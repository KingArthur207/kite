"""Depth sources for Kite behind one small interface, plus the geometry that makes any of them metric.

    from perception import depth_zoo as dz
    K = dz.Intrinsics.from_fov(640, 480, 60)            # a C270-class webcam: 60 degrees diagonal
    be = dz.get("da2-metric-outdoor-s+anchor", "mps")    # any registered name, "+anchor" / "+temporal" suffixes
    r = be.estimate(frame_bgr, K)                        # DepthResult: depth_m, confidence (0-1), latency_ms
    dz.anchor_to_ground(depth, K, cam_h=1.2)             # (depth_metric, scale, shift, pitch, inlier_ratio)

Every adapter imports its framework lazily and raises BackendUnavailable with the reason (missing package,
weights not downloaded, third-party code not cloned), so `available()` can list what runs on this machine.
Nothing downloads at runtime unless KITE_DEPTH_DOWNLOAD=1 (perception/bench_depth --setup sets it).

Three pieces are ours and pure numpy + OpenCV (tests run them without a model):

1. Ground anchoring. Floor pixels lie on a plane n.X = h in camera coordinates (n the unit "down" vector,
   h the camera height). RANSAC finds that plane in a depth map; the known chest height (1.2 m) then fixes
   the scale, and n gives pitch and roll. For a relative model that predicts affine-invariant disparity
   (1/z = a q + b) the floor is an affine function of the pixel in q, and that fixes only three of the four
   numbers (a, b, pitch, height): a pitch prior (calibration, IMU) resolves it, or we assume b = 0.
2. Motion stereo, after ARCore's depth-from-motion (Valentin et al., "Depth from motion for smartphone AR",
   SIGGRAPH Asia 2018) and the DepthLab utilities (Du et al., UIST 2020): pick a past keyframe with enough
   parallax, essential matrix for the pose up to scale, DIS optical flow for dense correspondences, per-pixel
   triangulation, scale from the floor at the known height, confidence from parallax, reprojection error,
   forward-backward consistency and texture, and a per-pixel Kalman filter over time (motion-compensated).
   ARCore has VIO poses; we have only images, so the pose comes from the frames themselves. Straight ahead
   is the focus of expansion when walking: parallax there is near zero, which is exactly the corridor Kite
   cares about. The confidence map says so honestly.
3. Fusion: inverse-variance (per pixel, in log depth) of an anchored dense model and motion stereo, after a
   robust global scale correction of the dense model against confident motion-stereo pixels.
"""

from __future__ import annotations

import functools
import math
import os
import sys
import time
import types
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_DEPTH = 80.0                       # m; sky and "infinitely far" are clipped here
MIN_DEPTH = 0.05
CHEST_H = 1.2                          # m, the chest webcam's height above the ground


def hf_cache() -> str:
    return os.environ.get("KITE_HF_CACHE", str(ROOT / "weights" / "hf"))


def third_party() -> Path:
    return Path(os.environ.get("KITE_DEPTH_THIRD_PARTY", str(ROOT / "weights" / "third_party")))


def _may_download() -> bool:
    return os.environ.get("KITE_DEPTH_DOWNLOAD") == "1"


class BackendUnavailable(RuntimeError):
    """A backend cannot run here; the message says why and how to fix it."""


# ------------------------------------------------------------------ the interface
@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, diag_fov_deg: float = 60.0) -> Intrinsics:
        """Square pixels, principal point at the centre, focal length from the diagonal field of view
        (Logitech C270: 60 degrees diagonal, C920: 78)."""
        f = 0.5 * math.hypot(width, height) / math.tan(math.radians(diag_fov_deg) / 2)
        return cls(f, f, width / 2, height / 2, width, height)

    def scaled(self, width: int, height: int) -> Intrinsics:
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float64)

    def rays(self) -> np.ndarray:
        """(h, w, 3) rays with z = 1: a pixel's 3D point is depth * ray. Cached per intrinsics; do not modify."""
        return _rays(self)


@functools.lru_cache(maxsize=16)
def _rays(K: Intrinsics) -> np.ndarray:
    u = (np.arange(K.width) + 0.5 - K.cx) / K.fx
    v = (np.arange(K.height) + 0.5 - K.cy) / K.fy
    r = np.empty((K.height, K.width, 3), np.float64)
    r[..., 0], r[..., 1], r[..., 2] = u[None, :], v[:, None], 1.0
    r.setflags(write=False)
    return r


@dataclass
class DepthResult:
    depth_m: np.ndarray                 # (h, w) float32, z-depth in metres (model units if the backend is not metric)
    confidence: np.ndarray              # (h, w) float32 in 0..1
    latency_ms: float
    meta: dict = field(default_factory=dict)


class DepthBackend:
    """`estimate(frame_bgr, intrinsics)` -> DepthResult at the frame's resolution. Stateful backends (video,
    motion stereo, temporal smoothing) keep state between calls; `reset()` starts a new sequence."""
    name = "base"
    metric = True                       # False: depth_m has an unknown scale (and maybe shift); see meta["disparity"]
    temporal = False                    # True: uses previous frames

    def estimate(self, frame_bgr: np.ndarray, intrinsics: Intrinsics | None = None) -> DepthResult:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def close(self) -> None:
        """Free model memory (the benchmark loads one model at a time on an 8 GB laptop)."""


def default_intrinsics(frame: np.ndarray, intrinsics: Intrinsics | None) -> Intrinsics:
    h, w = frame.shape[:2]
    if intrinsics is None:
        return Intrinsics.from_fov(w, h, 60.0)
    if (intrinsics.width, intrinsics.height) != (w, h):
        return intrinsics.scaled(w, h)
    return intrinsics


# ------------------------------------------------------------------ registry
@dataclass
class Entry:
    factory: Callable[..., DepthBackend]
    metric: bool
    licence: str
    note: str = ""


REGISTRY: dict[str, Entry] = {}


def register(name: str, factory: Callable[..., DepthBackend], metric: bool, licence: str, note: str = "") -> None:
    REGISTRY[name] = Entry(factory, metric, licence, note)


def get(name: str, device: str = "cpu", **kw) -> DepthBackend:
    """A backend by name. Suffixes compose: "da2-relative-s+anchor", "da2-metric-outdoor-s+anchor+temporal".
    "fused" is the anchored outdoor Small model fused with motion stereo; "fused:<dense name>" picks the dense
    model. Keyword arguments go to the base factory (cam_h also to the anchoring)."""
    cam_h = kw.get("cam_h", CHEST_H)
    if name.startswith("fused"):
        dense = name.split(":", 1)[1] if ":" in name else "da2-metric-outdoor-s+anchor"
        if "+anchor" not in dense:
            dense += "+anchor"
        return Fused(get(dense, device, **kw), MotionStereo(cam_h=cam_h))
    parts = name.split("+")
    base, mods = parts[0], parts[1:]
    if base not in REGISTRY:
        raise KeyError(f"unknown depth backend {base!r}; known: {', '.join(sorted(REGISTRY))}")
    entry = REGISTRY[base]
    base_kw = {k: v for k, v in kw.items() if k not in ("cam_h", "pitch_prior")}
    if base in ("motion-stereo", "flat-ground"):
        base_kw["cam_h"] = cam_h
    be = entry.factory(device=device, **base_kw)
    for m in mods:
        if m == "anchor":
            be = Anchored(be, cam_h=cam_h, pitch_prior=kw.get("pitch_prior"))
        elif m == "temporal":
            be = Temporal(be)
        else:
            raise KeyError(f"unknown modifier +{m} (known: +anchor, +temporal)")
    return be


def available(device: str = "cpu", names: list[str] | None = None) -> dict[str, str]:
    """{name: "ok" or the reason it cannot run}. Loads each model once, so it is slow for the big ones."""
    out = {}
    for n in names or sorted(REGISTRY):
        try:
            get(n, device).close()
            out[n] = "ok"
        except (BackendUnavailable, ImportError, OSError) as e:
            out[n] = str(e).splitlines()[0][:200]
    return out


# ------------------------------------------------------------------ confidence for single-image models
def edge_confidence(depth: np.ndarray, tau: float = 0.08, base: float = 0.8) -> np.ndarray:
    """Single-image models have no confidence of their own. Their worst pixels sit on depth edges, where the
    network interpolates between the foreground and the background ("flying pixels"), so confidence falls with
    the gradient of log depth per pixel. `base` caps it: a dense model is never certain."""
    import cv2
    z = np.log(np.clip(depth, MIN_DEPTH, MAX_DEPTH)).astype(np.float32)
    g = np.hypot(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)) / 8
    g = cv2.dilate(g, np.ones((3, 3), np.uint8))
    return (base * np.exp(-(g / tau) ** 2)).astype(np.float32)


def sigma_to_conf(sigma_rel: np.ndarray, ref: float = 0.05) -> np.ndarray:
    """Relative depth std -> confidence: 0.5 at `ref` (5 %), 0.86 at 2 %, 0.06 at 20 %."""
    with np.errstate(over="ignore", invalid="ignore"):
        c = 1.0 / (1.0 + (sigma_rel / ref) ** 2)
    return np.nan_to_num(c, nan=0.0).astype(np.float32)


def conf_to_sigma(conf: np.ndarray, ref: float = 0.05) -> np.ndarray:
    c = np.clip(conf, 1e-4, 1.0)
    return (ref * np.sqrt(1.0 / c - 1.0 + 1e-6)).astype(np.float32)


# ------------------------------------------------------------------ ground anchoring
class Anchored5(NamedTuple):
    depth_m: np.ndarray
    scale: float
    shift: float
    pitch: float              # degrees, + = camera tilted down
    inlier_ratio: float


@dataclass
class GroundFit:
    ok: bool
    normal: np.ndarray        # unit "down" vector in camera coordinates (x right, y down, z forward)
    height: float             # plane distance in the input's units (metric fits: the camera height)
    scale: float              # multiply the input depth by this (disparity fits: 1/z = scale * q + shift)
    shift: float
    pitch: float              # degrees
    roll: float               # degrees
    inlier_ratio: float
    inliers: np.ndarray | None = None
    reason: str = ""


def down_vector(pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """The ground normal (pointing down) in camera coordinates for a camera pitched down by `pitch` and rolled
    by `roll`: n = (cos p sin r, cos p cos r, sin p). pitch = asin(n_z), roll = atan2(n_x, n_y)."""
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    return np.array([math.cos(p) * math.sin(r), math.cos(p) * math.cos(r), math.sin(p)])


def angles_of(n: np.ndarray) -> tuple[float, float]:
    return math.degrees(math.asin(max(-1.0, min(1.0, float(n[2]))))), math.degrees(math.atan2(n[0], n[1]))


def _candidates(shape, mask, rows: float, max_pts: int, rng) -> np.ndarray:
    h, w = shape
    ok = np.zeros(shape, bool)
    ok[int(h * rows):] = True
    if mask is not None:
        ok &= mask
    idx = np.flatnonzero(ok)
    if idx.size > max_pts:
        idx = np.sort(rng.choice(idx, max_pts, replace=False))
    return idx


def _failed(reason: str, n=None) -> GroundFit:
    return GroundFit(False, down_vector(0.0) if n is None else n, float("nan"), 1.0, 0.0, float("nan"),
                     float("nan"), 0.0, None, reason)


def fit_ground_plane(depth: np.ndarray, intrinsics: Intrinsics, cam_h: float = CHEST_H, mask: np.ndarray | None = None,
                     pitch_range=(-15.0, 45.0), max_roll: float = 20.0, rows: float = 0.4, iters: int = 300,
                     min_inliers: int = 150, min_ratio: float = 0.12, seed: int = 0) -> GroundFit:
    """RANSAC plane n.X = d over pixels in the lower part of the frame (row >= rows * h), with n the down vector.
    Planes tilted like walls or ceilings (pitch or roll outside the ranges) are rejected. The inlier band is
    3 % of d plus 1 % of the point's depth, so it does not depend on the depth map's scale. scale = cam_h / d."""
    rng = np.random.default_rng(seed)
    K = intrinsics.scaled(depth.shape[1], depth.shape[0]) if (intrinsics.width, intrinsics.height) != \
        (depth.shape[1], depth.shape[0]) else intrinsics
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    if mask is not None:
        valid &= mask
    idx = _candidates(z.shape, valid, rows, 4000, rng)
    if idx.size < min_inliers:
        return _failed(f"only {idx.size} candidate pixels")
    P = K.rays().reshape(-1, 3)[idx] * z.reshape(-1)[idx, None]
    best = (-1, None)
    s = rng.integers(0, idx.size, (iters, 3))
    a, b, c = P[s[:, 0]], P[s[:, 1]], P[s[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    good = nn > 1e-12
    n[good] /= nn[good, None]
    d = np.einsum("ij,ij->i", n, a)
    flip = d < 0
    n[flip], d[flip] = -n[flip], -d[flip]
    pitch = np.degrees(np.arcsin(np.clip(n[:, 2], -1, 1)))
    roll = np.degrees(np.arctan2(n[:, 0], n[:, 1]))
    good &= (n[:, 1] > 0) & (pitch >= pitch_range[0]) & (pitch <= pitch_range[1]) & (np.abs(roll) <= max_roll)
    good &= d > 1e-9
    if not good.any():
        return _failed("no floor-like plane among the samples")
    n, d = n[good], d[good]
    res = np.abs(P @ n.T - d[None, :])                       # (pts, hyps)
    thr = 0.03 * d[None, :] + 0.01 * P[:, 2:3]
    score = (res < thr).sum(axis=0)
    k = int(np.argmax(score))
    best = (int(score[k]), (n[k], d[k]))
    n_k, d_k = best[1]
    inl = None
    for _ in range(3):                                       # least-squares refinement on the inliers
        inl = np.abs(P @ n_k - d_k) < 0.03 * d_k + 0.01 * P[:, 2]
        if inl.sum() < 3:
            break
        Q = P[inl]
        cen = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - cen, full_matrices=False)
        n_new = vt[-1]
        if n_new[1] < 0:
            n_new = -n_new
        n_k, d_k = n_new, float(cen @ n_new)
    pitch_k, roll_k = angles_of(n_k)
    ratio = float(inl.sum()) / idx.size
    full = np.zeros(z.size, bool)
    full[idx[inl]] = True
    fit = GroundFit(True, n_k, d_k, cam_h / d_k if d_k > 0 else 1.0, 0.0, pitch_k, roll_k, ratio,
                    full.reshape(z.shape))
    if d_k <= 0 or not (pitch_range[0] <= pitch_k <= pitch_range[1]) or abs(roll_k) > max_roll:
        fit.ok, fit.reason = False, f"refined plane is not a floor (pitch {pitch_k:.1f}, roll {roll_k:.1f})"
    elif inl.sum() < min_inliers or ratio < min_ratio:
        fit.ok, fit.reason = False, f"too few floor pixels ({int(inl.sum())}, {ratio:.0%})"
    return fit


def fit_ground_disparity(disp: np.ndarray, intrinsics: Intrinsics, cam_h: float = CHEST_H,
                         pitch_prior: float | None = None, mask: np.ndarray | None = None, rows: float = 0.4,
                         iters: int = 300, min_inliers: int = 150, min_ratio: float = 0.12, seed: int = 0,
                         pitch_range=(-15.0, 45.0), max_roll: float = 20.0) -> GroundFit:
    """Affine-invariant disparity q (larger = nearer) with true inverse depth 1/z = a q + b. On the floor
    1/z = n.r / h for the ray r = (x, y, 1), so q is affine in (x, y) there: q = A x + B y + C. RANSAC fits
    (A, B, C); then n is parallel to (aA, aB, aC + b) with |n| = 1. That is one equation for two unknowns, so:
    with `pitch_prior` (degrees) a and b both follow; without it we take b = 0 (the model's own horizon, where
    q -> 0, is the true one) and get the pitch from the fit. Returns scale = a, shift = b."""
    rng = np.random.default_rng(seed)
    K = intrinsics.scaled(disp.shape[1], disp.shape[0]) if (intrinsics.width, intrinsics.height) != \
        (disp.shape[1], disp.shape[0]) else intrinsics
    q = disp.astype(np.float64)
    valid = np.isfinite(q) & (q > 0)
    if mask is not None:
        valid &= mask
    idx = _candidates(q.shape, valid, rows, 4000, rng)
    if idx.size < min_inliers:
        return _failed(f"only {idx.size} candidate pixels")
    r = K.rays().reshape(-1, 3)[idx]
    X = np.stack([r[:, 0], r[:, 1], np.ones(idx.size)], 1)
    Q = q.reshape(-1)[idx]
    s = rng.integers(0, idx.size, (iters, 3))
    A = X[s]                                                  # (iters, 3, 3)
    ok = np.abs(np.linalg.det(A)) > 1e-12
    coef = np.zeros((iters, 3))
    coef[ok] = np.linalg.solve(A[ok], Q[s][ok][..., None])[..., 0]
    ok &= coef[:, 1] > 0                                      # disparity grows down the image on a floor
    if not ok.any():
        return _failed("no floor-like disparity ramp among the samples")
    coef = coef[ok]
    pred = X @ coef.T
    score = (np.abs(pred - Q[:, None]) < 0.04 * np.abs(pred) + 1e-9).sum(axis=0)
    c = coef[int(np.argmax(score))]
    inl = None
    for _ in range(3):
        pr = X @ c
        inl = np.abs(pr - Q) < 0.04 * np.abs(pr) + 1e-9
        if inl.sum() < 3:
            break
        c = np.linalg.lstsq(X[inl], Q[inl], rcond=None)[0]
    A_, B_, C_ = c
    if pitch_prior is None:
        norm = math.sqrt(A_ * A_ + B_ * B_ + C_ * C_)
        a, b = 1.0 / (cam_h * norm), 0.0
        n = np.array([A_, B_, C_]) / norm
    else:
        p = math.radians(pitch_prior)
        a = math.cos(p) / (cam_h * math.hypot(A_, B_))
        b = math.sin(p) / cam_h - a * C_
        n = np.array([a * A_ * cam_h, a * B_ * cam_h, math.sin(p)])
        n /= np.linalg.norm(n)
    pitch_k, roll_k = angles_of(n)
    ratio = float(inl.sum()) / idx.size
    full = np.zeros(q.size, bool)
    full[idx[inl]] = True
    fit = GroundFit(True, n, cam_h, a, b, pitch_k, roll_k, ratio, full.reshape(q.shape))
    if B_ <= 0 or not (pitch_range[0] <= pitch_k <= pitch_range[1]) or abs(roll_k) > max_roll:
        fit.ok, fit.reason = False, f"not a floor (pitch {pitch_k:.1f}, roll {roll_k:.1f})"
    elif inl.sum() < min_inliers or ratio < min_ratio:
        fit.ok, fit.reason = False, f"too few floor pixels ({int(inl.sum())}, {ratio:.0%})"
    return fit


def disparity_to_depth(disp: np.ndarray, scale: float, shift: float) -> np.ndarray:
    w = scale * disp.astype(np.float64) + shift
    with np.errstate(divide="ignore"):
        z = np.where(w > 1.0 / MAX_DEPTH, 1.0 / np.maximum(w, 1e-9), MAX_DEPTH)
    return z.astype(np.float32)


def anchor_to_ground(depth: np.ndarray, intrinsics: Intrinsics, cam_h: float = CHEST_H, kind: str = "depth",
                     pitch_prior: float | None = None, mask: np.ndarray | None = None) -> Anchored5:
    """Make a depth map metric by putting its floor at the known camera height.

    kind="depth": `depth` is z up to an unknown scale (a metric model whose scale is off, or motion stereo).
    Returns (depth * scale, scale, 0, pitch, inlier_ratio).
    kind="disparity": `depth` is affine-invariant disparity q (a relative model); 1/z = scale * q + shift, see
    fit_ground_disparity for why the pitch prior matters. On failure the input comes back with scale 1 (for
    disparity: 1/q), pitch nan and inlier ratio 0, so callers can tell and fall back."""
    if kind == "disparity":
        fit = fit_ground_disparity(depth, intrinsics, cam_h, pitch_prior, mask)
        if not fit.ok:
            return Anchored5(disparity_to_depth(depth, 1.0, 0.0), 1.0, 0.0, float("nan"), 0.0)
        return Anchored5(disparity_to_depth(depth, fit.scale, fit.shift), fit.scale, fit.shift, fit.pitch,
                         fit.inlier_ratio)
    fit = fit_ground_plane(depth, intrinsics, cam_h, mask)
    if not fit.ok:
        return Anchored5(depth.astype(np.float32), 1.0, 0.0, float("nan"), 0.0)
    return Anchored5((depth * fit.scale).astype(np.float32), fit.scale, 0.0, fit.pitch, fit.inlier_ratio)


class Anchored(DepthBackend):
    """Any backend, rescaled every frame so its floor sits at cam_h. Across frames the scale (and the shift of a
    relative model) is smoothed in log space; a failed fit holds the last good one for `hold` frames."""
    temporal = False

    def __init__(self, base: DepthBackend, cam_h: float = CHEST_H, pitch_prior: float | None = None,
                 alpha: float = 1.0, hold: int = 15):
        self.base, self.cam_h, self.pitch_prior, self.alpha, self.hold = base, cam_h, pitch_prior, alpha, hold
        self.name, self.metric = f"{base.name}+anchor", True
        self.temporal = base.temporal
        self.last: tuple[float, float] | None = None
        self.miss = 0

    def reset(self):
        self.base.reset()
        self.last, self.miss = None, 0

    def close(self):
        self.base.close()

    def estimate(self, frame_bgr, intrinsics=None) -> DepthResult:
        K = default_intrinsics(frame_bgr, intrinsics)
        r = self.base.estimate(frame_bgr, K)
        ts = time.perf_counter()
        disp = r.meta.get("disparity")
        relative = not self.base.metric and disp is not None
        if relative:
            fit = fit_ground_disparity(disp, K, self.cam_h, self.pitch_prior)
        else:
            fit = fit_ground_plane(r.depth_m, K, self.cam_h)
        conf = r.confidence
        if fit.ok:
            s, b = fit.scale, fit.shift
            if self.last is not None and self.alpha < 1.0:
                s = math.exp((1 - self.alpha) * math.log(self.last[0]) + self.alpha * math.log(s))
                b = (1 - self.alpha) * self.last[1] + self.alpha * b
            self.last, self.miss = (s, b), 0
        else:
            self.miss += 1
            if self.last is not None and self.miss <= self.hold:
                s, b = self.last
            else:
                s, b = (None, None)
            conf = conf * 0.5
        if relative:
            depth = disparity_to_depth(disp, s, b) if s is not None else r.depth_m
        else:
            depth = (r.depth_m * s).astype(np.float32) if s is not None else r.depth_m
        meta = {**r.meta, "anchor": {"ok": fit.ok, "scale": s, "shift": b, "pitch": fit.pitch, "roll": fit.roll,
                                     "inlier_ratio": fit.inlier_ratio, "reason": fit.reason}}
        meta.pop("disparity", None)
        return DepthResult(np.clip(depth, MIN_DEPTH, MAX_DEPTH).astype(np.float32), conf,
                           r.latency_ms + (time.perf_counter() - ts) * 1000, meta)


# ------------------------------------------------------------------ flat ground (the stand-in's prior)
class FlatGround(DepthBackend):
    """No model: every pixel below the horizon is floor at cam_h, everything else is far. This is the depth the
    stand-in's flat-ground rule implies; it never sees an obstacle, which is the point of the baseline."""
    name, metric = "flat-ground", True

    def __init__(self, device: str = "cpu", cam_h: float = CHEST_H, pitch: float = 0.0, roll: float = 0.0):
        self.cam_h, self.pitch, self.roll = cam_h, pitch, roll

    def estimate(self, frame_bgr, intrinsics=None) -> DepthResult:
        ts = time.perf_counter()
        K = default_intrinsics(frame_bgr, intrinsics)
        n = down_vector(self.pitch, self.roll)
        nr = K.rays() @ n
        with np.errstate(divide="ignore"):
            z = np.where(nr > self.cam_h / MAX_DEPTH, self.cam_h / np.maximum(nr, 1e-9), MAX_DEPTH)
        z = z.astype(np.float32)
        conf = np.where(z < MAX_DEPTH, 0.3, 0.05).astype(np.float32)
        return DepthResult(z, conf, (time.perf_counter() - ts) * 1000)


# ------------------------------------------------------------------ torch helpers
def _torch():
    try:
        import torch
    except ImportError as e:
        raise BackendUnavailable("needs torch (see perception/README.md, depth shoot-out, for the uv command)") from e
    return torch


def _free(device: str) -> None:
    import gc
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _sync(torch, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _fit14(w: int, h: int, long_side: int) -> tuple[int, int]:
    s = long_side / max(w, h)
    return max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)


# ------------------------------------------------------------------ Depth Anything V2 (transformers)
DA2 = {
    "da2-metric-outdoor-s": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", True, "Apache-2.0"),
    "da2-metric-indoor-s": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", True, "Apache-2.0"),
    "da2-metric-outdoor-b": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf", True, "CC-BY-NC-4.0"),
    "da2-metric-indoor-b": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", True, "CC-BY-NC-4.0"),
    "da2-relative-s": ("depth-anything/Depth-Anything-V2-Small-hf", False, "Apache-2.0"),
}


class HFDepth(DepthBackend):
    """Depth Anything V2 through transformers. Metric fine-tunes return metres; the relative model returns
    affine-invariant disparity (meta["disparity"]), with depth_m = 1/disparity normalised to median 1."""

    def __init__(self, name: str, device: str = "cpu", long_side: int = 448):
        torch = _torch()
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as e:
            raise BackendUnavailable("needs transformers") from e
        repo, self.metric, _ = DA2[name]
        self.name, self.device, self.long_side, self.torch = name, device, long_side, torch
        try:
            self.model = AutoModelForDepthEstimation.from_pretrained(
                repo, cache_dir=hf_cache(), local_files_only=not _may_download()).to(device).eval()
        except OSError as e:
            raise BackendUnavailable(f"{repo} not in {hf_cache()}: run python -m perception.bench_depth --setup") from e
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def close(self):
        del self.model
        _free(self.device)

    def estimate(self, frame_bgr, intrinsics=None) -> DepthResult:
        import cv2
        torch = self.torch
        h, w = frame_bgr.shape[:2]
        ts = time.perf_counter()
        x = cv2.resize(frame_bgr, _fit14(w, h, self.long_side), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
        t = torch.from_numpy(x).to(self.device).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            d = self.model(pixel_values=(t - self.mean) / self.std).predicted_depth
            d = torch.nn.functional.interpolate(d.reshape(1, 1, *d.shape[-2:]), (h, w), mode="bilinear",
                                                align_corners=False)[0, 0]
        d = d.float().cpu().numpy()
        ms = (time.perf_counter() - ts) * 1000
        if self.metric:
            z = np.clip(d, MIN_DEPTH, MAX_DEPTH).astype(np.float32)
            return DepthResult(z, edge_confidence(z), ms)
        q = np.maximum(d, 0).astype(np.float32)
        med = float(np.median(q[q > 0])) if (q > 0).any() else 1.0
        z = np.clip(med / np.maximum(q, 1e-6), MIN_DEPTH, MAX_DEPTH).astype(np.float32)
        return DepthResult(z, edge_confidence(z), ms, {"disparity": q})


for _n, (_repo, _metric, _lic) in DA2.items():
    register(_n, lambda device="cpu", _n=_n, **kw: HFDepth(_n, device, **kw), _metric, _lic, _repo)


# ------------------------------------------------------------------ Apple Depth Pro (transformers)
class DepthPro(DepthBackend):
    """apple/DepthPro-hf: metric depth at 1536x1536 from one image. It predicts its own field of view; with
    known intrinsics we use the real focal length instead (that is what makes it metric)."""
    name, metric = "depth-pro", True

    def __init__(self, device: str = "cpu", fp16: bool = True):
        torch = _torch()
        try:
            from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
        except ImportError as e:
            raise BackendUnavailable("needs transformers >= 4.49 (DepthPro)") from e
        self.torch, self.device = torch, device
        self.dtype = torch.float16 if fp16 and device != "cpu" else torch.float32
        try:
            kw = dict(cache_dir=hf_cache(), local_files_only=not _may_download())
            self.proc = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf", **kw)
            self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", torch_dtype=self.dtype,
                                                                    **kw).to(device).eval()
        except OSError as e:
            raise BackendUnavailable("apple/DepthPro-hf not downloaded: python -m perception.bench_depth --setup") from e

    def close(self):
        del self.model
        _free(self.device)

    def estimate(self, frame_bgr, intrinsics=None) -> DepthResult:
        torch = self.torch
        K = default_intrinsics(frame_bgr, intrinsics)
        h, w = frame_bgr.shape[:2]
        ts = time.perf_counter()
        inp = self.proc(images=frame_bgr[:, :, ::-1].copy(), return_tensors="pt")
        with torch.no_grad():
            out = self.model(pixel_values=inp["pixel_values"].to(self.device, self.dtype))
            canon = torch.nn.functional.interpolate(out.predicted_depth.float().reshape(1, 1, *out.predicted_depth.shape[-2:]),
                                                    (h, w), mode="bilinear", align_corners=False)[0, 0]
            fov = float(out.field_of_view.float().reshape(-1)[0]) if out.field_of_view is not None else float("nan")
        canon = canon.cpu().numpy()
        # transformers' post-processing: depth = 1 / (canonical * width / f); with the real f instead of the predicted
        z = K.fx / np.maximum(canon * w, 1e-6)
        ms = (time.perf_counter() - ts) * 1000
        z = np.clip(z, MIN_DEPTH, MAX_DEPTH).astype(np.float32)
        return DepthResult(z, edge_confidence(z), ms, {"pred_hfov_deg": fov})


register("depth-pro", DepthPro, True, "apple-amlr (research use only)", "apple/DepthPro-hf, 952M params")


# ------------------------------------------------------------------ UniDepth v2 (third-party code)
def _third_party_path(name: str, how: str) -> Path:
    p = third_party() / name
    if not p.exists():
        raise BackendUnavailable(f"{name} code not found at {p}: {how}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


class UniDepth(DepthBackend):
    """UniDepth v2 ViT-S (lpiccinelli/unidepth-v2-vits14). Its pip package pins triton and xformers, which do
    not exist on macOS; the code itself runs without them (wandb is only used for training logs, stubbed)."""
    name, metric = "unidepth-v2-s", True

    def __init__(self, device: str = "cpu"):
        torch = _torch()
        _third_party_path("UniDepth", "python -m perception.bench_depth --setup (git clone)")
        sys.modules.setdefault("wandb", types.ModuleType("wandb"))
        try:
            from unidepth.models import UniDepthV2
        except ImportError as e:
            raise BackendUnavailable(f"UniDepth import failed ({e}); needs einops, timm, huggingface_hub") from e
        self.torch, self.device = torch, device
        try:
            self.model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14", cache_dir=hf_cache(),
                                                    local_files_only=not _may_download()).to(device).eval()
        except Exception as e:  # noqa: BLE001  (hub mixin raises several kinds)
            raise BackendUnavailable(f"UniDepth weights unavailable: {e}") from e

    def close(self):
        del self.model
        _free(self.device)

    def estimate(self, frame_bgr, intrinsics=None) -> DepthResult:
        torch = self.torch
        K = default_intrinsics(frame_bgr, intrinsics)
        ts = time.perf_counter()
        rgb = torch.from_numpy(frame_bgr[:, :, ::-1].copy()).permute(2, 0, 1)
        with torch.no_grad():
            p = self.model.infer(rgb, torch.from_numpy(K.matrix()).float())
        z = p["depth"][0, 0].float().cpu().numpy()
        c = p.get("confidence")
        ms = (time.perf_counter() - ts) * 1000
        z = np.clip(z, MIN_DEPTH, MAX_DEPTH).astype(np.float32)
        conf = edge_confidence(z)
        if c is not None:     # UniDepth's confidence is an unbounded score; rank-normalise it per frame
            c = c[0, 0].float().cpu().numpy()
            conf = conf * (np.argsort(np.argsort(c.ravel())).reshape(c.shape) / max(c.size - 1, 1)).astype(np.float32)
        return DepthResult(z, conf.astype(np.float32), ms)


register("unidepth-v2-s", UniDepth, True, "CC-BY-NC-4.0", "lpiccinelli/unidepth-v2-vits14, takes intrinsics")


# ------------------------------------------------------------------ Metric3D v2 (third-party code, torch.hub)
class Metric3D(DepthBackend):
    """Metric3D v2 ViT-S through its hubconf. It needs mmcv only for a config class and an env dump (mmengine's
    Config stands in), and one decoder method hard-codes device="cuda" (patched to the model's device). Depth is
    predicted for a canonical camera (f = 1000 px) and rescaled with the real focal length."""
    name, metric = "metric3d-v2-s", True
    INPUT = (616, 1064)

    def __init__(self, device: str = "cpu"):
        torch = _torch()
        repo = _third_party_path("Metric3D", "python -m perception.bench_depth --setup (git clone)")
        try:
            from mmengine import Config, DictAction
        except ImportError as e:
            raise BackendUnavailable("Metric3D needs mmengine (pip install mmengine)") from e
        mm, mu = types.ModuleType("mmcv"), types.ModuleType("mmcv.utils")
        mu.Config, mu.DictAction, mu.collect_env, mu.get_git_hash = Config, DictAction, dict, lambda *a, **k: "none"
        mm.utils = mu
        sys.modules.setdefault("mmcv", mm)
        sys.modules.setdefault("mmcv.utils", mu)
        from mono.model.decode_heads import RAFTDepthNormalDPTDecoder5 as R

        def get_bins(self, n):
            dev = next(self.parameters()).device
            return torch.exp(torch.linspace(math.log(self.min_val), math.log(self.max_val), n, device=dev))
        R.RAFTDepthNormalDPT5.get_bins = get_bins
        torch.hub.set_dir(str(Path(hf_cache()).parent / "torchhub"))
        try:
            self.model = torch.hub.load(str(repo), "metric3d_vit_small", pretrain=True, source="local").to(device).eval()
        except Exception as e:  # noqa: BLE001
            raise BackendUnavailable(f"Metric3D load failed: {e}") from e
        self.torch, self.device = torch, device

    def close(self):
        del self.model
        _free(self.device)

    def estimate(self, frame_bgr, intrinsics=None) -> DepthResult:
        import cv2
        torch = self.torch
        K = default_intrinsics(frame_bgr, intrinsics)
        h, w = frame_bgr.shape[:2]
        ts = time.perf_counter()
        s = min(self.INPUT[0] / h, self.INPUT[1] / w)
        nh, nw = int(h * s), int(w * s)
        rgb = cv2.resize(frame_bgr[:, :, ::-1], (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mean, std = np.array([123.675, 116.28, 103.53], np.float32), np.array([58.395, 57.12, 57.375], np.float32)
        ph, pw = self.INPUT[0] - nh, self.INPUT[1] - nw
        top, left = ph // 2, pw // 2
        pad = np.empty((*self.INPUT, 3), np.float32)
        pad[:] = mean
        pad[top:top + nh, left:left + nw] = rgb
        t = torch.from_numpy(((pad - mean) / std).transpose(2, 0, 1)).float()[None].to(self.device)
        with torch.no_grad():
            d, conf, _ = self.model.inference({"input": t})
        d = d[0, 0, top:top + nh, left:left + nw].float().cpu().numpy()
        c = conf[0, 0, top:top + nh, left:left + nw].float().cpu().numpy()
        z = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) * (K.fx * s / 1000.0)
        c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
        ms = (time.perf_counter() - ts) * 1000
        z = np.clip(z, MIN_DEPTH, MAX_DEPTH).astype(np.float32)
        return DepthResult(z, (edge_confidence(z) * np.clip(c, 0, 1)).astype(np.float32), ms)


register("metric3d-v2-s", Metric3D, True, "BSD-2-Clause", "Metric3D v2 ViT-S (JUGGHM/Metric3D weights)")


# ------------------------------------------------------------------ Video Depth Anything (streaming, metric)
class VideoDepthAnything(DepthBackend):
    """Metric Video Depth Anything Small in its streaming mode (one frame in, cached temporal attention states
    from earlier frames). Temporally consistent by design; the streaming mode is training-free and the authors
    report it below the offline model."""
    name, metric, temporal = "vda-metric-s", True, True

    def __init__(self, device: str = "cpu", long_side: int = 448):
        torch = _torch()
        _third_party_path("Video-Depth-Anything", "python -m perception.bench_depth --setup (git clone)")
        try:
            from huggingface_hub import hf_hub_download
            from video_depth_anything.video_depth_stream import VideoDepthAnything as VDA
        except ImportError as e:
            raise BackendUnavailable(f"Video Depth Anything import failed ({e}); needs easydict, einops") from e
        try:
            ck = hf_hub_download("depth-anything/Metric-Video-Depth-Anything-Small",
                                 "metric_video_depth_anything_vits.pth", cache_dir=hf_cache(),
                                 local_files_only=not _may_download())
        except Exception as e:  # noqa: BLE001
            raise BackendUnavailable(f"Metric-Video-Depth-Anything-Small weights unavailable: {e}") from e
        self.cls, self.ck, self.torch, self.device, self.long_side = VDA, ck, torch, device, long_side
        self._load()

    def _load(self):
        m = self.cls(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        m.load_state_dict(self.torch.load(self.ck, map_location="cpu"), strict=True)
        self.model = m.to(self.device).eval()

    def reset(self):
        self.close()
        self._load()

    def close(self):
        self.model = None
        _free(self.device)

    def estimate(self, frame_bgr, intrinsics=None) -> DepthResult:
        ts = time.perf_counter()
        z = self.model.infer_video_depth_one(frame_bgr[:, :, ::-1].copy(), input_size=self.long_side,
                                             device=self.device, fp32=True)
        ms = (time.perf_counter() - ts) * 1000
        z = np.clip(np.asarray(z, np.float32), MIN_DEPTH, MAX_DEPTH)
        return DepthResult(z, edge_confidence(z), ms)


register("vda-metric-s", VideoDepthAnything, True, "Apache-2.0", "Metric-Video-Depth-Anything-Small, streaming")


# ------------------------------------------------------------------ flow and image helpers (OpenCV)
def _gray(frame_bgr: np.ndarray, width: int) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    h, w = g.shape
    if w != width:
        g = cv2.resize(g, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return g


def _remap(img: np.ndarray, flow: np.ndarray, interp=None, border=np.nan) -> np.ndarray:
    """Sample img at (x + flow_x, y + flow_y)."""
    import cv2
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR if interp is None else interp,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def push_pull_fill(depth: np.ndarray, weight: np.ndarray, levels: int = 6) -> np.ndarray:
    """Fill low-weight pixels from confident neighbours, coarse to fine (normalised convolution over a pyramid).
    Done in log depth so a hole between 2 m and 20 m fills with a geometric, not arithmetic, mean."""
    import cv2
    w = np.clip(weight, 0, 1).astype(np.float32)
    v = np.where(w > 0, np.log(np.clip(depth, MIN_DEPTH, MAX_DEPTH)), 0).astype(np.float32) * w
    pyr = [(v, w)]
    for _ in range(levels):
        v, w = cv2.pyrDown(v), cv2.pyrDown(w)
        pyr.append((v, w))
    val = pyr[-1][0] / np.maximum(pyr[-1][1], 1e-6)
    for v, w in reversed(pyr[:-1]):
        up = cv2.resize(val, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = np.clip(w * 4, 0, 1)
        val = a * (v / np.maximum(w, 1e-6)) + (1 - a) * up
    return np.exp(val).astype(np.float32)


# ------------------------------------------------------------------ motion stereo (depth from motion)
@dataclass
class _Frame:
    gray: np.ndarray          # work resolution (dense flow)
    full: np.ndarray          # pose resolution (sparse tracks): pose errors halve from 320 to 640 px wide
    t: int


class MotionStereo(DepthBackend):
    """Depth from a single moving camera (see the module docstring). Works at `work_width` (320 px) and returns
    the frame's resolution. meta: sigma_rel (per-pixel relative std), gap (frames to the keyframe), parallax_px
    (median, rotation removed), scale_ok, pitch (from the floor), still (True when there is no usable motion)."""
    name, metric, temporal = "motion-stereo", True, True

    def __init__(self, device: str = "cpu", cam_h: float = CHEST_H, work_width: int = 320, max_gap: int = 15,
                 min_parallax: float = 8.0, sigma_px: float = 0.5, temporal_fusion: bool = True,
                 pose_width: int = 640):
        import cv2
        self.cv2, self.cam_h, self.work_width, self.max_gap = cv2, cam_h, work_width, max_gap
        self.pose_width = pose_width
        self.min_parallax, self.sigma_px, self.fusion = min_parallax, sigma_px, temporal_fusion
        self.process = 0.006                                         # relative depth drift per frame of the prior
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.dis_fast = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.reset()

    def reset(self):
        self.buf: deque[_Frame] = deque(maxlen=self.max_gap + 1)
        self.n = 0
        self.state: tuple[np.ndarray, np.ndarray] | None = None       # fused z and variance at work resolution
        self.step_m: float | None = None                             # metric translation per frame (EMA)
        self.fwd_per_frame = 0.0
        self.pitch: float | None = None                              # running floor pitch, gates bad poses
        self.last_gap = 4

    # -------------------------------------------------- pose from sparse tracks
    def _pose(self, g, key, Kw, pts):
        cv2 = self.cv2
        lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p2, st, _ = cv2.calcOpticalFlowPyrLK(g, key, pts, None, **lk)
        p1b, st2, _ = cv2.calcOpticalFlowPyrLK(key, g, p2, None, **lk)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((p1b - pts)[:, 0], axis=1) < 0.7)
        if ok.sum() < 30:
            return None
        a, b = pts[ok, 0].astype(np.float64), p2[ok, 0].astype(np.float64)
        Km = Kw.matrix()
        E, m = cv2.findEssentialMat(a, b, Km, method=cv2.USAC_MAGSAC, prob=0.999, threshold=0.5)
        if E is None or E.shape[0] < 3:
            return None
        n_in, R, t, m = cv2.recoverPose(E[:3], a, b, Km, mask=m)
        inl = m[:, 0] > 0
        if n_in < 25 or abs(float(t[2, 0])) < 0.6:     # a walker's camera moves mostly along its axis
            return None
        H = Km @ R @ np.linalg.inv(Km)
        ah = np.c_[a[inl], np.ones(inl.sum())] @ H.T
        par = np.linalg.norm(b[inl] - ah[:, :2] / ah[:, 2:3], axis=1)
        return R, t[:, 0], float(np.median(par)), int(n_in)

    # -------------------------------------------------- one stereo measurement against a keyframe
    def _measure(self, g, Kw, gf, Kf):
        cv2 = self.cv2
        best = None
        gaps = [k for k in (1, 2, 3, 4, 6, 8, 10, 12, 15) if k < len(self.buf) and k <= self.max_gap]
        pts = cv2.goodFeaturesToTrack(gf, 1000, 0.005, 5, blockSize=7)
        if pts is None or len(pts) < 30 or not gaps:
            return None, {"still": True, "parallax_px": None}
        # Keyframe choice, as in depth from motion: the nearest past frame with enough parallax (baseline) and so the
        # most overlap. The search starts two steps below the last gap that worked and walks up: 1 to 5 solves.
        i = max(0, min(range(len(gaps)), key=lambda j: abs(gaps[j] - self.last_gap)) - 2)
        for gap in gaps[i:i + 5]:
            pose = self._pose(gf, self.buf[-1 - gap].full, Kf, pts)
            if pose is None:
                continue
            pose = (pose[0], pose[1], pose[2] * Kw.width / Kf.width, pose[3])     # parallax in work pixels
            if best is None or pose[2] > best[1][2]:
                best = (gap, pose)
            if pose[2] >= self.min_parallax:
                best = (gap, pose)
                break
        if best is not None:
            self.last_gap = best[0]
        if best is None or best[1][2] < 1.0:
            return None, {"still": True, "parallax_px": None if best is None else round(best[1][2], 2)}
        gap, (R, t, par_med, n_in) = best
        key = self.buf[-1 - gap].gray
        fw = self.dis.calc(g, key, None)
        bw = self.dis.calc(key, g, None)
        h, w = g.shape
        rays = Kw.rays().astype(np.float64)                       # m1 = K^-1 x1 with z = 1
        gx, gy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        x2 = np.stack([gx + fw[..., 0], gy + fw[..., 1]], -1)
        m2 = np.stack([(x2[..., 0] - Kw.cx) / Kw.fx, (x2[..., 1] - Kw.cy) / Kw.fy, np.ones((h, w))], -1)
        Rm1 = rays @ R.T
        a = np.cross(m2, Rm1)
        b = np.cross(m2, np.broadcast_to(t, m2.shape))
        aa = np.einsum("...i,...i", a, a)
        z = -np.einsum("...i,...i", a, b) / np.maximum(aa, 1e-15)
        X2 = Rm1 * z[..., None] + t
        with np.errstate(divide="ignore", invalid="ignore"):
            pu = Kw.fx * X2[..., 0] / X2[..., 2] + Kw.cx
            pv = Kw.fy * X2[..., 1] / X2[..., 2] + Kw.cy
        reproj = np.hypot(pu - x2[..., 0], pv - x2[..., 1])
        Hm = Kw.matrix() @ R @ np.linalg.inv(Kw.matrix())
        xr = np.stack([gx, gy, np.ones((h, w))], -1) @ Hm.T
        par = np.hypot(x2[..., 0] - xr[..., 0] / xr[..., 2], x2[..., 1] - xr[..., 1] / xr[..., 2])
        back = _remap(bw, fw.astype(np.float32), border=np.nan)
        fb = np.hypot(fw[..., 0] + back[..., 0], fw[..., 1] + back[..., 1])
        gm = cv2.GaussianBlur(np.hypot(cv2.Scharr(g, cv2.CV_32F, 1, 0), cv2.Scharr(g, cv2.CV_32F, 0, 1)) / 16,
                              (5, 5), 0)
        valid = np.isfinite(z) & (z > 0) & (X2[..., 2] > 0) & np.isfinite(fb) & np.isfinite(reproj)
        c_geo = np.exp(-(np.nan_to_num(reproj, nan=9) / 1.0) ** 2) * np.exp(-(np.nan_to_num(fb, nan=9) / 0.8) ** 2)
        c_tex = 1 - np.exp(-(gm / 6.0) ** 2)
        good = valid & (par > 2.0) & (c_geo > 0.5) & (c_tex > 0.5)
        fit = fit_ground_plane(np.where(good, z, np.nan), Kw, self.cam_h, mask=good, min_inliers=80, min_ratio=0.1)
        scale_ok = fit.ok and (self.pitch is None or abs(fit.pitch - self.pitch) < 8.0)
        # Scale: the floor fit, steadied by a constant-walking-speed prior (metres per frame, EMA). A fit more than
        # 15 % off the prior is taken as a bad fit (a tilted plane through repetitive floor tiles, a ramp).
        pred = None if self.step_m is None else self.step_m * gap
        if scale_ok:
            self.pitch = fit.pitch if self.pitch is None else 0.8 * self.pitch + 0.2 * fit.pitch
            s = fit.scale
            if pred is not None:
                if abs(math.log(s / pred)) < 0.15:
                    s = math.exp(0.5 * math.log(s) + 0.5 * math.log(pred))
                    self.step_m = 0.8 * self.step_m + 0.2 * fit.scale / gap
                else:
                    s, scale_ok = pred, False
                    self.step_m = 0.95 * self.step_m + 0.05 * fit.scale / gap
            else:
                self.step_m = s / gap
        elif pred is not None:
            s = pred
        else:
            return None, {"still": False, "scale_ok": False, "gap": gap, "parallax_px": round(par_med, 2)}
        c_key = -(R.T @ t) * s                                     # keyframe centre in current camera coordinates
        self.fwd_per_frame = float(-c_key[2] / gap)
        zm = np.where(valid, z * s, np.nan)
        sigma_rel = (self.sigma_px + np.nan_to_num(reproj, nan=9) + 0.5 * np.nan_to_num(fb, nan=9)) / np.maximum(par, 0.05)
        sigma_rel = sigma_rel / np.maximum(np.sqrt(c_tex), 0.05)
        if not scale_ok:
            sigma_rel = sigma_rel * 1.5 + 0.05
        sigma_rel = np.where(valid & (zm > MIN_DEPTH) & (zm < MAX_DEPTH), sigma_rel, np.inf)
        info = {"still": False, "scale_ok": scale_ok, "gap": gap, "parallax_px": round(par_med, 2),
                "inliers": n_in, "pitch": None if not scale_ok else round(fit.pitch, 2),
                "fwd_m_per_frame": round(self.fwd_per_frame, 4)}
        return (zm, sigma_rel), info

    # -------------------------------------------------- temporal fusion (per-pixel Kalman, motion-compensated)
    def _fuse(self, g, meas):
        cv2 = self.cv2
        prev = self.buf[-2].gray if len(self.buf) >= 2 else None
        if self.state is not None and prev is not None:
            zf, vf = self.state
            flow = self.dis_fast.calc(g, prev, None)
            zw = _remap(zf, flow, cv2.INTER_NEAREST)
            vw = _remap(vf, flow, cv2.INTER_NEAREST)
            photo = np.abs(g.astype(np.float32) - _remap(prev.astype(np.float32), flow))
            zp = zw - self.fwd_per_frame
            vp = vw + (self.process * zp) ** 2
            bad = ~np.isfinite(zp) | (zp <= MIN_DEPTH) | ~np.isfinite(photo) | (photo > 25)
            vp = np.where(bad, np.inf, vp)
            zp = np.where(bad, np.nan, zp)
        else:
            zp = vp = None
        if meas is not None:
            zm, sr = meas
            vm = np.where(np.isfinite(sr), (sr * np.nan_to_num(zm, nan=1.0)) ** 2, np.inf)
            zm = np.nan_to_num(zm, nan=1.0)
        if zp is None and meas is None:
            return None
        if zp is None:
            z, v = zm, vm
        elif meas is None:
            z, v = zp, vp
        else:
            have_p, have_m = np.isfinite(vp), np.isfinite(vm)
            e = zm - np.nan_to_num(zp, nan=0.0)
            S = np.where(have_p & have_m, vp + vm, np.inf)
            gate = have_p & have_m & (e * e > 9 * S)
            k = np.where(have_p & have_m, vp / np.where(np.isfinite(S), S, 1.0), np.where(have_m, 1.0, 0.0))
            z = np.where(have_p, np.nan_to_num(zp, nan=0.0), 0.0) + k * np.where(have_m, e, 0.0)
            v = np.where(have_p & have_m, vp * vm / np.where(np.isfinite(S), S, 1.0), np.where(have_m, vm, vp))
            # an innovation far outside both errors: something moved or the prior is stale; trust the sharper one
            z = np.where(gate, np.where(vm < vp, zm, zp), z)
            v = np.where(gate, np.minimum(vm, vp) * 4, v)
            z = np.where(have_p | have_m, z, np.nan)
        zz = np.nan_to_num(z, nan=1.0)
        # errors of successive measurements are correlated (same pose bias, same texture): never below 1.5 %
        v = np.where(np.isfinite(z) & (z > MIN_DEPTH), np.clip(v, (0.015 * zz) ** 2, (10 * zz) ** 2), np.inf)
        self.state = (z.astype(np.float32), v.astype(np.float32))
        return self.state

    def estimate(self, frame_bgr, intrinsics=None) -> DepthResult:
        cv2 = self.cv2
        ts = time.perf_counter()
        K = default_intrinsics(frame_bgr, intrinsics)
        h, w = frame_bgr.shape[:2]
        g = _gray(frame_bgr, self.work_width)
        gf = _gray(frame_bgr, min(w, self.pose_width))
        Kw, Kf = K.scaled(g.shape[1], g.shape[0]), K.scaled(gf.shape[1], gf.shape[0])
        self.buf.append(_Frame(g, gf, self.n))
        self.n += 1
        meas, info = (None, {"still": True}) if len(self.buf) < 2 else self._measure(g, Kw, gf, Kf)
        if self.fusion:
            st = self._fuse(g, meas)
        else:
            st = None if meas is None else (meas[0], (meas[1] * np.nan_to_num(meas[0], nan=1.0)) ** 2)
        if st is None:
            z = np.full((h, w), MAX_DEPTH, np.float32)
            return DepthResult(z, np.zeros((h, w), np.float32), (time.perf_counter() - ts) * 1000,
                               {**info, "sigma_rel": np.full((h, w), np.inf, np.float32)})
        zf, vf = st
        with np.errstate(invalid="ignore", divide="ignore"):
            sr = np.sqrt(vf) / np.maximum(np.nan_to_num(zf, nan=1.0), MIN_DEPTH)
        sr = np.where(np.isfinite(zf), sr, np.inf).astype(np.float32)
        conf = sigma_to_conf(sr)
        zfill = push_pull_fill(np.nan_to_num(zf, nan=1.0), conf)
        zfill = np.where(conf > 0.05, np.nan_to_num(zf, nan=1.0), zfill)
        z = cv2.resize(np.clip(zfill, MIN_DEPTH, MAX_DEPTH), (w, h), interpolation=cv2.INTER_LINEAR)
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
        sr_up = cv2.resize(np.minimum(sr, 1e3), (w, h), interpolation=cv2.INTER_LINEAR)
        info["sigma_rel"] = sr_up
        return DepthResult(z.astype(np.float32), conf.astype(np.float32), (time.perf_counter() - ts) * 1000, info)


register("motion-stereo", MotionStereo, True, "ours (MIT, this repo)",
         "keyframe essential matrix + DIS flow triangulation, floor-anchored scale, per-pixel Kalman")
register("flat-ground", FlatGround, True, "ours (MIT, this repo)", "the flat-ground prior, no model")


# ------------------------------------------------------------------ temporal smoothing for any backend
class Temporal(DepthBackend):
    """DepthLab-style temporal smoothing for a per-frame model: the previous output is warped into the current
    frame with optical flow, then blended per pixel in log depth. The blend trusts the new frame more where the
    warp fails the photometric check (disocclusion, something moving) and where the new frame is more confident.
    `forward_m` (metres per frame the camera moved forward, e.g. from motion stereo) removes the lag a plain EMA
    has while walking: without it a 2 m obstacle reads ~alpha_lag * 9 cm too far at 15 fps and 1.3 m/s."""

    def __init__(self, base: DepthBackend, alpha: float = 0.5, work_width: int = 320):
        import cv2
        self.base, self.alpha, self.work_width = base, alpha, work_width
        self.name, self.metric, self.temporal = f"{base.name}+temporal", base.metric, True
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.prev = None
        self.forward_m = 0.0

    def reset(self):
        self.base.reset()
        self.prev = None

    def close(self):
        self.base.close()

    def estimate(self, frame_bgr, intrinsics=None) -> DepthResult:
        import cv2
        r = self.base.estimate(frame_bgr, intrinsics)
        ts = time.perf_counter()
        h, w = r.depth_m.shape
        g = _gray(frame_bgr, self.work_width)
        sw, sh = g.shape[1], g.shape[0]
        ld = cv2.resize(np.log(np.clip(r.depth_m, MIN_DEPTH, MAX_DEPTH)), (sw, sh), interpolation=cv2.INTER_AREA)
        cf = cv2.resize(r.confidence, (sw, sh), interpolation=cv2.INTER_AREA)
        if self.prev is None:
            self.prev = (g, ld, cf)
            return r
        pg, pld, pcf = self.prev
        flow = self.dis.calc(g, pg, None)
        wld = _remap(pld, flow, cv2.INTER_LINEAR)
        wcf = _remap(pcf, flow, cv2.INTER_LINEAR, border=0.0)
        photo = np.abs(g.astype(np.float32) - _remap(pg.astype(np.float32), flow))
        if self.forward_m:
            wld = np.log(np.clip(np.exp(wld) - self.forward_m, MIN_DEPTH, MAX_DEPTH))
        ok = np.isfinite(wld) & np.isfinite(photo)
        trust = np.where(ok, np.exp(-(np.nan_to_num(photo, nan=255) / 12.0) ** 2), 0.0)
        a = 1 - (1 - self.alpha) * trust * (wcf / np.maximum(wcf + cf, 1e-6)) * 2
        a = np.clip(a, self.alpha * 0.5, 1.0)
        out = a * ld + (1 - a) * np.nan_to_num(wld, nan=0.0)
        conf = np.clip(a * cf + (1 - a) * wcf + 0.1 * trust * (1 - a), 0, 1)
        self.prev = (g, out, conf)
        depth = cv2.resize(np.exp(out), (w, h), interpolation=cv2.INTER_LINEAR)
        conf_up = cv2.resize(conf.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        # keep the full-resolution detail where nothing changed: add the base frame's high-frequency residual
        detail = np.log(np.clip(r.depth_m, MIN_DEPTH, MAX_DEPTH)) - cv2.resize(ld, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = np.exp(np.log(np.clip(depth, MIN_DEPTH, MAX_DEPTH)) + detail)
        return DepthResult(np.clip(depth, MIN_DEPTH, MAX_DEPTH).astype(np.float32), conf_up,
                           r.latency_ms + (time.perf_counter() - ts) * 1000, r.meta)


# ------------------------------------------------------------------ fusion: dense model + motion stereo
class Fused(DepthBackend):
    """Inverse-variance fusion, per pixel in log depth, of an anchored dense model and motion stereo.
    Step 1: the dense model's scale is corrected by the median ratio to motion stereo over pixels where both are
    confident (smoothed over frames): motion stereo's geometry is exact up to the floor-anchored scale, the
    network's is not. Step 2: sigma_dense = conf_to_sigma(conf) * dense_sigma_scale, sigma_ms from its own error
    model; the fused sigma becomes the confidence."""
    temporal = True
    metric = True

    def __init__(self, dense: DepthBackend, motion: MotionStereo, dense_sigma: float = 0.12,
                 scale_align: bool = True):
        self.dense, self.motion, self.dense_sigma, self.scale_align = dense, motion, dense_sigma, scale_align
        self.name = f"fused({dense.name}, {motion.name})"
        self.corr = 1.0

    def reset(self):
        self.dense.reset()
        self.motion.reset()
        self.corr = 1.0

    def close(self):
        self.dense.close()

    def estimate(self, frame_bgr, intrinsics=None) -> DepthResult:
        rd = self.dense.estimate(frame_bgr, intrinsics)
        rm = self.motion.estimate(frame_bgr, intrinsics)
        ts = time.perf_counter()
        zd = rd.depth_m.astype(np.float64)
        sm = rm.meta.get("sigma_rel")
        sm = np.full(zd.shape, np.inf) if sm is None else sm.astype(np.float64)
        both = (rm.confidence > 0.5) & (rd.confidence > 0.3) & (rm.depth_m < 20) & (zd > MIN_DEPTH)
        if self.scale_align and both.sum() > 0.01 * both.size:
            r = float(np.median(rm.depth_m[both] / zd[both]))
            if 0.5 < r < 2.0:
                self.corr = math.exp(0.8 * math.log(self.corr) + 0.2 * math.log(r))
        zd = zd * self.corr
        sd = np.maximum(conf_to_sigma(rd.confidence, 1.0) * self.dense_sigma, 0.03)
        wd, wm = 1 / sd ** 2, np.where(np.isfinite(sm), 1 / np.maximum(sm, 1e-3) ** 2, 0.0)
        lz = (wd * np.log(np.clip(zd, MIN_DEPTH, MAX_DEPTH)) + wm * np.log(np.clip(rm.depth_m, MIN_DEPTH, MAX_DEPTH))) \
            / (wd + wm)
        sf = 1 / np.sqrt(wd + wm)
        meta = {"dense": rd.meta, "motion": {k: v for k, v in rm.meta.items() if k != "sigma_rel"},
                "scale_corr": round(self.corr, 4), "ms_weight": float(np.mean(wm / (wd + wm))),
                "latency_parts_ms": {"dense": rd.latency_ms, "motion": rm.latency_ms}}
        return DepthResult(np.clip(np.exp(lz), MIN_DEPTH, MAX_DEPTH).astype(np.float32), sigma_to_conf(sf),
                           rd.latency_ms + rm.latency_ms + (time.perf_counter() - ts) * 1000, meta)


# ------------------------------------------------------------------ for perception/pipeline.py
class PipelineDepth:
    """Adapter with models.MetricDepth's call signature (img -> metric z at DEPTH_SIZE), so any backend can be
    dropped into Pipeline(depth=...) (`python -m perception --depth-backend NAME`). `fx` is the focal length in
    pixels at 960 wide, as the CLI uses it; unset, a C270-like 60 degree diagonal field of view is assumed."""

    def __init__(self, backend: DepthBackend, size=(448, 252), diag_fov_deg: float = 60.0, fx: float | None = None):
        self.backend, self.size, self.fov, self.fx = backend, size, diag_fov_deg, fx

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import cv2
        h, w = img.shape[:2]
        K = Intrinsics.from_fov(w, h, self.fov) if not self.fx else \
            Intrinsics(self.fx * w / 960, self.fx * w / 960, w / 2, h / 2, w, h)
        r = self.backend.estimate(img, K)
        return cv2.resize(r.depth_m, self.size, interpolation=cv2.INTER_LINEAR)


def describe() -> list[dict]:
    """One row per registered backend: name, metric, licence, note (for docs)."""
    return [{"name": n, "metric": e.metric, "licence": e.licence, "note": e.note} for n, e in sorted(REGISTRY.items())]
