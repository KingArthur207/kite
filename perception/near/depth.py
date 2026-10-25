"""Dense depth for the near system: a small backend interface, ground-anchored metric scale, per-pixel
confidence and ego-motion-compensated temporal smoothing.

Backends implement `estimate(frame) -> (depth, confidence | None)` and say what the numbers mean in `kind`:

    metric            metres along the optical axis, scale possibly off (Depth Anything V2 Metric)
    relative          an affine-invariant depth (z_true = s z + t)
    relative_inverse  an affine-invariant inverse depth / disparity (1 / z_true = a d + b; MiDaS, DA-V2 relative)

`perception/depth_zoo.py` (a parallel branch) is expected to provide more backends behind the same call; the
`zoo:<name>` spec loads one from there if that module exists.

Ground anchoring (what makes any of them metric on a chest camera): the lower middle of the frame is mostly
pavement, and for the ground the true depth of every row is known from geometry alone,
z_plane(v) = cam_h / (yn cos p + sin p). A consensus fit (RANSAC over candidate scales, and over a pitch grid
when the pitch is unknown) finds the scale (and shift) that makes the most ground candidates agree with the
plane; obstacles standing on the pavement are outliers and do not bias it. The fitted pitch feeds the ground
cue too. This builds on perception/fuse.py's fit_ground (line fit of Y against Z, trimmed) by replacing the
trimmed least squares with consensus, adding the affine and inverse-depth models, and reporting a residual
spread that becomes the depth cue's variance.

Confidence (ARCore's Raw Depth API ships one per pixel; Depth Anything does not, so we derive one): low at
depth discontinuities (a pixel straddling a pole's edge is half pole, half street), low far away, and low where
this frame disagrees with the motion-compensated previous one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .geometry import plane_depth
from .intrinsics import CameraProfile


class DepthBackend(Protocol):
    name: str
    kind: str

    def estimate(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]: ...


class DAv2Backend:
    """Depth Anything V2 Small, metric (perception/models.py MetricDepth: weights in weights/hf, never fetched)."""
    kind = "metric"

    def __init__(self, which: str = "outdoor", device: str = "cpu"):
        from ..models import MetricDepth
        self.name = f"depth-anything-v2-metric-{which}-small"
        self.model = MetricDepth(which, device)

    def estimate(self, frame):
        return self.model(frame), None


class ArrayBackend:
    """A backend from a function frame -> depth (tests, synthetic renders, replayed depth)."""

    def __init__(self, fn, kind: str = "metric", name: str = "array"):
        self.fn, self.kind, self.name = fn, kind, name

    def estimate(self, frame):
        out = self.fn(frame)
        return out if isinstance(out, tuple) else (out, None)


class ZooBackend:
    """Any perception/depth_zoo.py backend (feat/depth-shootout): dz.get(name).estimate(frame) -> DepthResult with
    depth_m and confidence. A non-metric one is treated as an affine disparity when its meta says so, else as a
    depth with an unknown scale; either way the ground anchor makes it metric. Its own "+anchor" suffix also works
    (then the near anchor sees a nearly metric map and fits a scale close to 1)."""

    def __init__(self, name: str, device: str = "cpu"):
        try:
            from .. import depth_zoo as dz  # type: ignore[attr-defined]
        except ImportError as e:
            raise SystemExit(f"--depth zoo:{name}: perception/depth_zoo.py is not on this branch ({e})")
        self.name = f"zoo:{name}"
        self.backend = dz.get(name, device)
        self.kind = "metric" if getattr(self.backend, "metric", True) else "relative"

    def estimate(self, frame):
        r = self.backend.estimate(frame)
        if not getattr(self.backend, "metric", True) and r.meta.get("disparity"):
            self.kind = "relative_inverse"
        return r.depth_m, getattr(r, "confidence", None)


def make_backend(spec: str, device: str = "cpu") -> DepthBackend | None:
    """--depth value -> backend: outdoor | indoor (DA-V2 metric small) | off | zoo:<name>."""
    if spec in (None, "", "off", "none"):
        return None
    if spec in ("outdoor", "indoor"):
        return DAv2Backend(spec, device)
    if spec.startswith("zoo:"):
        return ZooBackend(spec[4:], device)
    raise SystemExit(f"--depth {spec!r}: outdoor, indoor, off or zoo:<backend>")


# ------------------------------------------------------------------ ground anchoring
@dataclass
class Anchor:
    scale: float = 1.0          # z_true = scale * z + shift   (inverse: 1/z_true = scale * d + shift)
    shift: float = 0.0
    pitch: float = 0.0
    inlier_frac: float = 0.0
    sigma_log: float = 0.35     # spread of the ground's log residuals: the depth cue's own noise
    n: int = 0
    ok: bool = False
    mode: str = "scale"

    def apply(self, depth: np.ndarray) -> np.ndarray:
        d = depth.astype(np.float64)
        if self.mode == "inverse":
            inv = self.scale * d + self.shift
            return np.where(inv > 1e-3, 1.0 / np.maximum(inv, 1e-3), 200.0)
        return np.maximum(self.scale * d + self.shift, 0.05)


INLIER_LOG = 0.07          # a ground pixel within 7 % of the plane counts as agreeing
GROUND_COLS = (0.2, 0.8)
MAX_CAND = 3000


def ground_candidates(depth: np.ndarray, prof: CameraProfile, pitch: float, exclude: np.ndarray | None,
                      ignore_bottom: float = 0.03):
    """Pixels that could be pavement: below the horizon (with a margin), the middle columns, not inside a box."""
    h, w = depth.shape
    hz = prof.cy - prof.fy * math.tan(math.radians(pitch))
    r0 = int(max(hz + 0.08 * h, 0.45 * h))
    r1 = int(h * (1 - ignore_bottom))
    c0, c1 = int(w * GROUND_COLS[0]), int(w * GROUND_COLS[1])
    if r1 - r0 < 4:
        return None
    vv, uu = np.mgrid[r0:r1, c0:c1]
    z = depth[r0:r1, c0:c1].astype(np.float64)
    ok = np.isfinite(z) & (z > 1e-4)
    if exclude is not None:
        ok &= ~exclude[r0:r1, c0:c1]
    v, z = vv[ok], z[ok]
    if v.size > MAX_CAND:
        idx = np.linspace(0, v.size - 1, MAX_CAND).astype(int)
        v, z = v[idx], z[idx]
    return v.astype(np.float64), z


def anchor_ground(depth: np.ndarray, prof: CameraProfile, cam_h: float, pitch_prior: float,
                  exclude: np.ndarray | None = None, kind: str = "metric", fit_pitch: bool = True,
                  pitch_span: float = 15.0, iters: int = 96, seed: int = 0) -> Anchor:
    """Consensus fit of the depth model's scale (and shift) to the ground plane. `prof` is at the depth map's
    size. kind: metric -> scale only, and when fit_pitch the pitch too (1 degree grid over +-pitch_span, then
    0.25 degrees around the best); relative -> scale and shift; relative_inverse -> inverse-depth scale and
    shift (pitch fixed at the prior for both: with a free shift the pitch is barely observable)."""
    mode = {"metric": "scale", "relative": "affine", "relative_inverse": "inverse"}.get(kind, "scale")
    fail = Anchor(1.0, 0.0, pitch_prior, 0.0, 0.35, 0, False, mode)
    cand = ground_candidates(depth, prof, pitch_prior + (pitch_span if fit_pitch else 0.0), exclude)
    if cand is None or cand[0].size < 120:
        return fail
    v, z = cand
    rng = np.random.default_rng(seed)

    def at_pitch(p: float):
        zp = plane_depth(prof, v, cam_h, p)
        valid = np.isfinite(zp) & (zp < 40.0)
        if valid.sum() < 120:
            return None
        if mode == "scale":
            rho = np.where(valid, np.log(np.maximum(zp, 1e-6)) - np.log(z), np.nan)
            pool = rho[valid]
            cands = pool[rng.integers(0, pool.size, iters)]
            cnt = np.nansum(np.abs(rho[None, :] - cands[:, None]) < INLIER_LOG, axis=1)
            inl = valid & (np.abs(rho - cands[int(np.argmax(cnt))]) < INLIER_LOG)
            s, sh = float(np.exp(np.mean(rho[inl]))), 0.0
            pred = s * z
        else:
            idx = rng.integers(0, v.size, (iters, 2))
            x1, x2 = z[idx[:, 0]], z[idx[:, 1]]
            y = 1.0 / zp if mode == "inverse" else zp
            y1, y2 = y[idx[:, 0]], y[idx[:, 1]]
            den = np.where(np.abs(x2 - x1) > 1e-6, x2 - x1, np.nan)
            a = (y2 - y1) / den
            b = y1 - a * x1
            predm = a[:, None] * z[None, :] + b[:, None]
            if mode == "inverse":
                predm = np.where(predm > 1e-4, 1.0 / np.maximum(predm, 1e-4), np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                res = np.abs(np.log(predm / zp[None, :]))
            cnt = np.nansum(np.where(valid[None, :], res < INLIER_LOG, False), axis=1)
            cnt = np.where(np.isfinite(a) & (a > 0), cnt, -1)
            j = int(np.argmax(cnt))
            s, sh = float(a[j]), float(b[j])
            pred = s * z + sh if mode == "affine" else 1.0 / np.maximum(s * z + sh, 1e-4)
            with np.errstate(invalid="ignore", divide="ignore"):
                inl = valid & (np.abs(np.log(np.maximum(pred, 1e-6) / zp)) < INLIER_LOG)
        n_in = int(inl.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.log(np.maximum(pred[inl], 1e-6) / zp[inl])
        spread = float(1.4826 * np.median(np.abs(r - np.median(r)))) if n_in else 1.0
        return (n_in, -abs(p - pitch_prior)), Anchor(s, sh, p, n_in / v.size, max(0.04, spread), n_in, True, mode)

    def search(pitches):
        best = None
        for p in pitches:
            r = at_pitch(float(p))
            if r is not None and (best is None or r[0] > best[0]):
                best = r
        return best

    if fit_pitch and mode == "scale":
        best = search(np.arange(pitch_prior - pitch_span, pitch_prior + pitch_span + 1e-6, 1.0))
        if best is not None:
            p0 = best[1].pitch
            fine = search(np.arange(p0 - 1.0, p0 + 1.0 + 1e-6, 0.25))
            best = fine if fine is not None and fine[0] >= best[0] else best
    else:
        best = search([pitch_prior])
    if best is None:
        return fail
    a = best[1]
    if a.inlier_frac < 0.3 or a.n < 100 or (mode == "scale" and not 0.1 <= a.scale <= 10.0):
        return Anchor(a.scale, a.shift, pitch_prior, a.inlier_frac, 0.35, a.n, False, mode)
    a.pitch = round(a.pitch, 2)
    return a


class AnchorFilter:
    """Smooths per-frame anchors (log-scale and pitch, alpha 0.35) and holds the last good one `hold` frames."""

    def __init__(self, pitch: float, alpha: float = 0.35, hold: int = 30):
        self.given, self.alpha, self.hold = pitch, alpha, hold
        self.cur: Anchor | None = None
        self.miss = 0

    def update(self, a: Anchor) -> Anchor:
        if a.ok:
            self.miss = 0
            if self.cur is None or self.cur.mode != a.mode or a.mode != "scale":
                self.cur = a
            else:
                k = self.alpha
                c = self.cur
                self.cur = Anchor(math.exp((1 - k) * math.log(c.scale) + k * math.log(a.scale)), 0.0,
                                  (1 - k) * c.pitch + k * a.pitch, a.inlier_frac,
                                  (1 - k) * c.sigma_log + k * a.sigma_log, a.n, True, a.mode)
        else:
            self.miss += 1
            if self.miss > self.hold:
                self.cur = None
        if self.cur is not None:
            return self.cur
        return Anchor(1.0, 0.0, self.given, 0.0, 0.35, 0, False, a.mode)


# ------------------------------------------------------------------ confidence and smoothing
def confidence_map(z: np.ndarray, prev: np.ndarray | None = None, given: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel confidence in [0, 1]: depth edges (log-depth gradient over 8 % per pixel -> low), range (fades
    from 15 m to 40 m), temporal agreement with the motion-compensated previous map, and the backend's own."""
    lz = np.log(np.maximum(z, 1e-3))
    gy, gx = np.gradient(lz)
    edge = np.exp(-(np.hypot(gx, gy) / 0.08) ** 2)
    rng = np.clip((40.0 - z) / 25.0, 0.0, 1.0)
    c = edge * rng
    if prev is not None and prev.shape == z.shape:
        c = c * (0.4 + 0.6 * np.exp(-((lz - np.log(np.maximum(prev, 1e-3))) / 0.2) ** 2))
    if given is not None and given.shape == z.shape:
        c = c * np.clip(given, 0.0, 1.0)
    return c.astype(np.float32)


@dataclass
class DepthFrame:
    z: np.ndarray               # metric optical-axis depth, smoothed, at the depth map's size
    conf: np.ndarray
    anchor: Anchor
    prof: CameraProfile         # intrinsics at the depth map's size
    raw: np.ndarray             # the backend's output, unanchored
    backend: str = ""


class DepthProcessor:
    """backend -> anchored, confidence-weighted, temporally smoothed metric depth, once per frame."""

    def __init__(self, backend: DepthBackend, cam_h: float, pitch: float, fit_pitch: bool = True,
                 smooth: float = 0.6, anchor: bool = True):
        self.backend, self.cam_h, self.fit_pitch, self.smooth = backend, cam_h, fit_pitch, smooth
        self.anchor_on = anchor              # False: the model's own scale (the "depth model alone" baseline)
        self.filter = AnchorFilter(pitch)
        self.prev: np.ndarray | None = None
        self.prev_t: float | None = None
        self._maps = None

    def process(self, frame: np.ndarray, t: float, prof: CameraProfile, boxes=(), v_ego: float = 0.0
                ) -> DepthFrame:
        raw, given_conf = self.backend.estimate(frame)
        raw = np.asarray(raw, np.float64)
        pd = prof.resized(raw.shape[1], raw.shape[0])
        if pd.has_distortion:
            raw, given_conf, pd = self._undistort(raw, given_conf, pd)
        excl = np.zeros(raw.shape, bool)
        sx, sy = raw.shape[1] / prof.width, raw.shape[0] / prof.height
        for b in boxes:
            excl[int(b[1] * sy):int(math.ceil(b[3] * sy)), int(b[0] * sx):int(math.ceil(b[2] * sx))] = True
        if self.anchor_on:
            a = anchor_ground(raw, pd, self.cam_h, self.filter.cur.pitch if self.filter.cur else self.filter.given,
                              excl, getattr(self.backend, "kind", "metric"), self.fit_pitch)
            anchor = self.filter.update(a)
        else:
            anchor = Anchor(1.0, 0.0, self.filter.given, 0.0, 0.25, 0, False, "scale")
        z = anchor.apply(raw)
        prev = None
        if self.prev is not None and self.prev_t is not None:
            # the walker moved v * dt closer to everything ahead since the last map: predict, then blend
            dt = max(0.0, min(0.5, t - self.prev_t))
            prev = np.maximum(self.prev - v_ego * dt, 0.1)
        conf = confidence_map(z, prev, given_conf)
        if prev is not None and self.smooth > 0:
            lz, lp = np.log(np.maximum(z, 1e-3)), np.log(prev)
            jump = np.abs(lz - lp) > 0.3                 # new content (something stepped in): no smoothing
            k = np.where(jump, 1.0, 1.0 - self.smooth * (1.0 - conf) - 0.25 * self.smooth)
            z = np.exp(lp + np.clip(k, 0.2, 1.0) * (lz - lp))
        self.prev, self.prev_t = z, t
        return DepthFrame(z.astype(np.float32), conf, anchor, pd, raw, getattr(self.backend, "name", ""))

    def _undistort(self, raw, conf, pd):
        import cv2
        if self._maps is None or self._maps[0] != raw.shape:
            m1, m2 = cv2.initUndistortRectifyMap(pd.K, np.asarray(pd.dist, float), None, pd.K,
                                                 (raw.shape[1], raw.shape[0]), cv2.CV_32FC1)
            self._maps = (raw.shape, m1, m2)
        _, m1, m2 = self._maps
        raw = cv2.remap(raw.astype(np.float32), m1, m2, cv2.INTER_NEAREST).astype(np.float64)
        if conf is not None:
            conf = cv2.remap(conf.astype(np.float32), m1, m2, cv2.INTER_LINEAR)
        from dataclasses import replace
        return raw, conf, replace(pd, dist=[0.0] * 5)
