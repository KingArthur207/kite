"""Synthetic renders with known geometry: the only frames where every cue's error can be measured exactly.

A scene is flat ground, a camera at (cam_h, pitch) with known intrinsics (optionally with lens distortion), and
upright rectangles (people, poles, cars, ...) of known real size at known forward / lateral positions. render()
returns the detector boxes a perfect detector would give (plus jitter), a depth map as a depth model with a
wrong global scale, per-object bias and pixel noise would give, and the ground truth.

World frame: forward f, lateral l (+ right), up u. Camera axes: z = (cos p, -sin p) and y (down) =
(-sin p, -cos p) in (forward, up), x = lateral. So z_c = f cos p - (u - h) sin p, y_c = -f sin p - (u - h) cos p.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .intrinsics import CameraProfile


@dataclass
class Obj:
    kind: str
    forward: float          # m to the object's front face
    lateral: float          # m, centre, + right
    h: float                # real height (m)
    w: float                # real width (m)
    bottom: float = 0.0     # height of its lowest edge above the ground (0 = standing on it)
    label: str = ""


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str
    conf: float = 0.9
    label: str = ""
    tid: str | None = None
    mask: np.ndarray | None = None


@dataclass
class Render:
    dets: list[Det]
    depth: np.ndarray | None
    truth: list[Obj]
    visible: list[bool] = field(default_factory=list)


def project(prof: CameraProfile, f: np.ndarray, l: np.ndarray, u: np.ndarray, cam_h: float, pitch: float,
            distort: bool = True):
    p = math.radians(pitch)
    zc = f * math.cos(p) - (u - cam_h) * math.sin(p)
    yc = -f * math.sin(p) - (u - cam_h) * math.cos(p)
    px = np.stack([prof.fx * l / zc + prof.cx, prof.fy * yc / zc + prof.cy], axis=-1)
    if distort and prof.has_distortion:
        px = prof.distort_points(px.reshape(-1, 2)).reshape(px.shape)
    return px, zc


def box_of(prof: CameraProfile, o: Obj, cam_h: float, pitch: float, n: int = 9):
    """Bounding box (distorted pixels) of the object's outline, clipped to the frame; None if not visible."""
    s = np.linspace(0, 1, n)
    l0, l1 = o.lateral - o.w / 2, o.lateral + o.w / 2
    u0, u1 = o.bottom, o.bottom + o.h
    ls = np.concatenate([l0 + (l1 - l0) * s, l0 + (l1 - l0) * s, np.full(n, l0), np.full(n, l1)])
    us = np.concatenate([np.full(n, u0), np.full(n, u1), u0 + (u1 - u0) * s, u0 + (u1 - u0) * s])
    px, zc = project(prof, np.full(ls.shape, o.forward), ls, us, cam_h, pitch)
    if np.any(zc <= 0.05):
        return None
    x1, y1 = px[:, 0].min(), px[:, 1].min()
    x2, y2 = px[:, 0].max(), px[:, 1].max()
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(prof.width - 1.0, x2), min(prof.height - 1.0, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return x1, y1, x2, y2


def depth_map(prof: CameraProfile, objs: list[Obj], cam_h: float, pitch: float, size=(448, 252), sky: float = 80.0
              ) -> np.ndarray:
    """True optical-axis depth at the depth model's resolution (rays through distorted pixels)."""
    pd = prof.resized(*size)
    vv, uu = np.mgrid[0:size[1], 0:size[0]].astype(np.float64)
    und = pd.undistort_points(np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5], 1)).reshape(size[1], size[0], 2)
    xn = (und[..., 0] - pd.cx) / pd.fx
    yn = (und[..., 1] - pd.cy) / pd.fy
    p = math.radians(pitch)
    down = yn * math.cos(p) + math.sin(p)
    fwd_per_z = math.cos(p) - yn * math.sin(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(down > 1e-4, cam_h / down, sky)
    z = np.minimum(z, sky)
    for o in objs:
        with np.errstate(divide="ignore", invalid="ignore"):
            zo = np.where(fwd_per_z > 1e-4, o.forward / fwd_per_z, np.inf)
        lat = zo * xn
        hgt = cam_h - zo * down
        hit = (np.abs(lat - o.lateral) <= o.w / 2) & (hgt >= o.bottom) & (hgt <= o.bottom + o.h) & (zo < z)
        z = np.where(hit, zo, z)
    return z


def model_depth(z_true: np.ndarray, rng: np.random.Generator, scale: float = 1.0, noise: float = 0.05,
                kind: str = "metric") -> np.ndarray:
    """What a monocular depth model might say: a wrong global scale, smooth multiplicative noise, and for
    relative_inverse an affine disparity."""
    n = rng.normal(0.0, noise, z_true.shape)
    k = np.ones(5) / 5
    n = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, n)
    n = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, n) * math.sqrt(5)
    z = z_true * scale * np.exp(n)
    if kind == "relative_inverse":
        return 0.7 / z + 0.02
    return z


def render(prof: CameraProfile, objs: list[Obj], cam_h: float = 1.2, pitch: float = 5.0,
           rng: np.random.Generator | None = None, jitter: float = 0.02, jitter_px: float = 1.5,
           depth: bool = True, depth_scale: float = 1.0, depth_noise: float = 0.05, obj_bias: float = 0.08,
           depth_kind: str = "metric") -> Render:
    rng = rng or np.random.default_rng(0)
    dets, vis = [], []
    for i, o in enumerate(objs):
        b = box_of(prof, o, cam_h, pitch)
        vis.append(b is not None)
        if b is None:
            continue
        x1, y1, x2, y2 = b
        bw, bh = x2 - x1, y2 - y1
        W, H = prof.width - 1.0, prof.height - 1.0
        cut = (x1 <= 0.0, y1 <= 0.0, x2 >= W, y2 >= H)
        j = lambda s: float(rng.normal(0.0, jitter * s + jitter_px))
        e = lambda: float(rng.uniform(0.0, 6.0))        # detectors stop a few px short of a frame edge
        x1 = e() if cut[0] else x1 + j(bw)
        y1 = e() if cut[1] else y1 + j(bh)
        x2 = W - e() if cut[2] else x2 + j(bw)
        y2 = H - e() if cut[3] else y2 + j(bh)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        dets.append(Det(x1, y1, x2, y2, o.kind, 0.9, o.label or o.kind, f"s{i}"))
    dm = None
    if depth:
        biased = [Obj(o.kind, o.forward * math.exp(rng.normal(0.0, obj_bias)), o.lateral, o.h, o.w, o.bottom)
                  for o in objs]
        z = depth_map(prof, biased, cam_h, pitch)
        dm = model_depth(z, rng, depth_scale, depth_noise, depth_kind)
    return Render(dets, dm, objs, vis)


def image_for(prof: CameraProfile) -> np.ndarray:
    """A blank frame of the profile's size (the system only needs its shape when detections are given)."""
    return np.zeros((prof.height, prof.width, 3), np.uint8)
