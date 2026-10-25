"""The independent distance cues, each with its own variance in log-distance, plus looming.

Every cue returns a Cue(forward distance in m, sigma in natural-log units) or None when it does not apply.
Working in log-distance makes the errors multiplicative, which is what they are: a 10 % height prior error is
10 % of range at 1 m and at 10 m.

  size_h   Z = fy H cos(p) / h_px          known real height (priors.py); needs the box uncut top and bottom
  size_w   Z = fx W / w_px                 known real width; needs the box uncut left and right
  ground   ray through the bottom-centre pixel meets the ground plane at cam_h, pitch p
  depth    a low percentile of the anchored dense depth inside the box (its middle), confidence-weighted
  motion   v_ego * tau for a fixed object: the walker's own speed turns looming into metric distance
           (motion stereo with one camera; the same idea as ARCore's depth from motion, per object)
  loom     tau = s / (ds/dt) from box scale expansion: time to contact with no metric distance at all

Variances: size cues from the prior's log-sigma plus 2 px of box noise; ground from the pitch uncertainty and
2 px of foot-row noise through d ln F / d theta = 1 / (sin theta cos theta), plus 5 % for uneven ground; depth
from the anchor's ground residual spread, the box's depth confidence and a 12 % model floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import angle_below_horizon, ground_point, norm_xy, z_to_forward
from .intrinsics import CameraProfile
from .priors import Prior

BOX_PX_NOISE = 2.0
MIN_BOX_PX = 6.0
MIN_D, MAX_D = 0.3, 60.0


@dataclass
class Cue:
    d: float            # forward distance (m)
    sigma: float        # log-sigma

    @property
    def log(self) -> float:
        return math.log(self.d)


@dataclass
class Box:
    """One detection in undistorted pixels, with which edges touch the frame (the object continues past them)."""
    x1: float
    y1: float
    x2: float
    y2: float
    cut_top: bool = False
    cut_bottom: bool = False
    cut_left: bool = False
    cut_right: bool = False

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


def _clamp(d: float) -> float:
    return float(min(MAX_D, max(MIN_D, float(d))))


def size_height(box: Box, prof: CameraProfile, prior: Prior, pitch: float = 0.0) -> Cue | None:
    if box.cut_top or box.cut_bottom or box.h < MIN_BOX_PX:
        return None
    d = prof.fy * prior.h * math.cos(math.radians(pitch)) / box.h
    return Cue(_clamp(d), math.hypot(prior.sh, BOX_PX_NOISE / box.h))


def size_width(box: Box, prof: CameraProfile, prior: Prior) -> Cue | None:
    if box.cut_left or box.cut_right or box.w < MIN_BOX_PX:
        return None
    return Cue(_clamp(prof.fx * prior.w / box.w), math.hypot(prior.sw, BOX_PX_NOISE / box.w))


def size_bound(box: Box, prof: CameraProfile, prior: Prior) -> float | None:
    """A box cut by the frame's top or bottom is taller than it looks: its height only says the object is NEARER
    than fy H / h_px. Returned as an upper bound for the fusion."""
    if box.h < MIN_BOX_PX or not (box.cut_top or box.cut_bottom):
        return None
    return _clamp(prof.fy * prior.h * math.exp(2 * prior.sh) / box.h)


def ground(box: Box, prof: CameraProfile, cam_h: float, pitch: float, pitch_sigma_deg: float,
           elevated: bool = False) -> Cue | None:
    """Flat-ground distance of the box's bottom-centre point. elevated: kinds that do not stand on the ground at
    their box bottom (overhead signs) get no ground cue."""
    if box.cut_bottom or elevated:
        return None
    gp = ground_point(prof, box.cx, box.y2, cam_h, pitch)
    if gp is None:
        return None
    theta = angle_below_horizon(prof, box.y2, pitch)
    s_theta = math.hypot(math.radians(pitch_sigma_deg), BOX_PX_NOISE / prof.fy)
    sig = s_theta / max(1e-3, math.sin(theta) * math.cos(theta))
    return Cue(_clamp(gp[0]), math.hypot(min(sig, 2.0), 0.05))


def depth(box: Box, prof: CameraProfile, z: np.ndarray, conf: np.ndarray, pd: CameraProfile, pitch: float,
          anchor_sigma: float, anchored: bool, mask: np.ndarray | None = None, pct: float = 25.0) -> Cue | None:
    """Anchored dense depth inside the box. `pd` is the intrinsics at the depth map's size. A thin upright box
    (a pole) uses the whole box and a lower percentile, since the object is a sliver of it."""
    dh, dw = z.shape
    sx, sy = dw / prof.width, dh / prof.height
    x1, x2 = int(box.x1 * sx), int(math.ceil(box.x2 * sx))
    y1, y2 = int(box.y1 * sy), int(math.ceil(box.y2 * sy))
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(dw, max(x2, x1 + 1)), min(dh, max(y2, y1 + 1))
    if x2 <= x1 or y2 <= y1:
        return None
    zz, cc = z[y1:y2, x1:x2], conf[y1:y2, x1:x2]
    if mask is not None and mask.shape == z.shape and mask[y1:y2, x1:x2].sum() >= 4:
        sel = mask[y1:y2, x1:x2]
    elif box.h >= 3 * box.w:
        sel, pct = np.ones_like(zz, bool), min(pct, 10.0)
    else:
        bh, bw = zz.shape
        sel = np.zeros_like(zz, bool)
        sel[bh // 4: max(bh // 4 + 1, 3 * bh // 4), bw // 4: max(bw // 4 + 1, 3 * bw // 4)] = True
    sel &= np.isfinite(zz) & (zz > 0)
    if sel.sum() < 2:
        return None
    vals, w = zz[sel], cc[sel]
    zq = float(np.percentile(vals, pct))
    c = float(np.mean(w))
    rows = np.nonzero(sel)[0]
    v_px = (y1 + float(np.median(rows)) + 0.5) / sy
    yn = norm_xy(prof, box.cx, v_px)[1]
    f = z_to_forward(zq, yn, pitch)
    # a box filling the frame leaves the model no scene context and sits far from the ground pixels that anchored
    # it: on the doorway clip it read a face at ~0.9 m as 3 m. Its weight drops as the box fills the frame.
    fill = box.w * box.h / float(prof.width * prof.height)
    sig = math.sqrt(anchor_sigma ** 2 + 0.12 ** 2 + (0.35 * (1.0 - c)) ** 2 + (0.8 * max(0.0, fill - 0.25)) ** 2)
    return Cue(_clamp(f), sig * (1.0 if anchored else 2.0))


# ------------------------------------------------------------------ looming
@dataclass
class Loom:
    tau: float | None       # s to contact (None when not expanding significantly)
    rate: float             # d ln(scale) / dt (1/s); positive = getting closer
    rate_sigma: float
    n: int


def box_scale(box: Box) -> float | None:
    """A size measure that does not change when the frame cuts the box: sqrt(w h) uncut, else the uncut side."""
    vert = not (box.cut_top or box.cut_bottom)
    horiz = not (box.cut_left or box.cut_right)
    if vert and horiz:
        return math.sqrt(max(box.w, 1.0) * max(box.h, 1.0))
    if horiz:
        return max(box.w, 1.0)
    if vert:
        return max(box.h, 1.0)
    return None


def looming(hist: list[tuple[float, float, str]], window_s: float = 0.9, min_n: int = 4) -> Loom:
    """hist: (t, scale, which-measure) per frame. Least-squares slope of ln(scale) over the last window_s seconds,
    on the samples that used the same measure as the newest one. tau = 1 / slope when the slope is positive and
    at least 2 sigma from zero."""
    if not hist:
        return Loom(None, 0.0, math.inf, 0)
    t_end, _, kind = hist[-1]
    pts = [(t, s) for t, s, k in hist if k == kind and t_end - t <= window_s]
    n = len(pts)
    if n < min_n or pts[-1][0] - pts[0][0] < 0.25:
        return Loom(None, 0.0, math.inf, n)
    t = np.array([p[0] for p in pts])
    y = np.log(np.array([p[1] for p in pts]))
    t = t - t.mean()
    sxx = float(np.sum(t * t))
    k = float(np.sum(t * (y - y.mean())) / sxx)
    res = y - y.mean() - k * t
    s_res = math.sqrt(max(float(np.sum(res * res)) / max(1, n - 2), 0.01 ** 2))   # >= 1 % box jitter
    sk = s_res / math.sqrt(sxx)
    tau = None
    if k > 0 and k > 2.0 * sk:
        # the least-squares slope is the expansion rate at the window's mean time, not now; at a constant closing
        # speed tau falls one second per second, so subtract the lag (half the window): 0.4 s earlier warnings
        tau = max(0.05, 1.0 / k - (pts[-1][0] - float(np.mean([p[0] for p in pts]))))
    return Loom(tau, k, sk, n)


def tau_upper(loom: Loom) -> float | None:
    """tau at one sigma of the expansion rate on the slow side: the override only fires when even this says
    contact is close (a jittery box can fake a large slope for a frame)."""
    if loom.tau is None:
        return None
    k_lo = loom.rate - loom.rate_sigma
    return math.inf if k_lo <= 0 else loom.tau + (1.0 / k_lo - 1.0 / loom.rate)


def motion(loom: Loom, v_ego: float, v_sigma_rel: float = 0.25) -> Cue | None:
    """Distance of a FIXED object from the walker's own speed and its looming: F = v * tau."""
    if loom.tau is None or v_ego < 0.3:
        return None
    rel_tau = loom.rate_sigma / max(loom.rate, 1e-6)
    return Cue(_clamp(v_ego * loom.tau), math.hypot(v_sigma_rel, min(rel_tau, 1.5)))
