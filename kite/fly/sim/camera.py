"""Pinhole projection of the walker and a fake person detector with the Tello stream's timing (DESIGN 2.3, 8 R5).

Projection. The camera is fixed to the body and pitched with it (+ = nose up). A point (forward, right, up) in the
level drone frame lands at camera depth z = f cos p + u sin p and image row cy + fy (f sin p - u cos p) / z.

Person box (R5): the union of a head part (head_w wide, head top to chin) and a torso part (shoulder width, from the
shoulder line 0.25 m under the head top down to the feet), each clipped to the frame. At follow range the feet are
far below the frame, so the box is cut by the bottom edge, which is what WalkerTracker acquires; at 1.0 m behind and
0.30 m above only the top of the head is in frame (the shoulder line is 29 deg down, the frame ends at 21 deg).

Detectable: the head top is inside the frame (column in [0, W], row in [0, 716)) and >= 6 px of the person are
visible. Then, with h the visible height:
  head only (no torso rows in frame)   p = p_head min(1, (h / 40)^2)   confidence U(0.25, 0.6)
                                       (p_head: U(0.3, 0.8) in train / stress, 0.55 in demo; DESIGN 10 R10)
  head and torso                       p = min(1, (h / 20)^2)       confidence U(0.4, 0.9)
Each edge gets Gaussian noise and the box is clipped to the frame again.

Timing. The detector processes frames at rate_hz (random phase). A frame captured at t_c reaches the laptop at
t_c + video_latency (the time it carries, Track.t), and its boxes reach the tracker det_time later. Frames lose their
walker box to iid dropout and to bursts (a Poisson process of 0.2-1.5 s blackouts; the video keeps flowing); any
frame may also carry one false person-sized box (half of them touching the bottom edge). Every per-frame random
number is drawn at reset from the detector's own stream and indexed by frame number, so the detector's luck is the
same for every arm.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ...follow import Detection

W, H = 960, 720
DETECT_ROW_PX = 716.0  # the head top must be above this row to be detectable
MIN_VISIBLE_PX = 6.0
SHOULDER_DROP_M = 0.25  # head top -> shoulder line
FAR = 1e9  # image row of a point at or behind the camera plane (below the frame)
HEAD_ONLY = (40.0, 0.25, 0.6)  # (full-detection height px, confidence lo, hi); max p is DetectorParams.p_head
WITH_TORSO = (20.0, 0.4, 0.9)  # max p 1

# (head column u, head-top row, chin row, head half width px, shoulder row, torso half width px, feet row)
Geom = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class DetectorParams:
    rate_hz: float = 15.0
    video_latency_s: float = 0.10
    det_time_s: float = 0.05
    noise_px: float = 2.0  # sigma per box edge
    dropout: float = 0.0
    burst_rate_hz: float = 0.0
    false_frac: float = 0.0
    burst_len_s: tuple[float, float] = (0.2, 1.5)
    false_w_px: tuple[float, float] = (60.0, 300.0)
    false_h_px: tuple[float, float] = (80.0, 400.0)
    false_conf: tuple[float, float] = (0.3, 0.9)
    p_head: float = 0.55  # head-only detection probability once the head is >= 40 px tall


def project(fwd: float, right: float, up_top: float, head_m: float, head_half_w: float, torso_half_w: float,
            cam_z: float, pitch_deg: float, fx: float, fy: float, cx: float = W / 2, cy: float = H / 2) -> Geom | None:
    """The walker in the level drone frame (m): its axis `fwd` ahead and `right` of the camera, head top `up_top`
    above it (negative from above), feet at -cam_z. None when the head top is behind the camera."""
    p = math.radians(pitch_deg)
    c, s = math.cos(p), math.sin(p)
    fs, fc = fwd * s, fwd * c
    z = fc + up_top * s
    if z < 0.05:
        return None
    inv = 1.0 / z
    u_chin, u_sh = up_top - head_m, up_top - SHOULDER_DROP_M
    zc, zs, zf = fc + u_chin * s, fc + u_sh * s, fc - cam_z * s
    return (cx + fx * right * inv, cy + fy * (fs - up_top * c) * inv,
            cy + fy * (fs - u_chin * c) / zc if zc > 0.05 else FAR,
            fx * head_half_w * inv,
            cy + fy * (fs - u_sh * c) / zs if zs > 0.05 else FAR,
            fx * torso_half_w / zs if zs > 0.05 else 0.0,
            cy + fy * (fs + cam_z * c) / zf if zf > 0.05 else FAR)


def head_in_view(g: Geom | None) -> bool:
    return g is not None and 0.0 <= g[0] <= W and 0.0 <= g[1] < DETECT_ROW_PX


def visible_box(g: Geom | None) -> tuple[float, float, float, float, bool] | None:
    """(x1, y1, x2, y2, torso_in_frame): the union of the visible parts, clipped; None if not detectable."""
    if not head_in_view(g):
        return None
    u, vt, vc, hwh, vs, hwt, vf = g
    x1, x2 = u - hwh, u + hwh
    y2 = vc if vc < H else float(H)
    torso = vs <= H - MIN_VISIBLE_PX and u + hwt > 0.0 and u - hwt < W
    if torso:
        if u - hwt < x1:
            x1 = u - hwt
        if u + hwt > x2:
            x2 = u + hwt
        yb = vf if vf < H else float(H)
        if yb > y2:
            y2 = yb
    if y2 - vt < MIN_VISIBLE_PX:
        return None
    return (x1 if x1 > 0.0 else 0.0), (vt if vt > 0.0 else 0.0), (x2 if x2 < W else float(W)), y2, torso


def _lerp(a: Geom | None, b: Geom | None, w: float) -> Geom | None:
    if a is None or b is None:
        return b if w >= 0.5 else a
    v = 1.0 - w
    return tuple(v * x + w * y for x, y in zip(a, b))


class FakeDetector:
    """Frames captured in [t0, t1]; capture() makes their boxes, ready() hands them out when they reach the tracker.

    A ready item is (t_available, frame_t, detections, true_conf): true_conf is the walker box's confidence, 0 when
    the frame has no walker box."""

    def __init__(self, p: DetectorParams, rng: np.random.Generator, t0: float, t1: float):
        self.p = p
        period = 1.0 / p.rate_hz
        n = int((t1 - t0) * p.rate_hz) + 2
        self._t = (t0 + float(rng.uniform(0.0, period)) + np.arange(n) * period).tolist() + [math.inf]
        u = rng.random((9, n))
        z = rng.standard_normal((4, n)) * p.noise_px
        gaps, lens = rng.exponential(1.0, 64), rng.uniform(p.burst_len_s[0], p.burst_len_s[1], 64)
        drop = u[0] < p.dropout
        if p.burst_rate_hz > 0.0:
            tt = np.asarray(self._t[:n])
            for start, ln in zip(t0 + np.cumsum(gaps) / p.burst_rate_hz, lens):
                if start > t1:
                    break
                drop |= (tt >= start) & (tt < start + ln)
        self._drop = drop.tolist()
        self._udet = u[1].tolist()
        self._uconf = u[2].tolist()
        self._noise = z.T.tolist()
        self._false = (u[3] < p.false_frac).tolist()
        fw = p.false_w_px[0] + u[5] * (p.false_w_px[1] - p.false_w_px[0])
        fh = p.false_h_px[0] + u[6] * (p.false_h_px[1] - p.false_h_px[0])
        fx1 = u[4] * (W - fw)
        fy1 = np.where(u[7] < 0.5, H - fh, (u[7] - 0.5) * 2.0 * (H - fh))  # half of them touch the bottom edge
        fconf = p.false_conf[0] + u[8] * (p.false_conf[1] - p.false_conf[0])
        self._fbox = np.stack([fx1, fy1, fx1 + fw, fy1 + fh, fconf], axis=1).tolist()
        self._k = 0
        self._lat = p.video_latency_s
        self._avail = p.video_latency_s + p.det_time_s
        self._q: deque = deque()
        self.n_frames = 0
        self.n_true = 0

    def capture(self, t_prev: float, t_now: float, g_prev: Geom | None, g_now: Geom | None) -> None:
        """Every frame captured in (t_prev, t_now]; the geometry is interpolated between the two ticks."""
        ts, k = self._t, self._k
        if ts[k] > t_now:
            return
        span = t_now - t_prev
        while ts[k] <= t_now:
            tc = ts[k]
            dets: list[Detection] = []
            true_conf = 0.0
            if not self._drop[k]:
                box = visible_box(_lerp(g_prev, g_now, (tc - t_prev) / span if span > 0.0 else 1.0))
                if box is not None:
                    x1, y1, x2, y2, torso = box
                    h_full, c_lo, c_hi = WITH_TORSO if torso else HEAD_ONLY
                    p_max = 1.0 if torso else self.p.p_head
                    hv = y2 - y1
                    if self._udet[k] < p_max * min(1.0, (hv / h_full) ** 2):
                        n0, n1, n2, n3 = self._noise[k]
                        x1, y1 = max(0.0, x1 + n0), max(0.0, y1 + n1)
                        x2, y2 = min(float(W), max(x1 + 1.0, x2 + n2)), min(float(H), max(y1 + 1.0, y2 + n3))
                        true_conf = c_lo + self._uconf[k] * (c_hi - c_lo)
                        dets.append(Detection(x1, y1, x2, y2, true_conf))
                        self.n_true += 1
            if self._false[k]:
                dets.append(Detection(*self._fbox[k]))
            self._q.append((tc + self._avail, tc + self._lat, dets, true_conf))
            self.n_frames += 1
            k += 1
        self._k = k

    def ready(self, t: float) -> list[tuple[float, float, list[Detection], float]]:
        q, out = self._q, []
        while q and q[0][0] <= t:
            out.append(q.popleft())
        return out
