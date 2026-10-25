"""A fake YOLO pose detector: people as the app's PoseDetector returns them, from simulator truth (DESIGN section 3).

Each person is a 3D body: 17 COCO keypoints (nose, eyes and ears at the head, the face points 0.09 m in front of the
head centre; shoulders at 0.82 H, elbows 0.63 H, wrists 0.48 H, hips 0.53 H, knees 0.29 H, ankles 0.04 H, widths
from the shoulder width) and a silhouette (a horizontal ellipse per height, 1.2 x the shoulder width across the arms,
0.28 m deep), projected through the drone's TRUE pose and the TRUE camera (fx = fy drawn, a mount pitch drawn, cx 480,
cy 360). The follow believes BehindCfg.camera_fx and a level mount, as live.

Projection, the convention of CameraModel.ray inverted (the camera pitched up by p = body pitch + mount, + nose up):
a point (forward F, right R, up U) in the level drone frame is at camera depth z = F cos p + U sin p, column
cx + fx R / z and row cy + fy (F sin p - U cos p) / z. Nose up moves the scene DOWN the frame.

Per frame:
  bbox       the extent of the visible body: the head top to the lowest visible point, across the silhouette's
             widest visible part, clipped to the frame; each edge that is inside the frame + N(0, box_noise_px)
             (an edge cut by the frame stays on it, as YOLO's boxes do, so a cut box never passes for a whole one)
  keypoints  (17, 3) float32 x, y, conf; position noise N(0, kp_noise_px). Conf: in frame and facing the camera side
             U(0.6, 0.95); face points seen from behind U(0.05, 0.35); ears from behind U(0.4, 0.85), the far ear in
             a side view U(0.05, 0.35); a body joint on the far side in a side view U(0.35, 0.75); out of frame
             U(0, 0.25) at the projected (extrapolated) position. The shoulder labels swap with p swap_p.
             Seen from behind the person's RIGHT shoulder is on the image right (follow_behind's facing()).
  detected   with p = det_base_p * min(1, visible body height / 80 px)^2, except in a dropout burst (Poisson
             burst_rate_hz, 0.2-1.0 s each); a person whose head is hidden behind a nearer person's body is not
             detected. One false box (random keypoints) with p false_box_p. Most confident first.
The detector is the `detect` callable kite.follow_behind.Perception is given: the world load()s the frame's snapshot
(the truth at its capture time) before perception runs, and __call__ ignores the pixels. Which source made each
detection (SRC_PERSON, SRC_SECOND, SRC_FALSE) is kept per frame seq in `sources`, outside PersonDet.

Every per-frame random number is drawn at reset and indexed by the capture number, from the pose stream (the
person, false boxes, bursts) and the second person's own stream, so the detector's luck is the same for every arm.
Pure-Python floats per frame: only frames handed to perception (about 20 per second) are ever built.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ...follow_behind import PersonDet

W, H = 960, 720
CX, CY = 480.0, 360.0
FAR = 1e6  # image row/column of a point at or behind the camera plane
Z_MIN = 0.05  # camera depth below which a point is "behind the camera"
SRC_PERSON, SRC_SECOND, SRC_FALSE = 1, 2, 3
FULL_DET_PX = 80.0  # visible body height at which detection reaches det_base_p
MIN_VIS_W_PX = 8.0
MIN_SEEN_PX = 6.0  # less of a person than this (behind someone nearer) is never detected
BURST_LEN_S = (0.2, 1.0)
CONF_PERSON = (0.55, 0.92)
CONF_FALSE = (0.40, 0.70)
FALSE_W_PX, FALSE_H_PX = (60.0, 300.0), (80.0, 400.0)
SOURCES_KEPT = 8  # frames whose sources are remembered (the world reads the newest)

# keypoint conf ranges
KP_SEEN = (0.6, 0.95)
KP_FACE_BEHIND = (0.05, 0.35)
KP_EAR_BEHIND = (0.4, 0.85)
KP_FAR_SIDE = (0.35, 0.75)
KP_OUT = (0.0, 0.25)
_FACE, _EAR, _BODY = 0, 1, 2
# per keypoint (COCO order): kind and side (-1 the person's left, +1 right, 0 centre)
KP_KIND = (_FACE, _FACE, _FACE, _EAR, _EAR) + (_BODY,) * 12
KP_SIDE = (0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1)
N_KP = 17
# per-frame uniforms / normals per person: det, conf, swap, spare, 17 keypoint confs / 4 box edges, 34 keypoint coords
_NU, _NZ = 4 + N_KP, 4 + 2 * N_KP
_NF = 8 + 3 * N_KP  # false box: on?, x, w, h, at bottom?, y, conf, spare, then 17 x (x, y, conf)


class Snap(NamedTuple):
    """Truth at one capture time: the drone (x, y, z m, heading rad clockwise from north, body pitch deg + nose up),
    the person (head-top x, y, facing rad) and the second person (nan when nobody)."""

    x: float
    y: float
    z: float
    heading: float
    pitch_deg: float
    px: float
    py: float
    pf: float
    qx: float
    qy: float
    qf: float


def lerp_snap(a: Snap, b: Snap, w: float) -> Snap:
    """Linear between two physics ticks; headings the short way round."""
    v = 1.0 - w
    out = [v * p + w * q for p, q in zip(a, b)]
    for i in (3, 7, 10):
        d = (b[i] - a[i] + math.pi) % (2.0 * math.pi) - math.pi
        out[i] = a[i] + w * d
    if math.isnan(a.qx) or math.isnan(b.qx):  # the second person appears / leaves: take the nearer tick
        src = b if w >= 0.5 else a
        out[8], out[9], out[10] = src.qx, src.qy, src.qf
    return Snap(*out)


class Body:
    """One person's keypoints and silhouette in their own frame: (right, forward, up) m, the axis through the head
    top at the origin of the floor."""

    def __init__(self, height_m: float, shoulder_m: float):
        h, sw = float(height_m), float(shoulder_m)
        hs = 0.5 * sw
        self.height = h
        self.kp = ((0.0, 0.09, h - 0.135), (-0.032, 0.09, h - 0.105), (0.032, 0.09, h - 0.105),  # nose, eyes
                   (-0.075, -0.005, h - 0.12), (0.075, -0.005, h - 0.12),  # ears
                   (-hs, 0.0, 0.82 * h), (hs, 0.0, 0.82 * h),  # shoulders
                   (-(hs + 0.03), -0.02, 0.63 * h), (hs + 0.03, -0.02, 0.63 * h),  # elbows
                   (-(hs + 0.02), 0.0, 0.48 * h), (hs + 0.02, 0.0, 0.48 * h),  # wrists
                   (-0.25 * sw, 0.0, 0.53 * h), (0.25 * sw, 0.0, 0.53 * h),  # hips
                   (-0.22 * sw, 0.0, 0.29 * h), (0.22 * sw, 0.0, 0.29 * h),  # knees
                   (-0.2 * sw, -0.02, 0.04 * h), (0.2 * sw, -0.02, 0.04 * h))  # ankles
        # silhouette: (height, half width across, half depth), heights increasing. Across the arms at the shoulders
        # 1.2 x the shoulder width (0.48 m for 0.40: BehindCfg.body_width_m), 0.28 m deep (BODY_DEPTH_M)
        self.prof = ((0.0, 0.30 * sw, 0.13), (0.29 * h, 0.30 * sw, 0.08), (0.48 * h, 0.60 * sw, 0.14),
                     (0.82 * h, 0.60 * sw, 0.14), (min(0.82 * h + 0.03, h - 0.25), 0.42 * sw, 0.11),
                     (h - 0.23, 0.07, 0.07), (h - 0.115, 0.085, 0.10), (h - 0.02, 0.04, 0.05), (h, 0.0, 0.0))

    def width_at(self, z: float) -> tuple[float, float]:
        pr = self.prof
        if z <= pr[0][0]:
            return pr[0][1], pr[0][2]
        for (z0, a0, b0), (z1, a1, b1) in zip(pr, pr[1:]):
            if z <= z1:
                w = (z - z0) / (z1 - z0) if z1 > z0 else 1.0
                return a0 + w * (a1 - a0), b0 + w * (b1 - b0)
        return pr[-1][1], pr[-1][2]


class View(NamedTuple):
    """One person seen from one camera pose (no noise)."""

    visible: bool
    bbox: tuple[float, float, float, float]  # clipped to the frame
    cut: tuple[bool, bool, bool, bool]  # each edge lies on the frame border
    vis_h: float  # visible body height, px
    kps: tuple  # 17 x (col, row, in_frame)
    c_face: float  # facing . (person -> camera): +1 facing the camera, -1 back to it
    s_right: float  # right . (person -> camera): +1 the camera on their right
    dist_m: float  # horizontal camera -> head-top axis
    head: tuple[float, float, bool]  # head top (col, row, inside the frame)


class Camera3:
    """The true camera at one drone pose: project(X, Y, Z) world -> (col, row, depth)."""

    __slots__ = ("x", "y", "z", "sh", "ch", "cp", "sp", "fx", "fy", "cx", "cy")

    def __init__(self, snap: Snap, fx: float, mount_pitch_deg: float, fy: float | None = None, cx: float = CX,
                 cy: float = CY):
        p = math.radians(snap.pitch_deg + mount_pitch_deg)
        self.x, self.y, self.z = snap.x, snap.y, snap.z
        self.sh, self.ch = math.sin(snap.heading), math.cos(snap.heading)
        self.cp, self.sp = math.cos(p), math.sin(p)
        self.fx, self.fy, self.cx, self.cy = fx, (fx if fy is None else fy), cx, cy

    def project(self, wx: float, wy: float, wz: float) -> tuple[float, float, float]:
        dx, dy = wx - self.x, wy - self.y
        f = dx * self.sh + dy * self.ch
        r = dx * self.ch - dy * self.sh
        u = wz - self.z
        zc = f * self.cp + u * self.sp
        if zc < Z_MIN:
            return (self.cx + (FAR if r > 0 else -FAR), (FAR if u < 0 else -FAR), zc)
        return self.cx + self.fx * r / zc, self.cy + self.fy * (f * self.sp - u * self.cp) / zc, zc

    def height_at_row(self, f: float, row: float) -> float:
        """World height of the point imaged at `row` on a vertical line `f` m ahead (level-frame forward)."""
        k = (row - self.cy) / self.fy
        den = self.cp + k * self.sp
        if den <= 1e-6:
            return math.inf if k < 0 else -math.inf
        return self.z + f * (self.sp - k * self.cp) / den


def head_in_view(cam: Camera3, px: float, py: float, height_m: float) -> bool:
    c, r, zc = cam.project(px, py, height_m)
    return zc >= Z_MIN and 0.0 <= c <= W and 0.0 <= r < H


def view(cam: Camera3, body: Body, px: float, py: float, facing: float) -> View:
    """Where `body` (head-top axis at px, py, facing `facing`) appears, and what of it is visible."""
    sf, cf = math.sin(facing), math.cos(facing)  # forward (sf, cf), right (cf, -sf)
    tx, ty = cam.x - px, cam.y - py
    d = math.hypot(tx, ty)
    if d > 1e-6:
        tx, ty = tx / d, ty / d
    c_face, s_right = sf * tx + cf * ty, cf * tx - sf * ty
    hgt = body.height
    head = cam.project(px, py, hgt)
    head_in = head[2] >= Z_MIN and 0.0 <= head[0] <= W and 0.0 <= head[1] < H
    kps = []
    for lat, fwd, z in body.kp:
        c, r, zc = cam.project(px + lat * cf + fwd * sf, py - lat * sf + fwd * cf, z)
        kps.append((c, r, zc >= Z_MIN and 0.0 <= c <= W and 0.0 <= r <= H))
    # visible band of heights on the axis: rows 0 .. H
    fa = (px - cam.x) * cam.sh + (py - cam.y) * cam.ch  # forward distance of the axis
    none = View(False, (0.0, 0.0, 0.0, 0.0), (False,) * 4, 0.0, tuple(kps), c_face, s_right, d, head[:2] + (head_in,))
    if fa < 0.1:
        return none
    z_top = min(hgt, cam.height_at_row(fa, 0.0))
    z_bot = max(0.0, cam.height_at_row(fa, float(H)))
    if z_top <= z_bot:
        return none
    # the silhouette's edges at the knots inside the band and at the band's ends, across the line of sight
    nx, ny = -ty, tx  # horizontal, perpendicular to camera -> person, pointing to the camera's right
    rn, fn = cf * nx - sf * ny, sf * nx + cf * ny
    cols = []
    zs = [z for z, _, _ in body.prof if z_bot <= z <= z_top]
    if z_bot > 0.0:
        zs.append(z_bot)
    if z_top < hgt:
        zs.append(z_top)
    for z in zs:
        a, b = body.width_at(z)
        w = math.sqrt((a * rn) ** 2 + (b * fn) ** 2)
        cols.append(cam.project(px - w * nx, py - w * ny, z)[0])
        cols.append(cam.project(px + w * nx, py + w * ny, z)[0])
    x1, x2 = min(cols), max(cols)
    y1 = head[1] if head[2] >= Z_MIN else -FAR
    foot = cam.project(px, py, 0.0)
    y2 = foot[1] if foot[2] >= Z_MIN else FAR
    cut = (x1 <= 0.0, y1 <= 0.0, x2 >= W, y2 >= H)
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
    if x2 - x1 < MIN_VIS_W_PX or y2 - y1 < 2.0:
        return none
    return View(True, (x1, y1, x2, y2), cut, y2 - y1, tuple(kps), c_face, s_right, d, head[:2] + (head_in,))


def _kp_range(kind: int, side: int, c_face: float, s_right: float) -> tuple[float, float]:
    s_side = side * s_right  # > 0: this side of the person faces the camera
    if kind == _FACE:
        if side == 0:  # the nose: seen unless the back is turned
            return KP_SEEN if c_face > -0.3 else KP_FACE_BEHIND
        return KP_SEEN if (c_face > 0.3 or (c_face > -0.3 and s_side > 0.3)) else KP_FACE_BEHIND
    if kind == _EAR:
        if s_side > 0.5:
            return KP_SEEN
        if s_side < -0.5:
            return KP_FACE_BEHIND  # hidden behind the head
        return KP_EAR_BEHIND if c_face < 0.0 else KP_SEEN
    return KP_FAR_SIDE if s_side < -0.6 else KP_SEEN


def _bursts(rng: np.random.Generator, rate_hz: float, t_frames: list[float]) -> list[bool]:
    """Per-frame flags of a Poisson burst process (64 gaps and lengths always drawn: the stream never shifts)."""
    gaps = rng.exponential(1.0, 64)
    lens = rng.uniform(BURST_LEN_S[0], BURST_LEN_S[1], 64)
    drop = [False] * len(t_frames)
    if rate_hz <= 0.0 or not t_frames:
        return drop
    t0, t_end = t_frames[0], t_frames[-1]
    tt = np.asarray(t_frames)
    out = np.zeros(len(t_frames), bool)
    for start, ln in zip(t0 + np.cumsum(gaps) / rate_hz, lens):
        if start > t_end:
            break
        out |= (tt >= start) & (tt < start + ln)
    return out.tolist()


class FakePose:
    """The `detect` callable of kite.follow_behind.Perception for one episode."""

    def __init__(self, p, rng: np.random.Generator, rng_q: np.random.Generator, t_frames: list[float], body: Body,
                 q_body: Body | None = None):
        """p: profiles.BehindParams. t_frames: capture time of every frame (index = capture number)."""
        n = len(t_frames)
        self.p, self.body, self.q_body = p, body, q_body
        self.fx, self.mount = float(p.fx_px), float(p.mount_pitch_deg)
        self._u = rng.random((n, _NU))
        self._z = rng.standard_normal((n, _NZ))
        self._f = rng.random((n, _NF))
        self._drop = _bursts(rng, float(p.burst_rate_hz), t_frames)
        self._uq = rng_q.random((n, _NU))
        self._zq = rng_q.standard_normal((n, _NZ))
        self.sources: dict[int, list[int]] = {}
        self._pending: tuple[int, int, Snap] | None = None
        self.n_frames = 0  # frames detect() ran on
        self.n_person = 0  # ... with the person detected

    # -- the world's side
    def load(self, seq: int, j: int, snap: Snap) -> None:
        """The frame perception will run on next: its seq, capture number and truth."""
        self._pending = (seq, j, snap)

    def camera(self, snap: Snap) -> Camera3:
        return Camera3(snap, self.fx, self.mount)

    # -- Perception's side
    def __call__(self, image: np.ndarray) -> list[PersonDet]:
        if self._pending is None:
            return []
        seq, j, snap = self._pending
        dets, srcs = self.detect(j, snap)
        self.sources[seq] = srcs
        if len(self.sources) > SOURCES_KEPT:
            for k in sorted(self.sources)[:-SOURCES_KEPT]:
                del self.sources[k]
        return dets

    def detect(self, j: int, snap: Snap) -> tuple[list[PersonDet], list[int]]:
        """The detections of capture j (most confident first) and the source of each."""
        p = self.p
        self.n_frames += 1
        cam = self.camera(snap)
        vp = view(cam, self.body, snap.px, snap.py, snap.pf)
        vq = None
        if self.q_body is not None and not math.isnan(snap.qx):
            vq = view(cam, self.q_body, snap.qx, snap.qy, snap.qf)
            if not vq.visible:
                vq = None
        p_ok = vp.visible and not self._drop[j]
        hp = vp.vis_h
        hq = vq.vis_h if vq is not None else 0.0
        if vp.visible and vq is not None:  # the nearer body hides what of the farther one is behind it
            if vq.dist_m < vp.dist_m:
                hp = _unhidden_h(vp, vq)
            else:
                hq = _unhidden_h(vq, vp)
        out: list[tuple[PersonDet, int]] = []
        base = float(p.det_base_p)
        if p_ok and hp >= MIN_SEEN_PX:
            u = self._u[j].tolist()
            if u[0] < base * min(1.0, hp / FULL_DET_PX) ** 2:
                out.append((self._person_det(vp, u, self._z[j].tolist()), SRC_PERSON))
                self.n_person += 1
        if vq is not None and hq >= MIN_SEEN_PX:
            u = self._uq[j].tolist()
            if u[0] < base * min(1.0, hq / FULL_DET_PX) ** 2:
                out.append((self._person_det(vq, u, self._zq[j].tolist()), SRC_SECOND))
        f = self._f[j].tolist()
        if f[0] < float(p.false_box_p):
            out.append((_false_det(f), SRC_FALSE))
        out.sort(key=lambda d: -d[0].conf)
        return [d for d, _ in out], [s for _, s in out]

    def _person_det(self, v: View, u: list[float], z: list[float]) -> PersonDet:
        p = self.p
        bn, kn = float(p.box_noise_px), float(p.kp_noise_px)
        x1, y1, x2, y2 = v.bbox
        c1, c2, c3, c4 = v.cut
        x1 = x1 if c1 else min(float(W), max(0.0, x1 + bn * z[0]))
        y1 = y1 if c2 else min(float(H), max(0.0, y1 + bn * z[1]))
        x2 = x2 if c3 else min(float(W), max(0.0, x2 + bn * z[2]))
        y2 = y2 if c4 else min(float(H), max(0.0, y2 + bn * z[3]))
        if x2 < x1 + 2.0:
            x2 = min(float(W), x1 + 2.0)
        if y2 < y1 + 2.0:
            y2 = min(float(H), y1 + 2.0)
        rows = []
        cf, sr = v.c_face, v.s_right
        for i in range(N_KP):
            col, row, inside = v.kps[i]
            lo, hi = _kp_range(KP_KIND[i], KP_SIDE[i], cf, sr) if inside else KP_OUT
            col = max(-W, min(2.0 * W, col)) + kn * z[4 + 2 * i]
            row = max(-H, min(2.0 * H, row)) + kn * z[5 + 2 * i]
            rows.append((col, row, lo + u[4 + i] * (hi - lo)))
        if u[2] < float(p.swap_p):
            rows[5], rows[6] = rows[6], rows[5]
        conf = CONF_PERSON[0] + u[1] * (CONF_PERSON[1] - CONF_PERSON[0])
        return PersonDet("person", conf, (x1, y1, x2, y2), np.array(rows, np.float32))


def _unhidden_h(far: View, near: View) -> float:
    """How much of the far person stays visible (px of height) with the near person in front: all of it unless their
    head is in front of the near person's box, then only the strip above the near person's head."""
    col = far.head[0]
    x1, y1, x2, _ = near.bbox
    if not x1 <= col <= x2:
        return far.vis_h
    return max(0.0, min(far.vis_h, y1 - far.bbox[1]))


def _false_det(f: list[float]) -> PersonDet:
    w = FALSE_W_PX[0] + f[2] * (FALSE_W_PX[1] - FALSE_W_PX[0])
    h = FALSE_H_PX[0] + f[3] * (FALSE_H_PX[1] - FALSE_H_PX[0])
    x1 = f[1] * (W - w)
    y1 = (H - h) if f[4] < 0.5 else f[5] * (H - h)  # half of them touch the bottom edge
    conf = CONF_FALSE[0] + f[6] * (CONF_FALSE[1] - CONF_FALSE[0])
    kp = [(x1 + f[8 + 3 * i] * w, y1 + f[9 + 3 * i] * h, 0.5 * f[10 + 3 * i]) for i in range(N_KP)]
    return PersonDet("person", conf, (x1, y1, x1 + w, y1 + h), np.array(kp, np.float32))
