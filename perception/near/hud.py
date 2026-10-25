"""Review HUD: the camera frame with boxes, priorities and the corridor on the ground, side by side with the
anchored depth map in Turbo (Google's improved rainbow colormap), plus a top-down radar strip.

Two DepthLab ideas, adapted: the false-colour depth view, and an oriented reticle: a ring drawn on the ground
where the walker's line meets the P1 boundary (2 m) and the P2 boundary (5 m), squashed by the ground's
foreshortening so it lies on the pavement, and coloured by the priority of whatever is ahead.
Never used on live cameras by default (frames are not stored; --hud is for files).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import horizon_row
from .intrinsics import CameraProfile
from .system import NearFrame

BGR = {1: (40, 40, 235), 2: (0, 190, 255), 3: (230, 200, 60), None: (150, 150, 150)}
DEPTH_MAX = 12.0


def _ground_px(prof: CameraProfile, forward: float, lateral: float, cam_h: float, pitch: float):
    p = math.radians(pitch)
    zc = forward * math.cos(p) + cam_h * math.sin(p)
    yc = -forward * math.sin(p) + cam_h * math.cos(p)
    if zc <= 0.05:
        return None
    pt = np.array([[prof.fx * lateral / zc + prof.cx, prof.fy * yc / zc + prof.cy]])
    if prof.has_distortion:
        pt = prof.distort_points(pt)
    return int(round(pt[0, 0])), int(round(pt[0, 1]))


def corridor_lines(img, prof, cam_h, pitch, half=0.5, color=(255, 255, 255)):
    for side in (-half, half):
        pts = [_ground_px(prof, f, side, cam_h, pitch) for f in np.linspace(0.6, 12.0, 30)]
        pts = [q for q in pts if q is not None and 0 <= q[1] < prof.height * 2]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, color, 1, cv2.LINE_AA)


def reticle(img, prof, cam_h, pitch, forward, color, label):
    c = _ground_px(prof, forward, 0.0, cam_h, pitch)
    e = _ground_px(prof, forward, 0.5, cam_h, pitch)
    n = _ground_px(prof, forward + 0.5, 0.0, cam_h, pitch)
    if c is None or e is None or n is None or not (0 <= c[1] < prof.height):
        return
    ax, ay = max(2, abs(e[0] - c[0])), max(1, abs(c[1] - n[1]))
    cv2.ellipse(img, c, (ax, ay), 0, 0, 360, color, 2, cv2.LINE_AA)
    cv2.line(img, (c[0] - ax // 3, c[1]), (c[0] + ax // 3, c[1]), color, 1, cv2.LINE_AA)
    cv2.putText(img, label, (c[0] + ax + 4, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def depth_view(nf: NearFrame, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if nf.depth is None:
        v = np.zeros((h, w, 3), np.uint8)
        cv2.putText(v, "no depth (--depth off)", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return v
    z = np.clip(nf.depth.z / DEPTH_MAX, 0, 1)
    turbo = cv2.applyColorMap(((1 - z) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)   # near = red, far = blue
    turbo = (turbo * (0.35 + 0.65 * nf.depth.conf[..., None])).astype(np.uint8)       # dim what we doubt
    v = cv2.resize(turbo, (w, h), interpolation=cv2.INTER_NEAREST)
    a = nf.depth.anchor
    txt = f"{nf.depth.backend or 'depth'}  x{a.scale:.2f}  pitch {a.pitch:.1f}  ground {100 * a.inlier_frac:.0f}%"
    cv2.putText(v, txt + ("" if a.ok else "  (not anchored)"), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return v


def radar_strip(nf: NearFrame, width: int, height: int = 150, far: float = 8.0) -> np.ndarray:
    s = np.full((height, width, 3), 18, np.uint8)
    px_m = (width - 40) / far
    y0 = height // 2
    for d, name in ((2.0, "2 m"), (5.0, "5 m")):
        x = 20 + int(d * px_m)
        cv2.line(s, (x, 10), (x, height - 10), (70, 70, 70), 1)
        cv2.putText(s, name, (x + 3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    lat_px = (height - 30) / 4.0
    for side in (-0.5, 0.5):
        y = y0 + int(side * lat_px)
        cv2.line(s, (20, y), (width - 20, y), (90, 90, 90), 1)
    cv2.circle(s, (14, y0), 5, (255, 255, 255), -1)
    for o in nf.objects:
        if o["distance_m"] is None or o["distance_m"] > far:
            continue
        x = 20 + int(o["distance_m"] * px_m)
        y = y0 + int(max(-2, min(2, o["lateral_m"])) * lat_px)
        cv2.circle(s, (x, y), 7 if o["priority"] else 4, BGR[o["priority"]], -1)
        if o["priority"]:
            cv2.putText(s, f"P{o['priority']}", (x - 8, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BGR[o["priority"]], 1)
    return s


def draw(frame: np.ndarray, nf: NearFrame, prof: CameraProfile, dets, cam_h: float) -> np.ndarray:
    h, w = frame.shape[:2]
    prof = prof.scaled(w, h)
    img = frame.copy()
    hz = horizon_row(prof, nf.pitch)
    if 0 <= hz < h:
        cv2.line(img, (0, int(hz)), (w, int(hz)), (120, 120, 120), 1, cv2.LINE_AA)
    corridor_lines(img, prof, cam_h, nf.pitch)
    top = nf.top["priority"] if nf.top else None
    reticle(img, prof, cam_h, nf.pitch, 2.0, BGR[1 if top == 1 else None], "2 m")
    reticle(img, prof, cam_h, nf.pitch, 5.0, BGR[2 if top == 2 else None], "5 m")
    by_box = {}
    for d, o in zip(dets, [o for o in nf.objects if not o["id"].startswith("occ:")]):
        by_box[id(d)] = o
    for d in dets:
        o = by_box.get(id(d))
        p = o["priority"] if o else None
        col = BGR[p]
        cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, 3 if p == 1 else 2)
        if o:
            txt = f"{'P' + str(p) if p else '--'} {o['kind']} {o['distance_m'] or 0:.1f}m"
            if o.get("ttc_s") is not None:
                txt += f" ttc {o['ttc_s']:.1f}s"
            ty = int(d.y1) - 6 if d.y1 > 60 else int(d.y1) + 58          # under the banner: inside the box
            cv2.putText(img, txt, (int(d.x1) + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    for o in nf.objects:
        if o["id"].startswith("occ:") and o["priority"]:
            q = _ground_px(prof, o["distance_m"], o["lateral_m"], cam_h, nf.pitch)
            if q and 0 <= q[1] < h:
                cv2.drawMarker(img, q, BGR[o["priority"]], cv2.MARKER_TILTED_CROSS, 22, 3)
                cv2.putText(img, f"P{o['priority']} {o['kind']} {o['distance_m']:.1f}m", (q[0] + 12, q[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, BGR[o["priority"]], 2, cv2.LINE_AA)
    banner = f"P{top}  {nf.top['kind']}" if top else "clear ahead"
    cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(img, f"t {nf.t:5.1f}s   {banner}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BGR[top], 2,
                cv2.LINE_AA)
    right = depth_view(nf, (w, h))
    corridor_lines(right, prof, cam_h, nf.pitch, color=(255, 255, 255))
    top_row = cv2.hconcat([img, right])
    return cv2.vconcat([top_row, radar_strip(nf, top_row.shape[1])])
