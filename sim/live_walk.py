"""Live walk on a real chest camera: smoothing, the stop zone, a depth fallback and the reveal screen.

Used by standin_detect.py for every chest frame (sim, file or live camera). Nothing here
writes frames to disk: the reveal buffer is JPEG bytes in RAM and goes away with the process.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque

STOP_M = 1.6            # something in the corridor this close -> critical ("Stop.")
DEPTH_STOP_M = 2.2      # 3D depth test (sim): a depth-only "obstacle" stops from here; its distance is the least
                        # certain (on the sim's plain bin depth read ~2.0 m from 2.3 m in to 1.2 m)
CORRIDOR_M = 0.7        # half-width of the walker's path, a bit wider than the earpiece's 0.55 + 0.25 margin
STOP_TTC = 0.8          # s; the earpiece goes critical at ttc <= 1.0 s, so a stop is sent as ttc 0.8
MEDIAN_N = 3            # frames in the distance median (each adds ~0.1 s of lag)
SPEED_WINDOW = 1.0      # s of median distances used for closing speed
BANNER_HOLD = 1.5       # s the STOP banner stays up after the last critical frame
REPLAY_S = 20.0
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_SIZE = (336, 252)  # w, h (multiples of 14); ~18 ms on an M5 with mps
DEPTH_BAND = 0.2        # band test (live cameras): centre band, +-20 % of the width either side of the middle
DEPTH_FILL = 0.08       # ... fraction of the band that must be close and not floor
GROUND_EPS = 0.2        # m: depth points lower than this are ground, curb steps included
HEAD_CLEAR = 2.0        # m: higher than this the walker passes under
DEPTH_CORRIDOR = 0.4    # m half-width for depth points: body and margin (Moss's walls run ~0.5 m off the path)
DEPTH_MIN_FRAC = 0.005  # fraction of the depth map that must be above-ground mass in the corridor
SCALE_PCT = 85          # percentile of depth / flat-floor depth below the horizon that sets the depth scale
PLANE_ITERS, PLANE_TOL, PLANE_MIN_UP, PLANE_MIN_IN = 80, 0.06, 0.8, 0.3   # ground_plane RANSAC


# ------------------------------------------------------------------ smoothing and the stop zone
class RangeFilter:
    """Per track id: median of the last MEDIAN_N distances, closing speed from a least-squares
    fit of those medians over SPEED_WINDOW (positive = getting closer)."""

    def __init__(self):
        self.raw: dict[str, deque] = {}
        self.med: dict[str, deque] = {}
        self.seen: dict[str, float] = {}

    def update(self, oid: str, t: float, d: float) -> tuple[float, float | None]:
        raw = self.raw.setdefault(oid, deque(maxlen=MEDIAN_N))
        med = self.med.setdefault(oid, deque(maxlen=60))
        raw.append(d)
        m = statistics.median(raw)
        med.append((t, m))
        self.seen[oid] = t
        for k in [k for k, s in self.seen.items() if t - s > 2.0]:     # forget lost tracks
            del self.raw[k], self.med[k], self.seen[k]
        pts = [(tt, dd) for tt, dd in med if t - tt <= SPEED_WINDOW]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < 0.3:
            return m, None
        mt = sum(p[0] for p in pts) / len(pts)
        md = sum(p[1] for p in pts) / len(pts)
        den = sum((p[0] - mt) ** 2 for p in pts)
        slope = sum((p[0] - mt) * (p[1] - md) for p in pts) / den
        return m, max(-5.0, min(10.0, -slope))


def in_corridor(bearing: float, distance: float) -> bool:
    b = math.radians(bearing)
    return distance * math.cos(b) > 0 and abs(distance * math.sin(b)) <= CORRIDOR_M


def is_stop(bearing: float, distance: float, closing: float | None, stop_m: float = STOP_M) -> bool:
    """Spec 5.3: stops are immediate for something large and close. Close in the corridor,
    or approaching fast enough that contact is under a second away."""
    if not in_corridor(bearing, distance):
        return False
    return distance <= stop_m or (closing is not None and closing > 0.3 and distance / closing <= 1.0)


def chest_step(objs: list[dict], filt: RangeFilter, t: float, stop_m: float = STOP_M,
               depth_stop_m: float | None = None) -> list[dict]:
    """Raw objects -> smoothed objects with closing_speed and a `stop` flag (not sent).
    A stop is sent with a closing speed that puts the earpiece's time-to-contact at STOP_TTC,
    so its sonar goes critical and says "Stop. Person ahead." at stop_m instead of 0.75 m.
    depth_stop_m, when given, is the stop range for depth-only "obstacle"s (DEPTH_STOP_M with the 3D depth test)."""
    out = []
    for o in objs:
        d, v = filt.update(o["id"], t, o["distance"])
        stop = is_stop(o["bearing"], d, v, max(stop_m, depth_stop_m) if depth_stop_m and o["kind"] == "obstacle"
                       else stop_m)
        if stop:
            v = max(v or 0.0, d / STOP_TTC)
        out.append({**o, "distance": round(d, 2), "closing_speed": None if v is None else round(v, 2),
                    "stop": stop})
    return out


def to_event_objects(objs: list[dict]) -> list[dict]:
    return [{k: v for k, v in o.items() if k not in ("cut_off", "stop") and v is not None} for o in objs]


# ------------------------------------------------------------------ depth fallback (class-agnostic)
def floor_depth(rows: int, fx: float, cam_height: float, pitch: float):
    """Per row of a depth map: the z-depth the floor would have there (inf at or above the horizon)."""
    import numpy as np
    off = np.arange(rows) - rows / 2
    a = math.radians(pitch) + np.arctan(off / fx)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(a > 0.02, cam_height / np.sin(a) * np.cos(np.arctan(off / fx)), np.inf)
    return z


def ground_scale(depth, floor) -> float:
    """Metric depth models misjudge scale outdoors (sim frames: 0.76x to 1.48x). Along a ray below the horizon the
    ground is the farthest thing it can hit, so a high percentile of depth / flat-floor depth there is the scale."""
    import numpy as np
    floor = np.broadcast_to(floor, depth.shape)
    m = np.isfinite(floor) & (floor < 10.0)
    if not m.any():
        return 1.0
    r = (depth / np.where(m, floor, 1.0))[m]
    return float(np.clip(np.percentile(r, SCALE_PCT), 0.5, 2.0))


def ground_plane(pts, n0, d0: float, seed: int = 0):
    """RANSAC plane n.P = d through 3D points (camera frame: x right, y down, z forward) that should mostly be ground.
    The plane must face within ~37 degrees of the configured down n0, so a wall can't win. Returns (n, d), or the
    configured (n0, d0) when no plane holds PLANE_MIN_IN of the points (the floor is hidden, e.g. by a close bin).
    A phone on a shirt never sits at exactly its configured pitch: the live walk bench's level stock footage, read at
    15 degrees down, put the floor 0.2 to 0.5 m "up" and stopped the walker for nothing."""
    import numpy as np
    if len(pts) < 50:
        return n0, d0
    rng = np.random.default_rng(seed)
    best, best_in = None, 0
    for _ in range(PLANE_ITERS):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if n @ n0 < 0:
            n = -n
        if n @ n0 < PLANE_MIN_UP:
            continue
        d = float(n @ a)
        if d <= 0:
            continue
        inl = np.abs(pts @ n - d) < PLANE_TOL * d
        if inl.sum() > best_in:
            best, best_in = inl, int(inl.sum())
    if best is None or best_in < PLANE_MIN_IN * len(pts):
        return n0, d0
    q = pts[best]
    c = q.mean(axis=0)
    n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
    n = n if n @ n0 > 0 else -n
    return n, float(n @ c)


def side_walls(nonground, below, foot_lat):
    """Columns whose nearest non-ground thing is a wall beside the path, not in it. Each column's foot (the lowest
    non-ground row that looks at the ground) gives, where that ray meets the ground plane, the wall's true sideways
    offset, which a depth model gets wrong on a wall seen edge-on (it read a wall 0.5 m to the side as 1 m ahead).
    Columns cut off by the bottom edge have no foot; a run of them at the image's side belongs to the wall next to it."""
    import numpy as np
    h, w = nonground.shape
    nb = nonground & below
    has = nb.any(axis=0)
    y0 = h - 1 - np.argmax(nb[::-1], axis=0)
    cut = has & (y0 >= h - 1)
    lat = foot_lat[y0, np.arange(w)]
    # 0.85: the foot row is quantised at depth-map resolution; Moss's wall, 0.5 m off, read 0.40 at the image edge
    wall = has & ~cut & (np.abs(lat) > 0.85 * DEPTH_CORRIDOR)
    for cols in (range(w - 1, -1, -1), range(w)):           # from the right edge, then from the left
        run = []
        for u in cols:
            if cut[u]:
                run.append(u)
                continue
            if wall[u]:
                wall[run] = True
            break
    return wall


def band_obstacle(depth, fx: float, cam_height: float, pitch: float, max_d: float = 2.5) -> dict | None:
    """Live cameras: one obstacle when something that is not floor and is closer than max_d fills DEPTH_FILL of the
    centre band. fx is in depth-map pixels. Loose on purpose: it passes the live walk bench (37/40, as without depth),
    where center_obstacle, which trusts the camera's pitch and height, stopped for walls and floor in 25 of 40 runs."""
    import numpy as np
    h, w = depth.shape
    lo, hi = int(w * (0.5 - DEPTH_BAND)), int(w * (0.5 + DEPTH_BAND))
    band = depth[:, lo:hi]
    close = (band < 0.75 * floor_depth(h, fx, cam_height, pitch)[:, None]) & (band < max_d)
    if close.mean() < DEPTH_FILL:
        return None
    cx = lo + float(np.nonzero(close)[1].mean())
    return {"id": "depth", "kind": "obstacle", "bearing": round(math.degrees(math.atan((cx - w / 2) / fx)), 1),
            "distance": round(float(np.percentile(band[close], 20)), 2), "confidence": 0.6, "cut_off": False}


def center_obstacle(depth, fx: float, cam_height: float, pitch: float, max_d: float = 2.5,
                    fy: float | None = None) -> dict | None:
    """Metric depth map (h, w) -> one obstacle when above-ground mass sits in the walker's corridor within max_d.
    Every pixel goes to 3D; the ground plane is fitted to the lower image (ground_plane; the camera's height and pitch
    are the fallback), which also fixes the depth scale (sim frames: 0.76x to 1.48x). Ground and curb steps (under
    GROUND_EPS), things over HEAD_CLEAR and walls beside the path (side_walls) drop out; what is left within
    DEPTH_CORRIDOR of the path and closer than max_d must fill DEPTH_MIN_FRAC of the map. fx and fy are in depth-map
    pixels; fy defaults to fx, and they differ when a 16:9 frame (the live camera) is squeezed into the 4:3 map."""
    import numpy as np
    h, w = depth.shape
    p = math.radians(pitch)
    fy = fy or fx
    rays = np.stack(np.broadcast_arrays(((np.arange(w) - w / 2) / fx)[None, :],
                                        ((np.arange(h) - h / 2) / fy)[:, None], 1.0), axis=-1)   # (h, w, 3), z = 1
    n0 = np.array([0.0, math.cos(p), math.sin(p)])                 # configured "down" in the camera frame
    d0 = cam_height * ground_scale(depth, floor_depth(h, fy, cam_height, pitch)[:, None])
    pts = rays * depth[..., None]
    cand = (rays @ n0 > 0.05) & (np.arange(h)[:, None] > h * 0.55)  # well below the horizon, lower image
    sample = pts[cand][:: max(1, int(cand.sum()) // 3000)]
    n, d = ground_plane(sample, n0, d0)
    pts = pts * (cam_height / d)                                    # metric: the ground plane sits cam_height down
    f = np.array([0.0, 0.0, 1.0]) - n[2] * n
    f /= np.linalg.norm(f)
    lat_axis = np.cross(n, f)
    height = cam_height - pts @ n
    fwd, lat = pts @ f, pts @ lat_axis
    down = rays @ n
    below = down > 0.02
    with np.errstate(divide="ignore", invalid="ignore"):
        foot_lat = np.where(below, cam_height / down * (rays @ lat_axis), np.inf)
    nonground = height >= GROUND_EPS
    wall = side_walls(nonground, below, foot_lat)
    mass = nonground & (height < HEAD_CLEAR) & (fwd > 0) & (fwd <= max_d) & (np.abs(lat) <= DEPTH_CORRIDOR) \
        & ~wall[None, :]
    if mass.mean() < DEPTH_MIN_FRAC:
        return None
    dist = float(np.percentile(fwd[mass], 20))
    b = math.degrees(math.atan2(float(np.median(lat[mass])), dist))
    return {"id": "depth", "kind": "obstacle", "bearing": round(b, 1), "distance": round(dist, 2), "confidence": 0.6,
            "cut_off": False}


class DepthFallback:
    """Depth Anything V2 Small (metric, indoor) at 336x252: catches walls, doors and anything
    YOLO has no class for. Weights come from `--download`; never fetched at runtime."""

    def __init__(self, device: str, ground_3d: bool = False):
        import torch
        from transformers import AutoModelForDepthEstimation
        self.torch, self.device = torch, device
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, local_files_only=True).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.ground_3d = ground_3d   # center_obstacle (needs a known camera pose: sim frames), else band_obstacle
        self.prev = False       # 3D: 2-frame onset, one frame's depth noise (a wall edge) must not stop the walker

    def __call__(self, img, fx: float, cam_height: float, pitch: float, objs: list[dict]) -> dict | None:
        import cv2
        x = cv2.resize(img, DEPTH_SIZE)[:, :, ::-1].copy()
        t = self.torch.from_numpy(x).to(self.device).permute(2, 0, 1)[None].float() / 255
        with self.torch.no_grad():
            depth = self.model(pixel_values=(t - self.mean) / self.std).predicted_depth[0].cpu().numpy()
        dfx = fx * depth.shape[1] / img.shape[1]
        if self.ground_3d:
            o = center_obstacle(depth, dfx, cam_height, pitch, fy=fx * depth.shape[0] / img.shape[0])
            onset, self.prev = self.prev, o is not None
            if not onset:
                return None
        else:
            o = band_obstacle(depth, dfx, cam_height, pitch)
        if o is None:
            return None
        # YOLO already has it: one sound per thing
        if any(abs(p["bearing"] - o["bearing"]) < 20 and p["distance"] < o["distance"] + 0.7 for p in objs):
            return None
        return o


def download_depth() -> None:
    from transformers import AutoModelForDepthEstimation
    AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL)
    print(f"cached {DEPTH_MODEL}")


# ------------------------------------------------------------------ reveal screen
def blur_heads(img, boxes, objs: list[dict], fx: float):
    """Pixelate the head of every person box: the top of the box, sized from the distance
    (a head is ~0.3 m, so fx * 0.3 / d px). No face model; works for profiles and backs too."""
    import cv2
    for (x1, y1, x2, _y2, *_), o in zip(boxes, objs):
        if o["kind"] != "person":
            continue
        side = int(min(x2 - x1, fx * 0.3 / max(o["distance"], 0.3)))
        x, y = max(0, int((x1 + x2 - side) / 2)), max(0, int(y1))
        roi = img[y:y + side, x:x + side]
        if roi.size:
            img[y:y + side, x:x + side] = cv2.resize(cv2.resize(roi, (6, 6)), roi.shape[1::-1],
                                                     interpolation=cv2.INTER_NEAREST)
    return img


def stop_text(objs: list[dict]) -> str | None:
    s = [o for o in objs if o.get("stop")]
    if not s:
        return None
    o = min(s, key=lambda o: o["distance"])
    return f"STOP - {o['kind']} {o['distance']:.1f} m"


def draw_banner(img, text: str):
    import cv2
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 90), (w, h), (0, 0, 200), -1)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
    cv2.putText(img, text, ((w - tw) // 2, h - 30), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    return img


class Reveal:
    """cv2 window: live chest view, or (key r) a replay of the last REPLAY_S seconds from RAM.
    Keys: r replay, l live, q / Esc quit."""

    WIN = "Kite chest view"

    def __init__(self):
        self.buf: deque = deque()           # (t, jpeg bytes)
        self.banner_until, self.banner = -1.0, ""
        self.replay_t0: float | None = None
        self.replay_from = self.replay_to = 0.0

    def frame(self, img, objs: list[dict], t: float, fps: float) -> bool:
        """Annotated live frame in; shows live or replay. Returns False when the user quits."""
        import cv2
        text = stop_text(objs)
        if text:
            self.banner, self.banner_until = text, t + BANNER_HOLD
        if t < self.banner_until:
            draw_banner(img, self.banner)
        cv2.putText(img, f"{fps:4.1f} fps", (img.shape[1] - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        self.buf.append((t, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()))
        while self.buf and t - self.buf[0][0] > REPLAY_S:
            self.buf.popleft()
        show = img
        if self.replay_t0 is not None:
            want = self.replay_from + (time.monotonic() - self.replay_t0)
            past = [jpg for tt, jpg in self.buf if tt <= want]
            if want >= self.replay_to or not past:
                self.replay_t0 = None
            else:
                import numpy as np
                show = cv2.imdecode(np.frombuffer(past[-1], np.uint8), cv2.IMREAD_COLOR)
                cv2.putText(show, f"REPLAY  {want - self.replay_to:+.1f} s", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                            (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.WIN, show)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("r") and self.buf:
            self.replay_t0, self.replay_from, self.replay_to = time.monotonic(), self.buf[0][0], t
        elif k == ord("l"):
            self.replay_t0 = None
        return k not in (ord("q"), 27)
