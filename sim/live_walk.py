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
DEPTH_STOP_M = 2.2      # ... for a depth-only "obstacle": its distance is the least certain (on the sim's plain bin,
                        # depth read ~2.0 m from 2.3 m in to 1.2 m), so it gets a longer stop zone
CORRIDOR_M = 0.7        # half-width of the walker's path, a bit wider than the earpiece's 0.55 + 0.25 margin
STOP_TTC = 0.8          # s; the earpiece goes critical at ttc <= 1.0 s, so a stop is sent as ttc 0.8
MEDIAN_N = 3            # frames in the distance median (each adds ~0.1 s of lag)
SPEED_WINDOW = 1.0      # s of median distances used for closing speed
BANNER_HOLD = 1.5       # s the STOP banner stays up after the last critical frame
REPLAY_S = 20.0
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_SIZE = (336, 252)  # w, h (multiples of 14); ~18 ms on an M5 with mps
GROUND_EPS = 0.2        # m: depth points lower than this are ground, curb steps included
HEAD_CLEAR = 2.0        # m: higher than this the walker passes under
DEPTH_CORRIDOR = 0.4    # m half-width for depth points: body and margin (Moss's walls run ~0.5 m off the path)
DEPTH_MIN_FRAC = 0.005  # fraction of the depth map that must be above-ground mass in the corridor
SCALE_PCT = 85          # percentile of depth / flat-floor depth below the horizon that sets the depth scale


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


def chest_step(objs: list[dict], filt: RangeFilter, t: float, stop_m: float = STOP_M) -> list[dict]:
    """Raw objects -> smoothed objects with closing_speed and a `stop` flag (not sent).
    A stop is sent with a closing speed that puts the earpiece's time-to-contact at STOP_TTC,
    so its sonar goes critical and says "Stop. Person ahead." at stop_m instead of 0.75 m."""
    out = []
    for o in objs:
        d, v = filt.update(o["id"], t, o["distance"])
        stop = is_stop(o["bearing"], d, v, max(stop_m, DEPTH_STOP_M) if o["kind"] == "obstacle" else stop_m)
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


def side_walls(nonground, below, a, xc, cam_height: float):
    """Columns whose nearest non-ground thing is a wall beside the path, not in it. Each column's foot (the lowest
    non-ground row below the horizon) gives its distance on flat ground and so its true sideways offset, which a
    depth model gets wrong on a wall seen edge-on (it read a wall 0.5 m to the side as 1 m ahead). Columns cut off
    by the bottom edge have no foot; a run of them at the image's side belongs to the wall next to it."""
    import numpy as np
    h, w = nonground.shape
    nb = nonground & below
    has = nb.any(axis=0)
    y0 = h - 1 - np.argmax(nb[::-1], axis=0)
    cut = has & (y0 >= h - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        foot = cam_height / np.tan(a[y0])
    # 0.85: the foot row is quantised at depth-map resolution; Moss's wall, 0.5 m off, read 0.40 at the image edge
    wall = has & ~cut & (np.abs(foot * xc) > 0.85 * DEPTH_CORRIDOR)
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


def center_obstacle(depth, fx: float, cam_height: float, pitch: float, max_d: float = 2.5) -> dict | None:
    """Metric depth map (h, w) -> one obstacle when above-ground mass sits in the walker's corridor within max_d.
    Every pixel goes to 3D with the camera's height and pitch (scale fixed by ground_scale). Ground and curb steps
    (under GROUND_EPS), things over HEAD_CLEAR and walls beside the path (side_walls) drop out; what is left within
    DEPTH_CORRIDOR of the path and closer than max_d must fill DEPTH_MIN_FRAC of the map. fx in depth-map pixels."""
    import numpy as np
    h, w = depth.shape
    p = math.radians(pitch)
    yc = ((np.arange(h) - h / 2) / fx)[:, None]
    xc = (np.arange(w) - w / 2) / fx
    a = p + np.arctan(yc[:, 0])
    z = depth / ground_scale(depth, floor_depth(h, fx, cam_height, pitch)[:, None])
    height = cam_height - z * (yc * math.cos(p) + math.sin(p))
    fwd = z * (math.cos(p) - yc * math.sin(p))
    lat = z * xc[None, :]
    nonground = height >= GROUND_EPS
    wall = side_walls(nonground, (a > 0.02)[:, None], a, xc, cam_height)
    mass = nonground & (height < HEAD_CLEAR) & (fwd > 0) & (fwd <= max_d) & (np.abs(lat) <= DEPTH_CORRIDOR) \
        & ~wall[None, :]
    if mass.mean() < DEPTH_MIN_FRAC:
        return None
    d = float(np.percentile(fwd[mass], 20))
    b = math.degrees(math.atan2(float(np.median(lat[mass])), d))
    return {"id": "depth", "kind": "obstacle", "bearing": round(b, 1), "distance": round(d, 2), "confidence": 0.6,
            "cut_off": False}


class DepthFallback:
    """Depth Anything V2 Small (metric, indoor) at 336x252: catches walls, doors and anything
    YOLO has no class for. Weights come from `--download`; never fetched at runtime."""

    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForDepthEstimation
        self.torch, self.device = torch, device
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, local_files_only=True).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.prev = False       # 2-frame onset: one frame's depth noise (a wall edge) must not stop the walker

    def __call__(self, img, fx: float, cam_height: float, pitch: float, objs: list[dict]) -> dict | None:
        import cv2
        x = cv2.resize(img, DEPTH_SIZE)[:, :, ::-1].copy()
        t = self.torch.from_numpy(x).to(self.device).permute(2, 0, 1)[None].float() / 255
        with self.torch.no_grad():
            depth = self.model(pixel_values=(t - self.mean) / self.std).predicted_depth[0].cpu().numpy()
        o = center_obstacle(depth, fx * depth.shape[1] / img.shape[1], cam_height, pitch)
        onset, self.prev = self.prev, o is not None
        if o is None or not onset:
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
