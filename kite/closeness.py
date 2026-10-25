"""Camera-only closeness: what is directly ahead, as a priority (1 close, 2 mid, 3 far), never a raw distance.

    python -m kite.closeness --source webcam:0 [--show] [--jsonl out.jsonl] [--udp host:port]
    python -m kite.closeness --source tello --show
    python -m kite.closeness --source clip.mp4 --fast

priority 1: something in the walking corridor closer than ~2 m       (Arthur: highest beep)
priority 2: ~2 to 5 m
priority 3: ~5 to MAX_RANGE_M (12 m)
null:       nothing in the corridor within MAX_RANGE_M (no alert)

Distance per YOLO box, the more conservative (smaller) of: pinhole from a known class height, the ground-plane
distance of the box bottom (camera height + pitch), and, when the box is cut off, the head row / shoulder width.
Only boxes overlapping a corridor CORRIDOR_M either side of the walking line count; any class counts (people too).
Wide things escalate earlier: both thresholds scale by up to SIZE_MAX_K with the object's width in metres.
Escalation is immediate; relaxing waits RELAX_S and needs HYST_M of margin past the boundary, so it doesn't flicker.

priority_for() is pure (no numpy, no YOLO): boxes in, priority out, so it can run on the UNO Q's Linux side.
Intrinsics: webcams default to a Logitech C920 (70.42 deg horizontal FOV), fx from the real frame width at runtime;
tello / files use kite.sources' defaults (kite/camera.py); --hfov overrides any of them.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "yolo11n.pt"

C920_HFOV_DEG = 70.42       # Logitech C920/C922: 78 deg diagonal, 70.42 horizontal at 16:9 (Brio at 90 deg: --hfov 82)
CAM_HEIGHT_M = 1.3          # chest-worn webcam
CORRIDOR_M = 0.5            # half-width of the walking corridor
NEAR_M, FAR_M = 2.0, 5.0    # priority 1 / 2 boundaries for a person-sized object
MAX_RANGE_M = 12.0          # farther than this: no alert at all
SIZE_REF_M, SIZE_MAX_K = 1.0, 1.5   # width (m) at which thresholds start growing, and the cap on that growth
HYST_M = 0.3                # relaxing must clear a boundary by this much (x size factor)
RELAX_S = 0.6               # a calmer level must hold this long before we switch to it
HEARTBEAT_S = 1.0
EDGE_PX = 16                # box this close to the frame edge counts as cut off (YOLO stops a few px short)
GROUND_MIN_RAD = 0.015      # box bottom must be this far below the horizon to trust the ground plane
PERSON_W = 0.45             # m shoulder width: distance when a person's head and feet are both out of frame
MIN_D = 0.3

# typical heights (m); anything else: DEFAULT_H. ponytail: COCO only; walls/doors need --depth
CLASS_H = {"person": 1.7, "bicycle": 1.1, "car": 1.5, "motorcycle": 1.2, "bus": 3.0, "truck": 3.0,
           "fire hydrant": 0.8, "stop sign": 2.2, "parking meter": 1.3, "bench": 0.9, "chair": 0.9, "dog": 0.6,
           "potted plant": 0.8, "suitcase": 0.7, "backpack": 0.5, "couch": 0.9, "dining table": 0.75,
           "traffic light": 3.0, "skateboard": 0.15, "umbrella": 1.0, "bird": 0.3, "cat": 0.3}
DEFAULT_H = 1.0
IGNORE = {"traffic light"}   # overhead: you walk under it


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def box_distance(box: tuple, cam: Intrinsics, cam_height: float, pitch_deg: float) -> float:
    """(x1, y1, x2, y2, label) -> conservative metres to the object. pitch + = camera tilted down."""
    x1, y1, x2, y2, label = box[:5]
    real_h = CLASS_H.get(label, DEFAULT_H)
    p = math.radians(pitch_deg)
    below = p + math.atan((y2 - cam.cy) / cam.fy)
    d = real_h * cam.fy / max(y2 - y1, 1.0)                    # pinhole from the class height
    if below > GROUND_MIN_RAD:
        d = min(d, cam_height / math.tan(below))               # foot point on flat ground
    top_cut, bottom_cut = y1 <= EDGE_PX, y2 >= cam.height - EDGE_PX
    up = -(p + math.atan((y1 - cam.cy) / cam.fy))
    if not top_cut and real_h - cam_height > 0.2 and up > GROUND_MIN_RAD:
        d = min(d, (real_h - cam_height) / math.tan(up))       # head row, for things taller than the camera
    if top_cut and bottom_cut:
        # fills the frame top to bottom: no row says how close. Shoulder width for people, else assume close.
        d = min(d, PERSON_W * cam.fx / max(x2 - x1, 1.0) if label == "person" else NEAR_M)
    return max(MIN_D, d)


def level(d: float, k: float, current: int | None = None) -> int | None:
    """Distance + size factor -> 1/2/3/None. Boundaries a calmer level than `current` would cross move out by HYST_M."""
    near, far, top = NEAR_M * k, FAR_M * k, MAX_RANGE_M
    m = HYST_M * k
    cur = current or 4
    near += m if cur <= 1 else 0
    far += m if cur <= 2 else 0
    top += m if cur <= 3 else 0
    return 1 if d < near else 2 if d < far else 3 if d < top else None


def priority_for(frame_dets: list[tuple], cam: Intrinsics, cam_height: float = CAM_HEIGHT_M, pitch_deg: float = 0.0,
                 current: int | None = None, corridor_m: float = CORRIDOR_M) -> dict:
    """frame_dets: [(x1, y1, x2, y2, label, ...)] in pixels -> {priority, label, est_m, bearing_deg} of the most
    urgent object in the corridor (priority None when there is none). `current`: the level now playing (hysteresis)."""
    best = {"priority": None, "label": None, "est_m": None, "bearing_deg": None}
    rank = (5, math.inf)
    for box in frame_dets:
        x1, _, x2, _, label = box[:5]
        if label in IGNORE:
            continue
        d = box_distance(box, cam, cam_height, pitch_deg)
        left, right = (x1 - cam.cx) / cam.fx * d, (x2 - cam.cx) / cam.fx * d   # lateral metres at that distance
        if right < -corridor_m or left > corridor_m:
            continue
        k = min(SIZE_MAX_K, max(1.0, (right - left) / SIZE_REF_M))
        pr = level(d, k, current)
        if pr is None or (pr, d) >= rank:
            continue
        rank = (pr, d)
        best = {"priority": pr, "label": label, "est_m": round(d, 1),
                "bearing_deg": round(math.degrees(math.atan(((x1 + x2) / 2 - cam.cx) / cam.fx)))}
    return best


class Steady:
    """Escalate at once; relax only after RELAX_S of calmer frames, to the most urgent level seen in that time."""

    def __init__(self):
        self.priority: int | None = None
        self.calm_since: float | None = None
        self.calm_worst = 5

    def step(self, raw: int | None, t: float) -> int | None:
        r, cur = raw or 4, self.priority or 4
        if r <= cur:
            self.priority, self.calm_since = raw, None
            return self.priority
        if self.calm_since is None:
            self.calm_since, self.calm_worst = t, r
        self.calm_worst = min(self.calm_worst, r)
        if t - self.calm_since >= RELAX_S:
            self.priority = None if self.calm_worst >= 4 else self.calm_worst
            self.calm_since = None
        return self.priority


# ------------------------------------------------------------------ runtime (YOLO, sources, outputs)
COLORS = {1: (0, 0, 255), 2: (0, 165, 255), 3: (0, 220, 0), None: (160, 160, 160)}


def draw(img, dets, cam: Intrinsics, out: dict, cam_height: float, pitch: float, fps: float):
    import cv2
    h = img.shape[0]
    for d in (1.0, 2.0, 5.0, MAX_RANGE_M):          # corridor edges on the ground at these distances
        a = math.atan(cam_height / d) - math.radians(pitch)
        y = int(cam.cy + cam.fy * math.tan(a))
        xs = [int(cam.cx + s * CORRIDOR_M / d * cam.fx) for s in (-1, 1)]
        if 0 <= y < h:
            cv2.line(img, (xs[0], y), (xs[1], y), (255, 255, 0), 1)
            cv2.putText(img, f"{d:g}m", (xs[1] + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    a0 = math.atan(cam_height / 0.8) - math.radians(pitch)
    y0 = min(h - 1, int(cam.cy + cam.fy * math.tan(a0)))
    a1 = math.atan(cam_height / MAX_RANGE_M) - math.radians(pitch)
    y1 = int(cam.cy + cam.fy * math.tan(a1))
    for s in (-1, 1):
        cv2.line(img, (int(cam.cx + s * CORRIDOR_M / 0.8 * cam.fx), y0),
                 (int(cam.cx + s * CORRIDOR_M / MAX_RANGE_M * cam.fx), y1), (255, 255, 0), 2)
    for x1, y1_, x2, y2, label, conf in dets:
        cv2.rectangle(img, (int(x1), int(y1_)), (int(x2), int(y2)), (200, 200, 200), 1)
        cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1_) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
    p = out["priority"]
    cv2.rectangle(img, (0, 0), (230, 110), COLORS[p], -1)
    cv2.putText(img, f"P{p}" if p else "--", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6)
    if p:
        cv2.putText(img, f"{out['label']} ~{out['est_m']}m", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[p], 2)
    cv2.putText(img, f"{fps:.1f} fps", (img.shape[1] - 110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def detect(model, frame, conf: float, device: str) -> list[tuple]:
    r = model.predict(frame, verbose=False, conf=conf, imgsz=640, device=device)[0]
    if r.boxes is None:
        return []
    return [(*xyxy, r.names[int(c)], cf) for xyxy, c, cf in
            zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="webcam:0", help="webcam:<index|name>, tello, a video file (kite.sources)")
    ap.add_argument("--hfov", type=float, help=f"horizontal FOV, deg (webcam default {C920_HFOV_DEG}: Logitech C920)")
    ap.add_argument("--cam-height", type=float, help=f"camera height, m (default {CAM_HEIGHT_M}, or the Tello's)")
    ap.add_argument("--pitch", type=float, default=0.0, help="camera tilt, deg, + = down")
    ap.add_argument("--corridor", type=float, default=CORRIDOR_M, help="corridor half-width, m")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--device", help="mps / cpu (default mps if available)")
    ap.add_argument("--depth", action="store_true", help="also Depth Anything (sim/live_walk.py) for walls/doors")
    ap.add_argument("--show", action="store_true", help="window: corridor, boxes, priority (q quits)")
    ap.add_argument("--jsonl", help="also append every output line here")
    ap.add_argument("--udp", help="host:port of the earpiece: send a `hazard` (kind obstacle) on priority 1/2")
    ap.add_argument("--fast", action="store_true", help="files: don't pace to real time")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = forever)")
    a = ap.parse_args(argv)

    import cv2
    import torch
    from ultralytics import YOLO

    from kite.sources import open_source, parse_spec

    if not Path(a.weights).is_file():
        sys.exit(f"missing {a.weights}; run: python sim/standin_detect.py --download")
    kind = parse_spec(a.source)[0]
    opts = {}
    if a.hfov or kind == "webcam":
        opts["hfov_deg"] = a.hfov or C920_HFOV_DEG
    if kind == "file":
        opts["realtime"] = not a.fast
    try:
        src = open_source(a.source, **opts)
    except (LookupError, RuntimeError, OSError, ValueError) as e:
        sys.exit(f"cannot open {a.source}: {e} (macOS: allow Camera for this terminal)")
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(a.weights)
    depth = None
    if a.depth:
        sys.path.insert(0, str(ROOT / "sim"))
        import live_walk
        depth = live_walk.DepthFallback(device)
    ear = None
    if a.udp:
        sys.path.insert(0, str(ROOT / "earpiece"))
        from earpiece.client import EarpieceClient
        host, _, port = a.udp.rpartition(":")
        ear = EarpieceClient(host or "127.0.0.1", int(port), source="closeness")
    jsonl = open(a.jsonl, "a") if a.jsonl else None
    import numpy as np
    model.predict(np.zeros((540, 960, 3), np.uint8), verbose=False, imgsz=640, device=device)  # mps warm-up, ~seconds

    steady, last_print, last_pr, n, t0 = Steady(), -math.inf, "init", 0, time.monotonic()
    fps, fps_t, fps_n = 0.0, time.monotonic(), 0

    def emit(rec: dict) -> None:
        line = json.dumps(rec, separators=(",", ":"))
        print(line, flush=True)
        if jsonl:
            jsonl.write(line + "\n")

    try:
        while not (a.seconds and time.monotonic() - t0 > a.seconds):
            ok, img = src.read()
            if not ok:
                break
            c = src.camera
            cam = Intrinsics(img.shape[1], img.shape[0], c.fx, c.fy, c.cx, c.cy)
            pose = src.pose or {}
            cam_h = a.cam_height or (pose.get("cam_height") if kind == "tello" else None) or CAM_HEIGHT_M
            dets = detect(model, img, a.conf, device)
            if depth is not None:
                o = depth(img, cam.fx, cam_h, a.pitch, [])
                if o:      # depth-only obstacle -> a synthetic box at its distance, 1 m wide, on the ground
                    d, b = o["distance"], math.radians(o["bearing"])
                    xc = cam.cx + cam.fx * math.tan(b)
                    yb = cam.cy + cam.fy * math.tan(math.atan(cam_h / d) + math.radians(a.pitch))
                    half = 0.5 / d * cam.fx
                    dets.append((xc - half, yb - DEFAULT_H / d * cam.fy, xc + half, yb, "obstacle", 0.6))
            t = time.monotonic() - t0
            raw = priority_for(dets, cam, cam_h, a.pitch, steady.priority, a.corridor)
            pr = steady.step(raw["priority"], t)
            out = raw if raw["priority"] == pr else {**raw, "priority": pr}
            if pr is None:
                out = {"priority": None, "label": None, "est_m": None, "bearing_deg": None}
            if pr != last_pr or t - last_print >= HEARTBEAT_S:
                emit({"t": round(t, 2), **out})
                if ear is not None and pr in (1, 2):
                    what = (out["label"] or "obstacle").capitalize()
                    ear.send({"type": "hazard", "kind": "obstacle", "key": "closeness",
                              "state": "stop" if pr == 1 else None,
                              "text": f"Stop. {what} ahead." if pr == 1 else f"{what} ahead.",
                              "bearing": out["bearing_deg"], "distance": out["est_m"], "ttl": 2.0,
                              "priority": "critical" if pr == 1 else "hazard"})
                last_pr, last_print = pr, t
            n += 1
            fps_n += 1
            if time.monotonic() - fps_t >= 2.0:
                fps, fps_t, fps_n = fps_n / (time.monotonic() - fps_t), time.monotonic(), 0
            if a.show:
                cv2.imshow("kite closeness", draw(img, dets, cam, out, cam_h, a.pitch, fps))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        if jsonl:
            jsonl.close()
        if a.show:
            cv2.destroyAllWindows()
        el = time.monotonic() - t0
        print(f"[closeness] {n} frames in {el:.1f} s = {n / max(el, 1e-9):.1f} fps on {device}, "
              f"camera {src.camera.source}", file=sys.stderr, flush=True)
    return 0


def _selfcheck() -> None:
    """python -c 'import kite.closeness as c; c._selfcheck()'"""
    f = 1280 / (2 * math.tan(math.radians(C920_HFOV_DEG / 2)))
    cam = Intrinsics(1280, 720, f, f, 640, 360)

    def person(d, x=0.0):       # a 1.7 m, 0.45 m wide person standing d m ahead, x m to the right, on flat ground
        yb = cam.cy + cam.fy * CAM_HEIGHT_M / d
        yt = cam.cy - cam.fy * (1.7 - CAM_HEIGHT_M) / d
        xc = cam.cx + cam.fx * x / d
        return (xc - 0.225 * cam.fx / d, max(0, yt), xc + 0.225 * cam.fx / d, min(720, yb), "person", 0.9)

    for d, want in ((1.5, 1), (3.0, 2), (7.0, 3), (20.0, None)):
        got = priority_for([person(d)], cam)
        assert got["priority"] == want, (d, got)
    assert abs(priority_for([person(3.0)], cam)["est_m"] - 3.0) < 0.3
    assert priority_for([person(3.0, x=2.0)], cam)["priority"] is None          # off to the side
    car = (cam.cx - 1.8 * cam.fx / 2.8 / 2, cam.cy - 0.2 * cam.fy / 2.8, cam.cx + 1.8 * cam.fx / 2.8 / 2,
           cam.cy + cam.fy * CAM_HEIGHT_M / 2.8, "car", 0.9)
    assert priority_for([car], cam)["priority"] == 1                           # wide: escalates at 2.8 m
    assert priority_for([person(2.1)], cam, current=1)["priority"] == 1         # hysteresis holds
    s = Steady()
    assert [s.step(p, t) for p, t in ((3, 0), (1, 0.1), (2, 0.2), (2, 0.5), (2, 0.85))] == [3, 1, 1, 1, 2]
    print("closeness selfcheck ok")


if __name__ == "__main__":
    raise SystemExit(main())
