"""Stand-in until streetseg lands; same events, same FrameSource.

YOLO11n boxes -> earpiece events, so "real detector on sim frames -> beep in AirPods"
works before Rex's streetseg exists. Swap this file out; nothing downstream changes.

    python sim/standin_detect.py --download                  # once, fetches weights/yolo11n.pt
    python sim/standin_detect.py                             # sim chest + drone via sim/relay.py
    python sim/standin_detect.py --source clip.mp4 --camera drone --jsonl out.jsonl --overlay out.mp4

chest camera -> `obstacles` (every frame, ~8-10 Hz) + `heartbeat` when empty
drone camera -> `hazard` crowd / vehicle (3-frame onset, silent ttl expiry) + `scene` every 10 s

Geometry assumes pitch 0 and square pixels (fy = fx): bearing from bbox centre x,
distance from the bbox bottom row on flat ground, else from a typical object height.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "earpiece"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from earpiece.events import clock_phrase, distance_phrase  # noqa: E402  (needs numpy, scipy)

WEIGHTS = ROOT / "weights" / "yolo11n.pt"
DEFAULT_FX = 1000.0
DEFAULT_HEIGHT = {"chest": 1.2, "drone": 2.0}     # sim: CHEST_HEIGHT, HEAD_TOP + DRONE_ABOVE_HEAD
DRONE_BEHIND_M = 1.0
MIN_D, MAX_D = 0.3, 50.0
GROUND_MIN_PX = 12          # bbox bottom must be this far below the horizon to trust ground geometry
EDGE_PX = 8                 # bbox this close to the bottom edge counts as cut off

KIND = {"person": "person", "bicycle": "bike", "car": "vehicle", "motorcycle": "vehicle",
        "bus": "vehicle", "truck": "vehicle", "fire hydrant": "fire_hydrant", "parking meter": "pole",
        "bench": "bench", "dog": "dog", "chair": "chair", "potted plant": "plant", "suitcase": "suitcase"}
TYPICAL_H = {"person": 1.7, "bike": 1.1, "vehicle": 1.5, "bus": 3.0, "truck": 3.0}

# drone hazard rules
CROWD_N, CROWD_NEAR, CROWD_FAR, CROWD_HALF_ANGLE = 3, 1.0, 15.0, 35.0
VEHICLE_RANGE, VEHICLE_LANE = 12.0, 3.0           # m ahead, m either side of the walker's path
ONSET_FRAMES, HAZARD_TTL, SCENE_EVERY = 3, 4.0, 10.0


# ------------------------------------------------------------------ pure geometry
def bearing_deg(cx: float, fx: float = DEFAULT_FX, width: int = 960) -> float:
    """Bbox centre x -> degrees, 0 ahead, + right."""
    return math.degrees(math.atan((cx - width / 2) / fx))


def distance_m(y_bottom: float, box_h: float, cam_height: float, fx: float = DEFAULT_FX,
               height: int = 720, real_h: float = 1.7) -> float:
    """Flat-ground distance from the bbox bottom row; size-based when the foot point is
    at or above the horizon (or too close to it to be stable). Clamped to [0.3, 50] m."""
    below = y_bottom - height / 2
    if below > GROUND_MIN_PX:
        d = cam_height * fx / below
    else:
        d = real_h * fx / max(box_h, 1.0)
    return min(MAX_D, max(MIN_D, d))


def drone_to_walker(bearing: float, distance: float, behind: float = DRONE_BEHIND_M) -> tuple[float, float]:
    """Drone-frame (bearing, ground distance) -> walker frame; the walker is `behind` m ahead
    of the drone, same heading."""
    fwd = distance * math.cos(math.radians(bearing)) - behind
    lat = distance * math.sin(math.radians(bearing))
    return math.degrees(math.atan2(lat, fwd)), math.hypot(fwd, lat)


def objects_from_boxes(boxes, fx: float, cam_height: float, width: int, height: int) -> list[dict]:
    """boxes: [(x1, y1, x2, y2, coco_name, conf, track_id|None)] -> earpiece objects."""
    out = []
    for i, (x1, y1, x2, y2, name, conf, tid) in enumerate(boxes):
        kind = KIND.get(name)
        if kind is None:
            continue
        real_h = TYPICAL_H.get(name, TYPICAL_H.get(kind, 1.0))
        out.append({"id": str(tid if tid is not None else i), "kind": kind,
                    "bearing": round(bearing_deg((x1 + x2) / 2, fx, width), 1),
                    "distance": round(distance_m(y2, y2 - y1, cam_height, fx, height, real_h), 2),
                    "confidence": round(float(conf), 2), "cut_off": y2 >= height - EDGE_PX})
    return out


def drone_hazards(objs: list[dict]) -> dict[str, dict]:
    """Drone objects -> candidate hazards {kind: {bearing, distance}} in the walker frame.
    A person cut off by the bottom edge is the walker and never counts."""
    people, cars = [], []
    for o in objs:
        if o["kind"] == "person" and o["cut_off"]:
            continue
        b, d = drone_to_walker(o["bearing"], o["distance"])
        if o["kind"] == "person" and CROWD_NEAR <= d <= CROWD_FAR and abs(b) <= CROWD_HALF_ANGLE:
            people.append((b, d))
        elif o["kind"] == "vehicle" and abs(b) < 90 and d <= VEHICLE_RANGE \
                and abs(d * math.sin(math.radians(b))) <= VEHICLE_LANE:
            cars.append((b, d))
    out = {}
    if len(people) >= CROWD_N:
        out["crowd"] = {"bearing": sum(b for b, _ in people) / len(people), "distance": min(d for _, d in people)}
    if cars:
        b, d = min(cars, key=lambda c: c[1])
        out["vehicle"] = {"bearing": b, "distance": d}
    return out


class HazardGate:
    """A new hazard, or a new direction for a live one, must hold ONSET_FRAMES frames.
    While held it is refreshed with the same text every ttl/2 (the earpiece only re-speaks
    when the text changes). Clearing sends nothing: the ttl runs out."""

    def __init__(self):
        self.streak: dict[str, int] = {}
        self.live: dict[str, dict] = {}     # kind -> {"text", "dir", "sent", "pending", "n"}

    def step(self, cands: dict[str, dict], now: float) -> list[dict]:
        out = []
        for k in list(self.streak):
            if k not in cands:
                del self.streak[k]
                self.live.pop(k, None)
        for k, c in cands.items():
            self.streak[k] = self.streak.get(k, 0) + 1
            if self.streak[k] < ONSET_FRAMES:
                continue
            direction = clock_phrase(c["bearing"])
            live = self.live.get(k)
            if live:
                live["n"] = live["n"] + 1 if direction == live["pending"] else 1
                live["pending"] = direction
                turned = direction != live["dir"] and live["n"] >= ONSET_FRAMES
                if not turned and now - live["sent"] < HAZARD_TTL / 2:
                    continue
            if not live or turned:
                live = {"text": f"{'Crowd' if k == 'crowd' else 'Car'} {direction}, "
                                f"{distance_phrase(c['distance'])}.", "dir": direction,
                        "pending": direction, "n": ONSET_FRAMES}
                self.live[k] = live
            live["sent"] = now
            out.append({"type": "hazard", "kind": k, "text": live["text"], "bearing": round(c["bearing"], 1),
                        "distance": round(c["distance"], 1), "ttl": HAZARD_TTL})
        return out


# ------------------------------------------------------------------ runtime
def download() -> None:
    from ultralytics.utils.downloads import attempt_download_asset
    WEIGHTS.parent.mkdir(exist_ok=True)
    print(attempt_download_asset(str(WEIGHTS)))


def detect(model, frame, track: bool) -> list[tuple]:
    kw = dict(verbose=False, conf=0.35, imgsz=640)
    r = (model.track(frame, persist=True, **kw) if track else model.predict(frame, **kw))[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return []
    ids = b.id.int().tolist() if b.id is not None else [None] * len(b)
    return [(*xyxy, r.names[int(c)], cf, tid)
            for xyxy, c, cf, tid in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist(), ids)]


def draw(frame, objs, boxes, label):
    import cv2
    img = frame.copy()
    kept = [bx for bx in boxes if bx[4] in KIND]
    for (x1, y1, x2, y2, *_), o in zip(kept, objs):
        col = (0, 0, 255) if o["kind"] == "vehicle" else (0, 200, 255) if o["cut_off"] else (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, f"{o['kind']} {o['bearing']:+.0f}d {o['distance']:.1f}m", (int(x1), max(15, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--download", action="store_true", help="fetch weights/yolo11n.pt and exit")
    ap.add_argument("--source", help="video file (default: sim frames from the relay)")
    ap.add_argument("--camera", choices=("chest", "drone", "both"), default="both")
    ap.add_argument("--relay", default="ws://127.0.0.1:5020/sub")
    ap.add_argument("--fx", type=float, default=DEFAULT_FX)
    ap.add_argument("--cam-height", type=float, help="override camera height (m); sim uses its pose")
    ap.add_argument("--device", default=None, help="mps / cpu (default: mps if available)")
    ap.add_argument("--jsonl", help="also write events with a t field (scripts/replay.py)")
    ap.add_argument("--overlay", help="write an mp4 with boxes and bearing/distance labels")
    ap.add_argument("--no-send", action="store_true", help="do not send to the earpiece")
    ap.add_argument("--fast", action="store_true", help="video: do not pace to real time")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this long (0 = forever)")
    a = ap.parse_args()
    if a.download:
        return download()
    if not WEIGHTS.exists():
        sys.exit(f"missing {WEIGHTS}; run: python sim/standin_detect.py --download")

    import cv2
    import numpy as np
    import torch
    from earpiece.client import EarpieceClient
    from earpiece.config import _load_env  # noqa: F401  loads the repo .env
    from ultralytics import YOLO

    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    cams = ("chest", "drone") if a.camera == "both" else (a.camera,)
    models = {c: YOLO(str(WEIGHTS)) for c in cams}           # one tracker per camera
    ear = EarpieceClient(host=os.environ.get("KITE_EARPIECE_HOST", "127.0.0.1"), source="standin")
    jsonl = open(a.jsonl, "w") if a.jsonl else None
    t0 = time.monotonic()

    if a.source:
        cap = cv2.VideoCapture(a.source)
        if not cap.isOpened():
            sys.exit(f"cannot open {a.source}")
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(vfps / 10))                      # ~10 Hz
        srcs, fidx = None, 0
    else:
        from frames import SimFrameSource
        srcs = {c: SimFrameSource(c, a.relay) for c in cams}

    def emit(ev: dict, t: float) -> None:
        if not a.no_send:
            ear.send(dict(ev))
        if jsonl:
            jsonl.write(json.dumps({"t": round(t, 3), **ev}, separators=(",", ":")) + "\n")

    gate, last_scene, writer = HazardGate(), -SCENE_EVERY, None
    n, infer_s, warm = 0, 0.0, None
    print(f"[standin] cams={cams} device={device} -> {ear.addr} source={a.source or a.relay}", flush=True)
    try:
        while True:
            if a.max_seconds and time.monotonic() - t0 > a.max_seconds:
                break
            frames, poses = {}, {}
            if srcs is None:
                for _ in range(step - 1):
                    cap.grab()
                ok, img = cap.read()
                if not ok:
                    break
                t = fidx * step / vfps
                fidx += 1
                if not a.fast:
                    lag = t - (time.monotonic() - t0)
                    if lag > 0:
                        time.sleep(lag)
                if img.shape[1] != 960:
                    img = cv2.resize(img, (960, round(img.shape[0] * 960 / img.shape[1])))
                frames = {c: img for c in cams}
            else:
                for c, s in srcs.items():
                    ok, img = s.read()
                    if ok:
                        frames[c], poses[c] = img, s.pose or {}
                if not frames:
                    print("[standin] no sim frames yet (relay up? sim page open?)", flush=True)
                    continue
                t = time.monotonic() - t0
            tiles = {}
            for c, img in frames.items():
                h, w = img.shape[:2]
                cam_h = a.cam_height or poses.get(c, {}).get("cam_height") or DEFAULT_HEIGHT[c]
                ts = time.perf_counter()
                boxes = detect(models[c], img, track=True)
                dt = time.perf_counter() - ts
                if warm is None:
                    warm = dt
                else:
                    infer_s += dt
                    n += 1
                objs = objects_from_boxes(boxes, a.fx, cam_h, w, h)
                if c == "chest":
                    emit({"type": "obstacles", "source": "standin-chest", "seq": n,
                          "objects": [{k: v for k, v in o.items() if k != "cut_off"} for o in objs]}, t)
                    if not objs:
                        emit({"type": "heartbeat", "source": "standin-chest"}, t)
                else:
                    for ev in gate.step(drone_hazards(objs), t):
                        emit({**ev, "source": "standin-drone"}, t)
                        print(f"[standin] t={t:.1f} hazard {ev['kind']}: {ev['text']}", flush=True)
                    if t - last_scene >= SCENE_EVERY:
                        last_scene = t
                        cnt = {}
                        for o in objs:
                            if not (o["kind"] == "person" and o["cut_off"]):
                                cnt[o["kind"]] = cnt.get(o["kind"], 0) + 1
                        text = "Drone view: " + (", ".join(f"{v} {k}" for k, v in sorted(cnt.items())) or "clear") + "."
                        emit({"type": "scene", "source": "standin-drone", "text": text}, t)
                if a.overlay:
                    tiles[c] = cv2.resize(draw(img, objs, boxes, c), (960, 720))
            if a.overlay and tiles:
                blank = np.zeros((720, 960, 3), np.uint8)            # a camera that missed this tick
                frame = cv2.hconcat([tiles.get(c, blank) for c in cams])
                if writer is None:
                    writer = cv2.VideoWriter(a.overlay, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
        if jsonl:
            jsonl.close()
        if n:
            print(f"[standin] {n} inferences, {infer_s / n * 1000:.1f} ms each ({n / infer_s:.1f} fps) "
                  f"on {device}; first call {warm * 1000:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
