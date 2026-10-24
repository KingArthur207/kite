"""Stand-in until streetseg lands; same events, same FrameSource.

YOLO11n boxes -> earpiece events, so "real detector on sim frames -> beep in AirPods"
works before the streetseg perception package exists. Swap this file out; nothing downstream changes.

    python sim/standin_detect.py --download                  # once, fetches weights/yolo11n.pt
    python sim/standin_detect.py                             # sim chest + drone via sim/relay.py
    python sim/standin_detect.py --source clip.mp4 --camera drone --jsonl out.jsonl --overlay out.mp4
    python sim/standin_detect.py --source webcam:iPhone --show   # live chest camera, by name or index (kite.sources)
    python sim/standin_detect.py --source streetview:route --camera drone --jsonl sv.jsonl --no-send   # real imagery

chest camera -> `obstacles` (every frame, ~8-10 Hz) + `heartbeat` when empty; static things stay in them for a
                while after they leave the view (world_frame.ObstacleMemory); --world adds YOLO-World street classes
drone camera -> `hazard` crowd / vehicle (3-frame onset, silent ttl expiry) + `scene` every 10 s; a car closing
                on the crossing ahead -> `hazard` vehicle/corner, state wait (world_frame.CornerWatch)

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
sys.path.insert(0, str(ROOT))                        # kite.sources: one way to open a camera
sys.path.insert(0, str(Path(__file__).resolve().parent))

from earpiece.events import clock_phrase, distance_phrase  # noqa: E402  (needs numpy, scipy)
import live_walk  # noqa: E402
import world_frame  # noqa: E402

WEIGHTS = ROOT / "weights" / "yolo11n.pt"
# Open vocabulary for the chest camera (--world): street things COCO has no class for. `--download --world` bakes
# this vocabulary into WORLD_WEIGHTS once (it needs CLIP then); at runtime nothing is fetched.
WORLD_BASE = ROOT / "weights" / "yolov8s-worldv2.pt"
WORLD_WEIGHTS = ROOT / "weights" / "yolov8s-worldv2-street.pt"
WORLD_KIND = {"trash can": "bin", "bin": "bin", "sign": "sign", "pole": "pole", "bollard": "bollard",
              "fire hydrant": "fire_hydrant", "bench": "bench", "scooter": "scooter", "stroller": "stroller"}
WORLD_CONF = 0.3        # sim: the street's trash cans score 0.7; the metric's plain green box at most 0.23
DEFAULT_FX = 1000.0
LIVE_FX = 700.0             # iPhone 1x / MacBook camera at 960 wide (~69 deg across); calibrate with --fx
DEFAULT_HEIGHT = {"chest": 1.2, "drone": 2.0}     # sim: CHEST_HEIGHT, HEAD_TOP + DRONE_ABOVE_HEAD
DRONE_BEHIND_M = 1.0
MIN_D, MAX_D = 0.3, 50.0
GROUND_MIN_PX = 12          # bbox bottom must be this far below the horizon to trust ground geometry
EDGE_PX = 16                # bbox this close to the bottom (or top) edge counts as cut off (YOLO boxes stop
                            # a few px short of the edge: at 8 px one person flickered 1.3 m / 3.2 m)
PERSON_W = 0.45             # m, a person's box width face-on; only used when head and feet are both out of frame

KIND = {"person": "person", "bicycle": "bike", "car": "vehicle", "motorcycle": "vehicle",
        "bus": "vehicle", "truck": "vehicle", "fire hydrant": "fire_hydrant", "parking meter": "pole",
        "bench": "bench", "dog": "dog", "chair": "chair", "potted plant": "plant", "suitcase": "suitcase",
        **WORLD_KIND}
TYPICAL_H = {"person": 1.7, "bike": 1.1, "vehicle": 1.5, "bus": 3.0, "truck": 3.0, "bin": 1.0, "sign": 0.5,
             "pole": 3.0, "bollard": 0.9, "fire_hydrant": 0.8, "bench": 0.9, "scooter": 1.1, "stroller": 1.0}

# drone hazard rules
CROWD_N, CROWD_NEAR, CROWD_FAR, CROWD_HALF_ANGLE = 3, 1.0, 15.0, 35.0
VEHICLE_RANGE, VEHICLE_LANE = 12.0, 1.5           # m ahead, m either side of the walker's path (Moss is a 6 m alley:
                                                  # 3 m flagged oncoming cars while the walker was on the sidewalk)
CROSS_HORIZON, CROSS_MIN_V = 3.0, 0.5             # a car further out counts if it reaches the path within 3 s
ONSET_FRAMES, HAZARD_TTL, SCENE_EVERY = 3, 4.0, 10.0
# parked cars (Street View on Moss: a car parked by the corner kept warning): a vehicle more than PARKED_LANE off the
# path whose world position holds within STILL_M for STILL_S is parked and never a hazard. Off-path cars wait
# STILL_S before they count, so a parked one never gets its 3-frame onset first.
PARKED_LANE, STILL_M, STILL_S, STILL_KEEP_S = 0.8, 1.5, 0.5, 10.0


# ------------------------------------------------------------------ pure geometry
def bearing_deg(cx: float, fx: float = DEFAULT_FX, width: int = 960) -> float:
    """Bbox centre x -> degrees, 0 ahead, + right."""
    return math.degrees(math.atan((cx - width / 2) / fx))


def below_horizon(y: float, fx: float, height: int, pitch: float = 0.0) -> float:
    """Radians the ray through image row y points below horizontal; pitch in degrees, + = tilted down."""
    return math.radians(pitch) + math.atan((y - height / 2) / fx)


def distance_m(y_bottom: float, box_h: float, cam_height: float, fx: float = DEFAULT_FX,
               height: int = 720, real_h: float = 1.7, pitch: float = 0.0, y_top: float | None = None) -> float:
    """Flat-ground distance from the bbox bottom row; size-based when the foot point is
    at or above the horizon (or too close to it to be stable). Clamped to [0.3, 50] m.
    Pass y_top when the box is cut off by the bottom or the top edge (a person close to a chest camera):
    the object is taller in the image than its box, so the bottom row and box height only bound the
    distance from above, and the top row (head at real_h, or above it) gives it when the object is
    taller than the camera."""
    stable = math.atan(GROUND_MIN_PX / fx)
    a = below_horizon(y_bottom, fx, height, pitch)
    size_d = real_h * fx / max(box_h, 1.0)
    d = cam_height / math.tan(a) if a > stable else size_d
    if y_top is not None:
        d = min(d, size_d)
        up = -below_horizon(y_top, fx, height, pitch)
        if real_h - cam_height > 0.2 and up > stable:
            d = min(d, (real_h - cam_height) / math.tan(up))
    return min(MAX_D, max(MIN_D, d))


def drone_to_walker(bearing: float, distance: float, behind: float = DRONE_BEHIND_M) -> tuple[float, float]:
    """Drone-frame (bearing, ground distance) -> walker frame; the walker is `behind` m ahead
    of the drone, same heading."""
    fwd = distance * math.cos(math.radians(bearing)) - behind
    lat = distance * math.sin(math.radians(bearing))
    return math.degrees(math.atan2(lat, fwd)), math.hypot(fwd, lat)


def drone_obj_to_walker(o: dict, pose: dict | None) -> tuple[float, float]:
    """Drone object -> (bearing, distance) from the walker. While the drone scouts a corner its pose says where it
    is and which way it looks (cam_x, cam_y, cam_heading); else it is DRONE_BEHIND_M behind, looking the same way."""
    if pose and "cam_x" in pose and world_frame.walker_of(pose):
        x, y = world_frame.to_world(world_frame.drone_cam(pose), o["bearing"], o["distance"])
        return world_frame.to_walker(world_frame.walker_of(pose), x, y)
    return drone_to_walker(o["bearing"], o["distance"])


def objects_from_boxes(boxes, fx: float, cam_height: float, width: int, height: int,
                       pitch: float = 0.0) -> list[dict]:
    """boxes: [(x1, y1, x2, y2, coco_name, conf, track_id|None)] -> earpiece objects."""
    out = []
    for i, (x1, y1, x2, y2, name, conf, tid) in enumerate(boxes):
        kind = KIND.get(name)
        if kind is None:
            continue
        real_h = TYPICAL_H.get(name, TYPICAL_H.get(kind, 1.0))
        cut, top_cut = y2 >= height - EDGE_PX, y1 <= EDGE_PX
        # Head above the frame: the top row still bounds the distance, even when the bottom row looks like feet
        # (a box ending 9 px above the edge read 3.2 m for someone at 1.3 m; live_walk_bench).
        d = distance_m(y2, y2 - y1, cam_height, fx, height, real_h, pitch, y_top=y1 if cut or top_cut else None)
        if cut and top_cut and kind == "person":
            # head and feet both out of frame: no row can say how close (at 15 deg down on 960x540 every distance
            # under 1.65 m read 1.65, so "Stop" never fired). Shoulder width still can.
            d = max(MIN_D, min(d, PERSON_W * fx / max(x2 - x1, 1.0)))
        out.append({"id": str(tid if tid is not None else i), "kind": kind,
                    "bearing": round(bearing_deg((x1 + x2) / 2, fx, width), 1),
                    "distance": round(d, 2), "confidence": round(float(conf), 2), "cut_off": cut})
    return out


class StillVehicles:
    """Where off-path vehicles were seen, in world metres. judge() -> "parked" when one was seen within STILL_M of
    the same spot at least STILL_S earlier (any track id: Street View steps break tracks), "moving" once its track
    is STILL_S old and not parked, else "unknown"."""
    # ponytail: a car creeping under STILL_M / STILL_S (3 m/s) counts as parked; fine off the path, where this applies

    def __init__(self):
        self.seen: list[tuple[float, float, float]] = []
        self.first: dict[str, float] = {}

    def judge(self, tid: str, t: float, x: float, y: float) -> str:
        self.seen = [s for s in self.seen if t - s[0] <= STILL_KEEP_S]
        parked = any(t - st >= STILL_S and math.hypot(sx - x, sy - y) <= STILL_M for st, sx, sy in self.seen)
        self.seen.append((t, x, y))
        first = self.first.setdefault(tid, t)
        return "parked" if parked else "moving" if t - first >= STILL_S else "unknown"


def drone_hazards(objs: list[dict], lateral: live_walk.RangeFilter | None = None, t: float = 0.0,
                  pose: dict | None = None, still: StillVehicles | None = None) -> dict[str, dict]:
    """Drone objects -> candidate hazards {kind: {bearing, distance}} in the walker frame.
    A person cut off by the bottom edge is the walker and never counts. A vehicle counts when it is
    within VEHICLE_LANE of the walker's path, or (with `lateral`, a RangeFilter fed each car's
    sideways offset per track) when it is closing on the path fast enough to cross it within CROSS_HORIZON.
    A car driving along the street beside the walker keeps its offset and never counts.
    With `still` and the walker's `pose` (x, y metres, compass heading), a vehicle more than PARKED_LANE off the
    path counts only once StillVehicles has seen it move: parked cars never do."""
    people, cars = [], []
    for o in objs:
        if o["kind"] == "person" and o["cut_off"]:
            continue
        b, d = drone_obj_to_walker(o, pose)
        if o["kind"] == "person" and CROWD_NEAR <= d <= CROWD_FAR and abs(b) <= CROWD_HALF_ANGLE:
            people.append((b, d))
        elif o["kind"] == "vehicle" and abs(b) < 90 and d <= VEHICLE_RANGE:
            lat, v = abs(d * math.sin(math.radians(b))), None
            if lateral is not None:
                lat, v = lateral.update(o["id"], t, lat)
            if still is not None and pose and "x" in pose and lat > PARKED_LANE:
                h = math.radians(pose["heading"] + b)
                if still.judge(o["id"], t, pose["x"] + d * math.sin(h), pose["y"] + d * math.cos(h)) != "moving":
                    continue
            if lat <= VEHICLE_LANE or (v is not None and v >= CROSS_MIN_V and lat / v <= CROSS_HORIZON):
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
def download(world: bool = False) -> None:
    from ultralytics.utils.downloads import attempt_download_asset
    WEIGHTS.parent.mkdir(exist_ok=True)
    print(attempt_download_asset(str(WEIGHTS)))
    if world:
        from ultralytics import YOLOWorld
        print(attempt_download_asset(str(WORLD_BASE)))
        m = YOLOWorld(str(WORLD_BASE))
        m.set_classes(list(WORLD_KIND))         # CLIP text encoder, only here
        m.save(str(WORLD_WEIGHTS))
        print(WORLD_WEIGHTS)


def detect(model, frame, track: bool, conf: float = 0.35) -> list[tuple]:
    kw = dict(verbose=False, conf=conf, imgsz=640)
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
        col = (0, 0, 255) if o["kind"] == "vehicle" or o.get("stop") else (0, 200, 255) if o["cut_off"] \
            else (0, 255, 0)
        v = o.get("closing_speed")
        text = f"{o['kind']} {o['bearing']:+.0f}d {o['distance']:.1f}m" + (f" {v:+.1f}m/s" if v is not None else "")
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, text, (int(x1) + 2, int(y1) - 6 if y1 > 24 else int(y1) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--download", action="store_true", help="fetch weights/yolo11n.pt and exit")
    ap.add_argument("--source", help="video file, or a live chest camera: an index or webcam:<index or name>, "
                                     "e.g. webcam:iPhone (default: sim frames)")
    ap.add_argument("--list-cameras", action="store_true", help="print the cameras (index, name, size) and exit")
    ap.add_argument("--camera", choices=("chest", "drone", "both"), default="both")
    ap.add_argument("--relay", default="ws://127.0.0.1:5020/sub")
    ap.add_argument("--ear-port", type=int, default=5010, help="earpiece UDP port")
    ap.add_argument("--fx", type=float, help=f"focal length in px at 960 wide (default {DEFAULT_FX:.0f}, "
                                              f"{LIVE_FX:.0f} for a live camera)")
    ap.add_argument("--cam-height", type=float, help="override camera height (m); sim uses its pose")
    ap.add_argument("--pitch", type=float, default=0.0, help="camera tilt in degrees, + = down (chest phone); "
                                                             "streetview: also the tilt the images are fetched at")
    ap.add_argument("--hfov", type=float, help="streetview: horizontal FOV to fetch, deg (default the Tello's)")
    ap.add_argument("--cache-dir", help="streetview: reuse responses between runs (under /tmp; delete it after)")
    ap.add_argument("--person-height", type=float, help="height of the person who steps in (default 1.7 m)")
    ap.add_argument("--stop-m", type=float, default=live_walk.STOP_M, help="chest: critical inside this range")
    ap.add_argument("--depth", action="store_true", help="chest: Depth Anything fallback for close non-YOLO things")
    ap.add_argument("--world", action="store_true", help="chest: also YOLO-World for bins, signs, poles, bollards ... "
                                                         "(--download --world fetches it; needs CLIP for that step: "
                                                         "--with 'clip @ git+https://github.com/ultralytics/CLIP.git')")
    ap.add_argument("--world-every", type=int, default=1, help="--world: every Nth chest frame (~11 ms on mps)")
    ap.add_argument("--no-memory", action="store_true", help="chest: forget obstacles as soon as they leave the view")
    ap.add_argument("--crossing", help="x,y of the crossing ahead in the pose's metres, for the corner car watch "
                                       "(default 0,0 on sim frames: the Moss/Howard zebra)")
    ap.add_argument("--show", action="store_true", help="reveal window: live chest view, r = replay last 20 s")
    ap.add_argument("--no-blur", action="store_true", help="--show: do not pixelate heads")
    ap.add_argument("--device", default=None, help="mps / cpu (default: mps if available)")
    ap.add_argument("--jsonl", help="also write events with a t field (scripts/replay.py)")
    ap.add_argument("--overlay", help="write an mp4 with boxes and bearing/distance labels")
    ap.add_argument("--no-send", action="store_true", help="do not send to the earpiece")
    ap.add_argument("--fast", action="store_true", help="video: do not pace to real time")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this long (0 = forever)")
    a = ap.parse_args()
    if a.download:
        download(a.world)
        return live_walk.download_depth() if a.depth else None
    if a.list_cameras:
        from kite.sources.open import list_cameras
        return list_cameras(probe=True)
    live, sv, kind = False, None, None
    if a.source:
        from kite.sources import parse_spec
        try:
            kind = parse_spec(a.source)[0]
        except ValueError as e:
            sys.exit(str(e))
        if kind not in ("webcam", "file", "streetview"):
            sys.exit(f"--source {a.source}: use a video file, a camera or streetview:... (sim frames are the default)")
        live = kind == "webcam"
    if kind == "streetview" and a.overlay:
        sys.exit("--overlay is off for streetview: Google's terms do not allow storing the images")
    if live and a.overlay:
        sys.exit("--overlay is off for live cameras: we don't store frames during the live demo (spec 5.3)")
    fx = a.fx or (LIVE_FX if live else DEFAULT_FX)
    if a.person_height:
        TYPICAL_H["person"] = a.person_height
    if not WEIGHTS.exists():
        sys.exit(f"missing {WEIGHTS}; run: python sim/standin_detect.py --download")
    if a.world and not WORLD_WEIGHTS.exists():
        sys.exit(f"missing {WORLD_WEIGHTS}; run: python sim/standin_detect.py --download --world")
    crossing = tuple(float(v) for v in a.crossing.split(",")) if a.crossing else None if a.source else (0.0, 0.0)

    import cv2
    import numpy as np
    import torch
    from earpiece.client import EarpieceClient
    from earpiece.config import _load_env  # noqa: F401  loads the repo .env
    from ultralytics import YOLO

    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    cams = ("chest",) if live or (a.show and a.camera == "both") else \
        ("chest", "drone") if a.camera == "both" else (a.camera,)
    if a.show and cams != ("chest",):
        sys.exit("--show is the chest view; use --camera chest")
    models = {c: YOLO(str(WEIGHTS)) for c in cams}           # one tracker per camera
    world = YOLO(str(WORLD_WEIGHTS)) if a.world and "chest" in cams else None
    ear = EarpieceClient(host=os.environ.get("KITE_EARPIECE_HOST", "127.0.0.1"), port=a.ear_port, source="standin")
    jsonl = open(a.jsonl, "w") if a.jsonl else None
    t0 = time.monotonic()

    if kind == "streetview":
        from kite.sources import open_source
        opts = {k: v for k, v in (("hfov_deg", a.hfov), ("cache_dir", a.cache_dir)) if v is not None}
        try:
            sv = open_source(a.source, pitch=-a.pitch, **opts)  # pose pitch: + = up
        except (RuntimeError, ValueError) as e:
            sys.exit(str(e))
        fx, srcs = sv.camera.fx, None
    elif live:
        from kite.sources import open_source
        try:
            cap = open_source(a.source, timeout=2.0)          # newest frame only, so no growing lag
        except (LookupError, RuntimeError, OSError) as e:
            sys.exit(f"cannot open camera {a.source}: {e}\ntry --list-cameras (macOS: allow Camera for the terminal)")
        srcs = None
    elif a.source:
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

    gate, lateral, last_scene, writer = HazardGate(), live_walk.RangeFilter(), -SCENE_EVERY, None
    still = StillVehicles()
    memory = None if a.no_memory else world_frame.ObstacleMemory()
    corner, chest_n, corner_on = world_frame.CornerWatch(crossing), 0, False
    n, infer_s, warm = 0, 0.0, None
    rf, depth = live_walk.RangeFilter(), live_walk.DepthFallback(device) if a.depth else None
    reveal = live_walk.Reveal() if a.show else None
    fps, fps_t, fps_n, was_stop = 0.0, time.monotonic(), 0, False
    print(f"[standin] cams={cams} device={device} -> {ear.addr} source={a.source or a.relay}", flush=True)
    for m in [*models.values(), *([world] if world else [])]:   # the first call takes seconds (mps kernels):
        m.predict(np.zeros((540, 960, 3), np.uint8), verbose=False, imgsz=640)   # pay it before the walk
    t0 = time.monotonic()                 # video t=0 is now: the bench lines frames up with speech by this
    print(f"[standin] t0_wall={time.time():.3f}", flush=True)
    try:
        while True:
            if a.max_seconds and time.monotonic() - t0 > a.max_seconds:
                break
            frames, poses = {}, {}
            if sv is not None:
                ok, img = sv.read()
                if not ok:
                    break
                t, fx = sv.pose["t"], sv.camera.fx
                frames, poses = {c: img for c in cams}, {c: sv.pose for c in cams}
                p = sv.pano
                print(f"[standin] t={t:.1f} streetview {p['lat']:.6f},{p['lng']:.6f} heading {sv.pose['heading']} "
                      f"{p['date']} {p['copyright']}", flush=True)
            elif live:
                ok, img = cap.read()
                if not ok:
                    print("[standin] camera read failed", flush=True)
                    break
                t = time.monotonic() - t0
                if img.shape[1] != 960:
                    img = cv2.resize(img, (960, round(img.shape[0] * 960 / img.shape[1])))
                frames = {"chest": img}
            elif srcs is None:
                want = fidx + step - 1
                if not a.fast:      # behind real time: skip to the current frame, as a live camera would
                    want = max(want, int((time.monotonic() - t0) * vfps))
                while fidx < want and cap.grab():
                    fidx += 1
                ok, img = cap.read()
                if not ok:
                    break
                t = fidx / vfps
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
                boxes = detect(models[c], img, track=sv is None)   # streetview: stills ~10 m apart, and
                # ByteTrack only reports a new object from its second frame, so it hid most of them
                dt = time.perf_counter() - ts
                if warm is None:
                    warm = dt
                else:
                    infer_s += dt
                    n += 1
                pc = poses.get(c) or {}
                # sim/streetview_bridge.py sends each camera's fx and pitch (+ = up) in its pose
                cfx = pc.get("fx") or fx
                cpitch = -pc["pitch"] if pc.get("source") == "streetview" else a.pitch
                objs = objects_from_boxes(boxes, cfx, cam_h, w, h, cpitch)
                if c == "chest":
                    chest_n += 1
                    if world is not None and chest_n % max(1, a.world_every) == 0:
                        wboxes = [(*b[:6], b[6] if b[6] is None else f"w{b[6]}")      # own track ids
                                  for b in detect(world, img, True, WORLD_CONF)]
                        objs, boxes = objs + objects_from_boxes(wboxes, cfx, cam_h, w, h, cpitch), boxes + wboxes
                    extra = depth(img, cfx, cam_h, cpitch, objs) if depth else None
                    raw = objs + ([extra] if extra else [])
                    if memory is not None:
                        raw = memory.step(raw, pc, t)
                    objs = sent = live_walk.chest_step(raw, rf, t, a.stop_m)
                    emit({"type": "obstacles", "source": "standin-chest", "seq": n,
                          "objects": live_walk.to_event_objects(sent)}, t)
                    if not sent:
                        emit({"type": "heartbeat", "source": "standin-chest"}, t)
                    stop = live_walk.stop_text(sent)
                    if stop and not was_stop:
                        print(f"[standin] t={t:.1f} {stop}", flush=True)
                    was_stop = bool(stop)
                    if reveal:
                        kept = [bx for bx in boxes if bx[4] in KIND]
                        view = draw(img if a.no_blur else live_walk.blur_heads(img.copy(), kept, objs, cfx),
                                    objs, boxes, "chest")
                        if extra:
                            cv2.putText(view, f"depth: obstacle {extra['bearing']:+.0f}d {extra['distance']:.1f}m",
                                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        if not reveal.frame(view, sent, t, fps):
                            return
                else:
                    for ev in gate.step(drone_hazards(objs, lateral, t, pc, still), t):
                        emit({**ev, "source": "standin-drone"}, t)
                        print(f"[standin] t={t:.1f} hazard {ev['kind']}: {ev['text']}", flush=True)
                    cam = world_frame.drone_cam(pc)
                    cars = [(o["id"], *world_frame.to_world(cam, o["bearing"], o["distance"]))
                            for o in objs if o["kind"] == "vehicle"] if cam else []
                    ev = corner.step(cars, pc, t)
                    if ev:
                        emit({**ev, "source": "standin-drone"}, t)
                        if not corner_on:
                            print(f"[standin] t={t:.1f} hazard corner: {ev['text']}", flush=True)
                    corner_on = bool(ev)
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
            fps_n += 1
            if time.monotonic() - fps_t >= 5.0:
                fps = fps_n / (time.monotonic() - fps_t)
                fps_t, fps_n = time.monotonic(), 0
                print(f"[standin] {fps:.1f} fps end to end", flush=True)
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
        if reveal:
            cv2.destroyAllWindows()
        if live:
            cap.release()
        if sv is not None:
            sv.release()                                      # prints the request count
        if n:
            print(f"[standin] {n} inferences, {infer_s / n * 1000:.1f} ms each ({n / infer_s:.1f} fps) "
                  f"on {device}; first call {warm * 1000:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
