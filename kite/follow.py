"""Follow mode: keep the Tello about 0.3 m above and ~1.2 m behind the walker's head, on rc sticks.

    python -m kite.follow --source recordings/<take>              # rehearsal: what follow WOULD send, per frame
    python -m kite.follow --source clip.mp4 --overlay --preview    # any video; overlay mp4 next to the csv
    python -m kite.record --source tello --out recordings --fly --follow   # live: g toggles follow
    python -m kite.record --source tello --out recordings --fly --follow --follow-predict   # steer on the prediction
    python -m kite.follow --source recordings/<take> --yaw pd      # opt in: a kite.fly yaw source (simulation only)

Geometry. The camera is fixed and level, so from above and behind only the top of the walker's head (and, further
back, the shoulders) is in frame, cut off by the bottom edge: that cut-off person is the walker (spec 5.3). The head
top's image row gives the distance: a camera `above_m` over the head top sees it at depression angle
atan(above_m / distance), i.e. row cy + fy * above_m / distance (590 px at 0.3 m / 1.2 m, fx 920).

Pieces (each testable without a drone):
  Detector          person boxes from one BGR frame (YoloPersonDetector: ultralytics YOLO11n, optional extra
                    `follow`; tests use a fake)
  WalkerTracker     picks the walker among the boxes: acquires only a box touching the bottom edge, then follows
                    the box nearest the previous one; tolerates missed frames
  FollowController  track + telemetry -> rc (lr, fb, ud, yaw): yaw centres the walker, fb holds the distance
                    (plus feed-forward of the walker's estimated speed and a clamped integral for wind), ud holds the
                    altitude, small lr. Deadbands, per-channel clamps, slew limits (braking is never rate limited).
                    Track older than lost_s -> rc 0 (hover); it never searches by moving.
                    Video lag: the range seen in a frame is carried forward to now with the walker's estimated speed
                    and the drone's modelled travel since the frame (stick -> speed dead time + lag).
                    Head guard (last, on every command, the planner's too): forward speed never above the walker's
                    speed plus a closing speed it can still brake from before min_range_m; inside the standoff fb
                    never increases; closer than min_range_m no forward stick and no descent.
                    Speed limit: the Tello does ~0.96 m/s at full stick (0.0096 m/s per unit), and SafetyCfg.max_rc
                    caps it lower. A walker faster than that for outpace_s (or out past max_follow_m) puts follow in
                    'outpaced': it hovers in place (fb, lr 0; yaw keeps facing the walker, altitude held) and warns,
                    and resumes once the walker is slower again. It never chases at full stick.
                    Prediction (kite/predict.py, opt in: FollowCfg.predict, --follow-predict / --predict): while
                    following, a walker motion model (IMM Kalman filter) and a short-horizon drone planner replace
                    lr, fb and yaw; the reactive command stays when the filter is not confident yet. Coast, lost
                    (hover), ud, clamps and slew are the same either way.
                    Yaw source (kite/fly, opt in: --follow-yaw / --yaw pd|fly, extra `fly`): a predictive PD or the
                    fruit-fly pursuit circuit decides the yaw stick alone while following; neither passed its
                    simulation ship gate (kite/fly/DESIGN.md section 11). Exclusive with prediction.
  FollowMode       the live glue for kite.record: tracker thread, start checks, decisions -> events.jsonl.
Every rc goes through the SafetyGovernor (kite.record sends it), which clamps to SafetyCfg.max_rc.

rc signs (Drone.rc, TelloDrone -> djitellopy send_rc_control(lr, fb, ud, yaw) -> "rc lr fb ud yaw"):
+lr right, +fb forward, +ud up, +yaw clockwise (= turn right; kite.types: heading clockwise positive).

Weights (one time, on normal internet; never fetched at runtime):
    pip install -e ".[follow]"
    mkdir -p data/models && curl -L -o data/models/yolo11n.pt \\
        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
data/models/yolo11n.pt is in git. KITE_YOLO_WEIGHTS or --weights point elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .camera import Camera, load_camera
from .predict import DronePlanner, Plan, PlannerCfg, Prediction, WalkerFilter, fwd_right, route_turn_ahead
from .types import Telemetry

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "data" / "models" / "yolo11n.pt"
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
FETCH_HINT = (f"one time, on normal internet: pip install -e \".[follow]\" && mkdir -p data/models && "
              f"curl -L -o data/models/yolo11n.pt {WEIGHTS_URL}")


@dataclass
class FollowCfg:
    # where to fly relative to the walker
    behind_m: float = 1.2  # horizontal distance camera -> top of the walker's head (1.5 shows more head + shoulders)
    above_m: float = 0.3  # camera height above the head top; turns the head's image row into a distance
    height_m: float | None = None  # altitude to hold, m above the takeoff point (None: the height when follow starts)
    # tracker
    min_conf: float = 0.25  # person boxes below this confidence are ignored (a head seen from above scores low)
    edge_px: float = 16.0  # a box ending within this of the bottom edge "touches" it (at 720 px; scaled)
    gate_frac: float = 0.3  # a box whose centre is further than this x frame width from the track is someone else
    smooth: float = 0.6  # weight of a new detection in the smoothed box
    min_hits: int = 3  # detections before a new track is trusted
    keep_s: float = 2.0  # a track unseen this long is dropped (re-acquiring then needs the bottom edge again)
    # controller
    lost_s: float = 0.5  # no walker for this long -> hover (rc 0)
    fresh_s: float = 0.2  # a track older than this is coasted (last command, decaying), not steered on
    coast_decay: float = 0.7  # per 0.1 s while coasting
    k_yaw: float = 1.5  # yaw stick per degree of bearing to the walker
    k_lr: float = 10.0  # lr stick per metre of sideways offset (small: turns use yaw)
    k_fb: float = 35.0  # fb stick per metre of distance error
    k_ud: float = 50.0  # ud stick per metre of height error
    ff: float = 0.8  # fraction of the walker's estimated speed fed forward into fb
    # m/s per stick unit: 0.27-0.30 m/s at stick 30 in the team's step tests, ~0.96 m/s at 100 (extrapolated)
    stick_m_s: float = 0.0096
    response_s: float = 0.4  # drone speed lag (first order), for the speed estimate behind the feed-forward
    dead_s: float = 0.3  # stick -> speed dead time (step tests: 0.3-0.47 s on fb), in the same model
    lag_comp: bool = True  # carry the measured range forward to now (video lag + the drone's modelled travel)
    k_i_fb: float = 8.0  # fb stick per metre-second of range error: wind and model error (0: off)
    i_max: float = 20.0  # integral clamp, stick units (anti-windup: also frozen while fb is saturated)
    brake_m_s2: float = 0.8  # deceleration the closing-speed cap assumes (it must stop before min_range_m)
    # speed honesty: the drone cannot fly faster than min(max_fb, SafetyCfg.max_rc) x stick_m_s
    # outpaced (hover in place, warn) when for outpace_s fb is saturated (or the walker is past max_follow_m), the
    # walker is outpace_err_m behind the standoff, and the range is not progress_m shorter than at the start
    outpace_s: float = 4.0
    outpace_err_m: float = 1.0
    progress_m: float = 0.05
    outpace_fast_s: float = 1.5  # ... or sooner, after this long, if the range has grown by outpace_grow_m
    outpace_grow_m: float = 0.5
    max_follow_m: float = 4.0  # no resume from outpaced while the walker is further than this
    resume_frac: float = 0.7  # outpaced -> following once the walker is slower than this x top speed ...
    resume_s: float = 1.5  # ... for this long, with the range not growing by more than progress_m meanwhile
    video_lag_s: float = 0.25  # DroneCfg.video_lag_s: a frame shows the world this long ago
    speed_tau_s: float = 0.8  # smoothing of the walker speed estimate (a speed-up is believed this slowly) ...
    stop_tau_s: float = 0.25  # ... a slow-down this quickly: braking for a stop never waits for the slow filter
    db_yaw_deg: float = 2.0  # deadbands
    db_lat_m: float = 0.1
    db_range_m: float = 0.1
    db_h_m: float = 0.08
    max_yaw: int = 40  # per-channel clamps (the governor's max_rc also applies)
    max_lr: int = 20
    max_fb: int = 100  # follow speed is limited by the governor (SafetyCfg.max_rc): raise that to go faster
    max_ud: int = 30
    slew: float = 150.0  # stick units per second on every channel (rate limit); a hover is never rate limited
    # more forward stick, slower: speeding up tilts the nose (and camera) down, which reads as the walker further
    # away, which asks for more speed. 60/s keeps the tilt to ~3 deg. Braking is never rate limited (it reads closer)
    slew_fb_up: float = 60.0
    # closer than this: no feed-forward, no forward stick, no descent. The head top leaves the frame below
    # above_m / tan(half the vertical FOV) = 0.3 / (360 / 920) = 0.77 m, so the guard must act before that
    min_range_m: float = 0.9
    # reactive standoff = behind_m + headway_s x walker speed (constant time gap, as in adaptive cruise control).
    # A stop shows up ~0.85 s late (video 0.25 + detection + estimate + 0.3 dead time) and the drone needs ~0.45 s
    # x speed more to brake: a dead stop costs ~1.3 s x speed. behind_m + headway_s x v >= min_range_m + 1.3 v + 0.1
    # needs headway_s ~1.0 s at 0.6 m/s (0.8 s at 0.4 m/s)
    headway_s: float = 1.0
    max_range_m: float = 5.0  # head near or above the horizon reads as this far
    pitch_comp: float = 0.0  # 0: off. +1/-1 adds telemetry pitch (sign unverified, see recording.PITCH_SIGN)
    # A quadcopter tilts to accelerate (a = g tan(tilt)) and the camera tilts with it: 2.5 deg moves the head row as
    # much as 0.4 m of range at 1.6 m, and the range rate turns that into a false walker speed (a feedback loop).
    # tilt_comp adds back this fraction of the tilt the drone's modelled acceleration needs, at the frame's time.
    # It needs no telemetry sign. A steady tilt (wind, drag) is only removed by pitch_comp (once its sign is known).
    tilt_comp: float = 0.0  # off by default: it helps when the model matches, hurts under dead-time / lag mismatch
    tilt_tau_s: float = 0.15  # attitude-loop lag in that model (the tilt cannot jump)
    auto_resume: bool = True  # after a loss, resume when a walker touching the bottom edge is tracked again
    # prediction (kite/predict.py). It always runs (the overlay draws it); predict=True lets its plan steer
    predict: bool = False  # opt in: --follow-predict (kite.record), --predict (rehearsal)
    horizon_s: float = 1.5  # predicted walker path and planned drone path
    meas_px: float = 8.0  # head-top / box-centre noise of a detection (px), for the filter
    min_updates: int = 5  # detections in the filter before its plan is trusted
    max_std_m: float = 0.8  # predicted position 1-sigma at the horizon above this -> reactive controller instead
    yaw_deg_s: float = 1.0  # deg/s per yaw stick unit (dead reckoning when telemetry has no yaw; the planner)
    walker_accel: float = 0.8  # m/s^2: how sharply the filter lets the walker change speed (higher: quicker, noisier)
    gap_s: float = 0.4  # standoff grows by walker speed x this: braking room for a stop the video shows late


# ------------------------------------------------------------------ detection
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


Detector = Callable[[np.ndarray], list[Detection]]


class YoloPersonDetector:
    """Person boxes from ultralytics YOLO (COCO class 0). Weights must already be on disk (FETCH_HINT)."""

    def __init__(self, weights: str | Path | None = None, imgsz: int = 640, conf: float = 0.2,
                 device: str | None = None):
        path = Path(weights or os.environ.get("KITE_YOLO_WEIGHTS") or DEFAULT_WEIGHTS)
        if not path.is_file():
            raise FileNotFoundError(f"no detector weights at {path}; {FETCH_HINT}")
        os.environ.setdefault("YOLO_OFFLINE", "1")  # no update checks / downloads at runtime
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(f"ultralytics is not installed; {FETCH_HINT}") from None
        self.model = YOLO(str(path))
        if device is None:
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device, self.imgsz, self.conf = device, imgsz, conf
        self.weights = path

    def __call__(self, image: np.ndarray) -> list[Detection]:
        r = self.model.predict(image, classes=[0], conf=self.conf, imgsz=self.imgsz, device=self.device,
                               verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        return [Detection(*map(float, b), float(c)) for b, c in zip(xyxy, conf)]


def detector_available(weights: str | Path | None = None) -> str | None:
    """None if YoloPersonDetector can load, else why not (checked before the drone connects)."""
    import importlib.util

    path = Path(weights or os.environ.get("KITE_YOLO_WEIGHTS") or DEFAULT_WEIGHTS)
    if importlib.util.find_spec("ultralytics") is None:
        return f"ultralytics is not installed; {FETCH_HINT}"
    if not path.is_file():
        return f"no detector weights at {path}; {FETCH_HINT}"
    return None


# ------------------------------------------------------------------ tracking
@dataclass
class Track:
    box: tuple[float, float, float, float]  # smoothed x1, y1, x2, y2 (px)
    conf: float  # smoothed detection confidence
    t: float  # time of the frame the walker was last matched in
    size: tuple[int, int]  # frame width, height
    hits: int = 1
    misses: int = 0  # frames since the last match
    touches_bottom: bool = True
    min_hits: int = 3

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.min_hits

    def age(self, now: float) -> float:
        return now - self.t


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class WalkerTracker:
    """One walker among the person boxes. Acquire: a box touching the bottom edge, nearest the frame's centre column.
    Keep: the box overlapping / nearest the previous one, within gate_frac of the frame width. Misses keep the
    track (stale) for keep_s; FollowController decides what a stale track means."""

    def __init__(self, cfg: FollowCfg | None = None):
        self.cfg = cfg or FollowCfg()
        self.track: Track | None = None

    def reset(self) -> None:
        self.track = None

    def _touches(self, d: Detection, h: int) -> bool:
        return d.y2 >= h - self.cfg.edge_px * h / 720.0

    def update(self, dets: list[Detection], t: float, width: int, height: int) -> Track | None:
        c = self.cfg
        cands = [d for d in dets if d.conf >= c.min_conf and d.x2 > d.x1 and d.y2 > d.y1]
        tr = self.track
        if tr is not None and t - tr.t > c.keep_s:
            tr = self.track = None
        best = None
        if tr is not None:
            px, gate = (tr.box[0] + tr.box[2]) / 2, c.gate_frac * width
            scored = []
            for d in cands:
                dist = abs((d.x1 + d.x2) / 2 - px)
                iou = _iou(d.box, tr.box)
                if iou > 0.05 or dist < gate:
                    scored.append((iou + 0.2 * self._touches(d, height) - dist / width, d))
            if scored:
                best = max(scored, key=lambda s: s[0])[1]
        else:
            edge = [d for d in cands if self._touches(d, height)]
            if edge:
                best = min(edge, key=lambda d: (abs((d.x1 + d.x2) / 2 - width / 2), -(d.x2 - d.x1)))
        if best is None:
            if tr is not None:
                tr.misses += 1
            return self.track
        if tr is None:
            self.track = Track(best.box, best.conf, t, (width, height), min_hits=c.min_hits)
        else:
            a = c.smooth
            tr.box = tuple(a * n + (1 - a) * o for n, o in zip(best.box, tr.box))
            tr.conf = a * best.conf + (1 - a) * tr.conf
            tr.t, tr.size = t, (width, height)
            tr.hits += 1
            tr.misses = 0
            tr.touches_bottom = self._touches(best, height)
        return self.track


# ------------------------------------------------------------------ control
@dataclass
class FollowCmd:
    lr: int = 0
    fb: int = 0
    ud: int = 0
    yaw: int = 0
    state: str = "off"  # following | coast | outpaced | lost | off
    range_m: float | None = None  # horizontal distance to the walker's head (estimate, as measured in the frame)
    range_now_m: float | None = None  # the same carried forward to now (lag compensation)
    warning: str | None = None  # outpaced: what to tell the pilot / walker
    guard: str | None = None  # which head-guard rule cut fb or ud this tick (None: none)
    bearing_deg: float | None = None  # + = walker to the right
    walker_m_s: float | None = None  # estimated walker speed (feed-forward)
    height_err_m: float | None = None
    planner: str | None = None  # predict | reactive: which one steered (while following)
    # prediction for display, in the drone's frame now: (forward m, right m) points
    walker_path: np.ndarray | None = None
    walker_std: np.ndarray | None = None
    drone_path: np.ndarray | None = None
    trail: np.ndarray | None = None

    @property
    def rc(self) -> tuple[int, int, int, int]:
        return (self.lr, self.fb, self.ud, self.yaw)


def _db(x: float, band: float) -> float:
    """Deadband with no jump at its edge."""
    return 0.0 if abs(x) <= band else x - math.copysign(band, x)


def scaled_camera(cam: Camera, width: int, height: int) -> Camera:
    """The same field of view at another frame size (a resized clip, a test frame)."""
    if (width, height) == (cam.width, cam.height):
        return cam
    sx, sy = width / cam.width, height / cam.height
    return Camera(width, height, cam.fx * sx, cam.fy * sy, cam.cx * sx, cam.cy * sy, cam.source)


class YawSource(Protocol):
    """Replaces the yaw law while following (kite.fly.steer.FlySteer). Called every control step with a fresh track:
    the walker's bearing (deg, + right) and range (m) measured on the frame that arrived at frame_t. Returns the yaw
    stick before the clamps and slew limit; `fallback` is the law's own value (returned if the source fails).
    It runs on the control loop and nothing times it out: keep a call to a few ms (FlySteer: ~3 ms, under 6 ms worst)."""

    def reset(self) -> None: ...

    def yaw(self, t: float, frame_t: float, bearing_deg: float, range_m: float, fallback: float) -> float: ...


class FollowController:
    """Track + telemetry -> rc. Call update() at the control rate (kite.record: 10 Hz); it is rate independent.
    yaw_source (optional, reactive mode only) decides the yaw stick while following; coasting and lost handling stay
    the same. If it raises or returns a non-finite value, the yaw law takes over for good (yaw_source_error says why)."""

    def __init__(self, cfg: FollowCfg | None = None, camera: Camera | None = None, max_rc: int = 100,
                 route: np.ndarray | None = None, open_loop: bool = False, yaw_source: YawSource | None = None):
        self.cfg = cfg or FollowCfg()
        self.camera = camera or load_camera()
        self.max_rc = max_rc
        self.route = route  # route polyline in sim metres (kite.predict.load_route), with walker_xy: the prior
        # open_loop (rehearsal on a file): the rc is never flown, so the camera moved only as it did in the video and
        # the dead reckoning ignores the sticks. ponytail: a flown take could dead-reckon on telemetry vgx/vgy instead
        self.open_loop = open_loop
        if yaw_source is not None and self.cfg.predict:  # the planner owns yaw in predict mode; never mix the two
            raise ValueError("a yaw source and FollowCfg.predict are exclusive: pick one")
        for name in ("behind_m", "video_lag_s"):  # a source built for another geometry would steer on wrong numbers
            v = getattr(yaw_source, name, None)
            if v is not None and abs(float(v) - getattr(self.cfg, name)) > 1e-9:
                raise ValueError(f"yaw source {name} {v} != follow cfg {name} {getattr(self.cfg, name)}: "
                                 f"build it from the controller's FollowCfg")
        self.yaw_source = yaw_source
        self.yaw_source_error: str | None = None
        self.reset()

    def _drop_yaw_source(self, e: Exception) -> None:
        self.yaw_source_error = f"{type(e).__name__}: {e}"
        self.yaw_source = None

    def _reset_yaw_source(self) -> None:
        if self.yaw_source is not None:
            try:
                self.yaw_source.reset()
            except Exception as e:  # a broken source must never stop following: the law takes over
                self._drop_yaw_source(e)

    def _yaw(self, t: float, track: Track, bearing: float, rng: float, law: float) -> float:
        if self.yaw_source is None:
            return law
        try:
            y = float(self.yaw_source.yaw(t, track.t, bearing, rng, law))
            if not math.isfinite(y):
                raise ValueError(f"yaw source returned {y}")
            return y
        except Exception as e:
            self._drop_yaw_source(e)
            return law

    def reset(self, height_target_m: float | None = None) -> None:
        self._reset_yaw_source()
        self.height_target = self.cfg.height_m if self.cfg.height_m is not None else height_target_m
        self.state = "following"
        self._t: float | None = None
        self._out = np.zeros(4)  # last command, float, lr fb ud yaw
        self._v_cmd = 0.0  # modelled forward speed of the drone (m/s)
        self._a_cmd = 0.0  # modelled forward acceleration (m/s^2): its tilt is atan(a / g)
        self._v_hist: deque = deque(maxlen=64)  # (t, modelled speed): the speed when the frame was taken
        self._fb_hist: deque = deque(maxlen=64)  # (t, fb sent): the model's input, dead_s later
        self._a_hist: deque = deque(maxlen=256)  # (t, modelled forward acceleration m/s^2): the tilt at a frame time
        self._v_rel = 0.0  # d(range)/dt, smoothed
        self._v_rel_fast = 0.0  # the same, smoothed over stop_tau_s
        self._last_meas: tuple[float, float] | None = None  # (frame t, range)
        self._i_fb = 0.0  # integral of the range error (stick units)
        self._slow_t: float | None = None  # outpaced: since when the walker has looked slow enough to resume
        self._sat: tuple[float, float] | None = None  # (since when fb has been saturated far behind, range then)
        self._rng_s: float | None = None  # smoothed range, for the progress test
        self._slow_rng = 0.0  # outpaced: the smoothed range when the walker first looked slow
        # prediction: dead-reckoned drone pose (x, y, heading) from the sticks sent and telemetry yaw
        self.filter = WalkerFilter(accel=self.cfg.walker_accel)
        self.planner = DronePlanner(PlannerCfg(stick_m_s=self.cfg.stick_m_s, response_s=self.cfg.response_s,
                                               yaw_deg_s=self.cfg.yaw_deg_s, slew=self.cfg.slew,
                                               behind_m=self.cfg.behind_m, min_range_m=self.cfg.min_range_m,
                                               k_yaw=self.cfg.k_yaw, db_yaw_deg=self.cfg.db_yaw_deg,
                                               gap_s=self.cfg.gap_s))
        self._pose = np.zeros(3)
        self._vb = np.zeros(2)  # modelled forward, right speed (m/s)
        self._pose_hist: deque = deque(maxlen=64)  # (t, x, y, heading)
        self._trail: deque = deque(maxlen=80)  # filtered walker positions, for the standoff point
        self._filter_t: float | None = None
        self.prediction: Prediction | None = None
        self.plan: Plan | None = None

    def target_row(self, cam: Camera | None = None) -> float:
        """Image row where the head top sits at the target distance."""
        cam = cam or self.camera
        return cam.cy + cam.fy * self.cfg.above_m / self.cfg.behind_m

    def measure(self, track: Track, pitch_deg: float | None = None, tilt_deg: float = 0.0) -> tuple[float, float, float]:
        """(range m, bearing deg, sideways offset m) of the walker's head top. tilt_deg: the body's nose-down tilt
        when the frame was taken (the camera is fixed to the body: tilted down, the head top shows higher, i.e.
        further away), added back to the depression."""
        c = self.cfg
        cam = scaled_camera(self.camera, *track.size)
        x1, y1, x2, _ = track.box
        bearing = math.atan2((x1 + x2) / 2 - cam.cx, cam.fx)
        down = math.atan2(y1 - cam.cy, cam.fy)  # depression of the head top below the optical axis
        down += math.radians(tilt_deg)
        if c.pitch_comp and pitch_deg is not None:
            down += c.pitch_comp * math.radians(pitch_deg)
        rng = c.max_range_m if down <= math.atan2(c.above_m, c.max_range_m) else c.above_m / math.tan(down)
        rng = max(0.2, min(c.max_range_m, rng))
        return rng, math.degrees(bearing), rng * math.tan(bearing)

    def _speed_at(self, t: float) -> float:
        for tt, v in reversed(self._v_hist):
            if tt <= t:
                return v
        return self._v_hist[0][1] if self._v_hist else 0.0

    def update(self, track: Track | None, tel: Telemetry | None, t: float, walker_xy=None) -> FollowCmd:
        """walker_xy: the walker's GPS position in sim metres, when known (route prior); else None."""
        c = self.cfg
        dt = 0.1 if self._t is None else max(0.0, min(0.5, t - self._t))
        self._t = t
        self._dead_reckon(tel, t, dt)
        age = math.inf if track is None else track.age(t)
        cmd = FollowCmd()
        if self.state == "lost":
            # resume only on a confirmed walker touching the bottom edge, seen just now
            if c.auto_resume and track is not None and track.confirmed and track.touches_bottom and age <= c.fresh_s:
                self.state = "following"
                self._last_meas = None
                self._slow_t = self._sat = self._rng_s = None
                self._i_fb = 0.0
                self._reset_prediction()
                self._reset_yaw_source()  # its memory of the walker is from before the loss
        elif age > c.lost_s:
            self.state = "lost"
            self._i_fb = 0.0
        elif self.state != "outpaced":
            self.state = "following" if age <= c.fresh_s else "coast"

        if self.state == "lost":
            out = np.zeros(4)  # hover now: never rate limited, never searching
        elif self.state == "coast" or (self.state == "outpaced" and age > c.fresh_s):
            out = self._out * c.coast_decay ** (dt / 0.1)
            if self.state == "outpaced":
                out[0] = out[1] = 0.0
            out[2] = self._ud(tel, cmd)
            out = self._limit(out, dt)
        else:
            # (open loop, on a file: the sticks were never flown, so the model's tilt is not the camera's)
            tilt = 0.0 if self.open_loop else c.tilt_comp * math.degrees(
                math.atan2(self._at(self._a_hist, track.t - c.video_lag_s), 9.81))
            rng, bearing, lateral = self.measure(track, tel.pitch_deg if tel else None, tilt)
            cmd.range_m, cmd.bearing_deg = rng, bearing
            # walker speed = drone speed (modelled, at the frame's time) + rate of change of the range
            if self._last_meas is not None and track.t > self._last_meas[0] + 1e-3:
                dr = (rng - self._last_meas[1]) / (track.t - self._last_meas[0])
                # the head-row range's noise grows with range squared: smooth its rate as much more further out
                k = max(1.0, (rng / c.behind_m) ** 2)
                a = min(1.0, (track.t - self._last_meas[0]) / (c.speed_tau_s * k))
                self._v_rel += a * (max(-1.5, min(1.5, dr)) - self._v_rel)
                a = min(1.0, (track.t - self._last_meas[0]) / (c.stop_tau_s * k))
                self._v_rel_fast += a * (max(-1.5, min(1.5, dr)) - self._v_rel_fast)
            if self._last_meas is None or track.t > self._last_meas[0]:
                self._last_meas = (track.t, rng)
            t_cap = track.t - c.video_lag_s  # when the frame was taken
            # asymmetric: the slower of the two estimates, so a stop shows within ~stop_tau_s and a start is not
            # believed before speed_tau_s (the integral makes up the small low bias this leaves while walking)
            v_walker = max(0.0, min(1.2, self._speed_at(t_cap) + min(self._v_rel, self._v_rel_fast)))
            cmd.walker_m_s = v_walker
            rng_now = rng
            if c.lag_comp:  # the walker kept walking and the drone kept flying since the frame was taken
                rng_now += max(-0.5, min(0.5, v_walker * (t - t_cap) - self._travel(t_cap, t)))
            cmd.range_now_m = rng_now
            self._outpace_check(rng, v_walker, t)
            if self.state == "outpaced":
                # hover in place, keep facing the walker (so it can resume), hold the altitude; never chase
                out = np.array([0.0, 0.0, self._ud(tel, cmd), c.k_yaw * _db(bearing, c.db_yaw_deg)])
                self._i_fb = 0.0
            else:
                # constant time gap (as in adaptive cruise control): braking room for a stop the video shows late
                err = _db(rng_now - (c.behind_m + c.headway_s * v_walker), c.db_range_m)
                fb = c.k_fb * err
                if rng_now >= c.min_range_m:
                    fb += c.ff * v_walker / c.stick_m_s
                lim_fb = min(c.max_fb, self.max_rc)
                if c.k_i_fb and self.state == "following":  # conditional integration: frozen while saturated
                    pushing = (err > 0 and self._out[1] >= lim_fb - 0.5) or (err < 0 and self._out[1] <= -lim_fb + 0.5)
                    if not pushing:
                        self._i_fb = max(-c.i_max, min(c.i_max, self._i_fb + c.k_i_fb * err * dt))
                stopping = v_walker < 0.15 and self._v_rel_fast < -0.15  # the walker stopped: the gap is closing
                if (err < 0 or stopping) and self._i_fb > 0:
                    self._i_fb = 0.0  # no forward push survives inside the standoff or a stop (never fly into it)
                fb += self._i_fb
                yaw = self._yaw(t, track, bearing, rng, c.k_yaw * _db(bearing, c.db_yaw_deg))
                out = np.array([c.k_lr * _db(lateral, c.db_lat_m), fb, self._ud(tel, cmd), yaw])
                cmd.planner = "reactive"
                plan = self._predict(track, rng, lateral, t, walker_xy)
                if c.predict and plan is not None:
                    out[0], out[1], out[3] = plan.lr, plan.fb, plan.yaw
                    cmd.planner = "predict"
            out = self._limit(out, dt)
            out = self._guard(out, min(rng, rng_now), v_walker, cmd)
        if self.state == "outpaced":
            out[0] = out[1] = 0.0  # stop translating at once (like braking, never rate limited)
            v = self._v_walker_last if cmd.walker_m_s is None else cmd.walker_m_s
            cmd.warning = (f"walker {v:.1f} m/s, the drone tops out at {self.top_speed():.2f} m/s: hovering, ask the "
                           "walker to slow down")
        if cmd.walker_m_s is not None:
            self._v_walker_last = cmd.walker_m_s
        self._out = out
        # modelled drone speed, for the feed-forward: the fb sent dead_s ago through a first-order lag
        self._fb_hist.append((t, float(out[1])))
        n = max(1, int(math.ceil(dt / 0.02)))  # sub-steps: the attitude lag is shorter than a control tick
        for k in range(n):
            ts = t - dt + (k + 1) * dt / n
            u = self._at(self._fb_hist, ts - c.dead_s)
            a_want = (u * c.stick_m_s - self._v_cmd) / c.response_s  # the speed loop asks for this ...
            self._a_cmd += (a_want - self._a_cmd) * min(1.0, dt / n / c.tilt_tau_s)  # ... the attitude loop gives this
            self._v_cmd += self._a_cmd * dt / n
            self._a_hist.append((ts, self._a_cmd))
        self._v_hist.append((t, self._v_cmd))
        cmd.lr, cmd.fb, cmd.ud, cmd.yaw = (int(round(v)) for v in out)
        cmd.state = self.state
        if self.state != "lost" and self.prediction is not None:
            self._display(cmd)
        return cmd

    def top_speed(self) -> float:
        """The fastest the drone can follow (m/s): its forward stick limit x stick_m_s."""
        c = self.cfg
        return min(c.max_fb, self.max_rc, 100) * c.stick_m_s

    def _travel(self, t0: float, t1: float) -> float:
        """Modelled forward distance flown between t0 and t1 (speed piecewise constant between control ticks)."""
        if t1 <= t0 or not self._v_hist:
            return 0.0
        d, prev = 0.0, None
        for tt, v in self._v_hist:
            if prev is not None:
                a, b = max(prev[0], t0), min(tt, t1)
                if b > a:
                    d += prev[1] * (b - a)
            prev = (tt, v)
        if t1 > prev[0]:  # since the last tick: the current speed
            d += prev[1] * (t1 - max(prev[0], t0))
        return d

    @staticmethod
    def _at(hist: deque, t: float) -> float:
        """The value of a (t, value) history at time t (the last one at or before t; 0 before it starts)."""
        for tt, v in reversed(hist):
            if tt <= t:
                return v
        return 0.0

    def _outpace_check(self, rng: float, v_walker: float, t: float) -> None:
        """following <-> outpaced. Outpaced when the drone makes no progress for outpace_s: fb saturated (or the
        walker past max_follow_m) with the walker outpace_err_m behind the standoff, and the smoothed range not
        progress_m shorter than when that started (a walker or a headwind it cannot beat). It judges progress, not
        the walker-speed estimate: that one is a derivative of the head row, too noisy past 2 m to decide on.
        Back: the walker within max_follow_m, slower than resume_frac x top speed and the gap no longer growing, for
        resume_s (the drone then closes the gap at its own speed, head guard on). Further away it keeps hovering and
        warning: the head-row range is capped at max_range_m out there, so progress could not be judged."""
        c = self.cfg
        top = self.top_speed()
        a = min(1.0, 0.1 / 0.5)  # ~0.5 s smoothing of the range (per measurement), for the progress test
        self._rng_s = rng if self._rng_s is None else self._rng_s + a * (rng - self._rng_s)
        if self.state == "outpaced":
            # resume once the walker looks slow AND the gap has stopped growing (hovering, a drone that drifts back
            # in the wind while the walker walks on must not resume and fall back again)
            if v_walker >= c.resume_frac * top or rng >= c.max_follow_m:  # too far: the pilot brings it closer
                self._slow_t = None
            elif self._slow_t is None:
                self._slow_t, self._slow_rng = t, self._rng_s
            elif self._rng_s > self._slow_rng + c.progress_m:
                self._slow_t, self._slow_rng = t, self._rng_s  # still pulling away: start the window again
            elif t - self._slow_t >= c.resume_s:
                self.state, self._slow_t, self._sat = "following", None, None
            return
        lim_fb = min(c.max_fb, self.max_rc)
        behind = rng - (c.behind_m + c.headway_s * min(v_walker, top))
        # judged only at full speed: while it accelerates, the nose-down tilt reads as the walker pulling away
        at_speed = self._v_cmd >= 0.9 * top
        stuck = (self._out[1] >= lim_fb - 1 or rng > c.max_follow_m) and behind > c.outpace_err_m and at_speed
        if not stuck:
            self._sat = None
        elif self._sat is None:
            self._sat = (t, self._rng_s)
        elif t - self._sat[0] >= c.outpace_fast_s and self._rng_s - self._sat[1] > c.outpace_grow_m:
            self.state = "outpaced"  # clearly pulling away at full stick: no need to wait the whole window
        elif t - self._sat[0] >= c.outpace_s:
            if self._rng_s > self._sat[1] - c.progress_m:
                self.state = "outpaced"
            else:
                self._sat = (t, self._rng_s)  # gaining on it: start a new window
        if self.state == "outpaced":
            self._slow_t = self._sat = None

    def _guard(self, out: np.ndarray, rng: float, v_walker: float, cmd: FollowCmd) -> np.ndarray:
        """Head guard, after everything else (the planner's command too): never speed up toward the walker's head.
          * forward speed <= the walker's speed + the closing speed it can still brake from (brake_m_s2) over the gap
            to min_range_m, given the whole loop delay (video lag + dead time + response)
          * inside the standoff (closer than behind_m - deadband): fb never increases, and never goes above the
            walker's own feed-forward
          * closer than min_range_m: no forward stick at all, and no descent."""
        c = self.cfg
        out = out.copy()
        why = None
        delay = c.video_lag_s + c.dead_s + c.response_s
        gap = rng - c.min_range_m
        a = c.brake_m_s2
        v_close = 0.0 if gap <= 0 else -a * delay + math.sqrt((a * delay) ** 2 + 2 * a * gap)
        cap = (v_walker + v_close) / c.stick_m_s
        if out[1] > cap:
            out[1], why = cap, "closing speed"
        if rng < c.behind_m - c.db_range_m and out[1] > 0:
            hold = max(0.0, min(self._out[1], c.ff * v_walker / c.stick_m_s))
            if out[1] > hold:
                out[1], why = hold, "inside standoff"
        if rng < c.min_range_m:
            if out[1] > 0:
                out[1], why = 0.0, "too close"
            if out[2] < 0:
                out[2], why = 0.0, "too close"
        cmd.guard = why
        return out

    # -- prediction
    def _reset_prediction(self) -> None:
        self.filter.reset()
        self._trail.clear()
        self._filter_t = None
        self.prediction = self.plan = None

    def _dead_reckon(self, tel: Telemetry | None, t: float, dt: float) -> None:
        """Advance the modelled drone pose over the last dt with the sticks sent then."""
        c = self.cfg
        a = min(1.0, dt / c.response_s)
        target = np.zeros(2) if self.open_loop else np.array([self._out[1], self._out[0]]) * c.stick_m_s
        self._vb += (target - self._vb) * a
        if tel is not None and tel.yaw_deg is not None:
            self._pose[2] = tel.yaw_deg
        elif not self.open_loop:
            self._pose[2] += self._out[3] * c.yaw_deg_s * dt
        f, r = fwd_right(self._pose[2])
        self._pose[:2] += (self._vb[0] * f + self._vb[1] * r) * dt
        self._pose_hist.append((t, *self._pose))

    def _pose_at(self, t: float) -> np.ndarray:
        for p in reversed(self._pose_hist):
            if p[0] <= t:
                return np.array(p[1:])
        return np.array(self._pose_hist[0][1:]) if self._pose_hist else self._pose.copy()

    def _predict(self, track: Track, rng: float, lateral: float, t: float, walker_xy) -> Plan | None:
        """Feed the filter the new detection (at the frame's time), predict the walker, plan the drone.
        Returns the plan when it is trusted, else None (the reactive command stays)."""
        c = self.cfg
        cam = scaled_camera(self.camera, *track.size)
        t_meas = track.t - c.video_lag_s
        if self._filter_t is None or track.t > self._filter_t + 1e-3:
            self._filter_t = track.t
            x, y, h = self._pose_at(t_meas)
            f, r = fwd_right(h)
            z = np.array([x, y]) + rng * f + lateral * r
            s_f = rng * rng / (cam.fy * c.above_m) * c.meas_px + 0.03  # range from the head row
            s_r = rng * c.meas_px / cam.fx + 0.03
            J = np.stack([f, r], 1)
            self.filter.update(t_meas, z, J @ np.diag([s_f ** 2, s_r ** 2]) @ J.T)
            if not self._trail:  # seed the trail on the line walker -> drone, so the standoff starts where we are
                d = np.array([x, y]) - z
                self._trail.append(z + d / max(1e-6, np.hypot(*d)) * c.behind_m)
            xf = self.filter.combined()[0][:2]
            if np.hypot(*(xf - self._trail[-1])) > 0.05:
                self._trail.append(xf)
        turn = None
        if walker_xy is not None and self.route is not None:
            turn = route_turn_ahead(walker_xy, self.route)
        pred = self.prediction = self.filter.predict(t, c.horizon_s, 0.1, turn)
        if pred is None:
            return None
        lim = np.minimum([c.max_lr, c.max_fb, c.max_ud, c.max_yaw], self.max_rc).astype(float)
        self.plan = self.planner.plan(pred, np.array(self._trail)[:-1], self._pose[:2], self._pose[2], self._vb,
                                      self._out, lim)
        ok = pred.n_updates >= c.min_updates and pred.std[-1] <= c.max_std_m
        return self.plan if ok else None

    def _display(self, cmd: FollowCmd) -> None:
        """Prediction and plan in the drone's frame now (forward, right), for the overlay and the log."""
        f, r = fwd_right(self._pose[2])
        body = lambda p: np.stack([(p - self._pose[:2]) @ f, (p - self._pose[:2]) @ r], -1)  # noqa: E731
        cmd.walker_path, cmd.walker_std = body(self.prediction.path), self.prediction.std
        if self._trail:
            cmd.trail = body(np.array(self._trail))
        if self.plan is not None:
            cmd.drone_path = body(self.plan.drone_path)

    def _ud(self, tel: Telemetry | None, cmd: FollowCmd) -> float:
        if self.height_target is None or tel is None or tel.height_m is None:
            return 0.0
        err = self.height_target - tel.height_m
        cmd.height_err_m = err
        return self.cfg.k_ud * _db(err, self.cfg.db_h_m)

    def _limit(self, out: np.ndarray, dt: float) -> np.ndarray:
        c = self.cfg
        lim = np.minimum([c.max_lr, c.max_fb, c.max_ud, c.max_yaw], self.max_rc)
        out = np.clip(np.nan_to_num(out), -lim, lim)
        step = c.slew * max(dt, 1e-3)
        lo, hi = self._out - step, self._out + step
        if self._out[1] > 0:
            lo[1] = min(lo[1], -step)  # braking (less forward stick) is never rate limited
        hi[1] = min(hi[1], max(self._out[1], 0.0) + min(c.slew, c.slew_fb_up) * max(dt, 1e-3))
        return np.clip(out, lo, hi)


# ------------------------------------------------------------------ opt-in yaw sources (kite.fly)
YAW_CHOICES = ("law", "pd", "fly")  # kite.record --follow-yaw, rehearsal --yaw. law: FollowController's own (default)
FLY_HINT = 'pip install -e ".[fly]"'


def fly_available() -> str | None:
    """None if kite.fly's yaw sources can load (scipy, the `fly` extra), else why not. Imports nothing from kite.fly."""
    import importlib.util

    if importlib.util.find_spec("scipy") is None:
        return f"scipy is not installed (kite.fly needs it); one time, on normal internet: {FLY_HINT}"
    return None


def build_yaw_source(name: str, cfg: FollowCfg, params: str | Path | None = None) -> tuple[YawSource | None, str]:
    """--follow-yaw / --yaw: (the yaw source, built from this very cfg instance, and its startup line). law builds
    nothing and never imports kite.fly."""
    if name == "law":
        return None, "yaw source law (the follow controller's own yaw law, the default)"
    try:
        from .fly.steer import describe_yaw_source, make_yaw_source
    except ImportError as e:
        raise ImportError(f"yaw source {name}: kite.fly cannot load ({e}); {FLY_HINT}") from None
    src = make_yaw_source(name, cfg, params)
    return src, describe_yaw_source(src, cfg)


def yaw_source_label(src: YawSource | None) -> str:
    """How logs and the status line name a yaw source: law for none, else its name (pd, fly), arm or class."""
    if src is None:
        return "law"
    return str(getattr(src, "name", None) or getattr(src, "arm", None) or type(src).__name__)


def yaw_source_failure(ctrl: FollowController, src: YawSource | None) -> str | None:
    """Why the configured yaw source `src` is not steering now (None while it is, or when there is none): the
    controller dropped it for good (yaw_source_error), or it fell back to the law on its own until its next reset
    (kite.fly's `failed`, the same as its stats["failed"])."""
    if src is None:
        return None
    return ctrl.yaw_source_error or getattr(src, "failed", None) or None


# ------------------------------------------------------------------ live follow inside kite.record
def _link_error(drone) -> str | None:
    """TelloDrone.link_error behind any wrapper (SafetyGovernor.inner): set while Wi-Fi sends fail."""
    d = drone
    while d is not None:
        if getattr(d, "link_error", None):
            return d.link_error
        d = getattr(d, "inner", None)
    return None


class FollowMode:
    """Tracker thread + controller for kite.record --fly --follow.

    The tracker runs on every new frame from the start (the status line shows the walker before g). start() refuses
    unless flying, not busy / landing / safety-hovering, video is live, and a walker touching the bottom edge is
    tracked. tick() is called by the recorder's control loop while active; it returns the rc to send (the recorder
    sends it through the SafetyGovernor) and cancels itself on any safety action, landing or discrete command.
    log(msg, quiet=False, **extra) records to events.jsonl (quiet: no console line).
    yaw_source (opt in, --follow-yaw pd|fly): passed to the controller. With one, "follow on", every "follow" event
    and the status line name the source steering yaw (law once it fell back, with why), and the fallback is logged
    once when it happens. Without one, the per-tick events and the status line are as before.
    """

    def __init__(self, frames, detector_factory: Callable[[], Detector], cfg: FollowCfg | None = None,
                 camera: Camera | None = None, max_rc: int = 50, log: Callable | None = None,
                 yaw_source: YawSource | None = None):
        self.frames = frames
        self.cfg = cfg or FollowCfg()
        self.tracker = WalkerTracker(self.cfg)
        self.ctrl = FollowController(self.cfg, camera, max_rc, yaw_source=yaw_source)
        self.yaw_source = yaw_source  # as configured (the controller drops a failing one: ctrl.yaw_source is None)
        self.yaw_name = yaw_source_label(yaw_source)
        self._yaw_failure_logged: str | None = None
        self.log = log or (lambda msg, quiet=False, **kw: None)
        self._factory = detector_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.track: Track | None = None
        self.last_frame_t: float | None = None
        self.ready = False
        self.error: str | None = None
        self.active = False
        self.last: FollowCmd = FollowCmd()
        self.det_fps = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="follow-tracker")

    def start_tracker(self) -> FollowMode:
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            detect = self._factory()
        except Exception as e:
            self.error = f"detector failed to load: {e}"
            return
        self.ready = True
        last_seq, n, t_count = None, 0, time.time()
        while not self._stop.is_set():
            f = self.frames.read()
            if f is None or f.seq == last_seq:
                time.sleep(0.003)
                continue
            last_seq = f.seq
            try:
                dets = detect(f.image)
            except Exception as e:  # one bad frame must not kill tracking; a dead detector shows as a lost walker
                self.error = f"detector: {e!r}"
                dets = []
            with self._lock:
                self.track = self.tracker.update(dets, f.t, f.width, f.height)
                self.last_frame_t = f.t
            n += 1
            if time.time() - t_count >= 2.0:
                self.det_fps, n, t_count = n / (time.time() - t_count), 0, time.time()

    def _snapshot(self) -> Track | None:
        with self._lock:
            tr = self.track
            return None if tr is None else Track(tr.box, tr.conf, tr.t, tr.size, tr.hits, tr.misses, tr.touches_bottom,
                                                 tr.min_hits)

    def start(self, drone) -> str | None:
        """Start following; returns why not (None when started)."""
        now = time.time()
        tr = self._snapshot()
        why = None
        if not self.ready:
            why = self.error or "detector still loading"
        elif not drone.flying or drone.busy():
            why = "not flying (take off first)"
        elif getattr(drone, "land_requested", False):
            why = "landing"
        elif getattr(drone, "hovering_for_safety", False):
            why = "safety hover"
        elif _link_error(drone):
            why = f"Wi-Fi down ({_link_error(drone)})"
        elif self.last_frame_t is None or now - self.last_frame_t > 1.0:
            why = "no live video"
        elif tr is None or not tr.confirmed or tr.age(now) > self.cfg.fresh_s or not tr.touches_bottom:
            why = "no walker in view (a person cut off by the bottom edge)"
        if why:
            self.log("follow refused", reason=why)
            return why
        tel = drone.telemetry()
        self.ctrl.reset(tel.height_m)
        self.active = True
        self.last = FollowCmd(state="following")
        yaw = {"yaw_source": "law"} if self.yaw_source is None else {
            **self._yaw_fields(), "yaw_params": getattr(self.yaw_source, "source", None)}
        self.log("follow on", box=[round(v) for v in tr.box], conf=round(tr.conf, 2),
                 height_target=None if self.ctrl.height_target is None else round(self.ctrl.height_target, 2),
                 behind_m=self.cfg.behind_m, **yaw)
        self._log_yaw_fallback()
        return None

    # -- the yaw source (only with one configured)
    def yaw_state(self) -> tuple[str, str | None]:
        """(the yaw source steering now, why the configured one fell back to the law or None)."""
        why = yaw_source_failure(self.ctrl, self.yaw_source)
        return ("law" if why else self.yaw_name), why

    def _yaw_fields(self) -> dict:
        if self.yaw_source is None:
            return {}
        active, why = self.yaw_state()
        return {"yaw_source": active} if why is None else {"yaw_source": active, "yaw_source_error": why}

    def _log_yaw_fallback(self) -> None:
        """Log a fallback once when it happens (again only if it recovered, on a reset, and failed anew)."""
        if self.yaw_source is None:
            return
        _, why = self.yaw_state()
        if why is not None and self._yaw_failure_logged is None:
            self.log(f"follow: yaw source {self.yaw_name} failed -> the follow law steers yaw", reason=why)
        self._yaw_failure_logged = why

    def cancel(self, reason: str) -> None:
        if self.active:
            self.active = False
            self.last = FollowCmd()
            self.log("follow off", reason=reason)

    def toggle(self, drone) -> None:
        if self.active:
            self.cancel("g")
        else:
            self.start(drone)

    def tick(self, drone, action: str = "ok", busy: bool = False) -> tuple[int, int, int, int]:
        """The rc to send now (0s once cancelled)."""
        if not self.active:
            return (0, 0, 0, 0)
        if action != "ok":
            self.cancel(f"safety {action}")
        elif not drone.flying or busy or getattr(drone, "land_requested", False):
            self.cancel("not flying / busy / landing")
        elif _link_error(drone):  # rc cannot reach it; never resume on its own when the link comes back
            self.cancel(f"Wi-Fi down ({_link_error(drone)})")
        if not self.active:
            return (0, 0, 0, 0)
        tr = self._snapshot()
        prev = self.last.state
        cmd = self.ctrl.update(tr, drone.telemetry(), time.time())
        self.last = cmd
        if cmd.state != prev:
            if cmd.state == "lost":
                print("\a", end="", flush=True)  # terminal bell: the pilot is watching the drone, not the screen
                self.log(f"follow: walker lost for {self.cfg.lost_s:g} s -> hover (rc 0), waiting for the walker")
            elif cmd.state == "outpaced":
                print("\a", end="", flush=True)
                self.log(f"follow: OUTPACED -> hover in place (fb, lr 0), still facing the walker: {cmd.warning}",
                         walker_m_s=None if cmd.walker_m_s is None else round(cmd.walker_m_s, 2),
                         top_speed_m_s=round(self.ctrl.top_speed(), 2))
            elif prev == "outpaced" and cmd.state == "following":
                self.log("follow: walker slow enough again -> following")
            elif prev == "lost":
                self.log("follow: walker found again -> following")
        self._log_yaw_fallback()
        self.log("follow", quiet=True, state=cmd.state, rc=list(cmd.rc), conf=None if tr is None else round(tr.conf, 2),
                 box=None if tr is None else [round(v) for v in tr.box],
                 range_m=None if cmd.range_m is None else round(cmd.range_m, 2),
                 bearing_deg=None if cmd.bearing_deg is None else round(cmd.bearing_deg, 1),
                 walker_m_s=None if cmd.walker_m_s is None else round(cmd.walker_m_s, 2),
                 height_err_m=None if cmd.height_err_m is None else round(cmd.height_err_m, 2), planner=cmd.planner,
                 range_now_m=None if cmd.range_now_m is None else round(cmd.range_now_m, 2), guard=cmd.guard,
                 **self._yaw_fields())
        return cmd.rc

    def status(self) -> str:
        if self.error and not self.ready:
            return f"follow ERROR {self.error}"
        if not self.ready:
            return "follow loading"
        tr = self._snapshot()
        now = time.time()
        if tr is None or tr.age(now) > self.cfg.lost_s:
            walker = "no walker"
        else:
            walker = f"walker conf {tr.conf:.2f} box {','.join(str(round(v)) for v in tr.box)}"
            if tr.age(now) > self.cfg.fresh_s:
                walker += f" ({tr.age(now):.1f}s old)"
        yaw = ""
        if self.yaw_source is not None:
            active, why = self.yaw_state()
            yaw = f"yaw {active}" if why is None else f"yaw law ({self.yaw_name} failed: {why[:60]})"
        if not self.active:
            return f"follow off ({walker}, {self.det_fps:.0f} det/s{', ' + yaw if yaw else ''})"
        s = self.last
        line = f"FOLLOW {s.state.upper()} {walker}"
        if s.range_m is not None:
            line += f" r {s.range_m:.1f}m b {s.bearing_deg:+.0f}°"
        if s.state == "outpaced":
            line += f" - TOO FAST: walker slower than {self.ctrl.top_speed():.1f} m/s"
        return f"{line} {yaw}" if yaw else line


# ------------------------------------------------------------------ rehearsal on a file (open loop)
def _iter_source(path: Path):
    """(img, t, telemetry or None, flying or None, fps) for every frame of a kite.record folder or a video file."""
    if path.is_dir():
        from .sources.recording import RecordingSource

        rec = RecordingSource(path)
        dur = float(rec.times[-1] - rec.times[0]) if len(rec) > 1 else 0.0
        fps = (len(rec) - 1) / dur if dur > 0 else 30.0
        try:
            while True:
                ok, img = rec.read()
                if not ok:
                    return
                raw, pose = rec.telemetry, rec.pose or {}
                tel = None
                if raw is not None:
                    h = raw.get("h")
                    tel = Telemetry(t=rec.t, height_m=None if h is None else float(h) / 100.0,
                                    pitch_deg=pose.get("pitch"), yaw_deg=pose.get("heading"))
                yield img, rec.t, tel, pose.get("flying"), fps, rec.camera
        finally:
            rec.release()
    else:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        i = 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    return
                yield img, i / fps, None, None, fps, None
                i += 1
        finally:
            cap.release()


STATE_BGR = {"following": (0, 200, 0), "coast": (0, 200, 255), "lost": (0, 0, 255), "off": (160, 160, 160)}


def draw_overlay(img: np.ndarray, dets: list[Detection], track: Track | None, cmd: FollowCmd,
                 ctrl: FollowController) -> np.ndarray:
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    cam = scaled_camera(ctrl.camera, w, h)
    for d in dets:
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (150, 150, 150), 1)
    row = int(ctrl.target_row(cam))
    cv2.line(out, (0, row), (w, row), (255, 200, 0), 1)  # target head-top row
    cv2.line(out, (int(cam.cx), 0), (int(cam.cx), h), (255, 200, 0), 1)
    col = STATE_BGR.get(cmd.state, (255, 255, 255))
    if track is not None:
        x1, y1, x2, y2 = (int(v) for v in track.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
    text = f"{cmd.state.upper()}  rc lr {cmd.lr:+d} fb {cmd.fb:+d} ud {cmd.ud:+d} yaw {cmd.yaw:+d}"
    if cmd.range_m is not None:
        text += f"  r {cmd.range_m:.2f}m b {cmd.bearing_deg:+.1f}deg"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    c = (w // 2, h // 2)  # stick arrows: yaw/lr sideways, fb up
    cv2.arrowedLine(out, c, (c[0] + 2 * (cmd.yaw + cmd.lr), c[1] - 2 * cmd.fb), col, 2)
    if cmd.walker_path is not None:
        _draw_prediction(out, cmd, ctrl.cfg, cam)
    return out


WALKER_BGR, DRONE_BGR, TRAIL_BGR = (60, 220, 60), (255, 160, 0), (150, 150, 150)


def _draw_prediction(out: np.ndarray, cmd: FollowCmd, cfg: FollowCfg, cam: Camera) -> None:
    """Predicted walker path on the image (at head-top height) and a top-down inset, bottom right: walker trail
    (grey), predicted walker path with its 1-sigma circles (green), planned drone path (blue), 1 m grid, the drone
    at the bottom centre facing up. Points are (forward, right) m in the drone's frame now."""
    import cv2

    h, w = out.shape[:2]
    pts = [(cam.cx + cam.fx * r / f, cam.cy + cam.fy * cfg.above_m / f) for f, r in cmd.walker_path if f > 0.3]
    pts = np.array([p for p in pts if -w < p[0] < 2 * w and -h < p[1] < 2 * h], np.int32)
    if len(pts) > 1:
        cv2.polylines(out, [pts], False, WALKER_BGR, 2)
        cv2.circle(out, (int(pts[-1][0]), int(pts[-1][1])), 6, WALKER_BGR, 2)
    size = int(0.4 * min(w, h))
    scale = size / max(4.0, float(cmd.walker_path[:, 0].max()) + 1.0)  # 4 m across, more if the walker is further
    x0, y0 = w - size - 10, h - size - 10
    roi = out[y0:y0 + size, x0:x0 + size]
    roi[:] = (roi * 0.35).astype(np.uint8)
    org = np.array([size / 2, size - 0.5 * scale])

    def px(p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        return np.stack([org[0] + p[:, 1] * scale, org[1] - p[:, 0] * scale], 1).astype(np.int32)

    for m in range(1, int(size / scale)):
        cv2.line(roi, (0, int(org[1] - m * scale)), (size, int(org[1] - m * scale)), (70, 70, 70), 1)
    if cmd.trail is not None and len(cmd.trail) > 1:
        cv2.polylines(roi, [px(cmd.trail)], False, TRAIL_BGR, 1)
    wp = px(cmd.walker_path)
    for (u, v), sd in zip(wp[::5], cmd.walker_std[::5]):
        cv2.circle(roi, (int(u), int(v)), max(2, int(sd * scale)), WALKER_BGR, 1)
    cv2.polylines(roi, [wp], False, WALKER_BGR, 2)
    if cmd.drone_path is not None:
        cv2.polylines(roi, [px(cmd.drone_path)], False, DRONE_BGR, 2)
    cv2.drawMarker(roi, (int(org[0]), int(org[1])), DRONE_BGR, cv2.MARKER_TRIANGLE_UP, 14, 2)
    cv2.putText(roi, f"{cmd.planner or '-'}  next {cfg.horizon_s:g} s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1)


CSV_COLS = ["index", "t", "flying", "state", "conf", "x1", "y1", "x2", "y2", "n_dets", "range_m", "bearing_deg",
            "walker_m_s", "height_m", "lr", "fb", "ud", "yaw", "planner", "pred_speed", "pred_turn_deg_s", "p_stop"]


def _pred_cols(ctrl: FollowController | None, cmd: FollowCmd) -> list[str]:
    p = None if ctrl is None or cmd.walker_path is None else ctrl.prediction
    return ["", "", ""] if p is None else [f"{p.speed:.2f}", f"{p.turn_deg_s:.1f}", f"{p.p_stop:.2f}"]


def _walker_gps(take: Path) -> dict[int, tuple[float, float]]:
    """frame index -> walker (x, y) sim metres from `python -m kite.gps ... --recording <take>` (gps/frames_gps.csv)."""
    path = take / "gps" / "frames_gps.csv"
    if not path.is_file():
        return {}
    with open(path, newline="") as fh:
        return {int(r["index"]): (float(r["x"]), float(r["y"])) for r in csv.DictReader(fh) if r["x"] and r["y"]}


def rehearse(source: str | Path, detector: Detector, cfg: FollowCfg | None = None, out_csv: str | Path | None = None,
             overlay: str | Path | None = None, preview: bool = False, control_hz: float = 10.0,
             max_rc: int = 50, max_s: float = 0.0, yaw_source: YawSource | None = None) -> dict:
    """Run tracker + controller open loop on every frame of a take or clip; write the rc it WOULD send per frame.
    Follow engages at the first walker touching the bottom edge, like pressing g. Returns a summary.
    The prediction always runs (the overlay draws it); cfg.predict says whether its plan or the reactive rc is
    written. A take with gps/frames_gps.csv (kite.gps) also gets the route prior. yaw_source (opt in, built from this
    cfg; not with cfg.predict) decides yaw while following; the summary says which one and its first fallback."""
    import cv2

    src = Path(source)
    cfg = cfg or FollowCfg()
    gps = _walker_gps(src) if src.is_dir() else {}
    route = None
    if gps:
        from .predict import load_route

        route = load_route()
    if out_csv is None:
        out_csv = src / "follow_dry_run.csv" if src.is_dir() else src.with_suffix(".follow_dry_run.csv")
    tracker = WalkerTracker(cfg)
    ctrl: FollowController | None = None
    cmd, next_tick, t0 = FollowCmd(), None, None
    vw = None
    stats = {"frames": 0, "walker_frames": 0, "following_ticks": 0, "coast_ticks": 0, "lost_ticks": 0,
             "outpaced_ticks": 0, "ticks": 0, "losses": 0, "engaged_at_s": None,
             "yaw_source": yaw_source_label(yaw_source), "yaw_source_error": None}
    try:
        with open(out_csv, "w", newline="") as fh:
            out = csv.writer(fh)
            out.writerow(CSV_COLS)
            for i, (img, t, tel, flying, fps, cam) in enumerate(_iter_source(src)):
                t0 = t if t0 is None else t0
                if max_s and t - t0 > max_s:
                    break
                h, w = img.shape[:2]
                dets = detector(img)
                tr = tracker.update(dets, t, w, h)
                fresh = tr is not None and tr.t == t
                stats["frames"] += 1
                stats["walker_frames"] += fresh
                if ctrl is None and fresh and tr.confirmed and tr.touches_bottom:
                    ctrl = FollowController(cfg, cam or load_camera(), max_rc, route=route, open_loop=True,
                                            yaw_source=yaw_source)
                    ctrl.reset(tel.height_m if tel else None)
                    stats["engaged_at_s"] = round(t - t0, 2)
                    next_tick = t
                if ctrl is not None and t >= next_tick:
                    prev = cmd.state
                    cmd = ctrl.update(tr, tel, t, gps.get(i))
                    stats["yaw_source_error"] = stats["yaw_source_error"] or yaw_source_failure(ctrl, yaw_source)
                    next_tick += 1.0 / control_hz
                    stats["ticks"] += 1
                    stats[f"{cmd.state}_ticks"] += 1
                    stats["losses"] += cmd.state == "lost" and prev != "lost"
                box = tr.box if tr is not None else (None,) * 4
                out.writerow([i, f"{t:.3f}", "" if flying is None else int(bool(flying)), cmd.state,
                              "" if tr is None else f"{tr.conf:.2f}", *("" if v is None else f"{v:.0f}" for v in box),
                              len(dets), "" if cmd.range_m is None else f"{cmd.range_m:.2f}",
                              "" if cmd.bearing_deg is None else f"{cmd.bearing_deg:.1f}",
                              "" if cmd.walker_m_s is None else f"{cmd.walker_m_s:.2f}",
                              "" if tel is None or tel.height_m is None else f"{tel.height_m:.2f}", *cmd.rc,
                              cmd.planner or "", *_pred_cols(ctrl, cmd)])
                if overlay or preview:
                    vis = draw_overlay(img, dets, tr, cmd, ctrl or FollowController(cfg, cam or load_camera(), max_rc))
                    if overlay:
                        if vw is None:
                            vw = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vw.write(vis)
                    if preview:
                        cv2.imshow("kite follow rehearsal (q quits)", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
    finally:
        if vw is not None:
            vw.release()
        if preview:
            cv2.destroyAllWindows()
    stats["csv"] = str(out_csv)
    stats["overlay"] = str(overlay) if overlay else None
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kite.follow",
                                description="Follow-mode rehearsal: the rc follow WOULD send, per frame (no drone).")
    p.add_argument("--source", required=True, help="a kite.record folder or a video file")
    p.add_argument("--out", default=None, help="csv path (default: <take>/follow_dry_run.csv or <clip>.follow_dry_run.csv)")
    p.add_argument("--overlay", action="store_true", help="also write the annotated video next to the csv (.mp4)")
    p.add_argument("--preview", action="store_true", help="show the annotated frames (q quits)")
    p.add_argument("--weights", default=None, help=f"YOLO weights (default {DEFAULT_WEIGHTS.relative_to(ROOT)})")
    p.add_argument("--behind", type=float, default=FollowCfg.behind_m, help="target distance behind the head, m")
    p.add_argument("--above", type=float, default=FollowCfg.above_m, help="camera above the head top, m")
    p.add_argument("--height", type=float, default=None, help="altitude to hold, m (default: height at engage)")
    p.add_argument("--max-s", type=float, default=0.0, help="stop after this many seconds of the source")
    p.add_argument("--predict", action="store_true", help="write the prediction planner's rc instead of the reactive "
                   "controller's (the overlay draws the prediction either way)")
    p.add_argument("--yaw", choices=YAW_CHOICES, default="law", help="what steers yaw while following: law (default, "
                   "the controller's own yaw law), pd (kite.fly's predictive PD) or fly (kite.fly's fruit-fly pursuit "
                   "circuit). pd and fly are experimental, simulation only, and need the 'fly' extra; not with "
                   "--predict")
    p.add_argument("--yaw-params", default=None, metavar="PATH", help="with --yaw pd|fly: a parameter file instead of "
                   "the tuned one (kite/fly/data/pred_pd.json, fly_yaw.json)")
    args = p.parse_args(argv)

    src = Path(args.source)
    if not src.exists():
        p.error(f"no such file or folder: {src}")
    if args.yaw == "law" and args.yaw_params:
        p.error("--yaw-params needs --yaw pd or fly")
    if args.yaw != "law":
        if args.predict:
            p.error(f"--yaw {args.yaw} and --predict are exclusive: the prediction planner steers yaw itself")
        why = fly_available()
        if why:
            p.error(f"--yaw {args.yaw}: {why}")
        if args.yaw_params and not Path(args.yaw_params).is_file():
            p.error(f"--yaw-params: no such file: {args.yaw_params}")
    why = detector_available(args.weights)
    if why:
        print(why)
        return 1
    cfg = FollowCfg(behind_m=args.behind, above_m=args.above, height_m=args.height, predict=args.predict)
    yaw_source, line = build_yaw_source(args.yaw, cfg, args.yaw_params)
    print(line, flush=True)
    out = Path(args.out) if args.out else None
    overlay = None
    if args.overlay:
        base = out or (src / "follow_dry_run.csv" if src.is_dir() else src.with_suffix(".follow_dry_run.csv"))
        overlay = Path(base).with_suffix(".mp4")
    print(f"loading the detector ... (rehearsal on {src})", flush=True)
    det = YoloPersonDetector(args.weights)
    s = rehearse(src, det, cfg, out, overlay, args.preview, max_s=args.max_s, yaw_source=yaw_source)
    n = max(1, s["ticks"])
    print(f"{s['frames']} frames, walker seen in {s['walker_frames']} ({100 * s['walker_frames'] / max(1, s['frames']):.0f}%)")
    if s["engaged_at_s"] is None:
        print("follow never engaged: no person touching the bottom edge for 3 frames")
    else:
        print(f"engaged at {s['engaged_at_s']} s; control ticks: following {100 * s['following_ticks'] / n:.0f}%  "
              f"coast {100 * s['coast_ticks'] / n:.0f}%  lost {100 * s['lost_ticks'] / n:.0f}%  ({s['losses']} losses)")
    if s["yaw_source_error"]:
        print(f"yaw source {s['yaw_source']} fell back to the follow law: {s['yaw_source_error']}")
    print(f"rc per frame -> {s['csv']}" + (f"\noverlay -> {s['overlay']}" if s["overlay"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
