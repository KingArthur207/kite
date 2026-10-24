"""Follow mode: keep the Tello about 0.3 m above and ~1.2 m behind the walker's head, on rc sticks.

    python -m kite.follow --source recordings/<take>              # rehearsal: what follow WOULD send, per frame
    python -m kite.follow --source clip.mp4 --overlay --preview    # any video; overlay mp4 next to the csv
    python -m kite.record --source tello --out recordings --fly --follow   # live: g toggles follow
    python -m kite.record --source tello --out recordings --fly --follow --follow-predict   # steer on the prediction

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
                    (plus feed-forward of the walker's estimated speed), ud holds the altitude, small lr.
                    Deadbands, per-channel clamps, slew limits. Track older than lost_s -> rc 0 (hover); it
                    never searches by moving.
                    Prediction (kite/predict.py, opt in: FollowCfg.predict, --follow-predict / --predict): while
                    following, a walker motion model (IMM Kalman filter) and a short-horizon drone planner replace
                    lr, fb and yaw; the reactive command stays when the filter is not confident yet. Coast, lost
                    (hover), ud, clamps and slew are the same either way.
  FollowMode       the live glue for kite.record: tracker thread, start checks, decisions -> events.jsonl.
Every rc goes through the SafetyGovernor (kite.record sends it), which clamps to SafetyCfg.max_rc.

rc signs (Drone.rc, TelloDrone -> djitellopy send_rc_control(lr, fb, ud, yaw) -> "rc lr fb ud yaw"):
+lr right, +fb forward, +ud up, +yaw clockwise (= turn right; kite.types: heading clockwise positive).

Weights (one time, on normal internet; never fetched at runtime):
    pip install -e ".[follow]"
    mkdir -p data/models && curl -L -o data/models/yolo11n.pt \\
        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
data/models/ is gitignored. KITE_YOLO_WEIGHTS or --weights point elsewhere.
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
from dataclasses import dataclass, field
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
    stick_m_s: float = 0.01  # m/s per stick unit (~1 m/s at stick 100, kite-notes-02 / spec 3)
    response_s: float = 0.4  # drone speed lag (first order), for the speed estimate behind the feed-forward
    video_lag_s: float = 0.25  # DroneCfg.video_lag_s: a frame shows the world this long ago
    speed_tau_s: float = 0.8  # smoothing of the walker speed estimate
    db_yaw_deg: float = 2.0  # deadbands
    db_lat_m: float = 0.1
    db_range_m: float = 0.1
    db_h_m: float = 0.08
    max_yaw: int = 40  # per-channel clamps (the governor's max_rc also applies)
    max_lr: int = 20
    max_fb: int = 100  # follow speed is limited by the governor (SafetyCfg.max_rc): raise that to go faster
    max_ud: int = 30
    slew: float = 150.0  # stick units per second on every channel (rate limit); a hover is never rate limited
    min_range_m: float = 0.7  # closer than this: no feed-forward, back off
    max_range_m: float = 5.0  # head near or above the horizon reads as this far
    pitch_comp: float = 0.0  # 0: off. +1/-1 adds telemetry pitch (sign unverified, see recording.PITCH_SIGN)
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
    state: str = "off"  # following | coast | lost | off
    range_m: float | None = None  # horizontal distance to the walker's head (estimate)
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
    stick before the clamps and slew limit; `fallback` is the law's own value (returned if the source fails)."""

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
        self._v_hist: deque = deque(maxlen=64)  # (t, modelled speed): the speed when the frame was taken
        self._v_rel = 0.0  # d(range)/dt, smoothed
        self._last_meas: tuple[float, float] | None = None  # (frame t, range)
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

    def measure(self, track: Track, pitch_deg: float | None = None) -> tuple[float, float, float]:
        """(range m, bearing deg, sideways offset m) of the walker's head top."""
        c = self.cfg
        cam = scaled_camera(self.camera, *track.size)
        x1, y1, x2, _ = track.box
        bearing = math.atan2((x1 + x2) / 2 - cam.cx, cam.fx)
        down = math.atan2(y1 - cam.cy, cam.fy)  # depression of the head top below the optical axis
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
                self._reset_prediction()
                self._reset_yaw_source()  # its memory of the walker is from before the loss
        elif age > c.lost_s:
            self.state = "lost"
        else:
            self.state = "following" if age <= c.fresh_s else "coast"

        if self.state == "lost":
            out = np.zeros(4)  # hover now: never rate limited, never searching
        elif self.state == "coast":
            out = self._out * c.coast_decay ** (dt / 0.1)
            out[2] = self._ud(tel, cmd)
            out = self._limit(out, dt)
        else:
            rng, bearing, lateral = self.measure(track, tel.pitch_deg if tel else None)
            cmd.range_m, cmd.bearing_deg = rng, bearing
            # walker speed = drone speed (modelled, at the frame's time) + rate of change of the range
            if self._last_meas is not None and track.t > self._last_meas[0] + 1e-3:
                dr = (rng - self._last_meas[1]) / (track.t - self._last_meas[0])
                a = min(1.0, (track.t - self._last_meas[0]) / c.speed_tau_s)
                self._v_rel += a * (max(-1.5, min(1.5, dr)) - self._v_rel)
            if self._last_meas is None or track.t > self._last_meas[0]:
                self._last_meas = (track.t, rng)
            v_walker = max(0.0, min(1.2, self._speed_at(track.t - c.video_lag_s) + self._v_rel))
            cmd.walker_m_s = v_walker
            fb = c.k_fb * _db(rng - c.behind_m, c.db_range_m)
            if rng >= c.min_range_m:
                fb += c.ff * v_walker / c.stick_m_s
            yaw = self._yaw(t, track, bearing, rng, c.k_yaw * _db(bearing, c.db_yaw_deg))
            out = np.array([c.k_lr * _db(lateral, c.db_lat_m), fb, self._ud(tel, cmd), yaw])
            cmd.planner = "reactive"
            plan = self._predict(track, rng, lateral, t, walker_xy)
            if c.predict and plan is not None:
                out[0], out[1], out[3] = plan.lr, plan.fb, plan.yaw
                cmd.planner = "predict"
            out = self._limit(out, dt)
        self._out = out
        # modelled drone speed, for the feed-forward (first-order response to the fb actually sent)
        self._v_cmd += (out[1] * c.stick_m_s - self._v_cmd) * min(1.0, dt / c.response_s)
        self._v_hist.append((t, self._v_cmd))
        cmd.lr, cmd.fb, cmd.ud, cmd.yaw = (int(round(v)) for v in out)
        cmd.state = self.state
        if self.state != "lost" and self.prediction is not None:
            self._display(cmd)
        return cmd

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
        out = np.clip(out, -lim, lim)
        step = c.slew * max(dt, 1e-3)
        return np.clip(out, self._out - step, self._out + step)


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
    """

    def __init__(self, frames, detector_factory: Callable[[], Detector], cfg: FollowCfg | None = None,
                 camera: Camera | None = None, max_rc: int = 50, log: Callable | None = None):
        self.frames = frames
        self.cfg = cfg or FollowCfg()
        self.tracker = WalkerTracker(self.cfg)
        self.ctrl = FollowController(self.cfg, camera, max_rc)
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
        self.log("follow on", box=[round(v) for v in tr.box], conf=round(tr.conf, 2),
                 height_target=None if self.ctrl.height_target is None else round(self.ctrl.height_target, 2),
                 behind_m=self.cfg.behind_m)
        return None

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
            elif prev == "lost":
                self.log("follow: walker found again -> following")
        self.log("follow", quiet=True, state=cmd.state, rc=list(cmd.rc), conf=None if tr is None else round(tr.conf, 2),
                 box=None if tr is None else [round(v) for v in tr.box],
                 range_m=None if cmd.range_m is None else round(cmd.range_m, 2),
                 bearing_deg=None if cmd.bearing_deg is None else round(cmd.bearing_deg, 1),
                 walker_m_s=None if cmd.walker_m_s is None else round(cmd.walker_m_s, 2),
                 height_err_m=None if cmd.height_err_m is None else round(cmd.height_err_m, 2), planner=cmd.planner)
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
        if not self.active:
            return f"follow off ({walker}, {self.det_fps:.0f} det/s)"
        s = self.last
        line = f"FOLLOW {s.state.upper()} {walker}"
        if s.range_m is not None:
            line += f" r {s.range_m:.1f}m b {s.bearing_deg:+.0f}°"
        return line


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
             max_rc: int = 50, max_s: float = 0.0) -> dict:
    """Run tracker + controller open loop on every frame of a take or clip; write the rc it WOULD send per frame.
    Follow engages at the first walker touching the bottom edge, like pressing g. Returns a summary.
    The prediction always runs (the overlay draws it); cfg.predict says whether its plan or the reactive rc is
    written. A take with gps/frames_gps.csv (kite.gps) also gets the route prior."""
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
    stats = {"frames": 0, "walker_frames": 0, "following_ticks": 0, "coast_ticks": 0, "lost_ticks": 0, "ticks": 0,
             "losses": 0, "engaged_at_s": None}
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
                    ctrl = FollowController(cfg, cam or load_camera(), max_rc, route=route, open_loop=True)
                    ctrl.reset(tel.height_m if tel else None)
                    stats["engaged_at_s"] = round(t - t0, 2)
                    next_tick = t
                if ctrl is not None and t >= next_tick:
                    prev = cmd.state
                    cmd = ctrl.update(tr, tel, t, gps.get(i))
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
    args = p.parse_args(argv)

    src = Path(args.source)
    if not src.exists():
        p.error(f"no such file or folder: {src}")
    why = detector_available(args.weights)
    if why:
        print(why)
        return 1
    cfg = FollowCfg(behind_m=args.behind, above_m=args.above, height_m=args.height, predict=args.predict)
    out = Path(args.out) if args.out else None
    overlay = None
    if args.overlay:
        base = out or (src / "follow_dry_run.csv" if src.is_dir() else src.with_suffix(".follow_dry_run.csv"))
        overlay = Path(base).with_suffix(".mp4")
    print(f"loading the detector ... (rehearsal on {src})", flush=True)
    det = YoloPersonDetector(args.weights)
    s = rehearse(src, det, cfg, out, overlay, args.preview, max_s=args.max_s)
    n = max(1, s["ticks"])
    print(f"{s['frames']} frames, walker seen in {s['walker_frames']} ({100 * s['walker_frames'] / max(1, s['frames']):.0f}%)")
    if s["engaged_at_s"] is None:
        print("follow never engaged: no person touching the bottom edge for 3 frames")
    else:
        print(f"engaged at {s['engaged_at_s']} s; control ticks: following {100 * s['following_ticks'] / n:.0f}%  "
              f"coast {100 * s['coast_ticks'] / n:.0f}%  lost {100 * s['lost_ticks'] / n:.0f}%  ({s['losses']} losses)")
    print(f"rc per frame -> {s['csv']}" + (f"\noverlay -> {s['overlay']}" if s["overlay"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
