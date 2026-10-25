"""Follow behind: hover behind and above the person's head, facing their back (continuous rc). run/app.py flies this;
kite/follow.py's FollowController stays for kite.record and the rehearsals.

    mission = FollowMission(drone, Perception(cfg, PoseDetector(), camera), cfg, log)
    mission.begin()                       # right after takeoff: CLIMB -> FOLLOW
    every loop: mission.sense(frame, now) (perception on NEW frames only), then mission.step()

Flow (FollowMission):
    CLIMB      one discrete "up" to altitude_m above the floor (skipped when within 20 cm), then FOLLOW
    FOLLOW     FollowBehind below. Person unseen for 20 s (and following for 20 s) -> REACQUIRE
    REACQUIRE  go to LOOK height (just below head height: from follow height a person closer than ~1.5 m is out of the
               bottom of the frame), then turn 30 deg at a time (0.8 s look after each, 13 turns at most) until the
               person is seen -> CLIMB (one discrete up back to altitude_m) -> FOLLOW; not found -> HOLD
    HOLD       hover until told "follow" (-> REACQUIRE)
Every discrete command resets the tracks afterwards: boxes jump, the lock starts afresh.

Perception, every new frame (YOLO pose: people + 17 COCO keypoints):
    SimpleTracker   track ids across frames: IoU >= 0.2, or a centre jump < 15 % of the diagonal between consecutive
                    frames with a similar size
    TargetLock      ONE person: the largest; confirmed by 3 hits in the detector's last 5 runs; never jumps to another
                    person while the locked one is tracked; lost after 1.5 s unseen (then prefers someone near where
                    they were), released after 4.5 s (then anyone can be locked)
    PersonEstimator range (horizontal distance), cues fused by their expected errors in log space:
                      full_height  whole body in frame: height prior
                      head_elev    top of the head visible and the altitude known: (altitude - person height) /
                                   tan(depression). The cue that does NOT change when the person turns: the one that
                                   matters from above/behind (feet out of frame). person_height_m must be real
                      shoulders    both shoulders visible: shoulder-width prior, corrected by |cos facing|
                      bbox_width   body-width prior (fallback, shrinks when the person turns)
                    range_lo_m: the most conservative cue (used before moving TOWARD the person).
                    facing_deg (0 = back to the camera, +90 = facing image-right, +-180 = facing the camera) from the
                    shoulder vector, its sign from the nose or the ears
    The altitude is Telemetry.floor_altitude_m(): the downward ToF while it agrees with 'h' within 0.2 m (there is
    floor below), else 'h'. The height is held against it and the person's range is read with it.

FollowBehind, every fresh frame with the locked person:
    yaw      turns toward them (bearing, deadband), or an opt-in kite.fly yaw source
    forward  holds the follow distance; moves TOWARD them only when even range_lo_m says it is safe and their box does
             not fill the frame; too close -> back off (and climb a little). Feed-forward: they walk toward us -> back
             away at their speed right away
    up/down  holds altitude_m (above their head)
    lateral  orbits around them when they turn (facing_deg != 0), so the drone ends up behind them
Not seen: keep the last command 0.35 s, hold until 1 s, turn toward where they were last seen until 20 s, hover. Lost
while close (they walk under the camera): back off to the side away from them and climb, up to 2.5 s.

rc signs as everywhere in kite: +lr right, +fb forward, +ud up, +yaw clockwise. Every command goes through the
SafetyGovernor.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .camera import Camera
from .types import Frame, Telemetry

ROOT = Path(__file__).resolve().parent.parent
POSE_WEIGHTS = ROOT / "data" / "models" / "yolo11n-pose.pt"
POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
POSE_HINT = f"one time, on normal internet: mkdir -p data/models && curl -L -o data/models/yolo11n-pose.pt {POSE_URL}"

FLY_PARAMS = Path(__file__).resolve().parent / "flybrain" / "data" / "brains" / "trained" / "FLY-YAW_smooth2_best.json"

RUNNING, SUCCESS, FAILURE = "running", "success", "failure"
log = logging.getLogger("kite.follow_behind")


# ------------------------------------------------------------------ settings
@dataclass
class BehindCfg:
    # The defaults are the settings the follow flew well with on the real Tello.
    # Geometry (the Tello camera is fixed, +-21 deg vertical view): at 1.8 m behind the head and shoulders are in
    # frame (shoulders come into view from ~1.6 m behind); at 1.0 m only the head is.
    distance_m: float = 1.8  # horizontal distance behind the person
    altitude_m: float = 1.9  # hover height above the floor: >= person height + 0.2 m
    yaw_gain: float = 1.2  # rc yaw per degree of bearing error
    yaw_deadband_deg: float = 4.0
    dist_gain: float = 40.0  # rc forward per metre of distance error
    dist_deadband_m: float = 0.1
    alt_gain: float = 60.0  # rc up per metre of altitude error
    alt_deadband_m: float = 0.05
    orbit_gain: float = 0.0  # rc lateral per degree of facing error (to get behind the person); 0 = off, as flown
    # (it was switched off on site while diagnosing a drift to the right; 0.5 turns it on)
    orbit_deadband_deg: float = 25.0
    max_rc_yaw: int = 40
    max_rc_forward: int = 35  # ~0.35 m/s: keeps up with a slow walk
    max_rc_lateral: int = 30
    max_rc_up: int = 30
    max_rc_backoff: int = 50  # backing away from a person who comes closer (still clamped by safety.max_rc)
    approach_rate_mps: float = 0.2  # they come closer faster than this: back away at their speed right away
    min_range_m: float = 0.55  # the most conservative distance cue below this: back off (and never approach)
    max_box_width_frac: float = 0.75  # person box wider than this fraction of the frame: back off
    search_yaw_rc: int = 20  # yaw speed when the person is lost
    # who decides the yaw stick while the person is in view: "fly" (the fruit fly connectome controller below, as
    # flown) or "pid" (yaw_gain / yaw_deadband_deg above). Distance, altitude, orbit and safety are unchanged.
    steering: str = "fly"
    # fruit fly steering (kite.flybrain FlySteer: a 1,446-neuron pursuit circuit from the fruit fly connectome)
    fly_params_path: str = str(FLY_PARAMS)  # trainer best.json: trained, then fine-tuned for a smooth stick
    fly_brain_path: str = ""  # pursuit subgraph .npz; "" = the one named in the params file
    fly_latency_s: float | None = None  # capture -> frame arrival its latency filter predicts across; None = video_lag_s
    video_lag_s: float = 0.25  # DroneCfg.video_lag_s
    fly_max_rc_yaw: int = 40  # clamp on the fly's yaw stick (max_rc_yaw also applies; safety.max_rc last)
    fly_smoothing_ms: float = 0.0  # stick low-pass time constant (ms); 0 = off
    fly_deadband: float = 4.0  # subtracted from |yaw| (stick units)
    fly_slew: float = 300.0  # max stick change per second (0 = off)
    fly_hysteresis: float = 3.0  # the stick only moves when the new value differs by more than this
    fly_forward: bool = False  # the fly also drives forward/back: off (these weights have no trained forward readout)
    # person
    person_height_m: float = 1.70  # SET THIS to the wearer's real height WITH SHOES, to +-2 cm (it sets the distance)
    person_height_sd_m: float = 0.02  # 0.02 once person_height_m is measured (0.05 for an unknown person)
    shoulder_width_m: float = 0.40
    body_width_m: float = 0.48  # bbox width of a person seen from behind
    pitch_sign: int = 1  # telemetry pitch sign so that nose-up is positive (the Tello reports forward tilt as
    # NEGATIVE pitch: 1); 0 = do not use pitch
    camera_fx: float = 1077.0  # px at 960x720, fx = fy: median of calibrate_camera at 1.5, 2 and 3 m on the Tello
    # tracking
    iou_match: float = 0.2
    center_match_frac: float = 0.15
    max_age_s: float = 1.5  # >= lost_after_s, so re-locking works
    confirm_hits: int = 3
    confirm_window: int = 5
    lost_after_s: float = 1.5
    # mission
    min_altitude_m: float = 0.5  # SafetyCfg.min_altitude_m (the LOOK height stays 0.2 m above it)
    max_altitude_m: float = 2.3  # SafetyCfg.max_altitude_m
    lost_reacquire_s: float = 20.0  # FOLLOW -> REACQUIRE after the person is unseen this long

    def check(self) -> None:
        if self.altitude_m < self.person_height_m + 0.2:
            raise ValueError(f"altitude {self.altitude_m} m must be >= the person's height + 0.2 m "
                             f"({self.person_height_m + 0.2:.2f}): too close to head height, the range estimate breaks")
        if self.min_range_m >= self.distance_m:
            raise ValueError(f"the follow distance ({self.distance_m} m) must be more than min_range_m "
                             f"({self.min_range_m} m), the distance it backs off at")
        if self.altitude_m > self.max_altitude_m:
            raise ValueError(f"altitude {self.altitude_m} m is above the safety ceiling {self.max_altitude_m} m")
        if self.max_age_s < self.lost_after_s:
            raise ValueError("max_age_s must be >= lost_after_s (else re-locking a returning person fails)")


# ------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    pitch_deg: float = 0.0  # camera tilt relative to the body, up positive (Tello: ~0)

    @classmethod
    def from_camera(cls, cam: Camera) -> CameraModel:
        return cls(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)

    def for_frame(self, width: int, height: int) -> CameraModel:
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return CameraModel(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, self.pitch_deg)

    def ray(self, x: float, y: float, body_pitch_deg: float = 0.0) -> tuple[float, float, float]:
        """Direction of pixel (x, y) in the drone's LEVEL frame: (forward, right, up), not normalised."""
        r, u = (x - self.cx) / self.fx, -(y - self.cy) / self.fy
        p = math.radians(self.pitch_deg + body_pitch_deg)
        return math.cos(p) - u * math.sin(p), r, math.sin(p) + u * math.cos(p)

    def bearing_deg(self, x: float) -> float:
        """Horizontal angle of pixel column x from the optical axis, + right."""
        return math.degrees(math.atan((x - self.cx) / self.fx))

    def elevation_deg(self, y: float, body_pitch_deg: float = 0.0, x: float | None = None) -> float:
        """Angle of pixel (x, y) above the horizon, + up."""
        f, r, u = self.ray(self.cx if x is None else x, y, body_pitch_deg)
        return math.degrees(math.atan2(u, math.hypot(f, r)))

    def range_from_height(self, pixel_h: float, real_h_m: float, x: float | None = None) -> float | None:
        """Horizontal distance to an upright object of known height from its pixel height."""
        if pixel_h <= 1.0 or real_h_m <= 0:
            return None
        b = 0.0 if x is None else math.radians(self.bearing_deg(x))
        return self.fy * real_h_m / pixel_h / math.cos(b)

    def range_from_width(self, pixel_w: float, real_w_m: float, x: float | None = None) -> float | None:
        if pixel_w <= 1.0 or real_w_m <= 0:
            return None
        b = 0.0 if x is None else math.radians(self.bearing_deg(x))
        return self.fx * real_w_m / pixel_w / math.cos(b)

    def range_to_height(self, x: float, y: float, height_diff_m: float, body_pitch_deg: float = 0.0,
                        min_angle_deg: float = 1.5) -> float | None:
        """Horizontal distance to the point imaged at (x, y), knowing it lies `height_diff_m` below the camera
        (negative = above). None if the ray does not reach that height at a usable angle."""
        f, r, u = self.ray(x, y, body_pitch_deg)
        horiz = math.hypot(f, r)
        if height_diff_m == 0 or u * height_diff_m >= 0:  # the ray goes the wrong way
            return None
        if math.degrees(math.atan2(abs(u), horiz)) < min_angle_deg:
            return None
        return abs(height_diff_m) / abs(u) * horiz


# ------------------------------------------------------------------ detections
@dataclass
class PersonDet:
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    keypoints: np.ndarray | None = None  # (17, 3): x, y, confidence, COCO order
    track_id: int | None = None

    # the camera overlay (run/ui.py) draws x1..y2
    x1 = property(lambda self: self.bbox[0])
    y1 = property(lambda self: self.bbox[1])
    x2 = property(lambda self: self.bbox[2])
    y2 = property(lambda self: self.bbox[3])

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])

    @property
    def w(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def h(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.w * self.h


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pose_available(weights: str | Path | None = None) -> str | None:
    """None if the pose detector can load, else why not."""
    import importlib.util

    if importlib.util.find_spec("ultralytics") is None:
        return 'ultralytics is not installed: pip install -e ".[follow]"'
    p = Path(weights) if weights else POSE_WEIGHTS
    if not p.is_file():
        return f"no pose weights at {p}; {POSE_HINT}"
    return None


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class PoseDetector:
    """YOLO pose: people + 17 COCO keypoints, most confident first."""

    def __init__(self, weights: str | Path | None = None, conf: float = 0.4, imgsz: int = 640, iou: float = 0.5,
                 max_det: int = 50, device: str = "auto"):
        from ultralytics import YOLO

        self.model = YOLO(str(weights or POSE_WEIGHTS))
        names = {int(i): n for i, n in self.model.names.items()}
        self.device = _auto_device() if device == "auto" else device
        self._kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, classes=[i for i, n in names.items() if n == "person"],
                            device=self.device, max_det=max_det, verbose=False)
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), **self._kwargs)  # pay load / first-run latency now

    def __call__(self, image: np.ndarray) -> list[PersonDet]:
        r = self.model.predict(image, **self._kwargs)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kps = None
        if r.keypoints is not None and r.keypoints.data is not None:
            kps = r.keypoints.data.cpu().numpy()  # (N, 17, 3): x, y, conf
        out = [PersonDet("person", float(confs[i]), tuple(float(v) for v in xyxy[i]),
                         None if kps is None else kps[i].astype(np.float32)) for i in range(len(xyxy))]
        out.sort(key=lambda d: -d.conf)
        return out


# ------------------------------------------------------------------ tracking
@dataclass
class Track:
    id: int
    cls: str
    bbox: tuple[float, float, float, float]
    conf: float
    first_t: float
    last_t: float
    hits: int = 1
    seen_frames: list[int] = field(default_factory=list)  # frame indices where it was matched


class SimpleTracker:
    """Gives detections of the same object the same track_id across frames. Greedy, best pairs first: IoU >=
    iou_match, or centre distance below center_match_frac x the image diagonal (a fast yaw shifts the box more than
    its own width)."""

    def __init__(self, iou_match: float = 0.2, center_match_frac: float = 0.15, max_age_s: float = 1.5):
        self.iou_match = iou_match
        self.center_match_frac = center_match_frac
        self.max_age_s = max_age_s
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.frame_index = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_index = 0

    def _score(self, tr: Track, d: PersonDet, diag: float) -> float:
        """Association score in (0, 2]; 0 = no match."""
        o = iou(tr.bbox, d.bbox)
        if o >= self.iou_match:
            return 1.0 + o
        if not tr.seen_frames or tr.seen_frames[-1] != self.frame_index - 1:
            return 0.0  # the centre fallback is for fast motion between consecutive frames only
        tcx, tcy = 0.5 * (tr.bbox[0] + tr.bbox[2]), 0.5 * (tr.bbox[1] + tr.bbox[3])
        dist = math.hypot(d.cx - tcx, d.cy - tcy) / diag
        if dist < self.center_match_frac:
            # also require a similar size so a small far object does not steal a big near one
            ta = (tr.bbox[2] - tr.bbox[0]) * (tr.bbox[3] - tr.bbox[1])
            ratio = min(ta, d.area) / max(ta, d.area, 1e-9)
            if ratio > 0.35:
                return 1.0 - dist / self.center_match_frac
        return 0.0

    def update(self, detections: list[PersonDet], t: float, image_size: tuple[int, int]) -> list[PersonDet]:
        """Returns the detections with track_id set."""
        self.frame_index += 1
        diag = math.hypot(*image_size)
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_t > self.max_age_s]:
            del self.tracks[tid]
        pairs = []
        for di, d in enumerate(detections):
            for tid, tr in self.tracks.items():
                if tr.cls == d.cls:
                    s = self._score(tr, d, diag)
                    if s > 0:
                        pairs.append((s, di, tid))
        pairs.sort(reverse=True)
        used_d, used_t, assign = set(), set(), {}
        for s, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            assign[di] = tid
        out = []
        for di, d in enumerate(detections):
            tid = assign.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, d.cls, d.bbox, d.conf, t, t, 1, [self.frame_index])
            else:
                tr = self.tracks[tid]
                tr.bbox, tr.conf, tr.last_t = d.bbox, d.conf, t
                tr.hits += 1
                tr.seen_frames.append(self.frame_index)
                del tr.seen_frames[:-50]
            out.append(replace(d, track_id=tid))
        return out


def largest(cands: list[PersonDet]) -> PersonDet:
    return max(cands, key=lambda d: d.area)


@dataclass
class LockState:
    cls: str | None = None
    det: PersonDet | None = None  # detection in THIS frame (None if not seen this frame)
    last_det: PersonDet | None = None  # last detection of the locked person (display / memory)
    track_id: int | None = None
    confirmed: bool = False
    lost: bool = True
    unseen_s: float = math.inf
    hits: int = 0


class TargetLock:
    """Pick ONE object of a class, confirm it, stick to it, say when it is lost.

    Candidates: detections of the locked class; `select` picks one when nothing is locked. Once locked, the lock
    follows that track_id only: it does NOT jump to another object while the locked one is still tracked.
    Confirmed: matched in >= confirm_hits of the detector's last confirm_window runs, or one detection with conf >=
    confirm_conf. Lost: not seen for lost_after_s; while lost a new candidate close to the last position is preferred
    for re-locking; after 3 x lost_after_s the lock is released and any candidate can be locked."""

    def __init__(self, tracker: SimpleTracker, confirm_hits: int = 3, confirm_window: int = 5,
                 confirm_conf: float = 0.55, lost_after_s: float = 1.5,
                 select: Callable[[list[PersonDet]], PersonDet] = largest, relock_radius_frac: float = 0.25):
        self.tracker = tracker
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.confirm_conf = confirm_conf
        self.lost_after_s = lost_after_s
        self.select = select
        self.relock_radius_frac = relock_radius_frac
        self.cls: str | None = None
        self.state = LockState()
        self._last_seen_t: float | None = None
        self._confirmed = False
        self._needs_hits = False
        self._runs: deque[int] = deque(maxlen=confirm_window)  # tracker frame indices the detector ran on

    def unseen_s(self, t: float) -> float:
        return math.inf if self._last_seen_t is None else t - self._last_seen_t

    def set_class(self, cls: str | None) -> None:
        if cls != self.cls:
            self.cls = cls
            self.release()

    def release(self) -> None:
        self.state = LockState(cls=self.cls)
        self._last_seen_t = None
        self._confirmed = False
        self._needs_hits = False

    def _hits_ok(self, tid: int | None) -> bool:
        tr = self.tracker.tracks.get(tid)
        if tr is None:
            return False
        first = self._runs[0] if self._runs else self.tracker.frame_index - self.confirm_window + 1
        return len([f for f in tr.seen_frames if f >= first]) >= self.confirm_hits

    def _is_confirmed(self, tid: int, det: PersonDet) -> bool:
        if self._hits_ok(tid):
            self._needs_hits = False
            return True
        if self._needs_hits:  # after a re-lock, one confident frame is not enough
            return False
        return det.conf >= self.confirm_conf

    def _pick(self, cands: list[PersonDet]) -> PersonDet:
        repeated = [d for d in cands if self._hits_ok(d.track_id)]
        return self.select(repeated or cands)

    @staticmethod
    def _jumped(a: PersonDet, b: PersonDet) -> bool:
        size = max(a.w, a.h, 1.0)
        return math.hypot(b.cx - a.cx, b.cy - a.cy) > 2.0 * size

    def update(self, detections: list[PersonDet], t: float, image_size: tuple[int, int],
               ran: bool = True) -> LockState:
        """Call after tracker.update on every frame; `ran` = did the detector run on it. Returns a copy."""
        fi = self.tracker.frame_index
        if self._runs and fi < self._runs[-1]:  # the tracker was reset
            self._runs.clear()
        if ran and (not self._runs or self._runs[-1] != fi):
            self._runs.append(fi)
        return replace(self._update(detections, t, image_size))

    def _update(self, detections: list[PersonDet], t: float, image_size: tuple[int, int]) -> LockState:
        s = self.state
        if self.cls is None:
            return s
        cands = [d for d in detections if d.cls == self.cls]
        det = None
        if s.track_id is not None and not self._confirmed:
            # not committed yet: pick again every frame, so a one-frame false blob cannot block the real target
            # (tracks with repeated hits win over single sightings)
            if cands:
                det = self._pick(cands)
                if det.track_id != s.track_id:
                    s.track_id = det.track_id
        elif s.track_id is not None:
            det = next((d for d in cands if d.track_id == s.track_id), None)
            if det is not None and s.last_det is not None and self._jumped(s.last_det, det):
                self._confirmed, self._needs_hits = False, True  # our id glued onto something far away
            if det is None and cands and s.last_det is not None:
                tr = self.tracker.tracks.get(s.track_id)
                expired = tr is None
                brief = tr is None or len(tr.seen_frames) < self.confirm_hits
                repeated = [d for d in cands if self._hits_ok(d.track_id)]
                size = max(s.last_det.w, s.last_det.h, 1.0)
                diag = math.hypot(*image_size)

                def dist(d):
                    return math.hypot(d.cx - s.last_det.cx, d.cy - s.last_det.cy)

                # re-attach quickly only very close by; after a real loss, anywhere near where it was
                radius = self.relock_radius_frac * diag if s.lost else 3.0 * size
                near = [d for d in cands if dist(d) < radius]
                if brief and repeated:
                    # we had committed to something seen only briefly (a one-frame blob) while another candidate
                    # keeps being detected: take that one
                    det = self._pick(repeated)
                elif near or expired:
                    # back under a new track id near where we lost it, or our track is gone: take the nearest
                    # candidate, and confirm it again with REPEATED hits (not one confident frame)
                    det = min(near or cands, key=dist)
                    self._confirmed, self._needs_hits = False, True
                if det is not None:
                    s.track_id = det.track_id
        elif cands:
            det = self._pick(cands)
            s.track_id = det.track_id
            self._confirmed = False

        if det is not None:
            self._last_seen_t = t
            s.hits += 1
            self._confirmed = self._confirmed or self._is_confirmed(s.track_id, det)
            s.last_det = det
        s.det = det
        s.unseen_s = math.inf if self._last_seen_t is None else t - self._last_seen_t
        s.lost = s.unseen_s > self.lost_after_s or self._last_seen_t is None
        s.confirmed = self._confirmed and not s.lost
        if s.lost and self._last_seen_t is not None and s.unseen_s > 3 * self.lost_after_s:
            # gone for good: free the lock so any new candidate can be picked
            self.release()
            return self._update(detections, t, image_size) if cands else self.state
        return s


# ------------------------------------------------------------------ person geometry
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6
MIN_RANGE_M, MAX_RANGE_M = 0.3, 15.0
BODY_DEPTH_M = 0.28  # chest-to-back incl. arms, for profile views


@dataclass
class PersonObs:
    det: PersonDet
    bearing_deg: float
    elevation_deg: float
    range_m: float | None = None
    range_src: str = ""
    facing_deg: float | None = None  # 0 = back to the camera, +90 = facing image-right, +-180 = facing the camera
    facing_conf: float = 0.0
    range_lo_m: float | None = None  # smallest of the range cues: the safe value before moving toward them


class PersonEstimator:
    def __init__(self, person_height_m: float = 1.75, shoulder_width_m: float = 0.40, body_width_m: float = 0.48,
                 kp_conf: float = 0.4, edge_px: float = 3.0, min_height_diff_m: float = 0.15,
                 height_sd_m: float = 0.05, altitude_sd_m: float = 0.03, pitch_sd_deg: float = 1.0):
        self.person_height_m = person_height_m
        self.pitch_sd_deg = pitch_sd_deg  # uncertainty of the camera's vertical angle (3 deg if pitch is not used)
        self.height_sd_m = height_sd_m  # 0.05 for an unknown person; ~0.02 once the wearer is measured
        self.altitude_sd_m = altitude_sd_m
        self.shoulder_width_m = shoulder_width_m
        self.body_width_m = body_width_m
        self.kp_conf = kp_conf
        self.edge_px = edge_px
        self.min_height_diff_m = min_height_diff_m

    def _kp(self, det: PersonDet, i: int):
        k = det.keypoints
        if k is None or k.shape[0] <= i or k[i, 2] < self.kp_conf:
            return None
        return float(k[i, 0]), float(k[i, 1])

    @staticmethod
    def _ok(r: float | None) -> bool:
        return r is not None and MIN_RANGE_M <= r <= MAX_RANGE_M and math.isfinite(r)

    def facing(self, det: PersonDet, cam: CameraModel, range_hint_m: float | None) -> tuple[float | None, float]:
        """(facing_deg, confidence 0..1), or (None, 0) without both shoulders.
        cos from the signed shoulder vector: seen from behind the person's RIGHT shoulder is on the image right
        (dx > 0), from the front on the left; |dx| shrinks toward profile views. sign from the nose when the face is
        visible (nose right of the shoulder midpoint -> facing right), else from the ears (turning right from a back
        view shows the RIGHT ear)."""
        ls, rs = self._kp(det, L_SH), self._kp(det, R_SH)
        if ls is None or rs is None:
            return None, 0.0
        dx = rs[0] - ls[0]
        if range_hint_m:
            expected = cam.fx * self.shoulder_width_m / range_hint_m
        else:
            expected = 0.83 * det.w  # shoulders ~83 % of a back-view body box
        c = float(np.clip(dx / max(expected, 1e-6), -1.0, 1.0))
        k = det.keypoints
        face = float(np.mean(k[[NOSE, L_EYE, R_EYE], 2]))
        sign, sign_conf = 0.0, 0.0
        nose = self._kp(det, NOSE)
        mid_x = 0.5 * (ls[0] + rs[0])
        if nose is not None and face >= self.kp_conf:
            off = (nose[0] - mid_x) / max(abs(dx), 0.25 * expected, 1.0)
            if abs(off) > 0.08:
                sign, sign_conf = math.copysign(1.0, off), min(1.0, abs(off) * 2)
        if sign == 0.0:
            ear_diff = float(k[R_EAR, 2] - k[L_EAR, 2])
            if abs(ear_diff) > 0.15:
                sign, sign_conf = math.copysign(1.0, ear_diff), min(1.0, abs(ear_diff) * 2)
        mag = math.degrees(math.acos(c))  # 0 (back) .. 180 (front)
        conf = (0.8 if mag < 30 or mag > 150 else 0.3) if sign == 0.0 else 0.5 + 0.5 * sign_conf
        # independent front/back vote from the face: a left/right shoulder label swap would flip the answer by
        # 180 deg with full confidence, so disagreement means "don't trust it"
        if (dx < 0 and face < 0.45) or (dx > 0 and face > 0.6 and mag < 60):
            conf = min(conf, 0.25)
        if not range_hint_m:
            conf = min(conf, 0.3)  # magnitude normalised by the box width: rough
        return (mag if sign == 0.0 else sign * mag), conf

    def ranges(self, det: PersonDet, cam: CameraModel, altitude_m: float | None, body_pitch_deg: float = 0.0,
               facing_deg: float | None = None, facing_ok: bool = True) -> dict[str, float]:
        """facing_ok=False: the facing estimate is not trustworthy, so the shoulder cue is left out."""
        out: dict[str, float] = {}
        x1, y1, x2, y2 = det.bbox
        top_ok = y1 > self.edge_px
        bottom_ok = y2 < cam.height - self.edge_px
        side_ok = x1 > self.edge_px and x2 < cam.width - self.edge_px
        if top_ok and bottom_ok:
            r = cam.range_from_height(det.h, self.person_height_m, det.cx)
            if self._ok(r):
                out["full_height"] = r
        ls, rs = self._kp(det, L_SH), self._kp(det, R_SH)
        if ls is not None and rs is not None and facing_deg is not None and facing_ok:
            px = abs(rs[0] - ls[0])
            cos_f = abs(math.cos(math.radians(facing_deg)))
            if cos_f > 0.6:  # near front/back view: foreshortening is small and correctable
                r = cam.range_from_width(px / cos_f, self.shoulder_width_m, 0.5 * (ls[0] + rs[0]))
                if self._ok(r):
                    out["shoulders"] = r
        if top_ok and altitude_m is not None:
            diff = altitude_m - self.person_height_m
            if abs(diff) >= self.min_height_diff_m:
                r = cam.range_to_height(det.cx, y1, diff, body_pitch_deg)
                if self._ok(r):
                    out["head_elev"] = r
        if side_ok and bottom_ok:  # a box cut by the frame bottom may be narrower than the body
            # the body is ~0.28 m deep: seen in profile the box is much narrower than from behind
            if facing_deg is not None and facing_ok:
                f = math.radians(facing_deg)
                width = math.hypot(self.body_width_m * math.cos(f), BODY_DEPTH_M * math.sin(f))
            else:
                width = self.body_width_m
            r = cam.range_from_width(det.w, width, det.cx)
            if self._ok(r):
                out["bbox_width"] = r
        return out

    def rel_sigma(self, src: str, det: PersonDet, cam: CameraModel, altitude_m: float | None,
                  facing_deg: float | None, body_pitch_deg: float) -> float:
        """Expected relative error (1 sigma) of one range cue in the current geometry."""
        if src == "full_height":
            return 0.07  # people's heights vary ~4 %, box fit ~3 %
        if src == "shoulders":
            cos_f = abs(math.cos(math.radians(facing_deg))) if facing_deg is not None else 0.7
            return 0.10 + 0.5 * (1.0 - cos_f)  # shoulder widths vary ~10 %; worse when turned
        if src == "head_elev":
            diff = abs((altitude_m or 0.0) - self.person_height_m)
            e = math.radians(abs(cam.elevation_deg(det.bbox[1], body_pitch_deg, det.cx)))
            dz = math.hypot(self.height_sd_m, self.altitude_sd_m)  # height prior + altimeter
            # elevation error: pitch residual (or the whole pitch when telemetry pitch is not used) + box edge
            return math.hypot(dz / max(diff, 1e-3),
                              math.radians(self.pitch_sd_deg) / max(math.sin(e) * math.cos(e), 1e-3))
        return 0.2 if facing_deg is not None else 0.35  # bbox width (facing-corrected or not)

    @staticmethod
    def fuse(ranges: dict[str, float], sigmas: dict[str, float]) -> tuple[float | None, str]:
        """Inverse-variance average in log space (errors are multiplicative)."""
        if not ranges:
            return None, ""
        w = {k: 1.0 / sigmas[k] ** 2 for k in ranges}
        tot = sum(w.values())
        r = math.exp(sum(w[k] * math.log(v) for k, v in ranges.items()) / tot)
        return r, "+".join(sorted(ranges, key=lambda k: -w[k]))

    def estimate(self, det: PersonDet, cam: CameraModel, altitude_m: float | None = None,
                 body_pitch_deg: float = 0.0) -> PersonObs:
        """`cam` must match the image the detection came from (cam.for_frame(w, h)). range_src lists the fused cues,
        most trusted first."""
        bearing = cam.bearing_deg(det.cx)
        elevation = cam.elevation_deg(det.cy, body_pitch_deg)
        # pass 1: rotation-independent cues give the range hint that normalises the shoulder vector
        r0 = self.ranges(det, cam, altitude_m, body_pitch_deg)
        s0 = {k: self.rel_sigma(k, det, cam, altitude_m, None, body_pitch_deg) for k in r0}
        hint, _ = self.fuse({k: v for k, v in r0.items() if k in ("full_height", "head_elev")}, s0)
        facing, fconf = self.facing(det, cam, hint)
        # pass 2: shoulders corrected by the facing estimate (only when it is trustworthy), then fuse
        ok = fconf >= 0.5
        r = self.ranges(det, cam, altitude_m, body_pitch_deg, facing, ok)
        sig = {k: self.rel_sigma(k, det, cam, altitude_m, facing if ok else None, body_pitch_deg) for k in r}
        rng, src = self.fuse(r, sig)
        obs = PersonObs(det, bearing, elevation, rng, src, facing, fconf)
        # conservative lower bound (use it before moving TOWARD the person): the smallest cue, and a width bound:
        # with a trusted facing, the facing-corrected body width minus 15 % (people vary); otherwise the narrowest
        # a person can look (0.28 m deep, whatever way they face)
        lows = list(r.values())
        x1, _, x2, _ = det.bbox
        if x1 > self.edge_px and x2 < cam.width - self.edge_px:
            if ok and facing is not None:
                f = math.radians(facing)
                width = 0.85 * math.hypot(self.body_width_m * math.cos(f), BODY_DEPTH_M * math.sin(f))
            else:
                width = BODY_DEPTH_M
            wb = cam.range_from_width(det.w, width, det.cx)
            if wb is not None:
                lows.append(wb)
        obs.range_lo_m = min(lows) if lows else None
        return obs


# ------------------------------------------------------------------ perception
@dataclass
class PersonResult:
    t: float
    seq: int
    image_size: tuple[int, int]
    persons: list[PersonObs] = field(default_factory=list)
    person: PersonObs | None = None  # the locked person, when seen in this frame
    person_confirmed: bool = False
    person_unseen_s: float = math.inf


class Perception:
    """One frame (+ telemetry) in, one PersonResult out: pose detector -> tracker -> lock -> estimator."""

    def __init__(self, cfg: BehindCfg, detect: Callable[[np.ndarray], list[PersonDet]], camera: CameraModel):
        self.cfg = cfg
        self.detect = detect
        self.camera = camera
        self.tracker = SimpleTracker(cfg.iou_match, cfg.center_match_frac, cfg.max_age_s)
        self.lock = TargetLock(self.tracker, cfg.confirm_hits, cfg.confirm_window, 0.99, cfg.lost_after_s,
                               select=largest)
        self.lock.set_class("person")
        # without telemetry pitch (pitch_sign 0) the head-elevation cue must be trusted less
        self.estimator = PersonEstimator(cfg.person_height_m, cfg.shoulder_width_m, cfg.body_width_m,
                                         height_sd_m=cfg.person_height_sd_m,
                                         pitch_sd_deg=1.0 if cfg.pitch_sign else 3.0)
        self.last: PersonResult | None = None

    def reset_tracks(self) -> None:
        """After a big discontinuity (a discrete move or turn) old boxes cannot be associated anyway."""
        self.tracker.reset()
        self.lock.release()

    def update(self, frame: Frame, tel: Telemetry | None = None) -> PersonResult:
        if self.last is not None and frame.seq == self.last.seq and frame.t == self.last.t:
            return self.last  # already processed: a repeated frame must not count as a new sighting
        cam = self.camera.for_frame(frame.width, frame.height)
        tel = tel or Telemetry(frame.t)
        altitude = tel.floor_altitude_m()  # person range needs height above the FLOOR, not above a table
        pitch = (tel.pitch_deg or 0.0) * self.cfg.pitch_sign
        size = (frame.width, frame.height)
        dets = self.tracker.update(self.detect(frame.image), frame.t, size)
        res = PersonResult(frame.t, frame.seq, size)
        res.persons = [self.estimator.estimate(d, cam, altitude, pitch) for d in dets]
        ls = self.lock.update(dets, frame.t, size)
        if ls.det is not None:
            res.person = next((p for p in res.persons if p.det.track_id == ls.det.track_id), None)
        res.person_confirmed = ls.confirmed
        res.person_unseen_s = self.lock.unseen_s(frame.t)
        self.last = res
        return res


# ------------------------------------------------------------------ behaviours
def clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def deadband(v: float, db: float) -> float:
    return 0.0 if abs(v) < db else v - math.copysign(db, v)


@dataclass
class Ctx:
    """What a behaviour sees each step."""

    drone: object  # kite Drone (the SafetyGovernor)
    perception: Perception
    now: float = 0.0
    frame: Frame | None = None
    new_frame: bool = False  # `frame` arrived this loop (perception ran on it)
    res: PersonResult | None = None
    tel: Telemetry | None = None

    @property
    def altitude(self) -> float | None:
        """Height above the FLOOR (holding altitude)."""
        return None if self.tel is None else self.tel.floor_altitude_m()


_FLY_SHARED: dict = {}  # one FlySteer per settings per process: the brain loads once
_FLY_WARNED: set = set()


def _fly_warn_once(key: str, msg: str, warn: Callable[[str], None] | None) -> None:
    if key not in _FLY_WARNED:
        _FLY_WARNED.add(key)
        (warn or log.warning)(msg)


class FlyYaw:
    """Per-behaviour handle on the shared FlySteer. Angles in degrees here (+ = right). The fly decides ONLY the yaw
    stick; if it cannot load or raises, it warns once and the behaviour uses its own yaw law instead."""

    def __init__(self, steer, max_rc_yaw: int, fwd_enabled: bool = False, warn: Callable[[str], None] | None = None):
        self.steer = steer
        self.max_rc_yaw = max_rc_yaw
        self.fwd_enabled = fwd_enabled
        self.failed = False
        self.warn = warn

    @staticmethod
    def load(cfg: BehindCfg):
        """The shared FlySteer for these settings (loads the brain the first time)."""
        latency = cfg.video_lag_s if cfg.fly_latency_s is None else cfg.fly_latency_s
        key = (cfg.fly_params_path, cfg.fly_brain_path, latency, cfg.fly_smoothing_ms, cfg.fly_deadband, cfg.fly_slew,
               cfg.fly_hysteresis)
        steer = _FLY_SHARED.get(key)
        if steer is None:
            from .flybrain.flyfollow.steer import FlySteer

            steer = FlySteer(params_path=cfg.fly_params_path or None, brain_path=cfg.fly_brain_path or None,
                             max_yaw=cfg.fly_max_rc_yaw, latency_s=latency, viz=False,
                             smoothing_ms=cfg.fly_smoothing_ms, deadband=cfg.fly_deadband, slew=cfg.fly_slew,
                             hysteresis=cfg.fly_hysteresis)
            _FLY_SHARED[key] = steer
            log.info("fly steering loaded: %s", steer.stats)
        return steer

    @classmethod
    def for_behavior(cls, cfg: BehindCfg, kind: str, z_ref_m: float,
                     warn: Callable[[str], None] | None = None) -> FlyYaw | None:
        """A reset FlyYaw if cfg.steering is "fly" and the fly loads, else None (use the yaw law)."""
        if cfg.steering != "fly":
            return None
        try:
            steer = cls.load(cfg)
            steer.reset(z_ref_m=z_ref_m, kind=kind)  # fresh brain state + 1 s warmup (~35 ms)
        except Exception as e:  # any failure: keep flying on the yaw law
            _fly_warn_once(f"load{cfg.fly_params_path}", f"fly steering unavailable ({type(e).__name__}: {e}); "
                           "using the yaw law", warn)
            return None
        return cls(steer, cfg.fly_max_rc_yaw, fwd_enabled=(kind == "follow" and cfg.fly_forward), warn=warn)

    @property
    def target_valid(self) -> bool:
        """The fly still has a target (a detection in the last 0.5 s)."""
        return (not self.failed) and bool(self.steer.target_valid)

    def yaw(self, now: float, bearing_deg: float | None = None, range_m: float | None = None,
            frame_t: float | None = None, fallback: float = 0.0) -> float:
        """Yaw stick (+ = clockwise). Pass the bearing only for a detection on a NEW frame; call with no bearing on
        other loops so the brain keeps ticking. Returns `fallback` if the fly has failed. range_m sets the size the
        fly sees (z_ref / range); None = its standoff size, which follow uses."""
        if self.failed:
            return fallback
        try:
            b = None if bearing_deg is None else math.radians(bearing_deg)
            y = self.steer.yaw(now, b, range_m=range_m, t_frame=frame_t)
            return float(max(-self.max_rc_yaw, min(self.max_rc_yaw, y)))
        except Exception as e:
            self.failed = True
            _fly_warn_once(f"run{id(self)}", f"fly steering failed ({type(e).__name__}: {e}); back to the yaw law",
                           self.warn)
            return fallback

    def forward(self, fallback: float = 0.0) -> float:
        """The fly's forward/back stick from the last brain tick (fly_forward only; the caller clamps and gates it)."""
        if self.failed or not self.fwd_enabled:
            return fallback
        try:
            return float(self.steer.forward)
        except Exception as e:
            self.failed = True
            _fly_warn_once(f"run{id(self)}", f"fly forward failed ({type(e).__name__}: {e}); back to the laws",
                           self.warn)
            return fallback


class FollowBehind:
    status = ""
    fly: FlyYaw | None = None  # the fruit fly yaw (steering "fly"), set in start(); None = the yaw law

    def __init__(self, cfg: BehindCfg, yaw_source=None, warn: Callable[[str], None] | None = None):
        self.c = cfg
        self.warn = warn
        self.yaw_source = yaw_source  # opt in: a kite.fly yaw source replaces only the yaw while the person is seen
        self.yaw_source_error: str | None = None
        self.ranges: deque = deque(maxlen=5)  # (t, range): only recent ones count, never stale values
        self.facings: deque = deque(maxlen=5)
        self.range_hist: deque = deque(maxlen=30)  # (t, filtered range) for the approach-rate feed-forward
        self.fwd_hist: deque = deque(maxlen=60)  # (t, commanded forward rc)
        self.last_close = False
        self.cmd = (0, 0, 0, 0)
        self.last_seen_t: float | None = None
        self.last_bearing = 0.0

    def start(self, ctx: Ctx) -> None:
        self.ranges.clear()
        self.facings.clear()
        self.range_hist.clear()
        self.fwd_hist.clear()
        self.last_close = False
        self.cmd = (0, 0, 0, 0)
        self.last_seen_t = None
        self.fly = None if self.yaw_source is not None else FlyYaw.for_behavior(self.c, "follow", self.c.distance_m,
                                                                                self.warn)
        if self.yaw_source is not None:
            try:
                self.yaw_source.reset()
            except Exception as e:
                self.yaw_source_error, self.yaw_source = f"{type(e).__name__}: {e}", None

    def _source_yaw(self, ctx: Ctx, bearing: float, rng: float | None, fallback: float) -> float:
        if self.yaw_source is None:
            return fallback
        try:
            return float(self.yaw_source.yaw(ctx.now, ctx.frame.t, bearing,
                                             rng if rng is not None else self.c.distance_m, fallback))
        except Exception as e:  # a failing yaw source never stops the follow: the law steers from now on
            self.yaw_source_error, self.yaw_source = f"{type(e).__name__}: {e}", None
            return fallback

    def _range(self, now: float) -> float | None:
        """Median of the range estimates from the last 0.7 s (None when there are none)."""
        recent = [r for t, r in self.ranges if now - t <= 0.7]
        return statistics.median(recent) if recent else None

    def _range_rate(self, now: float) -> float | None:
        """Slope of the filtered range over the last ~1 s (m/s, negative = they come closer). None unless the
        history is recent and gap-free: a rate across a dropout is meaningless."""
        h = [x for x in self.range_hist if now - x[0] <= 1.0]
        if len(h) < 5 or h[-1][0] - h[0][0] < 0.5 or now - h[-1][0] > 0.3:
            return None
        if any(b[0] - a[0] > 0.4 for a, b in zip(h, h[1:])):
            return None
        ts = [t for t, _ in h]
        rs = [r for _, r in h]
        tm, rm = sum(ts) / len(ts), sum(rs) / len(rs)
        den = sum((t - tm) ** 2 for t in ts)
        return None if den <= 0 else sum((t - tm) * (r - rm) for t, r in zip(ts, rs)) / den

    def _climb_rc(self, ctx: Ctx) -> int:
        """Climb to get clear of a person, at most 0.25 m above the follow altitude (the governor also caps)."""
        alt = ctx.altitude
        return self.c.max_rc_up if (alt is None or alt < self.c.altitude_m + 0.25) else self._altitude_rc(ctx)

    def _altitude_rc(self, ctx: Ctx) -> int:
        alt = ctx.altitude
        if alt is None:
            return 0
        return int(clamp(deadband(self.c.altitude_m - alt, self.c.alt_deadband_m) * self.c.alt_gain, self.c.max_rc_up))

    def step(self, ctx: Ctx) -> str:
        c = self.c
        ud = self._altitude_rc(ctx)
        p = ctx.res.person if (ctx.res is not None and ctx.new_frame) else None
        if p is not None:
            self.last_seen_t, self.last_bearing = ctx.now, p.bearing_deg
            if p.range_m is not None:
                self.ranges.append((ctx.now, p.range_m))
            r = self._range(ctx.now)
            if r is not None:
                self.range_hist.append((ctx.now, r))
            r_lo = min(v for v in (p.range_lo_m, r) if v is not None) if (p.range_lo_m or r) else None
            yaw = clamp(deadband(p.bearing_deg, c.yaw_deadband_deg) * c.yaw_gain, c.max_rc_yaw)
            if self.fly is not None:  # fly steering: replaces only this yaw value (no range: it steers at size s = 1,
                # unless fly_forward is on: then the fly needs the true size to judge distance)
                rng = r if self.fly.fwd_enabled else None
                yaw = clamp(self.fly.yaw(ctx.now, p.bearing_deg, rng, ctx.frame.t, fallback=yaw), c.max_rc_yaw)
            else:
                yaw = clamp(self._source_yaw(ctx, p.bearing_deg, r, yaw), c.max_rc_yaw)
            fwd, approaching = 0.0, False
            width, height = ctx.res.image_size
            too_big = p.det.w > c.max_box_width_frac * width
            if r is not None:
                e = deadband(r - c.distance_m, c.dist_deadband_m)
                if e > 0:  # too far: only approach if even the most conservative cue says we are not too close
                    if r_lo is not None and r_lo > c.min_range_m and not too_big:
                        fwd = clamp(e * c.dist_gain, c.max_rc_forward)
                else:
                    fwd = clamp(e * c.dist_gain, c.max_rc_forward)
                # feed-forward: they walk toward us -> back away at their speed now (rc 100 ~ 1 m/s), don't wait for
                # the distance error to build up (the head leaves the frame at ~0.6 m). Our own forward motion also
                # shrinks the range (and video lags): subtract the speed we commanded so only THEIR motion counts
                rate = self._range_rate(ctx.now)
                own = [f for t, f in self.fwd_hist if 0.3 <= ctx.now - t <= 1.3]
                if rate is not None and own:
                    closing = rate + (sum(own) / len(own)) / 100.0
                    approaching = closing < -c.approach_rate_mps
                    if approaching:
                        fwd = min(fwd, clamp(100.0 * closing - 10, c.max_rc_backoff))
            if self.fly is not None and self.fly.fwd_enabled:
                # full fly following: the fly's forward stick replaces the distance law, under the same gates:
                # approach only when the most conservative range cue says it is safe, keep the feed-forward back-away,
                # and the too-close override below still wins
                f = self.fly.forward(fallback=fwd)
                if f > 0 and not (r_lo is not None and r_lo > c.min_range_m and not too_big):
                    f = 0.0
                if approaching:
                    f = min(f, fwd)
                fwd = max(-float(c.max_rc_backoff), min(float(c.max_rc_forward), f))
            if (r_lo is not None and r_lo < c.min_range_m) or too_big:
                fwd = -c.max_rc_backoff  # too close: back off fast and gain a little height
                ud = max(ud, self._climb_rc(ctx) // 2)
            # coming closer, nearly too close, or the top of their head at the bottom of the frame: if they vanish
            # now, they are walking under us (a plain dropout at the follow distance is not this)
            self.last_close = (approaching or (r is not None and r < c.min_range_m + 0.25)
                               or p.det.bbox[1] >= 0.88 * height)
            lat = 0.0
            if p.facing_deg is not None and p.facing_conf >= 0.5:
                self.facings.append(math.radians(p.facing_deg))
            if len(self.facings) >= 3:
                # circular mean of recent confident estimates: one mislabelled frame cannot swing the orbit
                f = math.degrees(math.atan2(sum(math.sin(a) for a in self.facings),
                                            sum(math.cos(a) for a in self.facings)))
                # facing +90 = they face image-right, so their back is to the image-left: move left
                lat = clamp(-c.orbit_gain * deadband(f, c.orbit_deadband_deg), c.max_rc_lateral)
            self.cmd = (int(lat), int(fwd), ud, int(yaw))
            self.fwd_hist.append((ctx.now, int(fwd)))
            rng = "?" if p.range_m is None else f"{p.range_m:.1f}"
            lo = "?" if r_lo is None else f"{r_lo:.1f}"
            face = "?" if p.facing_deg is None else f"{p.facing_deg:+.0f}"
            self.status = f"person {rng} m (lo {lo}) bearing {p.bearing_deg:+.0f} facing {face}"
            ctx.drone.rc(*self.cmd)
            return RUNNING
        unseen = math.inf if self.last_seen_t is None else ctx.now - self.last_seen_t
        if self.last_close and 0.2 < unseen < 2.5:
            # lost while close: they are walking under the camera's view. Back off, step out of their path (away from
            # the side they were on) and climb; never hover in their way
            side = -1 if self.last_bearing >= 0 else 1
            self.cmd = (side * c.max_rc_backoff, -c.max_rc_backoff, self._climb_rc(ctx), 0)
            ctx.drone.rc(*self.cmd)
            self.status = "person close and out of view: backing off to the side"
        elif unseen < 0.35:  # between frames: keep the last command (the fly keeps ticking)
            yaw = self.cmd[3] if self.fly is None else int(clamp(self.fly.yaw(ctx.now, fallback=self.cmd[3]),
                                                                 c.max_rc_yaw))
            ctx.drone.rc(*self.cmd[:2], ud, yaw)
        elif unseen < 1.0:
            ctx.drone.rc(0, 0, ud, 0)
            self.status = "person briefly lost: holding"
        elif unseen < 20.0 or self.last_seen_t is None:
            direction = 1 if self.last_bearing >= 0 else -1
            ctx.drone.rc(0, 0, ud, direction * c.search_yaw_rc)
            self.status = f"person lost {0 if self.last_seen_t is None else unseen:.0f} s: turning " \
                          f"{'right' if direction > 0 else 'left'}"
        else:
            ctx.drone.rc(0, 0, ud, 0)
            self.status = "person lost: hovering"
        return RUNNING


class Discrete:
    """One discrete drone command (move / rotate): issue once, wait for the reply, reset the tracks after it."""

    def __init__(self, kind: str, value: float, direction: str = ""):
        self.kind, self.value, self.direction = kind, value, direction  # kind: rotate | move
        self.issued = False
        self.result: str | None = None

    def step(self, ctx: Ctx) -> str:
        d = ctx.drone
        if not self.issued:
            if d.busy():
                return RUNNING  # something else still running (should not happen)
            if self.kind == "rotate":
                d.rotate(int(round(self.value)))
            else:
                d.move(self.direction, int(round(self.value)))
            self.issued = True
            if not d.busy():  # refused immediately (e.g. safety)
                self.result = d.last_result()
                return FAILURE if self.result != "ok" else SUCCESS
            return RUNNING
        if d.busy():
            return RUNNING
        self.result = d.last_result()
        ctx.perception.reset_tracks()  # boxes jump after a discrete command: start tracking afresh
        return SUCCESS if self.result == "ok" else FAILURE


class ReacquirePerson:
    """Turn step_deg at a time (looking dwell_s after each) until the person is seen; max_steps turns at most."""

    status = ""

    def __init__(self, step_deg: int = 30, dwell_s: float = 0.8, max_steps: int = 13):
        self.step_deg, self.dwell_s, self.max_steps = step_deg, dwell_s, max_steps

    def start(self, ctx: Ctx) -> None:
        self.cmd: Discrete | None = None
        self.steps = 0
        self.dwell_start = ctx.now

    def step(self, ctx: Ctx) -> str:
        if self.cmd is not None:
            r = self.cmd.step(ctx)
            if r == RUNNING:
                return RUNNING
            self.cmd = None
            self.dwell_start = ctx.now
            return RUNNING
        ctx.drone.rc(0, 0, 0, 0)
        if ctx.now - self.dwell_start < self.dwell_s:
            return RUNNING
        if ctx.new_frame and ctx.res is not None and ctx.res.person is not None:
            self.status = "person found"
            return SUCCESS
        if not ctx.new_frame:
            return RUNNING
        if self.steps >= self.max_steps:
            self.status = "person not found"
            return FAILURE
        self.steps += 1
        self.cmd = Discrete("rotate", self.step_deg)
        self.status = f"looking for the person ({self.steps})"
        return RUNNING


# ------------------------------------------------------------------ mission
@dataclass
class Shown:
    """The locked person for the camera overlay."""

    box: tuple[float, float, float, float]
    t: float  # frame time it was last seen
    confirmed: bool


class FollowMission:
    """CLIMB -> FOLLOW -> (lost 20 s) REACQUIRE -> FOLLOW | HOLD. begin() right after takeoff; sense() then step()
    every loop while flying; stop() before landing. Operator commands: follow_me(), cancel(), hold(), resume().
    log(text, kind) gets every state change and each change of what the behaviour is doing; announce(text) is what
    the drone says to the walker (say(text) when given, e.g. speech)."""

    def __init__(self, drone, perception: Perception, cfg: BehindCfg, log: Callable[[str, str], None] | None = None,
                 yaw_source=None, say: Callable[[str], None] | None = None):
        cfg.check()
        self.cfg = cfg
        self.ctx = Ctx(drone, perception)
        self.log = log or (lambda text, kind="info": None)
        self.say = say
        self.said: list[tuple[float, str]] = []
        self._resume_state: str | None = None
        self.yaw_source = yaw_source
        self.state = "IDLE"
        self.state_t = 0.0
        self.child = None
        self.cmd: Discrete | None = None
        self._said = ""

    # -- sensing
    def sense(self, frame: Frame | None, now: float) -> None:
        """The newest frame (perception runs only on NEW frames) and telemetry."""
        ctx = self.ctx
        ctx.now = now
        ctx.tel = ctx.drone.telemetry()
        ctx.new_frame = frame is not None and (ctx.frame is None or frame.seq != ctx.frame.seq)
        if ctx.new_frame:
            ctx.frame = frame
            ctx.res = ctx.perception.update(frame, ctx.tel)

    # -- states
    def _go(self, state: str, why: str = "", child=None) -> None:
        self.state, self.state_t = state, self.ctx.now
        self.child, self.cmd = child, None
        self._said = ""
        self.log(f"-> {state.lower()}" + (f": {why}" if why else ""), "info")
        if child is not None:
            child.start(self.ctx)

    def _follow(self, why: str) -> None:
        self._go("FOLLOW", why, FollowBehind(self.cfg, self.yaw_source, warn=lambda m: self.log(m, "warn")))

    def begin(self) -> None:
        """Airborne: climb to the follow altitude, then follow."""
        self._go("CLIMB", "airborne")

    def announce(self, text: str) -> None:
        self.said.append((self.ctx.now, text))
        self.log(f"says: {text}", "recv")
        if self.say is not None:
            try:
                self.say(text)
            except Exception:  # speech must never stop the flight
                pass

    def follow_me(self) -> bool:
        """'follow me': look for the person again, from HOLD, or from FOLLOW once they have been unseen for 2 s.
        False when it does not apply (climbing, searching, or the person is in view)."""
        res = self.ctx.res
        lost = res is None or res.person_unseen_s > 2.0
        if not (self.state == "HOLD" or (self.state == "FOLLOW" and lost)):
            return False
        self.ctx.drone.stop()
        self.announce("Coming back to you." if self.state != "FOLLOW" else "Looking for you.")
        self._go("REACQUIRE", "asked to follow")
        return True

    def cancel(self) -> bool:
        """'stop' / 'cancel': from HOLD, stop and look for the person again. Ignored in FOLLOW."""
        if self.state != "HOLD":
            return False
        self.ctx.drone.stop()
        self.announce("Okay, stopping. Coming back to you.")
        self._go("REACQUIRE", "cancelled")
        return True

    def hold(self) -> bool:
        """Operator pause: stop whatever runs and hover; resume() continues with a fresh search. Ignored while
        climbing (a stop there would confuse the drone's state) and when already holding."""
        if self.state not in ("FOLLOW", "REACQUIRE"):
            return False
        self._resume_state = self.state
        self.ctx.drone.stop()
        self._go("HOLD", "operator pause")
        return True

    def resume(self) -> bool:
        """After hold(): turn until the person is seen (at the current height), then follow."""
        if self.state != "HOLD":
            return False
        self._go("REACQUIRE", "resume", ReacquirePerson())
        return True

    def stop(self) -> None:
        """Landing / emergency: no more commands from the mission."""
        self.state, self.child, self.cmd = "IDLE", None, None

    @property
    def active(self) -> bool:
        return self.state == "FOLLOW"

    @property
    def status(self) -> str:
        s = getattr(self.child, "status", "") if self.child is not None else ""
        return f"{self.state}: {s}" if s else self.state

    def step(self) -> None:
        ctx, c, d = self.ctx, self.cfg, self.ctx.drone
        st = self.state
        if st == "CLIMB":
            if self.cmd is None:
                alt = ctx.altitude
                cm = int(round((c.altitude_m - alt) * 100)) if alt is not None else 0
                if cm < 20:
                    self._follow("at follow altitude")
                    return
                self.cmd = Discrete("move", min(cm, 200), "up")
            r = self.cmd.step(ctx)
            if r != RUNNING:
                self._follow("climbed" if r == SUCCESS else f"climb: {d.last_result()}")
        elif st == "FOLLOW":
            self.child.step(ctx)
            if ctx.res is not None and ctx.res.person_unseen_s > c.lost_reacquire_s and \
                    ctx.now - self.state_t > c.lost_reacquire_s:
                self._go("REACQUIRE", f"person lost for {c.lost_reacquire_s:.0f} s")
        elif st == "REACQUIRE":
            self._step_reacquire()
        elif st == "HOLD":
            if not d.busy():
                d.rc(0, 0, 0, 0)
        self._say_status()

    def _say_status(self) -> None:
        """Log what the behaviour is doing when that changes (not every new range reading)."""
        s = getattr(self.child, "status", "") if self.child is not None else ""
        if s.startswith("person ") and (s[7:8].isdigit() or s.startswith("person ?")):
            kind = "following the person"
        elif s.startswith("person lost") and "turning" in s:
            kind = "person lost: turning toward where they were"
        else:
            kind = s.split(" (")[0]
        if kind and kind != self._said:
            self._said = kind
            self.log(f"{self.state.lower()}: {kind}", "info")

    def _step_reacquire(self) -> None:
        """Go to LOOK height (just below head height: from follow height a person closer than ~1.5 m is out of the
        bottom of the frame), turn until the person is seen, then follow (FOLLOW climbs back up)."""
        ctx, c = self.ctx, self.cfg
        if self.child is None:
            if self.cmd is None:
                alt = ctx.altitude
                look = max(c.min_altitude_m + 0.2, c.person_height_m - 0.15)
                cm = int(round((look - alt) * 100)) if alt is not None else 0
                if abs(cm) >= 20:
                    self.cmd = Discrete("move", min(abs(cm), 200), "up" if cm > 0 else "down")
                else:
                    self.child = ReacquirePerson()
                    self.child.start(ctx)
                    return
            if self.cmd is not None and self.cmd.step(ctx) != RUNNING:
                self.cmd = None
                self.child = ReacquirePerson()
                self.child.start(ctx)
            return
        r = self.child.step(ctx)
        if r == SUCCESS:
            alt = ctx.altitude
            if alt is not None and c.altitude_m - alt >= 0.2:
                # back up to the follow height in one move, as after takeoff: holding it by rc from look height
                # settles a few cm short (rc under 1 is 0), too close to head height for the head-angle range
                self._go("CLIMB", "person found: climbing back to the follow height")
            else:
                self._follow("person found")
        elif r == FAILURE:
            self.announce("I can't see you. Say 'follow me' when you're in front of me, or 'land'.")
            self._go("HOLD", "person not found")
            self._resume_state = "FOLLOW"

    # -- the camera overlay
    def snapshot(self) -> Shown | None:
        """The locked person's last box (.box, .t, .confirmed), or None."""
        res = self.ctx.res
        det = self.ctx.perception.lock.state.last_det
        if det is None or res is None or not math.isfinite(res.person_unseen_s):
            return None
        return Shown(det.bbox, res.t - res.person_unseen_s, res.person_confirmed or self.state == "FOLLOW")
