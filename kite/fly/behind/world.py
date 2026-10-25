"""One behind-follow episode: the app's real follow mission against a simulated person, camera and Tello (DESIGN 3).

    res = run_behind_episode(seed, "demo", BehindCfg())                    # FLOWN: FollowMission + smooth2
    res = run_behind_episode(seed, "train", BehindCfg(steering="pid"))     # the yaw law instead of the fly
    res = run_behind_episode(seed, "train", tuned_cfg, arm="V3-JOINT", brain_x=x)   # TunedFollowMission, brain x
    res = run_behind_episode(seed, "demo", cfg, overrides={"speed_lo": 0.6, "speed_hi": 0.6}, record=True)

The loop is run/pilot.py's _tick with the real kite.follow_behind classes, every 0.05 + U(0, 0.01) s:
    the newest frame that has reached the laptop (and been through YOLO) -> Telemetry -> mission.sense(frame, now)
    [Perception: FakePose -> SimpleTracker -> TargetLock -> PersonEstimator] -> governor.check(now, last_frame_t)
    -> mission.step() while flying (it sends rc, or a discrete move it then waits on) -> SafetyGovernor -> SimTello
then kite.fly.sim.drone.DroneModel (reused, not edited) in 0.01 s sub-steps; a stick takes effect from the next one.

type(cfg) is BehindCfg: the flown FollowMission. A kite.follow_tuned.TunedCfg: TunedFollowMission (imported lazily:
the tuned follow is optional here). The world sets, per episode and on a copy (dataclasses.replace; the caller's cfg is
never touched), what the operator sets in the app: person_height_m = the person's height as typed (true + U(+-0.02),
to the cm) and altitude_m = that + the cfg's own clearance (0.2 m by default, the app's --height default, at most
2.2 m). With brain_x (34 FLY-YAW numbers in [0, 1]) the fly brain is that candidate: a params file
{"arm": "FLY-YAW", "x": ..., "brain": "pursuit_core1.npz"} in a per-process temp dir, named by a hash of x.

Start: directly in FOLLOW, as begin() + the climb would leave it (mission._follow("sim start")): the drone hovers at
altitude_m + U(-0.12, +0.08), the person stands 1.8 + U(-0.3, +0.6) m ahead (+-10 deg), facing away (+-20 deg), for
1.0-2.5 s. The lock needs its 3 confirming hits as live; start_failed when it never confirms on the person in 5 s.
After every follow start (the first, and each after a REACQUIRE) the fly brain's Poisson noise is reseeded from the
episode seed, so the brain's luck is common to every arm too.

Video: frames captured at 30 fps (random phase; the video runs from 1 s before the start, so a frame is there at
t = 0, as live) reach the laptop video_latency_s later (Frame.t) and perception YOLO U(0.02, 0.06) s after that; the
first control tick after that takes the newest one. The image is the truth at CAPTURE time, the telemetry is NOW: the
app reads a 0.3 s old frame with the current pitch, and so does the sim. Telemetry each tick: 'h' = z + bias + noise to
the cm, tof = what is below (the person's head within 0.3 m) to the cm, pitch (+ nose up) and yaw in whole degrees,
battery 80 %. The body pitch is DroneModel's acceleration-implied tilt through a first-order attitude lag (att_tau_s,
0.08-0.20 s; DroneModel's own pitch jumps with every stick step). Governor: the real SafetyGovernor(SimTello,
SafetyCfg(max_rc=65, max_altitude_m=2.3)), with the geofence off and the flight-time limit past the episode (nothing
else may land it). SimTello's discrete move / rotate run at 0.5 m/s / 60 deg/s and are busy() until done, so
REACQUIRE's turns and the climbs happen. Collision ends the episode: camera within 0.3 m horizontally of the head top
while above it, or 3D within 0.35 m.

Telemetry pitch error: an OU process (sd pitch_noise_deg, correlation pitch_noise_tau_s, profiles.py says why),
added to the lagged body pitch and rounded. The ToF reads the highest part of the person's silhouette (Body.width_at:
shoulders, arms, head) inside its 25 deg cone below the drone, else the floor: so the governor's no-descent over a
person can act in the band between the cone's reach and the collision rule. The shared FlySteer is keyed without
fly_max_rc_yaw (kite.follow_behind.FlyYaw.load), so every follow start sets its clamp to the episode's own.

Truth traces for kite.fly.behind.metrics.episode_metrics(phys, ctrl, info): `phys` per 0.01 s physics tick after the
start (pitch_deg is the lagged body pitch the camera has), `ctrl` per control tick: the sticks that reached SimTello
after the governor; the lock's identity (0 none or lost, 1 the person, 2 the second person, 3 a false box: the
locked track's latest detection matched to its source in FakePose); whether the follow saw its person on a new
frame; gov_event = check() was not "ok", the governor logged an event, or the rc it passed on differs from the rc the
mission asked for (a clamp, the ceiling / floor push, or the climb taper in the 0.4 m below the ceiling).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ... import follow_behind as fb
from ...config import SafetyCfg
from ...drone.base import Drone, check_move, check_rotate, clamp_rc
from ...drone.safety import SafetyGovernor
from ...follow_behind import BehindCfg, CameraModel, FollowBehind, FollowMission, Perception
from ...types import Frame, Telemetry
from ..interfaces import EpisodeResult
from ..sim.drone import PHYS_DT, DroneModel, DroneParams, noise_len
from . import profiles as pf
from .fitness import episode_cost
from .metrics import episode_metrics
from .pose import CX, CY, Z_MIN, Body, FakePose, Snap, lerp_snap
from .pose import H as IMG_H
from .pose import W as IMG_W
from .walker import Crossing, Person

CONTROL_S = 0.05  # run/pilot.py TICK_S ...
CONTROL_JITTER_S = 0.01  # ... plus U(0, this)
FPS = 30.0
VIDEO_PREROLL_S = 1.0  # the video ran before the follow started: a frame is there at t = 0 (perception starts at 0)
METRIC_START_S = 3.0
START_CONFIRM_S = 5.0
MOVE_MPS, ROTATE_DPS = 0.5, 60.0  # SimTello's discrete motions
BATTERY_PCT = 80.0
MAX_RC = 65  # SafetyCfg.max_rc in the sim: 0.6 m/s (stick 63) must be possible
CEILING_M = 2.3
MAX_ALTITUDE_M = 2.2  # run/app.py: --height is at most 2.2
COLLIDE_H_M, COLLIDE_3D_M = 0.3, 0.35
TOF_HALF_TAN = math.tan(math.radians(12.5))  # the downward ToF's cone (~25 deg full)
TOF_NEAR_M = 0.6  # farther than this from the head-top axis, nothing of the person is under the ToF
TOF_STEP_M = 0.02
FLY_CACHE_MAX = 4  # FlySteer instances kept per process (kite.follow_behind._FLY_SHARED)
BRAIN_SEED_STRIDE = 1_000_003  # brain seed of the n-th follow start: seed + n * this
IMAGE = np.zeros((IMG_H, IMG_W, 3), np.uint8)  # one shared frame buffer: the fake detector ignores the pixels
_EPS = 1e-9
_LOCK_NONE = 0
_FOLLOW_STATES = ("CLIMB", "FOLLOW", "REACQUIRE", "HOLD")  # run/pilot.py STATES: the mission steps in these


# ------------------------------------------------------------------ the drone
class SimTello(Drone):
    """The simulated Tello behind the real SafetyGovernor. rc() stores the sticks the physics applies from the next
    sub-step (ignored while a discrete command runs, as the Drone contract says); move() / rotate() start a timed
    motion that advance() plays out, busy() until it is done; telemetry() is the latest state packet."""

    name = "sim-behind"

    def __init__(self, move_mps: float = MOVE_MPS, rotate_dps: float = ROTATE_DPS):
        self.cmd = (0, 0, 0, 0)
        self.tel = Telemetry(t=0.0)
        self.move_mps, self.rotate_dps = move_mps, rotate_dps
        self.motion: list | None = None  # [kind, remaining (m or deg), direction or sign]
        self._result: str | None = "ok"
        self._flying = True
        self.n_discrete = 0

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        if self._flying and self.motion is None:
            self.cmd = (lr, fb, ud, yaw)

    def move(self, direction: str, cm: int) -> None:
        cm = check_move(direction, cm)
        self.motion, self.cmd, self._result = ["move", cm / 100.0, direction], (0, 0, 0, 0), None
        self.n_discrete += 1

    def rotate(self, deg: int) -> None:
        d = check_rotate(deg)
        self.motion, self.cmd, self._result = ["rotate", float(abs(d)), 1.0 if d > 0 else -1.0], (0, 0, 0, 0), None
        self.n_discrete += 1

    def advance(self, model: DroneModel, dt: float) -> None:
        """One physics sub-step of the running discrete motion (on top of a zero-stick model step: the Tello's own
        closed loop moves it at a set speed)."""
        m = self.motion
        if m[0] == "rotate":
            d = min(m[1], self.rotate_dps * dt)
            h = model.heading + m[2] * math.radians(d)
            model.heading = h - 2.0 * math.pi if h > math.pi else (h + 2.0 * math.pi if h <= -math.pi else h)
        else:
            d = min(m[1], self.move_mps * dt)
            s, c = math.sin(model.heading), math.cos(model.heading)
            way = m[2]
            if way == "up":
                model.z += d
            elif way == "down":
                model.z = max(0.0, model.z - d)
            elif way in ("forward", "back"):
                k = d if way == "forward" else -d
                model.x, model.y = model.x + k * s, model.y + k * c
            else:
                k = d if way == "right" else -d
                model.x, model.y = model.x + k * c, model.y - k * s
        m[1] -= d
        if m[1] <= 1e-9:
            self.motion, self._result = None, "ok"

    def stop(self) -> None:
        self.motion, self.cmd, self._result = None, (0, 0, 0, 0), "ok"

    def busy(self) -> bool:
        return self.motion is not None

    def last_result(self) -> str | None:
        return None if self.motion is not None else self._result

    def telemetry(self) -> Telemetry:
        return self.tel

    @property
    def flying(self) -> bool:
        return self._flying

    def takeoff(self) -> None:
        self._flying = True

    def land(self) -> None:  # the episode is over (nothing in the sim should ask for it)
        self._flying, self.motion, self.cmd = False, None, (0, 0, 0, 0)

    def emergency(self) -> bool | None:
        self.land()
        return True

    def frame_source(self):
        return None


class _Governor(SafetyGovernor):
    """The real SafetyGovernor; it only notes the rc the mission asked for, so the world can tell when it changed it."""

    def __init__(self, inner: Drone, cfg: SafetyCfg):
        super().__init__(inner, cfg)
        self.asked: tuple | None = None

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self.asked = (lr, fb, ud, yaw)
        super().rc(lr, fb, ud, yaw)


def drone_params(p: pf.BehindParams) -> DroneParams:
    return DroneParams(fwd_gain_mps=p.fwd_gain_mps, yaw_gain_dps=p.yaw_gain_dps, vz_gain_mps=p.vz_gain_mps,
                       tau_fwd_s=p.tau_fwd_s, tau_yaw_s=p.tau_yaw_s, tau_z_s=p.tau_z_s, fwd_dead_s=p.fwd_dead_s,
                       yaw_dead_s=p.yaw_dead_s, vz_dead_s=p.vz_dead_s, pitch_scale=p.pitch_scale,
                       drift_sigma_mps=p.drift_sigma_mps, wind_sigma_mps=p.wind_sigma_mps, hold_tau_s=p.hold_tau_s)


def safety_cfg(episode_s: float) -> SafetyCfg:
    """65 so 0.6 m/s is possible (the flown follow never sends more than 50); nothing but the altitude limits and the
    stale checks can act: battery 80 %, no geofence, the flight-time limit past the episode."""
    return SafetyCfg(max_rc=MAX_RC, max_altitude_m=CEILING_M, geofence_m=None,
                     max_flight_s=max(SafetyCfg().max_flight_s, episode_s + 60.0))


# ------------------------------------------------------------------ the follow under test
def mission_class(cfg: BehindCfg):
    """FollowMission for the flown BehindCfg, TunedFollowMission for a TunedCfg."""
    if type(cfg) is BehindCfg:
        return FollowMission
    from ...follow_tuned import TunedCfg, TunedFollowMission

    if isinstance(cfg, TunedCfg):
        return TunedFollowMission
    raise TypeError(f"cfg must be a BehindCfg or a kite.follow_tuned.TunedCfg, not {type(cfg).__name__}")


_TMP: tuple[int, Path] | None = None  # (pid, dir): a forked worker makes its own, so no process deletes another's


def _tmp_dir() -> Path:
    global _TMP
    if _TMP is None or _TMP[0] != os.getpid():
        d = Path(tempfile.mkdtemp(prefix=f"kite-behind-{os.getpid()}-"))
        atexit.register(shutil.rmtree, d, True)
        _TMP = (os.getpid(), d)
    return _TMP[1]


def brain_params_file(brain_x) -> Path:
    """A FlySteer params file for 34 FLY-YAW numbers in [0, 1] (param_space("FLY-YAW") order), named by their hash so
    the same candidate reuses its file (and its loaded brain in kite.follow_behind._FLY_SHARED)."""
    x = np.asarray(brain_x, dtype=np.float64).ravel()
    digest = hashlib.sha1(x.tobytes()).hexdigest()[:20]
    path = _tmp_dir() / f"FLY-YAW_{digest}.json"
    if not path.is_file():
        from ...flybrain.flyfollow.rl.params import param_space

        dim = param_space("FLY-YAW").dim
        if x.shape != (dim,) or not np.all(np.isfinite(x)):
            raise ValueError(f"brain_x must be {dim} finite FLY-YAW numbers, got shape {x.shape}")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"arm": "FLY-YAW", "x": [float(v) for v in np.clip(x, 0.0, 1.0)],
                                   "brain": "pursuit_core1.npz"}), encoding="utf-8")
        os.replace(tmp, path)
    return path


def _touch_fly_cache(steer) -> None:
    """Keep kite.follow_behind._FLY_SHARED to FLY_CACHE_MAX brains, least recently started first out (a worker that
    evaluates thousands of candidates must not hold thousands of brains). An evicted candidate's temp file goes too."""
    shared = fb._FLY_SHARED
    for k, v in list(shared.items()):
        if v is steer:
            shared[k] = shared.pop(k)  # most recent last
            break
    tmp = str(_TMP[1]) if _TMP is not None and _TMP[0] == os.getpid() else None
    while len(shared) > FLY_CACHE_MAX:
        k = next(iter(shared))
        del shared[k]
        if tmp is not None and isinstance(k, tuple) and str(k[0]).startswith(tmp):
            try:
                os.remove(k[0])
            except OSError:
                pass


def episode_cfg(cfg: BehindCfg, p: pf.BehindParams, brain_x=None) -> BehindCfg:
    """The cfg this episode flies: the operator's settings for this person (a copy; `cfg` is not changed)."""
    ph = p.operator_height_m
    clearance = cfg.altitude_m - cfg.person_height_m
    alt = min(round(ph + clearance, 2), MAX_ALTITUDE_M)
    if alt < ph + 0.2:  # rounding (1.87 + 0.2 = 2.0700000000000003): BehindCfg.check wants >= height + 0.2
        alt = ph + 0.2
    kw: dict = {"person_height_m": ph, "altitude_m": alt}
    if brain_x is not None and cfg.steering == "fly":
        kw["fly_params_path"] = str(brain_params_file(brain_x))
    return replace(cfg, **kw)


class _TimedSteer:
    """Passes a FlySteer through and adds up the wall time of its yaw() calls (EpisodeResult.brain_s)."""

    __slots__ = ("inner", "clock")

    def __init__(self, inner, clock: list[float]):
        self.inner, self.clock = inner, clock

    def yaw(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self.inner.yaw(*args, **kwargs)
        finally:
            self.clock[0] += time.perf_counter() - t0

    def __getattr__(self, name):
        return getattr(self.inner, name)


# ------------------------------------------------------------------ the episode
def tof_top(body: Body, right: float, fwd: float, z: float) -> float | None:
    """The highest point of the person's silhouette inside the ToF's cone below a drone at height z, offset (right,
    forward) m from their head-top axis in their own frame; None when the cone sees only the floor. Called only
    within TOF_NEAR_M of the axis (rare: the steps cost nothing on the common path)."""
    zs = min(body.height, z)
    while zs > 0.0:
        a, b = body.width_at(zs)
        if math.hypot(max(0.0, abs(right) - a), max(0.0, abs(fwd) - b)) <= (z - zs) * TOF_HALF_TAN:
            return zs
        zs -= TOF_STEP_M
    return None


def pitch_noise(bp: pf.BehindParams, normals: list[float], periods: list[float]) -> list[float]:
    """The telemetry pitch error per control tick: an OU process of sd bp.pitch_noise_deg and correlation time
    bp.pitch_noise_tau_s over the pre-drawn tick periods, driven by pre-drawn normals (common to every arm)."""
    sd, tau = float(bp.pitch_noise_deg), max(1e-3, float(bp.pitch_noise_tau_s))
    out = [0.0] * len(normals)
    v = sd * normals[0] if normals else 0.0
    for k in range(len(normals)):
        if k:
            a = math.exp(-periods[k - 1] / tau)
            v = a * v + sd * math.sqrt(1.0 - a * a) * normals[k]
        out[k] = v
    return out


def _in_view(x: float, y: float, z: float, sh: float, ch: float, cp: float, sp: float, fx: float, px: float,
             py: float, hz: float) -> bool:
    """The head top (px, py, hz) is inside the true camera's frame."""
    dx, dy = px - x, py - y
    f = dx * sh + dy * ch
    u = hz - z
    zc = f * cp + u * sp
    if zc < Z_MIN:
        return False
    col = CX + fx * (dx * ch - dy * sh) / zc
    row = CY + fx * (f * sp - u * cp) / zc
    return 0.0 <= col <= IMG_W and 0.0 <= row < IMG_H


def run_behind_episode(seed: int, profile: str, cfg: BehindCfg, *, arm: str = "FLOWN", brain_x=None,
                       episode_s: float = 60.0, overrides: dict | None = None, record: bool = False) -> EpisodeResult:
    """One episode; see the module docstring. ret = -kite.fly.behind.fitness.episode_cost(metrics)."""

    wall0 = time.perf_counter()
    bp = pf.draw(seed, profile, overrides)
    cls = mission_class(cfg)
    c = episode_cfg(cfg, bp, brain_x)
    dt = PHYS_DT
    n_phys = int(round(episode_s / dt))
    t_end = n_phys * dt
    n_ctrl = int(math.ceil(episode_s / CONTROL_S)) + 4
    n_frames = int((episode_s + VIDEO_PREROLL_S) * FPS) + 4

    # ---- every random number of the episode, drawn now from its own stream (common to every arm)
    periods = (CONTROL_S + CONTROL_JITTER_S * pf.stream(seed, profile, pf.S_TIMING).random(n_ctrl)).tolist()
    tn_h, tn_tof, tn_p = pf.stream(seed, profile, pf.S_TELEMETRY).standard_normal((3, n_ctrl)).tolist()
    pitch_err = pitch_noise(bp, tn_p, periods)
    vid = pf.stream(seed, profile, pf.S_VIDEO)
    phase = float(vid.uniform(0.0, 1.0 / FPS))
    yolo = vid.uniform(bp.yolo_lo_s, bp.yolo_hi_s, n_frames).tolist()
    t_frames = [phase - VIDEO_PREROLL_S + j / FPS for j in range(n_frames)]
    noise = pf.stream(seed, profile, pf.S_DRONE).standard_normal(noise_len(n_phys + 2)).tolist()

    # ---- the world at the start
    hp = float(bp.person_h_m)
    r0 = c.distance_m + bp.start_range_off_m
    b0 = math.radians(bp.start_bearing_deg)
    model = DroneModel(drone_params(bp), noise, dt)
    model.reset(0.0, 0.0, c.altitude_m + bp.alt_offset_m, 0.0)
    stretch = (bp.stretch_mps, bp.stretch_at, bp.stretch_s) if bp.stretch_mps > 0.0 else None
    person = Person(pf.stream(seed, profile, pf.S_WALK), r0 * math.sin(b0), r0 * math.cos(b0),
                    b0 + math.radians(bp.start_facing_deg), n_phys, bp.speed_lo, bp.speed_hi, bp.stand_s,
                    bp.p_step_back, stretch)
    crossing = (Crossing(bp.q_at * episode_s, bp.q_dist_m, bp.q_speed_mps, bp.q_side, bp.q_angle_deg, dt)
                if bp.second_person else None)
    body = Body(hp, bp.shoulder_m)
    pose = FakePose(bp, pf.stream(seed, profile, pf.S_POSE), pf.stream(seed, profile, pf.S_SECOND), t_frames,
                    body, Body(bp.q_height_m, bp.q_shoulder_m) if crossing is not None else None)
    sim = SimTello()
    gov = _Governor(sim, safety_cfg(episode_s))
    perception = Perception(c, pose, CameraModel(IMG_W, IMG_H, c.camera_fx, c.camera_fx, CX, CY))
    lines: list[tuple[float, str, str]] = []
    mission = cls(gov, perception, c, log=(lambda text, kind="info": lines.append((ctx.now, kind, text)))
                  if record else None)
    ctx = mission.ctx

    def packet(t: float, k: int, pitch: float) -> Telemetry:
        z = model.z
        dx, dy = model.x - person.x, model.y - person.y
        below = z
        if dx * dx + dy * dy < TOF_NEAR_M * TOF_NEAR_M:
            sf, cf = math.sin(person.facing), math.cos(person.facing)  # forward (sf, cf), right (cf, -sf)
            top = tof_top(body, dx * cf - dy * sf, dx * sf + dy * cf, z)
            if top is not None:
                below = z - top
        yaw = (math.degrees(model.heading) + 180.0) % 360.0 - 180.0
        return Telemetry(t=t, yaw_deg=float(round(yaw)), pitch_deg=float(round(pitch + pitch_err[k])),
                         roll_deg=0.0,
                         height_m=round(z + bp.alt_bias_m + bp.alt_noise_m * tn_h[k], 2),
                         tof_m=round(max(0.0, below) + bp.tof_noise_m * tn_tof[k], 2), battery_pct=BATTERY_PCT,
                         flight_time_s=t)

    brain_clock = [0.0]
    starts = {"child": None, "n": 0}

    def on_step() -> None:
        """After every mission step: a new follow behaviour gets the episode's brain seed (and the timer)."""
        child = mission.child
        if child is starts["child"]:
            return
        starts["child"] = child
        if isinstance(child, FollowBehind) and child.fly is not None:
            steer = child.fly.steer
            steer.reset(seed=int(seed) + starts["n"] * BRAIN_SEED_STRIDE, z_ref_m=c.distance_m, kind="follow")
            steer.max_yaw = int(c.fly_max_rc_yaw)  # the shared brain's clamp is this episode's, not the first one's
            _touch_fly_cache(steer)
            child.fly.steer = _TimedSteer(steer, brain_clock)
            starts["n"] += 1

    sim.tel = packet(0.0, 0, 0.0)
    ctx.now, ctx.tel = 0.0, sim.tel
    mission._follow("sim start")  # as begin() leaves it after the climb
    on_step()

    # ---- traces
    phys = {k: [] for k in ("t", "x", "y", "z", "heading", "pitch_deg", "vx", "vy", "px", "py", "p_heading",
                            "p_speed", "p_event", "in_view", "qx", "qy")}
    ctrl = {k: [] for k in ("t", "state", "lr", "fb", "ud", "yaw", "lock", "seen", "gov_event")}
    (a_t, a_x, a_y, a_z, a_h, a_p, a_vx, a_vy, a_px, a_py, a_ph, a_ps, a_pe, a_iv, a_qx, a_qy) = (
        phys[k].append for k in phys)
    fx_true, mount = float(bp.fx_px), math.radians(bp.mount_pitch_deg)

    frames: deque = deque()  # (capture j, ready time, Frame.t, Snap), captured and not yet handed to perception
    jc, last_j = 0, -1
    frame: Frame | None = None
    track_src: dict[int, int] = {}
    confirmed_t: float | None = None
    collided, collision_t, landed = False, None, False
    nan = math.nan
    q_now = (nan, nan, 0.0)
    # the body pitch: DroneModel's acceleration-implied tilt through the attitude loop's lag (DroneModel's pitch
    # jumps with every stick step; a real quadrotor tilts over ~0.1 s, and the camera and the telemetry see that)
    pitch, k_att = 0.0, 1.0 - math.exp(-dt / max(1e-3, bp.att_tau_s))
    prev = (model.x, model.y, model.z, model.heading, pitch, person.x, person.y, person.facing) + q_now
    lat = float(bp.video_latency_s)
    while jc < n_frames - 1 and t_frames[jc] <= _EPS:  # the pre-roll: the scene as it stands at the start
        frames.append((jc, t_frames[jc] + lat + yolo[jc], t_frames[jc] + lat, Snap(*prev)))
        jc += 1
    tick, k, t_c = 0, 0, 0.0
    while t_c < t_end - _EPS:
        # newest frame through YOLO by now (frames may finish out of order: newest only, never an older one later)
        best = None
        for it in frames:
            if it[1] <= t_c + _EPS and (best is None or it[0] > best[0]):
                best = it
        if best is not None:
            last_j = best[0]
            while frames and frames[0][0] <= last_j:
                frames.popleft()
            frame = Frame(IMAGE, best[2], last_j + 1, "sim")
            pose.load(frame.seq, last_j, best[3])
        sim.tel = packet(t_c, k, pitch)
        mission.sense(frame, t_c)
        res = ctx.res
        if ctx.new_frame and res is not None:
            srcs = pose.sources.get(frame.seq)
            if srcs is not None and len(srcs) == len(res.persons):
                for obs, s in zip(res.persons, srcs):
                    track_src[obs.det.track_id] = s
        ls = perception.lock.state
        lock = track_src.get(ls.track_id, _LOCK_NONE) if (ls.track_id is not None and not ls.lost) else _LOCK_NONE
        seen = bool(ctx.new_frame and res is not None and res.person is not None)
        if confirmed_t is None and res is not None and res.person_confirmed and lock == 1:
            confirmed_t = t_c
        n_ev = len(gov.events)
        gov.asked = None
        action = gov.check(t_c, frame.t if frame is not None else None)
        if action == "land" or not sim.flying:
            mission.stop()  # run/pilot.py: the governor lands; the episode is over
            landed = True
        elif mission.state in _FOLLOW_STATES:
            mission.step()
            on_step()
        cmd = sim.cmd
        asked = gov.asked
        intervened = action != "ok" or len(gov.events) != n_ev or (
            asked is not None and not sim.busy() and cmd != tuple(clamp_rc(_num(v), 100) for v in asked))
        ctrl["t"].append(t_c)
        ctrl["state"].append(mission.state)
        ctrl["lr"].append(cmd[0])
        ctrl["fb"].append(cmd[1])
        ctrl["ud"].append(cmd[2])
        ctrl["yaw"].append(cmd[3])
        ctrl["lock"].append(lock)
        ctrl["seen"].append(seen)
        ctrl["gov_event"].append(bool(intervened))
        if landed:
            break

        # ---- physics until the next control tick
        t_next = t_c + periods[k]
        t_stop = t_next if t_next < t_end else t_end
        while tick * dt < t_stop - _EPS and tick < n_phys:
            ox, oy = model.x, model.y
            if sim.motion is not None:
                model.step(0.0, 0.0, 0.0, 0.0)
                sim.advance(model, dt)
            else:
                model.step(*cmd)
            tick += 1
            t = tick * dt
            x, y, z, h = model.x, model.y, model.z, model.heading
            pitch += (model.pitch_deg - pitch) * k_att
            person.step(x, y)
            px, py = person.x, person.y
            if crossing is not None:
                crossing.step(t, x, y, h)
                q_now = (crossing.x, crossing.y, crossing.facing)
            cur = (x, y, z, h, pitch, px, py, person.facing) + q_now
            # frames captured in (t - dt, t]
            while jc < n_frames - 1 and t_frames[jc] <= t + _EPS:
                tc = t_frames[jc]
                snap = lerp_snap(Snap(*prev), Snap(*cur), (tc - (t - dt)) / dt)
                frames.append((jc, tc + lat + yolo[jc], tc + lat, snap))
                jc += 1
            prev = cur
            sh, ch = math.sin(h), math.cos(h)
            p_rad = math.radians(pitch) + mount
            a_t(t)
            a_x(x)
            a_y(y)
            a_z(z)
            a_h(h)
            a_p(pitch)
            a_vx((x - ox) / dt)
            a_vy((y - oy) / dt)
            a_px(px)
            a_py(py)
            a_ph(person.facing)
            a_ps(person.speed)
            a_pe(person.event)
            a_iv(_in_view(x, y, z, sh, ch, math.cos(p_rad), math.sin(p_rad), fx_true, px, py, hp))
            a_qx(q_now[0])
            a_qy(q_now[1])
            rh = math.hypot(px - x, py - y)
            dz = z - hp
            if (rh < COLLIDE_H_M and dz > 0.0) or rh * rh + dz * dz < COLLIDE_3D_M * COLLIDE_3D_M:
                collided, collision_t = True, t
                break
        if collided or tick >= n_phys:
            break
        t_c = t_next
        k += 1

    start_failed = confirmed_t is None or confirmed_t > START_CONFIRM_S
    info = {"distance_m": c.distance_m, "altitude_m": c.altitude_m, "person_height_m": hp,
            "operator_height_m": c.person_height_m, "metric_start_s": METRIC_START_S, "episode_s": float(episode_s),
            "collided": collided, "collision_t": collision_t, "start_failed": start_failed, "landed": landed}
    metrics = episode_metrics(phys, ctrl, info)
    ret = -float(episode_cost(metrics))
    trace = None
    if record:
        trace = {"phys": phys, "ctrl": ctrl, "info": info,
                 "meta": {"params": asdict(bp), "cfg": asdict(c), "segments": list(person.segments),
                          "follow_starts": starts["n"], "confirmed_t": confirmed_t, "frames_detected": pose.n_frames,
                          "person_detected": pose.n_person, "discrete": sim.n_discrete,
                          "governor_events": list(gov.events), "log": lines}}
    return EpisodeResult(seed=int(seed), profile=profile, arm=arm, ret=ret, terms={}, metrics=metrics,
                         n_ticks=int(round(tick * dt / CONTROL_S)), wall_s=time.perf_counter() - wall0,
                         brain_s=brain_clock[0], trace=trace)


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


__all__ = ["SimTello", "brain_params_file", "drone_params", "episode_cfg", "mission_class", "pitch_noise",
           "run_behind_episode", "safety_cfg", "tof_top"]
