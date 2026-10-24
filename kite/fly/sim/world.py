"""One follow episode: kite's real tracker, controller and safety governor against a simulated world (DESIGN 2, 8, 10).

    res = run_episode(seed, "demo")                          # KITE-LAW: FollowController's own yaw law
    res = run_episode(seed, "train", yaw_source=FlySteer())  # any kite.follow.YawSource steers the yaw
    res = run_episode(seed, "demo", overrides={"p_head": 0.3})   # change one world parameter (profiles.SimParams)
    cfg = dataclasses.replace(default_follow_cfg(), predict=True)   # FollowCfg.predict: kite/predict.py steers
    res = run_episode(seed, "demo", follow_cfg=cfg)

The loop, as kite.record drives FollowMode live:
  fake detector (camera.FakeDetector: boxes at their frame time, after latency) -> kite.follow.WalkerTracker
  -> kite.follow.FollowController(cfg, camera=<believed Camera, fx 920>, max_rc, yaw_source) every control step
  -> the real kite.drone.safety.SafetyGovernor(SafetyCfg(max_rc, max_altitude_m=3.0)) wrapping SimTello (a
     kite.drone.base.Drone), with governor.check(now, last_frame_t) every control step
  -> drone.DroneModel in 0.01 s sub-steps; a command takes effect from the next sub-step.
The control period is 0.100 + U(0, 0.012) s, as kite.record's loop measures it. The controller sees only boxes,
Telemetry.height_m (truth + bias + noise), pitch_deg (truth + noise) and yaw_deg (truth) in whole degrees as the
Tello's state packet sends them (yaw wrapped to +-180; only FollowCfg.pitch_comp and the prediction's dead reckoning
read them), and tof_m; reward and metrics use truth. No route is simulated, so the controller gets route None and
walker_xy None, as kite.record runs it (FollowCfg.predict then predicts without the route prior).

Start. The walker stands facing away (+-30 deg) at behind_m + U(-0.2, +0.8) m and +-15 deg off the drone's heading;
the drone hovers at head top + above_m + U(-0.10, +0.15) m. Draws whose head top is not >= 20 px inside the frame
from the level start pose are skipped. Each remaining candidate runs the pre-roll: the drone hovers (rc 0) on the
static scene with the tracker running, and follow starts at the first control step from 1 s on where FollowMode.start
would accept (a confirmed, fresh track touching the bottom edge, live video), up to 5 s. If it never accepts, the
next candidate is tried with fresh detector and drift streams (attempt index in the stream key). The pre-roll does
not depend on the arm, so the same start is used for every arm; if every attempt fails the episode runs from the
last one and metrics["start_failed"] = 1, to be excluded for every arm. Then FollowController.reset(tel.height_m),
and a yaw source whose reset() takes `seed` is reset with the episode seed (its noise is the episode's).

A collision ("overhead": true horizontal range < 0.3 m while above the head) ends the episode; the rest of it is
charged as lost at 90 deg in `ret` and in the metrics (metrics.EpisodeMetrics). If the governor ever answers anything
but "ok", follow is cancelled (rc 0) as FollowMode.tick does. n_ticks counts interfaces.DT (0.05 s) ticks of follow.
"""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass

from ...camera import Camera
from ...config import SafetyCfg
from ...drone.base import Drone
from ...drone.safety import SafetyGovernor
from ...follow import FollowCfg, FollowController, WalkerTracker, YawSource
from ...types import Telemetry
from ..interfaces import DT, METRIC_START_S, EpisodeResult
from .camera import DETECT_ROW_PX, H, W, DetectorParams, FakeDetector, head_in_view, project
from .drone import PHYS_DT, DroneModel, DroneParams, noise_len
from .metrics import EpisodeMetrics
from .profiles import (S_DETECTOR, S_DRONE, S_START, S_TELEMETRY, S_TIMING, S_WALKER, SimParams, draw_params,
                       stream)
from .reward import Reward, RewardCfg
from .walker import Walker, WalkerCfg

CONTROL_S = 0.100  # kite.record's control period ...
CONTROL_JITTER_S = 0.012  # ... plus U(0, this)
PREROLL_S = 1.0
PREROLL_MAX_S = 5.0
START_ATTEMPTS = 8  # start candidates tried before an episode is flagged start_failed
BELIEVED_FX = 920.0  # Camera() default: no calibration file today
STREET_CEILING_M = 3.0
CORNER_AFTER_S = 3.0  # corner_losses: a loss whose last true detection is in a turn or this long after it
_EPS = 1e-9


def default_follow_cfg() -> FollowCfg:
    """Every arm's configuration (DESIGN 8, R3)."""
    return FollowCfg(behind_m=1.0, above_m=0.30, max_yaw=70)


@dataclass(frozen=True)
class StartCfg:
    range_offset_m: tuple[float, float] = (-0.2, 0.8)  # added to behind_m
    bearing_deg: float = 15.0
    facing_deg: float = 30.0
    alt_offset_m: tuple[float, float] = (-0.10, 0.15)  # added to head top + above_m
    inside_px: float = 20.0  # the head top must be this far inside the frame from the level start pose
    max_draws: int = 64


@dataclass(frozen=True)
class StartPose:
    range_m: float  # horizontal, camera to head top
    bearing_rad: float  # + = walker right of the drone's heading
    facing_rad: float  # walker heading, clockwise from north
    alt_offset_m: float  # drone altitude - (head top + above_m)
    redraws: int  # rows of the start table skipped before this one


def start_candidates(seed: int, profile: str, p: SimParams, cfg: FollowCfg,
                     sc: StartCfg = StartCfg()) -> Iterator[StartPose]:
    """Start poses from the start stream whose head top is >= inside_px inside the frame (level drone, true fx)."""
    rows = stream(seed, profile, S_START).random((sc.max_draws, 4)).tolist()
    last = None
    for k, us in enumerate(rows):
        r0 = cfg.behind_m + sc.range_offset_m[0] + us[0] * (sc.range_offset_m[1] - sc.range_offset_m[0])
        b0 = math.radians((2.0 * us[1] - 1.0) * sc.bearing_deg)
        alt = sc.alt_offset_m[0] + us[3] * (sc.alt_offset_m[1] - sc.alt_offset_m[0])
        last = StartPose(r0, b0, b0 + math.radians((2.0 * us[2] - 1.0) * sc.facing_deg), alt, k)
        row = H / 2 + p.fx_px * (cfg.above_m + alt) / r0
        col = W / 2 + p.fx_px * math.tan(b0)
        if sc.inside_px <= row <= H - sc.inside_px and sc.inside_px <= col <= W - sc.inside_px:
            yield last
    if last is not None:
        yield last  # nothing passed: the last row, so a caller always gets a start


def start_pose(seed: int, profile: str, p: SimParams, cfg: FollowCfg, sc: StartCfg = StartCfg()) -> StartPose:
    """The first geometrically valid start (the one used unless its pre-roll fails)."""
    return next(start_candidates(seed, profile, p, cfg, sc))


class SimTello(Drone):
    """The simulated drone behind the real SafetyGovernor. rc() stores the sticks the physics applies from the next
    sub-step; telemetry() is the latest state packet. It never takes off, moves or rotates on command."""

    name = "sim"

    def __init__(self):
        self.cmd = (0, 0, 0, 0)
        self.tel = Telemetry(t=0.0)
        self._flying = True

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        if self._flying:
            self.cmd = (lr, fb, ud, yaw)

    def telemetry(self) -> Telemetry:
        return self.tel

    @property
    def flying(self) -> bool:
        return self._flying

    def takeoff(self) -> None:
        self._flying = True

    def land(self) -> None:  # the sim keeps hovering; follow is over
        self._flying = False
        self.cmd = (0, 0, 0, 0)

    def emergency(self) -> bool | None:
        self.land()
        return True

    def move(self, direction: str, cm: int) -> None:
        pass

    def rotate(self, deg: int) -> None:
        pass

    def stop(self) -> None:
        self.cmd = (0, 0, 0, 0)

    def busy(self) -> bool:
        return False

    def last_result(self) -> str | None:
        return "ok"

    def frame_source(self):
        return None


class _TimedYaw:
    """Passes a YawSource through and adds up the time spent in it."""

    def __init__(self, src: YawSource):
        self.src = src
        self.s = 0.0

    def reset(self) -> None:
        self.src.reset()

    def yaw(self, t: float, frame_t: float, bearing_deg: float, range_m: float, fallback: float) -> float:
        t0 = time.perf_counter()
        try:
            return self.src.yaw(t, frame_t, bearing_deg, range_m, fallback)
        finally:
            self.s += time.perf_counter() - t0


def _takes_seed(src: object) -> bool:
    try:
        return "seed" in inspect.signature(src.reset).parameters
    except (TypeError, ValueError, AttributeError):
        return False


def drone_params(p: SimParams) -> DroneParams:
    return DroneParams(fwd_gain_mps=p.fwd_gain_mps, yaw_gain_dps=p.yaw_gain_dps, vz_gain_mps=p.vz_gain_mps,
                       tau_fwd_s=p.tau_fwd_s, tau_yaw_s=p.tau_yaw_s, tau_z_s=p.tau_z_s, fwd_dead_s=p.fwd_dead_s,
                       yaw_dead_s=p.yaw_dead_s, vz_dead_s=p.vz_dead_s, pitch_scale=p.pitch_scale,
                       drift_sigma_mps=p.drift_sigma_mps, wind_sigma_mps=p.wind_sigma_mps, hold_tau_s=p.hold_tau_s)


def detector_params(p: SimParams) -> DetectorParams:
    return DetectorParams(rate_hz=p.det_rate_hz, video_latency_s=p.video_latency_s, det_time_s=p.det_time_s,
                          noise_px=p.box_noise_px, dropout=p.dropout, burst_rate_hz=p.burst_rate_hz,
                          false_frac=p.false_box_frac, p_head=p.p_head)


def believed_camera() -> Camera:
    return Camera(W, H, BELIEVED_FX, BELIEVED_FX, W / 2, H / 2, "sim: the uncalibrated published estimate")


class _World:
    """One start attempt: the walker, drone, detector, tracker and governor, with the pre-roll already run."""

    def __init__(self, seed: int, profile: str, p: SimParams, cfg: FollowCfg, sp: StartPose, attempt: int,
                 n_follow: int, episode_s: float, walker_cfg: WalkerCfg, periods: list[float], tel_h: list[float],
                 tel_p: list[float]):
        self.p, self.cfg, self.sp = p, cfg, sp
        self.head, self.fx = p.head_top_m, p.fx_px
        self.half_head, self.half_torso = 0.5 * p.head_w_m, 0.5 * p.shoulder_m
        self.tel_h, self.tel_p = tel_h, tel_p
        self.wx, self.wy = sp.range_m * math.sin(sp.bearing_rad), sp.range_m * math.cos(sp.bearing_rad)
        self.walker = Walker(stream(seed, profile, S_WALKER), self.wx, self.wy, sp.facing_rad, p.v_cap_mps,
                             (p.walk_speed_lo, p.walk_speed_hi), n_follow + 2, walker_cfg, PHYS_DT)
        n_noise = noise_len(int((PREROLL_MAX_S + episode_s + 1.0) / PHYS_DT))
        self.model = DroneModel(drone_params(p), stream(seed, profile, S_DRONE, attempt).standard_normal(
            n_noise).tolist(), PHYS_DT)
        self.model.reset(0.0, 0.0, self.head + cfg.above_m + sp.alt_offset_m, 0.0)
        self.det = FakeDetector(detector_params(p), stream(seed, profile, S_DETECTOR, attempt), 0.0,
                                PREROLL_MAX_S + episode_s + 1.0)
        self.sim = SimTello()
        self.gov = SafetyGovernor(self.sim, SafetyCfg(max_rc=p.max_rc, max_altitude_m=STREET_CEILING_M))
        self.tracker = WalkerTracker(cfg)
        self.last_true = -math.inf  # availability time of the last usable walker box
        self.last_frame_t: float | None = None  # newest frame the laptop has (the governor's video check)
        self.g = self.geom()
        self.tick, self.k, self.t_c = 0, 0, 0.0
        self.started = self._preroll(periods)

    def geom(self):
        d = self.model
        dx, dy = self.wx - d.x, self.wy - d.y
        s, c = math.sin(d.heading), math.cos(d.heading)
        return project(dx * s + dy * c, dx * c - dy * s, self.head - d.z, self.p.head_m, self.half_head,
                       self.half_torso, d.z, d.pitch_deg, self.fx, self.fx)

    def deliver(self, t: float) -> None:
        min_conf = self.cfg.min_conf
        for t_avail, frame_t, dets, true_conf in self.det.ready(t):
            self.tracker.update(dets, frame_t, W, H)
            self.last_frame_t = frame_t
            if true_conf >= min_conf:
                self.last_true = t_avail

    def packet(self, t: float, k: int) -> Telemetry:
        """The state packet: attitude in whole degrees (the Tello sends integers), yaw in -180..180."""
        m = self.model
        yaw = (math.degrees(m.heading) + 180.0) % 360.0 - 180.0
        return Telemetry(t=t, yaw_deg=float(round(yaw)), pitch_deg=float(round(m.pitch_deg + self.tel_p[k])),
                         height_m=m.z + self.tel_h[k], tof_m=m.z)

    def _preroll(self, periods: list[float]) -> bool:
        cfg, model, det = self.cfg, self.model, self.det
        while True:
            t_c = self.t_c
            self.deliver(t_c)
            self.sim.tel = self.packet(t_c, self.k)
            action = self.gov.check(t_c, self.last_frame_t)
            if t_c >= PREROLL_S - _EPS:
                tr, lf = self.tracker.track, self.last_frame_t
                if (action == "ok" and lf is not None and t_c - lf <= 1.0 and tr is not None and tr.confirmed
                        and tr.age(t_c) <= cfg.fresh_s and tr.touches_bottom):
                    return True
                if t_c >= PREROLL_MAX_S - _EPS:
                    return False
            self.gov.rc(0, 0, 0, 0)
            self.t_c = t_c + periods[self.k]
            self.k += 1
            while self.tick * PHYS_DT < self.t_c - _EPS:
                model.step(0.0, 0.0, 0.0, 0.0)
                self.tick += 1
                g_new = self.geom()
                det.capture((self.tick - 1) * PHYS_DT, self.tick * PHYS_DT, self.g, g_new)
                self.g = g_new


def _below_frame(g) -> bool:
    """The head left through the bottom edge (or went under the camera), not out of a side."""
    return g is None or (0.0 <= g[0] <= W and g[1] >= DETECT_ROW_PX)


def run_episode(seed: int, profile: str, follow_cfg: FollowCfg | None = None, yaw_source: YawSource | None = None,
                arm: str = "KITE-LAW", record: bool = False, episode_s: float = 60.0,
                overrides: dict[str, float] | None = None, reward_cfg: RewardCfg = RewardCfg(),
                walker_cfg: WalkerCfg = WalkerCfg(), start_cfg: StartCfg = StartCfg(),
                controller_max_rc: int | None = None) -> EpisodeResult:
    """controller_max_rc: FollowController's max_rc if not the governor's (FollowMode passes SafetyCfg.max_rc)."""
    wall0 = time.perf_counter()
    cfg = follow_cfg or default_follow_cfg()
    p = draw_params(seed, profile, overrides)
    pdt = PHYS_DT
    n_follow = int(round(episode_s / pdt))
    n_ctrl_max = int(math.ceil((PREROLL_MAX_S + episode_s) / CONTROL_S)) + 2
    above, behind, head = cfg.above_m, cfg.behind_m, p.head_top_m
    tn = stream(seed, profile, S_TELEMETRY).standard_normal((2, n_ctrl_max))
    tel_h = (p.alt_bias_m + p.alt_noise_m * tn[0]).tolist()
    tel_p = (p.pitch_noise_deg * tn[1]).tolist()
    periods = (CONTROL_S + CONTROL_JITTER_S * stream(seed, profile, S_TIMING).random(n_ctrl_max)).tolist()

    # ---- the world: the first start whose pre-roll lets follow start (the same for every arm)
    world = None
    for attempt, sp in enumerate(start_candidates(seed, profile, p, cfg, start_cfg)):
        world = _World(seed, profile, p, cfg, sp, attempt, n_follow, episode_s, walker_cfg, periods, tel_h, tel_p)
        if world.started or attempt + 1 >= START_ATTEMPTS:
            break
    w = world
    model, det, sim, gov, tracker, walker = w.model, w.det, w.sim, w.gov, w.tracker, w.walker
    tick, k, t_c, g = w.tick, w.k, w.t_c, w.g
    wx, wy = w.wx, w.wy
    t_start, tick0, k0 = t_c, tick, k
    t0f = tick0 * pdt  # the walker's clock and the follow / metric windows run on the physics grid from here
    tick_end = tick0 + n_follow
    t_end = tick_end * pdt

    timed = _TimedYaw(yaw_source) if yaw_source is not None else None
    ctrl = FollowController(cfg, camera=believed_camera(), max_rc=controller_max_rc or p.max_rc, yaw_source=timed)
    ctrl.reset(sim.tel.height_m)
    if yaw_source is not None and _takes_seed(yaw_source):
        yaw_source.reset(seed=seed)

    # ---- follow
    reward = Reward(reward_cfg, behind, cfg.min_range_m)
    ret = 0.0
    lost_after, collide = reward_cfg.lost_after_s, reward_cfg.collide_m
    m_phys0 = int(round(METRIC_START_S / pdt))
    corners = [(t0f + a * pdt, t0f + e * pdt + CORNER_AFTER_S) for kind, a, e in walker.segments if kind == "turn"]
    mx = EpisodeMetrics(behind, cfg.min_range_m, t0f + METRIC_START_S, corners, reward_cfg.band_frac)
    mx._was_in_view = head_in_view(g)
    phys = mx.phys
    prev_yaw = prev_fb = prev_lr = 0
    collided = cancelled = False
    trace: dict[str, list] | None = None
    if record:
        trace = {key: [] for key in ("t", "state", "lr", "fb", "ud", "yaw", "cmd_fb", "cmd_yaw", "x", "y", "z",
                                     "heading_deg", "pitch_deg", "walker_x", "walker_y", "range_m", "bearing_deg",
                                     "in_view", "v_top", "est_range_m", "est_bearing_deg", "walker_m_s", "lost",
                                     "action", "holding", "planner")}
        trace["meta"] = [{"t_start": t_start, "start": w.sp, "v_cap_mps": p.v_cap_mps, "params": p,
                          "segments": walker.segments, "step": walker.step_ticks, "started": w.started}]
    rng_t = math.hypot(wx - model.x, wy - model.y)
    t_stop = t_c
    while t_c < t_end - _EPS:
        w.deliver(t_c)
        sim.tel = w.packet(t_c, k)
        action = gov.check(t_c, w.last_frame_t)
        if action != "ok" or not sim.flying:
            cancelled = True  # FollowMode.tick: any safety action ends follow
        if cancelled:
            state, cmd = "off", None
        else:
            cmd = ctrl.update(tracker.track, gov.telemetry(), t_c)
            gov.rc(*cmd.rc)
            state = cmd.state
        lr, fb, ud, yaw = sim.cmd if not cancelled else (0, 0, 0, 0)
        t_next = t_c + periods[k]
        t_stop = t_next if t_next < t_end else t_end
        dt_c = t_stop - t_c

        # truth at the control step: reward and the controller-side metrics
        dx, dy = wx - model.x, wy - model.y
        s, c = math.sin(model.heading), math.cos(model.heading)
        bearing = math.atan2(dx * c - dy * s, dx * s + dy * c)
        gap = t_c - w.last_true > lost_after
        lost = gap or state == "lost"
        d_yaw = yaw - prev_yaw
        ret += reward.step(dt_c, math.hypot(dx, dy), bearing, d_yaw, lost)
        mx.control(t_c, dt_c, state, d_yaw, yaw, gap, w.last_true, fb - prev_fb, lr - prev_lr,
                   cmd is not None and cmd.planner == "predict")
        prev_yaw, prev_fb, prev_lr = yaw, fb, lr
        if trace is not None:
            for key, v in (("t", t_c - t_start), ("state", state), ("lr", lr), ("fb", fb), ("ud", ud), ("yaw", yaw),
                           ("cmd_fb", None if cmd is None else cmd.fb), ("cmd_yaw", None if cmd is None else cmd.yaw),
                           ("x", model.x), ("y", model.y), ("z", model.z), ("heading_deg", math.degrees(model.heading)),
                           ("pitch_deg", model.pitch_deg), ("walker_x", wx), ("walker_y", wy),
                           ("range_m", math.hypot(dx, dy)), ("bearing_deg", math.degrees(bearing)),
                           ("in_view", head_in_view(g)), ("v_top", None if g is None else g[1]),
                           ("est_range_m", None if cmd is None else cmd.range_m),
                           ("est_bearing_deg", None if cmd is None else cmd.bearing_deg),
                           ("walker_m_s", None if cmd is None else cmd.walker_m_s), ("lost", lost),
                           ("action", action), ("holding", model.holding),
                           ("planner", None if cmd is None else cmd.planner)):
                trace[key].append(v)

        while tick * pdt < t_stop - _EPS and tick < tick_end:
            model.step(lr, fb, ud, yaw)
            tick += 1
            wx, wy = walker.position(tick - tick0, model.x, model.y)
            w.wx, w.wy = wx, wy
            g_new = w.geom()
            det.capture((tick - 1) * pdt, tick * pdt, g, g_new)
            g = g_new
            dx, dy = wx - model.x, wy - model.y
            rng_t = math.hypot(dx, dy)
            dz = model.z - head
            s, c = math.sin(model.heading), math.cos(model.heading)
            phys(tick * pdt, rng_t, math.degrees(math.atan2(dx * c - dy * s, dx * s + dy * c)), head_in_view(g),
                 _below_frame(g), dz - above, math.sqrt(rng_t * rng_t + dz * dz), pdt)
            if rng_t < collide:
                collided = True
                break
        if collided:
            i_w = tick - tick0
            mx.collided(rng_t, math.sqrt(rng_t * rng_t + (model.z - head) ** 2),
                        max(0, n_follow - max(i_w, m_phys0)), max(0.0, t_end - max(tick * pdt, mx.t_metric)))
            ret += reward.collision(t_end - t_stop)
            break
        t_c = t_next
        k += 1

    # control steps in the full episode's metric window (the clock is the same for every arm)
    n_cm_full, t = 0, t_start
    for per in periods[k0:]:
        if t >= t_end - _EPS:
            break
        n_cm_full += t >= mx.t_metric - _EPS
        t += per
    metrics = mx.result(n_follow - m_phys0, n_cm_full)
    metrics["start_failed"] = float(not w.started)
    if trace is not None:
        trace["meta"][0].update(cancelled=cancelled, yaw_source_error=ctrl.yaw_source_error,
                                governor_events=list(gov.events), start_attempts=attempt + 1)
    return EpisodeResult(seed=int(seed), profile=profile, arm=arm, ret=ret, terms=dict(reward.sums), metrics=metrics,
                         n_ticks=int(round((tick - tick0) * pdt / DT)), wall_s=time.perf_counter() - wall0,
                         brain_s=timed.s if timed is not None else 0.0, trace=trace)
