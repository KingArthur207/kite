"""kite.follow_tuned: the trained follow's controller (kite/fly/behind/DESIGN.md, section 2).

The promise that matters most: with TunedCfg() defaults it IS the flown follow, rc for rc. The identity tests drive
FollowBehind and TunedFollowBehind through the same 60 s of scripted perception at 20 Hz (every branch of the step:
approach at the cap, receding, stopping, closing in fast, too close, box too big, dropouts of 0.2 / 0.6 / 1.5 / 3 s,
lost while close, no range, no altitude) and compare every stick, for the yaw law and for the fruit fly brain. The
two runs are sequential, not interleaved: FlyYaw shares one FlySteer per settings per process, and each follow start
resets it, so each run sees the same deterministic brain.

The speed tests close the loop with a 1-D kinematic model (the person walks along the drone's forward axis; the
drone's speed follows the stick through a lag; observations arrive late), enough to check "faster only when the
person is faster" without the full simulator.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import pytest

from kite.follow_behind import (BehindCfg, CameraModel, Ctx, FollowBehind, FollowMission, Perception, PersonDet,
                                PersonObs, PersonResult, deadband)
from kite.follow_tuned import (FWD_CAP_LIMIT, RC_PER_MPS, SMOOTH2_PATH, TunedCfg, TunedFollowBehind,
                               TunedFollowMission, load_model, save_model)
from kite.types import Frame, Telemetry

DT = 0.05  # run/pilot.py's 20 Hz tick
W, H = 960, 720
FX = 1077.0
IMAGE = np.zeros((1, 1, 3), np.uint8)  # the behaviour never reads pixels


class FakeDrone:
    """Records every rc; telemetry is whatever the test set last."""

    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.tel = Telemetry(0.0, height_m=1.9, tof_m=1.9, pitch_deg=0.0)

    def rc(self, lr, fb, ud, yaw) -> None:
        self.sent.append((lr, fb, ud, yaw))

    def telemetry(self) -> Telemetry:
        return self.tel

    def busy(self) -> bool:
        return False

    def last_result(self) -> str:
        return "ok"

    def stop(self) -> None:
        pass

    def move(self, direction: str, cm: int) -> None:
        pass

    def rotate(self, deg: int) -> None:
        pass


def person(range_m: float | None, bearing_deg: float = 0.0, *, lo: float | None = -1.0, est: float | None = -1.0,
           w_px: float | None = None, top_px: float = 300.0, facing: float | None = None,
           facing_conf: float = 0.0) -> PersonObs:
    """A PersonObs as PersonEstimator would give it. range_m sets the box; est (default = range_m) is the range
    estimate (None = no cue this frame), lo (default 0.95 x range) the conservative cue."""
    r = range_m if range_m is not None else 1.8
    w = FX * 0.48 / r if w_px is None else w_px
    cx = 480.0 + FX * math.tan(math.radians(bearing_deg))
    det = PersonDet("person", 0.9, (cx - w / 2, top_px, cx + w / 2, float(H)), None, 1)
    return PersonObs(det, bearing_deg, -5.0, range_m if est == -1.0 else est, "head_elev", facing, facing_conf,
                     0.95 * r if lo == -1.0 else lo)


@dataclass
class Tick:
    now: float
    new_frame: bool
    frame: Frame | None
    res: PersonResult | None
    tel: Telemetry | None


def _seen_at(t: float) -> dict | None:
    """The scripted person at time t: None = not detected on this frame (a dropout / lost), else person() kwargs."""
    if t < 1.0:
        return None  # never seen yet: the search turn
    if t < 3.0:
        return dict(range_m=2.6, bearing_deg=8.0)  # standing far: approach at the cap
    if t < 10.0:
        return dict(range_m=2.0 + 0.13 * (t - 3.0), bearing_deg=6.0 * math.sin(t))  # walking away
    if t < 13.0:
        return dict(range_m=2.9 - 0.3 * (t - 10.0), bearing_deg=-3.0)  # stopped: we close in
    if t < 16.0:
        return dict(range_m=2.0 - 0.45 * (t - 13.0), bearing_deg=4.0)  # closing in fast: back away
    if t < 18.0:
        return dict(range_m=0.5, bearing_deg=10.0, lo=0.45)  # too close: back off and climb
    if t < 20.0:
        return dict(range_m=1.0, bearing_deg=-5.0, w_px=0.8 * W)  # box too big
    for a, b in ((22.0, 22.2), (24.0, 24.6), (27.0, 28.5), (31.0, 34.0)):  # dropouts 0.2 / 0.6 / 1.5 / 3 s
        if a <= t < b:
            return None
    if t < 31.0:
        return dict(range_m=1.8 + 0.1 * math.sin(2 * t), bearing_deg=15.0 * math.sin(0.7 * t) - (12 if t > 28.5 else 0))
    if t < 36.0:
        return dict(range_m=1.8 - 0.3 * (t - 34.0), bearing_deg=3.0)  # coming closer
    if t < 37.0:
        return dict(range_m=0.75, bearing_deg=5.0)  # nearly too close
    if t < 40.0:
        return None  # lost while close: they walked under the camera
    if t < 42.0:
        return dict(range_m=1.6, bearing_deg=-6.0, top_px=0.9 * H)  # head at the bottom of the frame
    if t < 43.5:
        return None  # lost while close again (head-at-bottom case)
    if t < 48.0:  # no range cue on some frames, conservative cue missing on others, turned (facing)
        k = int(t / DT)
        return dict(range_m=2.1 + 0.2 * math.sin(t), bearing_deg=-8.0, est=None if k % 3 == 0 else -1.0,
                    lo=None if k % 4 == 0 else -1.0, facing=100.0 * math.sin(0.5 * t), facing_conf=0.8)
    if t < 51.0:
        return dict(range_m=2.0, bearing_deg=2.0, est=None, lo=None)  # no range at all
    if t < 55.0:
        return dict(range_m=1.9 - 0.35 * (t - 51.0), bearing_deg=-2.0)  # closing, no altitude: too close at the end
    return dict(range_m=1.8 + 0.25 * (t - 55.0), bearing_deg=5.0 * math.sin(3 * t))  # walking away fast


def scripted_inputs(seconds: float = 60.0, seed: int = 0) -> list[Tick]:
    """One input per 20 Hz control tick (with loop jitter), frames on most ticks (every 7th has none: between
    frames), range noise, a telemetry altitude that wanders, climbs past the back-off ceiling and goes missing."""
    rng = np.random.default_rng(seed)
    out, seq, frame, res = [], 0, None, None
    for i in range(int(round(seconds / DT))):
        t = i * DT + float(rng.uniform(0.0, 0.01))
        alt = 1.9 + 0.04 * math.sin(0.9 * t) + float(rng.normal(0.0, 0.01))
        if 16.0 <= t < 18.0:
            alt = 1.9 + 0.25 * (t - 16.0)  # climbing during the back-off, past altitude_m + 0.25
        tel = Telemetry(t, height_m=round(alt, 2), tof_m=round(alt + float(rng.normal(0, 0.01)), 2), pitch_deg=0.0)
        if 51.0 <= t < 55.0:
            tel = Telemetry(t)  # no altitude
        new = i % 7 != 6
        if new:
            seq += 1
            frame = Frame(IMAGE, t - float(rng.uniform(0.0, 0.03)), seq)
            s = _seen_at(t)
            if s is not None and s.get("range_m") is not None:
                s = {**s, "range_m": s["range_m"] * (1.0 + float(rng.normal(0.0, 0.02)))}
            p = person(**s) if s is not None else None
            res = PersonResult(frame.t, seq, (W, H), [p] if p else [], p, p is not None, 0.0 if p else math.inf)
        out.append(Tick(t, new, frame, res, tel))
    return out


def drive(behaviour, inputs: list[Tick]) -> list[tuple]:
    """Feed the ticks to a behaviour the way FollowMission.sense/step does; every rc it sent, in order."""
    drone = FakeDrone()
    ctx = Ctx(drone, None)
    behaviour.start(ctx)
    for it in inputs:
        ctx.now, ctx.tel, ctx.new_frame = it.now, it.tel, it.new_frame
        if it.new_frame:
            ctx.frame, ctx.res = it.frame, it.res
        behaviour.step(ctx)
    assert len(drone.sent) == len(inputs)
    return drone.sent


class StubYawSource:
    """A deterministic kite.fly-style yaw source (the --yaw retrained / pd path of the app)."""

    def reset(self) -> None:
        self.n = 0

    def yaw(self, now, frame_t, bearing, rng, fallback):
        self.n += 1
        return 0.8 * bearing + 0.1 * (self.n % 5) + 0.01 * rng


@pytest.fixture(scope="module")
def inputs() -> list[Tick]:
    return scripted_inputs()


def _typed(rcs: list[tuple]) -> list[tuple]:
    return [tuple((type(v).__name__, v) for v in rc) for rc in rcs]


# ------------------------------------------------------------------ identity with the flown follow
IDENTITY_CASES = {
    "pid": dict(steering="pid"),
    "fly": dict(),  # the flown app: smooth2 steering
    "pid-orbit": dict(steering="pid", orbit_gain=0.5),  # --orbit
    "pid-custom": dict(steering="pid", dist_gain=55.0, max_rc_forward=30, max_rc_backoff=45, dist_deadband_m=0.05,
                       alt_gain=45.0, max_rc_up=25, approach_rate_mps=0.15, min_range_m=0.7),
}


@pytest.mark.parametrize("case", list(IDENTITY_CASES))
def test_defaults_send_bit_identical_rc(inputs, case):
    kw = IDENTITY_CASES[case]
    flown = drive(FollowBehind(BehindCfg(**kw)), inputs)
    tuned = drive(TunedFollowBehind(TunedCfg(**kw)), inputs)
    assert _typed(tuned) == _typed(flown)
    # the script really reaches every branch of the step
    fb = [rc[1] for rc in flown]
    assert max(fb) == kw.get("max_rc_forward", 35)  # approach at the cap
    assert -kw.get("max_rc_backoff", 50) in fb  # too close / box too big / lost while close
    assert any(abs(rc[0]) == kw.get("max_rc_backoff", 50) for rc in flown)  # lost while close: to the side
    assert any(rc[3] == 20 for rc in flown) and any(rc[3] == -20 for rc in flown)  # search turns both ways
    assert max(rc[2] for rc in flown) == kw.get("max_rc_up", 30)  # climb (altitude unknown)
    if case == "pid-orbit":
        assert any(rc[0] not in (0, 50, -50) for rc in flown)


def test_yaw_source_path_is_identical(inputs):
    flown = drive(FollowBehind(BehindCfg(steering="pid"), yaw_source=StubYawSource()), inputs[:600])
    tuned = drive(TunedFollowBehind(TunedCfg(steering="pid"), yaw_source=StubYawSource()), inputs[:600])
    assert _typed(tuned) == _typed(flown)


def test_plain_behind_cfg_is_promoted_with_flown_defaults(inputs):
    cfg = BehindCfg(steering="pid", dist_gain=48.0)
    b = TunedFollowBehind(cfg)
    assert isinstance(b.c, TunedCfg) and b.c.dist_gain == 48.0 and b.c.fwd_cap_fast == 35
    assert _typed(drive(b, inputs[:400])) == _typed(drive(FollowBehind(cfg), inputs[:400]))
    t = TunedCfg()
    assert TunedFollowBehind(t).c is t


# ------------------------------------------------------------------ the mission
def _pose_det(r: float, bearing_deg: float, alt: float = 1.9, person_h: float = 1.70) -> PersonDet:
    """A box as the camera at `alt` sees a person `r` m away (head top from the elevation; feet below the frame)."""
    cx = 480.0 + FX * math.tan(math.radians(bearing_deg))
    y1 = 360.0 + FX * (alt - person_h) / r
    w = FX * 0.48 / r
    return PersonDet("person", 0.9, (max(0.0, cx - w / 2), y1, min(float(W), cx + w / 2), float(H)), None)


def _drive_mission(mission_cls, cfg, seconds: float = 20.0) -> list[tuple]:
    """FollowMission / TunedFollowMission through the real Perception (tracker, lock, estimator) on scripted boxes,
    as run/pilot.py's _tick drives it."""
    drone = FakeDrone()
    current: list = []
    perception = Perception(cfg, lambda image: list(current), CameraModel(W, H, FX, FX, 480.0, 360.0))
    m = mission_cls(drone, perception, cfg)
    rng = np.random.default_rng(3)
    seq = 0
    for i in range(int(seconds / DT)):
        t = 100.0 + i * DT
        drone.tel = Telemetry(t, height_m=1.9, tof_m=1.9, pitch_deg=0.0)
        r = 2.4 + 0.08 * i * DT if i * DT < 8 else max(1.0, 3.04 - 0.35 * (i * DT - 8))
        gone = 12.0 <= i * DT < 13.0
        current[:] = [] if gone else [_pose_det(r * (1 + float(rng.normal(0, 0.01))), 6.0 * math.sin(i * DT))]
        seq += 1
        m.sense(Frame(IMAGE, t, seq), t)
        if i == 0:
            m.begin()
        m.step()
    assert m.state == "FOLLOW"
    return drone.sent


@pytest.mark.parametrize("steering", ["pid", "fly"])
def test_mission_identical_through_real_perception(steering):
    flown = _drive_mission(FollowMission, BehindCfg(steering=steering))
    tuned = _drive_mission(TunedFollowMission, TunedCfg(steering=steering))
    assert len(flown) > 300 and _typed(tuned) == _typed(flown)
    assert -50 in [rc[1] for rc in flown] or min(rc[1] for rc in flown) < 0  # it backed away at the end


def test_mission_follows_with_the_tuned_behaviour():
    drone = FakeDrone()
    cfg = TunedCfg(steering="pid", fwd_cap_fast=63)
    m = TunedFollowMission(drone, Perception(cfg, lambda image: [], CameraModel(W, H, FX, FX, 480.0, 360.0)), cfg)
    m.sense(Frame(IMAGE, 1.0, 1), 1.0)
    m.begin()
    m.step()
    assert m.state == "FOLLOW" and type(m.child) is TunedFollowBehind and m.child.c is cfg
    m2 = FollowMission(drone, Perception(cfg, lambda image: [], CameraModel(W, H, FX, FX, 480.0, 360.0)), cfg)
    m2.sense(Frame(IMAGE, 1.0, 1), 1.0)
    m2.begin()
    m2.step()
    assert type(m2.child) is FollowBehind  # the flown mission is untouched


# ------------------------------------------------------------------ speed matching (closed loop)
class Line:
    """1-D closed loop: person and drone on one line, range = person - drone. The drone's speed follows the forward
    stick (gain m/s per stick, 0.96 m/s per 100 nominal) through a dead time (0.45 s: the Tello lag test's median is
    0.47) and a 0.45 s lag; the camera sees the range 0.25 s late with 2 % noise; a new frame every tick. The dead
    time is what makes "own speed" hard: the stick rises long before the range shows the drone moving."""

    def __init__(self, behaviour, speed_fn, start_range: float = 1.8, seed: int = 1, dead_s: float = 0.45,
                 gain: float = 0.0096, drone=None):
        self.b, self.speed_fn = behaviour, speed_fn
        self.rng = np.random.default_rng(seed)
        self.xp, self.xd, self.vd = start_range, 0.0, 0.0
        self.gain = gain
        self.drone = FakeDrone() if drone is None else drone
        self.ctx = Ctx(self.drone, None)
        self.b.start(self.ctx)
        self.cmd_q: deque = deque([0.0] * max(1, int(round(dead_s / DT))))
        self.obs_q: deque = deque([start_range] * 5)  # 0.25 s video latency
        self.log: list[dict] = []
        self.seq = 0

    def run(self, seconds: float) -> list[dict]:
        t0 = len(self.log) * DT
        for i in range(int(round(seconds / DT))):
            t = t0 + i * DT
            self.obs_q.append(self.xp - self.xd)
            r_seen = self.obs_q.popleft() * (1.0 + float(self.rng.normal(0.0, 0.02)))
            self.seq += 1
            p = person(r_seen)
            ctx = self.ctx
            ctx.now, ctx.new_frame = t, True
            ctx.tel = Telemetry(t, height_m=1.9, tof_m=1.9, pitch_deg=0.0)
            ctx.frame = Frame(IMAGE, t, self.seq)
            ctx.res = PersonResult(t, self.seq, (W, H), [p], p, True, 0.0)
            self.b.step(ctx)
            fb = self.drone.sent[-1][1]
            m = getattr(getattr(self.drone, "cfg", None), "max_rc", None)
            self.cmd_q.append(fb if m is None else max(-m, min(m, fb)))
            target = self.cmd_q.popleft() * self.gain
            self.vd += (1.0 - math.exp(-DT / 0.45)) * (target - self.vd)
            self.xd += self.vd * DT
            self.xp += self.speed_fn(t) * DT
            self.log.append(dict(t=t, fb=fb, range=self.xp - self.xd, vd=self.vd,
                                 v_est=getattr(self.b, "walker_speed_mps", None), r_filt=self.b._range(t),
                                 cap=getattr(self.b, "fwd_cap_rc", None), gate=getattr(self.b, "fast_gate", None)))
        return self.log


FAST = dict(steering="pid", fwd_cap_fast=63, fwd_cap_per_mps=150.0, fwd_ff_gain=0.8)


def walk(speed: float, start: float = 2.0, stop: float = math.inf):
    return lambda t: speed if start <= t < stop else 0.0


def test_fast_walker_gets_the_faster_cap():
    log = Line(TunedFollowBehind(TunedCfg(**FAST)), walk(0.6)).run(30.0)
    fb = [x["fb"] for x in log]
    opened = next(x["t"] for x in log if x["gate"])
    assert 2.0 + 1.0 < opened < 2.0 + 3.0  # the dwell (1 s) after the estimate passes v0, within 3 s of the walk
    assert all(x["fb"] <= 35 for x in log if x["t"] < opened)  # never above the flown cap before the gate
    assert max(fb[: int(7.0 / DT)]) > 55
    assert max(fb) <= FWD_CAP_LIMIT
    late = [x for x in log if x["t"] > 15.0]
    assert abs(np.mean([x["v_est"] for x in late]) - 0.6) < 0.03  # it measures the walker's speed
    # 63 ~ 0.6 m/s: it keeps pace (the gap opened while the estimate built up stays, it does not grow)
    assert late[-1]["range"] < 3.4 and late[-1]["range"] <= late[0]["range"] + 0.02
    flown = Line(FollowBehind(BehindCfg(steering="pid")), walk(0.6)).run(30.0)
    assert max(x["fb"] for x in flown) == 35 and flown[-1]["range"] > 5.0  # the flown follow falls behind


def test_brisk_walker_is_matched_near_the_distance():
    log = Line(TunedFollowBehind(TunedCfg(**FAST)), walk(0.5)).run(30.0)
    late = [x for x in log if x["t"] > 20.0]
    assert all(2.0 < x["range"] < 2.6 for x in late)
    assert abs(np.mean([x["fb"] for x in late]) - 0.5 * RC_PER_MPS) < 4  # 0.5 m/s worth of stick


def test_standing_person_never_gets_more_than_max_rc_forward():
    log = Line(TunedFollowBehind(TunedCfg(**FAST)), lambda t: 0.0, start_range=2.2).run(30.0)
    assert max(x["fb"] for x in log) <= 35
    assert 1.6 < log[-1]["range"] < 2.0  # it closed to the follow distance, at the flown speed


@pytest.mark.parametrize("dead_s", [0.33, 0.47, 0.61])  # the training profile's dead-time range and its median
@pytest.mark.parametrize("speed", [0.2, 0.25, 0.3])
def test_slow_walker_never_gets_the_fast_cap(dead_s, speed):
    """Below fwd_cap_v0_mps (0.35) the drone never goes faster than the flown cap, whatever the Tello's dead time
    (the own speed is the stick that reached the drone, through the same delay, over the past the range rate
    describes: the stick rising long before the range shows it no longer reads a slow walker as fast)."""
    for seed in range(3):
        log = Line(TunedFollowBehind(TunedCfg(**FAST)), walk(speed), seed=seed, dead_s=dead_s).run(40.0)
        assert max(x["fb"] for x in log) <= 35
        assert not any(x["gate"] for x in log)
        late = [x["v_est"] for x in log if x["t"] > 15.0]
        assert abs(np.mean(late) - speed) < 0.03  # it measures them right


def test_a_drone_gaining_on_them_never_opens_the_gate():
    """The gate needs the range not shrinking: however fast an estimate reads, a drone closing in on the person at
    its present speed does not get a higher cap; with the range steady it opens after fwd_cap_dwell_s."""
    def run(ranges):
        b = TunedFollowBehind(TunedCfg(**FAST))
        b._walker_speed = lambda now, rate, own: None if rate is None else 0.6  # "walking away fast"
        ticks = [Tick(i * DT, True, Frame(IMAGE, i * DT, i + 1),
                      PersonResult(i * DT, i + 1, (W, H), [person(r)], person(r), True, 0.0),
                      Telemetry(i * DT, height_m=1.9, tof_m=1.9)) for i, r in enumerate(ranges)]
        drone, ctx, gates = FakeDrone(), None, []
        ctx = Ctx(drone, None)
        b.start(ctx)
        for it in ticks:
            ctx.now, ctx.tel, ctx.new_frame, ctx.frame, ctx.res = it.now, it.tel, True, it.frame, it.res
            b.step(ctx)
            gates.append(b.fast_gate)
        return gates, [rc[1] for rc in drone.sent]

    gates, fb = run([3.5 - 0.3 * i * DT for i in range(80)])  # closing in at 0.3 m/s
    assert not any(gates) and max(fb) <= 35
    gates, fb = run([3.0] * 80)  # steady
    first = gates.index(True) * DT
    assert 0.5 + 1.0 - DT <= first <= 0.5 + 1.0 + 3 * DT  # the rate needs 0.5 s of history, then the 1 s dwell
    assert max(fb) == FWD_CAP_LIMIT


class GovDrone(FakeDrone):
    """A drone that, like kite.drone.safety.SafetyGovernor, says what it clamps every rc to."""

    def __init__(self, max_rc: int) -> None:
        super().__init__()
        self.cfg = type("SafetyCfgLike", (), {"max_rc": max_rc})()
        self.hovering_for_safety = False


def test_the_estimate_counts_the_stick_the_governor_passed():
    """A model asking 63 under a 50 governor: the drone gets 50 (~0.48 m/s). Counting the asked-for 63 as own speed
    would read the walker 0.13 m/s faster than they are and keep the cap high; the passed 50 reads them right."""
    gov = Line(TunedFollowBehind(TunedCfg(**FAST)), walk(0.6), drone=GovDrone(50)).run(30.0)
    assert max(x["fb"] for x in gov) == FWD_CAP_LIMIT  # it asks for 63 ...
    late = [x["v_est"] for x in gov if x["t"] > 15.0]
    assert abs(np.mean(late) - 0.6) < 0.04  # ... and still measures the walker right
    b = TunedFollowBehind(TunedCfg())
    drone = GovDrone(50)
    ctx = Ctx(drone, None, now=1.0)
    b._passed(ctx, 63)
    drone.hovering_for_safety = True
    ctx.now = 1.05
    b._passed(ctx, 63)
    assert list(b._cmd) == [(1.0, 50.0), (1.05, 0.0)]  # clamped; nothing while the governor holds a hover


FAST_W = dict(FAST, approach_model_w=1.0)


def flown_law(c: TunedCfg, x: dict) -> int:
    """What FollowBehind's distance law sends for the same filtered range."""
    e = deadband(x["r_filt"] - c.distance_m, c.dist_deadband_m)
    return int(max(-c.max_rc_forward, min(c.max_rc_forward, e * c.dist_gain)))


@pytest.mark.parametrize("w", [0.0, 1.0])
def test_stopping_person_falls_back_to_the_flown_law(w):
    t_stop = 14.0
    c = TunedCfg(**FAST, approach_model_w=w)
    log = Line(TunedFollowBehind(c), walk(0.6, stop=t_stop)).run(24.0)
    assert min(x["fb"] for x in log if t_stop - 3.0 < x["t"] < t_stop) > 45  # it was going fast
    after = [x for x in log if x["t"] >= t_stop]
    # the raised cap is gone ~1.5 s after the stop (the range lags 0.25 s of video, the estimate ~1 s more)
    assert all(x["fb"] <= 35 for x in after if x["t"] >= t_stop + 1.5)
    assert min(x["range"] for x in after) > 1.5  # no overrun
    assert abs(log[-1]["fb"]) <= 1 and 1.5 < log[-1]["range"] < 2.1
    late = [x for x in after if x["t"] >= t_stop + 2.5]
    if w == 1.0:
        # the approach feed-forward judges their speed with the aligned own speed: a drone braking for a person
        # who stopped is not read as them walking at it, so no needless back-off; then the flown law
        assert min(x["fb"] for x in after) >= -5
        assert all(x["fb"] <= flown_law(c, x) + 2 for x in late)
    else:
        # the flown closing speed (mean stick 0.3-1.3 s ago) reads the braking drone as the person coming at it
        assert min(x["fb"] for x in after) < -30


def test_slew_never_slows_braking():
    """fwd_slew limits speed-ups only: braking from 0.6 m/s for a person who stops is as quick as without it."""
    t_stop = 14.0
    free = Line(TunedFollowBehind(TunedCfg(**FAST_W)), walk(0.6, stop=t_stop)).run(20.0)
    slew = Line(TunedFollowBehind(TunedCfg(**FAST_W, fwd_slew=60.0)), walk(0.6, stop=t_stop)).run(20.0)
    rise = [b["fb"] - a["fb"] for a, b in zip(slew, slew[1:])]
    assert max(rise) <= 60.0 * DT + 1
    m_free = min(x["range"] for x in free if x["t"] >= t_stop)
    m_slew = min(x["range"] for x in slew if x["t"] >= t_stop)
    assert m_slew >= m_free - 0.1 and m_slew > 1.5
    b = TunedFollowBehind(TunedCfg(fwd_slew=60.0))
    b._sent_fwd(1.0, 63.0)
    assert b._slew(1.05, 0.0) == 0.0 and b._slew(1.05, -50.0) == -50.0 and b._slew(1.05, 70.0) == pytest.approx(66.0)


def test_unknown_speed_keeps_the_flown_cap():
    log = Line(TunedFollowBehind(TunedCfg(**FAST)), walk(0.6, start=0.0)).run(1.0)
    early = [x for x in log if x["t"] < 0.45]  # under 0.5 s of range history: no rate, no estimate
    assert early and all(x["v_est"] is None for x in early)
    assert max(x["fb"] for x in log) <= 35
    c = TunedCfg(**FAST)
    assert c.forward_cap(None) == 35 and c.forward_cap(0.3) == 35 and c.forward_cap(0.45) == pytest.approx(50.0)
    assert c.forward_cap(5.0) == FWD_CAP_LIMIT
    assert TunedCfg(fwd_cap_fast=80, fwd_cap_per_mps=500.0).forward_cap(2.0) == FWD_CAP_LIMIT
    assert TunedCfg(max_rc_forward=40, fwd_cap_fast=35, fwd_cap_per_mps=100.0).forward_cap(1.0) == 40  # never below
    wild = TunedCfg(max_rc_forward=80)  # a base cap past the limit is clamped too (and check() refuses it)
    assert wild.forward_cap(None) == FWD_CAP_LIMIT and wild.forward_cap(0.6) == FWD_CAP_LIMIT


@pytest.mark.parametrize("r, too_far", [(1.7, False), (1.85, False), (1.62, False), (2.2, True), (2.6, True)])
def test_feed_forward_only_when_too_far(r, too_far):
    """However fast the estimate says they walk, at or inside the distance (e <= 0) the tuned law is the flown law;
    too far, the feed-forward and the raised cap add to it."""
    tuned = TunedFollowBehind(TunedCfg(**FAST))
    tuned._walker_speed = lambda now, rate, own: 0.6  # an estimate that says: walking away fast
    ticks = [Tick(i * DT, True, Frame(IMAGE, i * DT, i + 1),
                  PersonResult(i * DT, i + 1, (W, H), [person(r)], person(r), True, 0.0),
                  Telemetry(i * DT, height_m=1.9, tof_m=1.9)) for i in range(30)]
    fb_t = [rc[1] for rc in drive(tuned, ticks)]
    fb_f = [rc[1] for rc in drive(FollowBehind(BehindCfg(steering="pid")), ticks)]
    if too_far:
        assert all(a > b for a, b in zip(fb_t, fb_f)) and max(fb_t) <= FWD_CAP_LIMIT
    else:
        assert fb_t == fb_f


# ------------------------------------------------------------------ slew and the back-off climb
def test_slew_limits_speed_ups_but_never_the_back_off():
    c = TunedCfg(steering="pid", fwd_slew=100.0)
    log = Line(TunedFollowBehind(c), lambda t: 0.0, start_range=3.0).run(3.0)
    fb = [0] + [x["fb"] for x in log]
    assert max(b - a for a, b in zip(fb, fb[1:])) <= 100.0 * DT + 1
    assert max(fb) == 35
    # too close right after going forward at full stick: -max_rc_backoff on the very next tick
    b = TunedFollowBehind(c)
    drone = FakeDrone()
    ctx = Ctx(drone, None)
    b.start(ctx)
    for i, r in enumerate([3.0] * 30 + [0.5] * 3):
        t = i * DT
        p = person(r, lo=0.45 if r < 1 else -1.0)
        ctx.now, ctx.new_frame, ctx.tel = t, True, Telemetry(t, height_m=1.9, tof_m=1.9)
        ctx.frame, ctx.res = Frame(IMAGE, t, i + 1), PersonResult(t, i + 1, (W, H), [p], p, True, 0.0)
        b.step(ctx)
    assert drone.sent[29][1] == 35 and drone.sent[30][1] == -50


def test_slew_never_slows_backing_away_from_an_approaching_person():
    c = TunedCfg(steering="pid", fwd_slew=60.0)
    b = TunedFollowBehind(c)
    drone = FakeDrone()
    ctx = Ctx(drone, None)
    b.start(ctx)
    for i in range(60):  # held at 1.8 m, then they walk toward us at 0.8 m/s
        t = i * DT
        r = 1.8 if t < 1.5 else 1.8 - 0.8 * (t - 1.5)
        p = person(r)
        ctx.now, ctx.new_frame, ctx.tel = t, True, Telemetry(t, height_m=1.9, tof_m=1.9)
        ctx.frame, ctx.res = Frame(IMAGE, t, i + 1), PersonResult(t, i + 1, (W, H), [p], p, True, 0.0)
        b.step(ctx)
    fb = [rc[1] for rc in drone.sent]
    drops = [a - b for a, b in zip(fb, fb[1:])]
    assert max(drops) > 60.0 * DT + 1  # a back-away step bigger than the slew allows
    assert min(fb) <= -40


def test_backoff_climb_ceiling():
    """backoff_climb_m replaces the flown 0.25: above altitude_m + it the climb holds altitude instead."""
    drone = FakeDrone()
    for alt in (1.9, 1.97, 2.1, 2.2):
        ctx = Ctx(drone, None, tel=Telemetry(0.0, height_m=alt, tof_m=alt))
        fb, tb = FollowBehind(BehindCfg()), TunedFollowBehind(TunedCfg(backoff_climb_m=0.05))
        assert fb._climb_rc(ctx) == (30 if alt < 2.15 else fb._altitude_rc(ctx))
        assert tb._climb_rc(ctx) == (30 if alt < 1.95 else tb._altitude_rc(ctx))
        assert TunedFollowBehind(TunedCfg())._climb_rc(ctx) == fb._climb_rc(ctx)
    ctx = Ctx(drone, None, tel=Telemetry(0.0, height_m=2.1, tof_m=2.1))
    assert FollowBehind(BehindCfg())._climb_rc(ctx) == 30
    assert TunedFollowBehind(TunedCfg(backoff_climb_m=0.05))._climb_rc(ctx) < 0  # descends back instead
    assert TunedFollowBehind(TunedCfg(backoff_climb_m=0.0))._climb_rc(Ctx(drone, None, tel=None)) == 30  # unknown


def test_check_rejects_bad_knobs():
    TunedCfg().check()
    for kw in (dict(walker_speed_tau_s=0.0), dict(fwd_slew=-1.0), dict(backoff_climb_m=-0.1),
               dict(fwd_ff_gain=float("nan")), dict(max_rc_forward=64), dict(fwd_cap_fast=70), dict(own_tau_s=0.0),
               dict(fwd_cap_dwell_s=-1.0), dict(approach_model_w=-0.5)):
        with pytest.raises(ValueError):
            TunedCfg(**kw).check()


# ------------------------------------------------------------------ model files
def _smooth2() -> dict:
    return json.loads(SMOOTH2_PATH.read_text(encoding="utf-8"))


def _yaws(steer, n: int = 160) -> list[int]:
    steer.reset(seed=0, z_ref_m=1.8, kind="follow")
    out = []
    for i in range(n):
        t = 50.0 + i * DT
        b = None if 60 <= i < 75 else math.radians(12.0 * math.sin(0.8 * t))
        out.append(steer.yaw(t, b, t_frame=None if b is None else t - 0.02))
    return out


def test_fly_model_round_trip_and_flysteer(tmp_path):
    from kite.flybrain.flyfollow.steer import FlySteer

    x = np.clip(np.asarray(_smooth2()["x"]) + 0.07 * np.sin(np.arange(34)), 0.0, 1.0).tolist()
    follow = {"dist_gain": 52.1, "max_rc_forward": 38.4, "fwd_cap_fast": 55, "fwd_cap_per_mps": 110.0,
              "fwd_ff_gain": 0.6, "fwd_slew": 400.0, "backoff_climb_m": 0.1, "fly_max_rc_yaw": 45,
              "fly_deadband": 3.0}
    path = save_model(tmp_path / "behind" / "v3-joint.json", name="v3-joint", follow=follow, brain_x=x,
                      steering="fly", trained={"behind_arm": "V3-JOINT", "cma_seed": 2, "gen": 140, "fitness": 0.71})
    rec = json.loads(path.read_text())
    assert rec["arm"] == "FLY-YAW" and rec["brain"] == "pursuit_core1.npz" and rec["x"] == x
    assert rec["steering"] == "fly" and rec["follow"]["max_rc_forward"] == 38 and "time" in rec["trained"]
    cfg = load_model(path)
    assert isinstance(cfg, TunedCfg) and cfg.steering == "fly" and cfg.model_name == "v3-joint"
    assert cfg.fly_params_path == str(path.resolve())
    assert (cfg.dist_gain, cfg.max_rc_forward, cfg.fwd_cap_fast, cfg.fwd_ff_gain) == (52.1, 38, 55, 0.6)
    assert cfg.fly_max_rc_yaw == 45 and cfg.max_rc_yaw == 45 and cfg.backoff_climb_m == 0.1
    assert cfg.person_height_m == BehindCfg().person_height_m  # untouched settings keep the flown defaults
    again = save_model(tmp_path / "again.json", name="v3-joint", follow={k: getattr(cfg, k) for k in follow},
                       brain_x=json.loads(path.read_text())["x"], steering="fly", trained=rec["trained"])
    assert json.loads(again.read_text()) == rec
    # the model file is a FlySteer params file: same brain as one built from the same x
    plain = tmp_path / "plain_best.json"
    plain.write_text(json.dumps({"arm": "FLY-YAW", "x": x, "brain": "pursuit_core1.npz"}))
    a, b = FlySteer(params_path=path), FlySteer(params_path=plain)
    assert a.arm == b.arm == "FLY-YAW"
    ya = _yaws(a)
    assert ya == _yaws(b) and any(v != 0 for v in ya)
    assert ya != _yaws(FlySteer(params_path=SMOOTH2_PATH))  # and it is not smooth2's brain


def test_follow_only_model_copies_smooth2_and_flies_like_flown(tmp_path, inputs):
    path = save_model(tmp_path / "v3-follow.json", name="v3-follow", follow={}, brain_x=None, steering="fly",
                      trained={"behind_arm": "V3-FOLLOW"})
    rec, s2 = json.loads(path.read_text()), _smooth2()
    assert rec["x"] == s2["x"] and rec["arm"] == s2["arm"] and rec["brain"] == s2["brain"]
    cfg = load_model(path)
    assert cfg.fly_params_path != BehindCfg().fly_params_path  # its own FlySteer, read from the model file
    part = inputs[:500]
    assert _typed(drive(TunedFollowBehind(cfg), part)) == _typed(drive(FollowBehind(BehindCfg()), part))


def test_law_model(tmp_path):
    path = save_model(tmp_path / "v3-law.json", name="v3-law", follow={"yaw_gain": 1.6, "max_rc_yaw": 50},
                      brain_x=None, steering="pid", trained={"behind_arm": "V3-LAW"}, signoff={"recommended": False})
    rec = json.loads(path.read_text())
    assert "x" not in rec and rec["steering"] == "pid" and rec["signoff"] == {"recommended": False}
    cfg = load_model(path)
    assert cfg.steering == "pid" and cfg.yaw_gain == 1.6 and cfg.max_rc_yaw == 50
    assert cfg.fly_params_path == BehindCfg().fly_params_path
    with pytest.raises(ValueError):
        save_model(tmp_path / "bad.json", name="bad", follow={}, brain_x=[0.5] * 34, steering="pid", trained={})


def test_model_file_caps_are_clamped(tmp_path):
    p = tmp_path / "wild.json"
    p.write_text(json.dumps({"name": "wild", "steering": "pid",
                             "follow": {"fwd_cap_fast": 80, "max_rc_forward": 70, "fwd_cap_per_mps": 1000}}))
    cfg = load_model(p)
    assert cfg.fwd_cap_fast == FWD_CAP_LIMIT and cfg.max_rc_forward == FWD_CAP_LIMIT
    log = Line(TunedFollowBehind(replace(cfg, fwd_ff_gain=1.2)), walk(1.0)).run(15.0)
    assert max(x["fb"] for x in log) <= FWD_CAP_LIMIT


def test_model_files_refuse_mistakes(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        save_model(tmp_path / "a.json", name="a", follow={"dist_gian": 40}, brain_x=None, steering="fly", trained={})
    with pytest.raises(ValueError, match="loader"):
        save_model(tmp_path / "a.json", name="a", follow={"steering": "pid"}, brain_x=None, steering="fly",
                   trained={})
    with pytest.raises(ValueError, match="34"):
        save_model(tmp_path / "a.json", name="a", follow={}, brain_x=[0.5] * 10, steering="fly", trained={})
    with pytest.raises(ValueError):
        save_model(tmp_path / "a.json", name="a", follow={}, brain_x=[1.5] * 34, steering="fly", trained={})
    with pytest.raises(ValueError):
        save_model(tmp_path / "a.json", name="a", follow={}, brain_x=None, steering="yolo", trained={})
    nobrain = tmp_path / "nobrain.json"
    nobrain.write_text(json.dumps({"name": "n", "steering": "fly", "follow": {}}))
    with pytest.raises(ValueError, match="brain"):
        load_model(nobrain)
    assert not list(tmp_path.glob("a.json*"))  # nothing half-written
    ok = save_model(tmp_path / "ok.json", name="ok", follow={}, brain_x=None, steering="pid", trained={})
    assert load_model(ok, person_height_m=1.82).person_height_m == 1.82
    with pytest.raises(ValueError):
        load_model(ok, not_a_field=1)
    assert RC_PER_MPS == 104.0


def test_model_files_record_the_governor_they_need(tmp_path):
    """0.6 m/s is stick 63: SafetyCfg's max_rc 50 would clamp it to ~0.48 m/s, so a model file says what it needs."""
    from kite.follow_tuned import model_governor, required_max_rc

    assert required_max_rc(BehindCfg()) == 50  # the flown follow never asks for more than SafetyCfg's 50
    assert required_max_rc(TunedCfg(**FAST)) == FWD_CAP_LIMIT
    assert required_max_rc(TunedCfg(steering="pid", fwd_cap_fast=63)) == 50  # no rise: the cap never passes 35
    path = save_model(tmp_path / "m.json", name="m", follow={"fwd_cap_fast": 63, "fwd_cap_per_mps": 100.0},
                      brain_x=None, steering="pid", trained={}, governor={"trained_max_rc": 65})
    assert model_governor(path) == {"needs_max_rc": 63, "trained_max_rc": 65}
    assert load_model(path).fwd_cap_fast == 63  # the loader ignores it: the app's governor is the app's


def _scripted(c: TunedCfg, rfun, n: int) -> list[int]:
    b = TunedFollowBehind(c)
    drone = FakeDrone()
    ctx = Ctx(drone, None)
    b.start(ctx)
    for i in range(n):
        t = i * DT
        p = person(rfun(t))
        ctx.now, ctx.new_frame, ctx.tel = t, True, Telemetry(t, height_m=1.9, tof_m=1.9)
        ctx.frame, ctx.res = Frame(IMAGE, t, i + 1), PersonResult(t, i + 1, (W, H), [p], p, True, 0.0)
        b.step(ctx)
    return [rc[1] for rc in drone.sent]


def test_approach_low_pass_damps_a_range_flicker_but_still_backs_away():
    """approach_tau_s: a range estimate swinging +-0.35 m at 0.9 Hz (what a camera tilting with the drone's own
    pitch, read with telemetry 0.3 s newer than the frame, looks like) sets the flown approach feed-forward backing
    off at -50; low-passed 0.6 s it does not. A person really walking at the drone at 0.8 m/s still gets the full
    back-off, ~0.3 s later (the too-close override below min_range_m is never filtered)."""
    flicker = lambda t: 2.0 + 0.35 * math.sin(2 * math.pi * 0.9 * t)  # noqa: E731
    walk_at = lambda t: 1.8 if t < 1.5 else max(0.9, 1.8 - 0.8 * (t - 1.5))  # noqa: E731
    flown, damped = TunedCfg(steering="pid"), TunedCfg(steering="pid", approach_tau_s=0.6)
    assert min(_scripted(flown, flicker, 120)) == -50 and min(_scripted(damped, flicker, 120)) > -10
    for c, by in ((flown, 2.0), (damped, 2.3)):
        fb = _scripted(c, walk_at, 80)
        first = next(i * DT for i, f in enumerate(fb) if f <= -30)
        assert first <= by and min(fb) == -50
