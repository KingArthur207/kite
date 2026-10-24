"""Seeded random-sequence tests for the SafetyGovernor (the hypothesis state machine is test_safety_properties.py),
focused on its escalation rules: random command / telemetry / video sequences, invariants checked after
every step. Seeded numpy generators (no extra dependency), so every failure reproduces from its seed.

Invariants:
  * every rc that reaches the drone is int, within +-max_rc on every channel (NaN / inf / huge inputs included)
  * above the ceiling the drone only ever gets ud <= 0 (pushed down), below the floor ud >= 0, and inside the soft
    band the climb / descent is tapered, never more than asked for
  * once a land is asked for (any reason): no stick other than 0 and no discrete move reaches the drone, and while
    it still flies and is not busy a land is re-sent at least every land_retry_s
  * stale video or telemetry: check() says 'hover' (or 'land' once a limit is reached), sends rc 0 and drops the
    mission's rc and moves; after telemetry_land_s / video_land_s of it, it lands
  * battery below min, flight time over max, temperature at max, dead-reckoned distance over the geofence -> land
  * no call ever raises
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kite.config import SafetyCfg
from kite.drone import SafetyGovernor
from kite.drone.base import Drone
from kite.drone.safety import SOFT_BAND_M
from kite.types import Telemetry

TICK = 0.1


class ScriptedDrone(Drone):
    """Records everything the governor lets through. Telemetry, temperature, busy and flying are set by the test."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.tel = Telemetry(t=0.0, height_m=1.5, tof_m=1.5, battery_pct=80.0, vx=0.0, vy=0.0)
        self.temp: float | None = 50.0
        self._flying = False
        self._busy_until = -1.0
        self.now = 0.0
        self.land_works = True  # False: the land is "lost" (Wi-Fi), it keeps flying

    @property
    def flying(self):
        return self._flying

    def temperature_c(self):
        return self.temp

    def takeoff(self):
        self.calls.append(("takeoff",))
        self._flying = True
        self._busy_until = self.now + 0.3

    def land(self):
        self.calls.append(("land", self.now))
        self._busy_until = self.now + 0.4
        if self.land_works:
            self._landing_at = self.now + 0.4

    def emergency(self):
        self.calls.append(("emergency",))
        self._flying = False
        return True

    def rc(self, lr, fb, ud, yaw):
        self.calls.append(("rc", lr, fb, ud, yaw, self.now))

    def move(self, direction, cm):
        self.calls.append(("move", direction, cm))
        self._busy_until = self.now + 0.5

    def rotate(self, deg):
        self.calls.append(("rotate", deg))
        self._busy_until = self.now + 0.3

    def stop(self):
        self.calls.append(("stop",))

    def busy(self):
        if getattr(self, "_landing_at", None) is not None and self.now >= self._landing_at:
            self._flying, self._landing_at = False, None
        return self.now < self._busy_until

    def last_result(self):
        return "ok"

    def telemetry(self):
        return self.tel

    def frame_source(self):
        return None


def random_value(rng):
    """A stick value: mostly in range, sometimes huge, fractional, NaN or inf."""
    r = rng.random()
    if r < 0.05:
        return float("nan")
    if r < 0.08:
        return float(rng.choice([math.inf, -math.inf]))
    if r < 0.15:
        return float(rng.uniform(-1000, 1000))
    if r < 0.2:
        return np.int64(rng.integers(-150, 150))
    return float(rng.uniform(-100, 100))


def run_sequence(seed: int, steps: int = 400, cfg: SafetyCfg | None = None):
    """Drive the governor with a random sequence; check the invariants after every step."""
    rng = np.random.default_rng(seed)
    cfg = cfg or SafetyCfg(max_rc=int(rng.choice([10, 30, 50, 70, 100])), max_flight_s=float(rng.uniform(5, 60)),
                           geofence_m=float(rng.choice([5.0, 20.0, 1e9])), telemetry_land_s=float(rng.uniform(1, 6)),
                           video_land_s=float(rng.uniform(1, 10)))
    d = ScriptedDrone()
    now = [100.0]
    gov = SafetyGovernor(d, cfg, clock=lambda: now[0])
    last_frame = 100.0
    tel_t = 100.0
    stale_since = {"video": None, "telemetry": None}
    took_off = False
    for step in range(steps):
        now[0] += TICK
        d.now = now[0]
        # the world: video and telemetry come and go, telemetry wanders through every band
        if rng.random() < 0.9 or stale_since["video"] is None and rng.random() < 0.5:
            last_frame = now[0] if rng.random() < 0.93 else last_frame
        if rng.random() < 0.93:
            tel_t = now[0]
        h = float(rng.choice([rng.uniform(0.0, 3.5), cfg.max_altitude_m + rng.uniform(-0.5, 0.5),
                              cfg.min_altitude_m + rng.uniform(-0.5, 0.5)]))
        tof = h if rng.random() < 0.7 else float(rng.uniform(0.1, 3.0))
        d.tel = Telemetry(t=tel_t, height_m=h, tof_m=tof, battery_pct=float(rng.choice([80.0, 50.0, 25.0, 19.0])),
                          vx=float(rng.normal(0, 0.6)), vy=float(rng.normal(0, 0.6)))
        d.temp = float(rng.choice([50.0, 70.0, 89.0, 91.0, np.nan])) if rng.random() < 0.3 else d.temp
        # the mission: a random command
        before = len(d.calls)
        op = rng.choice(["rc", "rc", "rc", "rc", "move", "rotate", "land", "takeoff", "stop"],
                        p=[.2, .2, .2, .2, .05, .05, .02, .05, .03])
        hover_before = gov.hovering_for_safety
        landing_before = gov.land_requested
        if op == "rc":
            vals = [random_value(rng) for _ in range(4)]
            gov.rc(*vals)
        elif op == "move":
            gov.move(str(rng.choice(["forward", "back", "up", "down", "left", "right"])), int(rng.integers(20, 200)))
        elif op == "rotate":
            gov.rotate(int(rng.integers(1, 360)) * int(rng.choice([-1, 1])))
        elif op == "land":
            gov.land()
        elif op == "takeoff" and not d.flying and not took_off:
            gov.takeoff()
            took_off = True
        elif op == "stop":
            gov.stop()
        new = d.calls[before:]
        # invariants on what the mission command let through
        for c in new:
            if c[0] == "rc":
                vals = c[1:5]
                assert all(type(v) is int for v in vals), (seed, step, c)
                assert all(abs(v) <= cfg.max_rc for v in vals), (seed, step, c, cfg.max_rc)
                if hover_before or landing_before:
                    raise AssertionError(f"seed {seed} step {step}: mission rc {c} reached the drone during "
                                         f"{'landing' if landing_before else 'safety hover'}")
                high, clear = d.tel.floor_altitude_m(), d.tel.altitude_m()
                above = d.flying and high is not None and high > cfg.max_altitude_m
                below = d.flying and clear is not None and clear < cfg.min_altitude_m
                if above and below:  # too high for the ceiling, but something tall is right below: hold
                    assert vals[2] == 0, (seed, step, "ceiling + floor", high, clear, c)
                elif above:
                    assert vals[2] < 0, (seed, step, "ceiling", high, c)
                elif below:
                    assert vals[2] >= 0, (seed, step, "floor", clear, c)
            if c[0] in ("move", "rotate"):
                assert not (hover_before or landing_before), (seed, step, c)
        # the periodic check
        before = len(d.calls)
        busy = d.busy()  # (may finish a landing)
        flying = d.flying
        action = gov.check(now[0], last_frame)
        new = d.calls[before:]
        assert action in ("ok", "hover", "land")
        for k, stale in (("video", now[0] - last_frame > cfg.frame_timeout_s),
                         ("telemetry", now[0] - tel_t > cfg.telemetry_timeout_s)):
            stale_since[k] = (stale_since[k] if stale_since[k] is not None else now[0]) if stale and flying else None
        if flying:
            must_land = (d.tel.battery_pct < cfg.min_battery_pct or (d.temp is not None and d.temp >= cfg.max_temp_c)
                         or (stale_since["telemetry"] is not None
                             and now[0] - stale_since["telemetry"] >= cfg.telemetry_land_s)
                         or (stale_since["video"] is not None and now[0] - stale_since["video"] >= cfg.video_land_s))
            if must_land:
                assert action == "land", (seed, step, action, d.tel, d.temp, stale_since)
            if gov.distance_m > cfg.geofence_m:
                assert action == "land", (seed, step, "geofence", gov.distance_m)
            if action == "hover" and not busy:
                assert ("rc", 0, 0, 0, 0, now[0]) in new, (seed, step, new)
            if any(stale_since.values()) and action == "ok":
                raise AssertionError(f"seed {seed} step {step}: stale data but 'ok'")
        if gov.land_requested and d.flying and not d.busy():
            lands = [c[1] for c in d.calls if c[0] == "land"]
            assert lands and now[0] - lands[-1] <= cfg.land_retry_s + TICK + 1e-9, (seed, step, lands[-1:], now[0])
        if gov.land_requested and rng.random() < 0.3:
            d.land_works = bool(rng.random() < 0.5)  # some lands get lost
    return gov, d


@pytest.mark.parametrize("seed", range(60))
def test_random_sequences_never_break_a_limit(seed):
    run_sequence(seed)


@pytest.mark.parametrize("max_rc", [5, 15])
def test_the_ceiling_push_never_exceeds_a_small_max_rc(max_rc):
    """Above the ceiling the governor pushes down, but never harder than max_rc (it used to send -20 regardless)."""
    d = ScriptedDrone()
    d._flying = True
    d.tel = Telemetry(t=0.0, height_m=3.0, tof_m=3.0, battery_pct=80.0)
    gov = SafetyGovernor(d, SafetyCfg(max_rc=max_rc), clock=lambda: 0.0)
    gov.rc(0, 0, 0, 0)
    assert d.calls[-1][1:5] == (0, 0, -max_rc, 0)


def test_nan_and_inf_sticks_are_zero_not_an_exception():
    d = ScriptedDrone()
    d._flying = True
    gov = SafetyGovernor(d, SafetyCfg(), clock=lambda: 0.0)
    gov.rc(float("nan"), float("inf"), -float("inf"), np.float64("nan"))
    assert d.calls[-1][1:5] == (0, 0, 0, 0)  # a non-finite stick is garbage: 0, not full stick
    gov.rc(None, "x", 3.7, np.int64(12))
    assert d.calls[-1][1:5] == (0, 0, 4, 12) and all(type(v) is int for v in d.calls[-1][1:5])


def test_soft_band_tapers_climb_and_descent():
    d = ScriptedDrone()
    d._flying = True
    cfg = SafetyCfg(max_rc=100)
    gov = SafetyGovernor(d, cfg, clock=lambda: 0.0)
    d.tel = Telemetry(t=0.0, height_m=cfg.max_altitude_m - SOFT_BAND_M / 2, tof_m=cfg.max_altitude_m - SOFT_BAND_M / 2)
    gov.rc(0, 0, 80, 0)
    assert d.calls[-1][3] == 40  # half way into the band: half the climb
    d.tel = Telemetry(t=0.0, height_m=cfg.min_altitude_m + SOFT_BAND_M / 4, tof_m=cfg.min_altitude_m + SOFT_BAND_M / 4)
    gov.rc(0, 0, -80, 0)
    assert d.calls[-1][3] == pytest.approx(-20, abs=1)


def flying_gov(**kw):
    d = ScriptedDrone()
    d._flying = True
    now = [10.0]
    gov = SafetyGovernor(d, SafetyCfg(**kw), clock=lambda: now[0])
    d.tel = Telemetry(t=now[0], height_m=1.5, tof_m=1.5, battery_pct=80.0, vx=0.0, vy=0.0)
    d.now = now[0]
    return gov, d, now


def test_stale_telemetry_hovers_then_lands():
    gov, d, now = flying_gov(telemetry_land_s=3.0)
    t_tel = now[0]
    actions = []
    for _ in range(50):
        now[0] += 0.1
        d.now = now[0]
        actions.append(gov.check(now[0], now[0]))  # video fine, telemetry frozen at t_tel
    first_hover = actions.index("hover")
    first_land = actions.index("land")
    assert all(a == "hover" for a in actions[first_hover:first_land])
    assert now[0] - 5.0 + (first_land + 1) * 0.1 - t_tel == pytest.approx(1.0 + 3.0 + 0.1, abs=0.15)
    assert ("land", pytest.approx(t_tel + 4.1, abs=0.15)) in [(c[0], c[1]) for c in d.calls if c[0] == "land"]
    gov.rc(0, 40, 0, 0)
    assert not any(c[0] == "rc" and c[2] == 40 for c in d.calls)


def test_stale_video_hovers_blocks_moves_then_lands():
    gov, d, now = flying_gov(video_land_s=4.0)
    frame_t = now[0]
    for i in range(20):
        now[0] += 0.1
        d.now = now[0]
        d.tel = Telemetry(t=now[0], height_m=1.5, tof_m=1.5, battery_pct=80.0)
        gov.check(now[0], frame_t)
    assert gov.hovering_for_safety
    gov.move("forward", 50)
    gov.rotate(30)
    assert gov.last_result().startswith("error: safety") and not any(c[0] in ("move", "rotate") for c in d.calls)
    actions = []
    for i in range(40):
        now[0] += 0.1
        d.now = now[0]
        d.tel = Telemetry(t=now[0], height_m=1.5, tof_m=1.5, battery_pct=80.0)
        actions.append(gov.check(now[0], frame_t))
    assert "land" in actions and any(c[0] == "land" for c in d.calls)
    assert "video stale" in gov.land_reason


def test_video_back_before_the_limit_resumes():
    gov, d, now = flying_gov(video_land_s=4.0)
    frame_t = now[0]
    for _ in range(25):
        now[0] += 0.1
        d.tel = Telemetry(t=now[0], height_m=1.5, tof_m=1.5, battery_pct=80.0)
        gov.check(now[0], frame_t)
    now[0] += 0.1
    assert gov.check(now[0], now[0]) == "ok" and not gov.hovering_for_safety
    for _ in range(25):  # stale again: the timer starts over
        now[0] += 0.1
        d.tel = Telemetry(t=now[0], height_m=1.5, tof_m=1.5, battery_pct=80.0)
        assert gov.check(now[0], now[0] - 1.5) == "hover"


@pytest.mark.parametrize("temp, want", [(85.0, "ok"), (90.0, "land"), (None, "ok"), (float("nan"), "ok")])
def test_temperature(temp, want):
    gov, d, now = flying_gov()
    d.temp = temp
    assert gov.check(now[0], now[0]) == want


def test_geofence_from_dead_reckoned_velocity_whatever_the_axis_signs():
    for sign in (1, -1):
        gov, d, now = flying_gov(geofence_m=10.0)
        gov.check(now[0], now[0])
        a = "ok"
        for i in range(200):  # 0.6 m/s along one axis (any sign), then along the other
            now[0] += 0.1
            v = (0.8 * sign, 0.0) if i < 100 else (0.0, -0.8 * sign)
            d.tel = Telemetry(t=now[0], height_m=1.5, tof_m=1.5, battery_pct=80.0, vx=v[0], vy=v[1])
            a = gov.check(now[0], now[0])
            if a == "land":
                break
        assert a == "land"
        assert gov.distance_m == pytest.approx(10.0, abs=0.1)
        assert "geofence" in gov.land_reason


def test_geofence_ignores_hover_noise_and_missing_velocity():
    gov, d, now = flying_gov(geofence_m=2.0)
    rng = np.random.default_rng(0)
    for _ in range(4000):  # 400 s of +-1 dm/s readings around a hover: nothing adds up
        now[0] += 0.1
        d.tel = Telemetry(t=now[0], height_m=1.5, tof_m=1.5, battery_pct=80.0,
                          vx=float(rng.choice([-0.1, 0.0, 0.1])), vy=float(rng.choice([-0.1, 0.0, 0.1])))
        assert gov.check(now[0], now[0]) in ("ok",) or gov.land_reason.startswith("flight time")
        if gov.land_requested:
            break
    assert gov.distance_m == 0.0
    gov2, d2, now2 = flying_gov(geofence_m=2.0)
    d2.tel = Telemetry(t=now2[0], height_m=1.5, battery_pct=80.0)  # no vgx / vgy at all
    assert gov2.check(now2[0], now2[0]) == "ok"


def test_a_lost_land_is_resent_every_retry_interval_until_it_is_down():
    gov, d, now = flying_gov(land_retry_s=2.0)
    d.land_works = False
    gov.land()
    sent = []
    for _ in range(80):
        now[0] += 0.1
        d.now = now[0]
        d.tel = Telemetry(t=now[0], height_m=1.5, tof_m=1.5, battery_pct=80.0)
        gov.check(now[0], now[0])
        sent = [c[1] for c in d.calls if c[0] == "land"]
    assert len(sent) >= 4 and max(np.diff(sent)) <= 2.0 + 0.1 + 1e-9
    d.land_works = True
    for _ in range(40):
        now[0] += 0.1
        d.now = now[0]
        d.busy()
        gov.check(now[0], now[0])
    assert not d.flying


def test_no_takeoff_after_a_land_in_the_same_run():
    gov, d, now = flying_gov()
    gov.land()
    d._flying = False
    gov.takeoff()
    assert ("takeoff",) not in d.calls and "already landed" in gov.last_result()
