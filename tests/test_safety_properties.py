"""Property tests of the SafetyGovernor (kite/drone/safety.py) on random command / time / telemetry sequences.

A fake drone and a fake clock, no sleeps. The invariants are the ones the module states in its docstring:
  * every rc that reaches the drone is clamped to +-max_rc (a NaN / inf stick is a hover, 0)
  * flying above max_altitude_m the climb is cut (and pushed down), below min_altitude_m the descent is cut
  * discrete up / down moves that would cross a limit are refused ('error: safety ...'), nothing reaches the drone
  * battery below min_battery_pct, or flight time above max_flight_s -> 'land'; a land that was asked for is sent at
    once, and re-sent every 2 s while it still flies and is not busy (never more often)
  * no new frame for frame_timeout_s, or stale telemetry -> 'hover' with rc 0, and the mission's rc is held back
    until a check says 'ok' again; NaN or far-future stamps count as stale
  * while landing, the mission's rc / moves / rotations never reach the drone
Telemetry may be None (unknown), NaN, +-inf, huge or negative: the governor must never raise on it.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from kite.config import SafetyCfg
from kite.drone import Drone, SafetyGovernor
from kite.drone.base import DIRECTIONS
from kite.drone.safety import SOFT_BAND_M
from kite.types import Telemetry

NAN, INF = float("nan"), float("inf")
WEIRD = [NAN, INF, -INF, 1e308, -1e308, 0.0, -0.0, -1.0, 1e-300]

FAST = settings(max_examples=60, stateful_step_count=40, deadline=None, database=None,
                suppress_health_check=[HealthCheck.too_slow])
THOROUGH = settings(max_examples=500, stateful_step_count=60, deadline=None, database=None,
                    suppress_health_check=[HealthCheck.too_slow])


def sensor(lo: float, hi: float):
    """A reading: mostly plausible, sometimes unknown (None) or garbage."""
    return st.one_of(st.floats(lo, hi), st.none(), st.sampled_from(WEIRD), st.floats(allow_nan=True, allow_infinity=True))


stick = st.one_of(st.integers(-1000, 1000), st.floats(-1e6, 1e6), st.sampled_from(WEIRD),
                  st.floats(allow_nan=True, allow_infinity=True))
stamp_age = st.one_of(st.floats(-0.05, 3.0), st.sampled_from(WEIRD), st.floats(allow_nan=True, allow_infinity=True))

cfgs = st.builds(
    lambda max_rc, lo, span, bat, flight, ft, tt: SafetyCfg(max_rc=max_rc, min_altitude_m=lo, max_altitude_m=lo + span,
                                                            min_battery_pct=bat, max_flight_s=flight,
                                                            frame_timeout_s=ft, telemetry_timeout_s=tt),
    st.integers(1, 100), st.floats(0.1, 1.5), st.floats(0.5, 3.0), st.floats(0.0, 50.0), st.floats(1.0, 600.0),
    st.floats(0.1, 3.0), st.floats(0.1, 3.0),
)


class FakeDrone(Drone):
    """Records what reaches it; the test sets flying / busy / telemetry."""

    def __init__(self):
        self.tel = Telemetry(t=0.0)
        self.rcs: list[tuple] = []
        self.moves: list[tuple] = []
        self.lands = 0
        self._flying = self._busy = False

    def takeoff(self):
        self._flying = True

    def land(self):
        self.lands += 1

    def emergency(self):
        self._flying = False
        return True

    def rc(self, lr, fb, ud, yaw):
        self.rcs.append((lr, fb, ud, yaw))

    def move(self, direction, cm):
        self.moves.append((direction, cm))

    def rotate(self, deg):
        self.moves.append(("rotate", deg))

    def stop(self):
        pass

    def busy(self):
        return self._busy

    def last_result(self):
        return "ok"

    def telemetry(self):
        return self.tel

    def frame_source(self):
        raise RuntimeError("no video")

    @property
    def flying(self):
        return self._flying


def finite(v) -> bool:
    return v is not None and math.isfinite(v)


def known(v) -> bool:
    """An altitude the governor can act on: None and NaN are unknown (no limit applies), +-inf is known."""
    return v is not None and not math.isnan(v)


def stale(now: float, t, timeout: float) -> bool:
    return t is None or not finite(t) or not finite(now) or abs(now - t) > timeout


class GovernorMachine(RuleBasedStateMachine):
    @initialize(cfg=cfgs)
    def setup(self, cfg):
        self.cfg = cfg
        self.t = 1000.0  # one fake clock: the governor's re-send clock and check()'s `now`
        self.drone = FakeDrone()
        self.drone.tel = Telemetry(t=self.t, height_m=1.0, battery_pct=80.0)
        self.gov = SafetyGovernor(self.drone, cfg, clock=lambda: self.t)
        # model
        self.land_requested = False
        self.hovering = False
        self.last_land_t: float | None = None  # when the last land reached the drone
        self.takeoff_t: float | None = None

    # ------------------------------------------------------------------ world
    @rule(flying=st.booleans())
    def set_flying(self, flying):
        self.drone._flying = flying

    @rule(busy=st.booleans())
    def set_busy(self, busy):
        self.drone._busy = busy

    @rule(h=sensor(-1.0, 6.0), tof=sensor(-1.0, 9.0), battery=sensor(-10.0, 110.0), age=stamp_age)
    def set_telemetry(self, h, tof, battery, age):
        self.drone.tel = Telemetry(t=self.t - age, height_m=h, tof_m=tof, battery_pct=battery)

    # ------------------------------------------------------------------ periodic check
    @rule(dt=st.one_of(st.floats(0.0, 3.0), st.floats(0.0, 1000.0)), frame_age=st.one_of(st.none(), stamp_age))
    def check(self, dt, frame_age):
        self.t += dt
        d, c = self.drone, self.cfg
        frame_t = None if frame_age is None else self.t - frame_age
        n_rc, n_land = len(d.rcs), d.lands
        action = self.gov.check(self.t, frame_t)
        assert action in ("ok", "hover", "land")
        if not d.flying:
            self.takeoff_t = None
            self.hovering = False
            assert action == "ok", "the governor must not land / hover a drone that is not flying"
            assert len(d.rcs) == n_rc and d.lands == n_land
            return
        if self.takeoff_t is None:
            self.takeoff_t = self.t
        bat = d.tel.battery_pct
        must_land = ((bat is not None and bat < c.min_battery_pct) or self.t - self.takeoff_t > c.max_flight_s
                     or self.land_requested)
        if must_land:
            assert action == "land"
            assert self.gov.land_requested
            assert len(d.rcs) == n_rc, "no stick while landing"
            due = not d.busy() and (self.last_land_t is None or self.t - self.last_land_t > 2.0)
            if not self.land_requested:
                assert d.lands == n_land + 1, "the first land is sent at once"
            elif due:
                assert d.lands == n_land + 1, "a lost land is re-sent after 2 s"
            else:
                assert d.lands == n_land, "a land is never re-sent within 2 s or while busy"
            if d.lands > n_land:
                self.last_land_t = self.t
            self.land_requested = True
            return
        assert d.lands == n_land
        if stale(self.t, frame_t, c.frame_timeout_s) or stale(self.t, d.tel.t, c.telemetry_timeout_s):
            assert action == "hover"
            self.hovering = True
            if d.busy():
                assert len(d.rcs) == n_rc  # an rc would abort the running discrete command
            else:
                assert d.rcs[n_rc:] == [(0, 0, 0, 0)]
        else:
            assert action == "ok"
            self.hovering = False
            assert len(d.rcs) == n_rc

    # ------------------------------------------------------------------ the mission's commands
    @rule(lr=stick, fb=stick, ud=stick, yaw=stick)
    def rc(self, lr, fb, ud, yaw):
        d, c = self.drone, self.cfg
        n = len(d.rcs)
        self.gov.rc(lr, fb, ud, yaw)
        if self.land_requested or self.hovering:
            assert len(d.rcs) == n, "the mission's rc must not reach the drone while landing / hovering for safety"
            return
        assert len(d.rcs) == n + 1
        out = d.rcs[-1]
        m = c.max_rc
        for v in out:
            assert isinstance(v, int) and -m <= v <= m, (out, m)
        for want, got in ((lr, out[0]), (fb, out[1]), (yaw, out[3])):
            if not math.isfinite(want):
                assert got == 0
            else:
                assert got == max(-m, min(m, round(want)))
        if not math.isfinite(ud):
            requested_ud = 0
        else:
            requested_ud = max(-m, min(m, round(ud)))
        got_ud = out[2]
        if not d.flying:
            assert got_ud == requested_ud  # on the ground only the clamp applies
            return
        high, clear = d.tel.floor_altitude_m(), d.tel.altitude_m()
        near_ceiling = known(high) and high > c.max_altitude_m - SOFT_BAND_M
        near_floor = known(clear) and clear < c.min_altitude_m + SOFT_BAND_M
        if known(high) and high >= c.max_altitude_m:
            assert got_ud <= 0, "no climb at or above the ceiling"
        if known(clear) and clear <= c.min_altitude_m:
            assert got_ud >= 0, "no descent at or below the floor"
        if known(high) and high > c.max_altitude_m and not near_floor:
            assert got_ud < 0, "pushed down above the ceiling (unless something is right below it)"
        if near_ceiling and requested_ud > 0:
            assert got_ud <= requested_ud  # the climb is tapered (or reversed above the ceiling), never boosted
            if high <= c.max_altitude_m:
                assert got_ud >= 0  # inside the band it only slows down
        if near_floor and requested_ud < 0 and not near_ceiling:
            assert requested_ud <= got_ud <= 0  # the descent is tapered, never boosted
        if not near_ceiling and not near_floor:
            assert got_ud == requested_ud  # away from both limits the stick passes unchanged

    @rule(direction=st.sampled_from(DIRECTIONS), cm=st.integers(20, 500))
    def move(self, direction, cm):
        d, c = self.drone, self.cfg
        n = len(d.moves)
        self.gov.move(direction, cm)
        high, clear = d.tel.floor_altitude_m(), d.tel.altitude_m()
        refuse = self.land_requested
        if direction == "up" and known(high) and high + cm / 100.0 > c.max_altitude_m:
            refuse = True
        if direction == "down" and known(clear) and clear - cm / 100.0 < c.min_altitude_m:
            refuse = True
        if refuse:
            assert len(d.moves) == n and self.gov.last_result().startswith("error: safety")
        else:
            assert d.moves[n:] == [(direction, cm)] and self.gov.last_result() == "ok"

    @rule(direction=st.sampled_from(DIRECTIONS), cm=st.one_of(st.integers(-10**6, 19), st.integers(501, 10**6)))
    def bad_move(self, direction, cm):
        n = len(self.drone.moves)
        with pytest.raises(ValueError):
            self.gov.move(direction, cm)
        assert len(self.drone.moves) == n

    @rule(deg=st.integers(-360, 360).filter(bool))
    def rotate(self, deg):
        n = len(self.drone.moves)
        self.gov.rotate(deg)
        if self.land_requested:
            assert len(self.drone.moves) == n and self.gov.last_result().startswith("error: safety")
        else:
            assert self.drone.moves[n:] == [("rotate", deg)]

    @rule()
    def pilot_lands(self):  # the 'l' key: always sent, also again
        n = self.drone.lands
        self.gov.land()
        assert self.drone.lands == n + 1 and self.gov.land_requested
        self.land_requested = True
        self.last_land_t = self.t

    @rule()
    def emergency(self):
        assert self.gov.emergency() is True
        assert not self.drone.flying

    # ------------------------------------------------------------------ always
    @invariant()
    def model_agrees(self):
        if hasattr(self, "gov"):
            assert self.gov.land_requested == self.land_requested
            assert self.gov.hovering_for_safety == self.hovering


test_governor_state_machine = GovernorMachine.TestCase
test_governor_state_machine.settings = FAST


class ThoroughGovernorMachine(GovernorMachine):
    pass


test_governor_state_machine_thorough = pytest.mark.slow(ThoroughGovernorMachine.TestCase)
test_governor_state_machine_thorough.settings = THOROUGH


# ------------------------------------------------------------------ single-call properties
@FAST
@given(cfg=cfgs, sticks=st.tuples(stick, stick, stick, stick), h=sensor(-1.0, 6.0), tof=sensor(-1.0, 9.0),
       flying=st.booleans())
def test_every_rc_that_reaches_the_drone_is_within_max_rc(cfg, sticks, h, tof, flying):
    d = FakeDrone()
    d._flying = flying
    d.tel = Telemetry(t=0.0, height_m=h, tof_m=tof)
    gov = SafetyGovernor(d, cfg, clock=lambda: 0.0)
    gov.rc(*sticks)
    (out,) = d.rcs
    assert all(isinstance(v, int) and abs(v) <= cfg.max_rc for v in out)


@FAST
@given(cfg=cfgs, now=st.one_of(st.floats(0.0, 1e10), st.sampled_from(WEIRD)),
       frame_t=st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True)),
       tel_t=st.floats(allow_nan=True, allow_infinity=True), battery=sensor(-10.0, 110.0))
def test_check_never_raises_and_lands_or_hovers_on_bad_data(cfg, now, frame_t, tel_t, battery):
    d = FakeDrone()
    d._flying = True
    d.tel = Telemetry(t=tel_t, height_m=1.0, battery_pct=battery)
    gov = SafetyGovernor(d, cfg, clock=lambda: 0.0)
    action = gov.check(now, frame_t)
    if battery is not None and battery < cfg.min_battery_pct:
        assert action == "land" and d.lands == 1
    elif stale(now, frame_t, cfg.frame_timeout_s) or stale(now, tel_t, cfg.telemetry_timeout_s):
        assert action == "hover" and d.rcs == [(0, 0, 0, 0)]
    else:
        assert action == "ok"
