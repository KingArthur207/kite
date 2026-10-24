"""Safety governor: wraps any Drone; limits always win over the mission.

    drone = SafetyGovernor(TelloDrone(...), cfg.safety)
    ...every loop:  action = drone.check(now, last_frame_t)   # 'ok' | 'hover' | 'land'

Rules (config.SafetyCfg):
  * every rc channel clamped to +-max_rc; a NaN / inf stick is 0 (never an exception in the control loop)
  * altitude ceiling/floor: rc climb is cut above max_altitude_m (and pushed down, within max_rc), descent is cut
    below min_altitude_m, both tapered over SOFT_BAND_M; discrete up/down moves that would cross a limit are refused.
    The floor wins: above the ceiling by 'h' but close to something below by the ToF (an awning) -> it holds
  * battery below min_battery_pct, flight time above max_flight_s, board temperature at max_temp_c, or the
    dead-reckoned distance from the takeoff point above geofence_m -> land
  * a land that was asked for (any reason) but failed / was never sent while it still flies is re-sent every
    land_retry_s
  * no new video frame for frame_timeout_s, or stale telemetry -> hover (rc 0) until they come back; still stale
    after video_land_s / telemetry_land_s -> land (the rc 0 keep-alive stops the Tello's own 15 s auto-land)
  * while hovering for safety or landing, rc and discrete moves from the mission are dropped / refused; a takeoff
    after a land in the same run is refused
The mission code keeps working through the wrapper; refused commands report 'error: safety ...'.
Time: check() judges staleness, flight time and odometry on the `now` it is given (the caller's clock, like
Frame.t and Telemetry.t); a NaN stamp, or one more than the timeout in the future (the wall clock stepped back),
counts as stale. The land re-send interval is timed on `clock` (time.monotonic by default), so a wall-clock step
cannot hold it back.
"""

from __future__ import annotations

import math
import time
from typing import Callable

from ..config import SafetyCfg
from .base import Drone, check_move, clamp_rc

SOFT_BAND_M = 0.4  # vertical speed is tapered to zero over this distance before a limit
CEILING_PUSH = 20  # ud stick (down) above the ceiling, never more than max_rc
MAX_PLAUSIBLE_M_S = 10.0  # a state-stream speed above this is garbage, not motion (geofence odometry)


def _finite_or_nan(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _finite(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


class SafetyGovernor(Drone):
    name = "safety"

    def __init__(self, inner: Drone, cfg: SafetyCfg, clock: Callable[[], float] = time.monotonic):
        self.inner = inner
        self.cfg = cfg
        self._clock = clock  # land re-send interval only; check()'s `now` is the caller's clock
        self.events: list[tuple[float, str]] = []
        self._takeoff_t: float | None = None
        self._refused: str | None = None
        self.hovering_for_safety = False
        self.land_requested = False
        self._land_sent = -1e9
        self.land_reason: str | None = None
        self._stale_since: dict[str, float] = {}
        self.odometry_m = (0.0, 0.0)  # dead-reckoned displacement since takeoff (state-stream axes)
        self._odo_t: float | None = None

    def _event(self, msg: str) -> None:
        if not self.events or self.events[-1][1] != msg:
            self.events.append((time.time(), msg))

    @property
    def distance_m(self) -> float:
        """Dead-reckoned horizontal distance from the takeoff point."""
        return math.hypot(*self.odometry_m)

    # ------------------------------------------------------------------ periodic check
    def check(self, now: float, last_frame_t: float | None) -> str:
        """Call every loop. Returns 'land' (and starts / re-sends landing), 'hover' (and holds rc 0) or 'ok'."""
        c = self.cfg
        tel = self.inner.telemetry()
        if not self.inner.flying:
            self._takeoff_t = self._odo_t = None
            self._stale_since.clear()
            self.hovering_for_safety = False
            return "ok"
        if self._takeoff_t is None:
            self._takeoff_t = now
            self.odometry_m = (0.0, 0.0)
        tel_stale = not abs(now - _finite_or_nan(tel.t)) <= c.telemetry_timeout_s  # NaN / future stamp: stale
        self._odometry(tel, now, tel_stale)
        temp = self._temperature()
        if tel.battery_pct is not None and tel.battery_pct < c.min_battery_pct:  # even a stale low figure lands
            return self._land(f"battery {tel.battery_pct:.0f}% < {c.min_battery_pct:.0f}%", now)
        if now - self._takeoff_t > c.max_flight_s:
            return self._land(f"flight time > {c.max_flight_s:.0f} s", now)
        if temp is not None and temp >= c.max_temp_c:
            return self._land(f"temperature {temp:.0f} C >= {c.max_temp_c:.0f} C", now)
        if c.geofence_m is not None and self.distance_m > c.geofence_m:
            return self._land(f"{self.distance_m:.0f} m from takeoff > geofence {c.geofence_m:.0f} m", now)
        if self.land_requested:  # e.g. the 'l' key during a Wi-Fi drop: keep asking until it is down
            return self._land(self.land_reason or "requested", now)
        frame_stale = last_frame_t is None or not abs(now - _finite_or_nan(last_frame_t)) <= c.frame_timeout_s
        stale = {"video": frame_stale, "telemetry": tel_stale}
        for k, s in stale.items():
            if not s:
                self._stale_since.pop(k, None)
            else:
                self._stale_since.setdefault(k, now)
        if frame_stale or tel_stale:
            kind = "video" if frame_stale else "telemetry"
            for k, limit in (("telemetry", c.telemetry_land_s), ("video", c.video_land_s)):
                if k in self._stale_since and now - self._stale_since[k] >= limit:
                    return self._land(f"{k} stale for {now - self._stale_since[k]:.0f} s", now)
            self._event(f"{kind} stale -> hover")
            self.hovering_for_safety = True
            if not self.inner.busy():
                self.inner.rc(0, 0, 0, 0)
            return "hover"
        self.hovering_for_safety = False
        return "ok"

    def _odometry(self, tel, now: float, tel_stale: bool) -> None:
        dt = 0.0 if self._odo_t is None else max(0.0, min(0.5, now - self._odo_t))
        self._odo_t = now
        if tel_stale or tel.vx is None or tel.vy is None:
            return
        db = self.cfg.geofence_db_m_s
        vx, vy = (_finite(v) for v in (tel.vx, tel.vy))
        # hover noise is ignored, and so is garbage: the Tello tops out at 8 m/s
        vx, vy = (0.0 if abs(v) <= db or abs(v) > MAX_PLAUSIBLE_M_S else v for v in (vx, vy))
        self.odometry_m = (self.odometry_m[0] + vx * dt, self.odometry_m[1] + vy * dt)

    def _temperature(self) -> float | None:
        """Board temperature (C) if the drone reports one (TelloDrone.temperature_c: the state stream's temph)."""
        d = self.inner
        while d is not None:
            fn = getattr(d, "temperature_c", None)
            if callable(fn):
                try:
                    v = fn()
                except Exception:
                    return None
                return None if v is None or not math.isfinite(v) else float(v)
            d = getattr(d, "inner", None)
        return None

    def _land(self, why: str, now: float | None = None) -> str:
        now = self._clock()  # the re-send interval runs on the governor's own clock
        self._event(f"LAND: {why}")
        # (re)send land if we have not asked yet, or a previous land failed / was lost and it still flies
        if not self.land_requested or (not self.inner.busy() and self.inner.flying
                                       and now - self._land_sent > self.cfg.land_retry_s):
            self.land_requested = True
            self.land_reason = self.land_reason or why
            self._land_sent = now
            self.inner.land()
        return "land"

    # ------------------------------------------------------------------ Drone interface (filtered)
    @property
    def flying(self) -> bool:
        return self.inner.flying

    def connect(self) -> None:
        self.inner.connect()

    def close(self) -> None:
        self.inner.close()

    def takeoff(self) -> None:
        if self.land_requested:
            self._refuse("it already landed in this run")
            return
        self._refused = None
        self.inner.takeoff()

    def land(self) -> None:
        self.land_requested = True
        self.land_reason = self.land_reason or "requested"
        self._land_sent = self._clock()
        self.inner.land()

    def emergency(self) -> bool | None:
        self._event("EMERGENCY")
        return self.inner.emergency()

    def stop(self) -> None:
        self.inner.stop()

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        if self.land_requested or self.hovering_for_safety:
            return
        m = self.cfg.max_rc
        lr, fb, ud, yaw = (clamp_rc(_finite(v), m) for v in (lr, fb, ud, yaw))
        tel = self.inner.telemetry()
        high, clear = tel.floor_altitude_m(), tel.altitude_m()  # above the floor / above what is below
        high = None if high is None or not math.isfinite(high) else high
        clear = None if clear is None or not math.isfinite(clear) else clear
        if self.inner.flying:
            hi, lo, band = self.cfg.max_altitude_m, self.cfg.min_altitude_m, SOFT_BAND_M
            if high is not None:
                if high > hi:
                    ud = min(ud, -min(CEILING_PUSH, m))
                    self._event(f"ceiling {high:.2f} m > {hi} m -> descend")
                elif ud > 0 and high > hi - band:  # taper the climb: the drone keeps rising ~0.2 m after rc 0
                    ud = int(ud * max(0.0, (hi - high) / band))
            if clear is not None:
                if clear < lo:
                    ud = max(ud, 0)
                    self._event(f"floor {clear:.2f} m < {lo} m -> no descent")
                elif ud < 0 and clear < lo + band:
                    ud = int(ud * max(0.0, (clear - lo) / band))
        self.inner.rc(lr, fb, ud, yaw)

    def move(self, direction: str, cm: int) -> None:
        cm = check_move(direction, cm)
        if self.land_requested:
            self._refuse("landing")
            return
        if self.hovering_for_safety:
            self._refuse("safety hover (stale video / telemetry)")
            return
        tel = self.inner.telemetry()
        high, clear = tel.floor_altitude_m(), tel.altitude_m()
        if direction == "up" and high is not None and high + cm / 100.0 > self.cfg.max_altitude_m:
            self._refuse(f"up {cm} would reach {high + cm / 100.0:.2f} m")
            return
        if direction == "down" and clear is not None and clear - cm / 100.0 < self.cfg.min_altitude_m:
            self._refuse(f"down {cm} would leave {clear - cm / 100.0:.2f} m below")
            return
        self._refused = None
        self.inner.move(direction, cm)

    def rotate(self, deg: int) -> None:
        if self.land_requested:
            self._refuse("landing")
            return
        if self.hovering_for_safety:
            self._refuse("safety hover (stale video / telemetry)")
            return
        self._refused = None
        self.inner.rotate(deg)

    def _refuse(self, why: str) -> None:
        self._refused = f"error: safety ({why})"
        self._event(f"refused: {why}")

    def busy(self) -> bool:
        return self.inner.busy()

    def last_result(self) -> str | None:
        return self._refused if self._refused else self.inner.last_result()

    def set_speed(self, cm_s: int) -> None:
        self.inner.set_speed(cm_s)

    def telemetry(self):
        return self.inner.telemetry()

    def frame_source(self):
        return self.inner.frame_source()
