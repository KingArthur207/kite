"""SimTello: kite.drone.base.Drone on kite's Tello response model, driven by a simulated clock.

The world calls step(dt) to advance the physics; everything else is the Drone interface the real TelloDrone has:
rc() streams sticks (ignored while a discrete command runs, as TelloDrone does), takeoff / land / move / rotate are
non-blocking and polled with busy() / last_result(), stop() cancels a move, telemetry() reads like the Tello's state
stream (whole degrees, height in cm, speeds in 0.1 m/s steps, forward = negative vx) stamped with the sim clock.

Discrete commands fly on internal sticks: takeoff and land at full climb / sink, moves and rotations on a P law
toward the target, done when within 5 cm / 3 deg and settled.
"""

from __future__ import annotations

import math

from kite.drone.base import Drone, check_move, check_rotate, clamp_rc
from kite.fly.sim.drone import PHYS_DT, DroneModel, DroneParams
from kite.types import Telemetry, wrap_deg

TAKEOFF_H_M = 1.0  # where the Tello's own takeoff stops climbing
DRAIN_PCT_S = 100.0 / 600.0  # battery: about 10 minutes from full


class SimTello(Drone):
    name = "sim-tello"

    def __init__(self, clock, frame_source, x: float = 0.0, y: float = 0.0, heading_deg: float = 0.0,
                 params: DroneParams | None = None, move_speed_cm_s: int = 50):
        self._clock = clock  # () -> sim time (s)
        self._source = frame_source
        self.model = DroneModel(params or DroneParams())
        self.model.reset(x, y, 0.0, math.radians(heading_deg))
        self.move_speed_cm_s = move_speed_cm_s
        self.battery = 100.0
        self._flying = False
        self._sticks = (0, 0, 0, 0)
        self._pending: dict | None = None
        self._result: str | None = "ok"
        self.log: list[tuple[float, str]] = []

    # ------------------------------------------------------------------ physics
    def step(self, dt: float) -> None:
        for _ in range(max(1, round(dt / PHYS_DT))):
            sticks = self._discrete_sticks() if self._pending else (self._sticks if self._flying else (0, 0, 0, 0))
            self.model.step(*sticks)
            if not self._flying and not self._pending:
                self.model.z = self.model.vz = 0.0
            if self._flying:
                self.battery = max(0.0, self.battery - DRAIN_PCT_S * PHYS_DT)

    def _discrete_sticks(self) -> tuple[int, int, int, int]:
        p, m = self._pending, self.model
        kind = p["kind"]
        if kind == "takeoff":
            if m.z >= TAKEOFF_H_M:
                self._done("ok")
                self._flying = True
                return (0, 0, 0, 0)
            return (0, 0, 100, 0)
        if kind == "land":
            if m.z <= 0.0:
                self._done("ok")
                self._flying = False
                return (0, 0, 0, 0)
            return (0, 0, -100, 0)
        if self._clock() - p["t0"] > p["timeout"]:
            self._done("error: timeout")
            return (0, 0, 0, 0)
        speed = self.move_speed_cm_s / 100.0
        if kind == "rotate":
            err = wrap_deg(p["target"] - math.degrees(m.heading))
            if abs(err) < 3.0 and abs(m.yaw_rate_dps) < 5.0:
                self._done("ok")
                return (0, 0, 0, 0)
            return (0, 0, 0, clamp_rc(err * 2.0))
        # move: target in the world frame, sticks in the body frame
        dx, dy, dz = p["target"][0] - m.x, p["target"][1] - m.y, p["target"][2] - m.z
        s, c = math.sin(m.heading), math.cos(m.heading)
        fwd, right = dx * s + dy * c, dx * c - dy * s
        v = math.hypot(m.vx, m.vy)
        if math.hypot(math.hypot(fwd, right), dz) < 0.05 and v < 0.05 and abs(m.vz) < 0.05:
            self._done("ok")
            return (0, 0, 0, 0)
        lim = 100.0 * speed / self.model.p.fwd_gain_mps

        def law(e, k=80.0, cap=lim):  # gentle: the stick reaches the speed ~0.9 s late (dead time + lag)
            return clamp_rc(max(-cap, min(cap, e * k)))

        # vertical: 3x the gain (0.32 m/s at full stick against 0.96 horizontally), at any speed setting
        return (law(right), law(fwd), law(dz, k=240.0, cap=100.0), 0)

    def _done(self, result: str) -> None:
        self.log.append((self._clock(), f"{self._pending['cmd']} -> {result}"))
        self._pending = None
        self._result = result

    def _start(self, kind: str, cmd: str, **kw) -> None:
        if self._pending is not None:
            raise RuntimeError(f"'{cmd}' while '{self._pending['cmd']}' is still running")
        self.log.append((self._clock(), cmd))
        self._result = None
        self._pending = {"kind": kind, "cmd": cmd, "t0": self._clock(), **kw}

    # ------------------------------------------------------------------ Drone interface
    @property
    def flying(self) -> bool:
        return self._flying

    def takeoff(self) -> None:
        self._start("takeoff", "takeoff")

    def land(self) -> None:
        if self._pending is not None and self._pending["kind"] == "land":
            return
        self._pending = None
        self._start("land", "land")

    def emergency(self) -> bool:
        self.log.append((self._clock(), "emergency"))
        self._pending, self._result, self._flying = None, "ok", False
        self.model.z = self.model.vz = 0.0
        return True

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        if self._pending is not None:
            return  # a stick would abort the running discrete command
        self._sticks = tuple(clamp_rc(v) for v in (lr, fb, ud, yaw))

    def move(self, direction: str, cm: int) -> None:
        cm = check_move(direction, cm)
        m, d = self.model, cm / 100.0
        s, c = math.sin(m.heading), math.cos(m.heading)
        f, r, u = {"forward": (d, 0, 0), "back": (-d, 0, 0), "right": (0, d, 0), "left": (0, -d, 0),
                   "up": (0, 0, d), "down": (0, 0, -d)}[direction]
        target = (m.x + f * s + r * c, m.y + f * c - r * s, m.z + u)
        speed = self.model.p.vz_gain_mps if u else self.move_speed_cm_s / 100.0
        self._start("move", f"{direction} {cm}", target=target, timeout=d / speed + 7.5)

    def rotate(self, deg: int) -> None:
        deg = check_rotate(deg)
        target = math.degrees(self.model.heading) + deg
        self._start("rotate", f"{'cw' if deg > 0 else 'ccw'} {abs(deg)}", target=target,
                    timeout=abs(deg) / 60.0 + 6.0)

    def stop(self) -> None:
        if self._pending is not None and self._pending["kind"] in ("takeoff", "land"):
            return
        self._pending, self._result = None, "ok"
        self._sticks = (0, 0, 0, 0)

    def busy(self) -> bool:
        return self._pending is not None

    def last_result(self) -> str | None:
        return self._result

    def set_speed(self, cm_s: int) -> None:
        self.move_speed_cm_s = int(cm_s)

    def telemetry(self) -> Telemetry:
        m = self.model
        s, c = math.sin(m.heading), math.cos(m.heading)
        v_fwd, v_right = m.vx * s + m.vy * c, m.vx * c - m.vy * s
        h = round(m.z * 100.0) / 100.0
        return Telemetry(
            t=self._clock(), yaw_deg=float(round(wrap_deg(math.degrees(m.heading)))),
            pitch_deg=float(round(m.pitch_deg)), roll_deg=0.0,
            height_m=h, tof_m=h if 0.1 <= h <= 8.0 else None,
            vx=-round(v_fwd * 10.0) / 10.0, vy=round(v_right * 10.0) / 10.0, vz=round(m.vz * 10.0) / 10.0,
            battery_pct=float(int(self.battery)), flight_time_s=None)

    def frame_source(self):
        return self._source
