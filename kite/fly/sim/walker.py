"""Street walker (DESIGN 2.2): a unicycle over random segments, pre-generated at reset.

The base path (head-top position per physics sub-step) is drawn from the walker's own stream when the episode
starts, so it is the same for every yaw arm. The one part that reads the drone is the optional step back toward it
(StepBack): when, how far and how fast are drawn at reset; only its direction is taken from where the drone is the
moment it begins, and it stops early once the walker is step_stop_m from the drone. It is an offset on the base path.

Segments: straight (gentle curvature, maybe a new speed), corner turn (walking through it, then a straight), stop
(a crossing). Speeds are capped at v_cap = profiles.walker_speed_cap so every episode can physically be followed; the
walker accelerates and brakes at accel_mps2. Index 0 is the follow start; the walker stands still before it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .drone import PHYS_DT


@dataclass(frozen=True)
class WalkerCfg:
    p_straight: float = 0.55
    p_turn: float = 0.25  # stop: the rest (0.20)
    straight_s: tuple[float, float] = (3.0, 15.0)
    curve_deg: tuple[float, float] = (10.0, 30.0)  # total heading change over a straight segment, either side
    p_new_speed: float = 0.6
    turn_deg: tuple[float, float] = (60.0, 100.0)
    turn_rate_dps: tuple[float, float] = (45.0, 120.0)
    stop_s: tuple[float, float] = (1.0, 10.0)
    # a corner is followed by a straight: back-to-back 60-100 deg turns at 45-120 deg/s walk tight circles
    # (radius v / rate, under 1 m), which no street walk does
    turn_then_straight: bool = True
    accel_mps2: float = 1.0
    step_prob: float = 0.2  # one step back toward the drone per episode
    step_m: tuple[float, float] = (0.3, 0.8)
    step_speed_mps: tuple[float, float] = (0.4, 0.8)
    step_stop_m: float = 0.8  # the walker never steps closer than this to the drone
    step_window: tuple[float, float] = (0.15, 0.85)  # where in the episode the step may start
    step_pause_s: float = 0.3  # standing still before and after it


@dataclass
class StepBack:
    """The walker's one step toward the drone, ticks [first, last). Direction fixed when it begins."""

    first: int
    last: int
    speed_mps: float
    stop_m: float
    dt: float
    heading: tuple[float, float] | None = None  # unit vector toward the drone, set at `first`
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def apply(self, i: int, base_x: float, base_y: float, drone_x: float, drone_y: float) -> tuple[float, float]:
        ox, oy = self.offset
        if self.first <= i < self.last:
            to_x, to_y = drone_x - (base_x + ox), drone_y - (base_y + oy)
            gap = math.hypot(to_x, to_y)
            if self.heading is None:
                self.heading = (to_x / gap, to_y / gap) if gap > 1e-6 else (0.0, 0.0)
            if gap > self.stop_m:
                stride = self.speed_mps * self.dt
                self.offset = [ox + self.heading[0] * stride, oy + self.heading[1] * stride]
        return base_x + self.offset[0], base_y + self.offset[1]


def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


class Walker:
    """Head-top (x, y) per physics sub-step. Heading in radians clockwise from north (kite.types)."""

    def __init__(self, rng: np.random.Generator, x0: float, y0: float, heading: float, v_cap: float,
                 speed_range: tuple[float, float], n_ticks: int, cfg: WalkerCfg = WalkerCfg(), dt: float = PHYS_DT):
        self.cfg, self.dt, self.v_cap = cfg, dt, v_cap
        # the step's draws come first and always happen, so they never shift the segment draws
        wants_step = bool(rng.random() < cfg.step_prob)
        u_when, u_len, u_speed = rng.random(3).tolist()
        lo, hi = cfg.step_window
        step_from = int((lo + u_when * (hi - lo)) * n_ticks)
        step_len = cfg.step_m[0] + u_len * (cfg.step_m[1] - cfg.step_m[0])
        step_speed = cfg.step_speed_mps[0] + u_speed * (cfg.step_speed_mps[1] - cfg.step_speed_mps[0])
        self.step: StepBack | None = None

        n = n_ticks + 1
        self.xs, self.ys, self.hs, self.vs = [x0] * n, [y0] * n, [heading] * n, [0.0] * n
        self.segments: list[tuple[str, int, int]] = []  # (kind, first tick, end tick)
        dv_max = cfg.accel_mps2 * dt
        pause = int(round(cfg.step_pause_s / dt))
        state = {"x": x0, "y": y0, "phi": heading, "v": 0.0}
        v_goal, kind, i = 0.0, "", 1
        while i < n:
            u = float(rng.random())  # always drawn, so forcing a straight never shifts the later draws
            if kind == "turn" and cfg.turn_then_straight:
                u = 0.0
            turn_per_tick = 0.0
            if wants_step and self.step is None and i >= step_from:
                kind = "step"
                begin = i + int(math.ceil(state["v"] / dv_max)) + pause  # brake to a stand, pause, step
                n_move = max(1, int(round(step_len / step_speed / dt)))
                self.step = StepBack(begin, begin + n_move, step_speed, cfg.step_stop_m, dt)
                ticks, v_goal = begin + n_move + pause - i, 0.0
            elif u < cfg.p_straight:
                kind = "straight"
                ticks = max(1, int(round(_u(rng, cfg.straight_s) / dt)))
                bend = math.radians(_u(rng, cfg.curve_deg)) * (1.0 if rng.random() < 0.5 else -1.0)
                turn_per_tick = bend / ticks
                if rng.random() < cfg.p_new_speed or v_goal <= 0.0:
                    v_goal = min(v_cap, _u(rng, speed_range))
            elif u < cfg.p_straight + cfg.p_turn:
                kind = "turn"
                angle = math.radians(_u(rng, cfg.turn_deg))
                rate = math.radians(_u(rng, cfg.turn_rate_dps))
                side = 1.0 if rng.random() < 0.5 else -1.0
                ticks = max(1, int(round(angle / rate / dt)))
                turn_per_tick = side * angle / ticks
                if v_goal <= 0.0:  # a corner is walked through: set off if standing
                    v_goal = min(v_cap, _u(rng, speed_range))
            else:
                kind = "stop"
                ticks = max(1, int(round(_u(rng, cfg.stop_s) / dt)))
                v_goal = 0.0
            end = min(n, i + ticks)
            self.segments.append((kind, i, end))
            self._walk(state, i, end, v_goal, turn_per_tick, dv_max)
            i = end

    def _walk(self, st: dict, first: int, end: int, v_goal: float, turn_per_tick: float, dv_max: float) -> None:
        x, y, phi, v, dt = st["x"], st["y"], st["phi"], st["v"], self.dt
        for k in range(first, end):
            v = min(v_goal, v + dv_max) if v < v_goal else max(v_goal, v - dv_max)
            phi += turn_per_tick
            x += v * math.sin(phi) * dt
            y += v * math.cos(phi) * dt
            self.xs[k], self.ys[k], self.hs[k], self.vs[k] = x, y, phi, v
        st.update(x=x, y=y, phi=phi, v=v)

    def position(self, i: int, drone_x: float, drone_y: float) -> tuple[float, float]:
        """Head-top (x, y) at tick i. Call once per tick, in order: the step back integrates here."""
        if self.step is None:
            return self.xs[i], self.ys[i]
        return self.step.apply(i, self.xs[i], self.ys[i], drone_x, drone_y)

    @property
    def step_ticks(self) -> tuple[int, int]:
        return (self.step.first, self.step.last) if self.step is not None else (-1, -1)
