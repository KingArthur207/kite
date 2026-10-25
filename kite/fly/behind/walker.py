"""The person walking ahead of the drone (DESIGN section 3): a unicycle over random segments, planned at reset.

The plan (speed, facing change and event code per 0.01 s tick) is drawn from the walker's own stream when the episode
starts and never looks at the drone, so it is the same for every arm (common random numbers). Two segments read the
drone, and only for a direction, the moment they begin:
  step back    one 0.3-0.8 m step toward the drone (per episode, p 0.15): brake to a stand, pause, step, pause.
               It stops early if the person gets within step_stop_m of the drone. The person keeps facing away
  turn around  turn 150-180 deg (slowing to a pivot), then walk toward a point beside the drone (0.7-1.3 m to one
               side) and on past it for 2-5 s: the drone has to get out of the way and turn round
Everything else is planned: straight (3-12 s, gentle curvature, a new speed with p 0.6), corner (60-100 deg at
45-90 deg/s, walked through, then a straight), stop (1-8 s, a crossing), speed change (ramps to a new speed of the
profile's range over 1-3 s), and in `demo` one faster stretch. Acceleration 0.8 m/s^2. There is no speed cap from the
drone's limits: a person who outwalks the drone leaves it behind, as live.

Each segment takes a fixed block of uniforms whatever its kind, so changing a profile's speeds (or forcing one in a
test) never shifts the segment sequence. Positions are the head top (x east, y north, m); headings in radians
clockwise from north (kite.types); index 0 is the follow start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PHYS_DT = 0.01
# event codes per tick (the metrics read them): what the person is doing
EV_WALK, EV_STOP, EV_SPEED, EV_TURN_AROUND, EV_STEP_BACK, EV_CORNER = range(6)
# motion modes: along the facing, toward the drone (step back), steered to the pass point (after a turn around)
_ALONG, _STEP, _BACK = 0, 1, 2
_SEG_U = 8  # uniforms per segment
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class WalkCfg:
    p_straight: float = 0.45  # segment kinds; normalized by their sum (the step back is per episode, not a kind)
    p_corner: float = 0.20
    p_stop: float = 0.17
    p_speed: float = 0.10
    p_turn_around: float = 0.03
    straight_s: tuple[float, float] = (3.0, 12.0)
    curve_deg: tuple[float, float] = (10.0, 25.0)  # total facing change over a straight, either side
    p_new_speed: float = 0.6
    corner_deg: tuple[float, float] = (60.0, 100.0)
    corner_rate_dps: tuple[float, float] = (45.0, 90.0)
    stop_s: tuple[float, float] = (1.0, 8.0)
    ramp_s: tuple[float, float] = (1.0, 3.0)  # speed change: ramp time ...
    speed_hold_s: tuple[float, float] = (2.0, 6.0)  # ... then walk straight at the new speed this long
    turn_around_deg: tuple[float, float] = (150.0, 180.0)
    turn_around_rate_dps: tuple[float, float] = (90.0, 180.0)
    turn_around_v_mps: float = 0.3  # pivoting: slower than this while turning round
    walk_back_s: tuple[float, float] = (2.0, 5.0)
    pass_offset_m: tuple[float, float] = (0.7, 1.3)  # beside the drone, on a side drawn at reset
    walk_back_turn_dps: float = 90.0  # facing steered toward the pass point at most this fast
    step_m: tuple[float, float] = (0.3, 0.8)
    step_speed_mps: tuple[float, float] = (0.4, 0.8)
    step_stop_m: float = 0.8  # never steps closer than this to the drone
    step_window: tuple[float, float] = (0.15, 0.85)  # fraction of the episode where the step may start
    step_pause_s: float = 0.3  # standing still before and after it
    accel_mps2: float = 0.8


def _lerp(lo_hi: tuple[float, float], u: float) -> float:
    return lo_hi[0] + u * (lo_hi[1] - lo_hi[0])


def wrap_rad(a: float) -> float:
    """Wrap to (-pi, pi]."""
    a = (a + math.pi) % _TWO_PI - math.pi
    return math.pi if a == -math.pi else a


class Person:
    """The walk of one episode. step(drone_x, drone_y) advances one tick; x, y, facing, speed and event are the state
    after it. `segments` lists (kind, first tick, end tick) of the plan."""

    def __init__(self, rng: np.random.Generator, x0: float, y0: float, facing0: float, n_ticks: int,
                 speed_lo: float, speed_hi: float, stand_s: float, p_step_back: float = 0.15,
                 stretch: tuple[float, float, float] | None = None, cfg: WalkCfg = WalkCfg(),
                 dt: float = PHYS_DT):
        """stretch: (speed m/s, start as a fraction of the episode, length s) of one faster stretch, or None."""
        self.cfg, self.dt = cfg, dt
        self.x, self.y, self.facing = float(x0), float(y0), float(facing0)
        self.speed, self.event, self.i = 0.0, EV_STOP, 0
        # per-episode draws first, always taken, so they never shift the segments
        g = rng.random(_SEG_U).tolist()
        wants_step = g[0] < p_step_back
        step_from = int(_lerp(cfg.step_window, g[1]) * n_ticks)
        step_len, step_speed = _lerp(cfg.step_m, g[2]), _lerp(cfg.step_speed_mps, g[3])
        stretch_ramp = _lerp(cfg.ramp_s, g[4])
        n = n_ticks + 1
        self.n = n
        vs, turn, ev, mode = [0.0] * n, [0.0] * n, [EV_STOP] * n, [_ALONG] * n
        self.vs, self.turn, self.ev, self.mode = vs, turn, ev, mode
        self.segments: list[tuple[str, int, int]] = []
        self._back: dict[int, tuple[float, float, float]] = {}  # walk-back first tick -> (side, offset, end)
        self._step: tuple[int, int] | None = None  # step-back ticks [first, last)
        acc = cfg.accel_mps2 * dt
        if stretch is not None and stretch[0] > 0.0:
            sa = int(stretch[1] * n_ticks)
            sw = (sa, sa + int(round(stretch[2] / dt)), float(stretch[0]), max(1, int(round(stretch_ramp / dt))))
        else:
            sw = None
        lo, hi = float(speed_lo), float(speed_hi)
        p = (cfg.p_straight, cfg.p_corner, cfg.p_stop, cfg.p_speed, cfg.p_turn_around)
        tot = sum(p)
        c1, c2, c3, c4 = (sum(p[:k]) / tot for k in (1, 2, 3, 4))

        state = {"v": 0.0, "goal": 0.0}

        def fill(first: int, end: int, goal: float, dv: float, d_face: float, code: int, m: int = _ALONG) -> None:
            """Ticks [first, end): speed toward `goal` at most dv per tick (the demo stretch replaces the goal of
            walking ticks inside its window and ramps in and out), facing change d_face per tick. A walking tick
            whose speed changes is a speed change (EV_SPEED)."""
            v = state["v"]
            walking = sw is not None and goal > 0.0 and (code in (EV_WALK, EV_CORNER, EV_SPEED) or m == _BACK)
            for k in range(first, end):
                g_k, dv_k = goal, dv
                if walking and sw[0] <= k < sw[1] + sw[3]:
                    dv_k = max(abs(sw[2] - goal), 1e-3) / sw[3]
                    if k < sw[1]:
                        g_k = sw[2]
                v_prev = v
                v = min(g_k, v + dv_k) if v < g_k else max(g_k, v - dv_k)
                vs[k], turn[k], mode[k] = v, d_face, m
                ev[k] = EV_SPEED if (code == EV_WALK and abs(v - v_prev) > 1e-12) else code
            state["v"] = v

        stand = min(n, max(1, int(round(stand_s / dt))))
        self.segments.append(("stand", 0, stand))
        fill(1, stand, 0.0, acc, 0.0, EV_STOP)
        i, prev = stand, "stand"
        while i < n:
            u = rng.random(_SEG_U).tolist()
            if wants_step and self._step is None and i >= step_from:
                kind = "step back"
                brake = int(math.ceil(state["v"] / acc)) if state["v"] > 0 else 0
                pause = int(round(cfg.step_pause_s / dt))
                n_move = max(1, int(round(step_len / step_speed / dt)))
                first = min(n, i + brake + pause)
                last = min(n, first + n_move)
                end = min(n, last + pause)
                fill(i, first, 0.0, acc, 0.0, EV_STEP_BACK)
                for k in range(first, last):
                    vs[k], mode[k], ev[k] = step_speed, _STEP, EV_STEP_BACK
                for k in range(last, end):
                    vs[k], ev[k] = 0.0, EV_STEP_BACK
                state["v"], state["goal"] = 0.0, 0.0
                self._step = (first, last)
            elif prev in ("corner", "walk back") or u[0] < c1:
                kind = "straight"
                ticks = max(1, int(round(_lerp(cfg.straight_s, u[1]) / dt)))
                bend = math.radians(_lerp(cfg.curve_deg, u[2])) * (1.0 if u[3] < 0.5 else -1.0)
                if u[4] < cfg.p_new_speed or state["goal"] <= 0.0:
                    state["goal"] = lo + u[5] * (hi - lo)
                end = min(n, i + ticks)
                fill(i, end, state["goal"], acc, bend / ticks, EV_WALK)
            elif u[0] < c2:
                kind = "corner"
                angle = math.radians(_lerp(cfg.corner_deg, u[2]))
                rate = math.radians(_lerp(cfg.corner_rate_dps, u[6]))
                ticks = max(1, int(round(angle / rate / dt)))
                if state["goal"] <= 0.0:  # a corner is walked through: set off if standing
                    state["goal"] = lo + u[5] * (hi - lo)
                end = min(n, i + ticks)
                fill(i, end, state["goal"], acc, (1.0 if u[3] < 0.5 else -1.0) * angle / ticks, EV_CORNER)
            elif u[0] < c3:
                kind = "stop"
                end = min(n, i + max(1, int(round(_lerp(cfg.stop_s, u[1]) / dt))))
                state["goal"] = 0.0
                fill(i, end, 0.0, acc, 0.0, EV_STOP)
            elif u[0] < c4:
                kind = "speed change"
                ramp = max(1, int(round(_lerp(cfg.ramp_s, u[1]) / dt)))
                hold = max(1, int(round(_lerp(cfg.speed_hold_s, u[2]) / dt)))
                new = lo + u[5] * (hi - lo)
                dv = min(acc, max(abs(new - state["v"]), 1e-3) / ramp)
                state["goal"] = new
                mid, end = min(n, i + ramp), min(n, i + ramp + hold)
                fill(i, mid, new, dv, 0.0, EV_SPEED)
                fill(mid, end, new, dv, 0.0, EV_WALK)
            else:
                kind = "turn around"
                angle = math.radians(_lerp(cfg.turn_around_deg, u[2]))
                rate = math.radians(_lerp(cfg.turn_around_rate_dps, u[6]))
                ticks = max(1, int(round(angle / rate / dt)))
                back = max(1, int(round(_lerp(cfg.walk_back_s, u[1]) / dt)))
                v_back = state["goal"] if state["goal"] > 0.0 else lo + u[5] * (hi - lo)
                state["goal"] = v_back
                mid, end = min(n, i + ticks), min(n, i + ticks + back)
                fill(i, mid, min(v_back, cfg.turn_around_v_mps), acc, (1.0 if u[3] < 0.5 else -1.0) * angle / ticks,
                     EV_TURN_AROUND)
                fill(mid, end, v_back, acc, 0.0, EV_TURN_AROUND, _BACK)
                if mid < end:
                    self._back[mid] = ((1.0 if u[4] < 0.5 else -1.0), _lerp(cfg.pass_offset_m, u[7]), float(end))
                self.segments.append((kind, i, mid))
                kind, i = "walk back", mid
            self.segments.append((kind, i, end))
            prev, i = kind, end
        self._step_dir: tuple[float, float] | None = None
        self._step_done = False
        self._back_heading: float | None = None
        self._max_turn = math.radians(cfg.walk_back_turn_dps) * dt

    @property
    def drone_dependent(self) -> bool:
        """The walk reads the drone somewhere in this episode (a step back or a turn around)."""
        return self._step is not None or bool(self._back)

    def step(self, drone_x: float, drone_y: float) -> None:
        """Advance one tick (call once per physics tick, in order)."""
        i = self.i + 1
        if i >= self.n:  # past the plan: stand
            self.i, self.speed, self.event = i, 0.0, EV_STOP
            return
        self.i = i
        v, m = self.vs[i], self.mode[i]
        self.event = self.ev[i]
        if m == _ALONG:
            self.facing = wrap_rad(self.facing + self.turn[i])
            h = self.facing
        elif m == _STEP:  # toward the drone, facing unchanged; the direction is fixed when the step begins
            to_x, to_y = drone_x - self.x, drone_y - self.y
            gap = math.hypot(to_x, to_y)
            if self._step_dir is None:
                self._step_dir = (to_x / gap, to_y / gap) if gap > 1e-6 else (0.0, 0.0)
            if self._step_done or gap <= self.cfg.step_stop_m:
                self._step_done = True
                v = 0.0
            self.speed = v
            self.x += self._step_dir[0] * v * self.dt
            self.y += self._step_dir[1] * v * self.dt
            return
        else:  # walking back past the drone: steer to the pass point fixed at the walk-back's first tick
            if i in self._back:
                side, offset, _ = self._back[i]
                to_x, to_y = drone_x - self.x, drone_y - self.y
                gap = math.hypot(to_x, to_y)
                if gap > 1e-6:
                    # the point `offset` beside the drone, perpendicular to the line person -> drone
                    tx = drone_x + side * offset * to_y / gap
                    ty = drone_y - side * offset * to_x / gap
                    self._back_heading = math.atan2(tx - self.x, ty - self.y)
                else:
                    self._back_heading = self.facing
            if self._back_heading is not None:
                d = wrap_rad(self._back_heading - self.facing)
                mt = self._max_turn
                self.facing = wrap_rad(self.facing + (mt if d > mt else (-mt if d < -mt else d)))
            h = self.facing
        self.speed = v
        self.x += v * math.sin(h) * self.dt
        self.y += v * math.cos(h) * self.dt


class Crossing:
    """A second person walking straight across the drone's view (DESIGN: the lock's distractor). The line is placed
    when the crossing begins, from where the drone is and looks then (direction only): dist_m ahead of the drone,
    entering from `side` (+1 = the drone's right) 3.5 m out and walking across at speed_mps, angle_deg off
    perpendicular, for 7 m. Before and after, nobody is there (x, y = nan)."""

    HALF_M = 3.5

    def __init__(self, t0: float, dist_m: float, speed_mps: float, side: float, angle_deg: float, dt: float = PHYS_DT):
        self.t0, self.dist, self.v, self.side, self.angle = t0, dist_m, speed_mps, side, math.radians(angle_deg)
        self.dt = dt
        self.t1 = t0 + 2.0 * self.HALF_M / max(speed_mps, 1e-3)
        self.x = self.y = math.nan
        self.facing = 0.0
        self._started = False

    @property
    def active(self) -> bool:
        return not math.isnan(self.x)

    def step(self, t: float, drone_x: float, drone_y: float, drone_heading: float) -> None:
        if t < self.t0 or t >= self.t1:
            if self._started:
                self.x = self.y = math.nan
            return
        if not self._started:
            self._started = True
            s, c = math.sin(drone_heading), math.cos(drone_heading)
            cx, cy = drone_x + self.dist * s, drone_y + self.dist * c  # the crossing point ahead of the drone
            rx, ry = c, -s  # the drone's right
            self.x, self.y = cx + self.side * self.HALF_M * rx, cy + self.side * self.HALF_M * ry
            # walking across toward the other side, angle_deg off perpendicular
            self.facing = wrap_rad(math.atan2(-self.side * rx, -self.side * ry) + self.angle)
            return
        self.x += self.v * math.sin(self.facing) * self.dt
        self.y += self.v * math.cos(self.facing) * self.dt
