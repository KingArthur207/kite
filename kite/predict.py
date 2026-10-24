"""Follow prediction: where the walker will be in the next 1.5 s, and the drone path that stays behind them.

Used by kite.follow.FollowController (FollowCfg.predict, `--follow-predict` / `--predict`). No drone, no camera here:
plain numpy on positions in the controller's dead-reckoned frame ("world": x, y metres, heading clockwise from +y,
like kite.types; the frame starts wherever follow was engaged and drifts slowly, which is fine because everything
here is relative over a couple of seconds).

  WalkerFilter   IMM of two linear Kalman filters over the walker's head position: CV (constant velocity, white
                 acceleration noise) and STOP (velocity forced to 0). STOP takes over within a few frames when the
                 walker stops at the curb; CV takes back over when they move off. A turn rate comes from the change of
                 the velocity's heading, so the predicted path bends (constant turn rate) at the Moss/Howard corner.
                 Optional route prior (GPS + route polyline, see route_turn_ahead): the predicted turn never goes past
                 the corner's angle.
  DronePlanner   sampling MPC over the same horizon: constant (fb, lr) candidates through the Tello's first-order
                 speed lag and slew limit, scored against the standoff point behind_m back along the walker's trail
                 (the drone flies where the walker already walked), with effort / change / too-close penalties. Yaw
                 looks from where the drone will be to where the walker will be (leads into turns), clamped so the
                 walker stays near the frame centre now.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROUTE_JSON = ROOT / "sim" / "public" / "osm" / "route.json"


def fwd_right(h_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Forward and right unit vectors of heading h (clockwise from +y)."""
    h = math.radians(h_deg)
    return np.array([math.sin(h), math.cos(h)]), np.array([math.cos(h), -math.sin(h)])


def compass_deg(v) -> float:
    return math.degrees(math.atan2(v[0], v[1]))


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------ route prior
def load_route(path: str | Path = ROUTE_JSON) -> np.ndarray:
    """The shoot route (sim/public/osm/route.json) as an (n, 2) polyline in sim metres (kite.gps.latlng_to_sim)."""
    from .gps import latlng_to_sim

    pts = json.loads(Path(path).read_text())["points"]
    return np.array([latlng_to_sim(p["lat"], p["lng"]) for p in pts])


def route_turn_ahead(xy, route: np.ndarray, within_m: float = 6.0, max_off_m: float = 8.0) -> float | None:
    """Signed turn (deg, + = right) at the next route corner less than within_m ahead of the walker's GPS position,
    0.0 if no corner is that close, None if the walker is more than max_off_m off the route (prior unusable)."""
    p = np.asarray(xy, float)
    a, b = route[:-1], route[1:]
    ab = b - a
    s = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    d = np.linalg.norm(p - (a + s[:, None] * ab), axis=1)
    i = int(d.argmin())
    if d[i] > max_off_m:
        return None
    ahead = (1 - s[i]) * np.linalg.norm(ab[i])
    for j in range(i + 1, len(route) - 1):
        if ahead > within_m:
            break
        turn = wrap(compass_deg(ab[j]) - compass_deg(ab[j - 1]))
        if abs(turn) > 20:
            return turn
        ahead += np.linalg.norm(ab[j])
    return 0.0


# ------------------------------------------------------------------ walker motion model
H_POS = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


def _kf(x, P, F, Q, z, R):
    """One Kalman predict + position update: (x, P, innovation, its covariance)."""
    x, P = F @ x, F @ P @ F.T + Q
    nu = z - H_POS @ x
    S = H_POS @ P @ H_POS.T + R
    K = P @ H_POS.T @ np.linalg.inv(S)
    return x + K @ nu, (np.eye(4) - K @ H_POS) @ P, nu, S


@dataclass
class Prediction:
    path: np.ndarray  # (n+1, 2) walker head position now, then every step_s
    std: np.ndarray  # (n+1,) 1-sigma position uncertainty (m)
    speed: float  # m/s
    heading_deg: float  # direction of travel (the last walking heading while stopped)
    turn_deg_s: float  # + = turning right
    p_stop: float  # IMM probability of the STOP model
    step_s: float
    n_updates: int


class WalkerFilter:
    """IMM (CV + STOP) over the walker's head position, measured at the frame's time."""

    def __init__(self, accel: float = 0.8, switch_s: tuple[float, float] = (4.0, 2.0), turn_window_s: float = 1.0,
                 turn_gate_deg_s: float = 10.0, max_turn_deg_s: float = 60.0, heading_accel: float = 0.25):
        self.accel = accel  # CV white acceleration noise (m/s^2)
        self.heading_accel = heading_accel
        self.switch_s = switch_s  # mean time in CV before a stop, in STOP before moving off (sets the Markov matrix)
        self.turn_window_s = turn_window_s
        self.turn_gate = turn_gate_deg_s  # the Moss/Howard corner is ~19 deg/s (0.5 m/s on a 1.5 m radius)
        self.max_turn = max_turn_deg_s
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None  # (2 models, 4) px py vx vy
        self.P: np.ndarray | None = None
        self.mu = np.array([0.5, 0.5])
        self.t: float | None = None
        self.n = 0
        self.turn = 0.0  # deg/s, smoothed
        self.heading: float | None = None
        self.turned = 0.0  # deg turned in the current turn (route prior)
        self._turn_h0: float | None = None
        self._hist: deque = deque()  # (t, heading) while walking
        self._s: np.ndarray | None = None  # the smoother CV filter, for the heading
        self._sP: np.ndarray | None = None

    # -- models
    def _models(self, dt: float, accel: float | None = None):
        F_cv = np.eye(4)
        F_cv[0, 2] = F_cv[1, 3] = dt
        q = (self.accel if accel is None else accel) ** 2
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q_cv = q * G @ G.T + 1e-6 * np.eye(4)
        F_st = np.diag([1.0, 1.0, 0.0, 0.0])
        Q_st = np.diag([0.02 * dt, 0.02 * dt, 1e-3, 1e-3])  # head sway; velocity pinned near 0
        return (F_cv, Q_cv), (F_st, Q_st)

    def _pi(self, dt: float) -> np.ndarray:
        a, b = (1 - math.exp(-dt / s) for s in self.switch_s)
        return np.array([[1 - a, a], [b, 1 - b]])  # rows: from CV, from STOP

    def combined(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.mu @ self.x
        P = sum(m * (Pi + np.outer(xi - x, xi - x)) for m, xi, Pi in zip(self.mu, self.x, self.P))
        return x, P

    def update(self, t: float, z: np.ndarray, R: np.ndarray) -> None:
        z = np.asarray(z, float)
        if self.x is None:
            x0 = np.array([z[0], z[1], 0.0, 0.0])
            P0 = np.diag([R[0, 0], R[1, 1], 0.5, 0.5])
            self.x, self.P = np.stack([x0, x0]), np.stack([P0, P0.copy()])
            self.mu, self.t, self.n = np.array([0.5, 0.5]), t, 1
            self._s, self._sP = x0.copy(), P0.copy()
            return
        dt = t - self.t
        if dt <= 1e-4:
            return
        self.t = t
        # IMM mixing
        pi = self._pi(dt)
        c = self.mu @ pi
        w = pi * self.mu[:, None] / np.maximum(c, 1e-12)  # w[i, j]: from i into j
        xm = w.T @ self.x
        Pm = np.stack([sum(w[i, j] * (self.P[i] + np.outer(self.x[i] - xm[j], self.x[i] - xm[j])) for i in range(2))
                       for j in range(2)])
        like = np.zeros(2)
        for j, (F, Q) in enumerate(self._models(dt)):
            self.x[j], self.P[j], nu, S = _kf(xm[j], Pm[j], F, Q, z, R)
            like[j] = math.exp(-0.5 * nu @ np.linalg.solve(S, nu)) / (2 * math.pi * math.sqrt(np.linalg.det(S)))
        mu = c * np.maximum(like, 1e-300)
        self.mu = mu / mu.sum() if mu.sum() > 0 else np.array([0.5, 0.5])
        self.n += 1
        # heading: of a second, smoother CV filter (heading_accel): the IMM's own velocity swings +-15 deg on a
        # straight walk. Turn rate: least-squares slope of that heading over turn_window_s. Only while walking.
        F, Q = self._models(dt, self.heading_accel)[0]
        self._s, self._sP, _, _ = _kf(self._s, self._sP, F, Q, z, R)
        if np.hypot(*self.combined()[0][2:]) > 0.2 and np.hypot(*self._s[2:]) > 0.15:
            self.heading = compass_deg(self._s[2:])
            self._hist.append((t, self.heading))
            while self._hist and t - self._hist[0][0] > self.turn_window_s:
                self._hist.popleft()
            if t - self._hist[0][0] >= 0.4:
                ts = np.array([p[0] for p in self._hist])
                hs = np.array([wrap(p[1] - self._hist[0][1]) for p in self._hist])
                rate = float(np.polyfit(ts - ts[0], hs, 1)[0])
                self.turn = max(-self.max_turn, min(self.max_turn, rate))
        else:
            self._hist.clear()
            self.turn = 0.0
        # how far into the current turn (heading now - heading when it began); hysteresis: on above 8, off below 4
        if abs(self.turn) > 8 and self._turn_h0 is None and self._hist:
            self._turn_h0 = self._hist[0][1]
        elif abs(self.turn) < 4:
            self._turn_h0 = None
        self.turned = 0.0 if self._turn_h0 is None or self.heading is None else wrap(self.heading - self._turn_h0)

    def predict(self, t_now: float, horizon_s: float = 1.5, step_s: float = 0.1,
                route_turn_deg: float | None = None) -> Prediction | None:
        """Path from t_now to t_now + horizon_s. route_turn_deg: the route prior's next corner (route_turn_ahead)."""
        if self.x is None:
            return None
        x, P = self.combined()
        lead = max(0.0, t_now - self.t)  # the filter state is at the last frame's time (video lag + age)
        speed = float(np.hypot(*x[2:]))
        heading = self.heading if self.heading is not None else compass_deg(x[2:])  # the smooth one
        turn = self.turn if speed > 0.2 else 0.0
        budget = math.inf  # how much more the path may turn: the rest of the route's corner, if turning into it
        gate = self.turn_gate  # a turn rate below this is heading noise: predict straight
        if route_turn_deg and turn and math.copysign(1, route_turn_deg) == math.copysign(1, turn):
            budget = max(0.0, abs(route_turn_deg) - abs(self.turned))
            gate /= 2  # the route says a corner this way is coming: believe a weaker turn
        if abs(turn) < gate:
            turn = 0.0
        n = int(round(horizon_s / step_s))
        F_cv, Q_cv = self._models(step_s)[0]
        pos, h, turned, tau0 = x[:2].copy(), heading, 0.0, 0.0
        path, std = [], []
        for k in range(n + 1):
            tau = lead + k * step_s
            steps = max(1, int(math.ceil((tau - tau0) / 0.02)))
            dt, tau0 = (tau - tau0) / steps, tau
            for _ in range(steps):
                d = turn * dt
                if abs(turned + d) > budget:
                    d = math.copysign(max(0.0, budget - abs(turned)), turn)
                turned += abs(d)
                h += d
                pos = pos + speed * fwd_right(h)[0] * dt
            path.append(pos.copy())
        Pk = P.copy()
        for _ in range(int(round(lead / step_s))):
            Pk = F_cv @ Pk @ F_cv.T + Q_cv
        for _ in range(n + 1):
            std.append(math.sqrt(max(0.0, (Pk[0, 0] + Pk[1, 1]) / 2)))
            Pk = F_cv @ Pk @ F_cv.T + Q_cv
        return Prediction(np.array(path), np.array(std), speed, heading, turn, float(self.mu[1]), step_s, self.n)


# ------------------------------------------------------------------ drone planner
@dataclass
class Plan:
    lr: float
    fb: float
    yaw: float
    drone_path: np.ndarray  # (n+1, 2) predicted drone positions under the chosen sticks
    target_path: np.ndarray  # (n+1, 2) standoff points the plan chases


def trail_point(trail: np.ndarray, back_m: float) -> np.ndarray:
    """The point back_m of arc length behind the end of a polyline (its first point if the trail is shorter)."""
    rem = back_m
    for i in range(len(trail) - 1, 0, -1):
        seg = trail[i] - trail[i - 1]
        L = float(np.hypot(*seg))
        if L >= rem and L > 0:
            return trail[i] - seg * (rem / L)
        rem -= L
    return trail[0].copy()


@dataclass
class PlannerCfg:
    stick_m_s: float = 0.01  # m/s per stick unit (FollowCfg.stick_m_s; measured ~0.0096)
    response_s: float = 0.4  # first-order speed lag of the Tello (FollowCfg.response_s)
    yaw_deg_s: float = 1.0  # deg/s per yaw stick unit. ponytail: guessed (the sim's value); measure on the Tello
    slew: float = 150.0  # stick units per second (FollowCfg.slew)
    behind_m: float = 1.2
    min_range_m: float = 0.7
    gap_s: float = 0.4  # standoff grows by walker speed x this (a cruise-control time gap): room to brake when the
    # walker stops, which the drone only sees video_lag_s + a few frames later
    close_m: float = 0.9  # plans closer than this to the walker are penalised (the head leaves the frame ~0.77 m)
    lead_s: float = 0.6  # yaw looks this far ahead
    max_lead_deg: float = 15.0  # yaw never aims further than this from where the walker is now
    k_yaw: float = 1.5
    db_yaw_deg: float = 2.0
    n_fb: int = 21  # candidate grid
    n_lr: int = 9
    w_effort: float = 0.02
    w_change: float = 0.3
    w_close: float = 50.0


class DronePlanner:
    def __init__(self, cfg: PlannerCfg | None = None):
        self.cfg = cfg or PlannerCfg()

    def plan(self, pred: Prediction, trail: np.ndarray, pos: np.ndarray, heading_deg: float, v_body: np.ndarray,
             prev: np.ndarray, lim: np.ndarray) -> Plan:
        """pred: walker path from now; trail: past walker positions (oldest first, ending near pred.path[0]);
        pos/heading_deg/v_body (fwd, right m/s): the drone now; prev: the last (lr, fb, ud, yaw) sent;
        lim: per-channel stick limits (lr, fb, ud, yaw)."""
        c = self.cfg
        n, dt = len(pred.path) - 1, pred.step_s
        full = np.vstack([trail, pred.path]) if len(trail) else pred.path
        k0 = len(full) - len(pred.path)
        back = c.behind_m + c.gap_s * pred.speed
        targets = np.array([trail_point(full[:k0 + k + 1], back) for k in range(n + 1)])

        # yaw: from where the drone will be to where the walker will be, near the walker's bearing now
        f0, r0 = fwd_right(heading_deg)
        k_lead = min(n, int(round(c.lead_s / dt)))
        v_world = v_body[0] * f0 + v_body[1] * r0
        look = pred.path[k_lead] - (pos + v_world * k_lead * dt)
        now = pred.path[0] - pos
        b_now = wrap(compass_deg(now) - heading_deg)
        err = wrap(compass_deg(look) - heading_deg) if np.hypot(*look) > 0.3 else b_now
        err = max(b_now - c.max_lead_deg, min(b_now + c.max_lead_deg, err))
        e = 0.0 if abs(err) <= c.db_yaw_deg else err - math.copysign(c.db_yaw_deg, err)
        yaw = max(-lim[3], min(lim[3], c.k_yaw * e))

        # sampling MPC over constant (fb, lr) candidates, through slew limit + first-order lag
        fb_c = np.linspace(-lim[1], lim[1], c.n_fb)
        lr_c = np.linspace(-lim[0], lim[0], c.n_lr)
        FB, LR = (a.ravel() for a in np.meshgrid(fb_c, lr_c))
        m = len(FB)
        u_fb, u_lr = np.full(m, prev[1], float), np.full(m, prev[0], float)
        vf, vr = np.full(m, v_body[0]), np.full(m, v_body[1])
        px, py = np.full(m, pos[0]), np.full(m, pos[1])
        h = heading_deg
        cost = c.w_effort * ((FB / 100) ** 2 + (LR / 100) ** 2) * n
        cost += c.w_change * (((FB - prev[1]) / 100) ** 2 + ((LR - prev[0]) / 100) ** 2) * n
        paths = np.zeros((m, n + 1, 2))
        paths[:, 0] = pos
        step = c.slew * dt
        a = min(1.0, dt / c.response_s)
        for k in range(1, n + 1):
            u_fb += np.clip(FB - u_fb, -step, step)
            u_lr += np.clip(LR - u_lr, -step, step)
            vf += (u_fb * c.stick_m_s - vf) * a
            vr += (u_lr * c.stick_m_s - vr) * a
            h += yaw * c.yaw_deg_s * dt
            f, r = fwd_right(h)
            px = px + (vf * f[0] + vr * r[0]) * dt
            py = py + (vf * f[1] + vr * r[1]) * dt
            paths[:, k, 0], paths[:, k, 1] = px, py
            w = 3.0 if k == n else 1.0
            cost += w * ((px - targets[k, 0]) ** 2 + (py - targets[k, 1]) ** 2)
            d = np.hypot(px - pred.path[k, 0], py - pred.path[k, 1])
            cost += c.w_close * np.maximum(0.0, max(c.min_range_m, c.close_m) - d) ** 2
        best = int(cost.argmin())
        return Plan(float(LR[best]), float(FB[best]), float(yaw), paths[best], targets)
