"""A harsher closed-loop follow simulation than test_follow.run_sim: a Tello plant with the numbers measured on the
team's drone (stick-30 step tests) and the disturbances a sidewalk adds.

    python tests/follow_sim.py            # metrics table for the standard scenarios (no drone, ~20 s)

Plant (per channel: dead time, first-order lag, hard saturation at stick 100):
  forward / right  0.0096 m/s per stick unit (0.27-0.30 m/s at stick 30 -> ~0.96 m/s at 100, extrapolated), dead time
                   0.30 s (0.47 s in the worst fit), time constant 0.45 s
  up               0.0032 m/s per unit (0.32 m/s at 100), dead time 0.41 s
  yaw              0.55 deg/s per unit (55 deg/s at 100), dead time 0.15 s
  pitch            the body tilts with acceleration and speed (nose down to speed up): the level camera is not level,
                   so the head-top row, and with it the range, moves while the drone accelerates
  wind             a steady push plus Ornstein-Uhlenbeck gusts on the drone (what the Tello's own optical-flow hold
                   leaves over), in the world frame
Sensing: video at 30 fps, each frame `lag` old (base + uniform jitter), dropped frames, a detector at 15 Hz with pixel
noise and a bystander box now and then; telemetry 'h' in whole 10 cm steps with noise, yaw with noise; optional GPS
(bias + noise) for the route prior.
Metrics: range error to the 1.2 m standoff, standoff error along the walker's path, the drone's top speed, time with
the fb stick saturated, time in each follow state, the closest the camera came to the walker's head (3-D) and whether
fb ever increased while the drone was already inside the standoff and closing on the head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from kite.camera import Camera
from kite.config import SafetyCfg
from kite.drone import SafetyGovernor
from kite.drone.base import Drone
from kite.follow import Detection, FollowCfg
from kite.types import Telemetry

CAM = Camera()
W, H = CAM.width, CAM.height
G = 9.81


@dataclass
class Plant:
    stick_m_s: float = 0.0096
    dead_s: float = 0.30
    tau_s: float = 0.45
    ud_m_s: float = 0.0032
    ud_dead_s: float = 0.41
    ud_tau_s: float = 0.25
    yaw_deg_s: float = 0.55
    yaw_dead_s: float = 0.15
    yaw_tau_s: float = 0.10
    att_tau_s: float = 0.15  # attitude loop: the tilt follows the tilt the speed loop asks for with this lag
    drag_tilt_deg: float = 1.5  # steady nose-down tilt per m/s of airspeed (rotor drag)
    pitch_sign: float = -1.0  # telemetry pitch of a nose-down tilt (the real sign is unverified)
    camera_tilt: float = 1.0  # 0: the camera stays level whatever the body does (to isolate the tilt effect)


@dataclass
class Scenario:
    speed: float = 0.5  # walker m/s on the walking legs
    walk: list = field(default_factory=lambda: [("stand", 1.0), ("walk", 12.0), ("stand", 3.0), ("turn", -90.0),
                                                ("walk", 8.0), ("stand", 3.0)])
    max_rc: int = 50
    plant: Plant = field(default_factory=Plant)
    wind_m_s: tuple = (0.0, 0.0)  # steady (east, north) push on the drone
    gust_m_s: float = 0.0  # 1-sigma of the gusts
    gust_tau_s: float = 2.0
    lag_s: float = 0.25  # base video lag
    lag_jitter_s: float = 0.0  # + uniform(0, jitter)
    drop: float = 0.1
    det_hz: float = 15.0
    noise_px: float = 3.0
    bystander: float = 0.0  # probability per detection of an extra box elsewhere in the frame
    h_step_m: float = 0.0  # telemetry height quantisation ('h' is whole cm; some firmwares report 10 cm steps)
    h_noise_m: float = 0.0
    yaw_noise_deg: float = 0.0
    blackout: tuple = (1e9, 1e9)
    seed: int = 0


def heading_axes(heading_deg):
    h = math.radians(heading_deg)
    return np.array([math.sin(h), math.cos(h), 0.0]), np.array([math.cos(h), -math.sin(h), 0.0])


def walker_points(height=1.70):
    pts = []
    for a in np.linspace(0, math.pi, 12):
        for b in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            pts.append((0.10 * math.sin(a) * math.cos(b), 0.08 * math.sin(a) * math.sin(b),
                        height - 0.11 + 0.11 * math.cos(a)))
    for z in np.arange(height - 0.25, height - 0.85, -0.02):
        for s in np.linspace(-0.22, 0.22, 23):
            pts += [(0.12, s, z), (-0.12, s, z)]
        for f in np.linspace(-0.12, 0.12, 13):
            pts += [(f, 0.22, z), (f, -0.22, z)]
    return np.array(pts)


def project(points, pos, heading_deg, tilt_deg, cam=CAM):
    """World points -> pixels through the drone camera, the body tilted nose-down by tilt_deg."""
    f, r = heading_axes(heading_deg)
    d = points - pos
    zf, xc, up = d @ f, d @ r, d[:, 2]
    a = math.radians(tilt_deg)  # nose down: the optical axis points a below the horizon
    zc = zf * math.cos(a) - up * math.sin(a)
    yc = -(up * math.cos(a) + zf * math.sin(a))
    with np.errstate(divide="ignore", invalid="ignore"):
        return cam.cx + cam.fx * xc / zc, cam.cy + cam.fy * yc / zc, zc


def detect(cam_pos, cam_heading, tilt, w_pos, w_heading, body, rng, sc: Scenario):
    f, r = heading_axes(w_heading)
    world = w_pos + body[:, :1] * f + body[:, 1:2] * r + body[:, 2:3] * np.array([0, 0, 1.0])
    u, v, zc = project(world, cam_pos, cam_heading, tilt)
    front = zc > 0.05
    inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = []
    if inside.sum() >= 5:
        x1, x2, y1, y2 = u[inside].min(), u[inside].max(), v[inside].min(), v[inside].max()
        if np.any(front & (v >= H) & (u >= x1) & (u <= x2)):
            y2 = H
        if x2 - x1 >= 15 and y2 - y1 >= 15:
            n = rng.normal(0, sc.noise_px, 4)
            out.append(Detection(x1 + n[0], y1 + n[1], x2 + n[2], min(H, y2 + n[3]), 0.6))
    if rng.random() < sc.bystander:  # someone further up the sidewalk, not touching the bottom edge
        x = rng.uniform(0, W - 60)
        out.append(Detection(x, 250, x + 50, 420, 0.7))
    return out


class HonestTello(Drone):
    """Kinematic Tello with per-channel dead time, lag and saturation, tilt and wind (see the module doc)."""

    def __init__(self, pos, heading, plant: Plant, sc: Scenario, rng):
        self.pos, self.heading = np.array(pos, float), float(heading)
        self.p, self.sc, self.rng = plant, sc, rng
        self.v = np.zeros(3)  # forward, right, up response (m/s), body frame, before wind
        self.w = 0.0
        self.cmd = (0, 0, 0, 0)
        self.hist: list = []  # (t, cmd): for the dead times
        self.now = 0.0
        self.gust = np.zeros(2)
        self.v_world = np.zeros(3)
        self.tilt = 0.0
        self.att = np.zeros(2)  # pitch (nose down +), roll from the attitude loop, rad

    def _cmd_at(self, t):
        for tt, c in reversed(self.hist):
            if tt <= t:
                return c
        return (0, 0, 0, 0)

    def step(self, dt):
        p, sc = self.p, self.sc
        lr, fb, _, _ = self._cmd_at(self.now - p.dead_s)
        ud = self._cmd_at(self.now - p.ud_dead_s)[2]
        yaw = self._cmd_at(self.now - p.yaw_dead_s)[3]
        sat = lambda x: max(-100, min(100, x))  # noqa: E731
        target = np.array([sat(fb) * p.stick_m_s, sat(lr) * p.stick_m_s, sat(ud) * p.ud_m_s])
        # speed loop -> tilt asked for -> attitude loop (lag) -> acceleration = g tan(tilt), per horizontal axis
        a_want = (target[:2] - self.v[:2]) / p.tau_s
        tilt_want = np.arctan2(a_want, G)
        self.att += (tilt_want - self.att) * min(1.0, dt / p.att_tau_s)
        a_xy = G * np.tan(self.att)
        acc = np.array([a_xy[0] * dt, a_xy[1] * dt, (target[2] - self.v[2]) * dt / p.ud_tau_s])
        self.v += acc
        self.w += (sat(yaw) * p.yaw_deg_s - self.w) * dt / p.yaw_tau_s
        self.heading += self.w * dt
        # gusts: OU process
        a = dt / sc.gust_tau_s
        self.gust += -self.gust * a + sc.gust_m_s * math.sqrt(2 * a) * self.rng.normal(0, 1, 2)
        wind = np.array(sc.wind_m_s) + self.gust
        f, r = heading_axes(self.heading)
        self.v_world = self.v[0] * f + self.v[1] * r + np.array([wind[0], wind[1], self.v[2]])
        self.pos += self.v_world * dt
        self.tilt = p.camera_tilt * (math.degrees(self.att[0]) + p.drag_tilt_deg * self.v[0])
        self.now += dt

    def rc(self, lr, fb, ud, yaw):
        self.cmd = (lr, fb, ud, yaw)
        self.hist.append((self.now, self.cmd))
        if len(self.hist) > 400:
            del self.hist[:200]

    def telemetry(self):
        sc = self.sc
        h = float(self.pos[2]) + self.rng.normal(0, sc.h_noise_m) if sc.h_noise_m else float(self.pos[2])
        if sc.h_step_m:
            h = round(h / sc.h_step_m) * sc.h_step_m
        yaw = self.heading + (self.rng.normal(0, sc.yaw_noise_deg) if sc.yaw_noise_deg else 0.0)
        return Telemetry(t=self.now, height_m=h, tof_m=h, yaw_deg=yaw, pitch_deg=self.p.pitch_sign * self.tilt,
                         battery_pct=80.0, vx=float(self.v_world[0]), vy=float(self.v_world[1]))

    flying = True

    def takeoff(self): ...
    def land(self): ...
    def emergency(self): ...
    def move(self, direction, cm): ...
    def rotate(self, deg): ...
    def stop(self): self.cmd = (0, 0, 0, 0)
    def busy(self): return False
    def last_result(self): return "ok"
    def frame_source(self): return None


def schedule(sc: Scenario, radius=1.5):
    out, t = [], 0.0
    for kind, x in sc.walk:
        if kind == "stand":
            t += x
            out.append((t, 0.0, 0.0))
        elif kind == "walk":
            t += x / sc.speed
            out.append((t, sc.speed, 0.0))
        else:
            dur = abs(math.radians(x)) * radius / sc.speed
            t += dur
            out.append((t, sc.speed, x / dur))
    return out


def legacy_follow(ref: str = "origin/main"):
    """kite/follow.py as it is at git `ref` (the controller before this change), imported as kite._follow_legacy."""
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    src = subprocess.run(["git", "show", f"{ref}:kite/follow.py"], cwd=root, capture_output=True, text=True,
                         check=True).stdout
    spec = importlib.util.spec_from_loader("kite._follow_legacy", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__, mod.__file__ = "kite", str(root / "kite" / "follow.py")
    sys.modules["kite._follow_legacy"] = mod  # dataclasses look their module up there
    exec(compile(src, f"{ref}:kite/follow.py", "exec"), mod.__dict__)
    return mod


def run(sc: Scenario | None = None, cfg=None, dt: float = 1 / 90, mod=None) -> list[dict]:
    """One closed-loop run; a log row per 10 Hz control tick after follow engages. mod: a module with FollowCfg,
    FollowController and WalkerTracker (default kite.follow; legacy_follow() for the old controller)."""
    import kite.follow as default_mod

    mod = mod or default_mod
    sc = sc or Scenario()
    cfg = cfg or mod.FollowCfg()
    rng = np.random.default_rng(sc.seed)
    body = walker_points(1.70)
    head_top = 1.70
    sched = schedule(sc)
    w_pos, w_head = np.zeros(3), 0.0
    sim = HonestTello([0.0, -cfg.behind_m, head_top + cfg.above_m], 0.0, sc.plant, sc, rng)
    gov = SafetyGovernor(sim, SafetyCfg(max_rc=sc.max_rc))
    tracker, ctrl = mod.WalkerTracker(cfg), mod.FollowController(cfg, CAM, sc.max_rc)
    history = []
    engaged, next_frame, next_det, next_tick = False, 0.0, 0.0, 0.0
    tr, log, t, prev_fb = None, [], 0.0, 0
    while t < sched[-1][0]:
        seg = next(s for s in sched if t < s[0])
        w_head += seg[2] * dt
        f, _ = heading_axes(w_head)
        w_pos = w_pos + f * seg[1] * dt
        sim.step(dt)
        t = sim.now
        history.append((t, sim.pos.copy(), sim.heading, w_pos.copy(), w_head, sim.tilt))
        if t >= next_frame:
            next_frame += 1 / 30
            if t >= next_det:  # the detector runs on the newest frame at det_hz
                next_det += 1 / sc.det_hz
                lag = sc.lag_s + rng.uniform(0, sc.lag_jitter_s)
                past = next((h for h in reversed(history) if h[0] <= t - lag), history[0])
                dets = detect(past[1], past[2], past[5], past[3], past[4], body, rng, sc)
                if rng.random() < sc.drop or sc.blackout[0] <= t < sc.blackout[1]:
                    dets = []
                tr = tracker.update(dets, t, W, H)
        if t >= next_tick:
            next_tick += 0.1
            gov.check(t, t)
            if not engaged and tr is not None and tr.confirmed:
                engaged = True
                ctrl.reset(float(sim.pos[2]))
            if engaged:
                cmd = ctrl.update(tr, gov.telemetry(), t)
                gov.rc(*cmd.rc)
                rel = w_pos - sim.pos
                dist = float(np.hypot(rel[0], rel[1]))
                fwd, _ = heading_axes(sim.heading)
                closing = float(np.dot(rel[:2], sim.v_world[:2] - (w_pos - history[-2][3])[:2] / dt)) / max(dist, 1e-6)
                head = w_pos + np.array([0, 0, head_top])
                log.append({"t": t, "state": cmd.state, "dist": dist, "head3d": float(np.linalg.norm(head - sim.pos)),
                            "height": float(sim.pos[2]), "rc": cmd.rc, "fb_inc": cmd.fb - prev_fb,
                            "closing": closing, "speed": float(np.hypot(*sim.v_world[:2])), "walker": seg[1],
                            "bearing": (math.degrees(math.atan2(rel[0], rel[1])) - sim.heading + 180) % 360 - 180,
                            "warning": getattr(cmd, "warning", None),
                            "standoff_err": float(np.hypot(*(trail_target(history, cfg.behind_m) - sim.pos[:2])))})
                prev_fb = cmd.fb
    return log


def trail_target(history, behind_m):
    pts, rem = history[-1][3][:2], behind_m
    for h in reversed(history[:-1]):
        d = float(np.hypot(*(h[3][:2] - pts)))
        if d >= rem:
            return pts + (h[3][:2] - pts) * rem / d
        rem -= d
        pts = h[3][:2]
    return pts


def metrics(log: list[dict], cfg: FollowCfg | None = None, max_rc: int = 50, after_s: float = 3.0) -> dict:
    """range_err: distance to the controller's own target standoff (behind_m + headway_s x the walker's true speed),
    only while following. accel_toward_head: ticks where the drone was really inside the standoff, closing on the
    walker, and fb went up (the head guard works on estimates, so this can be > 0 when the estimate lags)."""
    cfg = cfg or FollowCfg()
    run_ = [r for r in log if r["t"] > after_s]
    dist = np.array([r["dist"] for r in run_])
    fol = [r for r in run_ if r["state"] == "following"]
    err = np.abs(np.array([r["dist"] - (cfg.behind_m + cfg.headway_s * r["walker"]) for r in fol] or [np.nan]))
    fb = np.array([r["rc"][1] for r in run_])
    states = [r["state"] for r in run_]
    lim = min(max_rc, cfg.max_fb)
    toward = [r for r in run_ if r["dist"] < cfg.behind_m - 0.1 and r["closing"] > 0.05 and r["fb_inc"] > 0
              and r["rc"][1] > 0]
    out_t = [r["t"] for r in run_ if r["state"] == "outpaced"]
    rc = np.array([r["rc"] for r in run_], float)
    return {
        "range_err_mean": float(np.nanmean(err)), "range_err_p95": float(np.nanpercentile(err, 95)),
        "first_outpaced_s": out_t[0] if out_t else None,
        "jerk": float(np.abs(np.diff(rc, axis=0)).sum(1).mean()),
        "standoff_mean": float(np.mean([r["standoff_err"] for r in run_])),
        "min_dist": float(dist.min()), "max_dist": float(dist.max()),
        "min_head_3d": float(min(r["head3d"] for r in run_)),
        "max_speed": float(max(r["speed"] for r in run_)),
        "sat_frac": float(np.mean(np.abs(fb) >= lim - 1)),
        "max_bearing": float(max(abs(r["bearing"]) for r in run_)),
        "max_h_err": float(max(abs(r["height"] - 2.0) for r in run_)),
        "lost_ticks": states.count("lost"), "outpaced_ticks": states.count("outpaced"),
        "following_frac": states.count("following") / len(states),
        "accel_toward_head": len(toward),
        "warnings": sorted({r["warning"] for r in run_ if r["warning"]}),
    }


SCENARIOS = {  # max_rc 50 is ~0.48 m/s on this plant: walkers at 0.4 m/s; faster ones get a higher max_rc
    "nominal 0.4 m/s, max_rc 50": Scenario(speed=0.4),
    "headwind drift 0.1 m/s + gusts, max_rc 70": Scenario(speed=0.4, max_rc=70, wind_m_s=(0.0, -0.1),
                                                          gust_m_s=0.05, seed=1),
    "crosswind drift 0.1 m/s + gusts, max_rc 70": Scenario(speed=0.4, max_rc=70, wind_m_s=(0.1, 0.0),
                                                           gust_m_s=0.05, seed=8),
    "lag 0.25-0.45 s, 30% drops": Scenario(speed=0.4, lag_s=0.25, lag_jitter_s=0.2, drop=0.3, seed=2),
    "slow plant (0.47 s dead time), 0.35 m/s": Scenario(speed=0.35, plant=Plant(dead_s=0.47), seed=3),
    "noisy telemetry + bystanders": Scenario(speed=0.4, h_step_m=0.1, h_noise_m=0.03, yaw_noise_deg=1.0,
                                             bystander=0.1, noise_px=5.0, seed=4),
    "everything at once, max_rc 70": Scenario(speed=0.4, max_rc=70, wind_m_s=(0.07, -0.07), gust_m_s=0.05,
                                              lag_jitter_s=0.2, drop=0.2, plant=Plant(dead_s=0.4), h_step_m=0.1,
                                              h_noise_m=0.03, yaw_noise_deg=1.0, bystander=0.05, noise_px=5.0,
                                              seed=5),
    "0.6 m/s walker, max_rc 80": Scenario(speed=0.6, max_rc=80, seed=7),
    "1.2 m/s walker, max_rc 50 (outpaced)": Scenario(speed=1.2, walk=[("stand", 1.0), ("walk", 14.0),
                                                                      ("stand", 12.0)], seed=6),
    "headwind drift 0.3 m/s, max_rc 50 (outpaced)": Scenario(speed=0.4, wind_m_s=(0.0, -0.3), seed=9),
    "1.0 m/s walker stops after 4.2 m, max_rc 50 (outpaced, resumes)": Scenario(
        speed=1.0, walk=[("stand", 1.0), ("walk", 4.2), ("stand", 15.0)], seed=10),
}


def table(cfg: FollowCfg | None = None, legacy: bool = False) -> str:
    """Markdown table of the metrics per scenario. legacy: the controller on origin/main (before this change)."""
    mod = legacy_follow() if legacy else None
    keys = ["range_err_mean", "range_err_p95", "min_dist", "max_speed", "sat_frac", "lost_ticks", "outpaced_ticks",
            "first_outpaced_s", "accel_toward_head", "jerk"]
    lines = ["| scenario | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, sc in SCENARIOS.items():
        m = metrics(run(sc, None if legacy else cfg, mod=mod), cfg, max_rc=sc.max_rc)
        lines.append(f"| {name} | " + " | ".join(f"{m[k]:.2f}" if isinstance(m[k], float) else str(m[k])
                                                   for k in keys) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    print(table(legacy="--legacy" in sys.argv))
