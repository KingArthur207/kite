"""Tello response model in sent stick units (DESIGN 2.1, 8 R3, 10 R10): per-axis dead time, first-order lag, drift,
wind, a position-held hover, and pitch.

Per axis (lateral, forward, vertical, yaw) the sent stick goes through a command dead time and then a first-order
lag to a velocity (yaw: a clockwise rate). Lateral uses the forward gain, lag and dead time. The physics runs in
PHYS_DT = 0.01 s sub-steps and each dead time is a whole number of them (rounded: within 5 ms).

World frame: x east, y north, z up (m above the takeoff point). Heading in radians CLOCKWISE from north, like
kite.types; forward = (sin h, cos h), right = (cos h, -sin h). +yaw stick turns clockwise, +lr moves right.

Disturbances: on each horizontal axis an Ornstein-Uhlenbeck hover drift (tau 2 s) plus an OU street wind (tau 3 s),
each started from its stationary distribution, their normals drawn at reset (the same for every controller).
- Sticks off zero: they act as velocity offsets, so the drone is pushed along.
- Hover (the lateral and forward sticks reaching the drone are both 0): the Tello's vision position hold. The
  displacement the disturbances cause since the hover began is pulled back with time constant hold_tau_s, so the
  drone wanders a bounded 5-15 cm instead of random-walking. The commanded motion itself (braking) is not undone.

Pitch follows the commanded forward acceleration only: pitch = -pitch_scale * atan(a_fwd / g), capped at 25 deg,
+ = nose UP (interfaces.py). Accelerating forward tilts the nose down; braking tilts it up, which moves the walker
further down the fixed camera's frame.

Pure Python floats on purpose: this runs every sub-step of every training episode.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

PHYS_DT = 0.01  # s per physics sub-step (DESIGN 8, R3)
G = 9.81
N_NOISE_PER_TICK = 4  # drift x, drift y, wind x, wind y


@dataclass(frozen=True)
class DroneParams:
    fwd_gain_mps: float = 0.96  # m/s at stick 100 (defaults: the lag-test medians)
    yaw_gain_dps: float = 55.0
    vz_gain_mps: float = 0.32
    tau_fwd_s: float = 0.45
    tau_yaw_s: float = 0.02
    tau_z_s: float = 0.02
    fwd_dead_s: float = 0.47
    yaw_dead_s: float = 0.18
    vz_dead_s: float = 0.41
    pitch_scale: float = 0.75
    pitch_max_deg: float = 25.0
    drift_sigma_mps: float = 0.0
    drift_tau_s: float = 2.0
    wind_sigma_mps: float = 0.0
    wind_tau_s: float = 3.0
    hold_tau_s: float = 1.5


def noise_len(n_ticks: int) -> int:
    """Standard normals DroneModel needs for n_ticks sub-steps (plus the initial disturbance state)."""
    return N_NOISE_PER_TICK * (n_ticks + 1)


class DroneModel:
    """State: x, y, z (m), heading (rad, clockwise from north), commanded-response world velocity vx, vy, vz (m/s),
    yaw_rate_dps (clockwise), pitch_deg (+ nose up), holding (the position hold is on)."""

    def __init__(self, p: DroneParams, noise: Sequence[float] = (), dt: float = PHYS_DT):
        self.p = p
        self.dt = dt
        self._noise = list(noise)
        self._kf = 1.0 - math.exp(-dt / p.tau_fwd_s)
        self._ky = 1.0 - math.exp(-dt / p.tau_yaw_s)
        self._kz = 1.0 - math.exp(-dt / p.tau_z_s)
        self._kh = 1.0 / p.hold_tau_s
        # OU per process: d <- a d + b n, stationary std sigma
        self._ou = [(math.exp(-dt / tau), sigma * math.sqrt(1.0 - math.exp(-2.0 * dt / tau)), sigma)
                    for sigma, tau in ((p.drift_sigma_mps, p.drift_tau_s), (p.wind_sigma_mps, p.wind_tau_s))]
        # whole-sub-step dead time per axis (lr, fb, ud, yaw); the delay line holds the last `size` sent sticks
        self._lag = tuple(max(0, int(round(dead / dt))) for dead in (p.fwd_dead_s, p.fwd_dead_s, p.vz_dead_s,
                                                                         p.yaw_dead_s))
        self._size = max(self._lag) + 1
        self._pmax = p.pitch_max_deg
        self.reset(0.0, 0.0, 1.0, 0.0)

    def reset(self, x: float, y: float, z: float, heading: float) -> None:
        self.x, self.y, self.z, self.heading = x, y, z, heading
        self.vx = self.vy = self.vz = 0.0
        self.yaw_rate_dps = 0.0
        self.pitch_deg = 0.0
        self._line = [(0.0, 0.0, 0.0, 0.0)] * self._size
        self._n = 0
        self.holding = False
        self.ex = self.ey = 0.0  # disturbance displacement since the hover began
        n = self._noise
        (_, _, s_d), (_, _, s_w) = self._ou
        if len(n) >= N_NOISE_PER_TICK:  # stationary start: the drone is already drifting when follow begins
            self.dx, self.dy, self.wx, self.wy = s_d * n[0], s_d * n[1], s_w * n[2], s_w * n[3]
        else:
            self.dx = self.dy = self.wx = self.wy = 0.0

    def step(self, lr: float, fb: float, ud: float, yaw: float) -> None:
        """Advance one sub-step with these sticks sent at its start."""
        p, dt, line, size = self.p, self.dt, self._line, self._size
        n = self._n
        line[n % size] = (lr, fb, ud, yaw)
        l_lr, l_fb, l_ud, l_yaw = self._lag
        c_lr = line[(n - l_lr) % size][0] if n >= l_lr else 0.0
        c_fb = line[(n - l_fb) % size][1] if n >= l_fb else 0.0
        c_ud = line[(n - l_ud) % size][2] if n >= l_ud else 0.0
        c_yaw = line[(n - l_yaw) % size][3] if n >= l_yaw else 0.0
        self._n = n + 1

        # yaw first, then translate along the new heading
        self.yaw_rate_dps += (p.yaw_gain_dps * c_yaw * 0.01 - self.yaw_rate_dps) * self._ky
        h = self.heading + math.radians(self.yaw_rate_dps) * dt
        if h > math.pi:
            h -= 2.0 * math.pi
        elif h <= -math.pi:
            h += 2.0 * math.pi
        self.heading = h
        s, c = math.sin(h), math.cos(h)
        vf = p.fwd_gain_mps * c_fb * 0.01
        vr = p.fwd_gain_mps * c_lr * 0.01
        ox, oy = self.vx, self.vy
        k = self._kf
        self.vx = ox + (vf * s + vr * c - ox) * k
        self.vy = oy + (vf * c - vr * s - oy) * k
        self.vz += (p.vz_gain_mps * c_ud * 0.01 - self.vz) * self._kz

        nz = self._noise
        j = N_NOISE_PER_TICK * (n + 1)
        if j + 3 < len(nz):
            (ad, bd, _), (aw, bw, _) = self._ou
            self.dx = ad * self.dx + bd * nz[j]
            self.dy = ad * self.dy + bd * nz[j + 1]
            self.wx = aw * self.wx + bw * nz[j + 2]
            self.wy = aw * self.wy + bw * nz[j + 3]
        ux, uy = self.dx + self.wx, self.dy + self.wy  # disturbance velocity
        self.holding = c_lr == 0.0 and c_fb == 0.0
        if self.holding:  # position hold: the disturbance displacement relaxes back at 1 / hold_tau_s
            ux -= self.ex * self._kh
            uy -= self.ey * self._kh
            self.ex += ux * dt
            self.ey += uy * dt
        else:
            self.ex = self.ey = 0.0
        self.x += (self.vx + ux) * dt
        self.y += (self.vy + uy) * dt
        self.z += self.vz * dt
        if self.z < 0.0:
            self.z = self.vz = 0.0

        a_fwd = ((self.vx - ox) * s + (self.vy - oy) * c) / dt
        pt = -p.pitch_scale * math.degrees(math.atan(a_fwd / G))
        self.pitch_deg = self._pmax if pt > self._pmax else (-self._pmax if pt < -self._pmax else pt)

    def v_forward(self) -> float:
        """Commanded-response forward speed (m/s, no disturbances)."""
        return self.vx * math.sin(self.heading) + self.vy * math.cos(self.heading)
