"""Follow reward (DESIGN 4), from simulator truth, once per control step (dt 0.1 s):

    r = -dt (w_r e_z + w_x e_x^2 + w_j j + w_l l + w_c c) + events

    e_z  min(3, max(0, |Z - Z_ref| - b) / b + max(0, Z - Z_ref - b)^2 / far_quad_m)   (b = 0.15 Z_ref)
         (the cap bounds the sum, as e_z_cap did in the weights' original calibration: after a loss the walker
         walks on and an uncapped far term would grow without bound and swamp every other term)
    e_x  true bearing to the walker's head (rad)
    j    |change of the sent yaw stick| / 100
    l    no true detection for > lost_after_s, or the controller is in state lost
    c    true range < FollowCfg.min_range_m
    events: walker lost (l rising) -10, too close (c rising) -20, collision -200, which ends it.

A collision is "overhead": the true horizontal range < 0.3 m while the drone is above the head. The rest of the
episode is then charged as lost at a true bearing of 90 deg (the lost and e_x terms, DESIGN 10 R10), so ending
early never pays.

Each term and event is summed separately (terms()), each as its signed contribution, so sum(terms) == return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TERMS = ("e_z", "e_x", "jerk", "lost", "too_close", "ev_loss", "ev_too_close", "ev_collision")


@dataclass(frozen=True)
class RewardCfg:
    w_r: float = 2.0
    w_x: float = 7.6
    w_j: float = 6.6
    w_l: float = 5.0
    w_c: float = 10.0
    ev_loss: float = -10.0
    ev_too_close: float = -20.0
    ev_collision: float = -200.0
    band_frac: float = 0.15
    e_z_cap: float = 3.0
    far_quad_m: float = 1.0
    lost_after_s: float = 1.0
    collide_m: float = 0.3


def e_z(z: float, z_ref: float, band: float, cap: float = 3.0, far_quad_m: float = 1.0) -> float:
    d = z - z_ref
    ad = d if d >= 0.0 else -d
    e = (ad - band) / band if ad > band else 0.0
    if d > band:
        e += (d - band) * (d - band) / far_quad_m
    return e if e < cap else cap


class Reward:
    def __init__(self, cfg: RewardCfg, z_ref: float, min_range_m: float):
        self.cfg, self.z_ref, self.min_range = cfg, z_ref, min_range_m
        self.band = cfg.band_frac * z_ref
        self.sums = dict.fromkeys(TERMS, 0.0)
        self._lost = self._close = False
        self.n_loss = self.n_close = 0

    def step(self, dt: float, z: float, bearing_rad: float, d_yaw: float, lost: bool) -> float:
        c, s = self.cfg, self.sums
        close = z < self.min_range
        a = -dt * c.w_r * e_z(z, self.z_ref, self.band, c.e_z_cap, c.far_quad_m)
        b = -dt * c.w_x * bearing_rad * bearing_rad
        j = -dt * c.w_j * abs(d_yaw) * 0.01
        lo = -dt * c.w_l if lost else 0.0
        cl = -dt * c.w_c if close else 0.0
        s["e_z"] += a
        s["e_x"] += b
        s["jerk"] += j
        s["lost"] += lo
        s["too_close"] += cl
        r = a + b + j + lo + cl
        if lost and not self._lost:
            s["ev_loss"] += c.ev_loss
            self.n_loss += 1
            r += c.ev_loss
        if close and not self._close:
            s["ev_too_close"] += c.ev_too_close
            self.n_close += 1
            r += c.ev_too_close
        self._lost, self._close = lost, close
        return r

    def collision(self, remaining_s: float = 0.0) -> float:
        """The collision event plus the rest of the episode (remaining_s) charged as lost at 90 deg."""
        c, s = self.cfg, self.sums
        rest = max(0.0, remaining_s)
        lo, ex = -c.w_l * rest, -c.w_x * (0.5 * math.pi) ** 2 * rest
        s["ev_collision"] += c.ev_collision
        s["lost"] += lo
        s["e_x"] += ex
        return c.ev_collision + lo + ex

    @property
    def total(self) -> float:
        return sum(self.sums.values())
