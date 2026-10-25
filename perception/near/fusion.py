"""Robust cue fusion and the per-track Kalman filter, both in log-distance.

Per frame: the available cues are combined by inverse variance after an outlier test against their weighted
median (a depth reading that hit the wall behind a thin pole, a size cue on a child, a ground cue on a box whose
bottom is a shadow). If the survivors still disagree more than their sigmas allow, the fused variance is
inflated by the reduced chi-square, so disagreement shows up as uncertainty instead of false confidence.

Across frames: a constant-velocity Kalman filter on x = [l, r] with l = ln F and r = dl/dt. The rate state is
the reason for the log: r = -(closing speed) / F = -1 / TTC, so time to contact is a state of the filter, and
looming (d ln(box scale) / dt = -r, independent of any metric cue) is a direct measurement of it. Distance cues
correct l, looming corrects r, and each one helps the other through the covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .cues import Cue


@dataclass
class Fused:
    d: float
    sigma: float
    used: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def fuse(cues: dict[str, Cue | None], upper: float | None = None, floor_sigma: float = 0.04) -> Fused | None:
    """Inverse-variance mean of the cues in log-distance after rejecting outliers. `upper`: an upper bound on the
    distance (a box cut by the frame edge) that clamps the result."""
    items = [(k, c) for k, c in cues.items() if c is not None and c.d > 0 and math.isfinite(c.sigma)]
    if not items:
        if upper is not None:
            return Fused(upper * 0.8, 0.45, ["bound"], [])
        return None
    names = [k for k, _ in items]
    lx = np.array([math.log(c.d) for _, c in items])
    sg = np.maximum(np.array([c.sigma for _, c in items]), floor_sigma)
    w = 1.0 / sg ** 2
    med = weighted_median(lx, w)
    keep = np.abs(lx - med) <= np.maximum(2.5 * np.sqrt(sg ** 2 + 0.05 ** 2), 0.25)
    if not keep.any():
        keep = w == w.max()
    rejected = [n for n, k in zip(names, keep) if not k]
    lx, sg, w = lx[keep], sg[keep], w[keep]
    mean = float(np.sum(w * lx) / np.sum(w))
    var = 1.0 / float(np.sum(w))
    if lx.size > 1:
        chi2 = float(np.sum(w * (lx - mean) ** 2)) / (lx.size - 1)
        var *= max(1.0, chi2)
    if upper is not None and mean > math.log(upper):
        mean = math.log(upper)
    return Fused(math.exp(mean), math.sqrt(var), [n for n, k in zip(names, keep) if k], rejected)


class LogKalman:
    """x = [ln F, d ln F / dt]. `q` is the white-noise jerk of the log-rate (1/s^2): a person stepping out of a
    doorway changes the closing rate fast, so it is generous."""

    def __init__(self, l0: float, var0: float, t: float, q: float = 1.5, r0_sigma: float = 0.5):
        self.x = np.array([l0, 0.0])
        self.P = np.diag([var0, r0_sigma ** 2])
        self.t, self.q = t, q
        self.outliers = 0

    def predict(self, t: float) -> None:
        dt = max(0.0, min(1.0, t - self.t))
        self.t = t
        if dt == 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _update(self, z: float, var: float, H: np.ndarray, gate: float | None) -> bool:
        y = z - float(H @ self.x)
        S = float(H @ self.P @ H) + var
        if gate is not None and y * y / S > gate ** 2:
            return False
        K = self.P @ H / S
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        return True

    def update_distance(self, d: float, sigma: float, gate: float = 4.0) -> None:
        """A fused distance. Three gated-out readings in a row mean the track was wrong, not the cues: reset."""
        if self._update(math.log(d), sigma ** 2, np.array([1.0, 0.0]), gate):
            self.outliers = 0
            return
        self.outliers += 1
        if self.outliers >= 3:
            self.x[0] = math.log(d)
            self.P[0, 0] = sigma ** 2
            self.P[0, 1] = self.P[1, 0] = 0.0
            self.outliers = 0

    def update_rate(self, loom_rate: float, loom_sigma: float) -> None:
        """Looming: d ln(scale) / dt = -d ln F / dt."""
        if math.isfinite(loom_sigma):
            self._update(-loom_rate, loom_sigma ** 2, np.array([0.0, 1.0]), 5.0)

    @property
    def d(self) -> float:
        return math.exp(float(self.x[0]))

    @property
    def sigma(self) -> float:
        return math.sqrt(max(0.0, float(self.P[0, 0])))

    def ttc(self) -> tuple[float | None, float]:
        """(seconds to contact, its sigma) when the filter is confident the object is closing, else (None, inf)."""
        r, sr = float(self.x[1]), math.sqrt(max(1e-12, float(self.P[1, 1])))
        if r < 0 and -r > 2.0 * sr:
            return -1.0 / r, sr / (r * r)
        return None, math.inf
