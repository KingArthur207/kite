"""The fly's readout: steering-neuron rates -> yaw stick (DESIGN.md section 3).

    n   = lowpass_tau(rate) / norm                          10 DN rates, OUTPUT_GROUPS order; tau <= 150 ms
    yaw = YAW_SCALE * G * tanh(sum_t w_t (n_tR - n_tL) + b)  t over the 5 DN types
    yaw = sign(yaw) * max(0, |yaw| - deadzone)              soft deadzone (no jump at its edge)

`norm` holds each DN's mean rate over the hand-calibration stimuli (floored at NORM_FLOOR_HZ, so a silent cell does
not blow up its weight); the weights expect it, so a parameter file always travels with the norm it was fitted or
trained with. Sign: a walker on the right makes DNa02_R fire and DNa02_L fall silent, so w > 0 turns toward the
walker (yaw + = clockwise = turn right).
"""

from __future__ import annotations

import math

import numpy as np

from .interfaces import DN_TYPES, OUTPUT_GROUPS

YAW_SCALE = 60.0  # stick at G = 1 and a saturated tanh (the scale the readout was trained at, not a clamp)
TAU_MAX_MS = 150.0
NORM_FLOOR_HZ = 2.0

READOUT_NAMES = tuple(f"dec_w_yaw_{t}" for t in DN_TYPES) + ("dec_b_yaw", "dec_g_yaw", "dec_tau_yaw_ms",
                                                              "dec_deadzone")


def soft_deadzone(x: float, band: float) -> float:
    """Shrink |x| by band (0 inside it). A non-finite x passes through, so the caller's guards still see it."""
    if not math.isfinite(x):
        return x
    return math.copysign(max(0.0, abs(x) - band), x)


class YawReadout:
    """Stateful (the low-pass). reset() per episode, filter() during warmup, update() per brain tick."""

    def __init__(self, params: dict[str, float], norm: np.ndarray):
        missing = [k for k in READOUT_NAMES if k not in params]
        if missing:
            raise KeyError(f"readout parameters missing: {', '.join(missing)}")
        norm = np.asarray(norm, np.float64)
        if norm.shape != (len(OUTPUT_GROUPS),) or not np.all(np.isfinite(norm)):
            raise ValueError(f"norm must be {len(OUTPUT_GROUPS)} finite rates (OUTPUT_GROUPS order), got {norm}")
        self.norm = np.maximum(norm, NORM_FLOOR_HZ)
        w = np.array([float(params[f"dec_w_yaw_{t}"]) for t in DN_TYPES])
        signed = np.empty(len(OUTPUT_GROUPS))
        signed[0::2], signed[1::2] = -w, w  # OUTPUT_GROUPS is type-major, side-minor: L then R
        self.weights = signed / self.norm  # normalization folded into the weights
        self.bias = float(params["dec_b_yaw"])
        self.gain = float(params["dec_g_yaw"])
        self.tau_s = min(float(params["dec_tau_yaw_ms"]), TAU_MAX_MS) / 1000.0
        self.deadzone = float(params["dec_deadzone"])
        self.reset()

    def reset(self) -> None:
        self.filtered: np.ndarray | None = None
        self.drive = 0.0  # the tanh argument of the last update

    def filter(self, rates: np.ndarray, dt: float) -> np.ndarray:
        """Advance the low-pass by dt seconds with these rates (Hz); the first call initializes it."""
        r = np.asarray(rates, np.float64)
        if self.filtered is None:
            self.filtered = r.copy()
        else:
            self.filtered += (1.0 - math.exp(-dt / self.tau_s)) * (r - self.filtered)
        return self.filtered

    def update(self, rates: np.ndarray, dt: float) -> float:
        """Rates (Hz) of one brain tick -> yaw stick (+ = turn right), before any output conditioning. Raises
        FloatingPointError on a non-finite drive (NaN rates or parameters must never read as a 0 stick)."""
        self.drive = float(self.weights @ self.filter(rates, dt)) + self.bias
        if not math.isfinite(self.drive) or not math.isfinite(self.gain):
            raise FloatingPointError(f"non-finite readout drive {self.drive} (gain {self.gain})")
        return soft_deadzone(YAW_SCALE * self.gain * math.tanh(self.drive), self.deadzone)
