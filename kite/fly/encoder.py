"""The fly's eyes: TargetFeatures -> Poisson rates of the brain's 22 visual input channels (DESIGN.md section 3).

Channels (interfaces.CHANNELS): 8 LC10a azimuth bins per side, then LC9, LC11 and AROUSAL (P1) per side.

The retinotopy is ours (the connectome has none for LC10a): the right LC10a covers bearings from -overlap to
+THETA_MAX_DEG in 8 equal bins, bin 0 frontal, and the left side mirrors it. A walker at bearing theta (+ right),
normalized size s and bearing rate dtheta drives bin k of centre c_k with

    rate = r_max * gain_k * arousal * exp(-0.5 ((theta - c_k) / (sigma * s^a))^2) * s^b * (1 + k_v |dtheta|)

(Gaussian tuning that widens and brightens as the walker gets closer, and a little more drive when it moves),
capped at RATE_CAP_HZ. LC9 and LC11 get `enc_lc9_hz` / `enc_lc11_hz` times how strongly the spot falls in that
side's field, capped at LC_CAP_HZ: stronger LC9/LC11 drive swamps LC10a and breaks steering on this subgraph.
Arousal is a gain on LC10a ("gain" mode): P1 has no pathway onto LC10a here, so P1 itself only gets the fixed tonic
rate P1_TONIC_HZ (0). With no target only the tonic channels are on.
"""

from __future__ import annotations

import math

import numpy as np

from .interfaces import CHANNELS, N_AZIMUTH_BINS, N_CHANNELS, NO_TARGET, TargetFeatures

THETA_MAX_DEG = 40.0  # outermost bin edge on each side (the Tello sees about +-33 deg)
S_CLIP = (0.2, 5.0)  # normalized size range the size curves use
DTHETA_CLIP = 3.0  # rad/s
RATE_CAP_HZ = 400.0  # beyond the LIF refractory limit anyway
LC_CAP_HZ = 10.0  # hard cap on LC9 and LC11
P1_TONIC_HZ = 0.0  # P1 (AROUSAL) rate in "gain" mode, fixed, not trained
AROUSAL_MODE = "gain"

NB = N_AZIMUTH_BINS
CH_LC9 = 2 * NB  # LC9_L, LC9_R, LC11_L, LC11_R, AROUSAL_L, AROUSAL_R follow the 16 bins
CH_LC11 = CH_LC9 + 2
CH_AROUSAL = CH_LC9 + 4
assert CHANNELS[CH_LC9] == "LC9_L" and CHANNELS[CH_LC11] == "LC11_L" and CHANNELS[CH_AROUSAL] == "AROUSAL_L"

ENCODER_NAMES = (("enc_r_max_hz", "enc_width_deg", "enc_overlap_deg", "enc_size_width_exp", "enc_size_amp_exp",
                  "enc_vel_gain") + tuple(f"enc_bin_gain_{s}{k}" for s in "LR" for k in range(NB))
                 + ("enc_arousal", "enc_lc9_hz", "enc_lc11_hz"))


def bin_centres_deg(overlap_deg: float) -> np.ndarray:
    """Bin centres (deg, + right) in channel order L0..L7, R0..R7."""
    span = THETA_MAX_DEG + overlap_deg
    right = -overlap_deg + (np.arange(NB) + 0.5) * span / NB
    return np.concatenate([-right, right])


class TargetEncoder:
    """Stateless: channels(features) -> 22 rates (Hz). `params` holds the 25 enc_* values (kite.fly.params)."""

    def __init__(self, params: dict[str, float]):
        missing = [k for k in ENCODER_NAMES if k not in params]
        if missing:
            raise KeyError(f"encoder parameters missing: {', '.join(missing)}")
        p = {k: float(params[k]) for k in ENCODER_NAMES}
        self.params = p
        self.centres = np.radians(bin_centres_deg(p["enc_overlap_deg"]))
        self.sigma = math.radians(p["enc_width_deg"])
        self.a_width, self.a_amp, self.k_v = p["enc_size_width_exp"], p["enc_size_amp_exp"], p["enc_vel_gain"]
        gains = np.array([p[f"enc_bin_gain_{s}{k}"] for s in "LR" for k in range(NB)])
        self.amp_bins = p["enc_r_max_hz"] * gains * p["enc_arousal"]  # arousal "gain" mode: a gain on LC10a
        self.lc9, self.lc11 = p["enc_lc9_hz"], p["enc_lc11_hz"]
        self._rest = np.zeros(N_CHANNELS)
        self._rest[CH_AROUSAL:CH_AROUSAL + 2] = P1_TONIC_HZ

    def rest(self) -> np.ndarray:
        """The rates with no target (tonic channels only)."""
        return self._rest.copy()

    def channels(self, f: TargetFeatures = NO_TARGET) -> np.ndarray:
        """Features -> rates (Hz) per channel, a fresh float64 array of shape (22,)."""
        out = self._rest.copy()
        if not f.valid:
            return out
        s = min(max(float(f.s), S_CLIP[0]), S_CLIP[1])
        sigma = self.sigma * s ** self.a_width
        amp = s ** self.a_amp * (1.0 + self.k_v * min(abs(float(f.dtheta)), DTHETA_CLIP))
        z = (float(f.theta) - self.centres) / sigma
        tuning = np.exp(-0.5 * z * z)
        out[:2 * NB] = np.minimum(self.amp_bins * tuning * amp, RATE_CAP_HZ)
        side = (tuning[:NB].max() * amp, tuning[NB:].max() * amp)  # how strongly the spot falls in each side's field
        for i in range(2):
            out[CH_LC9 + i] = min(self.lc9 * side[i], LC_CAP_HZ)
            out[CH_LC11 + i] = min(self.lc11 * side[i], LC_CAP_HZ)
        return out

    def named(self, f: TargetFeatures) -> dict[str, float]:
        """Channel name -> rate, for inspection."""
        return dict(zip(CHANNELS, self.channels(f).tolist()))
