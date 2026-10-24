"""Binaural rendering with the Brown & Duda (1998) spherical-head model.

No HRTF dataset needed: each ear gets
  * an interaural time difference from the Woodworth/Brown-Duda delay formula, and
  * a head-shadow filter H(s) = (1 + a(th) s / 2w0) / (1 + s / 2w0), w0 = c / r,
    whose high-frequency gain a(th) falls from 2 (+6 dB, ear facing the source)
    to 0.1 (-20 dB, ear in the head's shadow).
Front/back confusion is the model's known weakness, so sources behind the
wearer also get a gentle low-pass (the pinna dulls sound from behind), and the
cue layer uses distinct sounds for "behind" rather than relying on this alone.

Reference: C. P. Brown and R. O. Duda, "A Structural Model for Binaural Sound
Synthesis", IEEE Trans. Speech and Audio Processing 6(5), 1998.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.signal import lfilter

HEAD_RADIUS_M = 0.0875
SPEED_OF_SOUND = 343.0
ALPHA_MIN = 0.1
THETA_MIN_DEG = 150.0


def wrap_deg(a: float) -> float:
    """Any angle -> (-180, 180]."""
    a = math.fmod(a, 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a


def _ear_angle(azimuth_deg: float, ear: int) -> float:
    """Angle in degrees [0, 180] between the source direction and an ear's axis.
    ear = +1 right (axis at +90 deg), -1 left (axis at -90 deg)."""
    return abs(wrap_deg(azimuth_deg - 90.0 * ear))


def head_shadow_alpha(ear_angle_deg: float) -> float:
    """Brown-Duda a(th): 2.0 facing the ear, ~0.1 near 150 deg, back up slightly at 180."""
    return (1.0 + ALPHA_MIN / 2.0) + (1.0 - ALPHA_MIN / 2.0) * math.cos(
        math.radians(ear_angle_deg) / math.radians(THETA_MIN_DEG) * math.pi)


def ear_delay_s(ear_angle_deg: float) -> float:
    """Brown-Duda per-ear delay (shifted so the earliest possible arrival is 0)."""
    th = math.radians(ear_angle_deg)
    a_c = HEAD_RADIUS_M / SPEED_OF_SOUND
    if th < math.pi / 2:
        t = -a_c * math.cos(th)
    else:
        t = a_c * (th - math.pi / 2)
    return t + a_c


@lru_cache(maxsize=4096)
def _shadow_coeffs(alpha: float, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear-transform coefficients of H(s) = (beta + alpha s) / (beta + s), beta = 2 c / r."""
    beta = 2.0 * SPEED_OF_SOUND / HEAD_RADIUS_M
    k = 2.0 * rate
    b = np.array([(beta + alpha * k), (beta - alpha * k)]) / (beta + k)
    a = np.array([1.0, (beta - k) / (beta + k)])
    return b, a


@lru_cache(maxsize=64)
def _behind_lowpass(rate: int) -> tuple[np.ndarray, np.ndarray]:
    """One-pole low-pass at ~2.5 kHz, the extra 'behind you' dulling."""
    fc = 2500.0
    x = math.exp(-2.0 * math.pi * fc / rate)
    return np.array([1.0 - x]), np.array([1.0, -x])


def _frac_delay(x: np.ndarray, delay_samples: float, out_len: int) -> np.ndarray:
    """Delay by a fractional number of samples (linear interpolation), padded to out_len."""
    out = np.zeros(out_len, dtype=np.float32)
    whole = int(math.floor(delay_samples))
    frac = delay_samples - whole
    n = min(x.size, out_len - whole - 1)
    if n <= 0:
        return out
    out[whole:whole + n] += (1.0 - frac) * x[:n]
    out[whole + 1:whole + 1 + n] += frac * x[:n]
    return out


def render(mono: np.ndarray, azimuth_deg: float, rate: int, gain: float = 1.0,
           extra_ild_db: float = 6.0) -> np.ndarray:
    """Mono float32 -> stereo (n + ~30 samples, 2) float32 placed at azimuth_deg.

    Total energy is normalised to that of a source dead ahead, so a cue does not
    get louder just because it moved to the side. extra_ild_db adds a
    constant-power level difference of up to that many dB (at +-90 deg) on top
    of the physical model: the spherical head alone gives only ~5 dB for the
    sub-1.5 kHz tones that dominate short pips, which is too subtle in earbuds.
    """
    az = wrap_deg(float(azimuth_deg))
    pad = int(math.ceil(2 * HEAD_RADIUS_M / SPEED_OF_SOUND * rate)) + 2
    out_len = mono.size + pad
    src = mono.astype(np.float64)
    if abs(az) > 90.0:
        b, a = _behind_lowpass(rate)
        # fade the dulling in from 90 to 135 deg so it does not jump at the boundary
        w = min(1.0, (abs(az) - 90.0) / 45.0)
        src = (1 - w) * src + w * lfilter(b, a, src) * 0.85
    chans = []
    for ear in (-1, +1):
        th = _ear_angle(az, ear)
        b, a = _shadow_coeffs(round(head_shadow_alpha(th), 3), rate)
        shaded = lfilter(b, a, src).astype(np.float32)
        chans.append(_frac_delay(shaded, ear_delay_s(th) * rate, out_len))
    st = np.stack(chans, axis=1).astype(np.float64)
    if extra_ild_db:
        d = extra_ild_db * math.sin(math.radians(az)) / 20.0   # -1 left .. +1 right
        st[:, 0] *= 10 ** (-d / 2)
        st[:, 1] *= 10 ** (d / 2)
    e_in = float(np.sum(mono.astype(np.float64) ** 2))
    e_out = float(np.sum(st ** 2))
    if e_out > 0 and e_in > 0:
        st *= math.sqrt(2.0 * e_in / e_out)
    return (st * gain).astype(np.float32)


def distance_gain(distance_m: float | None, ref_m: float = 1.0, floor: float = 0.35) -> float:
    """Gentle loudness roll-off with distance. Distance is mostly carried by beep rate
    and pitch; loudness only reinforces it, and never drops a warning below `floor`."""
    if distance_m is None or not math.isfinite(distance_m):
        return 1.0
    return float(max(floor, min(1.0, (ref_m / max(distance_m, 0.1)) ** 0.5)))


def interaural_level_db(stereo: np.ndarray) -> float:
    """Right-minus-left RMS level in dB (positive = louder on the right). Test helper."""
    l = float(np.sqrt(np.mean(stereo[:, 0].astype(np.float64) ** 2)) + 1e-12)
    r = float(np.sqrt(np.mean(stereo[:, 1].astype(np.float64) ** 2)) + 1e-12)
    return 20.0 * math.log10(r / l)


def interaural_delay_s(stereo: np.ndarray, rate: int) -> float:
    """Right-minus-left arrival time from cross-correlation (positive = left ear first).
    Test helper."""
    l = stereo[:, 0].astype(np.float64)
    r = stereo[:, 1].astype(np.float64)
    n = 1 << int(math.ceil(math.log2(2 * l.size)))
    xc = np.fft.irfft(np.fft.rfft(r, n) * np.conj(np.fft.rfft(l, n)), n)
    lag = int(np.argmax(np.concatenate([xc[-l.size + 1:], xc[:l.size]]))) - (l.size - 1)
    return lag / rate
