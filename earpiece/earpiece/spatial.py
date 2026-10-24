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

Engines: when the Rust extension `kite_audio` is built (and KITE_AUDIO_ENGINE is
not "python"), `render` runs its native port. It reproduces `_render_py` (the
numpy/scipy reference kept here, and the fallback) sample for sample, about 4x
faster per cue (10x+ for `render_many`) and without holding the GIL.
`sinc_taps` selects an opt-in windowed-sinc fractional delay in both engines.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.signal import lfilter


def _load_native():
    """The kite_audio module if it is built and has the binaural renderer, else None."""
    if os.environ.get("KITE_AUDIO_ENGINE", "auto").strip().lower() == "python":
        return None
    try:
        import kite_audio
    except ImportError:
        return None
    return kite_audio if hasattr(kite_audio, "render_binaural") else None


_ka = _load_native()
NATIVE = _ka is not None              # True: render() places cues in Rust

HEAD_RADIUS_M = 0.0875
SPEED_OF_SOUND = 343.0
ALPHA_MIN = 0.1
THETA_MIN_DEG = 150.0
SINC_TAPS_MAX = 64
_KAISER_BETA = 6.0                    # ~-60 dB sidelobes, flat to ~0.4 fs at 16 taps
BEARINGS_5DEG = tuple(range(-180, 185, 5))   # every bearing the 5-degree cue caches can ask for


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


def _check_taps(sinc_taps: int) -> int:
    t = int(sinc_taps)
    if t != 0 and (t % 2 or not 2 <= t <= SINC_TAPS_MAX):
        raise ValueError(f"sinc_taps must be 0 (linear) or an even number in 2..={SINC_TAPS_MAX}, got {t}")
    return t


@lru_cache(maxsize=4096)
def _sinc_weights(frac: float, taps: int) -> np.ndarray:
    """Kaiser-windowed sinc for a delay of taps/2 + frac samples (frac in [0, 1)).
    w[j] multiplies x[n - whole - j], j = 0..taps; unit DC gain; integer delay = pure shift."""
    half = taps // 2
    t = np.arange(taps + 1, dtype=np.float64) - half - frac
    inside = np.abs(t) < half
    u = np.where(inside, t / half, 0.0)
    win = np.i0(_KAISER_BETA * np.sqrt(np.maximum(0.0, 1.0 - u * u))) / np.i0(_KAISER_BETA)
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(t == 0.0, 1.0, np.where(t == np.round(t), 0.0, np.sin(np.pi * t) / (np.pi * t)))
    w = np.where(inside, sinc * win, 0.0)
    total = w.sum()
    return w / total if total != 0 else w


def _frac_delay_sinc(x: np.ndarray, delay_samples: float, out_len: int, taps: int) -> np.ndarray:
    """Band-limited fractional delay (adds taps/2 samples of latency), float64, padded to out_len."""
    out = np.zeros(out_len, dtype=np.float64)
    whole = int(math.floor(delay_samples))
    if x.size == 0:
        return out
    y = np.convolve(x.astype(np.float64), _sinc_weights(delay_samples - whole, taps))
    m = min(y.size, out_len - whole)
    if m > 0:
        out[whole:whole + m] = y[:m]
    return out


def out_frames(n: int, rate: int, sinc_taps: int = 0) -> int:
    """Length of a rendered n-sample cue. Linear keeps Python's historical 2a/c pad (the
    far ear at +-90 deg loses up to ~5 trailing samples, always silence after a release);
    the sinc path reserves the full far-ear delay plus the filter length."""
    if sinc_taps:
        return n + int(math.ceil(ear_delay_s(180.0) * rate)) + 2 + sinc_taps
    return n + int(math.ceil(2 * HEAD_RADIUS_M / SPEED_OF_SOUND * rate)) + 2


def render(mono: np.ndarray, azimuth_deg: float, rate: int, gain: float = 1.0,
           extra_ild_db: float = 6.0, sinc_taps: int = 0) -> np.ndarray:
    """Mono float32 -> stereo (n + ~30 samples, 2) float32 placed at azimuth_deg.

    Total energy is normalised to that of a source dead ahead, so a cue does not
    get louder just because it moved to the side. extra_ild_db adds a
    constant-power level difference of up to that many dB (at +-90 deg) on top
    of the physical model: the spherical head alone gives only ~5 dB for the
    sub-1.5 kHz tones that dominate short pips, which is too subtle in earbuds.

    sinc_taps = 0 (default) delays each ear by linear interpolation, exactly as
    before; an even 2..64 uses a Kaiser-windowed sinc instead (flat top octave
    at every azimuth, sub-sample ITD, taps/2 samples of extra latency).

    Runs in Rust when kite_audio is built (the input is placed as float32).
    """
    if _ka is not None and np.ndim(mono) == 1:
        return _render_native(mono, azimuth_deg, rate, gain, extra_ild_db, sinc_taps)
    return _render_py(mono, azimuth_deg, rate, gain, extra_ild_db, sinc_taps)


def _render_native(mono: np.ndarray, azimuth_deg: float, rate: int, gain: float = 1.0,
                   extra_ild_db: float = 6.0, sinc_taps: int = 0) -> np.ndarray:
    raw = _ka.render_binaural(np.ascontiguousarray(mono, dtype="<f4").tobytes(), float(azimuth_deg),
                              int(rate), float(gain), float(extra_ild_db), int(sinc_taps))
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2).astype(np.float32)


def render_many(mono: np.ndarray, azimuths: Iterable[float], rate: int, gain: float = 1.0,
                extra_ild_db: float = 6.0, sinc_taps: int = 0) -> list[np.ndarray]:
    """One cue at many azimuths (cache pre-warming): one native call across threads when
    kite_audio is built, each result identical to render() at that azimuth."""
    azs = [float(a) for a in azimuths]
    if _ka is not None and np.ndim(mono) == 1 and hasattr(_ka, "render_binaural_batch"):
        raws = _ka.render_binaural_batch(np.ascontiguousarray(mono, dtype="<f4").tobytes(), azs,
                                         int(rate), float(gain), float(extra_ild_db), int(sinc_taps))
        return [np.frombuffer(r, dtype="<f4").reshape(-1, 2).astype(np.float32) for r in raws]
    return [render(mono, a, rate, gain, extra_ild_db, sinc_taps) for a in azs]


def _render_py(mono: np.ndarray, azimuth_deg: float, rate: int, gain: float = 1.0,
               extra_ild_db: float = 6.0, sinc_taps: int = 0) -> np.ndarray:
    """The numpy/scipy reference implementation of render() (and the fallback engine)."""
    taps = _check_taps(sinc_taps)
    az = wrap_deg(float(azimuth_deg))
    out_len = out_frames(mono.size, rate, taps)
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
        if taps:
            chans.append(_frac_delay_sinc(shaded, ear_delay_s(th) * rate, out_len, taps))
        else:
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
