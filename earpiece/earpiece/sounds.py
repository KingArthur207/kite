"""Synthesised cue sounds. No asset files: everything is generated at startup.

Design rules (from sonification practice for travel aids):
  * every cue starts with a 2-4 ms attack and decays exponentially: no clicks,
    and sharp onsets are what make rhythm (beep rate) easy to judge;
  * harmonically rich timbres localise better than pure tones (more
    high-frequency content for the head-shadow cue), but stay under ~6 kHz
    where Bluetooth AAC is clean;
  * each object class has its own timbre so "what" is heard without speech:
      person   -> soft marimba-like knock
      vehicle  -> rough, buzzy two-tone (urgent by design)
      obstacle -> clean sine pip
      drop     -> falling glide (stairs down, curb)
"""

from __future__ import annotations

import math

import numpy as np

RATE = 48000


def _t(dur_s: float, rate: int = RATE) -> np.ndarray:
    return np.arange(int(round(dur_s * rate)), dtype=np.float64) / rate


def _env(n: int, attack_s: float = 0.003, decay_tau_s: float | None = 0.05,
         release_s: float = 0.006, rate: int = RATE) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / rate
    e = np.ones(n)
    a = max(1, int(attack_s * rate))
    e[:a] = np.linspace(0.0, 1.0, a) ** 2
    if decay_tau_s:
        e *= np.exp(-np.maximum(t - attack_s, 0.0) / decay_tau_s)
    r = max(1, min(n // 2, int(release_s * rate)))
    e[-r:] *= np.linspace(1.0, 0.0, r)
    return e


def _norm(x: np.ndarray, peak: float) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x * (peak / m if m > 0 else 0.0)).astype(np.float32)


def sine_pip(freq: float, dur_s: float = 0.07, peak: float = 0.5) -> np.ndarray:
    t = _t(dur_s)
    x = np.sin(2 * math.pi * freq * t) + 0.25 * np.sin(2 * math.pi * 2 * freq * t)
    return _norm(x * _env(t.size, decay_tau_s=dur_s * 0.6), peak)


def knock(freq: float, dur_s: float = 0.09, peak: float = 0.5) -> np.ndarray:
    """Marimba-ish: fundamental + inharmonic 3.9x / 9.2x partials that die fast."""
    t = _t(dur_s)
    x = (np.sin(2 * math.pi * freq * t) * np.exp(-t / (dur_s * 0.5))
         + 0.5 * np.sin(2 * math.pi * 3.9 * freq * t) * np.exp(-t / 0.012)
         + 0.25 * np.sin(2 * math.pi * 9.2 * freq * t) * np.exp(-t / 0.005))
    return _norm(x * _env(t.size, attack_s=0.002, decay_tau_s=None), peak)


def buzz(freq: float, dur_s: float = 0.09, peak: float = 0.5) -> np.ndarray:
    """Band-limited square-ish wave with a quick upward bend: reads as 'machine, urgent'."""
    t = _t(dur_s)
    f = freq * (1.0 + 0.06 * np.minimum(t / 0.03, 1.0))
    ph = 2 * math.pi * np.cumsum(f) / RATE
    x = sum(np.sin(k * ph) / k for k in (1, 3, 5, 7))
    return _norm(x * _env(t.size, decay_tau_s=dur_s * 0.8), peak)


def glide(f0: float, f1: float, dur_s: float = 0.12, peak: float = 0.5) -> np.ndarray:
    t = _t(dur_s)
    f = f0 * (f1 / f0) ** (t / t[-1])
    ph = 2 * math.pi * np.cumsum(f) / RATE
    x = np.sin(ph) + 0.3 * np.sin(2 * ph)
    return _norm(x * _env(t.size, decay_tau_s=dur_s * 0.7), peak)


def proximity_cue(kind: str, freq: float) -> np.ndarray:
    """The repeated sonar pip for one object class at a given pitch."""
    if kind in ("person", "people", "pedestrian", "crowd"):
        return knock(freq * 0.8)
    if kind in ("vehicle", "car", "bike", "bicycle", "scooter", "motorcycle", "bus", "truck"):
        return buzz(freq * 0.75)
    if kind in ("stairs_down", "curb", "drop", "stairs", "step"):
        return glide(freq * 1.2, freq * 0.6)
    return sine_pip(freq)


def _seq(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate([p.astype(np.float32) for p in parts])


def _gap(s: float) -> np.ndarray:
    return np.zeros(int(s * RATE), dtype=np.float32)


# ---------------------------------------------------------------- beacon
def beacon_on_course() -> np.ndarray:
    """Bright double tick: 'you are pointed at the waypoint'."""
    return _seq(knock(1568, 0.06, 0.45), _gap(0.05), knock(2093, 0.07, 0.45))


def beacon_off_course() -> np.ndarray:
    """Single soft tick, spatialised toward the waypoint: 'turn toward me'."""
    return knock(1175, 0.07, 0.4)


def beacon_behind() -> np.ndarray:
    """Low two-tone: 'the waypoint is behind you, turn around'."""
    return _seq(sine_pip(392, 0.09, 0.45), _gap(0.04), sine_pip(330, 0.11, 0.45))


# ---------------------------------------------------------------- status
def chime(*freqs: float, step_s: float = 0.09, peak: float = 0.42) -> np.ndarray:
    parts = []
    for i, f in enumerate(freqs):
        parts.append(knock(f, 0.16 if i == len(freqs) - 1 else step_s, peak))
    return _seq(*parts)


def arrived() -> np.ndarray:
    return chime(784, 988, 1175, 1568)


def ready() -> np.ndarray:
    return chime(659, 988)


def listening() -> np.ndarray:
    """Rising chirp: 'mic is live, speak now'."""
    return glide(600, 1300, 0.16, 0.4)


def heard() -> np.ndarray:
    """Short double blip: 'got it'."""
    return _seq(sine_pip(880, 0.05, 0.38), _gap(0.035), sine_pip(1175, 0.06, 0.38))


def error() -> np.ndarray:
    """Low soft buzz: 'did not catch that' / 'not available'."""
    return buzz(196, 0.22, 0.4)


def sensor_lost() -> np.ndarray:
    """Descending minor third, twice: 'obstacle detection is down'."""
    return _seq(sine_pip(784, 0.1, 0.45), _gap(0.03), sine_pip(659, 0.14, 0.45),
                _gap(0.12), sine_pip(784, 0.1, 0.45), _gap(0.03), sine_pip(659, 0.14, 0.45))


def sensor_back() -> np.ndarray:
    return _seq(sine_pip(659, 0.08, 0.42), _gap(0.03), sine_pip(784, 0.12, 0.42))


def critical_alarm() -> np.ndarray:
    """Three fast high buzzes. Only for imminent collision."""
    b = buzz(1320, 0.06, 0.6)
    return _seq(b, _gap(0.035), b, _gap(0.035), b)


def crossing_go() -> np.ndarray:
    """Walk signal on: steady rapid ticks like an accessible pedestrian signal."""
    return _seq(*[x for _ in range(4) for x in (knock(2000, 0.03, 0.4), _gap(0.07))])


def crossing_wait() -> np.ndarray:
    """Don't walk: slow low locator tone."""
    return _seq(sine_pip(440, 0.05, 0.4), _gap(0.4), sine_pip(440, 0.05, 0.4))


def ear_check(side: str) -> np.ndarray:
    return knock(880 if side == "left" else 1320, 0.12, 0.5)


def nav_chime() -> np.ndarray:
    """Soft two-note lead-in before a navigation instruction: 'directions coming'."""
    return chime(988, 1319, step_s=0.07, peak=0.32)
