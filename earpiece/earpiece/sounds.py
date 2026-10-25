"""Synthesised cue sounds. No asset files: everything is generated at startup.

Design rules (each one traced to a source; measurements in tests/test_sounds.py):
  * Sonar cues start with a 3-4 ms attack and decay quickly: sharp onsets make
    beep rate easy to judge and carry the onset ITD. No clicks at either edge.
  * Every kind cue is harmonically or spectrally rich, with several components
    between 1 and 4 kHz (Patterson 1982, CAA Paper 82017: >= 4 components in
    1-4 kHz; Middlebrooks 1992: narrowband sounds are localised by their
    spectrum, not their position; Pec et al. 2008 via Bujacz & Strumillo 2016:
    instrument tones with a little noise localise better than clean tones).
    Partials stop at 7 kHz: above that Bluetooth AAC is least reliable and
    Zwicker sharpness (a main driver of annoyance) climbs steeply.
  * Each object class has its own timbre AND its own envelope or modulation,
    because timbre is the strongest earcon differentiator (Brewster, Wright &
    Edwards, CHI 1993) and register alone needs 2-3 octaves to be identified:
      person   -> wooden knock: inharmonic bar partials + a mallet click, short
      vehicle  -> band-limited buzz with ~60 Hz roughness, the only rough cue:
                  roughness (30-150 Hz modulation) is what makes screams and
                  alarms alarming (Arnal et al., Current Biology 2015), so it
                  is reserved for the most dangerous object
      obstacle -> clean harmonic pip, flat sustain (no decay, no roughness)
      drop     -> harmonic glide falling an octave (stairs down, curb)
  * Loudness is matched across kinds with BS.1770 K-weighting integrated over
    the cue (loudness_lu): below ~200 ms the ear integrates energy, so a short
    cue's loudness follows its K-weighted energy, not its peak. Before this the
    kinds differed by ~7 LU at equal peak (buzz loudest, knock quietest).
  * The waypoint beacon is a decaying harmonic "ping" (Walker & Lindsay 2006,
    Human Factors 48(2): the sonar-ping beacon beat pink noise and a pure tone),
    a different timbre from every sonar cue so "turn there" is never confused
    with "something is there". Its pitch and brightness rise as the wearer turns
    toward the waypoint (continuous feedback plus an on-target accent beat a
    single on-target ping: Ahmetovic et al., IJHCS 2023), and the bright,
    on-course ticks put energy in Blauert's 2.9-5.8 kHz "front" band while the
    "behind" cue keeps its energy in the 0.72-1.8 kHz "back" band.
  * Profiles: "standard" (default) and "gentle" (KITE_SOUND_PROFILE=gentle or
    set_profile("gentle")), which leaves street sound audible: -6 dB, ~40 %
    shorter cues (smaller duty cycle, gaps for traffic to be heard), and energy
    moved above 1.5 kHz, out of the tyre-noise band (~0.8-1.6 kHz) where the cue
    would mask approaching cars (upward spread of masking: Wegel & Lane 1924);
    the pitch survives as residue pitch of the remaining harmonics. Gentle
    shaping fades out with urgency and is gone at the critical pitch, so an
    imminent collision always sounds at full strength.

Engines: when the Rust extension `kite_audio` is built (and KITE_AUDIO_ENGINE is not
"python"), every public cue below is synthesised by its port in native/src/sounds.rs,
sample for sample the same as the numpy/scipy code here (tests/test_rust_sounds.py),
1.5 to 2.6x faster (libm's sin dominates both) and with the GIL released. The native
sonar thread (KITE_SONAR_THREAD=native) synthesises its beeps there directly. Inside
`with sounds.reference():` this module's own code runs instead (the parity tests, and
the fallback when the extension is missing).
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import os
import re
import sys
from typing import Callable, Iterable

import numpy as np
from scipy.signal import butter, lfilter

RATE = 48000

PROFILES = ("standard", "gentle")
_profile = os.environ.get("KITE_SOUND_PROFILE", "standard").strip().lower()
if _profile not in PROFILES:
    _profile = "standard"


def _load_native():
    """The kite_audio module if it is built with the cue synthesiser, else None."""
    if os.environ.get("KITE_AUDIO_ENGINE", "auto").strip().lower() == "python":
        return None
    try:
        import kite_audio
    except ImportError:
        return None
    return kite_audio if hasattr(kite_audio, "synth") else None


_ka = _load_native()
NATIVE = _ka is not None              # True: cues are synthesised in Rust
_reference = contextvars.ContextVar("kite_sounds_reference", default=False)
_profile_listeners: list[Callable[[str], None]] = []   # e.g. the native sonar thread's cue cache


@contextlib.contextmanager
def reference():
    """Synthesise with this module's numpy/scipy code inside the block (per thread)."""
    token = _reference.set(True)
    try:
        yield
    finally:
        _reference.reset(token)


def _native():
    return _ka if _ka is not None and not _reference.get() else None


def _f32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def _synth(name: str, *args: float) -> np.ndarray:
    return _f32(_ka.synth(name, [float(a) for a in args], _profile))

# Loudness targets (LU, BS.1770 K-weighted, one cue integrated over a 400 ms window).
SONAR_TARGET_LU = -21.0           # the mean of the old kinds at equal peak, so overall level is unchanged
BEACON_TARGET_LU = -24.0          # beacon is persistent background guidance: 3 LU under the warnings
BEHIND_TARGET_LU = -23.0          # "turn around" is a one-off instruction, a little above the ticks
GENTLE_DB = -6.0
GENTLE_HIGHPASS_HZ = 1500.0
MAX_PEAK = 0.8                    # percussive cues need crest-factor room to reach the target;
# after cue_gain (0.55) and master (0.9) that is 0.4 FS, and the mixer's limiter guards pile-ups
PARTIAL_MAX_HZ = 7000.0
# Sonar maps urgency to pitch linearly between these (SonarCfg defaults); gentle shaping uses
# the same map to stand aside as urgency rises. Kept here so sounds.py does not import config.
URGENCY_PITCH_HZ = (520.0, 1250.0)
FRONT_BAND_HZ = (2900.0, 5800.0)  # Blauert 1969/70 directional bands (front)
BACK_BAND_HZ = (720.0, 1800.0)    # ... and the low "back" band


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


# ---------------------------------------------------------------- loudness
# ITU-R BS.1770-4 K-weighting at 48 kHz: high-shelf (head) then RLB high-pass.
_K_SHELF = (np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
            np.array([1.0, -1.69065929318241, 0.73248077421585]))
_K_HPF = (np.array([1.0, -2.0, 1.0]), np.array([1.0, -1.99004745483398, 0.99007225036621]))


def loudness_lu(x: np.ndarray, rate: int = RATE) -> float:
    """BS.1770 momentary loudness of one cue: K-weighted energy over a 400 ms window.

    Integrated LUFS gates and needs >= 400 ms of audio; a 50-200 ms cue is shorter than
    one block, so its energy is integrated over one momentary window instead. Short
    sounds' loudness grows with energy up to ~200 ms (temporal integration), so this
    ranks cues of different length the way the ear does. Mono or (n, channels)."""
    if rate != RATE:
        raise ValueError("loudness_lu uses the 48 kHz K-weighting coefficients")
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    tail = np.zeros((int(0.05 * rate), y.shape[1]))      # let the filters ring out
    y = lfilter(*_K_HPF, lfilter(*_K_SHELF, np.vstack([y, tail]), axis=0), axis=0)
    ms = float(np.sum(y ** 2)) / (0.4 * rate)
    return -0.691 + 10.0 * math.log10(ms) if ms > 0 else -math.inf


def _to_loudness(x: np.ndarray, target_lu: float, peak_max: float = MAX_PEAK) -> np.ndarray:
    """Scale to target loudness, never above peak_max (then the cue lands a little quieter)."""
    lu = loudness_lu(x)
    if not math.isfinite(lu):
        return x.astype(np.float32)
    g = 10 ** ((target_lu - lu) / 20.0)
    m = float(np.max(np.abs(x)))
    if m * g > peak_max:
        g = peak_max / m
    return (x * g).astype(np.float32)


# ---------------------------------------------------------------- profiles
def profile() -> str:
    return _profile


def set_profile(name: str) -> None:
    """Switch the cue profile at runtime and drop every cached rendering of the old one."""
    global _profile
    name = name.strip().lower()
    if name not in PROFILES:
        raise ValueError(f"unknown sound profile {name!r}; expected one of {PROFILES}")
    _profile = name
    for mod_name in ("earpiece.sonar", "earpiece.beacon"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        getattr(mod, "_WARM", {}).clear()
        for fn in ("_cue_render", "_beacon_render"):
            if hasattr(getattr(mod, fn, None), "cache_clear"):
                getattr(mod, fn).cache_clear()
    for fn in list(_profile_listeners):
        fn(name)


def on_profile_change(fn: Callable[[str], None]) -> None:
    """Call fn(profile) after every set_profile (native caches that must re-render)."""
    _profile_listeners.append(fn)


def remove_profile_listener(fn: Callable[[str], None]) -> None:
    with contextlib.suppress(ValueError):
        _profile_listeners.remove(fn)


def _urgency(pitch_hz: float) -> float:
    lo, hi = URGENCY_PITCH_HZ
    return min(1.0, max(0.0, (pitch_hz - lo) / (hi - lo)))


def gentle_amount(pitch_hz: float | None = None) -> float:
    """How much gentle shaping applies: 0 in the standard profile; in gentle, 1 for calm cues,
    fading to 0 between 60 % and 100 % urgency (smoothstep) so critical cues are untouched."""
    if _profile != "gentle":
        return 0.0
    if pitch_hz is None:
        return 1.0
    u = (_urgency(pitch_hz) - 0.6) / 0.4
    u = min(1.0, max(0.0, u))
    return 1.0 - u * u * (3 - 2 * u)


def _dur(base_s: float, amount: float) -> float:
    """Gentle cues are ~40 % shorter, never under 40 ms (Brewster: >= 30 ms is still identifiable)."""
    return max(0.04, base_s * (1.0 - 0.4 * amount))


def _gentle_filter(x: np.ndarray, amount: float) -> np.ndarray:
    """Blend in a 4th-order 1.5 kHz high-pass by `amount`: clears most of the tyre-noise band."""
    if amount <= 0:
        return x
    b, a = butter(4, GENTLE_HIGHPASS_HZ, btype="highpass", fs=RATE)
    return (1 - amount) * x + amount * lfilter(b, a, x)


def _finish(x: np.ndarray, target_lu: float, amount: float, fade_s: float = 0.004) -> np.ndarray:
    """Common tail: gentle filter, click-free edges, loudness match (-6 dB in gentle)."""
    x = _gentle_filter(np.asarray(x, dtype=np.float64), amount)
    r = max(1, min(x.size // 4, int(fade_s * RATE)))
    x[-r:] *= np.linspace(1.0, 0.0, r)
    x[0] = 0.0
    return _to_loudness(x, target_lu + GENTLE_DB * amount)


# ---------------------------------------------------------------- building blocks (status cues)
def sine_pip(freq: float, dur_s: float = 0.07, peak: float = 0.5) -> np.ndarray:
    if _native():
        return _synth("sine_pip", freq, dur_s, peak)
    t = _t(dur_s)
    x = np.sin(2 * math.pi * freq * t) + 0.25 * np.sin(2 * math.pi * 2 * freq * t)
    return _norm(x * _env(t.size, decay_tau_s=dur_s * 0.6), peak)


def knock(freq: float, dur_s: float = 0.09, peak: float = 0.5) -> np.ndarray:
    """Marimba-ish: fundamental + inharmonic 3.9x / 9.2x partials that die fast."""
    if _native():
        return _synth("knock", freq, dur_s, peak)
    t = _t(dur_s)
    x = (np.sin(2 * math.pi * freq * t) * np.exp(-t / (dur_s * 0.5))
         + 0.5 * np.sin(2 * math.pi * 3.9 * freq * t) * np.exp(-t / 0.012)
         + 0.25 * np.sin(2 * math.pi * 9.2 * freq * t) * np.exp(-t / 0.005))
    return _norm(x * _env(t.size, attack_s=0.002, decay_tau_s=None), peak)


def buzz(freq: float, dur_s: float = 0.09, peak: float = 0.5) -> np.ndarray:
    """Band-limited square-ish wave with a quick upward bend: reads as 'machine, urgent'."""
    if _native():
        return _synth("buzz", freq, dur_s, peak)
    t = _t(dur_s)
    f = freq * (1.0 + 0.06 * np.minimum(t / 0.03, 1.0))
    ph = 2 * math.pi * np.cumsum(f) / RATE
    x = sum(np.sin(k * ph) / k for k in (1, 3, 5, 7))
    return _norm(x * _env(t.size, decay_tau_s=dur_s * 0.8), peak)


def glide(f0: float, f1: float, dur_s: float = 0.12, peak: float = 0.5) -> np.ndarray:
    if _native():
        return _synth("glide", f0, f1, dur_s, peak)
    t = _t(dur_s)
    f = f0 * (f1 / f0) ** (t / t[-1])
    ph = 2 * math.pi * np.cumsum(f) / RATE
    x = np.sin(ph) + 0.3 * np.sin(2 * ph)
    return _norm(x * _env(t.size, decay_tau_s=dur_s * 0.7), peak)


def _harmonics(ph: np.ndarray, f_top: float, amps: Iterable[float]) -> np.ndarray:
    """Sum of sin(k ph) * amps[k-1] for every harmonic whose frequency stays <= PARTIAL_MAX_HZ."""
    x = np.zeros_like(ph)
    for k, a in enumerate(amps, start=1):
        if k * f_top > PARTIAL_MAX_HZ:
            break
        x += a * np.sin(k * ph)
    return x


def _click(n: int, seed: int) -> np.ndarray:
    """A 4 ms band-passed (1.5-6 kHz) noise burst: the mallet contact of a knock. Seeded, so
    every rendering of a cue is bit-identical."""
    rng = np.random.default_rng(seed)
    m = int(0.004 * RATE)
    b, a = butter(2, (1500.0, 6000.0), btype="bandpass", fs=RATE)
    burst = lfilter(b, a, rng.standard_normal(m)) * np.hanning(m)
    out = np.zeros(n)
    out[:min(n, m)] = burst[:n]
    return out


# ---------------------------------------------------------------- sonar kinds
def person_cue(freq: float) -> np.ndarray:
    """Wooden knock: bar partials 1 : 2.76 : 3.9 (inharmonic, decaying faster the higher they
    are) + a mallet click. Short, percussive and soft-edged: the least alarming cue."""
    if _native():
        return _synth("person", freq)
    amount = gentle_amount(freq)
    f = freq * 0.8
    t = _t(_dur(0.08, amount))
    x = np.sin(2 * math.pi * f * t) * np.exp(-t / 0.045)
    for ratio, amp, tau in ((2.76, 0.5, 0.025), (3.9, 0.35, 0.015), (5.4, 0.25, 0.01)):
        if ratio * f <= PARTIAL_MAX_HZ:
            x += amp * np.sin(2 * math.pi * ratio * f * t) * np.exp(-t / tau)
    x += 0.25 * _click(t.size, seed=int(f))
    x *= _env(t.size, attack_s=0.002, decay_tau_s=None, release_s=0.008)
    return _finish(x, SONAR_TARGET_LU, amount)


def vehicle_cue(freq: float) -> np.ndarray:
    """Machine buzz: band-limited sawtooth (all harmonics, 1/k) with a quick upward bend and
    60 Hz amplitude modulation (roughness = 'alarm'). Sustained, so it reads as engine."""
    if _native():
        return _synth("vehicle", freq)
    amount = gentle_amount(freq)
    f0 = freq * 0.75
    t = _t(_dur(0.09, amount))
    f = f0 * (1.0 + 0.06 * np.minimum(t / 0.03, 1.0))
    ph = 2 * math.pi * np.cumsum(f) / RATE
    x = _harmonics(ph, f0 * 1.06, [1.0 / k for k in range(1, 40)])
    x *= 1.0 - 0.45 * (0.5 + 0.5 * np.cos(2 * math.pi * 60.0 * t))
    x *= _env(t.size, attack_s=0.003, decay_tau_s=t[-1] * 1.2, release_s=0.01)
    return _finish(x, SONAR_TARGET_LU, amount)


def obstacle_cue(freq: float) -> np.ndarray:
    """Clean pip: harmonics 1-6 falling ~4.5 dB/harmonic, flat sustain, no modulation. The
    steady envelope (vs. the knock's fast decay) is what tells pole from person."""
    if _native():
        return _synth("obstacle", freq)
    amount = gentle_amount(freq)
    t = _t(_dur(0.07, amount))
    ph = 2 * math.pi * freq * t
    x = _harmonics(ph, freq, [k ** -1.5 for k in range(1, 7)])
    x *= _env(t.size, attack_s=0.004, decay_tau_s=None, release_s=0.015)
    return _finish(x, SONAR_TARGET_LU, amount)


def drop_cue(freq: float) -> np.ndarray:
    """Falling glide, 1.2f -> 0.6f (one octave down): 'the ground drops away'. Harmonics 1-4."""
    if _native():
        return _synth("drop", freq)
    amount = gentle_amount(freq)
    t = _t(_dur(0.12, amount))
    f0, f1 = freq * 1.2, freq * 0.6
    f = f0 * (f1 / f0) ** (t / t[-1])
    ph = 2 * math.pi * np.cumsum(f) / RATE
    x = _harmonics(ph, f0, [1.0, 0.5, 0.33, 0.25])
    x *= _env(t.size, attack_s=0.004, decay_tau_s=t[-1] * 0.8, release_s=0.012)
    return _finish(x, SONAR_TARGET_LU, amount)


PERSON_KINDS = ("person", "people", "pedestrian", "crowd")
VEHICLE_KINDS = ("vehicle", "car", "bike", "bicycle", "scooter", "motorcycle", "bus", "truck")
DROP_KINDS = ("stairs_down", "curb", "drop", "stairs", "step")


def proximity_cue(kind: str, freq: float) -> np.ndarray:
    """The repeated sonar pip for one object class at a given pitch (urgency)."""
    if (ka := _native()) is not None:
        return _f32(ka.synth_proximity(kind, float(freq), _profile))
    if kind in PERSON_KINDS:
        return person_cue(freq)
    if kind in VEHICLE_KINDS:
        return vehicle_cue(freq)
    if kind in DROP_KINDS:
        return drop_cue(freq)
    return obstacle_cue(freq)


def _seq(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate([p.astype(np.float32) for p in parts])


def _gap(s: float) -> np.ndarray:
    return np.zeros(int(s * RATE), dtype=np.float32)


# ---------------------------------------------------------------- beacon
# Every beacon cue is told apart by pitch contour, rhythm and timbre alone, so they survive
# mono playback (table speakers, one earbud) where left/right panning collapses; the
# spatialisation on top is a redundant cue. Turn pairs step a perfect fourth:
#   on      same-pitch double ping, bright        ("keep going")
#   right   rising pair, low -> high              (bearing > 0)
#   left    falling pair, high -> low             (bearing < 0)
#   behind  three low, dull, same-pitch pulses    (back band, no bright partials)
BEACON_PITCH_HZ = {"on": 2093.0, "near": 1175.0, "side": 880.0, "behind": 784.0}
TURN_INTERVAL = 4.0 / 3.0         # a perfect fourth: an unmistakable step that stays in one register
BRIGHT_SPAN_DEG = 55.0            # brightness falls from 1 on axis to 0 here (Soundscape's 'near' edge)


def beacon_brightness(bearing_deg: float, span_deg: float = BRIGHT_SPAN_DEG) -> float:
    """Raised-cosine on-axis brightness: 1 straight ahead, 0 at +-span and beyond.
    Flat at the top, so compass jitter of a few degrees while on course does not make
    the tick flicker; steepest at span/2, where the turn correction is being made."""
    a = min(abs(float(bearing_deg)), span_deg)
    return 0.5 * (1.0 + math.cos(math.pi * a / span_deg))


def beacon_ping(freq: float, brightness: float, dur_s: float = 0.11) -> np.ndarray:
    """One beacon ping, un-normalised: harmonics with a spectral slope set by brightness
    (k^-2.6 dull .. k^-0.8 bright) and a ringing exponential decay."""
    t = _t(dur_s)
    slope = 2.6 - 1.8 * min(1.0, max(0.0, brightness))
    x = _harmonics(2 * math.pi * freq * t, freq, [k ** -slope for k in range(1, 8)])
    return x * _env(t.size, attack_s=0.004, decay_tau_s=0.035, release_s=0.012)


def _notes(notes: list[np.ndarray], gap_s: float, target_lu: float, amount: float,
           fade_s: float = 0.004) -> np.ndarray:
    """Notes matched to each other, then the whole cue matched to target_lu (integrated)."""
    parts = [_finish(n, target_lu, amount, fade_s) for n in notes]
    seq = _seq(*[x for p in parts for x in (p, _gap(gap_s))][:-1])
    return _to_loudness(seq, target_lu + GENTLE_DB * amount)


def turn_pitches(f_low: float, bearing_deg: float) -> tuple[float, float]:
    """Rising pair for a right turn (bearing > 0), falling pair for a left turn."""
    hi = f_low * TURN_INTERVAL
    return (f_low, hi) if bearing_deg >= 0 else (hi, f_low)


def beacon_tick(region: str, bearing_deg: float = 0.0) -> np.ndarray:
    """The beacon cue for a Soundscape region ('on' | 'near' | 'side' | 'behind') at a bearing
    (positive = right). 'on' is a same-pitch double ping (an on-target accent on top of the
    continuous brightness, as in Ahmetovic et al.'s IS+P / MS+P); 'near' and 'side' are a
    rising pair when the waypoint is to the right and a falling pair when it is to the left;
    'behind' is the turn-around triple pulse."""
    if (ka := _native()) is not None:
        if region not in BEACON_PITCH_HZ:
            raise KeyError(region)
        return _f32(ka.synth_beacon(region, float(bearing_deg), _profile))
    if region == "behind":
        return beacon_behind()
    amount = gentle_amount(None)
    b = beacon_brightness(bearing_deg) if region != "side" else 0.0
    f = BEACON_PITCH_HZ[region]
    f1, f2 = (f, f) if region == "on" else turn_pitches(f, bearing_deg)
    notes = [beacon_ping(f1, b, _dur(0.07, amount)), beacon_ping(f2, b, _dur(0.09, amount))]
    return _notes(notes, 0.03, BEACON_TARGET_LU, amount)


def beacon_on_course() -> np.ndarray:
    """Bright same-pitch double ping: 'you are pointed at the waypoint'."""
    return beacon_tick("on", 0.0)


def beacon_off_course(bearing_deg: float = 35.0) -> np.ndarray:
    """Rising pair (waypoint to the right) or falling pair (to the left): 'turn toward me'."""
    return beacon_tick("near" if abs(bearing_deg) <= BRIGHT_SPAN_DEG else "side", bearing_deg)


def beacon_behind() -> np.ndarray:
    """'The waypoint is behind you, turn around': three low, same-pitch pulses (G5) whose
    energy sits in Blauert's 0.72-1.8 kHz back band, with no bright partials. Three beats and
    a dull timbre set it apart from the bright double 'on' ping and from both turn contours,
    in mono and even when front and back are confused."""
    if _native():
        return _synth("beacon_behind")
    lp = butter(2, 1800.0, btype="lowpass", fs=RATE)
    f = BEACON_PITCH_HZ["behind"]
    notes = []
    for _ in range(3):
        t = _t(0.075)
        x = _harmonics(2 * math.pi * f * t, f, [1.0, 0.6, 0.15])
        notes.append(lfilter(*lp, x * _env(t.size, attack_s=0.008, decay_tau_s=0.06, release_s=0.02)))
    return _notes(notes, 0.06, BEHIND_TARGET_LU, 0.0, fade_s=0.01)


# ---------------------------------------------------------------- status
def chime(*freqs: float, step_s: float = 0.09, peak: float = 0.42) -> np.ndarray:
    if _native() and freqs:
        return _synth("chime", step_s, peak, *freqs)
    parts = []
    for i, f in enumerate(freqs):
        parts.append(knock(f, 0.16 if i == len(freqs) - 1 else step_s, peak))
    return _seq(*parts)


def arrived() -> np.ndarray:
    if _native():
        return _synth("arrived")
    return chime(784, 988, 1175, 1568)


def ready() -> np.ndarray:
    if _native():
        return _synth("ready")
    return chime(659, 988)


def listening() -> np.ndarray:
    """Rising chirp: 'mic is live, speak now'."""
    if _native():
        return _synth("listening")
    return glide(600, 1300, 0.16, 0.4)


def heard() -> np.ndarray:
    """Short same-pitch double blip: 'got it'. Level, not rising: A5 -> D6 rising pips were
    the right-turn beacon's exact notes and contour (the dull side ping is nearly a sine)."""
    if _native():
        return _synth("heard")
    return _seq(sine_pip(1319, 0.05, 0.38), _gap(0.035), sine_pip(1319, 0.06, 0.38))


def error() -> np.ndarray:
    """Low soft buzz: 'did not catch that' / 'not available'."""
    if _native():
        return _synth("error")
    return buzz(196, 0.22, 0.4)


def sensor_lost() -> np.ndarray:
    """Descending minor third, twice: 'obstacle detection is down'."""
    if _native():
        return _synth("sensor_lost")
    return _seq(sine_pip(784, 0.1, 0.45), _gap(0.03), sine_pip(659, 0.14, 0.45),
                _gap(0.12), sine_pip(784, 0.1, 0.45), _gap(0.03), sine_pip(659, 0.14, 0.45))


def sensor_back() -> np.ndarray:
    if _native():
        return _synth("sensor_back")
    return _seq(sine_pip(659, 0.08, 0.42), _gap(0.03), sine_pip(784, 0.12, 0.42))


def critical_alarm() -> np.ndarray:
    """Three fast high buzzes. Only for imminent collision."""
    if _native():
        return _synth("critical_alarm")
    b = buzz(1320, 0.06, 0.6)
    return _seq(b, _gap(0.035), b, _gap(0.035), b)


def crossing_go() -> np.ndarray:
    """Walk signal on: steady rapid ticks like an accessible pedestrian signal."""
    if _native():
        return _synth("crossing_go")
    return _seq(*[x for _ in range(4) for x in (knock(2000, 0.03, 0.4), _gap(0.07))])


def crossing_wait() -> np.ndarray:
    """Don't walk: slow low locator tone."""
    if _native():
        return _synth("crossing_wait")
    return _seq(sine_pip(440, 0.05, 0.4), _gap(0.4), sine_pip(440, 0.05, 0.4))


def ear_check(side: str) -> np.ndarray:
    if _native():
        return _synth("ear_check_left" if side == "left" else "ear_check_right")
    return knock(880 if side == "left" else 1320, 0.12, 0.5)


def turn_side(text: str) -> int:
    """+1 if an instruction's first left/right word is 'right', -1 if 'left', 0 if neither."""
    m = re.search(r"\b(left|right)\b", text or "", re.IGNORECASE)
    return 0 if m is None else (1 if m.group(1).lower() == "right" else -1)


def nav_chime(text: str = "") -> np.ndarray:
    """Soft lead-in before a navigation instruction: 'directions coming'. It follows the beacon
    contour so it never contradicts the words: a rising pair before a right turn, a falling
    pair before a left, and a single knock when the instruction names no side."""
    side = turn_side(text)
    if _native():
        return _synth("nav_chime", side)
    if side == 0:
        return chime(1175, peak=0.32)
    return chime(*turn_pitches(988.0, side), step_s=0.07, peak=0.32)
