"""A standalone three-tone park-assist beeper. Renders WAV files only: it never opens an audio device.

    python -m perception.near.beeper --demo out/beeps      # p1.wav, p2.wav, p3.wav, approach.wav
    python -m perception.near --source clip.mp4 --wav out/clip_beeps.wav    # the beeps a run would have made

Three priorities, three pitches a fifth apart (each tone is 1.5x the one below, so they never blur into one):

    P3  E5  659 Hz   soft sine         every 1.10 s at 10 m  ->  0.70 s at 5 m
    P2  B5  988 Hz   sine + octave     every 0.55 s at 5 m   ->  0.25 s at 2 m
    P1  F#6 1480 Hz  bright, odd harm. every 0.20 s at 2 m   ->  a continuous tone at 0.7 m, or when ttc < 1 s

Like a car's parking sensor the rate rises inside each band as the distance shrinks (geometric interpolation),
so "P2 and getting closer" is audible without a new tone. Beeps are panned by bearing (constant power), with a
4 ms raised-cosine attack and release so they do not click.
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np

RATE = 22050
TONES = {1: 1479.98, 2: 987.77, 3: 659.26}
GAIN = {1: 0.9, 2: 0.65, 3: 0.42}
BEEP_S = {1: 0.06, 2: 0.07, 3: 0.09}
BAND = {1: (1.2, 0.5, 0.20, 0.0), 2: (3.0, 1.2, 0.55, 0.25), 3: (6.0, 3.0, 1.10, 0.70)}   # far, near, slow, fast
CONTINUOUS_TTC = 1.0


def interval(priority: int, distance: float | None, ttc: float | None = None) -> float:
    """Seconds between beep onsets (0 = continuous) for a priority at a distance."""
    far, near, slow, fast = BAND[priority]
    if priority == 1 and ((ttc is not None and ttc < CONTINUOUS_TTC) or (distance is not None and distance <= near)):
        return 0.0
    if distance is None:
        return slow
    f = min(1.0, max(0.0, (far - distance) / (far - near)))
    if fast <= 0:
        fast = BEEP_S[priority] * 1.6         # the last step before continuous
    return slow * (fast / slow) ** f


def tone(priority: int, dur: float, rate: int = RATE) -> np.ndarray:
    n = max(1, int(dur * rate))
    t = np.arange(n) / rate
    f = TONES[priority]
    if priority == 1:
        y = np.sin(2 * np.pi * f * t) + np.sin(2 * np.pi * 3 * f * t) / 3 + np.sin(2 * np.pi * 5 * f * t) / 6
        y /= 1.5
    elif priority == 2:
        y = 0.8 * np.sin(2 * np.pi * f * t) + 0.2 * np.sin(2 * np.pi * 2 * f * t)
    else:
        y = np.sin(2 * np.pi * f * t)
    a = min(n // 2, int(0.004 * rate))
    if a > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
        y[:a] *= ramp
        y[-a:] *= ramp[::-1]
    return (GAIN[priority] * y).astype(np.float32)


def pan(mono: np.ndarray, bearing: float) -> np.ndarray:
    """Constant-power pan: -90 deg hard left, +90 hard right."""
    x = (max(-90.0, min(90.0, bearing)) + 90.0) / 180.0 * (np.pi / 2)
    return np.stack([mono * math.cos(x), mono * math.sin(x)], axis=1)


def render(timeline: list[tuple], rate: int = RATE, tail_s: float = 0.3) -> np.ndarray:
    """timeline: [(t, priority | None, distance | None, ttc | None, bearing)], sorted by t. Each state holds
    until the next. Returns float32 stereo (N, 2)."""
    if not timeline:
        return np.zeros((0, 2), np.float32)
    end = timeline[-1][0] + tail_s
    out = np.zeros((int(end * rate) + rate // 2, 2), np.float32)
    nxt = 0.0
    times = [s[0] for s in timeline] + [end]
    for i, (t0, p, d, ttc, b) in enumerate(timeline):
        t1 = times[i + 1]
        if p not in TONES:
            nxt = max(nxt, t0)
            continue
        gap = interval(p, d, ttc)
        if gap == 0.0:                                    # continuous: fill the whole segment
            seg = pan(tone(p, t1 - t0, rate), b or 0.0)
            s0 = int(t0 * rate)
            out[s0:s0 + len(seg)] += seg
            nxt = t1
            continue
        nxt = max(nxt, t0) if nxt <= t0 + gap else t0 + gap   # speeding up: do not wait out a slower beat
        while nxt < t1:
            seg = pan(tone(p, BEEP_S[p], rate), b or 0.0)
            s0 = int(nxt * rate)
            out[s0:s0 + len(seg)] += seg[: max(0, len(out) - s0)]
            nxt += gap
    return np.clip(out, -1.0, 1.0)


def write_wav(path: str | Path, stereo: np.ndarray, rate: int = RATE) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return p


def timeline_from_events(events: list[dict]) -> list[tuple]:
    return [(e["t"], e.get("priority"), e.get("distance_m"), e.get("ttc_s"),
             next((o.get("bearing", 0.0) for o in e.get("objects", []) if o.get("id") == e.get("id")), 0.0))
            for e in events]


def demo(out: Path) -> list[Path]:
    paths = []
    for p, d in ((1, 1.4), (2, 3.5), (3, 7.0)):
        paths.append(write_wav(out / f"p{p}.wav", render([(0.0, p, d, None, 0.0), (2.5, None, None, None, 0.0)])))
    tl, t = [], 0.0                       # walking at 1.2 m/s at a pole from 9 m to 0.5 m
    while True:
        d = 9.0 - 1.2 * t
        if d < 0.5:
            break
        p = 1 if d < 2 else 2 if d < 5 else 3
        tl.append((t, p, d, d / 1.2, 0.0))
        t += 0.1
    paths.append(write_wav(out / "approach.wav", render(tl)))
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m perception.near.beeper", description=__doc__.split("\n")[0])
    ap.add_argument("--demo", default="out/beeps", help="folder for p1/p2/p3/approach.wav")
    a = ap.parse_args(argv)
    for p in demo(Path(a.demo)):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
