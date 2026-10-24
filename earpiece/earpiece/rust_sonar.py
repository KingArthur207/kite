"""Drop-in replacement for sonar.Sonar whose decisions run in the Rust `kite_audio` crate.

Same API (update_frame / update_one / clear / tick / summary / tracks / stats /
enabled / muted / labels), same cue stream: `kite_audio.Sonar` is a line-for-line
port of the Python decision core (tracks, closing speed, time-to-contact, urgency,
the two-source cap, beep scheduling, "Stop." and label decisions), and the parity
tests in tests/test_rust_sonar.py drive both with identical frames and require
identical beeps, gains, buses, pitches, bearings and speech.

What stays in Python is the playback of what Rust decided: looking up the rendered
cue (`sonar._cue`, pre-warmed for every critical beep) and handing it to the engine,
plus the speak callback. `tracks` and `tick()` return read-only snapshots
(`kite_audio.TrackState`) with the attributes of `sonar.Track`.

Select with KITE_SONAR_ENGINE=rust|python (default: rust when the extension is built).
"""

from __future__ import annotations

import time
from typing import Callable

from . import sonar as sonar_mod
from .config import SonarCfg
from .events import Obstacle

try:
    import kite_audio as _ka
except ImportError:            # extension not built: the Python sonar is used instead
    _ka = None

_I64 = (-(2 ** 63), 2 ** 63 - 1)


def available() -> bool:
    return _ka is not None and hasattr(_ka, "Sonar")


def _ob(o: Obstacle) -> tuple:
    # str(): Python builds the track key with an f-string, so a non-str id works there too
    return (str(o.id), str(o.kind), o.bearing, o.distance, o.closing_speed, o.confidence)


class RustSonar:
    native = True

    def __init__(self, engine, cfg: SonarCfg, speak: Callable[..., object] | None = None, log=print):
        if not available():
            raise RuntimeError("kite_audio extension is not built (or predates the Rust sonar)")
        self.engine = engine
        self.cfg = cfg
        self.log = log
        self._s = _ka.Sonar(engine.rate, cfg)
        self.speak = speak                 # speak(text, priority, key=, ttl=)

    # ------------------------------------------------------------------ flags
    @property
    def speak(self):
        return self._speak

    @speak.setter
    def speak(self, fn) -> None:
        self._speak = fn
        self._s.speak = fn is not None     # like Python: no callback, no speech decisions

    enabled = property(lambda s: s._s.enabled, lambda s, v: setattr(s._s, "enabled", bool(v)))
    muted = property(lambda s: s._s.muted, lambda s, v: setattr(s._s, "muted", bool(v)))
    labels = property(lambda s: s._s.labels, lambda s, v: setattr(s._s, "labels", bool(v)))

    @property
    def stats(self) -> dict:
        return self._s.stats()

    @property
    def tracks(self) -> dict:
        """Snapshot {id: TrackState}, in insertion order. Read-only."""
        return {t.id: t for t in self._s.tracks()}

    # ------------------------------------------------------------------ input
    def update_frame(self, source: str, objects: list[Obstacle], seq: int | None = None,
                     now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if seq is not None:
            seq = min(_I64[1], max(_I64[0], int(seq)))
        self._s.update_frame(source, [_ob(o) for o in objects], seq, now)

    def update_one(self, source: str, o: Obstacle, ttl: float | None, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._s.update_one(source, _ob(o), ttl, now)

    def clear(self, source: str | None = None) -> None:
        self._s.clear(source)

    # ------------------------------------------------------------------ tick
    def tick(self, now: float | None = None) -> list:
        """Call ~50 Hz. Rust decides; this plays what it decided, in the same order
        Python's Sonar.tick would. Returns the sonified tracks (snapshots)."""
        now = time.monotonic() if now is None else now
        eng = self.engine
        chosen, cues = self._s.tick(now, int(eng.now))
        for c in cues:
            if c[0] == "beep":
                _, at, kind, pitch_q, bear_q, critical, gain, tid = c
                eng.play(sonar_mod._cue(kind, pitch_q, bear_q, eng.rate), at=at,
                         bus="alert" if critical else "cue", gain=gain, tag=f"sonar:{tid}")
            elif self._speak is not None:
                _, text, priority, key, ttl = c
                self._speak(text, priority, key=key, ttl=ttl)
        return chosen

    def summary(self) -> list[dict]:
        return sonar_mod.summarize(self._s.tracks())
