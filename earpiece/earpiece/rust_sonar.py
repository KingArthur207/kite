"""Drop-in replacement for sonar.Sonar whose decisions run in the Rust `kite_audio` crate.

Same API (update_frame / update_one / clear / tick / summary / tracks / stats /
enabled / muted / labels), same cue stream: `kite_audio.Sonar` is a line-for-line
port of the Python decision core (tracks, closing speed, time-to-contact, urgency,
the two-source cap, beep scheduling, "Stop." and label decisions), and the parity
tests in tests/test_rust_sonar.py drive both with identical frames and require
identical beeps, gains, buses, pitches, bearings and speech.

Two ways to run it:

  * ticked from Python (default, KITE_SONAR_THREAD=python): the app's 50 Hz control
    loop calls tick(); Rust decides, and this module plays what it decided (the
    rendered cue `sonar._cue`, pre-warmed for every critical beep) and calls speak.
  * native thread (KITE_SONAR_THREAD=native, needs the Rust audio engine): a Rust
    thread ticks every 20 ms, synthesises and places each beep in Rust (sounds.rs,
    sample-identical to sounds.py) and schedules it straight into the mixer, and runs
    the stale-perception watchdog. Python only feeds frames; tick() then just polls
    the "speak" requests and watchdog transitions out of a lock-free ring. A stalled
    interpreter can delay what Kite says, but not a beep (tests/test_sonar_thread.py).

`tracks` and `tick()` return read-only snapshots (`kite_audio.TrackState`) with the
attributes of `sonar.Track`. Select the decisions with KITE_SONAR_ENGINE=rust|python
(default: rust when the extension is built).
"""

from __future__ import annotations

import collections
import time
from typing import Callable

from . import sonar as sonar_mod, sounds
from .config import SonarCfg
from .events import Obstacle

try:
    import kite_audio as _ka
except ImportError:            # extension not built: the Python sonar is used instead
    _ka = None

_I64 = (-(2 ** 63), 2 ** 63 - 1)
TICK_PERIOD_S = 0.02           # the native thread's step, the Python control loop's period


# every kite_audio.Sonar method RustSonar calls outside the native thread: a build older than these sources lacks
# some, and "auto" must then fall back to the Python sonar rather than fail on the first obstacle message
REQUIRED = ("clear", "note_perception", "perception_ok", "poll", "set_profile", "stats", "tick", "tracks",
            "update_frame", "update_one")


SONAR_ABI = 2                  # native/src/lib.rs SONAR_ABI: 2 = obstacles carry a priority (7-tuples)


def stale_build() -> list[str]:
    """What this build of kite_audio.Sonar lacks (empty when it is current or not built at all): missing methods,
    or an older obstacle signature (SONAR_ABI) that would reject the tuples this wrapper sends."""
    if _ka is None or not hasattr(_ka, "Sonar"):
        return []
    missing = [m for m in REQUIRED if not hasattr(_ka.Sonar, m)]
    if getattr(_ka, "SONAR_ABI", 1) < SONAR_ABI:
        missing.append(f"SONAR_ABI {SONAR_ABI}")
    return missing


def available() -> bool:
    return _ka is not None and hasattr(_ka, "Sonar") and not stale_build()


def native_available() -> bool:
    """kite_audio has the native sonar thread (and the cue synthesiser it needs)."""
    return available() and hasattr(_ka.Sonar, "start_native") and hasattr(_ka, "synth")


def _ob(o: Obstacle) -> tuple:
    # str(): Python builds the track key with an f-string, so a non-str id works there too
    return (str(o.id), str(o.kind), o.bearing, o.distance, o.closing_speed, o.confidence, o.priority)


class RustSonar:
    native = True

    def __init__(self, engine, cfg: SonarCfg, speak: Callable[..., object] | None = None, log=print,
                 thread: bool = False, perception_stale_s: float = 1.5, prewarm: bool = True):
        if not available():
            raise RuntimeError("kite_audio extension is not built (or predates the Rust sonar)")
        self.engine = engine
        self.cfg = cfg
        self.log = log
        self._s = _ka.Sonar(engine.rate, cfg)
        self.speak = speak                 # speak(text, priority, key=, ttl=)
        self.thread = False                # True: ticks on the native thread
        self._perception: collections.deque = collections.deque()
        if thread:
            self.start_native(perception_stale_s, prewarm=prewarm)

    # ------------------------------------------------------------------ native thread
    def start_native(self, perception_stale_s: float = 1.5, prewarm: bool = True) -> None:
        if not native_available():
            raise RuntimeError("kite_audio predates the native sonar thread")
        eng = getattr(self.engine, "_e", None)
        if eng is None:
            raise RuntimeError("the native sonar thread needs the Rust audio engine")
        self._s.start_native(eng, time.monotonic(), TICK_PERIOD_S, perception_stale_s,
                             sounds.profile(), prewarm)
        self.thread = True
        sounds.on_profile_change(self._s.set_profile)

    def close(self) -> None:
        """Stop the native thread (if any). Safe to call twice."""
        if self.thread:
            self.thread = False
            sounds.remove_profile_listener(self._s.set_profile)
            self._s.stop_native()

    def native_step(self, now: float) -> None:
        """Run one native step on this thread (deterministic tests): what the native thread
        does every period. poll() then hands out what it decided to say."""
        self._s.set_profile(sounds.profile())       # a no-op unless the profile changed
        self._s.native_step(self.engine._e, now)

    @property
    def native_watchdog(self) -> bool:
        """True when the stale-perception watchdog runs natively (the app only announces)."""
        return self.thread

    @property
    def perception_ok(self) -> bool | None:
        return self._s.perception_ok

    def note_perception(self, now: float | None = None) -> None:
        self._s.note_perception(time.monotonic() if now is None else now)

    def take_perception_events(self) -> list[tuple[bool, bool]]:
        """Native watchdog transitions since the last call: (ok, first), oldest first."""
        out = list(self._perception)
        self._perception.clear()
        return out

    def native_stats(self) -> dict:
        return self._s.native_stats()

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
        """Call ~50 Hz. Python-ticked: Rust decides, this plays what it decided, in the
        same order Python's Sonar.tick would. Native thread: the beeps are already
        scheduled; this hands the queued speech to speak() and returns the tracks the
        native thread last sonified. Returns the sonified tracks (snapshots)."""
        if self.thread:
            return self._poll()
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

    def poll(self) -> list:
        """Dispatch what native steps queued (speech, watchdog transitions); return the
        tracks the last native step sonified."""
        return self._poll()

    def _poll(self) -> list:
        chosen, events = self._s.poll()
        for e in events:
            if e[0] == "say":
                if self._speak is not None:
                    _, text, priority, key, ttl = e
                    self._speak(text, priority, key=key, ttl=ttl)
            else:
                _, ok, first, _at = e
                self._perception.append((ok, first))
        return chosen

    def summary(self) -> list[dict]:
        return sonar_mod.summarize(self._s.tracks())

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
