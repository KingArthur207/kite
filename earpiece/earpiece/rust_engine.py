"""Drop-in replacement for mixer.Engine backed by the Rust `kite_audio` crate.

Same control API (play / stop / speech_* / now / speaking / speech_events / render),
so sonar, beacon, speech and the app do not know which engine they drive. The
difference is where the real-time work happens: the device callback, mixing,
ducking and limiting all run on CoreAudio's thread inside Rust, fed through
lock-free rings, so nothing on the Python side (GIL, garbage collection, network
I/O, Claude or Moss calls) can starve the audio.

If the extension is not built, `available()` is False and the app uses the Python
engine instead. Build it with:  maturin develop --release -m native/Cargo.toml
"""

from __future__ import annotations

import itertools
import queue
import threading
import time

import numpy as np

try:
    import kite_audio as _ka
except ImportError:            # extension not built: the Python engine is used instead
    _ka = None

_EV = {0: "start", 1: "done", 2: "cut"}
_BUS = {"cue": 0, "alert": 1}


def available() -> bool:
    return _ka is not None


class RustEngine:
    native = True

    def __init__(self, rate: int = 48000, master_gain: float = 0.9, ceiling: float = 0.89,
                 cue_gain: float = 0.55, speech_gain: float = 1.0, duck: float = 0.45):
        if _ka is None:
            raise RuntimeError("kite_audio extension is not built")
        self.rate = rate
        self.ceiling = ceiling
        self._p = {"master_gain": master_gain, "cue_gain": cue_gain, "speech_gain": speech_gain,
                   "duck": duck, "cues_muted": False}
        self._e = _ka.Engine(rate, master_gain, ceiling, cue_gain, speech_gain, duck)
        self.speech_events: queue.SimpleQueue = queue.SimpleQueue()
        self._ids = itertools.count(1)
        self._uid: dict[str, int] = {}
        self._uid_rev: dict[int, str] = {}
        self._tag: dict[str, int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._pump, name="rust-events", daemon=True).start()

    # ------------------------------------------------------------------ ids
    def _uid_of(self, uid: str) -> int:
        with self._lock:
            n = self._uid.get(uid)
            if n is None:
                n = next(self._ids)
                self._uid[uid] = n
                self._uid_rev[n] = uid
            return n

    def _tag_of(self, tag: str | None) -> int:
        if tag is None:
            return 0
        with self._lock:
            n = self._tag.get(tag)
            if n is None:
                n = self._tag[tag] = next(self._ids)
            return n

    def _pump(self) -> None:
        """Move speech events from the Rust ring into the same queue the Python engine uses."""
        while not self._stop.wait(0.004):
            for kind, n in self._e.poll_events():
                with self._lock:
                    uid = self._uid_rev.get(n)
                    if kind in (1, 2) and uid is not None:          # finished: forget the mapping
                        self._uid_rev.pop(n, None)
                        self._uid.pop(uid, None)
                if uid is not None:
                    self.speech_events.put((_EV[kind], uid, time.monotonic()))

    # ------------------------------------------------------------------ params
    def _set(self, name: str, value) -> None:
        self._p[name] = value
        self._e.set_params(**{name: value})

    master_gain = property(lambda s: s._p["master_gain"], lambda s, v: s._set("master_gain", float(v)))
    cue_gain = property(lambda s: s._p["cue_gain"], lambda s, v: s._set("cue_gain", float(v)))
    speech_gain = property(lambda s: s._p["speech_gain"], lambda s, v: s._set("speech_gain", float(v)))
    duck = property(lambda s: s._p["duck"], lambda s, v: s._set("duck", float(v)))
    cues_muted = property(lambda s: s._p["cues_muted"], lambda s, v: s._set("cues_muted", bool(v)))

    # ------------------------------------------------------------------ control API
    def play(self, stereo: np.ndarray, at: int | None = None, bus: str = "cue",
             gain: float = 1.0, tag: str | None = None) -> None:
        if stereo.ndim == 1:
            stereo = np.repeat(stereo[:, None], 2, axis=1)
        data = np.ascontiguousarray(stereo, dtype=np.float32).tobytes()
        self._e.play(data, -1 if at is None else int(at), _BUS.get(bus, 0), float(gain), self._tag_of(tag))

    def stop(self, tag: str, fade_s: float = 0.03) -> None:
        self._e.stop(self._tag_of(tag), max(1, int(fade_s * self.rate)))

    def speech_begin(self, uid: str, gain: float = 1.0) -> None:
        self._e.speech_begin(self._uid_of(uid), float(gain))

    def speech_feed(self, uid: str, mono: np.ndarray) -> None:
        if mono.size:
            self._e.speech_feed(self._uid_of(uid), np.ascontiguousarray(mono, dtype=np.float32).tobytes())

    def speech_end(self, uid: str) -> None:
        self._e.speech_end(self._uid_of(uid))

    def speech_cancel(self, uid: str | None = None, fade_s: float = 0.04) -> None:
        self._e.speech_cancel(0 if uid is None else self._uid_of(uid), max(1, int(fade_s * self.rate)))

    @property
    def now(self) -> int:
        return self._e.now

    @property
    def speaking(self) -> bool:
        return self._e.speaking

    def seconds(self, samples: int) -> float:
        return samples / self.rate

    def samples(self, seconds: float) -> int:
        return int(round(seconds * self.rate))

    # ------------------------------------------------------------------ rendering / device
    def render(self, frames: int) -> np.ndarray:
        """Offline render (tests, simulation, WAV export)."""
        return np.frombuffer(self._e.render(frames), dtype=np.float32).reshape(-1, 2).copy()

    def open(self, substr: str = "", block: int = 512) -> tuple[str, int, int]:
        return self._e.open(substr, block)

    def close(self) -> None:
        self._e.close()

    def device(self):
        return self._e.device()

    def callback_age_ms(self) -> float | None:
        return self._e.callback_age_ms()

    def snapshot_stats(self) -> dict:
        s = self._e.stats()
        return {"blocks": int(s["blocks"]), "max_render_ms": s["max_render_ms"],
                "avg_render_ms": s["avg_render_ms"], "voices": int(s["voices"]),
                "limited_blocks": int(s["limited_blocks"]), "peak": s["peak"],
                "glitches": int(s["glitches"]), "contended": int(s["contended"]),
                "dropped_cmds": int(s["dropped_cmds"]), "engine": "rust"}

    @property
    def stats(self) -> dict:
        return self.snapshot_stats()

    def shutdown(self) -> None:
        self._stop.set()
        self._e.close()
