"""Getting the mix to the ears: the real audio device, or a clocked null sink.

DeviceOutput keeps a sounddevice stream running on the best output (AirPods if
present) and survives the things that happen in the street: AirPods taken out
or disconnected (macOS falls back to the speaker), reconnected, the stream
dying, or the default device changing. A watcher thread checks every couple of
seconds and re-opens the stream on the right device; the engine keeps its clock,
so scheduled cues simply continue.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from .mixer import Engine


class NullOutput:
    """Pumps engine.render() in real time (or as fast as possible) and optionally
    records what would have been played. Used for tests, simulation and CI."""

    def __init__(self, engine: Engine, block: int = 512, realtime: bool = True,
                 record: bool = False, max_record_s: float = 600.0):
        self.engine = engine
        self.block = block
        self.realtime = realtime
        self.record = record
        self._max = int(max_record_s * engine.rate)
        self.frames: list[np.ndarray] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.device_name = "null"
        self.is_airpods = False
        self.underflows = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="null-audio", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        period = self.block / self.engine.rate
        nxt = time.perf_counter()
        n = 0
        while not self._stop.is_set():
            out = self.engine.render(self.block)
            if self.record and n < self._max:
                self.frames.append(out.copy())
                n += out.shape[0]
            if self.realtime:
                nxt += period
                d = nxt - time.perf_counter()
                if d > 0:
                    time.sleep(d)
                elif d < -0.2:
                    nxt = time.perf_counter()

    def pump(self, seconds: float) -> np.ndarray:
        """Synchronous render (no thread): returns the stereo audio."""
        n = int(seconds * self.engine.rate)
        out = [self.engine.render(min(self.block, n - i)) for i in range(0, n, self.block)]
        a = np.concatenate(out) if out else np.zeros((0, 2), np.float32)
        if self.record:
            self.frames.append(a)
        return a

    def recorded(self) -> np.ndarray:
        return np.concatenate(self.frames) if self.frames else np.zeros((0, 2), np.float32)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _list_outputs():
    import sounddevice as sd
    return [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] >= 1]


def _refresh_portaudio() -> None:
    """PortAudio caches the device list at init; re-init to see (dis)connected AirPods.
    Only safe while no stream is open."""
    import sounddevice as sd
    try:
        sd._terminate()
    finally:
        sd._initialize()


class DeviceOutput:
    def __init__(self, engine: Engine, prefer_substr: str = "AirPods", block: int = 512,
                 poll_s: float = 2.0, on_change: Callable[[str, bool, str | None], None] | None = None,
                 log=print, device: str | int | None = None):
        self.engine = engine
        self.prefer = prefer_substr.lower()
        self.block = block
        self.poll_s = poll_s
        self.on_change = on_change         # (device_name, is_airpods, previous_name)
        self.log = log
        self.forced = device
        self._stream = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.device_name: str | None = None
        self.is_airpods = False
        self.underflows = 0
        self.callback_errors = 0
        self.restarts = 0
        self._last_cb = time.monotonic()

    # -------------------------------------------------------------- device choice
    def _pick(self) -> tuple[int | None, str]:
        import sounddevice as sd
        outs = _list_outputs()
        if self.forced is not None:
            for i, d in outs:
                if (isinstance(self.forced, int) and i == self.forced) or \
                   (isinstance(self.forced, str) and self.forced.lower() in d["name"].lower()):
                    return i, d["name"]
        for i, d in outs:
            if self.prefer and self.prefer in d["name"].lower():
                return i, d["name"]
        try:
            d = sd.query_devices(kind="output")
            return None, d["name"]
        except Exception:
            return (outs[0][0], outs[0][1]["name"]) if outs else (None, "none")

    def _callback(self, outdata, frames, t, status) -> None:
        if status.output_underflow:
            self.underflows += 1
        try:
            mix = self.engine.render(frames)
            if outdata.shape[1] == 2:
                outdata[:] = mix
            elif outdata.shape[1] == 1:
                outdata[:, 0] = mix.mean(axis=1)
            else:
                outdata.fill(0)
                outdata[:, :2] = mix
        except Exception:
            self.callback_errors += 1
            outdata.fill(0)
        self._last_cb = time.monotonic()

    def _open(self) -> None:
        import sounddevice as sd
        idx, name = self._pick()
        info = sd.query_devices(idx if idx is not None else sd.default.device[1])
        ch = 2 if info["max_output_channels"] >= 2 else 1
        stream = sd.OutputStream(samplerate=self.engine.rate, blocksize=self.block, channels=ch,
                                 dtype="float32", device=idx, latency="low", callback=self._callback)
        stream.start()
        prev = self.device_name
        self._stream = stream
        self.device_name = name
        self.is_airpods = self.prefer in name.lower() if self.prefer else False
        self._last_cb = time.monotonic()
        self.log(f"[audio] output: {name} ({ch} ch, {self.engine.rate} Hz, "
                 f"latency {stream.latency * 1000:.0f} ms)")
        if self.on_change is not None and prev != name:
            try:
                self.on_change(name, self.is_airpods, prev)
            except Exception as e:
                self.log(f"[audio] on_change failed: {e}")

    def _close(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.abort(ignore_errors=True)
                s.close(ignore_errors=True)
            except Exception:
                pass

    def start(self) -> None:
        with self._lock:
            self._open()
        threading.Thread(target=self._watch, name="audio-watch", daemon=True).start()

    def restart(self, why: str) -> None:
        with self._lock:
            self.log(f"[audio] reopening output ({why})")
            self._close()
            for attempt in range(20):
                try:
                    _refresh_portaudio()
                    self._open()
                    self.restarts += 1
                    return
                except Exception as e:
                    self.log(f"[audio] no usable output yet ({type(e).__name__}: {e}); retrying")
                    time.sleep(min(2.0, 0.25 * (attempt + 1)))

    def _watch(self) -> None:
        from . import devices
        last_default = devices.default_output_name()
        while not self._stop.wait(self.poll_s):
            try:
                s = self._stream
                if s is None or not s.active or time.monotonic() - self._last_cb > 1.5:
                    self.restart("stream stopped")
                    last_default = devices.default_output_name()
                    continue
                cur_default = devices.default_output_name()
                want_airpods = devices.airpods_output_present(self.prefer) if self.prefer else False
                if want_airpods and not self.is_airpods:
                    self.restart("AirPods appeared")
                elif self.is_airpods and want_airpods is False:
                    self.restart("AirPods gone")
                elif cur_default and last_default and cur_default != last_default and not self.is_airpods:
                    self.restart(f"default output changed to {cur_default}")
                last_default = cur_default
            except Exception as e:
                self.log(f"[audio] watcher: {type(e).__name__}: {e}")

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._close()
