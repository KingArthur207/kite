"""Hearing the wearer: microphone capture, end-of-speech detection, Cartesia STT.

Mic choice matters more than anything else here. Opening the AirPods' own
microphone switches them into the two-way call profile (mono, ~16 kHz) for as
long as it is open, which collapses the spatial cues to mono. So by default we
record from the laptop's built-in mic and leave the AirPods output-only
(Apple: support.apple.com/102217). `--mic airpods` overrides that.

Push-to-talk: AirPods stem press (or Enter) -> rising chirp -> speak -> the
utterance ends after ~0.7 s of silence (or a second press) -> "got it" blip ->
transcript. Audio streams to Cartesia while the user is still talking, so the
transcript is ready ~0.1 s after they stop (measured: 87 ms finalize latency).

Hands-free (optional): always listening; an utterance is only acted on if it
starts with the wake word ("Kite, ...").
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from typing import Callable

import numpy as np

from .config import ListenCfg


def pick_input(substr: str = "") -> tuple[int | None, str]:
    """Input device index and name. Prefers a non-AirPods mic unless asked otherwise."""
    import sounddevice as sd
    devs = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    if substr:
        for i, d in devs:
            if substr.lower() in d["name"].lower():
                return i, d["name"]
    try:
        default = sd.query_devices(kind="input")
        dname = default["name"]
    except Exception:
        dname = ""
    if substr.lower() != "airpods" and "airpods" in dname.lower():
        for i, d in devs:
            n = d["name"].lower()
            if "airpods" not in n and ("macbook" in n or "built-in" in n or "microphone" in n):
                return i, d["name"]
    return None, dname or "default"


class Endpointer:
    """Adaptive energy VAD with hysteresis (pattern proven in our vigil voice app).
    Learns the room's noise floor while nobody talks; an utterance starts at 3x the
    floor and continues while above 1.6x, ending after `end_silence_s` of quiet."""

    def __init__(self, rate: int, end_silence_s: float = 0.7, max_s: float = 8.0,
                 no_speech_s: float = 4.0):
        self.rate = rate
        self.end_silence_s = end_silence_s
        self.max_s = max_s
        self.no_speech_s = no_speech_s
        self.noise = 0.003
        self.speaking = False
        self.started = False
        self.quiet_s = 0.0
        self.elapsed = 0.0
        self.voiced_s = 0.0

    def push(self, block: np.ndarray) -> str | None:
        """Feed a block; returns 'start', 'end', 'timeout', or None."""
        dur = block.size / self.rate
        self.elapsed += dur
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) if block.size else 0.0
        start_thr = max(3.0 * self.noise, 0.006)
        keep_thr = max(1.6 * self.noise, 0.003)
        ev = None
        if not self.speaking and rms < start_thr:
            self.noise += 0.05 * (rms - self.noise)
            self.noise = max(self.noise, 1e-4)
        if rms >= (keep_thr if self.speaking else start_thr):
            if not self.speaking and not self.started:
                ev = "start"
            self.speaking = self.started = True
            self.quiet_s = 0.0
            self.voiced_s += dur
        elif self.speaking:
            self.quiet_s += dur
            if self.quiet_s >= self.end_silence_s:
                self.speaking = False
                return "end" if self.voiced_s >= 0.15 else "timeout"
        if self.started and self.elapsed >= self.max_s:
            return "end"
        if not self.started and self.elapsed >= self.no_speech_s:
            return "timeout"
        return ev


class CartesiaSTT:
    URL = ("wss://api.cartesia.ai/stt/websocket?model={m}&language=en&encoding=pcm_s16le"
           "&sample_rate={r}&cartesia_version=2026-08-14")

    def __init__(self, key: str | None, model: str, rate: int):
        self.key = key
        self.model = model
        self.rate = rate

    async def transcribe_stream(self, chunks: "asyncio.Queue[bytes | None]", timeout_s: float = 4.0) -> str:
        """Send PCM s16 chunks from the queue until None, then finalize and return the text."""
        import websockets
        async with websockets.connect(self.URL.format(m=self.model, r=self.rate),
                                      additional_headers={"X-API-Key": self.key},
                                      open_timeout=4, max_size=None) as ws:
            text = ""
            done = asyncio.Event()

            async def rx():
                nonlocal text
                async for raw in ws:
                    try:
                        m = json.loads(raw)
                    except ValueError:
                        continue
                    if m.get("type") == "transcript" and m.get("is_final"):
                        text += m.get("text", "")
                    elif m.get("type") in ("flush_done", "done"):
                        done.set()
                        return
                    elif m.get("type") == "error":
                        done.set()
                        raise RuntimeError(m.get("message", "stt error"))

            reader = asyncio.create_task(rx())
            while True:
                c = await chunks.get()
                if c is None:
                    break
                await ws.send(c)
            await ws.send("finalize")
            try:
                await asyncio.wait_for(done.wait(), timeout=timeout_s)
            finally:
                reader.cancel()
                try:
                    await ws.send("close")
                except Exception:
                    pass
            return text.strip()


class Listener:
    def __init__(self, cfg: ListenCfg, key: str | None, on_text: Callable[[str], None],
                 cue: Callable[[str], None], on_listening: Callable[[bool], None],
                 say: Callable[..., object], log=print):
        self.cfg = cfg
        self.stt = CartesiaSTT(key, cfg.stt_model, cfg.sample_rate) if key else None
        self.on_text = on_text             # final transcript (wake word already handled)
        self.cue = cue                     # cue("listening"|"heard"|"error")
        self.on_listening = on_listening   # True while the wearer is talking
        self.say = say
        self.log = log
        self._active = threading.Event()
        self._stop_now = threading.Event()
        self._mic_warned = False
        self.device_name: str | None = None
        self.stats = {"utterances": 0, "empty": 0, "errors": 0, "last_ms": None}
        self._hf_thread: threading.Thread | None = None
        self._hf_stop = threading.Event()

    @property
    def available(self) -> bool:
        return self.stt is not None

    @property
    def active(self) -> bool:
        return self._active.is_set()

    # ------------------------------------------------------------------ push-to-talk
    def push_to_talk(self) -> None:
        """Start listening, or (if already listening) end the utterance now."""
        if self._active.is_set():
            self._stop_now.set()
            return
        if self.stt is None:
            self.cue("error")
            self.say("Voice commands are offline.", "system", key="stt-offline", ttl=5)
            return
        self._active.set()
        self._stop_now.clear()
        threading.Thread(target=self._ptt_thread, name="ptt", daemon=True).start()

    def _ptt_thread(self) -> None:
        try:
            asyncio.run(self._capture_once(require_wake=False))
        except Exception as e:
            self.stats["errors"] += 1
            self.log(f"[listen] failed: {type(e).__name__}: {e}")
            self.cue("error")
        finally:
            self._active.clear()
            self.on_listening(False)

    async def _capture_once(self, require_wake: bool) -> None:
        import sounddevice as sd
        rate = self.cfg.sample_rate
        idx, name = pick_input(self.cfg.input_substr)
        self.device_name = name
        q: queue.Queue = queue.Queue()

        def cb(indata, frames, t, status):
            q.put(indata[:, 0].copy())

        self.on_listening(True)
        chunks: asyncio.Queue = asyncio.Queue()
        stt_task = asyncio.create_task(self.stt.transcribe_stream(chunks))
        ep = Endpointer(rate, self.cfg.end_silence_s, self.cfg.max_utterance_s, self.cfg.no_speech_timeout_s)
        block = int(rate * 0.1)
        pending = np.zeros(0, dtype=np.float32)
        peak = 0.0
        t_end_speech = None
        with sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=idx,
                            blocksize=int(rate * 0.05), callback=cb):
            self.cue("listening")
            t0 = time.monotonic()
            result = None
            while result is None:
                if self._stop_now.is_set():
                    result = "end" if ep.started else "timeout"
                    break
                try:
                    data = q.get(timeout=0.3)
                except queue.Empty:
                    if time.monotonic() - t0 > self.cfg.max_utterance_s + 1:
                        result = "timeout"
                    continue
                peak = max(peak, float(np.max(np.abs(data))) if data.size else 0.0)
                pending = np.concatenate([pending, data])
                while pending.size >= block:
                    b, pending = pending[:block], pending[block:]
                    pcm = (np.clip(b, -1, 1) * 32767).astype("<i2").tobytes()
                    await chunks.put(pcm)
                    ev = ep.push(b)
                    if ev in ("end", "timeout"):
                        result = ev
                        break
                if time.monotonic() - t0 > 1.0 and peak == 0.0 and not self._mic_warned:
                    self._mic_warned = True
                    self.log("[listen] microphone delivers silence: check System Settings > "
                             "Privacy & Security > Microphone for this terminal")
                    self.say("I can't hear the microphone. Check the microphone permission.",
                             "system", key="mic-perm", ttl=10)
                    result = "timeout"
                await asyncio.sleep(0)
            t_end_speech = time.monotonic()
        await chunks.put(None)
        if result == "timeout" and not ep.started:
            stt_task.cancel()
            self.stats["empty"] += 1
            self.cue("error")
            return
        self.cue("heard")
        try:
            text = await asyncio.wait_for(stt_task, timeout=6.0)
        except Exception as e:
            self.stats["errors"] += 1
            self.log(f"[listen] STT failed: {type(e).__name__}: {e}")
            self.say("Sorry, I couldn't reach speech recognition.", "system", key="stt-fail", ttl=5)
            return
        self.stats["last_ms"] = round((time.monotonic() - t_end_speech) * 1000)
        self._deliver(text, require_wake)

    def _deliver(self, text: str, require_wake: bool) -> None:
        from .brain import strip_wake
        text = text.strip()
        if not text:
            self.stats["empty"] += 1
            self.say("Sorry, I didn't catch that.", "system", ttl=4)
            return
        woke, rest = strip_wake(text, self.cfg.wake_words)
        if require_wake and not woke:
            self.log(f"[listen] ignored (no wake word): {text!r}")
            return
        self.stats["utterances"] += 1
        self.log(f"[listen] heard: {text!r} (+{self.stats['last_ms']} ms after speech)")
        self.on_text(rest if woke else text)

    # ------------------------------------------------------------------ hands-free
    def start_hands_free(self, speaking: Callable[[], bool]) -> None:
        """Always-on wake-word listening on a background thread. `speaking()` gates the
        mic while Kite talks through a loudspeaker (no self-transcription)."""
        if self.stt is None:
            self.log("[listen] hands-free needs the Cartesia key; disabled")
            return
        self._hf_stop.clear()
        self._hf_thread = threading.Thread(target=self._hf_run, args=(speaking,), name="handsfree",
                                           daemon=True)
        self._hf_thread.start()

    def _hf_run(self, speaking: Callable[[], bool]) -> None:
        while not self._hf_stop.is_set():
            if self._active.is_set() or speaking():
                time.sleep(0.1)
                continue
            try:
                self._active.set()
                asyncio.run(self._hf_once())
            except Exception as e:
                self.log(f"[listen] hands-free error: {type(e).__name__}: {e}")
                time.sleep(1.0)
            finally:
                self._active.clear()

    async def _hf_once(self) -> None:
        """Wait (mic open, nothing sent) for speech; then stream the utterance with a
        0.4 s pre-roll to STT and act on it only if it starts with the wake word."""
        import sounddevice as sd
        rate = self.cfg.sample_rate
        idx, _ = pick_input(self.cfg.input_substr)
        q: queue.Queue = queue.Queue()
        with sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=idx,
                            blocksize=int(rate * 0.05), callback=lambda d, f, t, s: q.put(d[:, 0].copy())):
            ep = Endpointer(rate, self.cfg.end_silence_s, self.cfg.max_utterance_s, no_speech_s=1e9)
            preroll: list[np.ndarray] = []
            block = int(rate * 0.1)
            pending = np.zeros(0, dtype=np.float32)
            chunks: asyncio.Queue | None = None
            task = None
            while not self._hf_stop.is_set():
                try:
                    data = q.get(timeout=0.3)
                except queue.Empty:
                    continue
                pending = np.concatenate([pending, data])
                while pending.size >= block:
                    b, pending = pending[:block], pending[block:]
                    ev = ep.push(b)
                    pcm = (np.clip(b, -1, 1) * 32767).astype("<i2").tobytes()
                    if chunks is None:
                        preroll.append(pcm)
                        del preroll[:-4]
                        if ev == "start":
                            chunks = asyncio.Queue()
                            for p in preroll:
                                chunks.put_nowait(p)
                            task = asyncio.create_task(self.stt.transcribe_stream(chunks))
                    else:
                        await chunks.put(pcm)
                        if ev in ("end", "timeout"):
                            await chunks.put(None)
                            text = await asyncio.wait_for(task, timeout=6.0)
                            self.stats["last_ms"] = None
                            self._deliver(text, require_wake=True)
                            return
                await asyncio.sleep(0)

    def stop(self) -> None:
        self._hf_stop.set()
        self._stop_now.set()
