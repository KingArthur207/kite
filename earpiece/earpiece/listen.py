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

Ink returns FINAL segments as the wearer pauses (measured: "Kite," arrives ~0.6 s into
the utterance; the last segment ~0.4 s after the last word). At the first 0.3 s pause we
also send a speculative `finalize` (Ink flushes what it has in ~0.15 s and the stream
stays open), so the words so far are in hand about half a second after they were said.
Two things use them while the wearer may still be talking:
  * early end: a transcript that is already a complete short command ("stop", "louder",
    "talk faster") and covers everything said ends the utterance at ~0.5 s of quiet
    instead of 0.7 s, and is acted on without waiting for the closing `finalize`;
  * `on_partial(text)`: the brain starts the Moss recall for the question so far.

Hands-free (optional): always listening; an utterance is only acted on if it
starts with the wake word ("Kite, ..."). With headphones the mic stays open while
Kite talks, so saying "Kite" is a voice barge-in: the wake-word segment cuts
non-safety speech before the wearer has finished the sentence.
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

QUICK_END_SILENCE_S = 0.35     # quiet needed to end a recognised short command early
FLUSH_AT_SILENCE_S = 0.3       # at a pause this long, ask Ink for the transcript so far
FLUSH = "flush"                # queue item: speculative finalize (the stream stays open)


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


class MicSource:
    """The microphone as a blocking source of float32 mono blocks (a context manager)."""

    def __init__(self, rate: int, device: int | None):
        self.rate = rate
        self.device = device
        self.q: queue.Queue = queue.Queue()
        self._stream = None

    def __enter__(self):
        import sounddevice as sd
        self._stream = sd.InputStream(samplerate=self.rate, channels=1, dtype="float32", device=self.device,
                                      blocksize=int(self.rate * 0.05),
                                      callback=lambda d, f, t, s: self.q.put(d[:, 0].copy()))
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def get(self, timeout: float) -> np.ndarray | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


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

    async def transcribe_stream(self, chunks: "asyncio.Queue[bytes | str | None]", timeout_s: float = 4.0,
                                on_segment: Callable[[str, float], None] | None = None) -> str:
        """Send PCM s16 chunks from the queue until None, then finalize and return the text.
        A FLUSH item asks Ink for the transcript so far without ending the stream (a
        speculative finalize at a pause). `on_segment(text_so_far, seconds_covered)` fires
        for every final segment as it arrives."""
        import websockets
        async with websockets.connect(self.URL.format(m=self.model, r=self.rate),
                                      additional_headers={"X-API-Key": self.key},
                                      open_timeout=4, max_size=None) as ws:
            text = ""
            covered = 0.0
            done = asyncio.Event()
            flushes = {"sent": 0, "done": 0, "ending": False}

            async def rx():
                nonlocal text, covered
                async for raw in ws:
                    try:
                        m = json.loads(raw)
                    except ValueError:
                        continue
                    if m.get("type") == "transcript" and m.get("is_final"):
                        text += m.get("text", "")          # Ink sends deltas: join exactly as received
                        covered += float(m.get("duration") or 0.0)
                        if on_segment is not None and text.strip():
                            try:
                                on_segment(text.strip(), covered)
                            except Exception:
                                pass
                    elif m.get("type") == "flush_done":
                        flushes["done"] += 1
                        if flushes["ending"] and flushes["done"] >= flushes["sent"]:
                            done.set()
                            return
                    elif m.get("type") == "done":
                        done.set()
                        return
                    elif m.get("type") == "error":
                        done.set()
                        raise RuntimeError(m.get("message") or m.get("title") or "stt error")

            reader = asyncio.create_task(rx())
            while True:
                c = await chunks.get()
                if c is None:
                    break
                if c == FLUSH:
                    flushes["sent"] += 1
                    await ws.send("finalize")
                    continue
                await ws.send(c)
            flushes["ending"] = True
            flushes["sent"] += 1
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
        self.on_partial: Callable[[str], None] | None = None   # text so far, while still talking
        self.turns = None                  # speech.VoiceTurns: end-of-speech -> reply latency
        self._active = threading.Event()
        self._listening = False
        self._stop_now = threading.Event()
        self._mic_warned = False
        self.device_name: str | None = None
        self.stats = {"utterances": 0, "empty": 0, "errors": 0, "last_ms": None, "stt_audio_s": 0.0,
                      "early_ends": 0, "voice_barge_ins": 0}
        self._hf_thread: threading.Thread | None = None
        self._hf_stop = threading.Event()

    @property
    def available(self) -> bool:
        return self.stt is not None

    @property
    def active(self) -> bool:
        return self._active.is_set()

    def _open_source(self):
        """Where audio comes from. Tests and the live check swap in recorded audio."""
        idx, name = pick_input(self.cfg.input_substr)
        self.device_name = name
        return MicSource(self.cfg.sample_rate, idx)

    def _set_listening(self, on: bool) -> None:
        """Tell the app once per change (the speech queue holds replies while this is on)."""
        if on != self._listening:
            self._listening = on
            self.on_listening(on)

    def _partial(self, text: str) -> None:
        if self.on_partial is not None:
            from .brain import strip_wake
            woke, rest = strip_wake(text, self.cfg.wake_words)
            if woke and not rest:
                return                          # just "Kite,": nothing to look up yet
            try:
                self.on_partial(rest if woke else text)   # the brain is asked without the wake word
            except Exception as e:
                self.log(f"[listen] partial handler failed: {type(e).__name__}: {e}")

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
            self._set_listening(False)

    async def _capture_once(self, require_wake: bool) -> None:
        from .brain import quick_command
        rate = self.cfg.sample_rate
        src = self._open_source()
        self._set_listening(True)
        chunks: asyncio.Queue = asyncio.Queue()
        seg = {"text": "", "covered": 0.0, "t": 0.0}

        def on_segment(text: str, covered: float) -> None:
            seg.update(text=text, covered=covered, t=time.monotonic())
            self._partial(text)

        stt_task = asyncio.create_task(self.stt.transcribe_stream(chunks, on_segment=on_segment))
        ep = Endpointer(rate, self.cfg.end_silence_s, self.cfg.max_utterance_s, self.cfg.no_speech_timeout_s)
        block = int(rate * 0.1)
        pending = np.zeros(0, dtype=np.float32)
        peak = 0.0
        sent_s = 0.0                 # audio seconds streamed so far
        voice_end_s = 0.0            # ... up to the end of the last voiced block
        t_last_voice: float | None = None
        flushed_for: float | None = None
        early = False
        with src:
            self.cue("listening")
            t0 = time.monotonic()
            result = None
            while result is None:
                if self._stop_now.is_set():
                    result = "end" if ep.started else "timeout"
                    break
                data = src.get(0.3)
                if data is None:
                    if time.monotonic() - t0 > self.cfg.max_utterance_s + 1:
                        result = "timeout"
                    continue
                peak = max(peak, float(np.max(np.abs(data))) if data.size else 0.0)
                pending = np.concatenate([pending, data])
                while pending.size >= block:
                    b, pending = pending[:block], pending[block:]
                    pcm = (np.clip(b, -1, 1) * 32767).astype("<i2").tobytes()
                    await chunks.put(pcm)
                    sent_s += b.size / rate
                    self.stats["stt_audio_s"] += b.size / rate          # Cartesia STT bills audio time
                    ev = ep.push(b)
                    if ep.speaking and ep.quiet_s == 0.0:
                        t_last_voice, voice_end_s = time.monotonic(), sent_s
                    if ev in ("end", "timeout"):
                        result = ev
                        break
                    if (ep.speaking and ep.quiet_s >= FLUSH_AT_SILENCE_S and t_last_voice is not None
                            and flushed_for != t_last_voice):
                        # a pause: get the words so far now (~150 ms) instead of whenever Ink
                        # decides the segment is over (~400 ms after the last word, measured)
                        flushed_for = t_last_voice
                        await chunks.put(FLUSH)
                    # a complete short command, transcribed up to the wearer's last word: go now
                    if (ep.speaking and ep.quiet_s >= QUICK_END_SILENCE_S and seg["text"]
                            and (seg["t"] >= (t_last_voice or 0.0) or seg["covered"] >= voice_end_s - 0.1)
                            and quick_command(seg["text"])):
                        result, early = "end", True
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
        # the wearer has finished: speech may flow again, so the reply is not held behind
        # STT finalize / socket close (barge-in already dropped the stale replies)
        self._set_listening(False)
        await chunks.put(None)
        if result == "timeout" and not ep.started:
            stt_task.cancel()
            self.stats["empty"] += 1
            self.cue("error")
            return
        if self.turns is not None:
            self.turns.begin(t_last_voice or t_end_speech, t_end_speech, "ptt")
        self.cue("heard")
        if early:
            text = seg["text"]
            self.stats["early_ends"] += 1
        else:
            try:
                text = await asyncio.wait_for(stt_task, timeout=6.0)
            except Exception as e:
                self.stats["errors"] += 1
                self.log(f"[listen] STT failed: {type(e).__name__}: {e}")
                if self.turns is not None:
                    self.turns.discard()
                self.say("Sorry, I couldn't reach speech recognition.", "system", key="stt-fail", ttl=5)
                return
        self.stats["last_ms"] = round((time.monotonic() - t_end_speech) * 1000)
        if self.turns is not None:
            self.turns.transcript(time.monotonic(), text, early)
        self._deliver(text, require_wake)
        if early:                                   # let the socket finalize and close quietly
            try:
                await asyncio.wait_for(stt_task, timeout=3.0)
            except Exception:
                pass

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
        0.4 s pre-roll to STT and act on it only if it starts with the wake word. The wake
        word arrives as its own segment while the wearer is still talking: that is the
        moment to barge in, not the end of the sentence."""
        from .brain import strip_wake
        rate = self.cfg.sample_rate
        barged = False

        def on_segment(text: str, covered: float) -> None:
            nonlocal barged
            woke, rest = strip_wake(text, self.cfg.wake_words)
            if woke and not barged:
                barged = True
                self.stats["voice_barge_ins"] += 1
                self._set_listening(True)        # cut non-safety speech, hold the rest
            if woke:
                self._partial(text)

        try:
            with self._open_source() as src:
                ep = Endpointer(rate, self.cfg.end_silence_s, self.cfg.max_utterance_s, no_speech_s=1e9)
                preroll: list[bytes] = []
                block = int(rate * 0.1)
                pending = np.zeros(0, dtype=np.float32)
                chunks: asyncio.Queue | None = None
                task = None
                t_last_voice = None
                while not self._hf_stop.is_set():
                    data = src.get(0.3)
                    if data is None:
                        continue
                    pending = np.concatenate([pending, data])
                    while pending.size >= block:
                        b, pending = pending[:block], pending[block:]
                        ev = ep.push(b)
                        if ep.speaking and ep.quiet_s == 0.0:
                            t_last_voice = time.monotonic()
                        pcm = (np.clip(b, -1, 1) * 32767).astype("<i2").tobytes()
                        if chunks is None:
                            preroll.append(pcm)
                            del preroll[:-4]
                            if ev == "start":
                                chunks = asyncio.Queue()
                                for p in preroll:
                                    chunks.put_nowait(p)
                                task = asyncio.create_task(self.stt.transcribe_stream(chunks, on_segment=on_segment))
                        else:
                            await chunks.put(pcm)
                            self.stats["stt_audio_s"] += b.size / rate
                            if ev in ("end", "timeout"):
                                t_end = time.monotonic()
                                self._set_listening(False)
                                await chunks.put(None)
                                text = await asyncio.wait_for(task, timeout=6.0)
                                self.stats["last_ms"] = round((time.monotonic() - t_end) * 1000)
                                if barged and self.turns is not None:
                                    self.turns.begin(t_last_voice or t_end, t_end, "wake")
                                    self.turns.transcript(time.monotonic(), text)
                                self._deliver(text, require_wake=True)
                                return
                    await asyncio.sleep(0)
        finally:
            self._set_listening(False)

    def stop(self) -> None:
        self._hf_stop.set()
        self._stop_now.set()
