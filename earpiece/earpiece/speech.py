"""Speech output: what gets said, in what order, and how fast it reaches the ear.

Synthesis, fastest first:
  1. phrase cache   on-disk PCM of every safety phrase, pre-rendered at startup
                    (survives restarts, works offline): ~0 ms
  2. Cartesia       one persistent WebSocket, audio streamed into the mixer as it
                    arrives: ~100 ms to first sound (measured)
  3. macOS `say`    offline last resort for dynamic text: ~1.5 s

Scheduling rules (the part that keeps a blind user safe and sane):
  * priority order critical > hazard > nav > answer > info > system;
  * critical / hazard speech PREEMPTS anything less important already playing
    (cut with a 40 ms fade); preempted navigation is re-queued if still valid;
  * every item has a time-to-live and is DROPPED if it could not start in time:
    "turn left now" said three seconds late is worse than silence;
  * same key or same text within the dedupe window is spoken once; a newer item
    with the same key replaces the queued one;
  * while the wearer is talking, only critical speech plays.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .config import SpeechCfg
from .events import PRIORITY_RANK
from .mixer import Engine


@dataclass
class Utterance:
    text: str
    priority: str = "info"
    key: str | None = None
    ttl: float | None = None
    created: float = field(default_factory=time.monotonic)
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    requeue_on_preempt: bool = False
    on_done: Callable[[str, str], None] | None = None   # (uid, "done"|"cut"|"dropped")
    pre_sound: np.ndarray | None = None                  # stereo earcon to play just before
    cancelled: bool = False

    @property
    def rank(self) -> int:
        return PRIORITY_RANK.get(self.priority, len(PRIORITY_RANK))

    def expired(self, now: float, default_ttl: float) -> bool:
        ttl = self.ttl if self.ttl is not None else (None if self.priority in ("answer", "system")
                                                       else default_ttl)
        return ttl is not None and now - self.created > ttl


def _norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


# ---------------------------------------------------------------------- synthesis
class PhraseCache:
    """PCM float32 at the engine rate, keyed by (voice, model, speed, text)."""

    def __init__(self, cfg: SpeechCfg, rate: int):
        self.dir = Path(cfg.cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rate = rate
        self.tag = f"{cfg.cartesia_model}|{cfg.voice_id}|{cfg.speed}|{rate}"
        self._mem: dict[str, np.ndarray] = {}

    def _path(self, text: str, tag: str | None = None) -> Path:
        h = hashlib.sha1(f"{tag or self.tag}|{text}".encode(), usedforsecurity=False).hexdigest()[:20]
        return self.dir / f"{h}.f32"

    def get(self, text: str) -> np.ndarray | None:
        if text in self._mem:
            return self._mem[text]
        for tag in (self.tag, f"offline|{self.rate}"):
            p = self._path(text, tag)
            if p.exists():
                try:
                    a = np.fromfile(p, dtype=np.float32)
                except OSError:
                    continue
                if a.size:
                    if tag == self.tag:
                        self._mem[text] = a
                    return a
        return None

    def has_neural(self, text: str) -> bool:
        return text in self._mem or self._path(text).exists()

    def put(self, text: str, pcm: np.ndarray, offline: bool = False) -> None:
        tag = f"offline|{self.rate}" if offline else self.tag
        tmp = self._path(text, tag).with_suffix(".tmp")
        pcm.astype(np.float32).tofile(tmp)
        os.replace(tmp, self._path(text, tag))          # atomic: never a half-written file
        if not offline:
            self._mem[text] = pcm.astype(np.float32)


def offline_tts(text: str, rate: int, voice: str = "Samantha", wpm: int = 210) -> np.ndarray:
    """macOS `say` -> float32 mono at `rate`. Raises if unavailable."""
    if shutil.which("say") is None:
        raise RuntimeError("no offline TTS on this system")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = ["say", "-r", str(wpm), "-o", path, f"--data-format=LEI16@{rate}", text]
        if voice:
            cmd[1:1] = ["-v", voice]
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        if r.returncode != 0 and voice:          # unknown voice name: use the system default
            subprocess.run(["say", "-r", str(wpm), "-o", path, f"--data-format=LEI16@{rate}", text],
                           capture_output=True, timeout=15, check=True)
        with wave.open(path, "rb") as w:
            n = w.getnframes()
            a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() == 2:
                a = a.reshape(-1, 2).mean(axis=1)
            if w.getframerate() != rate:
                from scipy.signal import resample_poly
                g = np.gcd(rate, w.getframerate())
                a = resample_poly(a, rate // g, w.getframerate() // g).astype(np.float32)
        return a
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class CartesiaStream:
    """One persistent Cartesia TTS WebSocket shared by all utterances (each is a context)."""

    URL = "wss://api.cartesia.ai/tts/websocket?cartesia_version={v}"

    def __init__(self, cfg: SpeechCfg, key: str | None, rate: int):
        self.cfg = cfg
        self.key = key
        self.rate = rate
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._ctx: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self.down_until = 0.0            # back off after failures instead of hammering
        self.failures = 0
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.key) and time.monotonic() >= self.down_until

    async def _connect(self):
        import websockets

        async with self._lock:
            if self._ws is not None:
                return self._ws
            self._ws = await asyncio.wait_for(
                websockets.connect(self.URL.format(v=self.cfg.cartesia_version),
                                   additional_headers={"X-API-Key": self.key},
                                   max_size=None, ping_interval=15, ping_timeout=10),
                timeout=self.cfg.connect_timeout_s)
            self._reader = asyncio.create_task(self._read(self._ws))
            return self._ws

    async def _read(self, ws) -> None:
        try:
            async for raw in ws:
                try:
                    m = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                q = self._ctx.get(m.get("context_id", ""))
                if q is not None:
                    q.put_nowait(m)
        except Exception as e:                 # connection dropped
            self.last_error = f"{type(e).__name__}: {e}"
        finally:
            if self._ws is ws:
                self._ws = None
            for q in self._ctx.values():
                q.put_nowait({"type": "error", "message": "connection closed"})

    def _fail(self, why: str) -> None:
        self.failures += 1
        self.last_error = why
        self.down_until = time.monotonic() + min(30.0, 2.0 * self.failures)

    async def stream(self, text: str, ctx_id: str):
        """Async generator of float32 mono chunks. Raises on failure / first-chunk timeout."""
        ws = await self._connect()
        q: asyncio.Queue = asyncio.Queue()
        self._ctx[ctx_id] = q
        req = {"model_id": self.cfg.cartesia_model, "transcript": text, "voice": self.cfg.voice_id,
               "language": "en", "context_id": ctx_id, "continue": False,
               "output_format": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": self.rate},
               "generation_config": {"speed": self.cfg.speed}}
        try:
            await ws.send(json.dumps(req))
            first = True
            while True:
                timeout = self.cfg.first_chunk_timeout_s if first else 5.0
                try:
                    m = await asyncio.wait_for(q.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    self._fail("timeout waiting for audio")
                    raise RuntimeError("cartesia timeout")
                t = m.get("type")
                if t == "chunk" and m.get("data"):
                    first = False
                    yield np.frombuffer(base64.b64decode(m["data"]), dtype=np.float32)
                elif t == "done":
                    self.failures = 0
                    return
                elif t == "error":
                    self._fail(str(m.get("message") or m.get("title") or "error"))
                    raise RuntimeError(f"cartesia: {self.last_error}")
        finally:
            self._ctx.pop(ctx_id, None)

    async def cancel(self, ctx_id: str) -> None:
        ws = self._ws
        if ws is not None:
            try:
                await ws.send(json.dumps({"context_id": ctx_id, "cancel": True}))
            except Exception:
                pass

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass


# ---------------------------------------------------------------------- manager
SAFETY_KINDS = ("Person", "Car", "Bike", "Obstacle", "Step down", "Curb")
FIXED_PHRASES = (
    "Stop.", "Stop. Person ahead.", "Stop. Obstacle ahead.", "Stop. Car.", "Careful.",
    "Obstacle detection paused.", "Obstacle detection is back.",
    "Voice commands are offline.", "Listening.", "Sorry, I didn't catch that.",
    "Guidance paused.", "Guidance resumed.", "Beeps muted.", "Beeps on.",
    "You have arrived.", "Don't walk.", "Walk signal is on.", "Crowd ahead.",
    "AirPods disconnected. Using the laptop speaker.", "AirPods connected.",
    "Kite is ready.", "Left ear.", "Right ear.", "Louder.", "Quieter.",
    "No internet. Using the backup voice.", "Internet is back.",
    "Turn around.", "On course.",
)


def safety_phrases() -> list[str]:
    """Every phrase the sonar can say, so none of them ever waits on the network."""
    from .events import clock_phrase
    dirs = sorted({clock_phrase(b) for b in range(-180, 181)})
    return list(FIXED_PHRASES) + [f"{k}, {d}." for k in SAFETY_KINDS for d in dirs]


class Speaker:
    def __init__(self, engine: Engine, cfg: SpeechCfg, key: str | None, log=print):
        self.engine = engine
        self.cfg = cfg
        self.log = log
        self.cache = PhraseCache(cfg, engine.rate)
        self.tts = CartesiaStream(cfg, key, engine.rate)
        self._queue: list[Utterance] = []
        self._qlock = threading.Lock()
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._current: Utterance | None = None
        self._recent: deque[tuple[float, str, str | None]] = deque(maxlen=64)   # (t, norm text, key)
        self.history: deque[dict] = deque(maxlen=50)
        self.last_nav: Utterance | None = None
        self.listening = False             # wearer is talking: hold non-critical speech
        self.paused = False                # guidance paused by the wearer
        self.online: bool | None = None
        self._seq = itertools.count()
        self.stats = {"spoken": 0, "dropped_ttl": 0, "deduped": 0, "preempted": 0,
                      "cache_hits": 0, "neural": 0, "offline": 0, "first_audio_ms": []}
        self.on_spoken: Callable[[dict], None] | None = None

    # ------------------------------------------------------------------ public API
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, name="speech", daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._loop is None and time.time() - t0 < 2:
            time.sleep(0.005)

    def say(self, text: str, priority: str = "info", key: str | None = None,
            ttl: float | None = None, requeue: bool | None = None,
            on_done: Callable[[str, str], None] | None = None,
            pre_sound: np.ndarray | None = None) -> str | None:
        """Thread-safe. Returns the utterance id, or None if deduplicated."""
        text = " ".join(text.split())
        if not text:
            return None
        u = Utterance(text, priority, key, ttl, requeue_on_preempt=(priority == "nav") if requeue is None
                      else requeue, on_done=on_done, pre_sound=pre_sound)
        now = time.monotonic()
        nt = _norm_text(text)
        with self._qlock:
            if priority != "critical":
                for t, txt, k in self._recent:
                    if now - t < self.cfg.dedupe_window_s and (txt == nt or (key and k == key)):
                        self.stats["deduped"] += 1
                        return None
                cur = self._current
                if cur is not None and (_norm_text(cur.text) == nt or (key and cur.key == key)):
                    self.stats["deduped"] += 1
                    return None
            same = [q for q in self._queue if _norm_text(q.text) == nt]
            if same and not u.key:                    # identical text: keep the queued item's key
                u.key = key = same[0].key
            if key:                                   # newer version replaces the queued one
                self._queue = [q for q in self._queue if q.key != key]
            self._queue = [q for q in self._queue if _norm_text(q.text) != nt]
            self._queue.append(u)
            self._queue.sort(key=lambda q: (q.rank, q.created))
            cur = self._current
            preempt = (cur is not None and u.rank < cur.rank and priority in ("critical", "hazard"))
        if preempt:
            self._call(self._preempt, cur)
        self._kick()
        return u.uid

    def cancel_all(self, keep_critical: bool = True) -> None:
        """Flush the queue and cut the current utterance (e.g. wearer said 'stop talking')."""
        with self._qlock:
            dropped = [q for q in self._queue if not (keep_critical and q.priority == "critical")]
            self._queue = [q for q in self._queue if keep_critical and q.priority == "critical"]
            cur = self._current
        for q in dropped:
            self._notify(q, "dropped")
        if cur is not None and not (keep_critical and cur.priority == "critical"):
            cur.cancelled = True
            cur.requeue_on_preempt = False
            self.engine.speech_cancel(cur.uid)

    def drop_where(self, pred: Callable[[Utterance], bool]) -> None:
        with self._qlock:
            gone = [q for q in self._queue if pred(q)]
            self._queue = [q for q in self._queue if not pred(q)]
        for q in gone:
            self._notify(q, "dropped")

    def set_listening(self, on: bool) -> None:
        """While the wearer talks, hold everything except critical and cut what is playing."""
        self.listening = on
        if on:
            cur = self._current
            if cur is not None and cur.priority != "critical":
                self._call(self._preempt, cur)
        else:
            self._kick()

    def repeat_last(self) -> bool:
        if self.last_nav is None:
            return False
        self._recent.clear()
        self.say(self.last_nav.text, "nav", key=None, ttl=30)
        return True

    @property
    def busy(self) -> bool:
        with self._qlock:
            return self._current is not None or bool(self._queue)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        t0 = time.time()
        while self.busy and time.time() - t0 < timeout:
            time.sleep(0.02)
        return not self.busy

    def prewarm(self, phrases: list[str] | None = None) -> None:
        """Render the safety phrases into the cache in the background."""
        self._call(self._prewarm, phrases or safety_phrases())

    def stop(self) -> None:
        if self._loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self.tts.close(), self._loop)
            try:
                fut.result(timeout=2)
            except Exception:
                pass
            def _shutdown():
                for t in asyncio.all_tasks(self._loop):
                    t.cancel()
                self._loop.call_soon(self._loop.stop)
            self._loop.call_soon_threadsafe(_shutdown)

    # ------------------------------------------------------------------ internals
    def _call(self, coro_fn, *args) -> None:
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(coro_fn(*args), self._loop)

    def _kick(self) -> None:
        if self._loop is not None and self._wake is not None:
            self._loop.call_soon_threadsafe(self._wake.set)

    def _notify(self, u: Utterance, how: str) -> None:
        if u.on_done is not None:
            try:
                u.on_done(u.uid, how)
            except Exception as e:
                self.log(f"[speech] on_done callback failed: {e}")

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._wake = asyncio.Event()
        self._loop = loop
        loop.create_task(self._consumer())
        loop.run_forever()

    async def _preempt(self, cur: Utterance) -> None:
        if self._current is not cur:
            return
        self.stats["preempted"] += 1
        cur.cancelled = True
        self.engine.speech_cancel(cur.uid, fade_s=0.04)
        await self.tts.cancel(cur.uid)
        # re-queueing happens in _consumer, and only if the line really was cut off: a line
        # that finished in the same few milliseconds must not be said twice

    def _requeue(self, cur: Utterance) -> None:
        if cur.requeue_on_preempt and not cur.expired(time.monotonic(), self.cfg.default_ttl_s):
            again = Utterance(cur.text, cur.priority, cur.key, cur.ttl, created=cur.created,
                              requeue_on_preempt=False, on_done=cur.on_done)
            with self._qlock:
                self._queue.append(again)
                self._queue.sort(key=lambda q: (q.rank, q.created))

    def _next(self) -> Utterance | None:
        now = time.monotonic()
        with self._qlock:
            while self._queue:
                if self.listening or self.paused:
                    allowed = [q for q in self._queue if q.priority == "critical"
                               or (self.paused and not self.listening and q.priority in ("hazard", "answer", "system"))]
                    if not allowed:
                        return None
                    u = allowed[0]
                    self._queue.remove(u)
                else:
                    u = self._queue.pop(0)
                if u.expired(now, self.cfg.default_ttl_s):
                    self.stats["dropped_ttl"] += 1
                    self._notify(u, "dropped")
                    continue
                self._current = u
                self._recent.append((now, _norm_text(u.text), u.key))
                return u
        return None

    async def _consumer(self) -> None:
        while True:
            u = self._next()
            if u is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass
                continue
            how = "cut"
            try:
                how = await self._speak(u)
            except Exception as e:
                self.log(f"[speech] failed ({type(e).__name__}: {e}): {u.text!r}")
            finally:
                with self._qlock:
                    self._current = None
            if how == "cut" and u.cancelled:
                self._requeue(u)
            if how == "done":
                self.stats["spoken"] += 1
                if u.priority == "nav":
                    self.last_nav = u
            rec = {"t": time.time(), "text": u.text, "priority": u.priority, "result": how}
            self.history.append(rec)
            if self.on_spoken is not None:
                try:
                    self.on_spoken(rec)
                except Exception:
                    pass
            self._notify(u, how)

    async def _await_end(self, uid: str, max_s: float) -> str:
        """Wait for the mixer to report this utterance finished playing or was cut."""
        t_end = time.monotonic() + max_s
        while time.monotonic() < t_end:
            try:
                ev, eid, _ = self.engine.speech_events.get_nowait()
            except Exception:
                await asyncio.sleep(0.01)
                continue
            if eid == uid and ev in ("done", "cut"):
                return ev
        self.engine.speech_cancel(uid)
        return "cut"

    async def _speak(self, u: Utterance) -> str:
        t0 = time.monotonic()
        if u.pre_sound is not None:
            self.engine.play(u.pre_sound, bus="alert" if u.priority == "critical" else "cue")
            await asyncio.sleep(len(u.pre_sound) / self.engine.rate * 0.8)
        self.engine.speech_begin(u.uid)
        pcm = self.cache.get(u.text)
        if pcm is not None:
            self.stats["cache_hits"] += 1
            self.engine.speech_feed(u.uid, pcm)
            self.engine.speech_end(u.uid)
            return await self._await_end(u.uid, pcm.size / self.engine.rate + 3.0)

        total = 0.0
        if self.tts.available:
            chunks = []
            try:
                async for c in self.tts.stream(u.text, u.uid):
                    if u.cancelled:                     # preempted while streaming
                        await self.tts.cancel(u.uid)
                        return "cut"
                    if not chunks:
                        self.stats["first_audio_ms"].append((time.monotonic() - t0) * 1000)
                        del self.stats["first_audio_ms"][:-50]
                    chunks.append(c)
                    total += c.size
                    self.engine.speech_feed(u.uid, c)
                self.engine.speech_end(u.uid)
                self.stats["neural"] += 1
                self._set_online(True)
                if len(u.text) <= 60 and chunks:          # short phrases are worth keeping offline
                    self.cache.put(u.text, np.concatenate(chunks))
                return await self._await_end(u.uid, total / self.engine.rate + 3.0)
            except Exception as e:
                self.log(f"[speech] cartesia failed ({e}); using the offline voice")
                self._set_online(False)
                if total > 0:              # audio already started: do not restart the sentence
                    self.engine.speech_end(u.uid)
                    return await self._await_end(u.uid, total / self.engine.rate + 3.0)
        if u.expired(time.monotonic(), self.cfg.default_ttl_s):
            self.engine.speech_cancel(u.uid)
            return "dropped"
        try:
            pcm = await asyncio.to_thread(offline_tts, u.text, self.engine.rate, self.cfg.offline_voice)
        except Exception as e:
            self.log(f"[speech] offline TTS failed too ({e})")
            self.engine.speech_cancel(u.uid)
            return "cut"
        if u.cancelled:
            return "cut"
        self.stats["offline"] += 1
        self.engine.speech_feed(u.uid, pcm)
        self.engine.speech_end(u.uid)
        return await self._await_end(u.uid, pcm.size / self.engine.rate + 3.0)

    def _set_online(self, ok: bool) -> None:
        if self.online is not None and ok != self.online:
            self.log(f"[speech] neural voice {'back online' if ok else 'offline'}")
        self.online = ok

    async def _prewarm(self, phrases: list[str]) -> None:
        missing = [p for p in phrases if not self.cache.has_neural(p)]
        if not missing:
            return
        t0 = time.monotonic()
        done = 0
        if self.tts.available:
            sem = asyncio.Semaphore(4)

            async def one(p: str) -> None:
                nonlocal done
                async with sem:
                    try:
                        chunks = [c async for c in self.tts.stream(p, "warm-" + uuid.uuid4().hex[:10])]
                        if chunks:
                            self.cache.put(p, np.concatenate(chunks))
                            done += 1
                    except Exception:
                        pass

            await asyncio.gather(*(one(p) for p in missing))
        if done < len(missing):
            # no neural voice: make sure the critical phrases at least exist offline
            for p in phrases:
                if self.cache.get(p) is None:
                    try:
                        self.cache.put(p, await asyncio.to_thread(
                            offline_tts, p, self.engine.rate, self.cfg.offline_voice), offline=True)
                    except Exception:
                        break
        self.log(f"[speech] phrase cache: {done}/{len(missing)} new neural phrases "
                 f"in {time.monotonic() - t0:.1f}s")
