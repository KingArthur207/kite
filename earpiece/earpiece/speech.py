"""Speech output: what gets said, in what order, and how fast it reaches the ear.

Synthesis, fastest first:
  1. phrase cache   on-disk PCM of every safety phrase, pre-rendered at startup
                    (survives restarts, works offline): ~0 ms
  2. Cartesia       one persistent WebSocket, audio streamed into the mixer as it
                    arrives: ~100 ms to first sound (measured)
  3. macOS `say`    offline last resort for dynamic text: 1.5 s or more

Scheduling rules (the part that keeps a blind user safe and sane):
  * priority order critical > hazard > nav > answer > info > system;
  * critical / hazard speech PREEMPTS anything less important already playing
    (cut with a 40 ms fade); preempted navigation is re-queued if still valid;
  * every item has a time-to-live and is DROPPED if it could not start in time:
    "turn left now" said three seconds late is worse than silence;
  * same key or same text within the dedupe window is spoken once; a newer item
    with the same key replaces the queued one;
  * BARGE-IN: when the wearer presses to talk (or says the wake word) anything that
    is not a safety warning stops at once and queued replies to the previous
    question are forgotten; while they talk, only critical speech plays.

The voice itself:
  * speaking rate is the wearer's choice ("talk faster" / "slower"), 0.7x to 1.5x
    (Sonic 3.6 accepts 0.6 to 1.5), remembered across runs. Cached safety phrases
    recorded at another rate still play instantly and are re-recorded in the background;
  * street names the voices get wrong ("Lyon Street" is LIE-un in San Francisco) are
    respelled in the text sent to the synthesiser only;
  * outages: a failed connection backs off and is retried by a background probe (a
    WebSocket handshake costs no credits), so live speech never waits on a dead network;
    the wearer hears once that the backup voice is in use, and again when it is back.
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

SAFETY = ("critical", "hazard")          # never cut by barge-in, never wait on the network
REPLIES = ("answer", "system")           # what the wearer hears in response to speaking
SPEED_MIN, SPEED_MAX, SPEED_STEP = 0.7, 1.5, 0.1


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


def _median(xs) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    return round(xs[len(xs) // 2], 1) if xs else None


# ---------------------------------------------------------------------- pronunciation
# Street names the voices read wrong near the venue and around San Francisco. Respellings
# rather than IPA so the same table also fixes the offline voice; applied only to the text
# that is synthesised (history, dedupe and the dashboard keep the real spelling). A Cartesia
# pronunciation dictionary can be layered on top with KITE_CARTESIA_PRONUNCIATION_DICT.
PRONUNCIATIONS: tuple[tuple[str, str], ...] = (
    (r"\bLyon(?=\s+St(?:reet)?\b)", "Lion"),           # LIE-un, not the French city
    (r"\bGough\b", "Goff"),
    (r"\bKearny\b", "Kerny"),                           # KUR-nee
    (r"\bNoe(?=\s+(?:St(?:reet)?|Valley)\b)", "Noey"),
)
_PRON = [(re.compile(p), r) for p, r in PRONUNCIATIONS]


def pronounce(text: str) -> str:
    """The text as the synthesiser should receive it."""
    for pat, rep in _PRON:
        text = pat.sub(rep, text)
    return text


# ---------------------------------------------------------------------- synthesis
class PhraseCache:
    """PCM float32 at the engine rate, keyed by (model, voice, speed, rate, text).

    `lookup` falls back from the current speaking rate to the default-rate copy, then to
    the offline voice, so a phrase that exists in any form plays at once."""

    def __init__(self, cfg: SpeechCfg, rate: int, extra_tag: str = ""):
        self.cfg = cfg
        self.dir = Path(cfg.cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rate = rate
        self.speed = round(float(cfg.speed), 2)
        self.extra = extra_tag
        self._mem: dict[tuple[str, str], np.ndarray] = {}

    def tag_for(self, speed: float) -> str:
        t = f"{self.cfg.cartesia_model}|{self.cfg.voice_id}|{round(float(speed), 2)}|{self.rate}"
        return t + (f"|{self.extra}" if self.extra else "")

    @property
    def tag(self) -> str:
        return self.tag_for(self.speed)

    @property
    def offline_tag(self) -> str:
        return f"offline|{self.rate}"

    def _path(self, text: str, tag: str | None = None) -> Path:
        h = hashlib.sha1(f"{tag or self.tag}|{text}".encode(), usedforsecurity=False).hexdigest()[:20]
        return self.dir / f"{h}.f32"

    def lookup(self, text: str, offline: bool = True) -> tuple[np.ndarray | None, str]:
        """(pcm, kind): kind is 'neural' (current rate), 'stale' (default rate), 'offline' or 'none'."""
        tags = [(self.tag, "neural")]
        base = self.tag_for(self.cfg.speed)
        if base != self.tag:
            tags.append((base, "stale"))
        if offline:
            tags.append((self.offline_tag, "offline"))
        for tag, kind in tags:
            k = (tag, text)
            if k in self._mem:
                return self._mem[k], kind
            p = self._path(text, tag)
            if p.exists():
                try:
                    a = np.fromfile(p, dtype=np.float32)
                except OSError:
                    continue
                if a.size:
                    if kind != "offline":
                        self._mem[k] = a
                    return a, kind
        return None, "none"

    def get(self, text: str) -> np.ndarray | None:
        return self.lookup(text)[0]

    def has_neural(self, text: str) -> bool:
        return (self.tag, text) in self._mem or self._path(text).exists()

    def put(self, text: str, pcm: np.ndarray, offline: bool = False, tag: str | None = None) -> None:
        tag = self.offline_tag if offline else (tag or self.tag)
        tmp = self._path(text, tag).with_suffix(".tmp")
        pcm.astype(np.float32).tofile(tmp)
        os.replace(tmp, self._path(text, tag))          # atomic: never a half-written file
        if not offline:
            self._mem[(tag, text)] = pcm.astype(np.float32)


def offline_tts(text: str, rate: int, voice: str = "Samantha", wpm: int = 210) -> np.ndarray:
    """macOS `say` -> float32 mono at `rate`. Raises if unavailable.

    stdin is closed on purpose: `say` inheriting a pipe (a launcher, a test runner) hangs
    until its timeout, and inheriting the terminal would let it race the keyboard controls."""
    if shutil.which("say") is None:
        raise RuntimeError("no offline TTS on this system")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = ["say", "-r", str(wpm), "-o", path, f"--data-format=LEI16@{rate}", text]
        if voice:
            cmd[1:1] = ["-v", voice]
        r = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=15)
        if r.returncode != 0 and voice:          # unknown voice name: use the system default
            subprocess.run(["say", "-r", str(wpm), "-o", path, f"--data-format=LEI16@{rate}", text],
                           capture_output=True, stdin=subprocess.DEVNULL, timeout=15, check=True)
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


def offline_wpm(speed: float) -> int:
    """The backup voice follows the wearer's rate (1.1x is about 210 words a minute)."""
    return int(round(190 * speed))


class UsageMeter:
    """Characters sent to Cartesia, persisted across runs so a hackathon team can see (and
    cap) its total burn. Written at most every few seconds; never on the audio path."""

    def __init__(self, path: Path, budget: int):
        self.path = Path(path)
        self.budget = budget
        self.session = 0
        self.total = 0
        self._dirty_t = 0.0
        try:
            self.total = int(json.loads(self.path.read_text()).get("chars", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            self.total = 0

    def add(self, n: int) -> None:
        self.session += n
        self.total += n
        if time.monotonic() - self._dirty_t > 3.0:
            self.flush()

    def flush(self) -> None:
        self._dirty_t = time.monotonic()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"chars": self.total, "updated": time.time()}))
            os.replace(tmp, self.path)
        except OSError:
            pass

    @property
    def over_budget(self) -> bool:
        return self.budget > 0 and self.total >= self.budget


class VoiceTurns:
    """Where the time goes between the wearer finishing a sentence and Kite answering.

      endpoint  last voiced audio -> end of utterance decided (silence wait or early end)
      stt       end of utterance -> final transcript in hand
      brain     transcript -> first reply line queued (local intent or Claude's first sentence)
      tts       reply queued -> its first audio handed to the mixer (cache, Cartesia or offline)
      total     last voiced audio -> first audio of the reply

    Thread-safe: the listener, the brain and the speech loop each report their stage."""

    STAGES = ("endpoint_ms", "stt_ms", "brain_ms", "tts_ms", "total_ms")

    def __init__(self, keep: int = 30):
        self._lock = threading.Lock()
        self._open: dict | None = None
        self.turns: deque[dict] = deque(maxlen=keep)
        self.abandoned = 0                 # turns that never got a spoken reply (e.g. "stop")

    def begin(self, t_voice_end: float, t_endpoint: float, mode: str = "ptt") -> None:
        with self._lock:
            if self._open is not None:
                self.abandoned += 1
            self._open = {"t_voice": t_voice_end, "t_end": t_endpoint, "mode": mode}

    def transcript(self, t: float, text: str = "", early: bool = False) -> None:
        with self._lock:
            o = self._open
            if o is not None and "t_text" not in o:
                o.update(t_text=t, text=text[:80], early=early)

    def reply_queued(self, uid: str, t: float) -> None:
        with self._lock:
            o = self._open
            if o is not None and "t_text" in o and "uid" not in o:
                o.update(uid=uid, t_reply=t)

    def discard(self) -> None:
        with self._lock:
            if self._open is not None:
                self.abandoned += 1
            self._open = None

    def audio(self, uid: str, t: float, source: str) -> dict | None:
        with self._lock:
            o = self._open
            if o is None or o.get("uid") != uid:
                return None
            self._open = None
            ms = lambda a, b: round((b - a) * 1000, 1)      # noqa: E731
            rec = {"mode": o["mode"], "text": o.get("text", ""), "source": source, "early_end": o["early"],
                   "endpoint_ms": ms(o["t_voice"], o["t_end"]), "stt_ms": ms(o["t_end"], o["t_text"]),
                   "brain_ms": ms(o["t_text"], o["t_reply"]), "tts_ms": ms(o["t_reply"], t),
                   "total_ms": ms(o["t_voice"], t)}
            self.turns.append(rec)
            return rec

    def summary(self) -> dict:
        with self._lock:
            turns = list(self.turns)
        return {"turns": len(turns), "abandoned": self.abandoned, "last": turns[-1] if turns else None,
                "median": {s: _median(t[s] for t in turns) for s in self.STAGES} if turns else None}


class ConcurrencyLimited(RuntimeError):
    """Cartesia refused a request because too many were in flight (our plan allows 2).
    Transient: retry shortly instead of falling back to the slow offline voice."""


class SlowFirstChunk(RuntimeError):
    """An urgent line's first audio did not arrive within its short deadline. Not an outage:
    the backup voice says this line, the network keeps its standing."""


class CartesiaStream:
    """One persistent Cartesia TTS WebSocket shared by all utterances (each is a context)."""

    URL = "wss://api.cartesia.ai/tts/websocket?cartesia_version={v}"
    QUOTA_BACKOFF_S = 300.0               # out of credits: do not retry every utterance
    CONNECT_BACKOFF_S = 30.0              # no handshake: wait for the probe rather than retry inline

    def __init__(self, cfg: SpeechCfg, key: str | None, rate: int, pronunciation_dict: str | None = None):
        self.cfg = cfg
        self.key = key
        self.rate = rate
        self.pronunciation_dict = pronunciation_dict
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._ctx: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self.down_until = 0.0            # back off after failures instead of hammering
        self.meter: "UsageMeter | None" = None
        self.failures = 0
        self.last_error: str | None = None
        self.quota_exhausted = False
        self.ttfb_ms: deque[float] = deque(maxlen=50)    # request sent -> first audio chunk

    @property
    def available(self) -> bool:
        return bool(self.key) and time.monotonic() >= self.down_until

    @property
    def connected(self) -> bool:
        return self._ws is not None

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

    async def probe(self) -> bool:
        """Open the socket without synthesising anything (no credits). True when it is up."""
        if not self.key:
            return False
        try:
            await self._connect()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return False
        self.down_until = 0.0
        self.failures = 0
        return True

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

    def _fail(self, why: str, backoff_s: float | None = None) -> None:
        self.failures += 1
        self.last_error = why
        self.down_until = time.monotonic() + (backoff_s if backoff_s is not None
                                              else min(30.0, 2.0 * self.failures))

    def request(self, text: str, ctx_id: str, speed: float | None = None) -> dict:
        req = {"model_id": self.cfg.cartesia_model, "transcript": text, "voice": self.cfg.voice_id,
               "language": "en", "context_id": ctx_id, "continue": False,
               "output_format": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": self.rate},
               "generation_config": {"speed": round(float(self.cfg.speed if speed is None else speed), 2)}}
        if self.pronunciation_dict:
            req["pronunciation_dict_id"] = self.pronunciation_dict
        return req

    async def stream(self, text: str, ctx_id: str, speed: float | None = None,
                     first_timeout: float | None = None):
        """Async generator of float32 mono chunks. Raises on failure / first-chunk timeout."""
        try:
            ws = await self._connect()
        except Exception as e:                 # no network / DNS / TLS: back off; the speaker's
            # background probe lifts this the moment a handshake succeeds again
            self._fail(f"connect: {type(e).__name__}: {e}", self.CONNECT_BACKOFF_S)
            raise RuntimeError(f"cartesia connect failed: {type(e).__name__}") from e
        q: asyncio.Queue = asyncio.Queue()
        self._ctx[ctx_id] = q
        try:
            await ws.send(json.dumps(self.request(text, ctx_id, speed)))
            t_sent = time.monotonic()
            if self.meter is not None:
                self.meter.add(len(text))
            first = True
            while True:
                timeout = (first_timeout or self.cfg.first_chunk_timeout_s) if first else 5.0
                try:
                    m = await asyncio.wait_for(q.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    if first and first_timeout is not None and first_timeout < self.cfg.first_chunk_timeout_s:
                        raise SlowFirstChunk(f"no audio within {first_timeout:.1f}s")
                    self._fail("timeout waiting for audio")
                    raise RuntimeError("cartesia timeout")
                t = m.get("type")
                if t == "chunk" and m.get("data"):
                    if first:
                        self.ttfb_ms.append((time.monotonic() - t_sent) * 1000)
                    first = False
                    yield np.frombuffer(base64.b64decode(m["data"]), dtype=np.float32)
                elif t == "done":
                    self.failures = 0
                    return
                elif t == "error":
                    msg = str(m.get("message") or m.get("title") or "error")
                    code = str(m.get("error_code") or "")
                    if code == "concurrency_limited" or m.get("status_code") == 429 or "concurrency" in msg.lower():
                        raise ConcurrencyLimited(msg)       # not an outage: no backoff
                    if code == "quota_exceeded" or "quota" in msg.lower() or "credit" in msg.lower():
                        self.quota_exhausted = True
                        self._fail(f"quota: {msg}", self.QUOTA_BACKOFF_S)
                    else:
                        self._fail(msg)
                    raise RuntimeError(f"cartesia: {self.last_error}")
        finally:
            self._ctx.pop(ctx_id, None)

    async def cancel(self, ctx_id: str) -> None:
        # Cartesia only cancels work not yet started; chunks already in flight are dropped
        # by the caller (the context queue is gone), so nothing late reaches the ear.
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
SAFETY_KINDS = ("Person", "Car", "Bike", "Obstacle", "Step down", "Curb", "Crowd", "Overhead obstacle",
                "Construction")
FIXED_PHRASES = (
    "Stop.", "Stop. Person ahead.", "Stop. Obstacle ahead.", "Stop. Car.", "Careful.",
    "Obstacle detection paused.", "Obstacle detection is back.",
    "Voice commands are offline.", "Listening.", "Sorry, I didn't catch that.",
    "Guidance paused.", "Guidance resumed.", "Beeps muted.", "Beeps on.",
    "You have arrived.", "Don't walk.", "Walk signal is on.", "Crowd ahead.",
    "AirPods disconnected. Using the laptop speaker.", "AirPods connected.",
    "Headphones disconnected. Using the laptop speaker.", "Headphones connected.",
    "Hold it steady. Reading.", "Reading text needs the internet right now.",
    "I can't see any text. Try holding it closer and a little steadier.",
    "Kite is ready.", "Left ear.", "Right ear.", "Louder.", "Quieter.",
    "No internet. Using the backup voice.", "Internet is back.",
    "Turn around.", "On course.",
    "Faster.", "Slower.", "Normal speed.", "That's as fast as I go.", "That's as slow as I go.",
    # the spec's fixed phrases (section 5.6): cached once, never streamed
    "Wait.", "Curb ahead.", "Bear left.", "Bear right.", "Path is clear. You can continue.",
    "Crowd ahead. Wait.", "Car ahead. Wait.", "Overhead obstacle ahead.",
)
VOICE_OFFLINE, VOICE_BACK = "No internet. Using the backup voice.", "Internet is back."


def safety_phrases() -> list[str]:
    """Every phrase the sonar can say, so none of them ever waits on the network."""
    from .events import clock_phrase
    dirs = sorted({clock_phrase(b) for b in range(-180, 181)})
    return list(FIXED_PHRASES) + [f"{k}, {d}." for k in SAFETY_KINDS for d in dirs]


class Speaker:
    PROBE_EVERY_S = 5.0          # while the neural voice is down, try a free handshake this often
    ANNOUNCE_GAP_S = 60.0        # at most one "backup voice" / "internet is back" per minute
    URGENT_FIRST_CHUNK_S = 1.0   # uncached warnings: give Cartesia this long, then the backup voice

    def __init__(self, engine: Engine, cfg: SpeechCfg, key: str | None, log=print):
        self.engine = engine
        self.cfg = cfg
        self.log = log
        pdict = os.environ.get("KITE_CARTESIA_PRONUNCIATION_DICT", "").strip() or None
        self.cache = PhraseCache(cfg, engine.rate, extra_tag=f"dict={pdict}" if pdict else "")
        self.tts = CartesiaStream(cfg, key, engine.rate, pronunciation_dict=pdict)
        self.meter = UsageMeter(Path(cfg.cache_dir).parent / "cartesia_usage.json", cfg.char_budget)
        self.tts.meter = self.meter
        self.prefs_path = Path(cfg.cache_dir).parent / "voice_prefs.json"
        self.cache.speed = self._load_speed()
        if key:
            self.log(f"[speech] Cartesia usage on this machine so far: {self.meter.total:,} characters"
                     + (f" of a {self.meter.budget:,} budget" if self.meter.budget else ""))
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
        self._announced_t = -1e9
        self._told_offline = False
        self._seq = itertools.count()
        self._warm_lock: asyncio.Lock | None = None
        self._warm_gen = 0
        self._stale_refresh: set[str] = set()
        self.turns = VoiceTurns()
        self.stats = {"spoken": 0, "dropped_ttl": 0, "deduped": 0, "preempted": 0, "barge_ins": 0,
                      "cache_hits": 0, "stale_rate_hits": 0, "neural": 0, "offline": 0, "first_audio_ms": [],
                      "speed": self.speed, "tts_first_byte_ms": None, "voice_loop": self.turns.summary()}
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
            replaced = [q for q in self._queue if (key and q.key == key) or _norm_text(q.text) == nt]
            if replaced:                              # newer version replaces the queued one
                self._queue = [q for q in self._queue if q not in replaced]
            self._queue.append(u)
            self._queue.sort(key=lambda q: (q.rank, q.created))
            replaced += self._shed_locked()
            cur = self._current
            preempt = (cur is not None and u.rank < cur.rank and priority in ("critical", "hazard"))
        if priority in REPLIES:
            self.turns.reply_queued(u.uid, now)
        for q in replaced:
            self._notify(q, "dropped")
        if preempt:
            self._call(self._preempt, cur)
        self._kick()
        return u.uid

    MAX_QUEUE = 32           # queued lines beyond this are shed, least important and oldest first
    MAX_QUEUE_CRITICAL = 128  # critical lines are only shed past this (a flood, never real traffic)

    def _shed_locked(self) -> list[Utterance]:
        """Keep the queue bounded (caller holds _qlock; the queue is sorted). "answer" and
        "system" lines have no default TTL, so without this a flood of distinct lines from the
        network would queue forever and be read out for hours, with say() getting slower each
        time. Sheds the oldest of the least important lines; critical lines only past a hard cap."""
        shed: list[Utterance] = []
        while len(self._queue) > self.MAX_QUEUE:
            pool = [q for q in self._queue if q.priority != "critical"]
            if not pool:
                if len(self._queue) <= self.MAX_QUEUE_CRITICAL:
                    break
                pool = self._queue
            victim = max(pool, key=lambda q: (q.rank, -q.created))
            self._queue.remove(victim)
            shed.append(victim)
        self.stats["shed"] = self.stats.get("shed", 0) + len(shed)
        return shed

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

    def barge_in(self) -> bool:
        """The wearer wants to talk. Stop non-safety speech now (40 ms fade, from the calling
        thread: no hop through the speech loop) and forget queued replies and chatter from
        before; they belong to a conversation the wearer has moved on from. Warnings are never
        cut, and a cut navigation line comes back afterwards if it is still fresh.
        Returns True if something was cut."""
        with self._qlock:
            gone = [q for q in self._queue if q.priority in ("answer", "info", "system")]
            if gone:
                self._queue = [q for q in self._queue if q not in gone]
            cur = self._current
        for q in gone:
            self._notify(q, "dropped")
        if cur is None or cur.priority in SAFETY or cur.cancelled:
            return False
        if cur.priority != "nav":
            cur.requeue_on_preempt = False
        cur.cancelled = True
        self.engine.speech_cancel(cur.uid, fade_s=0.04)
        self.stats["barge_ins"] += 1
        self._call(self.tts.cancel, cur.uid)
        return True

    def drop_where(self, pred: Callable[[Utterance], bool]) -> None:
        with self._qlock:
            gone = [q for q in self._queue if pred(q)]
            self._queue = [q for q in self._queue if not pred(q)]
        for q in gone:
            self._notify(q, "dropped")

    def set_listening(self, on: bool) -> None:
        """While the wearer talks, hold everything except critical. Starting to talk is a
        barge-in: non-safety speech stops; a warning already playing finishes."""
        self.listening = on
        if on:
            self.barge_in()
            self.warm()
        else:
            self._kick()

    def repeat_last(self) -> bool:
        if self.last_nav is None:
            return False
        self._recent.clear()
        self.say(self.last_nav.text, "nav", key=None, ttl=30)
        return True

    # ------------------------------------------------------------------ speaking rate
    @property
    def speed(self) -> float:
        return self.cache.speed

    def set_speed(self, speed: float) -> float:
        """Set and remember the speaking rate. Safety phrases recorded at the old rate keep
        playing instantly; the fixed ones are re-recorded in the background."""
        speed = round(min(SPEED_MAX, max(SPEED_MIN, float(speed))), 2)
        if speed == self.speed:
            return speed
        self.cache.speed = speed
        self.stats["speed"] = speed
        self._warm_gen += 1
        try:
            self.prefs_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.prefs_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"speed": speed}))
            os.replace(tmp, self.prefs_path)
        except OSError as e:
            self.log(f"[speech] could not save the speaking rate ({e})")
        self.log(f"[speech] speaking rate {speed:.1f}x")
        if self.tts.key:
            self.prewarm(list(FIXED_PHRASES))
        return speed

    def adjust_speed(self, step: int) -> str:
        """+1 faster, -1 slower, 0 back to the default. Returns what to say back."""
        if step == 0:
            self.set_speed(self.cfg.speed)
            return "Normal speed."
        new = round(self.speed + step * SPEED_STEP, 2)
        if (step > 0 and self.speed >= SPEED_MAX) or (step < 0 and self.speed <= SPEED_MIN):
            return "That's as fast as I go." if step > 0 else "That's as slow as I go."
        self.set_speed(new)
        return "Faster." if step > 0 else "Slower."

    def _load_speed(self) -> float:
        try:
            v = float(json.loads(self.prefs_path.read_text()).get("speed", self.cfg.speed))
            return round(min(SPEED_MAX, max(SPEED_MIN, v)), 2)
        except (OSError, ValueError, TypeError, AttributeError):
            return round(float(self.cfg.speed), 2)

    # ------------------------------------------------------------------ status
    @property
    def busy(self) -> bool:
        with self._qlock:
            return self._current is not None or bool(self._queue)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        t0 = time.time()
        while self.busy and time.time() - t0 < timeout:
            time.sleep(0.02)
        return not self.busy

    def latency(self) -> dict:
        """End-of-speech -> first-audio breakdown plus synthesis numbers, for state() / selftest."""
        return {"voice_loop": self.turns.summary(), "tts_first_byte_ms": _median(self.tts.ttfb_ms),
                "first_audio_ms": _median(self.stats["first_audio_ms"])}

    def prewarm(self, phrases: list[str] | None = None) -> None:
        """Render the safety phrases into the cache in the background."""
        self._call(self._prewarm, phrases or safety_phrases())

    def stop(self) -> None:
        self.meter.flush()
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
        self._warm_lock = asyncio.Lock()
        self._loop = loop
        loop.create_task(self._consumer())
        if self.tts.key:
            loop.create_task(self._probe_loop())
        loop.run_forever()

    async def _probe_loop(self) -> None:
        """Keep the TTS socket open (a handshake costs no credits; an open socket with no
        context does not count against the plan's concurrency). Measured: a reply on a cold
        socket reached the ear in ~300 ms, on a warm one in ~100 ms, and Cartesia closes
        sockets idle for 5 minutes. While the neural voice is down this is also what finds
        out it is back, so no sentence pays a connect timeout to discover that."""
        first = True
        while True:
            if not first:
                await asyncio.sleep(self.PROBE_EVERY_S)
            first = False
            if self.tts.connected or self.tts.quota_exhausted:
                continue
            if self.online is False or not self.tts.available:    # down (maybe since warm-up)
                if await self.tts.probe() and self.online is False:
                    self._set_online(True)
            elif self._current is None:
                await self.tts.probe()

    def warm(self) -> None:
        """The wearer started talking: a reply is coming, make sure the socket is open."""
        if self.tts.key and not self.tts.connected and self.tts.available:
            self._call(self.tts.probe)

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
                shed = self._shed_locked()
            for q in shed:
                self._notify(q, "dropped")

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
            started = time.time()
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
            rec = {"t": time.time(), "t_start": started, "text": u.text, "priority": u.priority, "result": how,
                   "key": u.key}
            self.history.append(rec)
            if self.on_spoken is not None:
                try:
                    self.on_spoken(rec)
                except Exception:
                    pass
            self._notify(u, how)
            if self._stale_refresh and not self.busy:
                todo, self._stale_refresh = list(self._stale_refresh), set()
                self.prewarm(todo)

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

    def _first_audio(self, u: Utterance, t0: float, source: str) -> None:
        now = time.monotonic()
        if source == "cartesia":                 # the neural number the dashboard has always shown
            self.stats["first_audio_ms"].append((now - t0) * 1000)
            del self.stats["first_audio_ms"][:-50]
            self.stats["tts_first_byte_ms"] = _median(self.tts.ttfb_ms)
        if u.priority in REPLIES and self.turns.audio(u.uid, now, source) is not None:
            self.stats["voice_loop"] = self.turns.summary()

    def _offline(self, text: str) -> np.ndarray:
        return offline_tts(text, self.engine.rate, self.cfg.offline_voice, wpm=offline_wpm(self.speed))

    async def _speak(self, u: Utterance) -> str:
        t0 = time.monotonic()
        if u.pre_sound is not None:
            self.engine.play(u.pre_sound, bus="alert" if u.priority == "critical" else "cue")
            await asyncio.sleep(len(u.pre_sound) / self.engine.rate * 0.8)
        if u.cancelled:                              # barged in during the earcon
            return "cut"
        wire = pronounce(u.text)
        urgent = u.priority in SAFETY
        self.engine.speech_begin(u.uid)
        # an offline-voice copy is only good enough when it is urgent or there is no network
        pcm, kind = self.cache.lookup(wire, offline=urgent or not self.tts.available)
        if pcm is not None:
            self.stats["cache_hits"] += 1
            if kind == "stale":
                self.stats["stale_rate_hits"] += 1
                self._stale_refresh.add(wire)      # re-record at the new rate when idle
            self.engine.speech_feed(u.uid, pcm)
            self._first_audio(u, t0, f"cache:{kind}")
            self.engine.speech_end(u.uid)
            if u.cancelled:                          # barge-in raced the start: make sure it is cut
                self.engine.speech_cancel(u.uid, fade_s=0.04)
            return await self._await_end(u.uid, pcm.size / self.engine.rate + 3.0)

        total = 0.0
        attempts = 3
        neural_ok = self.tts.available and (not self.meter.over_budget or urgent)
        if self.tts.available and not neural_ok and not getattr(self, "_budget_warned", False):
            self._budget_warned = True
            self.log("[speech] Cartesia character budget reached: non-urgent speech uses the built-in voice")
        backup: asyncio.Future | None = None
        if urgent and neural_ok:
            # a warning nobody cached: render the backup voice at the same time, so a slow or
            # dead network costs at most URGENT_FIRST_CHUNK_S before something is said
            backup = asyncio.ensure_future(asyncio.to_thread(self._offline, wire))
            backup.add_done_callback(lambda f: f.cancelled() or f.exception())   # never "unretrieved"
        speed = self.speed
        while neural_ok and attempts > 0:
            attempts -= 1
            chunks = []
            try:
                async for c in self.tts.stream(wire, u.uid if attempts == 2 else f"{u.uid}-{attempts}", speed,
                                               first_timeout=self.URGENT_FIRST_CHUNK_S if urgent else None):
                    if u.cancelled:                     # preempted while streaming
                        await self.tts.cancel(u.uid)
                        return "cut"
                    if not chunks:
                        self._first_audio(u, t0, "cartesia")
                    chunks.append(c)
                    total += c.size
                    self.engine.speech_feed(u.uid, c)
                self.engine.speech_end(u.uid)
                self.stats["neural"] += 1
                self._set_online(True)
                if len(wire) <= self.cfg.cache_max_chars and chunks:   # repeats cost nothing
                    self.cache.put(wire, np.concatenate(chunks), tag=self.cache.tag_for(speed))
                return await self._await_end(u.uid, total / self.engine.rate + 3.0)
            except ConcurrencyLimited:
                self.stats["concurrency_retries"] = self.stats.get("concurrency_retries", 0) + 1
                if total > 0 or u.cancelled:
                    break
                await asyncio.sleep(0.15 * (3 - attempts))  # another request is finishing: brief wait
                continue
            except SlowFirstChunk:
                self.stats["urgent_backup"] = self.stats.get("urgent_backup", 0) + 1
                await self.tts.cancel(u.uid if attempts == 2 else f"{u.uid}-{attempts}")
                break
            except Exception as e:
                self.log(f"[speech] cartesia failed ({e}); using the offline voice")
                self._set_online(False)
                if total > 0:              # audio already started: do not restart the sentence
                    self.engine.speech_end(u.uid)
                    return await self._await_end(u.uid, total / self.engine.rate + 3.0)
                break
        if total > 0:
            self.engine.speech_end(u.uid)
            return await self._await_end(u.uid, total / self.engine.rate + 3.0)
        if u.expired(time.monotonic(), self.cfg.default_ttl_s):
            self.engine.speech_cancel(u.uid)
            return "dropped"
        try:
            pcm = await (backup if backup is not None else asyncio.to_thread(self._offline, wire))
        except Exception as e:
            self.log(f"[speech] offline TTS failed too ({e})")
            self.engine.speech_cancel(u.uid)
            return "cut"
        if u.cancelled:
            return "cut"
        self.stats["offline"] += 1
        if len(wire) <= self.cfg.cache_max_chars:     # the next outage repeat is instant
            self.cache.put(wire, pcm, offline=True)
        self.engine.speech_feed(u.uid, pcm)
        self._first_audio(u, t0, "offline")
        self.engine.speech_end(u.uid)
        return await self._await_end(u.uid, pcm.size / self.engine.rate + 3.0)

    def _set_online(self, ok: bool) -> None:
        was = self.online
        self.online = ok
        if was is None or ok == was:
            return
        self.log(f"[speech] neural voice {'back online' if ok else 'offline'}")
        # The wearer hears the voice change, so say why (both lines are cached phrases): the
        # outage at most once a minute, and its end only if they were told about it.
        if ok:
            if self._told_offline:
                self._told_offline = False
                with self._qlock:
                    unheard = any(q.key == "voice-offline" for q in self._queue)
                if unheard:                            # the outage ended before they heard of it
                    self.drop_where(lambda q: q.key == "voice-offline")
                else:
                    self.say(VOICE_BACK, "system", key="voice-back", ttl=10)
            return
        now = time.monotonic()
        if now - self._announced_t >= self.ANNOUNCE_GAP_S:
            self._announced_t = now
            self._told_offline = True
            self.say(VOICE_OFFLINE, "system", key="voice-offline", ttl=10)

    async def _prewarm(self, phrases: list[str]) -> None:
        async with self._warm_lock:
            await self._prewarm_locked(phrases)

    async def _prewarm_locked(self, phrases: list[str]) -> None:
        gen = self._warm_gen
        phrases = [pronounce(p) for p in phrases]
        missing = [p for p in phrases if not self.cache.has_neural(p)]
        if not missing:
            return
        t0 = time.monotonic()
        done = 0
        if self.tts.available:
            # One at a time, and never while Kite is talking: the Cartesia plan allows only two
            # requests in flight, and live speech must always get one.
            speed, tag = self.speed, self.cache.tag
            for p in missing:
                if self.meter.over_budget or gen != self._warm_gen:     # rate changed: a newer run takes over
                    break
                while self._current is not None or self.listening:
                    await asyncio.sleep(0.05)
                for _try in range(3):
                    try:
                        chunks = [c async for c in self.tts.stream(p, "warm-" + uuid.uuid4().hex[:10], speed)]
                        if chunks:
                            self.cache.put(p, np.concatenate(chunks), tag=tag)
                            done += 1
                        break
                    except ConcurrencyLimited:
                        await asyncio.sleep(0.5)
                    except Exception:
                        break
                if not self.tts.available:
                    break
        if done < len(missing) and gen == self._warm_gen:
            # no neural voice: make sure the phrases at least exist offline
            for p in phrases:
                if self.cache.get(p) is None:
                    try:
                        self.cache.put(p, await asyncio.to_thread(self._offline, p), offline=True)
                    except Exception:
                        break
        self.log(f"[speech] phrase cache: {done}/{len(missing)} new neural phrases "
                 f"in {time.monotonic() - t0:.1f}s")
