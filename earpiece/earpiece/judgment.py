"""The judgment layer: Claude makes the careful calls, inside a box it cannot leave.

Spec 5.6. Input: active hazards, the route, the latest scene description, the
walker's pose and the nearest tracked objects. Output: at most one short spoken
decision ("Wait, the crosswalk is crowded.") and optional priority upgrades for
hazards that are already active.

The box, enforced here in code (the prompt asks for the same, but code decides):
  * Claude only ever makes Kite MORE cautious. It may tell the walker to wait or
    warn them; any reply that would clear a stop, send them on ("go", "continue",
    "safe to cross", "walk now", ...) is discarded, always.
  * It never reroutes or navigates: it has no tools, and the output schema has no
    route field.
  * Upgrades can only raise a hazard's priority, and only for hazards that exist.
  * Sonar, critical obstacles and "don't walk" are handled by deterministic code
    before this layer ever runs; while one of them is active, Claude stays quiet
    unless it is adding a warning.
  * It is rate-limited and runs off the audio path; if Claude is slow, down or
    refuses, nothing changes except that the decision is not spoken.

Scene description from images (Claude vision) lives here too: the drone or sim
view arrives as a JPEG and is turned into the `scene` text everyone uses.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .events import PRIORITY_RANK

MODEL = "claude-opus-5-5"

JUDGE_PROMPT = """You are the judgment layer of Kite, a navigation aid for a blind pedestrian. \
A drone above and behind the walker and a chest camera report what is around them; the JSON you get \
is everything Kite currently knows. Deterministic code already handles beeps, "stop" for imminent \
collisions, and "don't walk" signals.

Your only job: decide whether the walker should hear ONE short extra sentence that makes them safer, \
and whether any active hazard deserves a higher priority.

Rules:
- You may only make Kite more cautious: tell the walker to wait, slow down, or warn them about something.
- Never tell the walker to go, continue, cross, or that anything is safe or clear. Never suggest a route change.
- Speak only about things present in the JSON. If nothing needs saying, return an empty "say".
- At most 12 words, spoken English, no lists. Use clock directions or left/right/ahead and meters.
- "upgrade" may only list keys from "hazards" whose priority should be raised to "hazard" or "critical"."""

SCENE_PROMPT = """You describe a street scene for a blind pedestrian's navigation aid. The image comes \
from a camera {where}. In at most two short sentences, describe what matters for walking: sidewalk \
width and surface, people and crowds, vehicles, crossings and signals, obstacles, and anything at head \
height. Give directions as left, right or ahead, and rough distances in meters. Only describe what you \
can see; if unsure, say so. Never say anything is safe."""

READ_PROMPT = """Read the text in this photo aloud for a blind person who is pointing a camera at it \
(a menu, sign, label, book page, screen or document). Transcribe the readable text exactly, in natural \
reading order, as plain sentences ready to be spoken: no markdown, no bullet symbols, no commentary. \
For a menu, read each item with its price. Skip text that is too small or blurry to read reliably \
rather than guessing. If there is no readable text, reply exactly: NO_TEXT"""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "say": {"type": "string"},
        "wait": {"type": "boolean"},
        "upgrade": {"type": "array", "items": {"type": "object", "properties": {
            "key": {"type": "string"}, "priority": {"type": "string", "enum": ["hazard", "critical"]}},
            "required": ["key", "priority"], "additionalProperties": False}},
    },
    "required": ["say", "wait", "upgrade"],
    "additionalProperties": False,
}

# Anything that would clear a stop or send the walker on. Checked on every reply.
_GO = re.compile(
    r"\b(?:go|go ahead|continue|proceed|cross(?:ing)? now|you can (?:go|walk|cross|continue|proceed)|"
    r"(?:it'?s|it is|path is|way is|road is|street is|coast is)\s+(?:now\s+)?(?:safe|clear)|safe to|all clear|"
    r"walk now|start walking|keep walking|move forward|step (?:forward|off|out)|turn (?:left|right)|"
    r"reroute|detour|take (?:the )?(?:other|another)|instead go)\b", re.I)


def is_permitted(text: str) -> bool:
    """True if the sentence only adds caution."""
    t = " ".join(text.split())
    if not t:
        return False
    if _GO.search(t):
        return False
    return len(t.split()) <= 20


@dataclass
class Decision:
    say: str = ""
    wait: bool = False
    upgrade: list = field(default_factory=list)
    blocked: str | None = None          # why the spoken part was discarded, if it was


def sanitize(raw: dict, hazards: dict[str, dict]) -> Decision:
    """Apply the box to whatever the model returned."""
    d = Decision()
    say = raw.get("say") if isinstance(raw, dict) else ""
    if isinstance(say, str) and say.strip():
        if is_permitted(say):
            d.say = " ".join(say.split())[:160]
        else:
            d.blocked = "would clear a stop or move the walker"
    d.wait = bool(raw.get("wait")) if isinstance(raw, dict) else False
    ups = raw.get("upgrade") if isinstance(raw, dict) else []
    for u in ups if isinstance(ups, list) else []:
        if not isinstance(u, dict):
            continue
        key, pr = u.get("key"), u.get("priority")
        h = hazards.get(key) if isinstance(key, str) else None
        if h is None or pr not in ("hazard", "critical"):
            continue
        current = h.get("priority", "hazard")
        if PRIORITY_RANK[pr] < PRIORITY_RANK.get(current, 99):      # strictly more important only
            d.upgrade.append((key, pr))
    return d


class Judgment:
    """Runs Claude off the audio path. `snapshot()` supplies the world; `act(decision)`
    applies it. Triggered by new hazards / scenes, rate-limited by min_interval_s."""

    def __init__(self, api_key: str | None, snapshot: Callable[[], dict], act: Callable[[Decision], None],
                 log=print, model: str = MODEL, min_interval_s: float = 4.0, timeout_s: float = 8.0,
                 client=None):
        self.snapshot = snapshot
        self.act = act
        self.log = log
        self.model = model
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._client = client
        if self._client is None and api_key:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=1)
            except ImportError:
                self._client = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending = False
        self._busy = False
        self._last = 0.0
        self.enabled = True
        self.stats = {"calls": 0, "spoken": 0, "blocked": 0, "upgrades": 0, "errors": 0, "skipped": 0,
                      "last_ms": None, "scenes": 0}
        threading.Thread(target=self._run, name="judgment", daemon=True).start()
        t0 = time.time()
        while self._loop is None and time.time() - t0 < 2:
            time.sleep(0.002)

    @property
    def online(self) -> bool:
        return self._client is not None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_forever()

    # ------------------------------------------------------------------ triggering
    def poke(self, why: str = "") -> None:
        """Something changed (new hazard, new scene). Thread-safe, cheap, coalesced."""
        if not self.enabled or self._client is None or self._loop is None:
            return
        if self._busy or self._pending:
            self._pending = True
            return
        self._pending = True
        asyncio.run_coroutine_threadsafe(self._maybe_run(), self._loop)

    async def _maybe_run(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._pending = False
        self._busy = True
        try:
            await self.judge_once()
        finally:
            self._busy = False
            if self._pending:
                self._pending = False
                self.poke("coalesced")

    def judge_sync(self, timeout: float = 20.0) -> Decision | None:
        """For tests and the CLI."""
        return asyncio.run_coroutine_threadsafe(self.judge_once(), self._loop).result(timeout)

    async def judge_once(self) -> Decision | None:
        snap = self.snapshot()
        hazards = {h["key"]: h for h in snap.get("hazards", []) if isinstance(h, dict) and "key" in h}
        if not hazards and not snap.get("scene") and not any(
                (o.get("urgency") or 0) > 0.3 for o in snap.get("nearby", [])):
            self.stats["skipped"] += 1
            return None
        self._last = time.monotonic()
        self.stats["calls"] += 1
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(self._client.beta.messages.create(
                model=self.model, max_tokens=1024,
                system=[{"type": "text", "text": JUDGE_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": json.dumps(snap, separators=(",", ":"), default=str)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            ), timeout=self.timeout_s)
        except Exception as e:
            self.stats["errors"] += 1
            self.log(f"[judgment] Claude unavailable ({type(e).__name__}); nothing changes")
            return None
        self.stats["last_ms"] = round((time.monotonic() - t0) * 1000)
        if getattr(resp, "stop_reason", None) == "refusal":
            return None
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        try:
            raw = json.loads(text)
        except (TypeError, ValueError):
            self.stats["errors"] += 1
            return None
        d = sanitize(raw, hazards)
        if d.blocked:
            self.stats["blocked"] += 1
            self.log(f"[judgment] discarded {raw.get('say')!r}: {d.blocked}")
        if d.say:
            self.stats["spoken"] += 1
        self.stats["upgrades"] += len(d.upgrade)
        try:
            self.act(d)
        except Exception as e:
            self.log(f"[judgment] act failed: {e}")
        return d

    # ------------------------------------------------------------------ scene description
    def describe_image(self, jpeg: bytes, where: str = "flying just above and behind the walker's head",
                       timeout: float = 15.0) -> str | None:
        """Claude vision: JPEG -> one or two sentences for the `scene` context. Blocking;
        call from a worker thread."""
        if self._client is None or self._loop is None:
            return None
        fut = asyncio.run_coroutine_threadsafe(self._describe(jpeg, where), self._loop)
        try:
            return fut.result(timeout)
        except Exception as e:
            self.log(f"[judgment] scene description failed ({type(e).__name__})")
            return None

    def read_text(self, jpeg: bytes, timeout: float = 20.0) -> str | None:
        """Claude vision OCR: the readable text in the photo, ready to speak. "" if none."""
        if self._client is None or self._loop is None:
            return None
        fut = asyncio.run_coroutine_threadsafe(self._read(jpeg), self._loop)
        try:
            return fut.result(timeout)
        except Exception as e:
            self.log(f"[judgment] reading text failed ({type(e).__name__})")
            return None

    async def _read(self, jpeg: bytes) -> str | None:
        resp = await asyncio.wait_for(self._client.beta.messages.create(
            model=self.model, max_tokens=4096,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.standard_b64encode(jpeg).decode()}},
                {"type": "text", "text": READ_PROMPT}]}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        ), timeout=18)
        if getattr(resp, "stop_reason", None) == "refusal":
            return None
        text = " ".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        self.stats["reads"] = self.stats.get("reads", 0) + 1
        if not text or text.strip().upper().startswith("NO_TEXT"):
            return ""
        return re.sub(r"[*#_`>|]+", " ", " ".join(text.split()))[:2000]

    async def _describe(self, jpeg: bytes, where: str) -> str | None:
        resp = await asyncio.wait_for(self._client.beta.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.standard_b64encode(jpeg).decode()}},
                {"type": "text", "text": SCENE_PROMPT.format(where=where)}]}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        ), timeout=self.timeout_s + 6)
        if getattr(resp, "stop_reason", None) == "refusal":
            return None
        text = " ".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if not text:
            return None
        self.stats["scenes"] += 1
        # a description is context, never an instruction: strip anything that sounds like one
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if not _GO.search(s)]
        return " ".join(sentences)[:400] or None
