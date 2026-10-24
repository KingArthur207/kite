"""Claude's eyes: describe a camera frame, or read the text in it aloud.

Two jobs, both answering the walker or feeding context, never deciding anything:
  * describe_image: a drone, sim or body-camera frame becomes one or two sentences
    for the shared `scene` context, and answers "what's in front of me?";
  * read_text: OCR for a blind user. Point the body camera at a menu, a book, a
    street sign or a monitor and the text comes back ready to be spoken.

Safety: a description is context, never an instruction. Any sentence in it that
sounds like permission to move ("the road is clear, walk now", say on a sign) is
stripped before it is stored or spoken. The judgment layer (judgment.py) is the
separate, guard-railed place where Claude may add caution.
"""

from __future__ import annotations

import asyncio
import base64
import re
import threading
import time

MODEL = "claude-opus-5-5"

# Anything that would clear a stop or send the walker on. Checked on every reply.
_GO = re.compile(
    r"\b(?:go|go ahead|continue|proceed|cross(?:ing)? now|you can (?:go|walk|cross|continue|proceed)|"
    r"(?:it'?s|it is|path is|way is|road is|street is|coast is)\s+(?:now\s+)?(?:safe|clear)|safe to|all clear|"
    r"walk now|start walking|keep walking|move forward|step (?:forward|off|out)|turn (?:left|right)|"
    r"reroute|detour|take (?:the )?(?:other|another)|instead go)\b", re.I)



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


READ_PROMPT = """Read the text in this photo aloud for a blind person who is pointing a camera at it \
(a menu, sign, label, book page, screen or document). Transcribe the readable text exactly, in natural \
reading order, as plain sentences ready to be spoken: no markdown, no bullet symbols, no commentary. \
For a menu, read each item with its price. Skip text that is too small or blurry to read reliably \
rather than guessing. If there is no readable text, reply exactly: NO_TEXT"""


class Vision:
    def __init__(self, api_key: str | None, log=print, model: str = MODEL, timeout_s: float = 12.0,
                 client=None):
        self.log = log
        self.model = model
        self.timeout_s = timeout_s
        self._client = client
        if self._client is None and api_key:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=1)
            except ImportError:
                self._client = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.stats = {"scenes": 0, "reads": 0, "errors": 0}
        threading.Thread(target=self._run, name="vision", daemon=True).start()
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
            self.stats["errors"] += 1
            self.log(f"[vision] scene description failed ({type(e).__name__})")
            return None

    def read_text(self, jpeg: bytes, timeout: float = 20.0) -> str | None:
        """Claude vision OCR: the readable text in the photo, ready to speak. "" if none."""
        if self._client is None or self._loop is None:
            return None
        fut = asyncio.run_coroutine_threadsafe(self._read(jpeg), self._loop)
        try:
            return fut.result(timeout)
        except Exception as e:
            self.stats["errors"] += 1
            self.log(f"[vision] reading text failed ({type(e).__name__})")
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
        self.stats["reads"] += 1
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
