"""What happens after the wearer speaks.

Two tiers, so the important things never depend on the network:
  1. Local intents (instant, offline): stop talking, repeat, pause/resume,
     beeps on/off, louder/quieter, beacon on/off, "take me to X", cancel route,
     ear check, "what's around me", "where am I", "how far".
  2. Claude for everything else, with a live snapshot of what Kite knows (nearby
     objects, hazards, route, drone scene descriptions, position) and a few
     tools (navigate, control, alert the caregiver). Text is streamed and spoken
     sentence by sentence, so the first words arrive before the answer is done.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .events import clock_phrase, distance_phrase

MODEL = "claude-opus-5-5"
SYSTEM_PROMPT = """You are Kite, the voice of an assistive navigation system for a blind pedestrian. \
A small drone flies just above and behind the user and watches the street ahead, and a chest camera watches the path; their findings are given to you as JSON \
context with every question. You speak through the user's AirPods.

How to answer:
- Reply in one or two short spoken sentences, under 35 words. No lists, no markdown, no emoji.
- Give directions as clock positions or "left/right/ahead" and distances in meters, from the user's point of view.
- Only describe what the context contains. If something is not in the context, say you can't see it yet. Never guess about obstacles, traffic, or signals.
- Safety first: never tell the user it is safe to cross a street. If the context shows a walk signal, say the signal shows walk and remind them to listen for traffic.
- "memories" in the context are the user's own saved places and notes: use them to understand "my usual", "home", or a known hazard, and mention a remembered hazard when it matters.
- If the user wants to go somewhere, call navigate_to. If they want a setting changed, call control. If they are lost, hurt, scared, or ask for help, call alert_caregiver and tell them help has been notified.
- Be warm and calm, never chatty."""

TOOLS = [
    {"name": "navigate_to", "eager_input_streaming": True,
     "description": "Start walking directions to a destination the user named (a place, address, or landmark).",
     "input_schema": {"type": "object", "properties": {"destination": {"type": "string"}},
                      "required": ["destination"], "additionalProperties": False}},
    {"name": "control", "eager_input_streaming": True,
     "description": "Change how Kite sounds or behaves.",
     "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": [
         "mute_cues", "unmute_cues", "pause", "resume", "volume_up", "volume_down", "repeat",
         "beacon_on", "beacon_off", "cancel_route"]}},
         "required": ["action"], "additionalProperties": False}},
    {"name": "remember", "eager_input_streaming": True,
     "description": "Save something the user wants Kite to remember: a place at their current location, a hazard, or a note.",
     "input_schema": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["place", "hazard", "note"]}, "text": {"type": "string"}},
         "required": ["kind", "text"], "additionalProperties": False}},
    {"name": "alert_caregiver", "eager_input_streaming": True,
     "description": "Send an urgent message to the user's caregiver dashboard (lost, hurt, needs help).",
     "input_schema": {"type": "object", "properties": {"message": {"type": "string"}},
                      "required": ["message"], "additionalProperties": False}},
]


@dataclass
class Actions:
    """Callbacks into the app. Each returns a short spoken confirmation or None."""
    navigate: Callable[[str], str | None]
    control: Callable[[str], str | None]
    alert_caregiver: Callable[[str], str | None]
    stop_speaking: Callable[[], None]
    ear_check: Callable[[], None]
    remember: Callable[[str, str], str | None] = lambda kind, text: None
    recall: Callable[[str], list] = lambda query: []
    forget: Callable[[str], str | None] = lambda what: None
    rename: Callable[[str, str], str | None] = lambda old, new: None
    undo: Callable[[], str | None] = lambda: None
    clarify: Callable[[str], str | None] = lambda text: None    # answers a pending "which one?"
    describe_front: Callable[[], str | None] = lambda: None   # Claude on the chest camera frame
    read_text: Callable[[], str | None] = lambda: None        # OCR: read a menu / sign / screen aloud


_LOCAL = [
    (r"^(?:stop|quiet|shut up|be quiet|stop talking|silence|enough|cancel that)\b[.!]?$", "stop_speaking"),
    (r"\b(?:repeat|say (?:that|it) again|what did you say|come again|pardon)\b", "repeat"),
    (r"^(?:pause|hold on|pause guidance|stop guidance)\b", "pause"),
    (r"^(?:resume|continue|go on|unpause|resume guidance|keep going)\b", "resume"),
    (r"\b(?:unmute)\b|\b(?:turn on|start)\b.*\b(?:beeps?|beeping|sounds?|sonar)\b"
     r"|\b(?:beeps?|beeping|sounds?|sonar)\b.*\bon$", "unmute_cues"),
    (r"\b(?:mute|turn off|stop|no more)\b.*\b(?:beeps?|beeping|sounds?|sonar)\b"
     r"|\b(?:beeps?|beeping|sounds?|sonar)\b.*\boff$", "mute_cues"),
    (r"\b(?:louder|volume up|speak up|turn it up|can't hear)\b", "volume_up"),
    (r"\b(?:quieter|softer|volume down|too loud|turn it down)\b", "volume_down"),
    (r"\b(?:beacon|ticking)\b.*\b(?:off|stop)\b|\b(?:turn off|stop)\b.*\bbeacon\b", "beacon_off"),
    (r"\b(?:beacon|ticking)\b.*\bon\b|\b(?:turn on|start)\b.*\bbeacon\b", "beacon_on"),
    (r"\b(?:cancel|stop|end)\b.*\b(?:route|navigation|directions)\b", "cancel_route"),
    (r"\b(?:check|test)\b.*\b(?:ears?|audio|sound|airpods)\b|\bwhich ear\b", "ear_check"),
]
_REMEMBER_PLACE = re.compile(r"(?i)^(?:please\s+)?(?:remember|save|mark|bookmark)\s+(?:this place|this spot|"
                             r"this location|where i am|my location|this|here)\s*(?:as|is|called)?\s+(?P<name>.+?)[.!?]*$")
_REMEMBER_NOTE = re.compile(r"(?i)^(?:please\s+)?(?:remember|note|keep in mind|make a note)\s+(?:that\s+)?(?P<text>.+?)[.!?]*$")
_NOT_MEMORY = r"(?:it|that|this|everything|the (?:beacon|route|directions|navigation|sonar|beeps?|sounds?))"
_FORGET = re.compile(r"(?i)^(?:please\s+)?(?:forget|delete|remove|erase)\s+(?:about\s+)?(?:the memory (?:of|about)\s+|"
                     r"my memory of\s+)?(?!" + _NOT_MEMORY + r"[.!?]*$)(?P<what>.+?)[.!?]*$")
_RENAME = re.compile(r"(?i)^(?:please\s+)?(?:rename|change the name of)\s+(?P<old>.+?)\s+(?:to|as)\s+(?P<new>.+?)[.!?]*$")
_UNDO = re.compile(r"(?i)^(?:undo|undo that|bring (?:it|that) back|i didn'?t mean (?:it|that))[.!?]*$")
_HOME = re.compile(r"\b(?:take|bring|walk|guide|get|lead) me home\b|^(?:go|head|navigate|directions) home\b")
_READ = re.compile(r"\bread (?:this|that|it|me|the|out|aloud|what)\b|\bread\s*$|\bwhat does (?:this|that|it|the \w+) say\b"
                   r"|\bwhat(?:'s| is) (?:written|on (?:the|this|that) (?:sign|menu|screen|label|page|board))\b")
_RECALL = re.compile(r"\bwhat do you (?:remember|know) about\b|\bdo you remember\b")
_HAZARD_WORDS = re.compile(r"\b(curb|step|stairs?|construction|scaffold\w*|pole|hole|slippery|uneven|broken|"
                           r"crack\w*|ramp|puddle|narrow\w*|tall)\b", re.I)
_NAV = re.compile(r"(?i)\b(?:take me|navigate|guide me|walk me|get me|directions|route me|lead me|bring me|go)"
                  r"\s+(?:back\s+)?(?:to|towards?)\s+(?:the\s+)?(?P<dest>.+?)[.!?]*$")
_AROUND = re.compile(r"\b(?:what'?s|what is|anything|anyone|who'?s|who is)\b.*\b(?:around|ahead|in front|near|nearby|close)\b")
_WHERE = re.compile(r"\bwhere am i\b|\bwhere are we\b")
_HOWFAR = re.compile(r"\bhow (?:far|long)\b|\bare we there\b|\bwhere am i going\b")


def strip_wake(text: str, wake_words: tuple[str, ...]) -> tuple[bool, str]:
    """('kite, take me home') -> (True, 'take me home'). Tolerates STT variants."""
    t = text.strip().lower()
    t = re.sub(r"^(?:hey|okay|ok|hi)[, ]+", "", t)
    for w in ("kite", "kyte", "kight", "kate", "kit"):
        m = re.match(rf"^{w}\b[,.!?]?\s*", t)
        if m:
            return True, t[m.end():].strip()
    return False, t


class Brain:
    def __init__(self, context: Callable[[], dict], actions: Actions,
                 speak: Callable[..., object], api_key: str | None, log=print,
                 model: str = MODEL, timeout_s: float = 12.0):
        self.context = context
        self.actions = actions
        self.speak = speak                         # speak(text, priority, key=, ttl=)
        self.log = log
        self.model = model
        self.timeout_s = timeout_s
        self.api_key = api_key
        self._client = None
        self.history: list[dict] = []              # short memory: last few Q/A pairs as text
        self._loop: asyncio.AbstractEventLoop | None = None
        self._busy = threading.Lock()
        self.stats = {"local": 0, "llm": 0, "llm_errors": 0, "first_sentence_ms": []}
        if api_key:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=1)
            except ImportError:
                self.log("[brain] anthropic SDK not installed: local commands only")
        threading.Thread(target=self._run_loop, name="brain", daemon=True).start()
        while self._loop is None:
            time.sleep(0.002)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_forever()

    @property
    def online(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------ entry
    def handle(self, text: str) -> None:
        """Thread-safe: queue the utterance for handling."""
        asyncio.run_coroutine_threadsafe(self._handle(text), self._loop)

    def handle_sync(self, text: str, timeout: float = 20.0) -> str:
        """For tests and the CLI: returns 'local:<intent>' or 'llm' or 'fallback'."""
        return asyncio.run_coroutine_threadsafe(self._handle(text), self._loop).result(timeout)

    async def _handle(self, text: str) -> str:
        text = " ".join(text.split()).strip()
        if not text:
            return "empty"
        low = text.lower().strip(" .!?")
        answer = self.actions.clarify(text)          # "the park one", after "which bench?"
        if answer:
            self.stats["local"] += 1
            self.speak(answer, "answer", ttl=15)
            return "local:clarify"
        kind = self.local_intent(low, text.strip(" .!?"))
        if kind is not None:
            self.stats["local"] += 1
            self._do_local(kind, low)
            return f"local:{kind[0]}"
        if self._client is None:
            self.speak(self.offline_answer(low), "answer", ttl=20)
            return "fallback"
        try:
            await asyncio.wait_for(self._ask_llm(text), timeout=self.timeout_s + 3)
            self.stats["llm"] += 1
            return "llm"
        except Exception as e:
            self.stats["llm_errors"] += 1
            self.log(f"[brain] LLM failed ({type(e).__name__}: {e}); answering locally")
            self.speak(self.offline_answer(low), "answer", ttl=20)
            return "fallback"

    def local_intent(self, low: str, original: str | None = None) -> tuple[str, str | None] | None:
        o = original or low
        m = _REMEMBER_PLACE.search(o)
        if m:
            return ("remember_place", m.group("name").strip())
        m = _REMEMBER_NOTE.search(o)
        if m and not re.match(r"(?i)^remember (?:this|here)\b", o):
            return ("remember_note", m.group("text").strip())
        m = _RENAME.search(o)
        if m:
            return ("rename", m.group("old").strip() + "\n" + m.group("new").strip())
        m = _FORGET.search(o)
        if m:
            return ("forget", m.group("what").strip())
        if _UNDO.search(o):
            return ("undo", None)
        if _HOME.search(low):
            return ("navigate", "home")
        if _READ.search(low):
            return ("read", None)
        if _RECALL.search(low):
            return ("recall", None)
        if re.search(r"\b(?:take|bring|walk|guide|lead) me back\b|^go back\b", low) and not _NAV.search(o):
            return ("navigate", "back the way I came")
        m = _NAV.search(o)
        if m and len(m.group("dest")) >= 2:
            return ("navigate", m.group("dest").strip())
        for pat, name in _LOCAL:
            if re.search(pat, low):
                return (name, None)
        if _AROUND.search(low):
            return ("around", None)
        if _WHERE.search(low):
            return ("where", None)
        if _HOWFAR.search(low):
            return ("howfar", None)
        return None

    def _do_local(self, kind: tuple[str, str | None], low: str) -> None:
        name, arg = kind
        if name == "remember_place":
            msg = self.actions.remember("place", arg or "")
            if msg:
                self.speak(msg, "answer", ttl=15)
        elif name == "remember_note":
            kind = "hazard" if _HAZARD_WORDS.search(arg or "") else "note"
            msg = self.actions.remember(kind, arg or "")
            if msg:
                self.speak(msg, "answer", ttl=15)
        elif name in ("forget", "rename", "undo"):
            msg = (self.actions.forget(arg or "") if name == "forget" else
                   self.actions.rename(*(arg or "\n").split("\n", 1)) if name == "rename" else self.actions.undo())
            if msg:
                self.speak(msg, "answer", ttl=15)
        elif name == "recall":
            q = re.sub(r"^.*?\b(?:remember|know) about\s+", "", low)
            hits = self.actions.recall(q)
            self.speak(f"I remember: {hits[0]['text']}" if hits else "I don't have anything saved about that.",
                       "answer", ttl=20)
        elif name == "navigate":
            msg = self.actions.navigate(arg or "")
            if msg:
                self.speak(msg, "answer", ttl=15)
        elif name == "stop_speaking":
            self.actions.stop_speaking()
        elif name == "ear_check":
            self.actions.ear_check()
        elif name == "read":
            threading.Thread(target=self._read, daemon=True).start()
        elif name == "around" and re.search(r"\b(?:in front|ahead|front of me)\b", low):
            # spec 7: "what's in front of me?" is answered from the chest camera when possible
            threading.Thread(target=self._front, args=(low,), daemon=True).start()
        elif name in ("around", "where", "howfar"):
            self.speak(self.offline_answer(low), "answer", ttl=20)
        else:
            msg = self.actions.control(name)
            if msg:
                self.speak(msg, "answer", ttl=10)

    def _read(self) -> None:
        try:
            last = self.actions.read_text()
        except Exception as e:
            self.log(f"[brain] reading failed: {e}")
            last = "Sorry, I couldn't read that."
        if last:
            self.speak(last, "answer", ttl=120)

    def _front(self, low: str) -> None:
        text = None
        try:
            text = self.actions.describe_front()
        except Exception as e:
            self.log(f"[brain] chest camera description failed: {e}")
        self.speak(text or self.offline_answer(low), "answer", ttl=20)

    # ------------------------------------------------------------------ offline answers
    def offline_answer(self, low: str) -> str:
        ctx = self.context()
        if _WHERE.search(low):
            pos = ctx.get("position") or {}
            if pos.get("place"):
                return f"You are near {pos['place']}."
            return "I don't have your location yet."
        if _HOWFAR.search(low):
            b = ctx.get("route") or {}
            if b.get("label") and b.get("distance") is not None:
                return f"{b['label']} is {distance_phrase(b['distance'])} away, {clock_phrase(b['bearing'])}."
            if b.get("distance") is not None:
                return f"The next waypoint is {distance_phrase(b['distance'])} away, {clock_phrase(b['bearing'])}."
            return "There is no route right now. Say, take me to, and a place."
        if _AROUND.search(low) or "describe" in low or "see" in low:
            return self.describe(ctx)
        return "I can't answer that without the internet. I can still guide you and warn you about obstacles."

    @staticmethod
    def describe(ctx: dict) -> str:
        parts = []
        objs = [o for o in ctx.get("nearby", []) if o.get("distance") is not None][:3]
        names = {"person": "a person", "vehicle": "a car", "bike": "a bike", "curb": "a curb",
                 "stairs_down": "a step down"}
        for o in objs:
            parts.append(f"{names.get(o['kind'], 'an obstacle')} {clock_phrase(o['bearing'])}, "
                         f"{distance_phrase(o['distance'])}")
        hz = ctx.get("hazards") or []
        for h in hz[:2]:
            if h.get("text"):
                parts.append(h["text"].rstrip("."))
        scene = ctx.get("scene")
        if not parts and scene:
            return scene if len(scene) < 200 else scene[:200].rsplit(" ", 1)[0] + "."
        if not parts:
            if not ctx.get("perception_ok", True):
                return "Obstacle detection is not running, so I can't see around you right now."
            return "Nothing close to you that I can detect right now."
        s = "; ".join(parts)
        return s[0].upper() + s[1:] + "."

    # ------------------------------------------------------------------ Claude
    async def _ask_llm(self, text: str) -> None:
        import anthropic

        c = self.context()
        try:
            c["memories"] = self.actions.recall(text)
        except Exception:
            c["memories"] = []
        ctx = json.dumps(c, separators=(",", ":"), default=str)
        messages = list(self.history[-6:])
        messages.append({"role": "user", "content": f"<context>{ctx}</context>\n\nUser said: {text}"})
        t0 = time.monotonic()
        spoken: list[str] = []
        first = True

        def emit(sentence: str) -> None:
            nonlocal first
            s = sentence.strip()
            if not s:
                return
            if first:
                first = False
                self.stats["first_sentence_ms"].append((time.monotonic() - t0) * 1000)
                del self.stats["first_sentence_ms"][:-50]
            spoken.append(s)
            self.speak(s, "answer", ttl=30)

        for _turn in range(3):                     # at most 3 model turns (tool round-trips)
            buf = ""
            async with self._client.beta.messages.stream(
                model=self.model, max_tokens=1024, system=[{"type": "text", "text": SYSTEM_PROMPT,
                                                            "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS, messages=messages, output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            ) as stream:
                async for ev in stream:
                    if ev.type == "text":
                        buf += ev.text
                        while True:          # speak each finished sentence immediately
                            m = re.search(r"[.!?](?=\s)", buf)
                            if not m:
                                break
                            emit(buf[:m.end()])
                            buf = buf[m.end():]
                resp = await stream.get_final_message()
            if buf.strip():
                emit(buf)
            if resp.stop_reason == "refusal":
                emit("Sorry, I can't help with that one.")
                break
            uses = [b for b in resp.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": resp.content})
            if not uses or resp.stop_reason == "max_tokens":
                break
            results = []
            for u in uses:
                out, err = self._run_tool(u.name, u.input)
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": out,
                                **({"is_error": True} if err else {})})
            messages.append({"role": "user", "content": results})
        answer = " ".join(spoken)
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": answer or "(no answer)"}]
        self.history = self.history[-8:]

    def _run_tool(self, name: str, inp) -> tuple[str, bool]:
        if not isinstance(inp, dict):
            return "invalid input", True
        try:
            if name == "navigate_to":
                dest = str(inp.get("destination", "")).strip()[:120]
                if not dest:
                    return "missing destination", True
                return self.actions.navigate(dest) or "ok", False
            if name == "control":
                a = str(inp.get("action", ""))
                allowed = next(t for t in TOOLS if t["name"] == "control")["input_schema"]["properties"]["action"]["enum"]
                if a not in allowed:
                    return f"unknown action {a}", True
                return self.actions.control(a) or "ok", False
            if name == "remember":
                k = str(inp.get("kind", "note"))
                txt = str(inp.get("text", "")).strip()[:300]
                if k not in ("place", "hazard", "note") or not txt:
                    return "invalid memory", True
                return self.actions.remember(k, txt) or "saved", False
            if name == "alert_caregiver":
                msg = str(inp.get("message", "")).strip()[:300]
                return self.actions.alert_caregiver(msg or "The user asked for help.") or "sent", False
        except Exception as e:
            return f"failed: {e}", True
        return f"unknown tool {name}", True
