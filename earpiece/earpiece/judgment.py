"""Claude judgment layer (spec 5.6): one short spoken call about the street.

Deterministic code has already acted before this runs: sonar "Stop.", critical
obstacles and "don't walk" are spoken by app.py and sonar.py the moment they
arrive. Claude only adds a judgment on top ("Wait, the crosswalk is crowded.")
and may raise the priority of a hazard that is already on the list.

Guardrails, all enforced here and not left to the prompt:
  * Claude can warn or tell the walker to wait. Anything that sounds like
    permission ("safe to cross", "go ahead", "you can cross"), a reroute, or
    clearing a stop is thrown away.
  * Priority upgrades only go up, only for a hazard key that exists, and never
    to "critical" (that level belongs to the sonar).
  * At most one call every `min_gap_s`, none when hazards and scene are
    unchanged, never on the audio thread, and a slow or failed call is dropped
    without a word.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Callable

from .events import PRIORITY_RANK

MODEL = "claude-haiku-5-5"
MAX_SAY = 120
UPGRADE_TO = ("hazard", "nav", "answer", "info")   # "critical" stays with deterministic code

SYSTEM_PROMPT = """You are the judgment layer of Kite, an assistive navigation system for a blind pedestrian. \
A drone above and behind the walker and a chest camera report what they see. You get a JSON snapshot: \
active hazards (each with a key and priority), nearby obstacles, the route, the last spoken instruction, \
the drone's latest scene description, and the walker's position.

Deterministic safety code has already spoken every stop and every "don't walk". Your job is to add one short \
spoken judgment only when the snapshot shows a risk the walker has not been told about, such as a crowded \
crosswalk, a car turning across their path, or construction narrowing the sidewalk.

Always answer by calling the decide tool exactly once.
- say: one calm sentence under 120 characters, from the walker's point of view (left, right, ahead, meters). \
Use an empty string when there is nothing new worth saying, which is most of the time.
- warning: true when say tells them to wait or take care.
- upgrade_key / upgrade_priority: to make an existing hazard more urgent, give its key and a higher priority \
(hazard is the highest you may use). Empty strings for no upgrade.

Never say it is safe to cross, never tell them to go, walk, continue or proceed, never suggest another route, \
and never contradict a stop or a "don't walk". You may only warn or tell them to wait. Only use what the \
snapshot contains; never guess."""

DECIDE_TOOL = {
    "name": "decide",
    "description": "Give at most one short spoken judgment and an optional hazard priority upgrade.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "say": {"type": "string", "description": "One sentence to speak, or empty for nothing."},
            "warning": {"type": "boolean", "description": "True if say is a warning or a wait."},
            "upgrade_key": {"type": "string", "description": "Key of an existing hazard, or empty."},
            "upgrade_priority": {"type": "string", "enum": ["", *UPGRADE_TO]},
        },
        "required": ["say", "warning", "upgrade_key", "upgrade_priority"],
        "additionalProperties": False,
    },
}

# Permission, reroutes and stop-clearing. Matching any of these drops the sentence.
_FORBIDDEN = re.compile(
    r"\b(?:safe|clear|okay|ok|fine)\s+to\s+(?:cross|go|walk|proceed)\b"
    r"|\b(?:it'?s|it\s+is|path\s+is|road\s+is|street\s+is|coast\s+is|all)\s+(?:safe|clear)\b"
    r"|\bgo\s+ahead\b|\bgo\s+now\b|\bcross\s+now\b|\bstart\s+(?:crossing|walking)\b|\bwalk\s+now\b"
    r"|\byou\s+(?:can|may|should)\s+(?:now\s+)?(?:cross|go|walk|continue|proceed|keep)\b"
    r"|\b(?:proceed|continue|keep\s+(?:walking|going)|carry\s+on|move\s+on|go\s+on)\b"
    r"|\bno\s+(?:longer\s+)?need\s+to\s+(?:stop|wait)\b|\bdon'?t\s+(?:need\s+to\s+)?(?:stop|wait)\b"
    r"|\b(?:reroute|re-route|detour|another\s+route|different\s+route|other\s+way|instead)\b"
    r"|\bsafe\b",
    re.I)


def validate(inp, hazards: dict[str, str]) -> tuple[str | None, tuple[str, str] | None, bool, list[str]]:
    """Tool input -> (say, upgrade, warning, reasons rejected). hazards maps key -> current priority."""
    rejected: list[str] = []
    if not isinstance(inp, dict):
        return None, None, False, ["input is not an object"]
    say = inp.get("say")
    say = " ".join(say.split()) if isinstance(say, str) else ""
    if len(say) > MAX_SAY:
        rejected.append(f"say too long ({len(say)})")
        say = ""
    elif say and _FORBIDDEN.search(say):
        rejected.append(f"forbidden phrase: {say!r}")
        say = ""
    up = None
    key, pri = inp.get("upgrade_key"), inp.get("upgrade_priority")
    if key or pri:
        if not isinstance(key, str) or key not in hazards:
            rejected.append(f"unknown hazard key {key!r}")
        elif pri not in UPGRADE_TO:
            rejected.append(f"bad priority {pri!r}")
        elif PRIORITY_RANK[pri] >= PRIORITY_RANK.get(hazards[key], len(PRIORITY_RANK)):
            rejected.append(f"not an upgrade: {key} {hazards[key]} -> {pri}")
        else:
            up = (key, pri)
    return say or None, up, bool(inp.get("warning")), rejected


class Judgment:
    """Background worker. Call notify() when hazards or the scene change; it does the rest."""

    def __init__(self, snapshot: Callable[[], dict], say: Callable[..., object],
                 upgrade: Callable[[str, str], bool], event: Callable[[str, dict], None],
                 api_key: str | None = None, client=None, model: str = MODEL,
                 min_gap_s: float = 3.5, timeout_s: float = 2.5, log=print):
        self.snapshot = snapshot
        self.say = say
        self.upgrade = upgrade
        self.event = event
        self.model = model
        self.min_gap_s = min_gap_s
        self.timeout_s = timeout_s
        self.log = log
        self.client = client
        if client is None and api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)
            except ImportError:
                self.log("[judgment] anthropic SDK not installed: judgment off")
        self.stats = {"calls": 0, "said": 0, "upgrades": 0, "rejected": 0, "dropped": 0, "skipped": 0}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._last_call = -1e9
        self._last_seen: str | None = None
        if self.client is not None:
            threading.Thread(target=self._run, name="judgment", daemon=True).start()

    @property
    def online(self) -> bool:
        return self.client is not None

    def notify(self) -> None:
        """Cheap and thread-safe: just wakes the worker."""
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            if self._stop.is_set():
                return
            wait = self._last_call + self.min_gap_s - time.monotonic()
            if wait > 0 and self._stop.wait(wait):     # rate limit; changes meanwhile are coalesced
                return
            self._wake.clear()
            try:
                self.step()
            except Exception as e:
                self.log(f"[judgment] {type(e).__name__}: {e}")

    @staticmethod
    def fingerprint(snap: dict) -> str:
        hz = sorted((h.get("key"), h.get("state"), h.get("text"), h.get("priority"))
                    for h in snap.get("hazards") or [])
        return json.dumps([hz, snap.get("scene")], default=str)

    def step(self) -> str:
        """One decision. Returns what happened (for tests and logs)."""
        snap = self.snapshot()
        fp = self.fingerprint(snap)
        if fp == self._last_seen:
            self.stats["skipped"] += 1
            return "unchanged"
        if not snap.get("hazards") and not snap.get("scene"):
            self._last_seen = fp
            self.stats["skipped"] += 1
            return "empty"
        self._last_seen = fp
        self._last_call = time.monotonic()
        self.stats["calls"] += 1
        t0 = time.monotonic()
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=300,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                tools=[DECIDE_TOOL], tool_choice={"type": "auto"},
                thinking={"type": "disabled"}, output_config={"effort": "low"},
                messages=[{"role": "user", "content": json.dumps(snap, separators=(",", ":"), default=str)}],
            )
        except Exception as e:
            self.stats["dropped"] += 1
            self.log(f"[judgment] dropped ({type(e).__name__}: {e})")
            return "dropped"
        ms = (time.monotonic() - t0) * 1000
        if ms > self.timeout_s * 1000:
            self.stats["dropped"] += 1
            self.log(f"[judgment] dropped: answered in {ms:.0f} ms, too late to be useful")
            return "late"
        use = next((b for b in resp.content if getattr(b, "type", None) == "tool_use" and b.name == "decide"), None)
        if use is None:
            self.log(f"[judgment] no decision ({resp.stop_reason}, {ms:.0f} ms)")
            return "none"
        hazards = {h["key"]: h.get("priority") or "hazard" for h in snap.get("hazards") or [] if h.get("key")}
        text, up, warning, rejected = validate(use.input, hazards)
        if rejected:
            self.stats["rejected"] += len(rejected)
            self.log(f"[judgment] rejected: {'; '.join(rejected)}")
        if up and self.upgrade(*up):
            self.stats["upgrades"] += 1
        if text:
            self.stats["said"] += 1
            pri = "hazard" if warning or up else "nav"
            self.say(text, pri, key="judgment", ttl=4.0)
        if text or up:
            self.event("say", {"source": "judgment", "text": text, "priority": "hazard" if warning or up else "nav",
                               "upgrade": {"key": up[0], "priority": up[1]} if up else None,
                               "latency_ms": round(ms)})
        self.log(f"[judgment] {ms:.0f} ms: say={text!r} upgrade={up}")
        return "said" if text else ("upgraded" if up else "quiet")
