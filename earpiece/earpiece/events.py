"""The wire contract: JSON messages other Kite components send to the earpiece.

Every message is one JSON object with a "type". Field names accept the short
form ("bearing") or the explicit form ("bearing_deg"). Bearings are degrees in
the wearer's frame (0 ahead, +90 right, -90 left); distances are metres.

    obstacles  full set of objects in one perception frame (missing ids = gone)
               {"type":"obstacles","source":"webcam","seq":12,
                "objects":[{"id":"7","kind":"person","bearing":-20,"distance":2.4,
                            "closing_speed":0.4,"confidence":0.8}]}
    guidance   a navigation instruction to speak
               {"type":"guidance","text":"In 20 meters, turn left onto Lyon Street",
                "key":"step-3","ttl":8}
    beacon     direction of the next waypoint (Soundscape-style audio beacon)
               {"type":"beacon","bearing":-35,"distance":42,"label":"Palace of Fine Arts"}
               {"type":"beacon","active":false}
    hazard     something on the route: crosswalk signal, crowd, construction
               {"type":"hazard","kind":"crosswalk","state":"dont_walk","bearing":5,"distance":9}
    say        free text to speak   {"type":"say","text":"...","priority":"info"}
    scene      context for questions, not spoken  {"type":"scene","text":"Wide sidewalk..."}
    position   {"type":"position","lat":37.80,"lng":-122.45,"heading":270,"place":"Lyon St"}
    status     {"type":"status","source":"drone","battery":34,"state":"following"}
    heartbeat  {"type":"heartbeat","source":"webcam"}
    control    {"type":"control","action":"mute_cues"|"unmute_cues"|"pause"|"resume"
                |"volume_up"|"volume_down"|"repeat"|"beacon_on"|"beacon_off"}

Everything is validated and clamped here so a buggy or hostile sender can
never crash the audio loop or produce a misleading cue.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .spatial import wrap_deg

PRIORITIES = ("critical", "hazard", "nav", "answer", "info", "system")
PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}   # lower = more important
MAX_OBJECTS = 64
MAX_TEXT = 400
MAX_DISTANCE_M = 500.0
CONTROL_ACTIONS = {"mute_cues", "unmute_cues", "pause", "resume", "volume_up", "volume_down",
                   "repeat", "beacon_on", "beacon_off", "stop_speaking"}

_KIND_ALIASES = {
    "pedestrian": "person", "people": "person", "human": "person",
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle",
    "bicycle": "bike", "cyclist": "bike", "scooter": "bike",
    "stairs": "stairs_down", "step": "curb", "drop": "curb",
}


class BadMessage(ValueError):
    pass


def _num(d: dict, *names: str, default: float | None = None, lo: float | None = None,
         hi: float | None = None, required: bool = False) -> float | None:
    for n in names:
        if n in d and d[n] is not None:
            v = d[n]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise BadMessage(f"{n} must be a number")
            v = float(v)
            if not math.isfinite(v):
                raise BadMessage(f"{n} must be finite")
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v
    if required:
        raise BadMessage(f"missing {names[0]}")
    return default


def _text(d: dict, name: str = "text", required: bool = False) -> str | None:
    v = d.get(name)
    if v is None:
        if required:
            raise BadMessage(f"missing {name}")
        return None
    if not isinstance(v, str):
        raise BadMessage(f"{name} must be a string")
    v = " ".join(v.split())[:MAX_TEXT]
    if required and not v:
        raise BadMessage(f"empty {name}")
    return v or None


def _ident(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v)[:64]
    return re.sub(r"[^\w.:-]", "_", s)


def _kind(v: Any) -> str:
    k = re.sub(r"[^a-z_]", "", str(v or "obstacle").lower().replace(" ", "_").replace("-", "_"))[:32]
    return _KIND_ALIASES.get(k, k or "obstacle")


@dataclass
class Obstacle:
    id: str
    kind: str
    bearing: float
    distance: float
    closing_speed: float | None = None
    confidence: float = 1.0


@dataclass
class Msg:
    type: str
    source: str = ""
    seq: int | None = None
    objects: list[Obstacle] = field(default_factory=list)
    text: str | None = None
    key: str | None = None
    ttl: float | None = None
    priority: str = "info"
    bearing: float | None = None
    distance: float | None = None
    label: str | None = None
    active: bool = True
    kind: str | None = None
    state: str | None = None
    action: str | None = None
    data: dict = field(default_factory=dict)


def _priority(d: dict, default: str) -> str:
    p = str(d.get("priority", default)).lower()
    return p if p in PRIORITY_RANK else default


def parse(raw: bytes | str | dict) -> Msg:
    """Raw datagram / websocket frame / dict -> validated Msg. Raises BadMessage."""
    if isinstance(raw, (bytes, bytearray)):
        if len(raw) > 256 * 1024:
            raise BadMessage("message too large")
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise BadMessage("not utf-8") from e
    if isinstance(raw, str):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as e:
            raise BadMessage(f"bad json: {e.msg}") from e
    else:
        d = raw
    if not isinstance(d, dict):
        raise BadMessage("message must be a JSON object")
    t = str(d.get("type", "")).lower()
    src = _ident(d.get("source"), "unknown")
    seq = d.get("seq")
    seq = int(seq) if isinstance(seq, (int, float)) and not isinstance(seq, bool) and math.isfinite(seq) else None

    if t in ("obstacles", "obstacle"):
        items = d.get("objects") if t == "obstacles" else [d]
        if not isinstance(items, list):
            raise BadMessage("objects must be a list")
        objs: list[Obstacle] = []
        for i, o in enumerate(items[: MAX_OBJECTS * 4]):
            if not isinstance(o, dict):
                continue
            try:
                dist = _num(o, "distance", "distance_m", lo=0.0, hi=MAX_DISTANCE_M, required=True)
                bear = _num(o, "bearing", "bearing_deg", required=True)
            except BadMessage:
                continue                      # one bad object never drops the whole frame
            cs = _num(o, "closing_speed", "closing_speed_mps", lo=-30.0, hi=30.0)
            conf = _num(o, "confidence", "conf", default=1.0, lo=0.0, hi=1.0)
            objs.append(Obstacle(_ident(o.get("id"), f"{src}-{i}"), _kind(o.get("kind", o.get("class"))),
                                 wrap_deg(bear), dist, cs, conf))
        objs.sort(key=lambda o: o.distance)
        return Msg("obstacles" if t == "obstacles" else "obstacle", src, seq, objects=objs[:MAX_OBJECTS],
                   ttl=_num(d, "ttl", lo=0.05, hi=10.0))
    if t == "guidance":
        return Msg("guidance", src, seq, text=_text(d, required=True), key=_ident(d.get("key")) or None,
                   ttl=_num(d, "ttl", lo=0.5, hi=120.0), priority=_priority(d, "nav"),
                   bearing=_opt_bearing(d), distance=_num(d, "distance", "distance_m", lo=0.0, hi=1e5))
    if t == "beacon":
        active = bool(d.get("active", True))
        if not active:
            return Msg("beacon", src, seq, active=False)
        return Msg("beacon", src, seq, bearing=wrap_deg(_num(d, "bearing", "bearing_deg", required=True)),
                   distance=_num(d, "distance", "distance_m", lo=0.0, hi=1e5),
                   label=_text(d, "label"), active=True)
    if t == "hazard":
        return Msg("hazard", src, seq, kind=_kind(d.get("kind", "hazard")),
                   state=_ident(d.get("state")).lower() or None, text=_text(d),
                   bearing=_opt_bearing(d), distance=_num(d, "distance", "distance_m", lo=0.0, hi=1e4),
                   ttl=_num(d, "ttl", lo=0.5, hi=120.0), key=_ident(d.get("key")) or None,
                   priority=_priority(d, "hazard"))
    if t == "say":
        return Msg("say", src, seq, text=_text(d, required=True), priority=_priority(d, "info"),
                   key=_ident(d.get("key")) or None, ttl=_num(d, "ttl", lo=0.5, hi=120.0))
    if t == "scene":
        return Msg("scene", src, seq, text=_text(d, required=True))
    if t == "position":
        data = {k: _num(d, k) for k in ("lat", "lng", "heading", "accuracy")}
        data["place"] = _text(d, "place")
        return Msg("position", src, seq, data=data)
    if t == "status":
        data = {"battery": _num(d, "battery", lo=0, hi=100), "state": _text(d, "state")}
        return Msg("status", src, seq, text=_text(d), data=data)
    if t == "heartbeat":
        return Msg("heartbeat", src, seq)
    if t == "control":
        a = str(d.get("action", "")).lower()
        if a not in CONTROL_ACTIONS:
            raise BadMessage(f"unknown control action {a!r}")
        return Msg("control", src, seq, action=a)
    raise BadMessage(f"unknown type {t!r}")


def _opt_bearing(d: dict) -> float | None:
    b = _num(d, "bearing", "bearing_deg")
    return None if b is None else wrap_deg(b)


def clock_phrase(bearing: float) -> str:
    """Bearing -> spoken direction. Uses plain words near the axes and clock positions
    elsewhere (the convention O&M instructors teach)."""
    b = wrap_deg(bearing)
    a = abs(b)
    if a <= 12:
        return "ahead"
    if a >= 160:
        return "behind you"
    side = "right" if b > 0 else "left"
    if a <= 35:
        return f"slightly {side}"
    if 75 <= a <= 105:
        return f"on your {side}"
    hour = round(b / 30.0) % 12 or 12
    return f"at {hour} o'clock"


def distance_phrase(m: float | None) -> str:
    if m is None:
        return ""
    if m < 1.0:
        return "very close"
    if m < 10:
        v = round(m * 2) / 2
        return f"{v:g} meter" + ("" if v == 1 else "s")
    if m < 100:
        return f"{int(round(m / 5.0) * 5)} meters"
    return f"{int(round(m / 10.0) * 10)} meters"
