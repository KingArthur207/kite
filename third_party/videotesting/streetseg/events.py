"""Earpiece events.

A forward view (drone or sim) emits ``scene`` and, when the path is blocked,
``hazard``. The chest camera emits ``obstacle``. A direction the walker can
take is a ``say`` with priority ``nav`` ("Bear left.") and a ``hint`` the
judgment layer can read ("bear left"). A stop is said at once, with priority
``hazard``. There is no glasses turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Cue

_HINT = {
    "left": "bear left",
    "right": "bear right",
    "straight": "continue straight",
    "stop": "stop",
}


@dataclass(frozen=True)
class Event:
    kind: str
    source: str
    priority: str
    text: str
    direction: str = ""
    reason: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        body = {
            "kind": self.kind,
            "source": self.source,
            "priority": self.priority,
            "text": self.text,
        }
        if self.direction:
            body["direction"] = self.direction
        if self.reason:
            body["reason"] = self.reason
        if self.detail:
            body["detail"] = dict(self.detail)
        return body


def scene_text(cue: Cue) -> str:
    if not cue.columns:
        return "Street ahead."
    bits = []
    walkable = [col.name for col in cue.columns if col.selectable]
    if walkable:
        bits.append("walkable " + " and ".join(walkable))
    people = sum(col.people for col in cue.columns)
    vehicles = sum(col.vehicles for col in cue.columns)
    if people:
        bits.append(f"{people} person" if people == 1 else f"{people} people")
    if vehicles:
        bits.append(f"{vehicles} vehicle" if vehicles == 1 else f"{vehicles} vehicles")
    if cue.signal in ("red", "yellow", "green"):
        bits.append(f"{cue.signal} light")
    if cue.crosswalk:
        bits.append("crossing")
    if not bits:
        return "No walkable ground ahead."
    text = ", ".join(bits)
    return text[0].upper() + text[1:] + "."


def _columns(cue: Cue) -> dict:
    columns = {}
    if cue.columns:
        for col in cue.columns:
            columns[col.name] = {
                "walkable": round(col.walkable, 3),
                "blocked": round(col.blocked, 3),
                "people": col.people,
                "vehicles": col.vehicles,
                "selectable": col.selectable,
            }
    return {
        "signal": cue.signal,
        "crosswalk": cue.crosswalk,
        "crowd": cue.crowd,
        "wearer_excluded": cue.wearer_excluded,
        "columns": columns,
    }


def forward_events(
    raw: Cue,
    spoken: Cue,
    source: str,
    *,
    say: bool,
    pending: str,
    stable: int,
    extra: dict | None = None,
) -> list[Event]:
    """Scene from this frame. Say and hint follow the direction that has held."""
    detail = _columns(raw)
    detail["seen"] = raw.direction
    detail["pending"] = pending
    detail["stable_frames"] = stable
    if extra:
        detail.update(extra)
    events = [
        Event("scene", source, "info", scene_text(raw), spoken.direction, raw.reason, detail)
    ]
    holding = spoken.direction == "stop" and raw.direction != "stop"
    if spoken.direction == "stop":
        hazard = {"hazards": list(spoken.hazards), "holding": holding}
        events.append(Event("hazard", source, "hazard", spoken.speech, "stop", spoken.reason, hazard))
        if say:
            events.append(Event("say", source, "hazard", spoken.speech, "stop", spoken.reason))
        events.append(Event(
            "hint", source, "hazard", "stop", "stop", spoken.reason,
            {"pending": pending, "stable_frames": stable},
        ))
        return events
    if say:
        events.append(Event("say", source, "nav", spoken.speech, spoken.direction, spoken.reason))
    events.append(Event(
        "hint", source, "nav", _HINT[spoken.direction], spoken.direction, spoken.reason,
        {"pending": pending, "stable_frames": stable},
    ))
    return events


def obstacle_events(
    speech: str,
    reason: str,
    hazards: list[str],
    detail: dict,
    *,
    say: bool,
) -> list[Event]:
    events = [Event("obstacle", "chest", "hazard", speech, "stop", reason, detail)]
    if say:
        events.append(Event("say", "chest", "hazard", speech, "stop", reason))
    events.append(Event("hint", "chest", "hazard", "stop", "stop", reason, dict(detail)))
    return events
