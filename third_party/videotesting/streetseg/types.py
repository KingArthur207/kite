"""Shared records.

The earpiece consumes :class:`streetseg.events.Event`. :class:`Cue` is the
internal walking decision. Kite has no glasses, so nothing here is a turn of
-1, 0, or +1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """One detector box. xyxy is in pixels, x2/y2 exclusive."""

    label: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None
    signal: str | None = None
    source: str = "drone"

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass(frozen=True)
class ColumnScore:
    name: str
    walkable: float
    blocked: float
    people_pixels: float
    people: int
    vehicles: int
    score: float
    adjusted: float
    selectable: bool


@dataclass(frozen=True)
class Cue:
    """What the forward view decided, before it becomes earpiece events."""

    direction: str
    speech: str
    reason: str
    crowd: str
    signal: str
    crosswalk: bool
    hazards: tuple[str, ...] = ()
    columns: tuple[ColumnScore, ColumnScore, ColumnScore] | None = None
    wearer_excluded: bool = False

    @property
    def stop(self) -> bool:
        return self.direction == "stop"

    def to_dict(self) -> dict:
        columns = {}
        if self.columns is not None:
            for col in self.columns:
                columns[col.name] = {
                    "walkable": round(col.walkable, 3),
                    "blocked": round(col.blocked, 3),
                    "people": col.people,
                    "vehicles": col.vehicles,
                    "score": round(col.score, 3),
                    "selectable": col.selectable,
                }
        return {
            "direction": self.direction,
            "speech": self.speech,
            "reason": self.reason,
            "crowd": self.crowd,
            "signal": self.signal,
            "crosswalk": self.crosswalk,
            "hazards": list(self.hazards),
            "stop": self.stop,
            "wearer_excluded": self.wearer_excluded,
            "columns": columns,
        }


@dataclass
class FrameResult:
    events: tuple
    labels: np.ndarray | None = None
    boxes: list[Box] = field(default_factory=list)
    image: np.ndarray | None = None
    cue: Cue | None = None
