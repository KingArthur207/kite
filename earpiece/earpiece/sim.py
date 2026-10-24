"""Scripted street scenarios: run the whole earpiece without a camera or drone.

Each scenario is a generator of (time_s, message) pairs fed to the app exactly as
UDP datagrams would be, at the same rates the real perception produces (15 Hz).
Used for demos without hardware, for the integration tests, and for rendering
WAV files that prove what the wearer would hear.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

FPS = 15


def _frames(t0: float, t1: float, fn, source: str = "webcam") -> Iterator[tuple[float, dict]]:
    n = int((t1 - t0) * FPS)
    for i in range(n):
        t = t0 + i / FPS
        yield t, {"type": "obstacles", "source": source, "seq": int(t * FPS) + 1, "objects": fn(t)}


def walk() -> Iterator[tuple[float, dict]]:
    """A 40-second walk to the Palace of Fine Arts: route, beacon, a pedestrian
    coming straight at you, a pole to step around, a crosswalk, arrival."""
    yield 0.0, {"type": "position", "place": "Lyon Street and Bay Street", "lat": 37.8013, "lng": -122.4477}
    yield 0.2, {"type": "scene", "source": "drone",
                "text": "Wide sidewalk heading north. Light foot traffic. A crosswalk with a signal about 30 meters ahead."}
    yield 0.5, {"type": "guidance", "text": "Head north on Lyon Street for 60 meters.", "key": "step-1", "ttl": 10}
    for i in range(80):
        t = 1.0 + i * 0.5
        dist = max(2.0, 60.0 - t * 1.3)
        bearing = 8 * math.sin(t / 3) if t < 25 else (-45 if t < 28 else -5)
        yield t, {"type": "beacon", "bearing": bearing, "distance": dist, "label": "the Palace of Fine Arts"}

    def objs(t: float) -> list[dict]:
        out = []
        if 3 <= t <= 9:          # pedestrian walking toward the wearer, slightly left
            d = 7.0 - (t - 3) * 1.1
            if d > 0.4:
                out.append({"id": "p1", "kind": "person", "bearing": -12 + (t - 3) * 1.5, "distance": d,
                            "confidence": 0.9})
        if 11 <= t <= 17:        # a pole dead ahead, the wearer steps right of it
            d = 5.5 - (t - 11) * 1.0
            if d > 0.5:
                out.append({"id": "pole", "kind": "pole", "bearing": -4 - max(0, t - 14) * 12, "distance": d,
                            "confidence": 0.8})
        if 20 <= t <= 24:        # people standing off to the right, not in the path
            out.append({"id": "p2", "kind": "person", "bearing": 55, "distance": 3.0, "confidence": 0.7})
        return out

    yield from _frames(1.0, 40.0, objs)
    yield 18.0, {"type": "hazard", "kind": "crosswalk", "state": "dont_walk", "bearing": 3, "distance": 6, "ttl": 15}
    yield 22.5, {"type": "hazard", "kind": "crosswalk", "state": "walk", "bearing": 3, "distance": 4, "ttl": 15}
    yield 25.0, {"type": "guidance", "text": "Turn left onto Bay Street.", "key": "step-2", "ttl": 6,
                 "bearing": -60}
    yield 26.0, {"type": "hazard", "kind": "crowd", "text": "Small crowd ahead on the right.", "bearing": 30,
                 "ttl": 10, "priority": "nav"}
    yield 30.0, {"type": "status", "source": "drone", "battery": 28, "state": "following"}


def car() -> Iterator[tuple[float, dict]]:
    """A car backing out of a driveway from the right, fast. Must go critical."""
    def objs(t: float) -> list[dict]:
        d = max(0.6, 8.0 - (t - 0.5) * 3.0)
        return [{"id": "car1", "kind": "car", "bearing": 35 - (t - 0.5) * 8, "distance": d,
                 "closing_speed": 3.0, "confidence": 0.95}]
    yield from _frames(0.5, 4.0, objs)


def dropout() -> Iterator[tuple[float, dict]]:
    """Perception runs, dies for 3 s (camera unplugged), comes back."""
    def objs(t: float) -> list[dict]:
        return [{"id": "p", "kind": "person", "bearing": 0, "distance": 3.0, "confidence": 0.9}]
    yield from _frames(0.0, 2.0, objs)
    yield from _frames(5.0, 7.0, objs)


def crowd() -> Iterator[tuple[float, dict]]:
    """Twenty people around; only the two that matter may be sonified."""
    def objs(t: float) -> list[dict]:
        out = []
        for i in range(20):
            b = -80 + i * 8 + 3 * math.sin(t + i)
            d = 1.5 + (i % 5) * 1.2 + 0.3 * math.cos(t * 0.7 + i)
            out.append({"id": f"c{i}", "kind": "person", "bearing": b, "distance": d, "confidence": 0.8})
        return out
    yield from _frames(0.0, 6.0, objs)


SCENARIOS = {"walk": walk, "car": car, "dropout": dropout, "crowd": crowd}


def timeline(name: str) -> list[tuple[float, dict]]:
    return sorted(SCENARIOS[name](), key=lambda x: x[0])


def duration(name: str) -> float:
    return max(t for t, _ in timeline(name)) + 2.0


JSONL_GAP_S = 0.1   # lines without a "t" follow the previous one by this much


def load_jsonl(path: str | Path) -> list[dict]:
    """One event per line (earpiece wire format, optional "t"). Raises ValueError naming the bad line."""
    events = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: bad JSON ({e.msg})") from None
        if not isinstance(ev, dict) or "type" not in ev:
            raise ValueError(f"{path}:{n}: each line must be an event object with a type")
        events.append(ev)
    return events


def schedule(events: list[dict]) -> list[tuple[float, dict]]:
    """(seconds from start, event) with any absolute timestamps made relative, in file order."""
    ts = [e.get("t") for e in events]
    t0 = next((t for t in ts if isinstance(t, (int, float))), 0.0)
    out, last = [], 0.0
    for ev, t in zip(events, ts):
        last = t - t0 if isinstance(t, (int, float)) else last + JSONL_GAP_S
        out.append((last, {k: v for k, v in ev.items() if k != "t"}))
    return out


def jsonl_timeline(path: str | Path) -> list[tuple[float, dict]]:
    return sorted(schedule(load_jsonl(path)), key=lambda x: x[0])
