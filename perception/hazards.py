"""Drone view -> `hazard` and `scene` events.

Crowds and cars use the stand-in's rules unchanged (sim/standin_detect.py drone_hazards: parked-car
filter, lateral closing speed, the walker cut off by the bottom edge never counts). On top of them, a
fixed street object in the walker's path within PATH_RANGE becomes a hazard of its own kind:
construction, overhead (a branch or sign the cane passes under), curb / stairs_down, or obstacle
(a pole, bollard, e-scooter, bench... named in the spoken text). The earpiece names construction,
overhead, curb and stairs_down itself (app.py _HAZARD_NAMES); we always send text, so it says ours.

Same gating as the stand-in: a hazard must hold ONSET_FRAMES frames, its text is fixed when it first
fires (the earpiece re-speaks only when the text changes) and re-sent every ttl/2 while it holds; a
new direction must also hold ONSET_FRAMES; clearing sends nothing and the ttl runs out.
"""

from __future__ import annotations

import math

from earpiece.events import clock_phrase, distance_phrase
from standin_detect import HAZARD_TTL, ONSET_FRAMES, drone_hazards, drone_to_walker

from .vocab import MOVING, spoken

PATH_RANGE = 10.0          # m ahead of the walker
PATH_HALF = 0.75           # m either side of the walker's line
HAZARD_KIND = {"construction": "construction", "overhead": "overhead", "curb": "curb",
               "stairs_down": "stairs_down"}


def path_hazards(objs: list[dict]) -> dict[str, dict]:
    """Fixed objects in the walker's path -> {hazard kind: {bearing, distance, label}}, nearest per kind.
    Objects are in the drone frame (bearing, ground distance); the walker is 1 m ahead of the drone."""
    out: dict[str, dict] = {}
    for o in objs:
        if o["kind"] in MOVING or o.get("cut_off"):
            continue
        b, d = drone_to_walker(o["bearing"], o["distance"])
        ahead, lat = d * math.cos(math.radians(b)), d * math.sin(math.radians(b))
        if not (0 < ahead <= PATH_RANGE and abs(lat) <= PATH_HALF):
            continue
        k = HAZARD_KIND.get(o["kind"], "obstacle")
        if k not in out or d < out[k]["distance"]:
            out[k] = {"bearing": b, "distance": d, "label": o["kind"]}
    return out


def hazard_text(kind: str, c: dict) -> str:
    where, far = clock_phrase(c["bearing"]), distance_phrase(c["distance"])
    name = {"crowd": "Crowd", "vehicle": "Car", "construction": "Construction", "overhead": "Overhead obstacle",
            "curb": "Curb", "stairs_down": "Step down"}.get(kind) or spoken(c.get("label", "obstacle")).capitalize()
    return f"{name} {where}, {far}."


class Gate:
    """HazardGate for any hazard kind (the stand-in's only words crowds and cars)."""

    def __init__(self, onset: int = ONSET_FRAMES, ttl: float = HAZARD_TTL):
        self.onset, self.ttl = onset, ttl
        self.streak: dict[str, int] = {}
        self.live: dict[str, dict] = {}

    def step(self, cands: dict[str, dict], now: float) -> list[dict]:
        out = []
        for k in list(self.streak):
            if k not in cands:
                del self.streak[k]
                self.live.pop(k, None)
        for k, c in cands.items():
            self.streak[k] = self.streak.get(k, 0) + 1
            if self.streak[k] < self.onset:
                continue
            direction = clock_phrase(c["bearing"])
            live = self.live.get(k)
            turned = False
            if live:
                live["n"] = live["n"] + 1 if direction == live["pending"] else 1
                live["pending"] = direction
                turned = direction != live["dir"] and live["n"] >= self.onset
                if not turned and now - live["sent"] < self.ttl / 2:
                    continue
            if not live or turned:
                live = {"text": hazard_text(k, c), "dir": direction, "pending": direction, "n": self.onset}
                self.live[k] = live
            live["sent"] = now
            out.append({"type": "hazard", "kind": k, "text": live["text"], "bearing": round(c["bearing"], 1),
                        "distance": round(c["distance"], 1), "ttl": self.ttl})
        return out


def all_hazards(objs: list[dict], lateral=None, t: float = 0.0, pose: dict | None = None, still=None) -> dict:
    """Crowd and vehicle (stand-in rules) plus fixed objects in the path."""
    return {**drone_hazards(objs, lateral, t, pose, still), **path_hazards(objs)}


def scene_text(objs: list[dict]) -> str:
    """'Drone view: 2 persons, 1 pole, 1 e-scooter.' (the walker, cut off at the bottom, is left out)."""
    cnt: dict[str, int] = {}
    for o in objs:
        if not (o["kind"] == "person" and o.get("cut_off")):
            cnt[o["kind"]] = cnt.get(o["kind"], 0) + 1
    return "Drone view: " + (", ".join(f"{v} {spoken(k)}" for k, v in sorted(cnt.items())) or "clear") + "."
