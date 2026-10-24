"""World-frame perception: short-term obstacle memory (chest) and the corner car watch (drone).

World frame: metres east (x) and north (y), compass headings (0 north, 90 east, like the sim's pose). The sim
sends the walker's pose; on real hardware without positions, DeadReckoner integrates heading and walking speed.
Both rules are deterministic (spec 5.3): no model and no Claude decides a stop or clears one.
"""

from __future__ import annotations

import math
from collections import deque

MEMORY_S = 5.0          # an obstacle out of view is forgotten after this many seconds of walking
PASSED_M = 0.3          # ... or once it is this far behind the camera
MEMORY_MAX_M = 8.0      # ... or this far away
MOVING_MPS = 0.2        # slower than this, the walker is standing: memory does not age (the thing is still there)
MOVERS = {"person", "vehicle", "bike", "dog"}     # they move: never remembered where they were
MERGE_M = 0.7           # a sighting this close to a remembered obstacle of the same kind is that obstacle
MOVER_R, MOVER_KEEP_S = 1.0, 2.0   # no memory within MOVER_R of where a person or car was in the last MOVER_KEEP_S:
                                   # depth sees a crossing pedestrian too, and must not pin her spot as an obstacle

CORNER_NEAR_M = 20.0    # walker this close to the crossing: watch for cars approaching it
CORNER_CAR_M = 45.0     # cars this close to the crossing count
APPROACH_MPS = 1.0      # closing on the crossing at least this fast
PASS_MARGIN_S = 1.5     # held this long past the car's predicted arrival at the crossing
MAX_HOLD_S = 12.0       # never hold longer than this per sighting (a crawling car is re-judged when seen again)
CORNER_TTL = 1.0        # s; the hazard is refreshed every drone frame and lapses this long after the last
CORNER_TEXT = "Car approaching the corner."
WALK_SPEED = 1.2        # m/s, dead reckoning (the sim walker's speed; calibrate per wearer)


def to_world(cam: tuple[float, float, float], bearing: float, distance: float) -> tuple[float, float]:
    """(x, y, compass heading) of the camera + bearing (deg, + right) and ground distance -> world point."""
    h = math.radians(cam[2] + bearing)
    return cam[0] + distance * math.sin(h), cam[1] + distance * math.cos(h)


def to_walker(walker: tuple[float, float, float], x: float, y: float) -> tuple[float, float]:
    """World point -> (bearing deg in [-180, 180), + right; distance m) from the walker."""
    dx, dy = x - walker[0], y - walker[1]
    b = (math.degrees(math.atan2(dx, dy)) - walker[2] + 540) % 360 - 180
    return b, math.hypot(dx, dy)


def walker_of(pose: dict | None) -> tuple[float, float, float] | None:
    if not pose or "x" not in pose or "heading" not in pose:
        return None
    return float(pose["x"]), float(pose["y"]), float(pose["heading"])


def drone_cam(pose: dict | None, behind: float = 1.0) -> tuple[float, float, float] | None:
    """Where the drone camera is and looks: its own position when the pose has one (scouting a corner),
    else `behind` m behind the walker, looking where the walker looks."""
    w = walker_of(pose)
    if w is None:
        return None
    if "cam_x" in pose:
        return float(pose["cam_x"]), float(pose["cam_y"]), float(pose.get("cam_heading", w[2]))
    h = math.radians(w[2])
    return w[0] - behind * math.sin(h), w[1] - behind * math.cos(h), w[2]


class DeadReckoner:
    """Pose for a camera that has no position: heading (from the phone, or 0) integrated at walking speed.
    ponytail: assumes the wearer is always walking at `speed`; feed it a step detector if standing matters."""

    def __init__(self, speed: float = WALK_SPEED):
        self.speed, self.x, self.y, self.t = speed, 0.0, 0.0, None

    def pose(self, t: float, heading: float = 0.0) -> dict:
        if self.t is not None:
            d = self.speed * max(0.0, t - self.t)
            self.x += d * math.sin(math.radians(heading))
            self.y += d * math.cos(math.radians(heading))
        self.t = t
        return {"x": self.x, "y": self.y, "heading": heading}


class ObstacleMemory:
    """Static obstacles the chest camera saw, kept in the world frame so they still count after they drop out of
    view (a bin under a level camera vanishes inside ~1 m). Each frame: step(objects, pose, t) returns the objects
    plus every remembered one not in them, with its bearing and distance recomputed from the walker's pose."""

    def __init__(self):
        self.mem: dict[str, dict] = {}
        self.last: tuple[float, float, float] | None = None    # t, x, y of the previous step
        self.n = 0
        self.movers: list[tuple[float, float, float]] = []      # t, x, y

    def _key(self, kind: str, x: float, y: float) -> str:
        """Matched by place, not by id: depth reports every obstacle as "depth", trackers renumber."""
        near = [(math.hypot(m["x"] - x, m["y"] - y), k) for k, m in self.mem.items() if m["kind"] == kind]
        if near and min(near)[0] <= MERGE_M:
            return min(near)[1]
        self.n += 1
        return f"{kind}{self.n}"

    def step(self, objs: list[dict], pose: dict | None, t: float) -> list[dict]:
        w = walker_of(pose)
        if w is None:
            return objs
        dt = 0.0 if self.last is None else t - self.last[0]
        moving = dt > 0 and math.hypot(w[0] - self.last[1], w[1] - self.last[2]) / dt >= MOVING_MPS
        self.last = (t, w[0], w[1])
        seen = set()
        self.movers = [m for m in self.movers if t - m[0] <= MOVER_KEEP_S] + \
            [(t, *to_world(w, o["bearing"], o["distance"])) for o in objs if o["kind"] in MOVERS]
        for o in objs:
            if o["kind"] in MOVERS:
                continue
            x, y = to_world(w, o["bearing"], o["distance"])
            if any(math.hypot(mx - x, my - y) <= MOVER_R for _, mx, my in self.movers):
                continue
            k = self._key(o["kind"], x, y)
            old = self.mem.get(k)
            if old and math.hypot(old["x"] - w[0], old["y"] - w[1]) < o["distance"]:
                x, y = old["x"], old["y"]   # keep the nearer estimate: depth read a bin 1.3 m off as 1.9 m in the
            self.mem[k] = {**o, "x": x, "y": y, "age": 0.0}   # last frames before it left the view
            seen.add(k)
        out = list(objs)
        for k, m in list(self.mem.items()):
            if k in seen:
                continue
            m["age"] += dt if moving else 0.0
            b, d = to_walker(w, m["x"], m["y"])
            ahead = d * math.cos(math.radians(b))
            if m["age"] > MEMORY_S or ahead < -PASSED_M or d > MEMORY_MAX_M:
                del self.mem[k]
                continue
            out.append({**{f: v for f, v in m.items() if f not in ("x", "y", "age")},
                        "id": f"mem:{k}", "bearing": round(b, 1), "distance": round(d, 2), "cut_off": False})
        return out


class CornerWatch:
    """The drone sees a car closing on the crossing ahead -> hazard `vehicle` with state wait, held until the car
    should have passed it (its last distance over its speed, plus PASS_MARGIN_S). Only while the walker is within
    CORNER_NEAR_M of the crossing, which comes from the route (in the sim: the zebra at the origin).
    ponytail: a car driving straight past the corner on the cross street also holds the walker until it is by;
    tell turning from straight-on with the lane it is in if those waits get annoying."""

    def __init__(self, crossing: tuple[float, float] | None):
        self.c = crossing
        self.hist: dict[str, deque] = {}
        self.until = -math.inf
        self.car: tuple[float, float] | None = None

    def _closing(self, h: deque) -> float | None:
        pts = [p for p in h if h[-1][0] - p[0] <= 1.0]
        if len(pts) < 3 or pts[-1][0] - pts[0][0] < 0.3:
            return None
        mt = sum(p[0] for p in pts) / len(pts)
        mr = sum(p[1] for p in pts) / len(pts)
        den = sum((p[0] - mt) ** 2 for p in pts)
        return -sum((p[0] - mt) * (p[1] - mr) for p in pts) / den

    def step(self, cars: list[tuple[str, float, float]], pose: dict | None, t: float) -> dict | None:
        """cars: (track id, world x, world y) seen this drone frame. Returns the hazard event to send, or None."""
        w = walker_of(pose)
        if self.c is None or w is None:
            return None
        near = math.hypot(w[0] - self.c[0], w[1] - self.c[1]) <= CORNER_NEAR_M
        for k in [k for k, h in self.hist.items() if t - h[-1][0] > 2.0]:
            del self.hist[k]
        for cid, x, y in cars:
            r = math.hypot(x - self.c[0], y - self.c[1])
            h = self.hist.setdefault(cid, deque(maxlen=16))
            h.append((t, r))
            v = self._closing(h)
            if near and v is not None and v >= APPROACH_MPS and r <= CORNER_CAR_M:
                self.until = max(self.until, t + min(MAX_HOLD_S, r / v + PASS_MARGIN_S))
                self.car = (x, y)
        if t >= self.until or self.car is None:
            return None
        b, d = to_walker(w, *self.car)
        return {"type": "hazard", "kind": "vehicle", "key": "corner", "state": "wait", "text": CORNER_TEXT,
                "bearing": round(b, 1), "distance": round(d, 1), "ttl": CORNER_TTL}
