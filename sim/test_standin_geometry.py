"""Stand-in detector geometry and hazard gating, no model needed.  python sim/test_standin_geometry.py"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from standin_detect import (HazardGate, bearing_deg, distance_m, drone_hazards, drone_to_walker,  # noqa: E402
                            objects_from_boxes)

# bearing: centre is ahead, +x is right, fx=1000 -> 480 px off-centre is atan(0.48)
assert bearing_deg(480) == 0
assert math.isclose(bearing_deg(960), math.degrees(math.atan(0.48)))
assert bearing_deg(0) < 0 < bearing_deg(700)

# ground: chest 1.2 m, foot 120 px below horizon -> 10 m; 600 px -> 2 m
assert math.isclose(distance_m(480, 200, 1.2), 10.0)
assert math.isclose(distance_m(720, 400, 1.2), 1200 / 360)
# foot at/above horizon -> size-based: 1.7 m person 170 px tall -> 10 m
assert math.isclose(distance_m(360, 170, 1.2), 10.0)
assert math.isclose(distance_m(365, 170, 1.2), 10.0)      # too close to horizon to trust ground
# clamps
assert distance_m(361, 1, 1.2) == 50.0 and distance_m(720, 5000, 1.2) >= 0.3

# drone -> walker: something 5 m straight ahead of the drone is 4 m ahead of the walker
b, d = drone_to_walker(0, 5)
assert abs(b) < 1e-9 and math.isclose(d, 4.0)
b, d = drone_to_walker(30, 4)
assert b > 30 and d < 4

# the walker (person cut off by the bottom edge) is never part of a crowd
walker = (380, 500, 580, 720, "person", 0.9, 1)
far = [(250 + 150 * i, 350, 310 + 150 * i, 520, "person", 0.8, 10 + i) for i in range(3)]
objs = objects_from_boxes([walker] + far, 1000, 2.0, 960, 720)
assert objs[0]["cut_off"] and not any(o["cut_off"] for o in objs[1:])
assert "crowd" in drone_hazards(objs)
assert "crowd" not in drone_hazards(objects_from_boxes([walker] + far[:2], 1000, 2.0, 960, 720))
car = objects_from_boxes([(400, 400, 560, 560, "car", 0.9, 5)], 1000, 2.0, 960, 720)
assert "vehicle" in drone_hazards(car)
side = objects_from_boxes([(900, 450, 960, 560, "car", 0.9, 5)], 1000, 2.0, 960, 720)   # far off the path
assert "vehicle" not in drone_hazards(side)
assert objects_from_boxes([(0, 0, 10, 10, "tv", 0.9, None)], 1000, 1.2, 960, 720) == []

# vehicles over time (Moss St): `lateral` tracks each car's sideways offset from the walker's path
from live_walk import RangeFilter  # noqa: E402


def drone_car(fwd: float, lat: float, tid: str = "c") -> dict:
    """A car `fwd` m ahead of the walker and `lat` m to the side, as the drone (1 m behind) sees it."""
    return {"id": tid, "kind": "vehicle", "bearing": math.degrees(math.atan2(lat, fwd + 1.0)),
            "distance": math.hypot(fwd + 1.0, lat), "confidence": 0.9, "cut_off": False}


def vehicle_frames(path, dt=0.1):
    rf = RangeFilter()
    return ["vehicle" in drone_hazards([drone_car(f, l)], rf, i * dt) for i, (f, l) in enumerate(path)]


# (a) walker on the sidewalk, oncoming car in the lane 2.5 m to the side: never a hazard
assert not any(vehicle_frames([(11.0 - 0.6 * i, 2.5 + (0.1 if i % 2 else -0.1)) for i in range(15)]))
assert not any(vehicle_frames([(8.0, -2.5)] * 10))                 # parked / waiting car on the other side
assert not any(vehicle_frames([(6.0, 2.0 + 0.3 * i) for i in range(10)]))   # driving away from the path
# (b) a car crossing ahead: flagged once its approach is measured, well before it is in the path
cross = vehicle_frames([(6.0, 7.0 - 0.4 * i) for i in range(10)])  # 4 m/s toward the path from 7 m out
assert not cross[0] and cross[-1] and cross.index(True) <= 5       # flagged while still > 4 m to the side
assert all(vehicle_frames([(5.0, 0.8)] * 3))                       # in the path: flagged from the first frame

# (c) parked cars (Street View, the Moss/Howard corner): with the walker's pose, a car 1.2 m off the path that
# stays put is never a hazard, while the walker stands or walks past it; the same offset while driving still counts
from standin_detect import StillVehicles  # noqa: E402


def walk_past(car_xy, walker_y, dt=0.2, tid=lambda i: "c"):
    """Walker heading north along x = 0; `car_xy(i)` is the car's world position at frame i."""
    rf, still, out = RangeFilter(), StillVehicles(), []
    for i, wy in enumerate(walker_y):
        cx, cy = car_xy(i)
        o = drone_car(cy - wy, cx, tid(i))
        out.append("vehicle" in drone_hazards([o], rf, i * dt, {"x": 0.0, "y": wy, "heading": 0.0}, still))
    return out


assert any(vehicle_frames([(6.0, 1.2)] * 10))                                    # no pose: flagged as before
assert not any(walk_past(lambda i: (1.2, 8.0), [0.0] * 15))                      # standing, car parked
assert not any(walk_past(lambda i: (1.2, 9.0), [0.25 * i for i in range(20)]))   # walking past it
assert not any(walk_past(lambda i: (1.2, 8.0), [0.0] * 15, tid=lambda i: f"t{i // 4}"))  # tracks break
moving = walk_past(lambda i: (1.2, 11.0 - 1.0 * i), [0.0] * 8)                   # oncoming at 5 m/s
assert not moving[0] and moving[3] and all(moving[3:])                           # counted after STILL_S
assert all(walk_past(lambda i: (0.5, 6.0), [0.0] * 5))                           # in the path: always

# gate: 3-frame onset, refresh keeps the same text, clearing sends nothing
g, cand = HazardGate(), {"crowd": {"bearing": 0.0, "distance": 6.0}}
assert g.step(cand, 0.0) == [] and g.step(cand, 0.1) == []
first = g.step(cand, 0.2)
assert len(first) == 1 and first[0]["kind"] == "crowd" and first[0]["text"] == "Crowd ahead, 6 meters."
assert g.step({"crowd": {"bearing": 20.0, "distance": 4.0}}, 0.3) == []   # within ttl/2
again = g.step({"crowd": {"bearing": 20.0, "distance": 4.0}}, 2.5)
assert again[0]["text"] == first[0]["text"] and again[0]["distance"] == 4.0
left = {"crowd": {"bearing": -60.0, "distance": 4.0}}      # new direction: spoken only after 3 frames
assert g.step(left, 2.51) == [] and g.step(left, 2.52) == []
turned = g.step(left, 2.53)
assert turned and turned[0]["text"] != first[0]["text"] and "left" not in first[0]["text"]
assert g.step({}, 2.6) == []
assert g.step(cand, 2.7) == []          # must hold 3 frames again

# pitch (+ = tilted down): the horizon moves up the image by fx*tan(pitch)
assert math.isclose(distance_m(480, 200, 1.2, pitch=0.0), 10.0)
assert math.isclose(distance_m(360, 200, 1.2, pitch=10), 1.2 / math.tan(math.radians(10)))  # optical centre
# a foot 1.5 m away from a 1.2 m chest camera tilted 20 deg down: 38.7 deg below horizontal
y = 360 + 1000 * math.tan(math.atan(1.2 / 1.5) - math.radians(20))
assert math.isclose(distance_m(y, 300, 1.2, pitch=20), 1.5)
assert distance_m(y, 300, 1.2) > 1.5            # ignoring the tilt reads farther (the danger)

# cut off at the bottom: the head top (1.7 m) gives the distance; size and bottom row only bound it
fx, d_true = 700, 1.5
y_top = 360 - fx * 0.5 / d_true                  # head 0.5 m above a 1.2 m camera, pitch 0
assert math.isclose(distance_m(720, 720 - y_top, 1.2, fx, 720, 1.7, 0.0, y_top=y_top), d_true)
assert distance_m(720, 720 - y_top, 1.2, fx, 720, 1.7) > 2.0     # bottom row alone says too far
cut = objects_from_boxes([(300, y_top, 660, 720, "person", 0.9, 3)], fx, 1.2, 960, 720)
assert cut[0]["cut_off"] and math.isclose(cut[0]["distance"], 1.5, abs_tol=0.01)

# live geometry (scripts/demo.sh live): iPhone 1280x720 -> 960x540, fx 700, 1.2 m up, 15 deg down.
# Closer than ~1.65 m the head is above the frame and the feet below it; no row can tell 1.2 m from 1.6 m,
# so the shoulders (box width) must. Before: 1.65 m whatever the distance, and "Stop" (1.6 m) never fired.
for d_true in (1.0, 1.2, 1.4):
    half = 700 * 0.53 / d_true / 2                 # a 0.53 m wide person (the bench's sprite)
    o = objects_from_boxes([(480 - half, 3, 480 + half, 530, "person", 0.9, 1)], 700, 1.2, 960, 540, 15)[0]
    assert o["distance"] < 1.6 and o["distance"] <= d_true, (d_true, o)
# head above the frame, box bottom 7 px or 20 px above the edge: the top row still bounds the distance.
# Before: 3.2 m on one frame and 1.3 m on the next for the same person (bench clip "approach").
far, near = (objects_from_boxes([(410, 1, 613, y2, "person", 0.9, 1)], 700, 1.2, 960, 540)[0]["distance"]
             for y2 in (520, 533))
assert far < 1.4 and math.isclose(far, near, abs_tol=0.05), (far, near)
# feet in frame, head in frame: unchanged (ground row)
assert math.isclose(objects_from_boxes([(450, 200, 510, 470, "person", 0.9, 1)], 700, 1.2, 960, 540)[0]["distance"],
                    1.2 * 700 / 200, abs_tol=0.01)

# live walk: median smoothing, closing speed, stop zone
from live_walk import center_obstacle, chest_step, is_stop, to_event_objects  # noqa: E402

rf = RangeFilter()
for i in range(10):
    d, v = rf.update("a", i * 0.1, 3.0 - 0.1 * i + (2.0 if i == 5 else 0.0))   # 1 m/s with one outlier
assert math.isclose(d, 2.2, abs_tol=0.05) and v is not None and 0.6 < v < 1.4
assert is_stop(0, 1.5, None) and not is_stop(0, 2.0, 0.2) and is_stop(0, 2.0, 2.5)
assert not is_stop(45, 1.2, None)               # 0.85 m to the side: not in the path

# what the earpiece does with it: a person stepping in 1.5 m ahead must go critical and say Stop
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "earpiece"))
from earpiece.config import SonarCfg  # noqa: E402
from earpiece.events import parse  # noqa: E402
from earpiece.sonar import Sonar  # noqa: E402

sonar, rf, said = Sonar(None, SonarCfg()), RangeFilter(), []
sonar.speak = lambda text, prio, **kw: said.append((text, prio))
for i, dist in enumerate([3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6, 1.5, 1.4]):
    objs = chest_step([{"id": "7", "kind": "person", "bearing": 3.0, "distance": dist, "confidence": 0.9,
                        "cut_off": True}], rf, i * 0.1)
    msg = parse({"type": "obstacles", "source": "standin-chest", "seq": i + 1, "objects": to_event_objects(objs)})
    sonar.update_frame(msg.source, msg.objects, msg.seq, now=i * 0.1)
    tr = sonar.tracks["standin-chest:7"]
    sonar._score(tr)
    sonar._maybe_speak(tr, i * 0.1)
    if objs[0]["distance"] > 2.0:
        assert not tr.critical, (dist, objs)
assert tr.critical and ("Stop. Person ahead.", "critical") in said

# depth fallback: flat floor is not an obstacle, a wall 1.2 m ahead is
import numpy as np  # noqa: E402
from live_walk import floor_depth  # noqa: E402

h, w, f = 252, 336, 245.0
floor = np.repeat(floor_depth(h, f, 1.2, 15)[:, None], w, axis=1)
floor[~np.isfinite(floor)] = 20.0
assert center_obstacle(floor, f, 1.2, 15) is None
wall = floor.copy()
yc = (np.arange(h) - h / 2) / f        # a vertical face 1.2 m ahead (z-depth along a camera tilted down 15 deg)
wall[:200, 100:240] = (1.2 / (math.cos(math.radians(15)) - yc * math.sin(math.radians(15))))[:200, None]
o = center_obstacle(wall, f, 1.2, 15)
assert o and math.isclose(o["distance"], 1.2) and abs(o["bearing"]) < 5

print("standin geometry ok")
