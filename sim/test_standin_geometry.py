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

print("standin geometry ok")
