"""Is it directly in front of me? The walking corridor, projected from the camera.

The walker's line is the camera's heading on the ground (a chest clip points where the torso points). The
corridor is HALF_M = 0.5 m either side of it: an adult's shoulders are about 0.40 to 0.46 m across
(biacromial breadth), arms swinging add a few centimetres each side, and gait sways the trunk about +-2 to
3 cm, so 1.0 m is the width a body sweeps with a little to spare. It matches the 0.5 m walkable corridor of
published travel aids (the earpiece's sonar uses 0.55 m), and it is the width the user asked for.

An object is in the corridor when its footprint [left edge, right edge] at its estimated range overlaps
[-HALF_M, +HALF_M]. A SOFT_M margin beyond that counts as "edge" (reported, never above P2). The corridor widens
by WIDEN_PER_MPS for every m/s of walking speed above 1 m/s (less time to correct course).

Without a metric range (no cue fired, only an angle): the corridor is an angular cone of +-atan(HALF_M / 5 m),
about 5.7 deg, i.e. what +-0.5 m subtends at the P2/P3 boundary. Closer than 5 m that cone is too narrow (it
misses a person 0.4 m off the line at 2 m, who subtends 11 deg), farther it is too wide; this is why every
object gets a range first and the cone is only a fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import lateral_at
from .intrinsics import CameraProfile

HALF_M = 0.5
SOFT_M = 0.2
WIDEN_PER_MPS = 0.15
CONE_RANGE_M = 5.0


@dataclass
class CorridorHit:
    inside: bool            # overlaps the corridor proper
    edge: bool              # only overlaps the soft margin
    lateral: float          # centre offset (m), + right
    left: float
    right: float
    bearing: float          # deg, + right, of the footprint point nearest the walker's line


def half_width(v_ego: float = 1.2) -> float:
    return HALF_M + WIDEN_PER_MPS * max(0.0, v_ego - 1.0)


def test(prof: CameraProfile, x1: float, x2: float, v_foot: float, forward: float, pitch: float,
         v_ego: float = 1.2, half: float | None = None) -> CorridorHit:
    """Footprint of a box (undistorted columns x1..x2, foot row v_foot) at `forward` metres."""
    hw = half_width(v_ego) if half is None else half
    left = lateral_at(prof, x1, v_foot, forward, pitch)
    right = lateral_at(prof, x2, v_foot, forward, pitch)
    centre = lateral_at(prof, (x1 + x2) / 2, v_foot, forward, pitch)
    near = 0.0 if left <= 0 <= right else (left if left > 0 else right)
    inside = right >= -hw and left <= hw
    edge = not inside and right >= -(hw + SOFT_M) and left <= hw + SOFT_M
    return CorridorHit(inside, edge, centre, left, right, math.degrees(math.atan2(near, max(forward, 0.1))))


def test_angular(prof: CameraProfile, x1: float, x2: float, v_ego: float = 1.2) -> CorridorHit:
    """No range: the box's angular extent against the +-atan(half / 5 m) cone."""
    hw = half_width(v_ego)
    cone = math.atan(hw / CONE_RANGE_M)
    a1 = math.atan((x1 - prof.cx) / prof.fx)
    a2 = math.atan((x2 - prof.cx) / prof.fx)
    inside = a2 >= -cone and a1 <= cone
    soft = math.atan((hw + SOFT_M) / CONE_RANGE_M)
    edge = not inside and a2 >= -soft and a1 <= soft
    near = 0.0 if a1 <= 0 <= a2 else (a1 if a1 > 0 else a2)
    lat = CONE_RANGE_M * math.tan((a1 + a2) / 2)
    return CorridorHit(inside, edge, lat, CONE_RANGE_M * math.tan(a1), CONE_RANGE_M * math.tan(a2),
                       math.degrees(near))
