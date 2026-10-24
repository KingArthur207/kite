"""Earth geometry for walking: distances, bearings, and one shared local map frame.

Two worlds have to agree on where things are: the real street (GPS latitude and
longitude) and the sim (the three.js world, in meters). `LocalFrame` is the one
mapping between them: a flat east/north plane tangent to the earth at an anchor
point (WGS84 ENU), optionally rotated so the sim's axes need not face true north.
Over a few city blocks the flat-plane error is millimetres, far below GPS noise.

Anchoring on landmarks (the same two or three used to line up the splat):

    marks = [((37.80131, -122.44772), (0.0, 0.0)),      # (lat, lng) -> sim (x, y)
             ((37.80214, -122.44801), (-8.1, 93.6))]
    frame = LocalFrame.from_landmarks(marks)            # solves anchor + rotation
    frame.fit_rms_m                                      # how well the marks agree
    frame.to_xy(lat, lng)  -> (x, y) in sim meters
    frame.to_latlng(x, y)  -> (lat, lng)

Sim coordinates are a right-handed ground plane: +x to the right of +y when seen
from above. A three.js world with y up and -z "forward" maps as (x, y) = (x, -z).
With rotation_deg = 0, +x is east and +y is north; otherwise rotation_deg is the
compass bearing the sim's +y axis points to. Headings everywhere in Kite are
compass degrees (0 north, 90 east); `frame.heading_of(dx, dy)` turns a sim
direction into one, so the sim can send real-world `position.heading`.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

EARTH_R_M = 6371008.8                  # mean radius, for great-circle distances
_A = 6378137.0                         # WGS84 semi-major axis
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)

LatLng = tuple[float, float]


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in meters."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Initial compass bearing from point 1 to point 2, in [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lng2 - lng1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def relative_bearing(target_compass: float, heading: float) -> float:
    """Compass bearing of a target -> bearing in the walker's frame (0 ahead, +90 right)."""
    return (target_compass - heading + 180.0) % 360.0 - 180.0


def _ecef(lat: float, lng: float, h: float = 0.0) -> tuple[float, float, float]:
    p, l = math.radians(lat), math.radians(lng)
    n = _A / math.sqrt(1 - _E2 * math.sin(p) ** 2)
    return ((n + h) * math.cos(p) * math.cos(l), (n + h) * math.cos(p) * math.sin(l),
            (n * (1 - _E2) + h) * math.sin(p))


def _geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    lng = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - _E2))
    h = 0.0
    for _ in range(6):                  # converges to well under a millimetre
        n = _A / math.sqrt(1 - _E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - _E2 * n / (n + h)))
    return math.degrees(lat), math.degrees(lng), h


class LocalFrame:
    """Flat local map (meters) tangent to the earth at an anchor; see the module docstring."""

    def __init__(self, anchor_lat: float, anchor_lng: float, rotation_deg: float = 0.0):
        self.anchor_lat, self.anchor_lng = float(anchor_lat), float(anchor_lng)
        self.rotation_deg = float(rotation_deg)
        self.fit_rms_m: float | None = None
        p, l = math.radians(self.anchor_lat), math.radians(self.anchor_lng)
        self._sp, self._cp, self._sl, self._cl = math.sin(p), math.cos(p), math.sin(l), math.cos(l)
        r = math.radians(self.rotation_deg)
        self._sr, self._cr = math.sin(r), math.cos(r)
        self._o = _ecef(self.anchor_lat, self.anchor_lng)

    def __repr__(self) -> str:
        return f"LocalFrame({self.anchor_lat:.7f}, {self.anchor_lng:.7f}, rotation_deg={self.rotation_deg:.2f})"

    def _enu(self, lat: float, lng: float) -> tuple[float, float, float]:
        x, y, z = _ecef(lat, lng)
        dx, dy, dz = x - self._o[0], y - self._o[1], z - self._o[2]
        e = -self._sl * dx + self._cl * dy
        n = -self._sp * self._cl * dx - self._sp * self._sl * dy + self._cp * dz
        u = self._cp * self._cl * dx + self._cp * self._sl * dy + self._sp * dz
        return e, n, u

    def to_xy(self, lat: float, lng: float) -> tuple[float, float]:
        e, n, _ = self._enu(lat, lng)
        return e * self._cr - n * self._sr, e * self._sr + n * self._cr

    def to_latlng(self, x: float, y: float) -> LatLng:
        e, n = x * self._cr + y * self._sr, -x * self._sr + y * self._cr
        # The plane sits above the curved ground away from the anchor; slide the point
        # down the anchor's vertical until it is back on the ellipsoid, so
        # to_latlng(to_xy(p)) == p exactly rather than to a few centimetres.
        u = 0.0
        for _ in range(3):
            dx = -self._sl * e - self._sp * self._cl * n + self._cp * self._cl * u
            dy = self._cl * e - self._sp * self._sl * n + self._cp * self._sl * u
            dz = self._cp * n + self._sp * u
            lat, lng, h = _geodetic(self._o[0] + dx, self._o[1] + dy, self._o[2] + dz)
            u -= h
        return lat, lng

    def heading_of(self, dx: float, dy: float) -> float:
        """A direction in this frame's (x, y) -> compass heading, for `position.heading`."""
        return (math.degrees(math.atan2(dx, dy)) + self.rotation_deg) % 360.0

    def direction_of(self, heading: float) -> tuple[float, float]:
        """Compass heading -> unit direction in this frame's (x, y)."""
        a = math.radians(heading - self.rotation_deg)
        return math.sin(a), math.cos(a)

    @classmethod
    def from_landmarks(cls, marks: Sequence[tuple[LatLng, tuple[float, float]]],
                       rotation_deg: float = 0.0) -> "LocalFrame":
        """Fit the anchor (the lat/lng of sim (0, 0)) and rotation from landmarks seen in both
        worlds. Two or more marks solve the rotation (least squares, no scaling: both worlds
        are in meters); a single mark needs `rotation_deg`. `fit_rms_m` reports how far the
        marks disagree after the fit, which is the first thing to check if cues feel off."""
        if not marks:
            raise ValueError("need at least one landmark")
        probe = cls(*marks[0][0])
        enu = [probe.to_xy(*ll) for ll, _ in marks]
        sim = [(float(x), float(y)) for _, (x, y) in marks]
        k = len(marks)
        ce = (sum(p[0] for p in enu) / k, sum(p[1] for p in enu) / k)
        cs = (sum(p[0] for p in sim) / k, sum(p[1] for p in sim) / k)
        if k >= 2:
            num = den = 0.0
            for (e, n), (x, y) in zip(enu, sim):
                e, n, x, y = e - ce[0], n - ce[1], x - cs[0], y - cs[1]
                num += e * y - n * x
                den += e * x + n * y
            rotation_deg = math.degrees(math.atan2(num, den))
        r = math.radians(rotation_deg)
        c, s = math.cos(r), math.sin(r)
        # sim = R(r) * enu + t, so sim (0, 0) sits at enu = centroid_enu - R^T * centroid_sim
        e0 = ce[0] - (c * cs[0] + s * cs[1])
        n0 = ce[1] - (-s * cs[0] + c * cs[1])
        frame = cls(*probe.to_latlng(e0, n0), rotation_deg=rotation_deg)
        frame.fit_rms_m = math.sqrt(sum((fx - x) ** 2 + (fy - y) ** 2 for (ll, (x, y)) in marks
                                        for fx, fy in [frame.to_xy(*ll)]) / k)
        return frame


def heading_from_track(points: Iterable[tuple[float, float, float]], window_s: float = 1.0,
                       min_move_m: float = 3.0, max_window_s: float = 8.0) -> float | None:
    """Walking direction (compass degrees) from recent (t, lat, lng) fixes, newest last.

    Uses the last ~`window_s` of movement, but only trusts a displacement of at least
    `min_move_m`: phone GPS wanders several meters while standing still, and a heading
    computed from that wander would swing the beacon around at random. So it reaches
    further back (up to `max_window_s`) for slow walkers, and returns None when the
    walker has not really moved; callers then send no beacon rather than a wrong one."""
    pts = list(points)
    if len(pts) < 2:
        return None
    t1, lat1, lng1 = pts[-1]
    for t0, lat0, lng0 in reversed(pts[:-1]):
        age = t1 - t0
        if age > max_window_s:
            break
        if age >= window_s and haversine_m(lat0, lng0, lat1, lng1) >= min_move_m:
            return bearing_deg(lat0, lng0, lat1, lng1)
    return None
