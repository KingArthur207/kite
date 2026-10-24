"""Walking directions: a Google route turned into spoken turns and a beacon.

    compute_route()  Google Routes API v2 (the only directions API on our key) -> Route
    RouteFollower    positions in, cues out: what to say, when, and where the beacon points
    RouteGuide       the app's live route: routing off-thread, rerouting, cancelling

What the walker hears, for a turn 80 m away:
    "Head toward Lyon Street, slightly left."        at the start
    "In 30 meters, turn left onto Bay Street."        once, about 25 s before the turn
    "Now turn left onto Bay Street."                  once, about 6 m before it
and the beacon ticks from the corner the whole time, then swings round it.

Phone GPS wanders 5 to 10 m, so nothing here reacts to a single fix: progress along
the route is the median of recent fixes and only ever moves forward, every
announcement is said at most once, and "off the route" has to hold for seconds,
not samples. Bearings sent to the beacon are in the walker's frame, so they need
a heading; without one Kite sends no beacon rather than a wrong one.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Callable

from . import geo
from .events import Msg, clock_phrase, distance_phrase

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELD_MASK = ("routes.distanceMeters,routes.duration,routes.legs.steps.distanceMeters,"
              "routes.legs.steps.navigationInstruction,routes.legs.steps.startLocation,"
              "routes.legs.steps.endLocation,routes.legs.steps.polyline.encodedPolyline")
MAX_WORDS = 12
WALK_SPEED_MPS = 1.2

LatLng = tuple[float, float]


class RouteError(RuntimeError):
    pass


class NoKey(RouteError):
    """No GOOGLE_MAPS_API_KEY: retrying cannot help."""


# ====================================================================== route data
@dataclass
class Step:
    instruction: str                 # as Google wrote it; may hold several lines
    maneuver: str                    # DEPART, TURN_LEFT, NAME_CHANGE, ...
    distance_m: float
    start: LatLng
    end: LatLng
    points: list[LatLng] = field(default_factory=list)


@dataclass
class Route:
    steps: list[Step]
    distance_m: float
    duration_s: float | None
    destination: LatLng
    label: str | None = None

    def points(self) -> list[LatLng]:
        """The whole route as one polyline, without the duplicated joints between steps."""
        return self.polyline()[0]

    def polyline(self) -> tuple[list[LatLng], list[int]]:
        """(points, index of the point where each step starts). Step starts come from the
        joints, not from searching the line, so a route that passes a spot twice still
        puts each turn in the right place."""
        out: list[LatLng] = []
        starts: list[int] = []
        for s in self.steps:
            for j, p in enumerate(s.points or [s.start, s.end]):
                fresh = not out or geo.haversine_m(*out[-1], *p) > 0.05
                if j == 0:
                    starts.append(len(out) if fresh else len(out) - 1)
                if fresh:
                    out.append(p)
        if len(out) == 1:
            out.append(out[0])
        return out, starts


def decode_polyline(encoded: str) -> list[LatLng]:
    """Google's encoded polyline format (1e-5 degree precision) -> [(lat, lng)]."""
    pts: list[LatLng] = []
    i = lat = lng = 0
    while i < len(encoded):
        vals = []
        for _ in range(2):
            shift = res = 0
            while True:
                b = ord(encoded[i]) - 63
                i += 1
                res |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            vals.append(~(res >> 1) if res & 1 else res >> 1)
        lat += vals[0]
        lng += vals[1]
        pts.append((lat / 1e5, lng / 1e5))
    return pts


def _latlng(loc: dict | None) -> LatLng | None:
    ll = (loc or {}).get("latLng") or {}
    if ll.get("latitude") is None or ll.get("longitude") is None:
        return None
    return float(ll["latitude"]), float(ll["longitude"])


def parse_routes(data: dict, label: str | None = None, destination: LatLng | None = None) -> Route:
    """A computeRoutes response -> Route. Raises RouteError when there is no usable route."""
    routes = data.get("routes") or []
    if not routes:
        raise RouteError("no walking route found")
    r = routes[0]
    steps: list[Step] = []
    for leg in r.get("legs") or []:
        for s in leg.get("steps") or []:
            start, end = _latlng(s.get("startLocation")), _latlng(s.get("endLocation"))
            if start is None or end is None:
                continue
            nav = s.get("navigationInstruction") or {}
            pts = decode_polyline((s.get("polyline") or {}).get("encodedPolyline") or "")
            steps.append(Step(instruction=nav.get("instructions") or "", maneuver=nav.get("maneuver") or "",
                              distance_m=float(s.get("distanceMeters") or 0.0), start=start, end=end,
                              points=pts or [start, end]))
    if not steps:
        raise RouteError("route has no steps")
    dur = str(r.get("duration") or "").rstrip("s")
    return Route(steps, float(r.get("distanceMeters") or sum(s.distance_m for s in steps)),
                 float(dur) if dur.replace(".", "", 1).isdigit() else None,
                 destination or steps[-1].end, label)


def straight_route(origin: LatLng, destination: LatLng, label: str | None = None) -> Route:
    """A one-step "route" straight at the destination, for when no directions can be had."""
    d = geo.haversine_m(*origin, *destination)
    return Route([Step(f"Head toward {label or 'your destination'}", "DEPART", d, tuple(origin),
                       tuple(destination), [tuple(origin), tuple(destination)])],
                 d, None, tuple(destination), label)


def _waypoint(p: LatLng | str) -> dict:
    if isinstance(p, str):
        return {"address": p}
    return {"location": {"latLng": {"latitude": float(p[0]), "longitude": float(p[1])}}}


def fetch_routes(origin: LatLng, destination: LatLng | str, key: str, timeout: float = 6.0) -> dict:
    """The raw computeRoutes JSON. The key travels in a header only, so it never shows up
    in a URL, a log line or an exception message."""
    import httpx
    body = {"origin": _waypoint(origin), "destination": _waypoint(destination), "travelMode": "WALK",
            "languageCode": "en-US", "units": "METRIC"}
    r = httpx.post(ROUTES_URL, json=body, timeout=timeout,
                   headers={"X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELD_MASK})
    if r.status_code != 200:
        try:
            why = (r.json().get("error") or {}).get("status") or r.reason_phrase
        except (ValueError, AttributeError):
            why = r.reason_phrase
        raise RouteError(f"Routes API {r.status_code} {why}")
    return r.json()


def compute_route(origin: LatLng, destination: LatLng | str, key: str | None = None,
                  label: str | None = None, timeout: float = 6.0) -> Route:
    """Walking route from a lat/lng to a lat/lng or an address (Google Routes API v2)."""
    key = key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise NoKey("no GOOGLE_MAPS_API_KEY")
    data = fetch_routes(origin, destination, key, timeout)
    return parse_routes(data, label, None if isinstance(destination, str) else tuple(destination))


# ====================================================================== speakable text
_STREET = {"St": "Street", "Ave": "Avenue", "Blvd": "Boulevard", "Rd": "Road", "Dr": "Drive", "Ln": "Lane",
           "Pl": "Place", "Ct": "Court", "Pkwy": "Parkway", "Hwy": "Highway", "Sq": "Square", "Ter": "Terrace",
           "Cir": "Circle", "Expy": "Expressway", "Fwy": "Freeway", "Aly": "Alley"}
# Only after a name and before the end, punctuation or a lowercase word: "Lyon St toward"
# is a street, "onto St Francis" is a saint.
_STREET_RE = re.compile(r"(?<=[A-Za-z0-9] )(" + "|".join(_STREET) + r")\.?(?=$|[,;:)]|\s+[a-z0-9])")
_DIR_RE = re.compile(r"\b(onto|on|toward|at|via|to) ([NSEW])\.? (?=[A-Z0-9])")
_DIRS = {"N": "North", "S": "South", "E": "East", "W": "West"}
_COMPASS_RE = re.compile(r"^Head (?:north|south|east|west)(?:east|west)?\b ?", re.I)
_TURNS = ("TURN_", "UTURN", "FORK", "ROUNDABOUT", "RAMP", "MERGE")
_TRAILING = {"onto", "on", "toward", "to", "and", "the", "at", "via", "then", "of", "a"}


def speakable(line: str) -> str:
    """One Google instruction line -> how a person would say it: abbreviations spelled out
    (TTS reads "St" as "saint" or "S T"), slashes as "and", no stray spaces."""
    s = " ".join(line.replace("/", " and ").split())
    s = _STREET_RE.sub(lambda m: _STREET[m.group(1)], s)
    s = _DIR_RE.sub(lambda m: f"{m.group(1)} {_DIRS[m.group(2)]} ", s)
    s = re.sub(r"^(?:Turn )?[Ss]light (left|right)\b", r"Bear \1", s)       # the O&M phrase
    return re.sub(r"\bDestination will be on the (left|right)\b", r"Destination on your \1", s)


def sentences(instruction: str) -> list[str]:
    """"Continue onto Lyon St\\nTake the stairs" -> ["Continue onto Lyon Street", "Take the stairs"]."""
    return [speakable(x).rstrip(". ") for x in re.split(r"[\n\r]+", instruction or "") if x.strip()]


def fit(text: str, max_words: int = MAX_WORDS) -> str:
    """Keep a cue short enough to hear while walking: drop "toward ..." tails first, then
    trim words, never leaving a sentence hanging on "onto" or "and"."""
    text = " ".join(text.split())
    if len(text.split()) <= max_words:
        return text
    head, sep, _ = text.partition(" toward ")
    if sep and len(head.split()) <= max_words:
        return head.rstrip(",") + "."
    words = text.split()[:max_words]
    while words and words[-1].lower().strip(",.") in _TRAILING:
        words.pop()
    return " ".join(words).rstrip(",.") + "."


def _notes(lines: list[str]) -> list[str]:
    """Extra lines worth saying at the maneuver ("Take the stairs"). Not the destination
    side (that goes with the arrival heads-up) and not "Pass by <shop> (in 550m)", which
    is about somewhere far down the street."""
    return [n for n in lines[1:] if not n.startswith(("Destination", "Pass by"))]


def _lc(s: str) -> str:
    return s[:1].lower() + s[1:] if s[:2] != s[:2].upper() else s


def _sentence(*parts: str) -> str:
    return " ".join(p.rstrip(". ") + "." for p in parts if p)


# ====================================================================== following
@dataclass
class _Maneuver:
    index: int                       # step index (the maneuver happens at that step's start)
    s: float                         # meters along the route
    primary: str
    notes: list[str]
    turn: bool
    pre_said: bool = False
    now_said: bool = False


class RouteFollower:
    """Turns a stream of positions into spoken guidance and a beacon. Pure logic: all
    output goes through the callbacks, so it can be driven by a test, the sim, or GPS."""

    PRE_M = 30.0            # heads-up distance (~25 s of walking)
    SHORT_M = 12.0          # closer than this when it becomes next: skip the heads-up
    NOW_M = 6.0             # "now" distance (~5 s): time to slow and find the corner
    LATE_M = 10.0           # passed a turn without "now" (a GPS jump): still say it within this
    LOOKAHEAD_M = 15.0      # beacon aims this far ahead along the route...
    MIN_AIM_M = 8.0         # ...but at the corner itself until this close, so it never cuts corners
    OFF_ROUTE_M = 25.0
    BACK_ON_M = 15.0        # hysteresis: back within this resets the off-route clock
    OFF_ROUTE_S = 8.0
    ARRIVE_M = 8.0
    SNAP_BACK_M = 25.0      # how far behind progress a fix may still match the route
    MAX_SPEED_MPS = 3.0     # progress never runs ahead faster than this (plus STEP_SLACK_M a fix)
    STEP_SLACK_M = 2.0
    MAX_ACCURACY_M = 40.0   # fixes worse than this are ignored
    BEACON_MIN_S = 0.2      # sim positions can come at 30 Hz; the beacon needs ~5 Hz
    START_TTL_S = 30.0      # cues at the start wait for the walker, who is waiting for them

    def __init__(self, route: Route,
                 on_guidance: Callable[[str, str, float], None],
                 on_beacon: Callable[[float, float, str | None], None],
                 on_reroute: Callable[[], None] | None = None,
                 on_arrived: Callable[[str | None], None] | None = None,
                 label: str | None = None, key_prefix: str = "route"):
        self.route = route
        self.label = label or route.label
        self.on_guidance, self.on_beacon = on_guidance, on_beacon
        self.on_reroute, self.on_arrived = on_reroute, on_arrived
        self.prefix = key_prefix
        pts, starts = route.polyline()
        self.frame = geo.LocalFrame(*pts[0])
        self._xy = [self.frame.to_xy(*p) for p in pts]
        self._cum = [0.0]
        for a, b in zip(self._xy, self._xy[1:]):
            self._cum.append(self._cum[-1] + math.dist(a, b))
        self.length = self._cum[-1]
        self.dest_xy = self.frame.to_xy(*route.destination)
        # Steps without an instruction (Google emits them for some crossings) are walked, not announced.
        self.maneuvers = [self._maneuver(i, self._cum[starts[i]]) for i in range(1, len(route.steps))
                          if sentences(route.steps[i].instruction)]
        last = sentences(route.steps[-1].instruction)
        self._dest_side = next((m.group(1) for n in last
                                for m in [re.search(r"Destination on your (left|right)", n)] if m), None)
        self.progress = 0.0
        self.started = self.done = self.rerouting = False
        self.off_route = False
        self._joined = False            # been near the route at least once
        self._arrive_pre_said = False
        self._s_hist: deque[float] = deque(maxlen=3)
        self._xy_hist: deque[tuple[float, float]] = deque(maxlen=3)
        self._track: deque[tuple[float, float, float]] = deque(maxlen=64)
        self._off_since: float | None = None
        self._last_t: float | None = None
        self._last_beacon = -1e9
        self.heading: float | None = None
        self._track_h: float | None = None
        self._lock = threading.Lock()

    def _maneuver(self, i: int, s: float) -> _Maneuver:
        st = self.route.steps[i]
        lines = sentences(st.instruction)
        return _Maneuver(i, s, lines[0], _notes(lines), st.maneuver.startswith(_TURNS))

    # ------------------------------------------------------------ geometry
    def _project(self, p: tuple[float, float], lo: float, hi: float) -> tuple[float, float]:
        """Closest point of the route to p, among route points between lo and hi meters
        along it. Returns (distance from the route, meters along it)."""
        best = (math.inf, lo)
        for j in range(len(self._xy) - 1):
            s0, s1 = self._cum[j], self._cum[j + 1]
            seg = s1 - s0
            if s1 < lo or s0 > hi or seg <= 0:
                continue
            (ax, ay), (bx, by) = self._xy[j], self._xy[j + 1]
            t = ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / (seg * seg)
            t = min(max(t, (lo - s0) / seg, 0.0), (hi - s0) / seg, 1.0)
            d = math.hypot(p[0] - (ax + t * (bx - ax)), p[1] - (ay + t * (by - ay)))
            if d < best[0]:
                best = (d, s0 + t * seg)
        return best

    def point_at(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), self.length)
        for j in range(len(self._xy) - 1):
            if self._cum[j + 1] >= s:
                seg = self._cum[j + 1] - self._cum[j]
                t = 0.0 if seg <= 0 else (s - self._cum[j]) / seg
                (ax, ay), (bx, by) = self._xy[j], self._xy[j + 1]
                return ax + t * (bx - ax), ay + t * (by - ay)
        return self._xy[-1]

    def next_maneuver(self) -> _Maneuver | None:
        return next((m for m in self.maneuvers if m.s > self.progress), None)

    # ------------------------------------------------------------ the update
    def update(self, lat: float, lng: float, heading: float | None = None, now: float | None = None,
               accuracy: float | None = None) -> None:
        """Feed one position fix. Thread-safe; callbacks run after the lock is released."""
        out: list[tuple[Callable, tuple]] = []
        with self._lock:
            self._update(lat, lng, heading, time.monotonic() if now is None else now, accuracy, out)
        for fn, args in out:
            if fn is not None:
                fn(*args)

    def _update(self, lat, lng, heading, now, accuracy, out) -> None:
        if self.done or self.rerouting:
            return
        if accuracy is not None and accuracy > self.MAX_ACCURACY_M:
            return
        xy = self.frame.to_xy(lat, lng)
        self._track.append((now, lat, lng))
        self._xy_hist.append(xy)
        self.heading = heading if heading is not None else self._track_heading()
        dt = 1.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        hi = self.progress + self.SNAP_BACK_M + self.MAX_SPEED_MPS * dt
        cross, s = self._project(xy, self.progress - self.SNAP_BACK_M, hi if self.started else self.length)

        if cross > self.OFF_ROUTE_M:
            # Only once the walker has been on the route: if Google starts the route at the
            # nearest footpath 40 m away, rerouting would just return the same route forever.
            if self._joined and self._off_since is None:
                self._off_since = now
            if self._off_since is not None and now - self._off_since >= self.OFF_ROUTE_S:
                self.rerouting = self.off_route = True
                out.append((self.on_guidance, ("You're off the route. Finding a new one.",
                                               f"{self.prefix}:off-route", 10.0)))
                out.append((self.on_reroute, ()))
                return
        else:
            self._joined = True
            if cross <= self.BACK_ON_M:
                self._off_since = None
            self._s_hist.append(s)
            # Median of the last few fixes, so one wild fix can't move us; only ever forward, so
            # jitter can't repeat or rewind a turn; and no faster than a person moves, because
            # near a corner a GPS bias toward the cross street snaps the fix 10+ m round the
            # corner, which would leap straight over the "now" window.
            target = median(self._s_hist)
            if self.started:
                target = min(target, self.progress + self.MAX_SPEED_MPS * dt + self.STEP_SLACK_M)
            self.progress = max(self.progress, target)
        self.off_route = self._off_since is not None

        here = tuple(sum(c) / len(self._xy_hist) for c in zip(*self._xy_hist))
        aim = self._aim()
        rel = None
        if self.heading is not None:
            ax, ay = self.point_at(aim)
            compass = self.frame.heading_of(ax - here[0], ay - here[1])
            rel = geo.relative_bearing(compass, self.heading)

        if not self.started:
            self.started = True
            out.append((self.on_guidance, (self._start_text(rel), f"{self.prefix}:start",
                                           self.START_TTL_S)))

        if math.dist(here, self.dest_xy) <= self.ARRIVE_M or (
                self.length - self.progress <= self.ARRIVE_M and cross <= self.BACK_ON_M):
            self.done = True
            out.append((self.on_arrived, (self.label,)))
            return

        self._announce(out)

        if rel is not None and now - self._last_beacon >= self.BEACON_MIN_S:
            self._last_beacon = now
            ax, ay = self.point_at(aim)
            dist = math.hypot(ax - here[0], ay - here[1]) + (self.length - aim)
            out.append((self.on_beacon, (round(rel, 1), round(dist, 1), self.label)))

    def _track_heading(self) -> float | None:
        """Walking direction from the fixes, smoothed: a heading from GPS displacement swings
        with every wobble of the fix, and the beacon would swing with it."""
        h = geo.heading_from_track(self._track)
        if h is None or self._track_h is None:
            self._track_h = h
        else:
            self._track_h = (self._track_h + 0.5 * geo.relative_bearing(h, self._track_h)) % 360.0
        return self._track_h

    def _aim(self) -> float:
        nxt = self.next_maneuver()
        corner = nxt.s if nxt else self.length
        if corner - self.progress >= self.MIN_AIM_M:
            return min(self.progress + self.LOOKAHEAD_M, corner)
        return min(self.progress + self.LOOKAHEAD_M, self.length)

    def _announce(self, out: list) -> None:
        for m in self.maneuvers:
            if m.s > self.progress or m.now_said:
                continue
            m.now_said = True                     # passed it without a "now" (GPS jump)
            if self.progress - m.s <= self.LATE_M:
                out.append(self._now(m))
        m = self.next_maneuver()
        if m is not None:
            d = m.s - self.progress
            if (not m.pre_said and not m.now_said and (m.turn or m.notes)
                    and self.SHORT_M < d <= self.PRE_M):
                m.pre_said = True
                text = fit(_sentence(f"In {distance_phrase(d)}, {_lc(m.primary)}"))
                out.append((self.on_guidance, (text, f"{self.prefix}:pre:{m.index}", self._pre_ttl(d))))
            if d <= self.NOW_M and not m.now_said:
                m.now_said = True
                out.append(self._now(m))
            return
        d = self.length - self.progress
        if not self._arrive_pre_said and self.SHORT_M < d <= self.PRE_M:
            self._arrive_pre_said = True
            who = self.label if self.label and len(self.label.split()) <= 5 else "Your destination"
            side = f", on your {self._dest_side}" if self._dest_side else ""
            out.append((self.on_guidance, (fit(f"{who} is {distance_phrase(d)} ahead{side}."),
                                           f"{self.prefix}:pre:arrive", self._pre_ttl(d))))

    def _pre_ttl(self, d: float) -> float:
        # a heads-up is only useful while there is still time to act on it
        return round(min(10.0, max(3.0, 0.5 * (d - self.NOW_M) / WALK_SPEED_MPS)), 1)

    def _now(self, m: _Maneuver) -> tuple[Callable, tuple]:
        lead = f"Now {_lc(m.primary)}" if m.turn else m.primary
        text = _sentence(lead, *m.notes)
        if len(text.split()) > MAX_WORDS:
            text = fit(_sentence(lead))
        # A maneuver due right at the start (Google often begins with a 2 m step) queues behind
        # the start instruction and any memory heads-up; the walker hasn't moved yet, so it can
        # wait as long as they do. Later, a "now" more than 5 s late is worse than none.
        ttl = self.START_TTL_S if len(self._track) <= 1 else 5.0
        return self.on_guidance, (text, f"{self.prefix}:now:{m.index}", ttl)

    def _start_text(self, rel: float | None) -> str:
        lines = sentences(self.route.steps[0].instruction) or ["Follow the route"]
        first, notes = lines[0], _notes(lines)
        if rel is not None and _COMPASS_RE.match(first):
            # "northwest" means nothing without sight of the street; say it in the walker's frame
            where = {"ahead": "straight ahead", "on your left": "to your left",
                     "on your right": "to your right"}.get(clock_phrase(rel), clock_phrase(rel))
            rest = _COMPASS_RE.sub("", first)
            first = f"Head {rest}, {where}" if rest else f"Head {where}"
        nxt = self.maneuvers[0].s if self.maneuvers else self.length
        if nxt - self.progress >= 40 and " toward " not in first:
            first += f" for {distance_phrase(nxt - self.progress)}"
        text = _sentence(first, *notes)
        return text if len(text.split()) <= MAX_WORDS else fit(_sentence(first))

    def summary(self) -> dict:
        """What the voice brain and judgment layer need to know about the route right now."""
        m = self.next_maneuver()
        return {"destination": self.label, "remaining_m": round(max(0.0, self.length - self.progress)),
                "next": m.primary if m else "arrive",
                "next_in_m": round((m.s if m else self.length) - self.progress),
                "off_route": self.off_route, "rerouting": self.rerouting, "arrived": self.done}


# ====================================================================== the app's live route
class RouteGuide:
    """Owns the live route for the earpiece. Routing runs on a background thread so the
    network never blocks event handling; every cue goes out through `emit` as an ordinary
    guidance / beacon message, so route cues are spoken, mirrored and logged exactly like
    ones from any other producer."""

    RETRY_S = 15.0
    MAX_ATTEMPTS = 3

    def __init__(self, emit: Callable[[Msg], None], event: Callable[[str, dict], None] | None = None,
                 log: Callable[[str], None] = print, key: str | None = None):
        self.emit, self.event, self.log = emit, event or (lambda k, d: None), log
        self.key = key
        self.follower: RouteFollower | None = None
        self.destination: LatLng | None = None
        self.label: str | None = None
        self.last: dict | None = None
        self._gen = 0
        self._pending = False
        self._busy = False
        self._attempts = 0
        self._retry_at = 0.0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.destination is not None

    def start(self, destination: LatLng, label: str | None = None) -> None:
        """Route to destination from the latest position (or the first one to arrive)."""
        with self._lock:
            self._gen += 1
            self.destination, self.label = (float(destination[0]), float(destination[1])), label
            self.follower = None
            self._attempts, self._retry_at, self._pending = 0, 0.0, True
        self._maybe_route(time.monotonic())

    def cancel(self) -> None:
        with self._lock:
            was = self.destination is not None
            self._gen += 1
            self.follower, self.destination, self._pending = None, None, False
        if was:
            self.event("route", {"state": "cancelled"})

    def position(self, pos: dict, now: float | None = None) -> None:
        """Every `position` event, route or not: the next route starts from the latest fix."""
        if pos.get("lat") is None or pos.get("lng") is None:
            return
        now = time.monotonic() if now is None else now
        self.last = pos
        f = self.follower
        if f is not None:
            f.update(pos["lat"], pos["lng"], pos.get("heading"), now, pos.get("accuracy"))
        self._maybe_route(now)

    def summary(self) -> dict | None:
        f = self.follower
        if f is not None:
            return f.summary()
        return {"destination": self.label, "state": "routing"} if self.destination else None

    # ------------------------------------------------------------ internals
    def _maybe_route(self, now: float) -> None:
        with self._lock:
            if not self._pending or self._busy or self.last is None or now < self._retry_at:
                return
            self._pending, self._busy = False, True
            self._attempts += 1
            job = (self._gen, (self.last["lat"], self.last["lng"]), self.destination, self.label,
                   self._attempts)
        threading.Thread(target=self._route, args=job, name="route", daemon=True).start()

    def _route(self, gen: int, origin: LatLng, dest: LatLng, label: str | None, attempt: int) -> None:
        try:
            r = compute_route(origin, dest, key=self.key, label=label)
        except Exception as e:
            self.log(f"[route] no route ({type(e).__name__}: {e})")
            with self._lock:
                self._busy = False
                if gen != self._gen:
                    return
                if attempt < self.MAX_ATTEMPTS and not isinstance(e, NoKey):
                    self._pending, self._retry_at = True, time.monotonic() + self.RETRY_S
            self.event("route", {"state": "failed", "error": type(e).__name__})
            if attempt == 1:
                # Soundscape's answer to "no directions": a beacon straight at the place, so the
                # walker still knows which way it is. Replaced if a retry gets a real route.
                self._say(fit(f"I couldn't get walking directions. The beacon points straight at "
                              f"{label or 'your destination'}."), gen)
                self._follow(straight_route(origin, dest, label), gen, label, fallback=True)
            return
        with self._lock:
            self._busy = False
        self._follow(r, gen, label)

    def _follow(self, r: Route, gen: int, label: str | None, fallback: bool = False) -> None:
        with self._lock:
            if gen != self._gen:
                return                            # cancelled or replaced while we were routing
            f = self.follower = RouteFollower(
                r, on_guidance=lambda t, k, ttl: self._guidance(t, k, ttl, gen),
                on_beacon=lambda b, d, lab: self._beacon(b, d, lab, gen),
                on_reroute=lambda: self._reroute(gen), on_arrived=lambda lab: self._arrived(lab, gen),
                label=label, key_prefix=f"route{gen}")
            if fallback:
                f.OFF_ROUTE_M = math.inf          # streets never follow a straight line
                f.started = True                  # the apology above was the start
        pts = r.points()
        step = max(1, len(pts) // 200)
        self.event("route", {"state": "fallback" if fallback else "started", "label": label,
                             "distance_m": round(r.distance_m), "duration_s": r.duration_s,
                             "points": [[round(a, 6), round(b, 6)] for a, b in pts[::step] + pts[-1:]]})
        self.log(f"[route] {'straight line' if fallback else 'walking route'}: {round(r.distance_m)} m, "
                 f"{len(r.steps)} steps to {label or r.destination}")
        last = self.last
        if last is not None:
            f.update(last["lat"], last["lng"], last.get("heading"), None, last.get("accuracy"))

    def _live(self, gen: int) -> bool:
        return gen == self._gen

    def _guidance(self, text: str, key: str, ttl: float, gen: int) -> None:
        if self._live(gen):
            self.emit(Msg("guidance", "route", text=text, key=key, ttl=ttl, priority="nav"))

    def _say(self, text: str, gen: int) -> None:
        if self._live(gen):
            self.emit(Msg("say", "route", text=text, priority="answer", ttl=15.0))

    def _beacon(self, bearing: float, distance: float, label: str | None, gen: int) -> None:
        if self._live(gen):
            self.emit(Msg("beacon", "route", bearing=bearing, distance=distance, label=label))

    def _reroute(self, gen: int) -> None:
        with self._lock:
            if gen != self._gen:
                return
            self._gen += 1                        # the old follower's cues stop here
            self.follower = None
            self._attempts, self._retry_at, self._pending = 0, 0.0, True
        self.event("route", {"state": "rerouting"})
        self._maybe_route(time.monotonic())

    def _arrived(self, label: str | None, gen: int) -> None:
        if not self._live(gen):
            return
        # the beacon goes first: it has its own arrival chime and must not say it twice
        self.emit(Msg("beacon", "route", active=False))
        self.emit(Msg("guidance", "route", text=f"You have arrived{' at ' + label if label else ''}.",
                      key="arrived", ttl=10.0, priority="nav"))
        with self._lock:
            if gen == self._gen:
                self.destination, self.follower = None, None
        self.event("route", {"state": "arrived", "label": label})
