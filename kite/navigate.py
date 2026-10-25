"""'Take me to X': clarify the destination, then a walking route from the phone's GPS to it (Google Maps).

    python -m kite.navigate "take me to the nearest pharmacy"     # from the phone's GPS (python -m phone.latest)
    python -m kite.navigate "Doe Library" --from 37.8667,-122.2577
    python -m kite.navigate "Starbucks" --pick 2                  # answer "which one?" with the 2nd choice

    from kite.navigate import plan, choose
    p = plan("take me to Starbucks")          # p.status: "route" | "clarify" | "error"; p.say is what to speak
    if p.status == "clarify": p = choose(p, 1)                    # the walker picked the first one
    p.points                                  # [(lat, lng), ...] the path, for the map
    p.steps                                   # ["Head north on Bowditch St", ...]

"Nearest X" goes to the closest match; a named place with several matches asks which one. Standard library only
(it runs in the drone app's environment); the Google Maps key comes from GOOGLE_MAPS_API_KEY or the repo's .env.
Same Places and Routes requests as earpiece/earpiece/destinations.py and route.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
ROUTES_FIELDS = ("routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline,"
                 "routes.legs.steps.distanceMeters,routes.legs.steps.navigationInstruction")
MAX_CHOICES = 3
SEARCH_RESULTS = 8                 # ask for more, keep the closest MAX_CHOICES
NEARBY_RADIUS_M = 1500.0
NAMED_RADIUS_M = 5000.0

_ASK = re.compile(r"^(?:please\s+)?(?:take|bring|walk|guide|get)\s+me\s+(?:to|towards?)\s+|^(?:go|navigate|head)\s+"
                  r"(?:to|towards?)\s+|^(?:i want to go to|how do i get to|directions to)\s+", re.I)
_NEARBY = re.compile(r"^(?:the |a |an |any )?(?:nearest|closest|nearby|close by|local)\s+(?P<cat>.+)$"
                     r"|^(?P<cat2>.+?)\s+(?:nearby|near me|close by|around here)$", re.I)


@dataclass
class Place:
    name: str
    address: str | None
    lat: float
    lng: float
    distance_m: float | None = None


@dataclass
class Plan:
    status: str                                    # route | clarify | error
    query: str
    say: str                                       # one line to speak to the walker
    origin: tuple[float, float] | None = None
    destination: Place | None = None
    choices: list[Place] = field(default_factory=list)
    points: list[tuple[float, float]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    distance_m: float | None = None
    duration_s: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- parsing and geometry
def destination_text(text: str) -> tuple[str, bool]:
    """'take me to the nearest pharmacy' -> ('pharmacy', True); 'take me to Doe Library' -> ('Doe Library', False)."""
    q = _ASK.sub("", " ".join(text.strip().rstrip(".!?").split()))
    m = _NEARBY.match(q)
    if m:
        return (m.group("cat") or m.group("cat2")).strip(), True
    return re.sub(r"^the\s+", "", q, flags=re.I), False


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Google's encoded polyline (1e-5 degrees) -> [(lat, lng)]."""
    pts, i, lat, lng = [], 0, 0, 0
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


# ---------------------------------------------------------------- Google
def _key() -> str:
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    env = Path(__file__).resolve().parent.parent / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("GOOGLE_MAPS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise RuntimeError("no GOOGLE_MAPS_API_KEY (set it in the environment or the repo's .env)")
    return key


def _ssl() -> ssl.SSLContext:
    try:                                           # python.org builds on macOS ship without CA certificates
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _post(url: str, body: dict, fields: str, timeout: float = 6.0) -> dict:
    """POST JSON with the key in a header (never in a URL or a log line)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "X-Goog-Api-Key": _key(), "X-Goog-FieldMask": fields})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl()) as r:   # nosec B310: fixed https hosts
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            why = json.load(e).get("error", {}).get("status") or e.reason
        except ValueError:
            why = e.reason
        raise RuntimeError(f"Google said {e.code} {why}") from None


def search_places(query: str, near: tuple[float, float], nearby: bool) -> list[Place]:
    body = {"textQuery": query, "maxResultCount": SEARCH_RESULTS,
            "locationBias": {"circle": {"center": {"latitude": near[0], "longitude": near[1]},
                                        "radius": NEARBY_RADIUS_M if nearby else NAMED_RADIUS_M}}}
    if nearby:
        body["rankPreference"] = "DISTANCE"
    data = _post(PLACES_URL, body, "places.displayName,places.shortFormattedAddress,places.location")
    out = []
    for p in data.get("places") or []:
        loc = p.get("location") or {}
        name = (p.get("displayName") or {}).get("text")
        if name and loc.get("latitude") is not None:
            out.append(Place(name, p.get("shortFormattedAddress"), loc["latitude"], loc["longitude"],
                             haversine_m(*near, loc["latitude"], loc["longitude"])))
    return sorted(out, key=lambda p: p.distance_m)[:MAX_CHOICES]   # a chain's nearest branch, not the top-ranked one


def walking_route(origin: tuple[float, float], dest: Place) -> tuple[list, list[str], float | None, float | None]:
    """(path points, step instructions, metres, seconds) for a walk from origin to dest."""
    def wp(lat: float, lng: float) -> dict:
        return {"location": {"latLng": {"latitude": lat, "longitude": lng}}}
    data = _post(ROUTES_URL, {"origin": wp(*origin), "destination": wp(dest.lat, dest.lng), "travelMode": "WALK",
                              "languageCode": "en-US", "units": "METRIC"}, ROUTES_FIELDS)
    routes = data.get("routes") or []
    if not routes:
        raise RuntimeError("no walking route found")
    r = routes[0]
    steps = [". ".join(x.strip() for x in ((s.get("navigationInstruction") or {}).get("instructions") or "").splitlines()
                       if x.strip())
             for leg in r.get("legs") or [] for s in leg.get("steps") or []]
    dur = str(r.get("duration") or "").rstrip("s")
    return (decode_polyline((r.get("polyline") or {}).get("encodedPolyline") or ""), [s for s in steps if s],
            float(r.get("distanceMeters") or 0) or None, float(dur) if dur.replace(".", "", 1).isdigit() else None)


# ---------------------------------------------------------------- the API
def phone_position() -> tuple[float, float] | None:
    """The newest phone fix from SpacetimeDB (python -m phone.latest), or None."""
    try:
        from phone.latest import latest
        fixes = latest()
    except Exception as e:                         # no network, no phone yet: plan() says so
        print(f"[navigate] no phone position: {e}")
        return None
    return (fixes[0]["lat"], fixes[0]["lng"]) if fixes else None


def _how_far(m: float | None) -> str:
    if m is None:
        return ""
    return f", {round(m / 10) * 10:.0f} meters away" if m < 1000 else f", {m / 1000:.1f} kilometers away"


def _failed(query: str, say: str, origin, e: Exception) -> Plan:
    print(f"[navigate] {type(e).__name__}: {e}")
    return Plan("error", query, say, origin)


def plan(text: str, origin: tuple[float, float] | None = None) -> Plan:
    """What the walker said -> a route, a 'which one?' question, or an error to speak."""
    query, nearby = destination_text(text)
    if not query:
        return Plan("error", text, "Where would you like to go?")
    origin = origin or phone_position()
    if origin is None:
        return Plan("error", query, "I don't have your location yet. Is the phone connected?")
    try:
        places = search_places(query, origin, nearby)
    except (OSError, RuntimeError) as e:
        return _failed(query, "I can't look up places right now.", origin, e)
    if not places:
        return Plan("error", query, f"I couldn't find {query} near you.", origin)
    if nearby or len(places) == 1:
        return _routed(query, origin, places[0])                 # closest first
    names = [p.name + (f" on {p.address.split(',')[0]}" if p.address else "") for p in places]
    ask = "Which one? " + ". ".join(f"{i}: {n}" for i, n in enumerate(names, 1)) + "."
    return Plan("clarify", query, ask, origin, choices=places)


def choose(p: Plan, n: int) -> Plan:
    """Answer a 'which one?' with choice n (1-based)."""
    if p.status != "clarify" or not 1 <= n <= len(p.choices) or p.origin is None:
        return Plan("error", p.query, "Sorry, which one?", p.origin, choices=p.choices)
    return _routed(p.query, p.origin, p.choices[n - 1])


def _routed(query: str, origin: tuple[float, float], dest: Place) -> Plan:
    try:
        points, steps, dist, dur = walking_route(origin, dest)
    except (OSError, RuntimeError) as e:
        return _failed(query, f"I found {dest.name} but can't get walking directions.", origin, e)
    n = max(1, round(dur / 60)) if dur else 0
    mins = f", about {n} minute{'s' if n != 1 else ''}" if n else ""
    return Plan("route", query, f"Going to {dest.name}{_how_far(dist)}{mins}.", origin, dest,
                points=points or [origin, (dest.lat, dest.lng)], steps=steps, distance_m=dist, duration_s=dur)


def _lat_lng(text: str) -> tuple[float, float]:
    """--from as "lat,lng": two finite numbers on the globe, or a usage error before any network call."""
    try:
        lat, lng = (float(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected lat,lng, got {text!r}") from None
    if not (math.isfinite(lat) and math.isfinite(lng) and abs(lat) <= 90 and abs(lng) <= 180):
        raise argparse.ArgumentTypeError(f"not a place on the globe: {text!r}")
    return lat, lng


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser(prog="python -m kite.navigate", description=__doc__.splitlines()[0])
    a.add_argument("text", help='what the walker said, e.g. "take me to the nearest pharmacy"')
    a.add_argument("--from", dest="origin", type=_lat_lng, help="lat,lng instead of the phone's GPS")
    a.add_argument("--pick", type=int, help="answer 'which one?' with this choice (1-based)")
    a.add_argument("--json", action="store_true", help="print the whole plan as JSON")
    args = a.parse_args(argv)
    p = plan(args.text, args.origin)
    if p.status == "clarify" and args.pick:
        p = choose(p, args.pick)
    if args.json:
        print(json.dumps(p.to_dict(), indent=1))
    else:
        print(f"[{p.status}] {p.say}")
        for i, c in enumerate(p.choices if p.status == "clarify" else [], 1):
            print(f"  {i}. {c.name}  {c.address or ''}{_how_far(c.distance_m)}")
        for s in p.steps:
            print(f"  - {s}")
        if p.points:
            print(f"  path: {len(p.points)} points, {p.distance_m or 0:.0f} m")
    return 0 if p.status != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
