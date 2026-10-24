"""Understanding where the wearer wants to go.

    "my coffee shop"              -> memory   (Moss: the wearer's own saved places)
    "the end of the street"       -> relative (only the drone / route side can see it)
    "the next corner", "the crosswalk", "across the street", "back the way I came"
    "the nearest pharmacy"        -> nearby   (Google Places, ranked by distance from the wearer)
    "Palace of Fine Arts"         -> named    (Google Places, biased to the wearer's area)

Relative targets are sent on as structured requests ({"type":"relative",
"relation":"end_of_street"}) because they depend on what the drone and route
see right now; places are resolved to a name, address and coordinates here, so
the navigation side receives something it can route to directly.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

_RELATIVE = [
    (r"\b(?:end|bottom|top) of (?:the |this )?(?:street|road|block|sidewalk|path)\b", "end_of_street",
     "the end of the street"),
    (r"^(?:the )?(?:next |nearest |closest )?(?:corner|intersection)(?: please)?$", "next_intersection",
     "the next corner"),
    (r"^(?:the )?(?:next |nearest |closest )?(?:cross ?walk|crossing|pedestrian crossing)(?: please)?$",
     "nearest_crosswalk", "the crosswalk"),
    (r"\b(?:across|other side of) (?:the |this )?(?:street|road)\b", "cross_street", "the other side of the street"),
    (r"\bback (?:the way i came|where i (?:came from|started)|to (?:the )?start)\b|\bwhere i came from\b",
     "back", "where you started"),
    (r"^(?:the )?(?:entrance|door|way in)(?: please)?$", "nearest_entrance", "the entrance"),
]
_NEARBY = re.compile(r"^(?:the |a |an |any )?(?:nearest|closest|nearby|close by|local)\s+(?P<cat>.+)$"
                     r"|^(?:a |an |any |some )(?P<cat2>.+?)(?:\s+(?:nearby|near me|close by|around here))?$"
                     r"|^(?P<cat3>.+?)\s+(?:nearby|near me|close by|around here)$")


@dataclass
class Destination:
    kind: str                       # memory | relative | nearby | named
    query: str
    relation: str | None = None     # relative targets
    spoken: str | None = None       # how to say it back
    name: str | None = None
    address: str | None = None
    lat: float | None = None
    lng: float | None = None
    distance_m: float | None = None
    extra: dict = field(default_factory=dict)

    def request(self) -> dict:
        """The voice_request payload for the navigation side."""
        d = {"intent": "navigate", "destination": self.query, "target_type": self.kind}
        if self.relation:
            d["relation"] = self.relation
        for k in ("name", "address", "lat", "lng", "distance_m"):
            v = getattr(self, k)
            if v is not None:
                d[k] = round(v, 6) if isinstance(v, float) else v
        d.update(self.extra)
        return d


def classify(text: str) -> Destination:
    q = " ".join(text.strip().rstrip(".!?").split())
    low = q.lower()
    for pat, rel, spoken in _RELATIVE:
        if re.search(pat, low):
            return Destination("relative", q, relation=rel, spoken=spoken)
    m = _NEARBY.match(low)
    if m:
        cat = (m.group("cat") or m.group("cat2") or m.group("cat3") or "").strip()
        if cat:
            return Destination("nearby", cat)
    return Destination("named", re.sub(r"^(?:the)\s+", "", q, flags=re.I) if q.lower().startswith("the ") else q)


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def places_search(query: str, lat: float | None, lng: float | None, nearby: bool,
                  key: str | None = None, timeout: float = 4.0) -> list[dict]:
    """Google Places API (New) text search. Returns [{name, address, lat, lng}]."""
    import httpx
    key = key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise RuntimeError("no GOOGLE_MAPS_API_KEY")
    body: dict = {"textQuery": query, "maxResultCount": 3}
    if lat is not None and lng is not None:
        body["locationBias"] = {"circle": {"center": {"latitude": lat, "longitude": lng},
                                           "radius": 1500.0 if nearby else 20000.0}}
        if nearby:
            body["rankPreference"] = "DISTANCE"
    r = httpx.post("https://places.googleapis.com/v1/places:searchText", json=body, timeout=timeout,
                   headers={"X-Goog-Api-Key": key,
                            "X-Goog-FieldMask": "places.displayName,places.shortFormattedAddress,places.location"})
    r.raise_for_status()
    out = []
    for p in r.json().get("places", []) or []:
        loc = p.get("location") or {}
        out.append({"name": (p.get("displayName") or {}).get("text"), "address": p.get("shortFormattedAddress"),
                    "lat": loc.get("latitude"), "lng": loc.get("longitude")})
    return [p for p in out if p["name"] and p["lat"] is not None]


def resolve(dest: Destination, lat: float | None, lng: float | None, search=places_search) -> Destination:
    """Fill in name / address / coordinates for nearby and named destinations (network)."""
    if dest.kind not in ("nearby", "named"):
        return dest
    hits = search(dest.query, lat, lng, dest.kind == "nearby")
    if not hits:
        return dest
    h = hits[0]
    dest.name, dest.address, dest.lat, dest.lng = h["name"], h["address"], h["lat"], h["lng"]
    if lat is not None and lng is not None:
        dest.distance_m = haversine_m(lat, lng, h["lat"], h["lng"])
    return dest
