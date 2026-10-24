"""Fetch Moss St + the Howard St mouth from OpenStreetMap and write sim geometry.

    python scripts/osm_corner.py            # fetch from Overpass
    python scripts/osm_corner.py raw.json   # reuse a saved Overpass response

Writes sim/public/osm/moss.json (local metres, x east, y north, origin = the zebra
across Moss at Howard) and sim/public/osm/route.json (the demo walk as lat/lng).
Projection matches toLatLng() in sim/src/config.ts. Data (c) OpenStreetMap
contributors, ODbL.
"""
import json
import os
import math
import ssl
import statistics
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ORIGIN = (37.778352, -122.408768)          # OSM node 12144593726, zebra across Moss
BBOX = (-80, -150, 140, 70)                # local metres: west, south, east, north
OVERPASS = "https://overpass-api.de/api/interpreter"
UA = "kite-calhacks13-sim/0.1 (OSM corner extract for a blind-walker drone sim)"
OUT = Path(__file__).resolve().parent.parent / "sim" / "public" / "osm"

M_PER_DEG = 111320
LEVEL_H = 3.2
DEFAULT_LEVELS = 3
LANE_W = 3.2
PARKING_W = 2.4
TRACK_W, TRACK_BUFFER = 2.0, 0.9
ALLEY_W = 6.1                  # untagged residential: 20 ft roadway, SF 35 ft alley standard
DEFAULT_SIDEWALK = 3.0
MIN_SIDEWALK = 1.5
SIDEWALK_OVERDRAW = 3.0        # sidewalk slab runs this far under buildings so no gaps show
CROSSING_HALF = 1.5            # curb ramp half-width at a marked crossing
CAR_ROADS = {"primary", "secondary", "tertiary", "residential", "unclassified",
             "living_street", "primary_link", "secondary_link", "tertiary_link"}
MPH = 0.44704
DEFAULT_MPH = {"residential": 15}   # CA alley limit; Moss has no maxspeed tag
MOSS, HOWARD = 8916291, 1469149564
SPAWN_BEFORE_CROSSING = 70.0
ROUTE_ALONG_HOWARD = 20.0


def to_local(lat, lng):
    k = M_PER_DEG * math.cos(math.radians(ORIGIN[0]))
    return ((lng - ORIGIN[1]) * k, (lat - ORIGIN[0]) * M_PER_DEG)


def to_latlng(x, y):
    k = M_PER_DEG * math.cos(math.radians(ORIGIN[0]))
    return {"lat": round(ORIGIN[0] + y / M_PER_DEG, 7), "lng": round(ORIGIN[1] + x / k, 7)}


def fetch():
    s, n = (ORIGIN[0] + v / M_PER_DEG for v in (BBOX[1], BBOX[3]))
    w, e = (to_latlng(v, 0)["lng"] for v in (BBOX[0], BBOX[2]))
    bb = f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"
    q = (f'[out:json][timeout:25];(way["building"]({bb});way["highway"]({bb});'
         f'node["highway"="crossing"]({bb}););out geom tags;')
    req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": UA})
    # python.org builds on macOS ship without CA certs; the system bundle works.
    cafile = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else None
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context(cafile=cafile)) as r:
        return json.load(r)


# --- 2D helpers (points are (x, y) tuples)
def sub(a, b): return (a[0] - b[0], a[1] - b[1])
def add(a, b): return (a[0] + b[0], a[1] + b[1])
def mul(a, k): return (a[0] * k, a[1] * k)
def dot(a, b): return a[0] * b[0] + a[1] * b[1]
def cross(a, b): return a[0] * b[1] - a[1] * b[0]
def unit(a):
    d = math.hypot(*a)
    return (a[0] / d, a[1] / d)
def left(u): return (-u[1], u[0])
def lerp(a, b, t): return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
def r2(p): return [round(p[0], 2), round(p[1], 2)]


def offset(pts, off):
    """Polyline shifted `off` metres to the left of travel (mitred, miter capped at 3x)."""
    segs = [left(unit(sub(b, a))) for a, b in zip(pts, pts[1:])]
    out = []
    for i, p in enumerate(pts):
        n1, n2 = segs[max(i - 1, 0)], segs[min(i, len(segs) - 1)]
        m = unit(add(n1, n2)) if dot(n1, n2) > -0.9 else n1
        out.append(add(p, mul(m, off * min(3.0, 1 / max(dot(m, n1), 1e-3)))))
    return out


def ray_hit(o, d, polys, maxd):
    best = maxd
    for poly in polys:
        for a, b in zip(poly, poly[1:]):
            e = sub(b, a)
            den = cross(d, e)
            if abs(den) < 1e-9:
                continue
            t, s = cross(sub(a, o), e) / den, cross(sub(a, o), d) / den
            if 0 < t < best and 0 <= s <= 1:
                best = t
    return best


def measure_row(pts, buildings, maxd=20.0):
    """Median distance from the centreline to building faces on each side, or None."""
    hits = ([], [])
    for a, b in zip(pts, pts[1:]):
        u = unit(sub(b, a))
        for k in range(1, int(math.dist(a, b) / 2)):
            o = add(a, mul(u, 2 * k))
            for side, n in enumerate((left(u), mul(left(u), -1))):
                d = ray_hit(o, n, buildings, maxd)
                if d < maxd:
                    hits[side].append(d)
    if min(len(h) for h in hits) < 5:
        return None
    return statistics.median(hits[0]), statistics.median(hits[1])


def clip_inside(a, b, quad):
    """Parameter interval of segment a->b inside a convex quad (Cyrus-Beck), or None."""
    t0, t1 = 0.0, 1.0
    area = sum(cross(quad[i], quad[(i + 1) % 4]) for i in range(4))
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        e = sub(q, p)
        num = cross(e, sub(a, p)) * area   # >0 inside for either winding
        den = cross(e, sub(b, a)) * area
        if abs(den) < 1e-12:
            if num < 0:
                return None
            continue
        t = -num / den
        if den > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    return (t0, t1) if t1 - t0 > 1e-6 else None


def keep_intervals(cuts):
    """Complement of the union of cut intervals within [0, 1]."""
    out, t = [], 0.0
    for a, b in sorted(cuts):
        if a > t:
            out.append((t, a))
        t = max(t, b)
    if t < 1:
        out.append((t, 1.0))
    return [(a, b) for a, b in out if b - a > 1e-4]


def strip_quads(pts, a, b):
    la, lb = offset(pts, a), offset(pts, b)
    return [[la[i], la[i + 1], lb[i + 1], lb[i]] for i in range(len(pts) - 1)]


def project(p, pts):
    """(distance along, signed left offset) of p against polyline pts."""
    best, along = None, 0.0
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        u = unit(sub(b, a))
        t = max(0.0, min(L, dot(sub(p, a), u)))
        d = math.dist(p, add(a, mul(u, t)))
        if best is None or d < best[0]:
            best = (d, along + t, cross(u, sub(p, a)))
        along += L
    return best


def point_at(pts, s, off=0.0):
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        if s <= L:
            u = unit(sub(b, a))
            return add(add(a, mul(u, s)), mul(left(u), off))
        s -= L
    raise ValueError("past end of polyline")


def layout(way, buildings):
    t = way["tags"]
    oneway = t.get("oneway") in ("yes", "-1")
    pts = way["pts"] if t.get("oneway") != "-1" else way["pts"][::-1]
    if "lanes" in t or t["highway"] != "residential":
        lanes = int(t.get("lanes", "1" if oneway else "2").split(";")[0])
        lane_w = LANE_W
    else:
        lanes = 1 if oneway else 2
        lane_w = ALLEY_W / lanes
    def side(s):
        return t.get(f"parking:lane:{s}", t.get("parking:lane:both")) == "parallel", \
               t.get(f"cycleway:{s}", t.get("cycleway:both")) == "track"
    (park_l, track_l), (park_r, track_r) = side("left"), side("right")
    W = lanes * lane_w + PARKING_W * (park_l + park_r) + (TRACK_W + TRACK_BUFFER) * (track_l + track_r)
    row = measure_row(pts, buildings)
    if row:
        c = (row[0] - row[1]) / 2
        sidewalk = max(MIN_SIDEWALK, (row[0] + row[1] - W) / 2)
    else:
        c, sidewalk = 0.0, DEFAULT_SIDEWALK
    lo, hi = c - W / 2, c + W / 2       # right curb, left curb (left of travel is +)
    mph = float(t["maxspeed"].split()[0]) if "maxspeed" in t else DEFAULT_MPH.get(t["highway"], 25)
    # Lane centres from the left curb inward: track, buffer, parking, then travel lanes.
    x = hi - (TRACK_W + TRACK_BUFFER) * track_l - PARKING_W * park_l
    lane_list = []
    for i in range(lanes):
        o = x - lane_w * (i + 0.5)
        forward = oneway or i >= lanes / 2   # two-way: left half of lanes runs backwards
        lp = [r2(p) for p in offset(pts, o)]
        lane_list.append({"road": way["id"], "name": t.get("name", ""), "width": round(lane_w, 2),
                          "speed_mps": round(mph * MPH, 2), "pts": lp if forward else lp[::-1]})
    tracks = []
    if track_l:
        tracks.append((hi - TRACK_W, hi))
    if track_r:
        tracks.append((lo, lo + TRACK_W))
    return {"id": way["id"], "name": t.get("name", ""), "highway": t["highway"], "oneway": oneway,
            "lanes": lanes, "maxspeed_mph": mph, "pts": pts, "curbs": (lo, hi), "row": row,
            "sidewalk": sidewalk, "tracks": tracks, "lane_list": lane_list,
            "carriage": strip_quads(pts, lo, hi)}


def clipped(pts, a, b, others, cut_along=()):
    """Strip quads between offsets a and b, minus parts inside other carriageways
    or within CROSSING_HALF of the given along-distances (curb ramps)."""
    la, lb, mid = offset(pts, a), offset(pts, b), offset(pts, (a + b) / 2)
    quads, s0 = [], 0.0
    for i in range(len(pts) - 1):
        L = math.dist(pts[i], pts[i + 1])
        cuts = [iv for q in others if (iv := clip_inside(mid[i], mid[i + 1], q))]
        cuts += [((s - CROSSING_HALF - s0) / L, (s + CROSSING_HALF - s0) / L)
                 for s in cut_along if s0 - CROSSING_HALF < s < s0 + L + CROSSING_HALF]
        for t0, t1 in keep_intervals(cuts):
            quads.append([lerp(la[i], la[i + 1], t0), lerp(la[i], la[i + 1], t1),
                          lerp(lb[i], lb[i + 1], t1), lerp(lb[i], lb[i + 1], t0)])
        s0 += L
    return quads


def build(raw):
    els = raw["elements"]
    loc = lambda g: [to_local(p["lat"], p["lon"]) for p in g]
    buildings = []
    for e in els:
        t = e.get("tags", {})
        if e["type"] != "way" or "building" not in t:
            continue
        try:
            h = float(t["height"].split()[0]) if "height" in t else \
                float(t.get("building:levels", DEFAULT_LEVELS)) * LEVEL_H
        except ValueError:
            h = DEFAULT_LEVELS * LEVEL_H
        buildings.append({"id": e["id"], "h": round(h, 1), "pts": loc(e["geometry"])})
    polys = [b["pts"] for b in buildings]

    roads = [layout({"id": e["id"], "tags": e["tags"], "pts": loc(e["geometry"])}, polys)
             for e in els if e["type"] == "way" and e.get("tags", {}).get("highway") in CAR_ROADS]
    by_id = {r["id"]: r for r in roads}

    crossings = []
    for e in els:
        t = e.get("tags", {})
        if e["type"] != "node" or t.get("highway") != "crossing":
            continue
        p = to_local(e["lat"], e["lon"])
        near = [(project(p, r["pts"]), r) for r in roads]
        (d, along, side), r = min(near, key=lambda x: x[0][0])
        if d > 3:
            continue
        r.setdefault("crossings", []).append(along)
        mk = t.get("crossing:markings", t.get("crossing_ref", ""))
        crossings.append({"id": e["id"], "x": round(p[0], 2), "y": round(p[1], 2), "road": r["id"],
                          "markings": mk, "signals": t.get("crossing") == "traffic_signals",
                          "tactile": t.get("tactile_paving") == "yes",
                          "u": r2(unit(sub(point_at(r["pts"], min(along + 0.5, along_len(r["pts"]))),
                                           point_at(r["pts"], max(along - 0.5, 0))))),
                          "across": [round(r["curbs"][0] - side, 2), round(r["curbs"][1] - side, 2)]})

    sidewalks, curbs, road_quads, track_quads = [], [], [], []
    for r in roads:
        others = [q for o in roads if o is not r for q in o["carriage"]]
        lo, hi = r["curbs"]
        sw = r["sidewalk"] + SIDEWALK_OVERDRAW
        road_quads += r["carriage"]
        track_quads += [q for a, b in r["tracks"] for q in strip_quads(r["pts"], a, b)]
        sidewalks += clipped(r["pts"], hi, hi + sw, others) + clipped(r["pts"], lo - sw, lo, others)
        for off in (lo, hi):
            for q in clipped(r["pts"], off, off, others, r.get("crossings", ())):
                curbs.append(r2(q[0]) + r2(q[1]))

    # Demo route: Moss NE sidewalk, NW to Howard, left across the Moss zebra, SW along Howard.
    moss, howard = by_id[MOSS], by_id[HOWARD]
    m_lo, m_hi = moss["curbs"]
    h_lo, h_hi = howard["curbs"]
    sw_off = m_lo - moss["sidewalk"] / 2
    hs_off = h_hi + howard["sidewalk"] / 2          # Howard runs SW, so left (+) is the south side
    zebra = next(c for c in crossings if c["road"] == MOSS and c["markings"] == "zebra")
    s_cross = project((zebra["x"], zebra["y"]), moss["pts"])[1]
    # Footprints are roof outlines, so bay windows poke into the sidewalk. Walk midway
    # between the curb and the narrowest NE face along the walk (Moss runs SE: left (+) is NE).
    faces = []
    for k in range(int(SPAWN_BEFORE_CROSSING) + 3):
        o = point_at(moss["pts"], s_cross + k, m_hi)
        u = unit(sub(point_at(moss["pts"], s_cross + k + 0.5), point_at(moss["pts"], s_cross + k)))
        faces.append(m_hi + ray_hit(o, left(u), polys, 20.0))
    ne_face = min(faces)
    ne_off = (m_hi + ne_face) / 2
    hpts = offset(howard["pts"], hs_off)
    def on_howard(off):   # where a Moss offset line meets Howard's south sidewalk centreline
        a, b = offset(moss["pts"], off)[:2]
        best = min((intersect(a, b, p, q) for p, q in zip(hpts, hpts[1:])),
                   key=lambda r: abs(r[1]) if r else 1e9)
        return best[0]
    spawn = point_at(moss["pts"], s_cross + SPAWN_BEFORE_CROSSING, ne_off)
    corner, far = on_howard(ne_off), on_howard(sw_off)
    end = add(far, mul(unit(sub(far, corner)), ROUTE_ALONG_HOWARD))
    route = [spawn, corner, far, end]
    names = ["spawn: Moss NE sidewalk", "corner: Moss NE sidewalk at Howard",
             "across the Moss zebra", "Howard south sidewalk, 20 m SW"]
    heading = math.degrees(math.atan2(*sub(corner, spawn))) % 360
    length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))

    measurements = {
        "moss_roadway_m": round(m_hi - m_lo, 2),
        "moss_row_building_to_building_m": round(sum(moss["row"]), 2) if moss["row"] else None,
        "moss_sidewalk_m": round(moss["sidewalk"], 2),
        "moss_ne_sidewalk_narrowest_m": round(ne_face - m_hi, 2),
        "moss_ne_walk_line_from_curb_m": round(ne_off - m_hi, 2),
        "moss_length_m": round(along_len(moss["pts"]), 1),
        "zebra_curb_to_curb_m": round(zebra["across"][1] - zebra["across"][0], 2),
        "zebra_walk_sidewalk_centre_to_centre_m": round(math.dist(corner, far), 2),
        "howard_roadway_m": round(h_hi - h_lo, 2),
        "howard_row_building_to_building_m": round(sum(howard["row"]), 2) if howard["row"] else None,
        "howard_south_sidewalk_m": round(howard["sidewalk"], 2),
        "route_length_m": round(length, 1),
        "spawn_to_corner_m": round(math.dist(spawn, corner), 1),
    }
    rounded = lambda qs: [[r2(p) for p in q] for q in qs]
    data = {
        "attribution": "Map data (c) OpenStreetMap contributors, ODbL 1.0. https://www.openstreetmap.org/copyright",
        "frame": "metres, x east, y north; origin = OSM node 12144593726 (zebra across Moss at Howard)",
        "origin": {"lat": ORIGIN[0], "lng": ORIGIN[1]},
        "spawn": {"x": round(spawn[0], 2), "y": round(spawn[1], 2), "heading": round(heading, 1)},
        "route": [r2(p) for p in route],
        "measurements": measurements,
        "buildings": [{"id": b["id"], "h": b["h"], "pts": [r2(p) for p in b["pts"]]} for b in buildings],
        "roads": [{"id": r["id"], "name": r["name"], "highway": r["highway"], "oneway": r["oneway"],
                   "lanes": r["lanes"], "maxspeed_mph": r["maxspeed_mph"],
                   "curbs": [round(v, 2) for v in r["curbs"]], "sidewalk": round(r["sidewalk"], 2),
                   "pts": [r2(p) for p in r["pts"]]} for r in roads],
        "lanes": [l for r in roads for l in r["lane_list"]],
        "crossings": crossings,
        "road_quads": rounded(road_quads),
        "track_quads": rounded(track_quads),
        "sidewalk_quads": rounded(sidewalks),
        "curbs": curbs,
    }
    route_ll = {"attribution": data["attribution"], "length_m": round(length, 1),
                "points": [dict(name=n, **to_latlng(*p)) for n, p in zip(names, route)]}
    return data, route_ll


def along_len(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def intersect(a, b, p, q):
    """Intersection of infinite line a-b with segment p-q: (point, distance outside p-q) or None."""
    d, e = sub(b, a), sub(q, p)
    den = cross(d, e)
    if abs(den) < 1e-9:
        return None
    s = cross(sub(a, p), d) / -den
    pt = add(p, mul(e, s))
    return pt, (0 if 0 <= s <= 1 else min(abs(s), abs(s - 1)) * math.hypot(*e))


def self_check(data):
    m = data["measurements"]
    assert 150 < m["moss_length_m"] < 210, m
    assert 4 < m["zebra_curb_to_curb_m"] < 9, m
    assert 1.5 <= m["moss_sidewalk_m"] < 4, m
    sx, sy = data["spawn"]["x"], data["spawn"]["y"]
    assert sx > 0 and sy < 0 and 300 < data["spawn"]["heading"] < 330, data["spawn"]
    assert 60 < m["route_length_m"] < 130, m


def main():
    raw = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else fetch()
    data, route = build(raw)
    self_check(data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "moss.json").write_text(json.dumps(data, separators=(",", ":")))
    (OUT / "route.json").write_text(json.dumps(route, indent=2) + "\n")
    print(json.dumps(data["measurements"], indent=2))
    print(f"{len(data['buildings'])} buildings, {len(data['roads'])} roads, "
          f"{len(data['crossings'])} crossings -> {OUT}")


if __name__ == "__main__":
    main()
