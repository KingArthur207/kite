"""Fetch Moss St + the Howard St mouth from OpenStreetMap and write sim geometry.

    python scripts/osm_corner.py            # fetch from Overpass
    python scripts/osm_corner.py raw.json   # reuse a saved Overpass response

Writes sim/public/osm/moss.json (local metres, x east, y north, origin = the zebra
across Moss at Howard) and sim/public/osm/route.json (the demo walk as lat/lng).
Projection matches toLatLng() in sim/src/config.ts. Data (c) OpenStreetMap
contributors, ODbL. Where DataSF has better numbers (scripts/sf_open_data.py: LiDAR building heights, real
curb lines, surveyed sidewalk widths, street trees, parking meters) they replace the OSM estimates; `sources`
in moss.json says which. `--osm-only` skips DataSF.
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

import sf_open_data

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
# Street furniture nodes kept in moss.json `furniture` (kind = the first matching tag value).
FURNITURE_FILTERS = ['"highway"~"^(street_lamp|traffic_signals|stop|give_way)$"', '"natural"="tree"',
                     '"emergency"="fire_hydrant"',
                     '"amenity"~"^(waste_basket|bicycle_parking|parking_meter|vending_machine|bench)$"',
                     '"traffic_sign"', '"barrier"~"^(bollard|kerb)$"', '"power"="pole"', '"man_made"="utility_pole"']
FURNITURE_KEYS = ["highway", "natural", "emergency", "amenity", "barrier", "power", "man_made"]
FURNITURE_KINDS = {"street_lamp", "traffic_signals", "stop", "give_way", "tree", "fire_hydrant", "waste_basket",
                   "bicycle_parking", "parking_meter", "vending_machine", "bench", "bollard", "kerb", "pole",
                   "utility_pole"}
FURNITURE_TAGS = ("direction", "traffic_sign", "colour", "fire_hydrant:type", "tactile_paving", "vending", "kerb")
BUILDING_TAGS = ["building", "building:levels", "height", "building:material", "building:colour",
                 "roof:shape", "roof:colour", "shop", "amenity", "addr:housenumber"]
BAY_MIN, BAY_MAX, BAY_Z = 0.15, 2.0, 3.0
BAY_SPREAD = 0.5   # roof-outline overhang past the property line: clipped, from 3 m up
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
    furniture = "".join(f'node[{f}]({bb});' for f in FURNITURE_FILTERS)
    q = (f'[out:json][timeout:25];(way["building"]({bb});way["highway"]({bb});node["highway"="crossing"]({bb});'
         f'{furniture}node["shop"]({bb});node["amenity"]({bb}););out geom tags;')
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
    """Polyline shifted `off` metres to the left of travel (mitred, miter capped at 3x). `off` is a number or a
    list with one offset per point (real curbs vary along a road)."""
    offs = off if isinstance(off, (list, tuple)) else [off] * len(pts)
    segs = [left(unit(sub(b, a))) for a, b in zip(pts, pts[1:])]
    out = []
    for i, p in enumerate(pts):
        n1, n2 = segs[max(i - 1, 0)], segs[min(i, len(segs) - 1)]
        m = unit(add(n1, n2)) if dot(n1, n2) > -0.9 else n1
        out.append(add(p, mul(m, offs[i] * min(3.0, 1 / max(dot(m, n1), 1e-3)))))
    return out


def vadd(a, k):
    """Per-point offsets plus a constant (offsets may be a number or a list)."""
    return [v + k for v in a] if isinstance(a, list) else a + k


def vmid(a, b):
    return [(x + y) / 2 for x, y in zip(a, b)] if isinstance(a, list) else (a + b) / 2


def cumlen(pts):
    s = [0.0]
    for a, b in zip(pts, pts[1:]):
        s.append(s[-1] + math.dist(a, b))
    return s


def curbs_at(r, s):
    """(lo, hi) curb offsets of road r at distance s along it."""
    if "lo" not in r:
        return r["curbs"]
    ss = cumlen(r["pts"])
    for i in range(len(ss) - 1):
        if s <= ss[i + 1] or i == len(ss) - 2:
            t = 0 if ss[i + 1] == ss[i] else max(0.0, min(1.0, (s - ss[i]) / (ss[i + 1] - ss[i])))
            return (r["lo"][i] + (r["lo"][i + 1] - r["lo"][i]) * t, r["hi"][i] + (r["hi"][i + 1] - r["hi"][i]) * t)
    return r["curbs"]


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


def real_curbs(pts, sf):
    """pts with DataSF curb stations merged in, and per-point (lo, hi) offsets; or (pts, None)."""
    offs = sf_open_data.curb_offsets(pts, sf["curbs"]) if sf else None
    if not offs:
        return pts, None
    ss = cumlen(pts)
    merged = sorted({(round(s, 2), tuple(p)) for s, p in zip(ss, pts)} |
                    {(s, tuple(point_at(pts, min(s, ss[-1] - 1e-6)))) for s, _, _ in offs if 0.5 < s < ss[-1] - 0.5})
    def interp(s, k):
        for a, b in zip(offs, offs[1:]):
            if s <= b[0]:
                t = (s - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0
                return a[k] + (b[k] - a[k]) * max(0.0, t)
        return offs[-1][k]
    return [p for _, p in merged], [(interp(s, 1), interp(s, 2)) for s, _ in merged]


def layout(way, buildings, sf=None):
    t = way["tags"]
    oneway = t.get("oneway") in ("yes", "-1")
    pts = way["pts"] if t.get("oneway") != "-1" else way["pts"][::-1]
    pts, real = real_curbs(pts, sf)
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
    los, his = lo, hi                   # per point when DataSF has the real curbs
    sw_data = sf["sidewalks"].get(sf_open_data.street_key(t.get("name", ""))) if sf else None
    if sw_data:
        sidewalk = sw_data
    if real:
        los, his = [round(a, 2) for a, _ in real], [round(b, 2) for _, b in real]
        lo, hi = statistics.median(los), statistics.median(his)
        if lane_w != LANE_W:   # untagged alley: the lane fills the real roadway
            lane_w = (hi - lo) / lanes
    mph = float(t["maxspeed"].split()[0]) if "maxspeed" in t else DEFAULT_MPH.get(t["highway"], 25)
    # Lane centres from the left curb inward: track, buffer, parking, then travel lanes.
    x = vadd(his, -(TRACK_W + TRACK_BUFFER) * track_l - PARKING_W * park_l)
    lane_list = []
    for i in range(lanes):
        o = vadd(x, -lane_w * (i + 0.5))
        forward = oneway or i >= lanes / 2   # two-way: left half of lanes runs backwards
        lp = [r2(p) for p in offset(pts, o)]
        lane_list.append({"road": way["id"], "name": t.get("name", ""), "width": round(lane_w, 2),
                          "speed_mps": round(mph * MPH, 2), "pts": lp if forward else lp[::-1]})
    tracks = []
    if track_l:
        tracks.append((vadd(his, -TRACK_W), his))
    if track_r:
        tracks.append((los, vadd(los, TRACK_W)))
    out = {"id": way["id"], "name": t.get("name", ""), "highway": t["highway"], "oneway": oneway,
           "lanes": lanes, "maxspeed_mph": mph, "pts": pts, "curbs": (lo, hi), "row": row,
           "sidewalk": sidewalk, "tracks": tracks, "lane_list": lane_list,
           "carriage": strip_quads(pts, los, his), "curb_src": "DataSF emxt-b6yg" if real else "estimate",
           "sidewalk_src": "DataSF 4g86-grxu" if sw_data else "estimate"}
    if real:
        out["lo"], out["hi"] = los, his
    return out


def clip_half(poly, p, n):
    """Sutherland-Hodgman: the part of closed polygon `poly` where (x - p) . n >= 0."""
    ring = poly[:-1] if poly[0] == poly[-1] else poly
    out = []
    for i, a in enumerate(ring):
        b = ring[(i + 1) % len(ring)]
        da, db = dot(sub(a, p), n), dot(sub(b, p), n)
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            out.append(lerp(a, b, da / (da - db)))
    return out + out[:1] if len(out) >= 3 else None


def ground_footprints(buildings, roads):
    """Footprints are roof outlines, so SF bay windows (and some cornices) stick out over the sidewalk. Where a
    road has DataSF curbs and a surveyed sidewalk width, the property line is the curb plus that width; a
    footprint that crosses it by BAY_MIN..BAY_MAX metres is clipped there for the ground floor (`ground`), and
    the part outside overhangs from BAY_Z up."""
    lines = []
    for r in roads:
        if "lo" not in r or r["sidewalk_src"] == "estimate":
            continue
        for offs, sgn in ((r["hi"], 1), (r["lo"], -1)):
            prop = offset(r["pts"], [o + sgn * r["sidewalk"] for o in offs])
            lines.append((r, prop, sgn))
    for b in buildings:
        g = b["pts"]
        c = (sum(p[0] for p in g[:-1]) / (len(g) - 1), sum(p[1] for p in g[:-1]) / (len(g) - 1))
        for r, prop, sgn in lines:
            d, along, _ = project(c, prop)
            if d > 30 or along <= 0.5 or along >= along_len(prop) - 0.5:
                continue
            q = point_at(prop, along)
            u = unit(sub(point_at(prop, min(along + 0.5, along_len(prop))), q))
            n = mul(left(u), sgn)             # away from the road
            if dot(sub(c, q), n) <= 0:
                continue                      # building centre on the road side: not this road's frontage
            depth = -min(dot(sub(p, q), n) for p in g)
            if not BAY_MIN < depth < BAY_MAX:
                continue
            # Bays make the front step in and out. A front that sticks out by the same amount all along is an
            # offset footprint (or a narrower sidewalk there): keep it whole.
            front = [-dot(sub(p, q), n) for p in g if dot(sub(p, q), n) < 4.0]
            if max(front) - min(front) < BAY_SPREAD:
                continue
            cut = clip_half(g, q, n)
            if cut and poly_area(cut) > 0.6 * poly_area(b["pts"]):
                g = cut
        if g is not b["pts"]:
            b["ground"] = g


def poly_area(pts):
    return abs(sum(cross(a, b) for a, b in zip(pts, pts[1:]))) / 2


def clipped(pts, a, b, others, cut_along=()):
    """Strip quads between offsets a and b, minus parts inside other carriageways
    or within CROSSING_HALF of the given along-distances (curb ramps)."""
    la, lb, mid = offset(pts, a), offset(pts, b), offset(pts, vmid(a, b))
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


def build(raw, sf=None):
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
        b = {"id": e["id"], "h": round(h, 1), "pts": loc(e["geometry"]),
             "tags": {k: t[k] for k in BUILDING_TAGS if k in t}, "h_src": "osm" if "height" in t or
             "building:levels" in t else "default"}
        lidar = sf_open_data.building_heights(b["pts"], sf["footprints"]) if sf else None
        if lidar:
            b.update(h=lidar["h"], h_max=lidar["h_max"], h_osm=round(h, 1), h_src="DataSF ynuv-fyni LiDAR")
        buildings.append(b)
    polys = [b["pts"] for b in buildings]

    furniture, shop_nodes = [], []
    for e in els:
        t = e.get("tags", {})
        if e["type"] != "node":
            continue
        p = to_local(e["lat"], e["lon"])
        if "shop" in t or ("amenity" in t and t["amenity"] not in FURNITURE_KINDS):
            shop_nodes.append(p)
        kind = next((t[k] for k in FURNITURE_KEYS if t.get(k) in FURNITURE_KINDS), None)
        if kind == "kerb" and t.get("kerb") != "lowered":
            kind = None
        if kind is None and "traffic_sign" in t:
            kind = "traffic_sign"
        if kind:
            furniture.append({"id": e["id"], "kind": kind, "x": round(p[0], 2), "y": round(p[1], 2),
                              "tags": {k: t[k] for k in FURNITURE_TAGS if k in t}})
    # A building with a shop/amenity tag, or a shop/amenity node inside it, gets a storefront.
    for b in buildings:
        b["storefront"] = "shop" in b["tags"] or "amenity" in b["tags"] or \
            any(inside(p, b["pts"]) for p in shop_nodes)

    if sf:
        furniture += sf["furniture"]
    roads = [layout({"id": e["id"], "tags": e["tags"], "pts": loc(e["geometry"])}, polys, sf)
             for e in els if e["type"] == "way" and e.get("tags", {}).get("highway") in CAR_ROADS]
    by_id = {r["id"]: r for r in roads}

    ground_footprints(buildings, roads)
    ground_polys = [b.get("ground", b["pts"]) for b in buildings]
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
        clo, chi = curbs_at(r, along)
        mk = t.get("crossing:markings", t.get("crossing_ref", ""))
        crossings.append({"id": e["id"], "x": round(p[0], 2), "y": round(p[1], 2), "road": r["id"],
                          "markings": mk, "signals": t.get("crossing") == "traffic_signals",
                          "tactile": t.get("tactile_paving") == "yes",
                          "u": r2(unit(sub(point_at(r["pts"], min(along + 0.5, along_len(r["pts"]))),
                                           point_at(r["pts"], max(along - 0.5, 0))))),
                          "across": [round(clo - side, 2), round(chi - side, 2)]})

    sidewalks, curbs, road_quads, track_quads = [], [], [], []
    for r in roads:
        others = [q for o in roads if o is not r for q in o["carriage"]]
        lo, hi = r.get("lo", r["curbs"][0]), r.get("hi", r["curbs"][1])
        sw = r["sidewalk"] + SIDEWALK_OVERDRAW
        road_quads += r["carriage"]
        track_quads += [q for a, b in r["tracks"] for q in strip_quads(r["pts"], a, b)]
        sidewalks += clipped(r["pts"], hi, vadd(hi, sw), others) + clipped(r["pts"], vadd(lo, -sw), lo, others)
        for off in (lo, hi):
            for q in clipped(r["pts"], off, off, others, r.get("crossings", ())):
                curbs.append(r2(q[0]) + r2(q[1]))

    # Demo route: Moss NE sidewalk, NW to Howard, left across the Moss zebra, SW along Howard.
    moss, howard = by_id[MOSS], by_id[HOWARD]
    zebra = next(c for c in crossings if c["road"] == MOSS and c["markings"] == "zebra")
    s_cross = project((zebra["x"], zebra["y"]), moss["pts"])[1]
    # Curbs where the walk is: the zebra end of Moss, Howard's south side
    m_lo, m_hi = curbs_at(moss, s_cross + SPAWN_BEFORE_CROSSING / 2)
    h_lo, h_hi = curbs_at(howard, project((zebra["x"], zebra["y"]), howard["pts"])[1])
    sw_off = curbs_at(moss, s_cross)[0] - moss["sidewalk"] / 2
    hs_off = vadd(howard.get("hi", h_hi), howard["sidewalk"] / 2)   # Howard runs SW, so left (+) is the south side
    # Footprints are roof outlines, so bay windows poke into the sidewalk. Walk midway
    # between the curb and the narrowest NE face along the walk (Moss runs SE: left (+) is NE).
    faces = []
    for k in range(int(SPAWN_BEFORE_CROSSING) + 3):
        hi_k = curbs_at(moss, s_cross + k)[1]
        o = point_at(moss["pts"], s_cross + k, hi_k)
        u = unit(sub(point_at(moss["pts"], s_cross + k + 0.5), point_at(moss["pts"], s_cross + k)))
        faces.append((hi_k + ray_hit(o, left(u), ground_polys, 20.0), hi_k))
    ne_face, ne_curb = min(faces, key=lambda f: f[0] - f[1])
    ne_off = (ne_curb + ne_face) / 2
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
        "moss_roadway_range_m": [round(min(h - l for l, h in zip(moss["lo"], moss["hi"])), 2),
                                 round(max(h - l for l, h in zip(moss["lo"], moss["hi"])), 2)] if "lo" in moss else None,
        "moss_row_building_to_building_m": round(sum(moss["row"]), 2) if moss["row"] else None,
        "moss_sidewalk_m": round(moss["sidewalk"], 2),
        "moss_ne_sidewalk_narrowest_m": round(ne_face - ne_curb, 2),
        "moss_ne_walk_line_from_curb_m": round(ne_off - ne_curb, 2),
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
        "sources": {"osm": "OpenStreetMap via Overpass (footprints, roads, crossings, furniture nodes)",
                    **({f"datasf_{ds}": what for ds, (_, what) in sf_open_data.DATASETS.items()} if sf else {}),
                    "buildings_lidar_height": sum(b["h_src"].startswith("DataSF") for b in buildings),
                    "roads_real_curbs": [r["name"] or str(r["id"]) for r in roads if "lo" in r]},
        "buildings": [{"id": b["id"], "h": b["h"], "pts": [r2(p) for p in b["pts"]], "tags": b["tags"],
                       "storefront": b["storefront"], "h_src": b["h_src"],
                       **({"ground": [r2(p) for p in b["ground"]], "bay_z": BAY_Z} if "ground" in b else {}),
                       **{k: b[k] for k in ("h_max", "h_osm") if k in b}} for b in buildings],
        "furniture": furniture,
        "roads": [{"id": r["id"], "name": r["name"], "highway": r["highway"], "oneway": r["oneway"],
                   "lanes": r["lanes"], "maxspeed_mph": r["maxspeed_mph"],
                   "curbs": [round(v, 2) for v in r["curbs"]], "sidewalk": round(r["sidewalk"], 2),
                   "curb_src": r["curb_src"], "sidewalk_src": r["sidewalk_src"],
                   **({"curbs_at": [[a, b] for a, b in zip(r["lo"], r["hi"])]} if "lo" in r else {}),
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


def inside(p, poly):
    """Even-odd point in polygon."""
    c = False
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
            c = not c
    return c


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
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert inside((0.5, 0.5), sq) and not inside((2, 0.5), sq)
    assert all("tags" in b for b in data["buildings"]) and data["furniture"], "building tags or furniture missing"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw = json.loads(Path(args[0]).read_text()) if args else fetch()
    sf = None if "--osm-only" in sys.argv else sf_open_data.load(to_local, BBOX, to_latlng)
    data, route = build(raw, sf)
    self_check(data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "moss.json").write_text(json.dumps(data, separators=(",", ":")))
    (OUT / "route.json").write_text(json.dumps(route, indent=2) + "\n")
    print(json.dumps(data["measurements"], indent=2))
    print(f"{len(data['buildings'])} buildings, {len(data['roads'])} roads, "
          f"{len(data['crossings'])} crossings, {len(data['furniture'])} furniture, "
          f"{sum(b['storefront'] for b in data['buildings'])} storefronts -> {OUT}")


if __name__ == "__main__":
    main()
