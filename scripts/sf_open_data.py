"""San Francisco open data (DataSF, Socrata API at data.sf.gov, no key) for the Moss St box.

    python scripts/sf_open_data.py          # fetch (cached in /tmp/kite_sfod), print what it would change in moss.json
    python scripts/osm_corner.py            # uses this module to merge it into moss.json

Datasets (all public domain / PDDL on DataSF; cited in sim/public/osm/CREDITS.md):
  ynuv-fyni  Building Footprints, with 2010 LiDAR heights (hgt_median_m, hgt_maxcm, ...)
  emxt-b6yg  Curbs and Islands (curb lines)
  4g86-grxu  Sidewalk Widths (2014, feet per street segment)
  tkzw-k3nq  Street Tree Inventory (position, species, DBH)
  8vzz-qzz9  Parking Meters

What osm_corner.py takes from it:
  - building heights: each OSM footprint is sampled every 0.5 m and each sample takes the LiDAR roof height of
    the DataSF footprint it falls in (hgt_median_m, the median roof above the lowest ground point). The building
    height is the median of its samples; `h_max` keeps the highest point (parapets, penthouses, stair towers).
  - curbs: the real curb line on each side of every road, every CURB_STEP metres along it (curb_offsets()).
  - sidewalk widths per street, trees and parking meters as `furniture`.
Street lights and utility poles are not in DataSF; scripts/streetview_facades.py places the Moss poles locally.
"""
from __future__ import annotations

import json
import math
import ssl
import statistics
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://data.sf.gov/resource"
CACHE = Path("/tmp/kite_sfod")
UA = "kite-calhacks13-sim/0.1 (Moss St open-data extract)"
FT = 0.3048
IN = 0.0254
DATASETS = {   # id: (geometry column, what)
    "ynuv-fyni": ("shape", "Building Footprints (LiDAR heights)"),
    "emxt-b6yg": ("the_geom", "Curbs and Islands"),
    "4g86-grxu": ("shape", "Sidewalk Widths (2014)"),
    "tkzw-k3nq": (None, "Street Tree Inventory"),
    "8vzz-qzz9": ("shape", "Parking Meters"),
}
SAMPLE_M = 0.5        # footprint sampling step for heights
CURB_STEP = 5.0       # road stations for curb offsets
CURB_WIN = 1.5        # curb points within this many metres along a station count for it
CURB_MAX = 15.0       # ignore curb points further than this from the centreline


def _ctx():
    try:
        import certifi   # the python.org macOS build has no CA bundle and data.sf.gov's chain isn't in /etc/ssl
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(box_ll, refresh=False):
    """{dataset id: rows} for everything intersecting box_ll = (south, west, north, east)."""
    s, w, n, e = box_ll
    poly = f"'POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))'"
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for ds, (geo, _) in DATASETS.items():
        path = CACHE / f"{ds}.json"
        if path.exists() and not refresh:
            out[ds] = json.loads(path.read_text())
            continue
        where = f"intersects({geo},{poly})" if geo else \
            f"latitude between {s} and {n} and longitude between {w} and {e}"
        url = f"{API}/{ds}.json?" + urllib.parse.urlencode({"$where": where, "$limit": 5000})
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=90, context=_ctx()) as r:
            rows = json.load(r)
        path.write_text(json.dumps(rows))
        out[ds] = rows
    return out


# ---------------------------------------------------------------- geometry helpers
def _rings(geom, to_local):
    """Polygon / MultiPolygon -> list of polygons, each a list of rings in local metres."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [[[to_local(lat, lng) for lng, lat in ring] for ring in poly] for poly in polys]


def _lines(geom, to_local):
    ls = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
    return [[to_local(lat, lng) for lng, lat in ln] for ln in ls]


def _inside(p, ring):
    c = False
    for a, b in zip(ring, ring[1:] + ring[:1]):
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
            c = not c
    return c


def _in_poly(p, rings):
    return _inside(p, rings[0]) and not any(_inside(p, h) for h in rings[1:])


def _bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------- buildings
def footprints(raw, to_local):
    out = []
    for r in raw["ynuv-fyni"]:
        if "hgt_median_m" not in r:
            continue
        for rings in _rings(r["shape"], to_local):
            out.append({"rings": rings, "bbox": _bbox(rings[0]), "id": r.get("sf16_bldgid"),
                        "median": float(r["hgt_median_m"]), "max": float(r.get("hgt_maxcm", 0)) / 100})
    return out


def building_heights(pts, fps):
    """LiDAR heights for one OSM footprint: {"h", "h_max", "cover"} or None if no DataSF footprint covers it."""
    x0, y0, x1, y1 = _bbox(pts)
    cands = [f for f in fps if f["bbox"][0] <= x1 and f["bbox"][2] >= x0 and f["bbox"][1] <= y1 and f["bbox"][3] >= y0]
    hs, mx, n = [], [], 0
    y = y0 + SAMPLE_M / 2
    while y < y1:
        x = x0 + SAMPLE_M / 2
        while x < x1:
            if _inside((x, y), pts):
                n += 1
                f = next((f for f in cands if _in_poly((x, y), f["rings"])), None)
                if f:
                    hs.append(f["median"])
                    mx.append(f["max"])
            x += SAMPLE_M
        y += SAMPLE_M
    if not hs or len(hs) < 0.3 * n:
        return None
    return {"h": round(statistics.median(hs), 1), "h_max": round(max(mx), 1), "cover": round(len(hs) / n, 2)}


# ---------------------------------------------------------------- curbs and sidewalks
def curb_points(raw, to_local, step=0.5):
    pts = []
    for r in raw["emxt-b6yg"]:
        for ln in _lines(r["the_geom"], to_local):
            for a, b in zip(ln, ln[1:]):
                n = max(1, int(math.dist(a, b) / step))
                pts += [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n)]
    return pts


def _project(p, pts):
    best, along = None, 0.0
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        if L < 1e-9:
            continue
        u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        t = max(0.0, min(L, (p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1]))
        d = math.hypot(p[0] - a[0] - u[0] * t, p[1] - a[1] - u[1] * t)
        if best is None or d < best[0]:
            best = (d, along + t, u[0] * (p[1] - a[1]) - u[1] * (p[0] - a[0]))
        along += L
    return best


def curb_offsets(pts, cpts):
    """Real curb offsets along a road polyline (left of travel +): [(along, lo, hi)] every CURB_STEP m, or None.
    A station takes the nearest curb point on each side within CURB_WIN along; stations at junctions (one side
    missing, or a width far from the road's median) are filled from their neighbours."""
    total = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    x0, y0, x1, y1 = _bbox(pts)
    near = [p for p in cpts if x0 - CURB_MAX <= p[0] <= x1 + CURB_MAX and y0 - CURB_MAX <= p[1] <= y1 + CURB_MAX]
    proj = [_project(p, pts) for p in near]
    n = max(1, round(total / CURB_STEP))
    stations = []
    for i in range(n + 1):
        s = total * i / n
        sides = [side for d, a, side in proj if abs(a - s) < CURB_WIN and d < CURB_MAX and abs(side) > 1.0]
        lo = max((v for v in sides if v < 0), default=None)
        hi = min((v for v in sides if v > 0), default=None)
        stations.append([s, lo, hi])
    good = [st for st in stations if st[1] is not None and st[2] is not None]
    if len(good) < 0.5 * len(stations):
        return None
    med = statistics.median(st[2] - st[1] for st in good)
    ok = [st[1] is not None and st[2] is not None and 0.7 * med <= st[2] - st[1] <= 1.4 * med for st in stations]
    keep = [st for st, k in zip(stations, ok) if k]
    for st, k in zip(stations, ok):
        if not k:   # nearest good station's offsets
            g = min(keep, key=lambda q: abs(q[0] - st[0]))
            st[1], st[2] = g[1], g[2]
    return [(round(s, 2), round(lo, 2), round(hi, 2)) for s, lo, hi in stations]


def sidewalk_widths(raw):
    """{"MOSS ST": metres} from the 2014 survey (sidewalk_f, feet; the median over the street's segments)."""
    by = {}
    for r in raw["4g86-grxu"]:
        if r.get("sidewalk_f"):
            by.setdefault(f"{r['street']} {r.get('st_type', '')}".strip(), []).append(float(r["sidewalk_f"]) * FT)
    return {k: round(statistics.median(v), 2) for k, v in by.items()}


def street_key(name):
    """OSM "7th Street" -> DataSF "07TH ST"."""
    parts = name.upper().replace("STREET", "ST").split()
    if parts and parts[0][:-2].isdigit():
        parts[0] = parts[0].zfill(4)
    return " ".join(parts)


# ---------------------------------------------------------------- trees, meters
def tree_size(species, dbh_in):
    """(height m, crown radius m) from DBH and species: urban street-tree allometry, roughly height = 2 + 0.55 *
    DBH(cm)^0.85 capped by species, crown radius about 0.22 m per DBH inch. ponytail: one curve for all
    species with a per-species cap; good to a metre or two, which is what the camera sees."""
    cap = {"Platanus": 18, "Magnolia": 12, "Lophostemon": 14, "Ficus": 10, "Pyrus": 9, "Tristaniopsis": 9,
           "Prunus": 7, "Callistemon": 6, "Arbutus": 8, "Pinus": 15}
    genus = (species or "").split()[0] if species else ""
    d = max(2.0, dbh_in or 3.0)
    h = min(cap.get(genus, 10), 2.0 + 0.55 * (d * 2.54) ** 0.85)
    r = max(0.8, min(5.0, 0.22 * d + 0.4))
    return round(h, 1), round(r, 1)


def furniture(raw, to_local, inside_box):
    out = []
    for r in raw["tkzw-k3nq"]:
        if not (r.get("latitude") and r.get("longitude")) or r.get("planttype", "Tree") != "Tree":
            continue
        x, y = to_local(float(r["latitude"]), float(r["longitude"]))
        if not inside_box(x, y):
            continue
        sp = (r.get("species") or "").split("::")[0].strip()
        dbh = float(r["mapdbh"]) if r.get("mapdbh") else None
        h, cr = tree_size(sp, dbh)
        out.append({"id": int(r["treeid"]), "kind": "tree", "x": round(x, 2), "y": round(y, 2),
                    "tags": {"species": sp, "dbh_in": dbh, "height_m": h, "crown_r_m": cr,
                             "source": "DataSF tkzw-k3nq"}})
    for r in raw["8vzz-qzz9"]:
        if r.get("active_meter_flag") not in ("M", "P", "T"):   # U = removed
            continue
        x, y = to_local(float(r["latitude"]), float(r["longitude"]))
        if inside_box(x, y):
            out.append({"id": int(float(r["objectid"])), "kind": "parking_meter", "x": round(x, 2), "y": round(y, 2),
                        "tags": {"meter_type": r.get("meter_type", ""), "source": "DataSF 8vzz-qzz9"}})
    return out


def load(to_local, box_local, to_latlng, refresh=False):
    """Everything osm_corner.py needs, or None (with a warning) when DataSF can't be reached."""
    w, s, e, n = box_local
    sw, ne = to_latlng(w, s), to_latlng(e, n)
    try:
        raw = fetch((sw["lat"], sw["lng"], ne["lat"], ne["lng"]), refresh)
    except OSError as ex:
        print(f"[sf_open_data] DataSF unreachable ({ex}); keeping OSM-only estimates", file=sys.stderr)
        return None
    return {"footprints": footprints(raw, to_local), "curbs": curb_points(raw, to_local),
            "sidewalks": sidewalk_widths(raw),
            "furniture": furniture(raw, to_local, lambda x, y: w <= x <= e and s <= y <= n),
            "counts": {ds: len(rows) for ds, rows in raw.items()}}


def _self_check():
    sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
    fp = [{"rings": [sq], "bbox": (0, 0, 4, 4), "id": "a", "median": 7.0, "max": 9.0}]
    assert building_heights([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)], fp)["h"] == 7.0
    assert building_heights([(10, 10), (12, 10), (12, 12), (10, 10)], fp) is None
    road = [(0, 0), (40, 0)]
    curbs = [(x / 2, y) for x in range(81) for y in (-2.0, 4.0)]
    offs = curb_offsets(road, curbs)
    assert all(abs(lo + 2) < 1e-6 and abs(hi - 4) < 1e-6 for _, lo, hi in offs), offs
    assert street_key("7th Street") == "07TH ST" and street_key("Moss Street") == "MOSS ST"
    assert tree_size("Platanus x hispanica", 18)[0] <= 18


if __name__ == "__main__":
    _self_check()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import osm_corner as oc
    d = load(oc.to_local, oc.BBOX, oc.to_latlng, refresh="--refresh" in sys.argv)
    if d:
        print(d["counts"], d["sidewalks"])
        print(f"{len(d['footprints'])} footprints, {len(d['curbs'])} curb points, "
              f"{sum(f['kind'] == 'tree' for f in d['furniture'])} trees, "
              f"{sum(f['kind'] == 'parking_meter' for f in d['furniture'])} meters")
