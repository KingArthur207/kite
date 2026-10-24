"""Per-building facade layouts for Moss St from Street View (permission: docs/google-permission.md; the images and
everything measured from them stay local, never committed).

    U="uv run --no-project --with numpy --with opencv-python-headless"
    $U python scripts/facade_layouts.py rectify [ids]                       # fronts
    $U --with anthropic python scripts/facade_layouts.py extract [ids]
    $U --with anthropic python scripts/facade_layouts.py review [ids]       # optional second look
    $U python scripts/facade_layouts.py ground                              # sidewalk to y 0 in an existing JSON
    $U python scripts/facade_layouts.py rectify --sides; ... extract [ids] --sides   # exposed side walls
    $U --with playwright python scripts/facade_layouts.py compare [ids]     # needs a vite on :5191 (cd sim; npx vite --port 5191)
    $U python scripts/facade_layouts.py --selftest

rectify: one straight-on image per Moss front (sim/public/osm/moss.json ground footprint, the edges facing Moss
  merged into one plane from the leftmost to the rightmost corner, as seen from the street), from the cached,
  pose-calibrated Street View tiles of scripts/streetview_facades.py (/tmp/kite_sv; no new requests). Each image
  comes from ONE panorama (the one that sees the front cleanest and squarest): tiles of one panorama share a camera
  centre, so there are no seams or ghosts between panoramas. Trees and cars stay in (no painted-out smears). The
  plane is moved towards the street where the panoramas agree the facade really is (front_depth). Written to
  /tmp/kite_sv_layout/rect/<id>.jpg at PX_PER_M, plus a copy with a 1 m grid for the model. --sides: the walls
  facing along Moss that show over a lower neighbour (or a corner building's other street face).
extract: Claude reads each rectified image (with its metre scale) and returns the layout as strict JSON; it is
  validated (openings inside the wall, no overlaps, sane sizes; stacked bays merged), retried once with the
  problems listed, and written to sim/public/local/moss_facade_layouts.json ("buildings" by id, "sides" by
  "<id>:s<edge>"). Fronts that don't validate are left out (the sim draws them procedurally). ANTHROPIC_API_KEY
  from .env, never printed.
review: Claude sees its layout drawn over the photo and returns a corrected one (kept only if it validates).
ground: the photos' y 0 sits 0.1-2 m under the real sidewalk; extract and review already ground each layout (the lowest
  garage door or entry is the sidewalk, doors and garages start there: no invented steps); this re-grounds an old file.
compare: sim renders of each front (sim/facadecam.html?look=2, square cameras in strips) under the rectified
  Street View front, in out/facades_cmp/<id>.jpg.

Layout JSON per building (metres; x along the front from its left end as seen from the street, y up from the
sidewalk):
  {"a", "b", "width", "x0", "x1", "roofline", "floors": [{"y0", "y1"}], "cornice": {"y0", "y1"}|null, "belts": [y],
   "openings": [{"type": window|bay|door|garage|storefront, "x0", "x1", "y0", "y1", "shape": rect|arch|segmental,
                 "spring": y|null, "style": rollup|panel|gate|null, "panes": 1-8, "frame": white|dark|wall}]}
A "bay" is a projecting bay window spanning floors y0..y1 (its windows are drawn on its faces).
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT if (ROOT / ".env").exists() else ROOT.parents[2] if (ROOT.parents[2] / ".env").exists() else ROOT
MOSS = ROOT / "sim/public/osm/moss.json"
LOCAL = ROOT / "sim/public/local"
OUT_JSON = LOCAL / "moss_facade_layouts.json"
CSV = LOCAL / "moss_buildings.csv"
WORK = Path("/tmp/kite_sv_layout")
CMP = ROOT / "out/facades_cmp"
PX_PER_M = 60
MODEL = "claude-sonnet-5-5"
MOSS_ROAD = 8916291
FACING = 0.35           # outward normal . (towards Moss) for an edge to be part of the front
FRONT_MAX_OFF = 14.0    # m from Moss's centreline
MIN_EDGE = 0.3
TYPES = {"window", "bay", "door", "garage", "storefront"}
SHAPES = {"rect", "arch", "segmental"}
STYLES = {"rollup", "panel", "gate", None}
GAP = 0.08              # m: openings closer than this overlap


# ---------------------------------------------------------------- fronts
def _ccw(pts):
    p = np.asarray(pts, float)
    return 0.5 * np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]) > 0


def moss_line(data):
    road = next(r for r in data["roads"] if r["id"] == MOSS_ROAD)
    p = np.asarray(road["pts"], float)
    u = (p[-1] - p[0]) / np.linalg.norm(p[-1] - p[0])
    return p[0], u


def fronts(data, ids):
    """{id: dict(a, b, normal, h)}: a -> b is the front's left to right as seen from Moss (sim x east, y north)."""
    o, u = moss_line(data)
    n_line = np.array([-u[1], u[0]])
    out = {}
    for b in data["buildings"]:
        if b["id"] not in ids:
            continue
        poly = b.get("ground") or b["pts"]
        ccw = _ccw(poly)
        ends = []
        for k in range(len(poly) - 1):
            a, c = np.asarray(poly[k], float), np.asarray(poly[k + 1], float)
            L = np.linalg.norm(c - a)
            if L < MIN_EDGE:
                continue
            e = (c - a) / L
            nrm = np.array([e[1], -e[0]]) if ccw else np.array([-e[1], e[0]])
            off = ((a + c) / 2 - o) @ n_line
            if abs(off) < FRONT_MAX_OFF and nrm @ (-np.sign(off) * n_line) > FACING:
                ends += [a, c]
        if not ends:
            continue
        ends = np.array(ends)
        side = np.sign((ends.mean(0) - o) @ n_line)
        t = (ends - o) @ u
        p0, p1 = ends[t.argmin()], ends[t.argmax()]
        # Seen from the street (looking along -side*n_line), left is ... pick so that the right-hand normal faces out.
        nrm = -side * n_line
        a, c = (p0, p1) if np.cross(np.append(p1 - p0, 0), np.append(nrm, 0))[2] < 0 else (p1, p0)
        out[b["id"]] = {"a": a, "b": c, "normal": nrm, "h": float(b["h"]), "side": "NE" if side > 0 else "SW"}
    return out


SIDE_FACING = 0.7      # outward normal . (along Moss) for a side wall
SIDE_DEPTH = 12.0      # m of a side wall behind the front that the street sees
SIDE_MIN_H = 1.5       # m of exposed side wall worth reading
PARTY_GAP = 0.6        # m: a neighbour's wall this close and parallel hides this one (osmFacades PARTY_GAP)


def sides(data, ids, heights):
    """Side walls of the Moss buildings that show from the street: walls facing along Moss within SIDE_DEPTH of the
    front, from the top of the neighbour against them (or the ground) to the roofline. heights: {id: m} (the layout
    rooflines where known). [dict(key, bid, a, b, normal, z0, h)], a -> b left to right seen from outside."""
    o, u = moss_line(data)
    n_line = np.array([-u[1], u[0]])
    F = fronts(data, ids)
    rings = {b["id"]: np.asarray(b.get("ground") or b["pts"], float) for b in data["buildings"]}
    hs = {b["id"]: heights.get(b["id"], float(b["h"])) for b in data["buildings"]}
    out = []
    for bid, f in F.items():
        poly = rings[bid]
        ccw = _ccw(poly)
        fa = f["a"]
        for k in range(len(poly) - 1):
            a, c = poly[k], poly[k + 1]
            L = np.linalg.norm(c - a)
            if L < 1.0:
                continue
            e = (c - a) / L
            nrm = np.array([e[1], -e[0]]) if ccw else np.array([-e[1], e[0]])
            if abs(nrm @ u) < SIDE_FACING:
                continue
            # depth of each end behind the front line
            da, dc = (fa - a) @ f["normal"], (fa - c) @ f["normal"]
            if min(da, dc) > SIDE_DEPTH or max(da, dc) < -0.5:
                continue
            ta, tc = np.clip([da, dc], 0, SIDE_DEPTH)
            a2 = a + e * ((da - ta) / (da - dc) * L if da != dc else 0)
            c2 = a + e * ((da - tc) / (da - dc) * L if da != dc else L)
            if np.linalg.norm(c2 - a2) < 1.0:
                continue
            z0 = 0.0
            for ob in data["buildings"]:
                if ob["id"] == bid:
                    continue
                r = rings[ob["id"]]
                for p, q in zip(r[:-1], r[1:]):
                    el = np.linalg.norm(q - p)
                    if el < 0.3 or abs((q - p) @ e) / el < 0.9:
                        continue
                    if abs((p - a2) @ nrm) < PARTY_GAP and min(max((p - a2) @ e, (q - a2) @ e), np.linalg.norm(c2 - a2)) - \
                            max(min((p - a2) @ e, (q - a2) @ e), 0) > 0.5:
                        z0 = max(z0, hs[ob["id"]])
            if hs[bid] - z0 < SIDE_MIN_H:
                continue
            # a -> b left to right seen from outside: the right-hand normal faces out
            s, t = (a2, c2) if np.cross(np.append(c2 - a2, 0), np.append(nrm, 0))[2] < 0 else (c2, a2)
            out.append({"key": f"{bid}:s{k}", "bid": bid, "a": s, "b": t, "normal": nrm, "z0": z0, "h": hs[bid],
                        "side": f["side"]})
    return out


def csv_ids():
    import csv
    return [int(r["building_id"]) for r in csv.DictReader(CSV.open())]


# ---------------------------------------------------------------- rectify (cached Street View only)
def rectify(ids=None):
    import cv2
    sys.path.insert(0, str(ROOT)), sys.path.insert(0, str(ROOT / "scripts"))
    import streetview_facades as sv
    from kite.sources.streetview import StreetViewSource

    data = json.loads(MOSS.read_text())
    src = StreetViewSource(single=(37.7783, -122.4087, 0), cache_dir=sv.CACHE, key="cache-only")   # no key: a tile missing from the cache fails
    walls = sv.street_walls(data)
    panos = sv.load_panos(data, src)
    frames = {p["id"]: sv.pano_street(p, data) for p in panos}
    shots = sv.fetch_tiles(sv.plan_tiles(walls, panos, data), src, None)
    poses = json.loads((sv.CACHE.parent / "poses.json").read_text())
    occ = sv.occluders(data)
    (WORK / "rect").mkdir(parents=True, exist_ok=True)
    side_mode = "--sides" in sys.argv
    if side_mode:   # needs the fronts' layouts first: their rooflines are the buildings' heights
        lays = json.loads(OUT_JSON.read_text())["buildings"] if OUT_JSON.exists() else {}
        faces = sides(data, set(ids or csv_ids()), {int(k): float(v["roofline"]) for k, v in lays.items() if "roofline" in v})
    else:
        faces = [dict(f, key=str(bid), bid=bid, z0=0.0) for bid, f in fronts(data, set(ids or csv_ids())).items()]
    meta = {}
    pmap = {p["id"]: p for p in panos}
    for f in faces:
        bid = f["bid"]
        # occluders: everything but this building (its own bays sit in front of the merged plane)
        occ_b = [e for e in occ if e[0] != bid]
        d, agree = 0.0, 0.0
        if not side_mode:
            d, agree, curve = front_depth(sv, f, shots, pmap, frames, occ_b, poses)
            if "--depth-only" in sys.argv:
                print(bid, f"best {d:+.2f} ({agree:.3f})", " ".join(f"{x:+.2f}:{r:.2f}" for x, r in curve))
                continue
        top = f["h"] + (2.0 if side_mode else IMG_ABOVE)
        base = max(0.0, f["z0"] - 1.0)
        w = {"building": bid, "edge": "front", "a": f["a"] + d * f["normal"], "b": f["b"] + d * f["normal"],
             "z0": base, "z1": top, "normal": f["normal"], "street": "moss"}
        pid, pick = best_pano(sv, w, shots, frames, poses, occ_b)
        if pid is None:
            print(f"{f['key']}: no panorama sees it")
            continue
        # One panorama only: no seams or ghosts between panoramas. Trees and cars stay in the picture (painted-out
        # smears read worse than a tree the model can look past); only the Google logo rows are masked.
        one = [dict(s, occ=s["occ"] & logo_rows(s["occ"])) for s in shots if s["pano"]["id"] == pid]
        tex, clean, holes, used = sv.wall_texture(w, one, frames, poses, occ_b, ppm=PX_PER_M)
        L = float(np.linalg.norm(f["b"] - f["a"]))
        name = f["key"].replace(":", "_")
        cv2.imwrite(str(WORK / "rect" / f"{name}.jpg"), tex, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(WORK / "rect" / f"{name}_grid.jpg"), grid(tex, L, top, base), [cv2.IMWRITE_JPEG_QUALITY, 90])
        meta[f["key"]] = {"bid": bid, "a": f["a"].round(3).tolist(), "b": f["b"].round(3).tolist(),
                          "normal": f["normal"].round(4).tolist(), "width": round(L, 2), "h": f["h"], "z0": f["z0"],
                          "img_base": base, "img_top": top, "clean": round(clean, 3), "pano": pid,
                          "pano_score": round(pick, 3), "side": f["side"], "depth": d, "agree": round(agree, 3)}
        print(f"{f['key']} {f['side']}: {L:5.1f} x {f['h']:4.1f} m (from {f['z0']:.1f}), plane {d:+.2f} m "
              f"(agree {agree:.2f}), clean {clean:.0%}, panorama score {pick:.2f}")
    if "--depth-only" in sys.argv:
        return
    (WORK / ("sides.json" if side_mode else "fronts.json")).write_text(json.dumps(meta, indent=1))
    print(f"Street View requests: {src.requests}")


DEPTHS = np.arange(-0.5, 3.01, 0.25)   # m towards the street from the footprint's front line
IMG_ABOVE = 6.0                        # m drawn over the footprint height (LiDAR heights run 2-3 m under the photos)
DEPTH_MIN_AGREE = 0.25


def front_depth(sv, f, shots, pmap, frames, occ, poses):
    """Where the facade really is: the plane offset (towards the street) at which the panoramas' views of it agree
    best (high-pass NCC, as in streetview_facades' pose refinement). (best, score, curve)."""
    walls = [{"building": -1, "edge": "front", "a": f["a"] + d * f["normal"], "b": f["b"] + d * f["normal"],
              "z0": 0.0, "z1": f["h"], "normal": f["normal"], "street": "moss"} for d in DEPTHS]
    sc = sv.Scorer(walls, shots, pmap, frames, occ, 12, max_oblique=sv.REFINE_OBLIQUE)
    curve = []
    for i, d in enumerate(DEPTHS):
        pairs = sc.wall_pairs(i, poses)
        n = sum(p[3] for p in pairs)
        curve.append((float(d), sum(p[2] * p[3] for p in pairs) / n if n else -1.0))
    d, r = max(curve, key=lambda c: c[1])
    if r < DEPTH_MIN_AGREE:   # no clear answer (bays, trees): the footprint line
        d = 0.0
    return d, r, curve


def logo_rows(occ, rows=24):
    m = np.zeros_like(occ)
    m[-rows:] = True
    return m


def best_pano(sv, w, shots, frames, poses, occ):
    """The one panorama that sees the front best: clean coverage (its own tiles, cars and trees masked) times how
    square it looks at the front's centre. Its tiles share one camera centre, so they rectify without seams."""
    m = (w["a"] + w["b"]) / 2
    best = (None, -1.0)
    for pid in sorted({s["pano"]["id"] for s in shots}):
        own = [s for s in shots if s["pano"]["id"] == pid]
        C = sv.cam_centre(own[0]["pano"], poses.get(pid, sv.ZERO), frames[pid])
        d = m - C[:2]
        cos = float(-(d @ w["normal"]) / np.linalg.norm(d))
        if cos < 0.3:
            continue
        _, clean, _, _ = sv.wall_texture(w, own, frames, poses, occ, ppm=6)
        score = clean * (0.4 + 0.6 * cos)
        if score > best[1]:
            best = (pid, score)
    return best


def grid(tex, L, H, base=0.0):
    """A copy with a line every metre (labelled) so the model can read positions in metres. Rows: base..H."""
    import cv2
    g = tex.copy()
    h, w = g.shape[:2]
    for m in range(int(L) + 1):
        x = int(round(m / L * w))
        cv2.line(g, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(g, str(m), (x + 2, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for m in range(math.ceil(base), int(H) + 1):
        y = int(round(h - (m - base) / (H - base) * h))
        cv2.line(g, (0, y), (w, y), (255, 0, 255), 1)
        cv2.putText(g, str(m), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    return g


# ---------------------------------------------------------------- extract (Claude) and validate
FRONT_INTRO = "This is a rectified, straight-on photo of ONE building front on Moss St, San Francisco"
# OpenStreetMap merged a row of buildings into one footprint (260354450: 34 m, two apartment blocks): read them all,
# else the front came back with one block's openings and the sim drew the rest as a blank wall.
WIDE_INTRO = ("This is a rectified, straight-on photo of a {W:.0f} m stretch of street front on Moss St, San Francisco. "
              "OpenStreetMap has it as one building but it may be SEVERAL buildings side by side: treat them all as one "
              "front, list every opening across the whole width (not just the biggest building), and set x0/x1 to "
              "the ends of the whole row")
WIDE_IDS = {260354450}
SIDE_INTRO = ("This is the SIDE wall of one building on Moss St, San Francisco (the wall facing along the street, seen "
              "obliquely from the street and rectified straight-on). Below about {Z0:.1f} m it is behind the "
              "neighbouring building or hidden; describe only what is on this wall above that (windows, bay windows, "
              "light-well openings, a projecting upper floor as a bay). It is often a plain wall: then openings is []. "
              "Ignore anything that is clearly another wall seen through or past it")
PROMPT = """{INTRO}, rectified from a \
single Street View panorama. Trees, cars, poles and wires may stand in front of it: look past them. Smooth smeared \
areas (often at the left or right end, or at the top) were not seen by the camera. It is {W:.2f} m wide: x runs 0 to \
{W:.2f} m left to right, y runs {Y0:.1f} (image bottom; 0 is the sidewalk) to {H:.2f} m (the image top). The second image is the \
same photo with a labelled 1 m grid (yellow x, magenta y). Neighbouring buildings may show at the left or right end: \
describe only the front(s) named above.

Return ONLY a JSON object, no prose:
{{"x0": left edge of this building's front (0 if it runs into an unseen area at the left), "x1": right edge (likewise {W:.2f}),
 "roofline": y of the top of the front wall / parapet, "roof_seen": true if the roofline is in the picture,
 "floors": [{{"y0": .., "y1": ..}}, ...]   storeys bottom to top, floor line to floor line, the first from 0,
 "cornice": {{"y0": .., "y1": ..}} or null   the projecting cornice / eave band under the roofline,
 "belts": [y, ...]   horizontal trim bands across the front between floors (centre heights), [] if none,
 "openings": [{{"type": "window"|"bay"|"door"|"garage"|"storefront", "x0": .., "x1": .., "y0": .., "y1": ..,
   "shape": "rect"|"arch"|"segmental", "spring": y where a curved top starts (null for rect),
   "style": "rollup"|"panel"|"gate"|null, "panes": 1-8, "frame": "white"|"dark"|"wall"}}, ...]}}

Rules:
- Every window, door, garage door and shop window of this front, left to right, floor by floor. Coordinates are the \
  opening itself (the glass, door leaf or garage opening, not its trim), in metres, to 0.05 m. Read them off the grid.
- "bay": a projecting bay window (angled or square) as ONE entry from its underside (y0) to its top (y1), even when \
  it is stacked over two or three floors; x0..x1 its width where it meets the wall. Do NOT list the windows inside a \
  bay separately.
- "shape": "arch" = round-arched top (half circle), "segmental" = a shallow curved top; "spring" = the height where \
  the curve starts (the straight sides end). Else "rect". Arched garage openings, arched entries and segmental \
  industrial windows are common on Moss: look carefully at each top edge.
- "style" (garage only): "rollup" = horizontal slat shutter, "panel" = sectional or panelled door, "gate" = iron \
  bars/grille. For a door: "gate" if an iron security gate covers it, else null. Windows: null.
- "panes": glass panes across the width (a sash window 1, a pair 2, industrial grid windows the number of columns).
- "frame": "white" painted trim/casing, "dark" dark trim, "wall" no visible casing.
- No two openings may overlap; a door with a transom or fanlight is one opening. Sizes are real: doors about \
  0.8-1.3 m wide and 2.0-2.8 m tall (more with a fanlight), garage doors 2.2-5 m wide.
- If part of the front is hidden, still give your best reading of the visible pattern."""

LIMITS = {"window": (0.3, 4.5, 0.4, 4.5), "bay": (1.2, 5.5, 1.5, 14.0), "door": (0.6, 2.6, 1.6, 5.5),
          "garage": (1.8, 6.5, 1.8, 5.8), "storefront": (0.8, 12.0, 1.5, 5.0)}   # min/max width, min/max height


def validate(lay, W, H, side=False):
    """Problems with a layout (empty list = good). Fixes small things in place (sorts, clamps 5 cm, merges bays)."""
    errs = []
    try:
        x0, x1, roof = float(lay["x0"]), float(lay["x1"]), float(lay["roofline"])
        ops = lay["openings"]
        floors = lay.get("floors") or []
    except (KeyError, TypeError, ValueError) as e:
        return [f"missing or bad field: {e}"]
    if not (-0.1 <= x0 < x1 <= W + 0.1 and (side or x1 - x0 > 0.5 * W)):
        errs.append(f"x0/x1 {x0}..{x1} should cover most of 0..{W:.2f}")
    if not 2.5 < roof <= H + 0.1:
        errs.append(f"roofline {roof} outside 2.5..{H:.2f}")
    if (not floors and not side) or any(f["y1"] <= f["y0"] for f in floors):
        errs.append("floors must be bottom-to-top with y1 > y0")
    for o in ops:
        t = o.get("type")
        if t not in TYPES:
            errs.append(f"unknown type {t}")
            continue
        if o.get("shape", "rect") not in SHAPES:
            errs.append(f"bad shape {o.get('shape')}")
        sp = o.get("spring")
        if o.get("shape") in ("arch", "segmental") and not (isinstance(sp, (int, float)) and float(o["y0"]) < sp < float(o["y1"])):
            o["spring"] = None   # the renderer picks a rise from the width
        for k in ("x0", "x1", "y0", "y1"):
            o[k] = float(o[k])
        o["x0"], o["x1"] = max(o["x0"], x0) if o["x0"] > x0 - 0.06 else o["x0"], min(o["x1"], x1) if o["x1"] < x1 + 0.06 else o["x1"]
        o["y0"] = max(0.0, o["y0"]) if o["y0"] > -0.06 else o["y0"]
        w, h = o["x1"] - o["x0"], o["y1"] - o["y0"]
        wl, wh, hl, hh = LIMITS[t]
        if not (wl <= w <= wh and hl <= h <= hh):
            errs.append(f"{t} at x {o['x0']:.2f}-{o['x1']:.2f} y {o['y0']:.2f}-{o['y1']:.2f}: size {w:.2f} x {h:.2f} m not plausible")
        if o["x0"] < x0 or o["x1"] > x1 or o["y0"] < 0 or o["y1"] > roof + 0.05:
            errs.append(f"{t} at x {o['x0']:.2f}-{o['x1']:.2f} y {o['y0']:.2f}-{o['y1']:.2f} is outside the front "
                        f"(x {x0:.2f}-{x1:.2f}, y 0-{roof:.2f})")
    # A bay stacked over floors sometimes comes back as one bay per floor: one bay, underside to top.
    bays = sorted((o for o in ops if o.get("type") == "bay"), key=lambda o: o["y0"])
    for i, a in enumerate(bays):
        for b in bays[i + 1:]:
            if b.get("merged") or a.get("merged"):
                continue
            ov = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            if ov > 0.7 * min(a["x1"] - a["x0"], b["x1"] - b["x0"]) and b["y0"] - a["y1"] < 0.8:
                a.update(x0=min(a["x0"], b["x0"]), x1=max(a["x1"], b["x1"]), y1=max(a["y1"], b["y1"]))
                b["merged"] = True
    ops[:] = [o for o in ops if not o.get("merged")]
    ops.sort(key=lambda o: (o["y0"], o["x0"]))
    for i, a in enumerate(ops):
        for b in ops[i + 1:]:
            if min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]) > GAP and min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]) > GAP:
                errs.append(f"{a['type']} x {a['x0']:.2f}-{a['x1']:.2f} y {a['y0']:.2f}-{a['y1']:.2f} overlaps "
                            f"{b['type']} x {b['x0']:.2f}-{b['x1']:.2f} y {b['y0']:.2f}-{b['y1']:.2f}")
    return errs


AT_GRADE = ("garage", "door")


def ground(lay):
    """Puts the sidewalk at y 0. The rectified photos' y 0 is the panorama's ground plane, which sits 0.1-2 m under
    the real sidewalk along Moss (it slopes; Street View camera heights vary), so Claude's openings come back lifted
    by that much and every door got a staircase in the sim. Garage doors and entries meet the sidewalk, so the
    lowest of them is the datum: everything moves down by it, and doors and garages start at 0 (a raised entry is
    an opening down to the sidewalk with the steps behind it). In place; returns the datum removed (also kept,
    cumulative, in lay["datum"]). Idempotent."""
    ops = lay.get("openings") or []
    low = [o["y0"] for o in ops if o["type"] in AT_GRADE]
    d = min(low) if low else 0.0
    if d > 0.0:
        for o in ops:
            o["y0"], o["y1"] = max(0.0, o["y0"] - d), o["y1"] - d
            if o.get("spring") is not None:
                o["spring"] = o["spring"] - d
        for f in lay.get("floors") or []:
            f["y0"], f["y1"] = max(0.0, f["y0"] - d), f["y1"] - d
        if lay.get("cornice"):
            lay["cornice"] = {k: v - d for k, v in lay["cornice"].items()}
        lay["belts"] = [y - d for y in lay.get("belts") or [] if y - d > 0.5]
        lay["roofline"] = lay["roofline"] - d
        lay["datum"] = round(lay.get("datum", 0.0) + d, 3)
    for o in ops:
        if o["type"] in AT_GRADE:
            o["y0"] = 0.0
    return d


def load_key():
    for line in (MAIN / ".env").read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*['\"]?([^'\"\s]+)", line)
        if m:
            return m.group(1)
    return os.environ.get("ANTHROPIC_API_KEY")


def ask(client, name, W, H, Y0, intro, feedback=None, prev=None):
    img = lambda p: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",   # noqa: E731
                                                 "data": base64.b64encode(p.read_bytes()).decode()}}
    content = [img(WORK / "rect" / f"{name}.jpg"), img(WORK / "rect" / f"{name}_grid.jpg"),
               {"type": "text", "text": PROMPT.format(W=W, H=H, Y0=Y0, INTRO=intro)}]
    msgs = [{"role": "user", "content": content}]
    if feedback:
        msgs += [{"role": "assistant", "content": prev},
                 {"role": "user", "content": "That layout has problems:\n- " + "\n- ".join(feedback) +
                  "\nLook again and return the corrected JSON only."}]
    r = client.messages.create(model=MODEL, max_tokens=6000, messages=msgs)
    return "".join(b.text for b in r.content if b.type == "text"), r.usage


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def extract(ids=None, side=False):
    import anthropic
    meta = json.loads((WORK / ("sides.json" if side else "fronts.json")).read_text())
    client = anthropic.Anthropic(api_key=load_key())
    out = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    out.setdefault("buildings", {})
    out.setdefault("sides", {})
    group = out["sides"] if side else out["buildings"]
    (WORK / "claude").mkdir(parents=True, exist_ok=True)
    calls = 0
    for key in sorted(meta):
        f = meta[key]
        if ids and f.get("bid", int(key.split(":")[0])) not in ids:
            continue
        name = key.replace(":", "_")
        if f["clean"] < (0.5 if side else 0.3):   # a side wall needs a clear view: most are seen at a slant
            print(f"{key}: barely seen ({f['clean']:.0%}), left procedural")
            continue
        W, H, Y0 = f["width"], f["img_top"], f.get("img_base", 0.0)
        intro = SIDE_INTRO.format(Z0=f["z0"]) if side else WIDE_INTRO.format(W=W) if int(key) in WIDE_IDS else FRONT_INTRO
        text, usage = ask(client, name, W, H, Y0, intro)
        calls += 1
        lay, errs = None, ["no JSON object in the answer"]
        try:
            lay = parse(text)
            errs = validate(lay, W, H, side) if lay else errs
        except (json.JSONDecodeError, ValueError) as e:
            errs = [f"invalid JSON: {e}"]
        if errs:
            print(f"{key}: retry ({len(errs)} problems: {errs[0]})")
            text, usage = ask(client, name, W, H, Y0, intro, errs, text)
            calls += 1
            try:
                lay = parse(text)
                errs = validate(lay, W, H, side) if lay else ["no JSON"]
            except (json.JSONDecodeError, ValueError) as e:
                errs = [f"invalid JSON: {e}"]
        (WORK / "claude" / f"{name}.txt").write_text(text)
        if errs:
            print(f"{key}: still invalid, left procedural: {errs[:3]}")
            group.pop(key, None)
            continue
        if side:   # only what shows over the neighbour
            lay["openings"] = [o for o in lay["openings"] if o["y0"] >= f["z0"] - 0.2]
            if not lay["openings"]:
                group.pop(key, None)
                print(f"{key}: plain")
                continue
        ground(lay)
        lay.update(width=W, img_top=H, a=f["a"], b=f["b"], depth=f["depth"], z0=f.get("z0", 0.0))
        group[key] = lay
        kinds = {}
        for o in lay["openings"]:
            kinds[o["type"]] = kinds.get(o["type"], 0) + 1
        print(f"{key}: ok, roofline {lay['roofline']} m, {len(lay.get('floors') or [])} floors, {kinds}, "
              f"sidewalk {lay.get('datum', 0.0):.2f} m up the photo")
        draw_overlay(name, lay, W, H, Y0)
    out["note"] = ("Moss St facade layouts read from Street View (scripts/facade_layouts.py; local only, never commit). "
                   "Metres along the front from its left end as seen from the street (a -> b, moss.json frame), "
                   "y up from the sidewalk.")
    out["model"] = MODEL
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"{len(out['buildings'])} layouts in {OUT_JSON}; {calls} Claude calls this run")


REVIEW = """Here is a building front photo (first image) and the same photo with a layout drawn over it (second \
image: windows cyan, bays red, doors green, garages magenta, shop windows yellow; white lines the roofline, belts \
and the building's left/right edges). The layout as JSON:
{layout}
The image spans x 0 to {W:.2f} m and y {Y0:.1f} to {H:.2f} m. Compare the boxes with the photo carefully: missing \
or extra openings, boxes in the wrong place or of the wrong size, bays that should be one stacked bay or two, wrong \
shapes (arched/segmental tops), wrong garage styles. Return ONLY the corrected JSON object in the same format \
(return it unchanged if it is right). Coordinates are the openings themselves, in metres."""


def review(ids=None):
    """One more look per front: Claude sees its layout drawn over the photo and corrects it (1 call each)."""
    import anthropic
    meta = json.loads((WORK / "fronts.json").read_text())
    client = anthropic.Anthropic(api_key=load_key())
    out = json.loads(OUT_JSON.read_text())
    img = lambda p: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",   # noqa: E731
                                                 "data": base64.b64encode(p.read_bytes()).decode()}}
    calls = 0
    for key, lay in sorted(out["buildings"].items()):
        if ids and int(key) not in ids:
            continue
        f = meta[key]
        W, H = f["width"], f["img_top"]
        keep = {k: lay[k] for k in ("width", "img_top", "a", "b", "depth", "z0", "datum") if k in lay}
        shown = {k: v for k, v in lay.items() if k not in keep}
        dt = lay.get("datum", 0.0)   # the layout is grounded: the photo spans y -datum .. H - datum in its frame
        text = REVIEW.format(layout=json.dumps(shown), W=W, H=H - dt, Y0=-dt)
        r = client.messages.create(model=MODEL, max_tokens=6000, messages=[{"role": "user", "content": [
            img(WORK / "rect" / f"{key}.jpg"), img(WORK / "overlay" / f"{key}.jpg"), {"type": "text", "text": text}]}])
        calls += 1
        answer = "".join(b.text for b in r.content if b.type == "text")
        (WORK / "claude" / f"{key}_review.txt").write_text(answer)
        try:
            new = parse(answer)
            errs = validate(new, W, H) if new else ["no JSON"]
        except (json.JSONDecodeError, ValueError) as e:
            errs = [f"invalid JSON: {e}"]
        if errs:
            print(f"{key}: review invalid, kept the first reading ({errs[0]})")
            continue
        new.update(keep)
        ground(new)
        out["buildings"][key] = new
        draw_overlay(key, new, W, H)
        print(f"{key}: reviewed, {len(lay['openings'])} -> {len(new['openings'])} openings")
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"{calls} Claude calls this run")


def draw_overlay(bid, lay, W, H, Y0=0.0):
    import cv2
    img = cv2.imread(str(WORK / "rect" / f"{bid}.jpg"))
    h, w = img.shape[:2]
    Y0 -= lay.get("datum", 0.0)   # the layout's sidewalk sits that far up the photo (ground)
    H -= lay.get("datum", 0.0)
    px = lambda x, y: (int(round(x / W * w)), int(round(h - (y - Y0) / (H - Y0) * h)))   # noqa: E731
    col = {"window": (255, 200, 0), "bay": (0, 0, 255), "door": (0, 255, 0), "garage": (255, 0, 255), "storefront": (0, 255, 255)}
    for o in lay["openings"]:
        cv2.rectangle(img, px(o["x0"], o["y1"]), px(o["x1"], o["y0"]), col[o["type"]], 2)
        cv2.putText(img, f"{o['type'][0]}{'A' if o.get('shape') == 'arch' else 'S' if o.get('shape') == 'segmental' else ''}",
                    px(o["x0"] + 0.05, o["y1"] - 0.3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col[o["type"]], 1)
    for y in [lay["roofline"]] + list(lay.get("belts") or []):
        cv2.line(img, px(lay["x0"], y), px(lay["x1"], y), (255, 255, 255), 1)
    for x in (lay["x0"], lay["x1"]):
        cv2.line(img, px(x, 0), px(x, H), (255, 255, 255), 1)
    (WORK / "overlay").mkdir(exist_ok=True)
    cv2.imwrite(str(WORK / "overlay" / f"{bid}.jpg"), img)


# ---------------------------------------------------------------- compare with the sim
CMP_D = 8.0             # m: camera distance from the front (Moss is about 12 m wall to wall, so stay on this side)
CMP_SEG = 6.0           # m: widest strip one camera films; wider fronts are filmed in strips and stitched
CMP_PPM = 60


def sim_poses(f, H):
    """Cameras square to the front, one per strip, each framing exactly its strip (0..H high)."""
    a, b, nrm = np.asarray(f["a"]), np.asarray(f["b"]), np.asarray(f["normal"])
    L = float(np.linalg.norm(b - a))
    u = (b - a) / L
    n = max(1, math.ceil(L / CMP_SEG))
    out = []
    for k in range(n):
        m = a + u * (L * (k + 0.5) / n)
        c = m + nrm * CMP_D
        out.append({"pos": [float(c[0]), float(c[1]), H / 2], "look": [float(m[0]), float(m[1]), H / 2],
                    "fov": math.degrees(2 * math.atan(H / 2 / CMP_D)),
                    "w": round(L / n * CMP_PPM), "h": round(H * CMP_PPM)})
    return out


def compare(ids=None, url=None, port=5191, tag=""):
    """Sim render of each front (square camera, cropped to the front) under its rectified Street View image."""
    import cv2
    from playwright.sync_api import sync_playwright
    meta = {int(k): v for k, v in json.loads((WORK / "fronts.json").read_text()).items()}
    url = url or f"http://localhost:{port}/facadecam.html?look=2"
    CMP.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        br = p.chromium.launch(args=["--use-angle=metal", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"])
        page = br.new_page(viewport={"width": 1280, "height": 960})
        page.on("console", lambda m: m.type == "error" and print("  [page]", m.text))
        page.goto(url)
        page.wait_for_function("window.kiteCam !== undefined", timeout=60000)
        page.evaluate("kiteCam.ready")
        for bid in ids or sorted(meta):
            f = meta.get(bid)
            if f is None:
                continue
            H = f["img_top"]
            strips = []
            for pose in sim_poses(f, H):
                page.evaluate("p => kiteCam.shot(p)", pose)   # first frame warms up shaders and AO
                png = page.evaluate("p => kiteCam.shot(p)", pose)
                strips.append(cv2.imdecode(np.frombuffer(base64.b64decode(png.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR))
            crop = np.hstack(strips)
            sv = cv2.imread(str(WORK / "rect" / f"{bid}.jpg"))
            crop = cv2.resize(crop, (sv.shape[1], sv.shape[0]), interpolation=cv2.INTER_AREA)
            sep = np.full((8, sv.shape[1], 3), 255, np.uint8)
            both = np.vstack([sv, sep, crop])
            cv2.putText(both, f"{bid} Street View (top) | sim look=2 (bottom)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(CMP / f"{bid}{tag}.jpg"), both, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"  {CMP / f'{bid}{tag}.jpg'}")
        br.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rectify"
    args = [int(a) for a in sys.argv[2:] if a.isdigit()]
    opt = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d   # noqa: E731
    if cmd == "rectify":
        rectify(args or None)
    elif cmd == "--selftest":
        win = lambda x0, x1, y0, y1, t="window": {"type": t, "x0": x0, "x1": x1, "y0": y0, "y1": y1}   # noqa: E731
        lay = {"x0": 0, "x1": 7.5, "roofline": 10, "floors": [{"y0": 0, "y1": 3}],
               "openings": [win(1, 2, 4, 5.8), win(1.5, 2.5, 5, 6.5), win(0.5, 3.5, 3.5, 6.5, "bay"), win(0.5, 3.5, 6.8, 9.5, "bay")]}
        errs = validate(lay, 7.5, 12)
        assert sum("overlaps" in e for e in errs) >= 2, errs                           # windows overlap each other and the bay
        assert sum(o["type"] == "bay" for o in lay["openings"]) == 1, lay["openings"]   # stacked bays merged
        assert validate({"x0": 0, "x1": 7.5, "roofline": 10, "floors": [{"y0": 0, "y1": 3}],
                         "openings": [win(1, 3.5, 0, 2.4, "garage"), win(4.5, 5.5, 0.5, 2.6, "door")]}, 7.5, 12) == []
        assert validate({"x0": 0, "x1": 7.5, "roofline": 10, "floors": [{"y0": 0, "y1": 3}],
                         "openings": [win(1, 1.3, 0, 0.5, "door")]}, 7.5, 12)              # a 30 x 50 cm door isn't
        g = {"roofline": 10.0, "floors": [{"y0": 0.0, "y1": 3.7}], "cornice": {"y0": 9.5, "y1": 10.0}, "belts": [3.7],
             "openings": [win(1, 3.5, 0.7, 3.1, "garage"), win(4.5, 5.5, 1.6, 4.0, "door"), win(1, 2, 4.7, 6.2)]}
        assert abs(ground(g) - 0.7) < 1e-9 and abs(g["roofline"] - 9.3) < 1e-9 and g["datum"] == 0.7   # garage meets the sidewalk
        assert [o["y0"] for o in g["openings"]] == [0.0, 0.0, 4.0] and abs(g["openings"][1]["y1"] - 3.3) < 1e-9   # door: no stoop
        assert ground(g) == 0.0 and g["datum"] == 0.7                                                 # idempotent
        print("selftest ok")
    elif cmd == "extract":
        extract(args or None, side="--sides" in sys.argv)
    elif cmd == "review":
        review(args or None)
    elif cmd == "ground":   # re-ground the layouts already in OUT_JSON (no Claude calls)
        out = json.loads(OUT_JSON.read_text())
        for group in ("buildings", "sides"):
            for key, lay in out.get(group, {}).items():
                d = ground(lay)
                print(f"{key}: sidewalk {d:+.2f} m (total {lay.get('datum', 0.0):.2f}), roofline {lay['roofline']:.2f}")
        OUT_JSON.write_text(json.dumps(out, indent=1))
    elif cmd == "compare":
        compare(args or None, url=opt("--url"), tag=opt("--tag", ""))
    elif cmd == "fronts":
        for k, v in fronts(json.loads(MOSS.read_text()), set(args or csv_ids())).items():
            print(k, v["side"], v["a"].round(2), v["b"].round(2), round(float(np.linalg.norm(v["b"] - v["a"])), 2))
