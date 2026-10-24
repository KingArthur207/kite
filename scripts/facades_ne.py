"""Verified Street View facades for the NE side of Moss St, Howard to the route start (permission:
docs/google-permission.md; derived pixels stay in sim/public/local/, never committed).

    uv run --no-project --with numpy --with opencv-python-headless --with ultralytics \
        python scripts/facades_ne.py [--out DIR] [--weights weights/yolo11n.pt] [--env .env]
    uv run --no-project --with numpy --with opencv-python-headless python scripts/facades_ne.py --selftest

Writes <out>/<building>_<edge>.jpg and <out>/manifest.json (conventions inside it). Raw API responses are
cached in /tmp/kite_sv_ne so a rerun is free (Google's terms: temp only, delete it when done).

1. Walls: every footprint edge (moss.json order: edge k = pts[k] -> pts[k+1]) of the NE buildings below whose
   outward normal faces Moss (bay sides included). The corner building's Howard face is someone else's.
2. Google panoramas along the Moss roadway (metadata, free), every ~10 m.
3. Views (Static API, billed): per building front (the 30 m corner building in 4 parts) the most frontal pano
   and the nearest ~35 degrees off square either side, aimed at the front's centre, fov 60, one or two pitches
   from the ground to the roofline. YOLO person/vehicle boxes on every view.
4. Camera calibration (calibrate()). The metadata poses don't agree with each other or with OSM: as given,
   neighbouring panos put brick #29 and blue #33 on the same wall. Each pano's views are unrolled onto one
   plane along the NE fronts and neighbouring strips cross-correlated along the street (blurred colour). One
   cross-street offset for all cameras (+1.5 m NE: the real fronts are 4-4.5 m from the pano line, OSM says
   5-6.6 m) removes most of it; a per-pano along-street term the rest. The absolute along-street offset is
   ALONG_ANCHOR_M, set by eye from the written textures (+2 m). Per front, cameras also
   move by the front's OSM depth minus the mean (the photos show one facade line, OSM scatters it by 1.5 m).
5. Texture per wall (100 px/m): every view warped onto the wall plane (homography, camera 2.5 m up),
   weighted (frontal cos / distance)^2 x border falloff, ^WEIGHT_POW; YOLO boxes get almost no weight so another
   view fills them; columns hidden by another footprint, texels over 20 m away or too grazing are dropped.
6. Verify per front, leaving one pano out: the pano that sees most of the front, other than its main
   contributor, is left out of a second blend, which is drawn into that pano's best view and compared
   (blurred-colour NCC, YOLO boxes left out) with the front at its place and moved -3..3 m along Moss over the
   same pixels. Verified: it matches its place better than anywhere CONTROL_M+ away, best within
   MAX_BEST_OFFSET_M of the place, and score >= MIN_SCORE (a floor for real similarity). Walls of a
   verified front get the all-pano texture if street-facing (bay sides pick up neighbours) and mostly seen.
   ponytail: edge-level (pixel) agreement isn't reachable with flat walls and 10 m pano spacing, so this
   checks placement to about a metre, not detail. Leakage: the held-out pano helped calibrate its own pose.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from facade_rectify import Camera, project  # noqa: E402

MOSS = ROOT / "sim/public/osm/moss.json"
OUT = ROOT / "sim/public/local/facades_ne"
WEIGHTS = ROOT / "weights/yolo11n.pt"
CACHE = Path("/tmp/kite_sv_ne")
MOSS_ROAD = 8916291
# NE side, Howard corner to just past the route start (spawn is ~70 m along Moss from the zebra).
BUILDINGS = (260351187, 260351183, 260351182, 260351170, 260351180, 260351184, 260351195)
CORNER_SPLIT = {260351187: 4}   # long walls split into this many view groups
PX_PER_M = 100
SIZE = 640
FOV = 60
CAM_Z = 2.5                     # kite.sources.streetview.SV_CAM_HEIGHT_M
OBLIQUE_DEG = 35
MIN_EDGE_M = 0.5
MIN_FACING = 0.3                # outward normal . (towards Moss)
WEIGHT_POW = 3.0
MASKED_W = 1e-4                 # weight left inside a YOLO box: used only if no other view sees the texel
MAX_DIST, MIN_COS = 20.0, 0.35  # texels further or more grazing than this aren't taken from a view
STRIP_PPM, MAX_SHIFT, MIN_PAIR_NCC = 20, 8.0, 0.4   # camera calibration: strip px/m, search +-m, pair peak to use
PAIR_MAX_M = 3.0               # pair disagreements past this are false matches (cost capped, left out of the solve)
CROSS_SEARCH = np.arange(-1.0, 3.01, 0.25)           # global cross-street camera offsets tried, m (NE +)
MIN_SCORE = 0.2                # held-out blurred-colour NCC floor at the true position
MIN_SCORE_PX = 400              # fewer comparable pixels than this = unverifiable
MIN_COVER = 0.8                # fraction of a wall seen by a kept view, else no texture
MIN_FACING_TEX = 0.9           # bay sides (about 0.5) are seen at grazing angles and pick up neighbours: no texture
COLOR_SIGMA_PX = 10            # blur before the colour NCC (about 10 cm at 5 m)
OFFSETS_M = np.arange(-3.0, 3.01, 0.5)   # front moved along Moss for the localisation test
MAX_BEST_OFFSET_M = 1.0        # the held-out view must match best within this of the true position
CONTROL_M = 1.5                # control_score: best match this far or further from the true position
MIN_HELD_SEEN = 0.15           # held-out pano: its best view must see this much of the front
# Pairwise alignment fixes panos relative to each other only; this moves them all along Moss (SE +) onto OSM.
# Set by eye from the written textures (anchor 0, +1, +2 m): at +2 #29 is all brick, #19 all light blue, #33 its
# garage; at 0 every small front shows ~2 m of its SE neighbour.
ALONG_ANCHOR_M = 2.0
YOLO_CLASSES = (0, 1, 2, 3, 5, 7)   # person, bicycle, car, motorcycle, bus, truck
CONVENTION = (
    "wall_index k is the footprint edge from moss.json buildings[].pts[k] to pts[k+1] (that file's order). "
    "corners_local_m are [x east, y north, z up] metres in moss.json's frame, as seen from the street: "
    "bottom-left, bottom-right, top-right, top-left (counter-clockwise, so the right-hand normal faces out). "
    "Texture: column 0 = left edge, row 0 = top (roofline at the wall's moss.json height h); in three.js UVs "
    "(flipY) BL=(0,0) BR=(1,0) TR=(1,1) TL=(0,1). texture null = rejected (score under min_score or "
    "unverifiable): keep the procedural facade there.")


def compass(dx, dy):
    return math.degrees(math.atan2(dx, dy)) % 360


def ne_walls(data):
    """[{bid, k, L (left ground pt), R, h, nrm, along, facing}] facing Moss for BUILDINGS."""
    road = next(r for r in data["roads"] if r["id"] == MOSS_ROAD)["pts"]
    u = np.subtract(road[2], road[1]); u /= np.linalg.norm(u)
    to_moss = -np.array([-u[1], u[0]])         # NE side looks SW, towards the roadway
    walls = []
    for b in data["buildings"]:
        if b["id"] not in BUILDINGS:
            continue
        pts = np.array(b["pts"], float)
        area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
        for k in range(len(pts) - 1):
            a, c = pts[k], pts[k + 1]
            Lm = np.linalg.norm(c - a)
            e = (c - a) / max(Lm, 1e-9)
            nrm = np.array([e[1], -e[0]]) if area > 0 else np.array([-e[1], e[0]])   # outward, as osmFacades.ts
            off = ((a + c) / 2) @ -to_moss
            if Lm < MIN_EDGE_M or nrm @ to_moss < MIN_FACING or off > 12:
                continue
            left = np.array([nrm[1], -nrm[0]])      # the viewer's left, standing outside facing the wall
            Lp, Rp = (c, a) if (c - a) @ left > 0 else (a, c)
            walls.append({"bid": b["id"], "k": k, "L": Lp, "R": Rp, "h": float(b["h"]), "nrm": nrm,
                          "along": float(((a + c) / 2) @ u), "facing": float(nrm @ to_moss)})
    return walls, u


def group_targets(walls, data):
    """View targets: (building id, part) -> (centre ground point, outward normal, height, half-width). One per
    building front; a front of one long edge (the corner building) is split into CORNER_SPLIT parts."""
    out = {}
    for bid in BUILDINGS:
        ws = [w for w in walls if w["bid"] == bid]
        if not ws:
            continue
        nrm = np.mean([w["nrm"] * np.linalg.norm(w["R"] - w["L"]) for w in ws], 0)
        nrm, h = nrm / np.linalg.norm(nrm), max(w["h"] for w in ws)
        P = np.array([p for w in ws for p in (w["L"], w["R"])])
        if len(ws) > 1:
            out[(bid, 0)] = (P.mean(0), nrm, h, np.linalg.norm(P.max(0) - P.min(0)) / 2)
            continue
        a, b, parts = ws[0]["L"], ws[0]["R"], CORNER_SPLIT.get(bid, 1)
        for k in range(parts):
            out[(bid, k)] = (a + (k + 0.5) / parts * (b - a), nrm, h, np.linalg.norm(b - a) / parts / 2)
    return out




def find_panos(src, data, u):
    from kite.gps import latlng_to_sim, sim_to_latlng
    from kite.sources.streetview import _is_google
    panos = {}
    for s in np.arange(-6, 92, 3.0):
        m = src._metadata(*sim_to_latlng(*(s * u)))   # raises outside SCOPE
        if m and _is_google(m):
            x, y = latlng_to_sim(m["location"]["lat"], m["location"]["lng"])
            panos[m["pano_id"]] = {"id": m["pano_id"], "x": x, "y": y, "date": m.get("date")}
    return list(panos.values())


def pitches(dist, h):
    """Pitches whose fov-60 square views cover the wall from the ground to h from CAM_Z, dist away."""
    lo, hi = math.degrees(math.atan2(-CAM_Z, dist)), math.degrees(math.atan2(h - CAM_Z, dist))
    half = FOV / 2 - 3
    if hi - lo <= 2 * half:
        return [round((lo + hi) / 2)]
    return [round(lo + half), round(min(hi - half, 60))]


def plan_views(targets, panos):
    """[(pano, heading, pitch)]: per group the most frontal pano and the nearest ~OBLIQUE_DEG either side."""
    views = []
    for g, (ctr, nrm, h, half) in targets.items():
        cand = []
        for p in panos:
            d = np.array([p["x"], p["y"]]) - ctr
            dist = np.linalg.norm(d)
            if d @ nrm < 2.0 or dist > 25:
                continue
            side = float(nrm[0] * d[1] - nrm[1] * d[0]) / dist
            cand.append((math.degrees(math.asin(np.clip(side, -1, 1))), dist, p))
        if not cand:
            continue
        picks = [min(cand, key=lambda c: abs(c[0]))]
        for sgn in (1, -1):
            sided = [c for c in cand if sgn * c[0] > 15]
            if sided:
                picks.append(min(sided, key=lambda c: abs(abs(c[0]) - OBLIQUE_DEG)))
        for ang, dist, p in picks:
            hd = round(compass(*(ctr - np.array([p["x"], p["y"]]))), 0)
            for pt in pitches(dist * math.cos(math.radians(ang)), h):
                views.append((p, hd, pt))
    seen, out = set(), []
    for p, hd, pt in views:
        key = (p["id"], hd, pt)
        if not any(k[0] == p["id"] and abs((k[1] - hd + 180) % 360 - 180) < 8 and abs(k[2] - pt) < 8 for k in seen):
            seen.add(key)
            out.append((p, hd, pt))
    return out


def yolo_mask(model, img):
    """1 inside (slightly grown) person/vehicle boxes."""
    m = np.zeros(img.shape[:2], np.float32)
    if model is None:
        return m
    r = model.predict(img, conf=0.25, classes=list(YOLO_CLASSES), verbose=False)[0]
    for x0, y0, x1, y1 in r.boxes.xyxy.cpu().numpy():
        gx, gy = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
        m[max(0, int(y0 - gy)):int(y1 + gy) + 1, max(0, int(x0 - gx)):int(x1 + gx) + 1] = 1
    return m


def blocked_cols(c, ground, edges, skip):
    """Per texture column ground point (N, 2): does the ray from camera c to it properly cross a footprint edge
    other than `skip`? (Logic from an earlier scripts/streetview_facades.py.)"""
    ids, P, Q = edges
    keep = np.array([i != skip for i in ids])
    P, Q = P[keep], Q[keep]
    tips = ground - 0.15 * (ground - c) / np.linalg.norm(ground - c, axis=1)[:, None]

    def orient(o, s, t):   # broadcasting (N, E)
        return (s[..., 0] - o[..., 0]) * (t[..., 1] - o[..., 1]) - (s[..., 1] - o[..., 1]) * (t[..., 0] - o[..., 0])
    C = np.broadcast_to(c, tips.shape)[:, None, :]
    T = tips[:, None, :]
    d1, d2 = orient(P[None], Q[None], C), orient(P[None], Q[None], T)
    d3, d4 = orient(C, T, P[None]), orient(C, T, Q[None])
    return ((d1 * d2 < 0) & (d3 * d4 < 0)).any(axis=1)


def wall_grid(w):
    Lm = float(np.linalg.norm(w["R"] - w["L"]))
    ppm = w.get("ppm", PX_PER_M)
    return Lm, max(2, round(Lm * ppm)), max(2, round(w["h"] * ppm))


def warp_view(w, cam, img, ok_img, edges=None):
    """Image -> wall texture. Returns (tex float32 (th,tw,3), weight (th,tw)) or None if the wall isn't in view.
    ok_img: per-pixel weight in the image (border falloff x YOLO), 0 outside."""
    Lm, tw, th = wall_grid(w)
    L, R, h, nrm = w["L"], w["R"], w["h"], w["nrm"]
    c = np.array([cam.x, cam.y])
    if (c - L) @ nrm < 0.5:
        return None
    s = (np.arange(tw) + 0.5) / tw
    ground = L[None] + s[:, None] * (R - L)[None]                         # (tw, 2)
    fwd = cam.axes()[0]
    depth = (ground - c) @ fwd[:2]                                        # depth of a column, at camera height
    front = depth - abs(fwd[2]) * max(h, CAM_Z) > 0.3                     # conservative for a pitched camera
    if not front.any():
        return None
    j0, j1 = np.flatnonzero(front)[[0, -1]]
    corners = np.array([[*ground[j0], h], [*ground[j1], h], [*ground[j1], 0.0], [*ground[j0], 0.0]])
    dst = project(cam, corners)
    if np.isnan(dst).any():
        return None
    src = np.array([[j0 + 0.5, 0], [j1 + 0.5, 0], [j1 + 0.5, th], [j0 + 0.5, th]], np.float32) - 0.5
    H = cv2.getPerspectiveTransform(src, (dst - 0.5).astype(np.float32))
    flags, border = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, cv2.BORDER_CONSTANT
    tex = cv2.warpPerspective(img.astype(np.float32), H, (tw, th), flags=flags, borderMode=border)
    wimg = cv2.warpPerspective(ok_img, H, (tw, th), flags=flags, borderMode=border)
    col_ok = front.copy()
    if edges is not None:
        col_ok &= ~blocked_cols(c, ground, edges, (w["bid"], w["k"]))
    z = h * (1 - (np.arange(th) + 0.5) / th)
    hd = np.linalg.norm(ground - c, axis=1)
    dist = np.sqrt(hd[None] ** 2 + (z[:, None] - CAM_Z) ** 2)
    cos = np.clip(((c - ground) @ nrm)[None] / dist, 0, 1)
    wgt = ((cos / dist) ** 2 * wimg) ** WEIGHT_POW * col_ok[None] * ((dist < MAX_DIST) & (cos > MIN_COS))
    return tex, wgt.astype(np.float32)


def render_into(w, cam, tex, valid, edges=None):
    """Texture -> camera image (the inverse of warp_view). Returns (render (H,W,3), mask (H,W) bool)."""
    ones = np.ones(cam.height * cam.width, np.float32).reshape(cam.height, cam.width)
    r = warp_view(w, cam, ones[..., None].repeat(3, -1), ones, edges)   # just for the visible-column test
    if r is None:
        return None, None
    Lm, tw, th = wall_grid(w)
    vis = (r[1] > 0) & valid
    c = np.array([cam.x, cam.y])
    s = (np.arange(tw) + 0.5) / tw
    ground = w["L"][None] + s[:, None] * (w["R"] - w["L"])[None]
    fwd = cam.axes()[0]
    front = ((ground - c) @ fwd[:2] - abs(fwd[2]) * max(w["h"], CAM_Z)) > 0.3
    j0, j1 = np.flatnonzero(front)[[0, -1]]
    corners = np.array([[*ground[j0], w["h"]], [*ground[j1], w["h"]], [*ground[j1], 0.0], [*ground[j0], 0.0]])
    dst = (project(cam, corners) - 0.5).astype(np.float32)
    src = np.array([[j0 + 0.5, 0], [j1 + 0.5, 0], [j1 + 0.5, th], [j0 + 0.5, th]], np.float32) - 0.5
    H = cv2.getPerspectiveTransform(src, dst)
    size = (cam.width, cam.height)
    out = cv2.warpPerspective(tex.astype(np.float32), H, size, flags=cv2.INTER_LINEAR)
    m = cv2.warpPerspective(vis.astype(np.float32), H, size, flags=cv2.INTER_NEAREST) > 0.5
    return out, m


def edge_map(img):
    g = cv2.GaussianBlur(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32),
                         (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def ncc(a, b, m):
    if m.sum() < MIN_SCORE_PX:
        return None
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / d) if d > 0 else None



def border_falloff(hgt, wid, px=40):
    yy, xx = np.mgrid[0:hgt, 0:wid] + 0.5
    return np.clip(np.minimum.reduce([xx, wid - xx, yy, hgt - yy]) / px, 0, 1).astype(np.float32)


def view_weight(img, yolo):
    return border_falloff(*img.shape[:2]) * np.where(yolo > 0.5, MASKED_W, 1.0).astype(np.float32)


def blend_wall(w, shots, edges, skip=None):
    """Weighted blend of every view not from pano `skip`. -> (texture uint8, valid mask, used pano ids) or None."""
    Lm, tw, th = wall_grid(w)
    acc, wsum, used = np.zeros((th, tw, 3), np.float32), np.zeros((th, tw), np.float32), set()
    for pano, cam, img, yolo in shots:
        if pano["id"] == skip:
            continue
        r = warp_view(w, cam, img, view_weight(img, yolo), edges)
        if r is None or not (r[1] > 0).any():
            continue
        acc += r[0] * r[1][..., None]
        wsum += r[1]
        used.add(pano["id"])
    valid = wsum > 0
    if not valid.any():
        return None
    tex = np.zeros_like(acc)
    tex[valid] = acc[valid] / wsum[valid, None]
    tex[~valid] = np.median(tex[valid], axis=0)
    return tex.clip(0, 255).astype(np.uint8), valid, sorted(used)


def render_front(ws, texs, shot, edges, u, offset_m=0.0):
    """A front's textures drawn over one view, the front moved offset_m along Moss. -> (image, wall index map)"""
    pano, cam, img, yolo = shot
    if offset_m:
        cam = shifted([shot], {pano["id"]: -offset_m * u})[0][1]     # camera back = wall forward
    rend = img.astype(np.float32)
    owner = np.full((cam.height, cam.width), -1)
    for i, (w, t) in enumerate(zip(ws, texs)):
        if t is not None:
            r, m = render_into(w, cam, t[0], t[1], edges)
            if r is not None:
                rend[m], owner[m] = r[m], i
    return rend, owner


def localise(ws, texs, shot, edges, u):
    """Held-out check: blurred-colour NCC between the view and the front's textures drawn into it, with the
    front at its place and moved along Moss by each of OFFSETS_M, all compared over the same pixels (the
    front's own region at its place, minus YOLO boxes). Blurred colour, not edges: it tells brick from blue
    paint (a texture on the wrong building) and tolerates the decimetres a flat wall can't get right on bay
    fronts seen 40-60 degrees off square. -> ({offset: score}, per-wall scores at the place)"""
    img, yolo = shot[2], shot[3]
    b = cv2.GaussianBlur(img.astype(np.float32), (0, 0), COLOR_SIGMA_PX)
    rend0, own0 = render_front(ws, texs, shot, edges, u)
    base = cv2.erode(((own0 >= 0) & (yolo < 0.5)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    def score(rend, m):
        if m.sum() < MIN_SCORE_PX:
            return None
        a = cv2.GaussianBlur(rend, (0, 0), COLOR_SIGMA_PX)
        return ncc(a[m].ravel(), b[m].ravel(), np.ones(3 * int(m.sum()), bool))
    curve = {}
    for o in OFFSETS_M:
        rend, own = (rend0, own0) if o == 0 else render_front(ws, texs, shot, edges, u, float(o))
        s = score(rend, base & (cv2.erode((own >= 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0))
        if s is not None:
            curve[round(float(o), 2)] = s
    return curve, [score(rend0, base & (own0 == i)) for i in range(len(ws))]


def build_front(ws, shots, edges, u, line_off):
    """One building's Moss front. -> (per-wall [(texture, valid, used)] or None from every pano, held-out
    score at the true position, per-wall scores, best offset m, control = best score CONTROL_M or more away,
    held-out pano id, depth correction m).
    Verification is leave-one-pano-out: the pano that sees most of the front, other than its main contributor,
    is left out of a second blend, which is drawn into that pano's best view of the front (localise()). The
    textures written are the blend of every pano (that test checks the geometry they share)."""
    n = np.array([-u[1], u[0]])
    depth = front_offset(ws, n) - line_off      # OSM scatters Moss fronts in depth; the photos show one line
    shots = shifted(shots, {s[0]["id"]: depth * n for s in shots})
    vis = {}   # pano id -> (blend weight, best single-view visible fraction, that shot index)
    for i, (pano, cam, img, yolo) in enumerate(shots):
        wsum, seen, area = 0.0, 0.0, 0.0
        for w in ws:
            r = warp_view(w, cam, img, view_weight(img, yolo), edges)
            a = np.linalg.norm(w["R"] - w["L"]) * w["h"]
            area += a
            if r is not None:
                wsum += float(r[1].sum())
                seen += a * float((r[1] > 0).mean())
        old = vis.get(pano["id"], (0.0, 0.0, None))
        vis[pano["id"]] = (old[0] + wsum, max(old[1], seen / area), i if seen / area > old[1] else old[2])
    main = max(vis, key=lambda p: vis[p][0])
    cands = [p for p in vis if p != main and vis[p][1] > MIN_HELD_SEEN]
    held = max(cands, key=lambda p: vis[p][1]) if cands else None
    texs = [blend_wall(w, shots, edges) for w in ws]
    if held is None:
        return texs, None, [None] * len(ws), None, None, None, depth
    loo = [blend_wall(w, shots, edges, skip=held) for w in ws]
    curve, per_wall = localise(ws, loo, shots[vis[held][2]], edges, u)
    if 0.0 not in curve:
        return texs, None, per_wall, None, None, held, depth
    best = max(curve, key=curve.get)
    far = [s for o, s in curve.items() if abs(o) >= CONTROL_M]
    return texs, curve[0.0], per_wall, best, max(far) if far else None, held, depth



def front_offset(ws, n):
    """Length-weighted mean distance (m) of walls from Moss's centreline, NE +."""
    Ls = np.array([np.linalg.norm(w["R"] - w["L"]) for w in ws])
    return float(np.sum(Ls * np.array([((w["L"] + w["R"]) / 2) @ n for w in ws])) / Ls.sum())


def pano_strips(shots, plane):
    """Per pano: its views unrolled onto `plane`, blurred colour (float32) and a valid mask."""
    strips = {}
    for pid in sorted({s[0]["id"] for s in shots}):
        acc, wsum = 0, 0
        for pano, cam, img, yolo in shots:
            if pano["id"] == pid:
                r = warp_view(plane, cam, img, border_falloff(*img.shape[:2]) * (yolo < 0.5))
                if r is not None:
                    acc, wsum = acc + r[0] * r[1][..., None], wsum + r[1]
        if np.ndim(wsum) and (wsum > 0).any():
            tex = acc / np.maximum(wsum, 1e-30)[..., None]
            strips[pid] = (cv2.GaussianBlur(tex, (0, 0), 0.3 * STRIP_PPM),
                           cv2.erode((wsum > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    return strips


def pair_shifts(strips):
    """[(p, q, peak NCC, d px)] with strip_p column x ~ strip_q column x + d, for pano pairs that overlap."""
    ids, D, out = sorted(strips), int(MAX_SHIFT * STRIP_PPM), []
    for i, p in enumerate(ids):
        for q in ids[i + 1:]:
            (gp, mp), (gq, mq) = strips[p], strips[q]
            best = (-1.0, 0)
            for d in range(-D, D + 1, 2):
                a, b = (slice(0, gp.shape[1] - d), slice(d, None)) if d >= 0 else (slice(-d, None), slice(0, d))
                m = mp[:, a] & mq[:, b]
                if m.sum() < 2 * 4.5 * STRIP_PPM ** 2:      # ~2 m x half the height in common
                    continue
                c = ncc(gp[:, a][m].ravel(), gq[:, b][m].ravel(), np.ones(3 * int(m.sum()), bool))
                if c is not None and c > best[0]:
                    best = (c, d)
            if best[0] >= MIN_PAIR_NCC:
                out.append((p, q, best[0], best[1]))
    return out


def calibrate(shots, walls, u, n):
    """Street View camera corrections -> {pano id: (dx, dy) metres} to add to the metadata position.
    On Moss the views disagree by metres: neighbouring panos put the brick #29 and the blue #33 on the same
    wall. Each pano's views are unrolled onto one plane at the NE fronts' mean OSM offset, neighbouring strips
    are cross-correlated along the street (blurred colour), and
    1. one cross-street offset for all panos is picked where those along-street disagreements are smallest
       (camera depth error makes them grow with viewing angle; found ~1.5 m NE of the metadata on Moss),
    2. what disagreement remains is solved per pano along the street (weighted least squares, mean zero: the
       absolute along-street offset isn't observable this way).
    ponytail: one global cross-street term; make it per pano if some views still ghost."""
    off = front_offset(walls, n)
    plane = {"bid": 0, "k": 0, "L": off * n + 90 * u, "R": off * n - 8 * u, "h": 9.0, "nrm": -n, "ppm": STRIP_PPM}
    if (plane["L"] - plane["R"]) @ np.array([-n[1], n[0]]) < 0:   # viewer's left = (nrm_y, -nrm_x), nrm = -n
        plane["L"], plane["R"] = plane["R"], plane["L"]
    sgn = 1.0 if (plane["R"] - plane["L"]) @ u > 0 else -1.0       # strip columns run with sgn * along
    best = None
    for sn in CROSS_SEARCH:
        pairs = pair_shifts(pano_strips(shifted(shots, {s[0]["id"]: sn * n for s in shots}), plane))
        if pairs:
            cost = sum(c * min(abs(d) / STRIP_PPM, PAIR_MAX_M) for _, _, c, d in pairs) / sum(c for _, _, c, _ in pairs)
            print(f"  cross {sn:+.2f} m: {len(pairs)} pano pairs, mean disagreement {cost:.2f} m", flush=True)
            if best is None or cost < best[0]:
                best = (cost, sn, pairs)
    if best is None:
        return {s[0]["id"]: np.zeros(2) for s in shots}, 0.0, 0.0
    _, sn, pairs = best
    ids = sorted({s[0]["id"] for s in shots})
    rows, rhs, wts = [np.ones(len(ids))], [0.0], [1.0]
    for p, q, c, d in pairs:          # p shows true(a + e_p) at a, so e_p - e_q = sgn * d / ppm
        if abs(d) / STRIP_PPM > PAIR_MAX_M:
            continue                  # a false match on repeating bay fronts: after the cross term, real ones are < 3 m
        row = np.zeros(len(ids))
        row[ids.index(p)], row[ids.index(q)] = 1, -1
        rows.append(row)
        rhs.append(sgn * d / STRIP_PPM)
        wts.append(c)
    W = np.sqrt(np.array(wts))
    e = np.linalg.lstsq(np.array(rows) * W[:, None], np.array(rhs) * W, rcond=None)[0]
    shifts = {p: sn * n + float(v) * u for p, v in zip(ids, e)}
    anchor = ALONG_ANCHOR_M
    return {p: s + anchor * u for p, s in shifts.items()}, sn, anchor


def shifted(shots, shifts):
    out = []
    for pano, cam, img, yolo in shots:
        x, y = np.array([cam.x, cam.y]) + shifts.get(pano["id"], np.zeros(2))
        out.append((pano, Camera(x=x, y=y, z=cam.z, heading=cam.heading, pitch=cam.pitch, focal=cam.focal,
                                 width=cam.width, height=cam.height), img, yolo))
    return out


def corners(w):
    L, R, h = w["L"], w["R"], w["h"]
    return [[round(float(v), 3) for v in p] for p in ([*L, 0], [*R, 0], [*R, h], [*L, h])]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--weights", type=Path, default=WEIGHTS)
    p.add_argument("--env", type=Path, default=ROOT / ".env", help="GOOGLE_MAPS_API_KEY from here if not set")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    from kite.sources.streetview import StreetViewSource
    if not os.environ.get("GOOGLE_MAPS_API_KEY") and a.env.is_file():
        for line in a.env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "GOOGLE_MAPS_API_KEY":
                os.environ[k.strip()] = v.strip().strip("'\"")
    data = json.loads(MOSS.read_text())
    walls, u = ne_walls(data)
    src = StreetViewSource(single=(37.7783, -122.4087, 0), cache_dir=CACHE)
    panos = find_panos(src, data, u)
    views = plan_views(group_targets(walls, data), panos)
    print(f"{len(walls)} NE walls on {len({w['bid'] for w in walls})} fronts, {len(panos)} Google panoramas, "
          f"{len(views)} views", flush=True)
    try:
        from ultralytics import YOLO
        model = YOLO(str(a.weights))
    except Exception as e:   # noqa: BLE001 - no detector means no masking, not no facades
        print(f"YOLO unavailable ({e}); cars and people are not masked", file=sys.stderr)
        model = None
    shots, masked = [], 0
    for pano, hd, pt in views:
        raw = src._get("image", {"pano": pano["id"], "heading": f"{hd:.1f}", "pitch": f"{pt}", "fov": f"{FOV}",
                                 "size": f"{SIZE}x{SIZE}", "return_error_code": "true"})
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Street View image: the response is not an image")
        cam = Camera(x=pano["x"], y=pano["y"], z=CAM_Z, heading=hd, pitch=pt,
                     focal=0.5 / math.tan(math.radians(FOV) / 2), width=img.shape[1], height=img.shape[0])
        ym = yolo_mask(model, img)
        masked += ym.mean()
        shots.append((pano, cam, img, ym))
    ek = [(b["id"], k) for b in data["buildings"] for k in range(len(b["pts"]) - 1)]
    edges = (ek, np.array([b["pts"][k] for b in data["buildings"] for k in range(len(b["pts"]) - 1)], float),
             np.array([b["pts"][k + 1] for b in data["buildings"] for k in range(len(b["pts"]) - 1)], float))
    n = np.array([-u[1], u[0]])
    shifts, cross, anchor = calibrate(shots, walls, u, n)
    along = {k: round(float(v @ u), 2) for k, v in shifts.items()}
    print(f"camera corrections: {cross:+.2f} m across Moss (NE +) for all, boundary anchor {anchor:+.2f} m; along: "
          + ", ".join(f"{k[:6]} {v:+.2f}" for k, v in along.items()), flush=True)
    shots = shifted(shots, shifts)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for bid in BUILDINGS:
        ws = [w for w in walls if w["bid"] == bid]
        if not ws:
            continue
        texs, score, per_wall, best, control, held, depth = build_front(ws, shots, edges, u, front_offset(walls, n))
        verified = (score is not None and score >= MIN_SCORE and abs(best) <= MAX_BEST_OFFSET_M
                    and (control is None or score > control))
        print(f" building {bid}: score {fmt(score)}, best match at {fmt(best)} m, best >= {CONTROL_M:g} m away "
              f"{fmt(control)}, held-out pano {held}, front depth correction {depth:+.2f} m -> {'verified' if verified else 'REJECTED'}", flush=True)
        for w, t, ws_score in zip(ws, texs, per_wall):
            Lm = float(np.linalg.norm(w["R"] - w["L"]))
            cover = 0.0 if t is None else float(t[1].mean())
            entry = {"building_id": w["bid"], "wall_index": w["k"], "corners_local_m": corners(w),
                     "width_m": round(Lm, 3), "height_m": w["h"], "texture": None, "score": rnd(score),
                     "best_offset_m": best, "control_score": rnd(control), "wall_score": rnd(ws_score),
                     "coverage": round(cover, 3), "source_pano_ids": [] if t is None else t[2],
                     "heldout_pano_id": held, "front_depth_correction_m": round(depth, 2),
                     "facing": round(w["facing"], 2)}
            if verified and cover >= MIN_COVER and w["facing"] >= MIN_FACING_TEX:
                entry["texture"] = f"{w['bid']}_{w['k']}.jpg"
                cv2.imwrite(str(a.out / entry["texture"]), t[0], [cv2.IMWRITE_JPEG_QUALITY, 88])
            manifest.append(entry)
            print(f"  {w['bid']}_{w['k']:<2} {Lm:5.2f} x {w['h']:4.1f} m  cover {cover:4.0%}  wall score "
                  f"{fmt(ws_score)}  {'ok' if entry['texture'] else 'no texture'}", flush=True)
    r = src.requests
    (a.out / "manifest.json").write_text(json.dumps({
        "convention": CONVENTION, "px_per_m": PX_PER_M, "min_score": MIN_SCORE,
        "max_best_offset_m": MAX_BEST_OFFSET_M, "min_coverage": MIN_COVER, "min_facing": MIN_FACING_TEX,
        "score": "Verified per building front (score is shared by its walls). The Street View panorama that sees "
                 "most of the front, other than its main contributor, is left out of a second blend, which is drawn "
                 "into that panorama's best view of the front and compared with it: NCC of blurred colour over the "
                 "visible front minus YOLO person/vehicle boxes. The front is also drawn moved "
                 f"{OFFSETS_M[0]:g}..{OFFSETS_M[-1]:g} m along Moss over the same pixels: best_offset_m is where it "
                 f"matches best, control_score the best match {CONTROL_M:g} m or more away. A wall gets a texture when "
                 "its front has score > control_score, |best_offset_m| <= max_best_offset_m and score >= min_score, and the wall itself "
                 "faces the street (facing >= min_facing; bay sides don't) and coverage (fraction seen by a view) >= "
                 "min_coverage. The texture written blends every panorama. wall_score: the same NCC over that wall "
                 "only (noisy on 1-2 m bay facets). Checks placement to about a metre, not pixel detail.",
        "camera_correction_m": {"across_moss_ne": cross, "along_anchor": anchor, "along_moss_se": along,
                                "per_front": "front_depth_correction_m in each wall",
                                "note": "added to Street View metadata positions before projecting (calibrate())"},
        "requests": r, "walls": manifest}, indent=1))
    ok = sum(1 for m in manifest if m["texture"])
    print(f"textured {ok}/{len(manifest)} walls; mean YOLO-masked image fraction {masked / max(1, len(shots)):.1%}; "
          f"Street View requests: {r['image']} image (billed), {r['metadata']} metadata (free), {r['cached']} cached")
    return 0


def fmt(v):
    return "n/a" if v is None else f"{v:.2f}"


def rnd(v):
    return None if v is None else round(float(v), 3)


def selftest():
    """A synthetic 24 m wall 6 m from a row of five panos (Moss-like), three headings each: blending four must
    re-project onto the fifth (high score, best at offset 0, lower when moved), and the camera calibration must
    find one pano whose metadata is 1 m off along the street."""
    from facade_rectify import _render_wall
    rng = np.random.default_rng(0)
    h, Lp, Rp = 6.0, np.array([0.0, 10.0]), np.array([24.0, 10.0])    # seen from the south: L is west
    w = {"bid": 1, "k": 0, "L": Lp, "R": Rp, "h": h, "nrm": np.array([0.0, -1.0])}
    blocks = cv2.resize(rng.uniform(0, 255, (12, 48, 3)), (1200, 300), interpolation=cv2.INTER_NEAREST)
    tex = cv2.GaussianBlur(blocks, (0, 0), 1)
    f = 0.5 / math.tan(math.radians(FOV) / 2)
    u, n = np.array([1.0, 0.0]), np.array([0.0, 1.0])      # "Moss" runs east here; the wall is its NE side
    shots = []
    for i, x in enumerate((0.0, 6.0, 12.0, 18.0, 24.0)):
        for hd in (-40.0, 0.0, 40.0):
            c = Camera(x=x, y=4.0, z=CAM_Z, heading=hd % 360, pitch=15, focal=f, width=SIZE, height=SIZE)
            with np.errstate(all="ignore"):
                img = _render_wall(tex, c, tuple(Lp), tuple(Rp), h)
            shots.append(({"id": f"p{i}"}, c, img, np.zeros((SIZE, SIZE), np.float32)))
    texs, score, per_wall, best, control, held, depth = build_front([w], shots, None, u, front_offset([w], np.array([0.0, 1.0])))
    out, valid = texs[0][0], texs[0][1]
    gt = cv2.resize(tex, (out.shape[1], out.shape[0]))
    err = np.abs(out.astype(float) - gt)[valid].mean()
    assert out.shape[:2] == (round(h * PX_PER_M), round(24 * PX_PER_M)) and err < 12, err
    assert score > 0.9 and best == 0.0 and control < score - 0.3, (score, best, control)
    assert corners(w)[0] == [0.0, 10.0, 0.0] and corners(w)[1] == [24.0, 10.0, 0.0]
    shifts, cross, _ = calibrate(shifted(shots, {"p2": np.array([1.0, 0.0])}), [w], u, n)   # p2's metadata 1 m east
    rel = (shifts["p2"] - np.mean([shifts[f"p{i}"] for i in (0, 1, 3, 4)], axis=0)) @ u
    assert abs(cross) <= 0.25 and abs(rel + 1.0) < 0.25, (cross, rel)
    print(f"facades_ne self-check ok: texture error {err:.1f}/255, held-out score {score:.2f} (best at {best:g} m, "
          f"{control:.2f} {CONTROL_M:g} m+ away); calibration: across {cross:+.2f} m, p2 corrected by {rel:+.2f} m "
          "(true -1.00)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
