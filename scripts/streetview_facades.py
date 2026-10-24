"""Facade textures for Moss St from Google Street View (permission: docs/google-permission.md).

    set -a; . ../kite/.env; set +a        # GOOGLE_MAPS_API_KEY, never printed
    uv run --no-project --with numpy --with opencv-python-headless python scripts/streetview_facades.py

Writes sim/public/local/facades/<building>_<edge>.jpg and facades.json (gitignored: Street View-derived pixels
are never committed or published). Raw API responses are cached in /tmp/kite_sv/cache so a rerun is free;
delete that folder when done.

1. Street-facing walls: OSM footprint edges in moss.json that face Moss St (Folsom to Howard) or Howard St
   within the corner, and that nothing else stands in front of.
2. Google panoramas from the metadata endpoint (free) every 4 m along Moss and Howard near the corner, all
   inside kite.sources.streetview.SCOPE.
3. Per panorama, one Static view (640x640, billed) per group of walls it sees within 15 m, aimed straight at
   them, pitched up to the middle of the wall, fov 65-100 by distance.
4. Every wall texel (40 px/m, at most 2048 px wide) is projected into every view that sees it
   (facade_rectify.project, camera 2.5 m up at the panorama), skipping views where another footprint is in the
   way. Views are blended per texel with weight (frontal cos / distance)^2 times an image-border falloff, raised
   to WEIGHT_POW, so the most frontal, closest view wins and the seams are feathered.
5. openings(): closed rectangular Canny contours of window/door size in each texture, for the recesses that
   sim/blender/build_real_moss.py cuts. `--openings-only` redoes just this step from the saved JPEGs.
   ponytail: a contour heuristic; it finds about half the windows and some mural/tree false positives.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from facade_rectify import Camera, project, sample  # noqa: E402
from kite.gps import latlng_to_sim, sim_to_latlng  # noqa: E402
from kite.sources.streetview import SV_CAM_HEIGHT_M, StreetViewSource, _is_google  # noqa: E402

MOSS = ROOT / "sim/public/osm/moss.json"
OUT = ROOT / "sim/public/local/facades"
CACHE = Path("/tmp/kite_sv/cache")
MOSS_ROAD, HOWARD_ROADS = 8916291, (1121644083, 1469149564)
PX_PER_M, MAX_W = 40, 2048
SIZE = 640
SEE_M = 15.0           # a wall further than this from a panorama isn't fetched from it
GROUP_DEG = 35.0       # walls within this of one heading share a view
WEIGHT_POW = 3.0
MIN_COVER = 0.5        # below this fraction seen, the wall keeps the procedural facade


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def compass(dx, dy):
    return math.degrees(math.atan2(dx, dy)) % 360


def seg_cross(p, q, a, b):
    """Do segments pq and ab properly intersect (2D)?"""
    def orient(o, s, t):
        return (s[0] - o[0]) * (t[1] - o[1]) - (s[1] - o[1]) * (t[0] - o[0])
    d1, d2, d3, d4 = orient(a, b, p), orient(a, b, q), orient(p, q, a), orient(p, q, b)
    return d1 * d2 < 0 and d3 * d4 < 0


def blocked(c, tips, edges, skip):
    """For each tip (N, 2): does the segment from c to it properly cross any footprint edge but `skip`?"""
    ids, P, Q = edges
    keep = np.array([i != skip for i in ids])
    P, Q = P[keep], Q[keep]
    def orient(o, s, t):   # broadcasting (N, E)
        return (s[..., 0] - o[..., 0]) * (t[..., 1] - o[..., 1]) - (s[..., 1] - o[..., 1]) * (t[..., 0] - o[..., 0])
    C = np.broadcast_to(c, tips.shape)[:, None, :]
    T = tips[:, None, :]
    d1, d2 = orient(P[None], Q[None], C), orient(P[None], Q[None], T)
    d3, d4 = orient(C, T, P[None]), orient(C, T, Q[None])
    return ((d1 * d2 < 0) & (d3 * d4 < 0)).any(axis=1)


def street_walls(data):
    """[(building id, edge index, a, b, h, outward normal)] facing Moss or the Howard corner."""
    roads = {r["id"]: r for r in data["roads"]}
    moss_u = unit(np.subtract(roads[MOSS_ROAD]["pts"][2], roads[MOSS_ROAD]["pts"][1]))
    hp = roads[HOWARD_ROADS[0]]["pts"]
    how_u = unit(np.subtract(roads[HOWARD_ROADS[1]]["pts"][0], hp[2]))
    how_o = np.array(hp[0], float)
    lines = [(np.zeros(2), moss_u, -6.0, 185.0, 12.0), (how_o - how_u * how_o @ how_u, how_u, -45.0, 30.0, 20.0)]
    edges = [(b["id"], k, b["pts"][k], b["pts"][k + 1]) for b in data["buildings"] for k in range(len(b["pts"]) - 1)]
    out = []
    for b in data["buildings"]:
        pts = np.array(b["pts"], float)
        area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
        for k in range(len(pts) - 1):
            a, c = pts[k], pts[k + 1]
            L = np.linalg.norm(c - a)
            if L < 0.8:
                continue
            e = (c - a) / L
            nrm = np.array([e[1], -e[0]]) if area > 0 else np.array([-e[1], e[0]])   # outward
            m = (a + c) / 2
            for o, u, lo, hi, far in lines:
                n_line = np.array([-u[1], u[0]])
                off, along = (m - o) @ n_line, (m - o) @ u
                if lo <= along <= hi and 1.5 < abs(off) < far and nrm @ (-np.sign(off) * n_line) > 0.35:   # bays too
                    foot = m - off * n_line
                    tip = m + 0.05 * nrm
                    if not any(seg_cross(tip, foot, p, q) for bid, kk, p, q in edges if (bid, kk) != (b["id"], k)):
                        out.append((b["id"], k, a, c, float(b["h"]), nrm))
                    break
    return out


def find_panos(data, src):
    roads = {r["id"]: r for r in data["roads"]}
    def samples(pts, step):
        res = []
        for p, q in zip(pts, pts[1:]):
            n = max(1, int(math.dist(p, q) // step))
            res += [(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n) for i in range(n)]
        return res + [tuple(pts[-1])]
    pts = samples(roads[MOSS_ROAD]["pts"], 4)
    pts += [p for rid in HOWARD_ROADS for p in samples(roads[rid]["pts"], 5) if math.hypot(*p) < 45]
    panos = {}
    for x, y in pts:
        m = src._metadata(*sim_to_latlng(x, y))   # raises outside SCOPE
        if m and _is_google(m):
            px, py = latlng_to_sim(m["location"]["lat"], m["location"]["lng"])
            panos[m["pano_id"]] = {"id": m["pano_id"], "x": px, "y": py, "date": m.get("date")}
    return list(panos.values())


def plan_views(walls, panos):
    """Per panorama, headings aimed at the walls it sees, grouped. -> [(pano, heading, pitch, fov)]"""
    views = []
    for p in panos:
        c = np.array([p["x"], p["y"]])
        aims = []
        for _bid, _k, a, b, h, nrm in walls:
            d = (c - a) @ nrm
            if d < 2.5 or d > SEE_M:
                continue
            e = (b - a) / np.linalg.norm(b - a)
            t = np.clip((c - a) @ e, 1.0, np.linalg.norm(b - a) - 1.0)
            tgt = a + t * e
            dist = np.linalg.norm(tgt - c)
            if dist > SEE_M or (c - tgt) @ nrm / dist < 0.5:
                continue
            aims.append((compass(*(tgt - c)), dist, h))
        aims.sort()
        groups = []
        for hd, dist, h in aims:
            if groups and abs((hd - groups[-1][0][0] + 180) % 360 - 180) < GROUP_DEG:
                groups[-1].append((hd, dist, h))
            else:
                groups.append([(hd, dist, h)])
        if len(groups) > 1 and abs((groups[0][0][0] - groups[-1][0][0] + 180) % 360 - 180) < GROUP_DEG:
            groups[0] += groups.pop()
        for g in groups:
            hs = np.radians([x[0] for x in g])
            hd = math.degrees(math.atan2(np.sin(hs).mean(), np.cos(hs).mean())) % 360
            dist = min(x[1] for x in g)
            pitch = float(np.clip(math.degrees(math.atan2(max(x[2] for x in g) / 2 - SV_CAM_HEIGHT_M, dist)), 0, 35))
            fov = 100 if dist < 9 else 80 if dist < 15 else 65
            views.append((p, round(hd, 1), round(pitch), fov))
    return views


def wall_texture(wall, shots, edges):
    bid, k, a, b, h, nrm = wall
    L = float(np.linalg.norm(b - a))
    ppm = min(PX_PER_M, MAX_W / L)
    tw, th = max(2, round(L * ppm)), max(2, round(h * ppm))
    s = (np.arange(tw) + 0.5) / tw
    z = h * (1 - (np.arange(th) + 0.5) / th)
    ground = a[None, :] + s[:, None] * (b - a)[None, :]                       # (tw, 2)
    pts = np.concatenate([np.repeat(ground[None], th, 0), np.repeat(z[:, None, None], tw, 1)], -1).reshape(-1, 3)
    acc = np.zeros((th * tw, 3))
    wsum = np.zeros(th * tw)
    used = []
    for pano, cam, img in shots:
        c = np.array([cam.x, cam.y])
        if (c - a) @ nrm < 1.0:
            continue
        uv = project(cam, pts)
        vals, ok = sample(img.astype(float), uv[:, 0], uv[:, 1])
        if ok.mean() < 0.02:
            continue
        # columns hidden behind another footprint from this panorama
        tips = ground - 0.15 * (ground - c) / np.linalg.norm(ground - c, axis=1)[:, None]
        ok &= ~np.tile(blocked(c, tips, edges, (bid, k)), th)
        d3 = pts - np.array([cam.x, cam.y, cam.z])
        dist = np.linalg.norm(d3, axis=1)
        cos = np.clip(-(d3[:, :2] @ nrm) / dist, 0, 1)
        border = np.clip(np.minimum.reduce([uv[:, 0], cam.width - uv[:, 0], uv[:, 1], cam.height - uv[:, 1]]) / 40, 0, 1)
        w = np.where(ok, ((cos / dist) ** 2 * np.nan_to_num(border)) ** WEIGHT_POW, 0)
        if w.sum() == 0:
            continue
        acc += vals * w[:, None]
        wsum += w
        used.append(pano["id"])
    seen = wsum > 0
    tex = np.zeros((th * tw, 3))
    tex[seen] = acc[seen] / wsum[seen, None]
    if seen.any():
        tex[~seen] = np.median(tex[seen], axis=0)
    return tex.reshape(th, tw, 3).clip(0, 255).astype(np.uint8), float(seen.mean()), sorted(set(used))


def openings(img, L, h):
    """Window, door and garage rectangles [kind, s0, z0, s1, z1] (metres along the wall from a, up from the
    ground) found as closed, rectangular Canny contours. Nested frame contours keep the biggest."""
    th, tw = img.shape[:2]
    ppm = tw / L
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    e = cv2.dilate(cv2.Canny(g, 40, 110), np.ones((2, 2), np.uint8))
    cand = []
    for c in cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, hh = cv2.boundingRect(c)
        wm, hm = w / ppm, hh / ppm
        if not (0.6 <= wm <= 6 and 0.7 <= hm <= 3.6) or hm > 0.8 * h or cv2.contourArea(c) < 0.75 * w * hh:
            continue
        z1 = h - y / ppm
        z0 = z1 - hm
        if z1 > h - 0.2 or not 0.35 < wm / hm < 3.5:
            continue
        if z0 < 0.6:
            if hm < 1.9:
                continue      # cars, bins, planters
            kind = "garage" if wm > 2.2 else "door"
        else:
            kind = "window"
        cand.append((wm * hm, [kind, round(x / ppm, 2), round(z0, 2), round((x + w) / ppm, 2), round(z1, 2)]))
    out = []
    for _, r in sorted(cand, key=lambda t: -t[0]):
        if all(min(r[3], o[3]) - max(r[1], o[1]) <= 0.05 or min(r[4], o[4]) - max(r[2], o[2]) <= 0.05 for o in out):
            out.append(r)
    return out


def add_openings(manifest):
    for entry in manifest:
        if entry["file"]:
            img = cv2.imread(str(OUT / entry["file"]))
            L = float(np.hypot(*np.subtract(entry["b"], entry["a"])))
            entry["openings"] = openings(img, L, entry["h"])
            top = img[: max(1, img.shape[0] // 12)].reshape(-1, 3)
            entry["top_rgb"] = [round(float(v) / 255, 3) for v in np.median(top, axis=0)[::-1]]


def main():
    if "--openings-only" in sys.argv:
        man = json.loads((OUT / "facades.json").read_text())
        add_openings(man["walls"])
        (OUT / "facades.json").write_text(json.dumps(man, indent=1))
        print(f"openings: {sum(len(w.get('openings', [])) for w in man['walls'])} in {len(man['walls'])} walls")
        return
    data = json.loads(MOSS.read_text())
    walls = street_walls(data)
    src = StreetViewSource(single=(37.7783, -122.4087, 0), cache_dir=CACHE)
    panos = find_panos(data, src)
    views = plan_views(walls, panos)
    print(f"{len(walls)} street-facing walls, {len(panos)} Google panoramas, {len(views)} views", flush=True)
    shots = []
    for pano, hd, pitch, fov in views:
        raw = src._get("image", {"pano": pano["id"], "heading": f"{hd:.1f}", "pitch": f"{pitch}", "fov": f"{fov}",
                                 "size": f"{SIZE}x{SIZE}", "return_error_code": "true"})
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Street View image: the response is not an image")
        cam = Camera(x=pano["x"], y=pano["y"], z=SV_CAM_HEIGHT_M, heading=hd, pitch=pitch,
                     focal=0.5 / math.tan(math.radians(fov) / 2), width=img.shape[1], height=img.shape[0])
        shots.append((pano, cam, img))
    ek = [(b["id"], k) for b in data["buildings"] for k in range(len(b["pts"]) - 1)]
    edges = (ek, np.array([b["pts"][k] for b in data["buildings"] for k in range(len(b["pts"]) - 1)], float),
             np.array([b["pts"][k + 1] for b in data["buildings"] for k in range(len(b["pts"]) - 1)], float))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, textured = [], 0
    for wall in walls:
        bid, k, a, b, h, nrm = wall
        tex, cover, used = wall_texture(wall, shots, edges)
        entry = {"building": bid, "edge": k, "a": a.round(3).tolist(), "b": b.round(3).tolist(), "h": h,
                 "normal": nrm.round(4).tolist(), "coverage": round(cover, 3), "views": used, "file": None}
        if cover >= MIN_COVER:
            entry["file"] = f"{bid}_{k}.jpg"
            cv2.imwrite(str(OUT / entry["file"]), tex, [cv2.IMWRITE_JPEG_QUALITY, 88])
            textured += 1
        manifest.append(entry)
        print(f"  {bid}_{k}: {np.linalg.norm(b - a):5.1f} m x {h:4.1f} m, seen {cover:4.0%} from {len(used)} panoramas"
              f"{'' if entry['file'] else '  -> fallback'}", flush=True)
    add_openings(manifest)
    r = src.requests
    (OUT / "facades.json").write_text(json.dumps({"walls": manifest, "requests": r, "panos": len(panos),
                                                  "views": len(views)}, indent=1))
    print(f"textured {textured}/{len(walls)} walls; Street View requests: {r['image']} image (billed), "
          f"{r['metadata']} metadata (free), {r['cached']} from cache")


if __name__ == "__main__":
    main()
