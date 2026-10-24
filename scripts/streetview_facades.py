"""Facade textures for Moss St from Google Street View, v2 (permission: docs/google-permission.md).

    set -a; . ../kite/.env; set +a        # GOOGLE_MAPS_API_KEY, never printed
    uv run --no-project --with numpy --with opencv-python-headless --with ultralytics \\
        python scripts/streetview_facades.py [--plan] [--score-only] [--before moss_before.json]

Writes sim/public/local/facades/<building>_<edge>.jpg and facades.json (gitignored: Street View-derived pixels
and anything measured from them are never committed or published). Raw API responses and segmentation masks
are cached in /tmp/kite_sv so a rerun is free; delete that folder when done.

1. Walls: footprint edges in moss.json that face Moss St (Folsom to Howard) or Howard St within the corner.
   Buildings with a `ground` footprint (bays clipped at the property line, scripts/osm_corner.py) give two
   kinds: roof-outline edges from `bay_z` up where they overhang, and the ground-floor edge under them.
2. Google panoramas from the metadata endpoint (free) every 4 m along Moss and Howard near the corner, all
   inside kite.sources.streetview.SCOPE.
3. Tiles: per panorama, narrow Static views (640x640, billed) square to each side of the street, 2 headings x
   3 pitches at fov 50 on Moss (about 150 px/m on the near walls), fov 36 on Howard. Only tiles that are the
   best tile of this panorama for some wall point are fetched.
4. Masks: ADE20K segmentation (weights/yolo26n-sem-ade20k.pt, as in sim/seg_standin.py) marks cars, people,
   trees, poles, signs and sky in every tile; those pixels only count where no view sees the wall.
5. Poses: Google snaps panorama positions to its road line, about 2 m off the lane the car drove in on Moss.
   Each panorama's position, height, yaw and tilt are refined so that the textures different panoramas make
   for the same wall agree (high-pass NCC): first one lateral shift per street, then per panorama.
6. Verification: a wall gets a photo only if at least two panoramas see it and their textures agree (score >=
   SCORE_MIN); otherwise it keeps a plain material. Wrong is worse than plain. Scores go in facades.json.
7. Texture at PX_PER_M from all tiles, weighted to the most frontal, closest, unmasked view; texels no view
   sees cleanly are inpainted (OpenCV Telea). Windows and doors come from ADE20K run on the rectified
   texture; the photo roofline (where sky starts) is measured per wall.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from facade_rectify import sample  # noqa: E402
from kite.gps import latlng_to_sim, sim_to_latlng  # noqa: E402
from kite.sources.streetview import SV_CAM_HEIGHT_M, StreetViewSource, _is_google  # noqa: E402

MOSS = ROOT / "sim/public/osm/moss.json"
OUT = ROOT / "sim/public/local/facades"
CACHE = Path("/tmp/kite_sv/cache")
SEG_CACHE = Path("/tmp/kite_sv/seg")
WEIGHTS = next((p for p in (ROOT / "weights/yolo26n-sem-ade20k.pt",
                            ROOT.parent / "kite/weights/yolo26n-sem-ade20k.pt") if p.exists()), None)
MOSS_ROAD, HOWARD_ROADS = 8916291, (1121644083, 1469149564)
TILE = 640
FOV = {"moss": 50.0, "howard": 36.0}
HEAD_OFFS = {"moss": (-60.0, -22.0, 22.0, 60.0), "howard": (-32.0, 0.0, 32.0)}
PITCHES = {"moss": (-12.0, 24.0, 58.0), "howard": (2.0, 30.0)}
SEE_M = {"moss": 16.0, "howard": 30.0}
MAX_OBLIQUE = 72.0     # degrees off square beyond which a view doesn't see a wall (the blend favours square views)
PX_PER_M = {"moss": 144, "howard": 64}
MAX_TEX = 4096
REF_PPM, SCORE_PPM = 8, 10
REFINE_OBLIQUE = 60.0  # pose refinement compares only views this square to a wall: depth errors grow with tan
SCORE_MIN = 0.30       # mean high-pass NCC between panoramas' textures of a wall (see verify())
MIN_PAIR_M2 = 3.0      # two panoramas must both see this much of a wall to be compared
MIN_COVER = 0.6
WEIGHT_POW = 3.0
MASKED_W = 1e-4        # weight of a masked pixel: used only where nothing better sees the texel
OCCLUDERS = ("car", "person", "tree", "plant", "pole", "streetlight", "truck", "van", "bicycle", "minibike",
             "bus", "signboard", "traffic light", "ashcan", "pot", "flower")
LOGO_ROWS = 24
OPENINGS = {"windowpane": "window", "door": "door", "screen door": "door"}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def compass(dx, dy):
    return math.degrees(math.atan2(dx, dy)) % 360


# ---------------------------------------------------------------- walls
def _on_boundary(p, poly, tol=0.05):
    for a, b in zip(poly, poly[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        e = b - a
        t = np.clip((p - a) @ e / max(e @ e, 1e-12), 0, 1)
        if np.linalg.norm(a + t * e - p) < tol:
            return True
    return False


def _ccw(pts):
    pts = np.asarray(pts, float)
    return 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]) > 0


def building_edges(b):
    """[(key, a, b, z0, z1, outward normal)] for a building: roof-outline edges (bays from bay_z up) and the
    ground-floor edges under them."""
    out, h = [], float(b["h"])
    ground = b.get("ground")
    bay_z = float(b.get("bay_z", 3.0))
    for poly, is_ground in ((b["pts"], False), (ground, True)):
        if poly is None:
            continue
        ccw = _ccw(poly)
        for k in range(len(poly) - 1):
            a, c = np.array(poly[k], float), np.array(poly[k + 1], float)
            L = np.linalg.norm(c - a)
            if L < 0.3:
                continue
            e = (c - a) / L
            nrm = np.array([e[1], -e[0]]) if ccw else np.array([-e[1], e[0]])
            m = (a + c) / 2
            if is_ground:
                if _on_boundary(m, b["pts"]):
                    continue                     # same as an outline edge, already listed
                out.append((f"g{k}", a, c, 0.0, min(bay_z, h), nrm))
            else:
                z0 = bay_z if ground is not None and not _on_boundary(m, ground) else 0.0
                if z0 < h:
                    out.append((k, a, c, z0, h, nrm))
    return out


def street_lines(data):
    roads = {r["id"]: r for r in data["roads"]}
    mp = roads[MOSS_ROAD]["pts"]
    moss_u = unit(np.subtract(mp[-1], mp[0]))                                   # SE, Howard to Folsom
    hp = roads[HOWARD_ROADS[1]]["pts"]                                          # runs SW to the corner
    how_u = unit(np.subtract(hp[0], hp[-1]))                                    # NE
    how_o = np.array(hp[-1], float)
    # (origin, unit, along min, along max, max offset, name)
    return [(np.zeros(2), moss_u, -6.0, 185.0, 12.0, "moss"),
            (how_o - how_u * (how_o @ how_u), how_u, -45.0, 30.0, 20.0, "howard")]


def street_walls(data):
    """Walls facing Moss or the Howard corner: dicts with building, edge, a, b, z0, z1, normal, street."""
    lines = street_lines(data)
    occl = [(b["id"], np.array(p, float), np.array(q, float)) for b in data["buildings"]
            for p, q in zip(b["pts"], b["pts"][1:])]
    out = []
    for b in data["buildings"]:
        for key, a, c, z0, z1, nrm in building_edges(b):
            m = (a + c) / 2
            for o, u, lo, hi, far, name in lines:
                n_line = np.array([-u[1], u[0]])
                off, along = (m - o) @ n_line, (m - o) @ u
                if not (lo <= along <= hi and 1.5 < abs(off) < far and nrm @ (-np.sign(off) * n_line) > 0.35):
                    continue
                foot, tip = m - off * n_line, m + 0.05 * nrm
                if not any(seg_cross(tip, foot, p, q) for bid, p, q in occl if bid != b["id"]):
                    out.append({"building": b["id"], "edge": key, "a": a, "b": c, "z0": z0, "z1": z1,
                                "normal": nrm, "street": name, "h_src": b.get("h_src", ""),
                                "h_max": b.get("h_max")})
                break
    return out


def seg_cross(p, q, a, b):
    def orient(o, s, t):
        return (s[0] - o[0]) * (t[1] - o[1]) - (s[1] - o[1]) * (t[0] - o[0])
    d1, d2, d3, d4 = orient(a, b, p), orient(a, b, q), orient(p, q, a), orient(p, q, b)
    return d1 * d2 < 0 and d3 * d4 < 0


def occluders(data):
    """Every outline / ground edge with the heights it covers: (building id, key, p, q, zlo, zhi)."""
    out = []
    for b in data["buildings"]:
        for key, a, c, z0, z1, _ in building_edges(b):
            out.append((b["id"], key, a, c, z0, z1))
    return out


# ---------------------------------------------------------------- panoramas, cameras
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


def pano_street(p, data):
    """'moss' or 'howard' (the street the panorama is on) and that street's unit direction and left normal."""
    o, u, *_ = street_lines(data)[0]
    c = np.array([p["x"], p["y"]])
    n = np.array([-u[1], u[0]])
    if abs(c @ n) < 6 and (c @ u) > 2:
        return "moss", u, n
    _, u2, *_ = street_lines(data)[1]
    return "howard", u2, np.array([-u2[1], u2[0]])


ZERO = {"perp": 0.0, "along": 0.0, "dz": 0.0, "yaw": 0.0, "tx": 0.0, "ty": 0.0}


def cam_centre(p, pose, frame):
    _, u, n = frame
    c = np.array([p["x"], p["y"]]) + pose["perp"] * n + pose["along"] * u
    return np.array([c[0], c[1], SV_CAM_HEIGHT_M + pose["dz"]])


def tilt_matrix(tx, ty):
    a, b = math.radians(tx), math.radians(ty)
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    return rx @ ry


def project(C, pose, heading, pitch, fov, pts, size=TILE):
    """World points (N, 3) to pixels (N, 2) of a Static view (heading clockwise from north, pitch up),
    with the panorama's pose correction (yaw, tilt). NaN behind the camera."""
    d = pts - C
    if pose["tx"] or pose["ty"]:
        d = d @ tilt_matrix(pose["tx"], pose["ty"]).T
    h, p = math.radians(heading + pose["yaw"]), math.radians(pitch)
    level, up = np.array([math.sin(h), math.cos(h), 0.0]), np.array([0.0, 0.0, 1.0])
    fwd = math.cos(p) * level + math.sin(p) * up
    right = np.array([math.cos(h), -math.sin(h), 0.0])
    down = math.sin(p) * level - math.cos(p) * up
    f = 0.5 * size / math.tan(math.radians(fov) / 2)
    depth = d @ fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([size / 2 + f * (d @ right) / depth, size / 2 + f * (d @ down) / depth], 1)
    uv[depth <= 0.1] = np.nan
    return uv


def wall_grid(w, ppm, z_top=None):
    """Texel centres of a wall: (pts (th*tw, 3), ground (tw, 2), z (th,), tw, th); row 0 at the top."""
    a, b = w["a"], w["b"]
    L = float(np.linalg.norm(b - a))
    z1 = w["z1"] if z_top is None else z_top
    ppm = min(ppm, MAX_TEX / L, MAX_TEX / max(z1 - w["z0"], 0.1))
    tw, th = max(2, round(L * ppm)), max(2, round((z1 - w["z0"]) * ppm))
    s = (np.arange(tw) + 0.5) / tw
    z = z1 - (np.arange(th) + 0.5) / th * (z1 - w["z0"])
    ground = a[None, :] + s[:, None] * (b - a)[None, :]
    pts = np.concatenate([np.repeat(ground[None], th, 0), np.repeat(z[:, None, None], tw, 1)], -1).reshape(-1, 3)
    return pts, ground, z, tw, th


def hidden(C, ground, z, occ, wall):
    """(th, tw) bool: texels whose ray from C is cut by another footprint edge within that edge's heights."""
    th, tw = len(z), len(ground)
    out = np.zeros((th, tw), bool)
    c2 = C[:2]
    tips = ground - 0.05 * (ground - c2) / np.linalg.norm(ground - c2, axis=1)[:, None]
    lo, hi = np.minimum(c2, tips.min(0)), np.maximum(c2, tips.max(0))
    for bid, key, p, q, zlo, zhi in occ:
        if (bid, key) == (wall["building"], wall["edge"]) or max(p[0], q[0]) < lo[0] or min(p[0], q[0]) > hi[0] \
                or max(p[1], q[1]) < lo[1] or min(p[1], q[1]) > hi[1]:
            continue
        d = tips - c2                                     # (tw, 2)
        e = q - p
        den = d[:, 0] * e[1] - d[:, 1] * e[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((p[0] - c2[0]) * e[1] - (p[1] - c2[1]) * e[0]) / den       # along the ray, 0..1
            s = ((p[0] - c2[0]) * d[:, 1] - (p[1] - c2[1]) * d[:, 0]) / den  # along the edge, 0..1
        cols = np.nonzero((t > 1e-3) & (t < 1) & (s >= 0) & (s <= 1))[0]
        for j in cols:   # ray height at the crossing: C.z + t (z - C.z) in [zlo, zhi]
            hz = C[2] + t[j] * (z - C[2])
            out[:, j] |= (hz >= zlo) & (hz <= zhi)
    return out


# ---------------------------------------------------------------- tiles and masks
def plan_tiles(walls, panos, data):
    """[(pano, heading, pitch, fov, street)]: per panorama, the tiles that are the most central tile for some
    wall point (1 m grid) it sees within SEE_M and MAX_OBLIQUE."""
    tiles = []
    for p in panos:
        frame = pano_street(p, data)
        name, u, n = frame
        C = cam_centre(p, ZERO, frame)
        fov = FOV[name]
        cands = {}
        for side in (n, -n):
            base = compass(*side)
            for dh in HEAD_OFFS[name]:
                for pt in PITCHES[name]:
                    cands[(round((base + dh) % 360, 1), pt)] = 0
        for w in walls:
            if (C[:2] - w["a"]) @ w["normal"] < 1.0:
                continue
            pts, *_ = wall_grid(w, 1.0)
            d = pts - C
            dist = np.linalg.norm(d[:, :2], axis=1)
            cos = (-(d[:, :2] @ w["normal"])) / dist
            ok = (dist < SEE_M[name]) & (cos > math.cos(math.radians(MAX_OBLIQUE)))
            if not ok.any():
                continue
            hd = np.degrees(np.arctan2(d[ok, 0], d[ok, 1])) % 360
            el = np.degrees(np.arctan2(d[ok, 2], dist[ok]))
            keys = list(cands)
            kh = np.array([k[0] for k in keys])
            kp = np.array([k[1] for k in keys])
            dh = np.abs((hd[:, None] - kh[None] + 180) % 360 - 180)
            dp = np.abs(el[:, None] - kp[None])
            inside = (dh < fov / 2 - 2) & (dp < fov / 2 - 2)
            score = np.where(inside, dh ** 2 + dp ** 2, np.inf)
            best = score.argmin(1)
            for i in best[np.isfinite(score.min(1))]:
                cands[keys[i]] += 1
        for (hd, pt), cnt in cands.items():
            if cnt >= 3:
                tiles.append((p, hd, pt, fov, name))
    return tiles


class Segmenter:
    def __init__(self):
        if WEIGHTS is None:
            raise SystemExit("missing weights/yolo26n-sem-ade20k.pt; run: python sim/seg_standin.py --download")
        import torch
        from ultralytics import YOLO
        self.model = YOLO(str(WEIGHTS))
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        names = self.model.names
        self.ids = {v: k for k, v in names.items()}

    def labels(self, img):
        r = self.model.predict(img, imgsz=max(img.shape[:2]) // 32 * 32 or 640, device=self.device, verbose=False)[0]
        lab = r.semantic_mask.data.cpu().numpy().astype(np.uint8)
        lab = lab[0] if lab.ndim == 3 else lab
        if lab.shape != img.shape[:2]:
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return lab

    def masks(self, img):
        lab = self.labels(img)
        occ = np.isin(lab, [self.ids[n] for n in OCCLUDERS if n in self.ids])
        occ = cv2.dilate(occ.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        sky = lab == self.ids["sky"]
        pole = np.where(lab == self.ids["streetlight"], 2, np.where(lab == self.ids["pole"], 1, 0)).astype(np.uint8)
        return occ, sky, pole


def fetch_tiles(tiles, src, seg):
    shots = []
    SEG_CACHE.mkdir(parents=True, exist_ok=True)
    for p, hd, pt, fov, name in tiles:
        params = {"pano": p["id"], "heading": f"{hd:.1f}", "pitch": f"{pt:g}", "fov": f"{fov:g}",
                  "size": f"{TILE}x{TILE}", "return_error_code": "true"}
        raw = src._get("image", params)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Street View image: the response is not an image")
        key = SEG_CACHE / (hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24] + ".npz")
        z = np.load(key) if key.exists() else None
        if z is not None and ("pole" in z or seg is None):
            occ, sky = z["occ"], z["sky"]
            pole = z["pole"] if "pole" in z else np.zeros(occ.shape, np.uint8)
        else:
            occ, sky, pole = seg.masks(img)
            np.savez_compressed(key, occ=occ, sky=sky, pole=pole)
        occ = occ.copy()
        occ[-LOGO_ROWS:] = True        # the Google logo and copyright line
        shots.append({"pano": p, "heading": hd, "pitch": pt, "fov": fov, "street": name, "img": img,
                      "occ": occ, "sky": sky, "pole": pole, "gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)})
    return shots


# ---------------------------------------------------------------- per-panorama textures, scores
def highpass(tex, valid, sigma):
    """Texture minus its local mean (normalized convolution over valid texels)."""
    v = valid.astype(np.float32)
    num = cv2.GaussianBlur(np.where(valid, tex, 0).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(v, (0, 0), sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(valid & (den > 0.2), tex - num / den, np.nan)


def pano_texture(w, pshots, C, pose, ppm, occ_edges, grays, max_oblique=MAX_OBLIQUE):
    """Gray texture of wall w from one panorama's tiles (most central pixel wins), NaN where unseen/masked."""
    pts, ground, z, tw, th = wall_grid(w, ppm)
    if (C[:2] - w["a"]) @ w["normal"] < 1.0:
        return None
    d3 = pts - C
    dist = np.linalg.norm(d3[:, :2], axis=1)
    cos = -(d3[:, :2] @ w["normal"]) / dist
    front = cos > math.cos(math.radians(max_oblique))
    if not front.any():
        return None
    hid = hidden(C, ground, z, occ_edges, w).ravel()
    best = np.full(len(pts), -1.0)
    val = np.full(len(pts), np.nan, np.float32)
    for s, g in zip(pshots, grays):
        uv = project(C, pose, s["heading"], s["pitch"], s["fov"], pts)
        k = g.shape[0] / TILE
        v, ok = sample(g[..., None], uv[:, 0] * k, uv[:, 1] * k)
        if not ok.any():
            continue
        iu, iv = np.clip(np.nan_to_num(uv[:, 0]), 0, TILE - 1).astype(int), np.clip(np.nan_to_num(uv[:, 1]), 0, TILE - 1).astype(int)
        ok &= ~s["occ"][iv, iu] & ~s["sky"][iv, iu]
        border = np.minimum.reduce([uv[:, 0], TILE - uv[:, 0], uv[:, 1], TILE - uv[:, 1]])
        take = ok & (np.nan_to_num(border, nan=-1) > best)
        val[take] = v[take, 0]
        best[take] = border[take]
    val[~front | hid] = np.nan
    return val.reshape(th, tw)


def ncc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    n = m.sum()
    if n < 30:
        return None, 0
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = math.sqrt(float(x @ x) * float(y @ y))
    return (float(x @ y) / den if den > 0 else 0.0), int(n)


class Scorer:
    """Pairwise agreement of per-panorama wall textures, for pose refinement and verification."""

    def __init__(self, walls, shots, panos, frames, occ, ppm, sigma_m=0.8, max_oblique=MAX_OBLIQUE):
        self.walls, self.panos, self.frames, self.occ, self.ppm = walls, panos, frames, occ, ppm
        self.max_oblique = max_oblique
        self.sigma = sigma_m * ppm
        self.by_pano, self.grays = {}, {}
        f = max(1, int(round(120 / ppm / 1.5)))   # tiles are ~100-200 px/m: a texel averages the pixels it covers
        for s in shots:
            self.by_pano.setdefault(s["pano"]["id"], []).append(s)
            g = cv2.resize(s["gray"], (TILE // f, TILE // f), interpolation=cv2.INTER_AREA) if f > 1 else s["gray"]
            self.grays.setdefault(s["pano"]["id"], []).append(g)
        self.cache = {}

    def texture(self, wi, pid, pose):
        key = (wi, pid, tuple(sorted(pose.items())))
        if key not in self.cache:
            w = self.walls[wi]
            p = self.panos[pid]
            C = cam_centre(p, pose, self.frames[pid])
            t = pano_texture(w, self.by_pano.get(pid, []), C, pose, self.ppm, self.occ, self.grays.get(pid, []),
                             self.max_oblique)
            if t is not None:
                valid = np.isfinite(t)
                t = highpass(np.nan_to_num(t), valid, self.sigma) if valid.mean() > 0.05 else None
            self.cache[key] = t
        return self.cache[key]

    def wall_pairs(self, wi, poses):
        texs = {pid: self.texture(wi, pid, poses[pid]) for pid in self.panos}
        texs = {k: v for k, v in texs.items() if v is not None}
        out = []
        ids = sorted(texs)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                r, n = ncc(texs[a], texs[b])
                if r is not None and n >= MIN_PAIR_M2 * self.ppm ** 2:
                    out.append((a, b, r, n))
        return out

    def pano_score(self, pid, pose, poses, walls_of):
        tot, wsum = 0.0, 0.0
        for wi in walls_of[pid]:
            ta = self.texture(wi, pid, pose)
            if ta is None:
                continue
            for q in self.panos:
                if q == pid:
                    continue
                tb = self.texture(wi, q, poses[q])
                if tb is None:
                    continue
                r, n = ncc(ta, tb)
                if r is not None and n >= MIN_PAIR_M2 * self.ppm ** 2:
                    tot += r * n
                    wsum += n
        return tot / wsum if wsum else 0.0


def refine_poses(scorer, panos, frames, log=print):
    """One lateral shift per street, then coordinate descent per panorama. Returns {pano id: pose}."""
    poses = {pid: dict(ZERO) for pid in panos}
    walls_of = {pid: [wi for wi in range(len(scorer.walls))
                      if scorer.texture(wi, pid, ZERO) is not None] for pid in panos}
    def total():
        return float(np.mean([scorer.pano_score(pid, poses[pid], poses, walls_of) for pid in panos]))
    log(f"  poses: raw Google positions score {total():.3f}")
    base = {}
    for street, rng in (("moss", np.arange(-1.0, 4.01, 0.25)), ("howard", np.arange(-3.0, 3.01, 0.5))):
        ids = [pid for pid in panos if frames[pid][0] == street]
        best = (None, -2)
        for dp in rng:
            for pid in ids:
                poses[pid]["perp"] = float(dp)
            sc = float(np.mean([scorer.pano_score(pid, poses[pid], poses, walls_of) for pid in ids]))
            if sc > best[1]:
                best = (float(dp), sc)
        for pid in ids:
            poses[pid]["perp"] = best[0]
        base[street] = best[0]
        log(f"  poses: {street} shared lateral shift {best[0]:+.2f} m, score {best[1]:.3f}")
    steps = {"perp": 0.25, "along": 0.25, "dz": 0.2, "yaw": 0.5, "tx": 0.5, "ty": 0.5}
    bounds = {"perp": 2.5, "along": 3.0, "dz": 0.6, "yaw": 4.0, "tx": 3.0, "ty": 3.0}
    # Coarse: each panorama's own lateral and along-street offset on a 0.5 m grid (GPS along-track error is
    # independent per panorama and a few metres of it leaves no gradient for the descent below).
    for rnd in range(2):
        for pid in panos:
            if not walls_of[pid]:
                continue
            b0 = base.get(frames[pid][0], 0.0)
            best = (dict(poses[pid]), scorer.pano_score(pid, poses[pid], poses, walls_of))
            for dp in np.arange(b0 - 2.0, b0 + 2.01, 0.5):
                for da in np.arange(-3.0, 3.01, 0.5):
                    trial = dict(poses[pid], perp=float(dp), along=float(da))
                    sc = scorer.pano_score(pid, trial, poses, walls_of)
                    if sc > best[1] + 1e-4:
                        best = (trial, sc)
            poses[pid] = best[0]
        log(f"  poses: coarse grid round {rnd + 1} score {total():.3f}")
    for rnd, scale in enumerate((1.0, 1.0, 0.5)):
        for pid in panos:
            if not walls_of[pid]:
                continue
            cur = scorer.pano_score(pid, poses[pid], poses, walls_of)
            for k in steps:
                for sgn in (1, -1):
                    while True:
                        trial = dict(poses[pid])
                        trial[k] = round(trial[k] + sgn * steps[k] * scale, 3)
                        ref = base.get(frames[pid][0], 0.0) if k == "perp" else 0.0
                        if abs(trial[k] - ref) > bounds[k]:
                            break
                        sc = scorer.pano_score(pid, trial, poses, walls_of)
                        if sc <= cur + 1e-4:
                            break
                        poses[pid], cur = trial, sc
        log(f"  poses: round {rnd + 1} score {total():.3f}")
    return poses


def verify(scorer, poses):
    """Per wall: (score, pairs, panoramas, scope). Score = overlap-weighted mean NCC over panorama pairs of the
    wall; a wall too small to compare on its own (bay facets) takes the score of its building's whole street
    front (all its walls on that street pooled), scope "front"."""
    per = []
    for wi in range(len(scorer.walls)):
        pairs = scorer.wall_pairs(wi, poses)
        per.append((sum(p[2] * p[3] for p in pairs), sum(p[3] for p in pairs), len(pairs),
                    sorted({x for p in pairs for x in p[:2]})))
    fronts = {}
    for w, (rn, n, k, ids) in zip(scorer.walls, per):
        f = fronts.setdefault((w["building"], w["street"]), [0.0, 0, 0])
        f[0] += rn
        f[1] += n
        f[2] += k
    out = []
    for w, (rn, n, k, ids) in zip(scorer.walls, per):
        if n:
            out.append((round(rn / n, 3), k, ids, "wall"))
            continue
        f = fronts[(w["building"], w["street"])]
        out.append((round(f[0] / f[1], 3), f[2], [], "front") if f[1] else (None, 0, [], None))
    return out


# ---------------------------------------------------------------- poles
POLE_MIN_PX = 60        # a pole must cover this many rows of a tile column
POLE_CELL = 0.25
POLE_MIN_PANOS = 3
POLE_MIN_H = 5.0        # sign posts and meters are shorter; utility poles and street lights are 8-12 m
POLE_CURB_M = 1.5


def pixel_ray(C, pose, s, u, v):
    """World direction through pixel (u, v) of shot s: the inverse of project()."""
    h, p = math.radians(s["heading"] + pose["yaw"]), math.radians(s["pitch"])
    level, up = np.array([math.sin(h), math.cos(h), 0.0]), np.array([0.0, 0.0, 1.0])
    fwd = math.cos(p) * level + math.sin(p) * up
    right = np.array([math.cos(h), -math.sin(h), 0.0])
    down = math.sin(p) * level - math.cos(p) * up
    f = 0.5 * TILE / math.tan(math.radians(s["fov"]) / 2)
    d = fwd + (u - TILE / 2) / f * right + (v - TILE / 2) / f * down
    if pose["tx"] or pose["ty"]:
        d = tilt_matrix(pose["tx"], pose["ty"]).T @ d
    return d / np.linalg.norm(d)


def find_poles(shots, poses, frames, data):
    """Utility poles and street lights: each tall run of ADE20K pole / streetlight pixels in a tile is a ray
    from its panorama; cells on a sidewalk where rays from >= POLE_MIN_PANOS panoramas meet are poles."""
    rays = []
    for s in shots:
        m = s["pole"] > 0
        cols = m.sum(0)
        on = cols >= POLE_MIN_PX
        j = 0
        while j < TILE:
            if not on[j]:
                j += 1
                continue
            k = j
            while k < TILE and on[k]:
                k += 1
            u = (j + k - 1) / 2
            rows = np.nonzero(m[:, j:k].any(1))[0]
            kind = "streetlight" if (s["pole"][:, j:k] == 2).sum() > 0.3 * m[:, j:k].sum() else "utility"
            pid = s["pano"]["id"]
            pose = poses.get(pid, ZERO)
            C = cam_centre(s["pano"], pose, frames[pid])
            top = pixel_ray(C, pose, s, u, rows.min())
            mid = pixel_ray(C, pose, s, u, (rows.min() + rows.max()) / 2)
            rays.append((pid, C, mid, top, kind, rows.min() <= 2))
            j = k
    lo = np.array([-80.0, -150.0])
    nx, ny = int(220 / POLE_CELL), int(220 / POLE_CELL)
    votes = {}
    for i, (pid, C, d, *_ ) in enumerate(rays):
        h = d[:2] / np.linalg.norm(d[:2])
        side = np.array([-h[1], h[0]])
        for t in np.arange(1.5, 25.0, POLE_CELL / 2):
            for o in (-POLE_CELL, 0.0, POLE_CELL):   # rays from different panoramas miss by a few decimetres
                q = C[:2] + h * t + side * o * (1 + t / 10)
                c = tuple(((q - lo) / POLE_CELL).astype(int))
                if 0 <= c[0] < nx and 0 <= c[1] < ny:
                    votes.setdefault(c, set()).add(pid)
    curbs = np.array(data["curbs"], float)
    def sidewalk(p):   # poles stand at the curb: within POLE_CURB_M of a curb line, not inside a building
        a, b = curbs[:, :2], curbs[:, 2:]
        e = b - a
        t = np.clip(((p - a) * e).sum(1) / np.maximum((e * e).sum(1), 1e-9), 0, 1)
        d = np.linalg.norm(a + t[:, None] * e - p, axis=1).min()
        return d < POLE_CURB_M and not any(_inside(p, b["pts"]) for b in data["buildings"])
    cand = sorted(((len(v), c) for c, v in votes.items() if len(v) >= POLE_MIN_PANOS), reverse=True)
    poles = []
    for n, c in cand:
        p = lo + (np.array(c) + 0.5) * POLE_CELL
        if any(np.linalg.norm(p - q["xy"]) < 4.0 for q in poles) or not sidewalk(p):
            continue
        near = [r for r in rays if _ray_dist(r[1], r[2], p) < 0.8]
        hs = []
        for pid, C, d, top, kind, clipped in near:
            dist = np.linalg.norm(p - C[:2])
            if not clipped:
                hs.append(C[2] + dist * top[2] / np.linalg.norm(top[:2]))
        hgt = float(max(hs)) if hs else None   # canopies hide pole tops: the highest unclipped view
        kinds = [r[4] for r in near]
        if hgt is not None and hgt < POLE_MIN_H:
            continue
        poles.append({"xy": p, "x": round(float(p[0]), 2), "y": round(float(p[1]), 2), "panos": n,
                      "h": None if hgt is None else round(hgt, 1),
                      "kind": max(set(kinds), key=kinds.count), "light": "streetlight" in kinds})
    for q in poles:
        del q["xy"]
    return poles


def _inside(p, poly):
    c = False
    for a, b in zip(poly, poly[1:]):
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
            c = not c
    return c


def _ray_dist(C, d, p):
    h = d[:2] / np.linalg.norm(d[:2])
    v = p - C[:2]
    t = v @ h
    return float(np.linalg.norm(v - t * h)) if t > 0 else 1e9


def _in_quad(p, q):
    s = [(q[(i + 1) % 4][0] - q[i][0]) * (p[1] - q[i][1]) - (q[(i + 1) % 4][1] - q[i][1]) * (p[0] - q[i][0]) for i in range(4)]
    return all(v >= 0 for v in s) or all(v <= 0 for v in s)


# ---------------------------------------------------------------- final textures
def wall_texture(w, shots, frames, poses, occ_edges, ppm=None):
    """Colour texture at PX_PER_M (or ppm), clean coverage, inpainted share, the panoramas used."""
    ppm = ppm or PX_PER_M[w["street"]]
    pts, ground, z, tw, th = wall_grid(w, ppm)
    acc = np.zeros((th * tw, 3), np.float32)
    wsum = np.zeros(th * tw, np.float32)
    clean = np.zeros(th * tw, bool)
    used = set()
    hid_cache = {}
    for s in shots:
        p = s["pano"]
        pose = poses.get(p["id"], ZERO)
        C = cam_centre(p, pose, frames[p["id"]])
        if (C[:2] - w["a"]) @ w["normal"] < 1.0:
            continue
        uv = project(C, pose, s["heading"], s["pitch"], s["fov"], pts)
        inb = np.isfinite(uv[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] < TILE) & (uv[:, 1] >= 0) & (uv[:, 1] < TILE)
        if inb.mean() < 0.01:
            continue
        if p["id"] not in hid_cache:
            hid_cache[p["id"]] = hidden(C, ground, z, occ_edges, w).ravel()
        vals, ok = sample(s["img"].astype(np.float32), uv[:, 0], uv[:, 1])
        ok &= ~hid_cache[p["id"]]
        iu = np.clip(np.nan_to_num(uv[:, 0]), 0, TILE - 1).astype(int)
        iv = np.clip(np.nan_to_num(uv[:, 1]), 0, TILE - 1).astype(int)
        bad = s["occ"][iv, iu] | s["sky"][iv, iu]
        d3 = pts - C
        dist = np.linalg.norm(d3, axis=1)
        cos = np.clip(-(d3[:, :2] @ w["normal"]) / np.linalg.norm(d3[:, :2], axis=1), 0, 1)
        ok &= cos > math.cos(math.radians(MAX_OBLIQUE))
        border = np.clip(np.minimum.reduce([uv[:, 0], TILE - uv[:, 0], uv[:, 1], TILE - uv[:, 1]]) / 40, 0, 1)
        wt = ((cos / dist) ** 2 * np.nan_to_num(border)) ** WEIGHT_POW
        wt = np.where(ok, np.where(bad, wt * MASKED_W, wt), 0).astype(np.float32)
        if wt.sum() == 0:
            continue
        acc += vals * wt[:, None]
        wsum += wt
        clean |= ok & ~bad
        used.add(p["id"])
    seen = wsum > 0
    tex = np.zeros((th * tw, 3), np.float32)
    tex[seen] = acc[seen] / wsum[seen, None]
    if seen.any():
        tex[~seen] = np.median(tex[seen], axis=0)
    tex = tex.reshape(th, tw, 3).clip(0, 255).astype(np.uint8)
    holes = (~clean).reshape(th, tw)
    if holes.any() and clean.any():   # cars, trees, poles every view had in front: inpaint at a lower res
        f = max(1, int(ppm // 24))
        small = cv2.resize(tex, (max(2, tw // f), max(2, th // f)), interpolation=cv2.INTER_AREA)
        hs = cv2.resize(holes.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        fill = cv2.resize(cv2.inpaint(small, hs, 5, cv2.INPAINT_TELEA), (tw, th), interpolation=cv2.INTER_LINEAR)
        soft = cv2.GaussianBlur(holes.astype(np.float32), (0, 0), 2)[..., None]
        tex = (tex * (1 - soft) + fill * soft).astype(np.uint8)
    return tex, float(clean.mean()), float(holes.mean()), sorted(used)


def plain_colour(w, shots, frames, poses, occ_edges):
    """Median colour (0-1 RGB) of a wall's clean texels, for walls whose photo didn't verify: a colour can't
    put the wrong window in the wrong place."""
    tex, clean, _, _ = wall_texture(w, shots, frames, poses, occ_edges, ppm=6)
    px = tex.reshape(-1, 3)
    return [round(float(v) / 255, 3) for v in np.median(px, axis=0)[::-1]] if clean > 0.05 else None


def photo_roofline(w, shots, frames, poses, occ_edges, above=4.0, ppm=8):
    """Height where sky starts over a wall in the photos (median over columns), or None: the walls are drawn
    up to z1 + `above` and each texel is sky or not in the most central view."""
    ww = dict(w)
    pts, ground, z, tw, th = wall_grid(ww, ppm, z_top=w["z1"] + above)
    votes = np.zeros(len(pts))
    n = np.zeros(len(pts))
    for s in shots:
        p = s["pano"]
        pose = poses.get(p["id"], ZERO)
        C = cam_centre(p, pose, frames[p["id"]])
        if (C[:2] - w["a"]) @ w["normal"] < 2.0:
            continue
        uv = project(C, pose, s["heading"], s["pitch"], s["fov"], pts)
        ok = np.isfinite(uv[:, 0]) & (uv[:, 0] >= 8) & (uv[:, 0] < TILE - 8) & (uv[:, 1] >= 8) & (uv[:, 1] < TILE - 8)
        iu = np.clip(np.nan_to_num(uv[:, 0]), 0, TILE - 1).astype(int)
        iv = np.clip(np.nan_to_num(uv[:, 1]), 0, TILE - 1).astype(int)
        votes += ok & s["sky"][iv, iu]
        n += ok & ~s["occ"][iv, iu]
    frac = np.where(n > 0, votes / np.maximum(n, 1), np.nan).reshape(th, tw)
    tops = []
    for j in range(tw):
        col = frac[:, j]   # top row first
        k = 0
        while k < th and np.isfinite(col[k]) and col[k] > 0.5:
            k += 1
        if 0 < k < th and np.isfinite(col[:k]).all():
            tops.append(z[k - 1] + 0.5 / ppm)   # first non-sky texel from the top
    return round(float(np.median(tops)), 2) if len(tops) >= max(3, tw // 3) else None


def openings(seg, img, L, z0, z1):
    """Window and door rectangles [kind, s0, zb, s1, zt] (metres along the wall from a, up from the ground)
    from ADE20K windowpane / door masks on the rectified texture."""
    th, tw = img.shape[:2]
    ppm = tw / L
    f = min(1.0, 1280 / max(th, tw))
    small = cv2.resize(img, (max(32, int(tw * f)), max(32, int(th * f))), interpolation=cv2.INTER_AREA)
    lab = seg.labels(small)
    out = []
    for name, kind in OPENINGS.items():
        if name not in seg.ids:
            continue
        m = (lab == seg.ids[name]).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            wm, hm = w / f / ppm, h / f / ppm
            if not (0.5 <= wm <= 6 and 0.6 <= hm <= 4.0) or area < 0.55 * w * h:
                continue
            zt = z1 - y / f / ppm
            zb = zt - hm
            k = kind
            if kind == "door" and wm > 2.2:
                k = "garage"
            if kind == "window" and zb < z0 + 0.3 and hm > 1.9 and z0 == 0.0:
                k = "door" if wm <= 2.2 else "garage"
            out.append([k, round(x / f / ppm, 2), round(max(zb, z0), 2), round((x + w) / f / ppm, 2), round(min(zt, z1), 2)])
    keep = []
    for r in sorted(out, key=lambda r: -(r[3] - r[1]) * (r[4] - r[2])):
        if all(min(r[3], o[3]) - max(r[1], o[1]) <= 0.05 or min(r[4], o[4]) - max(r[2], o[2]) <= 0.05 for o in keep):
            keep.append(r)
    return keep


# ---------------------------------------------------------------- main
def load_panos(data, src):
    path = CACHE.parent / "panos.json"
    if path.exists():
        return json.loads(path.read_text())
    panos = find_panos(data, src)
    path.write_text(json.dumps(panos))
    return panos


def run(data_path, src, seg, log=print, refine=True, write=True):
    data = json.loads(Path(data_path).read_text())
    walls = street_walls(data)
    panos = load_panos(json.loads(MOSS.read_text()), src)
    frames = {p["id"]: pano_street(p, data) for p in panos}
    tiles = plan_tiles(walls, panos, data)
    log(f"{len(walls)} street walls ({sum(isinstance(w['edge'], str) for w in walls)} ground floors under bays), "
        f"{len(panos)} Google panoramas, {len(tiles)} tiles")
    if not write and not refine and seg is None:
        return {"tiles": len(tiles)}
    shots = fetch_tiles(tiles, src, seg)
    occ = occluders(data)
    pmap = {p["id"]: p for p in panos}
    ref = Scorer(walls, shots, pmap, frames, occ, REF_PPM, max_oblique=REFINE_OBLIQUE)
    pose_file = CACHE.parent / "poses.json"
    if refine and "--reuse-poses" in sys.argv and pose_file.exists():
        poses = json.loads(pose_file.read_text())
    elif refine:
        poses = refine_poses(ref, pmap, frames, log)
        pose_file.write_text(json.dumps(poses))
    else:
        poses = {pid: dict(ZERO) for pid in pmap}
    sc = Scorer(walls, shots, pmap, frames, occ, SCORE_PPM, max_oblique=REFINE_OBLIQUE)
    scores = verify(sc, poses)
    result = {"walls": [], "poses": poses, "tiles": len(tiles), "panos": len(panos),
              "poles": find_poles(shots, poses, frames, data)}
    log(f"  poles: {len(result['poles'])} found: " + ", ".join(f"({q['x']}, {q['y']}) {q['kind']} {q['h']} m "
                                                            f"{q['panos']} panoramas" for q in result["poles"]))
    for w, (score, npairs, used, scope) in zip(walls, scores):
        entry = {"building": w["building"], "edge": w["edge"], "a": w["a"].round(3).tolist(),
                 "b": w["b"].round(3).tolist(), "z0": w["z0"], "h": w["z1"], "normal": w["normal"].round(4).tolist(),
                 "street": w["street"], "score": score, "pairs": npairs, "score_scope": scope, "file": None,
                 "h_max": w["h_max"]}
        result["walls"].append(entry)
    if not write:
        return result
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.jpg"):
        f.unlink()
    for w, entry in zip(walls, result["walls"]):
        L = float(np.linalg.norm(w["b"] - w["a"]))
        entry["roof_photo"] = photo_roofline(w, shots, frames, poses, occ) if w["z0"] < w["z1"] and not isinstance(w["edge"], str) else None
        ok = entry["score"] is not None and entry["score"] >= SCORE_MIN
        if not ok:
            entry["plain_rgb"] = plain_colour(w, shots, frames, poses, occ)
            log(f"  {w['building']}_{w['edge']}: {L:4.1f} m, score {entry['score']} ({entry['pairs']} pairs) -> plain "
                f"{entry['plain_rgb']}")
            continue
        tex, clean, holes, used = wall_texture(w, shots, frames, poses, occ)
        entry.update(coverage=round(clean, 3), inpainted=round(holes, 3), views=used)
        if clean < MIN_COVER:
            log(f"  {w['building']}_{w['edge']}: clean cover {clean:.0%} -> plain")
            continue
        entry["file"] = f"{w['building']}_{w['edge']}.jpg"
        cv2.imwrite(str(OUT / entry["file"]), tex, [cv2.IMWRITE_JPEG_QUALITY, 85])
        entry["openings"] = openings(seg, tex, L, w["z0"], w["z1"])
        top = tex[: max(1, tex.shape[0] // 12)].reshape(-1, 3)
        entry["top_rgb"] = [round(float(v) / 255, 3) for v in np.median(top, axis=0)[::-1]]
        log(f"  {entry['file']}: {L:4.1f} x {w['z1'] - w['z0']:4.1f} m, score {entry['score']:.2f}, clean {clean:.0%}, "
            f"inpainted {holes:.0%}, {len(entry['openings'])} openings, roof data {w['z1']} photo {entry['roof_photo']}")
    return result


def main():
    src = StreetViewSource(single=(37.7783, -122.4087, 0), cache_dir=CACHE)
    if "--plain-colours" in sys.argv:   # add plain_rgb to an existing facades.json (no new requests)
        man = json.loads((OUT / "facades.json").read_text())
        data = json.loads(MOSS.read_text())
        walls = {(w["building"], str(w["edge"])): w for w in street_walls(data)}
        panos = load_panos(data, src)
        frames = {p["id"]: pano_street(p, data) for p in panos}
        shots = fetch_tiles(plan_tiles(list(walls.values()), panos, data), src, None)
        occ = occluders(data)
        for e in man["walls"]:
            if not e.get("file"):
                e["plain_rgb"] = plain_colour(walls[(e["building"], str(e["edge"]))], shots, frames, man["poses"], occ)
        (OUT / "facades.json").write_text(json.dumps(man, indent=1, default=float))
        print(f"plain colours for {sum(1 for e in man['walls'] if e.get('plain_rgb'))} walls")
        return
    if "--plan" in sys.argv:
        run(MOSS, src, None, refine=False, write=False)
        return
    seg = Segmenter()
    if "--before" in sys.argv:   # the #63 geometry with raw Google positions, scored on the same tiles
        old = sys.argv[sys.argv.index("--before") + 1]
        res = run(old, src, seg, refine=False, write=False)
        (CACHE.parent / "scores_before.json").write_text(json.dumps(res, default=str))
        sc = [w["score"] for w in res["walls"] if w["score"] is not None]
        print(f"before: {len(res['walls'])} walls, {len(sc)} verifiable, median score {np.median(sc):.3f}, "
              f">= {SCORE_MIN}: {sum(s >= SCORE_MIN for s in sc)}")
        return
    res = run(MOSS, src, seg, write="--score-only" not in sys.argv)
    r = src.requests
    res["requests"] = r
    if "--score-only" not in sys.argv:
        (OUT / "facades.json").write_text(json.dumps(res, indent=1, default=float))
    else:
        (CACHE.parent / "scores_after.json").write_text(json.dumps(res, default=float))
    sc = [w["score"] for w in res["walls"] if w["score"] is not None]
    print(f"textured {sum(1 for w in res['walls'] if w.get('file'))}/{len(res['walls'])} walls; median score "
          f"{np.median(sc):.3f}; Street View requests: {r['image']} image (billed), {r['metadata']} metadata (free), "
          f"{r['cached']} from cache")


if __name__ == "__main__":
    main()
