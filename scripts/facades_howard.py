"""Verified facade textures for the Howard St walls at the Moss/Howard corner, from Google Street View
(permission: docs/google-permission.md; this street only, derived imagery stays local and is never committed).

    uv run --no-project --with numpy --with opencv-python-headless --with ultralytics \
        python scripts/facades_howard.py [--env .env] [--out sim/public/local/facades_howard] [--weights weights]

Writes <out>/<building>_<wall>.jpg, manifest.json (kept walls) and report.json (every wall, scores, requests).
Raw API responses are cached in /tmp/kite_sv_howard/cache so a rerun is free; delete that folder when done.

Walls (WALLS below, picked by hand from moss.json): the Howard-facing walls of the buildings on the SE side of
Howard at the corner and where the route ends, and the NW side across Howard. wall_index k is the footprint edge
from buildings[i].pts[k] to pts[k+1] in moss.json (pts is closed: last == first), the same order
sim/src/osmWorld.ts walks. Texture: column 0 is the wall's left end as seen from the street (outside), row 0 the
roofline (moss.json h). corners_local_m: [x east, y north, z up] metres, bottom-left, bottom-right, top-right,
top-left as seen from outside, so texture u runs BL->BR and v runs bottom->top.

1. Google panoramas (metadata, free) every 5 m along Howard, inside kite.sources.streetview.SCOPE.
2. Each wall is cut into <= CHUNK_M pieces; per piece the PER_TARGET most frontal, closest unblocked panoramas get
   one Static view (640x640, billed) aimed at its centre, pitched so the wall spans the frame. Views are shared.
3. Masks: YOLO11n boxes (people, vehicles) plus ADE20K semantic pixels (trees, plants, poles, signs, cars).
4. Per wall, the panorama of the second-best view is held out entirely (none of its views go into that wall).
5. Pose: Google snaps panoramas to its road line, metres off the lane the car drove in, and headings are off a
   degree or so. Per panorama (along Howard, across, yaw, camera height): the across offset by grid search
   (LATERAL_M; one shared, then each), then all four by coordinate descent, maximising the edge NCC between
   the same wall rectified (facade_rectify.project) from different panoramas, never pairing a wall with its
   held-out panorama. So the held-out pose comes from the other walls only.
6. Texture (PX_PER_M): views blended by ((frontal cos / distance)^2 * border falloff)^BLEND_POW, masked texels skipped,
   holes inpainted.
7. Verify: the texture is warped into the held-out image (plane homography) and the edge maps compared (NCC) over
   unmasked, unoccluded, observed pixels, best over slides within SLACK_M (reported as offset_m). The null is the
   best score, same search, of every other wall's texture and of this texture mirrored: the wrong-wall and
   flipped-wall mistakes. Kept if score >= MIN_SCORE and score - null >= MIN_MARGIN.
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
from facade_rectify import Camera, project, sample  # noqa: E402
from kite.gps import latlng_to_sim, sim_to_latlng  # noqa: E402
from kite.sources.streetview import SV_CAM_HEIGHT_M, StreetViewSource, _is_google, _require_scope  # noqa: E402

MOSS = ROOT / "sim/public/osm/moss.json"
CACHE = Path("/tmp/kite_sv_howard/cache")
DEBUG = Path("/tmp/kite_sv_howard/debug")
WALLS = [(260351187, 0), (260354468, 4), (260354474, 8),                       # SE side of Howard
         (260345917, 3), (942678853, 0), (256469977, 6), (256469961, 3),       # NW side, across Howard
         (256469968, 1), (256469963, 1), (256469979, 2), (256469967, 5)]
HOWARD_A, HOWARD_B = (36.5, 48.8), (-46.9, -34.2)   # moss.json Howard centreline points either side of Moss
MOSS_JOIN = np.array([-6.27, 6.39])                 # Moss meets Howard
ALONG_RANGE, PANO_STEP_M = (-35.0, 65.0), 5.0
SIZE, PX_PER_M, MAX_W, LOW_PPM = 640, 100, 4096, 10
CHUNK_M, PER_TARGET, SEE_M, MIN_COS = 12.0, 3, 30.0, 0.4
FOV_MIN, FOV_MAX = 40, 70
MIN_COVER, MIN_SCORE, MIN_MARGIN = 0.5, 0.25, 0.08
BLEND_POW = 4   # 12 (near winner-take-all) verified worse here: 0 walls kept
LATERAL_M = 6.0   # across-Howard search for each panorama's true lane, either way
SLACK_M, VER_PPM, LOW_POSE_PPM = (1.5, 0.75), 20, 12   # verify slack: along, up (m)
MASK_BOX = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
MASK_SEM = {"tree", "plant", "person", "car", "truck", "bus", "van", "bicycle", "minibike", "pole",
            "streetlight", "signboard", "traffic light", "palm"}


# ------------------------------------------------------------------ geometry
# unit, compass, blocked: copied from an earlier scripts/streetview_facades.py (shared script, since rewritten).
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def compass(dx, dy):
    return math.degrees(math.atan2(dx, dy)) % 360


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


def howard_frame():
    u = unit(np.subtract(HOWARD_B, HOWARD_A))
    return u, np.array([-u[1], u[0]])   # along (towards SW), across (towards SE)


def load_walls(data):
    by_id = {b["id"]: b for b in data["buildings"]}
    out = []
    for bid, k in WALLS:
        b = by_id[bid]
        pts = np.array(b["pts"], float)
        area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
        p, q = pts[k], pts[k + 1]
        e = unit(q - p)
        nrm = np.array([e[1], -e[0]]) if area > 0 else np.array([-e[1], e[0]])   # outward
        right = np.array([-nrm[1], nrm[0]])          # viewer's right, facing the wall from outside
        left, rgt = (p, q) if (q - p) @ right > 0 else (q, p)
        out.append({"bid": bid, "k": k, "left": left, "right": rgt, "h": float(b["h"]), "nrm": nrm,
                    "L": float(np.linalg.norm(q - p)), "key": f"{bid}_{k}"})
    return out


def texel_points(w, ppm):
    tw, th = max(2, round(w["L"] * ppm)), max(2, round(w["h"] * ppm))
    s = (np.arange(tw) + 0.5) / tw
    z = w["h"] * (1 - (np.arange(th) + 0.5) / th)
    ground = w["left"][None] + s[:, None] * (w["right"] - w["left"])[None]
    pts = np.concatenate([np.repeat(ground[None], th, 0), np.repeat(z[:, None, None], tw, 1)], -1)
    return pts, ground, (th, tw)


def corners3d(w):
    (lx, ly), (rx, ry), h = w["left"], w["right"], w["h"]
    return [[lx, ly, 0.0], [rx, ry, 0.0], [rx, ry, h], [lx, ly, h]]   # BL, BR, TR, TL from outside


def camera(view, pose):
    """pose: {pano id: [along Howard m, across m, yaw deg, dz m]} corrections to the metadata pose."""
    p, (hu, hn) = view["pano"], howard_frame()
    da, dp, dyaw, dz = pose.get(p["id"], (0.0, 0.0, 0.0, 0.0))
    x, y = np.array([p["x"], p["y"]]) + da * hu + dp * hn
    return Camera(x=x, y=y, z=SV_CAM_HEIGHT_M + dz, heading=view["heading"] + dyaw, pitch=view["pitch"],
                  focal=0.5 / math.tan(math.radians(view["fov"]) / 2), width=SIZE, height=SIZE)


# ------------------------------------------------------------------ Street View
def find_panos(src):
    hu, hn = howard_frame()
    panos = {}
    for t in np.arange(*ALONG_RANGE, PANO_STEP_M):
        for side in (-3.0, 3.0):
            x, y = MOSS_JOIN + t * hu + side * hn
            try:
                m = src._metadata(*sim_to_latlng(x, y))
            except ValueError:   # outside SCOPE
                continue
            if not m or not _is_google(m):
                continue
            px, py = latlng_to_sim(m["location"]["lat"], m["location"]["lng"])
            if abs((np.array([px, py]) - MOSS_JOIN) @ hn) < 8.5:   # in Howard's roadway, not up Moss
                panos[m["pano_id"]] = {"id": m["pano_id"], "x": float(px), "y": float(py), "date": m.get("date"),
                                       "lat": m["location"]["lat"], "lng": m["location"]["lng"]}
    return list(panos.values())


def plan_views(walls, panos, edges):
    views = []
    for w in walls:
        n = max(1, math.ceil(w["L"] / CHUNK_M))
        for i in range(n):
            tgt = w["left"] + (i + 0.5) / n * (w["right"] - w["left"])
            half = w["L"] / n / 2
            cands = []
            for p in panos:
                c = np.array([p["x"], p["y"]])
                d = c - tgt
                dist = float(np.linalg.norm(d))
                cos = d @ w["nrm"] / dist
                if dist > SEE_M or cos < MIN_COS or blocked(c, (tgt + 0.15 * w["nrm"])[None], edges,
                                                            (w["bid"], w["k"]))[0]:
                    continue
                cands.append((cos ** 2 / dist, p, dist, cos))
            for _, p, dist, cos in sorted(cands, key=lambda t: -t[0])[:PER_TARGET]:
                c = np.array([p["x"], p["y"]])
                lo = math.degrees(math.atan2(-SV_CAM_HEIGHT_M, dist))
                hi = math.degrees(math.atan2(w["h"] - SV_CAM_HEIGHT_M, dist))
                hspan = 2 * math.degrees(math.atan2(half * math.sqrt(max(0.0, 1 - cos ** 2)) + half * cos, dist))
                fov = int(np.clip(max(hi - lo, hspan) * 1.1, FOV_MIN, FOV_MAX))
                pitch = int(round(np.clip((hi + lo) / 2, 0, 35)))
                hd = round(compass(*(tgt - c)), 1)
                dup = next((v for v in views if v["pano"]["id"] == p["id"]
                            and abs((v["heading"] - hd + 180) % 360 - 180) < 8 and abs(v["pitch"] - pitch) < 8), None)
                if dup:
                    dup["fov"] = max(dup["fov"], fov)
                else:
                    views.append({"pano": p, "heading": hd, "pitch": pitch, "fov": fov})
    return views


def fetch(src, v):
    _require_scope(v["pano"]["lat"], v["pano"]["lng"])
    raw = src._get("image", {"pano": v["pano"]["id"], "heading": f"{v['heading']:.1f}", "pitch": f"{v['pitch']}",
                             "fov": f"{v['fov']}", "size": f"{SIZE}x{SIZE}", "return_error_code": "true"})
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Street View image: the response is not an image")
    return img


def masks(imgs, weights):
    """Per image a float mask, 1 = facade to keep, 0 = person/vehicle box or tree/pole/sign pixel."""
    from ultralytics import YOLO

    det, sem = YOLO(str(weights / "yolo11n.pt")), YOLO(str(weights / "yolo26n-sem-ade20k.pt"))
    out = []
    for img in imgs:
        m = np.ones(img.shape[:2], np.uint8)
        r = det(img, verbose=False, conf=0.25)[0]
        for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            if r.names[c] in MASK_BOX:
                m[max(0, int(y0) - 4):int(y1) + 4, max(0, int(x0) - 4):int(x1) + 4] = 0
        s = sem(img, verbose=False)[0]
        cls = _sem_classes(s, img.shape[:2])
        if cls is None:
            raise RuntimeError("semantic model: no per-pixel classes in the result (ultralytics version?)")
        bad = np.isin(cls, [i for i, nm in s.names.items() if nm in MASK_SEM])
        m[cv2.dilate(bad.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0] = 0
        out.append(m.astype(float))
    return out


def _sem_classes(res, shape):
    """Class index per pixel from an ultralytics semantic result, whatever attribute this version uses."""
    for name in ("semantic_mask", "sem_mask", "masks"):
        v = getattr(res, name, None)
        if v is None:
            continue
        a = getattr(v, "data", v)
        a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
        a = a.squeeze()
        if a.ndim == 3:   # per-class scores
            a = a.argmax(0)
        return cv2.resize(a.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST)
    return None


# ------------------------------------------------------------------ rectify / align / blend
_BLOCKED: dict = {}


def rectify(w, view, img, mask, edges, ppm, pose):
    """Wall texels seen from one view: (values (th, tw, 3), usable (th, tw), weight (th, tw))."""
    cam = camera(view, pose)
    pts, ground, (th, tw) = texel_points(w, ppm)
    c = np.array([cam.x, cam.y])
    if (c - w["left"]) @ w["nrm"] < 1.0:
        z = np.zeros((th, tw))
        return np.zeros((th, tw, 3)), z.astype(bool), z
    flat = pts.reshape(-1, 3)
    uv = project(cam, flat)
    vals, ok = sample(img.astype(float), uv[:, 0], uv[:, 1])
    mv, _ = sample(mask[..., None], uv[:, 0], uv[:, 1])
    key = (w["key"], view["pano"]["id"], tw)
    if key not in _BLOCKED:   # columns hidden behind another footprint; ponytail: from the first pose asked (open street,
        # so lane shifts barely change it), recompute per pose if occlusion edges look wrong
        tips = ground - 0.15 * (ground - c) / np.linalg.norm(ground - c, axis=1)[:, None]
        _BLOCKED[key] = blocked(c, tips, edges, (w["bid"], w["k"]))
    ok &= ~np.tile(_BLOCKED[key], th)
    d3 = flat - np.array([cam.x, cam.y, cam.z])
    dist = np.linalg.norm(d3, axis=1)
    cos = np.clip(-(d3[:, :2] @ w["nrm"]) / dist, 0, 1)
    border = np.clip(np.minimum.reduce([uv[:, 0], SIZE - uv[:, 0], uv[:, 1], SIZE - uv[:, 1]]) / 40, 0, 1)
    good = ok & (np.nan_to_num(mv[:, 0]) > 0.5)
    wt = np.where(good, ((cos / dist) ** 2 * np.nan_to_num(border)) ** BLEND_POW, 0)
    return vals.reshape(th, tw, 3), good.reshape(th, tw), wt.reshape(th, tw)


def edges_of(gray):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def ncc(a, b, valid):
    if valid.sum() < 200:
        return 0.0
    x, y = a[valid].astype(float), b[valid].astype(float)
    x, y = x - x.mean(), y - y.mean()
    den = math.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else 0.0


def gray(v):
    return np.asarray(v, float) @ np.array([0.114, 0.587, 0.299])


def erode(m, r=2):
    return cv2.erode(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


def refine(walls, views, imgs, mks, edges, held, pose):
    """Coordinate descent on per-panorama [along, across, yaw, dz]. Objective: overlap-weighted edge NCC between
    every pair of views from different panoramas of the same wall, held-out panorama excluded, minus a small pull
    towards the metadata pose. -> (objective before, after)."""
    pid = [v["pano"]["id"] for v in views]
    seen = {w["key"]: [i for i in range(len(views)) if pid[i] != held[w["key"]]
                       and rectify(w, views[i], imgs[i], mks[i], edges, LOW_POSE_PPM, pose)[1].mean() > 0.1]
            for w in walls}

    def objective(only=None):
        total = 0.0
        for w in walls:
            idx = seen[w["key"]]
            if only is not None and only not in (pid[i] for i in idx):
                continue
            rs = []
            for i in idx:
                vals, ok, _ = rectify(w, views[i], imgs[i], mks[i], edges, LOW_POSE_PPM, pose)
                if ok.mean() > 0.1:
                    rs.append((pid[i], edges_of(gray(vals)), erode(ok)))
            for a in range(len(rs)):
                for b in range(a + 1, len(rs)):
                    if rs[a][0] == rs[b][0] or (only is not None and only not in (rs[a][0], rs[b][0])):
                        continue
                    both = rs[a][2] & rs[b][2]
                    total += both.mean() * ncc(rs[a][1], rs[b][1], both)
        pull = [pose[only]] if only else list(pose.values())
        return total - 0.01 * sum(p[0] ** 2 + 0.1 * p[1] ** 2 + p[2] ** 2 + 4 * p[3] ** 2 for p in pull)

    before = objective()
    # Google snaps panoramas to its road line, metres off the lane the car drove in (Howard has four), so first
    # the across-street offset by grid search: one for all panoramas, then each on its own.
    grid = np.arange(-LATERAL_M, LATERAL_M + 1e-9, 1.0)

    def set_all(dp):
        for p in pose:
            pose[p][1] = dp
        return objective()
    common = max(grid, key=set_all)
    set_all(common)
    for p in pose:
        def try_dp(dp, p=p):
            pose[p][1] = dp
            return objective(p)
        pose[p][1] = max(common + grid, key=try_dp)
    steps, lim = np.array([1.0, 1.0, 1.0, 0.3]), np.array([3.0, 2 * LATERAL_M, 3.0, 0.8])
    for _sweep in range(4):
        for p in pose:
            for j in range(4):
                best = objective(p)
                for sgn in (1, -1):
                    old = pose[p].copy()
                    pose[p][j] = float(np.clip(old[j] + sgn * steps[j], -lim[j], lim[j]))
                    val = objective(p)
                    if val > best + 1e-4:
                        break
                    pose[p] = old
        steps /= 2
    return before, objective()


def build_texture(w, cands, views, imgs, mks, edges, pose):
    ppm = min(PX_PER_M, MAX_W / w["L"], MAX_W / w["h"])
    acc, wsum, used = None, None, set()
    for i in cands:
        vals, _, wt = rectify(w, views[i], imgs[i], mks[i], edges, ppm, pose)
        if acc is None:
            acc, wsum = np.zeros(vals.shape), np.zeros(wt.shape)
        if wt.sum() > 0:
            acc += vals * wt[..., None]
            wsum += wt
            used.add(views[i]["pano"]["id"])
    seen = wsum > 0
    tex = np.zeros(acc.shape, np.uint8)
    tex[seen] = (acc[seen] / wsum[seen, None]).clip(0, 255).astype(np.uint8)
    if seen.any() and not seen.all():
        tex = cv2.inpaint(tex, (~seen).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return tex, seen, sorted(used)


def verify(w, tex, seen, view, img, mask, edges, pose, others):
    """Warp the texture into the held-out image and compare edges there, best slide within SLACK_M. others:
    textures of the other walls; they and this one mirrored, given the same search, make the null."""
    cam = camera(view, pose)
    th, tw = round(w["h"] * VER_PPM), round(w["L"] * VER_PPM)
    small = lambda a: cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)  # noqa: E731
    observed = cv2.resize(seen.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0
    _, vis, _ = rectify(w, view, img, np.ones(img.shape[:2]), edges, VER_PPM, pose)   # on screen, unoccluded
    tval = (observed & vis).astype(np.uint8)
    quad = project(cam, np.array([corners3d(w)[i] for i in (3, 2, 1, 0)], float))   # TL, TR, BR, BL
    if np.isnan(quad).any():
        return None
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), np.float32(quad))
    e_img = edges_of(gray(img))
    on_wall = (cv2.warpPerspective(tval, H, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 0) & (mask > 0.5)
    gx, gy = (np.arange(-m, m + 1e-9, 0.25) for m in SLACK_M)

    def best(t):
        out = (-1.0, 0.0, 0.0, None, None)
        for sx in gx:
            for sy in gy:
                Hs = H @ np.array([[1, 0, sx * VER_PPM], [0, 1, sy * VER_PPM], [0, 0, 1]], float)
                wt = cv2.warpPerspective(t, Hs, (SIZE, SIZE))
                valid = erode(on_wall & (cv2.warpPerspective(tval, Hs, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 0), 3)
                s = ncc(edges_of(gray(wt)), e_img, valid)
                if s > out[0]:
                    out = (s, sx, sy, valid, wt)
        return out

    score, sx, sy, valid, wt = best(small(tex))
    null = max(best(small(o))[0] for o in [tex[:, ::-1], *others])
    return {"score": round(score, 3), "null": round(null, 3), "offset_m": [round(float(sx), 2), round(float(sy), 2)],
            "pixels": int(valid.sum()), "overlay": (img, wt, valid, quad)}


def save_debug(key, ver):
    img, wt, valid, quad = ver["overlay"]
    blend = img.copy()
    blend[valid] = (0.5 * img[valid] + 0.5 * wt[valid]).astype(np.uint8)
    cv2.polylines(blend, [quad.astype(np.int32)], True, (0, 255, 255), 2)
    cv2.imwrite(str(DEBUG / f"{key}_verify.jpg"), np.hstack([img, blend, wt]))


def load_env(path: Path) -> None:
    if "GOOGLE_MAPS_API_KEY" not in os.environ and path.is_file():
        for line in path.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k == "GOOGLE_MAPS_API_KEY":
                os.environ[k] = v.strip().strip("'\"")


def held_out(w, views, imgs, mks, edges):
    """Views that see the wall, best first, and the held-out panorama (that of the best view not on the best
    view's panorama)."""
    rank = []
    for i, v in enumerate(views):
        _, ok, wt = rectify(w, v, imgs[i], mks[i], edges, LOW_PPM, {})
        if ok.mean() > 0.1:
            rank.append((wt.sum(), i))
    rank = [i for _, i in sorted(rank, reverse=True)]
    pid = lambda i: views[i]["pano"]["id"]  # noqa: E731
    held = next((pid(i) for i in rank if pid(i) != pid(rank[0])), None) if rank else None
    return rank, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "sim/public/local/facades_howard"))
    ap.add_argument("--weights", default=str(ROOT / "weights"))
    ap.add_argument("--env", default=str(ROOT / ".env"), help="file with GOOGLE_MAPS_API_KEY=... (never printed)")
    a = ap.parse_args()
    load_env(Path(a.env))
    out, weights = Path(a.out), Path(a.weights)
    out.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)

    data = json.loads(MOSS.read_text())
    walls = load_walls(data)
    ek = [(b["id"], k) for b in data["buildings"] for k in range(len(b["pts"]) - 1)]
    edges = (ek, np.array([b["pts"][k] for b in data["buildings"] for k in range(len(b["pts"]) - 1)], float),
             np.array([b["pts"][k + 1] for b in data["buildings"] for k in range(len(b["pts"]) - 1)], float))
    src = StreetViewSource(single=(37.7783, -122.4087, 0), cache_dir=CACHE)
    panos = find_panos(src)
    views = plan_views(walls, panos, edges)
    print(f"{len(walls)} walls, {len(panos)} Google panoramas on Howard, {len(views)} views", flush=True)
    imgs = [fetch(src, v) for v in views]
    mks = masks(imgs, weights)
    print(f"masked (people, vehicles, trees, poles, signs): {1 - np.mean([m.mean() for m in mks]):.0%} of pixels")

    ranks = {w["key"]: held_out(w, views, imgs, mks, edges) for w in walls}
    pose = {v["pano"]["id"]: np.zeros(4) for v in views}
    before, after = refine(walls, views, imgs, mks, edges, {k: h for k, (_, h) in ranks.items()}, pose)
    print(f"pose refinement: cross-view edge agreement {before:.2f} -> {after:.2f}", flush=True)
    texs = {}
    for w in walls:
        rank, held = ranks[w["key"]]
        if held:
            texs[w["key"]] = build_texture(w, [i for i in rank if views[i]["pano"]["id"] != held], views, imgs, mks,
                                           edges, pose)

    report, manifest = [], []
    for w in walls:
        entry = {"building_id": w["bid"], "wall_index": w["k"], "width_m": round(w["L"], 2), "height_m": w["h"]}
        rank, held = ranks[w["key"]]
        ver, ok = None, False
        if held:
            tex, seen, used = texs[w["key"]]
            h_i = next(i for i in rank if views[i]["pano"]["id"] == held)
            entry.update(coverage=round(float(seen.mean()), 3), source_pano_ids=used)
            others = [t for k, (t, _, _) in texs.items() if k != w["key"]]
            ver = verify(w, tex, seen, views[h_i], imgs[h_i], mks[h_i], edges, pose, others)
        if ver:
            save_debug(w["key"], ver)
            entry.update(score=ver["score"], null_score=ver["null"], offset_m=ver["offset_m"],
                         verify_pixels=ver["pixels"], heldout_pano_id=held)
            ok = (entry["coverage"] >= MIN_COVER and ver["pixels"] >= 2000 and ver["score"] >= MIN_SCORE
                  and ver["score"] - ver["null"] >= MIN_MARGIN)
        entry["kept"] = bool(ok)
        f = out / f"{w['key']}.jpg"
        if ok:
            cv2.imwrite(str(f), tex, [cv2.IMWRITE_JPEG_QUALITY, 88])
            manifest.append({"building_id": w["bid"], "wall_index": w["k"],
                             "corners_local_m": [[round(c, 3) for c in p] for p in corners3d(w)],
                             "width_m": entry["width_m"], "height_m": w["h"], "texture": f.name,
                             "score": entry["score"], "source_pano_ids": entry["source_pano_ids"]})
        elif f.exists():
            f.unlink()
        report.append(entry)
        print(f"  {w['key']:>12}: {w['L']:5.1f} x {w['h']:4.1f} m  cover {entry.get('coverage', 0):4.0%}  "
              f"score {entry.get('score', float('nan')):.3f} (null {entry.get('null_score', float('nan')):.3f}, "
              f"offset {entry.get('offset_m')})  {'kept' if ok else 'REJECTED'}", flush=True)
    r = src.requests
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "report.json").write_text(json.dumps({
        "walls": report, "requests_this_run": r,
        "pose": {k: [round(float(x), 2) for x in v] for k, v in pose.items()}, "panos": len(panos), "views": len(views),
        "wall_index": "edge k of moss.json buildings[].pts, from pts[k] to pts[k+1]",
        "corners": "[x east, y north, z up] m: bottom-left, bottom-right, top-right, top-left seen from outside; "
                   "texture column 0 = left, row 0 = top"}, indent=1))
    print(f"kept {len(manifest)}/{len(walls)} walls; Street View requests this run: {r['image']} image (billed), "
          f"{r['metadata']} metadata (free), {r['cached']} from cache")


if __name__ == "__main__":
    main()
