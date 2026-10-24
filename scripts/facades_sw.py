"""Verified Street View facade textures for the SOUTHWEST side of Moss St, Howard to ~80 m down (permission and
rules: docs/google-permission.md; output is gitignored and never committed or published).

    uv run --no-project --with numpy --with opencv-python-headless --with ultralytics \
        python scripts/facades_sw.py [--out sim/public/local/facades_sw] [--weights weights/yolo11n.pt]
    python scripts/facades_sw.py --selftest        # scoring check on a synthetic wall, no network

GOOGLE_MAPS_API_KEY comes from the environment or the repo .env (never printed). Raw responses are cached in
/tmp/kite_sv_sw/cache so reruns are free; delete it when done.

1. Walls: streetview_facades.street_walls() edges that face Moss from its SW side (outward normal toward the NE
   side), along Moss from the Howard corner to ALONG_MAX. Howard-facing faces belong to another script.
2. Panoramas: Google's own, from the free metadata endpoint every 3 m along Moss (all inside SCOPE).
3. Views per panorama: 640x640, fov 60, headings straight at the SW side and +-35 deg, pitch 0 and 40 (the
   fronts are only 4-6 m from the car camera, so one narrow view can't hold a whole storey stack).
4. Per wall, the panorama second-closest to the wall centre is HELD OUT. Walls nearer the street than
   BUILDING_LINE_M are rectified on that line instead (facade_offset: OSM footprints there include bay windows).
   The texture (100 px/m) blends every
   other panorama's views per texel, weight (frontal cos / distance)^2 x image-border falloff, cubed, with YOLO
   person/vehicle boxes zero-weighted (filled from another view where one exists, else left as is).
5. Verify: one extra view from the held-out panorama, aimed at the wall centre (fov 75), and the texture rendered
   into it: NCC of blurred Sobel edge magnitude over the wall pixels (YOLO boxes excluded), best over image
   offsets up to SLACK_PX. The same score
   with the wall slid SHIFTS_M along itself is the baseline: a texture on the wrong wall, or a wrong camera, does
   no better at 0 than shifted. Kept if score >= MIN_SCORE and score - best shifted >= MIN_MARGIN; others get no
   texture (manifest entry with texture null).

Manifest convention: wall_index k is the footprint edge moss.json buildings[*].pts[k] -> pts[k+1] (pts is a closed
ring, last point = first). corners_local_m are [x east, y north, z up] metres in moss.json's frame, ordered
bottom-left, bottom-right, top-right, top-left AS SEEN FROM THE STREET (outside the building). The JPEG's column 0
is that left edge and row 0 the roofline: UV (0,0) = bottom-left corner, (1,1) = top-right.
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
from streetview_facades import blocked, compass, street_walls  # noqa: E402
from kite.gps import latlng_to_sim, sim_to_latlng  # noqa: E402

MOSS = ROOT / "sim/public/osm/moss.json"
CACHE = Path("/tmp/kite_sv_sw/cache")
MOSS_ROAD = 8916291
ALONG_MAX = 80.0            # m from the Howard corner along Moss: walls with their middle past this are skipped
PX_PER_M, MAX_W = 100, 4096
SIZE, FOV = 640, 60
VERIFY_FOV = 75             # the held-out view: one per wall, from the held-out panorama, aimed at the wall centre
HEADING_OFFSETS = (-35, 0, 35)
PITCHES = (0, 40)
SEE_M = 15.0
WEIGHT_POW = 3.0
YOLO_CLASSES = {0, 1, 2, 3, 5, 7}   # person, bicycle, car, motorcycle, bus, truck
SHIFTS_M = (-3.0, -1.5, 1.5, 3.0)
SLACK_PX = 40               # ~0.5 m at the held-out views' range
MIN_SCORE, MIN_MARGIN = 0.25, 0.05
BUILDING_LINE_M = -5.0      # SW facades' line across Moss (moss_axes n); see facade_offset()
CAM_H = 2.5                  # kite.sources.streetview.SV_CAM_HEIGHT_M (ASSUMED car roof)


def load_key():
    if os.environ.get("GOOGLE_MAPS_API_KEY"):
        return
    for env in (ROOT / ".env", ROOT.parents[2] / ".env"):   # the worktree's main checkout too
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("GOOGLE_MAPS_API_KEY="):
                    os.environ["GOOGLE_MAPS_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
                    return


def moss_axes(data):
    pts = next(r for r in data["roads"] if r["id"] == MOSS_ROAD)["pts"]
    u = np.subtract(pts[2], pts[1]).astype(float)
    u /= np.linalg.norm(u)                      # Howard -> Folsom (SE)
    return u, np.array([-u[1], u[0]])           # n points to the NE side


def sw_walls(data):
    u, n = moss_axes(data)
    out = []
    for bid, k, a, b, h, nrm in street_walls(data):
        m = (a + b) / 2
        if m @ n < 0 and nrm @ n > 0.7 and -5 < m @ u < ALONG_MAX:
            out.append((bid, k, a, b, h, nrm))
    return out


def find_panos(data, src):
    from kite.sources.streetview import _is_google
    u, n = moss_axes(data)
    panos = {}
    for s in np.arange(-2, ALONG_MAX + 4, 3.0):
        m = src._metadata(*sim_to_latlng(*(s * u + 0.5 * n)))   # raises outside SCOPE
        if m and _is_google(m):
            x, y = latlng_to_sim(m["location"]["lat"], m["location"]["lng"])
            panos[m["pano_id"]] = {"id": m["pano_id"], "x": x, "y": y, "date": m.get("date")}
    return list(panos.values())


def fetch_views(src, panos, walls, n):
    """Grid of views per panorama that sees any SW wall. -> [(pano, Camera, bgr)]"""
    base = compass(*-n)
    shots = []
    for p in panos:
        c = np.array([p["x"], p["y"]])
        if not any(2.0 < (c - a) @ nrm < SEE_M and np.linalg.norm((a + b) / 2 - c) < SEE_M + np.linalg.norm(b - a) / 2
                   for _, _, a, b, _, nrm in walls):
            continue
        for dh in HEADING_OFFSETS:
            for pitch in PITCHES:
                hd = (base + dh) % 360
                raw = src._get("image", {"pano": p["id"], "heading": f"{hd:.1f}", "pitch": f"{pitch}", "fov": f"{FOV}",
                                         "size": f"{SIZE}x{SIZE}", "return_error_code": "true"})
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("Street View image: the response is not an image")
                cam = Camera(x=p["x"], y=p["y"], z=CAM_H, heading=hd, pitch=pitch,
                             focal=0.5 / math.tan(math.radians(FOV) / 2), width=img.shape[1], height=img.shape[0])
                shots.append((p, cam, img))
    return shots


def yolo_masks(shots, weights):
    """Per shot, a bool (H, W) mask of person/vehicle boxes, grown 10%."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    masks = []
    for _, _, img in shots:
        m = np.zeros(img.shape[:2], bool)
        for box in model(img, verbose=False, conf=0.25)[0].boxes:
            if int(box.cls) in YOLO_CLASSES:
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                dx, dy = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
                m[max(0, int(y0 - dy)):int(y1 + dy) + 1, max(0, int(x0 - dx)):int(x1 + dx) + 1] = True
        masks.append(m)
    return masks


def wall_points(a, b, h, shift=0.0, px_per_m=PX_PER_M):
    L = float(np.linalg.norm(b - a))
    ppm = min(px_per_m, MAX_W / L)
    tw, th = max(2, round(L * ppm)), max(2, round(h * ppm))
    e = (b - a) / L
    s = (np.arange(tw) + 0.5) / tw * L + shift
    z = h * (1 - (np.arange(th) + 0.5) / th)
    ground = a[None, :] + s[:, None] * e[None, :]
    pts = np.concatenate([np.repeat(ground[None], th, 0), np.repeat(z[:, None, None], tw, 1)], -1).reshape(-1, 3)
    return pts, ground, tw, th


def wall_texture(wall, shots, masks, edges, skip_pano):
    """Blend of every shot not from skip_pano. Column 0 = a. -> (tex uint8, seen fraction, pano ids)"""
    bid, k, a, b, h, nrm = wall
    pts, ground, tw, th = wall_points(a, b, h)
    acc, wsum = np.zeros((th * tw, 3)), np.zeros(th * tw)
    acc_m, wsum_m = np.zeros((th * tw, 3)), np.zeros(th * tw)   # ignoring YOLO masks: the fallback
    used = set()
    for (pano, cam, img), mask in zip(shots, masks):
        c = np.array([cam.x, cam.y])
        if pano["id"] == skip_pano or (c - a) @ nrm < 1.0:
            continue
        uv = project(cam, pts)
        vals, ok = sample(img.astype(np.float32), uv[:, 0], uv[:, 1])
        if ok.mean() < 0.01:
            continue
        tips = ground - 0.15 * (ground - c) / np.linalg.norm(ground - c, axis=1)[:, None]
        ok &= ~np.tile(blocked(c, tips, edges, (bid, k)), th)
        d3 = pts - np.array([cam.x, cam.y, cam.z])
        dist = np.linalg.norm(d3, axis=1)
        cos = np.clip(-(d3[:, :2] @ nrm) / dist, 0, 1)
        border = np.clip(np.minimum.reduce([uv[:, 0], cam.width - uv[:, 0], uv[:, 1], cam.height - uv[:, 1]]) / 40, 0, 1)
        w = np.where(ok, ((cos / dist) ** 2 * np.nan_to_num(border)) ** WEIGHT_POW, 0)
        if w.sum() == 0:
            continue
        iu = np.clip(np.nan_to_num(uv[:, 0]).astype(int), 0, cam.width - 1)
        iv = np.clip(np.nan_to_num(uv[:, 1]).astype(int), 0, cam.height - 1)
        wm = np.where(mask[iv, iu], 0, w)
        acc += vals * w[:, None]
        wsum += w
        acc_m += vals * wm[:, None]
        wsum_m += wm
        used.add(pano["id"])
    seen = wsum > 0
    tex = np.zeros((th * tw, 3))
    clean = wsum_m > 0.05 * wsum   # a masked texel takes the unmasked views only if nothing else saw it
    tex[seen] = acc[seen] / wsum[seen, None]
    tex[clean & seen] = acc_m[clean & seen] / wsum_m[clean & seen, None]
    if seen.any():
        tex[~seen] = np.median(tex[seen], axis=0)
    return tex.reshape(th, tw, 3).clip(0, 255).astype(np.uint8), float(seen.mean()), sorted(used)


def render(tex, cam, a, b, h, shift=0.0):
    """The wall (texture column 0 at a, slid `shift` m along itself) as seen by cam. -> (bgr float, hit mask)"""
    fwd, right, down = cam.axes()
    vs, us = np.mgrid[0:cam.height, 0:cam.width] + 0.5
    rays = (fwd[:, None] * cam.f_px + right[:, None] * (us.ravel() - cam.width / 2)
            + down[:, None] * (vs.ravel() - cam.height / 2))
    L = float(np.linalg.norm(b - a))
    e = np.array([*(b - a) / L, 0.0])
    a3 = np.array([*(a + shift * e[:2]), 0.0])
    normal = np.array([-e[1], e[0], 0.0])
    o = np.array([cam.x, cam.y, cam.z])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((a3 - o) @ normal) / (normal @ rays)
    hit = o[:, None] + rays * t
    s, z = (hit - a3[:, None]).T @ e, hit[2]
    th, tw = tex.shape[:2]
    vals, ok = sample(tex.astype(np.float32), s / L * tw, (h - z) / h * th)
    ok &= (t > 0) & (s >= 0) & (s <= L) & (z >= 0) & (z <= h)
    return vals.reshape(cam.height, cam.width, 3), ok.reshape(cam.height, cam.width)


def edges(bgr):
    g = cv2.GaussianBlur(cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return cv2.GaussianBlur(mag, (0, 0), 2.5)   # a few px of slack for pose / footprint error


def ncc(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else 0.0


def score(tex, cam, img, mask, a, b, h, shift=0.0):
    """Edge NCC of the rendered wall vs the real image over wall pixels outside YOLO boxes, best over image
    offsets up to SLACK_PX (pose / non-planar facade slack, well under the SHIFTS_M baseline); None if too few."""
    ren, hit = render(tex, cam, a, b, h, shift)
    hit = cv2.erode(hit.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)   # no wall-outline edges
    if hit.sum() < 0.05 * hit.size:
        return None
    er, ei, keep = edges(ren), edges(img), ~mask
    best = None
    for dy in range(-SLACK_PX, SLACK_PX + 1, 8):
        for dx in range(-SLACK_PX, SLACK_PX + 1, 8):
            m = np.roll(hit, (dy, dx), (0, 1)) & keep
            if m.sum() > 0.04 * m.size:
                s = ncc(np.roll(er, (dy, dx), (0, 1))[m], ei[m])
                best = s if best is None else max(best, s)
    return best


def fetch_view(src, pano, target, fov=FOV):
    """One Static view from pano aimed at a 3D point. -> (Camera, bgr)"""
    d = np.subtract(target[:2], (pano["x"], pano["y"]))
    hd = compass(*d)
    pitch = float(np.clip(round(math.degrees(math.atan2(target[2] - CAM_H, np.linalg.norm(d)))), 0, 45))
    raw = src._get("image", {"pano": pano["id"], "heading": f"{hd:.1f}", "pitch": f"{pitch:g}", "fov": f"{fov}",
                             "size": f"{SIZE}x{SIZE}", "return_error_code": "true"})
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Street View image: the response is not an image")
    return Camera(x=pano["x"], y=pano["y"], z=CAM_H, heading=hd, pitch=pitch,
                  focal=0.5 / math.tan(math.radians(fov) / 2), width=img.shape[1], height=img.shape[0]), img


def verify(wall, tex, cam, img, mask):
    """Score in the held-out view -> (score, best score with the wall slid along itself)."""
    _, _, a, b, h, _ = wall
    s0 = score(tex, cam, img, mask, a, b, h)
    shifted = [s for d in SHIFTS_M if (s := score(tex, cam, img, mask, a, b, h, d)) is not None]
    return s0, (max(shifted) if shifted else None)


def facade_offset(wall, n):
    """How far to move a wall (m, along n) to where its facade really is in the panoramas.
    Measured on the held-out views: the OSM footprints of 10 and 14 Moss sit ~1.5 m proud of the real
    ground-floor facade (their bay windows), so rectifying on them misplaces everything; the corner building
    (OSM line -5.0 m) verifies unmoved. Walls nearer the street than BUILDING_LINE_M go back to it; walls already
    behind it (recesses) stay. ponytail: one line for the block; fit per wall if more streets get textured."""
    _, _, a, b, _, _ = wall
    off = float(((a + b) / 2) @ n)
    return min(0.0, BUILDING_LINE_M - off)


def held_out_pano(wall, panos):
    _, _, a, b, _, nrm = wall
    m = (a + b) / 2
    front = [p for p in panos if (np.array([p["x"], p["y"]]) - a) @ nrm > 1.0]
    front.sort(key=lambda p: math.dist((p["x"], p["y"]), m))
    return front[1]["id"] if len(front) > 1 else None


def corners(a, b, h, nrm):
    """[BL, BR, TR, TL] as seen from outside; True if a is the left end (texture column 0 = a)."""
    look = -nrm
    right = np.array([look[1], -look[0]])
    a_left = (b - a) @ right > 0
    lft, rgt = (a, b) if a_left else (b, a)
    c = [[*lft, 0.0], [*rgt, 0.0], [*rgt, h], [*lft, h]]
    return [[round(float(v), 3) for v in p] for p in c], bool(a_left)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "sim/public/local/facades_sw"))
    ap.add_argument("--weights", default=next((str(p) for p in (ROOT / "weights/yolo11n.pt", ROOT / "data/models/yolo11n.pt")
                                               if p.is_file()), None))
    ap.add_argument("--debug-dir", help="also write held-out comparisons here (keep it in /tmp; Google imagery)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.weights:
        sys.exit("facades_sw: no YOLO weights; pass --weights path/to/yolo11n.pt")
    load_key()
    from kite.sources.streetview import StreetViewSource
    data = json.loads(MOSS.read_text())
    _, n = moss_axes(data)
    walls = sw_walls(data)
    src = StreetViewSource(single=(37.7783, -122.4087, 0), cache_dir=CACHE, radius_m=6)
    panos = find_panos(data, src)
    shots = fetch_views(src, panos, walls, n)
    masks = yolo_masks(shots, args.weights)
    print(f"{len(walls)} SW walls, {len(panos)} Google panoramas, {len(shots)} views", flush=True)
    all_edges = [(bb["id"], k) for bb in data["buildings"] for k in range(len(bb["pts"]) - 1)]
    E = (all_edges, np.array([bb["pts"][k] for bb in data["buildings"] for k in range(len(bb["pts"]) - 1)], float),
         np.array([bb["pts"][k + 1] for bb in data["buildings"] for k in range(len(bb["pts"]) - 1)], float))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    held = {(w[0], w[1]): held_out_pano(w, panos) for w in walls}
    manifest = []
    for wall in walls:
        bid, k, a0, b0, h, nrm = wall
        delta = facade_offset(wall, n) * n
        a, b = a0 + delta, b0 + delta            # where the facade is in the panoramas' frame
        wall_s = (bid, k, a, b, h, nrm)
        tex, cover, used = wall_texture(wall_s, shots, masks, E, held[(bid, k)])
        hp = next(p for p in panos if p["id"] == held[(bid, k)])
        vcam, vimg = fetch_view(src, hp, np.array([*(a0 + b0) / 2, h / 2]), fov=VERIFY_FOV)
        vmask = yolo_masks([(hp, vcam, vimg)], args.weights)[0]
        s0, s_shift = verify(wall_s, tex, vcam, vimg, vmask)
        ok = s0 is not None and cover >= 0.5 and s0 >= MIN_SCORE and (s_shift is None or s0 - s_shift >= MIN_MARGIN)
        cs, a_left = corners(a0, b0, h, nrm)
        name = f"{bid}_{k}.jpg"
        if ok:
            cv2.imwrite(str(out / name), tex if a_left else tex[:, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        if args.debug_dir:
            dbg = Path(args.debug_dir)
            dbg.mkdir(parents=True, exist_ok=True)
            ren, hit = render(tex, vcam, a, b, h)
            over = np.where(hit[..., None], 0.5 * ren + 0.5 * vimg, vimg).astype(np.uint8)
            cv2.imwrite(str(dbg / f"{bid}_{k}_heldout.jpg"),
                        np.hstack([vimg, over, np.where(hit[..., None], ren, 0).astype(np.uint8)]))
            cv2.imwrite(str(dbg / f"{bid}_{k}_tex.jpg"), tex if a_left else tex[:, ::-1])
        L = float(np.linalg.norm(b - a))
        manifest.append({"building_id": bid, "wall_index": k, "corners_local_m": cs, "width_m": round(L, 3),
                         "height_m": h, "texture": name if ok else None,
                         "score": None if s0 is None else round(s0, 3),
                         "score_shifted_best": None if s_shift is None else round(s_shift, 3),
                         "coverage": round(cover, 3), "source_pano_ids": used, "heldout_pano_id": held[(bid, k)],
                         "pano_frame_offset_m": np.round(delta, 3).tolist()})
        print(f"  {bid}_{k}: {L:5.1f} x {h:4.1f} m  seen {cover:4.0%}  score {s0 if s0 is None else round(s0, 3)} "
              f"(shifted best {s_shift if s_shift is None else round(s_shift, 3)})  {'KEPT' if ok else 'rejected'}",
              flush=True)
    r = src.requests
    doc = {"convention": __doc__.split("Manifest convention: ")[1].strip(),
           "verification": f"edge NCC in one held-out panorama's view; kept if score >= {MIN_SCORE} and score - best "
                           f"score with the wall slid {list(SHIFTS_M)} m >= {MIN_MARGIN}",
           "px_per_m": PX_PER_M, "requests": r, "walls": manifest}
    (out / "manifest.json").write_text(json.dumps(doc, indent=1))
    kept = sum(1 for m in manifest if m["texture"])
    print(f"kept {kept}/{len(manifest)} walls; Street View requests: {r['image']} image (billed), "
          f"{r['metadata']} metadata (free), {r['cached']} from cache")


def selftest():
    """A synthetic wall seen from two cameras: the right placement scores high and beats a slid one."""
    from facade_rectify import _render_wall
    rng = np.random.default_rng(0)
    a, b, h = np.array([0.0, 5.0]), np.array([10.0, 5.0]), 8.0
    tex = np.full((round(h * 40), 400, 3), 200, np.uint8)
    for _ in range(40):   # random "windows"
        x, y = rng.integers(0, tex.shape[1] - 40), rng.integers(0, tex.shape[0] - 50)
        tex[y:y + 50, x:x + 40] = rng.integers(0, 120)
    cam = Camera(x=6.0, y=0.0, z=2.5, heading=-20, focal=0.8, width=640, height=640, pitch=20)
    img = _render_wall(tex.astype(float), cam, tuple(a), tuple(b), h).astype(np.uint8)
    mask = np.zeros(img.shape[:2], bool)
    s0 = score(tex, cam, img, mask, a, b, h)
    s1 = max(score(tex, cam, img, mask, a, b, h, d) for d in SHIFTS_M)
    assert s0 > 0.9 and s0 - s1 > 0.3, (s0, s1)
    cs, a_left = corners(a, b, h, np.array([0.0, -1.0]))   # wall faces south: viewer looks north, a (west) is left
    assert a_left and cs[0] == [0.0, 5.0, 0.0] and cs[2] == [10.0, 5.0, 8.0], cs
    assert not corners(a, b, h, np.array([0.0, 1.0]))[1]
    print(f"facades_sw self-check ok: score {s0:.2f}, best slid {s1:.2f}")


if __name__ == "__main__":
    main()
