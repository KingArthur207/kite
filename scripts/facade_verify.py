"""Does the Blender street look like Street View where it should? Per-wall check (local only).

    uv run --no-project --with numpy --with opencv-python-headless python scripts/facade_verify.py \\
        sim/public/local/moss_real.glb after

For every street wall, the Street View tile that sees most of it cleanly (refined panorama pose from
scripts/streetview_facades.py, /tmp/kite_sv) is rendered from the same camera in Blender
(sim/blender/build_real_moss.py --render-glb). Both images are rectified onto the wall (as moss.json has it)
at 16 px/m, with the tile's car/tree/pole/sky masks, and compared by high-pass NCC. A low score with a high
mirrored or shifted score names the failure: "mirrored", "offset", else "mismatch". Prints a table and writes
/tmp/kite_sv/verify_<label>.json. Needs the tiles in /tmp/kite_sv/cache (no new requests).

`--gate` also turns photo walls scoring under GATE back to plain in facades.json (then run
streetview_facades.py --plain-colours and rebuild). The renders use the tiles the textures came from, so this
catches geometry and UV faults; the cross-panorama score in streetview_facades.py is the independent check.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import streetview_facades as F  # noqa: E402

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
PPM = 16
GOOD = 0.35
GATE = 0.2


def best_shot(w, shots, pmap, frames, poses, occ):
    best, bn = None, 0
    pts, ground, z, tw, th = F.wall_grid(w, 4)
    for s in shots:
        pid = s["pano"]["id"]
        pose = poses.get(pid, F.ZERO)
        C = F.cam_centre(pmap[pid], pose, frames[pid])
        if (C[:2] - w["a"]) @ w["normal"] < 1.0:
            continue
        uv = F.project(C, pose, s["heading"], s["pitch"], s["fov"], pts)
        ok = np.isfinite(uv[:, 0]) & (uv[:, 0] > 4) & (uv[:, 0] < F.TILE - 4) & (uv[:, 1] > 4) & (uv[:, 1] < F.TILE - 30)
        iu = np.clip(np.nan_to_num(uv[:, 0]), 0, F.TILE - 1).astype(int)
        iv = np.clip(np.nan_to_num(uv[:, 1]), 0, F.TILE - 1).astype(int)
        ok &= ~s["occ"][iv, iu] & ~s["sky"][iv, iu] & ~F.hidden(C, ground, z, occ, w).ravel()
        d = pts - C
        cos = -(d[:, :2] @ w["normal"]) / np.linalg.norm(d[:, :2], axis=1)
        n = float((ok * np.clip(cos, 0, 1)).sum())
        if n > bn:
            best, bn = s, n
    return best if bn >= 8 else None


def rectified(w, s, gray, C, pose, occ):
    pts, ground, z, tw, th = F.wall_grid(w, PPM)
    uv = F.project(C, pose, s["heading"], s["pitch"], s["fov"], pts)
    v, ok = F.sample(gray[..., None].astype(np.float32), uv[:, 0], uv[:, 1])
    iu = np.clip(np.nan_to_num(uv[:, 0]), 0, F.TILE - 1).astype(int)
    iv = np.clip(np.nan_to_num(uv[:, 1]), 0, F.TILE - 1).astype(int)
    ok &= ~s["occ"][iv, iu] & ~s["sky"][iv, iu] & ~F.hidden(C, ground, z, occ, w).ravel()
    t = np.where(ok, v[:, 0], np.nan).reshape(th, tw)
    return t


def compare(a, b):
    va, vb = np.isfinite(a), np.isfinite(b)
    ha, hb = F.highpass(np.nan_to_num(a), va, 0.6 * PPM), F.highpass(np.nan_to_num(b), vb, 0.6 * PPM)
    r, n = F.ncc(ha, hb)
    if r is None or n < 2 * PPM * PPM:
        return None
    flip, _ = F.ncc(ha, hb[:, ::-1])
    shift, best = (0, 0), r
    for dx in range(-int(1.5 * PPM), int(1.5 * PPM) + 1, PPM // 4):
        for dy in range(-PPM, PPM + 1, PPM // 4):
            if dx == dy == 0 or abs(dx) >= hb.shape[1] or abs(dy) >= hb.shape[0]:
                continue
            sb = np.full_like(hb, np.nan)
            ys, xs = slice(max(dy, 0), hb.shape[0] + min(dy, 0)), slice(max(dx, 0), hb.shape[1] + min(dx, 0))
            yd, xd = slice(max(-dy, 0), hb.shape[0] + min(-dy, 0)), slice(max(-dx, 0), hb.shape[1] + min(-dx, 0))
            sb[ys, xs] = hb[yd, xd]
            rr, nn = F.ncc(ha, sb)
            if rr is not None and nn >= 2 * PPM * PPM and rr > best:
                best, shift = rr, (dx / PPM, dy / PPM)
    verdict = "ok" if r >= GOOD else "mirrored" if (flip or -1) > max(r + 0.15, GOOD) else \
        "offset" if best > max(r + 0.15, GOOD) else "mismatch"
    return {"ncc": round(r, 3), "mirrored_ncc": round(flip or 0, 3), "best_shift_m": shift,
            "shifted_ncc": round(best, 3), "verdict": verdict, "texels": n}


def main():
    glb, label = Path(sys.argv[1]).resolve(), sys.argv[2]
    from kite.sources.streetview import StreetViewSource
    src = StreetViewSource(single=(37.7783, -122.4087, 0), cache_dir=F.CACHE, key="cache-only")
    data = json.loads(F.MOSS.read_text())
    walls = F.street_walls(data)
    panos = json.loads((F.CACHE.parent / "panos.json").read_text())
    pmap = {p["id"]: p for p in panos}
    frames = {p["id"]: F.pano_street(p, data) for p in panos}
    poses = json.loads((F.CACHE.parent / "poses.json").read_text())
    shots = F.fetch_tiles(F.plan_tiles(walls, panos, data), src, None)
    if src.requests["image"]:
        raise SystemExit("facade_verify must not fetch: run streetview_facades.py first")
    occ = F.occluders(data)
    pick = {}
    for i, w in enumerate(walls):
        s = best_shot(w, shots, pmap, frames, poses, occ)
        if s is not None:
            pick[i] = s
    cams, names = [], {}
    for s in {id(s): s for s in pick.values()}.values():
        pid = s["pano"]["id"]
        pose = poses.get(pid, F.ZERO)
        C = F.cam_centre(pmap[pid], pose, frames[pid])
        name = f"{pid[:10]}_{s['heading']:.0f}_{s['pitch']:.0f}".replace("-", "m")
        names[id(s)] = name
        cams.append({"name": name, "x": float(C[0]), "y": float(C[1]), "z": float(C[2]),
                     "heading": s["heading"] + pose["yaw"], "pitch": s["pitch"], "fov": s["fov"],
                     "tx": pose["tx"], "ty": pose["ty"]})
    out = F.CACHE.parent / f"render_{label}"
    cam_file = F.CACHE.parent / f"cams_{label}.json"
    cam_file.write_text(json.dumps(cams))
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(ROOT / "sim/blender/build_real_moss.py"), "--",
                    "--render-glb", str(glb), str(cam_file), str(out)], check=True, capture_output=True)
    rows = []
    for i, w in enumerate(walls):
        row = {"wall": f"{w['building']}_{w['edge']}", "street": w["street"],
               "length_m": round(float(np.linalg.norm(w["b"] - w["a"])), 1)}
        s = pick.get(i)
        if s is None:
            row["verdict"] = "unseen"
            rows.append(row)
            continue
        pid = s["pano"]["id"]
        pose = poses.get(pid, F.ZERO)
        C = F.cam_centre(pmap[pid], pose, frames[pid])
        img = cv2.imread(str(out / f"{names[id(s)]}.png"))
        sv = rectified(w, s, s["gray"], C, pose, occ)
        rd = rectified(w, s, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), C, pose, occ)
        res = compare(sv, rd)
        row.update(res or {"verdict": "too little seen"})
        row["view"] = names[id(s)]
        rows.append(row)
    (F.CACHE.parent / f"verify_{label}.json").write_text(json.dumps(rows, indent=1))
    if "--gate" in sys.argv:
        man_path = F.OUT / "facades.json"
        man = json.loads(man_path.read_text())
        by = {r["wall"]: r for r in rows}
        for e in man["walls"]:
            r = by.get(f"{e['building']}_{e['edge']}", {})
            e["render_ncc"] = r.get("ncc")
            if e.get("file") and r.get("ncc") is not None and r["ncc"] < GATE:
                e["file"], e["dropped"] = None, "render check"
                print(f"  gate: {r['wall']} back to plain (render NCC {r['ncc']:+.2f})")
        man_path.write_text(json.dumps(man, indent=1))
    scored = [r for r in rows if "ncc" in r]
    from collections import Counter
    print(f"{label}: {len(rows)} walls, {len(scored)} compared; median NCC "
          f"{np.median([r['ncc'] for r in scored]):.3f}; verdicts {dict(Counter(r['verdict'] for r in rows))}")
    for r in rows:
        if "ncc" in r:
            print(f"  {r['wall']:>14} {r['street']:6} {r['length_m']:5.1f} m  ncc {r['ncc']:+.2f}  mirrored "
                  f"{r['mirrored_ncc']:+.2f}  shift {r['best_shift_m']} -> {r['shifted_ncc']:+.2f}  {r['verdict']}")


if __name__ == "__main__":
    main()
