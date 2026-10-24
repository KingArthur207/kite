"""Per-building look of Moss St (Folsom to Howard) from Street View, for the sim's procedural buildings
(permission: docs/google-permission.md; everything this writes is Street View-derived and stays local).

    set -a; . .env; set +a       # ANTHROPIC_API_KEY (GOOGLE_MAPS_API_KEY only if a building needs a new view)
    uv run --no-project --with numpy --with opencv-python-headless --with anthropic \\
        python scripts/moss_look.py [--repo DIR] [--fix corrections.json] [--model claude-sonnet-5-5]
    python scripts/moss_look.py --selftest      # colour + corrections logic on synthetic data, no network

Writes (both gitignored, never commit):
  <repo>/sim/public/local/moss_buildings.json   {"buildings": {"<osm id>": {main, trim, cladding, floors, bays,
                                                  garages, garage_colour, gate, hedge, fire_escape, mural,
                                                  confidence}}}
  <repo>/out/look/buildings.html                review page: one card per building, "wrong" toggle + note,
                                                  exports corrections.json for --fix
  <repo>/out/look/crops/<id>.jpg, claude/<id>.json   crops and cached Claude answers (reruns are free)

1. Crop: the building's Moss-facing walls (streetview_facades.street_walls) rectified from the cached tiles in
   /tmp/kite_sv (its refined poses, ADE20K occluder/sky masks; no new requests), laid side by side as seen from
   the street. No placement verification: this only needs the right building, checked by eye on the page.
2. Claude vision (one call per building, structured JSON output): cladding, floors, bays, garages, gate,
   hedge, fire escape, mural, rough main/trim/garage colours. A building with no usable crop (none, mostly
   unseen, or Claude under MIN_CONF from it) gets one Static API view aimed at its footprint centre from the
   nearest panorama (cached in /tmp/kite_sv_look, capped at MAX_NEW_IMAGES) and a second call with both.
3. Colours from pixels (wall_colours): Claude's colour picks WHICH pixels are the wall (same hue, or neutral
   near its lightness); their brighter half (shade) gives the colour, after a white-patch exposure gain and
   with L* kept near Claude's reading. Trim likewise. Garage colour is Claude's (doors aren't located).
4. --fix: corrections.json from the page. A note of key=value pairs ("main=#c8b48a bays=2 gate=no") overrides
   those fields; any other note text re-asks Claude (the Sonnet model) with the note as a hint.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PPM = 40                   # crop px per metre
FRONT_COS = 0.85           # walls this square to the building's mean normal vote for its colours
MIN_SEEN = 0.35            # crop share seen by some view below which the crop is dropped for a Static view
MIN_SEEN_SURE = 0.6        # below this the crop gets a Static view alongside it
MAX_NEW_IMAGES = 60        # hard cap on new Static API images (the brief allows < 80)
LOOK_CACHE = Path("/tmp/kite_sv_look")
# Haiku 5.5 was tried first: it called nearly every wall stucco and disagreed with Sonnet (and the photos) on
# floors and bays for ~half the buildings; Sonnet is still cheap at 26 calls
MODEL, FIX_MODEL = "claude-sonnet-5-5", "claude-sonnet-5-5"
HUE_TOL, NEUTRAL_C = 25.0, 12.0     # wall pixels: within this hue (deg) of Claude's; below this chroma = neutral
WHITE_Y, MAX_GAIN, L_TOL = 0.80, 1.5, 10.0   # exposure: white-patch target, max gain, L* slack around Claude
MIN_SHARE = 0.03                    # fewer matching wall pixels than this: Claude's colour stands
MIN_CHROMA = 0.75                   # wall chroma at least this share of Claude's (the mosaics are washed out)
MIN_CONF = 0.6                      # Claude less sure than this from the crop alone: fetch a perspective view
CLADDING = ("siding", "stucco", "brick", "concrete", "metal")
FIELDS = {"main": str, "trim": str, "cladding": str, "floors": int, "bays": int, "garages": int,
          "garage_colour": str, "gate": bool, "hedge": bool, "fire_escape": bool, "mural": bool,
          "confidence": float}


# ---------------------------------------------------------------- targets, crops
def targets(repo: Path) -> list[dict]:
    """The Moss-facing buildings, in street order per side: sim/public/local/moss_buildings.csv rows."""
    path = repo / "sim/public/local/moss_buildings.csv"
    with path.open() as f:
        rows = [{"id": int(r["building_id"]), "side": r["side"], "m": float(r["m_from_howard"]),
                 "address": r["address"].replace(";", ", ") or "-"} for r in csv.DictReader(f)]
    return sorted(rows, key=lambda r: (r["side"], r["m"]))


def _cached_tiles(tiles, cache: Path):
    """Tiles whose image and segmentation masks are already cached (never a new request here)."""
    import hashlib
    import urllib.parse
    import streetview_facades as sf
    out = []
    for t in tiles:
        p, hd, pt, fov, _ = t
        params = {"pano": p["id"], "heading": f"{hd:.1f}", "pitch": f"{pt:g}", "fov": f"{fov:g}",
                  "size": f"{sf.TILE}x{sf.TILE}", "return_error_code": "true"}
        img = cache / f"image-{hashlib.sha256(urllib.parse.urlencode(params).encode()).hexdigest()[:24]}"
        seg = sf.SEG_CACHE / (hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24] + ".npz")
        if img.is_file() and seg.is_file():
            out.append(t)
    return out


def make_crops(ids: set[int], crop_dir: Path, log=print) -> dict[int, float]:
    """Rectified Moss front of each building -> crop_dir/<id>.jpg; returns the share of it some view saw."""
    import cv2
    import streetview_facades as sf
    from kite.sources.streetview import StreetViewSource
    data = json.loads(sf.MOSS.read_text())
    walls = sf.street_walls(data)
    src = StreetViewSource(single=(37.7783, -122.4087, 0), cache_dir=sf.CACHE, key="cache-only")
    panos = sf.load_panos(data, src)
    frames = {p["id"]: sf.pano_street(p, data) for p in panos}
    tiles = sf.plan_tiles(walls, panos, data)
    cached = _cached_tiles(tiles, sf.CACHE)
    log(f"crops: {len(cached)}/{len(tiles)} planned tiles cached in {sf.CACHE}")
    shots = sf.fetch_tiles(cached, src, None)
    assert src.requests["image"] == 0, "crops must not make new Street View requests"
    poses_path = sf.CACHE.parent / "poses.json"
    poses = json.loads(poses_path.read_text()) if poses_path.exists() else {}
    occ = sf.occluders(data)
    crop_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for bid in sorted(ids):
        ws = [w for w in walls if w["building"] == bid and w["street"] == "moss"]
        if not ws:
            log(f"  {bid}: no Moss-facing wall in moss.json")
            seen[bid] = 0.0
            continue
        lens = np.array([np.linalg.norm(w["b"] - w["a"]) for w in ws])
        n = sf.unit(sum(w["normal"] * L for w, L in zip(ws, lens)))
        right = np.array([-n[1], n[0]])                      # viewer faces -n
        s = np.array([[w["a"] @ right, w["b"] @ right] for w in ws])
        x0, x1 = s.min(), s.max()
        top = max(w["z1"] for w in ws)
        W, H = max(2, round((x1 - x0) * PPM)), max(2, round(top * PPM))
        canvas = np.full((H, W, 3), 128, np.uint8)
        painted = np.zeros((H, W), bool)
        area = seen_area = 0.0
        mid = np.mean([(w["a"] + w["b"]) / 2 for w in ws], axis=0)
        near = [sh for sh in shots if math.dist((sh["pano"]["x"], sh["pano"]["y"]), mid) < 30]
        for w, (sa, sb) in sorted(zip(ws, s), key=lambda t: ((t[0]["a"] + t[0]["b"]) / 2) @ n):
            tex, clean, _, _ = sf.wall_texture(w, near, frames, poses, occ, ppm=PPM)
            if sa > sb:
                tex = tex[:, ::-1]
            c0, c1 = round((min(sa, sb) - x0) * PPM), round((max(sa, sb) - x0) * PPM)
            r0, r1 = round((top - w["z1"]) * PPM), round((top - w["z0"]) * PPM)
            if c1 - c0 < 2 or r1 - r0 < 2:
                continue
            canvas[r0:r1, c0:c1] = cv2.resize(tex, (c1 - c0, r1 - r0), interpolation=cv2.INTER_AREA)
            painted[r0:r1, c0:c1] = w["normal"] @ n > FRONT_COS   # bay sides pick up neighbours: no colour votes
            a = (c1 - c0) * (r1 - r0)
            area, seen_area = area + a, seen_area + a * clean
        seen[bid] = seen_area / area if area else 0.0
        cv2.imwrite(str(crop_dir / f"{bid}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(crop_dir / f"{bid}.mask.png"), painted.astype(np.uint8) * 255)
        log(f"  {bid}: {len(ws)} walls, {W}x{H} px, seen {seen[bid]:.0%}")
    return seen


def fetch_view(bid: int, crop_dir: Path, budget: list[int], log=print) -> bool:
    """One Static API view (cached in LOOK_CACHE) aimed at the footprint centre from the nearest Moss panorama
    at least 5 m away, for a building without a usable crop. budget[0] counts new images against MAX_NEW_IMAGES."""
    import cv2
    import streetview_facades as sf
    from kite.gps import sim_to_latlng
    from kite.sources.streetview import StreetViewSource, in_scope
    data = json.loads(sf.MOSS.read_text())
    b = next(b for b in data["buildings"] if b["id"] == bid)
    c = np.mean(np.array(b["pts"][:-1], float), axis=0)
    panos = json.loads((sf.CACHE.parent / "panos.json").read_text())
    p = min((p for p in panos if 5 < math.dist((p["x"], p["y"]), c) and in_scope(*sim_to_latlng(p["x"], p["y"]))),
            key=lambda p: math.dist((p["x"], p["y"]), c))
    params = {"pano": p["id"], "heading": f"{sf.compass(c[0] - p['x'], c[1] - p['y']):.1f}", "pitch": "12",
              "fov": "90", "size": "640x640", "return_error_code": "true"}
    src = StreetViewSource(single=(37.7783, -122.4087, 0), cache_dir=LOOK_CACHE)
    import hashlib
    import urllib.parse
    cached = (LOOK_CACHE / f"image-{hashlib.sha256(urllib.parse.urlencode(params).encode()).hexdigest()[:24]}").is_file()
    if not cached and budget[0] >= MAX_NEW_IMAGES:
        log(f"  {bid}: no new Static view, the {MAX_NEW_IMAGES} image cap is reached")
        return False
    img = cv2.imdecode(np.frombuffer(src._get("image", params), np.uint8), cv2.IMREAD_COLOR)
    budget[0] += src.requests["image"]
    if img is None:
        return False
    cv2.imwrite(str(crop_dir / f"{bid}.view.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    log(f"  {bid}: Static view from {p['id']} ({'cached' if cached else 'new'})")
    return True


# ---------------------------------------------------------------- Claude vision
PROMPT = """These are Street View images of ONE building on Moss St, San Francisco ({address}, about {h:.0f} m tall).
{images}
Describe the building in the middle of the views (ignore neighbours at the edges, cars, trees, smears).
Fields:
- main: hex colour of the main wall surface as it would look in even daylight (not shaded, not the windows)
- trim: hex colour of window/door frames, cornice and trim (often white); same as main if there is none
- cladding: siding (horizontal wood/vinyl boards), stucco (smooth/textured render), brick, concrete, or metal
- floors: number of storeys above the street, including a ground-floor garage level
- bays: number of projecting bay-window columns on the street face (a bay stacked over several floors counts once; 0 if flat)
- garages: number of garage doors / roll-up doors on the street face
- garage_colour: hex colour of the garage doors (use the main colour if there are none)
- gate: an ornate metal/iron security gate at the entry; hedge: a hedge or large planting along the front;
  fire_escape: a metal fire escape on the face; mural: a painted mural covering a real part of the wall
- confidence: 0..1, how sure you are this is one building seen clearly enough to fill these fields
{hint}"""
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(FIELDS),
          "properties": {"main": {"type": "string"}, "trim": {"type": "string"},
                         "cladding": {"type": "string", "enum": list(CLADDING)},
                         "floors": {"type": "integer"}, "bays": {"type": "integer"}, "garages": {"type": "integer"},
                         "garage_colour": {"type": "string"}, "gate": {"type": "boolean"},
                         "hedge": {"type": "boolean"}, "fire_escape": {"type": "boolean"},
                         "mural": {"type": "boolean"}, "confidence": {"type": "number"}}}


def ask_claude(t: dict, crop_dir: Path, cache_dir: Path, model: str, hint: str = "", stats=None) -> dict:
    """Claude's answer for one building (cached by model + hint)."""
    import hashlib
    imgs = [(f, what) for f, what in (
        (crop_dir / f"{t['id']}.jpg", "a rectified, flat front-on mosaic of its street face (grey = unseen; blurry "
                                      "smears are inpainted occluders)"),
        (crop_dir / f"{t['id']}.view.jpg", "a perspective photo aimed at it from the street")) if f.exists()]
    key = hashlib.sha256(f"{model}|{hint}|{[f.name for f, _ in imgs]}".encode()).hexdigest()[:8]
    path = cache_dir / f"{t['id']}.{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    import anthropic
    content, names = [], []
    for f, what in imgs:
        if f.exists():
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": base64.standard_b64encode(f.read_bytes()).decode()}})
            names.append(f"Image {len(names) + 1}: {what}.")
    if not content:
        raise RuntimeError(f"{t['id']}: no image to describe")
    text = PROMPT.format(address=f"No. {t['address']}" if t["address"] != "-" else "no street number",
                         h=t.get("h", 8.0), images="\n".join(names),
                         hint=f"\nA person who checked an earlier answer says: {hint}" if hint else "")
    content.append({"type": "text", "text": text})
    client = anthropic.Anthropic()
    r = client.messages.create(model=model, max_tokens=4000, output_config={
        "effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": content}])
    if stats is not None:
        stats["calls"] += 1
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"{t['id']}: Claude stopped with {r.stop_reason}")
    ans = json.loads(next(b.text for b in r.content if b.type == "text"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ans, indent=1))
    return ans


# ---------------------------------------------------------------- colours
def hex_to_rgb(h: str) -> np.ndarray | None:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(h).strip())
    return np.array([int(m[1][i:i + 2], 16) for i in (0, 2, 4)], float) if m else None


def rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{int(round(min(255, max(0, v)))):02x}" for v in rgb)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) sRGB 0-255 -> CIE Lab (L 0-100), D65."""
    c = np.asarray(rgb, float).reshape(-1, 3) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    L, a, b = np.asarray(lab, float).reshape(-1, 3).T
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], 1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    c = xyz @ np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]).T
    c = np.where(c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(c * 255, 0, 255)


def _pick(lab: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Pixels (Lab) of the surface Claude named: within HUE_TOL of its hue and at least a third of its chroma if
    it named a colour; neutral and within 25 L* if it named a grey/white. Then their brighter half (shade)."""
    C, Cg = np.hypot(lab[:, 1], lab[:, 2]), math.hypot(g[1], g[2])
    if Cg >= NEUTRAL_C:
        dh = np.abs((np.degrees(np.arctan2(lab[:, 2], lab[:, 1]) - math.atan2(g[2], g[1])) + 180) % 360 - 180)
        ok = (dh < HUE_TOL) & (C >= Cg / 3)
    else:
        ok = (C < max(NEUTRAL_C, Cg + 6)) & (np.abs(lab[:, 0] - g[0]) < 25)
    q = lab[ok & (lab[:, 0] > 20) & (lab[:, 0] < 97)]
    return q[q[:, 0] >= np.median(q[:, 0])] if len(q) else q


def wall_colours(rgb: np.ndarray, guess_main, guess_trim) -> tuple[str, str, float]:
    """(main hex, trim hex, share of pixels backing main) from the crop's front-wall pixels (N, 3 RGB) and
    Claude's guesses. The guesses say WHICH surface is the wall (windows, trim, cars and murals share the crop)
    but over-saturate; hue and chroma come from the matching pixels' brighter half. Exposure: a white-patch gain
    (the crop's 97th-percentile luminance to WHITE_Y, capped at MAX_GAIN: the SW fronts face NE, in shade), then
    L* kept within L_TOL of Claude's reading, chroma at least MIN_CHROMA of Claude's, and Claude's hue for
    near-neutral walls. Too few matching pixels: Claude's guess stands."""
    rgb = np.asarray(rgb, float).reshape(-1, 3)
    lin = np.where(rgb / 255 > 0.04045, ((rgb / 255 + 0.055) / 1.055) ** 2.4, rgb / 255 / 12.92)
    gain = float(np.clip(WHITE_Y / max(np.percentile(lin @ [0.2126, 0.7152, 0.0722], 97), 1e-3), 1.0, MAX_GAIN))
    lab = _lab(rgb)
    lab[:, 0] = np.clip((lab[:, 0] + 16) * gain ** (1 / 3) - 16, 0, 100)   # L* + 16 ~ Y^(1/3)
    out, share = [], 0.0
    for i, g in enumerate((guess_main, guess_trim)):
        if g is None:
            out.append(None)
            continue
        gl = _lab(g)[0]
        q = _pick(lab, gl)
        if len(q) * 2 < MIN_SHARE * len(lab):
            out.append(rgb_to_hex(g))
            continue
        c = np.median(q, axis=0)
        c[0] = np.clip(c[0], gl[0] - L_TOL, gl[0] + L_TOL)
        cg, cc = math.hypot(gl[1], gl[2]), math.hypot(c[1], c[2])
        if cg < NEUTRAL_C:
            c[1:] = gl[1:]          # a near-white/grey wall's hue in the photo is the light's cast (blue in shade)
        elif cc < MIN_CHROMA * cg:
            c[1:] *= MIN_CHROMA * cg / max(cc, 1e-6)   # blending views and haze wash the mosaic out
        out.append(rgb_to_hex(_from_lab(c)[0]))
        if i == 0:
            share = min(1.0, 2 * len(q) / len(lab))
    main = out[0] or rgb_to_hex(_from_lab(np.median(lab, axis=0))[0])
    return main, out[1] or main, round(share, 3)


# ---------------------------------------------------------------- corrections
def parse_note(note: str) -> tuple[dict, str]:
    """'main=#c8b48a bays=2 gate=no rest of text' -> ({main, bays, gate}, 'rest of text')."""
    over, rest = {}, []
    for tok in str(note or "").split():
        k, eq, v = tok.partition("=")
        typ = FIELDS.get(k.strip().lower())
        if not eq or typ is None:
            rest.append(tok)
            continue
        k = k.strip().lower()
        try:
            if typ is bool:
                over[k] = v.lower() in ("1", "y", "yes", "true")
            elif typ is str:
                if k == "cladding" and v.lower() in CLADDING:
                    over[k] = v.lower()
                elif k != "cladding" and hex_to_rgb(v) is not None:
                    over[k] = rgb_to_hex(hex_to_rgb(v))
                else:
                    rest.append(tok)
            else:
                over[k] = typ(v)
        except ValueError:
            rest.append(tok)
    return over, " ".join(rest)


def clean(ans: dict) -> dict:
    """Claude's answer -> the schema's types and ranges (bad values become safe defaults)."""
    out = {}
    for k, typ in FIELDS.items():
        v = ans.get(k)
        if typ is str:
            out[k] = (v if v in CLADDING else "stucco") if k == "cladding" else \
                rgb_to_hex(hex_to_rgb(v)) if hex_to_rgb(v) is not None else "#b0aca4"
        elif typ is bool:
            out[k] = bool(v)
        elif typ is int:
            out[k] = max(0, min(12, int(v or 0)))
        else:
            out[k] = round(max(0.0, min(1.0, float(v or 0))), 2)
    out["floors"] = max(1, out["floors"])
    return out


# ---------------------------------------------------------------- review page
PAGE = """<!doctype html><meta charset="utf-8"><title>Moss St buildings, review</title>
<style>body{font:14px system-ui;margin:16px;background:#f4f3f0}h2{margin:24px 0 8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}
.card{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px #0002}.card.wrong{outline:3px solid #d33}
.imgs{display:flex;gap:6px}.imgs img{height:190px;max-width:100%;object-fit:contain;background:#ddd;border-radius:4px}
.sw{display:inline-block;width:46px;height:22px;border:1px solid #0003;border-radius:3px;vertical-align:middle}
table{border-collapse:collapse;margin:6px 0}td{padding:1px 8px 1px 0}textarea{width:100%;height:38px}
.low{color:#c60;font-weight:600}button{font-size:15px;padding:6px 14px}</style>
<h1>Moss St buildings</h1><p>Tick <b>wrong</b> on the cards that don't match the photo; the note can fix fields
directly (<code>main=#c8b48a bays=2 gate=no cladding=brick</code>) or say what's wrong in words. Then
<button onclick="exportFix()">Export corrections.json</button> and run
<code>python scripts/moss_look.py --fix corrections.json</code>.</p><div id="root"></div>
<script>const DATA=__DATA__;
const st=JSON.parse(localStorage.getItem('mossfix')||'{}');const save=()=>localStorage.setItem('mossfix',JSON.stringify(st));
const root=document.getElementById('root');
for(const side of ['NE','SW']){const h=document.createElement('h2');h.textContent=side+' side, Howard to Folsom';root.append(h);
const g=document.createElement('div');g.className='grid';root.append(g);
for(const b of DATA.filter(b=>b.side===side)){const v=b.look,c=document.createElement('div');c.className='card';
const s=st[b.id]||{};if(s.wrong)c.classList.add('wrong');
c.innerHTML=`<b>${b.address}</b> &middot; ${b.id} &middot; ${b.m} m from Howard
<div class="imgs">${b.imgs.map(f=>`<a href="${f}" target="_blank"><img src="${f}"></a>`).join('')}</div>
<div>main <span class="sw" style="background:${v.main}"></span> ${v.main} &nbsp; trim <span class="sw" style="background:${v.trim}"></span> ${v.trim}
&nbsp; garage <span class="sw" style="background:${v.garage_colour}"></span></div>
<table><tr><td>${v.cladding}</td><td>${v.floors} floors</td><td>${v.bays} bays</td><td>${v.garages} garages</td></tr>
<tr><td>gate ${v.gate?'yes':'no'}</td><td>hedge ${v.hedge?'yes':'no'}</td><td>fire escape ${v.fire_escape?'yes':'no'}</td><td>mural ${v.mural?'yes':'no'}</td></tr>
<tr><td colspan=4 class="${v.confidence<0.6?'low':''}">confidence ${v.confidence}${b.fixed?' &middot; corrected':''}</td></tr></table>
<label><input type="checkbox" ${s.wrong?'checked':''}> wrong</label><textarea placeholder="optional: what's wrong">${s.note||''}</textarea>`;
const [cb,ta]=[c.querySelector('input'),c.querySelector('textarea')];
cb.onchange=()=>{st[b.id]={...st[b.id],wrong:cb.checked};c.classList.toggle('wrong',cb.checked);save()};
ta.oninput=()=>{st[b.id]={...st[b.id],note:ta.value};save()};g.append(c)}}
function exportFix(){const out={};for(const[id,s]of Object.entries(st))if(s.wrong)out[id]={wrong:true,note:s.note||''};
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));
a.download='corrections.json';a.click()}</script>"""


def write_page(path: Path, ts: list[dict], looks: dict, fixed: set, crop_dir: Path) -> None:
    rows = [{"id": t["id"], "side": t["side"], "m": t["m"], "address": html.escape(t["address"]),
             "look": looks[str(t["id"])], "fixed": t["id"] in fixed,
             "imgs": [f"crops/{f.name}" for f in (crop_dir / f"{t['id']}.jpg", crop_dir / f"{t['id']}.view.jpg")
                      if f.exists()]} for t in ts if str(t["id"]) in looks]
    path.write_text(PAGE.replace("__DATA__", json.dumps(rows).replace("</", "<\\/")))


# ---------------------------------------------------------------- main
def load_env(repo: Path) -> None:
    env = repo / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        k, eq, v = line.partition("=")
        if eq and k.strip() and not k.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=ROOT, help="checkout whose sim/public/local and out/ to use")
    ap.add_argument("--fix", type=Path, help="corrections.json exported from buildings.html")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--recrop", action="store_true", help="rebuild the crops (else reuse out/look/crops)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import cv2
    load_env(a.repo)
    look = a.repo / "out/look"
    crop_dir, claude_dir = look / "crops", look / "claude"
    ts = targets(a.repo)
    ids = {t["id"] for t in ts}
    seen_path = crop_dir / "seen.json"
    if a.recrop or not seen_path.exists():
        seen = make_crops(ids, crop_dir)
        seen_path.write_text(json.dumps({str(k): v for k, v in seen.items()}))
    seen = {int(k): v for k, v in json.loads(seen_path.read_text()).items()}
    data = json.loads((ROOT / "sim/public/osm/moss.json").read_text())
    heights = {b["id"]: float(b["h"]) for b in data["buildings"]}
    budget = [0]
    for t in ts:
        t["h"] = heights.get(t["id"], 8.0)
        if seen.get(t["id"], 0) < MIN_SEEN:
            (crop_dir / f"{t['id']}.jpg").unlink(missing_ok=True)       # mostly unseen: the Static view only
            fetch_view(t["id"], crop_dir, budget)
    fix_path = look / "fixes.json"
    fixes = json.loads(fix_path.read_text()) if fix_path.exists() else {}
    if a.fix:
        fixes.update({str(k): v for k, v in json.loads(a.fix.read_text()).items() if v.get("wrong")})
        fix_path.write_text(json.dumps(fixes, indent=1))
    stats = {"calls": 0}
    looks = {}
    for t in ts:
        sid = str(t["id"])
        over, words = parse_note(fixes.get(sid, {}).get("note", ""))
        try:
            ans = ask_claude(t, crop_dir, claude_dir, a.model, stats=stats)
            view = crop_dir / f"{sid}.view.jpg"
            if not view.exists() and (ans.get("confidence", 0) < MIN_CONF or seen.get(t["id"], 0) < MIN_SEEN_SURE):
                if fetch_view(t["id"], crop_dir, budget):        # unsure from the crop: add a perspective photo
                    ans = ask_claude(t, crop_dir, claude_dir, a.model, stats=stats)
            if words:
                ans = ask_claude(t, crop_dir, claude_dir, FIX_MODEL, hint=words, stats=stats)
        except Exception as e:                       # one bad building must not lose the others
            print(f"  {sid}: Claude failed: {type(e).__name__}: {e}")
            continue
        v = clean(ans)
        img = cv2.imread(str(crop_dir / f"{sid}.jpg"))
        if img is None:
            img = cv2.imread(str(crop_dir / f"{sid}.view.jpg"))
            mask = np.ones(img.shape[:2], bool) if img is not None else None
        else:
            m = cv2.imread(str(crop_dir / f"{sid}.mask.png"), cv2.IMREAD_GRAYSCALE)
            mask = m > 0 if m is not None and (m > 0).mean() > 0.2 else np.ones(img.shape[:2], bool)
        share = 0.0
        if img is not None:
            v["main"], v["trim"], share = wall_colours(img[mask][:, ::-1], hex_to_rgb(v["main"]),
                                                       hex_to_rgb(v["trim"]))
        if not seen.get(t["id"]) or share < MIN_SHARE:     # no crop, or the pixels don't back Claude's colour
            v["confidence"] = round(v["confidence"] * 0.8, 2)
        if sid in fixes:
            v.update(over)
            if not over and not words:
                v["confidence"] = min(v["confidence"], 0.2)    # flagged wrong, nothing said: mark it doubtful
            elif not words:
                v["confidence"] = max(v["confidence"], 0.9)
        looks[sid] = v
    out = a.repo / "sim/public/local/moss_buildings.json"
    out.write_text(json.dumps({"buildings": looks}, indent=1))
    look.mkdir(parents=True, exist_ok=True)
    write_page(look / "buildings.html", ts, looks, {int(k) for k in fixes}, crop_dir)
    print(f"\n{'id':>10} {'side':4} {'address':16} {'main':8} {'trim':8} {'cladding':9} fl bays gar  conf")
    for t in ts:
        v = looks.get(str(t["id"]))
        if v:
            print(f"{t['id']:>10} {t['side']:4} {t['address'][:16]:16} {v['main']:8} {v['trim']:8} {v['cladding']:9} "
                  f"{v['floors']:2} {v['bays']:4} {v['garages']:3}  {v['confidence']:.2f}")
    print(f"\n{len(looks)}/{len(ts)} buildings -> {out}\nreview: {look / 'buildings.html'}\n"
          f"Static API images fetched: {budget[0]}; Claude calls: {stats['calls']} (cached answers reused)")
    return 0


def selftest() -> int:
    rng = np.random.default_rng(1)
    wall, trim, glass = np.array([200, 170, 120]), np.array([245, 245, 240]), np.array([25, 30, 35])
    px = np.vstack([wall * rng.uniform(0.55, 1.0, (6000, 1)),            # sunlit and shaded wall
                    np.tile(trim, (1500, 1)), np.tile(glass, (2500, 1))]) + rng.normal(0, 3, (10000, 3))
    guess = np.array([225, 175, 105])                                     # right hue, a bit too saturated
    main, tr, share = wall_colours(px.clip(0, 255), guess, trim)
    assert np.abs(_lab(hex_to_rgb(main)) - _lab(wall)).max() < 8, (main, "should be the sunlit wall, not the guess")
    assert np.abs(_lab(hex_to_rgb(tr)) - _lab(trim)).max() < 6, tr
    assert 0.4 < share < 0.8, share
    dark = (px * 0.82).clip(0, 255)                                       # underexposed: the gain lifts it back
    assert abs(_lab(hex_to_rgb(wall_colours(dark, guess, trim)[0]))[0, 0] - _lab(hex_to_rgb(main))[0, 0]) < 5
    m2, _, s2 = wall_colours(np.tile(glass, (500, 1)).astype(float), np.array([40, 90, 200]), None)
    assert m2 == "#285ac8" and s2 == 0.0, (m2, s2)                         # nothing blue: Claude's colour stands
    assert np.allclose(_from_lab(_lab(np.array([10, 128, 250])))[0], [10, 128, 250], atol=1)
    over, rest = parse_note("main=#C8B48A bays=2 gate=no cladding=brick hedge=yes it has two garages")
    assert over == {"main": "#c8b48a", "bays": 2, "gate": False, "cladding": "brick", "hedge": True}, over
    assert rest == "it has two garages", rest
    v = clean({"main": "red", "cladding": "wood", "floors": 0, "bays": "2", "confidence": 3})
    assert v["main"] == "#b0aca4" and v["cladding"] == "stucco" and v["floors"] == 1 and v["bays"] == 2
    assert v["confidence"] == 1.0 and set(v) == set(FIELDS)
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
