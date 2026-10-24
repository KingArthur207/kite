"""Moss St (Howard to Folsom, plus the Howard corner) in Blender, with real facades, exported for the sim.

    blender -b -P sim/blender/build_real_moss.py -- --export          # from the repo root
    # or in a running Blender:  exec(open("<repo>/sim/blender/build_real_moss.py").read())

Reads sim/public/osm/moss.json (metres, x east, y north, origin the Moss/Howard zebra) and, if present, the
Street View facades from scripts/streetview_facades.py in sim/public/local/facades/ (gitignored). Builds a
"MossReal" scene: buildings extruded from the OSM footprints, street-facing walls textured with the rectified
photos (UVs: u along the wall, v up, 0..1 over the wall) with the detected windows recessed 12 cm (glossier
glass), doors and garages 25-30 cm with a step in front of doors, and a cornice coloured from the top of
the photo; other walls in the Poly Haven materials the sim uses; flat roofs; asphalt, 15 cm sidewalks with
curb faces, zebras and lane lines; OSM street furniture; wooden poles with a crossarm, a cobra-head light
and three wires along Moss's SW curb. Blender x east, y north, z up, metres; the glTF export (+Y up) maps
that to three.js (x, up, -north), the frame osmWorld.ts uses. `--export` writes
sim/public/local/moss_real.glb (embedded textures). This file is code only: no imagery is committed.
"""
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy

try:
    ROOT = Path(__file__).resolve().parents[2]
except NameError:  # exec() in a live Blender: set KITE_ROOT first, or run from the repo root
    ROOT = Path(globals().get("KITE_ROOT", ".")).resolve()
MOSS = ROOT / "sim/public/osm/moss.json"
FACADES = ROOT / "sim/public/local/facades"
TEX = ROOT / "sim/public/textures"
OUT_GLB = ROOT / "sim/public/local/moss_real.glb"
SCENE = "MossReal"

CURB_H = 0.15
BAY_Z = 3.0
MARK_Z = 0.012
RECESS = {"window": 0.12, "door": 0.25, "garage": 0.3}
CORNICE_H, CORNICE_OUT = 0.35, 0.25
MOSS_ROAD = 8916291
POLE_EVERY, POLE_FIRST = 32.0, 8.0
WIRE_MAX_SPAN = 45.0
WIRE_SAG = 0.35
STOREY, WIN_EVERY = 3.2, 2.6
FALLBACK = [("stucco", 2.0), ("plaster", 3.2), ("brick", 3.0), ("siding", 1.8), ("concrete", 2.7)]


# ---------------------------------------------------------------- materials
_mats = {}


def _principled(mat):
    return next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def _image(path, colour):
    img = bpy.data.images.load(str(path), check_existing=True)
    if not colour:
        img.colorspace_settings.is_data = True
    return img


def _tex_node(mat, path, colour, extend=False):
    n = mat.node_tree.nodes.new("ShaderNodeTexImage")
    n.image = _image(path, colour)
    if extend:
        n.extension = next(i.identifier for i in n.bl_rna.properties["extension"].enum_items if i.identifier == "EXTEND")
    return n


def solid(name, rgb, rough=0.8, metal=0.0):
    if name in _mats:
        return _mats[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = _principled(m)
    p.inputs["Base Color"].default_value = (*rgb, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    _mats[name] = m
    return m


def pbr(name, rough_map=False, tint=None):
    """Poly Haven set from sim/public/textures: <name>_diff.jpg, _nor.jpg (OpenGL), optional _rough.jpg."""
    key = f"pbr_{name}" + (f"_{tint}" if tint else "")
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt, p = m.node_tree, _principled(m)
    diff = _tex_node(m, TEX / f"{name}_diff.jpg", True)
    if tint:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = next(i.identifier for i in mix.bl_rna.properties["blend_type"].enum_items if i.identifier == "MULTIPLY")
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs[7].default_value = (*tint, 1)
        nt.links.new(diff.outputs["Color"], mix.inputs[6])
        nt.links.new(mix.outputs[2], p.inputs["Base Color"])
    else:
        nt.links.new(diff.outputs["Color"], p.inputs["Base Color"])
    if (TEX / f"{name}_nor.jpg").exists():
        nor = _tex_node(m, TEX / f"{name}_nor.jpg", False)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(nor.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], p.inputs["Normal"])
    if rough_map:
        r = _tex_node(m, TEX / f"{name}_rough.jpg", False)
        nt.links.new(r.outputs["Color"], p.inputs["Roughness"])
    else:
        p.inputs["Roughness"].default_value = 0.9
    _mats[key] = m
    return m


def photo(file, glass=False):
    key = f"facade_{file}" + ("_glass" if glass else "")
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    p = _principled(m)
    t = _tex_node(m, FACADES / file, True, extend=True)
    m.node_tree.links.new(t.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.12 if glass else 0.85
    _mats[key] = m
    return m


# ---------------------------------------------------------------- mesh builder
class Mesh:
    """Quads/ngons with per-corner UVs, one material slot per material."""

    def __init__(self, name):
        self.name, self.verts, self.faces, self.uvs, self.mats, self.slots = name, [], [], [], [], {}

    def face(self, pts, uv, mat):
        i = len(self.verts)
        self.verts += [tuple(p) for p in pts]
        self.faces.append(list(range(i, i + len(pts))))
        self.uvs.append(uv)
        self.mats.append(self.slots.setdefault(mat.name, (len(self.slots), mat))[0])

    def quad_facing(self, pts, uv, mat, normal):
        """A quad wound so its normal points along `normal` (x, y, z)."""
        a, b, c = pts[0], pts[1], pts[2]
        e1 = [b[k] - a[k] for k in range(3)]
        e2 = [c[k] - a[k] for k in range(3)]
        n = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
        if sum(n[k] * normal[k] for k in range(3)) < 0:
            pts, uv = pts[::-1], uv[::-1]
        self.face(pts, uv, mat)

    def box(self, centre, size, yaw, mat, uv_scale=1.0):
        """Axis box (sx along yaw direction, sy across, sz up) with its base centre at `centre`."""
        cx, cy, cz = centre
        sx, sy, sz = size
        c, s = math.cos(yaw), math.sin(yaw)
        def P(x, y, z):
            return (cx + x * c - y * s, cy + x * s + y * c, cz + z)
        hx, hy = sx / 2, sy / 2
        corners = [P(-hx, -hy, 0), P(hx, -hy, 0), P(hx, hy, 0), P(-hx, hy, 0)]
        tops = [P(-hx, -hy, sz), P(hx, -hy, sz), P(hx, hy, sz), P(-hx, hy, sz)]
        k = uv_scale
        self.face(tops, [(0, 0), (sx * k, 0), (sx * k, sy * k), (0, sy * k)], mat)
        for i in range(4):
            j = (i + 1) % 4
            L = math.dist(corners[i][:2], corners[j][:2])
            self.face([corners[i], corners[j], tops[j], tops[i]], [(0, 0), (L * k, 0), (L * k, sz * k), (0, sz * k)], mat)

    def cyl(self, centre, r0, r1, h, mat, n=10):
        cx, cy, cz = centre
        ring = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
        for i in range(n):
            (a, b), (c, d) = ring[i], ring[(i + 1) % n]
            self.face([(cx + a * r0, cy + b * r0, cz), (cx + c * r0, cy + d * r0, cz),
                       (cx + c * r1, cy + d * r1, cz + h), (cx + a * r1, cy + b * r1, cz + h)],
                      [(i / n, 0), ((i + 1) / n, 0), ((i + 1) / n, h), (i / n, h)], mat)
        self.face([(cx + a * r1, cy + b * r1, cz + h) for a, b in ring], [(a, b) for a, b in ring], mat)

    def tube(self, a, b, r0, r1, mat, n=6):
        """A tapered cylinder from point a to point b (3D)."""
        d = [b[i] - a[i] for i in range(3)]
        L = math.sqrt(sum(x * x for x in d)) or 1e-6
        w = [x / L for x in d]
        t = (1, 0, 0) if abs(w[0]) < 0.9 else (0, 1, 0)
        u = [w[1] * t[2] - w[2] * t[1], w[2] * t[0] - w[0] * t[2], w[0] * t[1] - w[1] * t[0]]
        un = math.sqrt(sum(x * x for x in u))
        u = [x / un for x in u]
        v = [w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0]]
        ring = lambda c, r, i: tuple(c[k] + r * (math.cos(2 * math.pi * i / n) * u[k] + math.sin(2 * math.pi * i / n) * v[k]) for k in range(3))  # noqa: E731
        for i in range(n):
            self.face([ring(a, r0, i), ring(a, r0, i + 1), ring(b, r1, i + 1), ring(b, r1, i)],
                      [(i / n, 0), ((i + 1) / n, 0), ((i + 1) / n, 1), (i / n, 1)], mat)

    def blob(self, centre, r, mat, n=12):
        """A low-poly squashed sphere (tree canopy)."""
        cx, cy, cz = centre
        rows = [(math.sin(math.pi * j / 5), math.cos(math.pi * j / 5)) for j in range(6)]
        for j in range(5):
            for i in range(n):
                a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
                (s0, c0), (s1, c1) = rows[j], rows[j + 1]
                pts = [(cx + r * s0 * math.cos(a0), cy + r * s0 * math.sin(a0), cz + 0.8 * r * c0),
                       (cx + r * s1 * math.cos(a0), cy + r * s1 * math.sin(a0), cz + 0.8 * r * c1),
                       (cx + r * s1 * math.cos(a1), cy + r * s1 * math.sin(a1), cz + 0.8 * r * c1),
                       (cx + r * s0 * math.cos(a1), cy + r * s0 * math.sin(a1), cz + 0.8 * r * c0)]
                if j == 0:
                    pts = [pts[0], pts[1], pts[2]]
                elif j == 4:
                    pts = [pts[0], pts[1], pts[3]]
                self.face(pts, [(0, 0)] * len(pts), mat)

    def build(self, coll):
        me = bpy.data.meshes.new(self.name)
        me.from_pydata(self.verts, [], self.faces)
        for _, m in sorted(self.slots.values(), key=lambda t: t[0]):
            me.materials.append(m)
        me.polygons.foreach_set("material_index", self.mats)
        uvl = me.uv_layers.new(name="UVMap")
        loop_uv = [c for uv in self.uvs for xy in uv for c in xy]
        uvl.data.foreach_set("uv", loop_uv)
        me.validate()
        me.update()
        ob = bpy.data.objects.new(self.name, me)
        coll.objects.link(ob)
        return ob


# ---------------------------------------------------------------- buildings
def signed_area(pts):
    return 0.5 * sum(pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1] for i in range(len(pts) - 1))


def seeded(i):
    a = (i * 2654435761) & 0xFFFFFFFF
    return ((a ^ (a >> 15)) % 10007) / 10007


def on_boundary(p, poly, tol=0.05):
    for a, b in zip(poly, poly[1:]):
        ex, ey = b[0] - a[0], b[1] - a[1]
        L2 = ex * ex + ey * ey or 1e-12
        t = max(0.0, min(1.0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / L2))
        if math.hypot(a[0] + t * ex - p[0], a[1] + t * ey - p[1]) < tol:
            return True
    return False


def building_edges(bd):
    """[(key, a, b, z0, z1, outward normal)], as scripts/streetview_facades.py building_edges(): roof-outline
    edges (bays overhanging from bay_z) and the ground-floor edges under them (keys "g<k>")."""
    out, h = [], float(bd["h"])
    ground, bay_z = bd.get("ground"), float(bd.get("bay_z", BAY_Z))
    for poly, is_ground in ((bd["pts"], False), (ground, True)):
        if poly is None:
            continue
        ccw = signed_area(poly) > 0
        for k in range(len(poly) - 1):
            a, b = poly[k], poly[k + 1]
            L = math.dist(a, b)
            if L < 0.3:
                continue
            ex, ey = (b[0] - a[0]) / L, (b[1] - a[1]) / L
            nrm = (ey, -ex) if ccw else (-ey, ex)
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if is_ground:
                if not on_boundary(mid, bd["pts"]):
                    out.append((f"g{k}", a, b, 0.0, min(bay_z, h), nrm))
            else:
                z0 = bay_z if ground is not None and not on_boundary(mid, ground) else 0.0
                if z0 < h:
                    out.append((k, a, b, z0, h, nrm))
    return out


def textured_wall(m, a, b, z0, h, nrm, entry):
    """Grid over the wall split at every opening edge; cells inside an opening are pushed in, with reveals."""
    L = math.dist(a, b)
    e = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
    wall, glass = photo(entry["file"]), photo(entry["file"], glass=True)
    ops = [o for o in entry.get("openings", []) if 0 <= o[1] < o[3] <= L and z0 <= o[2] < o[4] <= h]

    def P(s, z, depth=0.0):
        return (a[0] + e[0] * s - nrm[0] * depth, a[1] + e[1] * s - nrm[1] * depth, z)

    def UV(s, z):
        return (s / L, (z - z0) / (h - z0))

    out = (nrm[0], nrm[1], 0.0)
    xs = sorted({0.0, L, *[o[1] for o in ops], *[o[3] for o in ops]})
    zs = sorted({z0, h, *[o[2] for o in ops], *[o[4] for o in ops]})
    for s0, s1 in zip(xs, xs[1:]):
        for z0, z1 in zip(zs, zs[1:]):
            sc, zc = (s0 + s1) / 2, (z0 + z1) / 2
            hit = next((o for o in ops if o[1] < sc < o[3] and o[2] < zc < o[4]), None)
            d = RECESS[hit[0]] if hit else 0.0
            mat = glass if hit and hit[0] == "window" else wall
            m.quad_facing([P(s0, z0, d), P(s1, z0, d), P(s1, z1, d), P(s0, z1, d)],
                          [UV(s0, z0), UV(s1, z0), UV(s1, z1), UV(s0, z1)], mat, out)
    for kind, s0, z0, s1, z1 in ops:
        d = RECESS[kind]
        for (sa, za), (sb, zb), inward in (((s0, z0), (s1, z0), (0, 0, 1)), ((s1, z1), (s0, z1), (0, 0, -1)),
                                           ((s0, z1), (s0, z0), (e[0], e[1], 0)), ((s1, z0), (s1, z1), (-e[0], -e[1], 0))):
            m.quad_facing([P(sa, za), P(sb, zb), P(sb, zb, d), P(sa, za, d)],
                          [UV(sa, za), UV(sb, zb), UV(sb, zb), UV(sa, za)], wall, inward)
        if kind == "door" and 0.25 < z0 < 0.9:   # a raised door: a stoop, one 15 cm step per 15 cm of sill
            steps = max(1, round((z0 - CURB_H) / 0.15))
            for k in range(steps):
                sc = (s0 + s1) / 2
                depth = 0.3 * (steps - k)
                cx, cy = a[0] + e[0] * sc + nrm[0] * depth / 2, a[1] + e[1] * sc + nrm[1] * depth / 2
                m.box((cx, cy, CURB_H), (s1 - s0 + 0.3, depth, (z0 - CURB_H) * (k + 1) / steps),
                      math.atan2(e[1], e[0]), solid("stoop", (0.55, 0.54, 0.52), 0.9))
    if entry.get("top_rgb") and not str(entry["edge"]).startswith("g"):
        col = tuple(c * 0.85 for c in entry["top_rgb"])
        cx, cy = (a[0] + b[0]) / 2 + nrm[0] * CORNICE_OUT / 2, (a[1] + b[1]) / 2 + nrm[1] * CORNICE_OUT / 2
        m.box((cx, cy, h - CORNICE_H), (L + 0.2, CORNICE_OUT, CORNICE_H), math.atan2(e[1], e[0]),
              solid(f"cornice_{entry['building']}_{entry['edge']}", col, 0.85))


def build_buildings(data, manifest, coll):
    """Ground floor from `ground` (bays clipped at the property line) up to bay_z, the roof outline above it
    with a soffit under the overhang. Street walls that passed verification carry their photo; the ones that
    didn't get the plain material (a wrong photo is worse than none)."""
    m = Mesh("buildings")
    roof = pbr("roof")
    glass = solid("window_glass", (0.06, 0.08, 0.10), 0.1, 0.3)
    walls = {(w["building"], str(w["edge"])): w for w in manifest if w.get("file")}
    street = {(w["building"], str(w["edge"])) for w in manifest}
    plain = {(w["building"], str(w["edge"])): w["plain_rgb"] for w in manifest if w.get("plain_rgb")}
    for bd in data["buildings"]:
        h = float(bd["h"])
        kind, size = FALLBACK[int(seeded(bd["id"]) * len(FALLBACK))]
        tint = [(0.95, 0.92, 0.85), (0.85, 0.88, 0.9), (0.9, 0.85, 0.8), (0.8, 0.82, 0.78)][bd["id"] % 4]
        tops = [w["top_rgb"] for (bid, _), w in walls.items() if bid == bd["id"] and w.get("top_rgb")]
        tops = tops or [c for (bid, _), c in plain.items() if bid == bd["id"]]
        if tops:   # other walls of a building seen in the photos: plain stucco in its photo colour
            kind, size = "stucco", 2.0
            tint = tuple(round(min(1.0, 1.15 * sum(t[i] for t in tops) / len(tops)), 2) for i in range(3))
        fb = pbr(kind, tint=tint if kind != "brick" else None)
        for key, a, b, z0, z1, nrm in building_edges(bd):
            L = math.dist(a, b)
            entry = walls.get((bd["id"], str(key)))
            if entry:
                textured_wall(m, a, b, z0, z1, nrm, entry)
                continue
            c = plain.get((bd["id"], str(key)))   # a street wall whose photo didn't verify: its colour only
            mat = pbr("stucco", tint=tuple(round(min(1.0, 1.15 * v), 2) for v in c)) if c else fb
            m.quad_facing([(*a, z0), (*b, z0), (*b, z1), (*a, z1)],
                          [(0, z0 / size), (L / size, z0 / size), (L / size, z1 / size), (0, z1 / size)], mat, (*nrm, 0))
            if not tops and (bd["id"], str(key)) not in street and L > 3 and z1 > 4:
                ex, ey = (b[0] - a[0]) / L, (b[1] - a[1]) / L   # unphotographed, off-street: a plain window grid
                n = int((L - 0.8) // WIN_EVERY)
                for i in range(n):
                    sc = (L - (n - 1) * WIN_EVERY) / 2 + i * WIN_EVERY
                    for f in range(1, int(z1 // STOREY)):
                        zw = f * STOREY + 0.8
                        if zw < z0:
                            continue
                        win = [(a[0] + ex * (sc + dx) + nrm[0] * 0.03, a[1] + ey * (sc + dx) + nrm[1] * 0.03, zw + dz)
                               for dx, dz in ((-0.6, 0), (0.6, 0), (0.6, 1.6), (-0.6, 1.6))]
                        m.quad_facing(win, [(0, 0), (1, 0), (1, 1), (0, 1)], glass, (*nrm, 0))
        pts = bd["pts"]
        ring = pts[:-1] if pts[0] == pts[-1] else pts
        if signed_area(pts) < 0:
            ring = ring[::-1]
        m.face([(x, y, h) for x, y in ring], [(x / 4, y / 4) for x, y in ring], roof)
        if bd.get("ground"):   # soffit: the outline at bay_z facing down (inside the ground floor it's hidden)
            bz = float(bd.get("bay_z", BAY_Z))
            m.face([(x, y, bz) for x, y in ring[::-1]], [(x / size, y / size) for x, y in ring[::-1]], fb)
    return m.build(coll)


# ---------------------------------------------------------------- streets
def in_quad(p, q):
    s = [(q[(i + 1) % 4][0] - q[i][0]) * (p[1] - q[i][1]) - (q[(i + 1) % 4][1] - q[i][1]) * (p[0] - q[i][0]) for i in range(4)]
    return all(v >= 0 for v in s) or all(v <= 0 for v in s)


def ribbon(pts, w):
    out = []
    for a, b in zip(pts, pts[1:]):
        d = math.dist(a, b)
        if d < 1e-3:
            continue
        nx, ny = -(b[1] - a[1]) / d * w / 2, (b[0] - a[0]) / d * w / 2
        out.append([(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)])
    return out


def dashes(pts, w, on=3.0, off=9.0):
    out, phase = [], 0.0
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        s = -phase
        while s < L:
            s0, s1 = max(0, s), min(L, s + on)
            if s1 - s0 > 0.2:
                lerp = lambda t: (a[0] + (b[0] - a[0]) * t / L, a[1] + (b[1] - a[1]) * t / L)  # noqa: E731
                out += ribbon([lerp(s0), lerp(s1)], w)
            s += on + off
        phase = (phase + L) % (on + off)
    return out


def offset(pts, o):
    res = []
    for i, p in enumerate(pts):
        a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
        L = math.dist(a, b) or 1
        res.append((p[0] - (b[1] - a[1]) / L * o, p[1] + (b[0] - a[0]) / L * o))
    return res


def flat(m, quads, z, mat, size):
    for q in quads:
        m.quad_facing([(x, y, z) for x, y in q], [(x / size, y / size) for x, y in q], mat, (0, 0, 1))


def build_streets(data, coll):
    m = Mesh("streets")
    m.face([(-400, -400, -0.02), (400, -400, -0.02), (400, 400, -0.02), (-400, 400, -0.02)],
           [(0, 0), (1, 0), (1, 1), (0, 1)], solid("lot", (0.33, 0.32, 0.30), 0.95))
    flat(m, data["road_quads"], 0.0, pbr("asphalt", rough_map=True), 3.0)
    flat(m, data["track_quads"], 0.004, pbr("asphalt", rough_map=True, tint=(0.62, 0.71, 0.62)), 3.0)
    flat(m, data["sidewalk_quads"], CURB_H, pbr("sidewalk", rough_map=True), 1.8)
    curb = pbr("curb")
    for x1, y1, x2, y2 in data["curbs"]:
        L = math.hypot(x2 - x1, y2 - y1)
        for pts in ([(x1, y1, 0), (x2, y2, 0), (x2, y2, CURB_H), (x1, y1, CURB_H)],
                    [(x2, y2, 0), (x1, y1, 0), (x1, y1, CURB_H), (x2, y2, CURB_H)]):   # both sides
            m.face(pts, [(0, 0), (L / 2.5, 0), (L / 2.5, CURB_H / 2.5), (0, CURB_H / 2.5)], curb)
    white, yellow = [], []
    for r in data["roads"]:
        if r["lanes"] < 2:
            continue
        ls = []
        for ln in (l for l in data["lanes"] if l["road"] == r["id"]):
            fwd = math.dist(ln["pts"][0], r["pts"][0]) < math.dist(ln["pts"][-1], r["pts"][0])
            ls.append((fwd, ln["pts"] if fwd else ln["pts"][::-1]))
        for (fa, pa), (fb, pb) in zip(ls, ls[1:]):
            if len(pa) != len(pb):
                continue
            mid = [((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) for p, q in zip(pa, pb)]
            if fa == fb:
                white += dashes(mid, 0.12)
            else:
                yellow += ribbon(offset(mid, 0.12), 0.12) + ribbon(offset(mid, -0.12), 0.12)
    for c in data["crossings"]:
        if c["markings"] not in ("zebra", "yes", "ladder"):
            continue
        ux, uy = c["u"]
        nx, ny = -uy, ux
        at = lambda a, b: (c["x"] + ux * a + nx * b, c["y"] + uy * a + ny * b)  # noqa: E731
        o = c["across"][0] + 0.5
        while o < c["across"][1] - 0.3:
            white.append([at(-1.5, o - 0.25), at(1.5, o - 0.25), at(1.5, o + 0.25), at(-1.5, o + 0.25)])
            o += 1.0
    flat(m, white, MARK_Z, solid("paint", (0.80, 0.79, 0.75), 0.7), 1)
    flat(m, yellow, MARK_Z, solid("paint_yellow", (0.70, 0.45, 0.04), 0.7), 1)
    return m.build(coll)


# ---------------------------------------------------------------- furniture, poles, wires
def walk(pts, first, every):
    """(point, unit direction) every `every` metres along a polyline, starting `first` in."""
    s0, nxt = 0.0, first
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        if L < 1e-6:
            continue
        u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        while nxt <= s0 + L:
            t = nxt - s0
            yield (a[0] + u[0] * t, a[1] + u[1] * t), u
            nxt += every
        s0 += L


def ground_z(p, data):
    return CURB_H if any(in_quad(p, q) for q in data["sidewalk_quads"]) else 0.0


ICO = None


def icosphere():
    """Unit icosphere, one subdivision: (verts, tris)."""
    global ICO
    if ICO:
        return ICO
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    v = [tuple(c / math.sqrt(sum(x * x for x in p)) for c in p) for p in v]
    mid, out = {}, []
    def m_(i, j):
        k = (min(i, j), max(i, j))
        if k not in mid:
            p = [(v[i][c] + v[j][c]) / 2 for c in range(3)]
            n = math.sqrt(sum(x * x for x in p))
            v.append(tuple(x / n for x in p))
            mid[k] = len(v) - 1
        return mid[k]
    for a, b, c in f:
        ab, bc, ca = m_(a, b), m_(b, c), m_(c, a)
        out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    ICO = (v, out)
    return ICO


# genus: (crown base as a share of height, crown width / height, leaf colour)
GENUS = {"Pyrus": (0.38, 0.75, (0.17, 0.30, 0.10)), "Tristaniopsis": (0.40, 0.55, (0.12, 0.25, 0.08)),
         "Magnolia": (0.25, 0.75, (0.06, 0.16, 0.05)), "Platanus": (0.40, 1.0, (0.20, 0.32, 0.12)),
         "Lophostemon": (0.35, 0.80, (0.10, 0.22, 0.08)), "Ficus": (0.30, 0.95, (0.10, 0.24, 0.08)),
         "Prunus": (0.35, 0.95, (0.26, 0.10, 0.12)), "Callistemon": (0.30, 0.90, (0.15, 0.25, 0.10)),
         "Arbutus": (0.30, 0.90, (0.13, 0.24, 0.09)), "Pinus": (0.20, 0.55, (0.08, 0.17, 0.08))}


def street_tree(m, p, z, species, h, crown_r, dbh_in, seed, bark):
    """Trunk sized from DBH up to the crown base, 3-5 primary limbs, and a crown of noisy leaf clumps in an
    ellipsoid of the species' proportions. Two leaf shades per genus."""
    genus = species.split()[0] if species else ""
    base, ratio, col = GENUS.get(genus, (0.35, 0.8, (0.13, 0.24, 0.09)))
    leaves = [solid(f"leaves_{genus or 'x'}_{k}", tuple(c * f for c in col), 0.85) for k, f in enumerate((1.0, 1.35))]
    rnd = lambda k: seeded(seed * 31 + k)   # noqa: E731
    r_trunk = max(0.04, dbh_in * 0.0127)
    zb = z + h * base
    m.cyl((*p, z), r_trunk * 1.15, r_trunk * 0.8, zb - z + 0.3, bark, 8)
    rw = min(crown_r, max(0.6, h * ratio / 2))           # crown half-width
    rh = (h * (1 - base)) / 2                             # crown half-height
    cz = zb + rh
    for k in range(3 + int(rnd(1) * 3)):                  # limbs from the crown base out into the crown
        a = 2 * math.pi * (k / 4 + 0.2 * rnd(10 + k))
        tip = (p[0] + math.cos(a) * rw * 0.6, p[1] + math.sin(a) * rw * 0.6, cz + rh * 0.2)
        m.tube((p[0], p[1], zb - 0.1), tip, r_trunk * 0.55, r_trunk * 0.25, bark)
    vs, tris = icosphere()
    n = int(6 + 3 * rw * rh)
    for k in range(min(n, 16)):
        u, v, w = rnd(100 + 3 * k) * 2 - 1, rnd(101 + 3 * k) * 2 - 1, rnd(102 + 3 * k) * 2 - 1
        if u * u + v * v + w * w > 1:
            u, v, w = u * 0.6, v * 0.6, w * 0.6
        c = (p[0] + u * rw * 0.7, p[1] + v * rw * 0.7, cz + w * rh * 0.7)
        rc = max(0.35, 0.45 * min(rw, rh * 1.3)) * (0.75 + 0.5 * rnd(200 + k))
        pts = []
        for j, q in enumerate(vs):
            jit = 0.8 + 0.4 * seeded(seed * 977 + k * 131 + j)
            pts.append((c[0] + q[0] * rc * jit, c[1] + q[1] * rc * jit, c[2] + q[2] * rc * jit * 0.85))
        mat = leaves[int(rnd(300 + k) * 2)]
        for a_, b_, c_ in tris:
            m.face([pts[a_], pts[b_], pts[c_]], [(0, 0), (1, 0), (0, 1)], mat)


def street_light(m, p, yaw, z, steel):
    c, s_ = math.cos(yaw), math.sin(yaw)
    m.cyl((*p, z), 0.12, 0.08, 8.5, steel)
    m.box((p[0] + c * 1.1, p[1] + s_ * 1.1, z + 8.27), (2.2, 0.07, 0.07), yaw, steel)
    m.box((p[0] + c * 2.3, p[1] + s_ * 2.3, z + 8.18), (0.7, 0.32, 0.15), yaw, steel)
    m.box((p[0] + c * 2.3, p[1] + s_ * 2.3, z + 8.16), (0.6, 0.26, 0.02), yaw, solid("lamp", (0.9, 0.88, 0.8), 0.3))


def build_furniture(data, coll, extras=None):
    m = Mesh("furniture")
    wood = solid("wood_pole", (0.23, 0.17, 0.12), 0.9)
    steel = solid("steel", (0.45, 0.48, 0.47), 0.45, 0.6)
    dark = solid("signal", (0.10, 0.12, 0.11), 0.6, 0.3)
    hydrant = solid("hydrant", (0.80, 0.78, 0.72), 0.5)
    cap = solid("hydrant_cap", (0.06, 0.20, 0.55), 0.45)
    red = solid("stop_red", (0.62, 0.03, 0.03), 0.5)
    pay = solid("paystation", (0.12, 0.16, 0.21), 0.5, 0.4)
    tactile = solid("tactile", (0.75, 0.55, 0.05), 0.6)
    lamp = solid("lamp", (0.9, 0.88, 0.8), 0.3)

    def wood_pole(p, yaw, light):
        z = ground_z(p, data)
        m.cyl((p[0], p[1], z), 0.15, 0.12, 10.5, wood)
        m.box((p[0], p[1], z + 9.55), (2.4, 0.1, 0.1), yaw, wood)
        for off in (-1.0, 0.0, 1.0):
            m.cyl((p[0] + math.cos(yaw) * off, p[1] + math.sin(yaw) * off, z + 9.65), 0.04, 0.03, 0.18, dark, 6)
        if light:
            c, s = math.cos(yaw), math.sin(yaw)
            m.box((p[0] + c * 0.9, p[1] + s * 0.9, z + 7.37), (1.8, 0.06, 0.06), yaw, steel)
            m.box((p[0] + c * 1.9, p[1] + s * 1.9, z + 7.28), (0.6, 0.3, 0.14), yaw, steel)
            m.box((p[0] + c * 1.9, p[1] + s * 1.9, z + 7.26), (0.5, 0.24, 0.02), yaw, lamp)

    def toward_road(p):
        best = None
        for r in data["roads"]:
            for a, b in zip(r["pts"], r["pts"][1:]):
                L = math.dist(a, b) or 1
                u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
                t = max(0, min(L, (p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1]))
                q = (a[0] + u[0] * t, a[1] + u[1] * t)
                d = math.dist(p, q)
                if best is None or d < best[0]:
                    best = (d, math.atan2(q[1] - p[1], q[0] - p[0]))
        return best[1]

    for f in data["furniture"]:
        p, k = (f["x"], f["y"]), f["kind"]
        z = ground_z(p, data)
        yaw = toward_road(p)
        if k == "fire_hydrant":
            m.cyl((*p, z), 0.17, 0.17, 0.06, hydrant, 12)
            m.cyl((*p, z), 0.13, 0.12, 0.6, hydrant, 12)
            m.box((*p, z + 0.375), (0.42, 0.09, 0.09), yaw, hydrant)
            m.cyl((*p, z + 0.6), 0.15, 0.06, 0.12, cap, 12)
        elif k == "traffic_signals":
            m.cyl((*p, z), 0.12, 0.1, 5.2, dark)
            m.box((p[0] + math.cos(yaw) * 0.2, p[1] + math.sin(yaw) * 0.2, z + 3.78), (0.3, 0.36, 1.05), yaw, dark)
        elif k in ("stop", "traffic_sign"):
            m.cyl((*p, z), 0.03, 0.03, 2.3, steel, 6)
            c, s = math.cos(yaw), math.sin(yaw)
            ring = [(0.38 * math.cos(math.pi / 8 + i * math.pi / 4), 0.38 * math.sin(math.pi / 8 + i * math.pi / 4)) for i in range(8)]
            pts = [(p[0] + c * 0.04 - s * u, p[1] + s * 0.04 + c * u, z + 2.3 + v) for u, v in ring]
            m.face(pts, [(0, 0)] * 8, red)
            m.face(pts[::-1], [(0, 0)] * 8, red)
        elif k == "vending_machine":
            m.box((*p, z), (0.34, 0.24, 1.45), yaw + math.pi / 2, pay)
        elif k in ("pole", "utility_pole"):
            wood_pole(p, yaw, False)
        elif k == "kerb":
            c = min(data["crossings"], key=lambda c: math.hypot(c["x"] - p[0], c["y"] - p[1]))
            m.box((*p, z), (1.2, 0.6, 0.008), math.atan2(c["u"][1], c["u"][0]), tactile)

    # Poles and street lights: where scripts/streetview_facades.py triangulated them from the photos (local
    # only), else the old guess (wooden poles every POLE_EVERY m along Moss's SW curb, lights on Howard).
    poles = []
    found = (extras or {}).get("poles") or []
    for f in found:
        p = (f["x"], f["y"])
        if f["kind"] == "utility":
            yaw = toward_road(p)
            wood_pole(p, yaw + math.pi / 2, f.get("light", False))
            poles.append((p, yaw + math.pi / 2))
        else:
            street_light(m, p, toward_road(p), ground_z(p, data), steel)
    if not found:
        moss = next(r for r in data["roads"] if r["id"] == MOSS_ROAD)
        line = offset(moss["pts"], moss["curbs"][0] - 0.35)
        for p, u in walk(line, POLE_FIRST, POLE_EVERY):
            yaw = math.atan2(u[0], -u[1])     # across the street, toward the road
            wood_pole(p, yaw, True)
            poles.append((p, yaw))
        for r in (r for r in data["roads"] if r["name"] == "Howard Street"):
            for side, sign in ((r["curbs"][1], 1), (r["curbs"][0], -1)):
                for p, u in walk(offset(r["pts"], side + sign * 0.5), 6.0, 36.0):
                    if -80 < p[0] < 140 and -150 < p[1] < 70 and not any(
                            math.hypot(c["x"] - p[0], c["y"] - p[1]) < 5 for c in data["crossings"]):
                        street_light(m, p, math.atan2(-sign * u[0], sign * u[1]), ground_z(p, data), steel)
    # Street trees where DataSF's inventory has them, shaped by species and trunk diameter.
    trees = Mesh("trees")
    bark = solid("bark", (0.20, 0.16, 0.12), 0.95)
    for f in data["furniture"]:
        if f["kind"] == "tree":
            t = f["tags"]
            street_tree(trees, (f["x"], f["y"]), ground_z((f["x"], f["y"]), data), t.get("species", ""),
                        t.get("height_m", 5.0), t.get("crown_r_m", 1.5), t.get("dbh_in") or 3, f["id"], bark)
        elif f["kind"] == "parking_meter":
            p = (f["x"], f["y"])
            z = ground_z(p, data)
            m.cyl((*p, z), 0.04, 0.04, 1.15, steel, 8)
            m.box((*p, z + 1.15), (0.2, 0.16, 0.36), toward_road(p), pay)
    # Overhead wires between neighbouring utility poles (sorted along Moss), three per crossarm.
    moss = next(r for r in data["roads"] if r["id"] == MOSS_ROAD)
    mu = (moss["pts"][-1][0] - moss["pts"][0][0], moss["pts"][-1][1] - moss["pts"][0][1])
    poles.sort(key=lambda q: q[0][0] * mu[0] + q[0][1] * mu[1])
    spans = [(a, b) for a, b in zip(poles, poles[1:]) if math.dist(a[0], b[0]) < WIRE_MAX_SPAN]
    t = trees.build(coll)   # flat shaded: the jittered clumps read as foliage, not smooth blobs
    return [m.build(coll), t]


# ---------------------------------------------------------------- scene, cameras, export
def sim_camera(name, x, y, z, heading, coll):
    """A camera like the sim's (960x720, vfov 39.6 deg), at metres x east, y north, z up, compass heading."""
    cam = bpy.data.cameras.new(name)
    cam.sensor_fit = next(i.identifier for i in cam.bl_rna.properties["sensor_fit"].enum_items if i.identifier == "VERTICAL")
    cam.angle_y = 2 * math.atan(360 / 1000)
    cam.clip_start, cam.clip_end = 0.05, 500
    ob = bpy.data.objects.new(name, cam)
    ob.location = (x, y, z)
    ob.rotation_euler = (math.pi / 2, 0, -math.radians(heading))
    coll.objects.link(ob)
    return ob


def build():
    data = json.loads(MOSS.read_text())
    man_path = FACADES / "facades.json"
    extras = json.loads(man_path.read_text()) if man_path.exists() else {}
    manifest = extras.get("walls", [])
    scene = bpy.data.scenes.get(SCENE) or bpy.data.scenes.new(SCENE)
    for ob in list(scene.collection.all_objects):
        me = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if isinstance(me, bpy.types.Mesh) and me.users == 0:
            bpy.data.meshes.remove(me)
    for c in list(scene.collection.children):
        bpy.data.collections.remove(c)
    for m in list(bpy.data.materials):   # last build's materials, now unused
        if m.users == 0:
            bpy.data.materials.remove(m)
    coll = bpy.data.collections.new("MossRealModel")
    scene.collection.children.link(coll)
    rigs = bpy.data.collections.new("MossRealRig")      # cameras and light, not exported
    scene.collection.children.link(rigs)
    _mats.clear()
    objs = [build_buildings(data, manifest, coll), build_streets(data, coll), *build_furniture(data, coll, extras)]
    sun = bpy.data.objects.new("MossRealSun", bpy.data.lights.new("MossRealSun", "SUN"))
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(-130))
    rigs.objects.link(sun)
    world = bpy.data.worlds.get("MossRealWorld") or bpy.data.worlds.new("MossRealWorld")
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    env = next((n for n in world.node_tree.nodes if n.type == "TEX_ENVIRONMENT"), None) or world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(TEX / "urban_street_01_1k.hdr"), check_existing=True)
    world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    scene.world = world
    spawn = data["spawn"]
    h = math.radians(spawn["heading"])
    first = sim_camera("real_drone_spawn", spawn["x"] - math.sin(h), spawn["y"] - math.cos(h), 2.0, spawn["heading"], rigs)
    (x0, y0), (x1, y1) = data["route"][1], data["route"][2]
    hz = math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360
    sim_camera("real_chest_corner", x0 + 0.3 * math.sin(math.radians(hz)), y0 + 0.3 * math.cos(math.radians(hz)), 1.2, hz, rigs)
    (x2, y2), (x3, y3) = data["route"][2], data["route"][3]
    hh = math.degrees(math.atan2(x3 - x2, y3 - y2)) % 360
    sim_camera("real_drone_howard", x2 - math.sin(math.radians(hh)), y2 - math.cos(math.radians(hh)), 2.0, hh, rigs)
    scene.camera = first
    scene.render.resolution_x, scene.render.resolution_y = 960, 720
    textured = sum(1 for w in manifest if w.get("file"))
    scene["kite_built"] = True
    print(f"[build_real_moss] {len(data['buildings'])} buildings, {textured}/{len(manifest)} street walls with photos, "
          f"{sum(len(o.data.polygons) for o in objs)} faces")
    return scene, objs


def render_views(cams_path, out_dir):
    """Render the MossReal model from Street View tile cameras, for scripts/facade_verify.py. cams_path: JSON
    [{name, x, y, z, heading, pitch, fov, tx, ty}] (yaw already in heading; tilt as in streetview_facades.project:
    world directions d are seen as Rt d). Writes <out_dir>/<name>.png at 640x640."""
    from mathutils import Matrix
    scene = bpy.data.scenes[SCENE]
    cams = json.loads(Path(cams_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cam = bpy.data.cameras.get("verify_cam") or bpy.data.cameras.new("verify_cam")
    cam.sensor_fit = next(i.identifier for i in cam.bl_rna.properties["sensor_fit"].enum_items if i.identifier == "HORIZONTAL")
    cam.clip_start, cam.clip_end = 0.05, 500
    ob = bpy.data.objects.get("verify_cam") or bpy.data.objects.new("verify_cam", cam)
    if ob.name not in scene.collection.objects:
        scene.collection.objects.link(ob)
    scene.camera = ob
    r = scene.render
    r.resolution_x = r.resolution_y = 640
    r.resolution_percentage = 100
    engines = [i.identifier for i in r.bl_rna.properties["engine"].enum_items]
    r.engine = next(e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH") if e in engines)
    r.image_settings.file_format = "PNG"
    for c in cams:
        cam.angle_x = math.radians(c["fov"])
        h, p = math.radians(c["heading"]), math.radians(c["pitch"])
        # camera axes in world: right, up (= -down), back (= -fwd), as streetview_facades.project
        level = (math.sin(h), math.cos(h), 0.0)
        fwd = [math.cos(p) * level[k] + math.sin(p) * (0, 0, 1)[k] for k in range(3)]
        right = (math.cos(h), -math.sin(h), 0.0)
        down = [math.sin(p) * level[k] - math.cos(p) * (0, 0, 1)[k] for k in range(3)]
        R = Matrix([[right[k], -down[k], -fwd[k]] for k in range(3)])
        a, b = math.radians(c.get("tx", 0)), math.radians(c.get("ty", 0))
        rx = Matrix([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        ry = Matrix([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
        R = (rx @ ry).transposed() @ R
        ob.matrix_world = Matrix.Translation((c["x"], c["y"], c["z"])) @ R.to_4x4()
        r.filepath = str(out / f"{c['name']}.png")
        bpy.ops.render.render(write_still=True, scene=scene.name)
    print(f"[build_real_moss] rendered {len(cams)} views to {out}")


def export(scene, objs):
    """Only the model collection of the MossReal scene goes in the GLB."""
    win = bpy.context.window          # None in blender -b
    old = win.scene if win else None
    if win:
        win.scene = scene
    vl = scene.view_layers[0]
    vl.active_layer_collection = vl.layer_collection.children["MossRealModel"]
    OUT_GLB.parent.mkdir(parents=True, exist_ok=True)
    try:
        with bpy.context.temp_override(**({"window": win} if win else {}), scene=scene, view_layer=vl):
            bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), use_active_scene=True, use_active_collection=True,
                                      export_yup=True, export_apply=True, export_cameras=False, export_lights=False)
    finally:
        if win:
            win.scene = old
    print(f"[build_real_moss] wrote {OUT_GLB} ({OUT_GLB.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--render-glb" in args:       # render an exported model (scripts/facade_verify.py)
        i = args.index("--render-glb")
        scene = bpy.context.scene
        scene.name = SCENE
        for ob in list(scene.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        bpy.ops.import_scene.gltf(filepath=args[i + 1])
        world = bpy.data.worlds.new("MossRealWorld")
        world.use_nodes = True
        bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
        env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(TEX / "urban_street_01_1k.hdr"), check_existing=True)
        world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
        scene.world = world
        render_views(args[i + 2], args[i + 3])
    elif "--render-only" in args:      # render the scene already in the open .blend
        render_views(args[args.index("--render-only") + 1], args[args.index("--render-only") + 2])
    else:
        s, o = build()
        if "--export" in args:
            export(s, o)
        if "--render" in args:
            render_views(args[args.index("--render") + 1], args[args.index("--render") + 2])
