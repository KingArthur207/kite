// Building facades for the Moss world: OSM footprints become textured walls with a material per
// building (OSM tags, else the building type and a seeded random from its id), window grids per
// storey, ground floors (garages and doors on residential fronts, shop glass where OSM has a shop),
// SF bay windows on some residential street fronts, and a cornice at the roof line.
// style "victorian" (Moss's row houses, from Street View): pastel siding or stucco, white trim, hooded window
// frames, an angled bay per lot, a deep bracketed cornice, panel garage doors and doors up a stoop.
// style "midrise" (Howard's north side): pale walls, big windows, shop glass at street level.
// Windows, doors and garage doors are instanced: one draw call per kind.
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { Geo, facadeMaterial, pbr, seeded, type FacadeKind } from "./osmLook";
import { buildingSheet, look2 } from "./look2";
import { leafTexture } from "./streetProps";
import { LayoutFronts, facadeLayouts, facadeSides, gaps, layoutHeight, onFront } from "./facadeLayouts";

type P = [number, number];
type Quad = P[];
/** pts: the roof outline. ground: the ground-floor outline where bays overhang the sidewalk from bay_z up. */
export interface Building {
  id: number; h: number; pts: P[]; ground?: P[]; bay_z?: number; tags?: Record<string, string>; storefront?: boolean;
  style?: "victorian" | "midrise";
}
/** A garage door on a street front: centre on the wall, along-wall unit u, outward normal n, width. */
export interface Garage { x: number; n: number; u: P; out: P; w: number }

const STOREY = 3.2;
const PARTY_GAP = 0.6;          // m: a wall this close and parallel to another building's wall is shared
const STREET_LOOK = 20;         // m: a wall whose normal hits a roadway within this is a street front
const LOT = 7.6;                // m: SF 25 ft lot, one garage + door + bay per lot
const CORNICE_H = 0.4, CORNICE_OUT = 0.22;
const VIC_CORNICE_H = 1.0;      // m: Victorian cornice (0.55 m, 0.4 out) over a frieze with brackets
const BAY_D = 0.7;              // m: SF angled bay depth (real ones 0.6-0.9 m)
const STOOP_RISE = 0.5, STOOP_D = 1.0, STOOP_MIN = 0.45;   // m: entry stoop height, full depth, shallowest worth drawing
const WALK_CLEAR = 1.4;         // m: stoops end this far from the demo walk line (walker radius, plus the 1 m the metrics
                                // autopilot sidesteps toward the buildings on Moss, src/metrics.ts SIDE)
const STEP_GREY = new THREE.Color(0xc4c1ba);   // painted concrete stoops
const BIG_BAY = 3.2;            // m: a layout bay this wide gets the big bay windows
const VIC_WINDOW_TOP = 1.85 / 2 + 0.13 + 0.3;  // m from a "vic" window's centre to the top of its hood

// Street View on Moss: mint, pale blue, cream, pink, light grey, a pale yellow and a sage.
const VICTORIAN = [0xdcefe2, 0xd6e6f2, 0xf7efd8, 0xf6dfe0, 0xe6e7e7, 0xf5ecc8, 0xd4e3d3, 0xf8f6f0, 0xcfe6e7];
const MIDRISE = [0xf3f0e9, 0xe8e5de, 0xdedcd5, 0xefe8da, 0xe4e6e6];
const VIC_DOORS = [0x7a2e2a, 0x24324a, 0x1e1e1e, 0x2f5a46, 0xe8e4da, 0x5b3b2a];
const VIC_GARAGES = [0xf3f1ea, 0xdcdedb, 0x9fc7b5, 0x8fb2c4, 0xe9dfc4, 0x5f7f73];

const PASTEL = [0xf1ead8, 0xe6dfd0, 0xd5dfe3, 0xc6d0bd, 0xe8cfbd, 0xd6d6d0, 0xf4f1ea, 0xc9b9a5, 0xb3c0ca, 0xe4d4a4, 0xd9c2c6];
const TINTS: Record<FacadeKind, number[]> = {
  stucco: PASTEL, plaster: PASTEL, siding: PASTEL,
  brick: [0xffffff, 0xeadfd8, 0xd9cbc2, 0xf2e8e2],
  concrete: [0xffffff, 0xe8e4dc, 0xd2d0cc, 0xc8c1b4],
  metal: [0xd4d8dc, 0x9ca7ae, 0x8b9b8d, 0xc8c2b4, 0x737e86],
};
const TAG_MATERIAL: Record<string, FacadeKind> = {
  brick: "brick", concrete: "concrete", stone: "concrete", glass: "concrete", metal: "metal", steel: "metal",
  wood: "siding", timber_framing: "siding", plaster: "stucco", stucco: "stucco", cement_block: "concrete",
};
const RESIDENTIAL = new Set(["residential", "house", "apartments", "terrace", "detached", "semidetached_house"]);
const COMMERCIAL = new Set(["commercial", "retail", "industrial", "warehouse", "office", "garage", "garages"]);
const MIX: Record<string, [FacadeKind, number][]> = {
  residential: [["stucco", 45], ["siding", 30], ["plaster", 15], ["brick", 10]],
  commercial: [["concrete", 35], ["brick", 30], ["stucco", 20], ["metal", 15]],
  any: [["stucco", 34], ["plaster", 14], ["siding", 16], ["brick", 16], ["concrete", 12], ["metal", 8]],
};

function pickKind(b: Building, rnd: () => number): FacadeKind {
  const t = b.tags ?? {};
  const tagged = TAG_MATERIAL[t["building:material"] ?? ""];
  if (tagged) return tagged;
  if (b.style === "victorian") return rnd() < 0.6 ? "siding" : "stucco";
  if (b.style === "midrise") return "stucco";
  const type = t.building ?? "yes";
  let mix = RESIDENTIAL.has(type) ? MIX.residential : COMMERCIAL.has(type) ? MIX.commercial : MIX.any;
  if (b.h > 14) mix = mix.filter(([k]) => k !== "siding" && k !== "metal");   // no wood or tin towers
  let r = rnd() * mix.reduce((s, [, w]) => s + w, 0);
  for (const [k, w] of mix) if ((r -= w) < 0) return k;
  return mix[0][0];
}

// look 2: mean linear albedo each wall material ends up with before its tint (texture mean x material colour;
// siding is look2's own canvas boards at 0.92). Measured off public/textures/*_diff.jpg.
const ALBEDO: Record<FacadeKind, [number, number, number]> = {
  stucco: [0.68, 0.65, 0.63], plaster: [0.4, 0.38, 0.37], brick: [0.14, 0.077, 0.051],
  concrete: [0.27, 0.24, 0.16], metal: [0.1, 0.098, 0.083], siding: [0.72, 0.72, 0.72],
};
const MAX_TINT = 10, PAINTED_LUMA = 0.35;
const luma = (c: THREE.Color) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
/** Vertex tint that makes a wall of this kind come out at colour c (linear) on average. */
function albedoTint(c: THREE.Color, kind: FacadeKind) {
  const [r, g, b] = ALBEDO[kind];
  return new THREE.Color(Math.min(MAX_TINT, c.r / r), Math.min(MAX_TINT, c.g / g), Math.min(MAX_TINT, c.b / b));
}

function tint(b: Building, kind: FacadeKind, rnd: () => number) {
  const tag = b.tags?.["building:colour"];
  const pal = kind === "brick" ? TINTS.brick : b.style === "victorian" ? VICTORIAN : b.style === "midrise" ? MIDRISE : TINTS[kind];
  const c = new THREE.Color(pal[Math.floor(rnd() * pal.length)]);
  if (tag) { try { c.setStyle(tag); } catch { /* unknown colour name: keep the palette pick */ } }
  return c.convertSRGBToLinear();
}

// --- window/door kinds: an instanced frame (bars + sill + mullions) and an instanced pane.
/** hood: a Victorian window head (a frieze board under a projecting cap). */
interface Kind { w: number; h: number; f: number; d: number; mull: "v" | "h" | "vh" | "none"; pane: THREE.Material; uvMetres?: boolean; hood?: boolean }
interface Inst { m: THREE.Matrix4; frame: THREE.Color; pane: THREE.Color }

function frameGeometry(k: Kind) {
  const { w, h, f, d } = k;
  const bars: THREE.BufferGeometry[] = [];
  const bar = (bw: number, bh: number, bd: number, x: number, y: number) =>
    bars.push(new THREE.BoxGeometry(bw, bh, bd).translate(x, y, bd / 2));
  bar(w + 2 * f, f, d, 0, h / 2 + f / 2);                 // head
  if (k.hood) {
    bar(w + 2 * f, 0.22, d * 0.7, 0, h / 2 + f + 0.11);
    bar(w + 2 * f + 0.22, 0.08, d + 0.1, 0, h / 2 + f + 0.26);
  }
  bar(w + 2 * f + 0.12, f * 1.3, d + 0.06, 0, -h / 2 - f * 0.65);   // sill
  bar(f, h, d, -w / 2 - f / 2, 0);
  bar(f, h, d, w / 2 + f / 2, 0);
  const m = f * 0.55;
  if (k.mull.includes("v")) bar(m, h, d * 0.6, 0, 0);
  if (k.mull.includes("h")) bar(w, m, d * 0.6, 0, k.mull === "h" ? 0 : h * 0.18);
  return mergeGeometries(bars)!;
}

function paneGeometry(k: Kind) {
  const g = new THREE.PlaneGeometry(k.w, k.h).translate(0, 0, 0.015);
  if (k.uvMetres) {   // garage shutters: texture in metres
    const uv = g.getAttribute("uv");
    for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * k.w, uv.getY(i) * k.h);
  }
  return g;
}

/** A copy of b whose roof outline sticks out at most `cap` m past its ground floor (0: no overhang at all). */
export function capBays(b: Building, cap: number): Building {
  if (!b.ground || b.ground.length < 3) return b;
  if (cap <= 0) return { ...b, pts: b.ground, ground: undefined, bay_z: undefined };
  const g = b.ground;
  const pts = b.pts.map((p): P => {
    if (inPoly(p, g)) return p;
    let best = { d: Infinity, q: p };
    for (let i = 0; i < g.length; i++) {
      const a = g[i], c = g[(i + 1) % g.length];
      const dx = c[0] - a[0], dy = c[1] - a[1];
      const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
      const q: P = [a[0] + t * dx, a[1] + t * dy], d = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (d < best.d) best = { d, q };
    }
    if (best.d <= cap) return p;
    const k = cap / best.d;
    return [best.q[0] + (p[0] - best.q[0]) * k, best.q[1] + (p[1] - best.q[1]) * k];
  });
  return { ...b, pts };
}

/** A panelled sectional garage door, white (tinted per instance): 4 x 4 raised panels. */
function panelDoorTexture() {
  const S = 256, c = document.createElement("canvas");
  c.width = c.height = S;
  const g = c.getContext("2d")!;
  g.fillStyle = "#f4f4f2"; g.fillRect(0, 0, S, S);
  const pw = S / 4, ph = S / 4;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    const x = i * pw + 8, y = j * ph + 9, w = pw - 16, h = ph - 18;
    g.fillStyle = "#c9c9c6"; g.fillRect(x, y, w, h);                  // shadowed bevel
    g.fillStyle = "#ffffff"; g.fillRect(x + 4, y + 4, w - 6, h - 6);  // lit panel face
  }
  g.fillStyle = "#b4b4b0";
  for (let j = 1; j < 4; j++) g.fillRect(0, j * ph - 1, S, 2);         // section joints
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** skip: walls drawn elsewhere (photo facades), as `${building id}:${edge index}`; they get no wall, windows or cornice.
 *  walk: the demo walk line; Victorian stoops stop short of it. Returns stoops as wall segments for collision. */
export function buildFacades(scene: THREE.Object3D, input: Building[], roadQuads: Quad[], skip = new Set<string>(), walk: P[] = []) {
  const glass = new THREE.MeshStandardMaterial({ name: "glass", color: 0xffffff, roughness: 0.06, metalness: 0.65, envMapIntensity: 1.3 });
  const trimMat = new THREE.MeshStandardMaterial({ name: "frame", color: 0xffffff, roughness: 0.65 });
  const doorMat = new THREE.MeshStandardMaterial({ name: "door", color: 0xffffff, roughness: 0.5 });
  const shutter = pbr("shutter", 2.0, { roughness: 0.55 });
  const panels = new THREE.MeshStandardMaterial({ map: panelDoorTexture(), roughness: 0.6 });
  const KINDS = {
    vic: { w: 0.9, h: 1.85, f: 0.13, d: 0.14, mull: "h", pane: glass, hood: true },
    large: { w: 1.9, h: 2.2, f: 0.07, d: 0.1, mull: "v", pane: glass },
    vgarage: { w: 2.5, h: 2.3, f: 0.14, d: 0.12, mull: "none", pane: panels },
    res: { w: 1.05, h: 1.7, f: 0.09, d: 0.12, mull: "h", pane: glass },
    narrow: { w: 0.62, h: 1.6, f: 0.08, d: 0.1, mull: "h", pane: glass },
    bayc: { w: 1.35, h: 1.85, f: 0.1, d: 0.12, mull: "h", pane: glass },    // look 2 layout bays: the big front sash
    bays: { w: 0.72, h: 1.85, f: 0.08, d: 0.1, mull: "h", pane: glass },    // and the side ones
    ind: { w: 2.2, h: 1.9, f: 0.07, d: 0.1, mull: "vh", pane: glass },
    shop: { w: 2.7, h: 2.45, f: 0.08, d: 0.1, mull: "none", pane: glass },
    garage: { w: 2.6, h: 2.3, f: 0.1, d: 0.1, mull: "none", pane: shutter, uvMetres: true },
    rollup: { w: 3.6, h: 3.0, f: 0.1, d: 0.1, mull: "none", pane: shutter, uvMetres: true },
    door: { w: 0.95, h: 2.2, f: 0.1, d: 0.14, mull: "none", pane: doorMat },
  } satisfies Record<string, Kind>;
  type KindName = keyof typeof KINDS;
  const inst = Object.fromEntries(Object.keys(KINDS).map((k) => [k, [] as Inst[]])) as Record<KindName, Inst[]>;

  const walls = new Map<FacadeKind, Geo>();
  const trim = new Geo();
  const roofs: THREE.BufferGeometry[] = [];
  const garages: Garage[] = [];
  const stoops: [number, number, number, number][] = [];   // wall segments around Victorian stoops (and hedges)
  const gates: THREE.Matrix4[] = [];
  const hedges: Hedge[] = [];
  const escapes: THREE.BufferGeometry[] = [];
  const layoutFronts = look2 ? new LayoutFronts({ glass, shutter, panels }) : null;
  const open = (pts: P[]) => {   // ring without the closing point
    const r = pts.slice();
    if (r.length > 1 && r[0][0] === r[r.length - 1][0] && r[0][1] === r[r.length - 1][1]) r.pop();
    return look2 ? simplify(r) : r;
  };
  // look 2: a building with a Street View layout stands on its ground footprint (no capped bay outline in front of
  // the layout's wall), as tall as the photo shows.
  const buildings = input.map((b) => { const l = layoutFronts && facadeLayouts.get(b.id); return l ? { ...capBays(b, 0), h: layoutHeight(l, b.h) } : b; });
  const rings = buildings.map((b) => open(b.pts));
  const edges = rings.flatMap((r, bi) => r.map((a, i) => ({ bi, a, b: r[(i + 1) % r.length] })));

  const isParty = (bi: number, a: P, b: P) => {
    const m: P = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    return edges.some((e) => {
      if (e.bi === bi) return false;
      const ex = e.b[0] - e.a[0], en = e.b[1] - e.a[1], el = Math.hypot(ex, en);
      if (el < 0.3 || Math.abs((ex * (b[0] - a[0]) + en * (b[1] - a[1])) / (el * L)) < 0.9) return false;
      return segDist(m, e.a, e.b) < PARTY_GAP;
    });
  };
  const isStreet = (bi: number, m: P, nrm: P) => {
    for (let k = 0.5; k < STREET_LOOK; k += 0.75) {
      const p: P = [m[0] + nrm[0] * k, m[1] + nrm[1] * k];
      if (roadQuads.some((q) => inQuad(p, q))) return true;
      if (k > 1 && rings.some((r, j) => j !== bi && inPoly(p, r))) return false;
    }
    return false;
  };

  buildings.forEach((b, bi) => {
    const lay = layoutFronts ? facadeLayouts.get(b.id) : undefined;   // look 2: this front from Street View
    // look 2: the front and any exposed side walls read off Street View, each with the footprint edges it covers
    const faces = layoutFronts ? [...(lay ? [lay] : []), ...(facadeSides.get(b.id) ?? [])].map((l) => ({ lay: l, segs: [] as [P, P][] })) : [];
    const ring = rings[bi];
    if (ring.length < 3) return;
    const rnd = seeded(b.id);
    const sheet = look2 ? buildingSheet.get(b.id) : undefined;   // per-building look from Street View, if loaded
    const rnd2 = seeded(b.id + 1);                                  // look-2 extras, so rnd's sequence stays as it was
    const sheetColor = sheet?.main ? new THREE.Color(sheet.main) : null;
    // Painted brick (a light colour on brick) reads as a light masonry wall: the brick texture would drag it to brown.
    const kind: FacadeKind = sheet?.cladding === "brick" && sheetColor && luma(sheetColor) > PAINTED_LUMA ? "stucco" : sheet?.cladding ?? pickKind(b, rnd);
    // The sheet gives the wall colour as photographed; the material multiplies it by its texture, so divide that out.
    const color = sheetColor ? albedoTint(sheetColor, kind) : tint(b, kind, rnd);
    const vic = b.style === "victorian", midrise = b.style === "midrise";
    const residential = vic || (!midrise && (kind === "stucco" || kind === "plaster" || kind === "siding" || RESIDENTIAL.has(b.tags?.building ?? "")) && !b.storefront);
    const lv = parseFloat(b.tags?.["building:levels"] ?? "");
    const levels = sheet?.floors ?? Math.max(1, Number.isFinite(lv) ? Math.round(lv) : Math.round(b.h / STOREY));
    const sh = b.h / levels;
    const trimColor = new THREE.Color(vic ? 0xf7f5ef : midrise ? 0xcfcfca
      : residential ? (rnd() < 0.75 ? 0xf3f0e8 : 0x4a4f55) : rnd() < 0.5 ? 0x3b3833 : 0x8d9296);
    if (sheet?.trim) trimColor.set(sheet.trim);
    let bays = sheet?.bays ?? Infinity, garageCount = sheet?.garages ?? Infinity;
    const corniceH = vic ? VIC_CORNICE_H : CORNICE_H;
    const paneTint = () => new THREE.Color().setHSL(0.58, 0.12, rnd() < 0.15 ? 0.45 : 0.1 + rnd() * 0.08);
    const g = walls.get(kind) ?? walls.set(kind, new Geo()).get(kind)!;
    let s0 = rnd() * 10;

    const roof = new THREE.ShapeGeometry(new THREE.Shape(ring.map(([x, y]) => new THREE.Vector2(x, y))));
    roofs.push(roof.rotateX(-Math.PI / 2).translate(0, b.h, 0));

    // Where OSM's roof outline has bays over the sidewalk, the ground floor follows `ground` up to bay_z,
    // and the roof outline starts there, with a soffit under the overhang.
    const ground = b.ground ? open(b.ground) : null;
    const bayZ = Math.min(b.bay_z ?? STOREY, b.h);
    const parts = ground && ground.length >= 3
      ? [{ ring: ground, y0: 0, y1: bayZ, low: true, high: false }, { ring, y0: bayZ, y1: b.h, low: false, high: true }]
      : [{ ring, y0: 0, y1: b.h, low: true, high: true }];
    if (parts.length > 1) {
      const down = new THREE.Vector3(0, -1, 0);
      for (const [i, j, k] of THREE.ShapeUtils.triangulateShape(ring.map(([x, y]) => new THREE.Vector2(x, y)), [])) {
        const t = [ring[i], ring[j], ring[k], ring[k]];
        g.quad(t.map((q) => v3(q, bayZ)), down, t.map((q) => [q[0], q[1]] as [number, number]), color);
      }
    }

    for (const { ring, y0, y1, low, high } of parts) {
      const area = ring.reduce((s, p, i) => s + p[0] * ring[(i + 1) % ring.length][1] - ring[(i + 1) % ring.length][0] * p[1], 0);
      ring.forEach((a, i) => {
        const c = ring[(i + 1) % ring.length];
        const L = Math.hypot(c[0] - a[0], c[1] - a[1]);
        if (L < 0.05) return;
        const u: P = [(c[0] - a[0]) / L, (c[1] - a[1]) / L];
        const out: P = area > 0 ? [u[1], -u[0]] : [-u[1], u[0]];
        if (high && skip.has(`${b.id}:${i}`)) { s0 += L; return; }
        const face = faces.find((f) => onFront(f.lay, a, c, out, f.lay !== lay));
        if (face) {   // drawn from the layout below, left to right seen from the street
          face.segs.push(u[1] * out[0] - u[0] * out[1] > 0 ? [a, c] : [c, a]);
          s0 += L;
          return;
        }
        const n3 = new THREE.Vector3(out[0], 0, -out[1]);
        g.quad([v3(a, y0), v3(c, y0), v3(c, y1), v3(a, y1)], n3, [[s0, y0], [s0 + L, y0], [s0 + L, y1], [s0, y1]], color);
        s0 += L;
        if (isParty(bi, a, c)) return;   // shared wall: blank
        const at = (t: number, o = 0): P => [a[0] + u[0] * t + out[0] * o, a[1] + u[1] * t + out[1] * o];
        const street = L >= 1.4 && isStreet(bi, at(L / 2), out);
        // look 2: a cornice only wraps a corner that's open (not against a neighbour's wall), so it never runs into one
        const ends: [boolean, boolean] = look2
          ? [!isParty(bi, ring[(i + ring.length - 1) % ring.length], a), !isParty(bi, c, ring[(i + 2) % ring.length])] : [true, true];
        if (high && vic && street) {   // cornice 0.55 m deep 0.4 m, frieze board, a bracket every 0.9 m
          cornice(trim, a, c, out, b.h - 0.55, b.h, 0.4, trimColor, ends);
          cornice(trim, a, c, out, b.h - VIC_CORNICE_H, b.h - 0.55, 0.05, trimColor, ends);
          for (let t = 0.3; t < L - 0.2; t += 0.9) slab(trim, at, u, out, t - 0.06, t + 0.06, 0, 0.36, b.h - 0.85, b.h - 0.55, trimColor);
        } else if (high) cornice(trim, a, c, out, b.h - CORNICE_H, b.h, CORNICE_OUT, trimColor, ends);
        if (L < 1.4) return;
        const place = (k: KindName, t: number, y: number, frame = trimColor, pane = paneTint(), o = 0, oo = out) =>
          inst[k].push({ m: basis(at(t, o), y, oo), frame, pane });
        const busy: [number, number][] = [];   // along-wall spans taken by bays (upper floors)
        const wantEscape = look2 && low && street && levels >= 3 && L > 6 && (sheet?.fire_escape ?? (midrise && rnd2() < FIRE_ESCAPE_SHARE));

        // Ground floor on street fronts.
        if (!low) { /* upper part only */ } else if (street && vic) {
          const lots = Math.max(1, Math.round(L / LOT)), lw = L / lots;
          if (!look2) cornice(trim, a, c, out, sh - 0.12, sh + 0.04, 0.07, trimColor);   // belt course (look 2: after the bays, around them)
          for (let j = 0; j < lots; j++) {
            const t0 = j * lw;
            if (lw >= 4.6 && garageCount-- > 0) {
              const gt = t0 + lw * 0.32;
              const gc = new THREE.Color(VIC_GARAGES[Math.floor(rnd() * VIC_GARAGES.length)]);
              place("vgarage", gt, 0.15 + KINDS.vgarage.h / 2, trimColor, sheet?.garage_colour ? gc.set(sheet.garage_colour) : gc);
              garages.push({ x: at(gt)[0], n: at(gt)[1], u, out, w: KINDS.vgarage.w });
            } else if (lw >= 2.2) place("vic", t0 + lw * 0.3, look2 ? Math.min(1.6, sh - 0.17 - VIC_WINDOW_TOP) : 1.6, trimColor);   // look 2: hood under the belt
            if (lw >= 2.2) {   // front door up a stoop, where the stoop stays clear of the demo walk
              const dt = t0 + lw * (lw >= 4.6 ? 0.8 : 0.72), hw = KINDS.door.w / 2 + 0.25;
              const room = walk.length > 1 ? Math.min(...walk.slice(1).map((q, k) => segDist(at(dt), walk[k], q))) - WALK_CLEAR : STOOP_D;
              const depth = Math.min(STOOP_D, room), rise = depth >= STOOP_MIN ? STOOP_RISE : 0;
              if (rise) {
                for (let k = 0; k < 3; k++) {
                  slab(trim, at, u, out, dt - hw, dt + hw, 0, depth * (3 - k) / 3, 0.15, 0.15 + rise * (k + 1) / 3, STEP_GREY);
                }
                const s = [at(dt - hw), at(dt - hw, depth), at(dt + hw, depth), at(dt + hw)];
                for (let k = 0; k < 3; k++) stoops.push([...s[k], ...s[k + 1]] as [number, number, number, number]);
              }
              place("door", dt, 0.15 + rise + KINDS.door.h / 2, trimColor, new THREE.Color(VIC_DOORS[Math.floor(rnd() * VIC_DOORS.length)]));
              if (look2 && (sheet?.gate ?? rnd2() < GATE_SHARE)) {   // iron security gate in front of the door
                gates.push(basis(at(dt, KINDS.door.d + 0.03), 0.15 + rise + KINDS.door.h / 2 + 0.05, out));
              }
              const hedgeD = Math.min(HEDGE_D, room);
              if (look2 && hedgeD >= HEDGE_MIN_D && (sheet?.hedge ?? rnd2() < HEDGE_SHARE)) {   // hedge beside the stoop
                const h0 = dt + hw + 0.05, h1 = Math.min(t0 + lw - 0.1, h0 + 1.6);
                if (h1 - h0 > 0.5) {
                  hedges.push({ at, t0: h0, t1: h1, d: hedgeD, h: 0.9 + rnd2() * 0.5 });
                  const s = [at(h0), at(h0, hedgeD), at(h1, hedgeD), at(h1)];
                  for (let k = 0; k < 3; k++) stoops.push([...s[k], ...s[k + 1]] as [number, number, number, number]);
                }
              }
            }
            if (high && levels >= 2 && lw >= 4.2 && bays-- > 0) {   // an angled bay over the garage on every lot
              const cx = t0 + lw / 2;
              bay(g, trim, at, cx, sh - 0.15, b.h - corniceH - 0.05, color, trimColor, (k, p, y, oo) =>
                inst[k === "res" ? "vic" : k].push({ m: basis(p, y, oo), frame: trimColor, pane: paneTint() }), sh, levels);
              busy.push([cx - 1.7, cx + 1.7]);
            }
          }
          if (look2) for (const [t0, t1] of gaps(0, L, busy.map(([lo, hi]): [number, number] => [lo + 0.2, hi - 0.2]))) {
            cornice(trim, at(t0), at(t1), out, sh - 0.12, sh + 0.04, 0.07, trimColor, [t0 < 0.01 && ends[0], t1 > L - 0.01 && ends[1]]);
          }
        } else if (street && (b.storefront || midrise)) {
          const n = Math.max(1, Math.floor((L - 0.6) / 3.1));
          for (let j = 0; j < n; j++) place("shop", L / 2 + (j - (n - 1) / 2) * 3.1, 0.35 + KINDS.shop.h / 2, new THREE.Color(0x2c2b29));
          cornice(trim, a, c, out, 2.95, 3.35, 0.08, new THREE.Color(0x26282b));   // sign band
        } else if (street && residential) {
          const lots = Math.max(1, Math.round(L / LOT)), lw = L / lots;
          cornice(trim, a, c, out, sh - 0.12, sh + 0.04, 0.07, trimColor);         // belt course
          for (let j = 0; j < lots; j++) {
            const t0 = j * lw;
            if (lw >= 4.6) {
              const gt = t0 + lw * 0.32;
              place("garage", gt, 0.05 + KINDS.garage.h / 2, trimColor, new THREE.Color(rnd() < 0.5 ? 0xe8e6e0 : 0xa8adb0));
              garages.push({ x: at(gt)[0], n: at(gt)[1], u, out, w: KINDS.garage.w });
            } else if (lw >= 2.2) place("res", t0 + lw * 0.3, 1.5, trimColor);
            if (lw >= 2.2) place("door", t0 + lw * 0.78, 0.15 + KINDS.door.h / 2, trimColor,
              new THREE.Color([0x7a2e2a, 0x24324a, 0x1e1e1e, 0x6b4a32, 0xe8e4da][Math.floor(rnd() * 5)]));
            // Bay window over the garage, storey 1 to the top.
            if (high && levels >= 2 && lw >= 4.2 && rnd() < 0.65) {
              const cx = t0 + lw / 2;
              bay(g, trim, at, cx, sh - 0.15, b.h - CORNICE_H - 0.05, color, trimColor, (k, p, y, oo) =>
                inst[k].push({ m: basis(p, y, oo), frame: trimColor, pane: paneTint() }), sh, levels);
              busy.push([cx - 1.7, cx + 1.7]);
            }
          }
        } else if (street) {
          if (L > 6 && rnd() < 0.6) {
            place("rollup", L * (0.25 + rnd() * 0.5), 0.05 + KINDS.rollup.h / 2, trimColor, new THREE.Color(0xc0c4c6));
          } else rowOfWindows(L, 3.4, (t) => place("ind", t, Math.min(sh * 0.5, 1.9)));
        }
        if (wantEscape) {   // look 2: a fire escape where it clears the bays (3.6 m wide, 1 m out)
          const t = [0.4, 0.25, 0.6, 0.15, 0.75].map((k) => L * k).find((t) => t > 2 && t < L - 2 && !busy.some(([lo, hi]) => t + 2 > lo && t - 2 < hi));
          const fe = t === undefined ? null : fireEscape(levels, sh, b.h);
          if (fe && t !== undefined) escapes.push(fe.applyMatrix4(basis(at(t), 0, out)));
        }

        // Upper storeys (and the ground floor of back walls).
        const wk: KindName = vic ? "vic" : midrise ? "large" : residential ? "res" : "ind";
        const pitch = vic ? 2.0 : midrise ? 2.8 : residential ? 2.3 : 3.4;
        for (let k = street ? 1 : 0; k < levels; k++) {
          const y = k * sh + Math.min(sh * 0.55, 1.95);
          const head = look2 && wk === "vic" ? VIC_WINDOW_TOP : KINDS[wk].h / 2;   // look 2: the hood counts too
          if (y - KINDS[wk].h / 2 < y0 || y + head > y1 - (high ? corniceH : 0) - 0.1) continue;
          rowOfWindows(L, pitch, (t) => {
            if (!busy.some(([lo, hi]) => t > lo - 0.6 && t < hi + 0.6)) place(wk, t, y);
          });
        }
      });
    }
    for (const { lay, segs } of faces) {
      if (!layoutFronts || !segs.length) continue;
      const r = layoutFronts.front({
        b, lay, wall: g, trim, color, trimColor, segs,
        garageColor: sheet?.garage_colour ? new THREE.Color(sheet.garage_colour) : undefined,
        walkRoom: (p) => walk.length > 1 ? Math.min(...walk.slice(1).map((q, k) => segDist(p, walk[k], q))) - WALK_CLEAR : STOOP_D,
        bay: (at, x0, x1, y0, y1, ys) => bay(g, trim, at, (x0 + x1) / 2, y0, y1, color, trimColor, (k, p, y, oo) =>
          inst[x1 - x0 >= BIG_BAY ? (k === "res" ? "bayc" : "bays") : k].push({ m: basis(p, y, oo), frame: trimColor, pane: paneTint() }), sh, levels, { W: x1 - x0, ys }),
      });
      stoops.push(...r.stoops);
      for (const q of r.garages) garages.push({ x: q.p[0], n: q.p[1], u: q.u, out: q.out, w: q.w });
    }
  });
  layoutFronts?.finish(scene);

  for (const [kind, geo] of walls) {
    const mesh = new THREE.Mesh(geo.build(), facadeMaterial(kind));
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
  }
  const trimMesh = new THREE.Mesh(trim.build(), new THREE.MeshStandardMaterial({ name: "trim", roughness: 0.65, vertexColors: true }));
  trimMesh.castShadow = trimMesh.receiveShadow = true;
  scene.add(trimMesh);
  const roofMesh = new THREE.Mesh(mergeGeometries(roofs)!, pbr("roof", 2.2, { roughness: 0.95, color: 0x9a9a98 }));
  roofMesh.receiveShadow = true;
  scene.add(roofMesh);

  for (const [name, list] of Object.entries(inst) as [KindName, Inst[]][]) {
    if (!list.length) continue;
    const k: Kind = KINDS[name];
    const frame = new THREE.InstancedMesh(frameGeometry(k), trimMat, list.length);
    const pane = new THREE.InstancedMesh(paneGeometry(k), k.pane, list.length);
    list.forEach((it, i) => {
      frame.setMatrixAt(i, it.m); frame.setColorAt(i, it.frame);
      pane.setMatrixAt(i, it.m); pane.setColorAt(i, it.pane);
    });
    frame.castShadow = true;
    frame.receiveShadow = pane.receiveShadow = true;
    scene.add(frame, pane);
  }
  const windows = Object.values(inst).reduce((s, l) => s + l.length, 0);
  if (gates.length) scene.add(gateMesh(gates, KINDS.door.w + 0.1, KINDS.door.h + 0.1));
  if (hedges.length) scene.add(hedgeMesh(hedges));
  if (escapes.length) {
    const m = new THREE.Mesh(mergeGeometries(escapes)!, new THREE.MeshStandardMaterial({ name: "iron", color: 0x1a1a1a, roughness: 0.55, metalness: 0.5 }));
    m.castShadow = m.receiveShadow = true;
    scene.add(m);
  }
  return { garages, windows, stoops };
}

// --- look 2: iron gates and doorstep hedges
const GATE_SHARE = 0.55, HEDGE_SHARE = 0.45;   // Street View on Moss: most doors gated, many stoops with a hedge
const HEDGE_D = 0.6, HEDGE_MIN_D = 0.35;       // m: hedge depth out from the wall, and the shallowest worth drawing
interface Hedge { at: (t: number, o?: number) => P; t0: number; t1: number; d: number; h: number }

const FIRE_ESCAPE_SHARE = 0.5;   // Howard's north side: about every other mid-rise has one on the street

/** A painted-black fire escape, local frame (x along the wall, y up, z out): a grated landing on every floor
 *  from the second up, railings with balusters, and a stair between landings. */
function fireEscape(levels: number, sh: number, top: number) {
  const W = 3.6, D = 1.0, parts: THREE.BufferGeometry[] = [];
  const box = (w: number, h: number, d: number, x: number, y: number, z: number) => parts.push(new THREE.BoxGeometry(w, h, d).translate(x, y, z).toNonIndexed());
  for (let k = 1; k < levels; k++) {
    const y = k * sh - 0.1;
    if (y + 1 > top) break;
    box(W, 0.04, D, 0, y, D / 2 + 0.05);                                    // landing
    box(W, 0.04, 0.04, 0, y + 0.95, D + 0.03); box(0.04, 0.04, D, -W / 2, y + 0.95, D / 2 + 0.05); box(0.04, 0.04, D, W / 2, y + 0.95, D / 2 + 0.05);
    box(W, 0.03, 0.03, 0, y + 0.45, D + 0.03);
    for (let x = -W / 2; x <= W / 2 + 1e-6; x += 0.15) box(0.015, 0.95, 0.015, x, y + 0.47, D + 0.03);
    for (const sx of [-1, 1]) for (let z = 0.2; z < D; z += 0.15) box(0.015, 0.95, 0.015, sx * W / 2, y + 0.47, z);
    if (k + 1 < levels && (k + 1) * sh - 0.1 + 1 <= top) {                // stair up to the next landing
      const run = W * 0.7, rise = sh, len = Math.hypot(run, rise);
      const stair = new THREE.BoxGeometry(len, 0.05, 0.6).rotateZ(Math.atan2(rise, run)).translate(0, y + rise / 2, 0.4);
      parts.push(stair.toNonIndexed());
    }
  }
  return parts.length ? mergeGeometries(parts)! : null;
}

/** Iron security door: a frame, vertical bars every 11 cm, three rails and a band of rings. */
function gateMesh(ms: THREE.Matrix4[], w: number, h: number) {
  const parts: THREE.BufferGeometry[] = [];
  const bar = (bw: number, bh: number, x: number, y: number) => parts.push(new THREE.BoxGeometry(bw, bh, 0.025).translate(x, y, 0));
  bar(0.05, h, -w / 2, 0); bar(0.05, h, w / 2, 0); bar(w, 0.05, 0, h / 2); bar(w, 0.05, 0, -h / 2);
  for (const y of [-h * 0.3, h * 0.12, h * 0.32]) bar(w, 0.03, 0, y);
  for (let x = -w / 2 + 0.11; x < w / 2 - 0.05; x += 0.11) bar(0.016, h, x, 0);
  for (let x = -w / 2 + 0.12; x < w / 2 - 0.08; x += 0.17) parts.push(new THREE.TorusGeometry(0.06, 0.008, 4, 12).translate(x + 0.03, h * 0.22, 0));
  const g = mergeGeometries(parts.map((p) => (p.index ? p.toNonIndexed() : p)))!;
  const m = new THREE.InstancedMesh(g, new THREE.MeshStandardMaterial({ name: "iron", color: 0x121212, roughness: 0.45, metalness: 0.6 }), ms.length);
  ms.forEach((x, i) => m.setMatrixAt(i, x));
  m.castShadow = m.receiveShadow = true;
  return m;
}

/** Clipped hedges: a dark green box core under small leaf cards scattered over its front, top and ends. */
function hedgeMesh(hs: Hedge[]) {
  const core: THREE.BufferGeometry[] = [];
  const pos: number[] = [], nor: number[] = [], uv: number[] = [];
  const r = seeded(31);
  const UV = [[0, 0], [1, 0], [1, 1], [0, 1]];
  for (const hd of hs) {
    const a = hd.at(hd.t0), b = hd.at(hd.t1), mid = hd.at((hd.t0 + hd.t1) / 2, hd.d / 2);
    const out = hd.at(hd.t0, 1);
    const len = hd.t1 - hd.t0;
    // Local frame: x along the wall, z out of it, y up (three.js: north is -z).
    const x = new THREE.Vector3(b[0] - a[0], 0, -(b[1] - a[1])).normalize();
    const z = new THREE.Vector3(out[0] - a[0], 0, -(out[1] - a[1])).normalize();
    const m = new THREE.Matrix4().makeBasis(x, new THREE.Vector3(0, 1, 0), z).setPosition(mid[0], 0, -mid[1]);
    if (m.determinant() < 0) m.makeBasis(x.negate(), new THREE.Vector3(0, 1, 0), z).setPosition(mid[0], 0, -mid[1]);
    core.push(new THREE.BoxGeometry(len - 0.1, hd.h - 0.08, hd.d - 0.1).translate(0, (hd.h - 0.08) / 2, 0).applyMatrix4(m));
    const n = Math.ceil(len * hd.h * 30);
    for (let k = 0; k < n; k++) {
      const face = r();
      const lp = face < 0.55 ? new THREE.Vector3((r() - 0.5) * len, r() * hd.h, hd.d / 2)
        : face < 0.85 ? new THREE.Vector3((r() - 0.5) * len, hd.h, (r() - 0.5) * hd.d)
        : new THREE.Vector3((r() < 0.5 ? -1 : 1) * len / 2, r() * hd.h, (r() - 0.5) * hd.d);
      const nrm = face < 0.55 ? new THREE.Vector3(0, 0.3, 1) : face < 0.85 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(Math.sign(lp.x), 0.3, 0);
      nrm.add(new THREE.Vector3(r() - 0.5, r() - 0.5, r() - 0.5).multiplyScalar(0.9)).normalize();
      const p = lp.applyMatrix4(m), wn = nrm.transformDirection(m);
      const t = new THREE.Vector3().crossVectors(wn, Math.abs(wn.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)).normalize();
      const bt = new THREE.Vector3().crossVectors(wn, t);
      const s = 0.2 + r() * 0.15;
      const q = UV.map(([i, j]) => p.clone().addScaledVector(t, (i * 2 - 1) * s).addScaledVector(bt, (j * 2 - 1) * s));
      for (const i of [0, 1, 2, 0, 2, 3]) { pos.push(q[i].x, q[i].y, q[i].z); nor.push(wn.x, wn.y, wn.z); uv.push(UV[i][0], UV[i][1]); }
    }
  }
  const coreMesh = new THREE.Mesh(mergeGeometries(core.map((g) => g.toNonIndexed()))!, new THREE.MeshStandardMaterial({ color: 0x1d3018, roughness: 0.95 }));
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  lg.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
  lg.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  const leaves = new THREE.Mesh(lg, new THREE.MeshStandardMaterial({ map: leafTexture(), alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.8, color: 0xc0d0b0 }));
  coreMesh.castShadow = coreMesh.receiveShadow = leaves.castShadow = leaves.receiveShadow = true;
  return new THREE.Group().add(coreMesh, leaves);
}

// --- helpers
const v3 = (p: P, y: number) => new THREE.Vector3(p[0], y, -p[1]);

/** look 2: a footprint ring without slivers: corners closer than 10 cm merged, corners on a straight line
 *  (under 8 degrees of turn) dropped. Bay clipping leaves fronts in collinear pieces a few cm apart, and every
 *  piece got its own lot (a door and a window each), crowding three doors onto one house. */
export function simplify(r: P[]): P[] {
  let pts = r.filter((p, i) => i === 0 || Math.hypot(p[0] - r[i - 1][0], p[1] - r[i - 1][1]) > 0.1);
  for (let changed = true; changed && pts.length > 3;) {
    changed = false;
    for (let i = 0; i < pts.length && pts.length > 3; i++) {
      const p = pts[(i + pts.length - 1) % pts.length], q = pts[i], s = pts[(i + 1) % pts.length];
      const a1 = Math.atan2(q[1] - p[1], q[0] - p[0]), a2 = Math.atan2(s[1] - q[1], s[0] - q[0]);
      const turn = Math.abs(((a2 - a1 + 3 * Math.PI) % (2 * Math.PI)) - Math.PI);
      if (turn < THREE.MathUtils.degToRad(8) || Math.hypot(s[0] - q[0], s[1] - q[1]) < 0.1) { pts = pts.filter((_, j) => j !== i); changed = true; break; }
    }
  }
  return pts;
}

/** Instance matrix: local +z out of the wall, +y up. x = y cross z keeps it a rotation (never mirrored),
 *  since InstancedMesh doesn't flip face winding for mirrored instances. */
function basis(p: P, y: number, out: P) {
  const z = new THREE.Vector3(out[0], 0, -out[1]), up = new THREE.Vector3(0, 1, 0);
  return new THREE.Matrix4().makeBasis(up.clone().cross(z), up, z).setPosition(p[0], y, -p[1]);
}

function rowOfWindows(L: number, pitch: number, put: (t: number) => void) {
  const n = Math.floor((L - 0.8) / pitch);
  for (let j = 0; j < n; j++) put(L / 2 + (j - (n - 1) / 2) * pitch);
}

/** A band along a wall from a to b at heights y0..y1, sticking out `o`: front, top and soffit faces. */
function cornice(g: Geo, a: P, b: P, out: P, y0: number, y1: number, o: number, color: THREE.Color, ends: [boolean, boolean] = [true, true]) {
  const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const u: P = [(b[0] - a[0]) / L, (b[1] - a[1]) / L];
  const ea = ends[0] ? o : 0, eb = ends[1] ? o : 0;
  const A: P = [a[0] - u[0] * ea, a[1] - u[1] * ea], B: P = [b[0] + u[0] * eb, b[1] + u[1] * eb];   // cover open corners
  const Ao: P = [A[0] + out[0] * o, A[1] + out[1] * o], Bo: P = [B[0] + out[0] * o, B[1] + out[1] * o];
  const n = new THREE.Vector3(out[0], 0, -out[1]);
  const len = L + ea + eb;
  g.quad([v3(Ao, y0), v3(Bo, y0), v3(Bo, y1), v3(Ao, y1)], n, [[0, y0], [len, y0], [len, y1], [0, y1]], color);
  g.quad([v3(A, y1), v3(B, y1), v3(Bo, y1), v3(Ao, y1)], new THREE.Vector3(0, 1, 0), [[0, 0], [len, 0], [len, o], [0, o]], color);
  g.quad([v3(A, y0), v3(B, y0), v3(Bo, y0), v3(Ao, y0)], new THREE.Vector3(0, -1, 0), [[0, 0], [len, 0], [len, o], [0, o]], color);
}

/** A block against a wall: t0..t1 along it, o0..o1 out of it, y0..y1 up; front, top, both ends and soffit. */
function slab(g: Geo, at: (t: number, o?: number) => P, u: P, out: P, t0: number, t1: number, o0: number, o1: number,
              y0: number, y1: number, color: THREE.Color) {
  const A = at(t0, o0), B = at(t1, o0), C = at(t1, o1), D = at(t0, o1);
  const n = (p: P) => new THREE.Vector3(p[0], 0, -p[1]);
  const uv: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
  g.quad([v3(D, y0), v3(C, y0), v3(C, y1), v3(D, y1)], n(out), uv, color);
  g.quad([v3(A, y1), v3(B, y1), v3(C, y1), v3(D, y1)], new THREE.Vector3(0, 1, 0), uv, color);
  g.quad([v3(A, y0), v3(B, y0), v3(C, y0), v3(D, y0)], new THREE.Vector3(0, -1, 0), uv, color);
  g.quad([v3(A, y0), v3(D, y0), v3(D, y1), v3(A, y1)], n([-u[0], -u[1]]), uv, color);
  g.quad([v3(B, y0), v3(C, y0), v3(C, y1), v3(B, y1)], n(u), uv, color);
}

type PlaceFn = (k: "res" | "narrow", p: P, y: number, out: P) => void;

/** An SF angled bay: a trapezoid 3 m wide at the wall, 1.5 m at the front, BAY_D deep. */
function bay(g: Geo, trim: Geo, at: (t: number, o?: number) => P, c: number, y0: number, y1: number,
             color: THREE.Color, trimColor: THREE.Color, place: PlaceFn, sh: number, levels: number,
             opts: { W?: number; ys?: number[] } = {}) {
  const W = opts.W ?? 3.0, Wf = W / 2, D = BAY_D;
  const ys = opts.ys ?? Array.from({ length: Math.max(0, levels - 1) }, (_, k) => (k + 1) * sh + Math.min(sh * 0.55, 1.95));
  const q = [at(c - W / 2), at(c - Wf / 2, D), at(c + Wf / 2, D), at(c + W / 2)];
  const back = at(c);
  for (let i = 0; i < 3; i++) {
    const a = q[i], b = q[i + 1];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const su: P = [(b[0] - a[0]) / L, (b[1] - a[1]) / L];
    let so: P = [su[1], -su[0]];
    const mid: P = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    if (so[0] * (mid[0] - back[0]) + so[1] * (mid[1] - back[1]) < 0) so = [-so[0], -so[1]];
    g.quad([v3(a, y0), v3(b, y0), v3(b, y1), v3(a, y1)], new THREE.Vector3(so[0], 0, -so[1]),
      [[0, y0], [L, y0], [L, y1], [0, y1]], color);
    cornice(trim, a, b, so, y1 - 0.3, y1, 0.1, trimColor);
    for (const y of ys) {
      if (opts.ys ? y + 1.05 > y1 - 0.3 || y - 1.05 < y0 : y + 0.85 > y1 - 0.3) continue;
      place(i === 1 ? "res" : "narrow", mid, y, so);
    }
  }
  const down = new THREE.Vector3(0, -1, 0), up = new THREE.Vector3(0, 1, 0);
  const uv: [number, number][] = q.map((p) => [p[0], p[1]]);
  g.quad(q.map((p) => v3(p, y0)), down, uv, color);
  g.quad(q.map((p) => v3(p, y1)), up, uv, color);
}

export function segDist(p: P, a: P, b: P) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

export function inQuad(p: P, q: Quad) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4];
    const c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if (c !== 0) { if (sign && Math.sign(c) !== sign) return false; sign = Math.sign(c); }
  }
  return true;
}

export function inPoly(p: P, r: P[]) {
  let c = false;
  for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
    if ((r[i][1] > p[1]) !== (r[j][1] > p[1]) && p[0] < r[i][0] + ((p[1] - r[i][1]) * (r[j][0] - r[i][0])) / (r[j][1] - r[i][1])) c = !c;
  }
  return c;
}
