// ?look=2: Moss building fronts drawn from per-building layouts read off Street View
// (public/local/moss_facade_layouts.json, scripts/facade_layouts.py; local only, never committed).
// A layout lists every opening of one front in metres: windows, bay windows, doors, garage doors and shop windows,
// each rectangular, round-arched or segmental-arched, with garage styles (roll-up, panel, iron gate), plus the floors,
// cornice and belt courses. The front becomes one wall with real holes in it: a reveal (the recess) around each
// opening, the glass, door or shutter set back in it, a casing around it that never runs into a neighbour's, and
// projecting angled bays where the layout has them. Fronts without a layout keep the procedural facade (osmFacades).
import * as THREE from "three";
import { Geo } from "./osmLook";

type P = [number, number];
export interface Opening {
  type: "window" | "bay" | "door" | "garage" | "storefront";
  x0: number; x1: number; y0: number; y1: number;
  shape?: "rect" | "arch" | "segmental"; spring?: number | null; style?: "rollup" | "panel" | "gate" | null; panes?: number;
  frame?: "white" | "dark" | "wall";
}
export interface FacadeLayout {
  x0: number; x1: number; roofline: number; floors: { y0: number; y1: number }[];
  cornice: { y0: number; y1: number } | null; belts: number[]; openings: Opening[];
  width: number; a: P; b: P;
}
export const facadeLayouts = new Map<number, FacadeLayout>();

/** Load the layouts if they're there (look 2 calls this with the building sheet). */
export async function loadFacadeLayouts() {
  const r = await fetch("local/moss_facade_layouts.json").catch(() => null);
  if (!r?.ok || !r.headers.get("content-type")?.includes("json")) return;   // vite answers a missing file with index.html
  try {
    const j = (await r.json()) as { buildings?: Record<string, FacadeLayout> };
    for (const [id, l] of Object.entries(j.buildings ?? {})) if (usable(l)) facadeLayouts.set(Number(id), l);
    console.info(`[look2] facade layouts: ${facadeLayouts.size} fronts`);
  } catch (e) { console.error("[look2] moss_facade_layouts.json unreadable, procedural fronts", e); }
}

function usable(l: FacadeLayout) {
  return Array.isArray(l.openings) && Array.isArray(l.a) && Array.isArray(l.b) && l.x1 > l.x0 && l.roofline > 2;
}

/** Roofline from the photo when it's believable (OSM/LiDAR heights run 2-3 m under Street View on Moss). */
export function layoutHeight(l: FacadeLayout, h: number) {
  return l.roofline > h * 0.7 && l.roofline < h * 1.7 ? l.roofline : h;
}

/** Does footprint edge a->c lie on the layout's front (same line, facing the same way)? */
export function onFront(l: FacadeLayout, a: P, c: P, out: P) {
  const fx = l.b[0] - l.a[0], fy = l.b[1] - l.a[1], L = Math.hypot(fx, fy);
  const n: P = [fy / L, -fx / L];
  const nOut = n[0] * out[0] + n[1] * out[1];
  if (Math.abs(nOut) < 0.9) return false;
  const off = (p: P) => Math.abs((p[0] - l.a[0]) * n[0] + (p[1] - l.a[1]) * n[1]);
  const t = (p: P) => ((p[0] - l.a[0]) * fx + (p[1] - l.a[1]) * fy) / L;
  return off(a) < FRONT_TOL && off(c) < FRONT_TOL && Math.min(t(a), t(c)) > -FRONT_TOL && Math.max(t(a), t(c)) < L + FRONT_TOL;
}
const FRONT_TOL = 0.8;   // m

// --- sizes (m)
const RECESS = { window: 0.12, door: 0.28, garage: 0.22, storefront: 0.14, bay: 0 };
const CASING = 0.11, CASING_D = 0.05, SILL_H = 0.07, SILL_D = 0.07, SILL_OVER = 0.05;
const EDGE = 0.04;             // holes stay this far inside the wall outline (triangulation needs them apart)
const BAY_D = 0.7;             // SF angled bay depth
const BELT_H = 0.16, BELT_OUT = 0.06;
const CORNICE_OUT = 0.32, BRACKET_PITCH = 0.9;
const STOOP_MIN_RISE = 0.25, STOOP_D = 1.0, STOOP_MIN_D = 0.45;
const GATE_BAR = 0.11;         // m between gate bars
const ARC_SEGS = 14;
const DARK = new THREE.Color(0x2a2b2d), IRON = new THREE.Color(0x141414), INTERIOR = new THREE.Color(0x34322f);
const ARCH_GLASS = new THREE.Color(0x8a99a6);   // the arched top of a window (plain glass, reflections over a lighter tint)
const DOORS = [0x7a2e2a, 0x24324a, 0x1e1e1e, 0x2f5a46, 0x5b3b2a];

export interface FrontCtx {
  b: { id: number; h: number }; lay: FacadeLayout;
  wall: Geo; trim: Geo; color: THREE.Color; trimColor: THREE.Color; garageColor?: THREE.Color;
  walkRoom: (p: P) => number;                 // m free in front of the wall at p before the demo walk's clearance
  bay: (x0: number, x1: number, y0: number, y1: number, ys: number[]) => [number, number];   // draws an angled bay, returns its along-wall span
}

/** Accumulates every layout front, then adds their meshes. */
export class LayoutFronts {
  private glass: THREE.Matrix4[] = [];
  private archGlass = new Geo();
  private doors = new Geo();
  private shutters = new Geo();
  private panels = new Geo();
  private iron = new Geo();
  constructor(private mats: { glass: THREE.Material; shutter: THREE.MeshStandardMaterial; panels: THREE.MeshStandardMaterial }) {}

  /** Draw one front. Returns stoop collision segments and garage doors (for parked-car clearances). */
  front(ctx: FrontCtx) {
    const { lay, color, trimColor } = ctx;
    const A = lay.a, B = lay.b;
    const L = Math.hypot(B[0] - A[0], B[1] - A[1]);
    const u: P = [(B[0] - A[0]) / L, (B[1] - A[1]) / L];
    const out: P = [u[1], -u[0]];                          // a -> b runs left to right seen from the street
    const H = layoutHeight(lay, ctx.b.h);
    // The photo is rectified in metres along the same line (a -> b), so its coordinates are the wall's.
    const X = (x: number) => x, Y = (y: number) => y, sy = 1;
    const F = new Frame(A, u, out);
    const ops = lay.openings.map((o) => ({ ...o, x0: X(o.x0), x1: X(o.x1), y0: Y(o.y0), y1: Y(o.y1) }))
      .map((o) => ({ ...o, x0: Math.max(EDGE, o.x0), x1: Math.min(L - EDGE, o.x1), y0: Math.max(o.type === "bay" ? 0.5 : EDGE, o.y0),
        y1: Math.min(H - EDGE - (lay.cornice ? (lay.cornice.y1 - lay.cornice.y0) * sy : 0), o.y1) }))
      .filter((o) => o.x1 - o.x0 > 0.25 && o.y1 - o.y0 > 0.25);
    const holes = ops.filter((o) => o.type !== "bay");
    const bays = ops.filter((o) => o.type === "bay");
    const floors = lay.floors.map((f) => ({ y0: Y(f.y0), y1: Y(f.y1) }));
    const rnd = mulberry(ctx.b.id);

    // Wall with the holes cut out.
    const shape = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(L, 0), new THREE.Vector2(L, H), new THREE.Vector2(0, H)]);
    const paths = holes.map((o) => holePath(o));
    for (const p of paths) shape.holes.push(new THREE.Path(p.map(([x, y]) => new THREE.Vector2(x, y)).reverse()));
    F.shape(ctx.wall, new THREE.ShapeGeometry(shape), 0.0, [0, 0, 1], color, (x, y) => [x, y]);

    const stoops: [number, number, number, number][] = [];
    const garages: { t: number; w: number }[] = [];
    holes.forEach((o, i) => {
      const path = paths[i], R = RECESS[o.type];
      F.reveal(ctx.wall, path, R, color);
      const c = casingWidth(o, holes, L, H);
      const frameCol = o.frame === "dark" ? DARK : trimColor;
      if (o.frame !== "wall" && c > 0.03) F.casing(ctx.trim, path, c, CASING_D, frameCol);
      const w = o.x1 - o.x0;
      if (o.type === "window") {
        const sw = Math.min(w + 2 * c + 2 * SILL_OVER, w + 2 * casingWidth(o, holes, L, H, true));
        F.box(ctx.trim, (o.x0 + o.x1) / 2 - sw / 2, (o.x0 + o.x1) / 2 + sw / 2, o.y0 - SILL_H, o.y0, -0.02, SILL_D, frameCol);
      }
      this.infill(F, o, path, R, frameCol, ctx, rnd, color);
      if (o.type === "door" && o.y0 > STOOP_MIN_RISE) stoops.push(...this.stoop(F, ctx, o));
      if (o.type === "garage") garages.push({ t: (o.x0 + o.x1) / 2, w });
    });

    // Bays, with a window per floor on each face.
    const busy: [number, number][] = [];
    for (const o of bays) {
      const ys = floors.filter((f) => f.y0 >= o.y0 - 0.5 && f.y1 <= o.y1 + 0.5).map((f) => f.y0 + Math.min((f.y1 - f.y0) * 0.55, 1.95));
      busy.push(ctx.bay(o.x0, o.x1, o.y0, o.y1, ys));
    }
    // Belt courses and the cornice, broken around bays and openings so no band runs through them.
    for (const y0 of lay.belts ?? []) {
      const y = Y(y0);
      if (y < 0.5 || y > H - 0.5) continue;
      const cuts: [number, number][] = [...busy, ...holes.filter((o) => o.y0 < y + BELT_H / 2 && o.y1 + CASING > y - BELT_H / 2)
        .map((o): [number, number] => [o.x0 - CASING, o.x1 + CASING])];
      for (const [t0, t1] of gaps(0, L, cuts)) F.box(ctx.trim, t0, t1, y - BELT_H / 2, y + BELT_H / 2, 0, BELT_OUT, trimColor);
    }
    if (lay.cornice) {
      const y0 = Math.min(Y(lay.cornice.y0), H - 0.25), y1 = H;
      F.box(ctx.trim, 0, L, y1 - Math.min(0.3, y1 - y0), y1, 0, CORNICE_OUT, trimColor);
      if (y1 - y0 > 0.45) {   // frieze board and brackets under the crown
        F.box(ctx.trim, 0, L, y0, y1 - 0.3, 0, 0.04, trimColor);
        for (let t = 0.3; t < L - 0.2; t += BRACKET_PITCH) F.box(ctx.trim, t - 0.06, t + 0.06, Math.max(y0, y1 - 0.6), y1 - 0.3, 0, CORNICE_OUT - 0.04, trimColor);
      }
    }
    return { stoops, garages: garages.map((g) => ({ p: F.at(g.t), u, out, w: g.w })) };
  }

  private infill(F: Frame, o: Opening, path: P[], R: number, frameCol: THREE.Color, ctx: FrontCtx, rnd: () => number, wall: THREE.Color) {
    const w = o.x1 - o.x0, cx = (o.x0 + o.x1) / 2;
    const shape = new THREE.Shape(path.map(([x, y]) => new THREE.Vector2(x, y)));
    const spring = springLine(o);
    if (o.type === "window" || o.type === "storefront") {
      // Glass: the rectangle up to the spring line is room glass (instanced, interior mapping), the arch cap dark glass.
      this.glass.push(F.matrix(cx, (o.y0 + spring) / 2, -R, w, spring - o.y0));
      if (spring < o.y1 - 0.02) {
        const cap = new THREE.Shape(path.filter(([, y]) => y >= spring - 1e-6).map(([x, y]) => new THREE.Vector2(x, y)));
        F.shape(this.archGlass, new THREE.ShapeGeometry(cap), -R + 0.005, [0, 0, 1], ARCH_GLASS, (x, y) => [x, y]);
      }
      const bar = o.type === "storefront" ? 0.06 : 0.045, col = o.type === "storefront" ? DARK : frameCol;
      const n = Math.max(1, Math.min(8, Math.round(o.panes ?? (w > 1.6 ? 2 : 1))));
      for (let k = 1; k < n; k++) {   // mullions, up into the arch
        const x = o.x0 + (w * k) / n;
        F.box(ctx.trim, x - bar / 2, x + bar / 2, o.y0, topAt(path, x, spring) - 0.01, -R, -R + 0.04, col);
      }
      if (o.type === "window") {   // glazing bars: a meeting rail on sashes, a square grid on industrial windows
        const cell = n >= 3 ? w / n : spring - o.y0 > 1.1 ? (spring - o.y0) / 2 : Infinity;
        for (let y = o.y0 + cell; y < o.y1 - 0.08; y += cell) {
          const [l, r] = spanAt(path, y);
          if (r - l > 0.1) F.box(ctx.trim, l, r, y - bar / 2, y + bar / 2, -R, -R + 0.04, col);
        }
      }
      if (spring < o.y1 - 0.02) F.box(ctx.trim, o.x0, o.x1, spring - bar / 2, spring + bar / 2, -R, -R + 0.04, col);   // transom bar
      return;
    }
    if (o.type === "door") {
      const dc = new THREE.Color(DOORS[Math.floor(rnd() * DOORS.length)]);
      F.shape(this.doors, new THREE.ShapeGeometry(shape), -R, [0, 0, 1], dc, (x, y) => [x - o.x0, y - o.y0]);
      if (spring < o.y1 - 0.02) {   // fanlight over the door
        const cap = new THREE.Shape(path.filter(([, y]) => y >= spring - 1e-6).map(([x, y]) => new THREE.Vector2(x, y)));
        F.shape(this.archGlass, new THREE.ShapeGeometry(cap), -R + 0.01, [0, 0, 1], ARCH_GLASS, (x, y) => [x, y]);
      }
      if (o.style === "gate") this.gate(F, path, o, -0.06);
      return;
    }
    // garage
    const style = o.style ?? "rollup";
    if (style === "gate") {
      F.shape(this.doors, new THREE.ShapeGeometry(shape), -R, [0, 0, 1], INTERIOR, (x, y) => [x, y]);
      this.gate(F, path, o, -0.08);
    } else {
      const tint = ctx.garageColor ?? (style === "panel" ? new THREE.Color(0xeeeeea) : new THREE.Color(0xb8bcbe));
      const geo = style === "panel" ? this.panels : this.shutters;
      F.shape(geo, new THREE.ShapeGeometry(shape), -R, [0, 0, 1], tint,
        style === "panel" ? (x, y) => [(x - o.x0) / w, (y - o.y0) / (spring - o.y0)] : (x, y) => [x, y]);
    }
    void wall;
  }

  /** Iron security gate filling an opening's outline: frame, vertical bars, two rails. */
  private gate(F: Frame, path: P[], o: Opening, z: number) {
    const spring = springLine(o), top = (x: number) => topAt(path, x, spring);
    for (let x = o.x0 + GATE_BAR / 2; x < o.x1 - 0.02; x += GATE_BAR) F.box(this.iron, x - 0.008, x + 0.008, o.y0, top(x) - 0.02, z, z + 0.02, IRON);
    for (const y of [o.y0 + 0.08, o.y0 + (spring - o.y0) * 0.45, spring - 0.06]) F.box(this.iron, o.x0, o.x1, y - 0.02, y + 0.02, z, z + 0.025, IRON);
    F.box(this.iron, o.x0, o.x0 + 0.04, o.y0, spring, z, z + 0.03, IRON);
    F.box(this.iron, o.x1 - 0.04, o.x1, o.y0, spring, z, z + 0.03, IRON);
  }

  /** Steps up to a raised door, stopping short of the demo walk. */
  private stoop(F: Frame, ctx: FrontCtx, o: Opening): [number, number, number, number][] {
    const t0 = o.x0 - 0.15, t1 = o.x1 + 0.15, rise = o.y0;
    const depth = Math.min(STOOP_D + 0.3 * Math.max(0, Math.round(rise / 0.18) - 3), ctx.walkRoom(F.at((t0 + t1) / 2)));
    if (depth < STOOP_MIN_D) return [];
    const n = Math.max(1, Math.round(rise / 0.18));
    for (let k = 0; k < n; k++) F.box(ctx.trim, t0, t1, 0, (rise * (k + 1)) / n, 0, (depth * (n - k)) / n, new THREE.Color(0xc4c1ba));
    const s = [F.at(t0), F.at(t0, depth), F.at(t1, depth), F.at(t1)];
    return [0, 1, 2].map((k) => [...s[k], ...s[k + 1]] as [number, number, number, number]);
  }

  finish(scene: THREE.Object3D) {
    const add = (g: Geo, m: THREE.Material) => {
      if (g.empty) return;
      const mesh = new THREE.Mesh(g.build(), m);
      mesh.castShadow = mesh.receiveShadow = true;
      scene.add(mesh);
    };
    const vc = (m: THREE.MeshStandardMaterial) => { const c = m.clone(); c.vertexColors = true; return c; };
    add(this.doors, new THREE.MeshStandardMaterial({ name: "door", vertexColors: true, roughness: 0.5 }));
    add(this.shutters, vc(this.mats.shutter));
    add(this.panels, vc(this.mats.panels));
    add(this.archGlass, new THREE.MeshStandardMaterial({ name: "archglass", vertexColors: true, roughness: 0.08, metalness: 0.6, envMapIntensity: 1.4 }));
    add(this.iron, new THREE.MeshStandardMaterial({ name: "iron", vertexColors: true, roughness: 0.45, metalness: 0.6 }));
    if (this.glass.length) {
      const panes = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1), this.mats.glass, this.glass.length);
      this.glass.forEach((m, i) => panes.setMatrixAt(i, m));
      panes.receiveShadow = true;
      scene.add(panes);
    }
  }
}

// --- opening outlines (local: x along the wall, y up), counter-clockwise from the bottom-left corner
function springLine(o: Opening) {
  const w = o.x1 - o.x0, h = o.y1 - o.y0;
  if (o.shape && o.shape !== "rect" && o.spring != null && o.spring > o.y0 + 0.1 && o.spring < o.y1 - 0.05) {
    return Math.max(o.spring, o.y1 - w / 2);   // never more than a half circle
  }
  if (o.shape === "arch") return Math.max(o.y0 + 0.2, o.y1 - Math.min(w / 2, h * 0.6));
  if (o.shape === "segmental") return o.y1 - Math.min(0.18 * w, 0.3 * h);
  return o.y1;
}

/** Left and right edge of an opening's outline at height y. */
function spanAt(path: P[], y: number): [number, number] {
  const xs: number[] = [];
  path.forEach((p, i) => {
    const q = path[(i + 1) % path.length];
    if ((y - p[1]) * (y - q[1]) <= 0 && p[1] !== q[1]) xs.push(p[0] + ((y - p[1]) / (q[1] - p[1])) * (q[0] - p[0]));
  });
  return xs.length >= 2 ? [Math.min(...xs), Math.max(...xs)] : [0, 0];
}

export function holePath(o: Opening): P[] {
  const { x0, x1, y0, y1 } = o, cx = (x0 + x1) / 2, hw = (x1 - x0) / 2;
  const ys = springLine(o);
  if (ys >= y1 - 1e-6) return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  const rise = y1 - ys;
  // Circle through (x0, ys), (x1, ys) and (cx, y1): a half circle when rise = hw, flatter when less.
  const R = (hw * hw + rise * rise) / (2 * rise), cy = y1 - R;
  const a0 = Math.atan2(ys - cy, hw);
  const pts: P[] = [[x0, y0], [x1, y0], [x1, ys]];
  for (let k = 1; k < ARC_SEGS; k++) {
    const a = a0 + ((Math.PI - 2 * a0) * k) / ARC_SEGS;
    pts.push([cx + R * Math.cos(a), cy + R * Math.sin(a)]);
  }
  pts.push([x0, ys]);
  return pts;
}

/** Height of an opening's top edge at x (the arch where there is one). */
function topAt(path: P[], x: number, spring: number) {
  let best = spring;
  for (let i = 2; i < path.length - 1; i++) {
    const [ax, ay] = path[i], [bx, by] = path[i + 1];
    if ((x - ax) * (x - bx) <= 0 && ax !== bx) best = Math.max(best, ay + ((x - ax) / (bx - ax)) * (by - ay));
  }
  return best;
}

/** Casing width: CASING, narrowed so it stops halfway to the next opening and inside the wall. */
function casingWidth(o: Opening, all: Opening[], L: number, H: number, sideOnly = false) {
  let c = Math.min(CASING, o.x0 - 0.01, L - o.x1 - 0.01, sideOnly ? CASING : H - o.y1 - 0.01);
  for (const q of all) {
    if (q === o) continue;
    const dx = Math.max(q.x0 - o.x1, o.x0 - q.x1), dy = Math.max(q.y0 - o.y1, o.y0 - q.y1);
    if (dy < 2 * CASING + SILL_H && dx >= 0) c = Math.min(c, dx / 2 - 0.01);
    if (dx < 2 * CASING && dy >= 0 && !sideOnly) c = Math.min(c, (dy - SILL_H) / 2 - 0.01);
  }
  return Math.max(0, c);
}

/** Spans of lo..hi not covered by cuts. */
export function gaps(lo: number, hi: number, cuts: [number, number][]) {
  const out: [number, number][] = [];
  let t = lo;
  for (const [a, b] of [...cuts].sort((p, q) => p[0] - q[0])) {
    if (a > t + 0.05) out.push([t, Math.min(a, hi)]);
    t = Math.max(t, b);
  }
  if (hi > t + 0.05) out.push([t, hi]);
  return out.filter(([a, b]) => b > a + 0.05);
}

function mulberry(seed: number) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** A wall's local frame: t along it from A (u), y up, z out of it. World: x east, y up, z south. */
class Frame {
  constructor(private A: P, private u: P, private o: P) {}
  at(t: number, z = 0): P { return [this.A[0] + this.u[0] * t + this.o[0] * z, this.A[1] + this.u[1] * t + this.o[1] * z]; }
  v(t: number, y: number, z: number) { const p = this.at(t, z); return new THREE.Vector3(p[0], y, -p[1]); }
  n(nx: number, ny: number, nz: number) {
    return new THREE.Vector3(this.u[0] * nx + this.o[0] * nz, ny, -(this.u[1] * nx + this.o[1] * nz));
  }
  /** Instance matrix for a unit plane: centre (t, y, z), size w x h, facing out. */
  matrix(t: number, y: number, z: number, w: number, h: number) {
    const x = this.n(1, 0, 0), up = new THREE.Vector3(0, 1, 0), zz = this.n(0, 0, 1);
    const m = new THREE.Matrix4().makeBasis(x, up, zz);
    if (m.determinant() < 0) m.makeBasis(x.negate(), up, zz);   // keep it a rotation (no mirrored instances)
    return m.scale(new THREE.Vector3(w, h, 1)).setPosition(this.v(t, y, z));
  }
  /** A flat (non-indexed) shape geometry in local x/y at depth z into g, UVs from uv(x, y). */
  shape(g: Geo, geo: THREE.BufferGeometry, z: number, n: [number, number, number], color: THREE.Color, uv: (x: number, y: number) => [number, number]) {
    const ng = geo.index ? geo.toNonIndexed() : geo;
    const pos = ng.getAttribute("position"), wn = this.n(...n);
    for (let i = 0; i < pos.count; i += 3) {
      const p = [0, 1, 2].map((k) => this.v(pos.getX(i + k), pos.getY(i + k), z));
      const uvs = [0, 1, 2].map((k) => uv(pos.getX(i + k), pos.getY(i + k)));
      tri(g, p, wn, uvs, color);
    }
  }
  /** The recess: faces from the wall face (z 0) back to -R along an opening's outline, facing into the opening. */
  reveal(g: Geo, path: P[], R: number, color: THREE.Color) {
    const cx = path.reduce((s, p) => s + p[0], 0) / path.length, cy = path.reduce((s, p) => s + p[1], 0) / path.length;
    let s = 0;
    path.forEach((p, i) => {
      const q = path[(i + 1) % path.length];
      const dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy);
      let nx = -dy / len, ny = dx / len;
      if (nx * (cx - (p[0] + q[0]) / 2) + ny * (cy - (p[1] + q[1]) / 2) < 0) { nx = -nx; ny = -ny; }
      g.quad([this.v(p[0], p[1], 0), this.v(q[0], q[1], 0), this.v(q[0], q[1], -R), this.v(p[0], p[1], -R)], this.n(nx, ny, 0),
        [[s, 0], [s + len, 0], [s + len, R], [s, R]], color);
      s += len;
    });
  }
  /** Casing: a band of width c and depth d around the outline, except along the bottom edge (sill or threshold). */
  casing(g: Geo, path: P[], c: number, d: number, color: THREE.Color) {
    const pts = [...path.slice(1), path[0]];                          // bottom-right, up, over, down to bottom-left
    const segN = (p: P, q: P): P => { const dx = q[0] - p[0], dy = q[1] - p[1], l = Math.hypot(dx, dy); return [dy / l, -dx / l]; };   // outward (ccw)
    const outer = pts.map((p, i) => {
      const n1 = i > 0 ? segN(pts[i - 1], p) : null, n2 = i < pts.length - 1 ? segN(p, pts[i + 1]) : null;
      if (!n1) return [p[0] + n2![0] * c, p[1]] as P;                   // ends: square to the bottom edge
      if (!n2) return [p[0] + n1[0] * c, p[1]] as P;
      const k = c / (1 + n1[0] * n2[0] + n1[1] * n2[1]);
      return [p[0] + (n1[0] + n2[0]) * k, p[1] + (n1[1] + n2[1]) * k] as P;
    });
    for (let i = 0; i + 1 < pts.length; i++) {
      const p = pts[i], q = pts[i + 1], P0 = outer[i], Q0 = outer[i + 1];
      g.quad([this.v(p[0], p[1], d), this.v(q[0], q[1], d), this.v(Q0[0], Q0[1], d), this.v(P0[0], P0[1], d)], this.n(0, 0, 1),
        [[0, 0], [1, 0], [1, 1], [0, 1]], color);
      const [nx, ny] = segN(p, q);
      g.quad([this.v(P0[0], P0[1], 0), this.v(Q0[0], Q0[1], 0), this.v(Q0[0], Q0[1], d), this.v(P0[0], P0[1], d)], this.n(nx, ny, 0),
        [[0, 0], [1, 0], [1, 1], [0, 1]], color);
    }
    for (const [p, o] of [[pts[0], outer[0]], [pts[pts.length - 1], outer[pts.length - 1]]] as [P, P][]) {   // end caps
      g.quad([this.v(p[0], p[1], 0), this.v(o[0], o[1], 0), this.v(o[0], o[1], d), this.v(p[0], p[1], d)], this.n(0, -1, 0),
        [[0, 0], [1, 0], [1, 1], [0, 1]], color);
    }
  }
  /** Box t0..t1, y0..y1, z0..z1 (z out of the wall): all six faces. */
  box(g: Geo, t0: number, t1: number, y0: number, y1: number, z0: number, z1: number, color: THREE.Color) {
    const v = (t: number, y: number, z: number) => this.v(t, y, z);
    const uv: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
    g.quad([v(t0, y0, z1), v(t1, y0, z1), v(t1, y1, z1), v(t0, y1, z1)], this.n(0, 0, 1), uv, color);
    g.quad([v(t0, y0, z0), v(t1, y0, z0), v(t1, y1, z0), v(t0, y1, z0)], this.n(0, 0, -1), uv, color);
    g.quad([v(t0, y1, z0), v(t1, y1, z0), v(t1, y1, z1), v(t0, y1, z1)], this.n(0, 1, 0), uv, color);
    g.quad([v(t0, y0, z0), v(t1, y0, z0), v(t1, y0, z1), v(t0, y0, z1)], this.n(0, -1, 0), uv, color);
    g.quad([v(t0, y0, z0), v(t0, y1, z0), v(t0, y1, z1), v(t0, y0, z1)], this.n(-1, 0, 0), uv, color);
    g.quad([v(t1, y0, z0), v(t1, y1, z0), v(t1, y1, z1), v(t1, y0, z1)], this.n(1, 0, 0), uv, color);
  }
}

function tri(g: Geo, p: THREE.Vector3[], n: THREE.Vector3, uv: [number, number][], color: THREE.Color) {
  const e1 = p[1].clone().sub(p[0]), e2 = p[2].clone().sub(p[0]);
  const order = e1.cross(e2).dot(n) >= 0 ? [0, 1, 2] : [0, 2, 1];
  for (const i of order) {
    g.pos.push(p[i].x, p[i].y, p[i].z); g.nor.push(n.x, n.y, n.z);
    g.uv.push(uv[i][0], uv[i][1]); g.col.push(color.r, color.g, color.b);
  }
}
