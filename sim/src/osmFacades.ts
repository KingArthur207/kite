// Building facades for the Moss world: OSM footprints become textured walls with a material per
// building (OSM tags, else the building type and a seeded random from its id), window grids per
// storey, ground floors (garages and doors on residential fronts, shop glass where OSM has a shop),
// SF bay windows on some residential street fronts, and a cornice at the roof line.
// Windows, doors and garage doors are instanced: one draw call per kind.
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { Geo, facadeMaterial, pbr, seeded, type FacadeKind } from "./osmLook";

type P = [number, number];
type Quad = P[];
export interface Building { id: number; h: number; pts: P[]; tags?: Record<string, string>; storefront?: boolean }
/** A garage door on a street front: centre on the wall, along-wall unit u, outward normal n, width. */
export interface Garage { x: number; n: number; u: P; out: P; w: number }

const STOREY = 3.2;
const PARTY_GAP = 0.6;          // m: a wall this close and parallel to another building's wall is shared
const STREET_LOOK = 20;         // m: a wall whose normal hits a roadway within this is a street front
const LOT = 7.6;                // m: SF 25 ft lot, one garage + door + bay per lot
const CORNICE_H = 0.4, CORNICE_OUT = 0.22;

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
  const type = t.building ?? "yes";
  let mix = RESIDENTIAL.has(type) ? MIX.residential : COMMERCIAL.has(type) ? MIX.commercial : MIX.any;
  if (b.h > 14) mix = mix.filter(([k]) => k !== "siding" && k !== "metal");   // no wood or tin towers
  let r = rnd() * mix.reduce((s, [, w]) => s + w, 0);
  for (const [k, w] of mix) if ((r -= w) < 0) return k;
  return mix[0][0];
}

function tint(b: Building, kind: FacadeKind, rnd: () => number) {
  const tag = b.tags?.["building:colour"];
  const c = new THREE.Color(TINTS[kind][Math.floor(rnd() * TINTS[kind].length)]);
  if (tag) { try { c.setStyle(tag); } catch { /* unknown colour name: keep the palette pick */ } }
  return c.convertSRGBToLinear();
}

// --- window/door kinds: an instanced frame (bars + sill + mullions) and an instanced pane.
interface Kind { w: number; h: number; f: number; d: number; mull: "v" | "h" | "vh" | "none"; pane: THREE.Material; uvMetres?: boolean }
interface Inst { m: THREE.Matrix4; frame: THREE.Color; pane: THREE.Color }

function frameGeometry(k: Kind) {
  const { w, h, f, d } = k;
  const bars: THREE.BufferGeometry[] = [];
  const bar = (bw: number, bh: number, bd: number, x: number, y: number) =>
    bars.push(new THREE.BoxGeometry(bw, bh, bd).translate(x, y, bd / 2));
  bar(w + 2 * f, f, d, 0, h / 2 + f / 2);                 // head
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

export function buildFacades(scene: THREE.Object3D, buildings: Building[], roadQuads: Quad[]) {
  const glass = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.06, metalness: 0.65, envMapIntensity: 1.3 });
  const trimMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.65 });
  const doorMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 });
  const shutter = pbr("shutter", 2.0, { roughness: 0.55 });
  const KINDS = {
    res: { w: 1.05, h: 1.7, f: 0.09, d: 0.12, mull: "h", pane: glass },
    narrow: { w: 0.62, h: 1.6, f: 0.08, d: 0.1, mull: "h", pane: glass },
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
  const rings = buildings.map((b) => {
    const r = b.pts.slice();
    if (r.length > 1 && r[0][0] === r[r.length - 1][0] && r[0][1] === r[r.length - 1][1]) r.pop();
    return r;
  });
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
    const ring = rings[bi];
    if (ring.length < 3) return;
    const rnd = seeded(b.id);
    const kind = pickKind(b, rnd);
    const color = tint(b, kind, rnd);
    const residential = (kind === "stucco" || kind === "plaster" || kind === "siding" || RESIDENTIAL.has(b.tags?.building ?? "")) && !b.storefront;
    const lv = parseFloat(b.tags?.["building:levels"] ?? "");
    const levels = Math.max(1, Number.isFinite(lv) ? Math.round(lv) : Math.round(b.h / STOREY));
    const sh = b.h / levels;
    const trimColor = new THREE.Color(residential ? (rnd() < 0.75 ? 0xf3f0e8 : 0x4a4f55) : rnd() < 0.5 ? 0x3b3833 : 0x8d9296);
    const paneTint = () => new THREE.Color().setHSL(0.58, 0.12, rnd() < 0.15 ? 0.45 : 0.1 + rnd() * 0.08);
    const g = walls.get(kind) ?? walls.set(kind, new Geo()).get(kind)!;
    const area = ring.reduce((s, p, i) => s + p[0] * ring[(i + 1) % ring.length][1] - ring[(i + 1) % ring.length][0] * p[1], 0);
    let s0 = rnd() * 10;

    const roof = new THREE.ShapeGeometry(new THREE.Shape(ring.map(([x, y]) => new THREE.Vector2(x, y))));
    roofs.push(roof.rotateX(-Math.PI / 2).translate(0, b.h, 0));

    ring.forEach((a, i) => {
      const c = ring[(i + 1) % ring.length];
      const L = Math.hypot(c[0] - a[0], c[1] - a[1]);
      if (L < 0.05) return;
      const u: P = [(c[0] - a[0]) / L, (c[1] - a[1]) / L];
      const out: P = area > 0 ? [u[1], -u[0]] : [-u[1], u[0]];
      const n3 = new THREE.Vector3(out[0], 0, -out[1]);
      g.quad([v3(a, 0), v3(c, 0), v3(c, b.h), v3(a, b.h)], n3, [[s0, 0], [s0 + L, 0], [s0 + L, b.h], [s0, b.h]], color);
      s0 += L;
      if (isParty(bi, a, c)) return;   // shared wall: blank
      const at = (t: number, o = 0): P => [a[0] + u[0] * t + out[0] * o, a[1] + u[1] * t + out[1] * o];
      cornice(trim, a, c, out, b.h - CORNICE_H, b.h, CORNICE_OUT, trimColor);
      if (L < 1.4) return;
      const street = isStreet(bi, at(L / 2), out);
      const place = (k: KindName, t: number, y: number, frame = trimColor, pane = paneTint(), o = 0, oo = out) =>
        inst[k].push({ m: basis(at(t, o), y, oo), frame, pane });
      const busy: [number, number][] = [];   // along-wall spans taken by bays (upper floors)

      // Ground floor on street fronts.
      if (street && b.storefront) {
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
          if (levels >= 2 && lw >= 4.2 && rnd() < 0.65) {
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

      // Upper storeys (and the ground floor of back walls).
      const wk: KindName = residential ? "res" : "ind";
      const pitch = residential ? 2.3 : 3.4;
      for (let k = street ? 1 : 0; k < levels; k++) {
        const y = k * sh + Math.min(sh * 0.55, 1.95);
        if (y + KINDS[wk].h / 2 > b.h - CORNICE_H - 0.1) continue;
        rowOfWindows(L, pitch, (t) => {
          if (!busy.some(([lo, hi]) => t > lo - 0.6 && t < hi + 0.6)) place(wk, t, y);
        });
      }
    });
  });

  for (const [kind, geo] of walls) {
    const mesh = new THREE.Mesh(geo.build(), facadeMaterial(kind));
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
  }
  const trimMesh = new THREE.Mesh(trim.build(), new THREE.MeshStandardMaterial({ roughness: 0.65, vertexColors: true }));
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
  return { garages, windows };
}

// --- helpers
const v3 = (p: P, y: number) => new THREE.Vector3(p[0], y, -p[1]);

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
function cornice(g: Geo, a: P, b: P, out: P, y0: number, y1: number, o: number, color: THREE.Color) {
  const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const u: P = [(b[0] - a[0]) / L, (b[1] - a[1]) / L];
  const A: P = [a[0] - u[0] * o, a[1] - u[1] * o], B: P = [b[0] + u[0] * o, b[1] + u[1] * o];   // cover corners
  const Ao: P = [A[0] + out[0] * o, A[1] + out[1] * o], Bo: P = [B[0] + out[0] * o, B[1] + out[1] * o];
  const n = new THREE.Vector3(out[0], 0, -out[1]);
  const len = L + 2 * o;
  g.quad([v3(Ao, y0), v3(Bo, y0), v3(Bo, y1), v3(Ao, y1)], n, [[0, y0], [len, y0], [len, y1], [0, y1]], color);
  g.quad([v3(A, y1), v3(B, y1), v3(Bo, y1), v3(Ao, y1)], new THREE.Vector3(0, 1, 0), [[0, 0], [len, 0], [len, o], [0, o]], color);
  g.quad([v3(A, y0), v3(B, y0), v3(Bo, y0), v3(Ao, y0)], new THREE.Vector3(0, -1, 0), [[0, 0], [len, 0], [len, o], [0, o]], color);
}

type PlaceFn = (k: "res" | "narrow", p: P, y: number, out: P) => void;

/** An SF angled bay: a trapezoid 3 m wide at the wall, 1.5 m at the front, 0.65 m deep. */
function bay(g: Geo, trim: Geo, at: (t: number, o?: number) => P, c: number, y0: number, y1: number,
             color: THREE.Color, trimColor: THREE.Color, place: PlaceFn, sh: number, levels: number) {
  const W = 3.0, Wf = 1.5, D = 0.65;
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
    for (let k = 1; k < levels; k++) {
      const y = k * sh + Math.min(sh * 0.55, 1.95);
      if (y + 0.85 > y1 - 0.3) continue;
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
