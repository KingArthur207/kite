// Streets for the Moss world: textured asphalt with lane lines, zebras and stop lines as decals,
// concrete sidewalks on 15 cm curbs, driveway aprons in front of garages, tactile pads at the
// lowered kerbs OSM maps, OSM street furniture, and (not in OSM, added by us) banner-carrying light poles,
// green cycle-track paint with edge lines and flex posts on Howard. Poles, wires, street trees, parked cars and the rest of the dressing are in
// streetDressing.ts. Nothing here collides; the walker's collision is the building faces and curbs.
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { Geo, pbr } from "./osmLook";
import { inPoly, inQuad, type Garage } from "./osmFacades";
import { bannerTexture } from "./streetProps";

type P = [number, number];
type Quad = P[];
interface Road { id: number; name: string; oneway: boolean; lanes: number; curbs: [number, number]; pts: P[]; curbs_at?: [number, number][] }
interface Lane { road: number; pts: P[] }
interface Crossing { x: number; y: number; markings: string; u: P; across: P; road: number }
interface Furniture { id: number; kind: string; x: number; y: number; tags: Record<string, string> }
export interface StreetData {
  roads: Road[]; lanes: Lane[]; crossings: Crossing[]; furniture: Furniture[];
  road_quads: Quad[]; track_quads: Quad[]; sidewalk_quads: Quad[];
  buildings: { pts: P[] }[];
}

const CURB_H = 0.15;
const MARK_Y = 0.012;
const DASH_ON = 3, DASH_OFF = 9;       // m: SF lane line, 10 ft on, 30 ft off
const LINE_W = 0.12;
const HOWARD_LIGHT_EVERY = 36;
const BANNER_KINDS = 3;
const TRACK_BUFFER = 0.9;          // m: painted buffer between the cycle track and traffic (osm_corner.py TRACK_BUFFER)
const TRACK_DETAIL_REACH = 120;    // m from the corner: edge lines and flex posts
const FLEX_POST_EVERY = 6;
const BBOX = { w: -80, s: -150, e: 140, n: 70 };
const v3 = (p: P, y: number) => new THREE.Vector3(p[0], y, -p[1]);
const UP = new THREE.Vector3(0, 1, 0);

function flatGeo(quads: Quad[], y: number) {
  const g = new Geo();
  for (const q of quads) g.quad(q.map((p) => v3(p, y)), UP, q.map((p) => [p[0], p[1]] as [number, number]));
  return g.build();
}

export function ribbon(pts: P[], w: number): Quad[] {
  const out: Quad[] = [];
  for (let i = 0; i + 1 < pts.length; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (d < 1e-3) continue;
    const nx = (-(b[1] - a[1]) / d) * w / 2, ny = ((b[0] - a[0]) / d) * w / 2;
    out.push([[a[0] + nx, a[1] + ny], [b[0] + nx, b[1] + ny], [b[0] - nx, b[1] - ny], [a[0] - nx, a[1] - ny]]);
  }
  return out;
}

function dashes(pts: P[], w: number): Quad[] {
  const out: Quad[] = [];
  let phase = 0;
  for (let i = 0; i + 1 < pts.length; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    for (let s = -phase; s < L; s += DASH_ON + DASH_OFF) {
      const s0 = Math.max(0, s), s1 = Math.min(L, s + DASH_ON);
      if (s1 - s0 < 0.2) continue;
      const lerp = (t: number): P => [a[0] + ((b[0] - a[0]) * t) / L, a[1] + ((b[1] - a[1]) * t) / L];
      out.push(...ribbon([lerp(s0), lerp(s1)], w));
    }
    phase = (phase + L) % (DASH_ON + DASH_OFF);
  }
  return out;
}

/** pts offset sideways (+ is left of travel) by o, or by o[i] at point i. */
export const offsetPts = (pts: P[], o: number | number[]): P[] => pts.map((p, i) => {
  const off = typeof o === "number" ? o : o[i];
  const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
  const L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
  return [p[0] - ((b[1] - a[1]) / L) * off, p[1] + ((b[0] - a[0]) / L) * off];
});

/** Per-point offsets of a road's curb (side 1 = left of travel), `inset` further out toward the buildings
 *  (negative: into the road). Uses the surveyed per-point curbs (`curbs_at`) when the road has them. */
export const curbOffsets = (r: Road, side: 0 | 1, inset: number): number[] =>
  r.pts.map((_, i) => (side ? (r.curbs_at?.[i]?.[1] ?? r.curbs[1]) + inset : (r.curbs_at?.[i]?.[0] ?? r.curbs[0]) - inset));

/** Arc-length samples along a polyline: point, unit direction, distance along. */
export function* along(pts: P[], first: number, every: number) {
  let s0 = 0, next = first;
  for (let i = 0; i + 1 < pts.length; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const u: P = [(b[0] - a[0]) / L, (b[1] - a[1]) / L];
    while (next <= s0 + L) {
      const t = next - s0;
      yield { p: [a[0] + u[0] * t, a[1] + u[1] * t] as P, u, s: next };
      next += every;
    }
    s0 += L;
  }
}

export function nearestOnRoads(roads: Road[], p: P) {
  let best = { d: Infinity, road: roads[0], q: p as P, u: [1, 0] as P, side: 0 };
  for (const r of roads) for (let i = 0; i + 1 < r.pts.length; i++) {
    const [a, b] = [r.pts[i], r.pts[i + 1]];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    const u: P = [(b[0] - a[0]) / L, (b[1] - a[1]) / L];
    const t = Math.max(0, Math.min(L, (p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1]));
    const q: P = [a[0] + u[0] * t, a[1] + u[1] * t];
    const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
    if (d < best.d) best = { d, road: r, q, u, side: u[0] * (p[1] - a[1]) - u[1] * (p[0] - a[0]) };
  }
  return best;
}

export function buildStreets(scene: THREE.Object3D, data: StreetData, garages: Garage[]) {
  const asphalt = pbr("asphalt", 3.0, { rough: true });
  const sidewalk = pbr("sidewalk", 1.8, { rough: true });
  const curb = pbr("curb", 2.5, { roughness: 0.9 });
  const paint = new THREE.MeshStandardMaterial({ color: 0xe6e4dc, roughness: 0.75, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
  const yellow = paint.clone(); yellow.color.set(0xd9a521);
  const add = (g: THREE.BufferGeometry, m: THREE.Material, shadow = false) => {
    const mesh = new THREE.Mesh(g, m);
    mesh.receiveShadow = true;
    mesh.castShadow = shadow;
    scene.add(mesh);
    return mesh;
  };

  // Ground beyond the modelled blocks: plain concrete-coloured lot.
  const ground = add(new THREE.PlaneGeometry(800, 800).rotateX(-Math.PI / 2).translate(0, -0.02, 0),
    new THREE.MeshStandardMaterial({ color: 0x77746e, roughness: 0.95 }));
  ground.receiveShadow = true;
  add(flatGeo(data.road_quads, 0), asphalt);
  // Howard's protected cycle track: green paint (Street View), a white line at its edge and the buffer's.
  const track = new THREE.MeshStandardMaterial({ color: 0x4f9a5c, roughness: 0.8, normalMap: asphalt.normalMap });
  add(flatGeo(data.track_quads, 0.004), track);

  // Lane lines between a road's lanes: white dashes between same-direction lanes, double yellow otherwise.
  const white: Quad[] = [], yell: Quad[] = [];
  for (const r of data.roads) {
    if (r.lanes < 2) continue;
    const ls = data.lanes.filter((l) => l.road === r.id).map((l) => {
      const fwd = Math.hypot(l.pts[0][0] - r.pts[0][0], l.pts[0][1] - r.pts[0][1]) <
        Math.hypot(l.pts[l.pts.length - 1][0] - r.pts[0][0], l.pts[l.pts.length - 1][1] - r.pts[0][1]);
      return { fwd, pts: fwd ? l.pts : [...l.pts].reverse() };
    });
    for (let i = 0; i + 1 < ls.length; i++) {
      const [a, b] = [ls[i], ls[i + 1]];
      if (a.pts.length !== b.pts.length) continue;
      const mid = a.pts.map((p, k): P => [(p[0] + b.pts[k][0]) / 2, (p[1] + b.pts[k][1]) / 2]);
      if (a.fwd === b.fwd) white.push(...dashes(mid, LINE_W));
      else yell.push(...ribbon(offsetPts(mid, 0.12), LINE_W), ...ribbon(offsetPts(mid, -0.12), LINE_W));
    }
  }
  // Zebra stripes, and a stop line before each marked crossing on a one-way road's approach side.
  for (const c of data.crossings) {
    if (!["zebra", "yes", "ladder"].includes(c.markings)) continue;
    const [ux, uy] = c.u, nx = -uy, ny = ux;
    const at = (a: number, b: number): P => [c.x + ux * a + nx * b, c.y + uy * a + ny * b];
    for (let o = c.across[0] + 0.5; o < c.across[1] - 0.3; o += 1.0) {
      white.push([at(-1.5, o - 0.25), at(1.5, o - 0.25), at(1.5, o + 0.25), at(-1.5, o + 0.25)]);
    }
    const road = data.roads.find((r) => r.id === c.road);
    const stop = -1.5 - 1.2;
    const mid = at(stop, (c.across[0] + c.across[1]) / 2);
    const inOtherRoad = data.road_quads.some((q) => inQuad(mid, q)) &&
      data.roads.some((r) => r.id !== c.road && nearestOnRoads([r], mid).d < Math.abs(r.curbs[1] - r.curbs[0]) / 2);
    if (road?.oneway && !inOtherRoad) {
      white.push([at(stop - 0.3, c.across[0] + 0.1), at(stop + 0.3, c.across[0] + 0.1), at(stop + 0.3, c.across[1] - 0.1), at(stop - 0.3, c.across[1] - 0.1)]);
    }
  }
  for (const t of trackEdges(data)) white.push(...ribbon([t.a, t.b], LINE_W), ...ribbon(offsetPts([t.a, t.b], t.side * TRACK_BUFFER), LINE_W));
  add(flatGeo(white, MARK_Y), paint);
  if (yell.length) add(flatGeo(yell, MARK_Y), yellow);

  // Sidewalks: concrete slab tops, curb faces on the sides.
  const top = new Geo(), side = new Geo();
  for (const q of data.sidewalk_quads) {
    top.quad(q.map((p) => v3(p, CURB_H)), UP, q.map((p) => [p[0], p[1]] as [number, number]));
    for (let i = 0; i < 4; i++) {
      const a = q[i], b = q[(i + 1) % 4], L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (L < 1e-3) continue;
      const cx = q.reduce((s, p) => s + p[0], 0) / 4, cy = q.reduce((s, p) => s + p[1], 0) / 4;
      let n: P = [(b[1] - a[1]) / L, -(b[0] - a[0]) / L];
      if (n[0] * (a[0] - cx) + n[1] * (a[1] - cy) < 0) n = [-n[0], -n[1]];
      side.quad([v3(a, 0), v3(b, 0), v3(b, CURB_H), v3(a, CURB_H)], new THREE.Vector3(n[0], 0, -n[1]),
        [[0, 0], [L, 0], [L, CURB_H], [0, CURB_H]]);
    }
  }
  add(top.build(), sidewalk);
  add(side.build(), curb);

  // Driveway aprons: a ramp from the curb top down into the gutter in front of each garage.
  const ramps = new Geo();
  for (const g of garages) {
    if (data.crossings.some((c) => Math.hypot(c.x - g.x, c.y - g.n) < 6)) continue;   // no apron over a crosswalk
    const near = nearestOnRoads(data.roads, [g.x, g.n]);
    const off = near.side > 0 ? near.road.curbs[1] : near.road.curbs[0];
    if (Math.abs(near.side) - Math.abs(off) > 8 || Math.abs(near.side) < Math.abs(off)) continue;
    const left: P = [-near.u[1], near.u[0]];
    const c: P = [near.q[0] + left[0] * off, near.q[1] + left[1] * off];
    const inward: P = [-Math.sign(near.side) * left[0], -Math.sign(near.side) * left[1]];
    const hw = g.w / 2 + 0.6;
    const p0: P = [c[0] - near.u[0] * hw, c[1] - near.u[1] * hw], p1: P = [c[0] + near.u[0] * hw, c[1] + near.u[1] * hw];
    const p2: P = [p1[0] + inward[0] * 0.7, p1[1] + inward[1] * 0.7], p3: P = [p0[0] + inward[0] * 0.7, p0[1] + inward[1] * 0.7];
    const n = new THREE.Vector3(inward[0] * 0.2, 1, -inward[1] * 0.2).normalize();
    ramps.quad([v3(p0, CURB_H + 0.004), v3(p1, CURB_H + 0.004), v3(p2, 0.006), v3(p3, 0.006)], n,
      [p0, p1, p2, p3].map((p) => [p[0], p[1]] as [number, number]));
  }
  if (!ramps.empty) add(ramps.build(), curb);

  buildFurniture(scene, data, garages);
}

/** The traffic-side edge of each cycle track quad (a to b), with side = +1/-1 for offsetPts toward the traffic. */
function trackEdges(data: StreetData) {
  const out: { a: P; b: P; side: number }[] = [];
  for (const q of data.track_quads) {
    const mid = (p: P, r: P): P => [(p[0] + r[0]) / 2, (p[1] + r[1]) / 2];
    const inner = nearestOnRoads(data.roads, mid(q[0], q[1])).d < nearestOnRoads(data.roads, mid(q[2], q[3])).d;
    const [a, b] = inner ? [q[0], q[1]] : [q[2], q[3]];
    const other = inner ? q[3] : q[0];
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) < 0.2 || Math.hypot(a[0], a[1]) > TRACK_DETAIL_REACH) continue;
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const left = (-(b[1] - a[1]) * (other[0] - a[0]) + (b[0] - a[0]) * (other[1] - a[1])) / L;   // >0: track on the left
    out.push({ a, b, side: left > 0 ? -1 : 1 });
  }
  return out;
}

/** Not in OSM: light poles along both Howard curbs, clear of crossings, garages and buildings.
 *  toRoad: unit vector from the pole toward the roadway. streetDressing.ts keeps trees off them. */
export function howardLightSpots(data: StreetData, garages: Garage[]) {
  const free = (p: P) =>
    p[0] > BBOX.w && p[0] < BBOX.e && p[1] > BBOX.s && p[1] < BBOX.n &&
    !data.road_quads.some((q) => inQuad(p, q)) && !data.buildings.some((b) => inPoly(p, b.pts)) &&
    !data.crossings.some((c) => Math.hypot(c.x - p[0], c.y - p[1]) < 5) &&
    !garages.some((g) => Math.hypot(g.x - p[0], g.n - p[1]) < 2.7);
  const out: { p: P; toRoad: P; u: P }[] = [];
  for (const r of data.roads.filter((x) => x.name === "Howard Street")) {
    for (const side of [0, 1] as const) {
      const off = side ? r.curbs[1] + 0.5 : r.curbs[0] - 0.5;
      for (const { p, u } of along(offsetPts(r.pts, off), 10 + side * 18, HOWARD_LIGHT_EVERY)) {
        if (free(p)) out.push({ p, u, toRoad: side ? [u[1], -u[0]] : [-u[1], u[0]] });
      }
    }
  }
  return out;
}

// --- furniture: geometry merged per material, so a handful of draw calls in all.
function buildFurniture(scene: THREE.Object3D, data: StreetData, garages: Garage[]) {
  const M = {
    wood: new THREE.MeshStandardMaterial({ color: 0x5a4a3a, roughness: 0.92 }),
    steel: new THREE.MeshStandardMaterial({ color: 0x8a918d, roughness: 0.45, metalness: 0.6 }),
    signal: new THREE.MeshStandardMaterial({ color: 0x2c332e, roughness: 0.6, metalness: 0.3 }),
    hydrant: new THREE.MeshStandardMaterial({ color: 0xe9e6dc, roughness: 0.5, metalness: 0.2 }),
    cap: new THREE.MeshStandardMaterial({ color: 0x2f5fa8, roughness: 0.45, metalness: 0.2 }),
    lamp: new THREE.MeshStandardMaterial({ color: 0xf4f1e2, emissive: 0x1a1810, roughness: 0.3 }),
    green: new THREE.MeshStandardMaterial({ color: 0x40d070, emissive: 0x20c050, emissiveIntensity: 1.5, roughness: 0.3 }),
    unlit: new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.3 }),
    paystation: new THREE.MeshStandardMaterial({ color: 0x2e3946, roughness: 0.5, metalness: 0.4 }),
    tactile: new THREE.MeshStandardMaterial({ color: 0xd8ae25, roughness: 0.6, polygonOffset: true, polygonOffsetFactor: -2 }),
    banner: new THREE.MeshStandardMaterial({ color: 0x23302a, roughness: 0.5, metalness: 0.4 }),
    base: new THREE.MeshStandardMaterial({ color: 0xa8322b, roughness: 0.55 }),
  };
  type MK = keyof typeof M;
  const parts = new Map<MK, THREE.BufferGeometry[]>();
  const put = (k: MK, g: THREE.BufferGeometry) => {
    g.deleteAttribute("uv");
    (parts.get(k) ?? parts.set(k, []).get(k)!).push(g.index ? g.toNonIndexed() : g);
  };
  /** The `u` for place() that turns local +z toward f. */
  const zdir = (f: P): P => [-f[1], f[0]];
  /** Geometry placed at (p, y) with local +x along `u` (east/north metres) and +z toward `f`. */
  const place = (g: THREE.BufferGeometry, p: P, y: number, u: P = [1, 0]) =>
    g.applyMatrix4(new THREE.Matrix4().makeRotationY(Math.atan2(u[1], u[0])).setPosition(p[0], y, -p[1]));
  const cyl = (r0: number, r1: number, h: number, seg = 10) => new THREE.CylinderGeometry(r1, r0, h, seg).translate(0, h / 2, 0);
  const box = (x: number, y: number, z: number) => new THREE.BoxGeometry(x, y, z);

  const ys = (p: P) => (data.road_quads.some((q) => inQuad(p, q)) ? 0 : CURB_H);

  const hydrant = (p: P) => {
    const y = ys(p);
    put("hydrant", place(cyl(0.17, 0.17, 0.06, 12), p, y));
    put("hydrant", place(cyl(0.13, 0.12, 0.6, 12), p, y));
    put("hydrant", place(box(0.42, 0.09, 0.09).translate(0, 0.42, 0), p, y));
    put("cap", place(new THREE.SphereGeometry(0.15, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2).translate(0, 0.62, 0), p, y));
  };
  const woodPole = (p: P, across: P, light: boolean) => {   // across: toward the road
    const y = ys(p), u = zdir(across);
    put("wood", place(cyl(0.15, 0.12, 10.5), p, y));
    put("wood", place(box(0.1, 0.1, 2.4).translate(0, 9.6, 0), p, y, u));              // crossarm across the street
    for (const z of [-1.0, 0, 1.0]) put("unlit", place(cyl(0.04, 0.03, 0.18, 6).translate(0, 9.65, z), p, y, u));
    if (light) {
      put("steel", place(box(0.06, 0.06, 1.8).translate(0, 7.4, 0.9), p, y, u));
      put("steel", place(box(0.3, 0.14, 0.6).translate(0, 7.35, 1.9), p, y, u));
      put("lamp", place(box(0.24, 0.02, 0.5).translate(0, 7.27, 1.9), p, y, u));
    }
  };
  const streetLight = (p: P, toRoad: P) => {
    const y = ys(p), u = zdir(toRoad);   // arm along local +z, over the road
    put("steel", place(cyl(0.12, 0.08, 8.5), p, y));
    put("steel", place(box(0.07, 0.07, 2.2).translate(0, 8.3, 1.1), p, y, u));
    put("steel", place(box(0.32, 0.15, 0.7).translate(0, 8.25, 2.3), p, y, u));
    put("lamp", place(box(0.26, 0.02, 0.55).translate(0, 8.17, 2.3), p, y, u));
  };
  const signal = (p: P, face: P) => {
    const y = ys(p), u = zdir(face);
    put("signal", place(cyl(0.12, 0.1, 5.2), p, y));
    put("signal", place(box(0.36, 1.05, 0.3).translate(0, 4.3, 0.2), p, y, u));
    put("unlit", place(new THREE.CircleGeometry(0.11, 12).translate(0, 4.62, 0.36), p, y, u));
    put("unlit", place(new THREE.CircleGeometry(0.11, 12).translate(0, 4.3, 0.36), p, y, u));
    put("green", place(new THREE.CircleGeometry(0.11, 12).translate(0, 3.98, 0.36), p, y, u));
  };
  const payStation = (p: P, u: P) => {
    const y = ys(p);
    put("paystation", place(box(0.34, 1.45, 0.24).translate(0, 0.725, 0), p, y, u));
    put("unlit", place(box(0.22, 0.16, 0.02).translate(0, 1.2, 0.13), p, y, u));
  };
  const stopSign = (p: P, face: P) => {
    const y = ys(p), u = zdir(face);
    put("steel", place(cyl(0.035, 0.035, 2.5, 6), p, y));
    const sign = new THREE.Mesh(new THREE.CircleGeometry(0.38, 8, Math.PI / 8), new THREE.MeshStandardMaterial({ map: stopTexture(), roughness: 0.5 }));
    sign.geometry.translate(0, 0, 0.05);
    sign.geometry.applyMatrix4(new THREE.Matrix4().makeRotationY(Math.atan2(u[1], u[0])));
    sign.position.set(p[0], y + 2.2, -p[1]);
    sign.castShadow = true;
    scene.add(sign);
  };

  const roadDir = (p: P) => nearestOnRoads(data.roads, p);
  for (const f of data.furniture) {
    const p: P = [f.x, f.y];
    const near = roadDir(p);
    const toRoad: P = near.d > 0 ? [(near.q[0] - p[0]) / near.d, (near.q[1] - p[1]) / near.d] : [1, 0];
    const upstream: P = [-near.u[0], -near.u[1]];
    if (f.kind === "fire_hydrant" && f.tags["fire_hydrant:type"] !== "wall") hydrant(p);
    else if (f.kind === "traffic_signals") signal(p, upstream);
    else if (f.kind === "traffic_sign" && (f.tags.traffic_sign ?? "").includes("stop")) stopSign(p, upstream);
    else if (f.kind === "stop" && !data.furniture.some((o) => o.kind === "traffic_sign" && Math.hypot(o.x - f.x, o.y - f.y) < 10)) stopSign(p, upstream);
    else if (f.kind === "vending_machine") payStation(p, near.u);
    else if (f.kind === "pole" || f.kind === "utility_pole") woodPole(p, toRoad, false);
    else if (f.kind === "street_lamp") streetLight(p, toRoad);
    else if (f.kind === "kerb" && f.tags.tactile_paving === "yes") {
      const c = data.crossings.reduce((a, b) => (Math.hypot(b.x - p[0], b.y - p[1]) < Math.hypot(a.x - p[0], a.y - p[1]) ? b : a));
      const n: P = [-c.u[1], c.u[0]];
      const sgn = Math.sign((p[0] - c.x) * n[0] + (p[1] - c.y) * n[1]) || 1;
      const q: P = [p[0] + n[0] * sgn * 0.35, p[1] + n[1] * sgn * 0.35];
      put("tactile", place(new THREE.PlaneGeometry(1.2, 0.6).rotateX(-Math.PI / 2).translate(0, 0.006, 0), q, ys(q), c.u));
    }
  }

  // Not in OSM: Howard's light poles, painted dark green with a red base and a pair of banners each
  // (Street View). Moss's poles and wires and the street trees are in streetDressing.ts.
  const banners = new Map<number, THREE.BufferGeometry[]>();
  howardLightSpots(data, garages).forEach(({ p, u, toRoad }, i) => {
    const y = ys(p), a = zdir(toRoad);
    put("banner", place(cyl(0.12, 0.08, 8.5), p, y));
    put("banner", place(box(0.07, 0.07, 2.2).translate(0, 8.3, 1.1), p, y, a));
    put("steel", place(box(0.32, 0.15, 0.7).translate(0, 8.25, 2.3), p, y, a));
    put("lamp", place(box(0.26, 0.02, 0.55).translate(0, 8.17, 2.3), p, y, a));
    put("base", place(cyl(0.2, 0.18, 0.9), p, y));
    for (const s of [-1, 1]) {   // banner arms across the sidewalk, banners face along Howard
      put("banner", place(box(0.04, 0.04, 0.95).translate(0, 5.9, s * 0.5), p, y, a));
      const g = new THREE.PlaneGeometry(0.8, 1.7).rotateY(Math.PI / 2).translate(0, 5.0, s * 0.55);
      place(g, p, y, a);
      const k = (i * 2 + (s > 0 ? 1 : 0)) % BANNER_KINDS;
      (banners.get(k) ?? banners.set(k, []).get(k)!).push(g);
    }
  });
  // Flex posts down the middle of the cycle track's buffer.
  for (const t of trackEdges(data)) {
    for (const { p } of along(offsetPts([t.a, t.b], t.side * TRACK_BUFFER / 2), 1, FLEX_POST_EVERY)) {
      if (data.crossings.some((c) => Math.hypot(c.x - p[0], c.y - p[1]) < 6)) continue;
      put("hydrant", place(cyl(0.05, 0.045, 0.9, 8), p, 0));
      put("unlit", place(cyl(0.1, 0.1, 0.04, 10), p, 0));
    }
  }
  for (const [k, gs] of banners) {
    const m = new THREE.Mesh(mergeGeometries(gs)!, new THREE.MeshStandardMaterial({ map: bannerTexture(k + 3), roughness: 0.8, side: THREE.DoubleSide }));
    m.castShadow = m.receiveShadow = true;
    scene.add(m);
  }

  for (const [k, gs] of parts) {
    const mesh = new THREE.Mesh(mergeGeometries(gs)!, M[k]);
    mesh.castShadow = k !== "tactile";
    mesh.receiveShadow = true;
    scene.add(mesh);
  }
}

let stopTex: THREE.CanvasTexture | null = null;
/** A STOP sign face for an octagon CircleGeometry(r, 8, PI/8): UVs map the circle to the full canvas. */
function stopTexture() {
  if (stopTex) return stopTex;
  const S = 256, cv = document.createElement("canvas");
  cv.width = cv.height = S;
  const ctx = cv.getContext("2d")!;
  const oct = (r: number) => {
    ctx.beginPath();
    for (let i = 0; i < 8; i++) {
      const a = Math.PI / 8 + (i * Math.PI) / 4;
      ctx.lineTo(S / 2 + Math.cos(a) * r, S / 2 - Math.sin(a) * r);
    }
    ctx.closePath();
  };
  ctx.fillStyle = "#f2f2f2"; oct(S / 2); ctx.fill();
  ctx.fillStyle = "#b3131b"; oct(S / 2 - 10); ctx.fill();
  ctx.fillStyle = "#f2f2f2";
  ctx.font = "bold 74px Helvetica, Arial, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("STOP", S / 2, S / 2 + 4);
  stopTex = new THREE.CanvasTexture(cv);
  stopTex.colorSpace = THREE.SRGBColorSpace;
  return stopTex;
}
