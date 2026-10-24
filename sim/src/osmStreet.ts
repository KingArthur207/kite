// Streets for the Moss world: textured asphalt with lane lines, zebras and stop lines as decals,
// concrete sidewalks on 15 cm curbs, driveway aprons in front of garages, tactile pads at the
// lowered kerbs OSM maps, OSM street furniture, and (not in OSM, added by us) cobra-head
// streetlights on Howard. Poles, wires, street trees, parked cars and the rest of the dressing are in
// streetDressing.ts. Nothing here collides; the walker's collision is the building faces and curbs.
import * as THREE from "three";
import { mergeGeometries, mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { Geo, pbr, seeded } from "./osmLook";
import { inPoly, inQuad, type Garage } from "./osmFacades";

type P = [number, number];
type Quad = P[];
interface Road { id: number; name: string; oneway: boolean; lanes: number; curbs: [number, number]; pts: P[] }
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

export const offsetPts = (pts: P[], o: number): P[] => pts.map((p, i) => {
  const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
  const L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
  return [p[0] - ((b[1] - a[1]) / L) * o, p[1] + ((b[0] - a[0]) / L) * o];
});

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
  const track = asphalt.clone(); track.color.set(0x9fb59f);   // Howard's cycle track, faint green
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
    bark: new THREE.MeshStandardMaterial({ color: 0x4d3e33, roughness: 0.95 }),
    leaves: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9, vertexColors: true }),
    soil: new THREE.MeshStandardMaterial({ color: 0x3b3027, roughness: 1, polygonOffset: true, polygonOffsetFactor: -2 }),
    tactile: new THREE.MeshStandardMaterial({ color: 0xd8ae25, roughness: 0.6, polygonOffset: true, polygonOffsetFactor: -2 }),
  };
  type MK = keyof typeof M;
  const parts = new Map<MK, THREE.BufferGeometry[]>();
  const put = (k: MK, g: THREE.BufferGeometry) => {
    if (k !== "leaves") g.deleteAttribute("uv");
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
  const free = (p: P, clear = 1.2) =>
    p[0] > BBOX.w && p[0] < BBOX.e && p[1] > BBOX.s && p[1] < BBOX.n &&
    !data.road_quads.some((q) => inQuad(p, q)) && !data.buildings.some((b) => inPoly(p, b.pts)) &&
    !data.crossings.some((c) => Math.hypot(c.x - p[0], c.y - p[1]) < 5) &&
    !garages.some((g) => Math.hypot(g.x - p[0], g.n - p[1]) < clear + 1.5);

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
  const tree = (p: P, id: number, height = 0) => {   // height: DataSF inventory estimate, 0 = typical
    const r = seeded(id), y = ys(p), sc = height > 0 ? height / 5.4 : 1;
    put("soil", place(new THREE.PlaneGeometry(1.1, 1.1).rotateX(-Math.PI / 2).translate(0, 0.004, 0), p, y));
    const h = 2.8 + r() * 0.8;
    put("bark", place(cyl(0.11, 0.07, h + 1.6, 8).scale(sc, sc, sc), p, y));
    const col = new THREE.Color().setHSL(0.22 + r() * 0.07, 0.3, 0.2 + r() * 0.07);
    for (let k = 0; k < 9; k++) {
      const ico = new THREE.IcosahedronGeometry(0.55 + r() * 0.5, 3);
      ico.deleteAttribute("normal"); ico.deleteAttribute("uv");
      const g = mergeVertices(ico);      // shared vertices: smooth normals, no facets
      g.setAttribute("uv", new THREE.Float32BufferAttribute(new Float32Array(g.getAttribute("position").count * 2), 2));
      const pos = g.getAttribute("position");
      for (let i = 0; i < pos.count; i++) {   // lumpy crown
        const s = 0.82 + 0.3 * Math.abs(Math.sin(pos.getX(i) * 3.1 + id) * Math.cos(pos.getZ(i) * 2.7 + k));
        pos.setXYZ(i, pos.getX(i) * s, pos.getY(i) * s * 0.85, pos.getZ(i) * s);
      }
      g.computeVertexNormals();
      const c = new Float32Array(pos.count * 3);
      for (let i = 0; i < pos.count; i++) {
        const shade = (0.7 + 0.35 * ((pos.getY(i) + 1.7) / 3.4)) * (0.75 + 0.5 * Math.abs((Math.sin(i * 12.9898 + k * 78.233) * 43758.5453) % 1));
        c.set([col.r * shade, col.g * shade, col.b * shade], i * 3);
      }
      g.setAttribute("color", new THREE.BufferAttribute(c, 3));
      const a = r() * Math.PI * 2, d = k === 0 ? 0 : 0.5 + r() * 0.7;
      put("leaves", place(g.translate(Math.cos(a) * d, h + 0.6 + r() * 1.6, Math.sin(a) * d).scale(sc, sc, sc), p, y));
    }
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
    else if (f.kind === "tree") tree(p, f.id, Number(f.tags.height_m ?? 0));
    else if (f.kind === "kerb" && f.tags.tactile_paving === "yes") {
      const c = data.crossings.reduce((a, b) => (Math.hypot(b.x - p[0], b.y - p[1]) < Math.hypot(a.x - p[0], a.y - p[1]) ? b : a));
      const n: P = [-c.u[1], c.u[0]];
      const sgn = Math.sign((p[0] - c.x) * n[0] + (p[1] - c.y) * n[1]) || 1;
      const q: P = [p[0] + n[0] * sgn * 0.35, p[1] + n[1] * sgn * 0.35];
      put("tactile", place(new THREE.PlaneGeometry(1.2, 0.6).rotateX(-Math.PI / 2).translate(0, 0.006, 0), q, ys(q), c.u));
    }
  }

  // Not in OSM: streetlights along both Howard curbs. Moss's poles and wires and the street trees
  // are in streetDressing.ts.
  const howard = data.roads.filter((r) => r.name === "Howard Street");
  for (const r of howard) {
    for (const side of [0, 1] as const) {
      const off = side ? r.curbs[1] + 0.5 : r.curbs[0] - 0.5;
      const toRoad = (u: P): P => (side ? [u[1], -u[0]] : [-u[1], u[0]]);
      for (const { p, u } of along(offsetPts(r.pts, off), 10 + side * 18, HOWARD_LIGHT_EVERY)) if (free(p)) streetLight(p, toRoad(u));
    }
  }

  for (const [k, gs] of parts) {
    const mesh = new THREE.Mesh(mergeGeometries(gs)!, M[k]);
    mesh.castShadow = k !== "soil" && k !== "tactile";
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
