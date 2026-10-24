// Street dressing for Moss St and the Howard corner, none of it in OSM: parked cars along Moss's NE
// curb with gaps at driveways, the hydrant and the corners; street trees in pits on Moss's SW
// sidewalk and both Howard sidewalks; wooden utility poles with sagging wires and service drops;
// no-parking and street-name signs; a hydrant, curbside carts, a litter can and a scooter;
// sidewalk joints, driveway cuts, stains, asphalt patches, tar seal and utility covers; and a
// brighter, Street View-like daylight. Static props return wall segments for the collision path in
// osmWorld.ts, and stay clear of the demo route and the sidewalk lines people walk in mossActors.ts.
import * as THREE from "three";
import { mergeGeometries, toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { loadCar, type CarName } from "./actors";
import { WALKER_RADIUS } from "./config";
import { pbr, seeded } from "./osmLook";
import { inPoly, inQuad, segDist, type Garage } from "./osmFacades";
import { along, nearestOnRoads, offsetPts, type StreetData } from "./osmStreet";
import {
  LeafSoup, blobTexture, cartGeo, coverTexture, hydrantGeo, leafTexture, limb, litterCanGeo, mulchTexture,
  noParkingTexture, scooterGeo, signPostGeo, streetNameTexture, streetTree, wireGeo,
} from "./streetProps";

type P = [number, number];
type Seg = [number, number, number, number];
type Data = StreetData & { route: P[] };

const CURB_H = 0.15;
const ROUTE_CLEAR = WALKER_RADIUS + 0.2;   // m: free gap kept between any prop and the demo route
const PARK_FROM_CURB = 1.0;                // m: parked car centre to curb face
const POLE_EVERY = 30, POLE_H = 10.5, ARM_H = 9.6, COMM_H = 7.2;
const AREA = { w: -60, e: 130, s: -135, n: 60 };   // where dressing goes: Moss and the Howard corner
const v3 = (p: P, y: number) => new THREE.Vector3(p[0], y, -p[1]);
const add = (a: P, b: P, k = 1): P => [a[0] + b[0] * k, a[1] + b[1] * k];
const dist = (a: P, b: P) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Arc length along a polyline of the point nearest p. */
function sOn(pts: P[], p: P) {
  let best = { d: Infinity, s: 0 }, s0 = 0;
  for (let i = 0; i + 1 < pts.length; i++) {
    const L = dist(pts[i], pts[i + 1]);
    const t = Math.max(0, Math.min(L, ((p[0] - pts[i][0]) * (pts[i + 1][0] - pts[i][0]) + (p[1] - pts[i][1]) * (pts[i + 1][1] - pts[i][1])) / (L || 1)));
    const d = segDist(p, pts[i], pts[i + 1]);
    if (d < best.d) best = { d, s: s0 + t };
    s0 += L;
  }
  return best.s;
}

/** Wall segments around an oriented rectangle: centre c, half sizes along u and across. */
function boxSegs(c: P, u: P, halfL: number, halfW: number): Seg[] {
  const n: P = [-u[1], u[0]];
  const k = [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => add(add(c, u, a * halfL), n, b * halfW));
  return k.map((p, i) => [...p, ...k[(i + 1) % 4]] as Seg);
}

/** Merged meshes per material, plus wall segments for collision. */
class Batch {
  geos = new Map<THREE.Material, THREE.BufferGeometry[]>();
  segs: Seg[] = [];
  put(m: THREE.Material, g: THREE.BufferGeometry | THREE.BufferGeometry[]) {
    const list = this.geos.get(m) ?? this.geos.set(m, []).get(m)!;
    for (const x of Array.isArray(g) ? g : [g]) list.push(x.index ? x.toNonIndexed() : x);
  }
  flush(group: THREE.Object3D, shadow: (m: THREE.Material) => boolean) {
    for (const [m, gs] of this.geos) {
      const keys = (g: THREE.BufferGeometry) => Object.keys(g.attributes).sort().join();
      const byLayout = new Map<string, THREE.BufferGeometry[]>();
      for (const g of gs) (byLayout.get(keys(g)) ?? byLayout.set(keys(g), []).get(keys(g))!).push(g);
      for (const list of byLayout.values()) {
        const mesh = new THREE.Mesh(mergeGeometries(list)!, m);
        mesh.castShadow = shadow(m);
        mesh.receiveShadow = true;
        group.add(mesh);
      }
    }
  }
}

/** Ground decals: quads with RGBA vertex colours over a soft texture, merged per texture. */
class Decals {
  pos: number[] = []; uv: number[] = []; col: number[] = [];
  constructor(private worldUV = false) {}
  /** Rectangle centred at c, `len` along u, `wid` across, at height y. */
  rect(c: P, u: P, len: number, wid: number, y: number, rgba: [number, number, number, number]) {
    const n: P = [-u[1], u[0]];
    const k = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => add(add(c, u, (a * len) / 2), n, (b * wid) / 2));
    for (const i of [0, 2, 1, 0, 3, 2]) {
      this.pos.push(k[i][0], y, -k[i][1]);
      if (this.worldUV) this.uv.push(k[i][0], k[i][1]);
      else this.uv.push([0, 1, 1, 0][i], [0, 0, 1, 1][i]);
      this.col.push(...rgba);
    }
  }
  mesh(map: THREE.Texture | null, opts: THREE.MeshStandardMaterialParameters = {}) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute("color", new THREE.Float32BufferAttribute(this.col, 4));
    g.computeVertexNormals();
    const m = new THREE.MeshStandardMaterial({
      map, vertexColors: true, transparent: true, depthWrite: false, roughness: 0.9,
      polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4, ...opts,
    });
    const mesh = new THREE.Mesh(g, m);
    mesh.receiveShadow = true;
    return mesh;
  }
}

/** Builds everything into `group` and returns the props' wall segments for collision. */
export function buildStreetDressing(group: THREE.Object3D, data: Data, garages: Garage[]): Seg[] {
  const root = new THREE.Group();
  root.name = "streetDressing";
  group.add(root);
  const scene = group.parent;
  if (scene instanceof THREE.Scene) brightDay(scene);

  const moss = data.roads.find((r) => r.name === "Moss Street")!;
  const howard = data.roads.filter((r) => r.name === "Howard Street" && r.pts.some((p) => Math.hypot(p[0], p[1]) < 80));
  const inArea = (p: P) => p[0] > AREA.w && p[0] < AREA.e && p[1] > AREA.s && p[1] < AREA.n;
  const onRoad = (p: P) => data.road_quads.some((q) => inQuad(p, q));
  const inBuilding = (p: P) => data.buildings.some((b) => inPoly(p, b.pts));
  const onSidewalk = (p: P) => !onRoad(p) && !inBuilding(p) && data.sidewalk_quads.some((q) => inQuad(p, q));
  const ys = (p: P) => (onRoad(p) ? 0 : CURB_H);

  // Lines people walk: the demo route, and the sidewalk lines in mossActors.ts.
  const [spawn, mossCorner, howardCorner, end] = data.route;
  const ext = (a: P, b: P, m: number): P => add(b, [(b[0] - a[0]) / dist(a, b), (b[1] - a[1]) / dist(a, b)], m);
  const mossDir: P = [(spawn[0] - mossCorner[0]) / dist(spawn, mossCorner), (spawn[1] - mossCorner[1]) / dist(spawn, mossCorner)];
  const walkLines: P[][] = [data.route, [howardCorner, add(howardCorner, mossDir, 30)], [ext(howardCorner, mossCorner, 35), end],
    [mossCorner, ext(mossCorner, spawn, 15)]];
  const lineDist = (p: P, line: P[]) => Math.min(...line.slice(1).map((b, i) => segDist(p, line[i], b)));
  const props: { p: P; r: number }[] = [];
  /** Is a prop of radius r at p out of everyone's way? */
  const clear = (p: P, r: number, opts: { garages?: boolean } = {}) =>
    inArea(p) && onSidewalk(p) &&
    lineDist(p, data.route) > r + ROUTE_CLEAR &&
    walkLines.slice(1).every((l) => lineDist(p, l) > r + 0.3) &&
    !data.crossings.some((c) => Math.hypot(c.x - p[0], c.y - p[1]) < 4.5) &&
    (opts.garages === false || !garages.some((g) => drivewayHit(g, p, r))) &&
    !props.some((o) => dist(o.p, p) < o.r + r + 0.4) &&
    !data.buildings.some((b) => b.pts.slice(1).some((q, i) => segDist(p, b.pts[i], q) < r + 0.05));
  /** Inside the strip between a garage door and the curb. */
  const drivewayHit = (g: Garage, p: P, r: number) => {
    const d: P = [p[0] - g.x, p[1] - g.n];
    const along = d[0] * g.u[0] + d[1] * g.u[1], out = d[0] * g.out[0] + d[1] * g.out[1];
    return Math.abs(along) < g.w / 2 + 0.6 + r && out > -0.5 && out < 7;
  };

  const batch = new Batch();
  const M = {
    bark: pbr("bark", 1.4, { roughness: 0.95 }),
    pole: pbr("polewood", 1.2, { roughness: 0.9, color: 0x9a8a78 }),
    leaves: new THREE.MeshStandardMaterial({ map: leafTexture(), alphaTest: 0.5, side: THREE.DoubleSide, vertexColors: true, roughness: 0.85 }),
    mulch: new THREE.MeshStandardMaterial({ map: mulchTexture(), roughness: 1, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3 }),
    plastic: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.05 }),
    metal: new THREE.MeshStandardMaterial({ color: 0x8c908e, roughness: 0.4, metalness: 0.7 }),
    porcelain: new THREE.MeshStandardMaterial({ color: 0x5e6b62, roughness: 0.3 }),
    wire: new THREE.MeshStandardMaterial({ color: 0x161616, roughness: 0.5 }),
    lamp: new THREE.MeshStandardMaterial({ color: 0xf4f1e2, emissive: 0x1a1810, roughness: 0.3 }),
  };
  M.mulch.map!.wrapS = M.mulch.map!.wrapT = THREE.RepeatWrapping;
  /** Geometry at (p, y) with local +x along u. */
  const place = (g: THREE.BufferGeometry, p: P, y: number, u: P = [1, 0]) =>
    g.applyMatrix4(new THREE.Matrix4().makeRotationY(Math.atan2(u[1], u[0])).setPosition(p[0], y, -p[1]));
  const mossU = (p: P) => nearestOnRoads([moss], p).u;
  const mossCurb = (side: 0 | 1, inset: number) => offsetPts(moss.pts, side ? moss.curbs[1] + inset : moss.curbs[0] - inset);

  // --- utility poles along Moss's SW curb: crossarm with three primaries, two comms cables lower
  // down, a cobra-head light, a transformer on every other pole, service drops to the houses.
  const poles: { p: P; u: P; y: number }[] = [];
  for (const { p, u } of along(mossCurb(0, 0.35), 8, POLE_EVERY)) {
    let q = p;
    for (const k of [0, 2, -2, 4, -4]) { q = add(p, u, k); if (clear(q, 0.2)) break; }
    if (!clear(q, 0.2)) continue;
    props.push({ p: q, r: 0.2 });
    poles.push({ p: q, u, y: ys(q) });
  }
  const wiresLine: number[] = [];
  const wire = (a: THREE.Vector3, b: THREE.Vector3, sag: number, r: number) => {
    const w = wireGeo(a, b, sag, r);
    batch.put(M.wire, w.mesh);
    w.line.slice(1).forEach((q, i) => wiresLine.push(w.line[i].x, w.line[i].y, w.line[i].z, q.x, q.y, q.z));
  };
  poles.forEach(({ p, u, y }, i) => {
    const across: P = [-u[1], u[0]];   // toward the road (NE)
    const rnd = seeded(i + 31);
    batch.put(M.pole, limb(v3(p, y - 0.05), v3(add(p, u, (rnd() - 0.5) * 0.15), y + POLE_H), 0.16, 0.12, 10));
    batch.put(M.pole, place(new THREE.BoxGeometry(0.1, 0.12, 2.4).translate(0, ARM_H, 0), p, y, u));
    for (const z of [-1, 0, 1]) batch.put(M.porcelain, place(new THREE.CylinderGeometry(0.035, 0.05, 0.2, 8).translate(0, ARM_H + 0.16, z), p, y, u));
    batch.put(M.metal, place(new THREE.BoxGeometry(1.8, 0.06, 0.06).translate(0.95, 7.62, 0), p, y, across));
    batch.put(M.metal, place(new THREE.BoxGeometry(0.6, 0.14, 0.3).translate(1.9, 7.55, 0), p, y, across));
    batch.put(M.lamp, place(new THREE.BoxGeometry(0.5, 0.02, 0.24).translate(1.9, 7.47, 0), p, y, across));
    if (i % 2 === 0) batch.put(M.metal, place(new THREE.CylinderGeometry(0.26, 0.26, 0.9, 14).translate(-0.4, 8.2, 0), p, y, across));
    batch.segs.push(...boxSegs(p, u, 0.17, 0.17));
    const next = poles[i + 1];
    if (next && dist(next.p, p) < POLE_EVERY * 2.2) {
      const span = dist(next.p, p);
      for (const z of [-1, 0, 1]) {
        const a = v3(add(p, across, z), y + ARM_H + 0.27), b = v3(add(next.p, [-next.u[1], next.u[0]], z), next.y + ARM_H + 0.27);
        wire(a, b, 0.02 * span, 0.011);
      }
      for (const [h, side] of [[COMM_H, 0.17], [COMM_H - 0.45, 0.17]] as const) {
        wire(v3(add(p, across, side), y + h), v3(add(next.p, [-next.u[1], next.u[0]], side), next.y + h), 0.03 * span, 0.018);
      }
    }
    // Service drops to the nearest facade across the street and behind the pole.
    for (const dir of [1, -1]) {
      const probe = add(p, across, dir * (dir > 0 ? 9 : 3));
      let best: { q: P; d: number } | null = null;
      for (const b of data.buildings) for (let k = 0; k + 1 < b.pts.length; k++) {
        const [a0, a1] = [b.pts[k], b.pts[k + 1]];
        const L = dist(a0, a1);
        if (L < 2) continue;
        const t = Math.max(0.1, Math.min(0.9, ((probe[0] - a0[0]) * (a1[0] - a0[0]) + (probe[1] - a0[1]) * (a1[1] - a0[1])) / (L * L)));
        const q: P = [a0[0] + (a1[0] - a0[0]) * t, a0[1] + (a1[1] - a0[1]) * t];
        const d = dist(q, probe);
        if (d < (best?.d ?? 6)) best = { q, d };
      }
      if (best) wire(v3(add(p, across, 0.17 * dir), y + COMM_H - 0.2), v3(best.q, 5.6), 0.35, 0.012);
    }
  });

  // --- street trees in pits: Moss SW sidewalk, both Howard sidewalks.
  const leaves = new LeafSoup();
  const tree = (p: P, seed: number, h: number) => {
    const rnd = seeded(seed);
    const y = ys(p), u = mossU(p);
    batch.put(M.bark, streetTree(v3(p, y), h, seed, leaves));
    batch.put(M.mulch, place(new THREE.PlaneGeometry(1.0, 1.0).rotateX(-Math.PI / 2).translate(0, 0.006, 0), p, y, u));
    for (const s of [-1, 1]) for (const o of [0, 1]) {   // thin steel edge around the basin
      const e = o ? new THREE.BoxGeometry(1.04, 0.02, 0.03).translate(0, 0.01, s * 0.51) : new THREE.BoxGeometry(0.03, 0.02, 1.04).translate(s * 0.51, 0.01, 0);
      batch.put(M.metal, place(e, p, y, u));
    }
    batch.segs.push(...boxSegs(p, u, 0.2, 0.2));
    return rnd;
  };
  let seed = 100;
  // DataSF's street tree inventory (moss.json furniture) puts real trees at their surveyed spots in osmStreet.ts;
  // only invent trees when that inventory isn't there.
  const inventoryTrees = data.furniture.some((f) => f.kind === "tree");
  if (!inventoryTrees) for (const { p } of along(mossCurb(0, 0.65), 5, 8.5)) {
    const rnd = seeded(seed++);
    if (rnd() < 0.3 || !clear(p, 0.5) || poles.some((o) => dist(o.p, p) < 3)) continue;
    props.push({ p, r: 0.5 });
    tree(p, seed * 13, 6 + rnd() * 1.5);
  }
  if (!inventoryTrees) for (const r of howard) for (const side of [0, 1] as const) {
    for (const { p } of along(offsetPts(r.pts, side ? r.curbs[1] + 0.85 : r.curbs[0] - 0.85), 4 + side * 5, 10)) {
      const rnd = seeded(seed++);
      if (!clear(p, 0.5) || data.furniture.some((f) => Math.hypot(f.x - p[0], f.y - p[1]) < 2.5)) continue;
      props.push({ p, r: 0.5 });
      tree(p, seed * 17, 7.5 + rnd() * 2);
    }
  }

  // --- signs, hydrant, carts, litter can, scooter.
  /** First spot along a curb line (from s0, both ways) where a prop of radius r fits. */
  const spot = (line: P[], s0: number, r: number, opts?: { garages?: boolean }) => {
    for (let k = 0; k < 40; k++) {
      const s = s0 + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 1.5;
      const hit = [...along(line, Math.max(0, s), 1e9)][0];
      if (hit && clear(hit.p, r, opts)) return hit;
    }
    return null;
  };
  const signAt = (p: P, face: P, tex: THREE.Texture, w: number, h: number, top: number) => {
    const y = ys(p);
    batch.put(M.metal, signPostGeo().map((g) => place(g, p, y)));
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.45, side: THREE.DoubleSide }));
    m.position.set(p[0], y + top - h / 2, -p[1]);
    m.rotation.y = Math.atan2(face[0], face[1]);   // plane normal (+z) toward face
    m.castShadow = true;
    root.add(m);
    batch.segs.push(...boxSegs(p, [1, 0], 0.05, 0.05));
    props.push({ p, r: 0.1 });
  };
  const mossLen = moss.pts.slice(1).reduce((s, b, i) => s + dist(moss.pts[i], b), 0);
  const zebraS = sOn(moss.pts, [0, 0]);
  for (const s of [zebraS + 12, zebraS + 70]) {
    const at = spot(mossCurb(0, 0.3), s, 0.1);
    if (at) signAt(at.p, [-at.u[0], -at.u[1]], noParkingTexture(), 0.32, 0.32, 2.6);
  }
  {
    const at = spot(mossCurb(0, 0.35), zebraS + 4, 0.1);
    if (at) {
      const nameMat = (t: THREE.Texture) => new THREE.MeshStandardMaterial({ map: t, roughness: 0.4, side: THREE.DoubleSide });
      signAt(at.p, [-at.u[1], at.u[0]], noParkingTexture(), 0.001, 0.001, 2.0);   // the post
      const y = ys(at.p);
      for (const [name, block, dir, up] of [["Moss St", "300", at.u, 2.85], ["Howard St", "1100", [at.u[1], -at.u[0]] as P, 3.05]] as const) {
        const sign = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.2), nameMat(streetNameTexture(name, block)));
        sign.geometry.attributes.uv.array.set([0, 1, 1, 1, 0, 0.75, 1, 0.75]);
        sign.position.set(at.p[0], y + up, -at.p[1]);
        sign.rotation.y = Math.atan2(-dir[1], dir[0]) + Math.PI / 2;
        sign.castShadow = true;
        root.add(sign);
      }
    }
  }
  const hydrantAt = spot(mossCurb(1, 0.35), zebraS + 48, 0.22) ?? spot(mossCurb(0, 0.35), zebraS + 48, 0.22);
  if (hydrantAt) {
    batch.put(M.plastic, hydrantGeo().map((g) => place(g, hydrantAt.p, ys(hydrantAt.p), hydrantAt.u)));
    batch.segs.push(...boxSegs(hydrantAt.p, hydrantAt.u, 0.2, 0.2));
    props.push({ p: hydrantAt.p, r: 0.22 });
  }
  // Curbside carts beside two garage doors on Moss's SW side, against the wall.
  const swGarages = garages.filter((g) => nearestOnRoads([moss], [g.x, g.n]).side < 0 && nearestOnRoads([moss], [g.x, g.n]).d < 8 && inArea([g.x, g.n]));
  let placedCarts = 0;
  for (const g of swGarages) {
    if (placedCarts >= 2) break;
    const base = add(add([g.x, g.n], g.out, 0.45), g.u, g.w / 2 + 0.95);
    const spots = [0, 0.7, 1.4].map((k) => add(base, g.u, k));
    if (!spots.every((p) => clear(p, 0.36))) continue;
    [0x222326, 0x1f4f9c, 0x2f7a3a].forEach((hex, k) => {
      batch.put(M.plastic, cartGeo(hex).map((x) => place(x, spots[k], ys(spots[k]), [-g.out[1], g.out[0]])));
      batch.segs.push(...boxSegs(spots[k], g.u, 0.33, 0.38));
      props.push({ p: spots[k], r: 0.36 });
    });
    placedCarts++;
  }
  if (howard.length) {
    const h = howard.find((r) => nearestOnRoads([r], end).d < 15) ?? howard[0];
    const can = spot(offsetPts(h.pts, h.curbs[1] + 0.5), sOn(h.pts, mossCorner) + 6, 0.32);
    if (can) {
      batch.put(M.plastic, litterCanGeo().map((g) => place(g, can.p, ys(can.p), [can.u[1], -can.u[0]])));
      batch.segs.push(...boxSegs(can.p, can.u, 0.3, 0.3));
      props.push({ p: can.p, r: 0.32 });
    }
  }
  const scooter = spot(mossCurb(0, 0.4), zebraS + 30, 0.55);
  if (scooter) {
    batch.put(M.plastic, scooterGeo(0x3aa655).map((g) => place(g.rotateX(0.12), scooter.p, ys(scooter.p), scooter.u)));
    batch.segs.push(...boxSegs(scooter.p, scooter.u, 0.55, 0.15));
    props.push({ p: scooter.p, r: 0.55 });
  }

  batch.flush(root, (m) => m !== M.mulch);
  const leafMesh = new THREE.Mesh(leaves.build(), M.leaves);
  leafMesh.castShadow = leafMesh.receiveShadow = true;
  root.add(leafMesh);
  const wg = new THREE.BufferGeometry();
  wg.setAttribute("position", new THREE.Float32BufferAttribute(wiresLine, 3));
  root.add(new THREE.LineSegments(wg, new THREE.LineBasicMaterial({ color: 0x202020 })));

  groundDetail(root, data, garages, moss, howard, { onRoad, onSidewalk, inArea });
  parkCars(root, data, garages, moss, [...props.map((o) => o.p)], hydrantAt?.p ?? null, batch.segs);
  return batch.segs;
}

// --- parked cars: NE curb of Moss, nose SE (Moss is one-way SE), gaps at driveways and corners.
const PARKED: CarName[] = ["sedan", "suv", "sedan2", "sedan", "suv", "sedan2", "sedan"];
const PAINT = [0xb9bcc0, 0xe9e8e4, 0x15171a, 0x4a4f56, 0x1d2a45, 0x6b1a1e, 0xa9a08c, 0x8d9499, 0xdcdcd8, 0x2a3a2f, 0x2b2d31];

function parkCars(root: THREE.Object3D, data: Data, garages: Garage[], moss: StreetData["roads"][number], _props: P[], hydrant: P | null, segs: Seg[]) {
  const line = offsetPts(moss.pts, moss.curbs[1] - PARK_FROM_CURB);
  const len = line.slice(1).reduce((s, b, i) => s + dist(line[i], b), 0);
  const blocked: [number, number][] = [];
  for (const c of data.crossings) if (c.road === moss.id) { const s = sOn(line, [c.x, c.y]); blocked.push([s - 9, s + 7]); }
  for (const g of garages) {
    const near = nearestOnRoads([moss], [g.x, g.n]);
    if (near.side <= 0 || near.road !== moss || Math.abs(near.side) - moss.curbs[1] > 8) continue;
    const s = sOn(line, [g.x, g.n]);
    blocked.push([s - g.w / 2 - 0.9, s + g.w / 2 + 0.9]);
  }
  if (hydrant) { const s = sOn(line, hydrant); blocked.push([s - 3, s + 3]); }
  const rnd = seeded(2024);
  const slots: { s: number; model: CarName; len: number }[] = [];
  const LEN: Record<string, number> = { sedan: 4.6, sedan2: 4.3, suv: 4.8 };
  let s = 0, k = 0;
  while (s < len) {
    const model = PARKED[k % PARKED.length], L = LEN[model];
    const hit = blocked.find(([a, b]) => s < b && s + L > a);
    if (hit) { s = hit[1] + 0.3; continue; }
    if (rnd() < 0.12) { s += 5; continue; }       // an empty space now and then
    slots.push({ s: s + L / 2, model, len: L });
    s += L + 0.7 + rnd() * 1.4;
    k++;
  }
  const at = (sv: number) => {
    let s0 = 0;
    for (let i = 0; i + 1 < line.length; i++) {
      const L = dist(line[i], line[i + 1]);
      if (sv <= s0 + L) { const u: P = [(line[i + 1][0] - line[i][0]) / L, (line[i + 1][1] - line[i][1]) / L]; return { p: add(line[i], u, sv - s0), u }; }
      s0 += L;
    }
    return null;
  };
  const shadows = new Decals();
  slots.forEach((slot, i) => {
    const pose = at(slot.s);
    if (!pose) return;
    const r2 = seeded(i * 7 + 3);
    const yaw = (r2() - 0.5) * 0.03;
    const u: P = [pose.u[0] * Math.cos(yaw) - pose.u[1] * Math.sin(yaw), pose.u[0] * Math.sin(yaw) + pose.u[1] * Math.cos(yaw)];
    const p = add(pose.p, [-u[1], u[0]], (r2() - 0.5) * 0.2);
    segs.push(...boxSegs(p, u, slot.len / 2, 0.92));
    shadows.rect(add(p, u, slot.len * 0.25), u, 1.0, 0.8, 0.008, [0.12, 0.11, 0.1, 0.35]);   // oil drip under the engine
    const paint = PAINT[Math.floor(r2() * PAINT.length)];
    loadCar(slot.model).then(({ obj }) => {
      dressCar(obj, paint);
      obj.position.set(p[0], 0, -p[1]);
      obj.rotation.y = Math.atan2(u[0], -u[1]);
      obj.name = `parked_${slot.model}`;
      root.add(obj);
    }).catch((e) => console.error(`[streetDressing] parked ${slot.model}`, e));
  });
  root.add(shadows.mesh(blobTexture()));
  console.info(`[streetDressing] ${slots.length} parked cars along Moss`);
}

// Low-poly bodies shade flat; creased normals round the panels and keep the hard edges.
const CREASE = THREE.MathUtils.degToRad(40);
const creasedCache = new WeakMap<THREE.BufferGeometry, THREE.BufferGeometry>();
const creased = (g: THREE.BufferGeometry) => creasedCache.get(g) ?? creasedCache.set(g, toCreasedNormals(g, CREASE)).get(g)!;

/** New materials for a parked car (the moving cars share the loaded ones): clear-coated paint,
 *  dark glass, rubber, satin trim. */
function dressCar(obj: THREE.Object3D, paint: number) {
  const metallic = paint !== 0xe9e8e4 && paint !== 0xdcdcd8;
  const body = new THREE.MeshPhysicalMaterial({ color: paint, metalness: metallic ? 0.55 : 0.1, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.08 });
  const glass = new THREE.MeshPhysicalMaterial({ color: 0x0b0e12, metalness: 0.2, roughness: 0.04, clearcoat: 1 });
  const rubber = new THREE.MeshStandardMaterial({ color: 0x141414, roughness: 0.9 });
  const trim = new THREE.MeshStandardMaterial({ color: 0x1e1f21, roughness: 0.6 });
  const chrome = new THREE.MeshStandardMaterial({ color: 0xc8c8c8, metalness: 1, roughness: 0.2 });
  const headlight = new THREE.MeshPhysicalMaterial({ color: 0xd8d6cf, metalness: 0.6, roughness: 0.1, clearcoat: 1 });
  const swap = (m: THREE.Material) => {
    const name = m.name;
    if (/window/i.test(name)) return glass;
    if (/^black$/i.test(name)) return rubber;
    if (/^grey$/i.test(name)) return trim;
    if (/material\.007/i.test(name)) return chrome;
    if (/head/i.test(name)) return headlight;
    if (/tail|brake/i.test(name)) {
      const c = (m as THREE.MeshStandardMaterial).clone();
      c.roughness = 0.15;
      return c;
    }
    return body;
  };
  obj.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    if ((mesh.material as THREE.MeshBasicMaterial).transparent) return;   // the blob shadow
    mesh.material = Array.isArray(mesh.material) ? mesh.material.map(swap) : swap(mesh.material);
    mesh.geometry = creased(mesh.geometry);
    mesh.castShadow = mesh.receiveShadow = true;
  });
}

// --- sidewalk joints, driveway cuts, stains; asphalt patches, tar seal, covers.
function groundDetail(root: THREE.Object3D, data: Data, garages: Garage[], moss: StreetData["roads"][number], howard: StreetData["roads"],
  f: { onRoad: (p: P) => boolean; onSidewalk: (p: P) => boolean; inArea: (p: P) => boolean }) {
  const rnd = seeded(77);
  const joints = new Decals(), stains = new Decals(), patches = new Decals(true), tar = new Decals(), covers = new Decals();
  const JOINT: [number, number, number, number] = [0.32, 0.31, 0.29, 0.55];
  for (const q of data.sidewalk_quads) {
    const L = dist(q[0], q[1]), W = dist(q[1], q[2]);
    if (L < 0.5 || !f.inArea(q[0])) continue;
    const u: P = [(q[1][0] - q[0][0]) / L, (q[1][1] - q[0][1]) / L];
    const n: P = [(q[3][0] - q[0][0]) / W, (q[3][1] - q[0][1]) / W];
    for (let t = 1.5; t < L; t += 1.5) joints.rect(add(add(q[0], u, t), n, W / 2), n, W, 0.03, CURB_H + 0.003, JOINT);
    for (let k = 0; k < L / 2.2; k++) {
      const c = add(add(q[0], u, rnd() * L), n, rnd() * W);
      if (!f.onSidewalk(c)) continue;
      const s = 0.25 + rnd() * rnd() * 1.4;
      stains.rect(c, [Math.cos(rnd() * 6.3), Math.sin(rnd() * 6.3)], s, s * (0.6 + rnd() * 0.5), CURB_H + 0.004,
        [0.18 + rnd() * 0.08, 0.17 + rnd() * 0.06, 0.15, 0.25 + rnd() * 0.35]);
    }
  }
  for (const g of garages) {
    if (!f.inArea([g.x, g.n])) continue;
    let depth = 0;
    while (depth < 7 && !f.onRoad(add([g.x, g.n], g.out, depth + 0.2))) depth += 0.2;
    if (depth < 0.5 || depth >= 7) continue;
    const c = add([g.x, g.n], g.out, depth / 2), w = g.w + 1.2;
    for (const o of [-w / 2, w / 2]) joints.rect(add(c, g.u, o), g.out, depth, 0.035, CURB_H + 0.003, JOINT);
    for (const o of [-0.75, 0.75]) stains.rect(add(c, g.u, o), g.out, depth, 0.45, CURB_H + 0.004, [0.2, 0.19, 0.17, 0.35]);
    stains.rect(c, g.out, depth, w, CURB_H + 0.0035, [0.95, 0.94, 0.9, 0.35]);
  }
  // Asphalt: utility-cut patches and a long trench on Moss, patches on Howard, tar-sealed cracks,
  // oil down the middle of each lane, covers.
  const roadPatch = (c: P, u: P, len: number, wid: number) => {
    patches.rect(c, u, len, wid, 0.004, [0.55, 0.55, 0.56, 1]);
    for (const [o, l, w] of [[wid / 2, len, 0.05], [-wid / 2, len, 0.05]] as const) tar.rect(add(c, [-u[1], u[0]], o), u, l, w, 0.006, [0.05, 0.05, 0.05, 0.7]);
    for (const o of [len / 2, -len / 2]) tar.rect(add(c, u, o), [-u[1], u[0]], wid, 0.05, 0.006, [0.05, 0.05, 0.05, 0.7]);
  };
  const roads = [moss, ...howard];
  for (const r of roads) {
    const width = r.curbs[1] - r.curbs[0];
    for (const { p, u } of along(r.pts, 6, r === moss ? 14 : 18)) {
      if (!f.inArea(p) || data.crossings.some((c) => Math.hypot(c.x - p[0], c.y - p[1]) < 6)) continue;
      const off = r.curbs[0] + 0.8 + rnd() * (width - 1.6);
      const c = add(p, [-u[1], u[0]], off);
      if (rnd() < 0.55) roadPatch(c, u, 1.2 + rnd() * 2.5, 0.9 + rnd() * 1.4);
      let q = add(p, [-u[1], u[0]], r.curbs[0] + 0.5 + rnd() * (width - 1)), dir = rnd() * 6.3;
      for (let k = 0; k < 8 + rnd() * 10; k++) {          // a wandering crack, tar-sealed
        dir += (rnd() - 0.5) * 0.9;
        const d: P = [Math.cos(dir), Math.sin(dir)], nq = add(q, d, 0.5);
        if (!f.onRoad(nq)) break;
        tar.rect(add(q, d, 0.25), d, 0.55, 0.04 + rnd() * 0.03, 0.005, [0.04, 0.04, 0.04, 0.75]);
        q = nq;
      }
    }
  }
  for (const l of data.lanes) {
    if (!roads.some((r) => r.id === l.road)) continue;
    for (const { p, u } of along(l.pts, 0, 1.6)) {
      if (!f.inArea(p) || !f.onRoad(p)) continue;
      stains.rect(p, u, 1.8, 0.7 + rnd() * 0.4, 0.003, [0.1, 0.1, 0.1, 0.18 + rnd() * 0.12]);
    }
  }
  {
    const u0 = nearestOnRoads([moss], [20, -20]).u;
    const pts = offsetPts(moss.pts, -1.2);
    const s0 = sOn(pts, [12, -12]);
    for (const { p } of along(pts, s0, 4)) if (sOn(pts, p) < s0 + 28) patches.rect(p, u0, 4.05, 0.8, 0.0045, [0.5, 0.5, 0.51, 1]);
  }
  for (const s of [30, 75, 110]) {
    const hit = [...along(moss.pts, s, 1e9)][0];
    if (hit) covers.rect(add(hit.p, [-hit.u[1], hit.u[0]], 0.4), hit.u, 0.75, 0.75, 0.007, [1, 1, 1, 1]);
  }
  for (const r of howard) for (const { p, u } of along(r.pts, 20, 45)) if (f.inArea(p)) covers.rect(add(p, [-u[1], u[0]], -2), u, 0.75, 0.75, 0.007, [1, 1, 1, 1]);

  const asphalt = pbr("asphalt", 3.0, { rough: true });
  root.add(joints.mesh(null, { roughness: 1 }), stains.mesh(blobTexture()), tar.mesh(null, { roughness: 0.35 }));
  root.add(patches.mesh(asphalt.map, { normalMap: asphalt.normalMap, transparent: false, depthWrite: true, roughness: 0.95 }));
  root.add(covers.mesh(coverTexture(), { metalness: 0.6, roughness: 0.55, alphaTest: 0.5, transparent: false }));
}

/** Bright, clear SF afternoon like Street View: stronger, whiter sun and a cooler sky fill. */
function brightDay(scene: THREE.Scene) {
  scene.traverse((o) => {
    if (!(o instanceof THREE.DirectionalLight)) return;
    o.intensity = 3.0;
    o.color.set(0xfff5e8);
    o.shadow.radius = 3;
  });
  const fill = new THREE.HemisphereLight(0xcfe0ff, 0x8a8478, 0.35);
  scene.add(fill);
  fill.castShadow = false;   // osmLook's childadded hook turns shadows on for everything added
  scene.environmentIntensity = 1.0;
  if (scene.fog instanceof THREE.Fog) { scene.fog.color.set(0xd4dde6); scene.fog.near = 120; }
}
