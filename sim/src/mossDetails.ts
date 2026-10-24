// ?look=2 only: street-level details on Moss St and its Howard corner that Street View (2026-09, walked with
// ?compare=1) shows and the dressing in streetDressing.ts doesn't: the second brick medallion mid-block, paper
// lanterns hanging over the street (instead of the bulb strings), grey brick pavers across Moss's last 14 m before
// Howard, two manholes, a red curb at the SW corner, the SW side's NO PARKING and 2-hour/street-cleaning signs,
// the fire department connection on 33 Moss, and the climbing shrubs against the NE houses. Everything is drawn here;
// positions are Street View estimates in the moss.json frame (x east, y north, metres from the zebra).
// Also mossLaneClear: Moss's traffic lane moved toward the SW where cars park on the NE curb (streetDressing parkCars).
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { WALKER_RADIUS } from "./config";
import { Geo, seeded } from "./osmLook";
import { segDist } from "./osmFacades";
import { along, curbOffsets, offsetPts, type StreetData } from "./osmStreet";
import { boxSegs, PARK_SWAP_AT, sOn } from "./streetDressing";
import { canvasTex, coverTexture, LeafSoup, leafTexture, limb, medallionTexture, signPostGeo, wireGeo } from "./streetProps";

type P = [number, number];
type Seg = [number, number, number, number];
type Data = StreetData & { route: P[]; buildings: { id: number; pts: P[]; ground?: P[] }[] };

const CURB_H = 0.15;
const MEDALLION_R = 1.9, MEDALLION2_AHEAD = 28;          // m: the second medallion, NW of the spawn (the first is at 6)
const LANTERN_WIRES = 4, LANTERN_FROM = 20, LANTERN_STEP = 5, LANTERN_H = 6.4, LANTERN_EVERY = 1.5;   // m, from the zebra
const PAVERS_FROM = 1.6, PAVERS_TO = 15;                  // m SE of the zebra: the grey brick paver patch
const RED_CURB: [number, number] = [2, 7];                // m SE of the zebra on the SW curb
const MANHOLES: P[] = [[22.8, -24.9], [8.3, -8.3]];
const MANHOLE_R = 0.45;
const NO_PARKING_SW: P = [23.5, -25.5];                   // NO PARKING ANY TIME, SW sidewalk, facing SE
const REG_SIGNS_AHEAD = 16;                               // m NW of the spawn, SW sidewalk: 2-hour and street cleaning
const FDC = { building: 260351180, near: [45.4, -36.7] as P };   // 33 Moss: red-capped fire department connection
const SHRUBS: { near: P; len: number; h: number }[] = [{ near: [35.5, -30], len: 3.2, h: 2.8 }, { near: [32, -25.5], len: 2.2, h: 3.2 }];
const LANE_SHIFT = 1.2, LANE_TAPER = 8;                   // m toward the SW, and over how far it eases back
const v3 = (p: P, y: number) => new THREE.Vector3(p[0], y, -p[1]);
const add = (a: P, b: P, k = 1): P => [a[0] + b[0] * k, a[1] + b[1] * k];
const dist = (a: P, b: P) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const at = (line: P[], s: number) => [...along(line, Math.max(0, s), 1e9)][0] ?? null;

/** Moss's traffic lane with its NW half moved LANE_SHIFT m toward the SW, clear of the NE parked cars. */
export function mossLaneClear<L extends { name: string; pts: P[] }>(lanes: L[]): L[] {
  return lanes.map((l) => {
    if (l.name !== "Moss Street") return l;
    const swap = sOn(l.pts, PARK_SWAP_AT);
    let s = 0;
    const off = l.pts.map((p, i) => {
      if (i) s += dist(l.pts[i - 1], p);
      return -LANE_SHIFT * Math.min(1, Math.max(0, (swap + LANE_TAPER - s) / LANE_TAPER));   // + is left of travel (NE)
    });
    return { ...l, pts: offsetPts(l.pts, off) };
  });
}

// --- textures
const paverTexture = () => canvasTex(512, (g, S) => {   // 1 m square: grey concrete pavers, 20 x 10 cm, running bond
  const r = seeded(41), bw = S / 5, bh = S / 10;
  g.fillStyle = "#4a4946"; g.fillRect(0, 0, S, S);
  for (let row = 0; row < 10; row++) for (let k = -1; k < 5; k++) {
    const x = k * bw + (row % 2) * bw / 2, l = 52 + r() * 16;
    g.fillStyle = `hsl(40, ${3 + r() * 4}%, ${l}%)`;
    g.fillRect(x + 2, row * bh + 2, bw - 4, bh - 4);
  }
  for (let i = 0; i < 1500; i++) { g.fillStyle = `rgba(30,28,25,${r() * 0.15})`; g.fillRect(r() * S, r() * S, 3, 3); }
});
const regSignTexture = (kind: "2hr" | "sweep") => canvasTex(256, (g, S) => {
  const c = kind === "2hr" ? "#1d6b3a" : "#c01d22";
  g.fillStyle = "#f4f3ee"; g.fillRect(0, 0, S, S);
  g.strokeStyle = c; g.lineWidth = 8; g.strokeRect(8, 8, S - 16, S - 16);
  g.fillStyle = c; g.textAlign = "center";
  const lines = kind === "2hr" ? [["2 HOUR", 44], ["PARKING", 36], ["8AM - 6PM", 26], ["MON THRU SAT", 22], ["EXCEPT AREA Q", 20], ["PERMITS", 20]]
    : [["NO PARKING", 34], ["12 NOON - 2 PM", 24], ["2ND & 4TH TUES", 22], ["STREET", 30], ["CLEANING", 30]];
  let y = 46;
  for (const [t, size] of lines as [string, number][]) { g.font = `bold ${size}px Helvetica, Arial`; g.fillText(t, S / 2, y); y += size + 10; }
});
const noParkingArrowTexture = () => canvasTex(256, (g, S) => {   // NO PARKING ANY TIME with the two-way arrow
  g.fillStyle = "#f4f3ee"; g.fillRect(0, 0, S, S);
  g.fillStyle = "#c01d22"; g.textAlign = "center";
  g.font = "bold 58px Helvetica, Arial"; g.fillText("NO", S / 2, 62);
  g.font = "bold 44px Helvetica, Arial"; g.fillText("PARKING", S / 2, 116);
  g.font = "bold 34px Helvetica, Arial"; g.fillText("ANY", S / 2, 160); g.fillText("TIME", S / 2, 198);
  g.fillRect(50, 222, 156, 8);
  for (const [x, d] of [[44, 1], [212, -1]]) { g.beginPath(); g.moveTo(x, 226); g.lineTo(x + d * 22, 212); g.lineTo(x + d * 22, 240); g.fill(); }
});

/** Builds the details into `group` (look 2 only; the caller checks) and returns wall segments for collision. */
export function buildMossDetails(group: THREE.Object3D, data: Data): Seg[] {
  const root = new THREE.Group();
  root.name = "mossDetails";
  group.add(root);
  const segs: Seg[] = [];
  const moss = data.roads.find((r) => r.name === "Moss Street");
  if (!moss) { console.error("[mossDetails] no Moss Street in moss.json; skipped"); return segs; }
  const zebraS = sOn(moss.pts, [0, 0]), spawnS = sOn(moss.pts, data.route[0]);
  const routeDist = (p: P) => Math.min(...data.route.slice(1).map((b, i) => segDist(p, data.route[i], b)));
  const curb = (side: 0 | 1, inset: number) => offsetPts(moss.pts, curbOffsets(moss, side, inset));
  const mid = offsetPts(moss.pts, moss.pts.map((_, i) => (curbOffsets(moss, 0, 0)[i] + curbOffsets(moss, 1, 0)[i]) / 2));
  const ground = (m: THREE.Material, g: THREE.BufferGeometry) => {
    const mesh = new THREE.Mesh(g, m);
    mesh.receiveShadow = true;
    root.add(mesh);
  };
  const solid = (m: THREE.Material, gs: THREE.BufferGeometry[]) => {
    if (!gs.length) return;
    const mesh = new THREE.Mesh(mergeGeometries(gs.map((g) => (g.index ? g.toNonIndexed() : g)))!, m);
    mesh.castShadow = mesh.receiveShadow = true;
    root.add(mesh);
  };
  const decal = (o: THREE.MeshStandardMaterialParameters) =>
    new THREE.MeshStandardMaterial({ roughness: 0.9, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3, ...o });

  // --- the second medallion, centred in the roadway (the first is streetDressing's, 6 m NW of the spawn)
  const med = at(mid, spawnS - MEDALLION2_AHEAD);
  if (med) ground(decal({ map: medallionTexture() }), new THREE.CircleGeometry(MEDALLION_R, 64).rotateX(-Math.PI / 2).translate(med.p[0], 0.006, -med.p[1]));

  // --- grey pavers curb to curb from the zebra to PAVERS_TO, manholes, red curb
  {
    const geo = new Geo(), sw = curb(0, 0), ne = curb(1, 0), up = new THREE.Vector3(0, 1, 0);
    for (let s = zebraS + PAVERS_FROM; s < zebraS + PAVERS_TO; s += 0.5) {
      const [a, b, c, d] = [at(sw, s), at(sw, s + 0.5), at(ne, s + 0.5), at(ne, s)];
      if (!a || !b || !c || !d) continue;
      const q = [a.p, b.p, c.p, d.p];
      geo.quad(q.map((p) => v3(p, 0.0075)), up, q.map((p) => [p[0], p[1]] as [number, number]));
    }
    const map = paverTexture();
    map.wrapS = map.wrapT = THREE.RepeatWrapping;
    if (!geo.empty) ground(decal({ map }), geo.build());
  }
  const cover = decal({ map: coverTexture(), metalness: 0.6, roughness: 0.55, polygonOffsetFactor: -5, polygonOffsetUnits: -5 });
  ground(cover, mergeGeometries(MANHOLES.map((p) => new THREE.CircleGeometry(MANHOLE_R, 24).rotateX(-Math.PI / 2).translate(p[0], 0.009, -p[1])))!);
  {
    const red: THREE.BufferGeometry[] = [], line = curb(0, 0);
    for (let s = zebraS + RED_CURB[0]; s < zebraS + RED_CURB[1]; s += 0.5) {
      const a = at(line, s);
      if (a) red.push(new THREE.BoxGeometry(0.52, CURB_H + 0.01, 0.2).applyMatrix4(
        new THREE.Matrix4().makeRotationY(Math.atan2(a.u[1], a.u[0])).setPosition(a.p[0] + a.u[0] * 0.25, CURB_H / 2, -(a.p[1] + a.u[1] * 0.25))));
    }
    solid(new THREE.MeshStandardMaterial({ color: 0xb3262a, roughness: 0.7 }), red);
  }

  // --- lanterns: white paper globes in drops of two or three, on wires zigzagging across mid-block
  {
    const wires: THREE.BufferGeometry[] = [], globes: THREE.BufferGeometry[] = [];
    const sw = curb(0, 2.0), ne = curb(1, 1.5), r = seeded(57);
    for (let k = 0; k < LANTERN_WIRES; k++) {
      const s = zebraS + LANTERN_FROM + k * LANTERN_STEP;
      const a = at(k % 2 ? ne : sw, s), b = at(k % 2 ? sw : ne, s + LANTERN_STEP);
      if (!a || !b) continue;
      const A = v3(a.p, LANTERN_H), B = v3(b.p, LANTERN_H + 0.3), span = A.distanceTo(B);
      const w = wireGeo(A, B, 0.06 * span, 0.008, Math.max(2, Math.round(span / LANTERN_EVERY)));
      wires.push(...w.mesh);
      for (const q of w.line.slice(1, -1)) {
        const n = 2 + (r() < 0.5 ? 1 : 0);
        let y = q.y - 0.25;
        wires.push(limb(q, new THREE.Vector3(q.x, q.y - 0.25 - n * 0.42, q.z), 0.005, 0.005, 3));
        for (let i = 0; i < n; i++) {
          const rad = 0.15 + r() * 0.05;
          globes.push(new THREE.SphereGeometry(rad, 12, 8).scale(1, 1.12, 1).translate(q.x, y - rad, q.z));
          y -= rad * 2.2 + 0.05;
        }
      }
    }
    solid(new THREE.MeshStandardMaterial({ color: 0x161616, roughness: 0.5 }), wires);
    solid(new THREE.MeshStandardMaterial({ color: 0xf6f3ea, emissive: 0x2a261c, roughness: 0.75 }), globes);
  }

  // --- signs on posts at the SW curb, facing walkers coming from the SE
  const metal = new THREE.MeshStandardMaterial({ color: 0x8c908e, roughness: 0.4, metalness: 0.7 });
  const posts: THREE.BufferGeometry[] = [];
  const sign = (s: number, faces: { tex: THREE.Texture; w: number; h: number; top: number }[]) => {
    const a = at(curb(0, 0.35), s);
    if (!a) return;
    posts.push(...signPostGeo().map((g) => g.translate(a.p[0], CURB_H, -a.p[1])));
    for (const f of faces) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(f.w, f.h), new THREE.MeshStandardMaterial({ map: f.tex, roughness: 0.45 }));
      m.add(new THREE.Mesh(m.geometry, metal).rotateY(Math.PI));
      m.position.set(a.p[0] + a.u[0] * 0.04, CURB_H + f.top - f.h / 2, -(a.p[1] + a.u[1] * 0.04));
      m.rotation.y = Math.atan2(a.u[0], -a.u[1]);   // normal toward +u (SE)
      m.castShadow = true;
      root.add(m);
    }
    segs.push(...boxSegs(a.p, a.u, 0.05, 0.05));
  };
  sign(sOn(moss.pts, NO_PARKING_SW), [{ tex: noParkingArrowTexture(), w: 0.3, h: 0.3, top: 2.6 }]);
  sign(spawnS - REG_SIGNS_AHEAD, [{ tex: regSignTexture("sweep"), w: 0.3, h: 0.3, top: 2.65 }, { tex: regSignTexture("2hr"), w: 0.3, h: 0.3, top: 2.3 }]);
  solid(metal, posts);

  // --- against the NE houses: the FDC on 33 Moss, climbing shrubs
  /** Nearest point on a building's ground-floor wall to p, with the outward normal (toward p's side). */
  const wallAt = (p: P, only?: number) => {
    let best: { q: P; u: P; n: P; d: number } | null = null;
    for (const b of data.buildings) {
      if (only !== undefined && b.id !== only) continue;
      const g = b.ground ?? b.pts;
      for (let k = 0; k + 1 < g.length; k++) {
        const [a, c] = [g[k], g[k + 1]], L = dist(a, c);
        if (L < 1) continue;
        const u: P = [(c[0] - a[0]) / L, (c[1] - a[1]) / L];
        const t = Math.max(0.3, Math.min(L - 0.3, (p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1]));
        const q = add(a, u, t), d = dist(p, q);
        if (d >= (best?.d ?? 4)) continue;
        let n: P = [-u[1], u[0]];
        if ((p[0] - q[0]) * n[0] + (p[1] - q[1]) * n[1] < 0) n = [u[1], -u[0]];
        best = { q, u, n, d };
      }
    }
    return best;
  };
  {
    const w = wallAt(FDC.near, FDC.building);
    if (w) {
      const chrome = new THREE.MeshStandardMaterial({ color: 0xd0d2d4, metalness: 1, roughness: 0.25 });
      const red = new THREE.MeshStandardMaterial({ color: 0xc2272d, roughness: 0.4 });
      const M = new THREE.Matrix4().makeRotationY(Math.atan2(w.n[1], w.n[0])).setPosition(w.q[0], CURB_H, -w.q[1]);   // local +x out of the wall
      const cyl = (r: number, h: number) => new THREE.CylinderGeometry(r, r, h, 14);
      const body = [cyl(0.05, 0.2).rotateZ(Math.PI / 2).translate(0.1, 0.75, 0), cyl(0.045, 0.75).translate(0.12, 0.375, 0),
        ...[-0.14, 0.14].map((z) => cyl(0.045, 0.16).rotateZ(Math.PI / 2).rotateY(z > 0 ? -0.5 : 0.5).translate(0.2, 0.78, z))];
      const caps = [-0.14, 0.14].map((z) => cyl(0.075, 0.05).rotateZ(Math.PI / 2).translate(0.28, 0.78, z * 1.25));
      solid(chrome, body.map((g) => g.applyMatrix4(M)));
      solid(red, caps.map((g) => g.applyMatrix4(M)));
      segs.push(...boxSegs(add(w.q, w.n, 0.17), w.u, 0.28, 0.17));
    }
  }
  {
    const leaves = new LeafSoup(), r = seeded(91);
    for (const sh of SHRUBS) {
      const w = wallAt(sh.near);
      if (!w) continue;
      const depth = Math.min(0.55, routeDist(w.q) - WALKER_RADIUS - 0.15);   // never into the walk
      if (depth < 0.2) continue;
      const c = add(w.q, w.n, depth / 2), crown = v3(c, CURB_H + sh.h / 2), crownR = Math.max(sh.len, sh.h) / 2;
      for (let i = 0; i < sh.len * sh.h * 14; i++) {
        const along = (r() - 0.5) * sh.len, out = r() * depth, up = r() ** 0.8 * sh.h;
        const taper = 1 - 0.35 * Math.abs(along) / (sh.len / 2) * (up / sh.h);   // rounder at the top corners
        const p = add(add(w.q, w.u, along * taper), w.n, out);
        const nrm = new THREE.Vector3(w.n[0] + (r() - 0.5), r() - 0.3, -(w.n[1] + (r() - 0.5))).normalize();
        leaves.card(v3(p, CURB_H + up), nrm, 0.3 + r() * 0.25, r() * Math.PI, crown, crownR, 0.75 + r() * 0.3);
      }
      segs.push(...boxSegs(c, w.u, sh.len / 2, depth / 2));
    }
    const mesh = new THREE.Mesh(leaves.build(), new THREE.MeshStandardMaterial({
      map: leafTexture(), alphaTest: 0.5, side: THREE.DoubleSide, vertexColors: true, roughness: 0.85 }));
    mesh.castShadow = mesh.receiveShadow = true;
    if (leaves.pos.length) root.add(mesh);
  }
  console.info(`[mossDetails] Street View details on Moss: ${segs.length / 4} colliders`);
  return segs;
}
