// Moss St at Howard St, San Francisco, built from OSM (scripts/osm_corner.py -> public/osm/moss.json).
// JSON frame: x east, y north, metres from the zebra across Moss. three.js: (x, up, -north).
import * as THREE from "three";
import moss from "../public/osm/moss.json";
import { spawnMossActors } from "./mossActors";

type P = [number, number];
type Quad = P[];
/** A wall segment [x1, n1, x2, n2] in metres east/north. Building faces and curbs. */
export type Wall = [number, number, number, number];
/** A car lane: points in travel order, metres east/north. */
export interface Lane { road: number; name: string; width: number; speed_mps: number; pts: P[] }

interface MossData {
  spawn: { x: number; y: number; heading: number };
  route: P[];
  buildings: { id: number; h: number; pts: P[] }[];
  lanes: Lane[];
  crossings: { x: number; y: number; markings: string; u: P; across: P }[];
  road_quads: Quad[]; track_quads: Quad[]; sidewalk_quads: Quad[];
  curbs: Wall[];
}
const data = moss as unknown as MossData;

/** What the Moss world adds over the placeholder world. main.ts reads these if present. */
export interface WorldExtras {
  spawn?: { x: number; n: number; heading: number };
  walls?: (x: number, n: number, r: number) => object[];
  overlay?: THREE.Object3D;   // screen-only helpers (route line), hidden in camera frames
  place?: string;
}

export const lanes: Lane[] = data.lanes;
export const route: P[] = data.route;
export const crossings = data.crossings;

const CURB_H = 0.15;
const v3 = (p: P, y: number) => new THREE.Vector3(p[0], y, -p[1]);
const mat = (color: number, extra: THREE.MeshLambertMaterialParameters = {}) =>
  new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide, ...extra });

// Flat quads at height y as one mesh.
function flat(quads: Quad[], y: number, m: THREE.Material) {
  const pos: number[] = [];
  for (const q of quads) for (const i of [0, 1, 2, 0, 2, 3]) pos.push(q[i][0], y, -q[i][1]);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return new THREE.Mesh(g, m);
}

// Quads raised to height h: top plus four side faces (the curb faces).
function slab(quads: Quad[], h: number, m: THREE.Material) {
  const pos: number[] = [];
  const push = (...ps: THREE.Vector3[]) => ps.forEach((p) => pos.push(p.x, p.y, p.z));
  for (const q of quads) {
    const t = q.map((p) => v3(p, h)), b = q.map((p) => v3(p, 0));
    push(t[0], t[1], t[2], t[0], t[2], t[3]);
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      push(b[i], b[j], t[j], b[i], t[j], t[i]);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return new THREE.Mesh(g, m);
}

function zebraQuads(): Quad[] {
  const out: Quad[] = [];
  for (const c of data.crossings) {
    if (!["zebra", "yes", "ladder"].includes(c.markings)) continue;
    const [ux, uy] = c.u, nx = -uy, ny = ux;          // u along the road, n across it
    for (let o = c.across[0] + 0.5; o < c.across[1] - 0.3; o += 1.0) {
      const cx = c.x + nx * o, cy = c.y + ny * o;
      const corner = (a: number, b: number): P => [cx + ux * a + nx * b, cy + uy * a + ny * b];
      out.push([corner(-1.5, -0.25), corner(1.5, -0.25), corner(1.5, 0.25), corner(-1.5, 0.25)]);
    }
  }
  return out;
}

function ribbon(pts: P[], w: number): Quad[] {
  const out: Quad[] = [];
  for (let i = 0; i + 1 < pts.length; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const nx = (-(b[1] - a[1]) / d) * w / 2, ny = ((b[0] - a[0]) / d) * w / 2;
    out.push([[a[0] + nx, a[1] + ny], [b[0] + nx, b[1] + ny], [b[0] - nx, b[1] - ny], [a[0] - nx, a[1] - ny]]);
  }
  return out;
}

function distToSeg(x: number, n: number, [x1, n1, x2, n2]: Wall) {
  const dx = x2 - x1, dn = n2 - n1;
  const t = Math.max(0, Math.min(1, ((x - x1) * dx + (n - n1) * dn) / (dx * dx + dn * dn || 1)));
  return Math.hypot(x - (x1 + t * dx), n - (n1 + t * dn));
}

/** actors: false leaves the street empty (metrics mode). walkers: people the cars must yield to. */
export function buildOsmWorld(opts: { actors?: boolean; walkers?: () => { x: number; n: number }[] } = {}) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9ec9ff);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x666655, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(30, 60, 20);
  scene.add(sun);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), mat(0x8a8780));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.02;
  scene.add(ground);
  scene.add(flat(data.road_quads, 0, mat(0x3a3a3c)));
  scene.add(flat(data.track_quads, 0.005, mat(0x4f6b52)));   // Howard's protected cycle track
  scene.add(flat(zebraQuads(), 0.01, mat(0xf2f2f2)));
  scene.add(slab(data.sidewalk_quads, CURB_H, mat(0xbdbdb5)));

  // Buildings: extruded footprints, a few shades so faces read in the camera frames.
  const shades = [0xc9a27e, 0xa7b4c2, 0xd8cfc4, 0xb98b73, 0x8f9aa8, 0xcfc39b, 0xb0a99f];
  const walls: Wall[] = [...data.curbs];
  data.buildings.forEach((b, i) => {
    const shape = new THREE.Shape(b.pts.map(([x, y]) => new THREE.Vector2(x, y)));
    const g = new THREE.ExtrudeGeometry(shape, { depth: b.h, bevelEnabled: false });
    g.rotateX(-Math.PI / 2);                                   // shape y (north) -> -z, depth -> up
    scene.add(new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: shades[i % shades.length] })));
    for (let k = 0; k + 1 < b.pts.length; k++) walls.push([...b.pts[k], ...b.pts[k + 1]] as Wall);
  });

  // Planned demo walk, drawn faintly on screen only (main.ts hides it for camera frames).
  const overlay = flat(ribbon(data.route, 0.3), CURB_H + 0.02,
    new THREE.MeshBasicMaterial({ color: 0x33ddff, transparent: true, opacity: 0.35, depthWrite: false, side: THREE.DoubleSide }));
  overlay.visible = false;
  scene.add(overlay);

  const extras: Required<WorldExtras> = {
    spawn: { x: data.spawn.x, n: data.spawn.y, heading: data.spawn.heading },
    // ponytail: brute-force scan of ~1000 segments per call; add a grid if the world grows a lot
    walls: (x, n, r) => walls.filter((w) => distToSeg(x, n, w) < r),
    overlay,
    place: "Moss St at Howard St",
  };
  const actors = opts.actors === false ? null : spawnMossActors(scene, opts.walkers ?? (() => []));
  return { scene, update: (dt: number) => actors?.update(dt), solids: () => actors?.solids() ?? [], ...extras };
}
