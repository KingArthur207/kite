// Moss St at Howard St, San Francisco, built from OSM (scripts/osm_corner.py -> public/osm/moss.json).
// JSON frame: x east, y north, metres from the zebra across Moss. three.js: (x, up, -north).
import * as THREE from "three";
import moss from "../public/osm/moss.json";
import { spawnMossActors } from "./mossActors";
import { addSkyAndSun, tuneRenderer } from "./osmLook";
import { buildFacades, type Building } from "./osmFacades";
import { buildStreets, ribbon, type StreetData } from "./osmStreet";
import { buildStreetDressing } from "./streetDressing";

type P = [number, number];
type Quad = P[];
/** A wall segment [x1, n1, x2, n2] in metres east/north. Building faces and curbs. */
export type Wall = [number, number, number, number];
/** A car lane: points in travel order, metres east/north. */
export interface Lane { road: number; name: string; width: number; speed_mps: number; pts: P[] }

interface MossData extends StreetData {
  spawn: { x: number; y: number; heading: number };
  route: P[];
  buildings: Building[];
  lanes: Lane[];
  crossings: StreetData["crossings"];
  curbs: Wall[];
}
const data = moss as unknown as MossData;

/** What the Moss world adds over the placeholder world. main.ts reads these if present. */
export interface WorldExtras {
  spawn?: { x: number; n: number; heading: number };
  walls?: (x: number, n: number, r: number) => object[];
  overlay?: THREE.Object3D;   // screen-only helpers (route line), hidden in camera frames
  place?: string;
  tune?: (r: THREE.WebGLRenderer) => void;   // renderer settings; main.ts applies them to both renderers
}

export const lanes: Lane[] = data.lanes;
export const route: P[] = data.route;
export const crossings = data.crossings;

const CURB_H = 0.15;
const v3 = (p: P, y: number) => new THREE.Vector3(p[0], y, -p[1]);

// Flat quads at height y as one mesh.
function flat(quads: Quad[], y: number, m: THREE.Material) {
  const pos: number[] = [];
  for (const q of quads) for (const i of [0, 1, 2, 0, 2, 3]) pos.push(...v3(q[i], y).toArray());
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return new THREE.Mesh(g, m);
}

function distToSeg(x: number, n: number, [x1, n1, x2, n2]: Wall) {
  const dx = x2 - x1, dn = n2 - n1;
  const t = Math.max(0, Math.min(1, ((x - x1) * dx + (n - n1) * dn) / (dx * dx + dn * dn || 1)));
  return Math.hypot(x - (x1 + t * dx), n - (n1 + t * dn));
}

/** actors: false leaves the street empty (metrics mode). walkers: people the cars must yield to.
 *  osmMeshes is the drawn street (ground, roads, buildings); the tiles world shows it to the capture cameras only. */
export function buildOsmWorld(opts: { actors?: boolean; walkers?: () => { x: number; n: number }[] } = {}) {
  const scene = new THREE.Scene();
  // Sun shadows cover the demo walk and the corner; the sky HDRI lights everything else.
  addSkyAndSun(scene, { x: 15, n: -25 }, 160);
  const visible = new THREE.Group();
  scene.add(visible);
  const { garages, windows } = buildFacades(visible, data.buildings, data.road_quads);
  buildStreets(visible, data, garages);
  console.info(`[osmWorld] ${data.buildings.length} buildings, ${windows} windows and doors, ${garages.length} garages`);

  const walls: Wall[] = [...data.curbs];
  for (const b of data.buildings) for (let k = 0; k + 1 < b.pts.length; k++) walls.push([...b.pts[k], ...b.pts[k + 1]] as Wall);
  walls.push(...buildStreetDressing(visible, data, garages));   // parked cars, trees, poles, signs: they collide too

  // Planned demo walk, drawn faintly on screen only (main.ts hides it for camera frames).
  const overlay = new THREE.Group();
  overlay.add(flat(ribbon(data.route, 0.3), CURB_H + 0.02,
    new THREE.MeshBasicMaterial({ color: 0x33ddff, transparent: true, opacity: 0.35, depthWrite: false, side: THREE.DoubleSide })));
  overlay.visible = false;
  scene.add(overlay);

  const extras: Required<WorldExtras> = {
    spawn: { x: data.spawn.x, n: data.spawn.y, heading: data.spawn.heading },
    // ponytail: brute-force scan of ~1000 segments per call; add a grid if the world grows a lot
    walls: (x, n, r) => walls.filter((w) => distToSeg(x, n, w) < r),
    overlay,
    place: "Moss St at Howard St",
    tune: tuneRenderer,
  };
  const actors = opts.actors === false ? null : spawnMossActors(scene, opts.walkers ?? (() => []));
  return { scene, osmMeshes: visible, update: (dt: number) => actors?.update(dt), solids: () => actors?.solids() ?? [], ...extras };
}
