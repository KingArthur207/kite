// ?world=photo: the OSM Moss world with each wall that has a Street View photo (public/local/facades_*/manifest.json,
// local only, never committed) drawn as one textured quad. Every other wall keeps the procedural osmFacades look,
// and streets, curbs and furniture stay osmStreet's. Collision, lanes, route, spawn and traffic are buildOsmWorld's.
// L (any world that has photos) labels walls `building_id/wall_index` on screen: photo walls green, the rest white.
import * as THREE from "three";
import moss from "../public/osm/moss.json";
import { buildOsmWorld } from "./osmWorld";

type P = [number, number];
/** One manifest entry. corners_local_m: bottom-left first (as seen from the street), moss.json metres (x east,
 *  y north, optional z up). texture is relative to the manifest's folder. */
export interface PhotoWall {
  building_id: number; wall_index: number; corners_local_m: number[][]; width_m: number; height_m: number;
  texture: string; score?: number; dir: string;
}

export const PHOTO_DIRS = ["local/facades_ne/", "local/facades_sw/", "local/facades_howard/"];
const BLEND_H = 0.6;        // m: the photo darkens toward the sidewalk over this height, like a contact shadow
const BLEND_DARK = 0.55;    // colour multiplier at the bottom edge
const NORMAL_SIZE = 3.2;    // m: plaster normal map tile, for a little relief under the photo
const MATCH_M = 1.5;        // m: manifest bottom corners must sit this close to an OSM footprint edge

/** Walls with a photo, filled by loadPhotoManifests() before main.ts runs (entry.ts). */
export let photoWalls: PhotoWall[] = [];
const key = (w: { building_id: number; wall_index: number }) => `${w.building_id}:${w.wall_index}`;

const finite = (v: unknown) => typeof v === "number" && Number.isFinite(v);
function valid(w: Partial<PhotoWall>): boolean {
  return finite(w.building_id) && Number.isInteger(w.wall_index) && finite(w.height_m) && w.height_m! > 0 &&
    Array.isArray(w.corners_local_m) && w.corners_local_m.length === 4 &&
    w.corners_local_m.every((c) => Array.isArray(c) && c.length >= 2 && c.every(finite)) &&
    typeof w.texture === "string" && /^[\w./-]+$/.test(w.texture) && !w.texture.includes("..") && !w.texture.startsWith("/");
}

export async function loadPhotoManifests() {
  const lists = await Promise.all(PHOTO_DIRS.map(async (dir) => {
    const r = await fetch(dir + "manifest.json").catch(() => null);
    // Vite answers a missing file with index.html, so anything that isn't JSON means "no manifest here".
    if (!r?.ok || !(r.headers.get("content-type") ?? "").includes("json")) return [];
    try {
      const j = await r.json();
      const all = (Array.isArray(j) ? j : j.walls ?? []) as Partial<PhotoWall>[];
      const ok = all.filter(valid);
      if (ok.length < all.length) console.warn(`[photoWorld] ${dir}manifest.json: ${all.length - ok.length} malformed walls skipped`);
      return ok.map((w) => ({ ...w, dir }) as PhotoWall);
    } catch (e) {
      console.warn(`[photoWorld] ${dir}manifest.json unreadable`, e);
      return [];
    }
  }));
  const best = new Map<string, PhotoWall>();   // two folders on one wall: the higher score wins
  for (const w of lists.flat()) {
    const o = best.get(key(w));
    if (!o || (w.score ?? 0) > (o.score ?? 0)) best.set(key(w), w);
  }
  photoWalls = [...best.values()];
}

interface Building { id: number; pts: P[] }
const buildings = (moss as unknown as { buildings: Building[] }).buildings;

function ring(b: Building) {
  const r = b.pts.slice();
  if (r.length > 1 && r[0][0] === r[r.length - 1][0] && r[0][1] === r[r.length - 1][1]) r.pop();
  return r;
}
const area = (r: P[]) => r.reduce((s, p, i) => s + p[0] * r[(i + 1) % r.length][1] - r[(i + 1) % r.length][0] * p[1], 0);
const dist = (a: P, b: P) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** The OSM edge a manifest wall sits on: wall_index if its ends match the photo's bottom corners, else the closest
 *  edge of that building (logged). Returns the edge index and its outward normal, or null. */
function matchEdge(w: PhotoWall, bl: P, br: P) {
  const b = buildings.find((x) => x.id === w.building_id);
  if (!b) return null;
  const r = ring(b), s = area(r) > 0 ? 1 : -1;
  const err = (i: number) => {
    const a = r[i], c = r[(i + 1) % r.length];
    return Math.min(Math.max(dist(a, bl), dist(c, br)), Math.max(dist(a, br), dist(c, bl)));
  };
  let i = w.wall_index;
  if (!(i >= 0 && i < r.length && err(i) < MATCH_M)) {
    i = r.map((_, k) => k).sort((p, q) => err(p) - err(q))[0];
    if (err(i) >= MATCH_M) return null;
    console.warn(`[photoWorld] ${key(w)}: corners match edge ${i}, using that`);
  }
  const a = r[i], c = r[(i + 1) % r.length], L = dist(a, c);
  const u: P = [(c[0] - a[0]) / L, (c[1] - a[1]) / L];
  return { edge: i, out: (s > 0 ? [u[1], -u[0]] : [-u[1], u[0]]) as P };
}

/** Quad from bl..br, z0..z1, split at the blend height; uv = photo (0..1), uv1 = metres for the normal map. */
function wallGeometry(bl: P, br: P, z0: number, z1: number, out: P) {
  const W = dist(bl, br), zb = Math.min(z1, z0 + BLEND_H);
  const pos: number[] = [], uv: number[] = [], uv1: number[] = [], col: number[] = [], nor: number[] = [];
  for (const z of [z0, zb, z1]) for (const t of [0, 1]) {
    const p: P = [bl[0] + (br[0] - bl[0]) * t, bl[1] + (br[1] - bl[1]) * t];
    pos.push(p[0], z, -p[1]);
    uv.push(t, (z - z0) / (z1 - z0));
    uv1.push((t * W) / NORMAL_SIZE, (z - z0) / NORMAL_SIZE);
    const c = z === z0 ? BLEND_DARK : 1;
    col.push(c, c, c);
    nor.push(out[0], 0, -out[1]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute("uv1", new THREE.Float32BufferAttribute(uv1, 2));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
  // Vertices: 0 bl-z0, 1 br-z0, 2 bl-zb, 3 br-zb, 4 bl-z1, 5 br-z1. Wind each triangle to face `out`.
  const e1 = new THREE.Vector3(pos[3] - pos[0], 0, pos[5] - pos[2]), up = new THREE.Vector3(0, 1, 0);
  const front = e1.cross(up).dot(new THREE.Vector3(out[0], 0, -out[1])) >= 0;   // bl->br then up faces out
  const tri = (a: number, b: number, c: number) => (front ? [a, b, c] : [a, c, b]);
  g.setIndex([...tri(0, 1, 3), ...tri(0, 3, 2), ...tri(2, 3, 5), ...tri(2, 5, 4)]);
  return g;
}

function label(text: string, color: string) {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 64;
  const x = c.getContext("2d")!;
  x.fillStyle = "rgba(0,0,0,0.6)"; x.fillRect(0, 0, 256, 64);
  x.fillStyle = color; x.font = "bold 34px monospace"; x.textAlign = "center"; x.textBaseline = "middle";
  x.fillText(text, 128, 32);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, fog: false }));
  s.scale.set(2.4, 0.6, 1);
  s.renderOrder = 999;
  return s;
}

/** Labels at every footprint edge of 2 m or more, 2.5 m up and 0.4 m out. */
function wallLabels(photo: Set<string>) {
  const g = new THREE.Group();
  for (const b of buildings) {
    const r = ring(b), s = area(r) > 0 ? 1 : -1;
    r.forEach((a, i) => {
      const c = r[(i + 1) % r.length], L = dist(a, c);
      if (L < 2) return;
      const k = `${b.id}:${i}`;
      const sp = label(`${b.id}/${i}`, photo.has(k) ? "#5f5" : "#fff");
      const o: P = [(s * (c[1] - a[1])) / L, (-s * (c[0] - a[0])) / L];
      sp.position.set((a[0] + c[0]) / 2 + o[0] * 0.4, 2.5, -((a[1] + c[1]) / 2 + o[1] * 0.4));
      g.add(sp);
    });
  }
  return g;
}

export function buildPhotoWorld(opts: Omit<Parameters<typeof buildOsmWorld>[0] & object, "skipWalls"> = {}) {
  const flag = window as unknown as { __kitePhoto: string };   // screenshot scripts wait for "ready"
  flag.__kitePhoto = "loading";
  const placed: { w: PhotoWall; bl: P; br: P; z0: number; z1: number; out: P }[] = [];
  let mirrored = 0;
  for (const w of photoWalls) {
    const c = w.corners_local_m, has3 = c.every((p) => p.length >= 3);
    const z = (p: number[]) => (has3 ? p[2] : 0);
    const bl = c[0].slice(0, 2) as P;
    const zMin = Math.min(...c.map(z));
    // Bottom-right: the other bottom corner farthest from bottom-left (2D manifests: the farthest corner).
    const br = c.slice(1).filter((p) => z(p) < zMin + 0.5).sort((p, q) => dist(q as P, bl) - dist(p as P, bl))[0].slice(0, 2) as P;
    const m = matchEdge(w, bl, br);
    if (!m) { console.warn(`[photoWorld] ${key(w)}: no OSM edge under its corners, skipped`); continue; }
    // From the street (looking along -out) the right-hand side is (-out) x up = (-out.y, out.x).
    if ((br[0] - bl[0]) * -m.out[1] + (br[1] - bl[1]) * m.out[0] < 0) mirrored++;
    const z1 = has3 ? Math.max(...c.map(z)) : zMin + w.height_m;
    placed.push({ w: { ...w, wall_index: m.edge }, bl, br, z0: Math.min(zMin, 0), z1, out: m.out });
  }
  const skip = new Set(placed.map((p) => key(p.w)));
  const world = buildOsmWorld({ ...opts, skipWalls: skip });

  const normal = new THREE.TextureLoader().load("textures/plaster_nor.jpg");
  normal.wrapS = normal.wrapT = THREE.RepeatWrapping;
  normal.channel = 1;
  let pending = placed.length;
  const done = () => { if (--pending <= 0) flag.__kitePhoto = "ready"; };
  if (!pending) flag.__kitePhoto = "ready";
  const loader = new THREE.TextureLoader();
  for (const p of placed) {
    const map = loader.load(p.w.dir + p.w.texture, done, undefined, () => {
      console.error(`[photoWorld] failed to load ${p.w.dir}${p.w.texture}`);
      done();
    });
    map.colorSpace = THREE.SRGBColorSpace;
    map.anisotropy = 8;
    const mat = new THREE.MeshStandardMaterial({
      map, normalMap: normal, normalScale: new THREE.Vector2(0.35, 0.35),
      roughness: 0.92, metalness: 0, envMapIntensity: 0.6, vertexColors: true,
    });
    const mesh = new THREE.Mesh(wallGeometry(p.bl, p.br, p.z0, p.z1, p.out), mat);
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.name = `photo ${key(p.w)}`;
    world.osmMeshes.add(mesh);   // drawn by the screen and the capture renderer alike
  }
  const procedural = buildings.reduce((s, b) => s + ring(b).length, 0) - placed.length;
  console.info(`[photoWorld] ${placed.length} photo walls, ${procedural} procedural walls` +
    (mirrored ? `; ${mirrored} manifests list bottom-left as seen from inside the building (would look mirrored)` : ""));
  (window as unknown as { __kitePhotoStats: unknown }).__kitePhotoStats = { photo: placed.length, procedural, mirrored };
  return world;
}

/** L toggles wall labels (screen only: they live in the overlay, which camera frames never draw). */
export function addWallLabelKey(overlay: THREE.Object3D) {
  let labels: THREE.Group | null = null;
  window.addEventListener("keydown", (e) => {
    if (e.key.toLowerCase() !== "l" || e.repeat) return;
    if (!labels) overlay.add(labels = wallLabels(new Set(photoWalls.map(key))));
    else labels.visible = !labels.visible;
  });
}
