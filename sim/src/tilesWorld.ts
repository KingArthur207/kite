// Moss St at Howard from Google Photorealistic 3D Tiles. The OSM world (osmWorld.ts) still gives the
// colliders, lanes, route, spawn and traffic.
//
// Google permission: at Cal Hacks (Oct 24, 2026) Google's reps allowed us to use Maps Platform here,
// including running our detector on rendered tile frames, "only to the extent we truly need it, which
// is just this one street" (Moss St at Howard). See docs/google-permission.md. So:
// - Tiles are geofenced to Moss St plus the Howard corner (fence.ts): tiles outside are never
//   requested (LoadRegionPlugin mask; every download is also checked against the fence and counted
//   in tilesStats().fenceViolations), tile pixels outside are discarded, the walker is clamped inside.
// - detector "tiles" (?detector=tiles): the capture cameras (frames to relay.py and the detector) see the tiles.
// - detector "osm" (default): the hybrid; also covers for the general Map Tiles policy, which lists
//   object detection as a prohibited use. Tiles on screen only; capture renders show the OSM meshes.
//   main.ts calls view("capture") before each capture render and view("screen") before the screen
//   render, and every tile mesh counts any draw by the capture renderer in tilesStats().captureLeaks.
// Frame: tiles reoriented to ENU at ORIGIN (the Moss zebra), then turned 180° about up so
// x = east, z = south (north is -z), and lowered so the road at the zebra sits at y = 0.
import * as THREE from "three";
import { TilesRenderer, OBB, WGS84_ELLIPSOID } from "3d-tiles-renderer";
import { GoogleCloudAuthPlugin, GLTFExtensionsPlugin, ReorientationPlugin, LoadRegionPlugin, OBBRegion } from "3d-tiles-renderer/plugins";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import moss from "../public/osm/moss.json";
import { buildOsmWorld, lanes, route } from "./osmWorld";
import { ORIGIN, CAM_W, CAM_H } from "./config";
import { FENCE, U, fenceSelfCheck } from "./fence";

export const TILES_KEY: string = import.meta.env.VITE_GOOGLE_MAPS_API_KEY ?? "";

const DEG = Math.PI / 180;
const GROUND_SAMPLE_M = 12;       // road under the lanes this close to the demo route levels the tiles
const GROUND_STEP_M = 3;
const RESAMPLE_S = 2;             // re-level as finer tiles arrive
const RESAMPLE_FOR_S = 90;

type P = [number, number];

export function buildTilesWorld(opts: {
  walkers: () => { x: number; n: number }[]; cameras: THREE.Camera[]; capture: THREE.WebGLRenderer;
  detector: "tiles" | "osm"; debug?: boolean;
}) {
  fenceSelfCheck();
  const world = buildOsmWorld({ walkers: opts.walkers });
  const { scene, osmMeshes } = world;
  const tilesLayer = new THREE.Group();    // tiles, the gray under-plane and blob shadows
  scene.add(tilesLayer);
  const hybrid = opts.detector === "osm";
  let captureLeaks = 0;
  /** detector "osm": tiles for the screen, OSM meshes for capture. detector "tiles": always tiles. */
  const view = (mode: "screen" | "capture") => {
    tilesLayer.visible = !hybrid || mode === "screen";
    osmMeshes.visible = !tilesLayer.visible;
  };
  view("screen");

  const tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: TILES_KEY, autoRefreshToken: true }));
  const draco = new DRACOLoader().setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.7/");
  tiles.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader: draco }));
  tiles.registerPlugin(new ReorientationPlugin({ lat: ORIGIN.lat * DEG, lon: ORIGIN.lng * DEG, height: 0 }));
  tiles.errorTarget = ERROR_TARGET_PX;
  const fenceObb = fenceInTileFrame();
  const region = new LoadRegionPlugin();
  region.addRegion(new OBBRegion({ obb: fenceObb, mask: true, errorTarget: ERROR_TARGET_PX }));
  tiles.registerPlugin(region);
  // By default the renderer also fetches every sibling (and ancestor) of a tile it refines, mask or
  // not, which pulled in tiles outside the fence. Off: only tiles the mask lets through load.
  tiles.loadSiblings = false;
  tiles.loadAncestors = false;
  for (const cam of opts.cameras) { tiles.setCamera(cam); tiles.setResolution(cam, CAM_W, CAM_H); }

  const root = new THREE.Group();     // ENU (x west, z north) -> sim (x east, z south)
  root.rotation.y = Math.PI;
  root.add(tiles.group);
  tilesLayer.add(root);
  // Under a photogrammetry tree there is often no ground at all; this shows through the cut instead of sky.
  const under = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), new THREE.MeshBasicMaterial({ color: 0x77736c }));
  under.rotation.x = -Math.PI / 2;
  under.position.y = -0.4;
  tilesLayer.add(under);
  const stats = { requests: 0, roots: 0, fenceViolations: 0 };
  tiles.addEventListener("tile-download-start", ({ tile }) => {     // one per tile fetch
    stats.requests++;
    // Assertion: the mask above should make this impossible. Count it loudly if it ever happens.
    const bv = (tile as unknown as { engineData?: { boundingVolume?: { intersectsOBB(o: OBB): boolean } } }).engineData?.boundingVolume;
    if (bv && !bv.intersectsOBB(fenceObb) && stats.fenceViolations++ === 0)
      console.error("[sim] requested a Google tile outside the Moss geofence");
  });
  tiles.addEventListener("load-root-tileset", () => stats.roots++);
  // Street trees and parked cars are solid melted blobs in the photogrammetry, and the OSM colliders
  // let the walker through them, so a camera can end up inside one. Cut tile geometry within
  // NEAR_CUT_M of whichever camera is rendering, and between that camera and the walker, except the
  // ground (see nearCut).
  tiles.addEventListener("load-model", ({ scene: model }) => model.traverse((o) => {
    const m = (o as THREE.Mesh).material as THREE.Material | undefined;
    if (m && !Array.isArray(m)) nearCut(m);
    // Leak check for detector "osm" (see top): this runs only when the mesh is actually drawn.
    o.onBeforeRender = (r) => {
      if (!hybrid || r !== opts.capture) return;
      if (captureLeaks++ === 0) console.error("[sim] Google tile drawn into a capture frame (Map Tiles policy)");
    };
  }));

  // Level the ground. Moss drops about 2 m from Howard to the spawn, so a single offset leaves
  // the far end of the walk floating. Fit a plane to the road under the lanes (lowest hit, so trees
  // and photogrammetry cars don't count), then tilt and shift the tiles so that plane is y = 0.
  const samples = laneSamples();
  const ray = new THREE.Raycaster();
  const down = new THREE.Vector3(0, -1, 0);
  /** Tile surface height under (x east, n north): the top hit, or the lowest (the ground under trees). */
  const heightAt = (x: number, n: number, lowest = false): number | null => {
    ray.set(new THREE.Vector3(x, 300, -n), down);
    const hits = ray.intersectObject(tiles.group, true);
    const hit = lowest ? hits[hits.length - 1] : hits[0];
    return hit ? hit.point.y : null;
  };
  const fit = { residual: NaN, tiltDeg: NaN, n: 0 };
  const level = () => {
    let pts = samples.map(([x, n]) => [x, n, heightAt(x, n, true)] as const).filter((p): p is [number, number, number] => p[2] !== null);
    if (pts.length < samples.length / 2) return;
    let plane = fitPlane(pts);
    pts = pts.filter(([x, n, y]) => Math.abs(y - plane(x, n)) < 0.5);       // drop curbs and holes
    plane = fitPlane(pts);
    const c = plane(0, 0), a = plane(1, 0) - c, b = plane(0, 1) - c;          // y = c + a*east + b*north
    const normal = new THREE.Vector3(-a, 1, b).normalize();                   // three z = -north
    const q = new THREE.Quaternion().setFromUnitVectors(normal, new THREE.Vector3(0, 1, 0));
    root.quaternion.premultiply(q);
    root.position.applyQuaternion(q);
    root.position.y -= c;
    root.updateMatrixWorld(true);
    const res = pts.map(([x, n, y]) => Math.abs(y - plane(x, n))).sort((u, v) => u - v);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(root.quaternion);
    Object.assign(fit, { residual: res[Math.floor(res.length * 0.9)], tiltDeg: Math.acos(Math.min(1, Math.abs(up.y))) / DEG, n: pts.length });
  };

  // Blob shadows under the glTF actors, which arrive asynchronously.
  const blobTex = (() => {
    const c = document.createElement("canvas"); c.width = c.height = 64;
    const g = c.getContext("2d")!, grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, "rgba(0,0,0,0.55)"); grad.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(c);
  })();
  const blobMat = new THREE.MeshBasicMaterial({ map: blobTex, transparent: true, depthWrite: false });
  const blobs: { actor: THREE.Object3D; blob: THREE.Mesh }[] = [];
  scene.addEventListener("childadded", ({ child }) => {
    if (!(child as THREE.Object3D).name) return;
    const size = new THREE.Box3().setFromObject(child).getSize(new THREE.Vector3());
    const blob = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), blobMat);
    blob.rotation.x = -Math.PI / 2;
    blob.scale.set(Math.max(size.x, size.z) * 1.1, Math.max(size.x, size.z) * 1.1, 1);
    blob.position.y = 0.03;
    blob.renderOrder = 1;
    tilesLayer.add(blob);
    blobs.push({ actor: child, blob });
  });

  // Ground truth for sim/yolo_check.py when the detector sees tiles: per camera (0 drone, 1 chest), is
  // the actor behind tile geometry? Both its middle and the top of its box must be hidden.
  type Gt = { min: number[]; max: number[]; occ?: boolean[] };
  const w = window as unknown as { __kiteActors?: () => Gt[] };
  const actorsGt = w.__kiteActors;
  if (actorsGt && !hybrid) {
    const occRay = new THREE.Raycaster();
    occRay.near = NEAR_CUT_M;                                   // geometry closer than this is cut away
    const blocked = (from: THREE.Vector3, to: THREE.Vector3) => {
      const d = to.clone().sub(from), len = d.length();
      occRay.set(from, d.normalize());
      occRay.far = len - 0.3;
      return occRay.intersectObject(tiles.group, true).length > 0;
    };
    w.__kiteActors = () => actorsGt().map((a) => {
      const mid = new THREE.Vector3((a.min[0] + a.max[0]) / 2, a.min[1] + (a.max[1] - a.min[1]) * 0.5, (a.min[2] + a.max[2]) / 2);
      const top = mid.clone().setY(a.min[1] + (a.max[1] - a.min[1]) * 0.9);
      return { ...a, occ: opts.cameras.slice(0, 2).map((c) => blocked(c.position, mid) && blocked(c.position, top)) };
    });
  }

  if (opts.debug) world.overlay.add(debugLines());
  const attribution = attributionBox(tiles);

  let t = 0, nextLevel = 0;
  return {
    ...world,
    place: world.place,
    tiles,
    view,
    detector: opts.detector,
    tilesStats: () => ({ ...stats, captureLeaks, fit: { ...fit }, rootY: root.position.y, progress: tiles.loadProgress, visible: tiles.visibleTiles.size }),
    heightAt,
    update(dt: number) {
      world.update(dt);
      const w = opts.walkers()[0];
      if (w) cutAt.value.set(w.x, 1, -w.n);
      for (const { actor, blob } of blobs) blob.position.set(actor.position.x, 0.03, actor.position.z);
      t += dt;
      if (t > nextLevel && t < RESAMPLE_FOR_S) { nextLevel = t + RESAMPLE_S; level(); }
      attribution(t);
    },
    /** Call once per frame after the cameras move, before rendering. */
    updateTiles() {
      for (const cam of opts.cameras) cam.updateMatrixWorld();
      tiles.update();
    },
  };
}

function laneSamples(): P[] {
  const out: P[] = [];
  for (const { pts } of lanes) for (let i = 0; i + 1 < pts.length; i++) {
    const [a, b] = [pts[i], pts[i + 1]], L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    for (let d = 0; d < L; d += GROUND_STEP_M) {
      const p: P = [a[0] + (b[0] - a[0]) * d / L, a[1] + (b[1] - a[1]) * d / L];
      if (distToRoute(p) < GROUND_SAMPLE_M) out.push(p);
    }
  }
  return out;
}

function distToRoute([x, n]: P) {
  let best = Infinity;
  for (let i = 0; i + 1 < route.length; i++) {
    const [[x1, n1], [x2, n2]] = [route[i], route[i + 1]], dx = x2 - x1, dn = n2 - n1;
    const t = Math.max(0, Math.min(1, ((x - x1) * dx + (n - n1) * dn) / (dx * dx + dn * dn)));
    best = Math.min(best, Math.hypot(x - x1 - t * dx, n - n1 - t * dn));
  }
  return best;
}

/** Least-squares plane y = c + a*x + b*n through the points. */
function fitPlane(pts: (readonly [number, number, number])[]) {
  const m = new THREE.Matrix3(), v = new THREE.Vector3();
  let sxx = 0, sxn = 0, sx = 0, snn = 0, sn = 0, sy = 0, sxy = 0, sny = 0;
  for (const [x, n, y] of pts) { sxx += x * x; sxn += x * n; sx += x; snn += n * n; sn += n; sy += y; sxy += x * y; sny += n * y; }
  m.set(sxx, sxn, sx, sxn, snn, sn, sx, sn, pts.length).invert();
  v.set(sxy, sny, sy).applyMatrix3(m);
  return (x: number, n: number) => v.x * x + v.y * n + v.z;
}

// OSM curbs (white) and building footprints (orange) on top of everything, screen only.
function debugLines() {
  const pos: number[] = [];
  const seg = (a: P, b: P, y: number) => pos.push(a[0], y, -a[1], b[0], y, -b[1]);
  for (const [x1, n1, x2, n2] of moss.curbs as number[][]) seg([x1, n1], [x2, n2], 0.15);
  const curbCount = pos.length / 3;
  for (const b of moss.buildings as unknown as { pts: P[] }[]) for (let i = 0; i + 1 < b.pts.length; i++) seg(b.pts[i], b.pts[i + 1], 0.3);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  const col: number[] = [];
  for (let i = 0; i < pos.length / 3; i++) col.push(...(i < curbCount ? [1, 1, 1] : [1, 0.55, 0]));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false, transparent: true, opacity: 0.8 }));
  lines.renderOrder = 10;
  return lines;
}

// Google Maps logo and the visible tiles' data attributions, bottom left (Maps Platform policy).
function attributionBox(tiles: TilesRenderer) {
  const box = document.createElement("div");
  box.style.cssText = "position:absolute;left:10px;bottom:5px;display:flex;align-items:flex-end;gap:12px;" +
    "pointer-events:none;font:11px/1.3 system-ui,sans-serif;color:#fff;text-shadow:0 0 3px #000,0 0 2px #000;";
  const logo = document.createElement("img");
  logo.src = "/google/maps_logo.svg";
  logo.alt = "Google Maps";
  logo.style.cssText = "height:18px;margin-top:10px;";
  const text = document.createElement("div");
  box.append(logo, text);
  document.body.appendChild(box);
  let next = 0;
  return (t: number) => {
    if (t < next) return;
    next = t + 1;
    const credits = tiles.getAttributions().filter((a) => a.type === "string").map((a) => a.value).join("; ");
    text.textContent = `${credits}  ·  3D city: Google Maps. Cars, people and the walker: Kite sim.`;
  };
}

/** The fence (fence.ts) as an OBB in the tiles' own frame (ECEF, TilesRenderer.group local), for
 *  LoadRegionPlugin. ENU at ORIGIN, turned so box x runs along Moss (U); 300 m tall so the
 *  leveling tilt and the ground height never matter. */
function fenceInTileFrame() {
  const enu = WGS84_ELLIPSOID.getEastNorthUpFrame(ORIGIN.lat * DEG, ORIGIN.lng * DEG, 0, new THREE.Matrix4());
  const turn = new THREE.Matrix4().makeRotationZ(Math.atan2(U[1], U[0]));     // ENU x (east) -> along Moss
  const box = new THREE.Box3(new THREE.Vector3(FENCE.uMin, FENCE.vMin, -100), new THREE.Vector3(FENCE.uMax, FENCE.vMax, 200));
  const obb = new OBB(box, enu.multiply(turn));
  obb.update();
  return obb;
}

const ERROR_TARGET_PX = 8;  // finer than the library's 16; about twice the tile requests
const NEAR_CUT_M = 3;       // trees and parked cars closer than this to a camera are cut away
// ... and anything in a cone from the camera down to the walker (radius CUT_R_CAM at the camera,
// CUT_R_WALKER at the walker), so the chase view sees the walker through street trees. A cone,
// not a circle round the walker, so the building wall beside the sidewalk stays.
const CUT_R_CAM = 1.5, CUT_R_WALKER = 0.8;
const GROUND_KEEP_M = 0.3;
const cutAt = { value: new THREE.Vector3() };   // the walker's middle, shared by every tile material
const f2 = (v: number) => v.toFixed(2);
function nearCut(m: THREE.Material) {
  m.onBeforeCompile = (s) => {
    s.uniforms.uKiteWalker = cutAt;
    s.vertexShader = "varying vec3 vKiteWorld;\n" + s.vertexShader.replace("#include <project_vertex>",
      "#include <project_vertex>\nvKiteWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;");
    s.fragmentShader = "varying vec3 vKiteWorld;\nuniform vec3 uKiteWalker;\n" + s.fragmentShader.replace("void main() {", `void main() {
  // Geofence (fence.ts): no tile pixels outside it, even from a coarse tile that spans it.
  float kiteU = vKiteWorld.x * ${U[0].toFixed(6)} - vKiteWorld.z * ${U[1].toFixed(6)};
  float kiteV = -vKiteWorld.x * ${U[1].toFixed(6)} - vKiteWorld.z * ${U[0].toFixed(6)};
  if (kiteU < ${f2(FENCE.uMin)} || kiteU > ${f2(FENCE.uMax)} || kiteV < ${f2(FENCE.vMin)} || kiteV > ${f2(FENCE.vMax)}) discard;
  if (vKiteWorld.y > ${f2(GROUND_KEEP_M)}) {
    vec3 ab = uKiteWalker - cameraPosition;
    float t = clamp(dot(vKiteWorld - cameraPosition, ab) / max(dot(ab, ab), 1e-4), 0.0, 1.0);
    if (distance(vKiteWorld, cameraPosition) < ${f2(NEAR_CUT_M)} ||
        distance(vKiteWorld, cameraPosition + t * ab) < mix(${f2(CUT_R_CAM)}, ${f2(CUT_R_WALKER)}, t)) discard;
  }`);
  };
  m.customProgramCacheKey = () => "kite-near-cut";
}
