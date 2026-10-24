import * as THREE from "three";
import {
  CAM_W, CAM_H, VFOV_DEG, EYE_HEIGHT, HEAD_TOP, DRONE_ABOVE_HEAD, DRONE_BEHIND, CHEST_HEIGHT,
  WALKER_RADIUS, WALK_SPEED, TURN_SPEED, RELAY_URL, EARPIECE_URL, FRAME_FPS, POSITION_HZ, toLatLng,
} from "./config";
import { buildWorld } from "./world";
import { buildOsmWorld, type WorldExtras } from "./osmWorld";
import { reconnectingSocket } from "./net";
import { startMetrics } from "./metrics";
import { buildTilesWorld, TILES_KEY } from "./tilesWorld";
import { clampToFence } from "./fence";

// ?world=tiles (needs VITE_GOOGLE_MAPS_API_KEY): Google Photorealistic 3D Tiles over the
// Moss colliders and traffic, geofenced to Moss St and the Howard corner (Google's permission, see
// tilesWorld.ts and docs/google-permission.md). By default the capture cameras render OSM Moss (YOLO recall on tile
// frames is far lower) and tiles show on screen only; ?detector=tiles puts tiles in capture too. ?world=moss (default): Moss St from OSM with textured facades
// everywhere. ?world=placeholder: the old box world.
// Metrics mode always runs on OSM Moss, with the street emptied of traffic (it scripts its own).
// ?relay= and ?ear= override the socket URLs (empty = offline).
const query = new URLSearchParams(location.search);
const metricsOn = query.get("metrics") === "1";
const aspect = CAM_W / CAM_H;
const droneCam = new THREE.PerspectiveCamera(VFOV_DEG, aspect, 0.05, 500);
const chestCam = new THREE.PerspectiveCamera(VFOV_DEG, aspect, 0.05, 500);
const cams = [droneCam, chestCam]; // index = camera id on the wire
// ?debug=1: T swaps the on-screen drone view for a north-up view from TOP_VIEW_M over the walker,
// low enough that its footprint (about 110 x 150 m) stays inside the tiles geofence.
const TOP_VIEW_M = 120;
const debug = query.get("debug") === "1";
const topCam = new THREE.PerspectiveCamera(50, aspect, 1, 1000);
let topView = false;
// Tiles world: the big view is a chase camera above and behind the walker. At drone height the
// photogrammetry in Moss's narrow alley is melted blobs; from up here it reads as a street.
// Screen only: it is never a capture camera.
const chaseCam = new THREE.PerspectiveCamera(55, aspect, 0.1, 1500);
// ?chase=behind,up,ahead (metres) tunes it.
const [CHASE_BEHIND, CHASE_UP, CHASE_AHEAD] = (query.get("chase") ?? "4,3,10").split(",").map(Number);
const FENCE_INSET_M = 10;   // walker stays this far inside the tiles geofence, so every camera is inside too

// --- capture renderer: exact 960x720 frames for perception
// ponytail: second WebGL context; switch to a render target + readRenderTargetPixels if the splat renderer needs a single context
const capture = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
capture.setPixelRatio(1);
capture.setSize(CAM_W, CAM_H, false);

let worldName = metricsOn ? "moss" : query.get("world") ?? "moss";   // textured OSM street by default; ?world=tiles for Google 3D (looks crumpled at street level)
if (worldName === "tiles" && !TILES_KEY) {
  console.warn("[sim] no VITE_GOOGLE_MAPS_API_KEY (run sh sim/tiles_env.sh); using ?world=moss");
  worldName = "moss";
}
const walkers = () => [walker];                                       // cars yield to the walker too
const world: ReturnType<typeof buildWorld> & WorldExtras & { updateTiles?: () => void; view?: (m: "screen" | "capture") => void } =
  worldName === "placeholder" ? buildWorld()
  : worldName === "tiles" ? buildTilesWorld({
    walkers, cameras: [...cams, chaseCam, ...(debug ? [topCam] : [])], capture, debug,
    detector: query.get("detector") === "tiles" ? "tiles" : "osm",   // OSM capture by default: YOLO recall on tile frames is far lower (PR #41)
  })
  : buildOsmWorld({ actors: !metricsOn, walkers });
world.tune?.(capture);   // tone mapping, colour space and shadows; the screen renderer gets the same
if ("tilesStats" in world) (window as unknown as { __kiteTiles: unknown }).__kiteTiles = world;
const { scene } = world;

// Walker: (x east, n north), heading in compass degrees (0 = north, 90 = east).
const walker = { x: 0, n: 0, heading: 0, ...world.spawn };
(window as unknown as { __kiteWalker: unknown }).__kiteWalker = walker;   // dev checks (screenshots) move it
const body = new THREE.Mesh(
  new THREE.CapsuleGeometry(0.25, HEAD_TOP - 0.5, 4, 8),
  new THREE.MeshLambertMaterial({ color: 0x4a6fa5 }),
);
scene.add(body);
const metrics = metricsOn ? startMetrics({ walker, scene, walls: world.walls! }) : null;

function forward() {
  const h = THREE.MathUtils.degToRad(walker.heading);
  return { x: Math.sin(h), z: -Math.cos(h) }; // three.js z points south
}

function placeCameras() {
  if (worldName === "tiles") Object.assign(walker, clampToFence(walker.x, walker.n, FENCE_INSET_M));
  const f = forward();
  const wx = walker.x, wz = -walker.n;
  body.position.set(wx, HEAD_TOP / 2, wz);
  const droneY = HEAD_TOP + DRONE_ABOVE_HEAD;
  droneCam.position.set(wx - f.x * DRONE_BEHIND, droneY, wz - f.z * DRONE_BEHIND);
  droneCam.lookAt(droneCam.position.x + f.x, droneY, droneCam.position.z + f.z);
  const chestOut = 0.3; // just in front of the body capsule
  chestCam.position.set(wx + f.x * chestOut, CHEST_HEIGHT, wz + f.z * chestOut);
  chestCam.lookAt(chestCam.position.x + f.x, CHEST_HEIGHT, chestCam.position.z + f.z);
  chaseCam.position.set(wx - f.x * CHASE_BEHIND, CHASE_UP, wz - f.z * CHASE_BEHIND);
  chaseCam.lookAt(wx + f.x * CHASE_AHEAD, 0.5, wz + f.z * CHASE_AHEAD);
  topCam.position.set(wx, TOP_VIEW_M, wz);
  topCam.up.set(0, 0, -1);                  // north up
  topCam.lookAt(wx, 0, wz);
}

// --- on-screen renderer: drone view letterboxed to 4:3, chest inset bottom right
const view = document.getElementById("view")!;
const hud = document.getElementById("hud")!;
const renderer = new THREE.WebGLRenderer({ antialias: true });
world.tune?.(renderer);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setScissorTest(true);
view.appendChild(renderer.domElement);
const resize = () => renderer.setSize(window.innerWidth, window.innerHeight);
window.addEventListener("resize", resize);
resize();

function drawScreen() {
  const W = window.innerWidth, H = window.innerHeight;
  renderer.setScissor(0, 0, W, H);
  renderer.setClearColor(0x000000);
  renderer.clear();
  const h = Math.min(H, W / aspect), w = h * aspect;
  const x = (W - w) / 2, y = (H - h) / 2;
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  world.view?.("screen");                          // tiles on (they may be off for ?detector=osm capture)
  if (world.overlay) world.overlay.visible = true; // screen only, never in camera frames
  renderer.render(scene, topView ? topCam : worldName === "tiles" ? chaseCam : droneCam);
  const iw = w * 0.25, ih = iw / aspect, pad = 10;
  renderer.setViewport(x + w - iw - pad, y + pad, iw, ih);
  renderer.setScissor(x + w - iw - pad, y + pad, iw, ih);
  renderer.render(scene, chestCam);
  if (world.overlay) world.overlay.visible = false;
}

// --- sockets
// Blind/oracle metrics stay offline so they never feed a shared earpiece or detector.
const relay = reconnectingSocket(metrics?.offline ? "" : query.get("relay") ?? RELAY_URL, true);
const ear = reconnectingSocket(metrics?.offline ? "" : query.get("ear") ?? EARPIECE_URL);

// Frames: alternate cameras so each gets FRAME_FPS. Wire format: 1 byte camera id + JPEG.
let nextCam = 0;
setInterval(() => {
  if (!relay.open || relay.buffered > 2_000_000) return; // skip rather than queue stale frames
  const id = nextCam;
  nextCam = 1 - nextCam;
  world.view?.("capture");     // ?detector=osm: no Google tile pixels in frames for the detector
  capture.render(scene, cams[id]);
  world.view?.("screen");
  capture.domElement.toBlob(async (blob) => {
    if (!blob) return;
    const jpeg = new Uint8Array(await blob.arrayBuffer());
    const msg = new Uint8Array(jpeg.length + 1);
    msg[0] = id;
    msg.set(jpeg, 1);
    relay.send(msg.buffer);
  }, "image/jpeg", 0.85);
}, 1000 / (FRAME_FPS * 2));

// Position (earpiece) and pose (perception) at POSITION_HZ.
setInterval(() => {
  const { lat, lng } = toLatLng(walker.x, walker.n);
  ear.json({ type: "position", source: "sim", heading: walker.heading, place: world.place ?? "sim corner",
             lat, lng, x: walker.x, y: walker.n });
  const camHeights = { drone: HEAD_TOP + DRONE_ABOVE_HEAD, chest: CHEST_HEIGHT };
  for (const [camera, cam_height] of Object.entries(camHeights)) {
    relay.json({ type: "pose", camera, x: walker.x, y: walker.n, heading: walker.heading, cam_height, pitch: 0 });
  }
}, 1000 / POSITION_HZ);

// --- input
const keys = new Set<string>();
let black = false;
window.addEventListener("keydown", (e) => {
  if (e.key.startsWith("Arrow")) e.preventDefault();
  keys.add(e.key);
  const k = e.key.toLowerCase();
  if (k === "b") black = true;
  if (k === "v") black = false;
  if (k === "t" && debug) topView = !topView;
  if (k === "h" && !e.repeat) {
    // DEBUG ONLY (skeleton gate): real hazards come from the detector, never the sim's object list.
    ear.json({ type: "hazard", source: "sim", kind: "crowd", bearing: 0, distance: 5 });
    lastDebug = performance.now();
  }
});
window.addEventListener("keyup", (e) => keys.delete(e.key));
window.addEventListener("blur", () => keys.clear());

// --- movement + collision (AABB on the ground plane)
let collisions = 0;
let touching = false;
let status = "";
let lastDebug = -1e9;

function hits(x: number, n: number, boxes: THREE.Box3[], r = WALKER_RADIUS): object[] {
  const z = -n;
  const inBoxes = boxes.filter((b) => x > b.min.x - r && x < b.max.x + r && z > b.min.z - r && z < b.max.z + r);
  return [...inBoxes, ...(world.walls?.(x, n, r) ?? [])]; // walls: building faces and curbs
}

function step(dt: number) {
  if (keys.has("ArrowLeft")) walker.heading -= TURN_SPEED * dt;
  if (keys.has("ArrowRight")) walker.heading += TURN_SPEED * dt;
  walker.heading = ((walker.heading % 360) + 360) % 360;

  const dir = (keys.has("ArrowUp") ? 1 : 0) - (keys.has("ArrowDown") ? 1 : 0);
  const boxes = world.solids();
  const here = hits(walker.x, walker.n, boxes);
  let blocked = false;
  let moved = false;
  if (dir !== 0) {
    const f = forward();
    const nx = walker.x + f.x * dir * WALK_SPEED * dt;
    const nn = walker.n - f.z * dir * WALK_SPEED * dt;
    // Block only on boxes we aren't already inside, so a car that drove into us can't pin us.
    blocked = hits(nx, nn, boxes).some((b) => !here.includes(b));
    if (!blocked) { walker.x = nx; walker.n = nn; moved = true; }
  }
  // Count each new contact once; the 10 cm margin stops creeping up to a wall from re-counting.
  const contact = hits(walker.x, walker.n, boxes, WALKER_RADIUS + 0.1).length > 0;
  if (contact && !touching) collisions++;
  touching = contact;

  const s = moved ? "walking" : "stopped";
  if (s !== status) { status = s; ear.json({ type: "status", source: "sim", state: s }); }
}

function drawHud() {
  const { lat, lng } = toLatLng(walker.x, walker.n);
  const dbg = performance.now() - lastDebug < 1500 ? "  (debug hazard sent)" : "";
  hud.textContent =
    `collisions: ${collisions}\n` +
    `pos: ${walker.x.toFixed(1)} E, ${walker.n.toFixed(1)} N   heading ${walker.heading.toFixed(0)}°\n` +
    `lat/lng: ${lat.toFixed(6)}, ${lng.toFixed(6)}\n` +
    `relay ${relay.open ? "on" : "off"}   earpiece ${ear.open ? "on" : "off"}   ${status}\n` +
    (black ? "BLACK SCREEN (V to reveal)\n" : "") +
    `\narrows walk/turn   B black screen   V reveal\nH  DEBUG: send one fake crowd hazard${dbg}`;
}

let last = performance.now();
function frame(now: number) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (metrics) metrics.step(dt);
  else { world.update(dt); step(dt); }
  placeCameras();
  world.updateTiles?.();
  view.classList.toggle("hidden", black);
  if (!black) drawScreen();
  if (metrics) hud.textContent = metrics.hud(); else drawHud();
  requestAnimationFrame(frame);
}
placeCameras();
requestAnimationFrame(frame);

// Tell anyone wiring this up which numbers the cameras use.
console.info(`[sim] camera ${CAM_W}x${CAM_H}, vfov ${VFOV_DEG.toFixed(2)}°, eye ${EYE_HEIGHT} m`);
