import * as THREE from "three";
import {
  CAM_W, CAM_H, VFOV_DEG, EYE_HEIGHT, HEAD_TOP, DRONE_ABOVE_HEAD, DRONE_BEHIND, CHEST_HEIGHT,
  WALKER_RADIUS, WALK_SPEED, TURN_SPEED, RELAY_URL, EARPIECE_URL, FRAME_FPS, POSITION_HZ, toLatLng,
} from "./config";
import { buildWorld } from "./world";
import { buildOsmWorld, type WorldExtras } from "./osmWorld";
import { reconnectingSocket } from "./net";

// ?world=placeholder keeps the old box world; default is Moss St from OSM.
const world: ReturnType<typeof buildWorld> & WorldExtras =
  new URLSearchParams(location.search).get("world") === "placeholder" ? buildWorld() : buildOsmWorld();
const { scene } = world;

// Walker: (x east, n north), heading in compass degrees (0 = north, 90 = east).
const walker = { x: 0, n: 0, heading: 0, ...world.spawn };
const body = new THREE.Mesh(
  new THREE.CapsuleGeometry(0.25, HEAD_TOP - 0.5, 4, 8),
  new THREE.MeshLambertMaterial({ color: 0x4a6fa5 }),
);
scene.add(body);

const aspect = CAM_W / CAM_H;
const droneCam = new THREE.PerspectiveCamera(VFOV_DEG, aspect, 0.05, 500);
const chestCam = new THREE.PerspectiveCamera(VFOV_DEG, aspect, 0.05, 500);
const cams = [droneCam, chestCam]; // index = camera id on the wire

function forward() {
  const h = THREE.MathUtils.degToRad(walker.heading);
  return { x: Math.sin(h), z: -Math.cos(h) }; // three.js z points south
}

function placeCameras() {
  const f = forward();
  const wx = walker.x, wz = -walker.n;
  body.position.set(wx, HEAD_TOP / 2, wz);
  const droneY = HEAD_TOP + DRONE_ABOVE_HEAD;
  droneCam.position.set(wx - f.x * DRONE_BEHIND, droneY, wz - f.z * DRONE_BEHIND);
  droneCam.lookAt(droneCam.position.x + f.x, droneY, droneCam.position.z + f.z);
  const chestOut = 0.3; // just in front of the body capsule
  chestCam.position.set(wx + f.x * chestOut, CHEST_HEIGHT, wz + f.z * chestOut);
  chestCam.lookAt(chestCam.position.x + f.x, CHEST_HEIGHT, chestCam.position.z + f.z);
}

// --- on-screen renderer: drone view letterboxed to 4:3, chest inset bottom right
const view = document.getElementById("view")!;
const hud = document.getElementById("hud")!;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setScissorTest(true);
view.appendChild(renderer.domElement);
const resize = () => renderer.setSize(window.innerWidth, window.innerHeight);
window.addEventListener("resize", resize);
resize();

// --- capture renderer: exact 960x720 frames for perception
// ponytail: second WebGL context; switch to a render target + readRenderTargetPixels if the splat renderer needs a single context
const capture = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
capture.setPixelRatio(1);
capture.setSize(CAM_W, CAM_H, false);

function drawScreen() {
  const W = window.innerWidth, H = window.innerHeight;
  renderer.setScissor(0, 0, W, H);
  renderer.setClearColor(0x000000);
  renderer.clear();
  const h = Math.min(H, W / aspect), w = h * aspect;
  const x = (W - w) / 2, y = (H - h) / 2;
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  if (world.overlay) world.overlay.visible = true; // screen only, never in camera frames
  renderer.render(scene, droneCam);
  const iw = w * 0.25, ih = iw / aspect, pad = 10;
  renderer.setViewport(x + w - iw - pad, y + pad, iw, ih);
  renderer.setScissor(x + w - iw - pad, y + pad, iw, ih);
  renderer.render(scene, chestCam);
  if (world.overlay) world.overlay.visible = false;
}

// --- sockets
const relay = reconnectingSocket(RELAY_URL, true);
const ear = reconnectingSocket(EARPIECE_URL);

// Frames: alternate cameras so each gets FRAME_FPS. Wire format: 1 byte camera id + JPEG.
let nextCam = 0;
setInterval(() => {
  if (!relay.open || relay.buffered > 2_000_000) return; // skip rather than queue stale frames
  const id = nextCam;
  nextCam = 1 - nextCam;
  capture.render(scene, cams[id]);
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
  world.update(dt);
  step(dt);
  placeCameras();
  view.classList.toggle("hidden", black);
  if (!black) drawScreen();
  drawHud();
  requestAnimationFrame(frame);
}
placeCameras();
requestAnimationFrame(frame);

// Tell anyone wiring this up which numbers the cameras use.
console.info(`[sim] camera ${CAM_W}x${CAM_H}, vfov ${VFOV_DEG.toFixed(2)}°, eye ${EYE_HEIGHT} m`);
