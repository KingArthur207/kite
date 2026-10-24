// ?shot=<name>: films one of the shots in shots.ts for the demo video. No HUD, no debug, no sockets; nothing
// moves on its own. scripts/record_shot.py drives it frame by frame through window.kiteShot:
//   await kiteShot.ready()            models, textures and (tiles world) the first tiles are in
//   await kiteShot.frame(dt)          advance dt seconds (0 for the first frame), wait for tiles, render, PNG data URL
// Every frame is the same on every run: the clock is the recorder's fixed step, the walker and the turning car
// are scripted, and the street's own traffic only moves when a frame is stepped.
// Google attribution is drawn into every frame of the tiles world (Maps Platform policy), and the shot camera
// must stay inside the tiles geofence (fence.ts); a frame outside it counts in stats().fenceViolations.
import * as THREE from "three";
import { WALK_SPEED, DRONE_BEHIND } from "./config";
import { Path, mossPaths } from "./mossActors";
import { loadCar, loadPerson, modelsSettled } from "./actors";
import { ribbon } from "./osmStreet";
import { inFence } from "./fence";
import { ROUTE, DRONE_HEIGHT, type Shot, type Cue, type ShotCtx } from "./shots";

export const SHOT_W = 1920, SHOT_H = 1080;
export const shotCam = new THREE.PerspectiveCamera(50, SHOT_W / SHOT_H, 0.1, 3000);

const WALK_DECEL = 1.4, WALK_ACCEL = 1.0;   // m/s^2, a person braking at a curb and setting off
const WALK_CYCLE_SPEED = 1.3;               // m/s the walk clip looks right at (actors.ts)
const DRONE_LAG_S = 0.35;                   // the drone trails its station behind the head a little
const CAR = { decel: 4, accel: 2.5 };       // m/s^2, as mossActors
const FENCE_MAX_Y = 190;                    // the geofence box is 200 m tall (tilesWorld fenceInTileFrame)

interface TilesLike {
  tiles: { stats: { queued: number; downloading: number; parsing: number; visible: number }; isLoading: boolean; root: object | null;
           setResolution(c: THREE.Camera, w: number, h: number): void;
           getAttributions(): { type: string; value: unknown }[] };
  tilesStats(): { requests: number; fenceViolations: number; captureLeaks: number; fit: object };
  level(): void;
}
interface MetricsLive {
  label: string; collisions: number; done: boolean;
  obstacles: { name: string; pos: THREE.Vector3; hit: boolean; top: number }[];
}

export function startShot(opts: {
  name: string; shot: Shot; scene: THREE.Scene; renderer: THREE.WebGLRenderer;
  world: { update(dt: number): void; updateTiles?: () => void; view?: (m: "screen" | "capture") => void; overlay?: THREE.Object3D } & Partial<TilesLike>;
  worldName: string;
  walker: { x: number; n: number; heading: number };
  metrics: { step(dt: number): void; live(): MetricsLive } | null;
  place: () => void;          // main.ts placeCameras: fence clamp for the walker, body capsule
}) {
  const { shot, scene, renderer, world, walker, metrics } = opts;
  const tiles = world.tiles && world.level ? (world as TilesLike) : null;
  shotCam.fov = shot.fov;
  shotCam.updateProjectionMatrix();
  tiles?.tiles.setResolution(shotCam, SHOT_W, SHOT_H);
  if (shot.fog && scene.fog instanceof THREE.Fog) [scene.fog.near, scene.fog.far] = shot.fog;
  if (world.overlay) world.overlay.visible = false;
  renderer.setPixelRatio(1);
  renderer.setSize(SHOT_W, SHOT_H, false);
  renderer.setScissorTest(false);

  // Output: the 3D frame plus labels and attribution, composited on a 2D canvas that is also what the page shows.
  const out = document.createElement("canvas");
  out.width = SHOT_W; out.height = SHOT_H;
  out.style.cssText = "position:fixed;inset:0;width:100%;height:100%;object-fit:contain;background:#000";
  for (const el of Array.from(document.body.children)) if (el instanceof HTMLElement && el.tagName !== "SCRIPT") el.style.display = "none";
  document.body.appendChild(out);
  const g = out.getContext("2d")!;
  const logo = new Image();
  const logoReady = new Promise((ok) => { logo.onload = ok; logo.onerror = ok; });
  logo.src = "/google/maps_logo.svg";

  // --- cast: the walker (with a white cane), the drone, the turning car
  const walkerObj = new THREE.Group();
  scene.add(walkerObj);
  walkerObj.visible = !shot.pov;
  walkerObj.add(cane());
  let walk: THREE.AnimationAction | null = null, idle: THREE.AnimationAction | null = null, mixer: THREE.AnimationMixer | null = null;
  const loads: Promise<unknown>[] = [
    loadPerson("casual", false).then((p) => {
      walkerObj.add(p.obj);
      mixer = p.mixer;
      mixer.stopAllAction();
      walk = mixer.clipAction(p.clips.find((c) => /walk/i.test(c.name)) ?? p.clips[0]).play();
      idle = mixer.clipAction(p.clips.find((c) => /idle/i.test(c.name)) ?? p.clips[0]).play();
    }),
    logoReady,
  ];
  const drone = buildDrone();
  drone.visible = !shot.hideDrone;
  scene.add(drone);

  const car = shot.carAtZebra === undefined ? null : scriptedCar(shot.carAtZebra);
  if (car) loads.push(loadCar("sedan").then(({ obj }) => { car.obj.add(obj); }));
  if (car) scene.add(car.obj);

  // --- route line and pulse rings
  const routeLine = shot.route ? routeMesh(shot.route.width, tiles ? 0.35 : 0.18) : null;
  if (routeLine) scene.add(routeLine.mesh);
  const rings = [0, 1].map(() => {
    const m = new THREE.Mesh(new THREE.RingGeometry(0.88, 1, 64), new THREE.MeshBasicMaterial({ transparent: true, depthTest: false, depthWrite: false }));
    m.rotation.x = -Math.PI / 2;
    m.renderOrder = 20;
    m.visible = false;
    scene.add(m);
    return m;
  });

  const marker = new THREE.Mesh(new THREE.RingGeometry(0.55, 0.75, 48), new THREE.MeshBasicMaterial({ color: 0x33ddff, transparent: true, opacity: 0.9, depthTest: false, depthWrite: false }));
  marker.rotation.x = -Math.PI / 2;
  marker.renderOrder = 19;
  marker.visible = !!shot.marker;
  scene.add(marker);

  // --- state
  let t = 0, s = shot.walk?.from ?? 0, v = 0, ready = false, fenceViolations = 0, frames = 0;
  const cues: Cue[] = [...(shot.cues ?? [])];
  let seenCollisions = 0;
  const smooth = { walker: new THREE.Vector3(), heading: 0, drone: new THREE.Vector3(), droneHeading: 0, init: false };

  function stepWalker(dt: number) {
    const w = shot.walk!;
    const stop = w.stops.find((st) => t < st.until && s <= st.at + 1e-3);
    const vMax = stop ? Math.sqrt(2 * WALK_DECEL * Math.max(0, stop.at - s)) : Infinity;
    v = Math.min(WALK_SPEED, vMax, v + WALK_ACCEL * dt);
    s = Math.min(ROUTE.len, s + v * dt);
    if (stop && s >= stop.at - 0.005) { s = stop.at; v = 0; }
    const p = ROUTE.at(s);
    walker.x = p.x; walker.n = p.n;
    walker.heading = headingAlong(ROUTE, s);
  }

  function ctx(dt: number): ShotCtx {
    const here = new THREE.Vector3(walker.x, 0, -walker.n);
    const k = smooth.init ? 1 - Math.exp(-dt / 0.25) : 1;
    smooth.walker.lerp(here, k);
    smooth.heading = lerpAngle(smooth.heading, walker.heading, smooth.init ? 1 - Math.exp(-dt / 0.4) : 1);
    const h = THREE.MathUtils.degToRad(smooth.heading);
    const station = here.clone().add(new THREE.Vector3(-Math.sin(h) * DRONE_BEHIND, DRONE_HEIGHT, Math.cos(h) * DRONE_BEHIND));
    const kd = smooth.init ? 1 - Math.exp(-dt / DRONE_LAG_S) : 1;
    smooth.drone.lerp(station, kd);
    smooth.droneHeading = lerpAngle(smooth.droneHeading, smooth.heading, kd);
    smooth.init = true;
    const bob = 0.012 * Math.sin(t * 2 * Math.PI * 0.6);
    return { t, walker: smooth.walker.clone(), heading: smooth.heading, drone: smooth.drone.clone().setY(smooth.drone.y + bob) };
  }

  function placeCast(k: ShotCtx, dt: number) {
    walkerObj.position.set(walker.x, 0, -walker.n);
    const h = THREE.MathUtils.degToRad(k.heading);
    walkerObj.rotation.y = Math.atan2(Math.sin(h), -Math.cos(h));
    const speed = metrics ? (dt > 0 ? walkerObj.userData.last?.distanceTo(walkerObj.position) / dt : 0) || 0 : v;
    walkerObj.userData.last = walkerObj.position.clone();
    marker.position.set(walker.x, 0.2, -walker.n);
    if (mixer && walk && idle) {
      const wgt = Math.min(1, speed / (WALK_SPEED * 0.6));
      walk.setEffectiveWeight(wgt);
      idle.setEffectiveWeight(1 - wgt);
      walk.timeScale = Math.max(speed, 0.4) / WALK_CYCLE_SPEED;
      mixer.update(dt);
    }
    drone.position.copy(k.drone);
    drone.rotation.y = Math.atan2(Math.sin(THREE.MathUtils.degToRad(smooth.droneHeading)), -Math.cos(THREE.MathUtils.degToRad(smooth.droneHeading)));
    for (const p of drone.userData.props as THREE.Object3D[]) p.rotation.y += dt * 140;   // motion-blur-free spin, reads as a blur disc
    car?.place(t);
    if (routeLine && shot.route) {
      const [a, b] = shot.route.reveal;
      routeLine.reveal(THREE.MathUtils.smoothstep(t, a, b));
    }
    // Rings: two staggered pulses at the walker while a ring cue is up.
    const ringCue = cues.find((c) => c.ring && t >= c.from && t < c.to);
    rings.forEach((r, i) => {
      r.visible = !!ringCue;
      if (!ringCue) return;
      const ph = (((t - ringCue.from) / 1.2 + i * 0.5) % 1 + 1) % 1;
      r.position.set(walker.x, tiles ? 0.25 : 0.2, -walker.n);
      r.scale.setScalar(0.4 + 2.6 * ph);
      const m = r.material as THREE.MeshBasicMaterial;
      m.color.set(ringCue.color ?? "#ffb020");
      m.opacity = (1 - ph) * fade(ringCue);
    });
  }

  const fade = (c: Cue) => Math.min(1, (t - c.from) / 0.3, (c.to - t) / 0.3);

  function draw() {
    g.drawImage(renderer.domElement, 0, 0, SHOT_W, SHOT_H);
    const head = new THREE.Vector3(walker.x, 2.5, -walker.n);
    let stack = 0;
    for (const c of cues) if (t >= c.from && t < c.to) label(head, c.text, c.color ?? "#fff", fade(c), stack++);
    if (metrics && !shot.pov) {
      const live = metrics.live();
      for (const o of live.obstacles) {
        const near = o.pos.distanceTo(new THREE.Vector3(walker.x, 0, -walker.n));
        if (near < 22 || o.hit) label(o.pos.clone().setY(o.top + 0.4), o.hit ? `Hit: ${o.name}` : o.name, o.hit ? "#ff6b5e" : "#e8eef5", 0.95, 0, 26);
      }
      caption(`Sim metrics walk, seed 1, 2.4x speed\n${live.label}\ncollisions: ${live.collisions}`);
    }
    if (tiles) attribution();
  }

  function label(at: THREE.Vector3, text: string, color: string, alpha: number, stack: number, size = 34) {
    const p = at.clone().project(shotCam);
    if (p.z > 1) return;
    const x = (p.x + 1) / 2 * SHOT_W, y = (1 - p.y) / 2 * SHOT_H - stack * (size * 1.9);
    g.save();
    g.globalAlpha = Math.max(0, alpha);
    g.font = `600 ${size}px -apple-system, "Helvetica Neue", Arial, sans-serif`;
    const w = g.measureText(text).width + size * 0.9, h = size * 1.5;
    g.fillStyle = "rgba(12,16,22,0.74)";
    g.beginPath(); g.roundRect(x - w / 2, y - h, w, h, h / 2); g.fill();
    g.fillStyle = color; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(text, x, y - h / 2 + 1);
    g.restore();
  }

  function caption(text: string) {
    g.save();
    g.font = `500 28px -apple-system, "Helvetica Neue", Arial, sans-serif`;
    const lines = text.split("\n");
    const w = Math.max(...lines.map((l) => g.measureText(l).width)) + 40;
    g.fillStyle = "rgba(12,16,22,0.7)";
    g.beginPath(); g.roundRect(36, 36, w, lines.length * 38 + 24, 14); g.fill();
    g.fillStyle = "#fff"; g.textBaseline = "top";
    lines.forEach((l, i) => g.fillText(l, 56, 48 + i * 38));
    g.restore();
  }

  // Google Maps logo and the visible tiles' data credits, bottom left, on every tiles frame.
  function attribution() {
    const credits = tiles!.tiles.getAttributions().filter((a) => a.type === "string").map((a) => String(a.value)).join("; ");
    g.save();
    const lh = 30, lw = logo.naturalWidth ? (logo.naturalWidth / logo.naturalHeight) * lh : 0;
    if (lw) g.drawImage(logo, 24, SHOT_H - lh - 18, lw, lh);
    g.font = `20px -apple-system, "Helvetica Neue", Arial, sans-serif`;
    g.fillStyle = "#fff"; g.shadowColor = "#000"; g.shadowBlur = 4; g.textBaseline = "bottom";
    g.fillText(`${credits}  ·  3D city: Google Maps. Cars, people and the walker: Kite sim.`, 24 + lw + 16, SHOT_H - 20);
    g.restore();
  }

  async function settle(timeoutMs = 20000) {
    if (!tiles) return;
    const end = performance.now() + timeoutMs;
    let quiet = 0;
    while (performance.now() < end && quiet < 2) {
      world.updateTiles?.();
      await sleep(30);
      const st = tiles.tiles.stats;
      const idle = tiles.tiles.root && st.visible > 0 && st.queued + st.downloading + st.parsing === 0 && !tiles.tiles.isLoading;
      quiet = idle ? quiet + 1 : 0;
    }
  }

  function aim(dt: number) {
    const k = ctx(dt);
    placeCast(k, dt);
    const cam = shot.camera(k);
    shotCam.position.copy(cam.pos);
    shotCam.up.copy(cam.up ?? new THREE.Vector3(0, 1, 0));
    shotCam.lookAt(cam.look);
    shotCam.updateMatrixWorld();
    if (tiles && !(inFence(cam.pos.x, -cam.pos.z) && cam.pos.y < FENCE_MAX_Y) && fenceViolations++ === 0)
      console.error(`[shot] camera left the tiles geofence at t=${t.toFixed(2)}`);
  }

  function advance(dt: number) {
    t += dt;
    if (metrics) {
      metrics.step(dt);
      if (shot.pov && shot.walk) stepWalker(dt);   // overrides the metrics walker; its obstacles still move
      const live = metrics.live();
      if (!shot.pov && live.collisions > seenCollisions) cues.push({ from: t, to: t + 1.6, text: "Collision", ring: true, color: "#ff5a4e" });
      seenCollisions = live.collisions;
    } else {
      world.update(dt);
      if (shot.walk) stepWalker(dt);
    }
    opts.place();
  }

  const init = (async () => {
    if (shot.walk) stepWalker(0);
    opts.place();
    await Promise.all([...loads, modelsSettled()]);
    const w = window as unknown as { __kiteLook?: string; __kiteReal?: string; __kitePhoto?: string };
    while (w.__kiteLook !== "ready" || w.__kiteReal === "loading" || w.__kitePhoto === "loading") await sleep(100);
    await sleep(800);   // let late model callbacks add their actors before frame 0
    if (tiles) {
      // Level the tiles once, then never again during the shot, so the ground can't jump between frames.
      // level() raycasts only the tiles on screen, so look straight down on the whole route (inside the fence).
      shotCam.fov = 75;
      shotCam.updateProjectionMatrix();
      shotCam.position.set(18, 80, 20);
      shotCam.up.set(-1, 0, -1).normalize();
      shotCam.lookAt(18, 0, 20);
      shotCam.updateMatrixWorld();
      await settle();
      scene.updateMatrixWorld(true);          // nothing has rendered yet, so new tiles still have stale matrices
      tiles.level();
      scene.updateMatrixWorld(true);
      tiles.level();
      shotCam.fov = shot.fov;
      shotCam.updateProjectionMatrix();
      const fit = tiles.tilesStats().fit as { n: number; residual: number; tiltDeg: number };
      if (!fit.n) throw new Error("[shot] could not level the tiles (no ground hits under the route)");
      console.info(`[shot] tiles levelled: ${fit.n} points, 90th pct residual ${fit.residual.toFixed(2)} m, tilt ${fit.tiltDeg.toFixed(2)} deg`);
    }
    ready = true;
  })();
  init.catch((e) => console.error("[shot] init failed", e));

  const api = {
    name: opts.name, world: opts.worldName, duration: shot.duration, still: shot.still,
    ready: () => ready,
    /** Advance dt seconds of shot time, wait for tiles, render, and return the frame as a PNG data URL. */
    async frame(dt: number, render = true): Promise<string> {
      if (!ready) throw new Error("kiteShot not ready");
      advance(dt);
      aim(dt);
      if (!render) return "";   // previews skip ahead (record_shot.py --preview); tiles then load late
      await settle();
      world.view?.("screen");
      renderer.render(scene, shotCam);
      draw();
      frames++;
      return out.toDataURL("image/png");
    },
    stats: () => ({ t, frames, fenceViolations, car: car?.obj.position.toArray(), walker: [walker.x, walker.n], tiles: tiles?.tilesStats() ?? null }),
  };
  (window as unknown as { kiteShot: typeof api }).kiteShot = api;
  console.info(`[shot] ${opts.name} on ${opts.worldName}, ${shot.duration} s`);
}

// --- helpers

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function lerpAngle(a: number, b: number, k: number) {
  const d = ((b - a + 540) % 360) - 180;
  return (a + d * k + 360) % 360;
}

/** Compass heading along a path at s, from a chord either side so corners turn smoothly. */
function headingAlong(path: Path, s: number, half = 0.8) {
  const a = path.at(s - half), b = path.at(s + half);
  return ((Math.atan2(b.x - a.x, b.n - a.n) * 180) / Math.PI + 360) % 360;
}

/** The car turning left from Howard's left lane into Moss, driven by shot time: cruise on Howard, slow for
 *  the turn, and its middle crosses the zebra at `atZebra` seconds. It does not yield (the demo hazard). */
function scriptedCar(atZebra: number) {
  const { turn, zebra } = mossPaths();
  const DS = 0.25, n = Math.ceil(turn.len / DS);
  // Speed limit along the path: each segment's speed, braking ahead of slower ones, accelerating out of them.
  const vs: number[] = [];
  for (let i = 0; i <= n; i++) {
    const sv = i * DS;
    let lim = turn.speed(sv);
    for (let k = 0; k < turn.pts.length; k++) {
      const d = turn.s[k] - sv;
      if (d > 0) lim = Math.min(lim, Math.sqrt(turn.v[k] ** 2 + 2 * CAR.decel * d));
    }
    vs.push(i ? Math.min(lim, Math.sqrt(vs[i - 1] ** 2 + 2 * CAR.accel * DS)) : lim);
  }
  const times = [0];
  for (let i = 1; i <= n; i++) times.push(times[i - 1] + (2 * DS) / (vs[i - 1] + vs[i]));
  const sAt = (time: number) => {
    if (time <= 0) return 0;
    let i = 1;
    while (i < n && times[i] < time) i++;
    const u = Math.min(1, (time - times[i - 1]) / (times[i] - times[i - 1]));
    return (i - 1 + u) * DS;
  };
  const tZebra = times[Math.round(turn.nearest(zebra.x, zebra.y).s / DS)];
  const obj = new THREE.Group();
  return {
    obj,
    place(t: number) {
      const sv = sAt(tZebra + t - atZebra);
      const p = turn.at(sv), h = headingAlong(turn, sv, 1.5) * Math.PI / 180;
      obj.position.set(p.x, 0, -p.n);
      obj.rotation.y = Math.atan2(Math.sin(h), -Math.cos(h));
    },
  };
}

/** The demo route as a ribbon, resampled every 0.5 m so it can be revealed from the start. */
function routeMesh(width: number, y: number) {
  const pts: [number, number][] = [];
  for (let sv = 0; sv < ROUTE.len; sv += 0.5) { const p = ROUTE.at(sv); pts.push([p.x, p.n]); }
  const last = ROUTE.at(ROUTE.len); pts.push([last.x, last.n]);
  const quads = ribbon(pts, width);
  const pos: number[] = [];
  for (const q of quads) for (const i of [0, 1, 2, 0, 2, 3]) pos.push(q[i][0], y, -q[i][1]);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    color: 0x33ddff, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: -4, fog: false,
  }));
  mesh.renderOrder = 5;
  return { mesh, reveal: (f: number) => geo.setDrawRange(0, 6 * Math.round(quads.length * f)) };
}

/** A white cane held in the right hand, tip on the ground about a metre ahead. Model frame: +z forward. */
function cane() {
  const hand = new THREE.Vector3(-0.24, 0.82, 0.22), tip = new THREE.Vector3(-0.12, 0.02, 1.0);
  const d = tip.clone().sub(hand), len = d.length();
  const g = new THREE.Group();
  const white = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, len * 0.85, 8), new THREE.MeshStandardMaterial({ color: 0xf4f4f4, roughness: 0.4 }));
  const red = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, len * 0.15, 8), new THREE.MeshStandardMaterial({ color: 0xc8202a, roughness: 0.4 }));
  white.position.y = len * 0.075; red.position.y = -len * 0.425;
  g.add(white, red);
  g.position.copy(hand).addScaledVector(d, 0.5);
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  return g;
}

/** A Tello-sized quadcopter from primitives (about 18 cm across with its prop guards). Front is +z. */
function buildDrone() {
  const g = new THREE.Group();
  const shell = new THREE.MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.45 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x1d1f22, roughness: 0.6 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.04, 0.095), shell);
  const top = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.01, 0.06), dark);
  top.position.y = 0.025;
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.01, 12), dark);
  lens.rotation.x = Math.PI / 2; lens.position.set(0, 0, 0.05);
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.004, 8, 6), new THREE.MeshBasicMaterial({ color: 0x40ff70 }));
  led.position.set(0.02, 0.0, 0.048);
  g.add(body, top, lens, led);
  const props: THREE.Object3D[] = [];
  for (const [x, z] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const cx = x * 0.06, cz = z * 0.06;
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.01, 0.085), shell);
    arm.position.set(cx / 2, 0.005, cz / 2); arm.rotation.y = Math.atan2(cx, cz);
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.018, 12), dark);
    motor.position.set(cx, 0.012, cz);
    const guard = new THREE.Mesh(new THREE.TorusGeometry(0.038, 0.003, 6, 32), shell);
    guard.rotation.x = Math.PI / 2; guard.position.set(cx, 0.012, cz);
    const prop = new THREE.Mesh(new THREE.BoxGeometry(0.068, 0.002, 0.009), dark);
    prop.position.set(cx, 0.023, cz); prop.rotation.y = x * z;
    g.add(arm, motor, guard, prop);
    props.push(prop);
  }
  g.userData.props = props;
  return g;
}
