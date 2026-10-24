// glTF cars and pedestrians, so the real detector (YOLO) has something to find in the sim frames.
// Models load async; until they arrive the street is just empty. Credits: public/models/CREDITS.md.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import * as SkeletonUtils from "three/examples/jsm/utils/SkeletonUtils.js";

// yaw turns the model so its front faces +z before we steer it (checked by eye in the sim).
const CARS = {
  sedan:  { file: "car_sedan.glb",  length: 4.6, yaw: 0 },
  sedan2: { file: "car_sedan2.glb", length: 4.3, yaw: 0 },
  suv:    { file: "car_suv.glb",    length: 4.8, yaw: 0 },
  pickup: { file: "car_pickup.glb", length: 5.3, yaw: 0 },
  rover:  { file: "car_rover.glb",  length: 4.9, yaw: 0 },
  bus:    { file: "bus.glb",        length: 12,  yaw: 0 },
};
const PEOPLE = {
  man:      { file: "ped_man.glb",          height: 1.78 },
  woman:    { file: "ped_casual_woman.glb", height: 1.65 },
  casual:   { file: "ped_casual.glb",       height: 1.75 },
  business: { file: "ped_business.glb",     height: 1.82 },
  dress:    { file: "ped_dress.glb",        height: 1.68 },
  woman2:   { file: "ped_woman2.glb",       height: 1.70 },
};
export type CarName = keyof typeof CARS;
export type PersonName = keyof typeof PEOPLE;

// Lanes match world.ts: main road x in [2.2, 9.2], cross street z in [-40, -33], crosswalk at x in [-2, 2].
// dir is the travel direction (+1 or -1 along axis); wrap is the loop range along the axis.
const CAR_SPAWNS: { model: CarName; axis: "x" | "z"; lane: number; at: number; dir: 1 | -1; speed: number; wrap: [number, number] }[] = [
  { model: "sedan",  axis: "z", lane: 4,     at: 10,  dir: -1, speed: 11, wrap: [-90, 30] }, // northbound
  { model: "suv",    axis: "z", lane: 4,     at: -40, dir: -1, speed: 11, wrap: [-90, 30] },
  { model: "bus",    axis: "z", lane: 7.5,   at: -60, dir: 1,  speed: 8,  wrap: [-90, 30] }, // southbound
  { model: "rover",  axis: "z", lane: 7.5,   at: 0,   dir: 1,  speed: 9,  wrap: [-90, 30] },
  { model: "pickup", axis: "x", lane: -35,   at: -40, dir: 1,  speed: 7,  wrap: [-60, 60] }, // cross street, eastbound
  { model: "sedan2", axis: "x", lane: -38.5, at: 30,  dir: -1, speed: 7,  wrap: [-60, 60] }, // westbound
];

// Sidewalk loop: north on the east half, back south on the west half, through the crosswalk both ways.
const SIDEWALK_LOOP: [number, number][] = [[1.0, 8], [1.0, -62], [-1.0, -62], [-1.0, 8]];
const WALKERS: { model: PersonName; s0: number; speed: number }[] = [
  { model: "man", s0: 20, speed: 1.4 },
  { model: "woman", s0: 45, speed: 1.25 },
  { model: "business", s0: 85, speed: 1.5 },
  { model: "dress", s0: 110, speed: 1.3 },
  { model: "woman2", s0: 128, speed: 1.35 },
];
const STANDERS: { model: PersonName; x: number; z: number; face: number }[] = [
  { model: "casual", x: -0.8, z: -43, face: Math.PI / 2 },
];

const STOP_LINE_X: [number, number] = [-2.8, 9.9]; // cross-street stop lines, before the crosswalk / past the main road
const CROSSWALK_HALF = 2.6;      // m: |x| that counts as on the crosswalk
const CROSSWALK_REACH = 4.5;     // m: how far from the car's lane a walker still blocks it
const MAIN_ROAD_LOOKAHEAD = 18;  // m: main-road car this close to the cross street keeps it shut
const FOLLOW_GAP = 4;            // m: bumper gap a car keeps behind the one ahead
const PERSON_RADIUS = 0.25;
const WALK_CYCLE_SPEED = 1.3; // m/s the walk clips look right at; timeScale = speed / this

interface Car { obj: THREE.Object3D; axis: "x" | "z"; dir: 1 | -1; speed: number; wrap: [number, number]; half: number; yields: boolean }
interface Person { obj: THREE.Object3D; mixer: THREE.AnimationMixer; s: number; speed: number }

const loader = new GLTFLoader();
const gltfs = new Map<string, ReturnType<typeof loader.loadAsync>>();
const load = (file: string) => {
  if (!gltfs.has(file)) gltfs.set(file, loader.loadAsync(`models/${file}`));
  return gltfs.get(file)!;   // each fit() clones, so one download serves every copy
};

/** A car model at real size, front along +z, sitting on y = 0, with its measured footprint. */
export async function loadCar(model: CarName) {
  const m = CARS[model];
  const obj = fit((await load(m.file)).scene, m.yaw, m.length, "length");
  obj.name = model;
  const size = measureBox(obj.children[0]).getSize(new THREE.Vector3());
  return { obj, length: size.z, width: size.x, height: size.y };
}

/** A person at real height, playing its walk (or idle) clip. */
export async function loadPerson(name: PersonName, idle: boolean) {
  const g = await load(PEOPLE[name].file);
  const obj = fit(g.scene, 0, PEOPLE[name].height, "height");
  obj.name = name;
  const mixer = new THREE.AnimationMixer(obj);
  const want = idle ? /idle|standing/i : /walk/i;
  const clip = g.animations.find((a) => want.test(a.name)) ?? g.animations[0];
  if (clip) mixer.clipAction(clip).play();
  return { obj, mixer };
}

// World box of a model as drawn: skinned meshes need their bones posed first, or they measure
// in bind space (Blender exports lie along z there).
function measureBox(o: THREE.Object3D) {
  o.updateMatrixWorld(true);
  o.traverse((m) => {
    if (m instanceof THREE.SkinnedMesh) { m.skeleton.update(); m.computeBoundingBox(); }
  });
  return new THREE.Box3().setFromObject(o);
}

// Scale so the given extent matches, then sit the model on y = 0, centered on its origin.
function fit(src: THREE.Object3D, yaw: number, size: number, measure: "height" | "length") {
  const inner = SkeletonUtils.clone(src);
  inner.rotation.y = yaw;
  const s = measureBox(inner).getSize(new THREE.Vector3());
  inner.scale.multiplyScalar(size / (measure === "height" ? s.y : Math.max(s.x, s.z)));
  // A car modelled along x gets turned to lie along z.
  if (measure === "length" && s.x > s.z) inner.rotation.y += Math.PI / 2;
  const box = measureBox(inner);
  const c = box.getCenter(new THREE.Vector3());
  inner.position.sub(new THREE.Vector3(c.x, box.min.y, c.z));
  const outer = new THREE.Group();
  outer.add(inner);
  // Blob shadow so things sit on the road instead of floating (the sim has no shadow maps).
  const ext = box.getSize(new THREE.Vector3());
  const blob = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), BLOB_MAT);
  blob.rotation.x = -Math.PI / 2;
  blob.position.y = 0.03;
  if (measure === "height") blob.scale.set(0.9, 0.9, 1);
  else blob.scale.set(ext.x * 1.25, ext.z * 1.15, 1);
  outer.add(blob);
  return outer;
}

const BLOB_MAT = (() => {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(32, 32, 4, 32, 32, 32);
  grad.addColorStop(0, "rgba(0,0,0,0.75)");
  grad.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false });
})();

export function spawnActors(scene: THREE.Scene) {
  const cars: Car[] = [];
  const people: Person[] = [];
  const standers: THREE.Object3D[] = [];

  for (const c of CAR_SPAWNS) {
    const m = CARS[c.model];
    loadCar(c.model).then(({ obj }) => {
      obj.rotation.y = c.axis === "z" ? (c.dir > 0 ? 0 : Math.PI) : (c.dir > 0 ? Math.PI / 2 : -Math.PI / 2);
      if (c.axis === "z") obj.position.set(c.lane, 0, c.at); else obj.position.set(c.at, 0, c.lane);
      scene.add(obj);
      cars.push({ obj, axis: c.axis, dir: c.dir, speed: c.speed, wrap: c.wrap, half: m.length / 2, yields: c.axis === "x" });
    }).catch((e) => console.error(`[actors] ${m.file}`, e));
  }

  const person = (name: PersonName, idle: boolean) => loadPerson(name, idle).then((p) => {
    scene.add(p.obj);
    return p;
  });

  for (const w of WALKERS) {
    person(w.model, false).then(({ obj, mixer }) => {
      mixer.timeScale = w.speed / WALK_CYCLE_SPEED;
      people.push({ obj, mixer, s: w.s0, speed: w.speed });
    }).catch((e) => console.error(`[actors] ${w.model}`, e));
  }
  const idleMixers: THREE.AnimationMixer[] = [];
  for (const p of STANDERS) {
    person(p.model, true).then(({ obj, mixer }) => {
      obj.position.set(p.x, 0, p.z);
      obj.rotation.y = p.face;
      standers.push(obj);
      idleMixers.push(mixer);
    }).catch((e) => console.error(`[actors] ${p.model}`, e));
  }

  const loopLen = SIDEWALK_LOOP.reduce((d, p, i) => {
    const q = SIDEWALK_LOOP[(i + 1) % SIDEWALK_LOOP.length];
    return d + Math.hypot(q[0] - p[0], q[1] - p[1]);
  }, 0);

  function placeOnLoop(o: THREE.Object3D, s: number) {
    s = ((s % loopLen) + loopLen) % loopLen;
    for (let i = 0; i < SIDEWALK_LOOP.length; i++) {
      const p = SIDEWALK_LOOP[i], q = SIDEWALK_LOOP[(i + 1) % SIDEWALK_LOOP.length];
      const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
      if (s <= len) {
        const t = s / len;
        o.position.set(p[0] + (q[0] - p[0]) * t, 0, p[1] + (q[1] - p[1]) * t);
        o.rotation.y = Math.atan2(q[0] - p[0], q[1] - p[1]);
        return;
      }
      s -= len;
    }
  }

  // Same-lane car ahead closer than FOLLOW_GAP: hold, so a faster car trails a slower one.
  const tooClose = (c: Car) => cars.some((o) => {
    if (o === c || o.axis !== c.axis || o.dir !== c.dir) return false;
    const pc = c.obj.position, po = o.obj.position;
    const side = c.axis === "x" ? Math.abs(po.z - pc.z) : Math.abs(po.x - pc.x);
    const gap = (po[c.axis] - pc[c.axis]) * c.dir - c.half - o.half;
    return side < 1 && gap > -1 && gap < FOLLOW_GAP;
  });

  // Cross-street cars wait at a stop line while someone is on the crosswalk or main-road traffic is near.
  function intersectionBusy(c: Car) {
    const lane = c.obj.position.z;
    const onCrosswalk = (o: THREE.Vector3) => Math.abs(o.x) < CROSSWALK_HALF && Math.abs(o.z - lane) < CROSSWALK_REACH;
    return people.some((q) => onCrosswalk(q.obj.position)) || standers.some((o) => onCrosswalk(o.position)) ||
      cars.some((o) => {
        if (o.axis !== "z") return false;
        const toLane = (lane - o.obj.position.z) * o.dir; // > 0 while approaching
        return toLane > -(o.half + 2) && toLane < MAIN_ROAD_LOOKAHEAD;
      });
  }

  function update(dt: number) {
    for (const c of cars) {
      const p = c.obj.position;
      if (tooClose(c)) continue;
      if (c.yields) {
        const front = p.x + c.dir * c.half;
        const line = c.dir > 0 ? STOP_LINE_X[0] : STOP_LINE_X[1];
        const reachesLine = (line - front) * c.dir >= 0 && (line - front) * c.dir < c.speed * dt + 0.05;
        if (reachesLine && intersectionBusy(c)) continue;
      }
      let v = p[c.axis] + c.dir * c.speed * dt;
      if (v < c.wrap[0]) v = c.wrap[1];
      if (v > c.wrap[1]) v = c.wrap[0];
      p[c.axis] = v;
    }
    for (const q of people) {
      q.s += q.speed * dt;
      placeOnLoop(q.obj, q.s);
      q.mixer.update(dt);
    }
    for (const m of idleMixers) m.update(dt);
  }

  const carBox = (c: Car) => new THREE.Box3().setFromObject(c.obj.children[0]); // model only, not its blob shadow
  const personBox = (o: THREE.Object3D) => new THREE.Box3(
    new THREE.Vector3(o.position.x - PERSON_RADIUS, 0, o.position.z - PERSON_RADIUS),
    new THREE.Vector3(o.position.x + PERSON_RADIUS, 1.8, o.position.z + PERSON_RADIUS),
  );
  const solids = () => [
    ...cars.map((c) => carBox(c)),
    ...people.map((q) => personBox(q.obj)),
    ...standers.map(personBox),
  ];

  // Ground truth for the detection check in sim/yolo_check.py; not used by the sim itself.
  (window as unknown as { __kiteActors: () => unknown }).__kiteActors = () => [
    ...cars.map((c) => ({ kind: "car", name: c.obj.name, box: carBox(c) })),
    ...[...people.map((q) => q.obj), ...standers].map((o) => ({ kind: "person", name: o.name, box: personBox(o) })),
  ].map(({ kind, name, box }) => ({ kind, name, min: box.min.toArray(), max: box.max.toArray() }));

  return { update, solids };
}
