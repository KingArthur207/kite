// Moss world traffic from the OSM lanes: Howard one-way SW in three lanes, Moss one-way SE, and
// cars turning left from Howard into Moss across the zebra (the demo's key hazard). Cars yield
// to anyone on a zebra and stop for people in their path. People walk the Moss and Howard
// sidewalks, some across the zebra. Placeholder-world traffic stays in actors.ts.
import * as THREE from "three";
import { lanes, route, crossings } from "./osmWorld";
import { loadCar, loadPerson, type CarName, type PersonName } from "./actors";

type P = [number, number];
export interface Pose { x: number; n: number; dx: number; dn: number }

const HOWARD_V = 10, MOSS_V = 5, TURN_V = 4;   // m/s
const TURN_R = 4;               // m: corner cut on each side of the Howard -> Moss turn
const ACCEL = 2.5, DECEL = 4;   // m/s^2
const LOOK = 25;                // m: how far ahead a car checks its path
const PATH_HALF = 1.5;          // m: someone this close to a car's path ahead is in its way
const CAR_GAP = 3, PERSON_GAP = 2.5;   // m: bumper gaps kept to a car / a person ahead
const ZEBRA_HALF = 1.5;         // m: half the stripe length along the road (osmWorld zebraQuads)
const ZEBRA_MARGIN = 0.6;       // m: someone this close to the stripes counts as on them
const PERSON_RADIUS = 0.25;
const WALK_CYCLE_SPEED = 1.3;   // m/s the walk clips look right at

/** A polyline with arc length. Each point carries the cruise speed of the segment it starts. */
export class Path {
  readonly s: number[] = [0];
  constructor(readonly pts: P[], readonly v: number[] = pts.map(() => 0)) {
    for (let i = 1; i < pts.length; i++) this.s.push(this.s[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  }
  get len() { return this.s[this.s.length - 1]; }
  private seg(s: number) {
    let i = 0;
    while (i < this.pts.length - 2 && s > this.s[i + 1]) i++;
    return i;
  }
  at(s: number): Pose {
    s = Math.max(0, Math.min(this.len, s));
    const i = this.seg(s), a = this.pts[i], b = this.pts[i + 1], L = this.s[i + 1] - this.s[i] || 1;
    const u = (s - this.s[i]) / L;
    return { x: a[0] + (b[0] - a[0]) * u, n: a[1] + (b[1] - a[1]) * u, dx: (b[0] - a[0]) / L, dn: (b[1] - a[1]) / L };
  }
  speed(s: number) { return this.v[this.seg(s)]; }
  /** Arc length of the closest point and the distance to it. */
  nearest(x: number, n: number) {
    let best = { s: 0, d: Infinity };
    for (let i = 0; i + 1 < this.pts.length; i++) {
      const [ax, an] = this.pts[i], [bx, bn] = this.pts[i + 1], L = this.s[i + 1] - this.s[i] || 1;
      const u = Math.max(0, Math.min(1, ((x - ax) * (bx - ax) + (n - an) * (bn - an)) / (L * L)));
      const d = Math.hypot(x - ax - (bx - ax) * u, n - an - (bn - an) * u);
      if (d < best.d) best = { s: this.s[i] + u * L, d };
    }
    return best;
  }
}

// Chain OSM lane pieces of one street end to start (OSM splits ways at every junction).
function chains(name: string): P[][] {
  const ls = lanes.filter((l) => l.name === name);
  const near = (a: P, b: P) => Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.5;
  const out: P[][] = [];
  for (const l of ls) {
    if (ls.some((o) => o !== l && near(o.pts[o.pts.length - 1], l.pts[0]))) continue;   // not a chain head
    let cur = l, pts = [...l.pts];
    for (let next; (next = ls.find((o) => o !== cur && near(cur.pts[cur.pts.length - 1], o.pts[0]))); cur = next) pts = [...pts, ...next.pts.slice(1)];
    out.push(pts);
  }
  return out;
}

/** Howard's three lanes (index 0 = the left lane, next to Moss) and the left turn into Moss. */
export function mossPaths() {
  const zebra = crossings.find((c) => Math.hypot(c.x, c.y) < 1)!;            // the origin zebra
  const howard = chains("Howard Street")
    .map((p) => new Path(p, p.map(() => HOWARD_V)))
    .sort((a, b) => a.nearest(zebra.x, zebra.y).d - b.nearest(zebra.x, zebra.y).d);
  const mossPts = chains("Moss Street")[0];
  const moss = new Path(mossPts, mossPts.map(() => MOSS_V));
  // Corner: where Howard's left lane crosses Moss's first segment. Cut it with a quadratic curve.
  const left = howard[0];
  const [a, b] = [moss.pts[0], moss.pts[1]];
  const side = (p: P) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
  const i = left.pts.findIndex((p, k) => k > 0 && Math.sign(side(p)) !== Math.sign(side(left.pts[k - 1])));
  const f = side(left.pts[i - 1]) / (side(left.pts[i - 1]) - side(left.pts[i]));
  const sCorner = left.s[i - 1] + f * (left.s[i] - left.s[i - 1]);
  const c = left.at(sCorner), sIn = moss.nearest(c.x, c.n).s;
  const p0 = left.at(sCorner - TURN_R), p2 = moss.at(sIn + TURN_R);
  const pts: P[] = [], v: number[] = [];
  left.pts.forEach((p, k) => { if (left.s[k] < sCorner - TURN_R) { pts.push(p); v.push(HOWARD_V); } });
  for (let k = 0; k <= 8; k++) {
    const t = k / 8, u = 1 - t;
    pts.push([u * u * p0.x + 2 * u * t * c.x + t * t * p2.x, u * u * p0.n + 2 * u * t * c.n + t * t * p2.n]);
    v.push(TURN_V);
  }
  moss.pts.forEach((p, k) => { if (moss.s[k] > sIn + TURN_R) { pts.push(p); v.push(MOSS_V); } });
  return { howard, turn: new Path(pts, v), zebra };
}

/** Ground-plane boxes for a car: axis-aligned squares along its axis, sized so the chain
 *  stays inside the car's width at any heading. */
// ponytail: AABB chain approximates an oriented box; swap to OBB tests if corner hits matter
export function carBoxes(p: Pose, length: number, width: number, height: number) {
  const a = width / (Math.abs(p.dx) + Math.abs(p.dn));
  const k = Math.max(2, Math.ceil((length - a) / (a / 2)) + 1);
  const out: THREE.Box3[] = [];
  for (let j = 0; j < k; j++) {
    const o = -(length - a) / 2 + ((length - a) * j) / (k - 1);
    const x = p.x + p.dx * o, z = -(p.n + p.dn * o);
    out.push(new THREE.Box3(new THREE.Vector3(x - a / 2, 0, z - a / 2), new THREE.Vector3(x + a / 2, height, z + a / 2)));
  }
  return out;
}

/** Eight corners of a car's oriented box, for projecting ground truth (sim/yolo_check.py). */
function carCorners(p: Pose, length: number, width: number, height: number) {
  const out: number[][] = [];
  for (const l of [-length / 2, length / 2]) for (const w of [-width / 2, width / 2]) for (const y of [0, height]) {
    out.push([p.x + p.dx * l + p.dn * w, y, -(p.n + p.dn * l - p.dx * w)]);
  }
  return out;
}

interface Zebra { x: number; n: number; ux: number; un: number; a0: number; a1: number }
interface Car {
  obj: THREE.Object3D; path: Path; s: number; v: number; vScale: number;
  length: number; width: number; height: number; zebraS: { z: Zebra; s: number }[];
}
interface Walker { obj: THREE.Object3D; mixer: THREE.AnimationMixer; path: Path; s: number; dir: 1 | -1; speed: number }

const onZebra = (z: Zebra, x: number, n: number) => {
  const ex = x - z.x, en = n - z.n, along = ex * z.ux + en * z.un, across = -ex * z.un + en * z.ux;
  return Math.abs(along) < ZEBRA_HALF + ZEBRA_MARGIN && across > z.a0 - ZEBRA_MARGIN && across < z.a1 + ZEBRA_MARGIN;
};

export function spawnMossActors(scene: THREE.Scene, others: () => { x: number; n: number }[]) {
  const { howard, turn } = mossPaths();
  const zebras: Zebra[] = crossings.filter((c) => ["zebra", "yes", "ladder"].includes(c.markings))
    .map((c) => ({ x: c.x, n: c.y, ux: c.u[0], un: c.u[1], a0: c.across[0], a1: c.across[1] }));

  // Sidewalk lines from the demo route: Moss NE walk line, the zebra, Howard's south walk line.
  const [spawn, mossCorner, howardCorner, end] = route;
  const ext = (a: P, b: P, m: number): P => {   // m metres past b, going a -> b
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    return [b[0] + ((b[0] - a[0]) / L) * m, b[1] + ((b[1] - a[1]) / L) * m];
  };
  const mossDir: P = [spawn[0] - mossCorner[0], spawn[1] - mossCorner[1]];
  const L = Math.hypot(...mossDir);
  const sw: P = [howardCorner[0] + (mossDir[0] / L) * 30, howardCorner[1] + (mossDir[1] / L) * 30];
  const PATHS = {
    howardSouth: new Path([ext(howardCorner, mossCorner, 35), end]),             // crosses the zebra
    mossNE: new Path([mossCorner, ext(mossCorner, spawn, 15)]),
    mossSW: new Path([howardCorner, sw]),
    route: new Path(route),                                                       // the demo walk, crosses the zebra
  };

  const CARS: { model: CarName; path: Path; s: number; vScale: number }[] = [
    { model: "sedan", path: turn, s: 30, vScale: 1 },
    { model: "suv", path: turn, s: 120, vScale: 0.95 },
    { model: "rover", path: turn, s: 215, vScale: 1.05 },
    { model: "sedan2", path: turn, s: 300, vScale: 1 },
    { model: "pickup", path: howard[0], s: 60, vScale: 0.95 },
    { model: "bus", path: howard[1], s: 20, vScale: 0.85 },
    { model: "sedan", path: howard[1], s: 250, vScale: 1 },
    { model: "suv", path: howard[2], s: 90, vScale: 1.05 },
    { model: "sedan2", path: howard[2], s: 330, vScale: 1 },
  ];
  const WALKERS: { model: PersonName; path: keyof typeof PATHS; s: number; speed: number }[] = [
    { model: "man", path: "howardSouth", s: 5, speed: 1.4 },
    { model: "woman", path: "howardSouth", s: 40, speed: 1.25 },
    { model: "business", path: "route", s: 50, speed: 1.5 },
    { model: "dress", path: "mossNE", s: 30, speed: 1.3 },
    { model: "woman2", path: "mossSW", s: 25, speed: 1.35 },
  ];

  const cars: Car[] = [];
  const walkers: Walker[] = [];
  const standers: THREE.Object3D[] = [];
  const idle: THREE.AnimationMixer[] = [];

  for (const c of CARS) {
    loadCar(c.model).then(({ obj, length, width, height }) => {
      scene.add(obj);
      const zebraS = zebras.map((z) => ({ z, ...c.path.nearest(z.x, z.n) }))
        .filter((h) => h.d < Math.max(-h.z.a0, h.z.a1)).map(({ z, s }) => ({ z, s }));
      cars.push({ obj, path: c.path, s: c.s, v: 0, vScale: c.vScale, length, width, height, zebraS });
    }).catch((e) => console.error(`[mossActors] ${c.model}`, e));
  }
  for (const w of WALKERS) {
    loadPerson(w.model, false).then(({ obj, mixer }) => {
      scene.add(obj);
      mixer.timeScale = w.speed / WALK_CYCLE_SPEED;
      walkers.push({ obj, mixer, path: PATHS[w.path], s: w.s, dir: 1, speed: w.speed });
    }).catch((e) => console.error(`[mossActors] ${w.model}`, e));
  }
  // One person waits on Howard's south sidewalk, building side, NE of the corner, facing the road.
  {
    const p = PATHS.howardSouth.at(20);
    loadPerson("casual", true).then(({ obj, mixer }) => {
      obj.position.set(p.x - p.dn * 1.2, 0, -(p.n + p.dx * 1.2));   // 1.2 m to the left of the SW-bound line
      obj.rotation.y = Math.atan2(p.dn, p.dx);                         // faces right of it: NW, toward Howard
      scene.add(obj);
      standers.push(obj);
      idle.push(mixer);
    }).catch((e) => console.error("[mossActors] casual", e));
  }

  const people = () => [
    ...walkers.map((w) => ({ x: w.obj.position.x, n: -w.obj.position.z })),
    ...standers.map((o) => ({ x: o.position.x, n: -o.position.z })),
    ...others(),
  ];
  const poseOf = (c: Car) => c.path.at(c.s);
  const boxesOf = (c: Car) => carBoxes(poseOf(c), c.length, c.width, c.height);

  // Target speed: cruise, slowing ahead of slower segments, stopping short of zebras with
  // someone on them, people in the path and the car ahead.
  function target(c: Car, ppl: { x: number; n: number }[]) {
    const stopAt = (sStop: number) => Math.sqrt(2 * DECEL * Math.max(0, sStop - c.s));
    let v = c.path.speed(c.s) * c.vScale;
    for (let k = 0; k < c.path.pts.length; k++) {
      const d = c.path.s[k] - c.s;
      if (d > 0 && d < LOOK) v = Math.min(v, Math.sqrt((c.path.v[k] * c.vScale) ** 2 + 2 * DECEL * d));
    }
    for (const { z, s } of c.zebraS) {
      const line = s - ZEBRA_HALF - 1 - c.length / 2;              // car centre at the stop line
      if (c.s <= line + 0.5 && line - c.s < LOOK && ppl.some((p) => onZebra(z, p.x, p.n))) v = Math.min(v, stopAt(line));
    }
    for (let d = c.length / 2; d < LOOK; d += 1) {
      const q = c.path.at(c.s + d);
      if (ppl.some((p) => Math.hypot(p.x - q.x, p.n - q.n) < PATH_HALF)) { v = Math.min(v, stopAt(c.s + d - c.length / 2 - PERSON_GAP)); break; }
      const ahead = cars.find((o) => o !== c && Math.hypot(o.obj.position.x - q.x, -o.obj.position.z - q.n) < PATH_HALF);
      if (ahead) { v = Math.min(v, stopAt(c.s + d - ahead.length / 2 - c.length / 2 - CAR_GAP)); break; }
    }
    return v;
  }

  function update(dt: number) {
    const ppl = people();
    for (const c of cars) {
      const v = target(c, ppl);
      c.v = v < c.v ? v : Math.min(v, c.v + ACCEL * dt);
      c.s += c.v * dt;
      if (c.s >= c.path.len) {
        // Back to the start of the lane once nobody is sitting there.
        const st = c.path.at(0);
        if (cars.every((o) => o === c || Math.hypot(o.obj.position.x - st.x, -o.obj.position.z - st.n) > 12)) { c.s = 0; c.v = c.path.speed(0) * c.vScale; }
        else c.s = c.path.len;
      }
      const p = poseOf(c);
      c.obj.position.set(p.x, 0, -p.n);
      c.obj.rotation.y = Math.atan2(p.dx, -p.dn);
    }
    const carBoxList = cars.flatMap(boxesOf);
    for (const w of walkers) {
      // Walk the line back and forth; wait while a car is right in front.
      let ns = w.s + w.dir * w.speed * dt;
      if (ns > w.path.len || ns < 0) { w.dir = w.dir > 0 ? -1 : 1; ns = Math.max(0, Math.min(w.path.len, ns)); }
      const q = w.path.at(ns), aheadX = q.x + q.dx * w.dir * 0.5, aheadN = q.n + q.dn * w.dir * 0.5;
      const blocked = carBoxList.some((b) => aheadX > b.min.x - 0.4 && aheadX < b.max.x + 0.4 && -aheadN > b.min.z - 0.4 && -aheadN < b.max.z + 0.4);
      if (!blocked) w.s = ns;
      const p = w.path.at(w.s);
      w.obj.position.set(p.x, 0, -p.n);
      w.obj.rotation.y = Math.atan2(p.dx * w.dir, -p.dn * w.dir);
      if (!blocked) w.mixer.update(dt);
    }
    for (const m of idle) m.update(dt);
  }

  const personBox = (o: THREE.Object3D) => new THREE.Box3(
    new THREE.Vector3(o.position.x - PERSON_RADIUS, 0, o.position.z - PERSON_RADIUS),
    new THREE.Vector3(o.position.x + PERSON_RADIUS, 1.8, o.position.z + PERSON_RADIUS),
  );
  const solids = () => [...cars.flatMap(boxesOf), ...walkers.map((w) => personBox(w.obj)), ...standers.map(personBox)];

  // Ground truth for sim/yolo_check.py; cars give oriented corners so their projected box is tight.
  (window as unknown as { __kiteActors: () => unknown }).__kiteActors = () => [
    ...cars.map((c) => {
      const corners = carCorners(poseOf(c), c.length, c.width, c.height);
      const b = new THREE.Box3().setFromPoints(corners.map((p) => new THREE.Vector3(...p)));
      return { kind: "car", name: c.obj.name, min: b.min.toArray(), max: b.max.toArray(), corners };
    }),
    ...[...walkers.map((w) => w.obj), ...standers].map((o) => {
      const b = personBox(o);
      return { kind: "person", name: o.name, min: b.min.toArray(), max: b.max.toArray() };
    }),
  ];

  return { update, solids };
}
