// Metrics mode (?metrics=1): scripted walks that count collisions with and without Kite.
//   mode=blind   walker never reacts
//   mode=kite    walker reacts ONLY to what the earpiece says/does (ws://127.0.0.1:5011, or ?ear=), real time
//   mode=oracle  UPPER BOUND, reacts to ground-truth obstacles 5 m ahead. Never report it as Kite.
// Params: trials (20), seed (1), speed (sim seconds per real second, blind/oracle only, 300).
// Runs on the Moss world's demo route with the street emptied; building faces and curbs still block.
import * as THREE from "three";
import { WALKER_RADIUS, WALK_SPEED, EARPIECE_URL } from "./config";
import { route } from "./osmWorld";
import { Path, mossPaths, carBoxes, type Pose } from "./mossActors";
import { loadCar, loadPerson } from "./actors";

const REACTION_S = 0.7;     // human reaction time before stopping
const CLEAR_S = 1.5;        // all-clear time before walking on
const STUCK_S = 8;          // waiting this long = a static thing: feel around it (cane). A passing car clears sooner
const BLOCKED_S = 1;        // pressed against something this long = feel around it
const SIDE_LEN = 7;         // meters walked on the sidestep line before rejoining
const IGNORE_LEN = 3;       // meters after a sidestep where cues are ignored (else static things hold forever)
const NEAR_MISS_M = 0.5;
const TRIAL_MAX_S = 150;
const DT = 0.02;            // fixed step for accelerated modes
const ORACLE_AHEAD = 5, ORACLE_HALF_W = 0.6, ORACLE_LOOKAHEAD_S = [0, 0.5, 1, 1.5, 2, 2.5, 3];
// The turning car: speed, seconds from its trigger to the zebra (+- jitter), seconds it drives on after.
const CAR = { v: 6, lead: 8, jitter: 1.2, after: 10, length: 4.6, width: 1.9, height: 1.5 };

const OBSTACLE_NAMES: Record<string, string> = {
  pedA: "Pedestrian crossing", bin: "Bin", head: "Head-height sign", car: "Car turning in", pedB: "Pedestrian from a doorway",
};

const STATUS_KEYS = ["perception-down", "perception-up", "continue"];   // earpiece app.py speech keys

type Mode ="blind" | "kite" | "oracle";
const LABEL: Record<Mode, string> = {
  blind: "BLIND (no cues)",
  kite: "KITE (reacts only to earpiece output)",
  oracle: "ORACLE: upper bound from ground truth, NOT Kite",
};

// The Moss demo route: NW up Moss's NE sidewalk, left across the zebra, SW along Howard.
// Sidestep per leg (sign +1 = left of the walking direction) and how far: Moss's NE sidewalk
// has about 1 m spare toward the buildings, the zebra and Howard's sidewalk about 1.5 m.
// ponytail: hand-picked from this route's wall clearances; recheck if the route changes
const ROUTE = new Path(route);
const SIDE = [{ sign: -1, m: 1.0 }, { sign: -1, m: 1.5 }, { sign: 1, m: 1.5 }];
const ROUTE_LEN = ROUTE.len;

function routeAt(s: number) {
  const p = ROUTE.at(s);
  let leg = 0;
  while (leg < SIDE.length - 1 && s > ROUTE.s[leg + 1]) leg++;
  const { sign, m } = SIDE[leg];
  return { ...p, side: { x: -p.dn * sign * m, n: p.dx * sign * m }, left: { x: -p.dn, n: p.dx } };
}

function mulberry32(a: number) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Same ground-plane AABB test as main.ts hits().
function hits(x: number, n: number, boxes: THREE.Box3[], r = WALKER_RADIUS) {
  const z = -n;
  return boxes.filter((b) => x > b.min.x - r && x < b.max.x + r && z > b.min.z - r && z < b.max.z + r);
}
function clearance(x: number, n: number, b: THREE.Box3) {
  const z = -n;
  const dx = Math.max(b.min.x - x, 0, x - b.max.x), dz = Math.max(b.min.z - z, 0, z - b.max.z);
  return Math.hypot(dx, dz) - WALKER_RADIUS;
}

// A scripted obstacle: waits at its start, starts moving when the walker's route progress hits
// `trigger`, moves for `dur` seconds. `at(k)` is its pose after k seconds of moving.
interface Obs {
  name: string; obj: THREE.Group; y: number; at: (k: number) => Pose; half: number; car: boolean;
  dur: number; trigger: number; t: number; minClear: number; hit: boolean;
}

export function startMetrics(ctx: {
  walker: { x: number; n: number; heading: number };
  scene: THREE.Scene;
  walls: (x: number, n: number, r: number) => object[];
}, q = new URLSearchParams(location.search)) {
  if (q.get("metrics") !== "1") return null;
  const mode = (["blind", "kite", "oracle"].includes(q.get("mode") ?? "") ? q.get("mode") : "blind") as Mode;
  const trials = Math.max(1, Number(q.get("trials")) || 20);
  const seed = Number(q.get("seed")) || 1;
  const speed = mode === "kite" ? 1 : Math.max(1, Number(q.get("speed")) || 300);
  const { walker, scene, walls } = ctx;

  // Visuals only (collision comes from each obstacle's pose): boxes at first, glTF car and people
  // swapped in once loaded, so the detector in kite mode sees what it would see on the street.
  const group = (geo: THREE.BufferGeometry, color: number, y: number) => {
    const g = new THREE.Group();
    const m = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color }));
    m.position.y = y;
    g.add(m);
    scene.add(g);
    return g;
  };
  const objs = {
    pedA: group(new THREE.CapsuleGeometry(0.25, 1.2, 4, 8), 0x7a3e9d, 0.85),
    bin: group(new THREE.BoxGeometry(0.5, 1.0, 0.5), 0x2e5e3a, 0.5),
    head: group(new THREE.BoxGeometry(1.0, 0.4, 0.3), 0x8b5a2b, 0.2),     // overhanging sign: above cane height
    car: group(new THREE.BoxGeometry(CAR.width, CAR.height, CAR.length), 0xb22222, CAR.height / 2),
    pedB: group(new THREE.CapsuleGeometry(0.25, 1.2, 4, 8), 0x3e7a5d, 0.85),
  };
  const mixers: Partial<Record<keyof typeof objs, THREE.AnimationMixer>> = {};
  const swap = (g: THREE.Group, obj: THREE.Object3D) => { g.clear(); g.add(obj); };
  loadCar("sedan").then(({ obj }) => swap(objs.car, obj)).catch((e) => console.error("[metrics] car", e));
  for (const [k, model] of [["pedA", "man"], ["pedB", "woman"]] as const) {
    loadPerson(model, false).then(({ obj, mixer }) => { swap(objs[k], obj); mixers[k] = mixer; })
      .catch((e) => console.error(`[metrics] ${model}`, e));
  }

  // The car turns in from Howard's left lane and drives through the zebra without yielding.
  const { turn, zebra } = mossPaths();
  const carAtZebra = turn.nearest(zebra.x, zebra.y).s;
  const walkerAtZebra = ROUTE.nearest(zebra.x, zebra.y).s;

  // Seeded per trial. Movers are triggered so they arrive near the walker's arrival, +- jitter.
  function makeObstacles(trial: number): Obs[] {
    const r = mulberry32(seed * 1000 + trial);
    const jit = (w: number) => (r() * 2 - 1) * w;
    const lead = (sAt: number, secs: number) => sAt - WALK_SPEED * secs;   // trigger point in route meters
    const still = (s: number): Pose => { const p = routeAt(s); return { x: p.x, n: p.n, dx: p.dx, dn: p.dn }; };
    // Someone crossing the walker's line from `from` m left of it to `to` m right, at v m/s.
    const crosser = (s: number, from: number, v: number) => {
      const p = routeAt(s);
      return (k: number): Pose => {
        const off = from - v * k;
        return { x: p.x + p.left.x * off, n: p.n + p.left.n * off, dx: -p.left.x, dn: -p.left.n };
      };
    };
    const o = (name: keyof typeof objs, at: (k: number) => Pose, dur: number, trigger: number, half = 0.25, y = 0): Obs =>
      ({ name, obj: objs[name], y, at, half, car: name === "car", dur, trigger, t: 0, minClear: Infinity, hit: false });
    const sA = 17 + 2 * r(), sBin = 44 + 2 * r(), sHead = 56 + 2 * r(), sB = 86 + 2 * r();
    const car0 = carAtZebra - CAR.v * (CAR.lead + jit(CAR.jitter));
    return [
      // From Moss's far (SW) sidewalk across the road and the walker's line, to a doorway 1.5 m past it.
      o("pedA", crosser(sA, 8.1, 1.3), 9.6 / 1.3, lead(sA - 2, 8.1 / 1.3 + jit(0.7))),
      o("bin", () => still(sBin), 0, -1),
      o("head", () => still(sHead), 0, -1, 0.3, 1.45),
      o("car", (k) => turn.at(car0 + CAR.v * k), CAR.lead + CAR.jitter + CAR.after, lead(walkerAtZebra, CAR.lead)),
      // Out of a doorway on Howard's south side, across the walker's line toward the curb.
      o("pedB", crosser(sB, 1.6, 0.9), 3.2 / 0.9, lead(sB - 2, 1.6 / 0.9 + jit(0.7))),
    ];
  }
  const poseOf = (o: Obs, dt = 0) => o.at(Math.min(o.t + dt, o.dur));
  const boxesOf = (o: Obs, dt = 0): THREE.Box3[] => {
    const p = poseOf(o, dt);
    if (o.car) return carBoxes(p, CAR.length, CAR.width, CAR.height);
    const z = -p.n;
    return [new THREE.Box3(new THREE.Vector3(p.x - o.half, o.y, z - o.half), new THREE.Vector3(p.x + o.half, o.y + (o.y ? 0.4 : 1.8), z + o.half))];
  };
  const place = (o: Obs) => {
    const p = poseOf(o);
    o.obj.position.set(p.x, o.y, -p.n);
    o.obj.rotation.y = Math.atan2(p.dx, -p.dn);
  };

  // --- kite: listen to the earpiece. Alarm = spoken critical/hazard, a critical sonar track,
  // or a hazard holding the walker (dont_walk/wait/red/stop). Other hazards count via their `spoken`.
  let kiteAlarmAt = -1e9, earOpen = false, lastCue = "";
  if (mode === "kite") {
    const connect = () => {
      const ws = new WebSocket(q.get("ear") || EARPIECE_URL);
      ws.onopen = () => { earOpen = true; };
      ws.onclose = () => { earOpen = false; setTimeout(connect, 2000); };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => {
        let m: any;
        try { m = JSON.parse(e.data); } catch { return; }
        const now = performance.now() / 1000;
        // Status lines share the hazard priority but warn of nothing: "Obstacle detection paused/is back"
        // and "Path is clear. You can continue." must not stop the walker.
        if (m.type === "spoken" && ["critical", "hazard"].includes(m.priority) && m.result !== "dropped"
            && !STATUS_KEYS.includes(m.key)) {
          kiteAlarmAt = now; lastCue = `spoken ${m.priority}: ${m.text}`;
        } else if (m.type === "state") {
          const crit = (m.tracks ?? []).find((t: any) => t.critical);
          const hold = (m.context?.hazards ?? []).find((h: any) => ["dont_walk", "wait", "red", "stop"].includes(h.state));
          if (crit || hold) { kiteAlarmAt = now; lastCue = crit ? `sonar critical ${crit.kind}` : `hazard ${hold.kind} ${hold.state}`; }
        }
      };
    };
    connect();
  }

  // --- trial state
  const results: any[] = [];
  let trial = 0, obs: Obs[] = [], t = 0, s = 0, side = 0, sideTarget = 0, sideUntil = 0, ignoreUntil = 0;
  let stopped = false, stopAt = Infinity, lastAlarm = -1e9, waited = 0, blockedFor = 0;
  let touching = new Set<object>(), collisions = 0, wallContacts = 0, done = false, stops: number[] = [];

  function reset() {
    obs = makeObstacles(trial);
    t = 0; s = 0; side = 0; sideTarget = 0; sideUntil = 0; ignoreUntil = 0;
    stopped = false; stopAt = Infinity; lastAlarm = -1e9; waited = 0; blockedFor = 0;
    touching = new Set(); collisions = 0; wallContacts = 0; stops = [];
    obs.forEach(place);
    placeWalker();
  }
  function placeWalker() {
    const p = routeAt(s);
    walker.x = p.x + p.side.x * side;
    walker.n = p.n + p.side.n * side;
    walker.heading = ((Math.atan2(p.dx, p.dn) * 180) / Math.PI + 360) % 360;
  }
  // Everything the walker can touch at (x, n): obstacle boxes (tagged with their obstacle) and walls.
  function touches(x: number, n: number, ob: { o: Obs; b: THREE.Box3 }[], r: number) {
    const inBoxes = hits(x, n, ob.map((e) => e.b), r);
    return [...ob.filter((e) => inBoxes.includes(e.b)), ...walls(x, n, r)];
  }

  function oracleAlarm() {
    const p = routeAt(s);
    for (const o of obs) for (const dt of ORACLE_LOOKAHEAD_S) for (const b of boxesOf(o, dt)) {
      // corners of the box in walker frame (along, lateral)
      let aMin = Infinity, aMax = -Infinity, lMin = Infinity, lMax = -Infinity;
      for (const x of [b.min.x, b.max.x]) for (const n of [-b.min.z, -b.max.z]) {
        const ex = x - walker.x, en = n - walker.n;
        const a = ex * p.dx + en * p.dn, l = ex * p.dn - en * p.dx;
        aMin = Math.min(aMin, a); aMax = Math.max(aMax, a); lMin = Math.min(lMin, l); lMax = Math.max(lMax, l);
      }
      if (aMax > 0 && aMin < ORACLE_AHEAD && lMax > -ORACLE_HALF_W && lMin < ORACLE_HALF_W) return true;
    }
    return false;
  }

  function stepOnce(dt: number) {
    t += dt;
    for (const o of obs) if (s >= o.trigger && o.dur > 0 && o.t < o.dur) {
      o.t += dt;
      mixers[o.name as keyof typeof objs]?.update(dt);
    }
    const ob = obs.flatMap((o) => boxesOf(o).map((b) => ({ o, b })));

    // Reaction: any alarm starts a stop after REACTION_S; walk on after CLEAR_S of quiet.
    if (mode !== "blind" && s >= ignoreUntil) {
      const alarm = mode === "oracle" ? oracleAlarm() : performance.now() / 1000 - kiteAlarmAt < 0.3;
      if (alarm) { lastAlarm = t; if (!stopped && stopAt === Infinity) stopAt = t + REACTION_S; }
    }
    if (!stopped && t >= stopAt) { stopped = true; waited = 0; stops.push(+s.toFixed(1)); }
    if (stopped) {
      waited += dt;
      if (t - lastAlarm >= CLEAR_S) { stopped = false; stopAt = Infinity; }
      else if (waited > STUCK_S) sidestep();
    }

    if (!stopped) {
      const here = touches(walker.x, walker.n, ob, WALKER_RADIUS);
      const lateral = side !== sideTarget;
      const ns = lateral ? s : Math.min(ROUTE_LEN, s + WALK_SPEED * dt);
      const legSide = routeAt(s).side, step = (WALK_SPEED * dt) / Math.hypot(legSide.x, legSide.n);   // side is 0..1 of the leg's sidestep
      const nside = lateral ? side + Math.sign(sideTarget - side) * Math.min(Math.abs(sideTarget - side), step) : side;
      const p = routeAt(ns);
      const nx = p.x + p.side.x * nside, nn = p.n + p.side.n * nside;
      // Block only on things we aren't already touching, so a car that drove into us can't pin us.
      const key = (e: object) => ("o" in e && "b" in e ? (e as { o: Obs }).o : e);
      const hereKeys = new Set(here.map(key));
      // A sidestep only stops at walls and curbs: the route runs ~0.5 m from the building on Moss, and the bin's
      // axis-aligned box caught every sideways sweep at its corner, so no walk got past it.
      const blocked = (lateral ? walls(nx, nn, WALKER_RADIUS) : touches(nx, nn, ob, WALKER_RADIUS))
        .some((e) => !hereKeys.has(key(e)));
      if (blocked && lateral) {
        sideTarget = side;   // the sidestep ran into something (a wall, a curb): stop there and walk on
      } else if (blocked) {
        blockedFor += dt;
        if (blockedFor > BLOCKED_S) sidestep();
      } else {
        blockedFor = 0; s = ns; side = nside;
        if (sideTarget !== 0 && s >= sideUntil) sideTarget = 0;
      }
      placeWalker();
    }
    obs.forEach(place);

    // Collisions (10 cm margin, as in main.ts): each obstacle counts once per walk, since a car's
    // box chain flickers in and out of contact; walls count each new contact. Near-miss per obstacle.
    const now = new Set<object>();
    for (const e of touches(walker.x, walker.n, ob, WALKER_RADIUS + 0.1)) {
      if ("o" in e && "b" in e) {
        const o = (e as { o: Obs }).o;
        if (!o.hit) collisions++;
        o.hit = true;
        continue;
      }
      now.add(e);
      if (!touching.has(e)) { collisions++; wallContacts++; }
    }
    touching = now;
    for (const { o, b } of ob) o.minClear = Math.min(o.minClear, clearance(walker.x, walker.n, b));

    if (s >= ROUTE_LEN || t >= TRIAL_MAX_S) finishTrial(s >= ROUTE_LEN);
  }

  function sidestep() {
    stopped = false; stopAt = Infinity; blockedFor = 0; waited = 0;
    // Still blocked after a sidestep that a wall or prop cut short: feel the other way (the street dressing
    // left too little room toward the buildings at the bin, and every walk stalled there until the 150 s cap).
    sideTarget = sideTarget > 0 ? -1 : 1; sideUntil = s + SIDE_LEN; ignoreUntil = s + IGNORE_LEN;
  }

  function finishTrial(completed: boolean) {
    results.push({
      trial, collisions, wall_contacts: wallContacts, near_misses: obs.filter((o) => !o.hit && o.minClear < NEAR_MISS_M).length,
      time_s: +t.toFixed(1), completed, stops_at_m: stops,   // route metres where the walker stopped on a cue
      hit: obs.filter((o) => o.hit).map((o) => o.name),
      near: obs.filter((o) => !o.hit && o.minClear < NEAR_MISS_M).map((o) => o.name),
    });
    trial++;
    (window as any).kiteMetrics = out();   // progress for the headless runner (kite mode is slow)
    if (trial >= trials) return finish();
    reset();
  }

  const summary = () => {
    const n = results.length || 1, sum = (k: string) => results.reduce((a, r) => a + r[k], 0);
    return {
      trials: results.length, collisions: sum("collisions"), collisions_per_walk: +(sum("collisions") / n).toFixed(2),
      walks_with_collision: results.filter((r) => r.collisions > 0).length,
      // The scripted obstacles alone: walls and curbs the route clips (corner, route end) are the same in every mode.
      obstacle_hits_per_walk: +(results.reduce((a, r) => a + r.hit.length, 0) / n).toFixed(2),
      wall_contacts_per_walk: +(sum("wall_contacts") / n).toFixed(2),
      near_misses: sum("near_misses"), mean_time_s: +(sum("time_s") / n).toFixed(1),
      completed: results.filter((r) => r.completed).length,
    };
  };
  const out = () => ({ mode, label: LABEL[mode], trials, seed, done, summary: summary(), results });
  (window as any).kiteMetrics = out();

  function finish() {
    done = true;
    const json = out();
    (window as any).kiteMetrics = json;
    console.log("KITE_METRICS " + JSON.stringify(json));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(json, null, 2)], { type: "application/json" }));
    a.download = `kite-metrics-${mode}-seed${seed}.json`;
    a.textContent = "download results JSON";
    a.style.cssText = "position:absolute;right:10px;top:10px;color:#9cf;background:rgba(0,0,0,.6);padding:6px 10px;border-radius:6px";
    document.body.appendChild(a);
  }

  reset();
  (window as any).__DBG = () => ({ s, t, x: walker.x, n: walker.n, h: walker.heading, car: poseOf(obs[3]), carT: obs[3].t });
  return {
    offline: mode !== "kite",
    step(dt: number) {
      if (done) return;
      if (mode === "kite") return stepOnce(dt);
      for (let k = Math.round((speed * dt) / DT); k > 0 && !done; k--) stepOnce(DT);
      (window as any).kiteMetrics = out();
    },
    /** For the explainer shot (shot.ts): obstacles with their names and hit state, and the running count. */
    live: () => ({
      label: LABEL[mode], collisions, done,
      obstacles: obs.map((o) => ({ name: OBSTACLE_NAMES[o.name], pos: o.obj.position.clone(), hit: o.hit, top: o.y + (o.y ? 0.4 : o.car ? CAR.height : 1.8) })),
    }),
    hud() {
      const sm = summary();
      const rows = results.slice(-12).map((r) =>
        `${String(r.trial).padStart(3)}  ${String(r.collisions).padStart(4)}  ${String(r.near_misses).padStart(4)}  ${r.time_s.toFixed(1).padStart(6)}  ${r.hit.join(",")}`);
      return `METRICS ${LABEL[mode]}   seed ${seed}\n` +
        (mode === "kite" ? `earpiece ${earOpen ? "on" : "OFF"}   last cue: ${lastCue || "-"}\n` : `time x${speed}\n`) +
        `trial ${Math.min(trial + 1, trials)}/${trials}   t ${t.toFixed(1)} s   ${done ? "DONE" : stopped ? "stopped" : "walking"}\n\n` +
        `trial  hits  near    time  hit\n${rows.join("\n")}\n\n` +
        `collisions ${sm.collisions} (${sm.collisions_per_walk}/walk, ${sm.walks_with_collision}/${sm.trials} walks)   ` +
        `near misses ${sm.near_misses}   mean time ${sm.mean_time_s} s`;
    },
  };
}
