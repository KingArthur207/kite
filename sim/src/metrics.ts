// Metrics mode (?metrics=1): scripted walks that count collisions with and without Kite.
//   mode=blind   walker never reacts
//   mode=kite    walker reacts ONLY to what the earpiece says/does (ws://127.0.0.1:5011), real time
//   mode=oracle  UPPER BOUND, reacts to ground-truth obstacles 5 m ahead. Never report it as Kite.
// Params: trials (20), seed (1), speed (sim seconds per real second, blind/oracle only, 300).
import * as THREE from "three";
import { WALKER_RADIUS, WALK_SPEED, EARPIECE_URL } from "./config";

const REACTION_S = 0.7;     // human reaction time before stopping
const CLEAR_S = 1.5;        // all-clear time before walking on
const STUCK_S = 5;          // waiting this long on a static thing = feel around it (cane)
const BLOCKED_S = 1;        // pressed against something this long = feel around it
const SIDE_M = 1.5;         // sidestep width
const SIDE_LEN = 7;         // meters walked on the sidestep line before rejoining
const IGNORE_LEN = 3;       // meters after a sidestep where cues are ignored (else static things hold forever)
const NEAR_MISS_M = 0.5;
const TRIAL_MAX_S = 150;
const DT = 0.02;            // fixed step for accelerated modes
const ORACLE_AHEAD = 5, ORACLE_HALF_W = 0.6, ORACLE_LOOKAHEAD_S = [0, 0.5, 1, 1.5];

type Mode = "blind" | "kite" | "oracle";
const LABEL: Record<Mode, string> = {
  blind: "BLIND (no cues)",
  kite: "KITE (reacts only to earpiece output)",
  oracle: "ORACLE: upper bound from ground truth, NOT Kite",
};

// Route in (x east, n north): up the sidewalk, over the crosswalk, around the corner west.
const ROUTE = [{ x: 0, n: 2 }, { x: 0, n: 41.5 }, { x: -10, n: 41.5 }];
const SIDE_DIR = [{ x: -1, n: 0 }, { x: 0, n: 1 }];   // per leg: sidestep away from the road
const LEGS = ROUTE.slice(1).map((b, i) => Math.hypot(b.x - ROUTE[i].x, b.n - ROUTE[i].n));
const ROUTE_LEN = LEGS.reduce((a, b) => a + b, 0);

function routeAt(s: number) {
  let i = 0;
  while (i < LEGS.length - 1 && s > LEGS[i]) s -= LEGS[i++];
  const a = ROUTE[i], b = ROUTE[i + 1], u = Math.min(1, s / LEGS[i]);
  const dx = (b.x - a.x) / LEGS[i], dn = (b.n - a.n) / LEGS[i];
  return { x: a.x + (b.x - a.x) * u, n: a.n + (b.n - a.n) * u, dx, dn, side: SIDE_DIR[i] };
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

// A scripted obstacle: waits at start, starts moving when the walker's route progress hits `trigger`.
interface Obs {
  name: string; mesh: THREE.Mesh; x0: number; n0: number; vx: number; vn: number;
  dur: number; trigger: number; t: number; minClear: number; hit: boolean;
}

export function startMetrics(ctx: {
  walker: { x: number; n: number; heading: number };
  scene: THREE.Scene;
  solids: () => THREE.Box3[];
}) {
  const q = new URLSearchParams(location.search);
  if (q.get("metrics") !== "1") return null;
  const mode = (["blind", "kite", "oracle"].includes(q.get("mode") ?? "") ? q.get("mode") : "blind") as Mode;
  const trials = Math.max(1, Number(q.get("trials")) || 20);
  const seed = Number(q.get("seed")) || 1;
  const speed = mode === "kite" ? 1 : Math.max(1, Number(q.get("speed")) || 300);
  const { walker, scene } = ctx;

  // Freeze the ambient world: hide its low movers/props (cars, people, bin), keep buildings as walls.
  // ponytail: height heuristic on the placeholder world; the splat gets an explicit collision layer.
  const walls = ctx.solids().filter((b) => b.max.y > 3);
  scene.traverse((o) => {
    if (!(o instanceof THREE.Mesh)) return;
    const b = new THREE.Box3().setFromObject(o);
    if (b.max.y > 0.5 && b.max.y < 3) o.visible = false;
  });

  const mk = (geo: THREE.BufferGeometry, color: number, y: number) => {
    const m = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color }));
    m.position.y = y;
    scene.add(m);
    return m;
  };
  const meshes = {
    pedA: mk(new THREE.CapsuleGeometry(0.25, 1.2, 4, 8), 0x7a3e9d, 0.85),
    bin: mk(new THREE.BoxGeometry(0.6, 1.0, 0.6), 0x2e5e3a, 0.5),
    head: mk(new THREE.BoxGeometry(1.2, 0.4, 0.3), 0x8b5a2b, 1.65),   // overhanging sign: above cane height
    car: mk(new THREE.BoxGeometry(4.4, 1.5, 1.8), 0xb22222, 0.75),
    pedB: mk(new THREE.CapsuleGeometry(0.25, 1.2, 4, 8), 0x3e7a5d, 0.85),
  };

  // Seeded per trial. Movers are triggered so they arrive near the walker's arrival, +- jitter.
  function makeObstacles(trial: number): Obs[] {
    const r = mulberry32(seed * 1000 + trial);
    const o = (name: keyof typeof meshes, x0: number, n0: number, vx: number, vn: number, dur: number, trigger: number): Obs =>
      ({ name, mesh: meshes[name], x0, n0, vx, vn, dur, trigger, t: 0, minClear: Infinity, hit: false });
    const nA = 6 + 2 * r(), nB = 15 + 2 * r(), nH = 22 + 2 * r(), xP = -(4 + 2 * r());
    const lead = (sAt: number, secs: number) => sAt - WALK_SPEED * secs;   // trigger point in route meters
    const carT = 6.67 + (r() * 2 - 1) * 1.2;                               // seconds for car to reach x=0
    const pedBs = 41.5 - 2 + -xP;                                           // route s where walker meets pedB
    return [
      o("pedA", 3, nA, -1, 0, 6, lead(nA - 2, 3 + (r() * 2 - 1) * 0.7)),
      o("bin", 0, nB, 0, 0, 0, -1),
      o("head", 0, nH, 0, 0, 0, -1),
      o("car", -8 * carT, 35, 8, 0, 15, 25),
      o("pedB", xP, 43.8, 0, -0.9, 4.8, lead(pedBs, 2.56 + (r() * 2 - 1) * 0.7)),
    ];
  }
  const pos = (o: Obs, t = o.t) => {
    const k = Math.min(t, o.dur);
    return { x: o.x0 + o.vx * k, n: o.n0 + o.vn * k };
  };
  const boxOf = (o: Obs, dt = 0) => {
    const p = pos(o, o.t + dt);
    o.mesh.position.set(p.x, o.mesh.position.y, -p.n);
    return new THREE.Box3().setFromObject(o.mesh);
  };

  // --- kite: listen to the earpiece. Alarm = spoken critical/hazard, a critical sonar track,
  // or a hazard holding the walker (dont_walk/wait/red/stop). Other hazards count via their `spoken`.
  let kiteAlarmAt = -1e9, earOpen = false, lastCue = "";
  if (mode === "kite") {
    const connect = () => {
      const ws = new WebSocket(EARPIECE_URL);
      ws.onopen = () => { earOpen = true; };
      ws.onclose = () => { earOpen = false; setTimeout(connect, 2000); };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => {
        let m: any;
        try { m = JSON.parse(e.data); } catch { return; }
        const now = performance.now() / 1000;
        if (m.type === "spoken" && ["critical", "hazard"].includes(m.priority) && m.result !== "dropped") {
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
  let touching = new Set<string>(), collisions = 0, done = false;

  function reset() {
    obs = makeObstacles(trial);
    t = 0; s = 0; side = 0; sideTarget = 0; sideUntil = 0; ignoreUntil = 0;
    stopped = false; stopAt = Infinity; lastAlarm = -1e9; waited = 0; blockedFor = 0;
    touching = new Set(); collisions = 0;
    placeWalker();
  }
  function placeWalker() {
    const p = routeAt(s);
    walker.x = p.x + p.side.x * side;
    walker.n = p.n + p.side.n * side;
    walker.heading = ((Math.atan2(p.dx, p.dn) * 180) / Math.PI + 360) % 360;
  }

  function oracleAlarm(boxes: { o: Obs; b: THREE.Box3 }[]) {
    const p = routeAt(s);
    let alarm = false;
    for (const { o } of boxes) for (const dt of ORACLE_LOOKAHEAD_S) {
      if (alarm) break;
      const b = boxOf(o, dt);
      // corners of the box in walker frame (along, lateral)
      const xs = [b.min.x, b.max.x], ns = [-b.min.z, -b.max.z];
      let aMin = Infinity, aMax = -Infinity, lMin = Infinity, lMax = -Infinity;
      for (const x of xs) for (const n of ns) {
        const ex = x - walker.x, en = n - walker.n;
        const a = ex * p.dx + en * p.dn, l = ex * p.dn - en * p.dx;
        aMin = Math.min(aMin, a); aMax = Math.max(aMax, a); lMin = Math.min(lMin, l); lMax = Math.max(lMax, l);
      }
      alarm = aMax > 0 && aMin < ORACLE_AHEAD && lMax > -ORACLE_HALF_W && lMin < ORACLE_HALF_W;
    }
    for (const { o } of boxes) boxOf(o);   // put meshes back at the present
    return alarm;
  }

  function stepOnce(dt: number) {
    t += dt;
    for (const o of obs) if (s >= o.trigger && o.dur > 0) o.t += dt;
    const ob = obs.map((o) => ({ o, b: boxOf(o) }));
    const boxes = [...walls, ...ob.map((x) => x.b)];

    // Reaction: any alarm starts a stop after REACTION_S; walk on after CLEAR_S of quiet.
    if (mode !== "blind" && s >= ignoreUntil) {
      const alarm = mode === "oracle" ? oracleAlarm(ob) : performance.now() / 1000 - kiteAlarmAt < 0.3;
      if (alarm) { lastAlarm = t; if (!stopped && stopAt === Infinity) stopAt = t + REACTION_S; }
    }
    if (!stopped && t >= stopAt) { stopped = true; waited = 0; }
    if (stopped) {
      waited += dt;
      if (t - lastAlarm >= CLEAR_S) { stopped = false; stopAt = Infinity; }
      else if (waited > STUCK_S) sidestep();
    }

    if (!stopped) {
      const here = hits(walker.x, walker.n, boxes);
      const lateral = side !== sideTarget;
      const ns = lateral ? s : Math.min(ROUTE_LEN, s + WALK_SPEED * dt);
      const nside = lateral ? side + Math.sign(sideTarget - side) * Math.min(Math.abs(sideTarget - side), WALK_SPEED * dt) : side;
      const p = routeAt(ns);
      const nx = p.x + p.side.x * nside, nn = p.n + p.side.n * nside;
      const blocked = hits(nx, nn, boxes).some((b) => !here.includes(b));
      if (blocked) {
        blockedFor += dt;
        if (blockedFor > BLOCKED_S) sidestep();
      } else {
        blockedFor = 0; s = ns; side = nside;
        if (sideTarget !== 0 && s >= sideUntil) sideTarget = 0;
      }
      placeWalker();
    }

    // Collisions: new contact onsets (10 cm margin, as in main.ts), per box. Near-miss per obstacle.
    const now = new Set<string>();
    const contacts = hits(walker.x, walker.n, boxes, WALKER_RADIUS + 0.1);
    for (const b of contacts) {
      const i = boxes.indexOf(b);
      const id = i < walls.length ? `wall${i}` : ob[i - walls.length].o.name;
      now.add(id);
      if (!touching.has(id)) collisions++;
      if (i >= walls.length) ob[i - walls.length].o.hit = true;
    }
    touching = now;
    for (const { o, b } of ob) o.minClear = Math.min(o.minClear, clearance(walker.x, walker.n, b));

    if (s >= ROUTE_LEN || t >= TRIAL_MAX_S) finishTrial(s >= ROUTE_LEN);
  }

  function sidestep() {
    stopped = false; stopAt = Infinity; blockedFor = 0; waited = 0;
    sideTarget = SIDE_M; sideUntil = s + SIDE_LEN; ignoreUntil = s + IGNORE_LEN;
  }

  function finishTrial(completed: boolean) {
    results.push({
      trial, collisions, near_misses: obs.filter((o) => !o.hit && o.minClear < NEAR_MISS_M).length,
      time_s: +t.toFixed(1), completed,
      hit: obs.filter((o) => o.hit).map((o) => o.name),
      near: obs.filter((o) => !o.hit && o.minClear < NEAR_MISS_M).map((o) => o.name),
    });
    trial++;
    if (trial >= trials) return finish();
    reset();
  }

  const summary = () => {
    const n = results.length || 1, sum = (k: string) => results.reduce((a, r) => a + r[k], 0);
    return {
      trials: results.length, collisions: sum("collisions"), collisions_per_walk: +(sum("collisions") / n).toFixed(2),
      walks_with_collision: results.filter((r) => r.collisions > 0).length,
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
  return {
    offline: mode !== "kite",
    step(dt: number) {
      if (done) return;
      if (mode === "kite") return stepOnce(dt);
      for (let k = Math.round((speed * dt) / DT); k > 0 && !done; k--) stepOnce(DT);
      (window as any).kiteMetrics = out();
    },
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
