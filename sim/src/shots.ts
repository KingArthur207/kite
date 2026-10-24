// Cinematic shots for the demo video (?shot=<name>, recorded by scripts/record_shot.py). The sim is a
// recreation shown next to the real Tello footage of Moss St; these are its camera moves.
// Frame: x east, y up, n north (three.js z = -n), metres from the Moss zebra. Route s = metres along the demo
// route: 0 at the spawn on Moss, about 70 at the zebra curb, 78 across it on Howard, 98 at the end.
import * as THREE from "three";
import { HEAD_TOP, DRONE_ABOVE_HEAD, VFOV_DEG } from "./config";
import { route } from "./osmWorld";
import { Path } from "./mossActors";

export const ROUTE = new Path(route);
export const S_CURB = ROUTE.s[1];          // the NE curb of the zebra across Moss
const DEG = Math.PI / 180;

/** What a shot's camera can follow, all in sim three.js coordinates. */
export interface ShotCtx {
  t: number;
  walker: THREE.Vector3;        // feet, smoothed
  heading: number;              // compass degrees, smoothed
  drone: THREE.Vector3;
}
export interface CamPose { pos: THREE.Vector3; look: THREE.Vector3; up?: THREE.Vector3 }

export interface Cue {
  from: number; to: number;     // shot seconds
  text: string;
  ring?: boolean;               // pulse rings on the ground at the walker
  color?: string;
}

export interface Shot {
  /** Query defaults; the page URL wins (so ?world=moss on any shot works). */
  query: Record<string, string>;
  duration: number;             // s
  fov: number;                  // vertical, degrees
  /** Scripted walker: starts at route s `from`, brakes to a stop at each `at` and walks on at `until`. */
  walk?: { from: number; stops: { at: number; until: number }[] };
  /** Scripted car turning in from Howard's left lane: its front crosses the zebra at this shot time. */
  carAtZebra?: number;
  route?: { width: number; reveal: [number, number] };   // drawn route line; reveal from/to shot seconds
  cues?: Cue[];
  fog?: [number, number];
  hideDrone?: boolean;          // the camera is the drone
  marker?: boolean;             // a steady ring under the walker, for wide shots
  still: number;                // shot second for the README still
  camera(k: ShotCtx): CamPose;
}

// --- keyframe helpers

const v = (x: number, n: number, y: number) => new THREE.Vector3(x, y, -n);
const routePt = (s: number, y = 0) => { const p = ROUTE.at(s); return v(p.x, p.n, y); };
/** A point `back` metres behind route s (along the route's direction there), `left` to its left, `y` up. */
const behind = (s: number, back: number, left: number, y: number) => {
  const p = ROUTE.at(s);
  return v(p.x - p.dx * back - p.dn * left, p.n - p.dn * back + p.dx * left, y);
};

/** Eased keyframe timing: monotone cubic (Fritsch-Butland) from key times to key index, flat at both ends,
 *  so the move eases in and out and never stops or reverses at an inner key. */
export function easedIndex(ts: number[], t: number): number {
  const n = ts.length;
  if (t <= ts[0]) return 0;
  if (t >= ts[n - 1]) return n - 1;
  const d = ts.slice(1).map((x, i) => 1 / (x - ts[i]));                     // secant slopes (index per second)
  const m = ts.map((_, i) => (i === 0 || i === n - 1 ? 0 : (2 * d[i - 1] * d[i]) / (d[i - 1] + d[i])));
  let i = 0;
  while (t > ts[i + 1]) i++;
  const h = ts[i + 1] - ts[i], u = (t - ts[i]) / h, u2 = u * u, u3 = u2 * u;
  return i * (2 * u3 - 3 * u2 + 1) + m[i] * h * (u3 - 2 * u2 + u) + (i + 1) * (-2 * u3 + 3 * u2) + m[i + 1] * h * (u3 - u2);
}

/** A Catmull-Rom camera path through keyframes {t, pos, look}, timed by easedIndex. */
export function track(keys: { t: number; pos: THREE.Vector3; look: THREE.Vector3 }[]) {
  const ts = keys.map((k) => k.t);
  const pos = new THREE.CatmullRomCurve3(keys.map((k) => k.pos), false, "centripetal");
  const look = new THREE.CatmullRomCurve3(keys.map((k) => k.look), false, "centripetal");
  return (t: number): CamPose => {
    const u = easedIndex(ts, t) / (keys.length - 1);   // CatmullRomCurve3.getPoint is by key index, not arc length
    return { pos: pos.getPoint(u), look: look.getPoint(u) };
  };
}

/** A scalar eased through keyframes [t, value]. */
export function keyed(keys: [number, number][]) {
  return (t: number) => {
    const f = easedIndex(keys.map((k) => k[0]), t), i = Math.min(keys.length - 2, Math.floor(f)), u = f - i;
    return keys[i][1] + (keys[i + 1][1] - keys[i][1]) * u;
  };
}

// --- the shots

// (a) Establishing: high over the Moss block, descending onto the walker at the start of the route, with the
// route drawn. Tiles only exist inside the geofence (fence.ts), so the camera stays over it and frames along
// Moss; past the fence edge there is no city, by design.
const establishing = (() => {
  const mid = v(62, -62, 0);                                  // middle of Moss between Howard and Folsom
  const cam = track([
    { t: 0, pos: v(78, -88, 175), look: v(52, -52, 0) },
    { t: 6, pos: v(92, -86, 105), look: mid.clone().lerp(v(25, -20, 0), 0.6) },
    { t: 11, pos: behind(14, 34, 4, 34), look: routePt(48, 0) },
    { t: 15, pos: behind(16, 20, 3.5, 15), look: routePt(42, 0) },
  ]);
  return {
    query: { world: "tiles" },
    duration: 15, fov: 50, fog: [400, 1400], still: 9,
    walk: { from: 0, stops: [] },
    route: { width: 1.6, reveal: [1, 7] },
    camera: (k) => cam(k.t),
  } satisfies Shot;
})();

// (b) Follow: the drone's own view (1 m behind, 0.3 m above the head, Tello vertical FOV), from the spawn up
// Moss to the zebra curb. The walker stops at the curb.
const follow: Shot = {
  query: { world: "tiles" },
  duration: 62, fov: VFOV_DEG, still: 20, hideDrone: true,
  walk: { from: 0, stops: [{ at: S_CURB - 0.8, until: 1e9 }] },
  camera: (k) => {
    const h = k.heading * DEG, f = new THREE.Vector3(Math.sin(h), 0, -Math.cos(h));
    return { pos: k.drone.clone(), look: k.drone.clone().addScaledVector(f, 10).setY(k.drone.y - 10 * Math.tan(4 * DEG)) };
  },
};

// (c) The corner: the walker stops short of the zebra as a car turns in from Howard without yielding, waits,
// then crosses. The camera orbits slowly over Howard (the open side of the corner; the other side is buildings).
const corner = (() => {
  const zebra = v(0, 0, 0);
  const angle = keyed([[0, 355], [24, 290]]);     // compass bearing of the camera from the zebra: N to WNW, over Howard
  const radius = keyed([[0, 17], [12, 13], [24, 15]]);
  const height = keyed([[0, 7], [24, 4.5]]);
  return {
    query: { world: "tiles" },
    duration: 24, fov: 42, still: 10.4,
    walk: { from: S_CURB - 11, stops: [{ at: S_CURB - 1.6, until: 13.6 }] },
    carAtZebra: 10.2,
    cues: [
      { from: 7.6, to: 12.6, text: "Car turning in", ring: true, color: "#ffb020" },
      { from: 8.3, to: 12.6, text: "Kite: “Stop.”", color: "#ffffff" },
    ],
    camera: (k) => {
      const a = angle(k.t) * DEG, r = radius(k.t);
      const pos = zebra.clone().add(new THREE.Vector3(Math.sin(a) * r, height(k.t), -Math.cos(a) * r));
      return { pos, look: k.walker.clone().lerp(zebra, 0.45).setY(1.1) };
    },
  } satisfies Shot;
})();

// (d) Top-down explainer of the metrics walk (metrics.ts, blind mode, one seeded walk at 2.4x): screen up is NW
// along Moss, so the walker goes up the screen, then left across the zebra and along Howard.
const explainer = (() => {
  const up = new THREE.Vector3(-1, 0, -1).normalize();                    // NW in three.js (x east, z south)
  const whole = v(18, -20, 0);                                            // route centre
  const height = keyed([[0, 115], [4.5, 30], [35.5, 30], [40, 115]]);
  const follow = keyed([[0, 0], [4.5, 1], [35.5, 1], [40, 0]]);               // 0 = whole route, 1 = on the walker
  return {
    query: { world: "moss", metrics: "1", mode: "blind", trials: "1", seed: "1", speed: "2.4" },
    duration: 40, fov: 45, still: 15, marker: true,
    route: { width: 0.5, reveal: [0, 3] },
    camera: (k) => {
      const c = whole.clone().lerp(k.walker.clone().addScaledVector(up, 6), follow(k.t)).setY(0);
      return { pos: c.clone().setY(height(k.t)), look: c, up };
    },
  } satisfies Shot;
})();

export const SHOTS: Record<string, Shot> = { establishing, follow, corner, explainer };
export const DRONE_HEIGHT = HEAD_TOP + DRONE_ABOVE_HEAD;
