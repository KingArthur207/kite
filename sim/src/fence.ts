// Geofence for Google 3D tiles: Moss St from Folsom to Howard plus the Howard corner, and nothing else.
// Google's Cal Hacks reps allowed Maps Platform use (tiles and the detector on rendered frames) only
// "to the extent we truly need it, which is just this one street" (see docs/google-permission.md).
// Enforced in tilesWorld.ts: tiles outside the fence are never requested (LoadRegionPlugin mask, plus a
// download check that counts violations), tile pixels outside it are discarded, and the walker is
// clamped inside it (main.ts).
//
// Frame: an oriented box in sim metres (x east, n north). u runs along Moss toward Howard (NW),
// v across it (NE). Built from moss.json: Moss's lane ends and the demo route, plus MARGIN_M.
import moss from "../public/osm/moss.json";

type P = [number, number];
const MARGIN_M = 30;
const HALF_WIDTH_M = 70;        // across Moss: the Howard corner and the buildings on both sides

const data = moss as unknown as { lanes: { name: string; pts: P[] }[]; route: P[] };
const mossLane = data.lanes.find((l) => l.name === "Moss Street")!.pts;
const [howardEnd, folsomEnd] = [mossLane[0], mossLane[mossLane.length - 1]];
const len = Math.hypot(howardEnd[0] - folsomEnd[0], howardEnd[1] - folsomEnd[1]);
/** Unit vector along Moss toward Howard, (east, north). */
export const U: P = [(howardEnd[0] - folsomEnd[0]) / len, (howardEnd[1] - folsomEnd[1]) / len];
const V: P = [-U[1], U[0]];

export const toUV = (x: number, n: number): P => [x * U[0] + n * U[1], x * V[0] + n * V[1]];
const us = [...mossLane, ...data.route].map(([x, n]) => toUV(x, n)[0]);

export const FENCE = {
  uMin: Math.min(...us) - MARGIN_M, uMax: Math.max(...us) + MARGIN_M,
  vMin: -HALF_WIDTH_M, vMax: HALF_WIDTH_M,
};

export function inFence(x: number, n: number, inset = 0): boolean {
  const [u, v] = toUV(x, n);
  return u >= FENCE.uMin + inset && u <= FENCE.uMax - inset && v >= FENCE.vMin + inset && v <= FENCE.vMax - inset;
}

/** The nearest point inside the fence, inset metres from its edge. */
export function clampToFence(x: number, n: number, inset = 0): { x: number; n: number } {
  const [u0, v0] = toUV(x, n);
  const u = Math.min(FENCE.uMax - inset, Math.max(FENCE.uMin + inset, u0));
  const v = Math.min(FENCE.vMax - inset, Math.max(FENCE.vMin + inset, v0));
  return { x: u * U[0] + v * V[0], n: u * U[1] + v * V[1] };
}

// Self-check, run when the tiles world starts: the route is inside, a point 500 m away is not,
// clamping lands inside, and the box is about the size we were allowed.
export function fenceSelfCheck(): void {
  const ok = (c: boolean, m: string) => { if (!c) throw new Error(`fence: ${m}`); };
  for (const [x, n] of data.route) ok(inFence(x, n, 10), `route point ${x},${n} not inside`);
  ok(!inFence(500, 0), "500 m east is inside");
  const c = clampToFence(500, 0, 10);
  ok(inFence(c.x, c.n, 9.9), "clamped point not inside");
  const w = FENCE.uMax - FENCE.uMin, h = FENCE.vMax - FENCE.vMin;
  ok(w > 200 && w < 300 && h > 100 && h < 200, `size ${w.toFixed(0)} x ${h.toFixed(0)} m`);
}
