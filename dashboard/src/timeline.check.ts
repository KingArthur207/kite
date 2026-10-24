// Self-check for timeline.ts: `node --experimental-strip-types src/timeline.check.ts` (Node 22+). Throws on failure.
import { happenedAt, lastWalk, liveStats, moments, speakerOf, type Ev } from './timeline.ts';

let n = 0;
const ev = (t: number, type: string, source: string, p: Record<string, unknown>, priority = 'info'): Ev =>
  ({ id: String(n++), t, type, source, priority, p: { type, ...p } });
function eq(actual: unknown, expected: unknown, what: string) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`${what}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`);
}

const T = 1_000_000;
const old = ev(T - 120_000, 'say', 'judgment', { text: 'From an older walk.' });
const walk = [
  old,
  ev(T, 'position', 'sim', { lat: 37.8, lng: -122.4 }),
  ev(T + 1000, 'guidance', 'route', { text: 'Head north.' }),
  ev(T + 1500, 'spoken', 'earpiece', { text: 'Head north.', priority: 'nav' }),      // echo of guidance: dropped
  ev(T + 2000, 'hazard', 'drone', { kind: 'crosswalk', state: 'dont_walk' }),
  ev(T + 2100, 'hazard', 'earpiece', { key: 'crosswalk', state: 'dont_walk', text: "Don't walk." }),  // same hazard
  ev(T + 3000, 'say', 'judgment', { text: 'Wait, the crosswalk is crowded.', latency_ms: 812 }, 'hazard'),
  ev(T + 3200, 'spoken', 'earpiece', { text: 'Wait, the crosswalk is crowded.', priority: 'hazard' }),
  ev(T + 4000, 'position', 'sim', { lat: 37.81, lng: -122.41 }),
  ev(T + 4100, 'spoken', 'earpiece', { text: 'Stop. Person ahead.', priority: 'critical' }),
  ev(T + 5000, 'say', 'caregiver', { text: "I'm on my way." }),
  ev(T + 5500, 'spoken', 'earpiece', { text: "Message from your caregiver: I'm on my way.", priority: 'info' }),
  ev(T + 6000, 'scene', 'scene-chest', { text: 'A person stands two meters ahead.' }),
  ev(T + 7000, 'guidance', 'route', { text: 'Head north.' }),                         // repeat guidance: dropped
];

const w = lastWalk(walk);
eq(w[0].p.type, 'position', 'walk starts after the quiet gap');

const ms = moments(w, 'sim');
eq(ms.map(m => [m.kind, m.who]), [
  ['guidance', 'kite'], ['hazard', 'kite'], ['judgment', 'claude'], ['stop', 'kite'], ['caregiver', 'caregiver'], ['scene', 'claude'],
], 'moments and who');
eq(ms[1].text, "Don't walk.", 'hazard keeps the earpiece wording');
eq(ms[2].detail, '812 ms', 'judgment latency');
eq(ms[3].pos, [37.81, -122.41], 'stop placed at the nearest position');

const desc = [...walk].reverse();
eq(speakerOf(walk[7], desc), 'claude', 'spoken judgment line is Claude');
eq(speakerOf(walk[9], desc), 'kite', 'spoken stop is Kite');
eq(speakerOf(walk[11], desc), 'caregiver', 'spoken caregiver line is the caregiver');

const stats = liveStats(desc, new Map([['sim', T + 4000], ['judgment', T + 3000]]), T + 7000);
eq(stats.perMin, walk.length - 1, 'events in the last minute');
eq(stats.sources.map(s => [s.source, s.perMin]), [['sim', 2], ['judgment', 1]], 'per-source rate');

const now = 1_800_000_000_000;
eq(happenedAt(ev(now, 'say', 'judgment', { t: now / 1000 - 4 })), now - 4000, 'producer epoch time wins');
eq(happenedAt(ev(now, 'say', 'judgment', { t: 12.5 })), now, 'relative replay time falls back to server time');

console.log('timeline.check: ok');
