// Self-check for clients.ts: `npm run check`. Throws on failure.
import { clientName, clientsOf, predict, type Fix } from './clients.ts';
import type { Ev } from './timeline.ts';

function ok(cond: boolean, what: string) { if (!cond) throw new Error(what); }
let n = 0;
const pos = (t: number, source: string, p: Record<string, unknown>): Ev => ({ id: String(n++), t, type: 'position', source, priority: 'info', p });

ok(clientName('phone-3fa2') === 'Phone 3fa2', 'phone name');
ok(clientName('phone') === 'Phone', 'plain phone name');

// Speed + course: 1 m/s due north for 10 s is about 10 m north.
const north = predict([{ lat: 37.8667, lng: -122.2577, t: 0, speed: 1, course: 0 }])!;
ok(Math.abs((north[0] - 37.8667) * 110_540 - 10) < 0.01 && Math.abs(north[1] + 122.2577) < 1e-9, 'dead reckoning north');
ok(predict([{ lat: 37, lng: -122, t: 0, speed: 0.2, course: 90 }]) === undefined, 'standing still: no prediction');

// No speed: from the fixes. 2 m east per second -> about 20 m east in 10 s.
const dLng = 2 / (111_320 * Math.cos((37 * Math.PI) / 180));
const fixes: Fix[] = [0, 1, 2].map(i => ({ lat: 37, lng: -122 + dLng * (2 - i), t: (2 - i) * 1000 }));
const east = predict(fixes)!;
ok(Math.abs((east[1] - fixes[0].lng) / dLng - 10) < 0.01, 'velocity from fixes');
ok(predict([fixes[0], { ...fixes[0], t: 0 }]) === undefined, 'same place twice: standing still');

// Grouping: one client per source, newest fix wins, bad coordinates skipped, stale gets no prediction.
const now = 100_000;
const desc = [
  pos(99_000, 'phone-a', { lat: 37.1, lng: -122, speed: 1, course: 0 }),
  pos(98_000, 'phone-a', { lat: 37.0, lng: -122 }),
  pos(97_000, 'phone-b', { lat: 'x', lng: -122 }),
  pos(60_000, 'phone-b', { lat: 37.2, lng: -122, speed: 1, course: 0, accuracy: -1 }),
];
const cs = clientsOf(desc, [], now);
ok(cs.length === 2 && cs[0].source === 'phone-a' && cs[1].source === 'phone-b', 'two clients sorted');
ok(cs[0].fix.lat === 37.1 && cs[0].trail.length === 2 && cs[0].ahead != null && cs[0].fresh === 'fresh', 'phone-a newest fix and prediction');
ok(cs[1].fresh === 'dead' && cs[1].ahead === undefined && cs[1].fix.accuracy === undefined, 'phone-b stale, -1 accuracy ignored');
ok(cs[0].color !== cs[1].color, 'distinct colours');
console.log('clients.check: ok');
