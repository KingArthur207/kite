// Dev-only fake clients for `?fixture=two-phones`: rendered locally, never written to SpacetimeDB.
// Two phones walk circles near Sproul Plaza (one sends speed and course, one only fixes), and a third
// went quiet 40 s ago.
import type { Ev } from './timeline.ts';

const CENTER = { lat: 37.8667, lng: -122.2577 };
const M_PER_DEG_LAT = 110_540;
const M_PER_DEG_LNG = 111_320 * Math.cos((CENTER.lat * Math.PI) / 180);
const HISTORY_S = 60;

type Walker = { source: string; radius: number; mps: number; phase: number; sendsSpeed: boolean; quietS: number; accuracy: number };
const WALKERS: Walker[] = [
  { source: 'phone-3fa2', radius: 28, mps: 1.3, phase: 0, sendsSpeed: true, quietS: 0, accuracy: 6 },
  { source: 'phone-9c01', radius: 45, mps: 1.1, phase: 2.4, sendsSpeed: false, quietS: 0, accuracy: 11 },
  { source: 'phone-7b2e', radius: 60, mps: 0, phase: 4.2, sendsSpeed: true, quietS: 40, accuracy: 18 },
];

export const fixtureOn = () => import.meta.env?.DEV && new URLSearchParams(location.search).get('fixture') === 'two-phones';

/** Position events for the last minute, newest first, one per second per phone. */
export function twoPhones(now: number): Ev[] {
  const out: Ev[] = [];
  const t0 = Math.floor(now / 1000) * 1000;
  for (let s = 0; s < HISTORY_S; s++) {
    for (const w of WALKERS) {
      const t = t0 - (s + w.quietS) * 1000;
      const theta = w.phase + (w.mps / w.radius) * (t / 1000);   // radians around the circle
      const north = w.radius * Math.cos(theta), east = w.radius * Math.sin(theta);
      const course = ((((theta * 180) / Math.PI + 90) % 360) + 360) % 360;   // tangent, compass degrees
      const p: Record<string, unknown> = {
        type: 'position', source: w.source, lat: CENTER.lat + north / M_PER_DEG_LAT, lng: CENTER.lng + east / M_PER_DEG_LNG,
        accuracy: w.accuracy, heading: course,
      };
      if (w.sendsSpeed) Object.assign(p, { speed: w.mps, course });
      out.push({ id: `fx-${w.source}-${t}`, type: 'position', source: w.source, priority: 'info', t, p });
    }
  }
  return out;
}
