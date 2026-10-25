// Every client that reports a position (one per phone: `phone-3fa2`, `phone-9c01`, or plain `phone`),
// with its trail and a short predicted path. Pure, so clients.check.ts runs it in Node.
import { num, type Ev, type LatLng } from './timeline.ts';
import { freshOf, type Fresh } from './view.ts';

export type Fix = { lat: number; lng: number; t: number; accuracy?: number; heading?: number; speed?: number; course?: number; place?: string };
export type Client = { source: string; name: string; color: string; fix: Fix; trail: LatLng[]; ahead?: LatLng; ageS: number; fresh: Fresh };

// Colours that read on the black map. Green and amber are left out: they mean live and stale.
export const PALETTE = ['#64D2FF', '#BF5AF2', '#FFFFFF', '#FFD60A', '#FF6FB5'];
const TRAIL_SIZE = 150;
const AHEAD_S = 10;          // predict this far ahead
const MOVING_MPS = 0.5;      // slower than this is standing still: no prediction
const RECENT_MS = 6_000;     // fixes this recent estimate the velocity when the phone sends no speed
const M_PER_DEG_LAT = 110_540;
const M_PER_DEG_LNG = 111_320;

const mPerDegLng = (lat: number) => M_PER_DEG_LNG * Math.cos((lat * Math.PI) / 180);
/** iOS reports -1 for an unknown speed or course. */
const valid = (v: unknown) => { const n = num(v); return n != null && n >= 0 ? n : undefined; };

export function fixOf(e: Ev): Fix | undefined {
  const lat = num(e.p.lat), lng = num(e.p.lng);
  if (lat == null || lng == null || Math.abs(lat) > 90 || Math.abs(lng) > 180) return undefined;
  const place = typeof e.p.place === 'string' ? e.p.place.slice(0, 60) : undefined;
  return { lat, lng, t: e.t, accuracy: valid(e.p.accuracy), heading: num(e.p.heading), speed: valid(e.p.speed), course: valid(e.p.course), place };
}

/** "phone-3fa2" -> "Phone 3fa2", "phone" -> "Phone", "sim" -> "Sim". */
export function clientName(source: string): string {
  const s = source.replace(/[_-]+/g, ' ').trim().slice(0, 40);
  return s ? s[0].toUpperCase() + s.slice(1) : 'Unknown';
}

/** Where this client will be AHEAD_S from its newest fix, or undefined when it is standing still.
 *  Uses the phone's speed and course when it sends them, else the last few fixes. `fixes` is newest first. */
export function predict(fixes: readonly Fix[]): LatLng | undefined {
  const f = fixes[0];
  if (!f) return undefined;
  if (f.speed != null && f.speed < MOVING_MPS) return undefined;
  if (f.speed != null && f.course != null) {
    const b = (f.course * Math.PI) / 180, d = f.speed * AHEAD_S;
    return [f.lat + (d * Math.cos(b)) / M_PER_DEG_LAT, f.lng + (d * Math.sin(b)) / mPerDegLng(f.lat)];
  }
  const recent = fixes.filter(x => f.t - x.t <= RECENT_MS);
  const o = recent[recent.length - 1];
  const dt = o ? (f.t - o.t) / 1000 : 0;
  if (dt < 0.5) return undefined;
  const vn = ((f.lat - o.lat) * M_PER_DEG_LAT) / dt, ve = ((f.lng - o.lng) * mPerDegLng(f.lat)) / dt;
  if (Math.hypot(vn, ve) < MOVING_MPS) return undefined;
  return [f.lat + (vn * AHEAD_S) / M_PER_DEG_LAT, f.lng + (ve * AHEAD_S) / mPerDegLng(f.lat)];
}

/** One client per position source. `desc` (event table) and `latestDesc` (latest table) are newest first.
 *  Sorted by name so the list and the colours stay put as fixes arrive. */
export function clientsOf(desc: readonly Ev[], latestDesc: readonly Ev[], now: number): Client[] {
  const bySource = new Map<string, Fix[]>();
  for (const e of [...latestDesc, ...desc]) {
    if (e.type !== 'position') continue;
    const f = fixOf(e);
    if (!f) continue;
    const list = bySource.get(e.source) ?? [];
    if (list.length < TRAIL_SIZE) list.push(f);
    bySource.set(e.source, list);
  }
  return [...bySource.keys()].sort().map((source, i) => {
    // The latest row repeats the newest event row; sort and drop same-time duplicates.
    const fixes = bySource.get(source)!.sort((a, b) => b.t - a.t).filter((f, j, a) => j === 0 || f.t !== a[j - 1].t);
    const fix = fixes[0];
    const ageS = Math.max(0, (now - fix.t) / 1000);
    const fresh = freshOf(ageS);
    return {
      source, name: clientName(source), color: PALETTE[i % PALETTE.length], fix, ageS, fresh,
      trail: fixes.map(f => [f.lat, f.lng] as LatLng).reverse(),
      ahead: fresh === 'fresh' ? predict(fixes) : undefined,   // never extrapolate a stale fix
    };
  });
}

/** "1.4 m/s" or '' (standing still reads as nothing). */
export const speedText = (f: Fix) => (f.speed != null && f.speed >= MOVING_MPS ? `${f.speed.toFixed(1)} m/s` : '');
/** "± 7 m" or ''. */
export const accuracyText = (f: Fix) => (f.accuracy != null ? `± ${Math.round(f.accuracy)} m` : '');
