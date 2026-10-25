// Pure display helpers: distances and directions, freshness and connection state.
// No React here, so view.check.ts runs them in Node.
import { num, type Payload } from './timeline.ts';

// ---- Freshness: green under FRESH_S since a source last wrote, amber until STALE_S, red after.
export const FRESH_S = 5;
export const STALE_S = 15;
export type Fresh = 'fresh' | 'stale' | 'dead';
export const freshOf = (ageS: number): Fresh => (ageS < FRESH_S ? 'fresh' : ageS < STALE_S ? 'stale' : 'dead');
/** Words for the dot colour, so freshness never depends on colour alone. */
export const FRESH_LABEL: Record<Fresh, string> = { fresh: 'live', stale: 'late', dead: 'silent' };

/** "just now", "12 s ago", "4 min ago", "2 h ago". */
export function ago(ageS: number): string {
  const s = Math.max(0, Math.round(ageS));
  if (s < 2) return 'just now';
  if (s < 90) return `${s} s ago`;
  if (s < 90 * 60) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}
export const ageS = (now: number, t: number) => Math.max(0, (now - t) / 1000);

// ---- Distances and directions in the walker's frame (bearing 0 ahead, +90 right).
export const MAX_RANGE_M = 500;   // anything farther (or negative, or not a number) is a bad payload, not a hazard

/** Bearing and distance when both are usable for the map, else undefined. */
export function relative(p: Payload): { bearing: number; distance: number } | undefined {
  const bearing = num(p.bearing), distance = num(p.distance);
  return bearing != null && distance != null && Math.abs(bearing) <= 360 && distance >= 0 && distance <= MAX_RANGE_M
    ? { bearing, distance } : undefined;
}

/** "under 1 m", "1.4 m", "8 m", "120 m"; '' when not a usable distance. */
export function meters(d: unknown): string {
  const v = num(d);
  if (v == null || v < 0 || v > MAX_RANGE_M * 20) return '';
  if (v < 1) return 'under 1 m';
  return v < 10 ? `${Number(v.toFixed(1))} m` : `${Math.round(v)} m`;
}

/** Walker-frame bearing as words: "ahead", "ahead right", "right", "behind right", "behind"... */
export function direction(bearing: unknown): string {
  const b0 = num(bearing);
  if (b0 == null) return '';
  const b = ((((b0 + 180) % 360) + 360) % 360) - 180;   // -180..180
  const a = Math.abs(b), side = b < 0 ? 'left' : 'right';
  if (a <= 15) return 'ahead';
  if (a <= 60) return `ahead ${side}`;
  if (a <= 120) return side;
  if (a <= 165) return `behind ${side}`;
  return 'behind';
}

const COMPASS = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west'];
/** Compass heading (0 north, 90 east) as "facing north-west". */
export function facing(heading: unknown): string {
  const h = num(heading);
  return h == null ? '' : `facing ${COMPASS[Math.round((((h % 360) + 360) % 360) / 45) % 8]}`;
}

/** "bike_lane" -> "bike lane". */
export const humanize = (s: string) => s.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();

// ---- Connection.
export type Link = 'connecting' | 'live' | 'reconnecting' | 'offline';
/** The SDK reconnects by itself; once a session was live, a drop is "reconnecting", not "offline". */
export function linkOf(isActive: boolean, everLive: boolean, error?: unknown): Link {
  if (isActive) return 'live';
  if (everLive) return 'reconnecting';
  return error ? 'offline' : 'connecting';
}
export const QUIET_S = 30;   // no row from any device for this long: say so instead of looking frozen

