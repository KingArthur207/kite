// What the app sends: each GPS fix as an earpiece `position` event (earpiece/earpiece/events.py), the payload of one
// SpacetimeDB `push` call. fix_t (the fix's own time, ms) marks it as the app's: the laptop forwards only rows that
// carry it (phone/cloud.py), never the earpiece's copy of a position it was already given.

export interface Fix {
  t: number; // ms epoch of the GPS fix
  lat: number;
  lng: number;
  acc: number | null; // horizontal accuracy, m
  alt: number | null;
  speed: number | null; // m/s
  course: number | null; // deg
}

// The subset of expo-location's LocationObject we read; kept local so this file stays pure TS.
export interface RawLocation {
  timestamp: number;
  coords: {
    latitude: number;
    longitude: number;
    accuracy: number | null;
    altitude: number | null;
    altitudeAccuracy?: number | null;
    speed: number | null;
    heading: number | null;
  };
}

export interface Position {
  type: 'position';
  source: string;
  lat: number;
  lng: number;
  fix_t: number;
  accuracy?: number;
  speed?: number;
  course?: number;
}

const round = (v: number, places: number): number => Number(v.toFixed(places));

// iOS reports -1 for an unknown speed, course or accuracy: that becomes null, never a sentinel.
function known(v: number | null | undefined, places: number): number | null {
  return v == null || !Number.isFinite(v) || v < 0 ? null : round(v, places);
}

export function toFix(loc: RawLocation): Fix {
  const c = loc.coords;
  const altKnown = c.altitude != null && Number.isFinite(c.altitude) && (c.altitudeAccuracy ?? 0) >= 0;
  return {
    t: Math.round(loc.timestamp),
    lat: round(c.latitude, 7),
    lng: round(c.longitude, 7),
    acc: known(c.accuracy, 1),
    alt: altKnown ? round(c.altitude as number, 1) : null,
    speed: known(c.speed, 2),
    course: known(c.heading, 1),
  };
}

// Unknown values are left out: the earpiece and the dashboard read a missing field as "unknown".
export function toPosition(fix: Fix, source = 'phone'): Position {
  const p: Position = { type: 'position', source, lat: fix.lat, lng: fix.lng, fix_t: fix.t };
  if (fix.acc != null) p.accuracy = fix.acc;
  if (fix.speed != null) p.speed = Math.min(fix.speed, 100); // events.py caps speed at 100 m/s
  if (fix.course != null) p.course = fix.course;
  return p;
}
