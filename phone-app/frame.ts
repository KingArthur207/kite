// The wire format the computer expects (phone/fix.py): one compact JSON fix per line, chunked to fit a BLE write.

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

const DEFAULT_WRITE = 20; // ATT default MTU 23 minus the 3-byte header
const MAX_WRITE = 512; // ATT attribute value limit

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

export function frameFix(fix: Fix): string {
  return JSON.stringify(fix) + '\n';
}

// Bytes per write for a negotiated MTU; ble-plx reports the MTU, the payload is 3 bytes less.
export function writeSize(mtu: number | null | undefined): number {
  return mtu && mtu > 23 ? Math.min(mtu - 3, MAX_WRITE) : DEFAULT_WRITE;
}

// The line is ASCII (JSON of numbers), so string length equals byte length.
export function chunk(line: string, size: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < line.length; i += size) out.push(line.slice(i, i + size));
  return out;
}
