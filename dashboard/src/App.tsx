import { useEffect, useState } from 'react';
import { tables, reducers } from './module_bindings';
import { useSpacetimeDB, useTable, useReducer } from 'spacetimedb/react';
import './App.css';

// Caregiver dashboard: read-only view of the shared Kite world state. No camera video (privacy).

type Payload = Record<string, unknown>;
type Row = { type: string; source: string; priority: string; payload: string; t: { toMillis(): bigint } };

const FEED_SIZE = 50;
const TRAIL_SIZE = 100;
const DEFAULT_HAZARD_TTL_S = 20;
const M_PER_DEG_LAT = 110_540;
const M_PER_DEG_LNG = 111_320;

function parse(json: string): Payload {
  try {
    const v = JSON.parse(json);
    return v && typeof v === 'object' ? v : {};
  } catch {
    return {};
  }
}

const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
const str = (v: unknown): string => (v == null ? '' : String(v));
const ms = (r: Row) => Number(r.t.toMillis());
const newestFirst = <T extends Row>(rows: readonly T[]) => [...rows].sort((a, b) => ms(b) - ms(a));
const m = (v: unknown) => (num(v) != null ? `${num(v)!.toFixed(1)} m` : false);
const deg = (v: unknown) => (num(v) != null ? `${Math.round(num(v)!)}°` : false);
const ago = (now: number, t: number) => `${Math.max(0, Math.round((now - t) / 1000))}s ago`;

function summary(type: string, p: Payload): string {
  switch (type) {
    case 'hazard':
      return [p.kind, p.state, p.text, m(p.distance), deg(p.bearing)]
        .filter(Boolean).join(' · ');
    case 'position':
      return [p.place, num(p.lat) != null && `${num(p.lat)!.toFixed(5)}, ${num(p.lng)?.toFixed(5)}`,
        num(p.heading) != null && `hdg ${deg(p.heading)}`].filter(Boolean).join(' · ');
    case 'status':
      return [p.state, p.battery != null && `${p.battery}%`, p.text].filter(Boolean).join(' · ');
    case 'beacon':
      return [p.label, m(p.distance), deg(p.bearing)].filter(Boolean).join(' · ');
    case 'obstacles':
      return `${Array.isArray(p.objects) ? p.objects.length : 0} objects`;
    default:
      return str(p.text ?? p.action ?? p.kind ?? '');
  }
}

// Project positions to local meters (or use x/y if the producer sends them). Missing coords are skipped.
function trailPoints(positions: Payload[]): [number, number][] {
  const pts: [number, number][] = [];
  const origin = positions.find(p => num(p.lat) != null && num(p.lng) != null);
  for (const p of positions) {
    const x = num(p.x), y = num(p.y), lat = num(p.lat), lng = num(p.lng);
    if (x != null && y != null) pts.push([x, y]);
    else if (lat != null && lng != null && origin) {
      const lat0 = origin.lat as number, lng0 = origin.lng as number;
      pts.push([(lng - lng0) * M_PER_DEG_LNG * Math.cos((lat0 * Math.PI) / 180), (lat - lat0) * M_PER_DEG_LAT]);
    }
  }
  return pts;
}

function Trail({ points }: { points: [number, number][] }) {
  if (points.length === 0) return <p className="muted">No coordinates yet (position events have no lat/lng or x/y).</p>;
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const pad = 5;
  const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
  const minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
  const w = maxX - minX, h = maxY - minY;
  // SVG y grows down; flip so north is up.
  const svgPts = points.map(([x, y]) => `${x - minX},${maxY - y}`).join(' ');
  const [lx, ly] = points[points.length - 1];
  return (
    <svg className="trail" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet">
      <polyline points={svgPts} fill="none" stroke="#4da3ff" strokeWidth={Math.max(w, h) / 150} vectorEffect="non-scaling-stroke" />
      <circle cx={lx - minX} cy={maxY - ly} r={Math.max(w, h) / 40} fill="#ffd34d" />
    </svg>
  );
}

function App() {
  const conn = useSpacetimeDB();
  const [events] = useTable(tables.event);
  const [latest] = useTable(tables.latest);
  const clear = useReducer(reducers.clear);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const sorted = newestFirst(events);
  const latestSorted = newestFirst(latest);

  const positionRows = sorted.filter(e => e.type === 'position').slice(0, TRAIL_SIZE).reverse();
  const lastPosRow = latestSorted.find(r => r.type === 'position');
  const pos = lastPosRow ? parse(lastPosRow.payload) : undefined;

  // Active hazards: newest per (source, kind) still inside its ttl.
  const hazards = new Map<string, { row: Row; p: Payload }>();
  for (const e of sorted) {
    if (e.type !== 'hazard') continue;
    const p = parse(e.payload);
    const key = `${e.source}:${str(p.kind)}`;
    if (hazards.has(key)) continue;
    const ttl = num(p.ttl) ?? DEFAULT_HAZARD_TTL_S;
    if (now - ms(e) <= ttl * 1000) hazards.set(key, { row: e, p });
  }

  const statuses = latestSorted.filter(r => r.type === 'status');
  const lastSpoken = latestSorted.find(r => ['guidance', 'spoken', 'say'].includes(r.type));

  const onClear = () => {
    if (confirm('Clear the event log and latest state for a fresh run?')) clear().catch(err => alert(`Clear failed: ${err}`));
  };

  return (
    <main>
      <header>
        <h1>Kite</h1>
        <span className={conn.isActive ? 'ok' : 'bad'}>
          {conn.isActive ? 'Live' : conn.connectionError ? `Disconnected: ${conn.connectionError.message}` : 'Connecting...'}
        </span>
        <button onClick={onClear} disabled={!conn.isActive}>Clear run</button>
      </header>

      <div className="grid">
        <section>
          <h2>Walker</h2>
          {pos ? (
            <>
              <div className="big">{str(pos.place) || 'Unknown place'}</div>
              <div className="muted">
                {num(pos.lat) != null ? `${num(pos.lat)!.toFixed(5)}, ${num(pos.lng)?.toFixed(5)}` : 'no coordinates'}
                {num(pos.heading) != null && ` · heading ${deg(pos.heading)}`}
                {lastPosRow && ` · ${ago(now, ms(lastPosRow))}`}
              </div>
            </>
          ) : <p className="muted">No position yet.</p>}
          <Trail points={trailPoints(positionRows.map(r => parse(r.payload)))} />
        </section>

        <section>
          <h2>Active hazards</h2>
          {hazards.size === 0 ? <p className="muted">None.</p> : (
            <ul className="hazards">
              {[...hazards.values()].map(({ row, p }) => (
                <li key={`${row.source}:${str(p.kind)}`} className={row.priority === 'critical' ? 'critical' : ''}>
                  <strong>{str(p.kind) || 'hazard'}</strong> {summary('hazard', { ...p, kind: undefined })}
                  <span className="muted"> · {row.source} · {ago(now, ms(row))}</span>
                </li>
              ))}
            </ul>
          )}

          <h2>Last spoken</h2>
          {lastSpoken ? (
            <p className="big">
              “{str(parse(lastSpoken.payload).text)}”
              <span className="muted small"> {lastSpoken.type} · {ago(now, ms(lastSpoken))}</span>
            </p>
          ) : <p className="muted">Nothing yet.</p>}

          <h2>Devices</h2>
          {statuses.length === 0 ? <p className="muted">No status yet.</p> : (
            <table>
              <tbody>
                {statuses.map(r => (
                  <tr key={r.key}>
                    <td><strong>{r.source}</strong></td>
                    <td>{summary('status', parse(r.payload))}</td>
                    <td className="muted">{ago(now, ms(r))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>

      <section>
        <h2>Live feed</h2>
        <table className="feed">
          <tbody>
            {sorted.slice(0, FEED_SIZE).map(e => (
              <tr key={e.id.toString()} className={`p-${e.priority}`}>
                <td className="muted">{new Date(ms(e)).toLocaleTimeString()}</td>
                <td>{e.type}</td>
                <td>{e.source}</td>
                <td>{e.priority}</td>
                <td>{summary(e.type, parse(e.payload))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}

export default App;
