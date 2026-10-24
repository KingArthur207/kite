import { useEffect, useState } from 'react';
import { tables, reducers } from './module_bindings';
import { useSpacetimeDB, useTable, useReducer } from 'spacetimedb/react';
import { WalkMap, PRIORITY_COLOR, type LatLng, type Pose, type Relative } from './WalkMap';
import './App.css';

// Caregiver dashboard: live view of the shared Kite world state, plus short messages to the walker
// (a `say` from source `caregiver`; the earpiece polls for them and speaks them). No camera video (privacy).

type Payload = Record<string, unknown>;
type Row = { type: string; source: string; priority: string; payload: string; t: { toMillis(): bigint } };

const FEED_SIZE = 40;
const TRAIL_SIZE = 150;
const DEFAULT_HAZARD_TTL_S = 20;
const BEACON_TTL_S = 30;
const FRESH_S = 5;   // device freshness: green under this, amber until STALE_S, red after
const STALE_S = 15;
const MAX_MESSAGE = 200;
const QUICK_MESSAGES = ["I'm on my way.", 'Call me when you can.', 'Take your time, no rush.'];
const SHOWN_MESSAGES = 3;

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
const secs = (now: number, t: number) => Math.max(0, Math.round((now - t) / 1000));
const ago = (now: number, t: number) => `${secs(now, t)}s ago`;
const freshClass = (s: number) => (s < FRESH_S ? 'fresh' : s < STALE_S ? 'stale' : 'dead');
// Hazards from sample data may carry no priority column; the payload or the type default decides.
const priorityOf = (row: Row, p: Payload) => str(p.priority) || row.priority || 'hazard';

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

const poseOf = (p: Payload): Pose | undefined => {
  const lat = num(p.lat), lng = num(p.lng);
  return lat != null && lng != null ? { lat, lng, heading: num(p.heading) } : undefined;
};

function App() {
  const conn = useSpacetimeDB();
  const [events] = useTable(tables.event);
  const [latest] = useTable(tables.latest);
  const clear = useReducer(reducers.clear);
  const push = useReducer(reducers.push);
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const sorted = newestFirst(events);
  const latestSorted = newestFirst(latest);

  // Walker: newest position (any source) that has coordinates; its trail comes from the same source.
  const lastPosRow = latestSorted.find(r => r.type === 'position' && poseOf(parse(r.payload)));
  const pos = lastPosRow ? parse(lastPosRow.payload) : undefined;
  const pose = pos && poseOf(pos);
  const trail: LatLng[] = sorted
    .filter(e => e.type === 'position' && e.source === lastPosRow?.source)
    .slice(0, TRAIL_SIZE)
    .map(e => poseOf(parse(e.payload)))
    .filter((p): p is Pose => p != null)
    .map(p => [p.lat, p.lng] as LatLng)
    .reverse();

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
  const hazardList = [...hazards.entries()];
  const hazardMarks: Relative[] = hazardList.flatMap(([key, { row, p }]) => {
    const bearing = num(p.bearing), distance = num(p.distance);
    return bearing == null || distance == null ? [] :
      [{ key, bearing, distance, label: [p.kind, p.state].filter(Boolean).join(' ') || 'hazard', priority: priorityOf(row, p) }];
  });

  const beaconRow = latestSorted.find(r => r.type === 'beacon' && now - ms(r) <= BEACON_TTL_S * 1000);
  const bp = beaconRow ? parse(beaconRow.payload) : undefined;
  const beacon: Relative | undefined = bp && bp.active !== false && num(bp.bearing) != null && num(bp.distance) != null
    ? { key: 'beacon', bearing: num(bp.bearing)!, distance: num(bp.distance)!, label: str(bp.label) || 'Next waypoint', priority: 'nav' }
    : undefined;

  // Devices: every source that has written anything, with its newest row.
  const devices = new Map<string, Row>();
  for (const r of latestSorted) if (!devices.has(r.source)) devices.set(r.source, r);
  const statusOf = (source: string) => latestSorted.find(r => r.type === 'status' && r.source === source);

  const lastSpoken = latestSorted.find(r => ['guidance', 'spoken', 'say'].includes(r.type));

  const sent = sorted.filter(e => e.type === 'say' && e.source === 'caregiver').slice(0, SHOWN_MESSAGES);

  const send = (raw: string) => {
    const text = raw.trim().slice(0, MAX_MESSAGE);
    if (!text || sending) return;
    const msg = { type: 'say', source: 'caregiver', priority: 'info', text, key: `caregiver-${Date.now()}` };
    setSending(true);
    push({ type: 'say', source: 'caregiver', priority: 'info', payload: JSON.stringify(msg) })
      .then(() => setDraft(''))
      .catch(err => alert(`Message not sent: ${err}`))
      .finally(() => setSending(false));
  };

  const onClear = () => {
    if (confirm('Clear the event log and latest state for a fresh run?')) clear().catch(err => alert(`Clear failed: ${err}`));
  };

  return (
    <div className="app">
      <header>
        <h1>Kite <span className="muted">caregiver view</span></h1>
        <span className={`conn ${conn.isActive ? 'ok' : 'bad'}`}>
          <i />{conn.isActive ? 'Live via SpacetimeDB' : conn.connectionError ? `Disconnected: ${conn.connectionError.message}` : 'Connecting...'}
        </span>
        <button onClick={onClear} disabled={!conn.isActive}>Clear run</button>
      </header>

      <div className="layout">
        <WalkMap pose={pose} trail={trail} hazards={hazardMarks} beacon={beacon}
          freshness={lastPosRow ? ago(now, ms(lastPosRow)) : ''} />

        <aside>
          <section>
            <h2>Walker</h2>
            {pos ? (
              <>
                <div className="big">{str(pos.place) || 'Unknown place'}</div>
                <div className="muted">
                  {summary('position', { ...pos, place: undefined })}
                  {lastPosRow && ` · ${ago(now, ms(lastPosRow))}`}
                </div>
                {beacon && <div className="beacon-line">Next: <strong>{beacon.label}</strong> · {Math.round(beacon.distance)} m</div>}
              </>
            ) : <p className="muted">No position yet.</p>}
          </section>

          <section>
            <h2>Active hazards</h2>
            {hazardList.length === 0 ? <p className="muted">None.</p> : (
              <ul className="hazards">
                {hazardList.map(([key, { row, p }]) => (
                  <li key={key} style={{ borderLeftColor: PRIORITY_COLOR[priorityOf(row, p)] ?? PRIORITY_COLOR.hazard }}
                    className={`p-${priorityOf(row, p)}`}>
                    <strong>{str(p.kind) || 'hazard'}</strong> {summary('hazard', { ...p, kind: undefined })}
                    <div className="muted small">{priorityOf(row, p)} · {row.source} · {ago(now, ms(row))}</div>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section>
            <h2>Message the walker</h2>
            <form className="message" onSubmit={e => { e.preventDefault(); send(draft); }}>
              <input value={draft} onChange={e => setDraft(e.target.value)} maxLength={MAX_MESSAGE}
                placeholder="Kite reads it aloud" aria-label="Message to the walker" disabled={!conn.isActive} />
              <button type="submit" disabled={!conn.isActive || sending || !draft.trim()}>Send</button>
            </form>
            <div className="quick">
              {QUICK_MESSAGES.map(q => (
                <button key={q} onClick={() => send(q)} disabled={!conn.isActive || sending}>{q}</button>
              ))}
            </div>
            {sent.map(e => (
              <div key={e.id.toString()} className="muted small">Sent “{str(parse(e.payload).text)}” · {ago(now, ms(e))}</div>
            ))}
          </section>

          <section>
            <h2>Last spoken</h2>
            {lastSpoken ? (
              <p className="spoken">
                “{str(parse(lastSpoken.payload).text)}”
                <span className="muted small"> {lastSpoken.type} · {ago(now, ms(lastSpoken))}</span>
              </p>
            ) : <p className="muted">Nothing yet.</p>}
          </section>

          <section>
            <h2>Devices</h2>
            {devices.size === 0 ? <p className="muted">Nothing connected yet.</p> : (
              <table className="devices">
                <tbody>
                  {[...devices.values()].map(r => {
                    const st = statusOf(r.source);
                    const s = secs(now, ms(r));
                    return (
                      <tr key={r.source}>
                        <td><span className={`dot ${freshClass(s)}`} /><strong>{r.source}</strong></td>
                        <td className="muted">{st ? summary('status', parse(st.payload)) : r.type}</td>
                        <td className={`age ${freshClass(s)}`}>{s}s ago</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </section>

          <section>
            <h2>Live feed</h2>
            <table className="feed">
              <tbody>
                {sorted.slice(0, FEED_SIZE).map(e => (
                  <tr key={e.id.toString()} className={`p-${e.priority} ${e.source === 'caregiver' ? 'caregiver' : ''}`}>
                    <td className="muted">{new Date(ms(e)).toLocaleTimeString()}</td>
                    <td>{e.type}</td>
                    <td className="muted">{e.source}</td>
                    <td>{summary(e.type, parse(e.payload))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </aside>
      </div>
    </div>
  );
}

export default App;
