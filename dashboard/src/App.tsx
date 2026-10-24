import { useEffect, useMemo, useState } from 'react';
import { tables, reducers } from './module_bindings';
import { useSpacetimeDB, useTable, useReducer } from 'spacetimedb/react';
import { WalkMap, PRIORITY_COLOR, type LatLng, type Pin, type Pose, type Relative } from './WalkMap';
import { happenedAt, lastWalk, liveStats, moments as walkMoments, num, parse, speakerOf, str, type Ev, type Moment, type Payload } from './timeline';
import { Badge, LiveState, Timeline } from './Reveal';
import './App.css';

// Caregiver dashboard: live view of the shared Kite world state, plus short messages to the walker
// (a `say` from source `caregiver`; the earpiece polls for them and speaks them). No camera video (privacy).
// Press P for presentation mode (the reveal): no raw feed or controls, bigger map, timeline and last line.

// `event` rows have an id, `latest` rows a "type:source" key.
type Row = { id?: bigint; key?: string; type: string; source: string; priority: string; payload: string; t: { toMillis(): bigint } };

const FEED_SIZE = 40;
const TRAIL_SIZE = 150;
const DEFAULT_HAZARD_TTL_S = 20;
const BEACON_TTL_S = 30;
const MAX_MESSAGE = 200;
const EVENT_TABLE_CAP = 500;   // the module keeps the newest 500 events
// The earpiece drops any message that could be heard as permission to move ("you can", "go", "clear"...),
// so the quick ones avoid those words.
const QUICK_MESSAGES = ["I'm on my way.", 'Call me when you get there.', 'Take your time, no rush.'];
// Same value as KITE_CAREGIVER_TOKEN on the earpiece. With it set, the earpiece speaks only messages signed
// with it. It ships in this page's JS, so serve the dashboard only on the demo laptop.
const CAREGIVER_TOKEN: string = import.meta.env.VITE_KITE_CAREGIVER_TOKEN ?? '';

// HMAC-SHA256(token, `${ts}:${text}`) as hex: earpiece/earpiece/spacetime.py CaregiverInbox checks it.
async function signMessage(text: string): Promise<{ ts: number; sig: string } | Record<string, never>> {
  if (!CAREGIVER_TOKEN) return {};
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey('raw', enc.encode(CAREGIVER_TOKEN), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const ts = Date.now();
  const mac = new Uint8Array(await crypto.subtle.sign('HMAC', key, enc.encode(`${ts}:${text}`)));
  return { ts, sig: Array.from(mac, b => b.toString(16).padStart(2, '0')).join('') };
}
const SHOWN_MESSAGES = 3;
const PRESENT_SOURCES = 6;   // busiest sources shown in presentation mode
// Moment colours on the map; stops are always red.
const PIN_COLOR: Record<string, string> = { stop: '#ef4444', kite: '#2dd4bf', claude: '#e8875f', caregiver: '#4da3ff' };

const toEv = (r: Row): Ev => ({ id: r.id?.toString() ?? r.key ?? '', type: r.type, source: r.source, priority: r.priority, t: Number(r.t.toMillis()), p: parse(r.payload) });
const newestFirst = (rows: readonly Ev[]) => [...rows].sort((a, b) => b.t - a.t);
const m = (v: unknown) => (num(v) != null ? `${num(v)!.toFixed(1)} m` : false);
const deg = (v: unknown) => (num(v) != null ? `${Math.round(num(v)!)}°` : false);
const secs = (now: number, t: number) => Math.max(0, Math.round((now - t) / 1000));
const ago = (now: number, t: number) => `${secs(now, t)}s ago`;
// Hazards from sample data may carry no priority column; the payload or the type default decides.
const priorityOf = (e: Ev) => str(e.p.priority) || e.priority || 'hazard';

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

const typing = (el: EventTarget | null) => el instanceof HTMLElement && (el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName));

function App() {
  const conn = useSpacetimeDB();
  const [events] = useTable(tables.event);
  const [latest] = useTable(tables.latest);
  const clear = useReducer(reducers.clear);
  const push = useReducer(reducers.push);
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [present, setPresent] = useState(() => location.hash === '#present');
  const [focusId, setFocusId] = useState<string>();

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== 'p' || e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return;
      setPresent(on => !on);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  useEffect(() => { history.replaceState(null, '', present ? '#present' : location.pathname + location.search); }, [present]);

  // Parse every payload once per table change, not once per render (the clock re-renders every second).
  const sorted = useMemo(() => newestFirst(events.map(toEv)), [events]);
  const latestSorted = useMemo(() => newestFirst(latest.map(toEv)), [latest]);

  // Walker: newest position (any source) that has coordinates; its trail comes from the same source.
  const lastPosRow = latestSorted.find(r => r.type === 'position' && poseOf(r.p));
  const pos = lastPosRow?.p;
  const pose = pos && poseOf(pos);
  const walkerSource = lastPosRow?.source;
  const trail: LatLng[] = useMemo(() => sorted
    .filter(e => e.type === 'position' && e.source === walkerSource)
    .slice(0, TRAIL_SIZE)
    .map(e => poseOf(e.p))
    .filter((p): p is Pose => p != null)
    .map(p => [p.lat, p.lng] as LatLng)
    .reverse(), [sorted, walkerSource]);

  // The last walk's key moments, oldest first.
  const walk = useMemo(() => lastWalk([...sorted].reverse()), [sorted]);
  const moments: Moment[] = useMemo(() => walkMoments(walk, walkerSource), [walk, walkerSource]);
  const pins: Pin[] = moments.flatMap(mo => mo.pos ? [{
    id: mo.id, at: mo.pos, stop: mo.kind === 'stop', label: mo.text,
    color: PIN_COLOR[mo.kind === 'stop' ? 'stop' : mo.who],
  }] : []);

  // Active hazards: newest per (source, kind) still inside its ttl.
  const hazards = new Map<string, Ev>();
  for (const e of sorted) {
    if (e.type !== 'hazard') continue;
    const key = `${e.source}:${str(e.p.kind)}`;
    if (hazards.has(key)) continue;
    const ttl = num(e.p.ttl) ?? DEFAULT_HAZARD_TTL_S;
    if (now - e.t <= ttl * 1000) hazards.set(key, e);
  }
  const hazardList = [...hazards.entries()];
  const hazardMarks: Relative[] = hazardList.flatMap(([key, e]) => {
    const bearing = num(e.p.bearing), distance = num(e.p.distance);
    return bearing == null || distance == null ? [] :
      [{ key, bearing, distance, label: [e.p.kind, e.p.state].filter(Boolean).join(' ') || 'hazard', priority: priorityOf(e) }];
  });

  const beaconRow = latestSorted.find(r => r.type === 'beacon' && now - r.t <= BEACON_TTL_S * 1000);
  const bp = beaconRow?.p;
  const beacon: Relative | undefined = bp && bp.active !== false && num(bp.bearing) != null && num(bp.distance) != null
    ? { key: 'beacon', bearing: num(bp.bearing)!, distance: num(bp.distance)!, label: str(bp.label) || 'Next waypoint', priority: 'nav' }
    : undefined;

  // Shared state: every source that has written anything, with its newest row time and last-minute rate.
  const newestBySource = new Map<string, number>();
  for (const r of latestSorted) if (!newestBySource.has(r.source)) newestBySource.set(r.source, r.t);
  const live = liveStats(sorted, newestBySource, now);
  // Every kept row is inside the last minute, so older ones were trimmed: the rate is a lower bound.
  const capped = sorted.length >= EVENT_TABLE_CAP && live.perMin >= sorted.length;
  const statusOf = (source: string) => {
    const st = latestSorted.find(r => r.type === 'status' && r.source === source);
    return st ? summary('status', st.p) : '';
  };

  const lastSpoken = latestSorted.find(r => ['guidance', 'spoken', 'say'].includes(r.type) && str(r.p.text));

  const sent = sorted.filter(e => e.type === 'say' && e.source === 'caregiver').slice(0, SHOWN_MESSAGES);

  const send = (raw: string) => {
    const text = raw.trim().slice(0, MAX_MESSAGE);
    if (!text || sending) return;
    setSending(true);
    signMessage(text)
      .then(sig => {
        const msg = { type: 'say', source: 'caregiver', priority: 'info', text, key: `caregiver-${Date.now()}`, ...sig };
        return push({ type: 'say', source: 'caregiver', priority: 'info', payload: JSON.stringify(msg) });
      })
      .then(() => setDraft(''))
      .catch(err => alert(`Message not sent: ${err}`))
      .finally(() => setSending(false));
  };

  const onClear = () => {
    if (confirm('Clear the event log and latest state for a fresh run?')) clear().catch(err => alert(`Clear failed: ${err}`));
  };

  return (
    <div className={`app ${present ? 'present' : ''}`}>
      <header>
        <h1>Kite <span className="muted">{present ? 'what Kite saw and did on your walk' : 'caregiver view'}</span></h1>
        <span className={`conn ${conn.isActive ? 'ok' : 'bad'}`}>
          <i />{conn.isActive ? 'Live via SpacetimeDB' : conn.connectionError ? `Disconnected: ${conn.connectionError.message}` : 'Connecting...'}
        </span>
        <button className="present-toggle" onClick={() => setPresent(on => !on)} aria-pressed={present}
          title="Toggle presentation mode (P)">{present ? 'Exit presentation' : 'Present'} <kbd>P</kbd></button>
        {!present && <button onClick={onClear} disabled={!conn.isActive}>Clear run</button>}
      </header>

      <div className="layout">
        <div className="main-col">
          <WalkMap pose={pose} trail={trail} hazards={hazardMarks} beacon={beacon}
            freshness={lastPosRow ? ago(now, lastPosRow.t) : ''} pins={pins} focusId={focusId} onPick={setFocusId} />
          <Timeline moments={moments} walkStart={walk.length ? Math.min(happenedAt(walk[0]), moments[0]?.t ?? Infinity) : undefined} focusId={focusId} onPick={setFocusId} />
        </div>

        <aside key={present ? 'present' : 'normal'}>{/* remount: each mode starts scrolled to the top */}
          <section className="last-spoken">
            <h2>Last spoken</h2>
            {lastSpoken ? (
              <p className="spoken">
                “{str(lastSpoken.p.text)}”
                <span className="spoken-meta"><Badge who={speakerOf(lastSpoken, sorted)} /><span className="muted small">{lastSpoken.type} · {ago(now, lastSpoken.t)}</span></span>
              </p>
            ) : <p className="muted">Nothing yet.</p>}
          </section>

          <section className="walker">
            <h2>Walker</h2>
            {pos ? (
              <>
                <div className="big">{str(pos.place) || 'Unknown place'}</div>
                <div className="muted">
                  {summary('position', { ...pos, place: undefined })}
                  {lastPosRow && ` · ${ago(now, lastPosRow.t)}`}
                </div>
                {beacon && <div className="beacon-line">Next: <strong>{beacon.label}</strong> · {Math.round(beacon.distance)} m</div>}
              </>
            ) : <p className="muted">No position yet.</p>}
          </section>

          <section>
            <h2>Active hazards</h2>
            {hazardList.length === 0 ? <p className="muted">None.</p> : (
              <ul className="hazards">
                {hazardList.map(([key, e]) => (
                  <li key={key} style={{ borderLeftColor: PRIORITY_COLOR[priorityOf(e)] ?? PRIORITY_COLOR.hazard }}
                    className={`p-${priorityOf(e)}`}>
                    <strong>{str(e.p.kind) || 'hazard'}</strong> {summary('hazard', { ...e.p, kind: undefined })}
                    <div className="muted small">{priorityOf(e)} · {e.source} · {ago(now, e.t)}</div>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <LiveState perMin={live.perMin} capped={capped} sources={live.sources} limit={present ? PRESENT_SOURCES : undefined} statusOf={statusOf} compact={present} />

          {!present && (
            <>
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
                  <div key={e.id} className="muted small">Sent “{str(e.p.text)}” · {ago(now, e.t)}</div>
                ))}
              </section>

              <section>
                <h2>Live feed</h2>
                <table className="feed">
                  <tbody>
                    {sorted.slice(0, FEED_SIZE).map(e => (
                      <tr key={e.id} className={`p-${e.priority} ${e.source === 'caregiver' ? 'caregiver' : ''}`}>
                        <td className="muted">{new Date(e.t).toLocaleTimeString()}</td>
                        <td>{e.type}</td>
                        <td className="muted">{e.source}</td>
                        <td>{summary(e.type, e.p)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

export default App;
