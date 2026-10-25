import { useEffect, useMemo, useRef, useState } from 'react';
import { tables, reducers } from './module_bindings';
import { useSpacetimeDB, useTable, useReducer } from 'spacetimedb/react';
import { WalkMap, PRIORITY_COLOR, type Pin, type Pose, type Relative } from './WalkMap';
import { accuracyText, clientsOf, speedText, type Client } from './clients';
import { fixtureOn, twoPhones } from './fixture';
import {
  clip, happenedAt, isStopText, lastWalk, liveStats, moments as walkMoments, num, parse, speakerOf, str,
  type Ev, type Moment, type Payload, type Who,
} from './timeline';
import { Badge, LiveState, StopCard, Timeline } from './Reveal';
import { FeedSettings, VideoPane, useFeeds } from './Feeds';
import {
  MAX_MESSAGE, QUIET_S, UNHEARD_AFTER_S, ago, ageS, blockedWord, deliveryOf, direction, facing, humanize,
  linkOf, meters, relative, type Link,
} from './view';
import './App.css';

// Default (analytics) view: what Kite knows (phones on the map, hazards, spoken lines, rates) beside the walker's
// and the drone's live camera feeds. `?view=walk` opens the walk view below.
// Caregiver dashboard: live view of the shared Kite world state, plus short messages to the walker
// (a `say` from source `caregiver`; the earpiece polls for them and speaks them). No camera video (privacy).
// Press P for presentation mode (the reveal): no raw feed or controls, bigger map, timeline and last line.
// Every payload is untrusted: it is parsed defensively and rendered as text by React, never as HTML.

// `event` rows have an id, `latest` rows a "type:source" key.
type Row = { id?: bigint; key?: string; type: string; source: string; priority: string; payload: string; t: { toMillis(): bigint } };

const FEED_SIZE = 40;
const DEFAULT_HAZARD_TTL_S = 20;
const MAX_HAZARD_TTL_S = 120;
const MAX_HAZARDS = 8;
const BEACON_TTL_S = 30;
const HELP_FRESH_S = 10 * 60;  // a help request stays on screen this long unless acknowledged
const EVENT_TABLE_CAP = 500;   // the module keeps the newest 500 events
// The earpiece drops any message that could be heard as permission to move ("you can", "go", "clear"...),
// so the quick ones avoid those words (view.check.ts checks them).
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
const PIN_COLOR: Record<string, string> = { stop: '#FF453A', help: '#FF453A', kite: '#63E6E2', claude: '#FF8A5B', caregiver: '#64D2FF' };
// `?view=walk` (and the older `?view=caregiver`) opens the walk view: map, walk timeline, messages and shared state.
const CAREGIVER = ['walk', 'caregiver'].includes(new URLSearchParams(location.search).get('view') ?? '');
const FIXTURE = fixtureOn();   // ?fixture=two-phones in dev: fake phones rendered locally, never written to SpacetimeDB

const toEv = (r: Row): Ev => ({ id: r.id?.toString() ?? r.key ?? '', type: r.type, source: r.source, priority: r.priority, t: Number(r.t.toMillis()), p: parse(r.payload) });
const newestFirst = (rows: readonly Ev[]) => [...rows].sort((a, b) => b.t - a.t);
const clock = (t: number) => new Date(t).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' });
// Hazards from sample data may carry no priority column; the payload or the type default decides.
const priorityOf = (e: Ev) => str(e.p.priority) || e.priority || 'hazard';
const kindOf = (p: Payload) => humanize(clip(p.kind, 40)) || 'hazard';
const whereOf = (p: Payload) => [meters(p.distance), direction(p.bearing)].filter(Boolean).join(' ');

function summary(type: string, p: Payload): string {
  switch (type) {
    case 'hazard':
      return [kindOf(p), humanize(clip(p.state, 30)), clip(p.text, 120), whereOf(p)].filter(Boolean).join(' · ');
    case 'position':
      return [clip(p.place, 60), num(p.lat) != null && num(p.lng) != null && `${num(p.lat)!.toFixed(5)}, ${num(p.lng)!.toFixed(5)}`,
        facing(p.heading)].filter(Boolean).join(' · ');
    case 'status':
      return [clip(p.state, 30), num(p.battery) != null && `battery ${Math.round(num(p.battery)!)}%`, clip(p.text, 60)].filter(Boolean).join(' · ');
    case 'beacon':
      return [clip(p.label, 60), whereOf(p)].filter(Boolean).join(' · ');
    case 'obstacles':
      return `${Array.isArray(p.objects) ? p.objects.length : 0} objects`;
    case 'caregiver_alert':
      return `Help request: ${clip(p.message, 160) || 'no message'}`;
    default:
      return clip(p.text ?? p.action ?? p.kind ?? '', 160);
  }
}


const typing = (el: EventTarget | null) => el instanceof HTMLElement && (el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName));

const LINK_TEXT: Record<Link, string> = { live: 'Live', connecting: 'Connecting…', reconnecting: 'Reconnecting…', offline: 'Offline' };
const SAID_BY: Record<Who, string> = { kite: 'Kite said', claude: 'Claude said', caregiver: 'The walker heard your message' };

// Local copies of messages until their row shows up in the event table.
type Outgoing = { key: string; text: string; at: number; state: 'sending' | 'failed'; error?: string };
type Note = { kind: 'ok' | 'error'; text: string };

/** Screen reader announcements: the same text twice still gets read (a trailing space toggles). */
function useAnnouncer() {
  const [text, setText] = useState('');
  return [text, (s: string) => setText(prev => (prev === s ? `${s} ` : s))] as const;
}

function App() {
  const conn = useSpacetimeDB();
  const [events, eventsReady] = useTable(tables.event);
  const [latest, latestReady] = useTable(tables.latest);
  const ready = eventsReady && latestReady;   // false again while reconnecting, though the cached rows stay
  const clear = useReducer(reducers.clear);
  const push = useReducer(reducers.push);
  const [draft, setDraft] = useState('');
  const [outbox, setOutbox] = useState<Outgoing[]>([]);
  const [note, setNote] = useState<Note>();
  const [now, setNow] = useState(Date.now());
  const [present, setPresent] = useState(() => location.hash === '#present');
  const [focusId, setFocusId] = useState<string>();
  const [follow, setFollow] = useState<string>();
  const [ackHelp, setAckHelp] = useState<string>();
  const [everLive, setEverLive] = useState(false);
  const [polite, announce] = useAnnouncer();
  const [urgent, announceUrgent] = useAnnouncer();
  const [feeds, saveFeeds] = useFeeds();

  useEffect(() => { if (conn.isActive) setEverLive(true); }, [conn.isActive]);
  const link = linkOf(conn.isActive, everLive, conn.connectionError);
  const live = link === 'live';

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return;
      if (e.key.toLowerCase() === 'p') setPresent(on => !on);
      else if (e.key === 'Escape') setPresent(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  useEffect(() => {
    history.replaceState(null, '', present ? '#present' : location.pathname + location.search);
    document.title = !CAREGIVER ? 'Kite live' : present ? 'Kite: the walk' : 'Kite caregiver dashboard';
    // The analytics view presents full-screen; Esc leaves full screen, which leaves presentation mode too.
    if (CAREGIVER) return;
    if (present && !document.fullscreenElement) document.documentElement.requestFullscreen?.().catch(() => {});
    else if (!present && document.fullscreenElement) document.exitFullscreen().catch(() => {});
  }, [present]);
  useEffect(() => {
    if (CAREGIVER) return;
    const onFs = () => { if (!document.fullscreenElement) setPresent(false); };
    document.addEventListener('fullscreenchange', onFs);
    return () => document.removeEventListener('fullscreenchange', onFs);
  }, []);

  // Parse every payload once per table change, not once per render (the clock re-renders every second).
  const fixtureTick = FIXTURE ? Math.floor(now / 1000) : 0;
  const sorted = useMemo(() => newestFirst([...events.map(toEv), ...(FIXTURE ? twoPhones(now) : [])]), [events, fixtureTick]);
  const latestSorted = useMemo(() => newestFirst(latest.map(toEv)), [latest]);
  const haveData = ready || sorted.length > 0 || latestSorted.length > 0;

  // Every client with a usable position, from both tables, so one bad payload in `latest` doesn't hide
  // anyone. The newest one is the walker that hazards, the beacon and the timeline are placed around.
  const clients = useMemo(() => clientsOf(sorted, latestSorted, now), [sorted, latestSorted, now]);
  const primary: Client | undefined = clients.reduce<Client | undefined>((a, c) => (!a || c.fix.t > a.fix.t ? c : a), undefined);
  const pose: Pose | undefined = primary && { lat: primary.fix.lat, lng: primary.fix.lng, heading: primary.fix.heading };
  const walkerSource = primary?.source;

  // The last walk's key moments, oldest first.
  const walk = useMemo(() => lastWalk([...sorted].reverse()), [sorted]);
  const moments: Moment[] = useMemo(() => walkMoments(walk, walkerSource), [walk, walkerSource]);
  const walkStart = walk.length ? Math.min(happenedAt(walk[0]), moments[0]?.t ?? Infinity) : undefined;
  const walkEnd = walk.length ? happenedAt(walk[walk.length - 1]) : undefined;
  const pins: Pin[] = useMemo(() => moments.flatMap(mo => mo.pos ? [{
    id: mo.id, at: mo.pos, stop: mo.kind === 'stop' || mo.kind === 'help', label: mo.text,
    color: PIN_COLOR[mo.kind === 'stop' || mo.kind === 'help' ? mo.kind : mo.who],
  }] : []), [moments]);
  const lastStop = useMemo(() => [...moments].reverse().find(m => m.kind === 'stop'), [moments]);
  const lastHelp = useMemo(() => [...moments].reverse().find(m => m.kind === 'help'), [moments]);
  const helpOpen = lastHelp && lastHelp.id !== ackHelp && ageS(now, lastHelp.t) < HELP_FRESH_S ? lastHelp : undefined;

  // Active hazards: newest per (source, kind) still inside its ttl.
  const hazards = new Map<string, Ev>();
  for (const e of sorted) {
    if (e.type !== 'hazard') continue;
    const key = `${e.source}:${kindOf(e.p)}`;
    if (hazards.has(key)) continue;
    const ttl = Math.min(MAX_HAZARD_TTL_S, Math.max(0, num(e.p.ttl) ?? DEFAULT_HAZARD_TTL_S));
    if (now - e.t <= ttl * 1000) hazards.set(key, e);
    if (hazards.size >= MAX_HAZARDS) break;
  }
  const hazardList = [...hazards.entries()];
  const hazardMarks: Relative[] = hazardList.flatMap(([key, e]) => {
    const rel = relative(e.p);
    return rel ? [{ key, ...rel, label: [kindOf(e.p), humanize(clip(e.p.state, 30))].filter(Boolean).join(' '), priority: priorityOf(e) }] : [];
  });

  const beaconRow = latestSorted.find(r => r.type === 'beacon' && now - r.t <= BEACON_TTL_S * 1000);
  const bp = beaconRow?.p;
  const beaconRel = bp && bp.active !== false ? relative(bp) : undefined;
  const beacon: Relative | undefined = beaconRel && { key: 'beacon', ...beaconRel, label: clip(bp?.label, 40) || 'Next waypoint', priority: 'nav' };

  // Shared state: every source that has written anything, with its newest row time and last-minute rate.
  const newestBySource = new Map<string, number>();
  for (const r of latestSorted) if (!newestBySource.has(r.source)) newestBySource.set(r.source, r.t);
  const stats = liveStats(sorted, newestBySource, now);
  // Every kept row is inside the last minute, so older ones were trimmed: the rate is a lower bound.
  const capped = sorted.length >= EVENT_TABLE_CAP && stats.perMin >= sorted.length;
  const statusOf = (source: string) => {
    const st = latestSorted.find(r => r.type === 'status' && r.source === source);
    return st ? summary('status', st.p) : '';
  };
  const newestT = latestSorted[0]?.t ?? sorted[0]?.t;
  const quietS = newestT != null ? ageS(now, newestT) : 0;
  const quiet = live && ready && newestT != null && quietS > QUIET_S;

  const lastSpoken = latestSorted.find(r => ['guidance', 'spoken', 'say'].includes(r.type) && clip(r.p.text));
  const speaker: Who | undefined = lastSpoken && speakerOf(lastSpoken, sorted);
  const spokenText = lastSpoken ? clip(lastSpoken.p.text) : '';

  // Announce what changes after the page has loaded (never the backlog): new spoken lines politely,
  // stops and help requests assertively.
  const seen = useRef<{ spoken?: string; stop?: string; help?: string }>();
  useEffect(() => {
    if (!ready) return;
    const cur = { spoken: lastSpoken && `${lastSpoken.id}@${lastSpoken.t}`, stop: lastStop?.id, help: lastHelp?.id };
    const was = seen.current;
    seen.current = cur;
    if (!was) return;
    if (cur.help && cur.help !== was.help && lastHelp) announceUrgent(`The walker asked for help: ${lastHelp.text}`);
    else if (cur.stop && cur.stop !== was.stop && lastStop) announceUrgent(`Stop. Kite said: ${lastStop.text}`);
    else if (cur.spoken && cur.spoken !== was.spoken && lastSpoken && speaker && !isStopText(spokenText) && lastSpoken.priority !== 'critical')
      announce(`${SAID_BY[speaker]}: ${spokenText.replace(/^message from your caregiver:\s*/i, '')}`);
  }, [ready, lastSpoken?.id, lastSpoken?.t, lastStop?.id, lastHelp?.id]);

  // Caregiver messages: the newest few from the table, plus local ones still on their way.
  const sentRows = sorted.filter(e => e.type === 'say' && e.source === 'caregiver').slice(0, SHOWN_MESSAGES);
  const arrivedKeys = useMemo(() => new Set(sorted.filter(e => e.type === 'say' && e.source === 'caregiver').map(e => str(e.p.key))), [sorted]);
  const pending = outbox.filter(o => !arrivedKeys.has(o.key));
  const sending = pending.some(o => o.state === 'sending');
  const draftText = draft.trim();
  const blocked = blockedWord(draftText);

  const send = (raw: string, fromDraft: boolean) => {
    const text = raw.trim().slice(0, MAX_MESSAGE);
    if (!text || sending || !live) return;
    const word = blockedWord(text);
    if (word) {
      setNote({ kind: 'error', text: `Not sent. Kite won't read a message with “${word}” aloud, because it could sound like permission to move.` });
      return;
    }
    const key = `caregiver-${Date.now()}`;
    setOutbox(o => [{ key, text, at: Date.now(), state: 'sending' as const }, ...o].slice(0, 10));
    setNote(undefined);
    signMessage(text)
      .then(sig => {
        const msg = { type: 'say', source: 'caregiver', priority: 'info', text, key, ...sig };
        return push({ type: 'say', source: 'caregiver', priority: 'info', payload: JSON.stringify(msg) });
      })
      .then(() => {
        if (fromDraft) setDraft('');
        setNote({ kind: 'ok', text: "Sent. Kite will read it aloud in the walker's earpiece." });
      })
      .catch((err: unknown) => {
        const name = err instanceof Error ? err.name : '';
        const error = name === 'UnknownCallResultError'
          ? 'The connection dropped before SpacetimeDB confirmed it. It may have arrived; check the list before sending again.'
          : `Not sent: ${err instanceof Error ? err.message : String(err)}`;
        setOutbox(o => o.map(x => (x.key === key ? { ...x, state: 'failed', error } : x)));
        setNote({ kind: 'error', text: error });
      });
  };

  const onClear = () => {
    if (confirm('Clear the event log and latest state for a fresh run?'))
      clear().then(() => { setOutbox([]); setFocusId(undefined); })
        .catch(err => setNote({ kind: 'error', text: `Clear failed: ${err instanceof Error ? err.message : String(err)}` }));
  };

  const waiting = !ready
    ? (link === 'live' ? 'Loading the walk from SpacetimeDB…' : link === 'offline'
      ? "Can't reach SpacetimeDB yet. The walk appears here as soon as it connects." : 'Connecting to SpacetimeDB…')
    : sorted.length === 0 && latestSorted.length === 0
      ? 'The database is empty. The map appears with the first position event that has lat/lng.'
      : 'No position with coordinates yet. The map appears as soon as a position event with lat/lng arrives.';

  const liveCount = clients.filter(c => c.fresh !== 'dead').length;
  const hero = !primary
    ? { word: 'Waiting', tone: 'idle', line: haveData ? 'No phone has connected yet.' : 'Loading…' }
    : primary.fresh === 'fresh'
      ? { word: primary.ahead ? 'Walking' : 'Live', tone: 'live',
        line: [liveCount === 1 ? 'One phone live' : `${liveCount} phones live`, primary.fix.place, ago(primary.ageS)].filter(Boolean).join(' · ') }
      : primary.fresh === 'stale'
        ? { word: 'Delayed', tone: 'warn', line: `Last position ${ago(primary.ageS)}` }
        : { word: 'Quiet', tone: 'idle', line: `Last position ${ago(primary.ageS)}` };

  return (
    <div className={`app ${present ? 'present' : ''}`}>
      <header>
        <img className="mark" src={`${import.meta.env.BASE_URL}kite-mark.svg`} alt="" width="34" height="34" />
        <h1>Kite <span className="subtitle">{!CAREGIVER ? 'live' : present ? 'The walk' : 'Caregiver'}</span></h1>
        <span className={`conn ${link}`} role="status">
          <i aria-hidden="true" />{LINK_TEXT[link]}<span className="conn-via"> · SpacetimeDB</span>
        </span>
        {FIXTURE && <span className="fixture-chip">Fixture data</span>}
        <span className="header-actions">
          {!CAREGIVER && !present && <FeedSettings feeds={feeds} onSave={saveFeeds} />}
          <button type="button" className="present-toggle" onClick={() => setPresent(on => !on)} aria-pressed={present}
            aria-keyshortcuts="P" title="Presentation mode (P, Esc to leave)">{present ? 'Exit' : 'Present'} <kbd aria-hidden="true">P</kbd></button>
          {!present && <button type="button" onClick={onClear} disabled={!live}>Clear run</button>}
        </span>
      </header>

      {link === 'reconnecting' || link === 'offline' ? (
        <div className="banner warn">
          {link === 'reconnecting'
            ? <>Connection to SpacetimeDB lost. {newestT != null ? <>Showing what arrived until {clock(newestT)}. </> : null}Reconnecting automatically.</>
            : <>Can't reach SpacetimeDB{conn.connectionError?.message ? ` (${clip(conn.connectionError.message, 120)})` : ''}. Retrying automatically.</>}
        </div>
      ) : quiet ? (
        <div className="banner quiet">No device has written for {ago(quietS).replace(' ago', '')}. The walk may be over, or the devices are offline.</div>
      ) : null}

      {CAREGIVER && helpOpen && (
        <div className="banner help">
          <strong>The walker asked for help</strong> <span>“{helpOpen.text}”</span>
          <span className="muted">{ago(ageS(now, helpOpen.t))}</span>
          <span className="banner-actions">
            {helpOpen.pos && <button type="button" onClick={() => setFocusId(helpOpen.id)}>Show on map</button>}
            <button type="button" onClick={() => setAckHelp(helpOpen.id)}>Got it</button>
          </span>
        </div>
      )}

      {!CAREGIVER ? (
        <main className="analytics">
          <WalkMap clients={clients} pose={pose} follow={follow} onFollow={setFollow} hazards={hazardMarks} beacon={beacon}
            pins={pins} focusId={focusId} onPick={setFocusId} initialView="all" waiting={waiting} />
          <VideoPane label="Walker camera" param="webcam" src={feeds.webcam} />
          <VideoPane label="Drone camera" param="drone" src={feeds.drone} />
          <div className="stats" role="group" aria-label="What Kite knows right now">
            <section className="stat">
              <h2>Phones live</h2>
              <p className={`stat-value ${liveCount ? 'accent' : ''}`}>{liveCount}</p>
              <p className="stat-sub">{clients.length - liveCount ? `${clients.length - liveCount} quiet` : hero.word}</p>
            </section>
            <section className="stat">
              <h2>Accuracy</h2>
              <p className="stat-value">{primary ? accuracyText(primary.fix) || '–' : '–'}</p>
              <p className="stat-sub">{primary ? primary.name : 'No fix yet'}</p>
            </section>
            <section className="stat">
              <h2>Speed</h2>
              <p className="stat-value">{primary?.fix.speed != null ? `${primary.fix.speed.toFixed(1)} m/s` : '–'}</p>
              <p className="stat-sub">{primary ? (speedText(primary.fix) ? 'Walking' : 'Standing') : '–'}</p>
            </section>
            <section className="stat">
              <h2>Events/min</h2>
              <p className="stat-value">{capped ? '≥' : ''}{stats.perMin}</p>
              <p className="stat-sub">{stats.sources.length} sources</p>
            </section>
            <section className="stat wide">
              <h2>Kite just said {speaker && <Badge who={speaker} />}</h2>
              {lastSpoken ? <p className={`spoken-now ${isStopText(spokenText) ? 'is-stop' : ''}`}>“{spokenText}” <span className="muted small">{ago(ageS(now, lastSpoken.t))}</span></p>
                : <p className="stat-sub">{haveData ? 'Nothing spoken yet.' : 'Loading…'}</p>}
            </section>
            <section className="stat wide">
              <h2>Active hazards</h2>
              <p className={`stat-value ${hazardList.length ? 'warn' : ''}`}>{hazardList.length}</p>
              <p className="stat-sub">{hazardList.length ? hazardList.slice(0, 3).map(([, e]) => [kindOf(e.p), whereOf(e.p)].filter(Boolean).join(' ')).join(' · ') : 'None right now'}</p>
            </section>
          </div>
        </main>
      ) : (
      <main className="layout">
        <div className="main-col">
          <WalkMap clients={clients} pose={pose} follow={follow} onFollow={setFollow} hazards={hazardMarks} beacon={beacon}
            pins={pins} focusId={focusId} onPick={setFocusId} initialView={present ? 'walk' : 'all'} waiting={waiting} />
          <Timeline moments={moments} walkStart={walkStart} walkEnd={walkEnd} focusId={focusId} onPick={setFocusId} />
        </div>

        <aside key={present ? 'present' : 'normal'} aria-label="Walk details" tabIndex={0}>{/* remount: each mode starts scrolled to the top */}
          <div className={`hero ${hero.tone}`}>
            <span className="ring" aria-hidden="true"><i /></span>
            <div>
              <p className="hero-word" role="status">{hero.word}</p>
              <p className="hero-line">{hero.line}</p>
            </div>
          </div>

          <section className="clients" aria-labelledby="clients-h">
            <h2 id="clients-h">{clients.length === 1 ? 'Phone' : 'Phones'}</h2>
            {clients.length === 0 ? <p className="muted">{haveData ? 'Each phone appears here when it sends its first position.' : 'Waiting for data…'}</p> : (
              <ul className="client-list">
                {clients.map(c => (
                  <li key={c.source}>
                    <button type="button" className={`client ${c.fresh}`} aria-pressed={follow === c.source}
                      title={follow === c.source ? 'Stop following' : 'Follow on the map'}
                      onClick={() => setFollow(f => (f === c.source ? undefined : c.source))}>
                      <span className="swatch" style={{ background: c.color }} aria-hidden="true" />
                      <span className="client-main">
                        <span className="client-name">{c.name}</span>
                        <span className="client-sub">{[accuracyText(c.fix), speedText(c.fix), c.fix.place].filter(Boolean).join(' · ') || 'Position only'}</span>
                      </span>
                      <span className={`client-age ${c.fresh}`}>{c.fresh === 'fresh' ? 'Live' : ago(c.ageS)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {beacon && <p className="beacon-line">Next: <strong>{beacon.label}</strong>, {meters(beacon.distance)} {direction(beacon.bearing)}</p>}
          </section>

          <section className="last-spoken" aria-labelledby="spoken-h">
            <h2 id="spoken-h">{present ? 'Kite just said' : 'Last spoken line'}</h2>
            {lastSpoken && speaker ? (
              <>
                <p className={`spoken ${isStopText(spokenText) ? 'is-stop' : ''}`}>“{spokenText}”</p>
                <p className="spoken-meta"><Badge who={speaker} /><span className="muted small">{ago(ageS(now, lastSpoken.t))}</span></p>
              </>
            ) : <p className="muted">{haveData ? 'Nothing spoken yet.' : 'Loading…'}</p>}
          </section>

          <StopCard stop={lastStop} walkStart={walkStart} now={now} onPick={setFocusId} focused={focusId === lastStop?.id} />

          <section aria-labelledby="hazards-h">
            <h2 id="hazards-h">Active hazards {hazardList.length > 0 && <span className="count-pill">{hazardList.length}</span>}</h2>
            {hazardList.length === 0 ? <p className="muted">{haveData ? 'None right now.' : 'Waiting for data…'}</p> : (
              <ul className="hazards">
                {hazardList.map(([key, e]) => {
                  const pri = priorityOf(e);
                  const where = whereOf(e.p);
                  return (
                    <li key={key} style={{ borderLeftColor: PRIORITY_COLOR[pri] ?? PRIORITY_COLOR.hazard }} className={`p-${pri}`}>
                      <strong>{kindOf(e.p)}</strong>{where && <span className="hz-where"> · {where}</span>}
                      {(clip(e.p.text, 140) || clip(e.p.state, 30)) && <div>{clip(e.p.text, 140) || humanize(clip(e.p.state, 30))}</div>}
                      <div className="muted small">{pri} · {e.source} · {ago(ageS(now, e.t))}</div>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

          {!present && (
            <section className="messages" aria-labelledby="msg-h">
              <h2 id="msg-h">Message the walker</h2>
              <p className="muted small" id="msg-hint">Kite reads it aloud in the walker's earpiece, after any safety cue. Words that could sound like permission to move (“go”, “cross”, “clear”, “you can”) are never spoken.</p>
              <form className="message" onSubmit={e => { e.preventDefault(); send(draft, true); }}>
                <label htmlFor="msg" className="sr-only">Message to the walker</label>
                <input id="msg" value={draft} onChange={e => setDraft(e.target.value)} maxLength={MAX_MESSAGE} autoComplete="off"
                  placeholder={live ? 'Type a short message' : 'Paused until reconnected'} disabled={!live}
                  aria-describedby="msg-hint msg-count msg-blocked" aria-invalid={blocked ? true : undefined} />
                <button type="submit" disabled={!live || sending || !draftText || !!blocked}>{sending ? 'Sending…' : 'Send'}</button>
              </form>
              <div className="msg-under">
                <span id="msg-blocked" className="blocked">{blocked && `Kite won't speak this: “${blocked}” could sound like permission to move.`}</span>
                <span id="msg-count" className={`count ${draft.length > MAX_MESSAGE - 20 ? 'near' : ''}`}>
                  {draft.length} / {MAX_MESSAGE}<span className="sr-only"> characters</span>
                </span>
              </div>
              <div className="quick" role="group" aria-label="Quick replies, sent right away">
                {QUICK_MESSAGES.map(q => (
                  <button type="button" key={q} onClick={() => send(q, false)} disabled={!live || sending}>{q}</button>
                ))}
              </div>
              <p className={`send-note ${note?.kind ?? ''}`} role="status">{note?.text}</p>
              {(pending.length > 0 || sentRows.length > 0) && (
                <ul className="sent" aria-label="Recent messages">
                  {pending.map(o => (
                    <li key={o.key} className={o.state}>
                      <span className="sent-text">“{clip(o.text)}”</span>
                      <span className="sent-state">{o.state === 'sending' ? 'Sending…' : o.error}</span>
                    </li>
                  ))}
                  {sentRows.map(e => {
                    const d = deliveryOf(e, sorted, now);
                    return (
                      <li key={e.id} className={d.state}>
                        <span className="sent-text">“{clip(e.p.text)}”</span>
                        <span className="sent-state">
                          {d.state === 'heard' ? <>Heard by the walker at {clock(d.heardAt!)}</>
                            : d.state === 'unheard' ? <>Sent {ago(ageS(now, e.t))}, not spoken yet after {UNHEARD_AFTER_S} s. Is the earpiece running with <code>--spacetime</code>?</>
                              : <>Sent {ago(ageS(now, e.t))}, waiting for the earpiece</>}
                        </span>
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>
          )}

          <LiveState perMin={stats.perMin} capped={capped} sources={stats.sources} limit={present ? PRESENT_SOURCES : undefined} statusOf={statusOf} compact={present} />

          {!present && (
            <details className="feed-card">
              <summary>Live feed <span className="muted small">newest first</span></summary>
              {sorted.length === 0 ? <p className="muted">{haveData ? 'No events yet.' : 'Loading…'}</p> : (
                <table className="feed">
                  <caption className="sr-only">Newest events first</caption>
                  <thead className="sr-only"><tr><th scope="col">Time</th><th scope="col">Type</th><th scope="col">Source</th><th scope="col">Details</th></tr></thead>
                  <tbody>
                    {sorted.slice(0, FEED_SIZE).map(e => (
                      <tr key={e.id} className={`p-${e.priority} ${e.source === 'caregiver' ? 'caregiver' : ''}`}>
                        <td className="muted">{new Date(e.t).toLocaleTimeString()}</td>
                        <td>{e.type}</td>
                        <td className="muted src">{e.source}</td>
                        <td className="detail">{summary(e.type, e.p)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </details>
          )}
        </aside>
      </main>
      )}

      <div className="sr-only" aria-live="polite" aria-atomic="true">{polite}</div>
      <div className="sr-only" aria-live="assertive" aria-atomic="true">{urgent}</div>
    </div>
  );
}

export default App;
