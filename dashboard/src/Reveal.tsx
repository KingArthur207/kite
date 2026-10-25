import { useEffect, useRef } from 'react';
import { KIND_LABEL, WHO_LABEL, type Moment, type SourceStat, type Who } from './timeline';
import { FRESH_LABEL, STALE_S, ago, freshOf } from './view';

// The pieces judges look at during the reveal: the walk timeline, who-acted badges, the latest stop
// and the shared-state panel. Colours live in App.css (--kite, --claude, --caregiver, --stop).

export const isLive = (s: SourceStat) => s.ageS < STALE_S;
const reducedMotion = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

export function Badge({ who }: { who: Who }) {
  return <span className={`badge ${who}`}>{WHO_LABEL[who]}</span>;
}

const WHO_ROLE: Record<Who, string> = {
  kite: 'safety and route code, no AI',
  claude: 'judgment, answers, scene',
  caregiver: 'messages from this screen',
};

/** Who acted on this walk, with counts: the legend and the tally in one line. */
export function WhoActed({ moments }: { moments: Moment[] }) {
  const count = (who: Who) => moments.filter(m => m.who === who).length;
  return (
    <ul className="who-acted" aria-label="Who acted on this walk">
      {(['kite', 'claude', 'caregiver'] as Who[]).map(who => (
        <li key={who}><Badge who={who} /> <b>{count(who)}</b> <span className="muted">{WHO_ROLE[who]}</span></li>
      ))}
    </ul>
  );
}

const clock = (t: number) => new Date(t).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' });
export const offset = (t: number, t0: number) => {
  const s = Math.max(0, Math.round((t - t0) / 1000));
  return `+${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** The newest stop, big and red: the moment the reveal is about. */
export function StopCard({ stop, walkStart, now, onPick, focused }: {
  stop?: Moment; walkStart?: number; now: number; onPick: (id: string) => void; focused: boolean;
}) {
  if (!stop) return null;
  return (
    <section className="stop-card" aria-labelledby="stop-h">
      <h2 id="stop-h">Latest stop</h2>
      <p className="stop-line">“{stop.text}”</p>
      <p className="stop-meta">
        <Badge who={stop.who} />
        <span>{stop.who === 'kite' ? 'Kite code stopped the walker, no AI in the loop' : stop.detail}</span>
      </p>
      <p className="muted small">
        {walkStart != null && <>{offset(stop.t, walkStart)} into the walk · </>}{clock(stop.t)} · {ago((now - stop.t) / 1000)}
        {stop.pos && <> · <button type="button" className="link" aria-pressed={focused} onClick={() => onPick(stop.id)}>Show on map</button></>}
      </p>
    </section>
  );
}

export function Timeline({ moments, walkStart, walkEnd, focusId, onPick }: {
  moments: Moment[]; walkStart?: number; walkEnd?: number; focusId?: string; onPick: (id: string | undefined) => void;
}) {
  const strip = useRef<HTMLOListElement>(null);
  const lastId = moments[moments.length - 1]?.id;
  // Keep the newest moment in view while the walk is live, unless one is selected.
  useEffect(() => {
    const el = strip.current;
    if (el && !focusId) el.scrollTo({ left: el.scrollWidth, behavior: reducedMotion() ? 'auto' : 'smooth' });
  }, [lastId, focusId]);
  // A moment picked on the map or the ribbon scrolls its card into view.
  useEffect(() => {
    if (!focusId) return;
    strip.current?.querySelector(`[data-id="${CSS.escape(focusId)}"]`)
      ?.scrollIntoView({ inline: 'center', block: 'nearest', behavior: reducedMotion() ? 'auto' : 'smooth' });
  }, [focusId]);

  const stops = moments.filter(m => m.kind === 'stop').length;
  const t0 = walkStart ?? moments[0]?.t ?? 0;
  const t1 = Math.max(walkEnd ?? 0, moments[moments.length - 1]?.t ?? 0, t0 + 1000);
  const frac = (t: number) => Math.min(100, Math.max(0, ((t - t0) / (t1 - t0)) * 100));
  // Label stops on the ribbon, skipping a label that would overlap the previous one.
  const labelled = new Set<string>();
  let lastLabel = -Infinity;
  for (const m of moments) {
    if (m.kind === 'stop' && frac(m.t) - lastLabel >= 7) { labelled.add(m.id); lastLabel = frac(m.t); }
  }

  return (
    <section className="timeline" aria-labelledby="timeline-h">
      <div className="timeline-head">
        <h2 id="timeline-h">Walk timeline</h2>
        {moments.length > 0 && (
          <span className="tally">
            <b className={stops ? 'stop-text' : ''}>{plural(stops, 'stop')}</b>
            {' · '}{plural(moments.length, 'key moment')}{walkStart != null && <> over {offset(t1, t0).slice(1)}</>}
          </span>
        )}
        {moments.length > 0 && <WhoActed moments={moments} />}
      </div>
      {moments.length === 0 ? (
        <p className="muted timeline-empty">No key moments yet. Stops, hazards, Claude's calls and dashboard messages show up here as the walk happens.</p>
      ) : (
        <>
          {/* The whole walk at a glance; the cards below are the keyboard and screen reader route. */}
          <div className="ribbon" aria-hidden="true">
            <div className="ribbon-track" />
            {moments.map(m => (
              <button key={m.id} type="button" tabIndex={-1} title={`${KIND_LABEL[m.kind]}: ${m.text}`}
                className={`tick k-${m.kind} w-${m.who} ${m.id === focusId ? 'on' : ''}`} style={{ left: `${frac(m.t)}%` }}
                onClick={() => onPick(m.id === focusId ? undefined : m.id)}>
                {labelled.has(m.id) && <span className="tick-label">Stop</span>}
              </button>
            ))}
            <span className="ribbon-end start">+0:00</span>
            <span className="ribbon-end end">{offset(t1, t0)}</span>
          </div>
          <ol className="strip" ref={strip}>
            {moments.map(m => (
              <li key={m.id} data-id={m.id}>
                <button type="button" className={`moment k-${m.kind} w-${m.who} ${m.id === focusId ? 'on' : ''}`}
                  aria-pressed={m.id === focusId} title={m.pos ? 'Show on the map' : 'No position recorded at this time'}
                  onClick={() => onPick(m.id === focusId ? undefined : m.id)}>
                  <span className="moment-top">
                    <span className="when">{walkStart != null ? offset(m.t, walkStart) : ''} <span className="muted">{clock(m.t)}</span></span>
                    <Badge who={m.who} />
                  </span>
                  <span className="moment-kind">{KIND_LABEL[m.kind]}{m.pos ? '' : ' · off map'}</span>
                  <span className="moment-text">{m.text}</span>
                  {m.detail && <span className="moment-detail muted">{m.detail}</span>}
                </button>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}

function Sparkline({ buckets, cls }: { buckets: number[]; cls: string }) {
  const W = 72, H = 18;
  const max = Math.max(1, ...buckets);
  const pts = buckets.map((v, i) => `${(i / (buckets.length - 1)) * W},${H - 1 - (v / max) * (H - 2)}`).join(' ');
  return (
    <svg className={`spark ${cls}`} width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <polyline points={pts} fill="none" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

export function LiveState({ perMin, capped, sources, statusOf, compact, limit }: {
  perMin: number; capped: boolean; sources: SourceStat[]; statusOf: (source: string) => string; compact: boolean; limit?: number;
}) {
  const live = sources.filter(isLive).length;
  const shown = sources.slice(0, limit);
  return (
    <section className="live" aria-labelledby="live-h">
      <h2 id="live-h">Devices <span className="via">shared live state on SpacetimeDB</span></h2>
      <div className="live-nums">
        <div><b>{live}</b><span className="muted">{live === 1 ? 'source' : 'sources'} live{sources.length > live ? ` of ${sources.length}` : ''}</span></div>
        <div><b>{capped ? '≥' : ''}{perMin}</b><span className="muted">events / min</span></div>
      </div>
      {sources.length === 0 ? <p className="muted">Nothing connected yet. Devices appear here as soon as they write.</p> : (
        <table className="devices">
          <caption className="sr-only">Each device, how often it writes and how long since it last wrote</caption>
          <thead className="sr-only"><tr><th scope="col">Device</th><th scope="col">Activity</th><th scope="col">Rate</th><th scope="col">Last write</th></tr></thead>
          <tbody>
            {shown.map(s => {
              const f = freshOf(s.ageS);
              const status = !compact && statusOf(s.source);
              return (
                <tr key={s.source} className={f}>
                  <th scope="row"><span className={`dot ${f}`} aria-hidden="true" /><strong className="src">{s.source}</strong>
                    {status && <div className="muted small">{status}</div>}</th>
                  <td><Sparkline buckets={s.buckets} cls={f} /></td>
                  <td className="rate">{s.perMin}<span className="muted">/min</span></td>
                  <td className={`age ${f}`}><span className="sr-only">{FRESH_LABEL[f]}, </span>{s.ageS < 600 ? `${s.ageS}s` : ago(s.ageS)}
                    {f === 'dead' && <span className="age-word" aria-hidden="true">silent</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {limit != null && sources.length > shown.length && <p className="muted small">and {sources.length - shown.length} more</p>}
      <p className="muted small honest">Counted from the <code>event</code> and <code>latest</code> tables every device writes to{capped ? ' (the event table keeps the newest 500)' : ''}.</p>
    </section>
  );
}
