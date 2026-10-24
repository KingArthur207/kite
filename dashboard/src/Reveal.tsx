import { useEffect, useRef } from 'react';
import { KIND_LABEL, WHO_LABEL, type Moment, type SourceStat, type Who } from './timeline';

// The pieces judges look at during the reveal: the walk timeline, who-said-what badges and the
// shared-state panel. Colours live in App.css (--kite, --claude, --caregiver, --stop).

const FRESH_S = 5;   // green under this, amber until STALE_S, red after
const STALE_S = 15;
export const freshClass = (s: number) => (s < FRESH_S ? 'fresh' : s < STALE_S ? 'stale' : 'dead');
export const isLive = (s: SourceStat) => s.ageS < STALE_S;

export function Badge({ who }: { who: Who }) {
  return <span className={`badge ${who}`}>{WHO_LABEL[who]}</span>;
}

export function Legend() {
  return (
    <div className="legend" aria-label="Who said what">
      <span><Badge who="kite" /> deterministic safety and route, no AI</span>
      <span><Badge who="claude" /> judgment, answers, scene</span>
      <span><Badge who="caregiver" /> messages from this screen</span>
    </div>
  );
}

const clock = (t: number) => new Date(t).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' });
const offset = (t: number, t0: number) => {
  const s = Math.max(0, Math.round((t - t0) / 1000));
  return `+${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

export function Timeline({ moments, walkStart, focusId, onPick }: {
  moments: Moment[]; walkStart?: number; focusId?: string; onPick: (id: string | undefined) => void;
}) {
  const strip = useRef<HTMLOListElement>(null);
  const lastId = moments[moments.length - 1]?.id;
  // Keep the newest moment in view while the walk is live, unless one is selected.
  useEffect(() => {
    const el = strip.current;
    if (el && !focusId) el.scrollTo({ left: el.scrollWidth, behavior: 'smooth' });
  }, [lastId, focusId]);

  const count = (f: (m: Moment) => boolean) => moments.filter(f).length;
  const stops = count(m => m.kind === 'stop');
  const claude = count(m => m.who === 'claude');
  const caregiver = count(m => m.who === 'caregiver');
  const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

  return (
    <section className="timeline" aria-label="Walk timeline">
      <div className="timeline-head">
        <h2>Walk timeline</h2>
        {moments.length > 0 && (
          <span className="tally">
            <b className="stop-text">{plural(stops, 'stop')}</b> · {plural(claude, 'Claude call')} · {plural(caregiver, 'caregiver message')}
          </span>
        )}
        <Legend />
      </div>
      {moments.length === 0 ? (
        <p className="muted timeline-empty">No key moments yet. Stops, hazards, Claude's calls and caregiver messages show up here as the walk happens.</p>
      ) : (
        <ol className="strip" ref={strip}>
          {moments.map(m => (
            <li key={m.id}>
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
  return (
    <section className="live">
      <h2>Shared live state <span className="via">SpacetimeDB</span></h2>
      <div className="live-nums">
        <div><b>{live}</b><span className="muted">{live === 1 ? 'source' : 'sources'} live{sources.length > live ? ` of ${sources.length}` : ''}</span></div>
        <div><b>{capped ? '≥' : ''}{perMin}</b><span className="muted">events / min</span></div>
      </div>
      {sources.length === 0 ? <p className="muted">Nothing connected yet.</p> : (
        <table className="devices">
          <tbody>
            {sources.slice(0, limit).map(s => (
              <tr key={s.source}>
                <td><span className={`dot ${freshClass(s.ageS)}`} /><strong>{s.source}</strong>
                  {!compact && statusOf(s.source) && <div className="muted small">{statusOf(s.source)}</div>}</td>
                <td><Sparkline buckets={s.buckets} cls={freshClass(s.ageS)} /></td>
                <td className="rate">{s.perMin}<span className="muted">/min</span></td>
                <td className={`age ${freshClass(s.ageS)}`}>{s.ageS}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="muted small honest">Counted from the <code>event</code> and <code>latest</code> tables every device writes to{capped ? ' (the event table keeps the newest 500)' : ''}.</p>
    </section>
  );
}
