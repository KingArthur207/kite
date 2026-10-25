// Pure helpers for the walk timeline and the shared-state panel. No React here, so check.ts can run them.

export type Payload = Record<string, unknown>;
export type LatLng = [number, number];
// One event-table row, parsed once. `t` is the server timestamp in ms.
export type Ev = { id: string; type: string; source: string; priority: string; t: number; p: Payload };

// Who acted: Kite's deterministic code or Claude (judgment, answers, scene text).
export type Who = 'kite' | 'claude';
export type Kind = 'stop' | 'hazard' | 'judgment' | 'answer' | 'scene' | 'guidance';
export type Moment = { id: string; t: number; kind: Kind; who: Who; text: string; detail: string; source: string; pos?: LatLng };

export const WHO_LABEL: Record<Who, string> = { kite: 'Kite', claude: 'Claude' };
export const KIND_LABEL: Record<Kind, string> = {
  stop: 'Stop', hazard: 'Hazard', judgment: 'Judgment', answer: 'Answer', scene: 'Scene', guidance: 'Guidance',
};

const WALK_GAP_MS = 45_000;     // a quiet gap this long starts a new walk
const REPEAT_MS = 15_000;       // the same hazard/scene/guidance again inside this window is one moment
const POS_MATCH_MS = 10_000;    // a moment gets the walker's position only if one is this close in time
const MAX_MOMENTS = 200;        // the event table keeps 500 rows; this bounds the strip and the map pins
export const MAX_TEXT = 280;    // longest payload text shown anywhere; the module allows 16 KB payloads

export const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
/** A payload field as text: strings, finite numbers and booleans only, so an object never shows as
 *  "[object Object]". React renders it as text, never as HTML. */
export const str = (v: unknown): string =>
  typeof v === 'string' ? v : (typeof v === 'number' && Number.isFinite(v)) || typeof v === 'boolean' ? String(v) : '';
/** str(), on one line, clipped to `max` characters. */
export function clip(v: unknown, max = MAX_TEXT): string {
  const s = str(v).replace(/\s+/g, ' ').trim();
  return s.length > max ? `${s.slice(0, max - 1).trimEnd()}…` : s;
}
export const norm = (s: string) => s.replace(/\s+/g, ' ').trim().toLowerCase();

/** A row's payload: always a plain object, whatever the row holds (bad JSON, null, an array, a string). */
export function parse(json: string): Payload {
  try {
    const v: unknown = JSON.parse(json);
    return v && typeof v === 'object' && !Array.isArray(v) ? (v as Payload) : {};
  } catch {
    return {};
  }
}

export const latLngOf = (p: Payload): LatLng | undefined => {
  const lat = num(p.lat), lng = num(p.lng);
  return lat != null && lng != null ? [lat, lng] : undefined;
};

/** Events of the most recent walk: everything after the last quiet gap. `asc` is oldest first. */
export function lastWalk(asc: readonly Ev[]): Ev[] {
  let start = 0;
  for (let i = 1; i < asc.length; i++) if (asc[i].t - asc[i - 1].t > WALK_GAP_MS) start = i;
  return asc.slice(start);
}

const SOURCE_CLOCK_MS = 5 * 60_000;
/** When it happened: the producer's epoch `t` (seconds) if it has a sane one, else the server time.
 *  A busy bridge can deliver a burst late, which would squash the timeline onto one second. */
export const happenedAt = (e: Ev): number => {
  const t = num(e.p.t);
  return t != null && t > 1e9 && Math.abs(t * 1000 - e.t) < SOURCE_CLOCK_MS ? t * 1000 : e.t;
};

export const isStopText = (s: string) => /^stop\b/i.test(s.trim());

function hazardText(p: Payload): string {
  const kind = str(p.kind) || str(p.key).split(':').pop() || 'hazard';
  const state = str(p.state) === 'dont_walk' ? "don't walk" : str(p.state).replace(/_/g, ' ');
  return clip(p.text) || clip([kind.replace(/_/g, ' '), state].filter(Boolean).join(' · '), 80);
}

/** One event -> a timeline moment, or undefined if it is not a key moment. Dedupe happens in moments(). */
function toMoment(e: Ev): Omit<Moment, 'pos'> | undefined {
  const { p } = e;
  const text = clip(p.text);
  const base = { id: e.id, t: happenedAt(e), source: e.source };
  switch (e.type) {
    case 'say':
      if (e.source === 'judgment') {
        const up = p.upgrade as Payload | null | undefined;
        const upText = up && typeof up === 'object' ? clip(`raised ${str(up.key)} to ${str(up.priority)}`, 80) : '';
        const lat = num(p.latency_ms);
        return text || upText ? {
          ...base, kind: 'judgment', who: 'claude', text: text || upText,
          detail: [text && upText, lat != null && `${Math.round(lat)} ms`].filter(Boolean).join(' · '),
        } : undefined;
      }
      return text && isStopText(text) ? { ...base, kind: 'stop', who: 'kite', text, detail: e.source } : undefined;
    case 'spoken': {
      // The earpiece's record of what it actually said. Stops come from sonar; `answer` lines are Claude
      // answering the walker. Everything else it speaks already shows up as its source event.
      if (!text) return undefined;
      if (e.priority === 'critical' || str(p.priority) === 'critical' || isStopText(text))
        return { ...base, kind: 'stop', who: 'kite', text, detail: 'spoken by Kite, no Claude' };
      if (str(p.priority) === 'answer') return { ...base, kind: 'answer', who: 'claude', text, detail: 'answer to the walker' };
      return undefined;
    }
    case 'hazard': {
      const pri = str(p.priority) || e.priority;
      const dontWalk = /dont_walk|don't walk|wait|red/i.test(str(p.state));
      if (p.upgraded_by === 'judgment') return { ...base, kind: 'judgment', who: 'claude', text: hazardText(p), detail: `raised to ${pri}` };
      return { ...base, kind: pri === 'critical' ? 'stop' : 'hazard', who: 'kite', text: hazardText(p), detail: dontWalk ? "don't walk" : pri || 'hazard' };
    }
    case 'scene':
      // Only Claude's scene text (scene-chest / scene-drone); raw perception scenes stay in the feed.
      return e.source.startsWith('scene-') && text ? { ...base, kind: 'scene', who: 'claude', text, detail: `sees (${e.source.slice(6)})` } : undefined;
    case 'guidance':
      return text ? { ...base, kind: 'guidance', who: 'kite', text, detail: 'route' } : undefined;
    default:
      return undefined;
  }
}

/** Key moments of a walk, oldest first, each with the walker's position nearest in time. */
export function moments(walk: readonly Ev[], walkerSource: string | undefined): Moment[] {
  const positions = walk
    .filter(e => e.type === 'position' && (walkerSource == null || e.source === walkerSource))
    .flatMap(e => { const at = latLngOf(e.p); return at ? [{ t: happenedAt(e), at }] : []; });
  const posAt = (t: number): LatLng | undefined => {
    let best: { t: number; at: LatLng } | undefined;
    for (const q of positions) if (!best || Math.abs(q.t - t) < Math.abs(best.t - t)) best = q;
    return best && Math.abs(best.t - t) <= POS_MATCH_MS ? best.at : undefined;
  };
  const out: Moment[] = [];
  const lastSeen = new Map<string, { t: number; i: number; hadText: boolean }>();
  for (const e of [...walk].sort((a, b) => happenedAt(a) - happenedAt(b))) {
    const m = toMoment(e);
    if (!m) continue;
    // Hazards repeat every frame and the earpiece re-mirrors them with rendered text; collapse the same
    // hazard (kind + state) or the same line inside REPEAT_MS, and guidance by text for the whole walk.
    // A critical hazard and the "Stop." the earpiece speaks for it are one stop, so stops share a key.
    const key = m.kind === 'stop' ? 'stop' : e.type === 'hazard' && m.kind !== 'judgment'
      ? `hazard:${str(e.p.kind) || str(e.p.key).split(':').pop()}:${str(e.p.state)}`
      : `${m.kind}:${norm(m.text)}`;
    const prev = lastSeen.get(key);
    if (prev && (m.kind === 'guidance' || m.t - prev.t < REPEAT_MS)) {
      const hasText = !!str(e.p.text);
      lastSeen.set(key, { ...prev, t: m.t, hadText: prev.hadText || hasText });
      // Keep the first time, but prefer the earpiece's rendered words when the first copy had none.
      // For a stop, the words the walker actually heard win.
      if (m.kind === 'stop' && e.type === 'spoken') out[prev.i] = { ...out[prev.i], text: m.text, detail: m.detail };
      else if (hasText && !prev.hadText) out[prev.i] = { ...out[prev.i], text: m.text };
      continue;
    }
    lastSeen.set(key, { t: m.t, i: out.length, hadText: !!str(e.p.text) });
    out.push({ ...m, pos: posAt(m.t) });
  }
  return out.slice(-MAX_MOMENTS);
}

export type SourceStat = { source: string; ageS: number; perMin: number; buckets: number[] };

/** Shared-state numbers, derived only from the tables: rows per source over the last minute. */
export function liveStats(desc: readonly Ev[], latestBySource: ReadonlyMap<string, number>, now: number, buckets = 12) {
  const WINDOW_MS = 60_000;
  const step = WINDOW_MS / buckets;
  const per = new Map<string, number[]>();
  let total = 0;
  for (const e of desc) {
    const age = Math.max(0, now - e.t);  // server clock may run slightly ahead of this browser
    if (age > WINDOW_MS) break;          // newest first, so the rest are older
    total++;
    const b = per.get(e.source) ?? new Array(buckets).fill(0);
    b[buckets - 1 - Math.min(buckets - 1, Math.floor(age / step))]++;
    per.set(e.source, b);
  }
  const sources: SourceStat[] = [...latestBySource.entries()]
    .map(([source, t]) => {
      const b = per.get(source) ?? new Array(buckets).fill(0);
      return { source, ageS: Math.max(0, Math.round((now - t) / 1000)), perMin: b.reduce((a, c) => a + c, 0), buckets: b };
    })
    .sort((a, b) => b.perMin - a.perMin || a.ageS - b.ageS);
  return { perMin: total, sources };
}

const SAME_LINE_MS = 10_000;

/** Who a spoken line came from. The earpiece speaks Claude's words too, so a
 *  `spoken` row is attributed by matching it to the `say` it came from. `desc` is newest first. */
export function speakerOf(e: Ev, desc: readonly Ev[]): Who {
  if (e.source === 'judgment' || e.source.startsWith('scene-')) return 'claude';
  if (e.type !== 'spoken') return 'kite';
  const text = norm(str(e.p.text));
  if (str(e.p.priority) === 'answer') return 'claude';
  const fromClaude = desc.some(s => s.type === 'say' && s.source === 'judgment' &&
    Math.abs(s.t - e.t) <= SAME_LINE_MS && norm(str(s.p.text)) === text);
  return fromClaude ? 'claude' : 'kite';
}
