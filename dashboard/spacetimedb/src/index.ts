import { schema, table, t, SenderError } from 'spacetimedb/server';

// Every Kite event (earpiece wire format, docs/internal/spec.md section 5.1) lands here.
// payload is the full event as JSON: its shape varies per type and is validated by
// earpiece/earpiece/events.py before anything acts on it.
const event = table(
  { name: 'event', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    type: t.string(),
    source: t.string(),
    priority: t.string(),
    payload: t.string(),
    t: t.timestamp(),
  }
);

// Newest event per (type, source), so a late joiner sees the whole world at once.
const latest = table(
  { name: 'latest', public: true },
  {
    key: t.string().primaryKey(),
    type: t.string(),
    source: t.string(),
    priority: t.string(),
    payload: t.string(),
    t: t.timestamp(),
  }
);

const spacetimedb = schema({ event, latest });
export default spacetimedb;

const KEEP_EVENTS = 500n;
const MAX_PAYLOAD = 16 * 1024;
const TYPES = new Set([
  'obstacles', 'hazard', 'guidance', 'beacon', 'position', 'status', 'scene', 'say',
  'heartbeat', 'control', 'memory', 'spoken', 'heard', 'voice_request', 'caregiver_alert',
  'gesture', 'output',
]);
const PRIORITIES = new Set(['critical', 'hazard', 'nav', 'answer', 'info', 'system', '']);
const IDENT = /^[\w.:-]{1,64}$/;

export const push = spacetimedb.reducer(
  { type: t.string(), source: t.string(), priority: t.string(), payload: t.string() },
  (ctx, { type, source, priority, payload }) => {
    if (!TYPES.has(type)) throw new SenderError(`unknown type ${type}`);
    if (!IDENT.test(source)) throw new SenderError('bad source');
    if (!PRIORITIES.has(priority)) throw new SenderError('bad priority');
    if (payload.length > MAX_PAYLOAD) throw new SenderError('payload too large');
    try {
      JSON.parse(payload);
    } catch {
      throw new SenderError('payload is not JSON');
    }

    const row = ctx.db.event.insert({ id: 0n, type, source, priority, payload, t: ctx.timestamp });
    if (row.id > KEEP_EVENTS) ctx.db.event.id.delete(row.id - KEEP_EVENTS);

    const key = `${type}:${source}`;
    const next = { key, type, source, priority, payload, t: ctx.timestamp };
    if (ctx.db.latest.key.find(key)) ctx.db.latest.key.update(next);
    else ctx.db.latest.insert(next);
  }
);

// Start a fresh demo run: drop the log and the latest state.
export const clear = spacetimedb.reducer(ctx => {
  for (const r of [...ctx.db.event.iter()]) ctx.db.event.id.delete(r.id);
  for (const r of [...ctx.db.latest.iter()]) ctx.db.latest.key.delete(r.key);
});
