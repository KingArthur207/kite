// The link to Kite through SpacetimeDB: each GPS fix is one `push` reducer call over HTTPS, the same call the Python
// bridge makes (earpiece/earpiece/spacetime.py). No Bluetooth, no board, no LAN: the phone's own internet is enough,
// and every computer reads the walker's position from the `latest` and `event` tables (python -m phone.latest,
// python -m phone.cloud, the live dashboard).
import { type Fix, toPosition } from './frame';

export const HOST = 'https://maincloud.spacetimedb.com';
export const DB = 'kite-sync-pl6rp';
export const SOURCE = 'phone';

export type LinkState = 'searching' | 'connected' | 'lost';

export const SEND_MS = 1000; // at most one push a second, the rate the Python bridge mirrors positions at
export const HEARTBEAT_MS = 2000; // standing still, iOS may go quiet: re-send the newest fix so Kite knows we're here
export const TIMEOUT_MS = 5000;
export const LOST_AFTER = 2; // failed pushes in a row before "lost"
export const MAX_BACKOFF_MS = 5000;

type Timer = ReturnType<typeof setTimeout>;

export interface Deps {
  fetch: typeof fetch;
  now: () => number;
  setTimeout: (fn: () => void, ms: number) => Timer;
  clearTimeout: (t: Timer | undefined) => void;
}

const realDeps: Deps = {
  fetch: (...a) => fetch(...a),
  now: () => Date.now(),
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (t) => clearTimeout(t),
};

interface LinkOptions {
  onState: (state: LinkState) => void;
  onSent: (count: number) => void;
  host?: string;
  db?: string;
  source?: string;
  deps?: Partial<Deps>;
}

export interface Link {
  send: (fix: Fix) => void; // a new fix: it goes out as soon as the rate allows; a newer one replaces it
  stop: () => void;
}

class Rejected extends Error {} // the reducer refused this row: sending it again won't help

export function startLink({ onState, onSent, host = HOST, db = DB, source = SOURCE, deps }: LinkOptions): Link {
  const d: Deps = { ...realDeps, ...deps };
  let alive = true;
  let state: LinkState = 'searching';
  let sent = 0;
  let failures = 0;
  let token: string | null = null;
  let newest: Fix | null = null; // not sent yet
  let last: Fix | null = null; // the last fix that went out, for the heartbeat
  let busy = false;
  let lastAt = -Infinity; // when the last push started
  let nextAt = -Infinity; // earliest start for the next push: the rate limit, or the backoff after a failure
  let timer: Timer | undefined;

  const setState = (next: LinkState) => {
    if (!alive || next === state) return;
    state = next;
    onState(next);
  };

  const post = async (url: string, body: string | undefined, headers: Record<string, string>): Promise<Response> => {
    const abort = new AbortController();
    const t = d.setTimeout(() => abort.abort(), TIMEOUT_MS);
    try {
      return await d.fetch(url, { method: 'POST', body, headers, signal: abort.signal });
    } finally {
      d.clearTimeout(t);
    }
  };

  const identity = async (): Promise<string> => {
    const r = await post(`${host}/v1/identity`, undefined, {});
    if (!r.ok) throw new Error(`identity ${r.status}`);
    const got = (await r.json()) as { token?: unknown };
    if (typeof got.token !== 'string') throw new Error('identity without a token');
    return got.token;
  };

  const push = async (fix: Fix) => {
    token ??= await identity();
    const args = { type: 'position', source, priority: '', payload: JSON.stringify(toPosition(fix, source)) };
    const r = await post(`${host}/v1/database/${db}/call/push`, JSON.stringify(args), {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    });
    if (r.ok) return;
    if (r.status === 401 || r.status === 403) {
      token = null; // a stale identity: the next try fetches a new one
      throw new Error(`push ${r.status}`);
    }
    if (r.status === 530 || (r.status >= 400 && r.status < 500)) throw new Rejected(`push ${r.status}`);
    throw new Error(`push ${r.status}`);
  };

  const pump = () => {
    if (!alive || busy) return;
    const fix = newest ?? last;
    if (!fix) return;
    const due = newest ? nextAt : Math.max(nextAt, lastAt + HEARTBEAT_MS);
    const now = d.now();
    if (now < due) {
      d.clearTimeout(timer);
      timer = d.setTimeout(pump, due - now);
      return;
    }
    void attempt(fix);
  };

  const attempt = async (fix: Fix) => {
    busy = true;
    newest = null;
    last = fix;
    lastAt = d.now();
    try {
      await push(fix);
      failures = 0;
      nextAt = lastAt + SEND_MS;
      if (alive) onSent(++sent);
      setState('connected');
    } catch (e) {
      if (e instanceof Rejected) {
        console.warn('[kite] dropped a fix:', e.message);
        nextAt = lastAt + SEND_MS;
      } else {
        failures += 1;
        newest ??= fix; // try this fix again unless a newer one came in meanwhile
        nextAt = d.now() + Math.min(MAX_BACKOFF_MS, 500 * 2 ** (failures - 1));
        if (failures >= LOST_AFTER && state === 'connected') setState('lost');
      }
    } finally {
      busy = false;
      pump();
    }
  };

  onState('searching');
  return {
    send: (fix) => {
      newest = fix;
      pump();
    },
    stop: () => {
      alive = false;
      d.clearTimeout(timer);
    },
  };
}
