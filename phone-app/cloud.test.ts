// bun test (from phone-app/): the SpacetimeDB link with a fake fetch and a fake clock, no network.
import { expect, test } from 'bun:test';
import { DB, HOST, type LinkState, startLink } from './cloud';
import type { Fix } from './frame';

const fix = (t: number, lat = 37.7783521): Fix => ({ t, lat, lng: -122.4087683, acc: 4.8, alt: null, speed: null, course: null });

type Reply = Response | Error;
const ok = () => new Response('', { status: 200 });
const identity = () => Response.json({ identity: 'c200abc', token: 'tok-1' });

const settle = async () => {
  for (let i = 0; i < 20; i++) await Promise.resolve();
};

function harness(onPush: (n: number) => Reply = ok) {
  let now = 1_000_000;
  let id = 0;
  const timers = new Map<number, { at: number; fn: () => void }>();
  const pushes: { body: Record<string, string>; auth: string }[] = [];
  let identities = 0;
  const states: LinkState[] = [];
  const sent: number[] = [];
  const link = startLink({
    onState: (s) => states.push(s),
    onSent: (n) => sent.push(n),
    deps: {
      now: () => now,
      setTimeout: ((fn: () => void, ms: number) => {
        timers.set(++id, { at: now + ms, fn });
        return id;
      }) as never,
      clearTimeout: ((k: number) => timers.delete(k)) as never,
      fetch: (async (url: string, init: RequestInit) => {
        let r: Reply;
        if (url === `${HOST}/v1/identity`) {
          identities += 1;
          r = identity();
        } else {
          expect(url).toBe(`${HOST}/v1/database/${DB}/call/push`);
          const headers = init.headers as Record<string, string>;
          pushes.push({ body: JSON.parse(init.body as string), auth: headers.Authorization });
          r = onPush(pushes.length);
        }
        if (r instanceof Error) throw r;
        return r;
      }) as never,
    },
  });
  const advance = async (ms: number) => {
    const end = now + ms;
    for (;;) {
      await settle();
      const due = [...timers].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      timers.delete(due[0]);
      now = Math.max(now, due[1].at);
      due[1].fn();
    }
    now = end;
    await settle();
  };
  const fixT = (i: number) => JSON.parse(pushes[i].body.payload).fix_t;
  return { link, pushes, states, sent, advance, fixT, identities: () => identities };
}

test('the first fix fetches an identity once, then pushes a position event with it', async () => {
  const h = harness();
  h.link.send(fix(1));
  await h.advance(0);
  expect(h.identities()).toBe(1);
  expect(h.pushes).toHaveLength(1);
  expect(h.pushes[0].auth).toBe('Bearer tok-1');
  expect(h.pushes[0].body).toMatchObject({ type: 'position', source: 'phone', priority: '' });
  expect(JSON.parse(h.pushes[0].body.payload)).toEqual({ type: 'position', source: 'phone', lat: 37.7783521, lng: -122.4087683, fix_t: 1, accuracy: 4.8 });
  expect(h.states).toEqual(['searching', 'connected']);
  expect(h.sent).toEqual([1]);
  h.link.send(fix(2));
  await h.advance(1000);
  expect(h.identities()).toBe(1);
});

test('at most one push a second, and only the newest fix of the ones that piled up', async () => {
  const h = harness();
  h.link.send(fix(1));
  await h.advance(100);
  h.link.send(fix(2));
  h.link.send(fix(3));
  await h.advance(500);
  expect(h.pushes).toHaveLength(1);
  await h.advance(400);
  expect(h.pushes).toHaveLength(2);
  expect(h.fixT(1)).toBe(3);
});

test('standing still, the newest fix is sent again every two seconds as a heartbeat', async () => {
  const h = harness();
  h.link.send(fix(1));
  await h.advance(1999);
  expect(h.pushes).toHaveLength(1);
  await h.advance(1);
  expect(h.pushes).toHaveLength(2);
  await h.advance(4000);
  expect(h.pushes).toHaveLength(4);
  expect([0, 1, 2, 3].map(h.fixT)).toEqual([1, 1, 1, 1]);
});

test('two failures in a row after connecting say lost; it backs off, keeps the newest fix, and comes back', async () => {
  let down = false;
  const h = harness(() => (down ? new TypeError('Network request failed') : ok()));
  h.link.send(fix(1));
  await h.advance(0);
  down = true;
  h.link.send(fix(2));
  await h.advance(1000); // push 2 fails: retry in 0.5 s
  expect(h.states.at(-1)).toBe('connected');
  await h.advance(500); // push 3 fails: lost, retry in 1 s
  expect(h.states.at(-1)).toBe('lost');
  h.link.send(fix(3));
  await h.advance(999);
  expect(h.pushes).toHaveLength(3);
  down = false;
  await h.advance(1);
  expect(h.pushes).toHaveLength(4);
  expect(h.fixT(3)).toBe(3);
  expect(h.states).toEqual(['searching', 'connected', 'lost', 'connected']);
  expect(h.sent).toEqual([1, 2]);
});

test('with no internet from the start it stays on connecting, never lost', async () => {
  const h = harness(() => new TypeError('Network request failed'));
  h.link.send(fix(1));
  await h.advance(20_000);
  expect(h.states).toEqual(['searching']);
  expect(h.pushes.length).toBeGreaterThan(4);
  expect(h.pushes.length).toBeLessThan(12); // backoff tops out at 5 s
});

test('a 401 drops the identity and the retry fetches a new one', async () => {
  const h = harness((n) => (n === 1 ? new Response('', { status: 401 }) : ok()));
  h.link.send(fix(1));
  await h.advance(500);
  expect(h.identities()).toBe(2);
  expect(h.pushes).toHaveLength(2);
  expect(h.states).toEqual(['searching', 'connected']);
});

test('a row the reducer refuses is dropped, not retried, and does not count as a failure', async () => {
  const h = harness((n) => (n === 1 ? new Response('bad source', { status: 530 }) : ok()));
  h.link.send(fix(1));
  await h.advance(999);
  expect(h.pushes).toHaveLength(1);
  h.link.send(fix(2));
  await h.advance(1);
  expect(h.pushes).toHaveLength(2);
  expect(h.fixT(1)).toBe(2);
  expect(h.states).toEqual(['searching', 'connected']);
});

test('stop ends every push and heartbeat', async () => {
  const h = harness();
  h.link.send(fix(1));
  await h.advance(0);
  h.link.stop();
  h.link.send(fix(2));
  await h.advance(10_000);
  expect(h.pushes).toHaveLength(1);
});
