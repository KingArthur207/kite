// Self-check for view.ts and the payload hardening in timeline.ts: `npm run check`. Throws on failure.
import { clip, moments, parse, str, type Ev } from './timeline.ts';
import { ago, blockedWord, deliveryOf, direction, facing, freshOf, humanize, linkOf, meters, relative } from './view.ts';

function eq(actual: unknown, expected: unknown, what: string) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`${what}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`);
}
let n = 0;
const ev = (t: number, type: string, source: string, p: Record<string, unknown>, priority = 'info'): Ev =>
  ({ id: String(n++), t, type, source, priority, p: { type, ...p } });

// Malformed payloads never throw and never leak objects or markup-as-HTML into the page.
eq(parse('not json'), {}, 'bad JSON');
eq(parse('null'), {}, 'null payload');
eq(parse('[1, 2]'), {}, 'array payload');
eq(parse('"text"'), {}, 'string payload');
eq(parse('{"text": "hi"}'), { text: 'hi' }, 'object payload');
eq(str({ nested: 1 }), '', 'objects are not text');
eq(str(['a']), '', 'arrays are not text');
eq(str(NaN), '', 'NaN is not text');
eq(str(3), '3', 'numbers are text');
eq(clip('a  \n b'), 'a b', 'whitespace collapsed');
eq(clip('x'.repeat(500), 10), 'xxxxxxxxx…', 'long text clipped');
eq(clip('<img src=x onerror=alert(1)>'), '<img src=x onerror=alert(1)>', 'markup stays literal text (React escapes it)');

const T = 1_800_000_000_000;
const junk = [
  ev(T, 'say', 'judgment', { text: { a: 1 }, upgrade: 'nope', latency_ms: 'fast' }),
  ev(T + 1, 'spoken', 'earpiece', { text: 42, priority: 5 }),
  ev(T + 2, 'hazard', 'drone', { kind: null, bearing: 'left', distance: -3, ttl: 'soon' }),
  ev(T + 3, 'caregiver_alert', 'earpiece', { message: { x: 1 } }),
  ev(T + 4, 'scene', 'scene-chest', {}),
];
const ms = moments(junk, undefined);
eq(ms.map(m => m.kind), ['hazard', 'help'], 'junk rows become safe moments or none');
eq(ms[1].text, 'The walker asked for help.', 'help request without a message');

// Map placement only for sane bearing/distance.
eq(relative({ bearing: 10, distance: 8 }), { bearing: 10, distance: 8 }, 'usable relative');
eq(relative({ bearing: 1e308, distance: 1e308 }), undefined, 'absurd relative');
eq(relative({ bearing: 'left', distance: 3 }), undefined, 'string bearing');
eq(relative({ bearing: 5, distance: -1 }), undefined, 'negative distance');

eq([meters(0.4), meters(1.44), meters(8.2), meters(120.6), meters(-3), meters('9')], ['under 1 m', '1.4 m', '8.2 m', '121 m', '', ''], 'meters');
eq([0, 10, 30, -30, 90, -100, 150, 180, -175, 370].map(direction),
  ['ahead', 'ahead', 'ahead right', 'ahead left', 'right', 'left', 'behind right', 'behind', 'behind', 'ahead'], 'direction');
eq([facing(0), facing(320), facing(-90), facing(undefined)], ['facing north', 'facing north-west', 'facing west', ''], 'facing');
eq(humanize('bike_lane'), 'bike lane', 'humanize');

eq([freshOf(0), freshOf(4.9), freshOf(5), freshOf(14), freshOf(15)], ['fresh', 'fresh', 'stale', 'stale', 'dead'], 'freshness bands');
eq([ago(0), ago(12), ago(240), ago(7200)], ['just now', '12 s ago', '4 min ago', '2 h ago'], 'ago');

eq([linkOf(true, false), linkOf(false, false), linkOf(false, false, new Error('x')), linkOf(false, true, new Error('x'))],
  ['live', 'connecting', 'offline', 'reconnecting'], 'connection state');

// Caregiver messages: blocked words mirror the earpiece's permission filter.
eq(blockedWord("I'm on my way."), '', 'quick reply 1 is spoken');
eq(blockedWord('Call me when you get there.'), '', 'quick reply 2 is spoken');
eq(blockedWord('Take your time, no rush.'), '', 'quick reply 3 is spoken');
eq(blockedWord("Don't cross yet."), '', 'a hold is fine');
eq(blockedWord('You can go now'), 'You can', 'permission is blocked');
eq(blockedWord('The street is clear'), 'clear', 'all-clear is blocked');
eq(blockedWord('Head home'), 'Head', 'movement is blocked');

const sent = ev(T, 'say', 'caregiver', { text: "I'm on my way." });
const desc = (rows: Ev[]) => [...rows].sort((a, b) => b.t - a.t);
eq(deliveryOf(sent, desc([sent]), T + 5000), { state: 'sent' }, 'just sent');
eq(deliveryOf(sent, desc([sent]), T + 30_000), { state: 'unheard' }, 'not spoken after a while');
const heard = ev(T + 1800, 'spoken', 'earpiece', { text: "Message from your caregiver:  I'm on my way." });
eq(deliveryOf(sent, desc([sent, heard]), T + 30_000), { state: 'heard', heardAt: T + 1800 }, 'heard');
const otherLine = ev(T + 1800, 'spoken', 'earpiece', { text: 'Message from your caregiver: Something else.' });
eq(deliveryOf(sent, desc([sent, otherLine]), T + 5000).state, 'sent', 'a different message is not this one');
const earlier = ev(T - 60_000, 'spoken', 'earpiece', { text: "Message from your caregiver: I'm on my way." });
eq(deliveryOf(sent, desc([sent, earlier]), T + 5000).state, 'sent', 'an older identical line does not count');

console.log('view.check: ok');
