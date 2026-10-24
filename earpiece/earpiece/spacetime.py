"""Mirror Kite events into SpacetimeDB (the `push` reducer in dashboard/spacetimedb).

    from earpiece.spacetime import SpacetimeBridge
    st = SpacetimeBridge()            # SPACETIME_HOST / SPACETIME_DB from .env, else Maincloud
    st.publish({"type": "hazard", "source": "drone", "kind": "crowd", "bearing": 30})

publish() never blocks: events go on a bounded queue and one thread POSTs them.
If the queue is full (network down) new events are dropped, so the caller's loop
never stalls. Fits Earpiece.bridges.

Maincloud bills per call and the demo runs until Sunday 9 AM, so the bridge is frugal:
  * heartbeats are never mirrored (they carry no state anyone else reads);
  * high-rate state (obstacles, position, beacon, status, output) keeps only the NEWEST
    pending event per (type, source) and sends it at most once a second (KITE_SPACETIME_HZ);
  * a token bucket caps every reducer call, retries included, at 5/s (KITE_SPACETIME_MAX_CPS);
  * events the module would reject (unknown type, bad source or priority, over 16 KB) are
    fixed or dropped here instead of costing a round trip;
  * network errors back off exponentially; a schema mismatch (no such reducer, changed
    arguments) disables the bridge with one log line instead of failing on every event.
stats["calls"] counts every reducer call this session.

CaregiverInbox is the one path back: it polls the `event` table for new `say` rows
with source "caregiver" (the dashboard's "Message the walker" box) and hands them to
Earpiece.handle, capped at priority "info".
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time

import httpx

from .config import _load_env
from .events import PRIORITY_RANK, BadMessage, parse

DEFAULT_HOST = "https://maincloud.spacetimedb.com"
DEFAULT_DB = "kite-sync-pl6rp"
QUEUE_MAX = 1000
TIMEOUT_S = 3.0
# Perception frames arrive at 10-30 Hz and the walker moves ~1.4 m/s; the dashboard only
# needs the latest value about once a second. Newest pending wins; nothing is sent late.
THROTTLED = frozenset({"obstacles", "position", "beacon", "status", "output"})
THROTTLE_S = 1.0                     # default; KITE_SPACETIME_HZ overrides (2 -> every 0.5 s)
NEVER_MIRRORED = frozenset({"heartbeat"})
MAX_CALLS_PER_S = 5.0                # global ceiling on reducer calls; KITE_SPACETIME_MAX_CPS
CAREGIVER = "caregiver"
CAREGIVER_PREFIX = "Message from your caregiver: "
CAREGIVER_SQL = "SELECT * FROM event WHERE source = 'caregiver'"   # fixed query, no inputs
POLL_S = 1.0
MIN_POLL_S = 0.5                     # SQL reads cost credits too: never poll faster than 2 Hz
MAX_BACKOFF_S = 30.0

# Mirrors of the module's own checks (dashboard/spacetimedb/src/index.ts).
TYPES = frozenset({
    "obstacles", "hazard", "guidance", "beacon", "position", "status", "scene", "say",
    "heartbeat", "control", "memory", "spoken", "heard", "voice_request", "caregiver_alert",
    "gesture", "output",
})
PRIORITIES = frozenset({"critical", "hazard", "nav", "answer", "info", "system", ""})
MAX_PAYLOAD = 16 * 1024              # module limit in UTF-16 units; our JSON is ASCII, so chars
MAX_TEXT = 2000                      # longest single string kept when a payload must shrink
_IDENT = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")     # the module's /^[\w.:-]{1,64}$/ (JS \w is ASCII)
_NOT_IDENT = re.compile(r"[^A-Za-z0-9_.:-]")


def _host_db(host: str | None, db: str | None) -> tuple[str, str]:
    _load_env()
    return ((host or os.environ.get("SPACETIME_HOST") or DEFAULT_HOST).rstrip("/"),
            db or os.environ.get("SPACETIME_DB") or DEFAULT_DB)


def _identity(http: httpx.Client, host: str) -> str:
    r = http.post(f"{host}/v1/identity")
    r.raise_for_status()
    return r.json()["token"]


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return v if math.isfinite(v) and v > 0 else default


def clean_source(v, default: str) -> str:
    """Any source -> the module's IDENT, so a stray space or slash never costs a rejected call."""
    s = "" if v is None else str(v)
    if _IDENT.match(s):
        return s
    return _NOT_IDENT.sub("_", s)[:64] or default


def _finite(v):
    """The server's JSON.parse rejects NaN and Infinity; Python's json happily writes them."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    return v


def _dumps(obj) -> str:
    try:
        return json.dumps(obj, separators=(",", ":"), default=str, allow_nan=False)
    except ValueError:
        return json.dumps(_finite(obj), separators=(",", ":"), default=str, allow_nan=False)


def encode_payload(obj: dict, limit: int = MAX_PAYLOAD) -> str:
    """The event as compact ASCII JSON that always fits the module's 16 KB payload cap.

    WHY not just drop it: a crowded obstacles frame is still worth showing. Long strings are
    cut first, then the longest lists are halved (nearest objects kept when they carry a
    distance) and `<list>_total` records how many there were."""
    s = _dumps(obj)
    if len(s) <= limit:
        return s
    o = _finite(json.loads(s))
    for k, v in list(o.items()):
        if isinstance(v, str) and len(v) > MAX_TEXT:
            o[k] = v[:MAX_TEXT]
            o["truncated"] = True
    while len(s := _dumps(o)) > limit:
        lists = [(len(v), k) for k, v in o.items() if isinstance(v, list) and v]
        if not lists:
            break
        n, k = max(lists)
        if f"{k}_total" not in o:
            o[f"{k}_total"] = n
            if all(isinstance(x, dict) and isinstance(x.get("distance"), (int, float)) for x in o[k]):
                o[k] = sorted(o[k], key=lambda x: x["distance"])
        o[k] = o[k][: n // 2]
        o["truncated"] = True
    if len(s) <= limit:
        return s
    return _dumps({"type": obj.get("type"), "source": obj.get("source"), "truncated": True, "bytes": len(s)})


class SpacetimeBridge:
    def __init__(self, host: str | None = None, db: str | None = None, source: str = "unknown", log=print,
                 *, throttle_s: float | None = None, max_calls_per_s: float | None = None,
                 backoff_s: float = 0.5, start: bool = True):
        self.host, self.db = _host_db(host, db)
        self.log = log
        self.source = source
        self.throttle_s = throttle_s or 1.0 / _env_float("KITE_SPACETIME_HZ", 1.0 / THROTTLE_S)
        self.max_calls_per_s = max_calls_per_s or _env_float("KITE_SPACETIME_MAX_CPS", MAX_CALLS_PER_S)
        self.backoff_s = backoff_s
        self.stats = {"sent": 0, "failed": 0, "dropped": 0, "calls": 0, "coalesced": 0, "skipped": 0,
                      "last_ms": None, "backlog": 0, "disabled": None}
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._token: str | None = None
        self._lock = threading.Lock()
        self._pending: dict[tuple, dict] = {}    # (type, source) -> its one queued copy, kept newest
        self._deferred: dict[tuple, dict] = {}   # pending copies taken off the queue, not due yet
        self._last: dict[tuple, float] = {}      # (type, source) -> when it was last sent
        self._bad_types: set[str] = set()
        self._noted: set[str] = set()
        self._tokens = min(self.max_calls_per_s, 5.0)
        self._tokens_t = time.monotonic()
        self._backoff = 0.0
        self._why = ""
        if start:
            threading.Thread(target=self._run, name="spacetime", daemon=True).start()

    def publish(self, obj: dict) -> None:
        kind = obj.get("type")
        if kind in NEVER_MIRRORED:
            self.stats["skipped"] += 1
            return
        if self.stats["disabled"] or kind not in TYPES or kind in self._bad_types:
            self.stats["dropped"] += 1
            if not self.stats["disabled"]:
                self._note(f"type:{kind}", f"[spacetime] not mirroring {kind!r} events "
                                           "(the module does not accept them)")
            return
        if kind in THROTTLED:
            key = (kind, clean_source(obj.get("source"), self.source))
            with self._lock:
                held = self._pending.get(key)
                if held is not None:             # still waiting to go: newest wins, in place
                    held.clear()
                    held.update(obj)
                    self.stats["coalesced"] += 1
                    return
                held = dict(obj)
                try:
                    self._q.put_nowait(held)
                except queue.Full:
                    self.stats["dropped"] += 1
                    return
                self._pending[key] = held
        else:
            try:
                self._q.put_nowait(obj)
            except queue.Full:
                self.stats["dropped"] += 1
                return
        self.stats["backlog"] = self._q.unfinished_tasks

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until everything queued has been sent (for scripts that exit right after)."""
        done = threading.Event()
        threading.Thread(target=lambda: (self._q.join(), done.set()), daemon=True).start()
        return done.wait(timeout)

    def _note(self, key: str, msg: str) -> None:
        if key not in self._noted:
            self._noted.add(key)
            self.log(msg)

    def _next(self) -> dict:
        """Next event to send. Throttled keys that are not due yet wait in _deferred (where
        publish still updates them in place) while ordered events keep flowing past."""
        while True:
            now = time.monotonic()
            wait = None
            with self._lock:
                for key, held in list(self._deferred.items()):
                    due = self._last.get(key, -1e9) + self.throttle_s
                    if due <= now:
                        del self._deferred[key]
                        return self._take(key, held, now)
                    wait = due - now if wait is None else min(wait, due - now)
            try:
                obj = self._q.get(timeout=wait)
            except queue.Empty:
                continue
            kind = obj.get("type")
            if kind in THROTTLED:
                key = (kind, clean_source(obj.get("source"), self.source))
                with self._lock:
                    if self._pending.get(key) is obj:
                        if time.monotonic() - self._last.get(key, -1e9) < self.throttle_s:
                            self._deferred[key] = obj    # its task_done comes when it is sent
                            continue
                        return self._take(key, obj, time.monotonic())
            return obj

    def _take(self, key: tuple, held: dict, now: float) -> dict:
        """Detach a pending copy (caller holds the lock); later publishes start a new one."""
        self._pending.pop(key, None)
        self._last[key] = now
        return dict(held)

    def _run(self) -> None:
        with httpx.Client(timeout=TIMEOUT_S) as http:
            while True:
                obj = self._next()
                try:
                    self._deliver(http, obj)
                except Exception as e:           # one bad event never kills the worker
                    self.stats["failed"] += 1
                    self.log(f"[spacetime] {type(e).__name__}: {e}")
                finally:
                    self._q.task_done()
                    self.stats["backlog"] = self._q.unfinished_tasks

    def _args(self, obj: dict) -> dict:
        p = str(obj.get("priority") or "").lower()
        return {"type": str(obj.get("type", "")), "source": clean_source(obj.get("source"), self.source),
                "priority": p if p in PRIORITIES else "", "payload": encode_payload(obj)}

    def _deliver(self, http: httpx.Client, obj: dict, tries: int = 4) -> None:
        if self.stats["disabled"] or obj.get("type") in self._bad_types:
            self.stats["dropped"] += 1
            return
        args = self._args(obj)
        for _ in range(tries):
            if self._backoff:
                time.sleep(self._backoff)
            self._take_token()
            try:
                ok = self._send(http, args)
                why = self._why
            except (httpx.HTTPError, OSError, ValueError, KeyError) as e:
                ok, why = None, f"{type(e).__name__}: {e}"
            if ok is not None:
                self._backoff = 0.0
                return
            self.stats["failed"] += 1
            self._backoff = min(MAX_BACKOFF_S, self._backoff * 2 if self._backoff else self.backoff_s)
            if self.stats["failed"] <= 5 or self.stats["failed"] % 100 == 0:
                self.log(f"[spacetime] {why}; retrying in {self._backoff:g}s")
        self.stats["dropped"] += 1

    def _take_token(self) -> None:
        """Token bucket over every reducer call: at most max_calls_per_s, bursts of up to 5."""
        rate = self.max_calls_per_s
        while True:
            now = time.monotonic()
            self._tokens = min(max(1.0, min(rate, 5.0)), self._tokens + (now - self._tokens_t) * rate)
            self._tokens_t = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            time.sleep((1.0 - self._tokens) / rate)

    def _send(self, http: httpx.Client, args: dict) -> bool | None:
        """True = landed, False = rejected for good (not retried), None = try again later."""
        if self._token is None:
            self._token = _identity(http, self.host)
        self.stats["calls"] += 1
        t0 = time.monotonic()
        r = http.post(f"{self.host}/v1/database/{self.db}/call/push", json=args,
                      headers={"Authorization": f"Bearer {self._token}"})
        if r.status_code < 300:
            self.stats["sent"] += 1
            self.stats["last_ms"] = round((time.monotonic() - t0) * 1000, 1)
            return True
        body = r.text[:120].strip()
        self._why = f"push {args['type']} -> {r.status_code} {body}"
        if r.status_code in (401, 403):          # stale identity: fetch a new one
            self._token = None
            return None
        if r.status_code == 530:                 # the reducer threw for this row: retrying won't help
            self.stats["failed"] += 1
            if body.startswith("unknown type"):
                self._bad_types.add(args["type"])
            self._note(f"rej:{args['type']}", f"[spacetime] module rejected {self._why}")
            return False
        if r.status_code in (400, 404):          # no such reducer or database, or its arguments changed
            self.stats["failed"] += 1
            self.stats["disabled"] = f"{r.status_code} {body}"
            self.log(f"[spacetime] disabled: {self._why}. Events stay local; "
                     "check the module and SPACETIME_DB.")
            return False
        return None                              # 5xx, 429: back off and retry


class CaregiverInbox:
    """Polls Spacetime about once a second for new caregiver `say` events and calls handle(msg).

    Rows already in the table at startup are skipped. Runs on its own daemon thread, so a slow
    or dead network never touches the audio loop; errors are logged a few times and backed off.
    Earpiece.handle does not mirror caregiver events back, so nothing loops.
    """

    def __init__(self, handle, host: str | None = None, db: str | None = None, log=print,
                 interval: float = POLL_S, start: bool = True):
        self.host, self.db = _host_db(host, db)
        self.handle = handle
        self.log = log
        self.interval = max(interval, MIN_POLL_S)
        self.last_id: int | None = None          # None until the first successful poll
        self.stats = {"polls": 0, "spoken": 0, "bad": 0, "failed": 0}
        self._token: str | None = None
        if start:
            threading.Thread(target=self._run, name="caregiver-inbox", daemon=True).start()

    def _run(self) -> None:
        fails = 0
        with httpx.Client(timeout=TIMEOUT_S) as http:
            while True:
                try:
                    self.poll(http)
                    fails = 0
                except Exception as e:
                    fails += 1
                    self._token = None                # cheap to re-fetch; covers an expired token
                    self.stats["failed"] += 1
                    if fails <= 3 or fails % 100 == 0:
                        self.log(f"[caregiver] {type(e).__name__}: {e}")
                time.sleep(min(MAX_BACKOFF_S, self.interval * 2 ** min(fails, 5)))

    def fetch(self, http: httpx.Client) -> list[tuple[int, str]]:
        """(id, payload) of every caregiver row, via the HTTP SQL endpoint."""
        if self._token is None:
            self._token = _identity(http, self.host)
        # Once we know where we are, ask only for newer rows: smaller replies, same answer.
        newer = f" AND id > {int(self.last_id)}" if self.last_id is not None and self.last_id >= 0 else ""
        r = http.post(f"{self.host}/v1/database/{self.db}/sql",
                      # fixed query; `newer` is built from an int only, so nothing is injectable
                      content=CAREGIVER_SQL + newer,  # nosec B608
                      headers={"Authorization": f"Bearer {self._token}"})
        r.raise_for_status()
        out = []
        for stmt in r.json():
            names = [(e.get("name") or {}).get("some") for e in stmt["schema"]["elements"]]
            out += [(int(row[names.index("id")]), str(row[names.index("payload")])) for row in stmt["rows"]]
        return out

    def poll(self, http: httpx.Client) -> None:
        rows = sorted(self.fetch(http))
        self.stats["polls"] += 1
        if self.last_id is None:                  # startup: old messages are history, not news
            self.last_id = rows[-1][0] if rows else -1
            return
        for rid, payload in rows:
            if rid <= self.last_id:
                continue
            self.last_id = rid
            try:
                msg = parse(payload)
            except BadMessage as e:
                self.stats["bad"] += 1
                self.log(f"[caregiver] ignored row {rid}: {e}")
                continue
            if msg.type != "say":                 # only words flow back; no controls, no hazards
                self.stats["bad"] += 1
                continue
            msg.source = CAREGIVER
            if PRIORITY_RANK[msg.priority] < PRIORITY_RANK["info"]:
                msg.priority = "info"             # can never outrank or interrupt a safety cue
            msg.text = CAREGIVER_PREFIX + msg.text
            self.stats["spoken"] += 1
            self.handle(msg)
