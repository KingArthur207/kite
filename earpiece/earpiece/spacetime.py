"""Mirror Kite events into SpacetimeDB (the `push` reducer in dashboard/spacetimedb).

    from earpiece.spacetime import SpacetimeBridge
    st = SpacetimeBridge()            # SPACETIME_HOST / SPACETIME_DB from .env, else Maincloud
    st.publish({"type": "hazard", "source": "drone", "kind": "crowd", "bearing": 30})

publish() never blocks: events go on a bounded queue and one thread POSTs them.
If the queue is full (network down) new events are dropped, so the caller's loop
never stalls. Fits Earpiece.bridges.

CaregiverInbox is the one path back: it polls the `event` table for new `say` rows
with source "caregiver" (the dashboard's "Message the walker" box) and hands them to
Earpiece.handle, capped at priority "info".
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time

import httpx

from .config import _load_env
from .events import PRIORITY_RANK, BadMessage, parse

DEFAULT_HOST = "https://maincloud.spacetimedb.com"
DEFAULT_DB = "kite-sync-pl6rp"
QUEUE_MAX = 1000
TIMEOUT_S = 3.0
# Perception frames arrive at 10-30 Hz; the dashboard only needs the latest few per second.
THROTTLED = {"obstacles", "heartbeat"}
THROTTLE_S = 0.5
CAREGIVER = "caregiver"
CAREGIVER_PREFIX = "Message from your caregiver: "
CAREGIVER_SQL = "SELECT * FROM event WHERE source = 'caregiver'"   # fixed query, no inputs
POLL_S = 1.0
MAX_BACKOFF_S = 30.0


def _host_db(host: str | None, db: str | None) -> tuple[str, str]:
    _load_env()
    return ((host or os.environ.get("SPACETIME_HOST") or DEFAULT_HOST).rstrip("/"),
            db or os.environ.get("SPACETIME_DB") or DEFAULT_DB)


def _identity(http: httpx.Client, host: str) -> str:
    r = http.post(f"{host}/v1/identity")
    r.raise_for_status()
    return r.json()["token"]


class SpacetimeBridge:
    def __init__(self, host: str | None = None, db: str | None = None, source: str = "unknown", log=print):
        self.host, self.db = _host_db(host, db)
        self.log = log
        self.source = source
        self.stats = {"sent": 0, "failed": 0, "dropped": 0}
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._token: str | None = None
        self._last: dict[tuple, float] = {}
        threading.Thread(target=self._run, name="spacetime", daemon=True).start()

    def publish(self, obj: dict) -> None:
        if obj.get("type") in THROTTLED:
            key = (obj.get("type"), obj.get("source"))
            now = time.monotonic()
            if now - self._last.get(key, 0.0) < THROTTLE_S:
                return
            self._last[key] = now
        try:
            self._q.put_nowait(obj)
        except queue.Full:
            self.stats["dropped"] += 1

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until everything queued has been sent (for scripts that exit right after)."""
        done = threading.Event()
        threading.Thread(target=lambda: (self._q.join(), done.set()), daemon=True).start()
        return done.wait(timeout)

    def _run(self) -> None:
        with httpx.Client(timeout=TIMEOUT_S) as http:
            while True:
                obj = self._q.get()
                try:
                    self._send(http, obj)
                    self.stats["sent"] += 1
                except Exception as e:
                    self.stats["failed"] += 1
                    if self.stats["failed"] <= 5 or self.stats["failed"] % 100 == 0:
                        self.log(f"[spacetime] {type(e).__name__}: {e}")
                finally:
                    self._q.task_done()

    def _send(self, http: httpx.Client, obj: dict) -> None:
        if self._token is None:
            self._token = _identity(http, self.host)
        args = {
            "type": str(obj.get("type", "")),
            "source": str(obj.get("source") or self.source)[:64],
            "priority": str(obj.get("priority") or ""),
            "payload": json.dumps(obj, separators=(",", ":"), default=str),
        }
        r = http.post(f"{self.host}/v1/database/{self.db}/call/push", json=args,
                      headers={"Authorization": f"Bearer {self._token}"})
        if r.status_code >= 400:
            raise RuntimeError(f"push {args['type']} -> {r.status_code} {r.text[:120]}")


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
        self.interval = interval
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
        r = http.post(f"{self.host}/v1/database/{self.db}/sql",
                      content=CAREGIVER_SQL,
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
