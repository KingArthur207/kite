"""Mirror Kite events into SpacetimeDB (the `push` reducer in dashboard/spacetimedb).

    from earpiece.spacetime import SpacetimeBridge
    st = SpacetimeBridge()            # SPACETIME_HOST / SPACETIME_DB from .env, else Maincloud
    st.publish({"type": "hazard", "source": "drone", "kind": "crowd", "bearing": 30})

publish() never blocks: events go on a bounded queue and one thread POSTs them.
If the queue is full (network down) new events are dropped, so the caller's loop
never stalls. Fits Earpiece.bridges.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time

import httpx

from .config import _load_env

DEFAULT_HOST = "https://maincloud.spacetimedb.com"
DEFAULT_DB = "kite-sync-pl6rp"
QUEUE_MAX = 1000
TIMEOUT_S = 3.0
# Perception frames arrive at 10-30 Hz; the dashboard only needs the latest few per second.
THROTTLED = {"obstacles", "heartbeat"}
THROTTLE_S = 0.5


class SpacetimeBridge:
    def __init__(self, host: str | None = None, db: str | None = None, source: str = "unknown", log=print):
        _load_env()
        self.host = (host or os.environ.get("SPACETIME_HOST") or DEFAULT_HOST).rstrip("/")
        self.db = db or os.environ.get("SPACETIME_DB") or DEFAULT_DB
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
            r = http.post(f"{self.host}/v1/identity")
            r.raise_for_status()
            self._token = r.json()["token"]
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
