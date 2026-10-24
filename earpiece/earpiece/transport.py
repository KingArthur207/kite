"""Network edges: how the rest of Kite talks to the earpiece, and how it talks back.

    UDP  :5010  JSON datagrams in (fire-and-forget; best for 10-30 Hz perception frames)
    WS   :5011  JSON both ways: events in; out go `state` (5 Hz), `spoken`, `heard`,
                `voice_request` (e.g. navigate), `caregiver_alert`
Both bind to localhost by default (--bind 0.0.0.0 to accept the drone laptop on
the LAN). Each peer is rate-limited (token bucket) so a runaway sender cannot
starve the audio loop, and every message is validated by events.parse().

Browsers let any web page open a WebSocket to 127.0.0.1, so the WebSocket also
checks the Origin header: only pages served from this machine (the sim and the
Street View bridge, any loopback port) and non-browser clients (no Origin: Python
senders, scripts/replay.py, the drone laptop) may connect. Anything else gets 403,
so a page in another tab can never make the earpiece speak. `allow_origins` (run
--allow-origin) adds more, e.g. a sim page opened from another machine.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import Callable

from .events import BadMessage, Msg, parse

LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?$")
MAX_BUCKETS = 4096          # per-peer rate limiters kept; idle ones are evicted beyond this


def origin_allowed(origin: str | None, extra: frozenset[str] = frozenset()) -> bool:
    """No Origin header (not a browser), a page served from this machine, or an explicit extra."""
    if origin is None:
        return True
    origin = origin.strip().rstrip("/")
    return bool(LOCAL_ORIGIN.match(origin)) or origin in extra


class _Bucket:
    def __init__(self, rate: float, burst: float):
        self.rate, self.burst = rate, burst
        self.tokens, self.t = burst, time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class Transport:
    def __init__(self, on_msg: Callable[[Msg], None], host: str = "127.0.0.1", udp_port: int = 5010,
                 ws_port: int = 5011, log=print, rate_per_peer: float = 400.0,
                 allow_origins: tuple[str, ...] = ()):
        self.on_msg = on_msg
        self.allow_origins = frozenset(o.strip().rstrip("/") for o in allow_origins)
        self.host = host
        self.udp_port = udp_port
        self.ws_port = ws_port
        self.log = log
        self.rate = rate_per_peer
        self._buckets: dict[str, _Bucket] = {}
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._servers: list = []
        self.stats = {"udp": 0, "ws": 0, "bad": 0, "limited": 0, "clients": 0, "last_error": None,
                      "origin_rejected": 0}
        self.state_provider: Callable[[], dict] | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="transport", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("transport failed to start")
        if self.stats["last_error"]:
            raise RuntimeError(self.stats["last_error"])

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except Exception as e:
            self.stats["last_error"] = f"{type(e).__name__}: {e}"
            self._ready.set()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    def stop(self, timeout: float = 2.0) -> None:
        """Close both sockets and every client, then end the transport thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        async def _close() -> None:
            for srv in self._servers:
                srv.close()
            for ws in list(self._clients):
                try:
                    await asyncio.wait_for(ws.close(), timeout=0.5)
                except Exception:
                    pass
            for srv in self._servers:
                if hasattr(srv, "wait_closed"):
                    try:
                        await asyncio.wait_for(srv.wait_closed(), timeout=0.5)
                    except Exception:
                        pass
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.wait(tasks, timeout=0.5)
            loop.stop()

        try:
            asyncio.run_coroutine_threadsafe(_close(), loop)
        except RuntimeError:                 # loop already closed
            return
        if self._thread is not None:
            self._thread.join(timeout)

    async def _serve(self) -> None:
        import websockets
        loop = asyncio.get_running_loop()
        udp, _ = await loop.create_datagram_endpoint(lambda: _Udp(self), local_addr=(self.host, self.udp_port))
        self._servers.append(udp)
        self._servers.append(await websockets.serve(
            self._ws_handler, self.host, self.ws_port, max_size=256 * 1024, ping_interval=20,
            ping_timeout=20, process_request=self._check_origin))
        loop.create_task(self._state_pump())
        self.log(f"[net] listening: udp://{self.host}:{self.udp_port}  ws://{self.host}:{self.ws_port}")

    def _check_origin(self, connection, request):
        origin = request.headers.get("Origin")
        if origin_allowed(origin, self.allow_origins):
            return None
        self.stats["origin_rejected"] += 1
        if self.stats["origin_rejected"] <= 5:
            self.log(f"[net] refused a WebSocket from origin {origin[:100]!r}")
        from http import HTTPStatus
        return connection.respond(HTTPStatus.FORBIDDEN, "origin not allowed\n")

    def _bucket(self, peer: str) -> _Bucket:
        b = self._buckets.get(peer)
        if b is None:
            if len(self._buckets) >= MAX_BUCKETS:
                # every UDP source port is a peer: forget the ones that have been quiet long
                # enough to be full again (they lose nothing), else start over
                now = time.monotonic()
                idle = [k for k, v in self._buckets.items() if now - v.t > v.burst / v.rate]
                for k in idle or list(self._buckets):
                    del self._buckets[k]
            b = self._buckets[peer] = _Bucket(self.rate, self.rate)
        return b

    def ingest(self, data, peer: str, via: str) -> None:
        b = self._bucket(peer)
        if not b.take():
            self.stats["limited"] += 1
            return
        try:
            msg = parse(data)
        except Exception as e:               # BadMessage, or a parser bug: never take the socket down
            self.stats["bad"] += 1
            if self.stats["bad"] <= 20 or self.stats["bad"] % 500 == 0:
                why = e if isinstance(e, BadMessage) else f"{type(e).__name__}: {e}"
                self.log(f"[net] rejected message from {peer}: {str(why)[:200]}")
            return
        self.stats[via] += 1
        try:
            self.on_msg(msg)
        except Exception as e:
            self.log(f"[net] handler error for {msg.type}: {type(e).__name__}: {e}")

    async def _ws_handler(self, ws) -> None:
        peer = f"ws:{ws.remote_address[0] if ws.remote_address else '?'}:{id(ws)}"
        self._clients.add(ws)
        self.stats["clients"] = len(self._clients)
        try:
            if self.state_provider is not None:
                await ws.send(json.dumps({"type": "state", **self.state_provider()}, default=str))
            async for raw in ws:
                self.ingest(raw, peer, "ws")
        except Exception:
            pass
        finally:
            self._clients.discard(ws)
            self._buckets.pop(peer, None)
            self.stats["clients"] = len(self._clients)

    async def _state_pump(self) -> None:
        while True:
            await asyncio.sleep(0.2)
            if self._clients and self.state_provider is not None:
                try:
                    await self._broadcast({"type": "state", **self.state_provider()})
                except Exception as e:
                    self.log(f"[net] state pump: {e}")

    async def _broadcast(self, obj: dict) -> None:
        if not self._clients:
            return
        data = json.dumps(obj, default=str)
        dead = []
        for ws in list(self._clients):
            try:
                await asyncio.wait_for(ws.send(data), timeout=0.5)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def publish(self, obj: dict) -> None:
        """Thread-safe broadcast to every connected WebSocket client."""
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(obj), self._loop)


class _Udp(asyncio.DatagramProtocol):
    def __init__(self, t: Transport):
        self.t = t

    def datagram_received(self, data: bytes, addr) -> None:
        self.t.ingest(data, f"udp:{addr[0]}:{addr[1]}", "udp")
