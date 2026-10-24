"""Network edges: how the rest of Kite talks to the earpiece, and how it talks back.

    UDP  :5010  JSON datagrams in (fire-and-forget; best for 10-30 Hz perception frames)
    WS   :5011  JSON both ways: events in; out go `state` (5 Hz), `spoken`, `heard`,
                `voice_request` (e.g. navigate), `caregiver_alert`
Both bind to localhost by default (--bind 0.0.0.0 to accept the drone laptop on
the LAN). Each peer is rate-limited (token bucket) so a runaway sender cannot
starve the audio loop, and every message is validated by events.parse().
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Callable

from .events import BadMessage, Msg, parse


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
                 ws_port: int = 5011, log=print, rate_per_peer: float = 400.0):
        self.on_msg = on_msg
        self.host = host
        self.udp_port = udp_port
        self.ws_port = ws_port
        self.log = log
        self.rate = rate_per_peer
        self._buckets: dict[str, _Bucket] = {}
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self.stats = {"udp": 0, "ws": 0, "bad": 0, "limited": 0, "clients": 0, "last_error": None}
        self.state_provider: Callable[[], dict] | None = None

    def start(self) -> None:
        threading.Thread(target=self._run, name="transport", daemon=True).start()
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
        loop.run_forever()

    async def _serve(self) -> None:
        import websockets
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: _Udp(self), local_addr=(self.host, self.udp_port))
        await websockets.serve(self._ws_handler, self.host, self.ws_port, max_size=256 * 1024,
                               ping_interval=20, ping_timeout=20)
        loop.create_task(self._state_pump())
        self.log(f"[net] listening: udp://{self.host}:{self.udp_port}  ws://{self.host}:{self.ws_port}")

    def ingest(self, data, peer: str, via: str) -> None:
        b = self._buckets.get(peer)
        if b is None:
            b = self._buckets[peer] = _Bucket(self.rate, self.rate)
        if not b.take():
            self.stats["limited"] += 1
            return
        try:
            msg = parse(data)
        except BadMessage as e:
            self.stats["bad"] += 1
            if self.stats["bad"] <= 20 or self.stats["bad"] % 500 == 0:
                self.log(f"[net] rejected message from {peer}: {e}")
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
