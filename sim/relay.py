"""Frame relay between the browser sim and perception.

    python sim/relay.py [--port 5020]

ws://127.0.0.1:5020/pub  the sim sends binary frames (1 byte camera id: 0 drone, 1 chest,
                         then JPEG bytes) and JSON text pose messages.
ws://127.0.0.1:5020/sub  receives the same messages. On connect it gets the latest frame
                         and pose per camera straight away.

Only the sim page served from this machine (an http://localhost or 127.0.0.1 origin) and
non-browser clients (no Origin header: the detector, the Street View bridge) may connect;
--origin adds one more allowed browser origin.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from http import HTTPStatus

import websockets

subs: set = set()
latest: dict = {}          # ("frame"|"pose", camera) -> last message
CAMERAS = ("drone", "chest")   # frame ids 0 and 1
# Browsers let any web page open a WebSocket to 127.0.0.1, so without this a page in another tab
# could feed fake frames to perception (or hide a real car) and watch both cameras.
LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?$")
extra_origins: set[str] = set()


def allowed(origin: str | None) -> bool:
    """No Origin header (not a browser), a loopback page, or one passed with --origin."""
    return origin is None or bool(LOCAL_ORIGIN.match(origin)) or origin in extra_origins


def check_origin(connection, request):
    if not allowed(request.headers.get("Origin")):
        return connection.respond(HTTPStatus.FORBIDDEN, "origin not allowed\n")
    return None


def remember(msg) -> None:
    if isinstance(msg, bytes):
        if len(msg) > 1 and msg[0] in (0, 1):
            latest[("frame", msg[0])] = msg
        return
    try:
        d = json.loads(msg)
    except json.JSONDecodeError:
        return
    if isinstance(d, dict) and d.get("type") == "pose" and d.get("camera") in CAMERAS:
        latest[("pose", d.get("camera"))] = msg


async def handler(ws) -> None:
    path = ws.request.path
    if path == "/sub":
        subs.add(ws)
        try:
            for m in list(latest.values()):
                await ws.send(m)
            await ws.wait_closed()
        finally:
            subs.discard(ws)
    elif path == "/pub":
        async for msg in ws:
            remember(msg)
            websockets.broadcast(subs, msg)   # skips slow subscribers instead of queueing
    else:
        await ws.close(1008, "use /pub or /sub")


async def serve(host: str, port: int) -> None:
    async with websockets.serve(handler, host, port, max_size=8 * 1024 * 1024,
                                process_request=check_origin):
        print(f"[relay] ws://{host}:{port}/pub -> /sub", flush=True)
        await asyncio.Future()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5020)
    ap.add_argument("--origin", action="append", default=[],
                    help="also accept browser connections from this origin, e.g. http://192.168.1.5:5173")
    a = ap.parse_args()
    extra_origins.update(o.rstrip("/") for o in a.origin)
    try:
        asyncio.run(serve(a.host, a.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
