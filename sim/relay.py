"""Frame relay between the browser sim and perception.

    python sim/relay.py [--port 5020]

ws://127.0.0.1:5020/pub  the sim sends binary frames (1 byte camera id: 0 drone, 1 chest,
                         then JPEG bytes) and JSON text pose messages.
ws://127.0.0.1:5020/sub  receives the same messages. On connect it gets the latest frame
                         and pose per camera straight away. Text it sends whose type is in UPSTREAM (drone
                         commands for ?drone=sim, the route walker's "go") goes to every /pub connection,
                         i.e. the page. Anything else from a subscriber is dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import websockets

subs: set = set()
pubs: set = set()
UPSTREAM = ("drone_cmd", "walker_cmd")
latest: dict = {}          # ("frame"|"pose", camera) -> last message


def remember(msg) -> None:
    if isinstance(msg, bytes):
        if len(msg) > 1 and msg[0] in (0, 1):
            latest[("frame", msg[0])] = msg
        return
    try:
        d = json.loads(msg)
    except json.JSONDecodeError:
        return
    if isinstance(d, dict) and d.get("type") == "pose":
        latest[("pose", d.get("camera"))] = msg


def upstream(msg) -> bool:
    if not isinstance(msg, str):
        return False
    try:
        d = json.loads(msg)
    except json.JSONDecodeError:
        return False
    return isinstance(d, dict) and d.get("type") in UPSTREAM


async def handler(ws) -> None:
    path = ws.request.path
    if path == "/sub":
        subs.add(ws)
        try:
            for m in list(latest.values()):
                await ws.send(m)
            async for msg in ws:
                if upstream(msg):
                    websockets.broadcast(pubs, msg)
        finally:
            subs.discard(ws)
    elif path == "/pub":
        pubs.add(ws)
        try:
            async for msg in ws:
                remember(msg)
                websockets.broadcast(subs, msg)   # skips slow subscribers instead of queueing
        finally:
            pubs.discard(ws)
    else:
        await ws.close(1008, "use /pub or /sub")


async def serve(host: str, port: int) -> None:
    async with websockets.serve(handler, host, port, max_size=8 * 1024 * 1024):
        print(f"[relay] ws://{host}:{port}/pub -> /sub", flush=True)
        await asyncio.Future()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5020)
    a = ap.parse_args()
    try:
        asyncio.run(serve(a.host, a.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
