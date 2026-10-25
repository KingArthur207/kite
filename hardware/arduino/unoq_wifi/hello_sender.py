#!/usr/bin/env python3
"""Runs on the Arduino UNO Q's Linux side: say hello once a second, over any Wi-Fi with internet.

Campus Wi-Fi like eduroam usually stops devices from reaching each other directly, so the hello goes
through the internet instead: the board pushes it to Kite's SpacetimeDB database, and listen.py on any
computer anywhere reads it from there (it also shows up as a live device on the demo dashboard).

    {"msg": "hello", "n": 12, "host": "<board name>"}     pushed as type=heartbeat, source=unoq

KITE_HELLO_MODE=udp sends a UDP broadcast on port 5099 instead (only for a shared network that allows it,
like a phone hotspot); KITE_HELLO_MODE=both does both. No processing yet: this only proves the board and
the computer can talk. Standard library only; install.sh makes it start at every power-on.
"""

import json
import os
import socket
import time
import urllib.error
import urllib.request

SPACETIME_HOST = os.environ.get("SPACETIME_HOST", "https://maincloud.spacetimedb.com").rstrip("/")
SPACETIME_DB = os.environ.get("SPACETIME_DB", "kite-sync-pl6rp")
SOURCE = os.environ.get("KITE_HELLO_SOURCE", "unoq")
MODE = os.environ.get("KITE_HELLO_MODE", "cloud")            # cloud | udp | both
EVERY_S = float(os.environ.get("KITE_HELLO_EVERY_S", "1"))
UDP_HOST = os.environ.get("KITE_HELLO_HOST", "255.255.255.255")
UDP_PORT = int(os.environ.get("KITE_HELLO_PORT", "5099"))


def post(url: str, body: bytes | None, token: str | None = None, timeout: float = 10) -> bytes:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body if body is not None else b"", headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:   # nosec B310: fixed https host
        return r.read()


class Cloud:
    def __init__(self) -> None:
        self.token = None

    def send(self, msg: dict) -> None:
        if self.token is None:          # an anonymous identity, minted once (same as the earpiece)
            self.token = json.loads(post(f"{SPACETIME_HOST}/v1/identity", None))["token"]
        args = {"type": "heartbeat", "source": SOURCE, "priority": "system", "payload": json.dumps(msg)}
        try:
            post(f"{SPACETIME_HOST}/v1/database/{SPACETIME_DB}/call/push", json.dumps(args).encode(), self.token)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):    # token expired or revoked: mint a new one next time
                self.token = None
            raise


def main() -> None:
    name = socket.gethostname()
    cloud = Cloud() if MODE in ("cloud", "both") else None
    udp = None
    if MODE in ("udp", "both"):
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    n = 0
    while True:
        n += 1
        msg = {"msg": "hello", "n": n, "host": name}
        sent = []
        if cloud:
            try:
                cloud.send(msg)
                sent.append("cloud")
            except (OSError, ValueError, KeyError) as e:     # no internet yet, captive portal, server down
                print(f"hello {n}: cloud send failed ({type(e).__name__}: {e}); will retry", flush=True)
        if udp:
            try:
                udp.sendto((json.dumps(msg) + "\n").encode(), (UDP_HOST, UDP_PORT))
                sent.append("udp")
            except OSError as e:
                print(f"hello {n}: udp send failed ({e})", flush=True)
        if sent:
            print(f"hello {n} -> {', '.join(sent)}", flush=True)
        time.sleep(EVERY_S if sent else max(EVERY_S, 2.0))


if __name__ == "__main__":
    main()
