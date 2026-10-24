#!/usr/bin/env python3
"""Runs on the Arduino UNO Q's Linux side: say hello on the network once a second.

Sends a small JSON line over UDP broadcast on port 5099, so any computer on the same Wi-Fi
(e.g. Rex's Windows computer running its Mobile Hotspot) hears it with no IP addresses to set up:

    {"from": "kite-unoq", "host": "<board name>", "n": 12, "t": 1761345678.2}

Set KITE_HELLO_HOST=192.168.137.1 to send to one computer instead of broadcasting (some networks drop
broadcasts). No processing yet: this only proves the board and the computer can talk wirelessly.
Standard library only; installed as a service by install.sh so it starts at power-on.
"""

import json
import os
import socket
import time

PORT = int(os.environ.get("KITE_HELLO_PORT", "5099"))
HOST = os.environ.get("KITE_HELLO_HOST", "255.255.255.255")
EVERY_S = float(os.environ.get("KITE_HELLO_EVERY_S", "1"))


def main() -> None:
    name = socket.gethostname()
    n = 0
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                while True:
                    n += 1
                    msg = {"from": "kite-unoq", "host": name, "n": n, "t": round(time.time(), 1)}
                    s.sendto((json.dumps(msg) + "\n").encode(), (HOST, PORT))
                    print(f"hello {n} -> {HOST}:{PORT}", flush=True)
                    time.sleep(EVERY_S)
        except OSError as e:          # no network yet (still joining Wi-Fi after boot): wait and retry
            print(f"network not ready ({e}); retrying in 2 s", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
