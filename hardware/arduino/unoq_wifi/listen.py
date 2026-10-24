"""Runs on any computer with internet (Windows, macOS, Linux): print every hello the UNO Q sends.

    python listen.py            # through the internet (works on eduroam): reads Kite's SpacetimeDB
    python listen.py --udp      # direct UDP broadcast on port 5099 (only on a shared network that allows it)

No installs needed (standard library only). Prints each hello with the time it arrived, and a warning when
the board goes quiet. Reads only new rows, once a second.
"""

import argparse
import json
import socket
import time
import urllib.request
from datetime import datetime

SPACETIME_HOST = "https://maincloud.spacetimedb.com"
SPACETIME_DB = "kite-sync-pl6rp"
QUIET_S = 5.0


def post(url: str, body: bytes, token: str | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:   # nosec B310: fixed https host
        return r.read()


def rows(reply: bytes) -> list[dict]:
    out = []
    for stmt in json.loads(reply):
        names = [(e.get("name") or {}).get("some") for e in stmt["schema"]["elements"]]
        out += [dict(zip(names, r)) for r in stmt["rows"]]
    return out


def show(text: str, who: str) -> None:
    print(f"{datetime.now():%H:%M:%S}  {who:15s}  {text}", flush=True)


def listen_cloud(source: str, host: str, db: str) -> None:
    token = json.loads(post(f"{host}/v1/identity", b""))["token"]
    # `source` is checked in main() to letters, digits and _ . : - only, so nothing can be injected
    base = f"SELECT * FROM event WHERE type = 'heartbeat' AND source = '{source}'"  # nosec B608
    last_id = max((int(r["id"]) for r in rows(post(f"{host}/v1/database/{db}/sql", base.encode(), token))),
                  default=-1)
    print(f"listening for '{source}' through SpacetimeDB ({db}); Ctrl+C to stop")
    last_seen, warned = None, False
    while True:
        try:
            # ids are unsigned: with no earlier row, ask for everything rather than "id > -1"
            query = f"{base} AND id > {last_id}" if last_id >= 0 else base
            new = rows(post(f"{host}/v1/database/{db}/sql", query.encode(), token))
        except OSError as e:
            show(f"... can't reach SpacetimeDB ({e}); retrying", "")
            time.sleep(3)
            continue
        for r in sorted(new, key=lambda r: int(r["id"])):
            last_id = max(last_id, int(r["id"]))
            try:
                m = json.loads(r["payload"])
                show(f"{m.get('msg')} {m.get('n')} from {m.get('host')}", "via internet")
            except ValueError:
                show(str(r["payload"]), "via internet")
            last_seen, warned = time.monotonic(), False
        if last_seen and time.monotonic() - last_seen > QUIET_S and not warned:
            show(f"... nothing for {QUIET_S:.0f} s (board off, or lost its Wi-Fi?)", "")
            warned = True
        time.sleep(1.0)


def listen_udp(port: int) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.settimeout(1.0)
    print(f"listening for the UNO Q on UDP {port}; Ctrl+C to stop")
    last_seen, warned = None, False
    while True:
        try:
            data, (ip, _) = s.recvfrom(2048)
        except socket.timeout:
            if last_seen and time.monotonic() - last_seen > QUIET_S and not warned:
                show(f"... nothing for {QUIET_S:.0f} s (board off, or left the Wi-Fi?)", "")
                warned = True
            continue
        last_seen, warned = time.monotonic(), False
        try:
            m = json.loads(data)
            show(f"{m.get('msg')} {m.get('n')} from {m.get('host')}", ip)
        except ValueError:
            show(data.decode("utf-8", errors="replace").strip(), ip)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--udp", action="store_true", help="listen for UDP broadcasts instead of the internet")
    ap.add_argument("--port", type=int, default=5099)
    ap.add_argument("--source", default="unoq", help="the board's source name (default unoq)")
    ap.add_argument("--db", default=SPACETIME_DB)
    ap.add_argument("--host", default=SPACETIME_HOST)
    a = ap.parse_args()
    if not a.source.replace("_", "").replace("-", "").replace(".", "").replace(":", "").isalnum():
        raise SystemExit("--source: letters, digits, _ . : - only")
    try:
        listen_udp(a.port) if a.udp else listen_cloud(a.source, a.host, a.db)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
