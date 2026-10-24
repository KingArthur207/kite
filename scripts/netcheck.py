"""Is my Kite setup wired? One command, PASS/FAIL per check, non-zero exit on any FAIL.

    python3 scripts/netcheck.py                    # earpiece at KITE_EARPIECE_HOST from .env
    python3 scripts/netcheck.py --host 169.254.12.34

Checks: this Mac's IPv4 addresses, the earpiece (WS 5011 state + a UDP 5010
heartbeat that must show up in net.udp), SpacetimeDB (identity + schema, no
writes), internet to Anthropic / Cartesia / Google, and which .env keys are set
(never their values). Stdlib only, so it runs on any teammate's laptop.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import ssl
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
REQUIRED_KEYS = ["ANTHROPIC_API_KEY", "CARTESIA_API_KEY", "GOOGLE_MAPS_API_KEY"]
OPTIONAL_KEYS = ["MOSS_PROJECT_ID", "MOSS_PROJECT_KEY"]
INTERNET = ["https://api.anthropic.com", "https://api.cartesia.ai", "https://routes.googleapis.com"]
TIMEOUT_S = 4.0
STATE_WAIT_S = 2.0          # state goes out at 5 Hz, so a couple of seconds is plenty

failures = 0


def report(ok: bool, what: str, hint: str = "") -> None:
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {what}" + ("" if ok or not hint else f"\n      fix: {hint}"))


def info(what: str) -> None:
    print(f"      {what}")


def load_env() -> dict[str, str]:
    """KEY=VALUE lines from the repo .env; real environment variables win, like python-dotenv."""
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
            if m and not line.lstrip().startswith("#"):
                env[m[1]] = m[2].strip().strip("'\"")
    return {**env, **os.environ}


def interfaces() -> None:
    print("== this Mac")
    out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    iface, found_bridge = None, False
    for line in out.splitlines():
        if line and not line[0].isspace():
            iface = line.split(":")[0]
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", line)
        if m and m[1] != "127.0.0.1":
            ip = m[1]
            link_local = ip.startswith("169.254.")
            found_bridge |= link_local or iface == "bridge0"
            mark = "  <- Thunderbolt bridge, give this to the drone laptop" if link_local or iface == "bridge0" else ""
            info(f"{iface:8} {ip}{mark}")
    if not found_bridge:
        info("no bridge0 / 169.254.x.x address (fine on one laptop; for two, plug in the Thunderbolt cable)")


def ws_connect(host: str, port: int) -> socket.socket:
    s = socket.create_connection((host, port), timeout=TIMEOUT_S)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = s.recv(1)
        if not chunk:
            raise ConnectionError("closed during handshake")
        head += chunk
    if b" 101 " not in head.split(b"\r\n")[0]:
        raise ConnectionError(head.split(b"\r\n")[0].decode(errors="replace"))
    return s


def recv_exact(s: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("earpiece closed the WebSocket")
        buf += chunk
    return buf


def ws_text(s: socket.socket) -> str:
    """Next text frame from the server (servers never mask; pings and fragments are skipped)."""
    while True:
        b0, b1 = recv_exact(s, 2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", recv_exact(s, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", recv_exact(s, 8))[0]
        data = recv_exact(s, n)
        if b0 & 0x0F == 1 and b0 & 0x80:
            return data.decode()


def next_state(s: socket.socket) -> dict:
    while True:
        m = json.loads(ws_text(s))
        if m.get("type") == "state":
            return m


def earpiece(host: str, udp_port: int, ws_port: int) -> None:
    print(f"== earpiece at {host}")
    try:
        s = ws_connect(host, ws_port)
        s.settimeout(STATE_WAIT_S)
        first = next_state(s)
    except Exception as e:
        report(False, f"WebSocket ws://{host}:{ws_port} ({type(e).__name__}: {e})",
               "start it on the wearable laptop: cd earpiece && .venv/bin/python -m earpiece run --bind 0.0.0.0 "
               "(from another laptop: right IP in KITE_EARPIECE_HOST? firewall allowed Python?)")
        report(False, f"UDP {host}:{udp_port}", "skipped, WebSocket check failed first")
        return
    net = first.get("net") or {}
    report(True, f"WebSocket ws://{host}:{ws_port}, got state (net.udp={net.get('udp')})")
    before = net.get("udp", 0)
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(
        b'{"type":"heartbeat","source":"netcheck"}', (host, udp_port))
    deadline, after = time.monotonic() + STATE_WAIT_S, before
    try:
        while after <= before and time.monotonic() < deadline:
            after = (next_state(s).get("net") or {}).get("udp", 0)
    except (OSError, ValueError):
        pass
    s.close()
    report(after > before, f"UDP {host}:{udp_port} heartbeat counted (net.udp {before} -> {after})",
           "UDP is blocked but TCP isn't: allow Python in System Settings > Network > Firewall, "
           "or check nothing else is bound to 5010")


def tls_context() -> ssl.SSLContext:
    """python.org Pythons ship with no CA certs until "Install Certificates" runs; use the macOS roots."""
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats()["x509_ca"]:
        roots = subprocess.run(["security", "find-certificate", "-a", "-p",
                                "/System/Library/Keychains/SystemRootCertificates.keychain"],
                               capture_output=True, text=True).stdout
        if roots:
            ctx.load_verify_locations(cadata=roots)
    return ctx


TLS = tls_context()


def http(method: str, url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "kite-netcheck"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=TLS) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:      # any HTTP status means DNS + TLS worked
        return e.code, e.read()


def spacetime(host: str, db: str) -> None:
    host = host.rstrip("/")
    print(f"== SpacetimeDB {host} / {db}")
    local = "127.0.0.1" in host or "localhost" in host
    hint = ("run scripts/offline.sh (starts the local server and publishes the module)" if local else
            "no internet? join the venue Wi-Fi, or switch to the offline fallback: scripts/offline.sh")
    try:
        code, _ = http("POST", f"{host}/v1/identity")
        report(code == 200, f"POST /v1/identity -> {code}", hint)
    except Exception as e:
        report(False, f"POST /v1/identity ({type(e).__name__}: {e})", hint)
        return
    try:
        code, body = http("GET", f"{host}/v1/database/{db}/schema?version=9")
        reducers = [r["name"] for r in json.loads(body).get("reducers", [])] if code == 200 else []
        report("push" in reducers, f"database {db}: {code}, reducers {reducers}",
               "module not published here: cd dashboard && npm run spacetime:publish:local" if local else
               "check SPACETIME_DB in .env (kite-sync-pl6rp)")
    except Exception as e:
        report(False, f"database schema ({type(e).__name__}: {e})", hint)


def internet() -> None:
    print("== internet")
    for url in INTERNET:
        try:
            code, _ = http("HEAD", url)
            report(True, f"{url} reachable (HTTP {code})")
        except Exception as e:
            report(False, f"{url} ({type(e).__name__}: {getattr(e, 'reason', e)})",
                   "no internet: join the venue Wi-Fi or a phone hotspot (the earpiece still runs cached phrases + say)")


def keys(env: dict[str, str]) -> None:
    print(f"== keys in {ENV_PATH}" + ("" if ENV_PATH.exists() else " (missing)"))
    for k in REQUIRED_KEYS:
        report(bool(env.get(k)), f"{k} {'set' if env.get(k) else 'missing'}",
               "cp .env.example .env and paste the key from the team vault")
    for k in OPTIONAL_KEYS:
        info(f"{k} {'set' if env.get(k) else 'not set (optional)'}")


def main(argv=None) -> int:
    env = load_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=env.get("KITE_EARPIECE_HOST", "127.0.0.1"), help="earpiece address")
    p.add_argument("--spacetime", default=env.get("SPACETIME_HOST", "https://maincloud.spacetimedb.com"))
    p.add_argument("--db", default=env.get("SPACETIME_DB", "kite-sync-pl6rp"))
    p.add_argument("--udp-port", type=int, default=int(env.get("KITE_EARPIECE_PORT", "5010")))
    p.add_argument("--ws-port", type=int, default=5011)
    a = p.parse_args(argv)

    interfaces()
    earpiece(a.host, a.udp_port, a.ws_port)
    spacetime(a.spacetime, a.db)
    internet()
    keys(env)
    print(f"\n{'all good' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
