#!/usr/bin/env python3
"""Kite link, computer side: talk to the UNO Q from Windows (or macOS, Linux). Standard library only.

    python kite_link.py token <token>     once: save the token install.sh printed on the board
    python kite_link.py doctor            check every step of the connection and say what is wrong
    python kite_link.py send "hello"      text to the board; it logs it and replies
    python kite_link.py hello | ping | status
    python kite_link.py listen            the board's recurring hello, live
    python kite_link.py watch             the board's camera in your browser
    python kite_link.py watch --save frames --seconds 10      or save the frames as JPEG files

Finding the board, in order (no IP to type):
  1. its Wi-Fi IP, read from the board's signed heartbeat in Kite's SpacetimeDB (direct, e.g. over eduroam);
  2. its Tailscale address or MagicDNS name `kite-unoq` (works across networks and through eduroam);
  3. for text only (send, hello, status, ping): through SpacetimeDB itself, slower but it always works.
Video needs 1 or 2. --board HOST[:PORT] skips discovery.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kite_common import (  # noqa: E402
    DEFAULT_PORT, DEFAULT_SOURCE, SPACETIME_DB, SPACETIME_HOST, TAILSCALE_NAME, Spacetime, check_source,
    clean_text, link_source, reply_source, row_time, sign, verify,
)

TOKEN_FILE = Path.home() / ".kite_unoq_token"
STALE_S = 30            # a heartbeat older than this means the board is off or offline
CLOUD_WAIT_S = 20
ISOLATION = ("eduroam blocks device-to-device here (or the two are on different networks). "
             "Set up Tailscale on both (README, plan B) or put both on a phone hotspot. "
             "Text still works through the internet; video needs a direct path.")


# ------------------------------------------------------------------------------------------- token

def read_token(path: Path = TOKEN_FILE) -> str:
    """The saved token. Tolerates a file written by PowerShell's `>` (UTF-16 with a BOM)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16").strip()
    return raw.decode("utf-8-sig", errors="replace").strip()


def save_token(value: str, path: Path = TOKEN_FILE) -> None:
    value = value.strip()
    if len(value) < 16:
        raise SystemExit("that doesn't look like the token (16+ characters); on the board: cat /etc/kite/token")
    path.write_text(value + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


# ------------------------------------------------------------------------------------------ direct

class Direct:
    """The board's HTTP API at host:port."""

    def __init__(self, host: str, port: int, token: str, label: str = "direct", timeout: float = 10.0) -> None:
        self.host, self.port, self.token, self.label, self.timeout = host, port, token, label, timeout

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"

    def url(self, path: str) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}{path}"

    def open(self, path: str, body: dict | None = None, timeout: float | None = None, auth: bool = True):
        headers = {"Authorization": f"Bearer {self.token}"} if auth and self.token else {}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url(path), data=data, headers=headers, method="POST" if data else "GET")
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)  # http to the board  # nosec B310

    def call(self, path: str, body: dict | None = None, auth: bool = True) -> tuple[int, dict]:
        """(HTTP status, JSON reply). Raises OSError when the board can't be reached at all."""
        try:
            with self.open(path, body, auth=auth) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {"error": e.reason}


def tcp_ok(host: str, port: int, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "connected"
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"


def reach(cands: list[tuple[str, str, int]], timeout: float = 2.0) -> list[tuple[str, str, int, bool, str]]:
    """Try every candidate at once (name lookups can hang); results in the candidates' order."""
    results: list = [None] * len(cands)

    def one(i: int, c: tuple[str, str, int]) -> None:
        results[i] = (*c, *tcp_ok(c[1], c[2], timeout))

    threads = [threading.Thread(target=one, args=(i, c), daemon=True) for i, c in enumerate(cands)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout + 1.5
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    return [r if r else (*cands[i], False, "timed out (name lookup)") for i, r in enumerate(results)]


def parse_board(value: str, port: int) -> tuple[str, int]:
    if value.startswith("[") and "]" in value:                 # [ipv6]:port
        host, _, rest = value[1:].partition("]")
        return host, int(rest[1:]) if rest.startswith(":") else port
    if value.count(":") == 1:
        host, p = value.split(":")
        return host, int(p)
    return value, port


# ------------------------------------------------------------------------------------------- cloud

def find_heartbeat(cloud: Spacetime, source: str, token: str, now: float | None = None) -> dict:
    """The board's newest heartbeat signed with our token.
    {"found": bool, "valid": bool, "hb": payload, "age_s": seconds, "unsigned": count of rows ignored}"""
    now = time.time() if now is None else now

    def parsed(row):
        try:
            d = json.loads(row["payload"])
        except (ValueError, KeyError, TypeError):
            return None
        return d if isinstance(d, dict) else None

    row = cloud.latest("heartbeat", source)
    d = parsed(row) if row else None
    if d is not None and verify(token, d):
        return {"found": True, "valid": True, "hb": d, "age_s": max(0.0, now - (row_time(row.get("t")) or 0)), "unsigned": 0}
    # the newest row isn't ours (or isn't signed): look back through the log for the newest that is
    rows = cloud.rows("heartbeat", source)
    unsigned = 0
    for r in reversed(rows):
        d = parsed(r)
        if d is not None and verify(token, d):
            return {"found": True, "valid": True, "hb": d, "age_s": max(0.0, now - (row_time(r.get("t")) or 0)),
                    "unsigned": unsigned}
        unsigned += 1
    return {"found": bool(rows or row), "valid": False, "hb": None, "age_s": None, "unsigned": unsigned}


def cloud_command(cloud: Spacetime, source: str, token: str, cmd: str, text: str | None = None,
                  wait_s: float = CLOUD_WAIT_S, poll_s: float = 1.0) -> dict:
    """Send one signed command through SpacetimeDB and wait for the board's signed reply."""
    after = cloud.last_id("control", reply_source(source))
    cid = secrets.token_hex(8)
    cloud.push("control", link_source(source),
               sign(token, {"to": source, "cmd": cmd, "text": text, "id": cid, "ts": int(time.time() * 1000)}))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        for r in cloud.rows("control", reply_source(source), after):
            after = max(after, int(r["id"]))
            try:
                d = json.loads(r["payload"])
            except (ValueError, TypeError):
                continue
            if isinstance(d, dict) and d.get("reply_to") == cid and verify(token, d):
                return d
    raise TimeoutError(f"no reply through SpacetimeDB in {wait_s:.0f} s (is the board on and online?)")


# --------------------------------------------------------------------------------------------- link

class Link:
    """Finds the board and runs commands on it, falling back as the module docstring says."""

    def __init__(self, a: argparse.Namespace, out=print) -> None:
        self.a, self.out = a, out
        self.source = check_source(a.source)
        self.token = a.token or os.environ.get("KITE_TOKEN", "") or read_token(Path(a.token_file))
        self.cloud = None if a.no_cloud else Spacetime(a.st_host, a.db)
        self._hb: dict | None = None
        self.tried: list = []

    def need_token(self) -> str:
        if not self.token:
            raise SystemExit("no token on this computer yet. The board printed it at install; save it with:\n"
                             "    python kite_link.py token <token>\n(or on the board: cat /etc/kite/token)")
        return self.token

    def heartbeat(self) -> dict | None:
        if self._hb is None and self.cloud and self.token:
            try:
                self._hb = find_heartbeat(self.cloud, self.source, self.token)
            except (OSError, ValueError, KeyError) as e:
                self._hb = {"found": False, "valid": False, "hb": None, "age_s": None, "error": str(e)}
        return self._hb

    def candidates(self) -> list[tuple[str, str, int]]:
        if self.a.board:
            host, port = parse_board(self.a.board, self.a.port)
            return [("--board", host, port)]
        out = []
        info = self.heartbeat()
        hb = info["hb"] if info and info.get("valid") else None
        port = int(hb.get("port") or self.a.port) if hb else self.a.port
        if hb:
            out += [("direct", ip, port) for ip in hb.get("ips") or [] if isinstance(ip, str)]
            if isinstance(hb.get("tailscale"), str):
                out.append(("tailscale", hb["tailscale"], port))
        out.append(("tailscale name", TAILSCALE_NAME, port))
        seen, uniq = set(), []
        for c in out:
            if (c[1], c[2]) not in seen:
                seen.add((c[1], c[2]))
                uniq.append(c)
        return uniq

    def direct(self) -> Direct | None:
        if self.a.cloud:
            return None
        self.tried = reach(self.candidates(), timeout=self.a.timeout)
        for label, host, port, ok, _ in self.tried:
            if ok:
                return Direct(host, port, self.token, label)
        return None

    def text_command(self, cmd: str, text: str | None = None) -> tuple[str, dict, float]:
        """(path used, reply, seconds). Direct first, then SpacetimeDB."""
        self.need_token()
        d = self.direct()
        if d:
            t0 = time.perf_counter()
            try:
                if cmd == "send":
                    code, reply = d.call("/send", {"text": text})
                else:
                    code, reply = d.call(f"/{cmd}")
                if code == 401:
                    raise SystemExit(f"the board at {d} rejected this computer's token (401): "
                                     "save it again with: python kite_link.py token <token>")
                if code != 200:
                    raise SystemExit(f"the board said {code}: {reply.get('error')}")
                return f"{d.label} {d}", reply, time.perf_counter() - t0
            except OSError as e:
                self.out(f"(direct path to {d} failed: {e}; trying the internet)")
        if not self.cloud:
            raise SystemExit("can't reach the board directly and --no-cloud is set. " + ISOLATION)
        t0 = time.perf_counter()
        reply = cloud_command(self.cloud, self.source, self.token, cmd, text)
        if not reply.get("ok", True):
            raise SystemExit(f"the board said: {reply.get('error')}")
        return "internet (SpacetimeDB)", reply, time.perf_counter() - t0


# ---------------------------------------------------------------------------------------- streams

def iter_mjpeg(stream, boundary: str = "kiteframe"):
    """JPEG frames from a multipart/x-mixed-replace stream (file-like with readline/read)."""
    marker = b"--" + boundary.encode()
    while True:
        line = stream.readline()
        if not line:
            return
        if line.strip() != marker:
            continue
        length = None
        while True:
            h = stream.readline()
            if not h:
                return
            if h in (b"\r\n", b"\n"):
                break
            k, _, v = h.decode("latin-1").partition(":")
            if k.strip().lower() == "content-length":
                length = int(v.strip())
        if length is None:                 # no length: read up to the JPEG end marker
            buf = b""
            while not buf.endswith(b"\xff\xd9"):
                ch = stream.read(1)
                if not ch:
                    return
                buf += ch
            yield buf
            continue
        data = stream.read(length)
        if len(data) < length:
            return
        yield data


def is_jpeg(b: bytes) -> bool:
    return len(b) > 4 and b[:2] == b"\xff\xd8" and b[-2:] == b"\xff\xd9"


def boundary_of(content_type: str) -> str:
    for part in content_type.split(";"):
        k, _, v = part.strip().partition("=")
        if k.lower() == "boundary":
            return v.strip('"')
    return "kiteframe"


def iter_events(stream):
    """JSON objects from a Server-Sent Events stream."""
    for raw in stream:
        line = raw.decode("utf-8", errors="replace").strip()
        if line.startswith("data:"):
            try:
                yield json.loads(line[5:].strip())
            except ValueError:
                continue


# ------------------------------------------------------------------------------------------ commands

def stamp() -> str:
    return time.strftime("%H:%M:%S")


def cmd_send(link: Link, a) -> int:
    text = clean_text(" ".join(a.text))
    path, reply, dt = link.text_command("send", text)
    link.out(f"board replied via {path} in {dt * 1000:.0f} ms: {reply.get('reply')} (message {reply.get('n')})")
    return 0


def cmd_simple(link: Link, a) -> int:
    path, reply, dt = link.text_command(a.cmd)
    if a.cmd == "status":
        link.out(f"status via {path}:")
        link.out(json.dumps(reply.get("status"), indent=2))
    else:
        link.out(f"board replied via {path} in {dt * 1000:.0f} ms: {reply.get('reply')} (host {reply.get('host')})")
    return 0


def cmd_ping(link: Link, a) -> int:
    link.need_token()
    d = link.direct()
    if not d:
        path, _, dt = link.text_command("ping")
        link.out(f"pong via {path}: {dt * 1000:.0f} ms (no direct path: {ISOLATION})")
        return 0
    times = []
    for _ in range(a.count):
        t0 = time.perf_counter()
        code, _ = d.call("/ping")
        ms = (time.perf_counter() - t0) * 1000
        if code != 200:
            raise SystemExit(f"ping: the board said {code}")
        times.append(ms)
        link.out(f"pong from {d} ({d.label}): {ms:.1f} ms")
        time.sleep(0.2)
    link.out(f"round trip min {min(times):.1f} / avg {sum(times) / len(times):.1f} / max {max(times):.1f} ms")
    return 0


def cmd_listen(link: Link, a) -> int:
    """The recurring hello, live: straight from the board when possible, else through SpacetimeDB.
    Reconnects when the board restarts."""
    link.need_token()
    got, end = 0, time.monotonic() + a.seconds if a.seconds else None
    d = None if a.cloud else link.direct()
    if d:
        link.out(f"listening to {d} ({d.label}); Ctrl+C to stop")
        while True:
            try:
                with d.open("/events", timeout=15) as r:
                    for ev in iter_events(r):
                        if ev.get("kind") == "hello":
                            link.out(f"{stamp()}  direct   {ev.get('msg')} {ev.get('n')} from {ev.get('host')}")
                            got += 1
                        elif ev.get("kind") == "got":
                            link.out(f"{stamp()}  direct   board got message {ev.get('n')}: {ev.get('text')}")
                        if (a.count and got >= a.count) or (end and time.monotonic() > end):
                            return 0
                link.out(f"{stamp()}  ... the board closed the stream; reconnecting")
            except (OSError, ValueError) as e:
                if end and time.monotonic() > end:
                    return 0 if got else 1
                link.out(f"{stamp()}  ... lost the board ({type(e).__name__}: {e}); reconnecting")
                time.sleep(1.0)
    if not link.cloud:
        raise SystemExit("no direct path and --no-cloud is set. " + ISOLATION)
    link.out(f"listening through SpacetimeDB for '{link.source}' (no direct path); Ctrl+C to stop")
    last = link.cloud.last_id("heartbeat", link.source)
    while True:
        try:
            rows = link.cloud.rows("heartbeat", link.source, last)
        except OSError as e:
            link.out(f"{stamp()}  ... can't reach SpacetimeDB ({e}); retrying")
            time.sleep(3)
            continue
        for r in rows:
            last = max(last, int(r["id"]))
            try:
                m = json.loads(r["payload"])
            except ValueError:
                continue
            if verify(link.token, m):
                link.out(f"{stamp()}  internet {m.get('msg')} {m.get('n')} from {m.get('host')}")
                got += 1
        if (a.count and got >= a.count) or (end and time.monotonic() > end):
            return 0 if got else 1
        time.sleep(1.0)


def cmd_watch(link: Link, a) -> int:
    link.need_token()
    d = link.direct()
    if not d:
        link.out("can't reach the board directly, and video has no internet fallback: " + ISOLATION)
        for label, host, port, _, why in link.tried:
            link.out(f"  tried {label} {host}:{port}: {why}")
        return 2
    if not a.save:
        url = d.url("/view?token=" + urllib.parse.quote(link.token, safe=""))
        link.out(f"opening the board's camera from {d} ({d.label}) in your browser")
        if not webbrowser.open(url):
            link.out(f"open this yourself: {d.url('/view?token=<your token>')}")
        return 0
    folder = Path(a.save)
    folder.mkdir(parents=True, exist_ok=True)
    n, total, bad, reconnects = 0, 0, 0, 0
    t0 = time.monotonic()
    end = t0 + (a.seconds or 1e9)
    first_t = last_t = None
    camera = "?"
    while time.monotonic() < end and (not a.frames or n < a.frames):
        try:
            with d.open("/video.mjpg", timeout=10) as r:
                if r.status != 200:
                    raise SystemExit(f"video: the board said {r.status}")
                camera = r.headers.get("X-Kite-Camera", camera)
                for jpeg in iter_mjpeg(r, boundary_of(r.headers.get("Content-Type", ""))):
                    now = time.monotonic()
                    first_t = first_t or now
                    last_t = now
                    n += 1
                    total += len(jpeg)
                    bad += 0 if is_jpeg(jpeg) else 1
                    (folder / f"frame_{n:05d}.jpg").write_bytes(jpeg)
                    if now >= end or (a.frames and n >= a.frames):
                        break
        except urllib.error.HTTPError as e:
            raise SystemExit(f"video: the board said {e.code} ({'token mismatch' if e.code == 401 else e.reason})")
        except OSError as e:
            if time.monotonic() >= end:
                break
            reconnects += 1
            link.out(f"{stamp()}  ... video dropped ({type(e).__name__}); reconnecting")
            time.sleep(1.0)
    span = (last_t - first_t) if first_t and last_t and n > 1 else 0
    fps = (n - 1) / span if span else 0.0
    link.out(f"saved {n} frames to {folder} from {d} ({d.label}, {camera}): {fps:.1f} fps, "
             f"{total / max(n, 1) / 1024:.0f} KB each, {bad} not JPEG, {reconnects} reconnects")
    return 0 if n and not bad else 1


def cmd_where(link: Link, a) -> int:
    link.need_token()
    info = link.heartbeat()
    if info and info.get("valid"):
        hb = info["hb"]
        link.out(f"heartbeat: {hb.get('host')} {info['age_s']:.0f} s ago, IPs {hb.get('ips')}, "
                 f"tailscale {hb.get('tailscale')}, port {hb.get('port')}")
    d = link.direct()
    for label, host, port, ok, why in link.tried:
        link.out(f"{'reachable' if ok else 'unreachable'}  {label} {host}:{port}  {'' if ok else why}")
    link.out(f"board: {d} ({d.label})" if d else "board: no direct path")
    return 0 if d else 2


def cmd_token(link: Link, a) -> int:
    save_token(a.value, Path(a.token_file))
    link.out(f"saved the token to {a.token_file}; now run: python kite_link.py doctor")
    return 0


# ----------------------------------------------------------------------------------------- doctor

class Report:
    def __init__(self, out) -> None:
        self.out, self.fails, self.warns = out, 0, 0

    def __call__(self, status: str, step: str, detail: str, means: str = "") -> None:
        self.out(f"{status:4s}  {step}: {detail}")
        if means:
            self.out(f"      means: {means}")
        self.fails += status == "FAIL"
        self.warns += status == "WARN"


def cmd_doctor(link: Link, a) -> int:
    r = Report(link.out)
    link.out(f"kite_link doctor: board source '{link.source}'")
    if link.token:
        r("PASS", "token", f"found ({len(link.token)} characters)")
    else:
        r("FAIL", "token", "none on this computer",
          "save the one install.sh printed: python kite_link.py token <token> (on the board: cat /etc/kite/token)")

    # 1. discovery through SpacetimeDB
    if a.board:
        r("SKIP", "discovery", f"--board {a.board} given")
    elif not link.cloud:
        r("SKIP", "discovery", "--no-cloud given; only the Tailscale name will be tried")
    elif link.token:
        info = link.heartbeat()
        if info.get("error"):
            r("FAIL", "internet", f"can't reach SpacetimeDB ({info['error']})",
              "this computer has no internet (or SpacetimeDB is down), so it can't look the board up; "
              "pass --board <ip> if you know it")
        elif not info["found"]:
            r("FAIL", "heartbeat", f"no heartbeat from '{link.source}' at all",
              "the board service isn't running or the board has never had internet. On the board: "
              "systemctl status kite-link, and check its Wi-Fi (nmcli dev status)")
        elif not info["valid"]:
            r("FAIL", "heartbeat", f"{info['unsigned']} heartbeats, none signed with your token",
              "token mismatch (or another board uses this source name): save the token again from "
              "the board's /etc/kite/token")
        else:
            hb, age = info["hb"], info["age_s"]
            where = f"{hb.get('host')} at {', '.join(hb.get('ips') or []) or 'no LAN IP'} port {hb.get('port')}" + \
                (f", tailscale {hb['tailscale']}" if hb.get("tailscale") else "")
            if age > STALE_S:
                r("FAIL", "heartbeat", f"last one {age:.0f} s ago from {where}",
                  "the board is off, asleep or lost its internet; the addresses may be out of date")
            else:
                r("PASS", "heartbeat", f"{age:.0f} s ago from {where}")

    # 2. the direct path
    d = link.direct()
    for label, host, port, ok, why in link.tried:
        if ok:
            r("PASS", "reach", f"TCP {host}:{port} ({label})")
        else:
            r("FAIL" if label in ("direct", "--board") else "WARN", "reach", f"TCP {host}:{port} ({label}): {why}",
              ISOLATION if label in ("direct", "--board") else "Tailscale isn't set up or isn't connected "
              "on one of the two (fine if the direct path works)")
    if d is None:
        r("FAIL", "direct path", "the board isn't reachable on any address", ISOLATION)
    else:
        try:
            code, body = d.call("/health", auth=False)
            if code == 200 and body.get("kite") == "unoq-link":
                r("PASS", "health", f"{d} is the Kite board {body.get('host')}")
            else:
                r("FAIL", "health", f"{d} answered {code} but isn't the Kite board",
                  "something else listens on that address and port (wrong IP, or another device took it)")
            code, body = d.call("/status")
            if code == 401:
                r("FAIL", "token check", f"{d} rejected this computer's token (401)",
                  "token mismatch: on the board run `cat /etc/kite/token`, here `python kite_link.py token <it>`")
            elif code == 200:
                st = body.get("status", {})
                w = st.get("wifi") or {}
                r("PASS", "token check", f"status ok: up {st.get('uptime_s')} s, Wi-Fi {w.get('ssid')} "
                  f"{w.get('signal_dbm')} dBm, CPU {st.get('cpu_temp_c')} C")
                t0 = time.perf_counter()
                code, body = d.call("/send", {"text": "doctor check"})
                ms = (time.perf_counter() - t0) * 1000
                if code == 200:
                    r("PASS", "text (direct)", f"board replied in {ms:.0f} ms: {body.get('reply')}")
                else:
                    r("FAIL", "text (direct)", f"board said {code}: {body.get('error')}")
                doctor_video(d, r, st.get("camera") or {})
            else:
                r("FAIL", "token check", f"{d} answered {code}: {body.get('error')}")
        except OSError as e:
            r("FAIL", "health", f"{d} accepted TCP but HTTP failed ({type(e).__name__}: {e})",
              "the network lets the connection open but drops it (a firewall or captive portal in between)")

    # 3. the internet path for text
    if link.cloud and link.token and not a.skip_cloud:
        try:
            t0 = time.perf_counter()
            reply = cloud_command(link.cloud, link.source, link.token, "ping", wait_s=a.cloud_wait)
            r("PASS", "text (internet)", f"board answered through SpacetimeDB in {time.perf_counter() - t0:.1f} s"
              if reply.get("ok") else f"board answered with an error: {reply.get('error')}")
        except TimeoutError as e:
            r("FAIL", "text (internet)", str(e), "the board isn't reading SpacetimeDB: it's off, offline, or "
              "its token differs from this computer's")
        except (OSError, ValueError) as e:
            r("FAIL", "text (internet)", f"SpacetimeDB error: {e}", "this computer can't reach SpacetimeDB")
    link.out("")
    if r.fails:
        link.out(f"{r.fails} check(s) failed, {r.warns} warning(s): see 'means' above")
        return 1
    link.out(f"all checks passed ({r.warns} warning(s))")
    return 0


def doctor_video(d: Direct, r: Report, cam: dict) -> None:
    try:
        with d.open("/snapshot.jpg", timeout=8) as resp:
            jpeg = resp.read()
    except (OSError, urllib.error.HTTPError) as e:
        r("FAIL", "snapshot", f"no frame ({e})", "the board's camera pipeline is stuck; restart: "
          "sudo systemctl restart kite-link")
        return
    if not is_jpeg(jpeg):
        r("FAIL", "snapshot", f"{len(jpeg)} bytes that aren't a JPEG")
        return
    source = cam.get("source", "?")
    if source == "synthetic":
        r("WARN", "camera", f"no camera: the board streams its test pattern ({cam.get('reason')})",
          "plug the USB camera in through a powered USB-C hub, then on the board: sudo systemctl restart kite-link")
    else:
        r("PASS", "camera", f"{source}")
    r("PASS", "snapshot", f"{len(jpeg) // 1024} KB JPEG")
    n, t_first, t_last = 0, None, None
    try:
        with d.open("/video.mjpg", timeout=8) as resp:
            end = time.monotonic() + 3.0
            for frame in iter_mjpeg(resp, boundary_of(resp.headers.get("Content-Type", ""))):
                if not is_jpeg(frame):
                    r("FAIL", "video", "a frame that isn't a JPEG")
                    return
                n += 1
                t_last = time.monotonic()
                t_first = t_first or t_last
                if t_last > end:
                    break
    except OSError as e:
        r("FAIL", "video", f"stream broke after {n} frames ({e})")
        return
    fps = (n - 1) / (t_last - t_first) if n > 1 and t_last > t_first else 0.0
    want = float(cam.get("fps") or 10)
    if fps >= 0.5 * want:
        r("PASS", "video", f"{n} frames, {fps:.1f} fps (board set to {want:g})")
    else:
        r("WARN" if n else "FAIL", "video", f"{n} frames, {fps:.1f} fps (board set to {want:g})",
          "the link is slow (a relayed Tailscale path, or weak Wi-Fi): lower KITE_VIDEO_FPS or KITE_VIDEO_SIZE")


# ------------------------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", help="the board's address, HOST[:PORT]; skips discovery")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--token", default="", help="the board's token (default: saved with `token`, or KITE_TOKEN)")
    ap.add_argument("--token-file", default=str(TOKEN_FILE))
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="the board's name in SpacetimeDB (default unoq)")
    ap.add_argument("--st-host", default=os.environ.get("SPACETIME_HOST", SPACETIME_HOST))
    ap.add_argument("--db", default=os.environ.get("SPACETIME_DB", SPACETIME_DB))
    ap.add_argument("--no-cloud", action="store_true", help="never use SpacetimeDB")
    ap.add_argument("--cloud", action="store_true", help="text through SpacetimeDB even if direct works")
    ap.add_argument("--timeout", type=float, default=2.0, help="seconds to wait for a TCP connection")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("send", help="send text; the board logs it and replies")
    p.add_argument("text", nargs="+")
    for name in ("hello", "status"):
        sub.add_parser(name)
    p = sub.add_parser("ping")
    p.add_argument("-n", "--count", type=int, default=5)
    p = sub.add_parser("listen", help="print the board's recurring hello")
    p.add_argument("--count", type=int, default=0, help="stop after this many")
    p.add_argument("--seconds", type=float, default=0)
    p = sub.add_parser("watch", help="the camera in your browser, or --save frames")
    p.add_argument("--save", help="folder to save JPEG frames into instead of opening the browser")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--frames", type=int, default=0)
    p = sub.add_parser("doctor", help="check every step of the connection")
    p.add_argument("--skip-cloud", action="store_true", help="don't test the text path through SpacetimeDB")
    p.add_argument("--cloud-wait", type=float, default=CLOUD_WAIT_S)
    sub.add_parser("where", help="find the board and show how it's reachable")
    p = sub.add_parser("token", help="save the board's token on this computer")
    p.add_argument("value")
    return ap


COMMANDS = {"send": cmd_send, "hello": cmd_simple, "status": cmd_simple, "ping": cmd_ping, "listen": cmd_listen,
            "watch": cmd_watch, "doctor": cmd_doctor, "where": cmd_where, "token": cmd_token}


def main(argv=None, out=print) -> int:
    for s in (sys.stdout, sys.stderr):             # a Windows console can't print every character
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="replace")
    a = build_parser().parse_args(argv)
    try:
        return COMMANDS[a.cmd](Link(a, out), a)
    except KeyboardInterrupt:
        return 130
    except (ValueError, TimeoutError) as e:
        out(f"error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
