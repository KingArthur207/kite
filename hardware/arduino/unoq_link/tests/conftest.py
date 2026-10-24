"""Fixtures for the Kite link tests: a fake SpacetimeDB over real HTTP, and a board on a real socket.

Run: uvx --with pillow pytest -q hardware/arduino/unoq_link/tests   (or scripts/check.sh hardware)
"""

from __future__ import annotations

import json
import re
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import board as board_mod  # noqa: E402

TYPES = {"heartbeat", "control", "say", "status", "output"}
IDENT = re.compile(r"^[\w.:-]{1,64}$")
EVENT_COLS = ["id", "type", "source", "priority", "payload", "t"]
LATEST_COLS = ["key", "type", "source", "priority", "payload", "t"]


class FakeSpacetime:
    """The parts of SpacetimeDB's HTTP API Kite uses, in memory: identity, the push reducer, and the
    exact SQL shapes kite_common sends."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.latest: dict[str, dict] = {}
        self.next_id = 1
        self.lock = threading.Lock()
        self.calls = 0
        self.down = False
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):  # noqa: N802
                fake.calls += 1
                if fake.down:
                    self.send_response(503)
                    self.end_headers()
                    return
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path == "/v1/identity":
                    return self._send(200, {"token": "fake-" + secrets.token_hex(4)})
                if not self.headers.get("Authorization", "").startswith("Bearer fake-"):
                    return self._send(401, {"error": "no identity"})
                m = re.match(r"/v1/database/([\w-]+)/(call/push|sql)$", self.path)
                if not m:
                    return self._send(404, {})
                if m.group(2) == "call/push":
                    a = json.loads(body)
                    if a["type"] not in TYPES or not IDENT.match(a["source"]):
                        return self._send(400, {"error": "rejected"})
                    json.loads(a["payload"])
                    fake.insert(a["type"], a["source"], a["payload"])
                    return self._send(200, {})
                return self._send(200, fake.sql(body.decode()))

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def insert(self, type_: str, source: str, payload, t: float | None = None, latest: bool = True) -> int:
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        with self.lock:
            row = {"id": self.next_id, "type": type_, "source": source, "priority": "system",
                   "payload": payload, "t": [int((time.time() if t is None else t) * 1e6)]}
            self.next_id += 1
            self.events.append(row)
            if latest:
                self.latest[f"{type_}:{source}"] = {"key": f"{type_}:{source}", **row}
            return row["id"]

    def sql(self, q: str) -> list:
        m = re.fullmatch(r"SELECT \* FROM event WHERE type = '([\w.:-]+)' AND source = '([\w.:-]+)'(?: AND id > (\d+))?", q)
        if m:
            after = int(m.group(3)) if m.group(3) else -1
            with self.lock:
                rows = [r for r in self.events if r["type"] == m.group(1) and r["source"] == m.group(2) and r["id"] > after]
            return [{"schema": {"elements": [{"name": {"some": c}} for c in EVENT_COLS]},
                     "rows": [[r[c] for c in EVENT_COLS] for r in rows]}]
        m = re.fullmatch(r"SELECT \* FROM latest WHERE key = '([\w.:-]+)'", q)
        if m:
            r = self.latest.get(m.group(1))
            return [{"schema": {"elements": [{"name": {"some": c}} for c in LATEST_COLS]},
                     "rows": [[r[c] for c in LATEST_COLS]] if r else []}]
        raise AssertionError(f"unexpected SQL: {q}")

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def spacetime():
    st = FakeSpacetime()
    yield st
    st.close()


@pytest.fixture
def token() -> str:
    return secrets.token_hex(16)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Running:
    """A board.py instance on 127.0.0.1 with the synthetic camera."""

    def __init__(self, token: str, st: FakeSpacetime | None, port: int = 0, source: str = "unoq-test", **env) -> None:
        e = {"KITE_TOKEN": token, "KITE_PORT": str(port), "KITE_BIND": "127.0.0.1", "KITE_SOURCE": source,
             "KITE_CAMERA": "synthetic", "KITE_VIDEO_FPS": "10", "KITE_ADVERTISE": "127.0.0.1",
             "KITE_HELLO_EVERY_S": "0.3", "KITE_CLOUD": "1" if st else "0",
             "SPACETIME_HOST": st.url if st else "http://127.0.0.1:9", "KITE_CLOUD_EVERY_S": "0.3",
             "KITE_CLOUD_POLL_S": "1", **env}
        self.cfg = board_mod.Config(e)
        self.board = board_mod.Board(self.cfg, run=lambda cmd: None)
        self.srv = board_mod.start(self.board)
        self.port = self.cfg.port

    def stop(self) -> None:
        board_mod.shutdown(self.board, self.srv)


@pytest.fixture
def make_board():
    boards = []

    def make(*a, **k) -> Running:
        b = Running(*a, **k)
        boards.append(b)
        return b

    yield make
    for b in boards:
        b.stop()
