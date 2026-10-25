"""kite.sources.mjpeg serves a live camera with no password: it stays on this computer unless asked, other web
pages can't read its frames (no CORS), and a DNS-rebinding page is refused by the Host check."""

from __future__ import annotations

import http.client
import threading
from http.server import ThreadingHTTPServer

import pytest

mjpeg = pytest.importorskip("kite.sources.mjpeg")


def test_listens_on_this_computer_only_by_default():
    a = mjpeg.parser().parse_args(["--source", "webcam:0"])
    assert a.host == "127.0.0.1"


@pytest.mark.parametrize("host,bound,ok", [
    ("127.0.0.1:8090", "127.0.0.1", True),
    ("localhost:8090", "127.0.0.1", True),
    ("[::1]:8090", "127.0.0.1", True),
    ("LOCALHOST", "127.0.0.1", True),
    ("evil.example:8090", "127.0.0.1", False),       # a page that rebound evil.example to 127.0.0.1
    ("127.0.0.1.evil.example", "127.0.0.1", False),
    ("", "127.0.0.1", False),
    (None, "127.0.0.1", False),
    ("192.168.1.20:8090", "0.0.0.0", True),          # shared on purpose with --host 0.0.0.0
])
def test_host_check(host, bound, ok):
    assert mjpeg.host_ok(host, bound) is ok


@pytest.fixture
def server():
    latest = mjpeg.Latest()
    latest.put(b"\xff\xd8frame\xff\xd9")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mjpeg.make_handler(latest, "127.0.0.1"))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], latest
    srv.shutdown()
    srv.server_close()


def get(port: int, host_header: str):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.putrequest("GET", "/video.mjpg", skip_host=True)
    c.putheader("Host", host_header)
    c.putheader("Origin", "https://evil.example")
    c.endheaders()
    return c, c.getresponse()


def test_stream_has_no_cors_header(server):
    port, latest = server
    threading.Timer(0.2, latest.put, args=(b"\xff\xd8next\xff\xd9",)).start()
    c, r = get(port, f"127.0.0.1:{port}")
    try:
        assert r.status == 200
        assert r.getheader("Access-Control-Allow-Origin") is None
        assert r.getheader("Content-Type").startswith("multipart/x-mixed-replace")
    finally:
        c.close()


def test_rebound_host_is_refused(server):
    port, _ = server
    c, r = get(port, f"evil.example:{port}")
    try:
        assert r.status == 403
    finally:
        c.close()
