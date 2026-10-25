"""The walk preview server listens on 127.0.0.1, which every web page can reach. It must not show the walker's
position to a DNS-rebinding page, spend Google Routes calls for another site, or choke on bad numbers."""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

import phone  # noqa: F401  (puts the earpiece package on sys.path)
from earpiece.route import Route, Step
from phone import walk_server

LAT, LNG = 37.8667, -122.2577


def _route(origin, destination, label=None, **_kw):
    a, b = origin, (origin[0] + 0.001, origin[1])
    return Route([Step("Head north", "DEPART", 111, a, b, [a, b])], 111, 90, b, label or destination)


@pytest.fixture
def server(monkeypatch):
    calls = []

    def compute(origin, destination, label=None, **kw):
        calls.append(destination)
        return _route(origin, destination, label)

    monkeypatch.setattr(walk_server, "SESSION", walk_server.WalkSession(compute=compute))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), walk_server.Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], calls
    srv.shutdown()
    srv.server_close()


def request(port, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.putrequest(method, path, skip_host=True)
    c.putheader("Host", host or f"127.0.0.1:{port}")
    raw = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    for k, v in hdrs.items():
        if v is not None:
            c.putheader(k, v)
    if method == "POST" and "Content-Length" not in hdrs:
        c.putheader("Content-Length", str(len(raw)))
    c.endheaders(raw or None)
    r = c.getresponse()
    out = r.status, json.loads(r.read() or b"{}")
    c.close()
    return out


START = {"lat": LAT, "lng": LNG, "destination": "Doe Library"}


def test_the_walk_page_works(server):
    port, calls = server
    code, body = request(port, "POST", "/api/start", START, {"Origin": f"http://127.0.0.1:{port}"})
    assert code == 200 and body["ok"] and calls == ["Doe Library"]
    code, body = request(port, "POST", "/api/tick", {"lat": LAT + 0.0001, "lng": LNG, "accuracy": 5, "heading": 0})
    assert code == 200 and body["navigating"]
    assert request(port, "GET", "/api/state", host=f"localhost:{port}")[0] == 200


def test_a_rebound_domain_cannot_read_the_position_or_the_map_token(server):
    port, _ = server
    request(port, "POST", "/api/start", START)
    for path in ("/api/state", "/api/config", "/walk.html"):
        code, body = request(port, "GET", path, host=f"evil.example:{port}")
        assert code == 403 and "lat" not in json.dumps(body) and "mapboxToken" not in body


def test_another_site_cannot_start_navigation(server):
    port, calls = server
    # a cross-site form post (text/plain needs no preflight) and a fetch from another origin
    assert request(port, "POST", "/api/start", START, {"Content-Type": "text/plain"})[0] == 403
    assert request(port, "POST", "/api/start", START, {"Origin": "https://evil.example"})[0] == 403
    assert request(port, "POST", "/api/start", START, host=f"evil.example:{port}")[0] == 403
    assert calls == []


@pytest.mark.parametrize("bad", [
    {"lat": "nan", "lng": LNG}, {"lat": LAT, "lng": "inf"}, {"lat": 91, "lng": LNG}, {"lat": LAT, "lng": 181},
    {"lat": True, "lng": LNG}, {"lat": None, "lng": LNG}, {"lng": LNG}, {"lat": [1], "lng": LNG},
    {"lat": LAT, "lng": LNG, "accuracy": "lots"}, {"lat": LAT, "lng": LNG, "heading": "1e999"},
    {"lat": LAT, "lng": LNG, "accuracy": -1},
])
def test_bad_numbers_are_a_400_not_a_crash(server, bad):
    port, calls = server
    assert request(port, "POST", "/api/start", {**bad, "destination": "Doe Library"})[0] == 400
    assert calls == []


def test_bodies_and_destinations_are_bounded(server):
    port, calls = server
    long = {**START, "destination": "x" * (walk_server.MAX_DESTINATION + 1)}
    assert request(port, "POST", "/api/start", long)[0] == 400
    assert request(port, "POST", "/api/start", b"[1, 2]")[0] == 400
    assert request(port, "POST", "/api/start", b"{}", {"Content-Length": "-1"})[0] == 400
    assert request(port, "POST", "/api/start", b"{}", {"Content-Length": str(walk_server.MAX_BODY + 1)})[0] == 413
    assert calls == []
