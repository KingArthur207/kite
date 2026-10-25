"""The walk preview server on 127.0.0.1: every request gets an answer, bad input never takes a thread down, and
positions round-trip. Google is never called: `compute` is a fake, and the real fetch is booby-trapped."""

import http.client
import json
import math
import socket
import threading

import pytest

import phone  # noqa: F401  (puts the earpiece package on sys.path)
from earpiece import route as route_mod
from earpiece.route import NoKey, Route, RouteDenied, RouteQuota, RouteUnavailable, Step
from phone import walk_server
from phone.walk_server import QUOTA_S, REROUTE_MIN_S, UNAVAILABLE_S, Handler, WalkSession, load_env

LAT, LNG = 37.8667, -122.2577
DEST = "Cloyne Court Hotel, 2600 Ridge Road, Berkeley, CA"


def _east(lat: float, lng: float, meters: float) -> float:
    return lng + meters / (111320 * math.cos(math.radians(lat)))


def _route_from(lat: float, lng: float) -> Route:
    a, b, c = (lat, lng), (lat + 0.0005, lng), (lat + 0.0010, lng)     # ~111 m due north
    steps = [Step("Head north on Bowditch Street", "DEPART", 56, a, b, [a, b]),
             Step("Continue north", "STRAIGHT", 55, b, c, [b, c])]
    return Route(steps, 111, 90, c, "Cloyne Court Hotel")


class FakeGoogle:
    """compute_route stand-in: answers from a script of routes / exceptions, records every call."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def __call__(self, origin, destination, label=None, **_kw):
        self.calls.append((origin, destination, label))
        step = self.script.pop(0) if self.script else "route"
        if isinstance(step, BaseException):
            raise step
        return _route_from(*origin)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this file may reach Google, whatever the code under test does."""
    def refuse(*_a, **_kw):
        raise AssertionError("test tried to call the Google Routes API")
    monkeypatch.setattr(route_mod, "fetch_routes", refuse)
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.setattr(Handler, "log_message", lambda self, fmt, *args: None)


@pytest.fixture
def google(monkeypatch):
    fake = FakeGoogle()
    monkeypatch.setattr(walk_server, "SESSION", WalkSession(compute=fake))
    return fake


@pytest.fixture
def server(google):
    httpd = walk_server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()
    t.join(timeout=5)
    assert not t.is_alive()


def call(port, method, path, body=None, headers=None):
    """One request on a fresh connection. A handler that never answers fails here (timeout), not by hanging."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        if headers is None:                      # what the walk page sends; main checks it
            headers = {"Content-Type": "application/json"} if method == "POST" else {}
        conn.request(method, path, body=body, headers=headers)
        r = conn.getresponse()
        data = r.read()
        ctype = r.getheader("Content-Type", "")
        return r.status, (json.loads(data) if ctype.startswith("application/json") else data), r
    finally:
        conn.close()


def raw_request(port, head: bytes) -> bytes:
    """Send raw request bytes, read until the server closes the connection."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(head)
        out = b""
        while chunk := s.recv(65536):
            out += chunk
        return out


# ---------------------------------------------------------------- pure helpers
def test_load_env_reads_only_the_given_root_and_never_overrides(tmp_path, monkeypatch):
    """Comments, blanks, quotes and lines without '=' are handled; a variable already set wins."""
    for k in ("KITE_FAKE_A", "KITE_FAKE_B", "KITE_FAKE_C", "KITE_FAKE_EMPTY"):
        monkeypatch.setenv(k, "")
        monkeypatch.delenv(k)                     # registers the key so monkeypatch removes it afterwards
    monkeypatch.setenv("KITE_FAKE_C", "already")
    (tmp_path / ".env").write_text('# comment\n\nKITE_FAKE_A = "one"\nKITE_FAKE_B=\'two\'\nnot a pair\n'
                                   "KITE_FAKE_C=three\nKITE_FAKE_EMPTY=\n", encoding="utf-8")
    monkeypatch.setattr(walk_server, "ROOT", tmp_path)
    load_env()
    import os
    assert (os.environ["KITE_FAKE_A"], os.environ["KITE_FAKE_B"], os.environ["KITE_FAKE_C"]) == ("one", "two",
                                                                                                "already")
    assert "KITE_FAKE_EMPTY" not in os.environ


def test_load_env_without_a_file_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(walk_server, "ROOT", tmp_path)
    load_env()                                    # no .env: nothing to read, nothing raised


def test_main_binds_localhost_and_honours_port(monkeypatch):
    seen = {}

    class FakeHTTPD:
        def __init__(self, addr, handler):
            seen["addr"], seen["handler"] = addr, handler

        def serve_forever(self):
            seen["served"] = True

    monkeypatch.setattr(walk_server, "ThreadingHTTPServer", FakeHTTPD)
    monkeypatch.setattr(walk_server, "load_env", lambda: None)
    monkeypatch.setattr("sys.argv", ["walk_server", "--port", "9999"])
    walk_server.main()
    assert seen == {"addr": ("127.0.0.1", 9999), "handler": Handler, "served": True}


# ---------------------------------------------------------------- GET
@pytest.mark.parametrize("path", ["/", "/walk.html", "/walk.html?debug=1"])
def test_page_is_served(server, path):
    status, body, r = call(server, "GET", path)
    assert status == 200 and r.getheader("Content-Type").startswith("text/html")
    assert body == walk_server.PAGE.read_bytes()
    assert r.getheader("Cache-Control") == "no-store"


def test_missing_page_is_a_json_404(server, monkeypatch, tmp_path):
    monkeypatch.setattr(walk_server, "PAGE", tmp_path / "walk.html")
    status, body, _ = call(server, "GET", "/")
    assert status == 404 and body["ok"] is False


def test_config_hands_over_only_the_mapbox_token(server, monkeypatch):
    monkeypatch.setenv("MAPBOX_TOKEN", "pk.fake-test-token")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-google-key")
    status, body, _ = call(server, "GET", "/api/config")
    assert status == 200 and body == {"mapboxToken": "pk.fake-test-token"}
    monkeypatch.delenv("MAPBOX_TOKEN")
    assert call(server, "GET", "/api/config")[1] == {"mapboxToken": ""}


def test_state_before_start(server):
    status, body, _ = call(server, "GET", "/api/state")
    assert status == 200 and body == {"ok": False, "error": "Navigation has not started.", "navigating": False,
                                      "google_calls": 0}


@pytest.mark.parametrize("method, path", [("GET", "/nope"), ("GET", "/api/start"), ("POST", "/"),
                                          ("POST", "/api/state"), ("POST", "/api/nope")])
def test_unknown_paths_are_json_404s(server, method, path):
    status, body, _ = call(server, method, path, body=b"{}" if method == "POST" else None)
    assert status == 404 and body == {"ok": False, "error": "Not found."}


# ---------------------------------------------------------------- POST: the happy path
def test_start_tick_state_round_trip_positions(server, google):
    status, view, _ = call(server, "POST", "/api/start", {"lat": LAT, "lng": LNG, "destination": f"  {DEST} ",
                                                          "accuracy": 5, "heading": 0})
    assert status == 200 and view["ok"] is True and view["navigating"] is True
    assert google.calls == [((LAT, LNG), DEST, "Cloyne Court Hotel")]
    assert view["label"] == "Cloyne Court Hotel" and view["google_calls"] == 1
    assert view["raw"] == [LAT, LNG] and view["accuracy_m"] == 5.0
    assert view["destination"] == [round(LAT + 0.0010, 6), LNG]
    assert view["distance_m"] == 111 and view["duration_s"] == 90

    lat2 = LAT + 0.00015
    status, view, _ = call(server, "POST", "/api/tick", {"lat": lat2, "lng": LNG, "accuracy": 4.0})
    assert status == 200 and view["raw"] == [round(lat2, 6), LNG]
    status, state, _ = call(server, "GET", "/api/state")
    assert state["raw"] == view["raw"] and state["google_calls"] == 1 and len(google.calls) == 1


def test_tick_before_start_is_answered_not_crashed(server):
    status, body, _ = call(server, "POST", "/api/tick", {"lat": LAT, "lng": LNG})
    assert status == 200 and body["ok"] is False and body["error"] == "Navigation has not started."


@pytest.mark.parametrize("extra", [{"accuracy": "bad"}, {"heading": {"deg": 3}}, {"accuracy": float("nan")},
                                   {"heading": True}])
def test_bad_optional_fields_are_a_400_and_the_walk_carries_on(server, extra):
    call(server, "POST", "/api/start", {"lat": LAT, "lng": LNG, "destination": DEST})
    status, reply, _ = call(server, "POST", "/api/tick", {"lat": LAT + 0.0001, "lng": LNG, **extra})
    assert status == 400 and reply["error"] == "accuracy and heading must be numbers."
    status, view, _ = call(server, "POST", "/api/tick", {"lat": LAT + 0.0001, "lng": LNG})
    assert status == 200 and view["raw"] == [round(LAT + 0.0001, 6), LNG]


def test_concurrent_ticks_all_get_answers(server):
    call(server, "POST", "/api/start", {"lat": LAT, "lng": LNG, "destination": DEST})
    statuses, errors = [], []

    def walk(i):
        try:
            for j in range(5):
                statuses.append(call(server, "POST", "/api/tick",
                                     {"lat": LAT + 0.00001 * (i + j), "lng": LNG})[0])
        except Exception as e:                  # surfaces in the assertion below, not as a lost thread
            errors.append(e)

    # 4 clients: under socketserver's listen backlog of 5, which macOS enforces with a reset.
    threads = [threading.Thread(target=walk, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == [] and statuses == [200] * 20


# ---------------------------------------------------------------- POST: malformed input
@pytest.mark.parametrize("body", [b"not json", b"\xff\xfe", b"null", b"[1, 2]", b'"lat"', b"5", b"[" * 60000])
def test_bad_json_is_a_400(server, body):
    status, reply, _ = call(server, "POST", "/api/tick", body=body)
    assert status == 400 and reply == {"ok": False, "error": "Expected a JSON object."}


@pytest.mark.parametrize("body", [b"{}", b'{"lat": 1}', b'{"lat": "north", "lng": 1}', b'{"lat": null, "lng": 1}',
                                  b'{"lat": NaN, "lng": 1}', b'{"lat": 1, "lng": Infinity}', b'{"lat": 91, "lng": 1}',
                                  b'{"lat": 1, "lng": -181}', b'{"lat": [1], "lng": 1}',
                                  b'{"lat": 1' + b"0" * 400 + b', "lng": 1}'])
def test_missing_or_impossible_coordinates_are_a_400(server, google, body):
    for path in ("/api/start", "/api/tick"):
        status, reply, _ = call(server, "POST", path, body=body)
        assert status == 400 and reply["error"] == "lat and lng are required."
    assert google.calls == []
    assert call(server, "GET", "/api/state")[0] == 200       # the server is still fine


@pytest.mark.parametrize("dest", [None, "", "   "])
def test_start_needs_a_destination(server, google, dest):
    status, reply, _ = call(server, "POST", "/api/start", {"lat": LAT, "lng": LNG, "destination": dest})
    assert status == 400 and reply["error"] == "A destination is required." and google.calls == []


def test_empty_body_is_an_empty_object(server):
    status, reply, _ = call(server, "POST", "/api/tick", body=b"")
    assert status == 400 and reply["error"] == "lat and lng are required."


def test_oversized_body_is_refused_and_the_connection_closed(server):
    """The unread body must never be parsed as a second request on the same keep-alive connection."""
    out = raw_request(server, b"POST /api/tick HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 70000\r\n\r\n")
    head, _, body = out.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 413")
    assert json.loads(body) == {"ok": False, "error": "Request is too large."}


@pytest.mark.parametrize("length", [b"-1", b"abc", b"1.5"])
def test_bad_content_length_is_a_400_not_a_hung_thread(server, length):
    out = raw_request(server, b"POST /api/tick HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: " + length + b"\r\n\r\n")
    head, _, body = out.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 400")
    assert json.loads(body)["ok"] is False


def test_an_unexpected_failure_in_routing_is_a_500_and_the_server_lives_on(server, google):
    google.script = [KeyError("legs")]                  # e.g. Google changed its JSON
    status, reply, _ = call(server, "POST", "/api/start", {"lat": LAT, "lng": LNG, "destination": DEST})
    assert status == 500 and reply == {"ok": False, "error": "Internal error."}
    status, view, _ = call(server, "POST", "/api/start", {"lat": LAT, "lng": LNG, "destination": DEST})
    assert status == 200 and view["ok"] is True


# ---------------------------------------------------------------- route failures
def test_no_key_uses_the_real_compute_route_and_never_reaches_the_network(server, monkeypatch):
    monkeypatch.setattr(walk_server, "SESSION", WalkSession())   # the default compute_route
    status, view, _ = call(server, "POST", "/api/start", {"lat": LAT, "lng": LNG, "destination": DEST})
    assert status == 200 and view["ok"] is False and view["google_calls"] == 1
    assert view["error"] == "No directions right now. The beacon points to Cloyne Court Hotel."
    assert view["guidance"] == view["error"]
    assert call(server, "GET", "/api/state")[1]["ok"] is False


@pytest.mark.parametrize("err", [RouteUnavailable("down"), RouteQuota("429"), NoKey("none"), RouteDenied("403")])
def test_a_failed_first_route_reports_and_holds_nothing(err):
    google = FakeGoogle(err)
    s = WalkSession(compute=google)
    view = s.start(LAT, LNG, DEST, now=1000)
    assert view["ok"] is False and view["holding"] is False and view["navigating"] is False
    assert "No directions right now." in view["error"]
    after = s.tick(LAT, LNG, now=2000)                  # ticks after a failed start never ask Google on their own
    assert after["ok"] is False and len(google.calls) == 1
    assert s.start(LAT, LNG, DEST, now=2001)["ok"] is True    # pressing start again does
    assert len(google.calls) == 2


def _walk_off(s: WalkSession, t0: float) -> float:
    """On the route for 20 s, then 40 m east of it until the follower asks for a new route. Returns the time."""
    s.tick(LAT + 0.00015, LNG, now=t0 + 10)
    s.tick(LAT + 0.00030, LNG, now=t0 + 20)
    off = _east(LAT + 0.00030, LNG, 40)
    s.tick(LAT + 0.00030, off, now=t0 + 30)
    s.tick(LAT + 0.00030, off, now=t0 + 39)
    return off


@pytest.mark.parametrize("err, delay", [(RouteUnavailable("timeout"), max(UNAVAILABLE_S, REROUTE_MIN_S)),
                                        (RouteQuota("429"), QUOTA_S)])
def test_a_failed_reroute_holds_for_its_full_backoff_then_retries(err, delay):
    """A 429 waits the whole QUOTA_S (it used to be cut to 20 s), and no retry beats the 20 s minimum."""
    google = FakeGoogle("route", err, "route")
    s = WalkSession(compute=google)
    s.start(LAT, LNG, DEST, now=1000)
    off = _walk_off(s, 1000)
    assert len(google.calls) == 2                       # the reroute was asked for and failed
    view = s.state()
    assert view["holding"] is True and view["hold_s"] == delay and view["on_route"] is False
    assert view["remaining"] == [view["raw"], view["destination"]]   # held: a straight line to the destination
    retry = 1039 + delay
    for t in range(1040, int(retry), 3):
        view = s.tick(LAT + 0.00030, off, now=t)
        assert len(google.calls) == 2 and view["holding"] is True
        assert view["hold_s"] == round(retry - t, 1)   # the page shows this as a countdown
    view = s.tick(LAT + 0.00030, off, now=retry)
    assert len(google.calls) == 3 and view["holding"] is False and view["ok"] is True
    assert google.calls[2][0] == (LAT + 0.00030, off)  # rerouted from where the walker actually is


@pytest.mark.parametrize("err", [NoKey("none"), RouteDenied("403")])
def test_a_reroute_that_can_never_work_is_not_retried(err):
    google = FakeGoogle("route", err)
    s = WalkSession(compute=google)
    s.start(LAT, LNG, DEST, now=1000)
    off = _walk_off(s, 1000)
    for t in range(1040, 2000, 30):
        s.tick(LAT + 0.00030, off, now=t)
    assert len(google.calls) == 2 and s.state()["holding"] is False


def test_a_poor_fix_is_shown_raw_and_never_moves_progress():
    s = WalkSession(compute=FakeGoogle())
    s.start(LAT, LNG, DEST, now=1000)
    before = s.tick(LAT + 0.00015, LNG, accuracy=5, now=1010)["progress_m"]
    view = s.tick(LAT + 0.00040, LNG, accuracy=100, now=1020)
    assert view["ignored_accuracy"] is True and view["on_route"] is False
    assert view["puck"] == view["raw"] == [round(LAT + 0.00040, 6), LNG]
    assert view["progress_m"] == before


def test_walking_to_the_end_arrives():
    s = WalkSession(compute=FakeGoogle())
    s.start(LAT, LNG, DEST, now=1000)
    view = {}
    for i in range(1, 25):                              # ~4.6 m every 4 s up to the destination
        view = s.tick(min(LAT + 0.0010, LAT + i * 0.00005), LNG, accuracy=4, now=1000 + 4 * i)
    assert view["arrived"] is True and view["navigating"] is False
    assert view["guidance"] == "You have arrived at Cloyne Court Hotel."
