"""SpacetimeBridge against a local fake of Maincloud's HTTP API.

The fake is a real HTTP server (the bridge's keep-alive client talks to it over a socket)
that implements /v1/identity and /call/push with the module's own checks, including its
530 replies when the reducer throws. Caregiver inbox tests live in test_caregiver_inbox.py.
"""

import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from conftest import wait_for
from earpiece import spacetime
from earpiece.config import Cfg
from earpiece.events import parse
from earpiece.spacetime import MAX_PAYLOAD, SpacetimeBridge, encode_payload

DB = "kite-test"


class FakeMaincloud:
    """Records reducer calls; `script` holds (status, body) replies to give first."""

    def __init__(self):
        self.calls: list[dict] = []
        self.times: list[float] = []
        self.auth: set = set()
        self.script: list[tuple[int, str]] = []
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"          # keep-alive, like Maincloud

            def log_message(self, *a):
                pass

            def _reply(self, code, body=""):
                b = (body if isinstance(body, str) else json.dumps(body)).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
                if self.path == "/v1/identity":
                    return self._reply(200, {"identity": "abc", "token": "tok-1"})
                m = re.fullmatch(rf"/v1/database/{DB}/call/(\w+)", self.path)
                if not m:
                    return self._reply(404, "not found")
                with fake.lock:
                    fake.times.append(time.monotonic())
                    fake.auth.add(self.headers.get("Authorization"))
                    if fake.script:
                        return self._reply(*fake.script.pop(0))
                    if m.group(1) != "push":
                        return self._reply(404, "No such procedure")
                    a = json.loads(body)
                    if a["type"] not in spacetime.TYPES:
                        return self._reply(530, f"unknown type {a['type']}")
                    if not re.fullmatch(r"[\w.:-]{1,64}", a["source"], re.ASCII):
                        return self._reply(530, "bad source")
                    if a["priority"] not in spacetime.PRIORITIES:
                        return self._reply(530, "bad priority")
                    if len(a["payload"]) > MAX_PAYLOAD:
                        return self._reply(530, "payload too large")
                    json.loads(a["payload"])
                    fake.calls.append(a)
                return self._reply(200)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def of(self, kind):
        return [json.loads(c["payload"]) for c in self.calls if c["type"] == kind]


@pytest.fixture
def fake():
    f = FakeMaincloud()
    yield f
    f.server.shutdown()


@pytest.fixture
def make(fake):
    def build(**kw):
        logs = kw.pop("logs", [])
        kw.setdefault("max_calls_per_s", 1000.0)   # the call ceiling has its own test
        b = SpacetimeBridge(fake.host, DB, kw.pop("source", "earpiece"), logs.append, **kw)
        b.logs = logs
        return b
    return build


# ------------------------------------------------------------------ the push call
def test_push_carries_type_source_priority_and_the_event_as_json(fake, make):
    b = make()
    ev = {"type": "hazard", "source": "drone", "kind": "crosswalk", "state": "dont_walk",
          "bearing": 5.0, "priority": "hazard"}
    b.publish(ev)
    b.publish({"type": "spoken", "t": 1.5, "text": "Kite is ready.", "priority": "system", "result": "said"})
    assert b.flush(5)
    hz, sp = fake.calls
    assert {k: hz[k] for k in ("type", "source", "priority")} == {"type": "hazard", "source": "drone",
                                                                   "priority": "hazard"}
    assert json.loads(hz["payload"]) == ev
    assert sp["source"] == "earpiece" and sp["priority"] == "system"     # own events: bridge source
    assert fake.auth == {"Bearer tok-1"}
    assert b.stats["sent"] == b.stats["calls"] == 2 and b.stats["last_ms"] is not None
    assert b.stats["backlog"] == 0


def test_source_and_priority_are_clamped_so_no_call_is_wasted(fake, make):
    b = make()
    b.publish({"type": "say", "source": "drone laptop/1 ✈", "priority": "URGENT", "text": "x"})
    b.publish({"type": "say", "source": "", "text": "y"})
    b.publish({"type": "guidance", "source": "s" * 200, "priority": "Nav", "text": "z"})
    b.publish({"type": "say", "priority": "critical", "text": "w", "n": float("nan")})
    assert b.flush(5)
    assert [c["source"] for c in fake.calls] == ["drone_laptop_1__", "earpiece", "s" * 64, "earpiece"]
    assert [c["priority"] for c in fake.calls] == ["", "", "nav", "critical"]
    assert json.loads(fake.calls[3]["payload"])["n"] is None             # NaN is not JSON
    assert b.stats["failed"] == 0


def test_types_the_module_does_not_know_are_dropped_locally_and_logged_once(fake, make):
    b = make()
    for _ in range(5):
        b.publish({"type": "obstacle", "source": "webcam", "id": "1", "bearing": 0, "distance": 1})
    b.publish({"type": "say", "text": "still works"})
    assert b.flush(5)
    assert [c["type"] for c in fake.calls] == ["say"] and b.stats["calls"] == 1
    assert sum("'obstacle'" in m for m in b.logs) == 1
    assert b.stats["dropped"] == 5


def test_heartbeats_are_never_mirrored(fake, make):
    b = make()
    for _ in range(20):
        b.publish({"type": "heartbeat", "source": "webcam"})
    b.publish({"type": "say", "text": "x"})
    assert b.flush(5)
    assert [c["type"] for c in fake.calls] == ["say"]
    assert b.stats["skipped"] == 20 and b.stats["dropped"] == 0 and b.stats["calls"] == 1


# ------------------------------------------------------------------ coalescing and the call budget
def test_100hz_obstacle_flood_sends_1hz_newest_wins_and_publish_never_blocks(fake, make):
    b = make()
    worst, seq, t0 = 0.0, 0, time.monotonic()
    while time.monotonic() - t0 < 2.5:
        seq += 1
        s = time.perf_counter()
        b.publish({"type": "obstacles", "source": "webcam", "seq": seq,
                   "objects": [{"id": "1", "kind": "person", "bearing": -20, "distance": 2.4}]})
        if seq % 25 == 0:
            b.publish({"type": "hazard", "source": "drone", "kind": "crowd", "n": seq})
        worst = max(worst, time.perf_counter() - s)
        time.sleep(0.01)
    assert b.flush(5)
    frames = fake.of("obstacles")
    assert 1 <= len(frames) <= 5                                         # ~1 Hz, not 100 Hz
    assert frames[-1]["seq"] == seq                                      # the newest frame lands
    assert [h["n"] for h in fake.of("hazard")] == list(range(25, seq + 1, 25))   # never coalesced
    assert b.stats["coalesced"] == seq - len(frames)
    gaps = [b2 - a for a, b2 in zip(fake.times, fake.times[1:])]
    # publish only enqueues: microseconds normally; the bound is loose so a loaded CI box
    # (GIL contention from other threads) can't fail it, while a blocking call (>= 80 ms) still would
    assert worst < 0.05 and gaps


def test_each_source_is_throttled_on_its_own_and_order_holds_for_the_rest(fake, make):
    b = make()
    for i in range(60):
        b.publish({"type": "say", "text": f"n{i}", "n": i})
        b.publish({"type": "position", "source": "sim", "lat": 37.8 + i * 1e-5, "lng": -122.4})
        b.publish({"type": "position", "source": "phone", "lat": 1.0 + i, "lng": 2.0})
        b.publish({"type": "spoken", "text": f"n{i}", "n": i})
    assert b.flush(10)
    assert [e["n"] for e in fake.of("say")] == list(range(60))
    assert [e["n"] for e in fake.of("spoken")] == list(range(60))
    pos = {c["source"]: json.loads(c["payload"]) for c in fake.calls if c["type"] == "position"}
    assert pos["sim"]["lat"] == pytest.approx(37.8 + 59 * 1e-5) and pos["phone"]["lat"] == 60.0
    assert len(fake.of("position")) <= 4


def test_global_ceiling_spreads_a_burst_and_counts_every_call(fake, make):
    b = make(max_calls_per_s=10.0)
    t0 = time.monotonic()
    for i in range(15):
        b.publish({"type": "say", "text": "x", "n": i})
    assert b.flush(10)
    took = time.monotonic() - t0
    assert 0.9 <= took < 3.0                                             # 5 at once, then 10/s
    assert b.stats["calls"] == 15 == len(fake.calls)
    assert max(sum(1 for t in fake.times if s <= t < s + 1.0) for s in fake.times) <= 15


def test_defaults_are_frugal_and_env_tunes_them(monkeypatch):
    b = SpacetimeBridge("http://127.0.0.1:9", DB, "earpiece", lambda *_: None, start=False)
    assert b.throttle_s == 1.0 and b.max_calls_per_s == 5.0
    monkeypatch.setenv("KITE_SPACETIME_HZ", "2")
    monkeypatch.setenv("KITE_SPACETIME_MAX_CPS", "3")
    b = SpacetimeBridge("http://127.0.0.1:9", DB, "earpiece", lambda *_: None, start=False)
    assert b.throttle_s == 0.5 and b.max_calls_per_s == 3.0
    monkeypatch.setenv("KITE_SPACETIME_HZ", "nonsense")
    assert SpacetimeBridge("http://127.0.0.1:9", DB, start=False).throttle_s == 1.0
    assert spacetime.CaregiverInbox(print, "http://127.0.0.1:9", DB, interval=0.01,
                                    start=False).interval == spacetime.MIN_POLL_S   # <= 2 Hz


# ------------------------------------------------------------------ payload cap
def test_huge_payloads_are_trimmed_to_fit_keeping_the_nearest_objects(fake, make):
    objs = [{"id": str(i), "kind": "person", "bearing": 0.0, "distance": 100.0 - i * 0.04, "note": "x" * 40}
            for i in range(2000)]
    s = encode_payload({"type": "obstacles", "source": "webcam", "objects": objs})
    d = json.loads(s)
    assert len(s) <= MAX_PAYLOAD and d["truncated"] and d["objects_total"] == 2000
    assert 10 < len(d["objects"]) < 2000 and d["objects"][0]["id"] == "1999"   # nearest first
    b = make()
    b.publish({"type": "obstacles", "source": "webcam", "objects": objs})
    b.publish({"type": "say", "text": "é" * 40000})                    # 6 chars each as é
    assert b.flush(5)
    assert len(fake.calls) == 2 and all(len(c["payload"]) <= MAX_PAYLOAD for c in fake.calls)
    assert len(fake.of("say")[0]["text"]) == spacetime.MAX_TEXT
    assert fake.of("obstacles")[0]["objects_total"] == 2000


# ------------------------------------------------------------------ failures
def test_server_errors_back_off_exponentially_and_the_event_still_lands(fake, make):
    fake.script = [(500, "boom"), (500, "boom"), (503, "busy")]
    b = make(backoff_s=0.05)
    b.publish({"type": "hazard", "kind": "crowd"})
    b.publish({"type": "say", "text": "after"})
    assert b.flush(10)
    assert [c["type"] for c in fake.calls] == ["hazard", "say"]          # retried, still in order
    assert b.stats["failed"] == 3 and b.stats["calls"] == 5
    gaps = [b2 - a for a, b2 in zip(fake.times, fake.times[1:4])]
    assert gaps[0] >= 0.04 and gaps[1] >= 0.09 and gaps[2] >= 0.18      # 0.05, 0.1, 0.2 s
    assert sum("retrying" in m for m in b.logs) == 3


def test_network_down_never_blocks_publish():
    b = SpacetimeBridge("http://127.0.0.1:9", DB, "earpiece", lambda *_: None, backoff_s=0.05,
                        max_calls_per_s=1000.0)
    t = time.perf_counter()
    for i in range(500):
        b.publish({"type": "say", "text": "x", "n": i})
    assert time.perf_counter() - t < 0.1
    assert wait_for(lambda: b.stats["failed"] >= 3, 5)
    assert b.stats["sent"] == 0 and b.stats["backlog"] >= 490


def test_no_such_reducer_disables_the_bridge_once_instead_of_spamming(fake, make):
    fake.script = [(404, "No such procedure")]
    b = make()
    b.publish({"type": "say", "text": "a"})
    assert wait_for(lambda: b.stats["disabled"], 5)
    for _ in range(50):
        b.publish({"type": "say", "text": "b"})
    assert b.flush(5)
    assert len(fake.times) == 1 and not fake.calls
    assert sum("disabled" in m for m in b.logs) == 1
    assert "No such procedure" in b.stats["disabled"] and b.stats["dropped"] == 50


def test_a_type_the_module_rejects_is_turned_off_but_others_keep_flowing(fake, make):
    fake.script = [(530, "unknown type gesture")]       # e.g. the module was republished without it
    b = make()
    b.publish({"type": "gesture", "gesture": "talk"})
    assert b.flush(5)
    for _ in range(5):
        b.publish({"type": "gesture", "gesture": "talk"})
    b.publish({"type": "hazard", "kind": "crowd"})
    assert b.flush(5)
    assert [c["type"] for c in fake.calls] == ["hazard"]
    assert len(fake.times) == 2                                          # one rejected try, one hazard
    assert b.stats["dropped"] == 5 and b.stats["failed"] == 1
    assert sum("rejected" in m for m in b.logs) == 1


# ------------------------------------------------------------------ wiring
def test_env_turns_the_bridge_on_and_points_it(monkeypatch):
    assert not Cfg().net.spacetime_enabled                               # off unless asked
    monkeypatch.setenv("KITE_SPACETIME", "1")
    monkeypatch.setenv("SPACETIME_HOST", "http://127.0.0.1:3000")
    monkeypatch.setenv("SPACETIME_DB", "other-db")
    net = Cfg().net
    assert net.spacetime_enabled
    assert (net.spacetime_host, net.spacetime_db) == ("http://127.0.0.1:3000", "other-db")


def test_earpiece_mirrors_inbound_and_spoken_events_through_the_bridge(fake, offline_speech):
    from earpiece.app import Earpiece
    from earpiece.memory import MemoryStore
    cfg = Cfg()
    cfg.speech.cache_dir = offline_speech / "cache"
    cfg.net.udp_port, cfg.net.ws_port = 0, 0
    mem = MemoryStore(project_id="", project_key="", store=offline_speech / "memory.json", log=lambda *_: None)
    a = Earpiece(cfg, output="null", record=True, log=lambda *_: None, memory=mem)
    b = SpacetimeBridge(fake.host, DB, source="earpiece", log=lambda *_: None)   # as `run --spacetime`
    a.bridges.append(b)
    a.start(prewarm=False, announce=False)
    try:
        a.handle(parse({"type": "guidance", "source": "sim", "text": "Turn left onto Lyon Street.",
                        "key": "s1"}))
        for i in range(30):
            a.handle(parse({"type": "obstacles", "source": "webcam", "seq": i,
                            "objects": [{"id": "1", "kind": "person", "bearing": 10, "distance": 3}]}))
            a.handle(parse({"type": "heartbeat", "source": "webcam"}))
        assert wait_for(lambda: fake.of("spoken"), 5)
        assert b.flush(5)
        g = fake.of("guidance")
        assert len(g) == 1 and g[0]["text"] == "Turn left onto Lyon Street." and g[0]["source"] == "sim"
        assert fake.of("spoken")[0]["text"] == "Turn left onto Lyon Street."
        assert fake.of("obstacles")[-1]["seq"] == 29 and len(fake.of("obstacles")) <= 2
        assert not fake.of("heartbeat") and b.stats["skipped"] == 30
    finally:
        a.stop()


# ------------------------------------------------------------------ live Maincloud
@pytest.mark.skipif(os.environ.get("KITE_LIVE_TESTS") != "1", reason="set KITE_LIVE_TESTS=1 to hit Maincloud")
def test_live_maincloud_round_trip():
    """A handful of pushes (about 4 reducer calls) under a unique source, read back by SQL,
    then `clear`, which wipes the demo tables so the run leaves the database empty."""
    host = os.environ.get("SPACETIME_HOST") or spacetime.DEFAULT_HOST
    db = os.environ.get("SPACETIME_DB") or spacetime.DEFAULT_DB
    tag = f"pytest-{uuid.uuid4().hex[:8]}"
    b = SpacetimeBridge(host, db, tag, print)
    b.publish({"type": "hazard", "kind": "crosswalk", "state": "dont_walk", "bearing": 5, "priority": "hazard"})
    b.publish({"type": "say", "text": "Live test.", "priority": "info"})
    b.publish({"type": "heartbeat"})                                     # never sent
    for i in range(3):
        b.publish({"type": "obstacles", "seq": i, "objects": []})       # one call, newest wins
    assert b.flush(15), b.stats
    with httpx.Client(timeout=10) as http:
        r = http.post(f"{host}/v1/database/{db}/sql",
                      content=f"SELECT type, payload FROM event WHERE source = '{tag}'".encode())
        rows = r.json()[0]["rows"]
        assert sorted(t for t, _ in rows) == ["hazard", "obstacles", "say"]
        assert json.loads([p for t, p in rows if t == "obstacles"][0])["seq"] == 2
        print(f"live: {b.stats}")
        http.post(f"{host}/v1/database/{db}/call/clear", json={},
                  headers={"Authorization": f"Bearer {b._token}"}).raise_for_status()
    assert b.stats["calls"] == 3


# ------------------------------------------------------------------ failure modes
def test_rate_limited_429_backs_off_and_still_lands(fake, make):
    fake.script = [(429, "too many requests"), (429, "too many requests")]
    b = make(backoff_s=0.05)
    b.publish({"type": "say", "text": "after a 429"})
    assert b.flush(10)
    assert [c["type"] for c in fake.calls] == ["say"]
    assert b.stats["failed"] == 2 and b.stats["dropped"] == 0 and not b.stats["disabled"]


def test_stale_identity_is_refetched(fake, make):
    fake.script = [(401, "bad token")]
    b = make(backoff_s=0.01)
    b.publish({"type": "say", "text": "x"})
    assert b.flush(10)
    assert len(fake.calls) == 1 and b.stats["sent"] == 1


def test_a_server_that_never_answers_bounds_the_queue_and_never_blocks(monkeypatch):
    """Accepts connections and then says nothing: every call times out."""
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(64)
    srv.settimeout(0.2)
    held = []
    stop = threading.Event()

    def accept():
        while not stop.is_set():
            try:
                held.append(srv.accept()[0])
            except OSError:
                pass
    threading.Thread(target=accept, daemon=True).start()
    monkeypatch.setattr(spacetime, "TIMEOUT_S", 0.2)
    b = SpacetimeBridge(f"http://127.0.0.1:{srv.getsockname()[1]}", DB, "earpiece", lambda *_: None,
                        backoff_s=0.01, max_calls_per_s=1000.0)
    try:
        t = time.perf_counter()
        for i in range(3 * spacetime.QUEUE_MAX):
            b.publish({"type": "say", "text": "x", "n": i})
        assert time.perf_counter() - t < 0.5
        assert b._q.qsize() <= spacetime.QUEUE_MAX
        assert b.stats["dropped"] >= spacetime.QUEUE_MAX
        assert wait_for(lambda: b.stats["failed"] >= 2, 5)       # timeouts count as failures
        assert b.stats["sent"] == 0
    finally:
        stop.set()
        srv.close()
        for c in held:
            c.close()


def test_garbage_identity_reply_is_a_retry_not_a_dead_worker():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/identity":
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(200, text="<html>captive portal</html>")
            return httpx.Response(200, json={"token": "t"})
        return httpx.Response(200)

    b = SpacetimeBridge("http://st.test", DB, "earpiece", lambda *_: None, backoff_s=0.01,
                        max_calls_per_s=1000.0, start=False)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        assert b.send_now({"type": "say", "text": "x"}, http, tries=4)
    assert calls["n"] == 3 and b.stats["failed"] == 2
