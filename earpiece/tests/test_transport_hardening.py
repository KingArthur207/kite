"""The network edge under abuse: browser origins, malformed frames, floods, concurrent
senders, reconnect storms, and a clean shutdown. Real sockets on loopback, no audio."""

from __future__ import annotations

import json
import socket
import threading

import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from conftest import free_port, wait_for
from earpiece.transport import MAX_BUCKETS, Transport, origin_allowed


@pytest.fixture
def net():
    got: list = []
    logs: list = []
    t = Transport(got.append, "127.0.0.1", free_port(), free_port(), log=logs.append,
                  allow_origins=("http://192.168.1.5:5173/",))
    t.state_provider = lambda: {"ok": True}
    t.start()
    t.got, t.logs = got, logs
    yield t
    t.stop()


def ws_url(t: Transport) -> str:
    return f"ws://127.0.0.1:{t.ws_port}"


def udp_send(t: Transport, data: bytes, sock: socket.socket | None = None) -> None:
    s = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(data, ("127.0.0.1", t.udp_port))
    if sock is None:
        s.close()


# ------------------------------------------------------------------ browser origins
@pytest.mark.parametrize("origin", [None, "http://localhost:5173", "http://127.0.0.1:5173",
                                    "http://localhost:5174", "http://[::1]:5173", "https://localhost",
                                    "http://192.168.1.5:5173"])
def test_local_pages_and_non_browser_clients_connect(net, origin):
    with connect(ws_url(net), origin=origin, open_timeout=3) as ws:
        assert json.loads(ws.recv(timeout=2))["type"] == "state"
        ws.send(json.dumps({"type": "say", "text": "hello", "source": "sim"}))
        assert wait_for(lambda: any(m.type == "say" for m in net.got), 3)


@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost.evil.example:5173",
                                    "http://127.0.0.1.nip.io:5173", "null", "file://", "http://192.168.1.6:5173",
                                    "chrome-extension://abcdef"])
def test_other_web_pages_get_403_and_nothing_is_spoken(net, origin):
    with pytest.raises(InvalidStatus) as e:
        with connect(ws_url(net), origin=origin, open_timeout=3) as ws:
            ws.send(json.dumps({"type": "say", "text": "walk now", "priority": "critical"}))
    assert e.value.response.status_code == 403
    assert net.got == [] and net.stats["origin_rejected"] == 1 and net.stats["clients"] == 0


def test_origin_allowed_is_strict():
    assert origin_allowed(None) and origin_allowed("http://localhost:5173/")
    assert not origin_allowed("http://localhost:5173.evil.example")
    assert not origin_allowed("http://user@localhost:5173")
    assert not origin_allowed("ws://localhost:5173")
    assert origin_allowed("http://10.0.0.2:5173", frozenset({"http://10.0.0.2:5173"}))


# ------------------------------------------------------------------ malformed input
NASTY = [
    b'{"type":"beacon","bearing":' + b"9" * 400 + b"}",                                # OverflowError
    b'{"type":"say","text":"x","seq":' + b"1" * 5000 + b"}",                          # int digit limit
    b"[" * 4000 + b"]" * 4000,                                                        # RecursionError
    b"\xff\xfe", b"{", b"null", b"[]", b'{"type":"control","action":"self_destruct"}',
]


def test_parser_crashes_never_kill_the_udp_socket(net):
    for raw in NASTY:
        udp_send(net, raw)
    udp_send(net, b'{"type":"heartbeat","source":"cam"}')
    assert wait_for(lambda: [m.type for m in net.got] == ["heartbeat"], 3)
    assert wait_for(lambda: net.stats["bad"] == len(NASTY), 2)


def test_parser_crashes_never_close_the_websocket(net):
    with connect(ws_url(net), open_timeout=3) as ws:
        ws.recv(timeout=2)
        for raw in NASTY:
            ws.send(raw.decode("utf-8", "replace"))
        ws.send(json.dumps({"type": "scene", "text": "still here"}))
        assert wait_for(lambda: [m.type for m in net.got] == ["scene"], 3), (net.got, net.stats, net.logs)
        assert net.stats["bad"] == len(NASTY) and net.stats["clients"] == 1


def test_oversized_websocket_frame_drops_only_that_client(net):
    with connect(ws_url(net), open_timeout=3) as big:
        big.recv(timeout=2)
        try:
            big.send("x" * (300 * 1024))
        except Exception:
            pass
    with connect(ws_url(net), open_timeout=3) as ws:
        ws.recv(timeout=2)
        ws.send(json.dumps({"type": "heartbeat"}))
        assert wait_for(lambda: len(net.got) == 1, 3)


def test_handler_exceptions_are_contained():
    def boom(msg):
        raise RuntimeError("handler bug")
    logs: list = []
    t = Transport(boom, "127.0.0.1", free_port(), free_port(), log=logs.append)
    t.start()
    try:
        for _ in range(3):
            udp_send(t, b'{"type":"heartbeat"}')
        assert wait_for(lambda: sum("handler error" in m for m in logs) == 3, 3)
    finally:
        t.stop()


# ------------------------------------------------------------------ floods and storms
def test_rate_limiter_table_stays_bounded():
    """Every UDP source port is its own peer; a sender cycling ports must not grow memory."""
    t = Transport(lambda m: None, log=lambda *_: None)
    for i in range(3 * MAX_BUCKETS):
        t.ingest(b'{"type":"heartbeat"}', f"udp:127.0.0.1:{i}", "udp")
        assert len(t._buckets) <= MAX_BUCKETS
    assert t.stats["udp"] == 3 * MAX_BUCKETS


def test_rate_limit_is_per_peer():
    t = Transport(lambda m: None, log=lambda *_: None, rate_per_peer=50)
    for _ in range(500):
        t.ingest(b'{"type":"heartbeat"}', "udp:a:1", "udp")
        t.ingest(b'{"type":"heartbeat"}', "udp:b:2", "udp")
    assert t.stats["limited"] > 800 and 100 <= t.stats["udp"] < 200


def test_concurrent_udp_senders(net):
    n_threads, per = 8, 600
    errors: list = []

    def sender(k):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for i in range(per):
                raw = (b"\x00garbage" if i % 5 == 0 else json.dumps(
                    {"type": "obstacles", "source": f"cam{k}", "seq": i,
                     "objects": [{"id": 1, "bearing": i % 180, "distance": 1 + k}]}).encode())
                s.sendto(raw, ("127.0.0.1", net.udp_port))
        except Exception as e:           # pragma: no cover - would fail the test below
            errors.append(e)
        finally:
            s.close()

    ths = [threading.Thread(target=sender, args=(k,)) for k in range(n_threads)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    total = n_threads * per
    st = net.stats
    # loopback UDP may drop datagrams when the receive buffer fills; it never invents them
    assert wait_for(lambda: st["udp"] + st["bad"] + st["limited"] >= total * 0.5, 5)
    assert not errors and st["udp"] + st["bad"] + st["limited"] <= total
    assert st["bad"] <= total // 5 and len(net.got) == st["udp"]
    assert all(m.type == "obstacles" and len(m.objects) == 1 for m in net.got)


def test_websocket_reconnect_storm_leaves_nothing_behind(net):
    def churn(k):
        for i in range(25):
            with connect(ws_url(net), open_timeout=5) as ws:
                ws.send(json.dumps({"type": "heartbeat", "source": f"c{k}"}))
                if i % 3 == 0:
                    ws.recv(timeout=2)
    ths = [threading.Thread(target=churn, args=(k,)) for k in range(8)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    assert wait_for(lambda: net.stats["clients"] == 0 and not net._clients, 5)
    assert wait_for(lambda: len(net.got) == 200, 5)
    assert not [k for k in net._buckets if k.startswith("ws:")]        # per-connection buckets are freed


def stalled_client(t: Transport) -> socket.socket:
    """A WebSocket client that completes the handshake and then never reads a byte."""
    import base64
    import os
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    s.connect(("127.0.0.1", t.ws_port))
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{t.ws_port}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    assert b" 101 " in s.recv(1024)
    return s


def test_a_stalled_client_is_closed_and_others_keep_receiving(net):
    """A client that stops reading (a frozen tab, a dead Wi-Fi link) fills its socket; the
    broadcast gives it 0.5 s, then closes it so it reconnects, instead of silently skipping it
    forever while it keeps sending. Everyone else gets every event."""
    slow = stalled_client(net)
    assert wait_for(lambda: net.stats["clients"] == 1, 3)
    try:
        with connect(ws_url(net), open_timeout=3) as fast:
            fast.recv(timeout=2)
            assert wait_for(lambda: net.stats["clients"] == 2, 3)
            blob = "x" * 20_000
            seen = set()

            def read_fast():
                while len(seen) < 200:
                    m = json.loads(fast.recv(timeout=5))
                    if m["type"] == "spoken":
                        seen.add(m["i"])
            reader = threading.Thread(target=read_fast)
            reader.start()
            for i in range(200):                     # ~4 MB: far more than the stalled socket takes
                net.publish({"type": "spoken", "i": i, "pad": blob})
            reader.join(20)
            assert seen == set(range(200))
            assert wait_for(lambda: net.stats.get("slow_dropped", 0) == 1, 5)
            assert wait_for(lambda: net.stats["clients"] == 1, 5)
    finally:
        slow.close()


# ------------------------------------------------------------------ lifecycle
def test_stop_frees_ports_and_ends_the_thread():
    udp, ws = free_port(), free_port()
    for _ in range(3):              # same ports every time: stop() really released them
        t = Transport(lambda m: None, "127.0.0.1", udp, ws, log=lambda *_: None)
        t.start()
        with connect(f"ws://127.0.0.1:{ws}", open_timeout=3) as c:
            c.recv(timeout=2) if t.state_provider else None
            t.stop()
        assert not t._thread.is_alive()
    t.stop()                        # idempotent


def test_app_stop_leaks_no_network_threads(offline_speech):
    from earpiece.app import Earpiece
    from earpiece.config import Cfg
    from earpiece.memory import MemoryStore

    def transports():
        return sum(1 for th in threading.enumerate() if th.name == "transport" and th.is_alive())
    before = transports()
    for _ in range(3):
        cfg = Cfg()
        cfg.speech.cache_dir = offline_speech / "cache"
        cfg.net.udp_port, cfg.net.ws_port = free_port(), free_port()
        mem = MemoryStore(project_id="", project_key="", store=offline_speech / "m.json", log=lambda *_: None)
        a = Earpiece(cfg, output="null", log=lambda *_: None, memory=mem)
        a.start(prewarm=False, announce=False)
        a.stop()
    assert wait_for(lambda: transports() == before, 3)
