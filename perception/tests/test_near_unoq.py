"""The UNO Q body camera as near's source, and the steady beat to the earpiece. Real board.py, no camera, no models."""

from __future__ import annotations

import http.client
import io
import secrets
import socket
import sys
import threading
import time
import types
import urllib.error
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from perception.near import output  # noqa: E402
from perception.near import unoq as U  # noqa: E402
from perception.near.unoq import UnoqSource, is_unoq, parse_spec  # noqa: E402

LINK = Path(__file__).resolve().parents[2] / "hardware" / "arduino" / "unoq_link"


# ------------------------------------------------------------------ the steady sender
class FakeEar:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_steady_sender_repeats_the_newest_frame_then_goes_quiet():
    ear, clock = FakeEar(), Clock()
    s = output.SteadySender(ear, hz=5, max_age=1.0, clock=clock)
    s._stop.set()                                       # drive it by hand
    s.update({"type": "obstacles", "objects": [{"id": "1", "priority": 1}]})
    assert len(ear.sent) == 1 and ear.sent[0]["seq"] == 1
    clock.t += 0.1
    assert not s.tick()                                 # inside the period: nothing extra
    for _ in range(4):
        clock.t += 0.2
        assert s.tick()                                 # 0.3, 0.5, 0.7, 0.9 s old: repeated
    clock.t += 0.2
    assert not s.tick()                                 # 1.1 s old: stale, the earpiece must hear silence
    assert [m["seq"] for m in ear.sent] == [1, 2, 3, 4, 5]
    s.update({"type": "obstacles", "objects": []})      # a new frame resets the age
    clock.t += 0.2
    assert s.tick() and ear.sent[-1]["objects"] == []


def test_spec_parsing():
    assert parse_spec("unoq") is None
    assert parse_spec("unoq:10.0.0.5") == "10.0.0.5"
    assert parse_spec("UNOQ:10.0.0.5:8765") == "10.0.0.5:8765"
    assert is_unoq("unoq") and is_unoq("unoq:h") and not is_unoq("webcam:Logitech") and not is_unoq(None)
    with pytest.raises(ValueError):
        parse_spec("unoq:")


# ------------------------------------------------------------------ against the real board server
class JpegCamera:
    """Counts frames: each read() is a new 64x48 JPEG whose grey level is the frame number."""
    name = "usb camera video0"

    def __init__(self):
        self.n = 0

    def read(self):
        self.n += 1
        ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), self.n % 250, np.uint8))
        return buf.tobytes()

    def close(self):
        pass


@pytest.fixture
def board_server(monkeypatch):
    sys.path.insert(0, str(LINK))
    import board
    token = secrets.token_urlsafe(24)
    b = board.Board(board.Config({"KITE_PORT": "0", "KITE_BIND": "127.0.0.1", "KITE_TOKEN": token,
                                  "KITE_CAMERA": "synthetic", "KITE_CLOUD": "0", "KITE_VIDEO_FPS": "20"}))
    srv = board.start(b)
    monkeypatch.setenv("KITE_NO_CLOUD", "1")
    yield b, f"unoq:127.0.0.1:{srv.server_address[1]}", token
    board.shutdown(b, srv)


def test_reads_the_board_camera_newest_frame_first(board_server, monkeypatch):
    b, spec, token = board_server
    b.hub.camera.close()
    b.hub.camera = JpegCamera()
    monkeypatch.setenv("KITE_TOKEN", token)
    src = UnoqSource(spec, out=lambda *a, **k: None)
    try:
        assert src.camera == "usb camera video0" and not src.test_pattern
        assert "Logitech" in src.name                  # picks the team's C270 profile
        ok, img = src.read()
        assert ok and img.shape == (48, 64, 3)
        time.sleep(0.6)                                # the board keeps sending while we "think"
        ok, img2 = src.read()
        assert ok
        assert src.frames_dropped > 0                  # stale frames were skipped, not queued
        assert int(img2[0, 0, 0]) > int(img[0, 0, 0]) + 3
    finally:
        src.release()


def test_no_frames_while_the_board_shows_its_test_pattern(board_server, monkeypatch):
    _, spec, token = board_server
    monkeypatch.setenv("KITE_TOKEN", token)
    said = []
    src = UnoqSource(spec, out=lambda *a, **k: said.append(a[0]))
    try:
        assert src.test_pattern
        assert src.read() == (False, None)             # never a distance from a "NO CAMERA" card
        assert any("test pattern" in s for s in said)
    finally:
        src.release()


def test_wrong_token_is_a_clear_error(board_server, monkeypatch):
    _, spec, _ = board_server
    monkeypatch.setenv("KITE_TOKEN", "x" * 32)
    with pytest.raises(SystemExit, match="401"):
        UnoqSource(spec, out=lambda *a, **k: None)


def test_unreachable_board_says_what_it_tried(monkeypatch):
    monkeypatch.setenv("KITE_TOKEN", "y" * 32)
    monkeypatch.setenv("KITE_NO_CLOUD", "1")
    with pytest.raises(SystemExit, match="cannot reach the UNO Q"):
        UnoqSource("unoq:127.0.0.1:9", out=lambda *a, **k: None, connect_timeout=0.3)


# ------------------------------------------------------------------ fault injection: fake board, no network
# The reader and status threads are driven by hand in the test's own thread: Thread.start() is a no-op and every
# wait returns at once, so a test that reconnects ten times takes microseconds and nothing can hang.
REAL_CAM = (200, {"ok": True, "status": {"camera": {"source": "usb camera video0", "width": 64, "height": 48}}})
PATTERN = (200, {"ok": True, "status": {"camera": {"source": "synthetic", "width": 64, "height": 48}}})


def jpeg(level: int = 7) -> bytes:
    return cv2.imencode(".jpg", np.full((48, 64, 3), level, np.uint8))[1].tobytes()


def part(body: bytes, length: int | str | None = None) -> bytes:
    n = len(body) if length is None else length
    return b"--kiteframe\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(n).encode() + b"\r\n\r\n" + body + b"\r\n"


class Response(io.BytesIO):
    """What urlopen returns for /video.mjpg: a context manager with headers and readline/read."""

    def __init__(self, data: bytes, ctype: str = "multipart/x-mixed-replace; boundary=kiteframe"):
        super().__init__(data)
        self.headers = {"Content-Type": ctype}


class FastEvent(threading.Event):
    """wait() never blocks; it records the timeouts asked for and sets itself after `limit` waits."""

    def __init__(self, limit: int | None = None):
        super().__init__()
        self.limit, self.timeouts = limit, []

    def wait(self, timeout=None):
        self.timeouts.append(timeout)
        if self.limit is not None and len(self.timeouts) >= self.limit:
            self.set()
        return self.is_set()


class NoThread:
    started: list = []

    def __init__(self, target=None, name=None, daemon=None):
        self.target, self.name = target, name

    def start(self):
        NoThread.started.append(self.name)


class FakeDirect:
    """The board's HTTP API: /status replies and /video.mjpg opens are scripted (an exception is raised)."""
    label = "direct"

    def __init__(self, streams=(), statuses=(), host="127.0.0.1"):
        self.streams, self.statuses, self.host = list(streams), list(statuses), host
        self.opens = 0
        self.stop: threading.Event | None = None

    def __str__(self):
        return f"{self.host}:8765"

    def call(self, path):
        assert path == "/status"
        s = self.statuses.pop(0) if self.statuses else REAL_CAM
        if isinstance(s, BaseException):
            raise s
        return s

    def open(self, path, timeout=None):
        assert path == "/video.mjpg" and timeout            # never an unbounded read
        self.opens += 1
        if not self.streams:                                # the script is over: end the reader
            self.stop.set()
            raise ConnectionRefusedError("script over")
        s = self.streams.pop(0)
        if isinstance(s, BaseException):
            raise s
        return s


@pytest.fixture
def fake_board(monkeypatch):
    """make(spec, directs) -> UnoqSource over FakeDirect(s); Link.direct() hands them out in order."""
    real = U._kite_link()
    NoThread.started = []
    monkeypatch.setattr(U, "threading", types.SimpleNamespace(Thread=NoThread, Condition=threading.Condition,
                                                              Event=FastEvent))
    monkeypatch.setattr(U, "STALE_S", 0)

    def make(directs, spec="unoq:127.0.0.1:8765", said=None):
        queue = list(directs)

        class Link:
            def __init__(self, args, out=None):
                self.a, self.tried = args, [("direct", "127.0.0.1", 8765, False, "refused")]

            def need_token(self):
                return "t" * 32

            def direct(self):
                return queue.pop(0) if queue else None

        kl = types.SimpleNamespace(build_parser=real.build_parser, Link=Link, boundary_of=real.boundary_of,
                                   iter_mjpeg=real.iter_mjpeg, is_jpeg=real.is_jpeg)
        monkeypatch.setattr(U, "_kite_link", lambda: kl)
        log = said if said is not None else []
        src = UnoqSource(spec, out=lambda *a, **k: log.append(a[0]))
        for d in directs:
            d.stop = src._stop
        return src

    return make


def drain(src) -> None:
    """Run the reader thread's loop to the end of its script, then let read() see what it left."""
    src._reader()
    src._stop.clear()


def test_the_reader_survives_every_network_failure_and_keeps_only_the_newest_frame(fake_board):
    d = FakeDirect([
        ConnectionResetError(54, "Connection reset by peer"),
        urllib.error.HTTPError("http://board/video.mjpg", 503, "busy", {}, None),
        socket.timeout("timed out"),                                   # a stalled stream past urlopen's timeout
        http.client.IncompleteRead(b"--kite"),
        Response(part(jpeg(), length=5000)),                           # truncated frame: the link dropped mid-JPEG
        Response(part(jpeg(), length="lots")),                         # a garbled header
        Response(part(b"not a jpeg at all") + part(jpeg(40)) + part(jpeg(90))),  # then frames, then the stream ends
    ])
    said = []
    src = fake_board([d], said=said)
    assert src.camera == "usb camera video0" and src.size == (64, 48) and not src.test_pattern
    drain(src)                                                      # returns: the script ended with stop set
    assert d.opens == 8 and src.reconnects == 7                        # every failure, and the end, reconnected
    assert src.frames_in == 2 and src.frames_dropped == 1              # the junk part was skipped, not decoded
    ok, img = src.read()
    assert ok and img.shape == (48, 64, 3) and abs(int(img[0, 0, 0]) - 90) < 4   # the newest frame, not a backlog
    assert src.read() == (False, None)                                # nothing new: no frame repeated
    logs = [s for s in said if "UNO Q video" in s]
    assert len(logs) == 2 and "ConnectionResetError" in logs[0] and "EOFError" not in logs[0]   # 1st and 5th only
    assert "EOFError" in logs[1] and "stream ended" in logs[1]


def test_backoff_grows_while_down_and_resets_once_frames_flow(fake_board):
    d = FakeDirect([OSError("down")] * 5 + [Response(b"")] + [Response(part(jpeg()))] + [OSError("down")])
    src = fake_board([d])
    src._stop = d.stop = FastEvent()
    drain(src)
    # 5 failures, then a stream that opens but ends before any frame (no reset), then one that delivers a frame
    assert src._stop.timeouts == [0.5, 1.0, 2.0, 4.0, 5.0, 5.0, 0.5, 1.0]


def test_a_corrupt_jpeg_is_a_skipped_frame_not_a_crash(fake_board):
    bad = b"\xff\xd8" + b"\x00garbage" * 20 + b"\xff\xd9"             # passes the marker check, does not decode
    d = FakeDirect([Response(part(bad))])
    src = fake_board([d])
    drain(src)
    assert src.frames_in == 1
    assert src.read() == (False, None)
    src._jpeg, src._seq = b"\xff\xd8\xff\xd9\x00", src._seq + 1       # even a near-empty buffer
    assert src.read() == (False, None)

    def raising(*_):
        raise cv2.error("imdecode: corrupt data")

    src._cv2 = types.SimpleNamespace(imdecode=raising, IMREAD_COLOR=cv2.IMREAD_COLOR, error=cv2.error)
    src._jpeg, src._seq = bad, src._seq + 1                           # a decoder that throws: still just a skip
    assert src.read() == (False, None)


def test_the_stream_never_flows_before_the_first_read_and_starts_once(fake_board):
    src = fake_board([FakeDirect()])
    assert NoThread.started == ["unoq-status"]
    assert src.read() == (False, None)                                # no frame yet: none, at once
    src.read()
    assert NoThread.started == ["unoq-status", "unoq-video"]
    src.release()
    src._jpeg, src._seq = jpeg(), src._seq + 1
    assert src.read() == (False, None)                                # released: never a frame again


def test_a_lost_board_is_looked_for_again_after_five_failures(fake_board):
    first = FakeDirect([OSError("no route to host")] * 5)
    second = FakeDirect([Response(part(jpeg(120)))], host="10.0.0.9")
    src = fake_board([first, second], spec="unoq")
    second.stop = src._stop
    drain(src)
    assert src.direct is second and src.frames_in == 1
    ok, img = src.read()
    assert ok and abs(int(img[0, 0, 0]) - 120) < 4


def test_rediscovery_that_finds_nothing_keeps_trying_the_old_address(fake_board):
    only = FakeDirect([OSError("down")] * 6)
    src = fake_board([only], spec="unoq")
    drain(src)                                                     # _find() raises SystemExit: swallowed
    assert src.direct is only and src.reconnects == 6


def test_a_named_board_is_never_rediscovered(fake_board):
    d = FakeDirect([OSError("down")] * 10)
    src = fake_board([d, FakeDirect(host="elsewhere")])
    drain(src)
    assert src.direct is d


def test_the_status_watch_survives_garbage_and_a_401_and_still_catches_the_test_pattern(fake_board):
    d = FakeDirect(statuses=[REAL_CAM])
    said = []
    src = fake_board([d], said=said)
    d.statuses = [ValueError("Expecting value: line 1 column 1"),     # a garbled JSON reply
                  http.client.IncompleteRead(b"{"),                    # a body cut off mid-reply
                  ConnectionResetError("reset"),
                  (401, {"error": "bad token"}),                       # token rotated while walking
                  (401, {"error": "bad token"}),
                  (500, {"error": "oops"}),
                  (200, ["not", "a", "dict"]),
                  PATTERN,                                             # the camera fell out: test pattern
                  REAL_CAM]                                            # and came back
    seen = []
    real_status = src._status

    def spy():
        st = real_status()
        seen.append(src.test_pattern)
        return st

    src._status = spy
    src._stop = FastEvent(limit=len(d.statuses) + 1)
    src._watch_status()                                                # returns: it ran to the end, never died
    assert not d.statuses
    assert seen[-1] is True and not src.test_pattern                   # flipped to the pattern, then back
    assert sum("401" in s for s in said) == 1                          # the same error is said once
    assert any("camera lost, test pattern" in s for s in said)
    assert any("camera usb camera video0 is back" in s for s in said)


def test_status_replies_that_cannot_be_read_are_unknown_not_fatal(fake_board):
    d = FakeDirect(statuses=[ValueError("bad json")])
    src = fake_board([d])                                             # startup with a garbled status reply
    assert src.camera == "unknown" and src.size == (640, 480) and not src.test_pattern
    for s in (http.client.IncompleteRead(b""), OSError("down"), (503, {}), (200, "text")):
        d.statuses = [s]
        assert src._status() is None
    d.statuses = [(200, {"camera": {"source": "x"}})]                  # an older board: status at the top level
    assert src._status() == {"camera": {"source": "x"}}
    d.statuses = [(401, {})]
    with pytest.raises(SystemExit, match="401"):
        src._status()


def test_a_board_that_starts_on_its_test_pattern_says_so(fake_board):
    said = []
    fake_board([FakeDirect(statuses=[PATTERN])], said=said)
    assert "test pattern" in said[-1]
