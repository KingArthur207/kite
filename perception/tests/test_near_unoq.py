"""The UNO Q body camera as near's source, and the steady beat to the earpiece. Real board.py, no camera, no models."""

from __future__ import annotations

import secrets
import sys
import time
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from perception.near import output  # noqa: E402
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
