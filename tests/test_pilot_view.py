"""BODYCAM and DRONE VIEW on the flying app: a picture switch. The follow keeps the drone frame.

Nothing here opens a camera, a drone, or a window. The body-camera opener is replaced.
"""

from __future__ import annotations

import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run"))

import demo  # noqa: E402
from pilot import Pilot  # noqa: E402


class _Picture:
    def copy(self):
        return "frame"


class _Reader:
    def __init__(self, name="unoq 10.31.226.210 usb camera /dev/video2 (Logitech)"):
        self.name = name
        self.released = False
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.reads == 1:
            return True, _Picture()
        return False, None

    def release(self):
        self.released = True


def _pilot(logs, board="", camera=""):
    return Pilot(type("A", (), {"board": board, "camera": camera})(),
                 lambda text, kind="info": logs.append((kind, text)))


def _wait(pred, seconds=2.0):
    end = time.time() + seconds
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_the_flying_app_starts_on_the_drone_picture():
    logs = []
    pilot = _pilot(logs)
    drone = type("F", (), {"image": "drone", "seq": 4})()
    pilot.last_frame = drone
    shown = {t["key"]: t["on"] for t in pilot.toggles()}
    assert shown == {"body": False, "drone": True}
    assert pilot.display_frame() is drone


def test_bodycam_and_drone_switch_the_picture_and_leave_flight_commands_alone():
    logs = []
    pilot = _pilot(logs)
    pilot.send("bodycam")
    assert pilot._view == "body"
    assert pilot._body_note == "opening"
    assert pilot._cmds.empty()
    pilot.toggle("drone")
    assert pilot._view == "drone"
    pilot.state = "idle"
    pilot.send("takeoff")
    assert pilot._cmds.get_nowait() == "takeoff"
    assert pilot._view == "drone"


def test_a_picture_switch_works_while_the_drone_is_offline():
    logs = []
    pilot = _pilot(logs)
    pilot.closed.set()
    pilot.send("body")
    assert pilot._view == "body"
    pilot.send("takeoff")
    assert pilot._cmds.empty()
    assert any("not connected" in text for _kind, text in logs)


def test_bodycam_shows_the_arduino_frame_and_keeps_the_drone_frame(monkeypatch):
    """The pane can show the body camera. last_frame, which takeoff and the follow read, stays the drone's."""
    reader = _Reader()
    seen = {}

    def open_body(board, log, webcam, unoq=None):
        seen["board"] = board
        seen["camera"] = webcam().name
        return reader, ""

    def webcam(requested):
        assert requested == "C270"
        return type("W", (), {"name": "webcam-not-used"})()

    monkeypatch.setattr(demo, "open_body_reader", open_body)
    monkeypatch.setattr(demo, "open_webcam_reader", webcam)
    logs = []
    pilot = _pilot(logs, board="10.31.226.210", camera="C270")
    drone = type("F", (), {"image": "drone", "seq": 7})()
    pilot.last_frame = drone
    pilot.send("bodycam")
    thread = threading.Thread(target=pilot._body_loop)
    thread.start()
    assert _wait(lambda: pilot._body_frame is not None)
    assert seen == {"board": "10.31.226.210", "camera": "webcam-not-used"}
    assert pilot._body_frame.image == "frame"
    assert pilot._body_frame.overlay is False
    assert pilot.display_frame() is pilot._body_frame
    assert pilot.last_frame is drone
    pilot.send("drone")
    assert _wait(lambda: reader.released)
    assert pilot.display_frame() is drone
    assert pilot.last_frame is drone
    pilot._stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_a_missing_body_camera_is_marked_unavailable_and_the_drone_picture_stays(monkeypatch):
    def open_body(board, log, webcam, unoq=None):
        pilot._stop.set()
        return None, "webcam"

    monkeypatch.setattr(demo, "open_body_reader", open_body)
    logs = []
    pilot = _pilot(logs)
    drone = type("F", (), {"image": "drone", "seq": 1})()
    pilot.last_frame = drone
    pilot._view = "body"
    pilot._body_loop()
    assert pilot._body_note == "unavailable"
    assert pilot.last_frame is drone
    assert pilot.display_frame() is drone


def test_the_laptop_webcam_is_the_fallback_and_the_button_says_so(monkeypatch):
    reader = _Reader(name="Logi C270")
    seen = {}

    def open_body(board, log, webcam, unoq=None):
        seen["board"] = board
        return webcam(), "webcam"

    def webcam(requested):
        seen["requested"] = requested
        return reader

    monkeypatch.setattr(demo, "open_body_reader", open_body)
    monkeypatch.setattr(demo, "open_webcam_reader", webcam)
    logs = []
    pilot = _pilot(logs, camera="Logi C270")
    pilot._view = "body"
    thread = threading.Thread(target=pilot._body_loop)
    thread.start()
    assert _wait(lambda: pilot._body_frame is not None)
    assert seen == {"board": "", "requested": "Logi C270"}
    assert pilot._body_note == "webcam"
    notes = {t["key"]: t["note"] for t in pilot.toggles()}
    assert notes["body"] == "webcam"
    assert pilot.last_frame is None
    pilot._stop.set()
    thread.join(timeout=2)
    assert reader.released
