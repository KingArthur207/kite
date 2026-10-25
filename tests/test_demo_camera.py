"""The webcam demo picks a Logitech C270 and does not talk to the drone."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run"))

import numpy as np  # noqa: E402

import demo  # noqa: E402
from demo import (  # noqa: E402
    Demo,
    body_attempts,
    camera_order,
    choose_camera,
    console_lines,
    dshow_video_names,
    looks_like_body_camera,
    meter_for,
    open_body_reader,
    switch_word,
)

LISTING = """
[dshow @ 0000019289f56080] "USB2.0 FHD UVC WebCam" (video)
[dshow @ 0000019289f56080]   Alternative name "@device_pnp_\\\\?\\usb#vid_3277"
[dshow @ 0000019289f56080] "Logi C270 HD WebCam" (video)
[dshow @ 0000019289f56080] "OBS Virtual Camera" (none)
[dshow @ 0000019289f56080] "Microphone (Logi C270 HD WebCam)" (audio)
"""


def test_dshow_listing_keeps_video_devices_only():
    assert dshow_video_names(LISTING) == ["USB2.0 FHD UVC WebCam", "Logi C270 HD WebCam"]


def test_choose_camera_prefers_the_c270_over_the_builtin():
    names = dshow_video_names(LISTING)
    assert choose_camera(names) == "Logi C270 HD WebCam"
    assert choose_camera(["Integrated Camera", "HD Pro Webcam C920"]) == "HD Pro Webcam C920"
    assert choose_camera(["OBS Virtual Camera"]) is None
    assert choose_camera(["USB2.0 FHD UVC WebCam"]) == "USB2.0 FHD UVC WebCam"


def test_the_c270_is_the_second_device_in_the_directshow_list():
    names = dshow_video_names(LISTING)
    assert names.index(choose_camera(names)) == 1


def test_camera_order_tries_the_c270_first_and_honors_a_request():
    names = dshow_video_names(LISTING)
    assert camera_order(names, None)[0] == "Logi C270 HD WebCam"
    assert "USB2.0 FHD UVC WebCam" in camera_order(names, None)
    assert camera_order(names, "Integrated Camera") == ["Integrated Camera"]


def test_console_lines_skip_the_repeated_hint():
    scene = type("E", (), {"kind": "scene", "text": "No walkable ground ahead."})()
    said = type("E", (), {"kind": "say", "text": "Stop. The path ahead looks blocked."})()
    hazard = type("E", (), {"kind": "hazard", "text": "Stop. The path ahead looks blocked."})()
    hint = type("E", (), {"kind": "hint", "text": "stop"})()
    assert console_lines([scene, hazard, said, hint]) == (
        "No walkable ground ahead.",
        "Stop. The path ahead looks blocked.",
    )


def test_flight_words_do_not_leave_the_camera_window():
    logs = []
    demo = Demo(type("A", (), {"camera": "", "device": ""})(), lambda text, kind="info": logs.append((kind, text)))
    demo.send("takeoff")
    demo.send("land")
    demo.send("e")
    assert demo.state == "emergency"
    assert demo.snap.state == "emergency"
    assert ("info", "this window is the camera") in logs
    assert not hasattr(demo, "drone") or demo.drone is None


def _demo(logs, **flags):
    args = type("A", (), {"camera": "", "device": "", **flags})()
    return Demo(args, lambda text, kind="info": logs.append((kind, text)))


def test_switch_words():
    assert switch_word("seg") == ("seg", None)
    assert switch_word("segmentation off") == ("seg", False)
    assert switch_word("yolo on") == ("yolo", True)
    assert switch_word("yolo maybe") is None
    assert switch_word("status") is None
    assert meter_for(True, True) == "seg+yolo"
    assert meter_for(False, True) == "yolo"
    assert meter_for(True, False) == "seg"


def test_the_window_opens_on_the_body_camera_with_yolo_only():
    demo_ = _demo([])
    assert [(t["key"], t["on"]) for t in demo_.toggles()] == [
        ("body", True), ("drone", False), ("yolo", True), ("seg", False),
    ]
    demo_ = _demo([], seg=True, no_yolo=True)
    assert [(t["key"], t["on"]) for t in demo_.toggles()] == [
        ("body", True), ("drone", False), ("yolo", False), ("seg", True),
    ]


def _loaders(monkeypatch, calls, seg=None):
    def yolo(device=""):
        calls.append("yolo")
        return type("P", (), {"segmenter": None, "lights": None})()

    def segmentation(device=""):
        calls.append("seg")
        if seg is not None:
            raise seg
        return "segmenter", "lights"

    monkeypatch.setattr(demo, "load_perception", yolo)
    monkeypatch.setattr(demo, "load_segmentation", segmentation)


def test_both_models_load_before_anything_runs(monkeypatch):
    calls = []
    _loaders(monkeypatch, calls)
    logs = []
    demo_ = _demo(logs)
    yolo = next(t for t in demo_.toggles() if t["key"] == "yolo")
    assert yolo["note"] == "loading"
    demo_.send("seg")
    assert calls == []  # a toggle never loads anything
    assert demo_._load_models()
    assert calls == ["yolo", "seg"]
    assert (demo_._perception.segmenter, demo_._perception.lights) == ("segmenter", "lights")
    assert [t["note"] for t in demo_.toggles()] == ["", "", "", ""]
    assert ("info", "models ready; on: YOLO, segmentation") in logs
    demo_.toggle("seg")
    demo_.toggle("seg")
    assert calls == ["yolo", "seg"]


def test_a_failed_segmentation_load_leaves_yolo_running(monkeypatch):
    calls = []
    _loaders(monkeypatch, calls, seg=FileNotFoundError("no weights"))
    logs = []
    demo_ = _demo(logs, seg=True)
    assert demo_._load_models()
    notes = {t["key"]: t["note"] for t in demo_.toggles()}
    assert notes["seg"] == "unavailable"
    assert notes["yolo"] == ""
    assert ("error", "segmentation unavailable: no weights") in logs


def test_raw_frames_come_back_when_guidance_is_off():
    demo_ = _demo([])
    demo_._publish("painted", True)
    demo_._publish("raw", False)
    assert demo_.last_frame.image == "painted"
    demo_.send("yolo off")
    demo_._publish("raw", False)
    assert demo_.last_frame.image == "raw"
    demo_._publish("late paint", True)
    assert demo_.last_frame.image == "raw"


def test_the_chest_webcam_goes_around_a_person_close_in_front(monkeypatch):
    demo._vendor_path()
    import streetseg.models
    from streetseg.types import Box

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return self

        def predict(self, frame):  # someone right in front, cut off by the bottom edge
            return [Box("person", 0.9, (300.0, 100.0, 580.0, 480.0))]

    monkeypatch.setattr(streetseg.models, "YoloDetector", FakeDetector)
    perception = demo.load_perception("")
    from streetseg.camera import Pose

    assert perception.cfg.exclude_wearer is False
    frame = np.zeros((480, 640, 3), np.uint8)
    events = perception.process(frame, demo.camera_for(frame.shape, 78.0), Pose(demo.WEBCAM_HEIGHT_M, demo.WEBCAM_PITCH_DEG))
    assert perception.cue.reason == "obstacle"
    assert perception.cue.speech in ("Bear left to go around the person.", "Bear right to go around the person.")
    assert any(e.kind == "say" for e in events)


def test_a_failed_import_logs_the_error_underneath(monkeypatch):
    try:
        try:
            raise ImportError("cannot import name 'dispatch_model'")
        except ImportError as exc:
            raise RuntimeError("Could not import module 'SegformerForSemanticSegmentation'") from exc
    except RuntimeError as wrapped:
        error = wrapped
    calls = []
    _loaders(monkeypatch, calls, seg=error)
    logs = []
    demo_ = _demo(logs)
    demo_._load_models()
    assert ("error", "segmentation unavailable: Could not import module 'SegformerForSemanticSegmentation' "
            "(ImportError: cannot import name 'dispatch_model')") in logs


class _Board:
    def __init__(self, camera="usb camera /dev/video2", pattern=False, name="unoq 10.31.226.210"):
        self.camera = camera
        self.test_pattern = pattern
        self.name = name
        self.released = False

    def release(self):
        self.released = True


def test_a_real_arduino_camera_is_the_bodycam(monkeypatch):
    board = _Board()
    monkeypatch.setattr(demo, "open_unoq", lambda spec, timeout: board)
    seen = []

    def webcam():
        raise AssertionError("webcam")

    reader, note = open_body_reader("", lambda text, kind="info": seen.append((kind, text)), webcam)
    assert reader is board and note == ""
    assert seen == []
    assert not board.released


def test_the_bodycam_falls_back_to_the_webcam_when_the_board_is_down():
    calls = []

    def down(spec, timeout):
        calls.append(spec)
        raise SystemExit("cannot reach the UNO Q")

    webcam = _Board(name="Integrated Camera")
    logs = []
    reader, note = open_body_reader("", lambda text, kind="info": logs.append((kind, text)), lambda: webcam, unoq=down)
    assert calls == ["unoq:10.31.226.210", "unoq"]
    assert reader is webcam and note == "webcam"
    assert logs[-1] == ("warn", "body camera: using the webcam")


def test_a_missing_token_does_not_try_the_board_twice():
    calls = []

    def down(spec, timeout):
        calls.append(spec)
        raise SystemExit("no token on this computer yet")

    webcam = _Board(name="webcam")
    reader, note = open_body_reader("", lambda *_a, **_k: None, lambda: webcam, unoq=down)
    assert calls == ["unoq:10.31.226.210"]
    assert reader is webcam and note == "webcam"


def test_the_test_pattern_is_not_the_body_camera():
    board = _Board(camera="synthetic", pattern=True)
    webcam = _Board(name="Logi C270 HD WebCam")
    reader, note = open_body_reader(
        "10.31.226.210", lambda *_a, **_k: None, lambda: webcam, unoq=lambda spec, timeout: board,
    )
    assert board.released
    assert reader is webcam and note == "webcam"
    assert body_attempts("10.31.226.210") == [("unoq:10.31.226.210", 1.5)]


def test_logitech_and_lenovo_count_as_the_body_camera():
    assert looks_like_body_camera("usb camera /dev/video2")
    assert looks_like_body_camera("Logi C270 HD WebCam")
    assert looks_like_body_camera("Lenovo EasyCamera")
    assert looks_like_body_camera("UVC Camera (046d:0825)")
    assert not looks_like_body_camera("synthetic")
    assert not looks_like_body_camera("")
    assert not looks_like_body_camera("unknown")


def test_the_stream_buttons_pick_one_view():
    logs = []
    demo_ = _demo(logs)
    demo_.send("drone")
    assert demo_._view == "drone"
    assert ("info", "DRONE VIEW") in logs
    assert [(t["key"], t["on"], t["note"]) for t in demo_.toggles()[:2]] == [
        ("body", False, ""), ("drone", True, "opening"),
    ]
    demo_.toggle("drone")  # already on: stays, the grab thread is what opens it
    assert demo_._view == "drone"
    demo_.send("bodycam")
    assert demo_._view == "body"
    assert demo_.toggles()[0]["note"] == "opening"
    demo_.send("takeoff")
    assert ("info", "this window is the camera") in logs


def test_a_painted_frame_from_the_other_stream_is_dropped():
    demo_ = _demo([])
    demo_._live = "drone"
    demo_._publish("body overlay", True, "body")
    assert demo_.last_frame is None
    demo_._publish("drone overlay", True, "drone")
    assert demo_.last_frame.image == "drone overlay"


def test_drone_view_reports_unavailable_without_opening_a_camera(monkeypatch):
    def boom():
        raise RuntimeError("no route to the drone")

    monkeypatch.setattr(demo, "open_drone_reader", boom)
    logs = []
    demo_ = _demo(logs)
    assert demo_._open_view("drone") is None
    assert demo_._drone_note == "unavailable"
    assert logs[-1][0] == "error"
