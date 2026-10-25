"""The webcam demo picks a Logitech C270 and does not talk to the drone."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run"))

from demo import Demo, camera_order, choose_camera, console_lines, dshow_video_names  # noqa: E402

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
