"""run/epic.py flies through the pilot. The demo's YOLO and segmentation only paint a copy of the picture."""

from __future__ import annotations

import importlib.util
import queue
import sys
import types
from pathlib import Path

import numpy as np
import pytest

RUN = Path(__file__).resolve().parent.parent / "run"


@pytest.fixture
def epic(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("kite_run_epic_under_test", RUN / "epic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fakes(monkeypatch):
    seen = {"pilots": [], "apps": [], "logs": []}

    class Pilot:
        def __init__(self, args, log):
            seen["pilots"].append((args, log))

    class App:
        def __init__(self, make_pilot, title, hint=None, opening=None):
            self.make_pilot, self.title, self.hint, self.opening = make_pilot, title, hint, opening
            seen["apps"].append(self)

        def log(self, msg, kind):
            seen["logs"].append((msg, kind))

        def run(self):
            return 7

    monkeypatch.setitem(sys.modules, "pilot", types.SimpleNamespace(Pilot=Pilot))
    monkeypatch.setitem(sys.modules, "ui", types.SimpleNamespace(App=App))
    return seen


def _open(epic):
    logs = []
    args = epic.build_parser().parse_args([])
    window = epic.Epic(args, lambda text, kind: logs.append((kind, text)))
    return window, logs


def test_scene_text_does_not_steer_and_the_follow_frame_stays(epic):
    """A guidance line, even the word takeoff, is logged. It is not queued, and the painted copy is not last_frame."""
    window, logs = _open(epic)
    source = np.zeros((8, 8, 3), np.uint8)
    window.pilot.last_frame = types.SimpleNamespace(image=source, seq=1)
    window.pilot._view = "drone"
    view, copied = window._raw_copy()

    class Perception:
        def set_modes(self, seg, yolo):
            self.modes = (seg, yolo)

        def process(self, frame, camera, pose):
            frame[:] = 9
            self.image = np.full_like(frame, 3)
            return [types.SimpleNamespace(kind="say", text="takeoff")]

    window._perception = Perception()
    window._loaded = True
    window._seg_state = "ready"
    window._intrinsics = lambda image, view: (None, None)
    window._on_frame(view, copied)

    assert window.pilot._cmds.empty()
    assert int(source.sum()) == 0
    assert window.pilot.last_frame.image is source
    shown = window.display_frame()
    assert shown is not window.pilot.last_frame
    assert int(shown.image.mean()) == 3
    assert ("info", "takeoff") in logs
    assert ("sent", "takeoff") not in logs
    assert window.pilot.drone is None and window.pilot.follow is None


def test_flight_words_reach_the_pilot_and_processing_words_do_not(epic):
    """hold and land go to the pilot. yolo and seg change only the picture switches."""
    window, logs = _open(epic)
    frame = window.pilot.last_frame
    window.send("hold")
    window.request_land()
    window.send("yolo off")
    window.send("seg on")
    window.toggle("body")
    queued = []
    while True:
        try:
            queued.append(window.pilot._cmds.get_nowait())
        except queue.Empty:
            break
    assert queued == ["hold", "land"]
    assert window.pilot.last_frame is frame
    assert window.pilot._view == "body"
    assert window._yolo_on is False and window._seg_on is True
    assert ("sent", "yolo off") in logs
    assert ("sent", "hold") in logs


def test_body_picture_is_not_the_follow_frame(epic):
    """The processor reads the body camera as a copy. last_frame stays the drone picture."""
    window, _logs = _open(epic)
    drone = np.zeros((4, 4, 3), np.uint8)
    body = np.full((4, 4, 3), 9, np.uint8)
    window.pilot.last_frame = types.SimpleNamespace(image=drone, seq=1)
    window.pilot._body_frame = types.SimpleNamespace(image=body, seq=2)
    window.pilot._view = "body"
    view, copied = window._raw_copy()
    copied[:] = 0
    assert view == "body"
    assert int(body.mean()) == 9
    assert window.pilot.last_frame.image is drone

    class Perception:
        def set_modes(self, seg, yolo):
            pass

        def process(self, frame, camera, pose):
            self.image = frame
            return []

    window._perception = Perception()
    window._intrinsics = lambda image, view: (None, None)
    window._on_frame(view, copied)
    shown = window.display_frame()
    assert shown.view == "body" and shown.overlay is False
    assert window.pilot.last_frame.image is drone


def test_pane_rate_is_not_written_onto_the_pilot(epic):
    window, _logs = _open(epic)
    window._pane_meter = True
    window._det_fps = 12
    window._meter = "seg+yolo"
    assert window.snap.det_fps == 12 and window.snap.meter == "seg+yolo"
    assert window.pilot.snap.det_fps == 0 and window.pilot.snap.yaw == ""


def test_shutdown_stops_the_picture_and_lands_through_the_pilot(epic):
    window, _logs = _open(epic)
    seen = {}
    window.pilot.shutdown = lambda land=True, timeout_s=30: seen.update(land=land)
    window._pilot_started = True
    window._perception = types.SimpleNamespace(close=lambda: seen.update(closed=True))
    window.shutdown(land=True)
    assert window._stop.is_set()
    assert seen == {"closed": True, "land": True}


def test_picture_models_load_before_the_pilot_thread(epic):
    """Transformers is imported before the flight thread imports its YOLO, and a failed picture still lets it fly."""
    window, _logs = _open(epic)
    order = []

    def load():
        order.append("load")
        assert window._pilot_started is False
        return True

    def start_pilot():
        order.append("pilot")
        window.pilot.state = "idle"
        window._stop.set()

    window._load = load
    window.pilot.start = start_pilot
    window._run()
    assert order == ["load", "pilot"]
    assert window._pilot_started is True


def test_a_failed_picture_model_still_starts_the_pilot(epic):
    window, _logs = _open(epic)
    started = []
    window._load = lambda: False

    def start_pilot():
        started.append(window._pilot_started)
        window.pilot.state = "offline"
        window._stop.set()

    window.pilot.start = start_pilot
    window._run()
    assert started == [False]
    assert window._pilot_started is True


def test_flags_cover_flight_and_the_picture(epic):
    args = epic.build_parser().parse_args(["--dry-run", "--yaw", "law", "--seg", "--no-yolo", "--device", "cpu"])
    assert (args.dry_run, args.yaw, args.seg, args.no_yolo, args.device) == (True, "law", True, True, "cpu")


def test_unsafe_follow_altitude_refuses_to_start(epic, fakes, capsys):
    assert epic.main(["--height", "0.7", "--walker-height", "0.4"]) == 2
    assert "0.8-2.2" in capsys.readouterr().out
    assert fakes["apps"] == [] and fakes["pilots"] == []


def test_main_hands_flight_args_to_the_pilot(epic, fakes):
    assert epic.main(["--behind", "2.0", "--yaw", "pd", "--seg"]) == 7
    window = fakes["apps"][0]
    assert window.title == "kite"
    assert "yolo" in window.hint and "takeoff" in window.hint
    window.make_pilot("the-log")
    (args, log), = fakes["pilots"]
    assert log == "the-log" and args.behind == 2.0 and args.yaw == "pd" and args.seg is True
    assert fakes["logs"][0] == ("follow at 1.90 m, 2.0 m behind, steering: pd", "info")
    assert fakes["logs"][1][0] == "camera processing is the picture only: it does not steer"
