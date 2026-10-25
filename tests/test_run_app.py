"""run/app.py, the main app's entry point, without a drone, a GUI, djitellopy or av: the flags and their defaults,
the follow-altitude checks that refuse to start, and the hand-off to run/ui.App and run/pilot.Pilot (both replaced by
fakes here, so no window opens and nothing connects)."""

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import pytest

from kite.config import VIDEO_BACKENDS

RUN = Path(__file__).resolve().parent.parent / "run"


@pytest.fixture
def app(monkeypatch):
    """run/app.py imported under a private name (it puts run/ on sys.path itself; undone after the test)."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("kite_run_app_under_test", RUN / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fakes(monkeypatch):
    """Fake pilot / ui / av / djitellopy modules: records what main() hands to them."""
    seen = {"pilots": [], "apps": [], "logs": [], "av_level": None, "tello_level": None}

    class Pilot:
        def __init__(self, args, log):
            seen["pilots"].append((args, log))

    class App:
        def __init__(self, make_pilot, title):
            self.make_pilot, self.title = make_pilot, title
            seen["apps"].append(self)

        def log(self, msg, kind):
            seen["logs"].append((msg, kind))

        def run(self):
            return 7

    class Logger:
        def setLevel(self, level):
            seen["tello_level"] = level

    av_logging = types.SimpleNamespace(PANIC=0, set_level=lambda lv: seen.__setitem__("av_level", lv))
    av = types.ModuleType("av")
    av.logging = av_logging
    monkeypatch.setitem(sys.modules, "pilot", types.SimpleNamespace(Pilot=Pilot))
    monkeypatch.setitem(sys.modules, "ui", types.SimpleNamespace(App=App))
    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setitem(sys.modules, "av.logging", av_logging)
    monkeypatch.setitem(sys.modules, "djitellopy", types.SimpleNamespace(Tello=types.SimpleNamespace(LOGGER=Logger())))
    monkeypatch.delenv("OPENCV_FFMPEG_LOGLEVEL", raising=False)
    return seen


def test_defaults_are_the_flown_configuration(app):
    """No flags: live (not dry run), the fruit fly brain on yaw, 1.7 m walker, 1.8 m behind, no orbit, no speech."""
    a = app.build_parser().parse_args([])
    assert (a.dry_run, a.yaw, a.walker_height, a.height, a.behind) == (False, "fly", 1.7, None, 1.8)
    assert (a.orbit, a.speak, a.video_backend, a.weights, a.yolo) == (False, False, None, None, False)


def test_flags_parse(app):
    """Every flag reaches the namespace with its type."""
    a = app.build_parser().parse_args(["--dry-run", "--yaw", "law", "--walker-height", "1.62", "--height", "2.0",
                                       "--behind", "2.5", "--orbit", "--speak", "--video-backend", "raw",
                                       "--weights", "w.pt", "--yolo"])
    assert (a.dry_run, a.yaw, a.walker_height, a.height, a.behind) == (True, "law", 1.62, 2.0, 2.5)
    assert (a.orbit, a.speak, a.video_backend, a.weights, a.yolo) == (True, True, "raw", "w.pt", True)


@pytest.mark.parametrize("argv", [["--yaw", "autopilot"], ["--video-backend", "gstreamer"], ["--height", "tall"]])
def test_bad_flags_are_usage_errors(app, argv, capsys):
    """Unknown steering, an unknown video reader or a non-number height: argparse exits 2."""
    with pytest.raises(SystemExit) as e:
        app.build_parser().parse_args(argv)
    assert e.value.code == 2


def test_video_backend_choices_come_from_kite_config(app):
    """The app offers exactly kite.config.VIDEO_BACKENDS."""
    for b in VIDEO_BACKENDS:
        assert app.build_parser().parse_args(["--video-backend", b]).video_backend == b


@pytest.mark.parametrize("argv,why", [
    (["--height", "0.7", "--walker-height", "0.4"], "0.8-2.2"),     # below the governor's floor margin
    (["--height", "2.3"], "0.8-2.2"),                                # above it
    (["--height", "1.8"], "at least --walker-height + 0.2"),         # at head height: cannot read the distance
    (["--walker-height", "2.05"], "at least --walker-height + 0.2"),  # default height capped at 2.2 < 2.25
])
def test_unsafe_follow_altitude_refuses_to_start(app, fakes, argv, why, capsys):
    """An altitude outside 0.8-2.2 m or not 0.2 m above the head exits 2 before anything is built."""
    assert app.main(argv) == 2
    assert why in capsys.readouterr().out
    assert fakes["apps"] == [] and fakes["pilots"] == []


@pytest.mark.parametrize("walker", [1.5, 1.7, 1.75, 1.81, 1.83, 1.85, 1.87, 1.9, 2.0])
def test_default_height_is_walker_plus_twenty_cm_and_is_accepted(app, fakes, walker):
    """Default --height = walker + 0.2 m (to the cm): it must never be refused by the app's own check, whatever
    the float rounding of walker + 0.2 (1.81 + 0.2 = 2.0100000000000002 > round(..., 2))."""
    assert app.main(["--walker-height", str(walker), "--dry-run"]) == 7
    fakes["apps"][0].make_pilot("log")
    (args, _), = fakes["pilots"]
    assert args.height == pytest.approx(min(round(walker + 0.2, 2), 2.2))


def test_an_explicit_height_exactly_twenty_cm_above_the_head_is_accepted(app, fakes):
    """--height 2.01 for a 1.81 m walker is the documented minimum, not a refusal."""
    assert app.main(["--walker-height", "1.81", "--height", "2.01"]) == 7


def test_main_hands_the_args_to_the_pilot_and_runs_the_app(app, fakes):
    """main() builds ui.App with a Pilot factory, logs the plan once and returns App.run()'s exit code."""
    assert app.main(["--behind", "2.0", "--yaw", "pd"]) == 7
    (a,) = fakes["apps"]
    assert a.title == "kite"
    assert fakes["pilots"] == []  # the pilot is built by the app, with its log, not by main()
    a.make_pilot("the-log")
    (args, log), = fakes["pilots"]
    assert log == "the-log" and args.height == 1.9 and args.behind == 2.0 and args.yaw == "pd"
    assert fakes["logs"] == [("follow at 1.90 m, 2.0 m behind, steering: pd", "info")]


def test_dry_run_says_so_in_the_title_and_the_first_line(app, fakes):
    """--dry-run is impossible to miss: in the window title and the first log line."""
    app.main(["--dry-run"])
    assert fakes["apps"][0].title == "kite  -  DRY RUN"
    assert fakes["logs"][0][0].startswith("DRY RUN: nothing reaches the motors. ")


def test_quiet_libraries_silences_ffmpeg_av_and_djitellopy(app, fakes):
    """FFmpeg's log level is set (not overridden if the user set one), av to PANIC, djitellopy to CRITICAL."""
    app.quiet_libraries()
    assert os.environ["OPENCV_FFMPEG_LOGLEVEL"] == "-8"
    assert fakes["av_level"] == 0 and fakes["tello_level"] == logging.CRITICAL


def test_quiet_libraries_keeps_a_user_log_level_and_survives_missing_libraries(app, monkeypatch):
    """No av, no djitellopy: still fine (they are optional here); a user's OPENCV_FFMPEG_LOGLEVEL stays."""
    monkeypatch.setenv("OPENCV_FFMPEG_LOGLEVEL", "24")
    monkeypatch.setitem(sys.modules, "av", None)
    monkeypatch.setitem(sys.modules, "av.logging", None)
    monkeypatch.setitem(sys.modules, "djitellopy", None)
    app.quiet_libraries()
    assert os.environ["OPENCV_FFMPEG_LOGLEVEL"] == "24"
