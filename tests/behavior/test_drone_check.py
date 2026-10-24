"""tests/drone_check.py's sequence against the protocol-level Tello emulator: real djitellopy over real UDP, kite's
TelloDrone + SafetyGovernor, the emulator's camera, no window."""

from __future__ import annotations

import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from kite.testing import EmulatorCamera

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from drone_check import Check  # noqa: E402

from .conftest import open_rig  # noqa: E402

MOTION = ("takeoff", "land", "emergency", "rc ", "forward", "back", "left", "right", "up", "down", "cw", "ccw", "stop")


def args(dry_run: bool, **kw) -> SimpleNamespace:
    a = dict(dry_run=dry_run, link_only=False, no_moves=False, mag=40, dur=0.6, settle=0.4, move_cm=30, min_battery=30)
    a.update(kw)
    return SimpleNamespace(**a)


def run_check(rig, a, abort_after_s: float | None = None) -> Check:
    cam = EmulatorCamera(rig.emu).start()  # the real drone's video is started by connect()
    try:
        assert cam.wait_first(5.0) is not None
        check = Check(rig.gov, cam, a)
        if abort_after_s is not None:
            threading.Timer(abort_after_s, check.abort.set).start()
        check.run()
    finally:
        cam.stop()
    return check


def failures(check: Check) -> list:
    return [(n, d) for n, ok, d in check.results if not ok]


def test_dry_test_passes_every_step_and_sends_no_motion():
    with open_rig(drone={"dry_run": True}) as rig:
        check = run_check(rig, args(dry_run=True))
        sent = rig.emu.commands()
    names = [n for n, _, _ in check.results]
    assert failures(check) == []
    assert {"command round trip", "video", "telemetry", "battery", "takeoff", "rc yaw right", "rc down",
            "move forward 30", "move down 30", "rotate +45", "rotate -45"} <= set(names)
    assert not [c for c in sent if c.startswith(MOTION)], sent  # dry run: nothing reached the motors


def test_flight_test_flies_every_axis_and_move_and_lands():
    with open_rig(time_scale=4.0) as rig:
        check = run_check(rig, args(dry_run=False))
        snap = rig.emu.snapshot()
        sent = rig.emu.commands()
    assert failures(check) == [], failures(check)
    video = next(d for n, _, d in check.results if n == "video")
    assert float(video.split()[0]) >= 22  # the emulator camera (~25 fps here), counted by frame number, not by our polling
    for cmd in ("takeoff", "forward 30", "back 30", "left 30", "right 30", "up 30", "down 30", "cw 45", "ccw 45", "land"):
        assert cmd in sent, (cmd, sent)
    assert not snap.flying and ("land", True, "down") in check.results


def test_ctrl_c_mid_flight_lands():
    with open_rig(time_scale=4.0) as rig:
        t0 = time.monotonic()
        check = run_check(rig, args(dry_run=False), abort_after_s=7.0)  # after the link checks and the takeoff
        snap = rig.emu.snapshot()
        sent = rig.emu.commands()
    assert "takeoff" in sent and ("land", True, "down") in check.results and not snap.flying
    assert not any(n.startswith("move") for n, _, _ in check.results)  # stopped before the moves
    assert time.monotonic() - t0 < 40


def test_takeoff_needs_live_video():
    class NoVideo:
        def read(self):
            return None

    with open_rig() as rig:
        check = Check(rig.gov, NoVideo(), args(dry_run=False, link_only=False))
        check.run()
        sent = rig.emu.commands()
    assert ("takeoff", False, "no live video: the safety governor would only hover") in check.results
    assert "takeoff" not in sent
