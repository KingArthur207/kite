"""tests/drone_check.py's sequence against the protocol-level Tello emulator: real djitellopy over real UDP, kite's
TelloDrone + SafetyGovernor, the emulator's camera, no window."""

from __future__ import annotations

import os
import sys
import threading
import time
from types import SimpleNamespace


from kite.testing import EmulatorCamera
from kite.types import Frame

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from drone_check import Check  # noqa: E402

from .conftest import open_rig  # noqa: E402

MOTION = ("takeoff", "land", "emergency", "rc ", "forward", "back", "left", "right", "up", "down", "cw", "ccw", "stop")


def args(dry_run: bool, **kw) -> SimpleNamespace:
    a = dict(dry_run=dry_run, link_only=False, no_moves=False, mag=40, dur=0.6, settle=0.4, move_cm=30, min_battery=30)
    a.update(kw)
    return SimpleNamespace(**a)


# Where the camera direction check is run: 2 m south of the takeoff point facing 210 deg, 3-6 m from a cluster of
# poles. From the takeoff pose the nearest poles are 6-11 m away over a uniform ground and sky: too little parallax
# for 30 cm up / down moves (the camera says "inconclusive" there, never the wrong way).
SCENE = {"north": -2.0, "east": 0.0, "heading": 210.0}
OVERLAY_ROWS = 22  # the emulator stamps "h .. hdg .." in the bottom rows of its 240-row view


class NoOverlay:
    """The emulator camera without its text overlay: real video has none, and static text holds the flow at zero."""

    def __init__(self, cam):
        self.cam, self._seq, self._out = cam, None, None

    def read(self):
        f = self.cam.read()
        if f is not None and f.seq != self._seq:
            img = f.image.copy()
            img[-OVERLAY_ROWS:] = img[-OVERLAY_ROWS - 1]
            self._seq, self._out = f.seq, Frame(img, f.t, f.seq, f.source)
        return self._out


def run_check(rig, a, abort_after_s: float | None = None, scene: dict | None = None) -> Check:
    """scene: place the drone there before takeoff and again before the moves (the rc yaw steps leave its heading
    some degrees off), and hide the overlay from the camera check."""
    cam = EmulatorCamera(rig.emu).start()  # the real drone's video is started by connect()
    try:
        assert cam.wait_first(5.0) is not None
        check = Check(rig.gov, NoOverlay(cam) if scene else cam, a)
        if scene:
            rig.emu.place(**scene)
            moves = check.moves

            def aimed_moves():
                rig.emu.place(**scene)
                time.sleep(0.2)  # let the camera draw the new view before the next step's hover baseline
                moves()

            check.moves = aimed_moves
        if abort_after_s is not None:
            threading.Timer(abort_after_s, check.abort.set).start()
        check.run()
    finally:
        cam.stop()
    return check


def invert_fb_lr(emu) -> None:
    """Make the emulated drone fly forward / back and left / right the opposite way of every command."""
    on_rc, start = emu._on_rc, emu._start_discrete
    swap = {"forward": "back", "back": "forward", "left": "right", "right": "left"}

    def rc(args):
        try:
            lr, fb, ud, yaw = (int(x) for x in args)
        except ValueError:
            return on_rc(args)
        return on_rc([str(-lr), str(-fb), str(ud), str(yaw)])

    emu._on_rc = rc
    emu._start_discrete = lambda w, n, addr: start(swap.get(w, w), n, addr)


def camera_says(check: Check, step: str) -> str:
    detail = next(d for n, _, d in check.results if n == step)
    return detail.split("camera: ")[1].split(" (")[0]


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


def test_flight_test_flies_every_axis_and_move_and_lands(capsys):
    with open_rig(time_scale=4.0) as rig:
        check = run_check(rig, args(dry_run=False), scene=SCENE)
        snap = rig.emu.snapshot()
        sent = rig.emu.commands()
    out = capsys.readouterr().out
    assert failures(check) == [], failures(check)
    assert "stand BEHIND the drone" in out.split("PASS  takeoff")[0]  # told before the takeoff
    for d in ("forward", "back", "left", "right", "up", "down"):  # the drone's own camera agrees with every move
        assert camera_says(check, f"move {d} 30") == d, next(x for x in check.results if x[0] == f"move {d} 30")
    assert check.direction_verdict.startswith("the drone moves as commanded"), out
    assert "DIRECTION SUMMARY" in out and "fb  " in out and "VERDICT: the drone moves as commanded" in out
    video = next(d for n, _, d in check.results if n == "video")
    assert float(video.split()[0]) >= 22  # the emulator camera (30 fps), counted by frame number, not by our polling
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


def test_flight_test_catches_a_drone_that_really_flies_fb_and_lr_backwards(capsys):
    with open_rig(time_scale=4.0) as rig:
        invert_fb_lr(rig.emu)
        check = run_check(rig, args(dry_run=False), scene=SCENE)
        snap = rig.emu.snapshot()
    out = capsys.readouterr().out
    assert check.direction_verdict == "the drone really moves opposite on fb, lr", out
    failed = {n for n, _ in failures(check)}
    assert {"move forward 30", "move back 30", "move left 30", "move right 30"} <= failed, out
    assert camera_says(check, "move forward 30") == "BACK - opposite of the command"
    assert not failed & {"move up 30", "move down 30", "rotate +45", "rotate -45", "land"}, out
    assert not snap.flying


def test_camera_motion_reads_each_direction_from_the_rendered_view():
    """The optical-flow reading on its own, on the emulator's rendered view (no flight): each direction, and
    'inconclusive' (never a guess) when the view has nothing to track."""
    import math

    import numpy as np

    from drone_check import CameraMotion
    from kite.testing.tello_emulator import Snapshot, render_view

    pose = dict(t_sim=0, phase="air", maneuver=None, motors=True, h=0.8, v_fwd=0, v_right=0, v_up=0, yaw_rate=0,
                tilt_fwd_deg=0, tilt_right_deg=0, sticks=(0, 0, 0, 0), battery=90, templ=50, temph=55, motor_time=0,
                speed_cm_s=50, sdk_mode=True, stream_on=True, last_cmd_age_sim=0)
    psi = math.radians(SCENE["heading"])

    def view(fwd=0.0, right=0.0, up=0.0, blank=False):
        img = render_view(Snapshot(**pose, heading=SCENE["heading"], north=SCENE["north"] + fwd * math.cos(psi)
                                   - right * math.sin(psi), east=SCENE["east"] + fwd * math.sin(psi)
                                   + right * math.cos(psi)) if not up else
                          Snapshot(**{**pose, "h": 0.8 + up}, heading=SCENE["heading"], north=SCENE["north"],
                                   east=SCENE["east"]), (320, 240))
        img[-OVERLAY_ROWS:] = img[-OVERLAY_ROWS - 1]
        return np.full_like(img, 90) if blank else img

    for step, axis, sign, motion in [("forward", "fb", 1, {"fwd": .3}), ("back", "fb", -1, {"fwd": -.3}),
                                     ("right", "lr", 1, {"right": .3}), ("left", "lr", -1, {"right": -.3}),
                                     ("up", "ud", 1, {"up": .3}), ("down", "ud", -1, {"up": -.3})]:
        for commanded in (sign, -sign):
            cm = CameraMotion()
            for _ in range(3):
                cm.add(view())
            cm.mark()
            for i in range(1, 9):
                cm.add(view(**{k: v * i / 8 for k, v in motion.items()}))
            v = cm.verdict(axis, commanded)
            assert v["measured"] == step, (step, commanded, v)
    cm = CameraMotion()
    for _ in range(10):
        cm.add(view(blank=True))
    cm.mark()
    for _ in range(10):
        cm.add(view(blank=True))
    v = cm.verdict("fb", 1)
    assert v["measured"] is None and "texture" in v["why"]
