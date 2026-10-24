"""tests/drone_fly.py against a fake djitellopy Tello: real TelloDrone + SafetyGovernor, scripted window keys."""

import time
from types import SimpleNamespace

import cv2
import pytest
from drone_fly import ARROWS, fly, key_name
from test_record import FakeTello, LiveImage

from kite.config import Config
from kite.drone import SafetyGovernor
from kite.drone.tello import TelloDrone

UP_MAC, UP_WIN, UP_LINUX = 63232, 2490368, 65362


class FakeGui:
    """The cv2 window calls fly() makes; waitKeyEx() returns scripted key codes (KeyboardInterrupt = Ctrl+C)."""

    WINDOW_NORMAL, WND_PROP_VISIBLE = cv2.WINDOW_NORMAL, cv2.WND_PROP_VISIBLE
    FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, script):
        self.script, self.t0, self.shown = list(script), None, 0

    def namedWindow(self, *a):
        pass

    def putText(self, *a, **k):
        pass

    def imshow(self, win, img):
        self.shown += 1

    def getWindowProperty(self, *a):
        return 1.0

    def destroyAllWindows(self):
        pass

    def waitKey(self, *a):
        return -1

    def waitKeyEx(self, *a):
        self.t0 = self.t0 or time.time()
        time.sleep(0.005)
        if time.time() - self.t0 > 20.0:  # backstop: a script that never ends the session is a failing test, not a hang
            raise KeyboardInterrupt
        if self.script and time.time() - self.t0 >= self.script[0][0]:
            key = self.script.pop(0)[1]
            if key is KeyboardInterrupt:
                raise KeyboardInterrupt
            return key if isinstance(key, int) else ord(key)
        return -1


def held(code, start, seconds, every=0.05):
    """A key held down: the OS repeats it every `every` s."""
    return [(start + k * every, code) for k in range(int(seconds / every))]


def drone(dry_run=False):
    fake = FakeTello()
    d = SafetyGovernor(TelloDrone(tello=fake, video=False, dry_run=dry_run), Config().safety)
    d.connect()
    time.sleep(0.25)  # first state packets
    return d, fake


def args(dry_run=False):
    return SimpleNamespace(speed=30, pulse=0.6, max_rc=50, dry_run=dry_run)


def motion(fake):
    return [s for s in fake.sent if s not in ("command", "speed 50") and not str(s).endswith("?")]


@pytest.mark.parametrize("code, name", [(UP_MAC, "up"), (UP_WIN, "up"), (UP_LINUX, "up"), (2424832, "left"),
                                        (63235, "right"), (65364, "down"), (ord("Q"), "q"), (ord("3"), "3"),
                                        (ord(" "), " "), (-1, None), (0x10FFFF, None)])
def test_key_names_cover_windows_macos_and_linux_arrows(code, name):
    assert key_name(code) == name


def test_every_arrow_code_maps_to_a_move():
    assert set(ARROWS.values()) == {"up", "down", "left", "right"}


@pytest.mark.parametrize("up", [UP_MAC, UP_WIN, UP_LINUX])
def test_takeoff_arrow_forward_then_q_lands_and_q_quits(up):
    d, fake = drone()
    gui = FakeGui([(0.3, "t")] + held(up, 1.0, 0.5) + [(2.2, "q"), (3.4, "q")])
    fly(d, LiveImage(), args(), cv2=gui)
    d.close()
    sent = motion(fake)
    assert sent[0] == "takeoff"
    assert ("rc", 0, 30, 0, 0) in sent  # forward at --speed, through the governor
    assert "land" in sent and sent.index("land") > sent.index(("rc", 0, 30, 0, 0))
    assert not d.flying and gui.shown > 50


def test_letting_go_stops_within_the_pulse():
    d, fake = drone()
    gui = FakeGui([(0.3, "t")] + held(ord("a"), 1.0, 0.3) + [(2.6, "q"), (3.6, "q")])
    fly(d, LiveImage(), args(), cv2=gui)
    d.close()
    rcs = [s for s in fake.sent if isinstance(s, tuple)]
    first_turn = rcs.index(("rc", 0, 0, 0, -30))
    assert ("rc", 0, 0, 0, 0) in rcs[first_turn:]  # back to hover after the key is released


def test_3_is_an_immediate_emergency():
    d, fake = drone()
    gui = FakeGui([(0.3, "t"), (1.0, "3"), (1.5, "q")])
    fly(d, LiveImage(), args(), cv2=gui)
    d.close()
    sent = motion(fake)
    assert "emergency" in sent and "land" not in sent[:sent.index("emergency")]
    assert not d.flying


def test_takeoff_is_refused_without_live_video():
    class NoVideo(LiveImage):
        def read(self):
            return None

    d, fake = drone()
    fly(d, NoVideo(), args(), cv2=FakeGui([(0.2, "t"), (0.8, "q")]))
    d.close()
    assert "takeoff" not in motion(fake)


def test_ctrl_c_lands_before_quitting():
    d, fake = drone()
    fly(d, LiveImage(), args(), cv2=FakeGui([(0.3, "t"), (1.2, "w"), (1.5, KeyboardInterrupt)]))
    d.close()
    sent = motion(fake)
    assert "takeoff" in sent and "land" in sent and not d.flying


def test_dry_run_never_sends_motion():
    d, fake = drone(dry_run=True)
    # dry-run takeoff "completes" after 5 s and landing after 4 s (TelloDrone's expected durations)
    gui = FakeGui([(0.3, "t")] + held(UP_WIN, 5.6, 0.5) + held(ord("d"), 6.2, 0.3) + [(6.8, "q"), (11.5, "q")])
    t0 = time.time()
    fly(d, LiveImage(), args(dry_run=True), cv2=gui)
    d.close()
    assert motion(fake) == []
    assert time.time() - t0 < 15  # ended on the second q, not on the backstop
