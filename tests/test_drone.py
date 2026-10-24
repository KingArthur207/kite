"""Tello adapter (against a fake djitellopy) and the safety governor (against a fake drone).

Ported from vigil's tests/test_drone.py and test_review_fixes_9_12.py; the governor tests ran on vigil's
simulator there and use FakeDrone here.
"""

import socket
import threading
import time

import numpy as np
import pytest

from kite.config import Config, DroneCfg, SafetyCfg
from kite.drone import Drone, SafetyGovernor, make_tello
from kite.drone import tello as tello_mod
from kite.drone.tello import TelloDrone
from kite.types import Telemetry


class FakeTello:
    """Mimics the djitellopy surface TelloDrone uses. Replies arrive asynchronously like the UDP thread's."""

    def __init__(self, delays=None, silent=()):
        self.sent = []
        self.udp = {"responses": [], "state": {}}
        self.is_flying = False
        self.delays = {"takeoff": 0.2, "land": 0.2, "move": 0.15, "rotate": 0.1, "stop": 0.02, "speed": 0.01}
        self.delays.update(delays or {})
        self.silent = set(silent)  # commands that never get a reply
        self.error_for = {}  # command prefix -> error text
        self.state = {"yaw": 10, "pitch": -2, "roll": 1, "h": 80, "tof": 85, "vgx": 3, "vgy": -1, "vgz": 0,
                      "bat": 87, "time": 5}
        self.ended = False

    def connect(self):
        self.sent.append("command")

    def send_control_command(self, cmd):
        self.sent.append(cmd)
        return True

    def streamon(self):
        self.sent.append("streamon")

    def send_command_with_return(self, cmd, timeout=7):
        self.sent.append(cmd)
        return next((e for p, e in self.error_for.items() if cmd.startswith(p)), "ok")

    def get_udp_video_address(self):
        return "udp://@0.0.0.0:11111"

    def get_own_udp_object(self):
        return self.udp

    def get_current_state(self):
        return self.state

    def send_rc_control(self, *v):
        self.sent.append(("rc",) + tuple(v))

    def reply_later(self, text, delay):
        threading.Timer(delay, lambda: self.udp["responses"].append(text.encode())).start()

    def send_command_without_return(self, cmd):
        self.sent.append(cmd)
        word = cmd.split()[0]
        if word in self.silent or word == "emergency":
            return
        kind = "move" if word in ("forward", "back", "left", "right", "up", "down") else ("rotate" if word in ("cw", "ccw") else word)
        err = next((e for p, e in self.error_for.items() if cmd.startswith(p)), None)
        self.reply_later(err or "ok", self.delays.get(kind, 0.05))

    def end(self):
        self.ended = True


def wait(d, timeout=3.0):
    t0 = time.time()
    while d.busy() and time.time() - t0 < timeout:
        time.sleep(0.005)
    return d.last_result()


def make(**kw):
    fake = FakeTello(**{k: v for k, v in kw.items() if k in ("delays", "silent")})
    d = TelloDrone(tello=fake, video=False, **{k: v for k, v in kw.items() if k not in ("delays", "silent")})
    d.connect()
    return d, fake


# ------------------------------------------------------------------ video start (the "can't find camera" crash)
class StreamingFakeTello(FakeTello):
    """Starts sending UDP 'video' datagrams to `port` on the n-th streamon (None: never, e.g. the phone app has it)."""

    def __init__(self, port, stream_on_nth=1):
        super().__init__()
        self.port, self.stream_on_nth, self.streamons = port, stream_on_nth, 0
        self._halt = threading.Event()

    def get_udp_video_address(self):
        return f"udp://@0.0.0.0:{self.port}"

    def streamon(self):
        super().streamon()
        self.streamons += 1
        if self.streamons == self.stream_on_nth:
            threading.Thread(target=self._send, daemon=True).start()

    def _send(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            while not self._halt.wait(0.02):
                s.sendto(b"\x00\x00\x00\x01", ("127.0.0.1", self.port))

    def end(self):
        super().end()
        self._halt.set()


class StubSource:
    def __init__(self, url, fps, backend):
        self.url = url

    def start(self):
        return self

    def stop(self):
        pass


def free_udp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def fast_video_check(monkeypatch):
    monkeypatch.setattr(tello_mod, "VIDEO_WAIT_S", 0.3)
    monkeypatch.setattr(tello_mod, "TelloVideoSource", StubSource)


def test_streamon_is_resent_until_video_packets_arrive(fast_video_check):
    fake = StreamingFakeTello(free_udp_port(), stream_on_nth=2)  # the drone ignores the first streamon
    d = TelloDrone(tello=fake, dry_run=True)
    d.connect()
    assert fake.streamons == 2 and d.frame_source().url == fake.get_udp_video_address()
    d.close()


def test_no_video_packets_gives_the_no_video_diagnosis(fast_video_check):
    fake = StreamingFakeTello(free_udp_port(), stream_on_nth=None)
    d = TelloDrone(tello=fake, dry_run=True)
    with pytest.raises(RuntimeError, match="sends no video"):
        d.connect()
    assert fake.streamons == tello_mod.STREAMON_TRIES
    d.close()


def test_video_port_held_by_another_program_is_named(fast_video_check):
    port = free_udp_port()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as other:
        other.bind(("", port))
        d = TelloDrone(tello=StreamingFakeTello(port), dry_run=True)
        with pytest.raises(RuntimeError, match=f"lsof -nP -iUDP:{port}"):
            d.connect()
        d.close()


# ------------------------------------------------------------------ video link settings (setbitrate & co)
def test_video_link_settings_go_after_command_and_before_streamon(fast_video_check):
    fake = StreamingFakeTello(free_udp_port())
    d = TelloDrone(tello=fake, dry_run=True, video_bitrate="2", video_resolution="low", video_stream_fps="middle")
    d.connect()
    assert fake.sent == ["command", "setbitrate 2", "setresolution low", "setfps middle", "streamon"]
    d.close()


def test_no_video_link_settings_by_default(fast_video_check):
    fake = StreamingFakeTello(free_udp_port())
    d = TelloDrone(tello=fake, dry_run=True)
    d.connect()
    assert fake.sent == ["command", "streamon"]
    d.close()


def test_old_firmware_error_is_logged_once_and_the_stream_still_starts(fast_video_check, caplog):
    fake = StreamingFakeTello(free_udp_port())
    fake.error_for = {"setbitrate": "error", "setfps": "unknown command: setfps"}
    d = TelloDrone(tello=fake, dry_run=True, video_bitrate="auto", video_resolution="high", video_stream_fps="low")
    with caplog.at_level("INFO", logger="kite.tello"):
        d.connect()
    assert fake.sent == ["command", "setbitrate 0", "setresolution high", "setfps low", "streamon"]
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("doesn't support setbitrate" in m and "keeping its default" in m for m in msgs) == 1
    assert sum("doesn't support setfps" in m for m in msgs) == 1
    assert "setresolution high -> ok" in msgs
    d.close()


def test_apply_video_link_never_raises():
    said = []

    def send(cmd):
        if cmd.startswith("setbitrate"):
            raise OSError("network unreachable")
        return None  # no reply at all

    replies = tello_mod.apply_video_link(send, bitrate="5", fps="high", say=said.append)
    assert list(replies) == ["setbitrate 5", "setfps high"]
    assert "not sent" in replies["setbitrate 5"] and len(said) == 2
    assert all("keeping the drone's default" in s for s in said)
    assert tello_mod.video_link_commands() == []


def test_make_tello_passes_the_video_link_flags():
    import argparse

    from kite.config import add_video_link_args, video_link_args

    p = argparse.ArgumentParser()
    add_video_link_args(p)
    cfg = Config()
    video_link_args(cfg.drone, p.parse_args(["--video-bitrate", "1", "--video-resolution", "low", "--video-fps", "low"]))
    d = make_tello(cfg, dry_run=True)
    assert d.video_link == {"bitrate": "1", "resolution": "low", "fps": "low"}
    with pytest.raises(SystemExit):
        p.parse_args(["--video-bitrate", "9"])


# ------------------------------------------------------------------ Tello adapter
def test_connect_sets_speed_and_takeoff_is_non_blocking():
    d, fake = make(move_speed_cm_s=40)
    assert fake.sent[:2] == ["command", "speed 40"]
    t0 = time.time()
    d.takeoff()
    assert time.time() - t0 < 0.15 and d.busy() and d.last_result() is None and not d.flying
    assert wait(d) == "ok" and d.flying and fake.is_flying


def test_move_reply_and_rc_blocked_while_busy():
    d, fake = make()
    d.takeoff()
    wait(d)
    d.move("forward", 100)
    d.rc(0, 0, 0, 50)  # must not reach the drone: it would abort the move
    assert not any(isinstance(s, tuple) for s in fake.sent)
    assert wait(d) == "ok"
    d.rc(0, 30, 0, 0)
    assert fake.sent[-1] == ("rc", 0, 30, 0, 0)
    d.rotate(-90)
    assert fake.sent[-1] == "ccw 90" and wait(d) == "ok"
    with pytest.raises(RuntimeError):
        d.move("up", 20)
        d.move("up", 20)  # second command while the first runs is a programming error


def test_rc_reaches_djitellopy_as_plain_ints():
    # djitellopy's @enforce_types rejects numpy scalars and floats with a TypeError mid-flight
    d, fake = make()
    d.takeoff()
    wait(d)
    d.rc(np.float64(12.6), np.int64(5), 0, -3.2)
    sent = fake.sent[-1]
    assert sent == ("rc", 13, 5, 0, -3) and all(type(v) is int for v in sent[1:])


def test_error_reply_and_timeout():
    d, fake = make()
    fake.error_for["forward"] = "error Not joystick"
    d.takeoff()
    wait(d)
    d.move("forward", 50)
    assert wait(d).startswith("error: error Not joystick")
    d2, fake2 = make(silent=("up",))
    d2._start("up 20", timeout_s=0.2, expected_s=0.1)
    assert wait(d2) == "error: timeout"


def test_stop_discards_the_late_reply_of_the_cancelled_move():
    d, fake = make(delays={"move": 0.4})
    d.takeoff()
    wait(d)
    d.move("forward", 200)
    time.sleep(0.05)
    d.stop()
    assert not d.busy() and d.last_result() == "ok" and fake.sent[-1] == "stop"
    fake.error_for["cw"] = "error No valid imu"
    t0 = time.time()
    d.rotate(45)  # sent right after the stop: waits out the settle window first (stale 'ok's arrive meanwhile)
    assert time.time() - t0 >= 0.8
    assert wait(d) == "error: error No valid imu"  # the stale "ok"s were not credited to the rotation


def test_stale_replies_are_cleared_before_a_new_command():
    d, fake = make(silent=("cw",))
    d.takeoff()
    wait(d)
    fake.udp["responses"].append(b"ok")  # leftover
    d._start("cw 30", timeout_s=0.3, expected_s=0.1)
    assert wait(d) == "error: timeout"


def test_min_gap_between_commands():
    d, fake = make(delays={"move": 0.0, "rotate": 0.0})
    d.takeoff()
    wait(d)
    t0 = time.time()
    for _ in range(3):
        d.rotate(10)
        wait(d)
    assert time.time() - t0 >= 0.2  # >= 0.1 s between sends


def test_telemetry_conversion():
    d, fake = make(yaw_sign=-1)
    tel = d.telemetry()
    assert tel.yaw_deg == -10 and tel.pitch_deg == -2 and tel.height_m == pytest.approx(0.8)
    assert tel.tof_m == pytest.approx(0.85) and tel.vx == pytest.approx(0.3) and tel.vy == pytest.approx(-0.1)
    assert tel.battery_pct == 87 and abs(tel.t - time.time()) < 2
    fake.state["tof"] = 6553  # out of range
    assert d.telemetry().tof_m is None
    fake.state = {}
    assert d.telemetry().yaw_deg is None


def test_dry_run_never_sends_motion():
    d, fake = make(dry_run=True)
    d.takeoff()
    assert d.busy()
    time.sleep(0.01)
    d.rc(10, 10, 10, 10)
    d.emergency()
    d.takeoff()
    d._pending.expected_s = 0.05
    assert wait(d) == "ok" and d.flying
    d.move("forward", 50)
    d._pending.expected_s = 0.05
    assert wait(d) == "ok"
    assert fake.sent == ["command"]  # not even 'speed'
    assert any("forward 50" in c for _, c in d.log) and not fake.is_flying


def test_land_is_idempotent_and_never_stops_a_landing():
    d, fake = make(delays={"land": 0.3})
    d.takeoff()
    wait(d)
    d.move("forward", 100)
    d.land()  # cancels the move with 'stop', then lands
    assert fake.sent[-2:] == ["stop", "land"]
    n = len(fake.sent)
    d.land()  # again while landing: nothing sent (a 'stop' would abort the landing)
    assert len(fake.sent) == n
    assert wait(d) == "ok" and not d.flying


def test_stop_never_cancels_takeoff_or_landing():
    d = TelloDrone(tello=FakeTello(delays={"takeoff": 0.3, "land": 0.3}), video=False)
    d.connect()
    d.takeoff()
    d.stop()
    assert d.busy() and "stop" not in d.tello.sent
    assert wait(d) == "ok" and d.flying
    d.land()
    d.stop()
    assert d.busy() and wait(d) == "ok" and not d.flying


def test_lost_land_reply_on_the_ground_counts_as_landed():
    d, fake = make()
    d.takeoff()
    wait(d)
    fake.error_for["land"] = "error"
    fake.state = dict(fake.state, h=0)  # it is down already
    d.land()
    assert wait(d) == "ok" and not d.flying


def test_close_lands_and_ends():
    d, fake = make()
    d.takeoff()
    wait(d)
    d.close()
    assert "land" in fake.sent and fake.ended


def test_dry_run_from_config_is_honoured():
    assert make_tello(Config(drone=DroneCfg(kind="dry_run"))).dry_run
    assert make_tello(Config(), dry_run=True).dry_run
    assert not make_tello(Config()).dry_run
    with pytest.raises(ValueError):
        make_tello(Config(drone=DroneCfg(kind="sim")))


def test_frozen_state_stream_is_detected_as_stale():
    d, fake = make()
    d.takeoff()
    wait(d)
    gov = SafetyGovernor(d, SafetyCfg())
    now = time.time()
    assert gov.check(now, now) == "ok"
    time.sleep(1.2)  # the state dict is never replaced: djitellopy's receiver thread died
    now = time.time()
    assert gov.check(now, now) == "hover"
    fake.state = dict(fake.state)  # a new packet arrives
    now = time.time()
    assert gov.check(now, now) == "ok"


# ------------------------------------------------------------------ safety governor
class FakeDrone(Drone):
    """Airborne drone whose telemetry the test sets; records what reaches it."""

    def __init__(self, height_m=1.5):
        self.tel = Telemetry(t=time.time(), height_m=height_m, battery_pct=80.0)
        self.rcs, self.moves, self.lands = [], [], 0
        self._flying, self._busy = True, False

    def takeoff(self):
        self._flying = True

    def land(self):
        self.lands += 1
        self._busy = True

    def emergency(self):
        self._flying = False

    def rc(self, lr, fb, ud, yaw):
        self.rcs.append((lr, fb, ud, yaw))

    def move(self, direction, cm):
        self.moves.append((direction, cm))

    def rotate(self, deg):
        self.moves.append(("rotate", deg))

    def stop(self):
        pass

    def busy(self):
        return self._busy

    def last_result(self):
        return "ok"

    def telemetry(self):
        return self.tel

    def frame_source(self):
        raise RuntimeError("no video in FakeDrone")

    @property
    def flying(self):
        return self._flying


def governed(height_m=1.5, **cfg):
    fake = FakeDrone(height_m)
    return SafetyGovernor(fake, SafetyCfg(**cfg)), fake


def test_governor_clamps_rc():
    gov, fake = governed(max_rc=30)
    gov.rc(100, -100, 0, 80)
    assert fake.rcs[-1] == (30, -30, 0, 30)


def test_non_finite_stick_values_hover_instead_of_raising():
    """A NaN / inf stick (a controller dividing by zero) used to raise out of rc() and end the flight loop."""
    nan, inf = float("nan"), float("inf")
    gov, fake = governed(max_rc=30)
    gov.rc(nan, inf, -inf, 10)
    assert fake.rcs[-1] == (0, 0, 0, 10)
    d, fake_tello = make()
    d.rc(nan, inf, -inf, 10)
    assert fake_tello.sent[-1] == ("rc", 0, 0, 0, 10)


def test_governor_ceiling_and_floor():
    gov, fake = governed(height_m=2.5, max_altitude_m=2.3, min_altitude_m=0.5)
    gov.rc(0, 0, 50, 0)
    assert fake.rcs[-1][2] == -20  # above the ceiling: pushed down
    fake.tel.height_m = 2.1
    gov.rc(0, 0, 50, 0)
    assert 0 < fake.rcs[-1][2] < 50  # climb tapered inside the 0.4 m band
    fake.tel.height_m = 0.4
    gov.rc(0, 0, -50, 0)
    assert fake.rcs[-1][2] == 0  # below the floor: no descent
    fake.tel.height_m = 0.7
    gov.rc(0, 0, -50, 0)
    assert -50 < fake.rcs[-1][2] < 0


def test_ceiling_push_down_stays_within_max_rc():
    """Above the ceiling the governor pushed down with a fixed -20, beyond a max_rc below 20."""
    gov, fake = governed(height_m=2.5, max_altitude_m=2.3, max_rc=10)
    gov.rc(0, 0, 50, 0)
    assert fake.rcs[-1][2] == -10


def test_governor_refuses_moves_across_the_limits():
    gov, fake = governed(height_m=2.0, max_altitude_m=2.3, min_altitude_m=0.5)
    gov.move("up", 50)
    assert gov.last_result().startswith("error: safety") and not fake.moves
    fake.tel.height_m = 0.8
    gov.move("down", 50)
    assert gov.last_result().startswith("error: safety") and not fake.moves
    gov.move("forward", 100)
    assert gov.last_result() == "ok" and fake.moves == [("forward", 100)]


def test_governor_lands_on_low_battery_and_resends_a_lost_land():
    gov, fake = governed(min_battery_pct=20)
    fake.tel.battery_pct = 15
    now = time.time()
    assert gov.check(now, now) == "land" and fake.lands == 1
    gov.rc(0, 50, 0, 0)  # ignored while landing
    assert not fake.rcs
    assert gov.check(now, now) == "land" and fake.lands == 1  # still landing: not re-sent
    fake._busy = False  # the land reply was lost and it is still flying
    gov._land_sent -= 3.0
    fake.tel.t = time.time()
    assert gov.check(time.time(), time.time()) == "land" and fake.lands == 2


def test_governor_lands_after_max_flight_time():
    gov, fake = governed(max_flight_s=10)
    now = time.time()
    assert gov.check(now, now) == "ok"
    fake.tel.t = now + 11
    assert gov.check(now + 11, now + 11) == "land"


def test_governor_hovers_when_video_or_telemetry_goes_stale_and_recovers():
    gov, fake = governed(frame_timeout_s=1.0, telemetry_timeout_s=1.0)
    now = time.time()
    fake.tel.t = now
    assert gov.check(now, now - 0.5) == "ok"
    assert gov.check(now, now - 2.0) == "hover" and fake.rcs[-1] == (0, 0, 0, 0)
    gov.rc(0, 40, 0, 0)  # the mission keeps asking, but the governor holds position
    assert fake.rcs[-1] == (0, 0, 0, 0)
    assert gov.check(now, now) == "ok"
    gov.rc(0, 40, 0, 0)
    assert fake.rcs[-1] == (0, 40, 0, 0)
    assert gov.check(now + 5.0, now + 5.0) == "hover"  # telemetry that stops updating also means hover


class _LinkFake:
    """The djitellopy surface TelloDrone needs; sends raise like a dropped Wi-Fi link when `down`."""

    def __init__(self, h_cm=10):
        self.down, self.sent, self.is_flying = False, [], False
        self.state = {"h": h_cm, "tof": h_cm, "bat": 80}
        self.udp = {"responses": []}

    def send_command_without_return(self, cmd):
        if self.down:
            raise OSError(51, "Network is unreachable")
        self.sent.append(cmd)

    def get_own_udp_object(self):
        return self.udp

    def get_current_state(self):
        return self.state


def test_a_land_that_never_left_the_laptop_is_not_counted_as_landed_even_low_down():
    fake = _LinkFake(h_cm=10)  # 0.1 m: under the 0.2 m "it is already down" rule
    d = TelloDrone(tello=fake, video=False)
    d._flying = fake.is_flying = True
    fake.down = True
    d.land()
    assert d.last_result().startswith("error: not sent")
    assert d.flying and fake.is_flying  # still flying: the governor, close() and djitellopy's end() keep trying
    fake.down = False
    d.land()
    assert "land" in fake.sent


def test_a_takeoff_whose_ok_was_lost_counts_as_flying_once_it_is_up():
    fake = _LinkFake(h_cm=80)
    d = TelloDrone(tello=fake, video=False)
    d.takeoff()
    d._complete("error: timeout")  # the 'ok' never came back
    assert d.last_result() == "ok" and d.flying and fake.is_flying  # so every landing path applies


def test_a_refused_takeoff_on_the_ground_stays_on_the_ground():
    fake = _LinkFake(h_cm=0)
    d = TelloDrone(tello=fake, video=False)
    d.takeoff()
    d._complete("error: battery low")
    assert d.last_result() == "error: battery low" and not d.flying


def test_an_rc_that_djitellopy_would_throttle_is_not_taken_as_news_about_the_link():
    class Throttling(_LinkFake):
        TIME_BTW_RC_CONTROL_COMMANDS = 0.001  # djitellopy's real gate
        last_rc_control_timestamp = 0.0

        def send_rc_control(self, *v):
            if time.time() - self.last_rc_control_timestamp > self.TIME_BTW_RC_CONTROL_COMMANDS:
                self.last_rc_control_timestamp = time.time()
                self.send_command_without_return("rc {} {} {} {}".format(*v))

    fake = Throttling()
    d = TelloDrone(tello=fake, video=False)
    fake.down = True
    d.rc(0, 10, 0, 0)  # sent, fails: the link is down
    assert d.link_error is not None
    fake.last_rc_control_timestamp = time.time()
    d.rc(0, 10, 0, 0)  # inside djitellopy's gate: kite waits it out and really sends, so the failure is seen
    assert d.link_error is not None


def test_nan_or_future_frame_and_telemetry_stamps_mean_hover():
    """NaN compared False with the timeout, so a NaN stamp (or one far in the future after the wall clock stepped
    back) passed as fresh video / telemetry and the governor never hovered."""
    nan = float("nan")
    gov, fake = governed(frame_timeout_s=1.0, telemetry_timeout_s=1.0)
    now = time.time()
    fake.tel.t = now
    assert gov.check(now, nan) == "hover"
    assert gov.check(now, now + 3600.0) == "hover"
    assert gov.check(now, now + 0.5) == "ok"  # within the timeout either way: a fresh frame
    fake.tel.t = nan
    assert gov.check(now, now) == "hover"
    fake.tel.t = now + 3600.0
    assert gov.check(now, now) == "hover"
    fake.tel.t = now
    assert gov.check(nan, now) == "hover"  # a broken clock is no evidence of live video either


def test_land_resend_is_not_held_back_by_a_wall_clock_step(monkeypatch):
    """The re-send interval was timed on time.time(): after the wall clock stepped back (NTP, a Wi-Fi switch) a
    lost land was not re-sent until the clock caught up again. It is timed on a monotonic clock now."""
    import kite.drone.safety as safety_mod

    gov, fake = governed()
    gov.land()
    assert fake.lands == 1
    fake._busy = False  # the land reply was lost, it still flies
    gov._land_sent -= 3.0  # the 2 s re-send interval has passed
    wall = time.time()
    monkeypatch.setattr(safety_mod.time, "time", lambda: wall - 3600.0)  # the wall clock steps back an hour
    fake.tel.t = wall
    assert gov.check(wall, wall) == "land" and fake.lands == 2


def test_land_resend_interval_runs_on_the_injected_clock():
    mono = [100.0]
    fake = FakeDrone()
    gov = SafetyGovernor(fake, SafetyCfg(), clock=lambda: mono[0])
    gov.land()
    fake._busy = False
    now = time.time()
    fake.tel.t = now
    mono[0] += 1.9
    assert gov.check(now, now) == "land" and fake.lands == 1  # within 2 s: not yet
    mono[0] += 0.2
    assert gov.check(now, now) == "land" and fake.lands == 2


def test_garbled_state_values_read_as_unknown():
    """telemetry() did float(v) on any value: a string or a huge int raised out of the governor's check()."""
    d, fake = make()
    fake.state = {"yaw": "x", "pitch": [1], "h": 10**400, "tof": float("inf"), "bat": float("nan"), "vgx": 3}
    tel = d.telemetry()
    assert (tel.yaw_deg, tel.pitch_deg, tel.height_m, tel.tof_m, tel.battery_pct) == (None,) * 5
    assert tel.vx == pytest.approx(0.3)
