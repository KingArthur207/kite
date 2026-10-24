"""Flight-safety review of `python -m kite.record --source tello --fly` (the 7 AM shoot path).

Each test reproduces one failure found in review, against the existing fakes (no drone, no network)."""

import errno
import json
import os
import termios
import time

import pytest
import test_drone as td
import test_record as tr

from kite.config import SafetyCfg
from kite.drone import SafetyGovernor
from kite.drone import tello as tello_mod
from kite.record import Recorder, RecordCfg, TerminalKeys


# ------------------------------------------------------------------ keyboard
def pipe_keys():
    r, w = os.pipe()
    k = TerminalKeys(fd=r)
    return k, w


def test_arrow_and_other_escape_sequences_never_become_flight_keys():
    k, w = pipe_keys()
    # Up = ESC [ A would read as 'a' (LEFT) after .lower(), Left = ESC [ D as 'd' (RIGHT), End = ESC [ F as 'f' (DOWN);
    # also application-mode arrows (ESC O A), Shift+Right (ESC [ 1 ; 2 C), F5, and Option+t sent as ESC t (Meta)
    os.write(w, b"\x1b[A\x1b[D\x1b[F\x1bOA\x1b[1;2C\x1b[15~\x1bt")
    assert k.read() == ""
    os.write(w, b"\x1b")  # a sequence split across two reads
    assert k.read() == ""
    os.write(w, b"[Aw")
    assert k.read() == "w"
    os.write(w, "é".encode()[:1])  # a multi-byte character split across reads, then a real key
    assert k.read() == ""
    os.write(w, "é".encode()[1:] + b"l")
    assert k.read() == "l"
    os.write(w, b"\x1a\x1c\x13W L")  # stray control chars dropped; caps-lock letters still pass (L must land)
    assert k.read() == "W L"


def test_a_lone_escape_does_not_swallow_the_next_key():
    k, w = pipe_keys()
    os.write(w, b"\x1b")
    assert k.read() == ""
    time.sleep(0.15)
    os.write(w, b"l")
    assert k.read() == "l"


def test_terminal_keys_disable_suspend_quit_and_output_freeze():
    """Ctrl+Z would suspend the process mid-flight (no keep-alive, the drone keeps its last stick), Ctrl+\\ would kill
    it without landing, Ctrl+S would freeze terminal output until a print() blocks the control loop."""
    master, slave = os.openpty()
    try:
        before = termios.tcgetattr(slave)
        with TerminalKeys(fd=slave):
            a = termios.tcgetattr(slave)
            off = os.fpathconf(slave, "PC_VDISABLE")
            assert a[6][termios.VSUSP] in (off, bytes([off]))
            assert a[6][termios.VQUIT] in (off, bytes([off]))
            assert not a[0] & termios.IXON
            assert a[3] & termios.ISIG and not a[3] & termios.ICANON  # Ctrl+C still lands
        after = termios.tcgetattr(slave)
        after[3] &= ~getattr(termios, "PENDIN", 0)  # macOS flags pending input on the switch back: not ours
        assert after == before
    finally:
        os.close(master)
        os.close(slave)


# ------------------------------------------------------------------ Wi-Fi drop (sendto raises OSError)
class FlakyTello(tr.FakeTello):
    """sendto fails like macOS does when the Wi-Fi drops or its queue is full (ENETUNREACH / ENOBUFS)."""

    down = False

    def _check(self):
        if self.down:
            raise OSError(errno.ENETUNREACH, "Network is unreachable")

    def send_rc_control(self, *v):
        self._check()
        super().send_rc_control(*v)

    def send_command_without_return(self, cmd):
        self._check()
        super().send_command_without_return(cmd)


def test_a_wifi_drop_mid_flight_does_not_end_the_take_or_abandon_the_drone(tmp_path):
    fake = FlakyTello()
    drone = SafetyGovernor(tello_mod.TelloDrone(tello=fake, video=False), SafetyCfg())
    drone.connect()
    time.sleep(0.25)

    class Keys(tr.ScriptedKeys):
        def read(self):
            t = time.time() - (self.t0 or time.time())
            fake.down = 1.3 <= t < 1.9  # the link drops for 0.6 s while flying
            return super().read()

    keys = Keys([(0.3, "t"), (1.0, "w"), (2.4, KeyboardInterrupt)])
    run = Recorder(tr.LiveImage(), drone, RecordCfg(out_dir=str(tmp_path), fly=True, speed=30), {"source_kind": "fake"},
                   keys=keys).run()
    drone.close()
    meta = json.loads((run / "meta.json").read_text())
    assert meta["summary"]["end_reason"] == "Ctrl+C"  # not "error: OSError": the take went on
    sent = fake.sent
    assert "land" in sent and not drone.flying
    events = [json.loads(line)["event"] for line in (run / "events.jsonl").read_text().splitlines()]
    assert "landed" in events


def test_a_land_that_cannot_be_sent_is_retried_when_the_link_returns():
    fake = td.FakeTello()
    d = tello_mod.TelloDrone(tello=fake, video=False)
    d.connect()
    d.takeoff()
    assert td.wait(d) == "ok"
    gov = SafetyGovernor(d, SafetyCfg())

    def down(*_):
        raise OSError(errno.ENOBUFS, "No buffer space available")

    real = fake.send_command_without_return
    fake.send_command_without_return = down
    gov.land()  # the 'l' key during a Wi-Fi drop: must not raise, must not sit 'busy' for 15 s
    assert not gov.busy() and gov.last_result().startswith("error") and gov.flying
    assert d.link_error
    fake.send_command_without_return = real
    gov._land_sent -= 3.0
    now = time.time()
    assert gov.check(now, now) == "land"  # the governor re-sends the land the pilot asked for
    assert fake.sent[-1] == "land" and td.wait(d) == "ok" and not gov.flying and d.link_error is None


def test_an_emergency_that_cannot_be_sent_is_reported_not_assumed():
    fake = td.FakeTello()
    d = tello_mod.TelloDrone(tello=fake, video=False)
    d.connect()
    d.takeoff()
    td.wait(d)

    def down(*_):
        raise OSError(errno.ENETUNREACH, "Network is unreachable")

    fake.send_command_without_return = down
    fake.send_rc_control = down
    gov = SafetyGovernor(d, SafetyCfg())
    gov.rc(0, 30, 0, 0)  # no exception out of the keep-alive
    assert gov.emergency() is False and gov.flying  # still flying: the pilot must be told, and can press !! again


def test_a_requested_land_that_failed_is_resent_while_still_flying():
    gov, fake = td.governed()
    gov.land()  # the 'l' key
    assert fake.lands == 1
    fake._busy = False  # the land got an error reply / timed out, and it still flies
    gov._land_sent -= 3.0
    now = time.time()
    fake.tel.t = now
    assert gov.check(now, now) == "land" and fake.lands == 2


# ------------------------------------------------------------------ takeoff preconditions
def test_takeoff_is_refused_when_telemetry_is_frozen(tmp_path):
    drone, fake = tr.fake_drone()

    class Keys(tr.ScriptedKeys):
        def read(self):
            if self.t0 and time.time() - self.t0 > 0.2:
                fake._halt.set()  # the state stream stops (djitellopy's receiver thread died on a bad packet)
            return super().read()

    keys = Keys([(1.6, "t")])
    Recorder(tr.LiveImage(), drone, RecordCfg(out_dir=str(tmp_path), fly=True, max_s=2.0), {"source_kind": "fake"},
             keys=keys).run()
    drone.close()
    assert "takeoff" not in fake.sent  # it would only have hovered, out of the pilot's control


# ------------------------------------------------------------------ wall-clock steps
class ClockTello(td.FakeTello):
    """Gates rc on time.time() like djitellopy's send_rc_control does."""

    last_rc_control_timestamp = 0.0

    def send_rc_control(self, *v):
        if time.time() - self.last_rc_control_timestamp > 0.001:
            self.last_rc_control_timestamp = time.time()
            super().send_rc_control(*v)


def test_a_wall_clock_step_neither_blocks_control_nor_times_out_a_takeoff(monkeypatch):
    fake = ClockTello()
    d = tello_mod.TelloDrone(tello=fake, video=False)
    d.connect()
    real = time.time
    d.takeoff()
    monkeypatch.setattr(time, "time", lambda: real() + 60)  # clock steps forward mid-takeoff
    assert d.busy()  # not a 60 s-old command: no false 'error: timeout' (which would leave it flying unmanaged)
    monkeypatch.setattr(time, "time", real)
    assert td.wait(d) == "ok" and d.flying
    d.rc(0, 20, 0, 0)
    monkeypatch.setattr(time, "time", lambda: real() - 5)  # clock steps back
    t0 = time.monotonic()
    d.rotate(10)  # would sleep 5 s in _send (main loop frozen, no keep-alive)
    assert time.monotonic() - t0 < 0.5
    td.wait(d)
    d.rc(0, 0, 0, 0)  # djitellopy would drop every rc for 5 s
    assert fake.sent[-1] == ("rc", 0, 0, 0, 0)


# ------------------------------------------------------------------ events file
def test_an_events_file_error_does_not_end_the_flight(tmp_path):
    class Broken:
        def write(self, s):
            raise OSError(errno.ENOSPC, "No space left on device")

        def flush(self):
            pass

    rec = Recorder(tr.LiveImage(), None, RecordCfg(out_dir=str(tmp_path)))
    rec._events = Broken()
    rec._event("rc", lr=0)  # disk full: printed, not raised into the control loop


def test_status_line_shows_a_wifi_drop(tmp_path):
    fake = td.FakeTello()
    d = tello_mod.TelloDrone(tello=fake, video=False)
    d.connect()
    gov = SafetyGovernor(d, SafetyCfg())
    rec = Recorder(tr.LiveImage(), gov, RecordCfg(out_dir=str(tmp_path), fly=True), keys=tr.ScriptedKeys([]))
    d.link_error = "[Errno 51] Network is unreachable"
    assert "WI-FI DOWN" in rec._status(time.monotonic(), 30.0, "hover")


if __name__ == "__main__":
    pytest.main([__file__])
