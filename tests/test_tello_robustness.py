"""TelloDrone rc robustness against fakes (no drone, no network): djitellopy's 1 ms rc gate, the rc dead-man and the
board temperature the governor lands on."""

import time

import test_drone as td

from kite.config import SafetyCfg
from kite.drone import SafetyGovernor
from kite.drone.tello import TelloDrone


class GatedTello(td.FakeTello):
    """send_rc_control drops a packet within 1 ms of the previous one, silently, like djitellopy 2.5."""

    def __init__(self):
        super().__init__()
        self.last_rc_control_timestamp = 0.0
        self.dropped = 0

    def send_rc_control(self, *v):
        if time.time() - self.last_rc_control_timestamp > 0.001:
            self.last_rc_control_timestamp = time.time()
            super().send_rc_control(*v)
        else:
            self.dropped += 1


def flying(fake, **kw):
    d = TelloDrone(tello=fake, video=False, **kw)
    d.connect()
    d.takeoff()
    assert td.wait(d) == "ok" and d.flying
    return d


def rcs(fake):
    return [s for s in fake.sent if isinstance(s, tuple)]


def test_a_hover_right_after_a_stick_is_not_dropped_by_the_1_ms_gate():
    fake = GatedTello()
    d = flying(fake, rc_hold_s=None)
    for _ in range(50):
        d.rc(0, 40, 0, 0)
        d.rc(0, 0, 0, 0)  # e.g. the governor's hover right after the mission's stick, same tick
    assert fake.dropped == 0
    assert rcs(fake)[-1] == ("rc", 0, 0, 0, 0) and len(rcs(fake)) == 100


def test_the_dead_man_sends_one_rc_0_when_the_loop_stops_refreshing_a_stick():
    fake = td.FakeTello()
    d = flying(fake, rc_hold_s=0.2)
    d.rc(0, 40, 0, 0)
    time.sleep(0.6)  # the control loop "stalls" with the stick held
    sent = rcs(fake)
    assert sent[-1] == ("rc", 0, 0, 0, 0) and sent.count(("rc", 0, 0, 0, 0)) == 1  # exactly once: no keep-alive
    assert d.watchdog_trips == 1 and any("rc watchdog" in m for _, m in d.log)
    time.sleep(0.3)
    assert rcs(fake) == sent  # nothing more: a dead program still lets the Tello land itself after 15 s
    d.close()


def test_the_dead_man_stays_quiet_while_the_loop_runs_and_for_a_zero_stick():
    fake = td.FakeTello()
    d = flying(fake, rc_hold_s=0.2)
    for _ in range(10):
        d.rc(0, 30, 0, 0)  # refreshed at 20 Hz
        time.sleep(0.05)
    d.rc(0, 0, 0, 0)
    time.sleep(0.5)  # a zero stick may sit
    assert d.watchdog_trips == 0 and rcs(fake)[-1] == ("rc", 0, 0, 0, 0)
    assert rcs(fake).count(("rc", 0, 0, 0, 0)) == 1
    d.close()


def test_the_dead_man_never_cuts_into_a_discrete_move():
    fake = td.FakeTello(delays={"move": 0.6})
    d = flying(fake, rc_hold_s=0.15)
    d.rc(0, 30, 0, 0)
    d.move("forward", 50)  # replaces the stick; rc would abort it
    time.sleep(0.4)
    assert d.busy() and d.watchdog_trips == 0
    assert td.wait(d) == "ok"
    d.close()


def test_the_dead_man_is_off_in_dry_run_and_on_the_ground():
    fake = td.FakeTello()
    d = TelloDrone(tello=fake, video=False, dry_run=True, rc_hold_s=0.1)
    d.connect()
    d.rc(0, 50, 0, 0)
    time.sleep(0.3)
    assert not rcs(fake) and d.watchdog_trips == 0


def test_temperature_lands_through_the_governor():
    fake = td.FakeTello()
    d = flying(fake, rc_hold_s=None)
    gov = SafetyGovernor(d, SafetyCfg())
    fake.state = dict(fake.state, temph=70)
    assert d.temperature_c() == 70.0
    now = time.time()
    assert gov.check(now, now) == "ok"
    fake.state = dict(fake.state, temph=92)
    now = time.time()
    assert gov.check(now, now) == "land" and fake.sent[-1] == "land"
    assert "temperature 92" in gov.land_reason
    fake.state = {k: v for k, v in fake.state.items() if k != "temph"}
    assert d.temperature_c() is None
