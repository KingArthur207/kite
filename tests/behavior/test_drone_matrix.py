"""Behaviour matrix for the drone layer: TelloDrone (kite/drone/tello.py), SafetyGovernor (kite/drone/safety.py),
the base helpers (kite/drone/base.py) and the Telemetry helpers (kite/types.py).

Everything runs at the djitellopy-object level against fakes defined in this file, on a fake clock: no sockets
(except the three video_packets_arrive tests, on loopback), no threads, no real sleeps. Each parametrize id reads
as the behaviour it pins. tests/test_drone.py and tests/test_flight_safety.py hold the original, thread-based
scenarios; this file extends them branch by branch.
"""

from __future__ import annotations

import errno
import logging
import math
import random
import socket
import sys
import types

import numpy as np
import pytest

from kite.config import Config, DroneCfg, SafetyCfg
from kite.drone import DIRECTIONS, Drone, SafetyGovernor, check_move, check_rotate, clamp_rc, make_tello
from kite.drone import safety as safety_mod
from kite.drone import tello as tello_mod
from kite.drone.safety import SOFT_BAND_M
from kite.drone.tello import MIN_GAP_S, NO_VIDEO, STOP_SETTLE_S, STREAMON_TRIES, VIDEO_WAIT_S, TelloDrone
from kite.types import Frame, Telemetry, wrap_deg

WALL0 = 1_800_000_000.0  # fake wall clock start (time.time())
MONO0 = 5_000.0  # fake monotonic start
NET_DOWN = OSError(errno.ENETUNREACH, "Network is unreachable")


# ====================================================================== fakes
class FakeClock:
    """Stands in for the `time` module inside kite.drone.tello and kite.drone.safety.

    sleep() advances both clocks (and runs on_sleep hooks, e.g. a late reply arriving during the stop settle
    window); wall-clock steps (step_wall) move time.time() only, like an NTP correction."""

    def __init__(self):
        self.mono = MONO0
        self.wall = WALL0
        self.sleeps: list[float] = []
        self.on_sleep: list = []

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.wall

    def sleep(self, s: float) -> None:
        assert s >= 0
        self.sleeps.append(s)
        self.advance(s)
        for hook in self.on_sleep:
            hook(s)

    def advance(self, s: float) -> None:
        self.mono += s
        self.wall += s

    def step_wall(self, s: float) -> None:
        self.wall += s


class FakeDjiTello:
    """The djitellopy.Tello surface TelloDrone touches. Synchronous: tests push replies with reply()."""

    def __init__(self, clock: FakeClock, host: str = "192.168.10.1"):
        self.clock = clock
        self.host = host
        self.cmds: list[str] = []  # send_command_without_return
        self.cmd_times: list[tuple[float, str]] = []  # (monotonic, cmd)
        self.rcs: list[tuple] = []  # send_rc_control
        self.control: list[str] = []  # send_control_command (blocking djitellopy call; only 'speed' at connect)
        self.streamons = 0
        self.connects = 0
        self.ends = 0
        self.responses: list = []
        self.state: dict | None = {"yaw": 10, "pitch": -2, "roll": 1, "h": 150, "tof": 152, "vgx": 3, "vgy": -1,
                                   "vgz": 0, "bat": 87, "time": 5}
        self.is_flying = False
        self.down = False  # Wi-Fi drop: every sendto raises OSError
        self.state_error: Exception | None = None
        self.end_error: Exception | None = None
        self.video_url = "udp://@0.0.0.0:11111"

    # -- djitellopy API
    def connect(self):
        self.connects += 1

    def send_control_command(self, cmd):
        self.control.append(cmd)
        return True

    def streamon(self):
        self.streamons += 1

    def get_udp_video_address(self):
        return self.video_url

    def get_own_udp_object(self):
        return {"responses": self.responses, "state": self.state}

    def get_current_state(self):
        if self.state_error is not None:
            raise self.state_error
        return self.state

    def send_command_without_return(self, cmd):
        if self.down:
            raise NET_DOWN
        self.cmds.append(cmd)
        self.cmd_times.append((self.clock.monotonic(), cmd))

    def send_rc_control(self, lr, fb, ud, yaw):
        for v in (lr, fb, ud, yaw):  # djitellopy's @enforce_types: a float / numpy scalar is a TypeError mid-flight
            if type(v) is not int:
                raise TypeError(f"rc value {v!r} is {type(v).__name__}, not int")
        if self.down:
            raise NET_DOWN
        self.rcs.append((lr, fb, ud, yaw))

    def end(self):
        self.ends += 1
        if self.end_error is not None:
            raise self.end_error

    # -- test helpers
    def reply(self, *texts):
        for t in texts:
            self.responses.append(t.encode() if isinstance(t, str) else t)

    def motion_sent(self) -> list:
        """Every packet that could move the drone (or change its flight state)."""
        return list(self.cmds) + [("rc",) + r for r in self.rcs] + list(self.control)


class RecordingSource:
    instances: list = []

    def __init__(self, url, fps, backend):
        self.url, self.fps, self.backend = url, fps, backend
        self.started = self.stopped = False
        RecordingSource.instances.append(self)

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True


class FakeInner(Drone):
    """Inner drone for the governor: the test sets flying/busy/telemetry; every call is recorded."""

    def __init__(self, clock: FakeClock, h=1.5, tof=None, bat=80.0, flying=True, busy=False):
        self.clock = clock
        self.tel = Telemetry(t=clock.time(), height_m=h, tof_m=tof, battery_pct=bat)
        self._flying, self._busy = flying, busy
        self.calls: list[tuple[str, tuple]] = []
        self.emergency_result: bool | None = True
        self.result = "ok"
        self.source = object()

    def _rec(self, name, *args):
        self.calls.append((name, args))

    def connect(self):
        self._rec("connect")

    def close(self):
        self._rec("close")

    def takeoff(self):
        self._rec("takeoff")
        self._busy = True

    def land(self):
        self._rec("land")
        self._busy = True

    def emergency(self):
        self._rec("emergency")
        if self.emergency_result:
            self._flying, self._busy = False, False
        return self.emergency_result

    def rc(self, lr, fb, ud, yaw):
        self._rec("rc", lr, fb, ud, yaw)

    def move(self, direction, cm):
        check_move(direction, cm)
        self._rec("move", direction, cm)

    def rotate(self, deg):
        check_rotate(deg)
        self._rec("rotate", deg)

    def stop(self):
        self._rec("stop")

    def set_speed(self, cm_s):
        self._rec("set_speed", cm_s)

    def busy(self):
        return self._busy

    def last_result(self):
        return self.result

    def telemetry(self):
        return self.tel

    def frame_source(self):
        return self.source

    @property
    def flying(self):
        return self._flying

    # helpers
    def named(self, name):
        return [a for n, a in self.calls if n == name]

    @property
    def rcs(self):
        return self.named("rc")

    @property
    def lands(self):
        return len(self.named("land"))


# ====================================================================== fixtures / helpers
@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(tello_mod, "time", c)
    monkeypatch.setattr(safety_mod, "time", c)
    return c


@pytest.fixture
def video(monkeypatch):
    """Patches the video check and the reader. Set video.arrivals to script video_packets_arrive."""
    ns = types.SimpleNamespace(arrivals=[True], calls=[])
    RecordingSource.instances = []

    def fake_arrive(port, timeout_s):
        ns.calls.append((port, timeout_s))
        r = ns.arrivals.pop(0) if ns.arrivals else False
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(tello_mod, "video_packets_arrive", fake_arrive)
    monkeypatch.setattr(tello_mod, "TelloVideoSource", RecordingSource)
    return ns


def make(clock, dry_run=False, connect=True, **kw) -> tuple[TelloDrone, FakeDjiTello]:
    fake = FakeDjiTello(clock)
    d = TelloDrone(tello=fake, video=kw.pop("video", False), dry_run=dry_run, **kw)
    if connect:
        d.connect()
    return d, fake


def finish(d: TelloDrone, fake: FakeDjiTello, clock: FakeClock, reply="ok"):
    """Complete the running discrete command: a reply (live) or its expected duration (dry run)."""
    if d.dry_run:
        clock.advance(d._pending.expected_s)
    else:
        fake.reply(reply)
    assert not d.busy()
    return d.last_result()


def fly(d, fake, clock):
    d.takeoff()
    assert finish(d, fake, clock) == "ok" and d.flying
    clock.advance(1.0)  # well past the min gap / any settle window


def to_state(d, fake, clock, state):
    """Put a TelloDrone into one of the named flight states."""
    if state == "ground":
        return
    if state == "taking_off":
        d.takeoff()
        return
    fly(d, fake, clock)
    if state == "flying":
        return
    if state == "moving":
        d.move("forward", 100)
    elif state == "rotating":
        d.rotate(90)
    elif state == "setting_speed":
        d.set_speed(30)
    elif state == "landing":
        d.land()
    elif state == "landed":
        d.land()
        assert finish(d, fake, clock) == "ok" and not d.flying
    elif state == "after_emergency":
        assert d.emergency() is True
    else:
        raise AssertionError(state)
    clock.advance(0.5)


def governed(clock, cfg=None, **inner_kw) -> tuple[SafetyGovernor, FakeInner]:
    inner = FakeInner(clock, **inner_kw)
    return SafetyGovernor(inner, cfg or SafetyCfg(), clock=clock.monotonic), inner


# ====================================================================== base.py: helpers and interface
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("cm, expected", [(20, 20), (500, 500), (100, 100), (20.4, 20), (19.6, 20), (499.6, 500),
                                          (500.4, 500), (19.5, 20), (20.5, 20), (np.int64(250), 250),
                                          (np.float64(33.3), 33)],
                         ids=lambda v: str(v))
def test_check_move_accepts_20_to_500_cm_after_rounding(direction, cm, expected):
    out = check_move(direction, cm)
    assert out == expected and type(out) is int


@pytest.mark.parametrize("direction, cm", [
    ("forward", 19), ("forward", 501), ("forward", 19.4), ("forward", 500.6), ("forward", 0), ("forward", -50),
    ("up", -100), ("Forward", 50), ("cw", 50), ("", 50), ("up ", 50), ("north", 100),
], ids=lambda v: repr(v))
def test_check_move_rejects_bad_direction_or_distance(direction, cm):
    with pytest.raises(ValueError):
        check_move(direction, cm)


@pytest.mark.parametrize("deg, expected", [(1, 1), (-1, -1), (360, 360), (-360, -360), (90, 90), (1.5, 2),
                                           (90.4, 90), (-359.6, -360), (359.5, 360), (np.float64(-45.2), -45)],
                         ids=lambda v: str(v))
def test_check_rotate_accepts_1_to_360_in_magnitude(deg, expected):
    out = check_rotate(deg)
    assert out == expected and type(out) is int


@pytest.mark.parametrize("deg", [0, 0.4, -0.4, 0.5, -0.5, 361, -361, 360.6, 1e6],
                         ids=lambda v: f"deg={v}")
def test_check_rotate_rejects_zero_and_more_than_a_turn(deg):
    with pytest.raises(ValueError):
        check_rotate(deg)


@pytest.mark.parametrize("v, limit, expected", [
    (0, 100, 0), (100, 100, 100), (-100, 100, -100), (101, 100, 100), (-101, 100, -100), (1e9, 100, 100),
    (-1e9, 100, -100), (12.6, 100, 13), (-3.2, 100, -3), (2.5, 100, 2), (3.5, 100, 4), (-2.5, 100, -2),
    (0.49, 100, 0), (np.float64(12.6), 100, 13), (np.int64(-7), 100, -7), (60, 50, 50), (-60, 50, -50),
    (50, 50, 50), (49.6, 50, 50), (50.4, 50, 50), (5, 0, 0), (-5, 0, 0), (30, 20, 20),
], ids=lambda v: str(v))
def test_clamp_rc_rounds_half_to_even_then_clamps_to_plain_int(v, limit, expected):
    out = clamp_rc(v, limit)
    assert out == expected and type(out) is int


@pytest.mark.parametrize("v", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_clamp_rc_turns_a_non_finite_stick_into_hover(v):
    # a controller that divided by zero must not raise out of the flight loop: that stick is 0 (hover)
    assert clamp_rc(v) == 0


def test_drone_base_optional_methods_are_no_ops_and_interface_is_abstract():
    with pytest.raises(TypeError):
        Drone()

    class Minimal(Drone):
        takeoff = land = emergency = rc = move = rotate = stop = busy = last_result = telemetry = \
            frame_source = lambda *a, **k: None
        flying = False

    m = Minimal()
    assert m.connect() is None and m.set_speed(30) is None and m.close() is None and m.name == "drone"
    assert DIRECTIONS == ("forward", "back", "left", "right", "up", "down")


# ====================================================================== types.py: Telemetry helpers, wrap_deg
@pytest.mark.parametrize("tof, h, kw, expected", [
    (1.0, 2.0, {}, 1.0),
    (None, 2.0, {}, 2.0),
    (0.09, 2.0, {}, 2.0),
    (0.1, 2.0, {}, 0.1),
    (3.0, 2.0, {}, 3.0),
    (3.01, 2.0, {}, 2.0),
    (5.0, None, {}, None),
    (None, None, {}, None),
    (1.0, None, {}, 1.0),
    (1.0, 2.0, {"prefer_tof": False}, 2.0),
    (5.0, 2.0, {"tof_max_m": 8.0}, 5.0),
    (-1.0, 2.0, {}, 2.0),
], ids=["tof_in_range_wins", "no_tof_uses_h", "tof_below_10cm_uses_h", "tof_exactly_10cm_kept",
        "tof_exactly_3m_kept", "tof_past_3m_uses_h", "tof_past_3m_and_no_h_is_none", "nothing_is_none",
        "tof_only", "prefer_tof_false_uses_h", "custom_tof_max", "negative_tof_uses_h"])
def test_altitude_m_truth_table(tof, h, kw, expected):
    assert Telemetry(t=0, tof_m=tof, height_m=h).altitude_m(**kw) == expected


@pytest.mark.parametrize("tof, h, kw, expected", [
    (1.0, None, {}, 1.0),
    (None, None, {}, None),
    (5.0, None, {}, None),
    (None, 1.5, {}, 1.5),
    (1.5, 1.5, {}, 1.5),
    (1.625, 1.5, {}, 1.625),
    (1.75, 1.5, {}, 1.5),
    (0.75, 1.5, {}, 1.5),
    (1.25, 1.0, {"agree_m": 0.25}, 1.25),
    (1.25, 1.0, {}, 1.0),
    (5.0, 5.0, {}, 5.0),
    (1.0, 1.0, {"agree_m": 0.0}, 1.0),
    (1.0625, 1.0, {"agree_m": 0.0}, 1.0),
], ids=["no_h_falls_back_to_tof", "nothing_is_none", "no_h_tof_out_of_altitude_range", "no_tof_uses_h",
        "tof_equals_h", "tof_agrees_within_20cm", "tof_disagrees_uses_h", "over_a_table_uses_h",
        "agreement_boundary_inclusive", "default_agreement_is_20cm", "agreeing_tof_has_no_range_check",
        "zero_agreement_exact_match", "zero_agreement_mismatch_uses_h"])
def test_floor_altitude_m_truth_table(tof, h, kw, expected):
    assert Telemetry(t=0, tof_m=tof, height_m=h).floor_altitude_m(**kw) == expected


def test_frame_size_comes_from_the_image():
    f = Frame(image=np.zeros((720, 960, 3), np.uint8), t=1.0, seq=3)
    assert (f.width, f.height, f.source) == (960, 720, "")
    assert type(f.width) is int and type(f.height) is int


def test_telemetry_defaults_are_all_unknown():
    tel = Telemetry(t=1.0)
    assert [tel.yaw_deg, tel.pitch_deg, tel.roll_deg, tel.height_m, tel.tof_m, tel.vx, tel.vy, tel.vz,
            tel.battery_pct, tel.flight_time_s] == [None] * 10


@pytest.mark.parametrize("a, expected", [
    (0, 0), (90, 90), (-90, -90), (179.5, 179.5), (-179.5, -179.5), (180, 180), (-180, 180), (181, -179),
    (-181, 179), (270, -90), (-270, 90), (360, 0), (-360, 0), (540, 180), (-540, 180), (720, 0), (1e6, -80),
    (359.75, -0.25), (-0.25, -0.25),
], ids=lambda v: str(v))
def test_wrap_deg_maps_into_minus_180_exclusive_to_180_inclusive(a, expected):
    assert wrap_deg(a) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("a", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_wrap_deg_of_non_finite_is_nan(a):
    assert math.isnan(wrap_deg(a))


def test_wrap_deg_property_random_angles():
    rng = random.Random(1234)
    for _ in range(5000):
        a = rng.uniform(-1e5, 1e5)
        w = wrap_deg(a)
        assert -180.0 < w <= 180.0
        assert math.isclose(math.remainder(a - w, 360.0), 0.0, abs_tol=1e-6)
        assert wrap_deg(w) == pytest.approx(w)


# ====================================================================== TelloDrone: connect / setup
@pytest.mark.parametrize("speed, sent", [(50, "speed 50"), (10, "speed 10"), (100, "speed 100"), (45.7, "speed 45")],
                         ids=lambda v: str(v))
def test_connect_live_sets_the_discrete_move_speed(clock, speed, sent):
    d, fake = make(clock, move_speed_cm_s=speed)
    assert fake.connects == 1 and fake.control == [sent] and fake.cmds == [] and fake.streamons == 0


def test_connect_dry_run_sends_no_speed_command(clock):
    d, fake = make(clock, dry_run=True)
    assert fake.connects == 1 and fake.control == [] and fake.cmds == []


@pytest.mark.parametrize("host", ["192.168.10.1", "10.0.0.7"])
def test_connect_without_a_tello_builds_djitellopy_with_quiet_logger(clock, monkeypatch, host):
    created = []

    class FakeTelloClass(FakeDjiTello):
        LOGGER = logging.getLogger("test.fake-djitellopy")

        def __init__(self, h):
            super().__init__(clock, h)
            created.append(self)

    FakeTelloClass.LOGGER.setLevel(logging.INFO)
    monkeypatch.setitem(sys.modules, "djitellopy", types.SimpleNamespace(Tello=FakeTelloClass))
    d = TelloDrone(video=False, host=host)
    assert d.tello is None
    d.connect()
    assert len(created) == 1 and d.tello is created[0] and created[0].host == host
    assert FakeTelloClass.LOGGER.level == logging.WARNING  # otherwise djitellopy prints every rc packet
    assert created[0].connects == 1 and created[0].control == ["speed 50"]


def test_connect_reuses_a_given_tello(clock):
    d, fake = make(clock)
    assert d.tello is fake


@pytest.mark.parametrize("dry_run", [False, True], ids=["live", "dry_run"])
@pytest.mark.parametrize("arrivals, streamons", [
    ([True], 1), ([False, True], 2), ([False, False, False, True], 4),
], ids=["first_streamon", "second_streamon", "last_streamon"])
def test_video_start_resends_streamon_until_packets_arrive(clock, video, dry_run, arrivals, streamons):
    video.arrivals = list(arrivals)
    d, fake = make(clock, dry_run=dry_run, video=True, video_fps=45, video_backend="opencv")
    assert fake.streamons == streamons <= STREAMON_TRIES
    assert video.calls == [(11111, VIDEO_WAIT_S)] * streamons
    src = d.frame_source()
    assert src is RecordingSource.instances[-1] and src.started
    assert (src.url, src.fps, src.backend) == ("udp://@0.0.0.0:11111", 45, "opencv")
    assert fake.control == ([] if dry_run else ["speed 50"])  # speed goes out before the stream starts


def test_video_port_is_parsed_from_djitellopy_address(clock, video):
    fake = FakeDjiTello(clock)
    fake.video_url = "udp://@0.0.0.0:22222"
    d = TelloDrone(tello=fake, dry_run=True)
    d.connect()
    assert video.calls[0][0] == 22222 and d.frame_source().url == "udp://@0.0.0.0:22222"


def test_video_never_arriving_raises_the_no_video_diagnosis_after_all_tries(clock, video):
    video.arrivals = [False] * STREAMON_TRIES
    d, fake = make(clock, dry_run=True, connect=False, video=True)
    with pytest.raises(RuntimeError) as e:
        d.connect()
    assert str(e.value) == NO_VIDEO.format(port=11111) and "firewall" in str(e.value)
    assert fake.streamons == STREAMON_TRIES and not RecordingSource.instances
    with pytest.raises(RuntimeError, match="video not started"):
        d.frame_source()


def test_video_port_held_by_another_program_raises_with_the_lsof_hint(clock, video):
    video.arrivals = [OSError(errno.EADDRINUSE, "Address already in use")]
    d, fake = make(clock, dry_run=True, connect=False, video=True)
    with pytest.raises(RuntimeError, match=r"lsof -nP -iUDP:11111") as e:
        d.connect()
    assert e.value.__cause__ is None and "held by another program" in str(e.value)
    assert fake.streamons == 1


@pytest.mark.parametrize("dry_run", [False, True], ids=["live", "dry_run"])
def test_frame_source_without_video_raises(clock, dry_run):
    d, _ = make(clock, dry_run=dry_run)
    with pytest.raises(RuntimeError, match="video not started"):
        d.frame_source()


def _free_udp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_video_packets_arrive_true_on_a_datagram():
    import threading

    port = _free_udp_port()
    stop = threading.Event()

    def blast():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            while not stop.wait(0.01):
                s.sendto(b"\x00\x00\x00\x01", ("127.0.0.1", port))

    t = threading.Thread(target=blast, daemon=True)
    t.start()
    try:
        assert tello_mod.video_packets_arrive(port, 2.0) is True
    finally:
        stop.set()


def test_video_packets_arrive_false_on_silence():
    assert tello_mod.video_packets_arrive(_free_udp_port(), 0.05) is False


def test_video_packets_arrive_oserror_when_the_port_is_held():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as other:
        other.bind(("", 0))
        with pytest.raises(OSError):
            tello_mod.video_packets_arrive(other.getsockname()[1], 0.05)


# ====================================================================== TelloDrone: close()
CLOSE_STATES = ["ground", "taking_off", "flying", "moving", "landing", "landed", "after_emergency"]
# state -> (live: 'land' sent by close, djitellopy is_flying after close)
CLOSE_LIVE = {"ground": (False, False), "taking_off": (True, True), "flying": (True, True), "moving": (True, True),
              "landing": (True, True), "landed": (False, False), "after_emergency": (False, False)}


@pytest.mark.parametrize("dry_run", [False, True], ids=["live", "dry_run"])
@pytest.mark.parametrize("state", CLOSE_STATES)
def test_close_lands_if_airborne_stops_video_and_ends_djitellopy(clock, video, dry_run, state):
    d, fake = make(clock, dry_run=dry_run, video=True)
    to_state(d, fake, clock, state)
    lands_before = fake.cmds.count("land")
    src = d.frame_source()
    d.close()
    land_sent, is_flying = CLOSE_LIVE[state] if not dry_run else (False, False)
    assert fake.cmds.count("land") - lands_before == int(land_sent)
    # djitellopy's end() lands again when is_flying: close() arms it as a safety net (never in dry run)
    assert fake.is_flying is is_flying
    assert src.stopped and d._source is None and fake.ends == 1
    if dry_run:
        assert fake.motion_sent() == []


def test_close_never_raises_when_djitellopy_end_fails(clock, caplog):
    d, fake = make(clock)
    fake.end_error = RuntimeError("socket already closed")
    with caplog.at_level(logging.WARNING, logger="kite.tello"):
        d.close()
    assert fake.ends == 1 and "tello.end(): socket already closed" in caplog.text


def test_close_while_flying_with_wifi_down_does_not_raise_and_still_ends(clock, video):
    d, fake = make(clock, video=True)
    fly(d, fake, clock)
    fake.down = True
    d.close()
    assert d.link_error and fake.ends == 1 and fake.is_flying is True and d._source is None


def test_close_before_connect_is_a_no_op():
    d = TelloDrone(video=False)
    d.close()
    assert d.tello is None


def test_close_twice_ends_twice_without_error(clock):
    d, fake = make(clock)
    d.close()
    d.close()
    assert fake.ends == 2 and fake.cmds == []


# ====================================================================== TelloDrone: _send, min gap, dry-run filter
@pytest.mark.parametrize("cmd, sent", [
    ("command", True), ("streamon", True), ("streamoff", True), ("battery?", True), ("tof?", True),
    ("takeoff", False), ("land", False), ("stop", False), ("speed 50", False), ("forward 20", False),
    ("cw 90", False), ("rc 1 2 3 4", False), ("emergency", False),
], ids=lambda v: str(v))
def test_dry_run_send_filter_only_lets_queries_and_link_commands_through(clock, cmd, sent):
    d, fake = make(clock, dry_run=True)
    assert d._send(cmd) is True
    assert fake.cmds == ([cmd] if sent else [])
    assert d.log[-1] == (clock.time(), cmd)  # logged either way


@pytest.mark.parametrize("dry_run", [False, True], ids=["live", "dry_run"])
def test_sends_closer_than_min_gap_wait_out_the_remainder(clock, dry_run):
    d, fake = make(clock, dry_run=dry_run)
    clock.sleeps.clear()
    d._send("battery?")
    assert clock.sleeps == []  # long after connect: no wait
    clock.advance(0.03)
    d._send("battery?")
    assert clock.sleeps == [pytest.approx(MIN_GAP_S - 0.03)]
    clock.advance(MIN_GAP_S)
    d._send("battery?")
    assert len(clock.sleeps) == 1


def test_min_gap_uses_monotonic_time_not_the_wall_clock(clock):
    d, fake = make(clock)
    d._send("battery?")
    clock.step_wall(-3600)  # NTP steps the wall clock back an hour
    clock.advance(0.2)
    d._send("battery?")
    clock.step_wall(+7200)
    d._send("battery?")
    assert clock.sleeps == [pytest.approx(MIN_GAP_S)] and len(fake.cmds) == 3


@pytest.mark.parametrize("dry_run", [False, True], ids=["live", "dry_run"])
def test_every_discrete_command_is_logged_with_wall_time(clock, dry_run):
    d, fake = make(clock, dry_run=dry_run)
    d.takeoff()
    assert d.log[-1] == (clock.time(), "takeoff")


# ====================================================================== TelloDrone: discrete command table
DISCRETE = [
    # id, call, sent text, timeout_s, expected_s   (move speed 50 cm/s unless said otherwise)
    ("takeoff", lambda d: d.takeoff(), "takeoff", 20.0, 5.0),
    ("land", lambda d: d.land(), "land", 15.0, 4.0),
    ("forward_100", lambda d: d.move("forward", 100), "forward 100", 100 / 50 + 7.5, 100 / 50 + 1.5),
    ("up_20", lambda d: d.move("up", 20), "up 20", 20 / 50 + 7.5, 20 / 50 + 1.5),
    ("back_500", lambda d: d.move("back", 500), "back 500", 500 / 50 + 7.5, 500 / 50 + 1.5),
    ("left_33_point_4_rounds", lambda d: d.move("left", 33.4), "left 33", 33 / 50 + 7.5, 33 / 50 + 1.5),
    ("down_50", lambda d: d.move("down", 50), "down 50", 50 / 50 + 7.5, 50 / 50 + 1.5),
    ("right_200", lambda d: d.move("right", 200), "right 200", 200 / 50 + 7.5, 200 / 50 + 1.5),
    ("cw_90", lambda d: d.rotate(90), "cw 90", 90 / 60 + 6.0, 90 / 60 + 1.0),
    ("ccw_90", lambda d: d.rotate(-90), "ccw 90", 90 / 60 + 6.0, 90 / 60 + 1.0),
    ("cw_360", lambda d: d.rotate(360), "cw 360", 360 / 60 + 6.0, 360 / 60 + 1.0),
    ("ccw_1", lambda d: d.rotate(-1), "ccw 1", 1 / 60 + 6.0, 1 / 60 + 1.0),
    ("speed_30", lambda d: d.set_speed(30), "speed 30", 5.0, 0.2),
]
DISCRETE_IDS = [r[0] for r in DISCRETE]


@pytest.mark.parametrize("name, call, text, timeout_s, expected_s", DISCRETE, ids=DISCRETE_IDS)
def test_discrete_command_is_sent_once_and_runs_non_blocking(clock, name, call, text, timeout_s, expected_s):
    d, fake = make(clock)
    t0 = clock.monotonic()
    call(d)
    assert fake.cmds == [text]
    assert d.busy() and d.last_result() is None
    p = d._pending
    assert (p.cmd, p.t_sent) == (text, t0)
    assert p.timeout_s == pytest.approx(timeout_s) and p.expected_s == pytest.approx(expected_s)
    assert clock.monotonic() == t0  # did not block (connect was long enough ago: no min-gap wait)


@pytest.mark.parametrize("speed, cm, expected_s", [(50, 100, 3.5), (10, 100, 11.5), (5, 100, 11.5), (0, 100, 11.5),
                                                   (100, 500, 6.5)], ids=lambda v: str(v))
def test_move_expected_duration_floors_speed_at_10_cm_s(clock, speed, cm, expected_s):
    d, fake = make(clock, move_speed_cm_s=speed)
    d.move("forward", cm)
    assert d._pending.expected_s == pytest.approx(expected_s)
    assert d._pending.timeout_s == pytest.approx(expected_s + 6.0)


REPLIES = [
    # id, replies, still busy, last_result
    ("ok", [b"ok"], False, "ok"),
    ("ok_upper_crlf", [b"OK\r\n"], False, "ok"),
    ("ok_as_str", ["ok"], False, "ok"),
    ("ok_bytearray", [bytearray(b"ok")], False, "ok"),
    ("ok_padded", [b"  ok  "], False, "ok"),
    ("ok_after_undecodable_byte", [b"\xffok"], False, "ok"),
    ("error_bare", [b"error"], False, "error: error"),
    ("error_not_joystick", [b"error Not joystick"], False, "error: error Not joystick"),
    ("error_upper", [b"ERROR No valid imu"], False, "error: ERROR No valid imu"),
    ("out_of_range", [b"out of range"], False, "error: out of range"),
    ("motor_stop", [b"Motor stop"], False, "error: Motor stop"),
    ("number_then_ok", [b"87", b"ok"], False, "ok"),
    ("tof_reading_then_error", [b"801mm", b"error x"], False, "error: error x"),
    ("first_answer_wins", [b"ok", b"error late"], False, "ok"),
    ("number_only_still_busy", [b"87"], True, None),
    ("empty_still_busy", [b""], True, None),
    ("whitespace_still_busy", [b"\r\n"], True, None),
    ("unknown_command_still_busy", [b"unknown command: foo"], True, None),
    ("garbage_still_busy", [b"\xff\xfe\x00"], True, None),
]


@pytest.mark.parametrize("cmd", ["move", "rotate", "takeoff"])
@pytest.mark.parametrize("_id, replies, busy, result", REPLIES, ids=[r[0] for r in REPLIES])
def test_reply_classification(clock, cmd, _id, replies, busy, result):
    d, fake = make(clock)
    {"move": lambda: d.move("forward", 100), "rotate": lambda: d.rotate(45), "takeoff": d.takeoff}[cmd]()
    fake.reply(*replies)
    assert d.busy() is busy and d.last_result() == result
    if result is not None:
        assert d._pending is None
    else:
        assert fake.responses == []  # non-answers are consumed, not left to confuse the next poll


def test_leftover_replies_after_an_answer_are_kept_until_the_next_command_clears_them(clock):
    d, fake = make(clock)
    d.rotate(45)
    fake.reply("ok", "ok")
    assert not d.busy() and fake.responses == [b"ok"]
    clock.advance(1)
    d.rotate(45)
    assert fake.responses == [] and d.busy()


@pytest.mark.parametrize("name, call, text, timeout_s, expected_s", DISCRETE, ids=DISCRETE_IDS)
def test_silent_drone_times_out_strictly_after_the_command_timeout(clock, name, call, text, timeout_s, expected_s):
    d, fake = make(clock)
    if name == "takeoff":
        fake.state = dict(fake.state, h=0, tof=0)  # still on the ground (up there, a lost 'ok' counts as flying)
    call(d)
    clock.advance(timeout_s)
    assert d.busy()  # exactly at the timeout: still waiting (strict >)
    clock.advance(1e-6)
    assert not d.busy() and d.last_result() == "error: timeout" and d._pending is None


def test_a_reply_that_lands_after_the_timeout_but_before_the_poll_still_counts(clock):
    d, fake = make(clock)
    d.move("forward", 100)
    clock.advance(60)
    fake.reply("ok")
    assert not d.busy() and d.last_result() == "ok"


def test_wall_clock_steps_never_time_out_or_stall_a_command(clock):
    d, fake = make(clock)
    fake.state = dict(fake.state, h=0, tof=0)  # on the ground: the timeout is not overridden by the height
    d.takeoff()
    clock.step_wall(+3600)
    assert d.busy()
    clock.step_wall(-7200)
    assert d.busy()
    clock.advance(20.0 + 1e-6)
    assert not d.busy() and d.last_result() == "error: timeout"


@pytest.mark.parametrize("name, call, text, timeout_s, expected_s", DISCRETE, ids=DISCRETE_IDS)
def test_stale_replies_queued_before_a_command_are_never_its_answer(clock, name, call, text, timeout_s, expected_s):
    d, fake = make(clock)
    fake.reply("ok", "error stale")
    call(d)
    assert d.busy() and d.last_result() is None


# -------------------------------------------------- state after completion
COMPLETION = [
    # id, setup state, command, reply (None: timeout), h (cm), result, flying after, djitellopy is_flying after
    ("takeoff_ok", "ground", "takeoff", "ok", 150, "ok", True, True),
    ("takeoff_error", "ground", "takeoff", "error", 0, "error: error", False, False),
    ("takeoff_timeout", "ground", "takeoff", None, 0, "error: timeout", False, False),
    ("takeoff_timeout_but_up", "ground", "takeoff", None, 150, "ok", True, True),  # the 'ok' was lost: it flies
    ("land_ok", "flying", "land", "ok", 150, "ok", False, False),
    ("land_error_airborne", "flying", "land", "error", 150, "error: error", True, True),
    ("land_error_on_ground", "flying", "land", "error", 20, "ok", False, False),
    ("land_error_just_above_ground", "flying", "land", "error", 21, "error: error", True, True),
    ("land_timeout_on_ground", "flying", "land", None, 0, "ok", False, False),
    ("land_timeout_airborne", "flying", "land", None, 150, "error: timeout", True, True),
    ("land_timeout_no_height", "flying", "land", None, None, "error: timeout", True, True),
    ("land_motor_stop_on_ground", "flying", "land", "Motor stop", 5, "ok", False, False),
    ("land_ok_from_ground", "ground", "land", "ok", 0, "ok", False, False),
    ("move_ok", "flying", "move", "ok", 150, "ok", True, True),
    ("move_error", "flying", "move", "error Not joystick", 150, "error: error Not joystick", True, True),
    ("move_error_low_does_not_count_as_landed", "flying", "move", "error", 10, "error: error", True, True),
    ("rotate_timeout", "flying", "rotate", None, 150, "error: timeout", True, True),
    ("speed_ok", "ground", "speed", "ok", 0, "ok", False, False),
]


@pytest.mark.parametrize("_id, state, cmd, reply, h, result, flying, is_flying", COMPLETION,
                         ids=[c[0] for c in COMPLETION])
def test_flight_state_after_a_command_completes(clock, _id, state, cmd, reply, h, result, flying, is_flying):
    d, fake = make(clock)
    to_state(d, fake, clock, state)
    {"takeoff": d.takeoff, "land": d.land, "move": lambda: d.move("forward", 100), "rotate": lambda: d.rotate(90),
     "speed": lambda: d.set_speed(40)}[cmd]()
    fake.state = dict(fake.state, h=h)
    if reply is None:
        clock.advance(d._pending.timeout_s + 0.01)
    else:
        fake.reply(reply)
    assert not d.busy()
    assert d.last_result() == result and d.flying is flying and fake.is_flying is is_flying


@pytest.mark.parametrize("name, call, text, timeout_s, expected_s", DISCRETE, ids=DISCRETE_IDS)
def test_dry_run_command_completes_ok_at_its_expected_duration_and_sends_nothing(clock, name, call, text, timeout_s,
                                                                                 expected_s):
    d, fake = make(clock, dry_run=True)
    if name == "land":
        fly(d, fake, clock)
    call(d)
    clock.advance(expected_s - 1e-6)
    assert d.busy() and d.last_result() is None
    clock.step_wall(+3600)  # the wall clock does not finish it early
    assert d.busy()
    clock.advance(1e-6)
    assert not d.busy() and d.last_result() == "ok"
    assert fake.motion_sent() == [] and any(c == text for _, c in d.log)


@pytest.mark.parametrize("seq, flying", [(["takeoff"], True), (["takeoff", "land"], False),
                                         (["takeoff", "move", "rotate"], True), (["land"], False)],
                         ids=lambda v: "-".join(v) if isinstance(v, list) else str(v))
def test_dry_run_tracks_flying_but_never_arms_djitellopy(clock, seq, flying):
    d, fake = make(clock, dry_run=True)
    for s in seq:
        {"takeoff": d.takeoff, "land": d.land, "move": lambda: d.move("up", 50), "rotate": lambda: d.rotate(30)}[s]()
        finish(d, fake, clock)
    assert d.flying is flying and fake.is_flying is False and fake.motion_sent() == []


# -------------------------------------------------- one discrete command at a time
SECOND = {
    "takeoff": lambda d: d.takeoff(),
    "move": lambda d: d.move("forward", 50),
    "rotate": lambda d: d.rotate(30),
    "speed": lambda d: d.set_speed(20),
}


@pytest.mark.parametrize("dry_run", [False, True], ids=["live", "dry_run"])
@pytest.mark.parametrize("first", ["taking_off", "moving", "rotating", "setting_speed", "landing"])
@pytest.mark.parametrize("second", list(SECOND))
def test_second_discrete_command_while_one_runs_raises_and_sends_nothing(clock, dry_run, first, second):
    d, fake = make(clock, dry_run=dry_run)
    to_state(d, fake, clock, first)
    pending, n = d._pending.cmd, len(fake.cmds)
    speed = d.move_speed_cm_s
    with pytest.raises(RuntimeError, match="still running"):
        SECOND[second](d)
    assert d._pending.cmd == pending and len(fake.cmds) == n
    assert d.move_speed_cm_s == speed  # a refused set_speed changes nothing


@pytest.mark.parametrize("call", [lambda d: d.move("forward", 10), lambda d: d.move("sideways", 50),
                                  lambda d: d.rotate(0), lambda d: d.rotate(400)],
                         ids=["move_too_short", "move_bad_direction", "rotate_zero", "rotate_too_far"])
def test_invalid_discrete_arguments_raise_before_anything_is_sent(clock, call):
    d, fake = make(clock)
    fly(d, fake, clock)
    n = len(fake.cmds)
    with pytest.raises(ValueError):
        call(d)
    assert len(fake.cmds) == n and d._pending is None and d.last_result() == "ok"


# -------------------------------------------------- network down at send time
@pytest.mark.parametrize("name, call, text, timeout_s, expected_s", DISCRETE, ids=DISCRETE_IDS)
def test_discrete_command_that_cannot_be_sent_fails_at_once_and_can_be_resent(clock, name, call, text, timeout_s,
                                                                              expected_s):
    d, fake = make(clock)
    fake.state = dict(fake.state, h=150)  # airborne height: a failed land is not mistaken for "already down"
    fake.down = True
    call(d)
    assert not d.busy() and d._pending is None
    assert d.last_result() == f"error: not sent ({NET_DOWN})" and d.link_error == str(NET_DOWN)
    fake.down = False
    clock.advance(1)
    call(d)  # free to re-send: no RuntimeError
    assert fake.cmds[-1] == text and d.busy() and d.link_error is None


def test_a_land_that_cannot_be_sent_is_never_counted_as_landed(clock):
    # the h <= 0.2 m rule needs a land that reached the drone: an unsent one (Wi-Fi down, telemetry frozen low) keeps
    # flying set, so the governor, close() and djitellopy's end() keep trying once the link is back
    d, fake = make(clock)
    fly(d, fake, clock)
    fake.state = dict(fake.state, h=0)
    fake.down = True
    d.land()
    assert d.last_result().startswith("error: not sent") and d.flying and fake.is_flying


def test_a_takeoff_whose_ok_is_lost_counts_as_flying_once_it_is_up(clock):
    d, fake = make(clock)
    fake.state = dict(fake.state, h=0, tof=0)
    d.takeoff()
    fake.state = dict(fake.state, h=90, tof=92)  # it climbed, but the 'ok' never came back
    clock.advance(20.0 + 1e-6)
    assert not d.busy() and d.last_result() == "ok" and d.flying and fake.is_flying


# ====================================================================== TelloDrone: land()
LAND_FROM = [
    # id, state, new commands sent by land(), flying right after the call, after a stop (cancelled command)
    ("idle_on_ground", "ground", ["land"], False, False),
    ("hovering", "flying", ["land"], True, False),
    ("during_takeoff", "taking_off", ["land"], True, False),
    ("during_move", "moving", ["stop", "land"], True, True),
    ("during_rotation", "rotating", ["stop", "land"], True, True),
    ("during_speed", "setting_speed", ["stop", "land"], True, True),
    ("already_landing", "landing", [], True, False),
    ("after_landing", "landed", ["land"], False, False),
]


@pytest.mark.parametrize("_id, state, sent, flying, settles", LAND_FROM, ids=[r[0] for r in LAND_FROM])
def test_land_from_every_state(clock, _id, state, sent, flying, settles):
    d, fake = make(clock)
    to_state(d, fake, clock, state)
    n = len(fake.cmds)
    clock.sleeps.clear()
    d.land()
    assert fake.cmds[n:] == sent and d.flying is flying
    assert d.busy() and d._pending.cmd == "land"
    # after a stop (or the takeoff packet) only the 0.1 s command gap applies: land never waits the stop's settle
    # window out. The cancelled command's replies are judged by busy() instead (test below).
    expected = MIN_GAP_S if settles or state == "taking_off" else 0.0
    assert sum(clock.sleeps) == pytest.approx(expected)
    assert d._pending.after_stop is (settles and True)


@pytest.mark.parametrize("_id, state, sent, flying, settles", LAND_FROM, ids=[r[0] for r in LAND_FROM])
def test_dry_run_land_from_every_state_sends_nothing_and_skips_the_settle_window(clock, _id, state, sent, flying,
                                                                                  settles):
    d, fake = make(clock, dry_run=True)
    to_state(d, fake, clock, state)
    clock.sleeps.clear()
    d.land()
    assert fake.motion_sent() == [] and d.flying is flying and d._pending.cmd == "land"
    # no settle window in dry run; the 0.1 s gap after the (logged) stop / takeoff still applies
    assert sum(clock.sleeps) == pytest.approx(MIN_GAP_S if settles or state == "taking_off" else 0.0)
    clock.advance(4.0)
    assert not d.busy() and d.last_result() == "ok" and not d.flying


def test_land_during_takeoff_drops_the_takeoff_wait_and_lands_on_the_land_reply(clock):
    d, fake = make(clock)
    d.takeoff()
    d.land()
    fake.reply("ok")  # whichever 'ok' comes first is taken as the land's: it was cleared before 'land' went out
    assert not d.busy() and not d.flying and fake.is_flying is False


def test_late_move_replies_after_a_cancelled_move_do_not_end_the_landing(clock):
    d, fake = make(clock)
    fly(d, fake, clock)
    d.move("forward", 200)
    clock.on_sleep.append(lambda s: fake.reply("ok"))  # the cancelled move / the stop answer meanwhile
    d.land()
    clock.on_sleep.clear()
    fake.reply("ok")
    assert d.busy() and d.flying  # those 'ok's did not end the landing: it is still 1.5 m up
    clock.advance(STOP_SETTLE_S)
    assert d.busy() and d.flying  # past the window too: an 'ok' with an airborne height is not a landing
    fake.state = dict(fake.state, h=0, tof=10)  # touched down
    fake.reply("ok")
    assert not d.busy() and not d.flying and d.last_result() == "ok"


def test_land_after_a_cancelled_move_ends_on_the_height_even_if_its_ok_came_early(clock):
    d, fake = make(clock)
    fly(d, fake, clock)
    d.move("forward", 200)
    d.land()
    fake.reply("ok")  # the stop's, the move's or the land's: it cannot tell
    assert d.busy()
    fake.state = dict(fake.state, h=0, tof=10)
    assert not d.busy() and not d.flying


def test_land_after_a_cancelled_move_without_a_height_waits_out_the_window(clock):
    d, fake = make(clock)
    fly(d, fake, clock)
    d.move("forward", 200)
    d.land()
    fake.state = {}  # no state stream
    fake.reply("ok")
    assert d.busy()
    clock.advance(STOP_SETTLE_S)
    assert not d.busy() and not d.flying


def test_land_after_a_cancelled_move_ignores_an_error_in_the_window_only(clock):
    d, fake = make(clock)
    fly(d, fake, clock)
    d.move("forward", 200)
    d.land()
    fake.reply("error Not joystick")  # the cancelled move's
    assert d.busy()
    clock.advance(STOP_SETTLE_S)
    fake.reply("error")  # the land's own
    assert not d.busy() and d.last_result() == "error: error" and d.flying


# ====================================================================== TelloDrone: stop()
STOP_FROM = [
    # id, state, 'stop' sent, still busy after
    ("idle_hover", "flying", True, False),
    ("on_ground", "ground", True, False),
    ("move", "moving", True, False),
    ("rotation", "rotating", True, False),
    ("speed", "setting_speed", True, False),
    ("takeoff_never_interrupted", "taking_off", False, True),
    ("landing_never_interrupted", "landing", False, True),
]


@pytest.mark.parametrize("dry_run", [False, True], ids=["live", "dry_run"])
@pytest.mark.parametrize("_id, state, sent, busy", STOP_FROM, ids=[r[0] for r in STOP_FROM])
def test_stop_from_every_state(clock, dry_run, _id, state, sent, busy):
    d, fake = make(clock, dry_run=dry_run)
    to_state(d, fake, clock, state)
    n, logged = len(fake.cmds), len(d.log)
    d.stop()
    assert d.busy() is busy
    if busy:
        assert d.last_result() is None and len(d.log) == logged
    else:
        assert d.last_result() == "ok" and d.log[-1][1] == "stop"
    assert fake.cmds[n:] == (["stop"] if sent and not dry_run else [])


@pytest.mark.parametrize("gap, waited", [(0.0, STOP_SETTLE_S), (0.25, 0.75), (0.999, 0.001), (1.0, 0.0), (5.0, 0.0)],
                         ids=lambda v: str(v))
def test_next_command_after_stop_waits_out_the_settle_window(clock, gap, waited):
    d, fake = make(clock)
    fly(d, fake, clock)
    d.stop()
    clock.advance(gap)
    clock.sleeps.clear()
    d.rotate(45)
    assert sum(clock.sleeps) == pytest.approx(waited, abs=1e-9)  # the settle wait also covers the 0.1 s gap
    assert d._ignore_until == 0.0 and fake.cmds[-1] == "cw 45"


def test_settle_window_is_skipped_in_dry_run(clock):
    d, fake = make(clock, dry_run=True)
    fly(d, fake, clock)
    d.stop()
    clock.advance(MIN_GAP_S)
    clock.sleeps.clear()
    d.rotate(45)
    assert clock.sleeps == [] and d._ignore_until == 0.0


def test_stop_with_wifi_down_does_not_raise_and_reports_ok(clock):
    d, fake = make(clock)
    fly(d, fake, clock)
    d.move("forward", 100)
    fake.down = True
    d.stop()
    assert d.last_result() == "ok" and not d.busy() and d.link_error


# ====================================================================== TelloDrone: emergency()
@pytest.mark.parametrize("dry_run", [False, True], ids=["live", "dry_run"])
@pytest.mark.parametrize("state", ["ground", "taking_off", "flying", "moving", "landing"])
def test_emergency_cuts_motors_from_every_state(clock, dry_run, state):
    d, fake = make(clock, dry_run=dry_run)
    to_state(d, fake, clock, state)
    n = len(fake.cmds)
    clock.advance(0.01)  # 10 ms after the last send: no min-gap wait for an emergency
    clock.sleeps.clear()
    assert d.emergency() is True
    assert fake.cmds[n:] == ([] if dry_run else ["emergency"]) and clock.sleeps == []
    assert not d.flying and not d.busy() and d.last_result() == "ok" and d.log[-1][1] == "emergency"


def test_emergency_with_wifi_down_reports_false_and_changes_nothing(clock):
    d, fake = make(clock)
    fly(d, fake, clock)
    d.move("forward", 100)
    fake.down = True
    assert d.emergency() is False
    assert d.flying and d.busy() and d._pending.cmd == "forward 100" and d.link_error


def test_dry_run_emergency_succeeds_even_with_wifi_down(clock):
    d, fake = make(clock, dry_run=True)
    fly(d, fake, clock)
    fake.down = True
    assert d.emergency() is True and not d.flying and d.link_error is None


def test_emergency_disarms_djitellopy(clock):
    # like djitellopy's own emergency(): is_flying is cleared, so close()->end() never sends 'land' to a fallen drone
    d, fake = make(clock)
    fly(d, fake, clock)
    d.emergency()
    assert fake.is_flying is False


# ====================================================================== TelloDrone: rc()
@pytest.mark.parametrize("sticks, sent", [
    ((0, 0, 0, 0), (0, 0, 0, 0)),
    ((10, -20, 30, -40), (10, -20, 30, -40)),
    ((100, -100, 100, -100), (100, -100, 100, -100)),
    ((150, -150, 1e6, -1e6), (100, -100, 100, -100)),
    ((12.6, -3.2, 2.5, 3.5), (13, -3, 2, 4)),
    ((np.float64(12.6), np.int64(5), np.float32(-7.7), np.int32(0)), (13, 5, -8, 0)),
    ((True, False, 0, 0), (1, 0, 0, 0)),
], ids=["zeros", "plain_ints", "full_scale", "clamped_to_100", "rounded_half_even", "numpy_scalars", "bools"])
def test_rc_reaches_djitellopy_clamped_rounded_plain_ints(clock, sticks, sent):
    d, fake = make(clock)
    fly(d, fake, clock)
    d.rc(*sticks)
    assert fake.rcs[-1] == sent and all(type(v) is int for v in fake.rcs[-1])


def test_rc_on_the_ground_is_still_sent(clock):
    d, fake = make(clock)
    d.rc(0, 0, 0, 0)  # the keep-alive also runs before takeoff
    assert fake.rcs == [(0, 0, 0, 0)]


@pytest.mark.parametrize("dry_run", [False, True], ids=["live", "dry_run"])
@pytest.mark.parametrize("state", ["taking_off", "moving", "rotating", "setting_speed", "landing"])
def test_rc_is_ignored_while_any_discrete_command_runs(clock, dry_run, state):
    d, fake = make(clock, dry_run=dry_run)
    to_state(d, fake, clock, state)
    logged = len(d.log)
    d.rc(0, 50, 0, 50)
    assert fake.rcs == [] and len(d.log) == logged


@pytest.mark.parametrize("sticks, logged", [((0, 0, 0, 0), False), ((0, 0, 0, 1), True), ((0.4, 0, 0, 0), False),
                                            ((300, 0, 0, 0), True)], ids=lambda v: str(v))
def test_dry_run_rc_is_logged_when_nonzero_and_never_sent(clock, sticks, logged):
    d, fake = make(clock, dry_run=True)
    fly(d, fake, clock)
    n = len(d.log)
    d.rc(*sticks)
    assert fake.rcs == []
    if logged:
        vals = tuple(clamp_rc(v) for v in sticks)
        assert d.log[n:] == [(clock.time(), "rc {} {} {} {}".format(*vals))]
    else:
        assert len(d.log) == n


@pytest.mark.parametrize("stamp_offset, reset", [(+5.0, True), (+0.001, True), (0.0, False), (-0.5, False)],
                         ids=["clock_stepped_back_5s", "clock_stepped_back_1ms", "same_instant", "normal"])
def test_rc_unblocks_djitellopy_after_the_wall_clock_steps_back(clock, stamp_offset, reset):
    d, fake = make(clock)
    fake.last_rc_control_timestamp = clock.time() + stamp_offset
    before = fake.last_rc_control_timestamp
    d.rc(0, 10, 0, 0)
    assert fake.last_rc_control_timestamp == (0.0 if reset else before)


def test_rc_tolerates_a_tello_without_an_rc_timestamp(clock):
    d, fake = make(clock)
    assert not hasattr(fake, "last_rc_control_timestamp")
    d.rc(1, 2, 3, 4)
    assert fake.rcs == [(1, 2, 3, 4)]


def test_rc_does_not_wait_for_or_count_toward_the_command_gap(clock):
    d, fake = make(clock)
    fly(d, fake, clock)
    d.rotate(10)
    fake.reply("ok")
    d.busy()
    last_send = d._last_send
    clock.sleeps.clear()
    d.rc(0, 0, 0, 10)
    assert clock.sleeps == [] and d._last_send == last_send and fake.rcs[-1] == (0, 0, 0, 10)


@pytest.mark.parametrize("v", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_rc_with_a_non_finite_stick_sends_hover_on_that_channel(clock, v):
    d, fake = make(clock)
    d.rc(10, v, 0, 0)
    assert fake.rcs[-1] == (10, 0, 0, 0)


# ====================================================================== TelloDrone: link_error
def test_link_error_is_set_on_failure_logged_once_and_cleared_on_recovery(clock, caplog):
    d, fake = make(clock)
    with caplog.at_level(logging.WARNING, logger="kite.tello"):
        d.rc(0, 0, 0, 0)
        assert d.link_error is None
        fake.down = True
        for _ in range(5):
            d.rc(0, 0, 0, 0)  # the keep-alive keeps retrying every tick
        assert d.link_error == str(NET_DOWN)
        fails = [r for r in caplog.records if "Wi-Fi send failed" in r.getMessage()]
        assert len(fails) == 1
        fake.down = False
        d.rc(0, 0, 0, 0)
        assert d.link_error is None
        assert sum("Wi-Fi sends work again" in r.getMessage() for r in caplog.records) == 1
        d.rc(0, 0, 0, 0)
        assert sum("Wi-Fi sends work again" in r.getMessage() for r in caplog.records) == 1
    assert fake.rcs == [(0, 0, 0, 0)] * 3


@pytest.mark.parametrize("err", [OSError(errno.ENOBUFS, "No buffer space available"),
                                 OSError(errno.EHOSTUNREACH, "No route to host"), socket.gaierror("dns"),
                                 ConnectionRefusedError("refused"), socket.timeout("timed out")],
                         ids=["ENOBUFS", "EHOSTUNREACH", "gaierror", "refused", "timeout"])
def test_every_oserror_flavour_becomes_link_error(clock, err):
    d, fake = make(clock)

    def boom(*_):
        raise err

    fake.send_rc_control = boom
    fake.send_command_without_return = boom
    d.rc(0, 0, 0, 0)
    assert d.link_error == str(err)
    d.takeoff()
    assert d.last_result() == f"error: not sent ({err})"


def test_non_network_errors_are_not_swallowed(clock):
    d, fake = make(clock)

    def boom(*_):
        raise TypeError("enforce_types")

    fake.send_rc_control = boom
    with pytest.raises(TypeError):
        d.rc(0, 0, 0, 0)
    assert d.link_error is None


# ====================================================================== TelloDrone: telemetry()
TEL_FIELDS = [
    # id, state key, raw, Telemetry attribute, expected
    ("yaw", "yaw", 10, "yaw_deg", 10.0),
    ("yaw_negative", "yaw", -179, "yaw_deg", -179.0),
    ("pitch", "pitch", -2, "pitch_deg", -2.0),
    ("roll", "roll", 1, "roll_deg", 1.0),
    ("h_cm_to_m", "h", 80, "height_m", 0.8),
    ("h_zero", "h", 0, "height_m", 0.0),
    ("h_below_takeoff_point", "h", -5, "height_m", -0.05),
    ("tof_cm_to_m", "tof", 85, "tof_m", 0.85),
    ("tof_10cm_on_ground_kept", "tof", 10, "tof_m", 0.1),
    ("tof_9cm_dropped", "tof", 9, "tof_m", None),
    ("tof_8m_kept", "tof", 800, "tof_m", 8.0),
    ("tof_801cm_dropped", "tof", 801, "tof_m", None),
    ("tof_6553_out_of_range", "tof", 6553, "tof_m", None),
    ("tof_zero_dropped", "tof", 0, "tof_m", None),
    ("tof_negative_dropped", "tof", -1, "tof_m", None),
    ("vgx_dm_s_to_m_s", "vgx", 3, "vx", 0.3),
    ("vgy_negative", "vgy", -1, "vy", -0.1),
    ("vgz", "vgz", 5, "vz", 0.5),
    ("bat", "bat", 87, "battery_pct", 87.0),
    ("bat_zero", "bat", 0, "battery_pct", 0.0),
    ("time", "time", 5, "flight_time_s", 5.0),
    ("string_number", "h", "85", "height_m", 0.85),
    ("float_value", "bat", 42.5, "battery_pct", 42.5),
]


@pytest.mark.parametrize("_id, key, raw, attr, expected", TEL_FIELDS, ids=[r[0] for r in TEL_FIELDS])
def test_telemetry_field_conversion(clock, _id, key, raw, attr, expected):
    d, fake = make(clock)
    fake.state = dict(fake.state, **{key: raw})
    got = getattr(d.telemetry(), attr)
    assert got == (None if expected is None else pytest.approx(expected))


ATTR_FOR = {"yaw": "yaw_deg", "pitch": "pitch_deg", "roll": "roll_deg", "h": "height_m", "tof": "tof_m", "vgx": "vx",
            "vgy": "vy", "vgz": "vz", "bat": "battery_pct", "time": "flight_time_s"}


@pytest.mark.parametrize("key", list(ATTR_FOR))
@pytest.mark.parametrize("raw", [None, float("nan"), "missing"], ids=["none", "nan", "missing"])
def test_telemetry_unknown_values_read_as_none(clock, key, raw):
    d, fake = make(clock)
    st = dict(fake.state)
    if raw == "missing":
        del st[key]
    else:
        st[key] = raw
    fake.state = st
    tel = d.telemetry()
    assert getattr(tel, ATTR_FOR[key]) is None
    others = [getattr(tel, a) for k, a in ATTR_FOR.items() if k != key]
    assert all(v is not None for v in others)  # one bad field never blanks the others


@pytest.mark.parametrize("scale, raw, expected", [(0.1, 3, 0.3), (0.01, 30, 0.3), (1.0, 2, 2.0)],
                         ids=["dm_s_default", "cm_s", "m_s"])
def test_telemetry_velocity_scale(clock, scale, raw, expected):
    d, fake = make(clock, velocity_scale=scale)
    fake.state = dict(fake.state, vgx=raw, vgy=-raw, vgz=0)
    tel = d.telemetry()
    assert (tel.vx, tel.vy, tel.vz) == (pytest.approx(expected), pytest.approx(-expected), 0.0)


@pytest.mark.parametrize("yaw_sign", [1, -1])
@pytest.mark.parametrize("yaw", [0, 10, -90, 179, -180])
def test_telemetry_yaw_sign(clock, yaw_sign, yaw):
    d, fake = make(clock, yaw_sign=yaw_sign)
    fake.state = dict(fake.state, yaw=yaw)
    assert d.telemetry().yaw_deg == yaw_sign * yaw


def test_telemetry_through_djitellopy_parse_state_drops_garbage_fields(clock):
    djitellopy = pytest.importorskip("djitellopy")
    d, fake = make(clock)
    logging.getLogger("djitellopy").disabled = True
    try:
        fake.state = djitellopy.Tello.parse_state(
            "pitch:0;roll:x;yaw:-45;vgx:0;vgy:0;vgz:0;templ:60;temph:62;tof:abc;h:120;bat:55;baro:12.3;time:9;"
            "agx:1.0;agy:2.0;agz:-998.0;\r\n")
    finally:
        logging.getLogger("djitellopy").disabled = False
    tel = d.telemetry()
    assert tel.roll_deg is None and tel.tof_m is None  # garbage never reaches float()
    assert (tel.yaw_deg, tel.height_m, tel.battery_pct, tel.flight_time_s) == (-45.0, 1.2, 55.0, 9.0)
    fake.state = djitellopy.Tello.parse_state("tof:6553;h:300;\r\n")
    assert d.telemetry().tof_m is None and d.telemetry().height_m == 3.0


def test_telemetry_stamp_is_first_sight_of_each_new_state_dict(clock):
    d, fake = make(clock)
    t0 = clock.time()
    assert d.telemetry().t == t0
    clock.advance(3.0)
    assert d.telemetry().t == t0  # same dict object: djitellopy's receiver has not replaced it (frozen stream)
    fake.state["bat"] = 50  # mutated in place is still the same packet as far as staleness goes
    assert d.telemetry().t == t0
    fake.state = dict(fake.state)
    assert d.telemetry().t == t0 + 3.0
    clock.step_wall(-100)  # stamps are wall-clock (time.time), compared against the caller's time.time()
    fake.state = dict(fake.state)
    assert d.telemetry().t == t0 + 3.0 - 100


@pytest.mark.parametrize("state, error", [({}, None), (None, None), ("unused", KeyError("192.168.10.1")),
                                          ("unused", RuntimeError("no state yet"))],
                         ids=["empty_dict", "none", "keyerror_after_end", "exception"])
def test_telemetry_without_state_is_all_none_and_stamped_never(clock, state, error):
    d, fake = make(clock)
    fake.state = state
    fake.state_error = error
    tel = d.telemetry()
    assert tel.t == 0.0
    assert all(getattr(tel, a) is None for a in ATTR_FOR.values())


# ====================================================================== make_tello
@pytest.mark.parametrize("kind, flag, dry", [("tello", False, False), ("tello", True, True), ("dry_run", False, True),
                                             ("dry_run", True, True)], ids=lambda v: str(v))
def test_make_tello_dry_run_from_flag_or_config(kind, flag, dry):
    cfg = Config(drone=DroneCfg(kind=kind, move_speed_cm_s=35, yaw_sign=-1, video_fps=40, video_backend="opencv"))
    d = make_tello(cfg, dry_run=flag)
    assert isinstance(d, TelloDrone) and d.dry_run is dry and d.tello is None
    assert (d.move_speed_cm_s, d.yaw_sign, d.video_fps, d.video_backend) == (35, -1, 40, "opencv")


@pytest.mark.parametrize("kind", ["sim", "", "TELLO", "webcam"])
def test_make_tello_rejects_other_kinds(kind):
    with pytest.raises(ValueError, match="cannot be used with a real Tello"):
        make_tello(Config(drone=DroneCfg(kind=kind)), dry_run=True)


def test_config_defaults_match_the_documented_limits():
    s, d = SafetyCfg(), DroneCfg()
    assert (s.max_rc, s.min_altitude_m, s.max_altitude_m, s.min_battery_pct, s.max_flight_s, s.frame_timeout_s,
            s.telemetry_timeout_s) == (50, 0.5, 2.3, 20.0, 420.0, 1.0, 1.0)
    assert (d.kind, d.move_speed_cm_s, d.yaw_sign, d.video_fps, d.video_backend, d.video_lag_s) == \
        ("tello", 50, 1, 60, "pyav", 0.25)
    c = Config()
    assert c.safety == s and c.drone == d and Config().safety is not c.safety  # no shared mutable default
    assert SOFT_BAND_M == 0.4 and MIN_GAP_S == 0.1 and STOP_SETTLE_S == 1.0


# ====================================================================== SafetyGovernor: rc rules
@pytest.mark.parametrize("max_rc", [20, 50, 100])
@pytest.mark.parametrize("channel", [0, 1, 2, 3], ids=["lr", "fb", "ud", "yaw"])
@pytest.mark.parametrize("v", [0, 1, -1, 19, 20, 49.6, -49.6, 50, -50, 51, -51, 100, -100, 1000, -1000],
                         ids=lambda v: f"v={v}")
def test_governor_clamps_every_channel_to_max_rc(clock, max_rc, channel, v):
    gov, inner = governed(clock, SafetyCfg(max_rc=max_rc), h=1.5)  # mid-band: no altitude rule applies
    sticks = [0, 0, 0, 0]
    sticks[channel] = v
    gov.rc(*sticks)
    expected = [0, 0, 0, 0]
    expected[channel] = clamp_rc(v, max_rc)
    assert inner.rcs == [tuple(expected)]
    assert all(type(x) is int for x in inner.rcs[0])


HI, LO = 2.5, 0.5
ALT_CFG = dict(max_rc=100, min_altitude_m=LO, max_altitude_m=HI)
ALT = [
    # id, h, tof, ud in, ud out
    ("cruise_climb_untouched", 1.5, None, 80, 80),
    ("cruise_descent_untouched", 1.5, None, -80, -80),
    ("above_ceiling_climb_becomes_descend_20", 3.0, None, 80, -20),
    ("above_ceiling_zero_becomes_descend_20", 3.0, None, 0, -20),
    ("above_ceiling_slow_descent_raised_to_20", 3.0, None, -10, -20),
    ("above_ceiling_fast_descent_kept", 3.0, None, -80, -80),
    ("at_ceiling_climb_cut_to_zero", HI, None, 80, 0),
    ("at_ceiling_descent_untouched", HI, None, -80, -80),
    ("in_ceiling_band_climb_tapered", 2.3, None, 80, int(80 * max(0.0, (HI - 2.3) / SOFT_BAND_M))),
    ("ceiling_band_edge_full_climb", HI - SOFT_BAND_M, None, 80, 80),
    ("below_floor_descent_cut", 0.25, None, -80, 0),
    ("below_floor_climb_allowed", 0.25, None, 80, 80),
    ("at_floor_descent_cut_to_zero", LO, None, -80, 0),
    ("in_floor_band_descent_tapered", 0.7, None, -80, int(-80 * max(0.0, (0.7 - LO) / SOFT_BAND_M))),
    ("floor_band_edge_full_descent", LO + SOFT_BAND_M, None, -80, -80),
    ("in_floor_band_climb_untouched", 0.7, None, 80, 80),
    ("over_a_table_tof_cuts_descent", 1.5, 0.3, -80, 0),
    ("over_a_table_climb_ok", 1.5, 0.3, 80, 80),
    ("tof_agreeing_with_h_rules_the_ceiling", 2.45, 2.6, 80, -20),
    ("tof_beyond_3m_ignored_h_tapers", 2.3, 3.5, 80, int(80 * max(0.0, (HI - 2.3) / SOFT_BAND_M))),
    ("no_h_tof_above_ceiling", None, 2.8, 80, -20),
    ("no_h_low_tof_cuts_descent", None, 0.3, -80, 0),
    ("no_altitude_at_all_climb_passes", None, None, 80, 80),
    ("no_altitude_at_all_descent_passes", None, None, -80, -80),
    ("above_ceiling_over_obstacle_floor_wins_hold", 3.0, 0.3, 80, 0),
    ("low_h_with_far_tof_cuts_descent", 0.4, 3.5, -80, 0),
]


@pytest.mark.parametrize("_id, h, tof, ud, out", ALT, ids=[a[0] for a in ALT])
def test_governor_altitude_rules_on_the_ud_stick_while_flying(clock, _id, h, tof, ud, out):
    gov, inner = governed(clock, SafetyCfg(**ALT_CFG), h=h, tof=tof)
    gov.rc(7, -7, ud, 9)
    assert inner.rcs == [(7, -7, out, 9)]  # the other sticks never change


@pytest.mark.parametrize("_id, h, tof, ud, out", ALT, ids=[a[0] for a in ALT])
def test_governor_altitude_rules_do_not_apply_on_the_ground(clock, _id, h, tof, ud, out):
    gov, inner = governed(clock, SafetyCfg(**ALT_CFG), h=h, tof=tof, flying=False)
    gov.rc(0, 0, ud, 0)
    assert inner.rcs == [(0, 0, ud, 0)] and gov.events == []


@pytest.mark.parametrize("max_rc, h, ud, out", [(30, 2.3, 100, int(30 * max(0.0, (HI - 2.3) / SOFT_BAND_M))),
                                               (30, 3.0, 100, -20), (30, 0.7, -100,
                                                                      int(-30 * max(0.0, (0.7 - LO) / SOFT_BAND_M)))],
                         ids=["taper_applies_after_clamp", "ceiling_push_within_max_rc", "floor_taper_after_clamp"])
def test_governor_clamps_before_the_altitude_taper(clock, max_rc, h, ud, out):
    gov, inner = governed(clock, SafetyCfg(max_rc=max_rc, min_altitude_m=LO, max_altitude_m=HI), h=h)
    gov.rc(0, 0, ud, 0)
    assert inner.rcs[-1][2] == out


@pytest.mark.parametrize("max_rc", [0, 5, 10, 19])
def test_governor_ceiling_push_respects_a_max_rc_below_20(clock, max_rc):
    gov, inner = governed(clock, SafetyCfg(max_rc=max_rc, max_altitude_m=HI), h=3.0)
    gov.rc(0, 0, 0, 0)
    assert abs(inner.rcs[-1][2]) <= max_rc


@pytest.mark.parametrize("h, tof, event", [
    (3.0, None, "ceiling 3.00 m > 2.5 m -> descend"),
    (0.25, None, "floor 0.25 m < 0.5 m -> no descent"),
    (1.5, 0.3, "floor 0.30 m < 0.5 m -> no descent"),
    (2.3, None, None),
    (0.7, None, None),
], ids=["ceiling", "floor", "floor_over_table", "ceiling_taper_is_silent", "floor_taper_is_silent"])
@pytest.mark.parametrize("ud", [50, 0, -50])
def test_governor_altitude_events(clock, h, tof, event, ud):
    gov, inner = governed(clock, SafetyCfg(**ALT_CFG), h=h, tof=tof)
    gov.rc(0, 0, ud, 0)
    assert [m for _, m in gov.events] == ([event] if event else [])
    if event:
        assert gov.events[0][0] == clock.time()


def test_governor_events_collapse_consecutive_repeats_only(clock):
    gov, inner = governed(clock, SafetyCfg(**ALT_CFG), h=3.0)
    for _ in range(3):
        gov.rc(0, 0, 50, 0)
    inner.tel.height_m = 3.1
    gov.rc(0, 0, 50, 0)
    inner.tel.height_m = 3.0
    gov.rc(0, 0, 50, 0)
    assert [m for _, m in gov.events] == ["ceiling 3.00 m > 2.5 m -> descend", "ceiling 3.10 m > 2.5 m -> descend",
                                          "ceiling 3.00 m > 2.5 m -> descend"]


# ====================================================================== SafetyGovernor: check() in every flight state
GOV_STATES = {
    # flying, busy, land already requested
    "ground": (False, False, False),
    "taking_off": (False, True, False),
    "flying": (True, False, False),
    "flying_discrete_move": (True, True, False),
    "landing": (True, True, True),
    "landing_lost_reply": (True, False, True),
    "landed": (False, False, True),
    "after_emergency": (False, False, False),
}
CONDITIONS = ["nominal", "low_battery", "flight_time_over", "video_stale", "no_frame_yet", "telemetry_stale",
              "battery_unknown"]


def _expected_check(state, cond):
    flying, busy, requested = GOV_STATES[state]
    if not flying:
        return "ok", 0, False
    if cond in ("low_battery", "flight_time_over"):
        return "land", int(not requested or not busy), False
    if requested:
        return "land", int(not busy), False
    if cond in ("video_stale", "no_frame_yet", "telemetry_stale"):
        return "hover", 0, not busy
    return "ok", 0, False


@pytest.mark.parametrize("cond", CONDITIONS)
@pytest.mark.parametrize("state", list(GOV_STATES))
def test_governor_check_matrix(clock, state, cond):
    cfg = SafetyCfg(max_flight_s=100.0)
    flying, busy, requested = GOV_STATES[state]
    gov, inner = governed(clock, cfg, flying=flying, busy=busy)
    if state == "after_emergency":
        inner._flying = True
        assert gov.emergency() is True
    t_prime = clock.time()
    inner.tel.t = t_prime
    gov.check(t_prime, t_prime)  # first look: starts the flight timer while flying
    if requested:
        gov.land()
        inner._busy = busy
    clock.advance(101.0 if cond == "flight_time_over" else 3.0)  # also > 2 s since any land was sent
    now = clock.time()
    inner.tel.t = now - (5.0 if cond == "telemetry_stale" else 0.1)
    inner.tel.battery_pct = {"low_battery": 10.0, "battery_unknown": None}.get(cond, 80.0)
    frame_t = None if cond == "no_frame_yet" else now - (5.0 if cond == "video_stale" else 0.1)
    lands, rcs = inner.lands, len(inner.rcs)
    action = gov.check(now, frame_t)
    exp_action, exp_lands, exp_rc0 = _expected_check(state, cond)
    assert action == exp_action
    assert inner.lands - lands == exp_lands
    assert inner.rcs[rcs:] == ([(0, 0, 0, 0)] if exp_rc0 else [])
    assert gov.hovering_for_safety is (action == "hover")
    if action == "land":
        assert gov.land_requested and gov.events[-1][1].startswith("LAND: ")
    if action == "ok" and not flying:
        assert gov._takeoff_t is None


@pytest.mark.parametrize("bat, action", [(None, "ok"), (0, "land"), (5, "land"), (19.9, "land"), (20, "ok"),
                                         (20.1, "ok"), (100, "ok")], ids=lambda v: f"bat={v}")
def test_governor_battery_threshold(clock, bat, action):
    gov, inner = governed(clock, bat=bat)
    now = clock.time()
    assert gov.check(now, now) == action
    if action == "land":
        assert gov.events[-1][1] == f"LAND: battery {bat:.0f}% < 20%" and inner.lands == 1


def test_governor_battery_zero_threshold_never_lands(clock):
    gov, inner = governed(clock, SafetyCfg(min_battery_pct=0.0), bat=0)
    now = clock.time()
    assert gov.check(now, now) == "ok"


@pytest.mark.parametrize("elapsed, action", [(0.0, "ok"), (99.999, "ok"), (100.0, "ok"), (100.001, "land")],
                         ids=lambda v: str(v))
def test_governor_flight_time_limit_is_strict_and_counts_from_the_first_airborne_check(clock, elapsed, action):
    gov, inner = governed(clock, SafetyCfg(max_flight_s=100.0))
    t0 = clock.time()
    assert gov.check(t0, t0) == "ok" and gov._takeoff_t == t0
    inner.tel.t = t0 + elapsed
    assert gov.check(t0 + elapsed, t0 + elapsed) == action
    if action == "land":
        assert gov.events[-1][1] == "LAND: flight time > 100 s"


def test_governor_flight_timer_restarts_after_touching_down(clock):
    gov, inner = governed(clock, SafetyCfg(max_flight_s=100.0))
    t0 = clock.time()
    gov.check(t0, t0)
    inner._flying = False
    assert gov.check(t0 + 50, t0 + 50) == "ok" and gov._takeoff_t is None
    inner._flying = True
    inner.tel.t = t0 + 120
    assert gov.check(t0 + 120, t0 + 120) == "ok" and gov._takeoff_t == t0 + 120


@pytest.mark.parametrize("frame_age, tel_age, action, event", [
    (0.0, 0.0, "ok", None),
    (1.0, 0.0, "ok", None),
    (1.001, 0.0, "hover", "video stale -> hover"),
    (None, 0.0, "hover", "video stale -> hover"),
    (0.0, 1.0, "ok", None),
    (0.0, 1.001, "hover", "telemetry stale -> hover"),
    (5.0, 5.0, "hover", "video stale -> hover"),
    (-0.5, 0.0, "ok", None),
], ids=["fresh", "frame_at_timeout", "frame_past_timeout", "no_frame", "telemetry_at_timeout",
        "telemetry_past_timeout", "both_stale_reports_video", "frame_from_the_future"])
def test_governor_staleness_boundaries(clock, frame_age, tel_age, action, event):
    gov, inner = governed(clock)
    now = clock.time()
    inner.tel.t = now - tel_age
    assert gov.check(now, None if frame_age is None else now - frame_age) == action
    assert [m for _, m in gov.events] == ([event] if event else [])
    assert inner.rcs == ([(0, 0, 0, 0)] if action == "hover" else [])


def test_governor_telemetry_never_stamped_means_hover(clock):
    gov, inner = governed(clock)
    inner.tel.t = 0.0  # TelloDrone.telemetry() before the first state packet
    now = clock.time()
    assert gov.check(now, now) == "hover"


def test_governor_hover_holds_rc_until_data_returns_then_releases(clock):
    gov, inner = governed(clock)
    now = clock.time()
    assert gov.check(now, now - 2) == "hover" and gov.hovering_for_safety
    for _ in range(3):
        gov.rc(0, 40, 0, 0)
        assert gov.check(now, now - 2) == "hover"
    assert inner.rcs == [(0, 0, 0, 0)] * 4  # one rc 0 per check, the mission's sticks never reach the drone
    assert [m for _, m in gov.events] == ["video stale -> hover"]  # repeated hovers log once
    assert gov.check(now, now) == "ok" and not gov.hovering_for_safety
    gov.rc(0, 40, 0, 0)
    assert inner.rcs[-1] == (0, 40, 0, 0)


def test_governor_hover_does_not_send_rc_over_a_running_discrete_command(clock):
    gov, inner = governed(clock, busy=True)
    now = clock.time()
    assert gov.check(now, None) == "hover" and inner.rcs == [] and gov.hovering_for_safety


def test_governor_land_keeps_a_previous_hover_flag(clock):
    # pinned: the land branch returns before hovering_for_safety is reset (rc is blocked by land_requested anyway)
    gov, inner = governed(clock)
    now = clock.time()
    gov.check(now, None)
    inner.tel.battery_pct = 5
    assert gov.check(now, None) == "land" and gov.hovering_for_safety


# -------------------------------------------------- combinations
COMBOS = [
    # id, battery, flight time over, frame stale, tel stale, land requested before -> action, first LAND reason
    ("low_battery_while_video_stale", 10, False, True, False, False, "land", "LAND: battery 10% < 20%"),
    ("low_battery_while_telemetry_stale", 10, False, False, True, False, "land", "LAND: battery 10% < 20%"),
    ("battery_beats_flight_time", 10, True, False, False, False, "land", "LAND: battery 10% < 20%"),
    ("flight_time_while_video_stale", 80, True, True, False, False, "land", "LAND: flight time > 100 s"),
    ("requested_while_video_stale", 80, False, True, False, True, "land", "LAND: requested"),
    ("requested_and_low_battery", 10, False, False, False, True, "land", "LAND: battery 10% < 20%"),
    ("everything_wrong_at_once", 0, True, True, True, True, "land", "LAND: battery 0% < 20%"),
]


@pytest.mark.parametrize("_id, bat, overtime, fstale, tstale, req, action, reason", COMBOS, ids=[c[0] for c in COMBOS])
def test_governor_landing_rules_beat_hovering(clock, _id, bat, overtime, fstale, tstale, req, action, reason):
    gov, inner = governed(clock, SafetyCfg(max_flight_s=100.0))
    t0 = clock.time()
    gov.check(t0, t0)
    if req:
        gov.land()
        inner._busy = False
    clock.advance(101.0 if overtime else 3.0)
    now = clock.time()
    inner.tel.t = now - (5 if tstale else 0)
    inner.tel.battery_pct = bat
    rcs = len(inner.rcs)
    assert gov.check(now, now - (5 if fstale else 0)) == action
    assert gov.events[-1][1] == reason
    assert inner.rcs[rcs:] == []  # no hover rc: landing wins


@pytest.mark.parametrize("h, ud, out", [(3.0, 100, -20), (2.5, 100, 0), (0.3, -100, 0), (0.5, -100, 0)],
                         ids=["ceiling_while_commanding_full_up", "at_ceiling_while_commanding_up",
                              "floor_while_commanding_full_down", "at_floor_while_commanding_down"])
def test_governor_altitude_limits_with_the_default_max_rc(clock, h, ud, out):
    gov, inner = governed(clock, SafetyCfg(max_altitude_m=2.5), h=h)
    gov.rc(0, 0, ud, 0)
    assert inner.rcs[-1][2] == out


# -------------------------------------------------- land re-send and latching
def test_governor_land_resend_every_two_seconds_only_while_flying_and_idle(clock):
    gov, inner = governed(clock, bat=10)
    t = clock.time
    assert gov.check(t(), t()) == "land" and inner.lands == 1
    sent_at = gov._land_sent
    inner._busy = False  # the land got an error / timed out: still flying
    clock.advance(2.0)
    inner.tel.t = t()
    assert gov.check(t(), t()) == "land" and inner.lands == 1  # exactly 2 s: not yet (strict >)
    clock.advance(0.001)
    assert gov.check(t(), t()) == "land" and inner.lands == 2 and gov._land_sent == clock.monotonic() != sent_at
    inner._busy = True  # landing now
    clock.advance(10)
    inner.tel.t = t()
    assert gov.check(t(), t()) == "land" and inner.lands == 2  # never re-sent over a running landing
    inner._busy, inner._flying = False, False  # it is down
    assert gov.check(t(), t()) == "ok" and inner.lands == 2


def test_governor_land_resend_uses_the_wall_clock_of_the_governor_not_the_callers_now(clock):
    gov, inner = governed(clock, bat=10)
    now = clock.time()
    gov.check(now, now)
    inner._busy = False
    assert gov.check(now + 100, now + 100) == "land"  # the caller's 'now' is far ahead...
    assert inner.lands == 1  # ...but the governor's own clock says < 2 s since the last land: no re-send


def test_explicit_land_latches_and_is_resent_while_it_still_flies(clock):
    gov, inner = governed(clock)
    gov.land()
    assert gov.land_requested and inner.lands == 1 and gov._land_sent == clock.monotonic()
    inner._busy = False
    clock.advance(1.0)
    inner.tel.t = clock.time()
    assert gov.check(clock.time(), clock.time()) == "land" and inner.lands == 1
    clock.advance(1.5)
    inner.tel.t = clock.time()
    assert gov.check(clock.time(), clock.time()) == "land" and inner.lands == 2
    assert gov.events[-1][1] == "LAND: requested"


def test_explicit_land_on_the_ground_is_still_passed_on(clock):
    gov, inner = governed(clock, flying=False)
    gov.land()
    assert inner.lands == 1 and gov.land_requested


def test_a_second_takeoff_after_a_land_is_refused(clock):
    # the land latch survives landing ("restart the recorder to fly again"): a new takeoff is refused outright
    gov, inner = governed(clock)
    gov.land()
    inner._flying, inner._busy = False, False
    now = clock.time()
    assert gov.check(now, now) == "ok" and gov.land_requested
    gov.takeoff()
    assert inner.named("takeoff") == []


# ====================================================================== SafetyGovernor: rc / move / rotate per state
GOV_RC_STATES = [
    # id, flying, land requested, hovering, forwarded
    ("ground", False, False, False, True),
    ("flying", True, False, False, True),
    ("hovering_for_safety", True, False, True, False),
    ("landing", True, True, False, False),
    ("landed", False, True, False, False),
]


@pytest.mark.parametrize("_id, flying, req, hover, forwarded", GOV_RC_STATES, ids=[s[0] for s in GOV_RC_STATES])
def test_governor_rc_per_state(clock, _id, flying, req, hover, forwarded):
    gov, inner = governed(clock, flying=flying)
    gov.land_requested, gov.hovering_for_safety = req, hover
    gov.rc(10, 20, 0, 30)
    assert inner.rcs == ([(10, 20, 0, 30)] if forwarded else [])


MOVES = [
    # id, direction, cm, h, tof, forwarded, refusal
    ("up_to_exactly_the_ceiling", "up", 50, 2.0, None, True, None),
    ("up_past_the_ceiling", "up", 51, 2.0, None, False, "up 51 would reach 2.51 m"),
    ("up_from_above_the_ceiling", "up", 20, 3.0, None, False, "up 20 would reach 3.20 m"),
    ("up_with_agreeing_tof", "up", 20, 2.2, 2.35, False, "up 20 would reach 2.55 m"),
    ("up_without_altitude", "up", 500, None, None, True, None),
    ("down_to_exactly_the_floor", "down", 50, 1.0, None, True, None),
    ("down_past_the_floor", "down", 51, 1.0, None, False, "down 51 would leave 0.49 m below"),
    ("down_over_a_table", "down", 20, 1.5, 0.6, False, "down 20 would leave 0.40 m below"),
    ("down_without_altitude", "down", 500, None, None, True, None),
    ("forward_above_the_ceiling", "forward", 100, 3.0, None, True, None),
    ("back_below_the_floor", "back", 100, 0.2, None, True, None),
    ("left", "left", 20, 1.5, None, True, None),
    ("right", "right", 500, 1.5, None, True, None),
]


@pytest.mark.parametrize("flying", [True, False], ids=["flying", "on_ground"])
@pytest.mark.parametrize("_id, direction, cm, h, tof, ok, why", MOVES, ids=[m[0] for m in MOVES])
def test_governor_move_limits(clock, flying, _id, direction, cm, h, tof, ok, why):
    gov, inner = governed(clock, SafetyCfg(max_altitude_m=HI, min_altitude_m=LO), h=h, tof=tof, flying=flying)
    inner.result = "inner-result"
    gov.move(direction, cm)
    if ok:
        assert inner.named("move") == [(direction, cm)] and gov.last_result() == "inner-result"
    else:
        assert inner.named("move") == [] and gov.last_result() == f"error: safety ({why})"
        assert gov.events[-1][1] == f"refused: {why}"


def test_governor_refusal_sticks_until_the_next_accepted_command(clock):
    gov, inner = governed(clock, SafetyCfg(max_altitude_m=HI), h=2.4)
    for accept in (lambda: gov.move("forward", 50), lambda: gov.rotate(30), gov.takeoff):
        gov.move("up", 50)
        assert gov.last_result().startswith("error: safety")
        assert gov.busy() is inner.busy()  # busy still reports the inner drone
        accept()
        assert gov.last_result() == "ok"


def test_governor_move_rounds_before_checking_and_forwards_the_rounded_cm(clock):
    gov, inner = governed(clock, SafetyCfg(max_altitude_m=HI), h=2.0)
    gov.move("up", 50.4)
    assert inner.named("move") == [("up", 50)]


@pytest.mark.parametrize("req", [False, True], ids=["normal", "landing"])
@pytest.mark.parametrize("direction, cm", [("up", 10), ("sideways", 50), ("forward", 501)])
def test_governor_invalid_move_raises_even_while_landing(clock, req, direction, cm):
    gov, inner = governed(clock)
    gov.land_requested = req
    with pytest.raises(ValueError):
        gov.move(direction, cm)
    assert inner.named("move") == []


@pytest.mark.parametrize("call, name", [(lambda g: g.move("forward", 100), "move"), (lambda g: g.rotate(90), "rotate")],
                         ids=["move", "rotate"])
def test_governor_refuses_discrete_commands_once_landing(clock, call, name):
    gov, inner = governed(clock)
    gov.land()
    call(gov)
    assert inner.named(name) == [] and gov.last_result() == "error: safety (landing)"


def test_governor_rotate_passes_through_and_invalid_angles_raise_from_the_inner_drone(clock):
    gov, inner = governed(clock)
    gov.rotate(-90)
    assert inner.named("rotate") == [(-90,)]
    with pytest.raises(ValueError):
        gov.rotate(0)


def test_governor_rotate_while_landing_refuses_even_an_invalid_angle_silently(clock):
    # pinned asymmetry: move() validates before the landing check, rotate() does not
    gov, inner = governed(clock)
    gov.land_requested = True
    gov.rotate(0)
    assert gov.last_result() == "error: safety (landing)"


@pytest.mark.parametrize("call, name", [(lambda g: g.move("forward", 100), "move"), (lambda g: g.rotate(90), "rotate")],
                         ids=["move", "rotate"])
def test_governor_refuses_discrete_commands_during_a_safety_hover(clock, call, name):
    # the stale-data hover blocks discrete moves and rotations too, not only rc
    gov, inner = governed(clock)
    now = clock.time()
    assert gov.check(now, None) == "hover"
    call(gov)
    assert inner.named(name) == []


# ====================================================================== SafetyGovernor: pass-throughs
@pytest.mark.parametrize("result", [True, False, None])
def test_governor_emergency_is_logged_and_returns_the_inner_answer(clock, result):
    gov, inner = governed(clock)
    inner.emergency_result = result
    assert gov.emergency() is result
    assert gov.events[-1][1] == "EMERGENCY" and inner.named("emergency") == [()]
    assert not gov.land_requested


def test_governor_repeated_emergency_is_passed_on_each_time_but_logged_once(clock):
    gov, inner = governed(clock)
    inner.emergency_result = False
    gov.emergency()
    gov.rc(0, 0, 0, 0)
    gov.emergency()
    assert [m for _, m in gov.events] == ["EMERGENCY"]  # consecutive duplicates collapse (rc logs nothing here)
    assert len(inner.named("emergency")) == 2


@pytest.mark.parametrize("call, name, args", [
    (lambda g: g.connect(), "connect", ()),
    (lambda g: g.close(), "close", ()),
    (lambda g: g.stop(), "stop", ()),
    (lambda g: g.set_speed(35), "set_speed", (35,)),
    (lambda g: g.takeoff(), "takeoff", ()),
], ids=["connect", "close", "stop", "set_speed", "takeoff"])
def test_governor_passes_calls_through(clock, call, name, args):
    gov, inner = governed(clock)
    call(gov)
    assert inner.named(name) == [args]


def test_governor_stop_passes_through_even_while_landing(clock):
    gov, inner = governed(clock)
    gov.land()
    gov.stop()
    assert inner.named("stop") == [()]  # TelloDrone.stop() itself never interrupts a landing


def test_governor_reads_flying_busy_telemetry_and_video_from_the_inner_drone(clock):
    gov, inner = governed(clock, flying=False, busy=True)
    assert gov.flying is False and gov.busy() is True and gov.telemetry() is inner.tel
    assert gov.frame_source() is inner.source and gov.name == "safety"
    inner._flying = True
    assert gov.flying is True


# ====================================================================== governor around a real TelloDrone
def test_governed_tello_low_battery_mid_move_stops_then_lands(clock):
    d, fake = make(clock)
    fly(d, fake, clock)
    gov = SafetyGovernor(d, SafetyCfg(), clock=clock.monotonic)
    d.move("forward", 200)
    fake.state = dict(fake.state, bat=12)
    now = clock.time()
    assert gov.check(now, now) == "land"
    assert fake.cmds[-2:] == ["stop", "land"] and d._pending.cmd == "land"
    fake.state = dict(fake.state, h=0, tof=10)  # touched down
    fake.reply("ok")
    assert not gov.busy() and not gov.flying


def test_governed_tello_safety_hover_sends_rc_zero_on_the_wire(clock):
    d, fake = make(clock)
    fly(d, fake, clock)
    gov = SafetyGovernor(d, SafetyCfg(), clock=clock.monotonic)
    now = clock.time()
    assert gov.check(now, now - 5) == "hover" and fake.rcs[-1] == (0, 0, 0, 0)


def test_governed_dry_run_hovers_and_lands_without_sending_anything(clock):
    d, fake = make(clock, dry_run=True)
    fly(d, fake, clock)
    gov = SafetyGovernor(d, SafetyCfg(), clock=clock.monotonic)
    fake.state = dict(fake.state)
    now = clock.time()
    assert gov.check(now, now - 5) == "hover"
    fake.state = dict(fake.state, bat=5)
    assert gov.check(now, now) == "land"
    clock.advance(4.0)
    assert not gov.busy() and not gov.flying and fake.motion_sent() == []


# ====================================================================== invariants (seeded random call sequences)
def _random_sticks(rng):
    return [rng.choice([0, rng.uniform(-300, 300), rng.randint(-100, 100), 100, -100]) for _ in range(4)]


def _random_op(rng, d, gov, fake, clock, live):
    """One random call on the governed stack (or the bare adapter). Returns nothing; may raise the two
    programming errors the API documents: RuntimeError (discrete command while busy) or ValueError (bad args)."""
    target = gov if d is None or rng.random() < 0.6 else d
    op = rng.choice(["takeoff", "land", "rc", "rc", "rc", "move", "rotate", "stop", "speed", "busy", "busy",
                     "emergency", "check", "check", "tick", "tick", "telemetry", "state", "reply", "wifi"])
    if op == "takeoff":
        target.takeoff()
    elif op == "land":
        target.land()
    elif op == "rc":
        target.rc(*_random_sticks(rng))
    elif op == "move":
        target.move(rng.choice(DIRECTIONS + ("sideways",)), rng.choice([20, 100, 500, 10, 600, rng.uniform(15, 510)]))
    elif op == "rotate":
        target.rotate(rng.choice([1, -90, 360, 0, 400, rng.uniform(-370, 370)]))
    elif op == "stop":
        target.stop()
    elif op == "speed":
        target.set_speed(rng.randint(10, 100))
    elif op == "busy":
        target.busy()
        target.last_result()
    elif op == "emergency":
        if rng.random() < 0.2:
            target.emergency()
    elif op == "check":
        now = clock.time()
        gov.check(now, rng.choice([now, now - 0.2, now - 3, None]))
    elif op == "tick":
        clock.advance(rng.choice([0.01, 0.05, 0.1, 0.5, 2.0, 7.0, 25.0]))
    elif op == "telemetry":
        target.telemetry()
    elif op == "state":
        fake.state = dict(fake.state, h=rng.choice([0, 10, 50, 150, 230, 260, None]),
                          tof=rng.choice([10, 60, 150, 6553, None]), bat=rng.choice([90, 50, 21, 19, 0, None]))
    elif op == "reply":
        if live:
            fake.reply(rng.choice(["ok", "ok", "error", "error Not joystick", "87", "out of range", b"\xff"]))
    elif op == "wifi":
        if live:
            fake.down = rng.random() < 0.3


@pytest.fixture
def patched_time(monkeypatch):
    """Lets one test install fresh FakeClocks repeatedly."""

    def install(c):
        monkeypatch.setattr(tello_mod, "time", c)
        monkeypatch.setattr(safety_mod, "time", c)

    return install


def _stack(patched_time, dry_run, seed, max_rc=50):
    clock = FakeClock()
    patched_time(clock)
    fake = FakeDjiTello(clock)
    d = TelloDrone(tello=fake, video=False, dry_run=dry_run)
    d.connect()
    gov = SafetyGovernor(d, SafetyCfg(max_rc=max_rc), clock=clock.monotonic)
    return random.Random(seed), clock, fake, d, gov


ALLOWED = (RuntimeError, ValueError)


def _step(rng, d, gov, fake, clock, live):
    try:
        _random_op(rng, d, gov, fake, clock, live)
    except ALLOWED as e:
        if isinstance(e, RuntimeError):
            assert "still running" in str(e)


@pytest.mark.parametrize("seed", range(40))
def test_invariant_dry_run_never_sends_a_motion_command(patched_time, seed):
    rng, clock, fake, d, gov = _stack(patched_time, True, seed)
    for _ in range(150):
        _step(rng, d, gov, fake, clock, live=False)
        assert fake.motion_sent() == [] and fake.is_flying is False
    gov.close()
    assert fake.motion_sent() == [] and fake.ends == 1


@pytest.mark.parametrize("seed", range(40))
def test_invariant_governor_never_lets_a_stick_exceed_max_rc(patched_time, seed):
    rng0 = random.Random(seed)
    max_rc = rng0.choice([20, 25, 30, 50, 70, 100])
    rng, clock, fake, d, gov = _stack(patched_time, False, seed, max_rc=max_rc)
    for _ in range(200):
        if rng.random() < 0.5:
            gov.rc(*_random_sticks(rng))
        else:
            _step(rng, None, gov, fake, clock, live=True)  # the mission only ever talks to the governor
    assert all(abs(v) <= max_rc for r in fake.rcs for v in r), max_rc


@pytest.mark.parametrize("seed", range(40))
def test_invariant_governor_output_within_max_rc_for_any_telemetry(clock, seed):
    rng = random.Random(seed)
    max_rc = rng.choice([20, 30, 50, 80, 100])
    gov, inner = governed(clock, SafetyCfg(max_rc=max_rc, min_altitude_m=rng.uniform(0.2, 1.0),
                                           max_altitude_m=rng.uniform(1.5, 3.0)))
    for _ in range(200):
        inner._flying = rng.random() < 0.8
        inner.tel.height_m = rng.choice([None, rng.uniform(-1, 5)])
        inner.tel.tof_m = rng.choice([None, rng.uniform(0, 8)])
        gov.rc(*_random_sticks(rng))
    assert inner.rcs and all(abs(v) <= max_rc for r in inner.rcs for v in r)


@pytest.mark.parametrize("dry_run", [False, True], ids=["live", "dry_run"])
@pytest.mark.parametrize("seed", range(25))
def test_invariant_after_land_requested_only_zero_sticks_reach_the_drone(patched_time, dry_run, seed):
    rng, clock, fake, d, gov = _stack(patched_time, dry_run, seed)
    for _ in range(rng.randint(0, 40)):
        _step(rng, d, gov, fake, clock, live=not dry_run)
    if rng.random() < 0.5:
        gov.land()
    else:
        fake.state = dict(fake.state, bat=5)
        d._flying = True
        now = clock.time()
        gov.check(now, now)
    assert gov.land_requested
    n_rc, n_log = len(fake.rcs), len(d.log)
    for _ in range(120):
        op = rng.random()
        if op < 0.5:
            gov.rc(*_random_sticks(rng))
        elif op < 0.7:
            now = clock.time()
            gov.check(now, rng.choice([now, None, now - 5]))
        elif op < 0.8:
            clock.advance(rng.choice([0.1, 1.0, 3.0]))
        elif op < 0.9:
            if not dry_run:
                fake.reply(rng.choice(["ok", "error"]))
            d.busy()
        else:
            try:
                rng.choice([lambda: gov.move("up", 50), lambda: gov.rotate(90), gov.stop])()
            except ALLOWED:
                pass
    assert all(r == (0, 0, 0, 0) for r in fake.rcs[n_rc:])
    assert not any(c.startswith("rc ") and c != "rc 0 0 0 0" for _, c in d.log[n_log:])


@pytest.mark.parametrize("seed", range(40))
def test_invariant_no_flight_call_raises_on_network_errors(patched_time, seed):
    rng, clock, fake, d, gov = _stack(patched_time, False, seed)
    for _ in range(150):
        fake.down = rng.random() < 0.5  # the link flaps between every call
        _step(rng, d, gov, fake, clock, live=True)  # anything but RuntimeError/ValueError fails the test
    fake.down = True
    d._flying = True
    gov.close()  # close() over a dead link: no raise, still ends djitellopy
    assert fake.ends == 1


@pytest.mark.parametrize("seed", range(30))
def test_invariant_commands_are_spaced_by_the_min_gap_and_rc_never_overlaps_a_discrete_command(patched_time, seed):
    rng, clock, fake, d, gov = _stack(patched_time, False, seed)
    for _ in range(150):
        pending_before = d._pending is not None
        n = len(fake.rcs)
        _step(rng, d, gov, fake, clock, live=True)
        if pending_before and d._pending is not None:
            assert len(fake.rcs) == n  # an rc packet would abort the running move on the drone
    times = [(t, c) for t, c in fake.cmd_times]
    for (t1, c1), (t2, c2) in zip(times, times[1:]):
        if "emergency" in (c1, c2):
            continue  # emergency goes out at once, by design
        assert t2 - t1 >= MIN_GAP_S - 1e-9, (c1, c2, t2 - t1)


@pytest.mark.parametrize("seed", range(20))
def test_invariant_busy_and_last_result_agree(patched_time, seed):
    rng, clock, fake, d, gov = _stack(patched_time, seed % 2 == 0, seed)
    for _ in range(150):
        _step(rng, d, gov, fake, clock, live=seed % 2 == 1)
        busy = d.busy()
        assert busy is (d._pending is not None)
        assert (d.last_result() is None) is busy
        r = d.last_result()
        assert r is None or r == "ok" or r.startswith("error")
