"""End to end against the protocol-level Tello emulator: kite's TelloDrone / SafetyGovernor / recorder / preflight ->
real djitellopy -> real UDP on loopback -> kite.testing.TelloEmulator (Tello SDK 2.0 text protocol, state stream,
H.264 video for the slow tests).

The emulator's numbers and sign conventions are ASSUMED where the real drone is unmeasured (see the docstring of
kite/testing/tello_emulator.py): these tests pin kite's own conventions and logic against the protocol, they do not
prove the real drone's. Non-video tests run the flight model 2-5x faster than real time; kite's timeouts stay real.
"""

from __future__ import annotations

import csv
import json
import math
import socket
import sys
import time

import cv2
import pytest

from kite.config import Config, DroneCfg
from kite.drone import make_tello
from kite.drone import tello as tello_mod
from kite.drone.tello import TelloDrone
from kite.record import Recorder, RecordCfg
from kite.testing import EmulatorCamera, TelloEmulator, red_pole_column
from kite.testing import tello_emulator as emu_mod

needs_ffmpeg = pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="needs ffmpeg")


def heading_change(a: float, b: float) -> float:
    return (b - a + 180.0) % 360.0 - 180.0


# ================================================================== protocol basics
def test_state_line_is_the_format_djitellopy_parses():
    from djitellopy import Tello

    for pads in (False, True):
        emu = TelloEmulator(mission_pad_fields=pads)
        try:
            emu.place(h=1.37, heading=-42.0)
            line = emu.state_line()
            assert line.endswith(";\r\n")
            st = Tello.parse_state(line)
            for k in ("pitch", "roll", "yaw", "vgx", "vgy", "vgz", "templ", "temph", "tof", "h", "bat", "time"):
                assert isinstance(st[k], int), k
            for k in ("baro", "agx", "agy", "agz"):
                assert isinstance(st[k], float), k
            assert st["h"] == 137 and st["tof"] == 137 and st["yaw"] == -42 and st["bat"] == 87
            assert ("mid" in st) == pads
        finally:
            emu.stop()


def test_connect_handshake(tello_emulator):
    rig = tello_emulator
    emu, link = rig.emu, rig.link
    assert emu.commands(include_rc=False)[:2] == ["command", "speed 50"]  # SDK mode, then kite's move speed
    first = emu.received[0]
    assert first[3] == ("127.0.0.1", link.client_port)  # sent from djitellopy's client socket, not 8889
    s = emu.snapshot()
    assert s.sdk_mode and s.speed_cm_s == 50 and not s.stream_on
    tel = rig.fresh_telemetry()
    assert abs(tel.battery_pct - emu.snapshot().battery) < 1.5  # whole percent in the state line
    assert tel.height_m == 0.0 and not rig.drone.flying and rig.drone.link_error is None
    assert time.time() - tel.t < 0.5
    assert emu.state_packets >= 1


def test_state_streams_at_about_10_hz(tello_emulator):
    rig = tello_emulator
    seen, last = 0, rig.drone.telemetry().t
    t_end = time.monotonic() + 1.0
    while time.monotonic() < t_end:
        t = rig.drone.telemetry().t
        if t != last:
            seen, last = seen + 1, t
        time.sleep(0.002)
    assert 7 <= seen <= 13


@pytest.mark.emulator(time_scale=2.0)
def test_telemetry_is_converted_to_si_units_with_kites_signs(tello_emulator):
    rig = tello_emulator
    rig.emu.place(h=1.23, heading=-30.0)
    tel = rig.fresh_telemetry()
    assert tel.height_m == pytest.approx(1.23) and tel.tof_m == pytest.approx(1.23)
    assert tel.yaw_deg == -30.0  # yaw_sign +1 (DroneCfg default)
    raw = rig.drone.tello.get_current_state()
    assert tel.battery_pct == raw["bat"] and tel.flight_time_s == raw["time"]
    rig.emu.place(h=0.0)
    rig.takeoff()
    rig.fly(1.0, fb=50)  # 0.5 m/s at stick 50 after the lag
    tel = rig.drone.telemetry()
    raw = rig.drone.tello.get_current_state()
    assert tel.vx == pytest.approx(raw["vgx"] * 0.1)  # dm/s -> m/s
    assert abs(tel.vx) == pytest.approx(rig.emu.snapshot().v_fwd, abs=0.11)
    assert tel.vx < 0  # the emulator follows vigil's log: forward reads as NEGATIVE vgx
    assert tel.pitch_deg < 0  # nose down while flying forward (reported pitch + = nose up)


def test_tof_out_of_range_and_on_the_ground_readings(tello_emulator):
    rig = tello_emulator
    tel = rig.fresh_telemetry()
    assert tel.tof_m == pytest.approx(0.1)  # 10 cm on the ground: kite keeps it (0.1..8 m)
    rig.emu.place(h=9.0)
    tel = rig.fresh_telemetry()
    assert rig.drone.tello.get_current_state()["tof"] == 6553 and tel.tof_m is None and tel.height_m == 9.0
    assert tel.altitude_m() == 9.0  # falls back to h


# ================================================================== takeoff / land
def test_takeoff_and_land_state_machine(tello_emulator):
    rig, emu, d = tello_emulator, tello_emulator.emu, tello_emulator.drone
    assert not d.flying and not d.busy()
    d.takeoff()
    assert d.busy() and d.last_result() is None
    emu.wait_until(lambda s: s.phase == "takeoff", 1.0)
    d.rc(0, 40, 0, 0)  # ignored while the takeoff runs (an rc would abort a discrete command)
    assert rig.wait() == "ok" and d.flying and d.tello.is_flying
    s = emu.snapshot()
    assert s.phase == "air" and s.h == pytest.approx(0.8, abs=0.02) and s.motors
    assert emu.rc_received() == []
    d.takeoff()
    assert rig.wait() == "error: error" and d.flying  # already flying
    d.land()
    assert rig.wait() == "ok" and not d.flying and not d.tello.is_flying
    s = emu.snapshot()
    assert s.phase == "ground" and not s.motors and s.h == 0.0
    d.land()  # on the ground: the drone answers error, h <= 0.2 m -> kite counts it as landed
    assert rig.wait() == "ok" and not d.flying


def test_land_during_takeoff_lands(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.gov.takeoff()
    emu.wait_until(lambda s: s.h > 0.2, 3.0)
    rig.gov.land()
    assert rig.wait() == "ok" and not rig.gov.flying
    assert emu.snapshot().phase == "ground"
    assert emu.commands(include_rc=False)[-2:] == ["takeoff", "land"]


def test_close_while_flying_lands_it(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    rig.drone.close()
    emu.wait_until(lambda s: s.phase == "ground", 5.0)
    assert "land" in emu.commands()


# ================================================================== sticks: kite's sign conventions vs the protocol
@pytest.mark.emulator(time_scale=5.0)
@pytest.mark.parametrize("start_heading", [0.0, 90.0])
def test_each_rc_axis_moves_the_drone_the_expected_way(tello_emulator, start_heading):
    """+lr = right, +fb = forward, +ud = up, +yaw = clockwise (Drone.rc docstring), through the governor."""
    rig, emu = tello_emulator, tello_emulator.emu
    emu.place(heading=start_heading)
    rig.takeoff()
    rig.fly(0.25, ud=50)  # to ~1.4 m: clear of the governor's floor and ceiling bands
    axes = [("fb", 1), ("fb", -1), ("lr", 1), ("lr", -1), ("ud", 1), ("ud", -1), ("yaw", 1), ("yaw", -1)]
    for axis, sign in axes if start_heading == 0.0 else axes[:4]:  # heading only matters for fb / lr
        rig.fly(0.25)  # settle
        s0, tel0 = emu.snapshot(), rig.drone.telemetry()
        rig.fly(0.3, **{axis: 30 * sign})
        rig.fly(0.25)
        s1, tel1 = emu.snapshot(), rig.drone.telemetry()
        psi = math.radians(s0.heading)
        dn, de = s1.north - s0.north, s1.east - s0.east
        fwd = dn * math.cos(psi) + de * math.sin(psi)
        right = -dn * math.sin(psi) + de * math.cos(psi)
        what = f"{axis} {30 * sign:+d}"
        if axis == "fb":
            assert sign * fwd > 0.2 and abs(right) < 0.05, what
        elif axis == "lr":
            assert sign * right > 0.2 and abs(fwd) < 0.05, what
        elif axis == "ud":
            assert sign * (s1.h - s0.h) > 0.15 and sign * (tel1.height_m - tel0.height_m) > 0.1, what
        else:
            assert sign * heading_change(s0.heading, s1.heading) > 20, what
            assert sign * heading_change(tel0.yaw_deg, tel1.yaw_deg) > 15, what  # kite's yaw_deg: + = cw
    assert max(max(abs(v) for v in rc) for rc in emu.rc_received()) <= 50  # governor clamp


@pytest.mark.parametrize("drone_sign,kite_sign", [(1, 1), (-1, -1), (-1, 1)])
def test_yaw_sign_config_makes_telemetry_yaw_clockwise_positive(tello_emulator_factory, drone_sign, kite_sign):
    with tello_emulator_factory(physics={"yaw_sign": drone_sign}, drone={"yaw_sign": kite_sign}) as rig:
        rig.takeoff()
        y0 = rig.fresh_telemetry().yaw_deg
        rig.gov.rotate(45)
        assert rig.wait() == "ok"
        y1 = rig.fresh_telemetry().yaw_deg
        change = heading_change(y0, y1)
        if drone_sign == kite_sign:
            assert change == pytest.approx(45, abs=2)
        else:  # DroneCfg.yaw_sign wrong for this drone: kite would read cw as a left turn
            assert change == pytest.approx(-45, abs=2)


def test_discrete_moves_and_rotations_fly_their_distance(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    plan = [("move", "forward", 100), ("move", "right", 50), ("move", "up", 50), ("rotate", None, -90),
            ("move", "forward", 60), ("move", "down", 40)]
    for kind, direction, n in plan:
        s0 = emu.snapshot()
        (rig.gov.move(direction, n) if kind == "move" else rig.gov.rotate(n))
        assert rig.wait() == "ok", (direction, n)
        s1 = emu.snapshot()
        psi = math.radians(s0.heading)
        dn, de = s1.north - s0.north, s1.east - s0.east
        fwd = dn * math.cos(psi) + de * math.sin(psi)
        right = -dn * math.sin(psi) + de * math.cos(psi)
        if direction == "forward":
            assert fwd == pytest.approx(n / 100, abs=0.02)
        elif direction == "right":
            assert right == pytest.approx(n / 100, abs=0.02)
        elif direction in ("up", "down"):
            assert s1.h - s0.h == pytest.approx((n if direction == "up" else -n) / 100, abs=0.02)
        else:
            assert heading_change(s0.heading, s1.heading) == pytest.approx(n, abs=1)
    assert emu.commands(include_rc=False)[-6:] == ["forward 100", "right 50", "up 50", "ccw 90", "forward 60",
                                                   "down 40"]


def test_rc_is_held_back_while_a_discrete_move_runs(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    rig.gov.move("forward", 100)
    while rig.gov.busy():
        rig.gov.rc(0, 0, 30, 0)  # the recorder's keep-alive would cancel the move on a real Tello
        time.sleep(0.05)
    assert rig.gov.last_result() == "ok"
    assert emu.rc_received() == []
    assert emu.snapshot().north == pytest.approx(1.0, abs=0.02)


# ================================================================== keep-alive and auto-land
@pytest.mark.emulator(time_scale=8.0)
def test_rc_keep_alive_prevents_auto_land_and_silence_triggers_it(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    rig.fly(emu.real_s(22.0))  # rc 0 0 0 0 at 10 Hz for 22 simulated s: well past 15 s
    assert emu.snapshot().phase == "air"
    s = emu.wait_until(lambda s: s.phase == "ground", emu.real_s(15.0 + 5.0))  # the laptop stops talking
    assert any(msg.startswith("auto-land") for _, msg in emu.events)
    auto = next(t for t, msg in emu.events if msg.startswith("auto-land"))
    last_heard = max(t_sim for _, t_sim, _, _, verdict in emu.received if verdict == "ok")
    assert auto - last_heard == pytest.approx(15.0, abs=0.3)
    assert s.h == 0.0
    # kite cannot see an auto-land: it still says flying until it asks to land; the drone answers 'error' and
    # TelloDrone's "h <= 0.2 m means landed" rule settles it
    assert rig.drone.flying
    assert rig.land() == "ok" and not rig.gov.flying


# ================================================================== safety governor against the emulator
def test_governor_clamps_every_stick(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    rig.fly(0.3, lr=100, fb=-100, ud=0, yaw=-100)
    assert emu.rc_received()[-1] == (50, -50, 0, -50)


@pytest.mark.emulator(time_scale=2.0)
def test_governor_ceiling_holds_a_climbing_drone_under_2_3_m(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    peak = [0.0]
    rig.fly(3.5, ud=50, on_tick=lambda: peak.__setitem__(0, max(peak[0], emu.snapshot().h)))
    h = emu.snapshot().h
    assert 1.95 <= h <= 2.4  # held around the ceiling (pushed down at stick -20 when above it)
    assert peak[0] <= 2.3 + 0.1, f"overshot the ceiling: {peak[0]:.2f} m"


@pytest.mark.emulator(time_scale=2.0)
def test_governor_floor_stops_a_descending_drone_at_0_5_m(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    rig.gov.move("up", 50)
    assert rig.wait() == "ok"
    low = [9.0]
    rig.fly(3.0, ud=-50, on_tick=lambda: low.__setitem__(0, min(low[0], emu.snapshot().h)))
    # a soft floor: below it descent is cut (never pushed back up), so the drone coasts ~speed x lag under it
    # (0.43 m in real time with the assumed 0.4 s lag, ~0.38 m at 2x where kite's 10 Hz loop is relatively slower)
    assert low[0] >= 0.5 - 0.15, f"went below the floor: {low[0]:.2f} m"
    assert emu.snapshot().h == pytest.approx(low[0], abs=0.01) and low[0] < 0.6  # and stays there


@pytest.mark.emulator(time_scale=2.0, terrain=lambda n, e: 0.6)
def test_governor_floor_uses_the_tof_over_an_obstacle(tello_emulator):
    """Over a 0.6 m table the downward ToF reads the clearance, so descent stops ~0.5 m above the TABLE."""
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    rig.gov.move("up", 60)
    assert rig.wait() == "ok"
    rig.fly(3.0, ud=-50)
    s = emu.snapshot()
    assert s.h >= 0.6 + 0.5 - 0.15
    tel = rig.drone.telemetry()
    assert tel.tof_m == pytest.approx(s.h - 0.6, abs=0.02) and tel.height_m == pytest.approx(s.h, abs=0.02)


def test_governor_refuses_a_discrete_climb_through_the_ceiling(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    rig.fresh_telemetry()
    rig.gov.move("up", 200)  # 0.8 + 2.0 > 2.3
    assert rig.gov.last_result().startswith("error: safety")
    assert "up 200" not in emu.commands()
    rig.gov.move("down", 40)  # 0.8 - 0.4 < 0.5
    assert rig.gov.last_result().startswith("error: safety") and "down 40" not in emu.commands()


def test_governor_lands_on_low_battery(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    emu.set_battery(19.4)
    actions = []
    t_end = time.monotonic() + 8
    while rig.gov.flying and time.monotonic() < t_end:
        actions += rig.fly(0.1)
        rig.gov.busy()
    assert "land" in actions and not rig.gov.flying
    assert emu.snapshot().phase == "ground" and "land" in emu.commands()
    assert any("battery 19%" in msg for _, msg in rig.gov.events)


@pytest.mark.emulator(safety={"max_flight_s": 1.0})
def test_governor_lands_after_the_flight_time_limit(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    t0 = time.monotonic()
    actions = []
    while rig.gov.flying and time.monotonic() - t0 < 6:
        actions += rig.fly(0.1)
    assert "land" in actions and not rig.gov.flying and emu.snapshot().phase == "ground"
    assert 1.0 <= time.monotonic() - t0 < 6


def test_takeoff_refused_by_the_drone_on_a_flat_battery_or_when_overheated(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    emu.set_battery(8)
    rig.gov.takeoff()
    assert rig.wait() == "error: error" and not rig.gov.flying and emu.snapshot().phase == "ground"
    emu.set_battery(80)
    emu.set_temperature(87, 91)
    rig.gov.takeoff()
    assert rig.wait() == "error: error" and not rig.gov.flying
    emu.set_temperature(60, 63)
    rig.takeoff()


def test_the_drone_lands_itself_at_critical_battery(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    emu.set_battery(4.9)
    rig.fly(0.5, governor=False)  # raw TelloDrone: the governor would have landed at 20 % long before
    s = emu.wait_until(lambda s: s.phase == "ground", 4.0)
    assert any("battery critical" in msg for _, msg in emu.events) and s.h == 0.0


# ================================================================== stale data -> hover
def test_frozen_state_stream_makes_the_governor_hover(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    rig.fly(0.5, fb=40)
    assert emu.snapshot().v_fwd > 0.2
    emu.pause_state()
    actions = rig.fly(1.6, fb=40)
    assert actions[0] == "ok" and actions[-1] == "hover"
    assert emu.rc_received()[-1] == (0, 0, 0, 0)  # the governor zeroes the sticks...
    assert emu.snapshot().v_fwd == pytest.approx(0.0, abs=0.05)  # ...and the drone stops
    assert any("telemetry stale" in msg for _, msg in rig.gov.events)
    emu.pause_state(False)
    rig.fresh_telemetry()
    actions = rig.fly(0.5, fb=40)
    assert actions[-1] == "ok" and emu.rc_received()[-1] == (0, 40, 0, 0)  # control resumes


def test_stale_video_makes_the_governor_hover(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    actions = rig.fly(0.4, fb=40, frame_age_s=1.5)
    assert set(actions) == {"hover"} and all(rc == (0, 0, 0, 0) for rc in emu.rc_received())


def test_one_garbage_state_packet_kills_djitellopys_state_receiver(tello_emulator):
    """djitellopy decodes state packets as ASCII and its receiver thread exits on the first error: telemetry freezes
    for the rest of the process. kite sees it as stale telemetry and hovers (the pilot must land and restart)."""
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    assert rig.fly(0.3)[-1] == "ok"
    emu.garbage("state", n=1, replace=True)
    time.sleep(0.3)
    assert emu_mod._state_thread is not None and not emu_mod._state_thread.is_alive()
    actions = rig.fly(1.5)
    assert actions[-1] == "hover"
    assert rig.land() == "ok" and emu.snapshot().phase == "ground"  # commands still work


# ================================================================== lossy link
def test_state_packet_loss_does_not_trigger_a_hover(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    emu.set_drop("state", 0.3)
    actions = rig.fly(3.0, fb=30)
    assert set(actions) == {"ok"}


@pytest.mark.emulator(time_scale=2.0)
def test_rc_packet_loss_still_flies_the_commanded_way(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    emu.set_drop("cmd", 0.3)
    s0 = emu.snapshot()
    rig.fly(2.0, fb=40, governor=False)
    s1 = emu.snapshot()
    lost = sum(1 for r in emu.received if r[2].startswith("rc ") and r[4] == "lost")
    assert lost > 0
    assert s1.north - s0.north > 0.6 * 0.4 * emu.time_scale * 2.0 * 0.8


def test_a_lost_reply_times_out_and_the_next_command_works(tello_emulator):
    rig, emu, d = tello_emulator, tello_emulator.emu, tello_emulator.drone
    rig.takeoff()
    emu.silence_replies()
    d.rotate(20)
    d._pending.timeout_s = 1.0  # kite's real timeout is 5 s + the expected duration; shortened to keep this fast
    assert rig.wait() == "error: timeout"
    assert emu.wait_until(lambda s: s.maneuver is None, 3.0)  # the drone did turn, only the reply was lost
    emu.silence_replies(False)
    d.rotate(-20)
    assert rig.wait() == "ok"


def test_garbage_replies_are_ignored_and_a_replaced_reply_times_out(tello_emulator):
    rig, emu, d = tello_emulator, tello_emulator.emu, tello_emulator.drone
    rig.takeoff()
    emu.garbage("cmd", n=1)  # a garbage datagram arrives before the real 'ok'
    d.rotate(15)
    assert rig.wait() == "ok"
    emu.garbage("cmd", n=1, replace=True)  # the 'ok' itself arrives corrupted
    d.rotate(15)
    d._pending.timeout_s = 1.0
    assert rig.wait() == "error: timeout"


# ================================================================== Wi-Fi drop
def test_wifi_drop_never_raises_hovers_then_control_resumes(tello_emulator):
    rig, emu, d = tello_emulator, tello_emulator.emu, tello_emulator.drone
    rig.takeoff()
    rig.fly(0.5, fb=30)
    n_before = len(emu.received)
    north_at_drop = emu.snapshot().north
    rig.link.wifi_down()
    actions = rig.fly(1.5, fb=30)  # sendto raises OSError inside djitellopy: never out of kite
    assert d.link_error and "unreachable" in d.link_error.lower()
    assert actions[-1] == "hover"  # no state packets -> stale telemetry
    assert len(emu.received) == n_before  # nothing reached the drone...
    assert emu.snapshot().north - north_at_drop > 0.5  # ...which kept its last stick (fb 30) all along
    rig.link.wifi_up()
    rig.fresh_telemetry()
    h0 = emu.snapshot().h
    actions = rig.fly(0.4, ud=40)
    assert actions[-1] == "ok" and d.link_error is None
    assert emu.rc_received()[-1] == (0, 0, 40, 0) and emu.snapshot().h > h0 + 0.2


def test_a_land_during_a_wifi_drop_is_resent_when_it_returns(tello_emulator):
    rig, emu, gov = tello_emulator, tello_emulator.emu, tello_emulator.gov
    rig.takeoff()
    rig.link.wifi_down()
    gov.land()
    assert not gov.busy() and gov.last_result().startswith("error: not sent") and gov.flying
    rig.fly(0.5)
    rig.link.wifi_up()
    t_end = time.monotonic() + 6
    while gov.flying and time.monotonic() < t_end:
        rig.fly(0.1)
    assert not gov.flying and emu.snapshot().phase == "ground"


@pytest.mark.emulator(time_scale=5.0)
def test_a_long_wifi_drop_ends_in_the_drones_own_auto_land(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    rig.link.wifi_down()
    rig.fly(emu.real_s(19.0))
    assert any(msg.startswith("auto-land") for _, msg in emu.events)
    rig.link.wifi_up()
    tel = rig.fresh_telemetry()
    assert tel.height_m == 0.0 and emu.snapshot().phase == "ground"
    assert rig.land() == "ok" and not rig.gov.flying


# ================================================================== emergency, errors, late replies, controllers
def test_emergency_cuts_the_motors_and_it_falls(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    rig.takeoff()
    rig.gov.move("up", 70)
    assert rig.wait() == "ok"
    assert rig.gov.emergency() is True and not rig.gov.flying
    emu.wait_until(lambda s: not s.motors, 0.5)
    s = emu.wait_until(lambda s: s.phase == "ground", 1.0)  # 1.5 m free fall ~0.55 s simulated
    assert s.h == 0.0 and any("hit the ground" in m for _, m in emu.events)
    tel = rig.fresh_telemetry()
    assert tel.height_m == 0.0
    n = len(emu.commands())
    rig.drone.close()  # must not send 'land' to a drone whose motors are off (djitellopy's end() would)
    time.sleep(0.3)
    assert "land" not in emu.commands()[n:]


def test_error_replies_reach_last_result(tello_emulator):
    rig, emu, d = tello_emulator, tello_emulator.emu, tello_emulator.drone
    d.move("forward", 50)  # motors off
    assert rig.wait() == "error: error Motor stop"
    d.set_speed(5)
    assert rig.wait() == "error: out of range"
    d.set_speed(40)
    assert rig.wait() == "ok" and emu.snapshot().speed_cm_s == 40
    assert d.tello.send_command_with_return("mon") == "unknown command: mon"  # EDU only
    assert d.tello.query_battery() == int(emu.snapshot().battery)


@pytest.mark.emulator(time_scale=2.0, cancelled_reply="ok", cancelled_reply_s=0.6)
def test_late_replies_after_stop_are_not_credited_to_the_next_command(tello_emulator):
    rig, emu, d = tello_emulator, tello_emulator.emu, tello_emulator.drone
    rig.takeoff()
    d.move("forward", 300)
    time.sleep(0.4)
    d.stop()
    assert emu.wait_until(lambda s: s.maneuver is None, 1.0)
    d.rotate(90)  # waits out kite's 1 s settle window first
    rig.wait()
    s = emu.snapshot()
    assert d.last_result() == "ok" and s.maneuver is None  # the rotation had really finished
    sent = [(t, text) for t, text, how in emu.replies if how == "sent"]
    assert [text for _, text in sent].count("ok") >= 4  # takeoff, stop, the cancelled move, the rotation
    cw = next(t for t, _, text, _, _ in emu.received if text == "cw 90")
    late = [t for t, text in sent if text == "ok" and t < cw]
    assert len(late) >= 3  # the stop's and the cancelled move's 'ok' had arrived (and were cleared) before it


@pytest.mark.xfail(strict=True, reason="KNOWN LIMIT: Tello replies carry no id; a cancelled move's reply arriving more "
                   "than STOP_SETTLE_S (1 s) after `stop` is credited to the next command, which then 'completes' "
                   "while the drone is still flying it. Real late-reply timing is unmeasured.")
@pytest.mark.emulator(time_scale=2.0, cancelled_reply="ok", cancelled_reply_s=1.6)
def test_a_reply_later_than_the_settle_window_is_not_credited_either(tello_emulator):
    rig, emu, d = tello_emulator, tello_emulator.emu, tello_emulator.drone
    rig.takeoff()
    d.move("forward", 300)
    time.sleep(0.4)
    d.stop()
    d.rotate(90)
    rig.wait()
    assert emu.snapshot().maneuver is None, "kite took the late 'ok' for the rotation"


@pytest.mark.emulator(min_command_gap_s=0.09)
def test_kite_spaces_discrete_commands_for_the_100_ms_rule(tello_emulator):
    rig, emu, d = tello_emulator, tello_emulator.emu, tello_emulator.drone
    rig.takeoff()
    d.move("forward", 100)
    d.stop()
    d.rotate(30)
    assert rig.wait() == "ok"
    d.move("back", 50)
    assert rig.wait() == "ok"
    dropped = [r[2] for r in emu.received if r[4].startswith("dropped")]
    assert dropped == []


def test_only_one_controller_at_a_time(tello_emulator):
    rig, emu = tello_emulator, tello_emulator.emu
    other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # a second program (or the phone app)
    other.settimeout(0.5)
    try:
        other.sendto(b"command", (emu.host, emu.command_port))
        with pytest.raises(socket.timeout):
            other.recvfrom(1024)
        other.sendto(b"takeoff", (emu.host, emu.command_port))
        time.sleep(0.2)
        assert emu.snapshot().phase == "ground"
    finally:
        other.close()
    assert [r[4] for r in emu.received if r[2] in ("command", "takeoff")][-2:] == ["ignored: another controller"] * 2
    rig.takeoff()  # kite keeps control


def test_no_video_packets_after_streamon_gives_the_phone_app_hint(tello_emulator_factory, monkeypatch):
    monkeypatch.setattr(tello_mod, "VIDEO_WAIT_S", 0.2)
    with tello_emulator_factory(connect=False, drone={"video": True}) as rig:
        rig.emu.phone_app(True)  # the drone answers us but streams to the phone
        with pytest.raises(RuntimeError, match="sends no video"):
            rig.drone.connect()
        assert rig.emu.commands(include_rc=False).count("streamon") == tello_mod.STREAMON_TRIES


def test_connect_fails_cleanly_when_no_state_arrives(tello_emulator_factory):
    from djitellopy import TelloException

    with tello_emulator_factory(connect=False) as rig:
        rig.emu.pause_state()
        with pytest.raises(TelloException, match="state packet"):
            rig.drone.connect()


# ================================================================== the recorder's --fly session (no video codec)
class Pilot:
    """Keys like a pilot reading the status line: t, hold r to 2.0 m, hold w, hold e, then l, then Ctrl+C."""

    def __init__(self, rig, target_h=2.0):
        self.rig, self.target_h = rig, target_h
        self.step, self.t_step, self.last_key = "wait", None, 0.0
        self.t0 = time.monotonic()
        self.log = []

    def _go(self, step):
        self.step, self.t_step = step, time.monotonic()
        self.log.append(step)

    def _repeat(self, key, now):  # terminal key repeat: one char every 0.1 s
        if now - self.last_key >= 0.1:
            self.last_key = now
            return key
        return ""

    def read(self) -> str:
        now = time.monotonic()
        gov = self.rig.gov
        tel = self.rig.drone.telemetry()
        if self.step == "wait" and now - self.t0 > 0.5:
            self._go("takeoff")
            return "t"
        if self.step == "takeoff" and gov.flying and not gov.busy():
            self._go("climb")
        if self.step == "climb":
            if tel.height_m is not None and tel.height_m >= self.target_h - 0.12:
                self._go("forward")
                return " "
            return self._repeat("r", now)
        if self.step == "forward":
            if now - self.t_step > 1.5:
                self._go("turn")
                return " "
            return self._repeat("w", now)
        if self.step == "turn":
            if now - self.t_step > 1.0:
                self._go("land")
                return " l"
            return self._repeat("e", now)
        if self.step == "land" and not gov.flying and not gov.busy():
            self._go("quit")
            raise KeyboardInterrupt
        if now - self.t0 > 40:
            raise KeyboardInterrupt
        return ""


def check_recording(run, emu, min_frames=30):
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    names = [e["event"] for e in events]
    for must in ("takeoff", "takeoff -> ok", "land (key)", "land -> ok"):
        assert must in names, (must, names)
    rcs = [(e["lr"], e["fb"], e["ud"], e["yaw"]) for e in events if e["event"] == "rc"]
    assert (0, 0, 50, 0) in rcs and (0, 50, 0, 0) in rcs and (0, 0, 0, 50) in rcs, rcs
    tel = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
    assert len(tel) > 50 and all(b["t_mono"] > a["t_mono"] for a, b in zip(tel, tel[1:]))
    h_max = max(t["h"] for t in tel) / 100
    assert 1.85 <= h_max <= 2.35, h_max
    yaws = [t["yaw"] for t in tel]
    assert max(yaws) - min(yaws) >= 20
    assert tel[-1]["h"] == 0
    with open(run / "frames.csv", newline="") as f:
        frames = list(csv.DictReader(f))
    assert len(frames) >= min_frames
    cap = cv2.VideoCapture(str(run / "video.mkv"))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n >= len(frames) - 5
    meta = json.loads((run / "meta.json").read_text())
    assert meta["summary"]["end_reason"] == "Ctrl+C" and meta["summary"]["telemetry_lines"] == len(tel)
    s = emu.snapshot()
    assert s.phase == "ground" and s.north > 0.3  # it flew forward, then landed
    return events, tel


@pytest.mark.emulator(time_scale=2.0)
def test_recorder_fly_session_takeoff_climb_forward_turn_land(tello_emulator, tmp_path):
    rig = tello_emulator
    cam = EmulatorCamera(rig.emu, size=(320, 240)).start()
    try:
        pilot = Pilot(rig)
        run = Recorder(cam, rig.gov, RecordCfg(out_dir=str(tmp_path), fly=True, speed=50, pulse_s=0.2),
                       {"source_kind": "emulator"}, keys=pilot).run()
    finally:
        cam.stop()
    assert pilot.log == ["takeoff", "climb", "forward", "turn", "land", "quit"]
    check_recording(run, rig.emu)


@pytest.mark.emulator(drone={"dry_run": True})
def test_dry_run_never_sends_motion(tello_emulator):
    """What `kite.record` without --fly, preflight and calibrate open: the keys 'work', the drone never moves."""
    rig, emu = tello_emulator, tello_emulator.emu
    rig.gov.takeoff()
    assert rig.wait() == "ok" and rig.gov.flying
    rig.fly(0.5, fb=40)
    rig.gov.rotate(10)
    rig.gov.stop()
    rig.gov.emergency()
    assert emu.commands() == ["command"] and emu.snapshot().phase == "ground"


@pytest.mark.emulator(time_scale=4.0)
def test_ctrl_c_in_flight_lands_before_the_recorder_exits(tello_emulator, tmp_path):
    rig = tello_emulator

    class Keys:
        def __init__(self):
            self.t0 = time.monotonic()

        def read(self):
            if time.monotonic() - self.t0 > 0.3 and not rig.gov.flying and not rig.gov.busy():
                return "t"
            if rig.gov.flying and not rig.gov.busy():
                raise KeyboardInterrupt
            return ""

    cam = EmulatorCamera(rig.emu).start()
    try:
        run = Recorder(cam, rig.gov, RecordCfg(out_dir=str(tmp_path), fly=True), {"source_kind": "emulator"},
                       keys=Keys()).run()
    finally:
        cam.stop()
    assert rig.emu.snapshot().phase == "ground" and not rig.gov.flying  # down before run() returned
    events = [json.loads(line)["event"] for line in (run / "events.jsonl").read_text().splitlines()]
    assert "land (exit)" in events and "landed" in events


# ================================================================== preflight's drone checks
class CameraTello(TelloDrone):
    """TelloDrone whose video is the emulator's in-process camera (the H.264 path is in the slow tests)."""

    def __init__(self, cam, after_connect=None, **kw):
        super().__init__(**kw)
        self._cam, self._after_connect = cam, after_connect

    def connect(self):
        super().connect()
        if self._after_connect is not None:
            self._after_connect()

    def frame_source(self):
        return self._cam


def run_preflight(rig, after_connect=None):
    from kite import preflight

    cam = EmulatorCamera(rig.emu, size=(960, 720)).start()
    try:
        return {c.name: c for c in preflight.drone_checks(
            lambda: CameraTello(cam, after_connect, host=rig.emu.host, video=False, dry_run=True),
            first_frame_s=3.0, measure_s=1.0)}
    finally:
        cam.stop()


@pytest.mark.emulator(connect=False)
def test_preflight_drone_checks_go_against_the_emulator(tello_emulator):
    from kite.preflight import GO

    checks = run_preflight(tello_emulator)
    bad = {n: (c.status, c.detail) for n, c in checks.items() if c.status != GO}
    assert not bad, bad
    assert set(checks) == {"drone link", "drone battery", "drone temperature", "video", "frame size", "telemetry",
                           "telemetry gaps", "imu", "command latency", "drone link closed"}
    sent = tello_emulator.emu.commands()
    assert sent[0] == "command" and set(sent) <= {"command", "battery?"}  # dry run: nothing that moves


@pytest.mark.emulator(connect=False, battery_pct=40)
def test_preflight_no_go_on_a_low_drone_battery(tello_emulator):
    checks = run_preflight(tello_emulator)
    assert checks["drone battery"].status == "NO-GO"


@pytest.mark.emulator(connect=False)
def test_preflight_no_go_when_hot_or_mute(tello_emulator):
    rig = tello_emulator
    rig.emu.set_temperature(85, 88)
    checks = run_preflight(rig, after_connect=lambda: rig.emu.silence_replies())
    assert checks["drone temperature"].status == "NO-GO"
    assert checks["command latency"].status == "NO-GO"


# ================================================================== slow: real H.264 video from the emulator
@pytest.mark.slow
@needs_ffmpeg
@pytest.mark.parametrize("backend", ["pyav", "opencv"])
def test_video_streams_960x720_and_shows_the_drones_motion(tello_emulator_factory, backend):
    with tello_emulator_factory(connect=False, video=True, drone={"video": True, "video_backend": backend}) as rig:
        rig.drone.connect()
        src = rig.drone.frame_source()
        first = src.wait_first(15.0)
        assert first is not None and first.image.shape == (720, 960, 3)
        seq0, sent0, t0 = src.read().seq, rig.emu.video_frames, time.time()
        time.sleep(2.0)
        fps = (src.read().seq - seq0) / (time.time() - t0)
        sent_fps = (rig.emu.video_frames - sent0) / (time.time() - t0)
        # the emulator draws in this process, so under load it may send under 30 fps: kite must keep up with it
        assert sent_fps > 12 and fps > 0.85 * sent_fps, (fps, sent_fps)
        col0 = red_pole_column(src.read().image)
        assert col0 is not None and abs(col0 - 480) < 40  # the red pole due north, dead ahead
        rig.takeoff()
        rig.gov.rotate(20)
        assert rig.wait() == "ok"
        time.sleep(0.5)  # newer frames than the end of the turn
        col1 = red_pole_column(src.read().image)
        expected = -920.0 * math.tan(math.radians(20))  # turning right moves the scene left
        assert col1 is not None and col1 - col0 == pytest.approx(expected, abs=60), (col0, col1)
        rig.emu.pause_video()
        time.sleep(1.5)
        now = time.time()
        assert now - src.read().t > 1.0
        assert rig.gov.check(now, src.read().t) == "hover"  # video stall -> hover
        rig.emu.pause_video(False)
        f = src.read()
        t_end = time.time() + 10
        while time.time() < t_end and (src.read() is None or src.read().seq == f.seq):
            time.sleep(0.05)
        now = time.time()
        assert now - src.read().t < 0.5 and rig.gov.check(now, src.read().t) == "ok"
        assert rig.land() == "ok"
    assert not rig.emu.snapshot().stream_on  # close() turned the stream off


@pytest.mark.slow
@needs_ffmpeg
def test_preflight_drone_checks_go_with_real_video(tello_emulator_factory):
    from kite import preflight

    with tello_emulator_factory(connect=False, video=True) as rig:
        cfg = Config(drone=DroneCfg(host=rig.emu.host))
        checks = preflight.drone_checks(lambda: make_tello(cfg, dry_run=True))
        bad = {c.name: (c.status, c.detail) for c in checks if c.status != preflight.GO}
        assert not bad, bad
        assert {c.name for c in checks} >= {"video", "frame size", "telemetry", "command latency"}
        assert not set(rig.emu.commands()) - {"command", "streamon", "battery?", "streamoff"}


@pytest.mark.slow
@needs_ffmpeg
def test_kite_record_main_fly_session_over_h264(tello_emulator_factory, tmp_path, monkeypatch):
    """python -m kite.record --source tello --fly, start to finish: the CLI builds the drone with make_tello, opens the
    H.264 stream, flies on scripted keys, lands on 'l', stops on Ctrl+C."""
    import io

    from kite import record

    with tello_emulator_factory(connect=False, video=True, time_scale=2.0) as rig:
        host = rig.emu.host
        made = {}

        def config():
            cfg = Config(drone=DroneCfg(host=host))
            made["cfg"] = cfg
            return cfg

        real_make = make_tello

        def make(cfg, dry_run=False):
            d = real_make(cfg, dry_run=dry_run)
            made["drone"] = d
            return d

        class Keys:
            def __enter__(self):
                d = made["drone"]  # the pilot reads the same drone the recorder flies
                self.pilot = Pilot(type("R", (), {"gov": d, "drone": d})())
                return self.pilot

            def __exit__(self, *exc):
                return False

        class Tty(io.StringIO):
            def isatty(self):
                return True

        monkeypatch.setattr(record, "Config", config)
        monkeypatch.setattr(record, "TerminalKeys", Keys)
        monkeypatch.setattr(sys, "stdin", Tty())
        import kite.drone

        monkeypatch.setattr(kite.drone, "make_tello", make)
        out = tmp_path / "rec"
        assert record.main(["--source", "tello", "--out", str(out), "--fly", "--speed", "50", "--pulse", "0.2"]) == 0
        runs = list(out.iterdir())
        assert len(runs) == 1
        check_recording(runs[0], rig.emu, min_frames=100)
        meta = json.loads((runs[0] / "meta.json").read_text())
        assert meta["dry_run"] is False and meta["video"]["width"] == 960 and meta["video"]["height"] == 720


@pytest.mark.slow
def test_a_lost_takeoff_reply_ends_on_the_ground_before_kites_timeout(tello_emulator_factory):
    """Real time (scale 1): the takeoff 'ok' is lost. kite stays busy (20 s timeout) and sends nothing, so the Tello's
    own 15 s auto-land brings it down before kite gives up; kite then reports error: timeout, not flying."""
    with tello_emulator_factory(time_scale=1.0) as rig:
        rig.emu.silence_replies()
        rig.gov.takeoff()
        rig.emu.wait_until(lambda s: s.phase == "air", 7.0)
        rig.emu.wait_until(lambda s: s.phase == "ground", 13.0)
        assert rig.gov.busy()  # kite is still waiting for the takeoff reply
        rig.emu.silence_replies(False)
        assert rig.wait(25.0) == "error: timeout" and not rig.gov.flying
        assert [c for c in rig.emu.commands() if c not in ("command", "speed 50")] == ["takeoff"]
