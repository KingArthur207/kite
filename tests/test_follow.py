"""Follow mode without a drone: controller signs and limits on synthetic boxes, the tracker, lost walker -> hover,
g / manual keys in kite.record against a fake Tello, FollowMode with an opt-in yaw source (passed through, named in
the events and the status line, a fallback logged once), the rehearsal on a file and its --yaw flags, and a
closed-loop kinematic simulation of the Moss St walk (straight, stop at the curb, left turn, crossing) seen through
kite.camera.Camera."""

import csv
import dataclasses
import importlib.util
import json
import math
import threading
import time

import cv2
import numpy as np
import pytest

import kite.follow as follow_mod
from kite.camera import Camera
from kite.config import Config, SafetyCfg
from kite.drone import SafetyGovernor
from kite.drone.base import Drone
from kite.drone.tello import TelloDrone
from kite.follow import (Detection, FollowCfg, FollowController, FollowMode, Track, WalkerTracker, draw_overlay,
                         rehearse, scaled_camera)
from kite.follow import main as follow_main
from kite.record import Recorder, RecordCfg
from kite.record import main as record_main
from kite.sources import FrameSource
from kite.types import Frame, Telemetry

CAM = Camera()  # 960 x 720, fx 920
W, H = CAM.width, CAM.height


def box_at(range_m, bearing_deg=0.0, cfg=FollowCfg(), half_w=70):
    """A walker box (head top at the row for range_m, centred on bearing_deg), cut off by the bottom edge."""
    y1 = CAM.cy + CAM.fy * cfg.above_m / range_m
    xc = CAM.cx + CAM.fx * math.tan(math.radians(bearing_deg))
    return (xc - half_w, y1, xc + half_w, H)


def track(box, t=10.0, conf=0.8, hits=5):
    return Track(box, conf, t, (W, H), hits=hits)


def tel(h=2.0, t=10.0):
    return Telemetry(t=t, height_m=h)


def settle(ctrl, tr, te, t0=10.0, n=20):
    """Run the controller long enough for the slew limit to let the command reach its value."""
    cmd = None
    for i in range(n):
        t = t0 + 0.1 * i
        tr.t = t
        cmd = ctrl.update(tr, te, t)
    return cmd


def controller(**kw):
    c = FollowController(FollowCfg(**kw), CAM, max_rc=50)
    c.reset(2.0)
    return c


def p_only(**kw):
    """The proportional part alone (no integral, lag / tilt compensation or time gap): for checking gains and signs."""
    return controller(**{"k_i_fb": 0.0, "lag_comp": False, "tilt_comp": 0.0, "headway_s": 0.0, **kw})


# ------------------------------------------------------------------ controller signs and magnitudes
def test_walker_at_target_gives_no_command():
    cmd = settle(controller(ff=0.0), track(box_at(1.2)), tel(2.0))
    assert cmd.rc == (0, 0, 0, 0) and cmd.state == "following"
    assert cmd.range_m == pytest.approx(1.2, abs=0.01) and cmd.bearing_deg == pytest.approx(0, abs=0.1)


def test_walker_drifts_right_yaws_right_and_left_yaws_left():
    right = settle(controller(ff=0.0), track(box_at(1.2, 10)), tel())
    left = settle(controller(ff=0.0), track(box_at(1.2, -10)), tel())
    assert right.yaw > 0 and left.yaw < 0  # + = clockwise = turn right (kite.types, Drone.rc)
    assert right.yaw == pytest.approx(1.5 * (10 - 2), abs=1)  # k_yaw x (bearing - deadband)
    assert right.lr >= 0 and left.lr <= 0 and abs(right.lr) < abs(right.yaw)  # turns use yaw; lr only small
    assert right.fb == 0 and right.ud == 0


def test_walker_small_and_high_in_frame_flies_forward_too_close_backs_off():
    far = settle(p_only(ff=0.0), track(box_at(2.0, half_w=40)), tel())  # head top higher, box smaller
    close = settle(p_only(ff=0.0), track(box_at(0.8, half_w=110)), tel())  # head top lower in the frame
    assert far.fb > 0 and close.fb < 0
    assert far.fb == pytest.approx(35 * (2.0 - 1.2 - 0.1), abs=1)
    assert far.yaw == 0 and close.yaw == 0


def test_altitude_hold_from_telemetry():
    low = settle(controller(), track(box_at(1.2)), tel(1.7))
    high = settle(controller(), track(box_at(1.2)), tel(2.3))
    assert low.ud > 0 and high.ud < 0
    assert controller().height_target == 2.0
    none = settle(controller(), track(box_at(1.2)), Telemetry(t=0))  # no height in telemetry: no altitude hold
    assert none.ud == 0


def test_feed_forward_from_a_receding_walker():
    """Walker starts walking away at 0.5 m/s: the range grows -> fb above what the distance error alone gives."""
    ff, no_ff = controller(slew=1000.0), p_only(slew=1000.0, ff=0.0)
    for i in range(8):
        t = 10.0 + 0.1 * i
        a = ff.update(track(box_at(1.2 + 0.05 * i), t=t), tel(), t)
        b = no_ff.update(track(box_at(1.2 + 0.05 * i), t=t), tel(), t)
    assert a.walker_m_s > 0.2 and b.fb == round(35 * (b.range_m - 1.2 - 0.1))
    assert a.fb > b.fb + 10


def test_clamps_and_rate_limit():
    ctrl = controller(outpace_s=1e9)  # a walker that never comes closer would soon be 'outpaced' (hover)
    tr = track(box_at(4.0, 30))
    first = ctrl.update(tr, tel(1.0), 10.0)
    assert all(abs(v) <= 150 * 0.1 + 1 for v in first.rc)  # one 0.1 s tick: at most slew x dt from 0
    cmd = settle(ctrl, tr, tel(1.0), t0=10.1, n=40)
    assert cmd.fb == 50 and cmd.yaw == 40 and cmd.ud == 30 and cmd.lr == 20  # FollowCfg max_* (and max_rc 50)
    small = FollowController(FollowCfg(outpace_s=1e9), CAM, max_rc=25)
    small.reset(2.0)
    assert max(abs(v) for v in settle(small, tr, tel(1.0), n=40).rc) <= 25


def test_missed_frames_coast_then_lost_hovers_and_resumes():
    ctrl = controller()
    tr = track(box_at(1.2, 15))
    cmd = settle(ctrl, tr, tel())
    assert cmd.yaw > 0
    t_seen = tr.t
    coast = ctrl.update(tr, tel(), t_seen + 0.3)  # 0.3 s without the walker: coasting, decaying
    assert coast.state == "coast" and 0 <= coast.yaw < cmd.yaw
    lost = ctrl.update(tr, tel(1.5), t_seen + 0.6)  # > lost_s: hover now, even with a height error
    assert lost.state == "lost" and lost.rc == (0, 0, 0, 0)
    assert ctrl.update(None, tel(1.5), t_seen + 2.0).rc == (0, 0, 0, 0)  # never searches by moving
    stale_edge = track(box_at(1.2, 15), t=t_seen + 2.0)
    stale_edge.touches_bottom = False
    assert ctrl.update(stale_edge, tel(), t_seen + 2.0).state == "lost"  # resume only at the bottom edge
    back = ctrl.update(track(box_at(1.2, 15), t=t_seen + 2.1), tel(), t_seen + 2.1)
    assert back.state == "following" and back.yaw > 0


def test_frame_size_is_scaled():
    """A 480x360 frame with the same field of view gives the same range and bearing."""
    ctrl = controller()
    x1, y1, x2, y2 = box_at(1.6, 12)
    small = Track((x1 / 2, y1 / 2, x2 / 2, y2 / 2), 0.8, 10.0, (480, 360), hits=5)
    rng, bearing, _ = ctrl.measure(small)
    assert rng == pytest.approx(1.6, abs=0.02) and bearing == pytest.approx(12, abs=0.1)
    assert scaled_camera(CAM, 480, 360).fx == pytest.approx(460)


# ------------------------------------------------------------------ tracker
def test_tracker_acquires_the_box_cut_by_the_bottom_edge_not_the_biggest():
    tr = WalkerTracker(FollowCfg())
    bystander = Detection(600, 100, 900, 650, 0.95)  # bigger, more confident, but not cut off by the bottom edge
    walker = Detection(400, 600, 540, 720, 0.5)
    for i in range(3):
        t = tr.update([bystander, walker], 0.033 * i, W, H)
    assert t.box == pytest.approx(walker.box) and t.confirmed


def test_tracker_follows_nearest_tolerates_misses_and_gates_jumps():
    cfg = FollowCfg()
    tr = WalkerTracker(cfg)
    for i in range(3):
        tr.update([Detection(400, 600, 540, 720, 0.8)], 0.033 * i, W, H)
    tr.update([], 0.1, W, H)  # missed frames keep the track
    tr.update([], 0.133, W, H)
    t = tr.update([Detection(430, 598, 570, 720, 0.7), Detection(0, 500, 120, 720, 0.9)], 0.166, W, H)
    assert t.misses == 0 and 430 > t.box[0] > 400  # the near box, smoothed
    t = tr.update([Detection(800, 600, 940, 720, 0.9)], 0.2, W, H)  # a jump across the frame is someone else
    assert t.misses == 1 and t.t == pytest.approx(0.166)
    assert tr.update([], 0.2 + cfg.keep_s + 0.01, W, H) is None  # dropped after keep_s


# ------------------------------------------------------------------ in kite.record: g, manual keys cancel, lost
class FakeTello:
    """The djitellopy surface TelloDrone uses (as in test_record): async 'ok' replies, a state dict every 0.1 s."""

    def __init__(self):
        self.sent = []
        self.udp = {"responses": [], "state": {}}
        self.is_flying = False
        self._halt = threading.Event()
        threading.Thread(target=self._states, daemon=True).start()

    def _states(self):
        n = 0
        while not self._halt.wait(0.1):
            n += 1
            self.udp["state"] = {"pitch": 0, "roll": 0, "yaw": 0, "h": 200 if self.is_flying else 0, "tof": 200,
                                 "vgx": 0, "vgy": 0, "vgz": 0, "bat": 87, "baro": 12.5, "time": n // 10}

    def connect(self):
        self.sent.append("command")

    def send_control_command(self, cmd):
        self.sent.append(cmd)
        return True

    def get_own_udp_object(self):
        return self.udp

    def get_current_state(self):
        return self.udp["state"]

    def send_rc_control(self, *v):
        self.sent.append(("rc",) + tuple(v))

    def send_command_without_return(self, cmd):
        self.sent.append(cmd)
        threading.Timer(0.2, lambda: self.udp["responses"].append(b"ok")).start()

    def end(self):
        self._halt.set()


class LiveImage(FrameSource):
    def __init__(self):
        self.t0 = time.time()

    def read(self):
        seq = int((time.time() - self.t0) * 30) + 1
        return Frame(np.zeros((360, 480, 3), np.uint8), self.t0 + seq / 30, seq, "live")


class ScriptedKeys:
    def __init__(self, script):
        self.script, self.t0 = list(script), None

    def read(self):
        self.t0 = self.t0 or time.time()
        out = ""
        while self.script and time.time() - self.t0 >= self.script[0][0]:
            key = self.script.pop(0)[1]
            if key is KeyboardInterrupt:
                raise KeyboardInterrupt
            out += key
        return out


class WalkerOnTheRight:
    """Fake detector: a walker cut off by the bottom edge, right of centre (480x360 frames), until `gone_after` s."""

    def __init__(self, gone_after=1e9):
        self.t0, self.gone_after = None, gone_after

    def __call__(self, img):
        self.t0 = self.t0 or time.time()
        if time.time() - self.t0 > self.gone_after:
            return []
        return [Detection(300, 290, 370, 360, 0.8)]


def fly_with_follow(tmp_path, script, detector):
    fake = FakeTello()
    drone = SafetyGovernor(TelloDrone(tello=fake, video=False), Config().safety)
    drone.connect()
    time.sleep(0.25)
    frames = LiveImage()
    rec_holder = {}
    follow = FollowMode(frames, lambda: detector, FollowCfg(), CAM, max_rc=50,
                        log=lambda msg, quiet=False, **kw: rec_holder["rec"]._event(msg, quiet=quiet, **kw))
    rec = Recorder(frames, drone, RecordCfg(out_dir=str(tmp_path), fly=True, speed=30), {"source_kind": "fake"},
                   keys=ScriptedKeys(script), follow=follow)
    rec_holder["rec"] = rec
    follow.start_tracker()
    try:
        run = rec.run()
    finally:
        follow.close()
        drone.close()
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    rcs = [s[1:] for s in fake.sent if isinstance(s, tuple)]
    return fake, drone, events, rcs


def test_g_follows_and_any_manual_key_cancels(tmp_path):
    fake, drone, events, rcs = fly_with_follow(tmp_path, [
        (0.2, "g"),  # refused: not flying yet
        (0.3, "t"), (1.0, "g"), (1.8, "w"),  # follow on, then w takes over at once
        (2.2, "g"), (2.6, " "),  # follow on again, space cancels (hover)
        (2.9, "g"), (3.3, KeyboardInterrupt)], WalkerOnTheRight())
    names = [e["event"] for e in events]
    refused = [e for e in events if e["event"] == "follow refused"]
    assert refused and "not flying" in refused[0]["reason"]
    offs = [e["reason"] for e in events if e["event"] == "follow off"]
    assert offs[:2] == ["key 'w'", "key ' '"] and "landing" in offs[2]  # Ctrl+C cancels too, then lands
    assert names.count("follow on") == 3
    decisions = [e for e in events if e["event"] == "follow"]
    assert decisions and all(e["state"] == "following" for e in decisions)
    assert decisions[0]["box"] and decisions[0]["conf"] == 0.8
    # walker right of centre -> yaw right through the governor and TelloDrone
    i_w = rcs.index((0, 30, 0, 0))
    assert any(r[3] > 0 for r in rcs[:i_w]) and all(r[3] >= 0 for r in rcs[:i_w])
    assert rcs[i_w + 1:i_w + 3] == [(0, 30, 0, 0)] * 2  # the manual stick wins: no follow rc after w
    assert "land" in fake.sent and not drone.flying
    assert all(abs(v) <= 50 for r in rcs for v in r)


def test_walker_lost_hovers_and_says_so(tmp_path):
    fake, drone, events, rcs = fly_with_follow(tmp_path, [(0.3, "t"), (1.0, "g"), (2.6, KeyboardInterrupt)],
                                               WalkerOnTheRight(gone_after=1.5))
    names = [e["event"] for e in events]
    assert "follow on" in names and any(n.startswith("follow: walker lost") for n in names)
    lost_t = next(e["t_mono"] for e in events if e["event"].startswith("follow: walker lost"))
    after = [e for e in events if e["event"] == "follow" and e["t_mono"] >= lost_t]
    assert after and all(e["rc"] == [0, 0, 0, 0] and e["state"] == "lost" for e in after)
    assert rcs[-1] == (0, 0, 0, 0)


def test_wifi_down_refuses_and_cancels_follow():
    """TelloDrone.link_error (a send failed) behind the governor: no start, and a running follow stops for good."""
    fake = FakeTello()
    tello = TelloDrone(tello=fake, video=False)
    drone = SafetyGovernor(tello, Config().safety)
    drone.connect()
    tello._flying = True
    time.sleep(0.25)
    frames = LiveImage()
    events = []
    follow = FollowMode(frames, lambda: WalkerOnTheRight(), FollowCfg(), CAM, max_rc=50,
                        log=lambda msg, quiet=False, **kw: events.append((msg, kw))).start_tracker()
    try:
        t_end = time.time() + 3
        while not (follow.ready and follow.track and follow.track.confirmed) and time.time() < t_end:
            time.sleep(0.02)
        tello.link_error = "[Errno 51] Network is unreachable"
        assert "Wi-Fi down" in follow.start(drone) and not follow.active
        tello.link_error = None
        assert follow.start(drone) is None and follow.active
        assert follow.tick(drone)[3] > 0  # walker on the right: yaw right
        tello.link_error = "[Errno 55] No buffer space available"
        assert follow.tick(drone) == (0, 0, 0, 0) and not follow.active
        tello.link_error = None
        assert follow.tick(drone) == (0, 0, 0, 0)  # the link is back: stays off until g
        assert events[-1] == ("follow off", {"reason": "Wi-Fi down ([Errno 55] No buffer space available)"})
    finally:
        follow.close()
        fake.end()


# ------------------------------------------------------------------ FollowMode with an opt-in yaw source
class FakeYaw:
    """A YawSource that always turns left (the law would turn right toward WalkerOnTheRight). After `after` calls it
    fails: fail="raise" raises (FollowController drops it for good), fail="own" falls back on its own like kite.fly
    (`failed` set, the law's value returned) until its next reset."""

    name, source = "fake", "fake_params.json"

    def __init__(self, after=None, fail="raise"):
        self.after, self.fail = after, fail
        self.calls = self.resets = 0
        self.failed = None

    def reset(self):
        self.resets += 1
        self.calls, self.failed = 0, None

    def yaw(self, t, frame_t, bearing_deg, range_m, fallback):
        self.calls += 1
        if self.after is not None and self.calls > self.after:
            if self.fail == "raise":
                raise RuntimeError("injected")
            self.failed = "RuntimeError: injected"
        return fallback if self.failed else -30.0


class FollowRig:
    """FollowMode on a fake Tello in the air with a walker right of centre; tick() runs it at 10 Hz."""

    def __init__(self, yaw_source=None):
        self.fake = FakeTello()
        tello = TelloDrone(tello=self.fake, video=False)
        self.drone = SafetyGovernor(tello, Config().safety)
        self.drone.connect()
        tello._flying = True
        time.sleep(0.25)
        self.events = []
        self.follow = FollowMode(LiveImage(), lambda: WalkerOnTheRight(), FollowCfg(), CAM, max_rc=50,
                                 log=lambda msg, quiet=False, **kw: self.events.append((msg, kw)),
                                 yaw_source=yaw_source).start_tracker()
        t_end = time.time() + 3
        while not (self.follow.ready and self.follow.track and self.follow.track.confirmed) and time.time() < t_end:
            time.sleep(0.02)

    def tick(self, n):
        rcs = []
        for _ in range(n):
            rcs.append(self.follow.tick(self.drone))
            time.sleep(0.1)
        return rcs

    def named(self, name):
        return [kw for msg, kw in self.events if msg == name]

    def failures(self):
        return [kw for msg, kw in self.events if msg.startswith("follow: yaw source")]

    def close(self):
        self.follow.close()
        self.fake.end()


def test_follow_mode_without_a_yaw_source_logs_as_before():
    rig = FollowRig()
    try:
        assert rig.follow.ctrl.yaw_source is None and rig.follow.start(rig.drone) is None
        rcs = rig.tick(5)
        assert rcs[-1][3] > 0  # the law: walker on the right, yaw right
        assert rig.named("follow on")[0]["yaw_source"] == "law" and "yaw_params" not in rig.named("follow on")[0]
        assert all("yaw_source" not in kw and "yaw_source_error" not in kw for kw in rig.named("follow"))
        assert "yaw" not in rig.follow.status() and not rig.failures()
    finally:
        rig.close()


def test_follow_mode_passes_a_yaw_source_through_and_names_it():
    src = FakeYaw()
    rig = FollowRig(src)
    try:
        assert rig.follow.ctrl.yaw_source is src and "yaw fake" in rig.follow.status()  # also while off
        resets = src.resets
        assert rig.follow.start(rig.drone) is None and src.resets == resets + 1  # follow on resets the source
        rcs = rig.tick(6)
        assert all(r[3] <= 0 for r in rcs) and rcs[-1][3] == -30  # the source's stick, through the clamps and slew
        on = rig.named("follow on")[0]
        assert on["yaw_source"] == "fake" and on["yaw_params"] == "fake_params.json"
        ticks = rig.named("follow")
        assert len(ticks) == 6 and all(kw["yaw_source"] == "fake" and "yaw_source_error" not in kw for kw in ticks)
        assert rig.follow.status().startswith("FOLLOW FOLLOWING") and rig.follow.status().endswith("yaw fake")
        assert not rig.failures()
    finally:
        rig.close()


def test_a_yaw_source_that_raises_falls_back_to_the_law_logged_once():
    rig = FollowRig(FakeYaw(after=3, fail="raise"))
    try:
        assert rig.follow.start(rig.drone) is None
        rcs = rig.tick(12)
        assert rcs[2][3] < 0 and rcs[-1][3] > 0  # left while the source steered, then the law turns right
        assert rig.follow.ctrl.yaw_source is None and rig.follow.ctrl.yaw_source_error == "RuntimeError: injected"
        assert rig.failures() == [{"reason": "RuntimeError: injected"}]  # once, not every tick
        assert ("follow: yaw source fake failed -> the follow law steers yaw", {"reason": "RuntimeError: injected"}) \
            in rig.events
        last = rig.named("follow")[-1]
        assert last["yaw_source"] == "law" and last["yaw_source_error"] == "RuntimeError: injected"
        assert "yaw law (fake failed: RuntimeError: injected)" in rig.follow.status()
        rig.follow.cancel("test")
        assert rig.follow.start(rig.drone) is None  # dropped for good: the law again, still not logged twice
        rig.tick(2)
        assert rig.named("follow on")[-1]["yaw_source"] == "law" and len(rig.failures()) == 1
    finally:
        rig.close()


def test_a_yaw_source_falling_back_on_its_own_is_logged_once_per_episode():
    """kite.fly's sources catch their own failures (failed set, the law's value returned) until the next reset; the
    controller still holds them, so FollowMode reads `failed` to say the law is steering."""
    src = FakeYaw(after=3, fail="own")
    rig = FollowRig(src)
    try:
        assert rig.follow.start(rig.drone) is None
        rcs = rig.tick(12)
        assert rcs[-1][3] > 0 and rig.follow.ctrl.yaw_source is src and rig.follow.ctrl.yaw_source_error is None
        assert rig.failures() == [{"reason": "RuntimeError: injected"}]
        assert rig.named("follow")[-1]["yaw_source"] == "law"
        rig.follow.cancel("test")
        assert rig.follow.start(rig.drone) is None  # the reset clears it: the source steers again ...
        assert rig.named("follow on")[-1]["yaw_source"] == "fake"
        rig.tick(2)
        assert rig.named("follow")[-1]["yaw_source"] == "fake"
        rig.tick(4)  # ... until it fails again: logged again, once
        assert len(rig.failures()) == 2 and rig.named("follow")[-1]["yaw_source"] == "law"
    finally:
        rig.close()


# ------------------------------------------------------------------ rehearsal on a file (open loop)
class DriftingWalker:
    """Fake detector for a 960x720 clip: the walker drifts right and away, frame by frame."""

    def __init__(self):
        self.i = 0

    def __call__(self, img):
        self.i += 1
        x = 430 + 6 * self.i
        return [Detection(x, 590 - 3 * self.i, x + 120, 720, 0.7)]


def make_clip(path, n=40, fps=30.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    for i in range(n):
        w.write(np.full((H, W, 3), 40 + i, np.uint8))
    w.release()
    return path


def test_rehearsal_on_a_clip_writes_rc_per_frame_and_overlay(tmp_path):
    clip = make_clip(tmp_path / "walk.avi")
    s = rehearse(clip, DriftingWalker(), FollowCfg(), overlay=tmp_path / "overlay.mp4")
    rows = list(csv.DictReader(open(s["csv"])))
    assert s["csv"].endswith("walk.follow_dry_run.csv") and len(rows) == 40 == s["frames"]
    assert s["engaged_at_s"] == pytest.approx(2 / 30, abs=0.01)  # third frame: confirmed at the bottom edge
    assert rows[0]["state"] == "off" and rows[-1]["state"] == "following"
    assert int(rows[-1]["yaw"]) > 0 and int(rows[-1]["fb"]) > 0  # drifted right and away
    assert rows[-1]["height_m"] == "" and int(rows[-1]["ud"]) == 0  # a plain clip has no telemetry
    cap = cv2.VideoCapture(str(tmp_path / "overlay.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    cap.release()


def test_rehearsal_on_a_kite_record_take(tmp_path):
    clip = make_clip(tmp_path / "walk.avi", n=30)
    assert record_main(["--source", str(clip), "--out", str(tmp_path / "rec")]) == 0
    take = next((tmp_path / "rec").iterdir())
    s = rehearse(take, DriftingWalker())
    rows = list(csv.DictReader(open(take / "follow_dry_run.csv")))
    assert len(rows) == 30 and s["following_ticks"] > 0 and int(rows[-1]["yaw"]) > 0


def test_rehearsal_with_a_yaw_source(tmp_path):
    clip = make_clip(tmp_path / "walk.avi")
    s = rehearse(clip, DriftingWalker(), FollowCfg(), yaw_source=FakeYaw())
    rows = list(csv.DictReader(open(s["csv"])))
    assert s["yaw_source"] == "fake" and s["yaw_source_error"] is None
    assert int(rows[-1]["yaw"]) < 0  # the source's left turn, though the walker drifted right
    s = rehearse(clip, DriftingWalker(), FollowCfg(), out_csv=tmp_path / "b.csv", yaw_source=FakeYaw(after=2))
    assert s["yaw_source_error"] == "RuntimeError: injected"
    assert int(list(csv.DictReader(open(s["csv"])))[-1]["yaw"]) > 0  # the law took over
    s = rehearse(clip, DriftingWalker(), FollowCfg(), out_csv=tmp_path / "c.csv")
    assert s["yaw_source"] == "law" and s["yaw_source_error"] is None


def test_rehearsal_yaw_flags(capsys, monkeypatch, tmp_path):
    clip = str(make_clip(tmp_path / "walk.avi"))

    def error(argv):
        with pytest.raises(SystemExit) as e:
            follow_main(argv)
        assert e.value.code == 2
        return capsys.readouterr().err.strip().splitlines()[-1]

    assert "--yaw pd and --predict are exclusive" in error(["--source", clip, "--yaw", "pd", "--predict"])
    assert "--yaw-params needs --yaw pd or fly" in error(["--source", clip, "--yaw-params", clip])
    assert "no such file" in error(["--source", clip, "--yaw", "fly", "--yaw-params", str(tmp_path / "none.json")])
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "scipy" else find_spec(name, *a, **k))
    assert 'scipy is not installed (kite.fly needs it); one time, on normal internet: pip install -e ".[fly]"' in \
        error(["--source", clip, "--yaw", "fly"])
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(follow_mod, "detector_available", lambda weights=None: None)  # no ultralytics needed
    monkeypatch.setattr(follow_mod, "YoloPersonDetector", lambda weights=None: DriftingWalker())
    assert follow_main(["--source", clip]) == 0  # the default: the law, as before
    out = capsys.readouterr().out
    assert "yaw source law (the follow controller's own yaw law, the default)" in out and "fell back" not in out


# ------------------------------------------------------------------ closed loop: the Moss St walk, simulated
def walker_points(height=1.70):
    """Surface points of a walker (walker frame: forward, right, up; feet at 0): head ellipsoid + torso box."""
    pts = []
    for a in np.linspace(0, math.pi, 12):  # head: centre 0.11 m below the top
        for b in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            pts.append((0.10 * math.sin(a) * math.cos(b), 0.08 * math.sin(a) * math.sin(b),
                        height - 0.11 + 0.11 * math.cos(a)))
    for z in np.arange(height - 0.25, height - 0.85, -0.02):  # shoulders and torso
        for s in np.linspace(-0.22, 0.22, 23):
            pts += [(0.12, s, z), (-0.12, s, z)]
        for f in np.linspace(-0.12, 0.12, 13):
            pts += [(f, 0.22, z), (f, -0.22, z)]
    return np.array(pts)


def heading_axes(heading_deg):
    """Forward and right unit vectors in the world (x east, y north), heading clockwise from north."""
    h = math.radians(heading_deg)
    return np.array([math.sin(h), math.cos(h), 0.0]), np.array([math.cos(h), -math.sin(h), 0.0])


def project(points, pos, heading_deg, cam=CAM):
    """World points -> pixel (u, v) and depth through a level camera at pos facing heading_deg."""
    f, r = heading_axes(heading_deg)
    d = points - pos
    zc, xc, yc = d @ f, d @ r, -d[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return cam.cx + cam.fx * xc / zc, cam.cy + cam.fy * yc / zc, zc


def simulated_detection(cam_pos, cam_heading, walker_pos, walker_heading, body, rng, noise_px=3.0):
    f, r = heading_axes(walker_heading)
    world = walker_pos + body[:, :1] * f + body[:, 1:2] * r + body[:, 2:3] * np.array([0, 0, 1.0])
    u, v, zc = project(world, cam_pos, cam_heading)
    front = zc > 0.05
    inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inside.sum() < 5:
        return []
    x1, x2, y1, y2 = u[inside].min(), u[inside].max(), v[inside].min(), v[inside].max()
    if np.any(front & (v >= H) & (u >= x1) & (u <= x2)):
        y2 = H  # the body goes on below the frame: cut off by the bottom edge
    if x2 - x1 < 15 or y2 - y1 < 15:
        return []
    n = rng.normal(0, noise_px, 4)
    return [Detection(x1 + n[0], y1 + n[1], x2 + n[2], min(H, y2 + n[3]), 0.6)]


class SimTello(Drone):
    """Kinematic Tello: body-frame velocities follow the rc (stick_m_s per unit, yaw_deg_s per unit) with a lag."""

    def __init__(self, pos, heading, stick_m_s=0.01, yaw_deg_s=1.0, tau=0.35, yaw_tau=0.2):
        self.pos, self.heading = np.array(pos, float), float(heading)
        self.v = np.zeros(3)  # forward, right, up (m/s)
        self.w = 0.0  # deg/s clockwise
        self.cmd = (0, 0, 0, 0)
        self.now = 0.0
        self.k, self.kw, self.tau, self.yaw_tau = stick_m_s, yaw_deg_s, tau, yaw_tau

    def step(self, dt):
        lr, fb, ud, yaw = self.cmd
        self.v += (np.array([fb, lr, ud]) * self.k - self.v) * dt / self.tau
        self.w += (yaw * self.kw - self.w) * dt / self.yaw_tau
        self.heading += self.w * dt
        f, r = heading_axes(self.heading)
        self.pos += (self.v[0] * f + self.v[1] * r + np.array([0, 0, self.v[2]])) * dt
        self.now += dt

    def rc(self, lr, fb, ud, yaw):
        self.cmd = (lr, fb, ud, yaw)

    def telemetry(self):
        return Telemetry(t=self.now, height_m=float(self.pos[2]), tof_m=float(self.pos[2]), yaw_deg=self.heading,
                         battery_pct=80.0)

    flying = True

    def takeoff(self): ...
    def land(self): ...
    def emergency(self): ...
    def move(self, direction, cm): ...
    def rotate(self, deg): ...
    def stop(self): self.cmd = (0, 0, 0, 0)
    def busy(self): return False
    def last_result(self): return "ok"
    def frame_source(self): return None


MOSS_WALK = [  # (seconds or None, speed m/s, turn deg/s clockwise, metres): NE sidewalk to the corner, stop, left turn
    ("stand", 1.0), ("walk", 12.0), ("stand", 3.0), ("turn", -90.0), ("walk", 8.0), ("stand", 3.0)]


def walker_schedule(speed, radius=1.5):
    """[(t_end, speed, turn_rate_deg_s)] for MOSS_WALK at `speed`."""
    out, t = [], 0.0
    for kind, x in MOSS_WALK:
        if kind == "stand":
            t += x
            out.append((t, 0.0, 0.0))
        elif kind == "walk":
            t += x / speed
            out.append((t, speed, 0.0))
        else:
            dur = abs(math.radians(x)) * radius / speed
            t += dur
            out.append((t, speed, x / dur))
    return out


def moss_route(radius=1.5):
    """MOSS_WALK's polyline in the sim world (the route prior): 12 m north, the corner, 8 m west."""
    return np.array([[0.0, 0.0], [0.0, 12.0 + radius], [-radius - 8.0, 12.0 + radius]])


SIM_PLANT = {"stick_m_s": 0.01, "dead_s": 0.0, "response_s": 0.35, "tilt_comp": 0.0}  # SimTello's own numbers


def run_sim(speed=0.4, max_rc=50, cfg=None, seed=0, drop=0.1, blackout=(20.0, 20.4), video_lag_s=0.25,
            dt=1 / 90, route=None, gps_noise_m=2.0):
    """route: give the controller the route prior, with the walker's GPS = truth + a bias of gps_noise_m.
    The controller's plant model is set to this kinematic SimTello (SIM_PLANT): a level camera that never tilts, no
    dead time. tests/follow_sim.py flies the measured Tello plant (dead time, tilt, saturation, wind) instead."""
    cfg = dataclasses.replace(cfg or FollowCfg(), **SIM_PLANT)
    rng = np.random.default_rng(seed)
    body = walker_points(1.70)
    sched = walker_schedule(speed)
    w_pos, w_head = np.zeros(3), 0.0  # walker heading north
    sim = SimTello([0.0, -cfg.behind_m, 1.70 + cfg.above_m], 0.0)
    gov = SafetyGovernor(sim, SafetyCfg(max_rc=max_rc))
    tracker, ctrl = WalkerTracker(cfg), FollowController(cfg, CAM, max_rc, route=route)
    gps_bias = rng.normal(0, gps_noise_m / math.sqrt(2), 2)
    history = []  # (t, cam pos, cam heading, walker pos, walker heading): the video shows the past
    engaged, next_frame, next_tick = False, 0.0, 0.0
    log = []
    t = 0.0
    while t < sched[-1][0]:
        seg = next(s for s in sched if t < s[0])
        w_head += seg[2] * dt
        f, _ = heading_axes(w_head)
        w_pos = w_pos + f * seg[1] * dt
        sim.step(dt)
        t = sim.now
        history.append((t, sim.pos.copy(), sim.heading, w_pos.copy(), w_head))
        if t >= next_frame:  # a frame arrives: it shows the world video_lag_s ago
            next_frame += 1 / 30
            past = next((h for h in reversed(history) if h[0] <= t - video_lag_s), history[0])
            dets = simulated_detection(past[1], past[2], past[3], past[4], body, rng)
            if rng.random() < drop or blackout[0] <= t < blackout[1]:
                dets = []
            tr = tracker.update(dets, t, W, H)
        if t >= next_tick:
            next_tick += 0.1
            assert gov.check(t, t) == "ok"
            if not engaged and tr is not None and tr.confirmed:
                engaged = True  # g pressed once the walker is tracked
                ctrl.reset(float(sim.pos[2]))
            if engaged:
                f_w, _ = heading_axes(w_head)
                gps = w_pos[:2] + gps_bias + rng.normal(0, 0.3, 2) if route is not None else None
                cmd = ctrl.update(tr, gov.telemetry(), t, gps)
                gov.rc(*cmd.rc)
                rel = w_pos - sim.pos
                fwd, _ = heading_axes(sim.heading)
                bearing = math.degrees(math.atan2(rel[0], rel[1])) - sim.heading
                log.append({"t": t, "state": cmd.state, "dist": float(np.hypot(rel[0], rel[1])),
                            "height": float(sim.pos[2]), "bearing": (bearing + 180) % 360 - 180, "rc": cmd.rc,
                            "range_est": cmd.range_m, "planner": cmd.planner,
                            "standoff_err": float(np.hypot(*(trail_target(history, cfg.behind_m) - sim.pos[:2])))})
    return log


def trail_target(history, behind_m):
    """The ideal standoff: behind_m of arc length back along the walker's true path (where it already walked)."""
    pts, rem = history[-1][3][:2], behind_m
    for h in reversed(history[:-1]):
        d = float(np.hypot(*(h[3][:2] - pts)))
        if d >= rem:
            return pts + (h[3][:2] - pts) * rem / d
        rem -= d
        pts = h[3][:2]
    return pts


def sim_metrics(log):
    """Report numbers for one closed-loop run (after the first 3 s)."""
    run = [r for r in log if r["t"] > 3.0]
    err = np.array([r["standoff_err"] for r in run])
    dist = np.array([r["dist"] for r in run])
    states = [r["state"] for r in run]
    rc = np.array([r["rc"] for r in run], float)
    flips = sum(int(np.sum((np.sign(a[1:]) * np.sign(a[:-1]) < 0) & (np.abs(a[1:] - a[:-1]) > 10)))
                for a in (rc[:, 0], rc[:, 1], rc[:, 3]))
    dur_min = (run[-1]["t"] - run[0]["t"]) / 60
    return {"standoff_mean": float(err.mean()), "standoff_p95": float(np.percentile(err, 95)),
            "max_lag": float(dist.max() - FollowCfg().behind_m), "min_dist": float(dist.min()),
            "max_bearing": max(abs(r["bearing"]) for r in run),
            "lost": sum(1 for a, b in zip(states, states[1:]) if b == "lost" and a != "lost"),
            "flips_per_min": flips / dur_min, "jerk": float(np.abs(np.diff(rc, axis=0)).sum(1).mean()),
            "predict_frac": sum(r["planner"] == "predict" for r in run) / len(run)}


def check_sim(log, cfg=FollowCfg(), tol_m=1.0, speed=0.4):
    """speed: the walker's walking speed; the reactive standoff grows by headway_s x speed while walking."""
    run = [r for r in log if r["t"] > 3.0]
    states = [r["state"] for r in run]
    dist = np.array([r["dist"] for r in run])
    assert "lost" not in states, "walker left the frame"
    assert states.count("following") / len(states) > 0.9
    assert np.all(dist - cfg.behind_m < tol_m + cfg.headway_s * speed), (dist.min(), dist.max())
    assert np.all(cfg.behind_m - dist < tol_m), (dist.min(), dist.max())
    assert dist.min() > 0.85  # never on top of the walker (the stop at the curb, the stop at the end)
    assert max(abs(r["bearing"]) for r in run) < 25  # walker stays well inside the 55 deg wide frame
    assert all(abs(r["height"] - 2.0) < 0.15 for r in run)
    return dist


def test_closed_loop_moss_walk_at_0_4_m_s_with_max_rc_50():
    """Stick 50 is ~0.48 m/s (0.0096 m/s per unit): a 0.4 m/s walker leaves the drone some speed to catch up with.
    (0.5 m/s at max_rc 50 is faster than the drone: follow would end up 'outpaced', hovering and warning.)"""
    log = run_sim(speed=0.4, max_rc=50)
    check_sim(log)
    assert abs(log[-1]["dist"] - 1.2) < 0.3  # settles back behind the walker once they stop
    assert max(abs(r["rc"][1]) for r in log) <= 50


def test_closed_loop_faster_walker_needs_a_higher_max_rc():
    """0.65 m/s is faster than stick 50 (~0.48 m/s): with SafetyCfg.max_rc 80 (~0.77 m/s) follow keeps up."""
    check_sim(run_sim(speed=0.65, max_rc=80, seed=1), speed=0.65)


def test_closed_loop_without_feed_forward_lags_but_keeps_the_walker():
    """ff=0 (the fallback knob if the speed estimate misbehaves) and no integral: distance error alone must drive
    fb, so it trails further back at 0.4 m/s, but never loses the walker. The integral (on by default) takes most
    of that lag back even without the feed-forward."""
    dist = check_sim(run_sim(speed=0.4, max_rc=50, cfg=FollowCfg(ff=0.0, k_i_fb=0.0), seed=2), tol_m=1.6)
    with_i = check_sim(run_sim(speed=0.4, max_rc=50, cfg=FollowCfg(ff=0.0), seed=2), tol_m=1.6)
    with_ff = check_sim(run_sim(speed=0.4, max_rc=50, seed=2))
    assert dist.max() > with_ff.max() + 0.5
    assert np.mean(np.abs(with_i - 1.2)) < np.mean(np.abs(dist - 1.2))


# ------------------------------------------------------------------ prediction (kite/predict.py) in the closed loop
def test_closed_loop_prediction_planner_holds_the_standoff_better_than_reactive():
    """Same walk, same seed: the planner keeps closer to the point 1.2 m back along the walker's path and keeps the
    walker well inside the frame through the corner, with every safety check of check_sim still holding. (The
    reactive controller now trails further back on purpose, by its time gap, which also shrinks its corner bearing.)"""
    reactive = sim_metrics(run_sim(seed=0))
    log = run_sim(cfg=FollowCfg(predict=True), seed=0)
    check_sim(log)
    pred = sim_metrics(log)
    assert pred["predict_frac"] > 0.95  # it steered (not the fallback) almost all the time
    assert pred["standoff_mean"] < reactive["standoff_mean"] - 0.05
    assert pred["max_bearing"] < 12.0
    assert pred["lost"] == 0 and pred["flips_per_min"] < 2
    assert max(abs(v) for r in log for v in r["rc"]) <= 50


def test_closed_loop_prediction_with_more_video_lag_and_dropped_frames():
    check_sim(run_sim(cfg=FollowCfg(predict=True), video_lag_s=0.4, drop=0.3, seed=3))


def test_closed_loop_prediction_with_the_route_prior():
    log = run_sim(cfg=FollowCfg(predict=True), route=moss_route(), seed=4)
    check_sim(log)
    assert sim_metrics(log)["standoff_mean"] < 0.5


def test_prediction_falls_back_to_the_reactive_controller_when_unsure():
    """max_std_m 0: the filter is never confident -> exactly the reactive controller's commands."""
    unsure = run_sim(cfg=FollowCfg(predict=True, max_std_m=0.0), seed=5)
    reactive = run_sim(seed=5)
    assert {r["planner"] for r in unsure} <= {"reactive", None}
    assert [r["rc"] for r in unsure] == [r["rc"] for r in reactive]


def test_prediction_still_hovers_when_the_walker_is_lost_and_resumes():
    """A 1 s blackout: rc 0 (hover) once lost_s has passed, even mid-plan; then the filter starts again."""
    log = run_sim(cfg=FollowCfg(predict=True), blackout=(20.0, 21.0), drop=0.0, seed=6)
    lost = [r for r in log if r["state"] == "lost"]
    assert lost and all(r["rc"] == (0, 0, 0, 0) for r in lost)
    after = [r for r in log if r["t"] > lost[-1]["t"]]
    assert after and after[-1]["state"] == "following" and after[-1]["planner"] == "predict"
    assert min(r["dist"] for r in log if r["t"] > 3) > 0.6


def test_rehearsal_draws_the_predicted_walker_and_planned_drone_paths(tmp_path):
    clip = make_clip(tmp_path / "walk.avi")
    s = rehearse(clip, DriftingWalker(), FollowCfg(predict=True), overlay=tmp_path / "overlay.mp4")
    rows = list(csv.DictReader(open(s["csv"])))
    assert rows[-1]["planner"] == "predict" and float(rows[-1]["pred_speed"]) > 0
    assert rows[0]["planner"] == "" and rows[0]["pred_speed"] == ""  # before follow engages
    # the same pieces by hand: a prediction and a plan, and the overlay draws them (the inset, bottom right)
    ctrl = FollowController(FollowCfg(predict=True), CAM, 50, open_loop=True)
    ctrl.reset(None)
    det, tr = DriftingWalker(), WalkerTracker(FollowCfg())
    for i in range(20):
        cmd = ctrl.update(tr.update(det(None), i / 10, W, H), None, i / 10)
    assert cmd.walker_path is not None and cmd.drone_path is not None and len(cmd.walker_path) == 16
    assert cmd.walker_path[-1][0] > cmd.walker_path[0][0]  # walking away from the camera
    img = np.full((H, W, 3), 60, np.uint8)
    with_pred = draw_overlay(img, [], tr.track, cmd, ctrl)
    cmd.walker_path = None
    plain = draw_overlay(img, [], tr.track, cmd, ctrl)
    assert np.abs(with_pred[H // 2:, W // 2:].astype(int) - plain[H // 2:, W // 2:]).sum() > 1000
