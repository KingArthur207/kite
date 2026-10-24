"""Follow mode without a drone: controller signs and limits on synthetic boxes, the tracker, lost walker -> hover,
g / manual keys in kite.record against a fake Tello, the rehearsal on a file, and a closed-loop kinematic simulation
of the Moss St walk (straight, stop at the curb, left turn, crossing) seen through kite.camera.Camera."""

import csv
import json
import math
import threading
import time

import cv2
import numpy as np
import pytest

from kite.camera import Camera
from kite.config import Config, SafetyCfg
from kite.drone import SafetyGovernor
from kite.drone.base import Drone
from kite.drone.tello import TelloDrone
from kite.follow import (Detection, FollowCfg, FollowController, FollowMode, Track, WalkerTracker, rehearse,
                         scaled_camera)
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
    far = settle(controller(ff=0.0), track(box_at(2.0, half_w=40)), tel())  # head top higher, box smaller
    close = settle(controller(ff=0.0), track(box_at(0.8, half_w=110)), tel())  # head top lower in the frame
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
    ff, no_ff = controller(slew=1000.0), controller(slew=1000.0, ff=0.0)
    for i in range(8):
        t = 10.0 + 0.1 * i
        a = ff.update(track(box_at(1.2 + 0.05 * i), t=t), tel(), t)
        b = no_ff.update(track(box_at(1.2 + 0.05 * i), t=t), tel(), t)
    assert a.walker_m_s > 0.2 and b.fb == round(35 * (b.range_m - 1.2 - 0.1))
    assert a.fb > b.fb + 10


def test_clamps_and_rate_limit():
    ctrl = controller()
    tr = track(box_at(4.0, 30))
    first = ctrl.update(tr, tel(1.0), 10.0)
    assert all(abs(v) <= 150 * 0.1 + 1 for v in first.rc)  # one 0.1 s tick: at most slew x dt from 0
    cmd = settle(ctrl, tr, tel(1.0), t0=10.1, n=40)
    assert cmd.fb == 50 and cmd.yaw == 40 and cmd.ud == 30 and cmd.lr == 20  # FollowCfg max_* (and max_rc 50)
    small = FollowController(FollowCfg(), CAM, max_rc=25)
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


def run_sim(speed=0.5, max_rc=50, cfg=None, seed=0, drop=0.1, blackout=(20.0, 20.4), video_lag_s=0.25,
            dt=1 / 90):
    cfg = cfg or FollowCfg()
    rng = np.random.default_rng(seed)
    body = walker_points(1.70)
    sched = walker_schedule(speed)
    w_pos, w_head = np.zeros(3), 0.0  # walker heading north
    sim = SimTello([0.0, -cfg.behind_m, 1.70 + cfg.above_m], 0.0)
    gov = SafetyGovernor(sim, SafetyCfg(max_rc=max_rc))
    tracker, ctrl = WalkerTracker(cfg), FollowController(cfg, CAM, max_rc)
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
                cmd = ctrl.update(tr, gov.telemetry(), t)
                gov.rc(*cmd.rc)
                rel = w_pos - sim.pos
                fwd, _ = heading_axes(sim.heading)
                bearing = math.degrees(math.atan2(rel[0], rel[1])) - sim.heading
                log.append({"t": t, "state": cmd.state, "dist": float(np.hypot(rel[0], rel[1])),
                            "height": float(sim.pos[2]), "bearing": (bearing + 180) % 360 - 180, "rc": cmd.rc,
                            "range_est": cmd.range_m})
    return log


def check_sim(log, cfg=FollowCfg(), tol_m=1.0):
    run = [r for r in log if r["t"] > 3.0]
    states = [r["state"] for r in run]
    dist = np.array([r["dist"] for r in run])
    assert "lost" not in states, "walker left the frame"
    assert states.count("following") / len(states) > 0.9
    assert np.all(np.abs(dist - cfg.behind_m) < tol_m), (dist.min(), dist.max())
    assert dist.min() > 0.6  # never on top of the walker (the stop at the curb, the stop at the end)
    assert max(abs(r["bearing"]) for r in run) < 25  # walker stays well inside the 55 deg wide frame
    assert all(abs(r["height"] - 2.0) < 0.15 for r in run)
    return dist


def test_closed_loop_moss_walk_at_0_5_m_s_with_max_rc_50():
    log = run_sim(speed=0.5, max_rc=50)
    dist = check_sim(log)
    assert abs(log[-1]["dist"] - 1.2) < 0.3  # settles back behind the walker once they stop
    assert max(abs(r["rc"][1]) for r in log) <= 50


def test_closed_loop_faster_walker_needs_a_higher_max_rc():
    """0.65 m/s is faster than stick 50 (0.5 m/s): with SafetyCfg.max_rc 70 follow keeps up."""
    check_sim(run_sim(speed=0.65, max_rc=70, seed=1))


def test_closed_loop_without_feed_forward_lags_but_keeps_the_walker():
    """ff=0 (the fallback knob if the speed estimate misbehaves): distance error alone must drive fb, so it trails
    ~1.2 m further back at 0.4 m/s, but never loses the walker."""
    dist = check_sim(run_sim(speed=0.4, max_rc=50, cfg=FollowCfg(ff=0.0), seed=2), tol_m=1.6)
    with_ff = check_sim(run_sim(speed=0.4, max_rc=50, seed=2))
    assert dist.max() > with_ff.max() + 0.5
