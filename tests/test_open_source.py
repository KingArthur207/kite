"""kite.sources.open_source: every spec kind against fakes (a synthetic clip, a dry-run take, a fake webcam, a fake
Tello, a fake sim module), camera scaling, the pose shape, and that the tello source never sends a motion command."""

import threading
import time
import types

import cv2
import numpy as np
import pytest

import kite.drone.tello as tello_mod
from kite.camera import Camera, load_camera
from kite.record import FileReplay, Recorder, RecordCfg
from kite.sources import POSE_KEYS, FrameSource, open_source, parse_spec
from kite.sources import cameras as cams
from kite.sources import open as op
from kite.types import Frame

N, FPS = 20, 60.0


def level(img) -> int:
    return int(round((float(img.mean()) - 10) / 5))


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Frame i: a flat grey of level 5*i + 10, 160x120 MJPG."""
    path = tmp_path_factory.mktemp("clip") / "ramp.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (160, 120))
    for i in range(N):
        w.write(np.full((120, 160, 3), 5 * i + 10, np.uint8))
    w.release()
    return path


def read_all(src):
    out = []
    while True:
        ok, img = src.read()
        if not ok:
            return out
        out.append((level(img), dict(src.pose)))


def test_parse_spec():
    assert parse_spec(0) == ("webcam", 0) and parse_spec("1") == ("webcam", 1) and parse_spec("webcam") == ("webcam", 0)
    assert parse_spec("webcam:2") == ("webcam", 2) and parse_spec("webcam:iPhone") == ("webcam", "iPhone")
    assert parse_spec("TELLO") == ("tello", None)
    assert parse_spec("sim") == ("sim", "drone") and parse_spec("sim:chest") == ("sim", "chest")
    with pytest.raises(ValueError):
        parse_spec("sim:car")
    with pytest.raises(ValueError):
        parse_spec("no/such/file.mp4")


def test_video_file_every_frame_in_order(clip):
    with open_source(str(clip)) as src:
        assert src.name == f"file:{clip}" and src.pose is None and len(src) == N
        got = read_all(src)
        assert [g[0] for g in got] == list(range(N))
        assert src.read() == (False, None)
    p = got[3][1]
    assert set(p) == set(POSE_KEYS) and p["t"] == pytest.approx(3 / FPS)
    assert p["cam_height"] is None and p["x"] is None
    # the Tello intrinsics, scaled to the clip with the same field of view
    cam, ref = src.camera, load_camera()
    assert (cam.width, cam.height) == (160, 120)
    assert cam.fx == pytest.approx(ref.fx * 160 / ref.width) and cam.hfov_deg == pytest.approx(ref.hfov_deg)
    assert "scaled to 160x120" in cam.source
    wide = op.tello_camera(1280, 720)  # a 16:9 stock clip: same horizontal FOV, square pixels, centred
    assert wide.hfov_deg == pytest.approx(ref.hfov_deg) and wide.fx / wide.fy == pytest.approx(ref.fx / ref.fy)
    assert (wide.cx, wide.cy) == (640, 360)


def test_video_file_realtime_and_options(clip):
    src = open_source(f"file:{clip}", realtime=True, speed=2.0, cam_height=1.4, pitch=-5.0, fx=200.0)
    t0 = time.monotonic()
    got = read_all(src)
    dt = time.monotonic() - t0
    src.release()
    assert len(got) == N and dt >= 0.9 * (N - 1) / FPS / 2  # paced at twice the clip's fps
    assert got[0][1]["cam_height"] == 1.4 and got[0][1]["pitch"] == -5.0
    assert src.camera.fx == 200.0 and src.camera.cx == 80.0
    with pytest.raises(TypeError):
        open_source(str(clip), max_gap_s=1.0)  # not an option of a file source


def test_recording_take(clip, tmp_path):
    rep = FileReplay(str(clip))
    try:
        run = Recorder(rep, None, RecordCfg(out_dir=str(tmp_path / "out"), fps=FPS), {"source_kind": "file"}).run()
    finally:
        rep.stop()
    for spec in (str(run), f"recording:{run}"):
        with open_source(spec) as src:
            assert src.name == f"recording:{run}"
            got = read_all(src)
            assert [g[0] for g in got] == list(range(N))
            assert set(got[0][1]) == set(POSE_KEYS) and got[0][1]["heading_ref"] == "tello_yaw"
            assert (src.camera.width, src.camera.height) == (160, 120)
    assert open_source(run, realtime=True).realtime


# ------------------------------------------------------------------ webcam
class FakeCapture:
    """cv2.VideoCapture stand-in: 1280x720 frames at ~100 fps, the index stamped in the pixel value."""

    opened: list = []

    def __init__(self, index, *a):
        self.index, self.n = index, 0
        FakeCapture.opened.append(index)

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def read(self):
        time.sleep(0.01)
        self.n += 1
        return True, np.full((720, 1280, 3), 10 * self.index + 1, np.uint8)

    def release(self):
        pass


@pytest.fixture
def fake_cams(monkeypatch):
    FakeCapture.opened = []
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cams, "opencv_cameras", lambda: [(0, "MacBook Pro Camera", "A"), (1, "Arthur's iPhone Camera", "B")])
    monkeypatch.setattr(cams, "system_cameras", lambda: ["MacBook Pro Camera", "Arthur's iPhone Camera",
                                                         "MacBook Pro Desk View Camera"])


def test_webcam_by_index(fake_cams):
    with open_source(0) as src:
        assert src.name == "webcam:0 MacBook Pro Camera"
        ok, img = src.read()
        assert ok and img.shape == (720, 1280, 3) and img.flags.writeable
        seq = src._seq
        ok, _ = src.read()
        assert ok and src._seq != seq  # a NEW frame each read
    p = src.pose
    assert set(p) == set(POSE_KEYS) and p["cam_height"] == op.CHEST_HEIGHT_M and p["pitch"] == 0.0
    assert p["x"] is None and p["heading"] is None and abs(p["t"] - time.time()) < 5
    cam = src.camera
    assert (cam.width, cam.height) == (1280, 720) and "uncalibrated" in cam.source
    assert cam.hfov_deg == pytest.approx(op.WEBCAM_HFOV_DEG) and cam.fx == cam.fy and cam.cx == 640
    assert FakeCapture.opened == [0]


def test_webcam_by_name_and_options(fake_cams):
    with open_source("webcam:iphone", cam_height=1.1, pitch=-10, hfov_deg=90) as src:
        ok, img = src.read()
        assert ok and int(img[0, 0, 0]) == 11  # index 1
        assert src.name == "webcam:1 Arthur's iPhone Camera"
        assert src.pose["cam_height"] == 1.1 and src.pose["pitch"] == -10
        assert src.camera.fx == pytest.approx(640.0)  # 90 deg over 1280 px
    with open_source("webcam:1", camera=Camera(640, 360, 500, 500, 320, 180, "calibrated")) as src:
        src.read()
        assert src.camera.fx == pytest.approx(1000) and "calibrated, scaled to 1280x720" == src.camera.source
    with pytest.raises(LookupError, match="MacBook Pro Camera"):
        open_source("webcam:android")


def test_webcam_only_in_avfoundation_list_opens_through_pyav(fake_cams, monkeypatch):
    seen = []

    class FakePyAV:
        def __init__(self, device, w, h):
            seen.append(device)

        def read(self):
            time.sleep(0.01)
            return True, np.zeros((1080, 1920, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(op, "_PyAVCamera", FakePyAV)
    with open_source("webcam:desk view") as src:
        ok, img = src.read()
        assert ok and seen == ["MacBook Pro Desk View Camera"] and src.camera.width == 1920
        assert "PyAV" in src.name


# ------------------------------------------------------------------ tello
class FakeTello:
    """The djitellopy surface TelloDrone uses; records every command; a new state dict every 0.05 s."""

    instances: list = []
    LOGGER = __import__("logging").getLogger("fake-tello")

    def __init__(self, host="192.168.10.1"):
        self.sent = []
        self.udp = {"responses": [], "state": {}}
        self.is_flying = False
        self._halt = threading.Event()
        FakeTello.instances.append(self)
        threading.Thread(target=self._states, daemon=True).start()

    def _states(self):
        n = 0
        while not self._halt.wait(0.05):
            n += 1
            self.udp["state"] = {"pitch": 3, "roll": -2, "yaw": -90, "h": 150, "tof": 148, "vgx": 0, "vgy": 0,
                                 "vgz": 0, "bat": 77, "time": 0}

    def connect(self):
        self.sent.append("command")

    def streamon(self):
        self.sent.append("streamon")

    def streamoff(self):
        self.sent.append("streamoff")

    def get_udp_video_address(self):
        return "udp://@0.0.0.0:11111"

    def get_current_state(self):
        return self.udp["state"]

    def get_own_udp_object(self):
        return self.udp

    def send_control_command(self, cmd):
        self.sent.append(cmd)
        return True

    def send_command_without_return(self, cmd):
        self.sent.append(cmd)

    def send_rc_control(self, *v):
        self.sent.append(("rc",) + tuple(v))

    def land(self):
        self.sent.append("land")

    def end(self):
        self.sent.append("end")
        self._halt.set()


class FakeVideo(FrameSource):
    """TelloVideoSource stand-in: 960x720 frames at ~60 fps."""

    def __init__(self, url, fps=60, backend="pyav"):
        self.t0 = time.time()
        self.stopped = False

    def read(self):
        seq = int((time.time() - self.t0) * 60)
        return None if seq < 1 else Frame(np.full((720, 960, 3), seq % 200, np.uint8), time.time(), seq, "tello")

    def stop(self):
        self.stopped = True


@pytest.fixture
def fake_tello(monkeypatch):
    import djitellopy

    FakeTello.instances = []
    monkeypatch.setattr(djitellopy, "Tello", FakeTello)
    monkeypatch.setattr(tello_mod, "video_packets_arrive", lambda port, timeout_s: True)
    monkeypatch.setattr(tello_mod, "TelloVideoSource", FakeVideo)


def test_tello_live_pose_and_no_motion(fake_tello):
    from kite.config import Config, DroneCfg

    cfg = Config(drone=DroneCfg(kind="tello"))  # a LIVE config: the source must still open it in dry run
    with open_source("tello", cfg=cfg) as src:
        assert src.name.startswith("tello") and src._drone.dry_run
        time.sleep(0.2)  # telemetry packets
        poses = []
        for _ in range(5):
            ok, img = src.read()
            assert ok and img.shape == (720, 960, 3) and img.flags.writeable
            poses.append(src.pose)
        # motion calls reaching the drone object are logged, never sent (belt and braces: nothing here calls them)
        src._drone.takeoff()
        src._drone.rc(0, 30, 0, 0)
    p = poses[-1]
    assert set(p) == set(POSE_KEYS) and p["heading_ref"] == "tello_yaw" and p["flying"] is False
    assert p["heading"] == pytest.approx(270.0) and p["cam_height"] == pytest.approx(1.48)  # ToF agrees with h
    assert p["pitch"] == pytest.approx(3.0) and p["roll"] == pytest.approx(-2.0) and p["battery"] == 77
    assert abs(p["tel_dt"]) <= 0.2
    assert src.camera.width == 960 and src.camera.fx == pytest.approx(load_camera().fx)
    fake = FakeTello.instances[0]
    allowed = {"command", "streamon", "streamoff", "end"}
    assert set(fake.sent) <= allowed, fake.sent
    assert fake.sent[:2] == ["command", "streamon"] and "end" in fake.sent


def test_tello_without_telemetry_has_empty_pose(fake_tello, monkeypatch):
    monkeypatch.setattr(FakeTello, "_states", lambda self: None)  # no state packets at all
    with open_source("tello") as src:
        ok, _ = src.read()
        assert ok and src.pose["cam_height"] is None and src.pose["heading"] is None and src.pose["t"] is not None
    assert set(FakeTello.instances[0].sent) <= {"command", "streamon", "streamoff", "end"}


# ------------------------------------------------------------------ sim
@pytest.fixture
def fake_sim(monkeypatch):
    made = []

    class SimFrameSource:
        def __init__(self, camera="drone", url="ws://127.0.0.1:5020/sub", timeout=2.0):
            self.camera, self.url, self.n = camera, url, 0
            self.pose = None
            made.append(self)

        def read(self):
            self.n += 1
            if self.n > 3:
                time.sleep(0.01)
                return False, None
            h = 1.2 if self.camera == "chest" else 2.0
            self.pose = {"type": "pose", "camera": self.camera, "x": 1.5 * self.n, "y": -2.0, "heading": 45.0,
                         "cam_height": h, "pitch": 0}
            return True, np.zeros((720, 960, 3), np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setitem(__import__("sys").modules, op.SIM_MODULE, types.SimpleNamespace(SimFrameSource=SimFrameSource))
    return made


def test_sim_sources(fake_sim):
    with open_source("sim:chest", url="ws://x/sub") as src:
        got = read_all(src)
    assert len(got) == 3 and fake_sim[0].camera == "chest" and fake_sim[0].url == "ws://x/sub" and fake_sim[0].released
    p = got[1][1]
    assert set(p) == set(POSE_KEYS) and p["x"] == 3.0 and p["y"] == -2.0 and p["heading"] == 45.0
    assert p["cam_height"] == 1.2 and p["pitch"] == 0 and p["heading_ref"] == "compass" and p["battery"] is None
    cam = src.camera
    assert (cam.width, cam.height, cam.cx) == (960, 720, 480) and cam.fx == op.sim_camera().fx
    assert "config.ts" in cam.source and cam.fx == pytest.approx(1000)  # sim/src/config.ts FX today
    with open_source("sim") as src:
        assert src.name == "sim:drone" and src.read()[0] and src.pose["cam_height"] == 2.0


# ------------------------------------------------------------------ CLI and camera list
def test_cli_check_on_a_file(clip, capsys):
    assert op.main([str(clip), "--seconds", "5"]) == 0
    out = capsys.readouterr().out
    assert f"frames  {N}" in out and "160x120" in out and "end of the file" in out and "pose    t 0.00" in out
    assert op.main(["no/such.mp4"]) == 1


def test_list_cameras_cli(fake_cams, monkeypatch, capsys):
    monkeypatch.setattr(cams, "camera_permission", lambda: "authorized")
    assert op.main(["--list-cameras"]) == 0
    out = capsys.readouterr().out
    assert "1  Arthur's iPhone Camera" in out and "Desk View" in out and "authorized" in out


def test_camera_name_match():
    assert cams._match("iphone", ["MacBook Pro Camera", "Arthur's iPhone Camera"]) == "Arthur's iPhone Camera"
    assert cams._match("macbook pro camera", ["MacBook Pro Camera Extra", "MacBook Pro Camera"]) == "MacBook Pro Camera"
    assert cams._match("zzz", ["a"]) is None
