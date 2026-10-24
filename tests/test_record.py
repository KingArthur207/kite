"""python -m kite.record: dry run on a video file, a hard crash mid-recording, keyboard flight against a fake Tello
(takeoff, a stick, and Ctrl+C landing it while the recorder keeps recording), and the --follow-yaw flags."""

import csv
import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from kite.config import Config
from kite.drone import SafetyGovernor
from kite.drone.tello import TelloDrone
from kite.record import Recorder, RecordCfg, build_parser, main
from kite.sources import FrameSource
from kite.types import Frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_FRAMES, FPS = 45, 60.0


def make_video(path, n=N_FRAMES, fps=FPS, size=(160, 120), textured=False):
    """Frame i: left half a flat grey of level 5*(i % 48) + 10 (order survives MJPG compression); textured=True fills
    the right half with noise so each frame is tens of KB like a real camera frame (flat frames are a few hundred
    bytes, which would sit in FFmpeg's write buffer)."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert w.isOpened()
    rng = np.random.default_rng(0)
    for i in range(n):
        img = np.full((size[1], size[0], 3), 5 * (i % 48) + 10, np.uint8)
        if textured:
            img[:, size[0] // 2:] = rng.integers(0, 256, (size[1], size[0] - size[0] // 2, 3), dtype=np.uint8)
        w.write(img)
    w.release()
    return str(path)


def read_back(path):
    cap = cv2.VideoCapture(str(path))
    levels = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        levels.append(int(round((float(img[:, : img.shape[1] // 2 - 8].mean()) - 10) / 5)))
    cap.release()
    return levels


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def only_run(out):
    runs = list(Path(out).iterdir())
    assert len(runs) == 1
    return runs[0]


def test_dry_run_on_a_file_records_every_frame(tmp_path):
    video = make_video(tmp_path / "ramp.avi")
    assert main(["--source", video, "--out", str(tmp_path / "rec")]) == 0
    run = only_run(tmp_path / "rec")
    for name in ("video.mkv", "frames.csv", "telemetry.jsonl", "events.jsonl", "meta.json"):
        assert (run / name).exists(), name

    frames = rows(run / "frames.csv")
    assert [int(r["index"]) for r in frames] == list(range(N_FRAMES))
    assert [int(r["seq"]) for r in frames] == list(range(1, N_FRAMES + 1))  # nothing dropped
    t = np.array([float(r["t_mono"]) for r in frames])
    assert np.all(np.diff(t) > 0)
    assert 0.5 * N_FRAMES / FPS < t[-1] - t[0] < 3 * N_FRAMES / FPS  # paced like a camera, not dumped at once

    levels = read_back(run / "video.mkv")
    assert len(levels) == N_FRAMES
    assert levels == list(range(N_FRAMES))  # same frames, same order, BGR round trip

    assert (run / "telemetry.jsonl").read_text() == ""
    meta = json.loads((run / "meta.json").read_text())
    assert meta["telemetry"].startswith("none") and meta["dry_run"] is True
    assert meta["summary"]["frames_written"] == N_FRAMES and meta["summary"]["end_reason"] == "end of file"
    assert meta["video"]["fps_nominal"] == FPS and meta["camera"]["fx_px"] is None


def test_a_hard_crash_mid_recording_leaves_a_playable_video(tmp_path):
    video = make_video(tmp_path / "long.avi", n=240, fps=60.0, size=(480, 360), textured=True)
    out = tmp_path / "rec"
    p = subprocess.Popen([sys.executable, "-m", "kite.record", "--source", video, "--out", str(out)], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 10:  # wait until it has written a good part of the clip
        csvs = list(out.glob("*/frames.csv"))
        if csvs and len(csvs[0].read_text().splitlines()) > 60:
            break
        time.sleep(0.05)
    p.send_signal(signal.SIGKILL)  # no cleanup at all: like a crash or a pulled battery
    p.wait()
    run = only_run(out)
    n_rows = len(rows(run / "frames.csv"))
    levels = read_back(run / "video.mkv")
    assert n_rows >= 60 and len(levels) >= n_rows - 5  # at most a few frames lost from the write buffers
    assert levels == [i % 48 for i in range(len(levels))]


# ------------------------------------------------------------------ flying through the recorder (fake Tello)
class FakeTello:
    """The djitellopy surface TelloDrone uses: replies arrive asynchronously; a new state dict every 0.1 s."""

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
            self.udp["state"] = {"pitch": 0, "roll": 0, "yaw": n, "h": 100 if self.is_flying else 0, "tof": 110,
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
    """A camera-like source: a new 160x120 frame every 1/30 s."""

    def __init__(self):
        self.t0 = time.time()

    def read(self):
        seq = int((time.time() - self.t0) * 30) + 1
        return Frame(np.full((120, 160, 3), seq % 200, np.uint8), self.t0 + seq / 30, seq, "live")


class ScriptedKeys:
    """Types keys at given times after the first read; KeyboardInterrupt stands for Ctrl+C."""

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


def fake_drone(dry_run=False):
    fake = FakeTello()
    drone = SafetyGovernor(TelloDrone(tello=fake, video=False, dry_run=dry_run), Config().safety)
    drone.connect()
    time.sleep(0.25)  # first state packets
    return drone, fake


def test_keys_fly_through_the_governor_and_ctrl_c_lands(tmp_path):
    drone, fake = fake_drone()
    keys = ScriptedKeys([(0.3, "t"), (0.9, "w"), (1.5, KeyboardInterrupt)])
    run = Recorder(LiveImage(), drone, RecordCfg(out_dir=str(tmp_path), fly=True, speed=30), {"source_kind": "fake"},
                   keys=keys).run()
    drone.close()
    sent = [s for s in fake.sent if s not in ("command", "speed 50")]
    assert sent[0] == "takeoff"
    assert ("rc", 0, 30, 0, 0) in sent  # the w key, through the governor
    assert "land" in sent and not drone.flying
    assert sent.index("land") > sent.index(("rc", 0, 30, 0, 0))

    events = [json.loads(line)["event"] for line in (run / "events.jsonl").read_text().splitlines()]
    assert "takeoff" in events and "takeoff -> ok" in events and "landed" in events
    tel = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
    assert len(tel) >= 10 and {"t_mono", "t_wall", "bat", "h", "baro"} <= set(tel[0])
    assert all(b["t_mono"] > a["t_mono"] for a, b in zip(tel, tel[1:]))
    assert len(rows(run / "frames.csv")) >= 30 and len(read_back(run / "video.mkv")) >= 30
    assert json.loads((run / "meta.json").read_text())["summary"]["end_reason"] == "Ctrl+C"


def test_record_only_sends_no_motion(tmp_path):
    drone, fake = fake_drone(dry_run=True)  # what --source tello without --fly builds
    Recorder(LiveImage(), drone, RecordCfg(out_dir=str(tmp_path), max_s=0.5), {"source_kind": "fake"}).run()
    drone.close()
    assert fake.sent == ["command"]


# ------------------------------------------------------------------ --follow-yaw (opt-in kite.fly yaw sources)
def parse_error(capsys, argv):
    """The message main() exits with on a bad command line (argparse: exit status 2, the error after the usage)."""
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    return capsys.readouterr().err.strip().splitlines()[-1]


def test_follow_yaw_defaults_to_the_law_and_pd_fly_need_follow_without_predict(capsys, monkeypatch, tmp_path):
    args = build_parser().parse_args(["--source", "tello", "--fly", "--follow"])
    assert args.follow_yaw == "law" and args.follow_yaw_params is None
    clip = str(tmp_path / "missing.mp4")  # the yaw checks come before the file check
    assert "--follow-yaw pd needs --follow" in parse_error(capsys, ["--source", clip, "--follow-yaw", "pd"])
    err = parse_error(capsys, ["--source", clip, "--follow", "--follow-predict", "--follow-yaw", "fly"])
    assert "--follow-yaw fly and --follow-predict are exclusive" in err
    assert "--follow-yaw-params needs --follow-yaw pd or fly" in parse_error(
        capsys, ["--source", clip, "--follow-yaw-params", str(tmp_path / "p.json")])
    assert "no such file" in parse_error(capsys, ["--source", clip, "--follow", "--follow-yaw", "pd",
                                                  "--follow-yaw-params", str(tmp_path / "p.json")])
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "scipy" else find_spec(name, *a, **k))
    err = parse_error(capsys, ["--source", clip, "--follow", "--follow-yaw", "fly"])
    assert "scipy is not installed" in err and 'pip install -e ".[fly]"' in err
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr("kite.follow.fly_available", lambda: None)  # scipy or not: the yaw checks pass ...
    err = parse_error(capsys, ["--source", clip, "--follow", "--follow-yaw", "pd"])
    assert "--follow needs --fly" in err and "yaw" not in err  # ... and the usual --follow checks follow


LAW_ONLY = textwrap.dedent("""
    import sys
    import kite.follow
    from kite.config import Config
    from kite.record import _make_follow, build_parser

    class NoFrames:
        def read(self):
            return None

    kite.follow.YoloPersonDetector = lambda weights=None: (lambda img: [])  # no ultralytics, no weights
    args = build_parser().parse_args(["--source", "tello", "--fly", "--follow"])
    follow = _make_follow(args, NoFrames(), Config())
    try:
        assert args.follow_yaw == "law"
        assert follow.yaw_source is None and follow.ctrl.yaw_source is None
    finally:
        follow.close()
    fly = sorted(m for m in sys.modules if m == "kite.fly" or m.startswith("kite.fly."))
    assert not fly, fly
    print("law ok")
""")


def test_follow_yaw_law_builds_no_source_and_never_imports_kite_fly():
    """A fresh interpreter (this one has imported kite.fly for its own tests): the default --follow builds no yaw
    source, says so in one line, and kite.fly never enters sys.modules."""
    r = subprocess.run([sys.executable, "-c", LAW_ONLY], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "law ok" in r.stdout
    assert "follow: yaw source law (the follow controller's own yaw law, the default)" in r.stdout


def test_follow_yaw_choices_match_kite_follow():
    from kite.follow import YAW_CHOICES
    from kite.record import build_parser

    act = next(a for a in build_parser()._actions if a.dest == "follow_yaw")
    assert tuple(act.choices) == YAW_CHOICES  # record keeps its own literal so --help never imports kite.follow
