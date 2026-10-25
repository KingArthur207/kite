"""`python -m perception.near` end to end on fakes: every source branch (still, video file, webcam, UNO Q), --bands,
--show, --hud, --jsonl, --wav and the earpiece sender. No model, no camera, no window, no sound, no network:
the detector is a perfect synthetic one, depth comes from perception/near/synth.py, cv2's window calls are
recorded instead of drawn, and the earpiece client only collects messages."""

from __future__ import annotations

import json
import sys
import types
import wave

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

import kite.sources  # noqa: E402
from perception.near import __main__ as M  # noqa: E402
from perception.near import depth as D  # noqa: E402
from perception.near import intrinsics as I  # noqa: E402
from perception.near import priority, synth, unoq  # noqa: E402

PROF = I.BUILTIN["c270"].scaled(640, 360)
PERSON_AT = 1.0                                       # m straight ahead: P1 at the default bands


def scene(forward: float = PERSON_AT):
    return synth.render(PROF, [synth.Obj("person", forward, 0.0, 1.7, 0.5)], cam_h=1.2, pitch=5.0,
                        rng=np.random.default_rng(0), jitter=0.0, jitter_px=0.0, depth_noise=0.0, obj_bias=0.0)


class FakeDetector:
    """What a perfect YOLO would return for the synthetic scene, whatever the frame shows."""

    def __init__(self, dets):
        self.dets, self.calls = dets, 0

    def __call__(self, img):
        self.calls += 1
        return list(self.dets)


class Window:
    """cv2.imshow / waitKey / destroyAllWindows, recorded. waitKey answers from `keys` (then -1: no key)."""

    def __init__(self, keys=()):
        self.shown, self.keys, self.destroyed = [], list(keys), 0

    def imshow(self, title, img):
        self.shown.append((title, img.shape))

    def waitKey(self, ms):  # noqa: N802  (cv2's name)
        return self.keys.pop(0) if self.keys else -1

    def destroyAllWindows(self):  # noqa: N802
        self.destroyed += 1


@pytest.fixture
def near(monkeypatch, capsys):
    """Patches every heavy or physical dependency of main() and returns a namespace to drive and inspect it."""
    r = scene()
    ns = types.SimpleNamespace(render=r, det=FakeDetector(r.dets), window=Window(), backends=[], sent=[],
                               clients=[], depth_calls=0)
    monkeypatch.setattr(priority, "P1_M", priority.P1_M)       # main() mutates the bands: restore them after
    monkeypatch.setattr(priority, "P2_M", priority.P2_M)
    monkeypatch.setattr(M, "build_detector", lambda kind, device: ns.det)
    monkeypatch.setattr(M, "pick_device", lambda want: want or "cpu")

    def make_backend(spec, device="cpu"):
        ns.backends.append((spec, device))
        if spec == "off":
            return None

        def fn(frame):
            ns.depth_calls += 1
            return r.depth
        return D.ArrayBackend(fn, "metric", "synthetic")

    monkeypatch.setattr(D, "make_backend", make_backend)
    for k in ("imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(cv2, k, getattr(ns.window, k))

    class Ear:
        def __init__(self, host, port, source):
            ns.clients.append((host, port, source))

        def send(self, msg):
            ns.sent.append(msg)

    import earpiece.client
    monkeypatch.setattr(earpiece.client, "EarpieceClient", Ear)

    def run(*argv):
        M.main(list(argv))
        return capsys.readouterr().out

    ns.run = run
    return ns


def frame():
    return synth.image_for(PROF)


def still(tmp_path):
    p = tmp_path / "photo.jpg"
    assert cv2.imwrite(str(p), frame())
    return p


# ------------------------------------------------------------------ argument parsing
def test_defaults_and_flags_parse():
    a = M.parse_args(["--source", "unoq"])
    assert a.depth == "outdoor" and a.detector == "coco" and a.bands is None and not a.show and a.ear_port
    a = M.parse_args(["--source", "x.mp4", "--bands", "1,2.5", "--show", "--no-send", "--detector", "hybrid"])
    assert a.bands == "1,2.5" and a.show and a.no_send and a.detector == "hybrid"
    with pytest.raises(SystemExit):
        M.parse_args(["--detector", "yolo-xxl"])


def test_list_profiles_prints_every_builtin_and_needs_no_source(capsys):
    M.main(["--list-profiles"])
    out = capsys.readouterr().out
    assert all(k in out for k in I.BUILTIN)


@pytest.mark.parametrize("bands,why", [("abc", "two distances"), ("1,2,3", "two distances"), ("1", "two distances"),
                                       ("3,1", "0 < P1 < P2"), ("0,1", "0 < P1 < P2"), ("2,2", "0 < P1 < P2")])
def test_bad_bands_are_refused_with_a_reason(near, bands, why):
    with pytest.raises(SystemExit, match=why):
        near.run("--source", "unoq", "--bands", bands)
    assert not near.backends                                  # refused before anything was opened or loaded


def test_bands_set_the_priority_thresholds(near, tmp_path):
    near.run("--source", str(still(tmp_path)), "--profile", "c270", "--no-send", "--bands", "0.4,0.8")
    assert (priority.P1_M, priority.P2_M) == (0.4, 0.8)


@pytest.mark.parametrize("argv,why", [((), "--source is required"),
                                      (("--source", "unoq", "--hud", "x.mp4"), "--hud is for files"),
                                      (("--source", "nope/missing.jpg"), "cannot read")])
def test_usage_errors_exit_with_a_message(near, argv, why):
    with pytest.raises(SystemExit, match=why):
        near.run(*argv, "--no-send")


def test_an_unknown_profile_is_a_clean_exit(near, tmp_path):
    with pytest.raises(SystemExit, match="--profile 'c9000'"):
        near.run("--source", str(still(tmp_path)), "--profile", "c9000", "--no-send")


# ------------------------------------------------------------------ a still: events, review image, beeps
def test_a_still_gives_one_p1_event_a_review_image_and_a_wav_that_is_never_played(near, tmp_path):
    jsonl, hud, wav = tmp_path / "near.jsonl", tmp_path / "out" / "hud.png", tmp_path / "beeps.wav"
    out = near.run("--source", str(still(tmp_path)), "--profile", "c270", "--no-send", "--depth", "indoor",
                   "--device", "cpu", "--jsonl", str(jsonl), "--hud", str(hud), "--wav", str(wav))
    assert near.backends == [("indoor", "cpu")] and near.depth_calls == 1 and near.det.calls == 1
    evs = [json.loads(ln) for ln in jsonl.read_text().splitlines()]
    assert len(evs) == 1 and evs[0]["priority"] == 1 and evs[0]["kind"] == "person"
    assert evs[0]["distance_m"] == pytest.approx(PERSON_AT, abs=0.3)
    img = cv2.imread(str(hud.with_suffix(".jpg")))           # a still's review is a JPEG next to --hud
    assert img is not None and img.shape[1] == 2 * PROF.width  # camera | depth side by side, radar below
    with wave.open(str(wav)) as w:
        assert w.getnchannels() == 2 and w.getnframes() > 0
    assert "(not played)" in out and "1 frames; P1 1" in out
    assert not near.window.shown and not near.clients        # no --show, --no-send: no window, no earpiece


def test_wider_bands_turn_the_same_scene_into_p3(near, tmp_path):
    jsonl = tmp_path / "near.jsonl"
    near.run("--source", str(still(tmp_path)), "--profile", "c270", "--no-send", "--bands", "0.3,0.6",
             "--jsonl", str(jsonl))
    ev = json.loads(jsonl.read_text())
    assert ev["priority"] == 3                                # 1 m is beyond P2 now


def test_depth_off_still_runs_on_the_detector_alone(near, tmp_path):
    out = near.run("--source", str(still(tmp_path)), "--profile", "c270", "--no-send", "--depth", "off")
    assert near.backends == [("off", "cpu")] and near.depth_calls == 0 and "1 frames" in out


# ------------------------------------------------------------------ a video file
class FakeCapture:
    """cv2.VideoCapture over n synthetic frames at `fps`."""
    instances: list = []

    def __init__(self, path, *args, n=7, fps=30.0, opened=True):
        self.path, self.n, self.fps, self.opened, self.i, self.released = path, n, fps, opened, 0, False
        self.props: dict = {}
        FakeCapture.instances.append(self)

    def isOpened(self):  # noqa: N802
        return self.opened

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: PROF.width,
                cv2.CAP_PROP_FRAME_HEIGHT: PROF.height}.get(prop, 0.0)

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def grab(self):
        if self.i >= self.n:
            return False
        self.i += 1
        return True

    def read(self):
        if not self.grab():
            return False, None
        return True, frame()

    def release(self):
        self.released = True


class FakeWriter:
    made: list = []

    def __init__(self, path, fourcc, fps, size):
        self.path, self.fps, self.size, self.frames, self.released = path, fps, size, [], False
        FakeWriter.made.append(self)

    def write(self, img):
        self.frames.append(img.shape)

    def release(self):
        self.released = True


def test_a_video_file_is_sampled_with_every_and_written_to_the_hud(near, tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")                                     # only has to exist: the capture is fake
    FakeCapture.instances, FakeWriter.made = [], []
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cv2, "VideoWriter", FakeWriter)
    jsonl = tmp_path / "near.jsonl"
    out = near.run("--source", str(clip), "--profile", "c270", "--no-send", "--fast", "--every", "2",
                   "--jsonl", str(jsonl), "--hud", str(tmp_path / "hud" / "clip.mp4"))
    cap = FakeCapture.instances[0]
    assert cap.released
    evs = [json.loads(ln) for ln in jsonl.read_text().splitlines()]
    assert [e["t"] for e in evs] == pytest.approx([1 / 30, 3 / 30, 5 / 30], abs=1e-3)   # frames 1, 3, 5 of 7
    assert evs[-1]["priority"] == 1                           # escalates after N_UP agreeing frames
    w = FakeWriter.made[0]
    assert w.fps == pytest.approx(15.0) and len(w.frames) == 3 and w.released
    assert (tmp_path / "hud").is_dir() and "3 frames" in out


def test_a_video_file_that_will_not_open_is_a_clean_exit(near, tmp_path, monkeypatch):
    clip = tmp_path / "broken.mp4"
    clip.write_bytes(b"")
    monkeypatch.setattr(cv2, "VideoCapture", lambda path, *a: FakeCapture(path, opened=False))
    with pytest.raises(SystemExit, match="cannot open"):
        near.run("--source", str(clip), "--no-send")


def test_max_seconds_stops_a_long_video(near, tmp_path, monkeypatch):
    clip = tmp_path / "long.mp4"
    clip.write_bytes(b"")
    monkeypatch.setattr(cv2, "VideoCapture", lambda path, *a: FakeCapture(path, n=10 ** 6))
    ticks = iter(range(10 ** 6))
    clock = types.SimpleNamespace(monotonic=lambda: float(next(ticks)), perf_counter=M.time.perf_counter,
                                  sleep=lambda s: pytest.fail("--fast never sleeps"))
    monkeypatch.setattr(M, "time", clock)                     # one "second" per look at the clock: no real waiting
    out = near.run("--source", str(clip), "--profile", "c270", "--no-send", "--fast", "--max-seconds", "3")
    assert "3 frames" in out and FakeCapture.instances[-1].i < 100


# ------------------------------------------------------------------ live sources, --show and the earpiece
class LiveCamera:
    """A kite.sources-like live camera: (False, None) now and then, frames otherwise, then Ctrl-C."""

    def __init__(self, name, frames=4, size=(320, 180)):
        self.name, self.left, self.size, self.reads, self.released = name, frames, size, 0, False

    def read(self):
        self.reads += 1
        if self.reads == 2:
            return False, None                                # a stale read: skipped, not a crash
        if self.left == 0:
            raise KeyboardInterrupt
        self.left -= 1
        return True, np.zeros((self.size[1], self.size[0], 3), np.uint8)

    def release(self):
        self.released = True


def test_show_draws_each_frame_with_its_tone_and_q_closes_the_window(near, monkeypatch):
    cam = LiveCamera("webcam:0 Logitech HD Webcam C270", frames=10)
    monkeypatch.setattr(kite.sources, "open_source", lambda spec: cam)
    near.window.keys = [-1, -1, ord("q")]
    out = near.run("--source", "webcam:Logitech", "--show", "--no-send")
    assert len(near.window.shown) == 3 and near.window.destroyed == 1 and cam.released
    title, shape = near.window.shown[0]
    assert "q to quit" in title
    assert shape[0] > 2 * 180 and shape[1] == 2 * 320         # boxes | depth, radar, then the tone banner
    assert "profile: USB name" in out and "3 frames" in out   # the C270 profile, picked by name, scaled


def test_esc_also_closes_the_window(near, monkeypatch):
    cam = LiveCamera("webcam:1 FaceTime HD Camera", frames=10)
    monkeypatch.setattr(kite.sources, "open_source", lambda spec: cam)
    near.window.keys = [27]
    near.run("--source", "webcam:1", "--show", "--no-send")
    assert len(near.window.shown) == 1 and near.window.destroyed == 1


def test_the_unoq_source_streams_to_the_earpiece_until_ctrl_c(near, monkeypatch):
    made = []

    class FakeUnoq(LiveCamera):
        def __init__(self, spec):
            super().__init__("unoq 127.0.0.1:8765 usb camera video0 (Logitech)", frames=3)
            made.append((spec, self))

    monkeypatch.setattr(unoq, "UnoqSource", FakeUnoq)
    monkeypatch.setattr(kite.sources, "open_source", lambda spec: pytest.fail("unoq must not go through kite.sources"))
    monkeypatch.setenv("KITE_EARPIECE_HOST", "10.0.0.7")
    out = near.run("--source", "unoq:127.0.0.1:8765", "--ear-port", "5099")
    (spec, cam), = made
    assert spec == "unoq:127.0.0.1:8765" and cam.released and cam.reads == 5
    assert near.clients == [("10.0.0.7", 5099, "near")]
    assert len(near.sent) >= 3 and all(m["type"] == "obstacles" for m in near.sent)
    assert [m["seq"] for m in near.sent] == list(range(1, len(near.sent) + 1))
    assert any(o.get("priority") == 1 for m in near.sent for o in m["objects"])
    assert "3 frames" in out and not near.window.shown


def test_a_camera_that_cannot_open_is_a_clean_exit_off_windows(near, monkeypatch):
    def boom(spec):
        raise LookupError("no camera matching 'Nope'")

    monkeypatch.setattr(kite.sources, "open_source", boom)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(SystemExit, match="cannot open webcam:Nope: no camera matching"):
        near.run("--source", "webcam:Nope", "--no-send")


# ------------------------------------------------------------------ open_live and the Windows fallback
def test_open_live_routes_unoq_specs_to_the_board(monkeypatch):
    class FakeUnoq:
        def __init__(self, spec):
            self.spec, self.name = spec, f"unoq board {spec}"

    monkeypatch.setattr(unoq, "UnoqSource", FakeUnoq)
    src, name = M.open_live("UNOQ")
    assert isinstance(src, FakeUnoq) and name == "unoq board UNOQ"


def test_open_live_falls_back_to_directshow_on_windows(monkeypatch):
    def boom(spec):
        raise RuntimeError("AVFoundation only")

    monkeypatch.setattr(kite.sources, "open_source", boom)
    monkeypatch.setattr(sys, "platform", "win32")
    seen = {}
    monkeypatch.setattr(M, "_open_windows", lambda query, why: seen.setdefault("args", (query, str(why))))
    M.open_live("webcam:Logitech")
    assert seen["args"] == ("Logitech", "AVFoundation only")


def _directshow(monkeypatch, names=None, opened=True):
    FakeCapture.instances = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda idx, api=None: FakeCapture(idx, opened=opened))
    if names is not None:
        graph = types.SimpleNamespace(FilterGraph=lambda: types.SimpleNamespace(get_input_devices=lambda: names))
        monkeypatch.setitem(sys.modules, "pygrabber", types.ModuleType("pygrabber"))
        monkeypatch.setitem(sys.modules, "pygrabber.dshow_graph", graph)
    else:
        monkeypatch.setitem(sys.modules, "pygrabber.dshow_graph", None)       # not installed


def test_directshow_picks_the_camera_by_name(monkeypatch):
    _directshow(monkeypatch, ["Integrated Webcam", "Logitech HD Webcam C270"])
    cap, label = M._open_windows("logitech", "no AVFoundation")
    assert cap.path == 1 and label == "webcam:1 Logitech HD Webcam C270"
    assert cap.props == {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720}


def test_directshow_without_pygrabber_asks_powershell_for_names(monkeypatch):
    import subprocess
    _directshow(monkeypatch)
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        assert kw["timeout"]                                   # never waits forever
        return types.SimpleNamespace(stdout="Logitech C270\r\n\r\n")

    monkeypatch.setattr(subprocess, "run", run)
    cap, label = M._open_windows("0", "why")
    assert calls and calls[0][0] == "powershell"
    assert cap.path == 0 and label == "webcam:0 Logitech C270"


def test_directshow_with_no_names_and_no_camera_says_what_it_saw(monkeypatch):
    import subprocess
    _directshow(monkeypatch, opened=False)

    def run(cmd, **kw):
        raise FileNotFoundError("powershell")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SystemExit, match="cameras seen: none"):
        M._open_windows("Logitech", "why")


# ------------------------------------------------------------------ device and detector choice (no torch needed)
def test_pick_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    def torch(cuda, mps):
        return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda),
                                     backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)))

    assert M.pick_device("mps") == "mps"
    monkeypatch.setitem(sys.modules, "torch", torch(True, True))
    assert M.pick_device(None) == "cuda"
    monkeypatch.setitem(sys.modules, "torch", torch(False, True))
    assert M.pick_device(None) == "mps"
    monkeypatch.setitem(sys.modules, "torch", torch(False, False))
    assert M.pick_device(None) == "cpu"
    monkeypatch.setitem(sys.modules, "torch", None)            # not installed
    assert M.pick_device(None) == "cpu"


def test_build_detector_wires_coco_and_the_open_vocabulary_model(monkeypatch):
    from perception import models, vocab
    made = []
    monkeypatch.setattr(models, "CocoDetector", lambda *a: made.append(("coco", a)) or "coco")
    monkeypatch.setattr(models, "OpenVocabDetector", lambda *a: made.append(("ov", a[1:])) or "ov")
    monkeypatch.setattr(models, "HybridDetector", lambda coco, ov: (coco, ov))
    monkeypatch.setattr(vocab, "load", lambda: {"pole": ["pole"]})
    assert M.build_detector("coco", "cpu") == ("coco", None)
    assert M.build_detector("hybrid", "mps") == ("coco", "ov")
    assert made == [("coco", ("cpu", 0.35, 640)), ("coco", ("mps", 0.35, 640)), ("ov", ("mps", 0.2, 640))]
