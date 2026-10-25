"""run/app.py --model (opt in): a trained follow model from kite/fly/data/behind flies through TunedFollowMission, and
without it every --yaw option builds exactly the follow it built before (FollowMission, BehindCfg, the same steering).
The pilot's _connect runs for real against fakes: no drone, no camera, no YOLO, no files written."""

import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "run"))

import app as appmod  # noqa: E402
import pilot as pilotmod  # noqa: E402

from kite.follow_behind import FLY_PARAMS, BehindCfg, FollowMission  # noqa: E402
from kite.follow_tuned import MODEL_DIR, TunedCfg, TunedFollowMission, app_follow_cfg, resolve_model  # noqa: E402
from kite.types import Telemetry  # noqa: E402


class FakeTello:
    flying = False

    def connect(self):
        pass

    def telemetry(self):
        return Telemetry(t=0.0, battery_pct=90.0, height_m=0.0)

    def frame_source(self):
        return None

    def close(self):
        pass

    def busy(self):
        return False


class FakeDetector:
    device = "cpu"

    def __init__(self, weights=None):
        pass

    def __call__(self, image):
        return []


@pytest.fixture
def connect(monkeypatch):
    import kite.drone
    import kite.follow_behind

    monkeypatch.setattr(kite.follow_behind, "pose_available", lambda weights=None: None)
    monkeypatch.setattr(kite.follow_behind, "PoseDetector", FakeDetector)
    monkeypatch.setattr(kite.drone, "make_tello", lambda cfg, dry_run=False: FakeTello())
    capture = types.ModuleType("capture")

    class SessionCapture:
        def __init__(self, root):
            self.dir = root

        def start(self):
            return self

    capture.SessionCapture = SessionCapture
    monkeypatch.setitem(sys.modules, "capture", capture)

    def run(argv):
        a = appmod.build_parser().parse_args(["--dry-run", *argv])
        a.height = a.height or min(round(a.walker_height + 0.2, 2), 2.2)
        lines = []
        p = pilotmod.Pilot(a, lambda text, kind="info": lines.append((kind, text)))
        ok = p._connect()
        return ok, p, lines

    return run


def test_model_is_off_by_default_and_parses(connect):
    assert appmod.build_parser().parse_args([]).model is None
    assert appmod.build_parser().parse_args(["--model", "v3b-joint"]).model == "v3b-joint"


@pytest.mark.parametrize("yaw, steering", [("fly", "fly"), ("law", "pid"), ("retrained", "pid"), ("pd", "pid")])
def test_without_model_every_yaw_builds_the_follow_it_built_before(connect, yaw, steering):
    ok, p, _ = connect(["--yaw", yaw])
    assert ok and type(p.follow) is FollowMission and type(p.follow.cfg) is BehindCfg
    cfg = p.follow.cfg
    assert cfg.steering == steering and cfg.fly_params_path == str(FLY_PARAMS)
    expected = BehindCfg(distance_m=1.8, altitude_m=1.9, person_height_m=1.7, steering=steering,
                         min_altitude_m=cfg.min_altitude_m, max_altitude_m=cfg.max_altitude_m,
                         video_lag_s=cfg.video_lag_s)
    assert asdict(cfg) == asdict(expected)
    assert (p.follow.yaw_source is not None) == (yaw in ("retrained", "pd"))
    assert p.snap.yaw == yaw


@pytest.mark.parametrize("name", ["v3b-joint", "v3b-follow"])
def test_a_model_flies_its_own_settings_through_the_tuned_mission(connect, name):
    ok, p, lines = connect(["--model", name, "--yaw", "law"])
    assert ok and type(p.follow) is TunedFollowMission and type(p.follow.cfg) is TunedCfg
    cfg = p.follow.cfg
    assert cfg.model_name == name and cfg.steering == "fly"
    assert cfg.fly_params_path == str((MODEL_DIR / f"{name}.json").resolve())  # its own brain, not smooth2's file
    assert (cfg.distance_m, cfg.altitude_m, cfg.person_height_m) == (1.8, 1.9, 1.7)  # the app's geometry
    assert cfg.fwd_cap_fast > 35 and cfg.max_rc_forward <= 35  # faster only when the person-speed gate opens
    assert p.follow.yaw_source is None and p.snap.yaw == f"model {name}"
    said = " ".join(t for _, t in lines)
    assert f"follow model {name}" in said and "--yaw law is not used" in said


def test_a_model_keeps_the_apps_distance_height_and_walker(connect):
    ok, p, _ = connect(["--model", "v3b-joint", "--behind", "2.0", "--walker-height", "1.62", "--height", "1.85"])
    cfg = p.follow.cfg
    assert ok and (cfg.distance_m, cfg.altitude_m, cfg.person_height_m) == (2.0, 1.85, 1.62)


def test_an_unknown_model_refuses_to_start_and_lists_the_models(connect):
    ok, p, lines = connect(["--model", "nope"])
    assert ok is False and p.follow is None and p.drone is None
    kind, text = lines[-1]
    assert kind == "error" and "v3b-joint" in text and "v3b-follow" in text


def test_resolve_model_by_name_file_name_or_path(tmp_path):
    p = MODEL_DIR / "v3b-joint.json"
    assert resolve_model("v3b-joint") == p and resolve_model("v3b-joint.json") == p and resolve_model(p) == p
    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path / "missing.json")
    with pytest.raises(ValueError):  # the app's settings are checked: the altitude must clear the head
        app_follow_cfg("v3b-joint", altitude_m=1.8, person_height_m=1.7)
