"""perception.near: intrinsics (FOV -> focal length, undistortion, profiles), ground geometry, the corridor."""

import json
import math

import numpy as np
import pytest

from perception.near import corridor, geometry, synth
from perception.near import intrinsics as I


# ------------------------------------------------------------------ FOV -> focal length
@pytest.mark.parametrize("key, f", [("c270", 1410.5), ("c310", 1271.8), ("c920", 1360.1)])
def test_spec_focal_lengths(key, f):
    p = I.BUILTIN[key]
    assert p.fx == pytest.approx(f, abs=0.1)
    assert p.fy == p.fx and p.cx == p.width / 2 and p.cy == p.height / 2
    assert I.dfov_from_focal(p.width, p.height, p.fx) == pytest.approx(p.dfov_deg)
    assert "calibrate for accuracy" in p.source and p.spec_url.startswith("https://www.logitech.com/")


def test_c310_horizontal_fov_matches_published_53_4_deg():
    # a retailer lists the C310's horizontal FOV as 53.4 deg next to Logitech's 60 deg diagonal: same lens
    assert I.BUILTIN["c310"].hfov_deg == pytest.approx(53.4, abs=0.1)


def test_focal_from_dfov_rejects_nonsense():
    with pytest.raises(ValueError):
        I.focal_from_dfov(1280, 720, 0.5)


def test_scaled_same_aspect_and_4_3_mode():
    p = I.BUILTIN["c920"].scaled(1280, 720)               # the C920's 720p mode is a downscale of 1080p
    assert p.fx == pytest.approx(1360.1 * 1280 / 1920, abs=0.1)
    q = I.BUILTIN["c270"].scaled(640, 480)                # 4:3: keep the horizontal FOV (assumed)
    assert q.hfov_deg == pytest.approx(I.BUILTIN["c270"].hfov_deg, abs=1e-6)
    assert q.cy == 240 and "assumed" in q.source


def test_resized_is_a_pure_resample():
    p = I.BUILTIN["c270"].resized(448, 252)
    assert p.fx == pytest.approx(1410.5 * 448 / 1280, abs=0.05) and p.fy == pytest.approx(1410.5 * 252 / 720, abs=0.05)


# ------------------------------------------------------------------ undistortion
def _lens():
    from dataclasses import replace
    return replace(I.BUILTIN["c270"], dist=[-0.21, 0.09, 0.001, -0.0015, 0.0])


def test_undistort_inverts_distort():
    p = _lens()
    pts = np.array([[5, 5], [640, 360], [1270, 710], [100, 600], [1000, 40]], float)
    back = p.undistort_points(p.distort_points(pts))
    assert np.abs(back - pts).max() < 0.05


def test_undistort_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    p = _lens()
    pts = np.array([[20, 30], [1260, 700], [640, 100], [300, 650]], float)
    ref = cv2.undistortPoints(pts.reshape(-1, 1, 2), p.K, np.array(p.dist), P=p.K).reshape(-1, 2)
    assert np.abs(p.undistort_points(pts) - ref).max() < 0.05


def test_no_distortion_is_identity_and_box_unchanged():
    p = I.BUILTIN["c270"]
    assert p.undistort_box(10, 20, 300, 400) == (10, 20, 300, 400)


def test_undistort_box_moves_corner_boxes_outward_for_barrel():
    p = _lens()
    x1, y1, x2, y2 = p.undistort_box(0, 0, 200, 150)
    assert x1 < 0 and y1 < 0                               # barrel pulls the corners in; undistorting pushes out


# ------------------------------------------------------------------ profiles
@pytest.mark.parametrize("name, key", [("Logitech HD Webcam C270", "c270"), ("HD Webcam C270", "c270"),
                                       ("HD Pro Webcam C920", "c920"), ("Logi C922 Pro Stream", "c920"),
                                       ("Logitech Webcam C310", "c310"), ("FaceTime HD Camera", None), (None, None)])
def test_match_profile(name, key):
    assert I.match_profile(name) == key


def test_resolve_auto_logitech_without_model_number_is_the_c270(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "PROFILE_DIR", tmp_path)
    p, why = I.resolve("auto", "Logitech USB Camera", (1280, 720))
    assert p.name == "Logitech C270" and "C270" in why


def test_resolve_prefers_a_calibrated_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "PROFILE_DIR", tmp_path)
    cal = I.CameraProfile("Logitech C270", 1280, 720, 1380.0, 1378.0, 650.0, 355.0, [-0.05, 0.1, 0, 0, 0],
                          "calibrated: charuco, 20 views")
    cal.save(I.calibrated_path("Logitech C270"))
    p, why = I.resolve("auto", "HD Webcam C270", (640, 360))
    assert p.calibrated and p.fx == pytest.approx(690.0) and "calibrated" in why


def test_resolve_json_and_errors(tmp_path):
    f = tmp_path / "cam.json"
    I.BUILTIN["c310"].save(f)
    assert json.loads(f.read_text())["hfov_deg"] == pytest.approx(53.4, abs=0.1)
    assert I.resolve(str(f))[0].name == "Logitech C310"
    with pytest.raises(ValueError):
        I.resolve("c9000")


def test_describe_is_useful():
    s = I.BUILTIN["c270"].describe()
    assert "1411" in s and "calibrate for accuracy" in s and "px tall" in s


# ------------------------------------------------------------------ ground geometry
@pytest.mark.parametrize("pitch", [0.0, 6.0, 15.0])
@pytest.mark.parametrize("F, L", [(3.0, 0.0), (6.5, -0.8), (12.0, 1.1)])
def test_ground_point_inverts_the_projection(pitch, F, L):
    p = I.BUILTIN["c920"]
    px, _ = synth.project(p, np.array([F]), np.array([L]), np.array([0.0]), 1.2, pitch)
    g = geometry.ground_point(p, px[0, 0], px[0, 1], 1.2, pitch)
    assert g[0] == pytest.approx(F, rel=1e-6) and g[1] == pytest.approx(L, abs=1e-6)


def test_ground_point_above_horizon_is_none():
    p = I.BUILTIN["c270"]
    assert geometry.ground_point(p, 640, geometry.horizon_row(p, 5.0) - 3, 1.2, 5.0) is None


def test_points_from_depth_heights():
    p = I.BUILTIN["c270"].resized(64, 36)
    z = geometry.plane_depth(p, np.arange(36), 1.2, 10.0)
    zz = np.repeat(np.where(np.isfinite(z), z, 50.0)[:, None], 64, 1)
    _, _, h = geometry.points_from_depth(zz, p, 1.2, 10.0)
    ground = np.isfinite(z)
    assert np.abs(h[ground]).max() < 1e-9


# ------------------------------------------------------------------ the corridor
def _box(p, F, L, W, pitch=5.0):
    b = synth.box_of(p, synth.Obj("pole", F, L, 1.0, W), 1.2, pitch)
    return b


@pytest.mark.parametrize("L, W, inside, edge", [(0.0, 0.3, True, False), (0.55, 0.15, True, False),
                                               (0.75, 0.2, False, True), (1.4, 0.3, False, False),
                                               (-0.7, 0.2, False, True)])
def test_corridor_metric(L, W, inside, edge):
    p = I.BUILTIN["c920"]
    b = _box(p, 4.0, L, W)
    hit = corridor.test(p, b[0], b[2], b[3], 4.0, 5.0, v_ego=1.0)
    assert (hit.inside, hit.edge) == (inside, edge)
    assert hit.lateral == pytest.approx(L, abs=0.02)


def test_corridor_widens_with_speed():
    assert corridor.half_width(1.0) == 0.5 and corridor.half_width(2.0) == pytest.approx(0.65)


def test_corridor_angular_fallback_is_about_5_7_deg():
    p = I.BUILTIN["c270"]
    cone = math.degrees(math.atan(0.5 / 5.0))
    assert cone == pytest.approx(5.71, abs=0.01)
    u_in = p.cx + p.fx * math.tan(math.radians(cone - 0.5))
    u_out = p.cx + p.fx * math.tan(math.radians(cone + 3.0))
    assert corridor.test_angular(p, u_in, u_in + 5).inside
    assert not corridor.test_angular(p, u_out, u_out + 5).inside
