"""depth_zoo: the interface, floor anchoring, motion stereo, fusion, temporal smoothing and the benchmark's
scoring rules. No model and no download: fake backends and the numpy ray caster stand in."""

import math
from pathlib import Path

import numpy as np
import pytest

from perception import depth_zoo as dz
from perception.bench_depth import metrics as M
from perception.bench_depth import synth

K = dz.Intrinsics.from_fov(640, 480, 60.0)


def floor_depth(K, pitch=8.0, roll=0.0, cam_h=1.2, far=80.0):
    """Exact z-depth of a flat floor seen from cam_h, pitched down `pitch` degrees; 0 above the horizon."""
    nr = K.rays() @ dz.down_vector(pitch, roll)
    z = np.where(nr > cam_h / far, cam_h / np.maximum(nr, 1e-9), 0.0)
    return z.astype(np.float32)


class Const(dz.DepthBackend):
    """A fake model: constant depth and confidence."""

    def __init__(self, depth=3.0, conf=0.5, name="const", metric=True, device="cpu"):
        self.d, self.c, self.name, self.metric = depth, conf, name, metric

    def estimate(self, frame_bgr, intrinsics=None):
        h, w = frame_bgr.shape[:2]
        return dz.DepthResult(np.full((h, w), self.d, np.float32), np.full((h, w), self.c, np.float32), 1.0)


class Plane(dz.DepthBackend):
    """A fake metric model that sees the floor exactly, at a wrong scale."""
    name, metric = "plane", True

    def __init__(self, scale=0.6, pitch=8.0, device="cpu"):
        self.scale, self.pitch = scale, pitch

    def estimate(self, frame_bgr, intrinsics=None):
        Kf = dz.default_intrinsics(frame_bgr, intrinsics)
        z = floor_depth(Kf, self.pitch)
        z = np.where(z > 0, z, 80.0) * self.scale
        return dz.DepthResult(z.astype(np.float32), np.full(z.shape, 0.6, np.float32), 2.0)


@pytest.fixture
def fake_registry():
    saved = dict(dz.REGISTRY)
    dz.register("plane", Plane, True, "test")
    dz.register("const", Const, True, "test")
    yield
    dz.REGISTRY.clear()
    dz.REGISTRY.update(saved)


# ------------------------------------------------------------------ interface
def test_intrinsics_from_fov_matches_a_c270():
    assert K.fx == pytest.approx(692.8, abs=0.1) and K.cx == 320 and K.cy == 240
    half = K.scaled(320, 240)
    assert half.fx == pytest.approx(K.fx / 2) and half.width == 320
    r = K.rays()
    assert r.shape == (480, 640, 3) and np.all(r[..., 2] == 1)


def test_registry_lists_every_promised_backend_with_a_licence():
    names = {d["name"] for d in dz.describe()}
    for want in ("da2-metric-outdoor-s", "da2-metric-indoor-s", "da2-metric-outdoor-b", "da2-metric-indoor-b",
                 "da2-relative-s", "depth-pro", "unidepth-v2-s", "metric3d-v2-s", "vda-metric-s", "motion-stereo",
                 "flat-ground"):
        assert want in names
    assert all(d["licence"] for d in dz.describe())


def test_get_composes_modifiers_and_returns_the_contract(fake_registry):
    be = dz.get("plane+anchor+temporal", cam_h=1.2)
    assert be.name == "plane+anchor+temporal" and be.metric
    img = np.zeros((480, 640, 3), np.uint8)
    r = be.estimate(img, K)
    assert isinstance(r, dz.DepthResult)
    assert r.depth_m.shape == (480, 640) and r.depth_m.dtype == np.float32
    assert r.confidence.shape == (480, 640) and r.confidence.dtype == np.float32
    assert 0 <= r.confidence.min() and r.confidence.max() <= 1 and r.latency_ms >= 0
    with pytest.raises(KeyError):
        dz.get("no-such-model")
    with pytest.raises(KeyError):
        dz.get("plane+sharpen")


def test_unavailable_backends_say_why_instead_of_crashing(monkeypatch, tmp_path):
    monkeypatch.setenv("KITE_DEPTH_THIRD_PARTY", str(tmp_path))
    with pytest.raises(dz.BackendUnavailable, match="not found|torch"):
        dz.get("metric3d-v2-s")


def test_anchored_wrapper_rescales_a_wrong_scale_model_to_the_true_floor(fake_registry):
    r = dz.get("plane+anchor", cam_h=1.2).estimate(np.zeros((480, 640, 3), np.uint8), K)
    truth = floor_depth(K, 8.0)
    m = truth > 0
    assert np.median(np.abs(r.depth_m[m] - truth[m]) / truth[m]) < 0.01
    assert r.meta["anchor"]["ok"] and r.meta["anchor"]["pitch"] == pytest.approx(8.0, abs=0.3)


def test_pipeline_adapter_returns_metric_depth_at_the_pipelines_size(fake_registry):
    d = dz.PipelineDepth(dz.get("plane+anchor"))(np.zeros((540, 960, 3), np.uint8))
    assert d.shape == (252, 448)


# ------------------------------------------------------------------ floor anchoring
@pytest.mark.parametrize("pitch,roll,scale", [(0.0, 0.0, 1.0), (8.0, 0.0, 0.5), (15.0, 3.0, 1.7), (25.0, -5.0, 0.8)])
def test_anchor_recovers_scale_and_pitch_of_a_metric_depth_map(pitch, roll, scale):
    z = floor_depth(K, pitch, roll) * scale
    a = dz.anchor_to_ground(z, K, cam_h=1.2)
    assert a.scale == pytest.approx(1 / scale, rel=0.01)
    assert a.shift == 0.0 and a.pitch == pytest.approx(pitch, abs=0.2) and a.inlier_ratio > 0.9
    fit = dz.fit_ground_plane(z, K, 1.2)
    assert fit.roll == pytest.approx(roll, abs=0.3)


@pytest.mark.parametrize("pitch,a,b", [(6.0, 2.5, 0.03), (12.0, 0.7, -0.01), (20.0, 4.0, 0.0)])
def test_anchor_recovers_scale_and_shift_of_affine_disparity_with_a_pitch_prior(pitch, a, b):
    z = floor_depth(K, pitch)
    q = np.where(z > 0, (1 / np.maximum(z, 1e-6) - b) / a, 0).astype(np.float32)
    out = dz.anchor_to_ground(q, K, cam_h=1.2, kind="disparity", pitch_prior=pitch)
    assert out.scale == pytest.approx(a, rel=0.01) and out.shift == pytest.approx(b, abs=2e-4)
    m = z > 0
    assert np.median(np.abs(out.depth_m[m] - z[m]) / z[m]) < 0.01


def test_without_a_pitch_prior_disparity_anchoring_assumes_zero_shift_and_finds_the_pitch():
    z = floor_depth(K, 10.0)
    q = np.where(z > 0, 1 / np.maximum(z, 1e-6) / 3.0, 0).astype(np.float32)
    out = dz.anchor_to_ground(q, K, cam_h=1.2, kind="disparity")
    assert out.scale == pytest.approx(3.0, rel=0.01) and out.shift == 0.0
    assert out.pitch == pytest.approx(10.0, abs=0.2)


def test_the_floor_alone_cannot_separate_shift_from_pitch():
    """Two (scale, shift, pitch) triples explain the same floor disparity: the reason for the pitch prior."""
    q = np.where(floor_depth(K, 10.0) > 0, 1 / np.maximum(floor_depth(K, 10.0), 1e-6), 0).astype(np.float32)
    f1 = dz.fit_ground_disparity(q, K, 1.2, pitch_prior=10.0)
    f2 = dz.fit_ground_disparity(q, K, 1.2, pitch_prior=14.0)
    assert f1.ok and f2.ok and abs(f1.shift - f2.shift) > 0.01
    assert f1.inlier_ratio == pytest.approx(f2.inlier_ratio, abs=0.02)


def test_a_wall_is_not_a_floor():
    z = np.full((480, 640), 3.0, np.float32)          # a fronto-parallel wall 3 m away fills the frame
    a = dz.anchor_to_ground(z, K, 1.2)
    assert a.scale == 1.0 and math.isnan(a.pitch) and a.inlier_ratio == 0.0
    assert not dz.fit_ground_plane(z, K, 1.2).ok


def test_anchor_ignores_an_obstacle_standing_on_the_floor():
    z = floor_depth(K, 8.0)
    z[200:420, 260:380] = 2.0                         # a bin 2 m ahead
    a = dz.anchor_to_ground(z * 0.7, K, 1.2)
    assert a.scale == pytest.approx(1 / 0.7, rel=0.02) and a.pitch == pytest.approx(8.0, abs=0.4)


def test_flat_ground_backend_matches_the_floor_geometry():
    r = dz.FlatGround(cam_h=1.2, pitch=8.0).estimate(np.zeros((480, 640, 3), np.uint8), K)
    truth = floor_depth(K, 8.0)
    m = truth > 0
    assert np.allclose(r.depth_m[m], truth[m], rtol=1e-4)


# ------------------------------------------------------------------ motion stereo on a rendered walk
@pytest.fixture(scope="module")
def walk():
    Ks = dz.Intrinsics.from_fov(320, 240, 60.0)
    sc = synth.scene("sidewalk")
    frames, depth = [], []
    for p in synth.path("sidewalk", 40)[::1]:
        img, d, _ = synth.render(sc, p, Ks, seed=3)
        frames.append(img)
        depth.append(d)
    return Ks, frames, depth


def test_motion_stereo_recovers_metric_depth_while_walking(walk):
    Ks, frames, depth = walk
    ms = dz.MotionStereo(cam_h=1.2, work_width=320, pose_width=320)
    errs, cover = [], []
    for k, f in enumerate(frames):
        r = ms.estimate(f, Ks)
        if k >= 25:
            gt = depth[k]
            c = (r.confidence > 0.5) & (gt > 0) & (gt < 30)
            cover.append(c.mean())
            errs.append(np.median(np.abs(r.depth_m[c] - gt[c]) / gt[c]))
    assert np.mean(cover) > 0.1                       # it measures a real share of the frame
    assert np.mean(errs) < 0.1                        # within 10 % where it says it is confident
    assert not r.meta["still"] and r.meta["gap"] >= 1


def test_motion_stereo_says_it_knows_nothing_when_standing_still():
    Ks = dz.Intrinsics.from_fov(320, 240, 60.0)
    sc = synth.scene("standstill")
    ms = dz.MotionStereo(cam_h=1.2, work_width=320, pose_width=320)
    confs = []
    for p in synth.path("standstill", 12):
        img, _, _ = synth.render(sc, p, Ks, seed=3)
        confs.append(float(ms.estimate(img, Ks).confidence.mean()))
    assert max(confs) < 0.05


# ------------------------------------------------------------------ fusion and temporal smoothing
class Fixed(dz.DepthBackend):
    def __init__(self, depth, conf, sigma=None):
        self.d, self.c, self.s = depth, conf, sigma
        self.name, self.metric = "fixed", True

    def estimate(self, frame_bgr, intrinsics=None):
        h, w = frame_bgr.shape[:2]
        meta = {} if self.s is None else {"sigma_rel": np.full((h, w), self.s, np.float32)}
        return dz.DepthResult(np.full((h, w), self.d, np.float32), np.full((h, w), self.c, np.float32), 1.0, meta)


@pytest.mark.parametrize("ms_sigma,expect", [(0.01, "motion"), (1.0, "dense"), (np.inf, "dense")])
def test_fusion_follows_the_more_confident_source(ms_sigma, expect):
    img = np.zeros((48, 64, 3), np.uint8)
    f = dz.Fused(Fixed(2.0, 0.5), Fixed(4.0, dz.sigma_to_conf(np.float32(ms_sigma)), ms_sigma), scale_align=False)
    d = float(f.estimate(img, dz.Intrinsics.from_fov(64, 48)).depth_m.mean())
    assert (abs(d - 4.0) < 0.15) if expect == "motion" else (abs(d - 2.0) < 0.15)


def test_fusion_with_equal_confidence_lands_between_and_is_more_confident():
    img = np.zeros((48, 64, 3), np.uint8)
    sigma = 0.12 * float(dz.conf_to_sigma(np.float32(0.5), 1.0))
    f = dz.Fused(Fixed(2.0, 0.5), Fixed(4.0, 0.5, sigma), scale_align=False)
    r = f.estimate(img, dz.Intrinsics.from_fov(64, 48))
    assert float(r.depth_m.mean()) == pytest.approx(math.sqrt(8.0), rel=0.02)     # mean in log depth
    assert float(r.confidence.mean()) > dz.sigma_to_conf(np.float32(sigma))


def test_fusion_corrects_the_dense_scale_against_confident_motion_stereo():
    img = np.zeros((48, 64, 3), np.uint8)
    f = dz.Fused(Fixed(2.0, 0.6), Fixed(3.0, 0.9, 0.01))
    for _ in range(30):
        f.estimate(img, dz.Intrinsics.from_fov(64, 48))
    assert f.corr == pytest.approx(1.5, rel=0.02)


def test_temporal_smoothing_keeps_a_static_scene_and_damps_flicker():
    rng = np.random.default_rng(0)
    img = (rng.random((120, 160, 3)) * 255).astype(np.uint8)

    class Flicker(dz.DepthBackend):
        name, metric, i = "flicker", True, 0

        def estimate(self, frame_bgr, intrinsics=None):
            self.i += 1
            d = 3.0 * (1.1 if self.i % 2 else 0.9)
            return dz.DepthResult(np.full((120, 160), d, np.float32), np.full((120, 160), 0.5, np.float32), 1.0)

    t = dz.Temporal(Flicker(), alpha=0.4, work_width=160)
    out = [float(t.estimate(img).depth_m.mean()) for _ in range(12)]
    raw_swing = 3.3 - 2.7
    assert max(out[4:]) - min(out[4:]) < 0.5 * raw_swing
    assert np.mean(out[4:]) == pytest.approx(3.0, rel=0.05)


# ------------------------------------------------------------------ benchmark rules
def test_scrollsafe_rule_bands_by_box_height_in_the_middle_half():
    W, H = 640, 480
    box = lambda x1, x2, h: {"x1": x1, "x2": x2, "y1": 400 - h, "y2": 400}   # noqa: E731
    assert M.scrollsafe_band([], W, H) == "clear"
    assert M.scrollsafe_band([box(300, 340, 0.45 * H)], W, H) == "P1"       # > 40 % of the frame: danger
    assert M.scrollsafe_band([box(300, 340, 0.30 * H)], W, H) == "P2"       # > 20 %: warning
    assert M.scrollsafe_band([box(300, 340, 0.10 * H)], W, H) == "P3"       # small but in the corridor
    assert M.scrollsafe_band([box(0, 100, 0.9 * H)], W, H) == "clear"       # tall, but its centre is at 8 %
    assert M.scrollsafe_band([box(0, 100, 0.9 * H), box(300, 340, 0.3 * H)], W, H) == "P2"


def test_priority_bands_and_scores():
    assert [M.band(d) for d in (None, 0.5, 1.99, 2.0, 4.9, 5.0, 10.0, 10.1)] == \
        ["clear", "P1", "P1", "P2", "P2", "P3", "P3", "clear"]
    s = M.priority_scores(["P1", "P2", "P1", "clear"], ["P1", "P1", "P3", "clear"])
    assert s["acc"] == 0.5 and s["p1_recall"] == 0.5 and s["false_p1"] == 0.5


def test_nearest_obstacle_finds_a_box_in_the_corridor_and_ignores_one_beside_it():
    sc = synth.Scene(ground_y=0.0, sky=True, objects=[synth.Box((-0.2, 0.0, 3.0), (0.2, 0.9, 3.4), (0.3, 0.3, 0.3))])
    pose = synth.Pose(np.array([0.0, 1.2, 0.0]), 0.0, 8.0, 0.0)
    _, d, _ = synth.render(sc, pose, K)
    near = M.nearest_obstacle(d, K, pose.down(), 1.2)
    assert near == pytest.approx(3.0, abs=0.1) and M.band(near) == "P2"
    sc.objects[0] = synth.Box((1.0, 0.0, 3.0), (1.4, 0.9, 3.4), (0.3, 0.3, 0.3))
    _, d, _ = synth.render(sc, pose, K)
    assert M.nearest_obstacle(d, K, pose.down(), 1.2) is None


def test_flat_ground_box_rule_ranges_a_box_by_its_bottom_row():
    pitch = 8.0
    n = dz.down_vector(pitch)
    # the pixel where the floor is 4 m ahead, straight on
    P = np.array([0.0, 0.0, 4.0])
    P = P + (1.2 - P @ n) * n            # point on the floor below (4 m along the optical axis direction)
    v = K.fy * P[1] / P[2] + K.cy
    d = M.flat_ground_box_distance([{"x1": 300, "x2": 340, "y1": v - 80, "y2": v}], K, 1.2, pitch)
    f = np.array([0.0, 0.0, 1.0]) - n[2] * n
    assert d == pytest.approx(float(P @ (f / np.linalg.norm(f))), rel=0.01)


def test_render_gives_exact_depth_for_the_floor():
    pose = synth.Pose(np.array([0.0, 1.2, 0.0]), 0.0, 8.0, 0.0)
    img, d, ids = synth.render(synth.Scene(), pose, K)
    truth = floor_depth(K, 8.0)
    m = (ids == -1) & (truth > 0) & (d > 0)
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    assert np.allclose(d[m], truth[m], rtol=1e-4)
    assert np.allclose(pose.down(), dz.down_vector(8.0), atol=1e-9)


@pytest.mark.slow
def test_real_depth_anything_small_anchors_on_a_rendered_sidewalk():
    """Needs weights/hf (python -m perception.bench_depth --setup) and torch + transformers."""
    pose = synth.path("sidewalk", 1)[0]
    img, d, _ = synth.render(synth.scene("sidewalk"), pose, K, seed=3)
    r = dz.get("da2-metric-outdoor-s+anchor").estimate(img, K)
    assert r.meta["anchor"]["ok"]
    assert r.meta["anchor"]["pitch"] == pytest.approx(pose.pitch, abs=4.0)


# ------------------------------------------------------------------ model-free plumbing (no torch, no weights)
def test_paths_and_the_download_switch_follow_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("KITE_HF_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("KITE_DEPTH_THIRD_PARTY", str(tmp_path / "tp"))
    monkeypatch.delenv("KITE_DEPTH_DOWNLOAD", raising=False)
    assert dz.hf_cache() == str(tmp_path / "hf") and dz.third_party() == tmp_path / "tp"
    assert not dz._may_download()
    monkeypatch.setenv("KITE_DEPTH_DOWNLOAD", "1")
    assert dz._may_download()
    monkeypatch.delenv("KITE_HF_CACHE")
    assert dz.hf_cache().endswith(str(Path("weights") / "hf"))


def test_the_base_backend_is_abstract_and_its_hooks_are_no_ops():
    be = dz.DepthBackend()
    with pytest.raises(NotImplementedError):
        be.estimate(np.zeros((4, 4, 3), np.uint8))
    assert be.reset() is None and be.close() is None


def test_default_intrinsics_fills_in_or_rescales_to_the_frame():
    f = np.zeros((240, 320, 3), np.uint8)
    assert dz.default_intrinsics(f, None) == dz.Intrinsics.from_fov(320, 240, 60.0)
    assert dz.default_intrinsics(f, K) == K.scaled(320, 240)
    Ks = K.scaled(320, 240)
    assert dz.default_intrinsics(f, Ks) is Ks
    assert np.allclose(K.matrix(), [[K.fx, 0, 320], [0, K.fy, 240], [0, 0, 1]])
    assert not K.rays().flags.writeable                # the cached rays are shared: read-only


def test_fit14_keeps_both_sides_multiples_of_14():
    assert dz._fit14(640, 480, 448) == (448, 336)
    w, h = dz._fit14(1920, 1080, 518)
    assert w % 14 == 0 and h % 14 == 0 and w == 518
    assert dz._fit14(10, 1, 14) == (14, 14)           # never below one patch


def test_get_builds_fused_and_passes_cam_h_to_the_geometry_backends(fake_registry):
    f = dz.get("fused:plane")
    assert isinstance(f, dz.Fused) and isinstance(f.motion, dz.MotionStereo)
    assert f.dense.name == "plane+anchor"              # the dense half is always anchored
    assert dz.get("fused:plane+anchor").dense.name == "plane+anchor"
    fg = dz.get("flat-ground", cam_h=1.5, pitch=4.0)
    assert isinstance(fg, dz.FlatGround) and fg.cam_h == 1.5 and fg.pitch == 4.0
    assert dz.get("motion-stereo", cam_h=0.9).cam_h == 0.9
    anch = dz.get("const+anchor", cam_h=1.4, pitch_prior=6.0)
    assert anch.cam_h == 1.4 and anch.pitch_prior == 6.0 and anch.base.name == "const"


def test_available_reports_ok_or_the_first_line_of_why(fake_registry):
    def broken(device="cpu"):
        raise dz.BackendUnavailable("weights missing\nrun the setup")

    def no_pkg(device="cpu"):
        raise ImportError("no module named magic")

    dz.register("broken", broken, True, "test")
    dz.register("nopkg", no_pkg, True, "test")
    got = dz.available(names=["const", "broken", "nopkg"])
    assert got == {"const": "ok", "broken": "weights missing", "nopkg": "no module named magic"}


def test_edge_confidence_is_capped_and_drops_on_depth_edges():
    flat = dz.edge_confidence(np.full((40, 40), 3.0, np.float32))
    assert flat.dtype == np.float32 and np.allclose(flat, 0.8)
    step = np.full((40, 40), 2.0, np.float32)
    step[:, 20:] = 10.0
    c = dz.edge_confidence(step)
    assert c[20, 19] < 0.05 and c[20, 20] < 0.05      # both sides of the edge are "flying pixels"
    assert c[20, 2] == pytest.approx(0.8) and c[20, 37] == pytest.approx(0.8)


def test_sigma_and_confidence_convert_both_ways():
    s = np.array([0.02, 0.05, 0.2], np.float32)
    c = dz.sigma_to_conf(s)
    assert c == pytest.approx([0.862, 0.5, 0.0588], abs=1e-3)       # the docstring's numbers
    assert dz.conf_to_sigma(c) == pytest.approx(s, rel=1e-3)
    assert dz.sigma_to_conf(np.array([np.nan]))[0] == 0.0
    assert dz.sigma_to_conf(np.array([np.inf]))[0] == 0.0
    assert np.isfinite(dz.conf_to_sigma(np.array([0.0, 1.0]))).all()  # clipped: no division by zero


def test_down_vector_and_angles_round_trip():
    for p, r in [(0.0, 0.0), (8.0, 0.0), (20.0, -4.0), (-10.0, 12.0)]:
        n = dz.down_vector(p, r)
        assert np.linalg.norm(n) == pytest.approx(1.0)
        assert dz.angles_of(n) == pytest.approx((p, r), abs=1e-9)
    assert dz.angles_of(np.array([0.0, 0.0, 1.5]))[0] == pytest.approx(90.0)   # clamped asin


def test_fit_ground_plane_says_why_it_failed():
    z = floor_depth(K, 8.0)
    none = dz.fit_ground_plane(z, K, mask=np.zeros(z.shape, bool))
    assert not none.ok and "candidate" in none.reason and math.isnan(none.pitch) and none.scale == 1.0
    ceiling = np.flipud(floor_depth(K, -8.0)).copy()    # a plane above the camera: its normal points up
    assert not dz.fit_ground_plane(ceiling, K).ok
    half = floor_depth(dz.Intrinsics.from_fov(320, 240, 60.0), 8.0) * 0.5
    fit = dz.fit_ground_plane(half, K, 1.2)             # full-size intrinsics on a half-size map are rescaled
    assert fit.ok and fit.scale == pytest.approx(2.0, rel=0.01) and fit.pitch == pytest.approx(8.0, abs=0.3)
    sparse = dz.fit_ground_plane(z, K, min_inliers=10 ** 6)
    assert not sparse.ok and "candidate" in sparse.reason
    thin = dz.fit_ground_plane(z, K, min_ratio=1.01)
    assert not thin.ok and "too few floor pixels" in thin.reason and thin.inliers.any()
    steep = dz.fit_ground_plane(z, K, pitch_range=(50.0, 60.0))     # a floor, but not at a plausible pitch
    assert not steep.ok and steep.reason == "no floor-like plane among the samples"


def test_fit_ground_disparity_rejects_a_ramp_the_wrong_way_and_an_empty_map():
    z = floor_depth(K, 10.0)
    q = np.where(z > 0, 1 / np.maximum(z, 1e-6), 0).astype(np.float32)
    assert "candidate" in dz.fit_ground_disparity(np.zeros_like(q), K).reason
    rows = np.linspace(2.0, 0.1, q.shape[0], dtype=np.float32)[:, None]
    falling = np.repeat(rows, q.shape[1], axis=1)       # nearer at the top: not a floor seen from above
    assert not dz.fit_ground_disparity(falling, K).ok
    half = q[::2, ::2].copy()
    fit = dz.fit_ground_disparity(half, K, 1.2, pitch_prior=10.0)
    assert fit.ok and fit.scale == pytest.approx(1.0, rel=0.02)
    left = np.zeros(q.shape, bool)
    left[:, :320] = True
    masked = dz.fit_ground_disparity(q, K, 1.2, pitch_prior=10.0, mask=left)
    assert masked.ok and not masked.inliers[:, 320:].any()            # only the masked-in pixels vote
    thin = dz.fit_ground_disparity(q, K, 1.2, min_ratio=1.01)
    assert not thin.ok and "too few floor pixels" in thin.reason


def test_disparity_to_depth_inverts_and_sends_the_horizon_far():
    q = np.array([[0.0, 1e-4, 0.5, 2.0]], np.float32)
    z = dz.disparity_to_depth(q, 1.0, 0.0)
    assert z.dtype == np.float32
    assert z[0, 0] == dz.MAX_DEPTH and z[0, 1] == dz.MAX_DEPTH        # at or past the horizon: far, not inf
    assert z[0, 2] == pytest.approx(2.0) and z[0, 3] == pytest.approx(0.5)
    assert dz.disparity_to_depth(q, 2.0, 0.5)[0, 2] == pytest.approx(1 / 1.5)


def test_failed_disparity_anchoring_falls_back_to_one_over_q():
    q = np.full((480, 640), 0.25, np.float32)           # a wall in disparity: no ramp
    a = dz.anchor_to_ground(q, K, 1.2, kind="disparity")
    assert a.scale == 1.0 and a.shift == 0.0 and math.isnan(a.pitch) and a.inlier_ratio == 0.0
    assert np.allclose(a.depth_m, 4.0)


class Script(dz.DepthBackend):
    """A fake model that plays a list of depth maps (and optional disparities) in order."""

    def __init__(self, maps, metric=True, disp=None):
        self.maps, self.disp, self.metric, self.name, self.i = maps, disp, metric, "script", 0
        self.resets = self.closes = 0

    def estimate(self, frame_bgr, intrinsics=None):
        z = self.maps[min(self.i, len(self.maps) - 1)]
        meta = {} if self.disp is None else {"disparity": self.disp[min(self.i, len(self.disp) - 1)], "x": 1}
        self.i += 1
        return dz.DepthResult(z.astype(np.float32), np.full(z.shape, 0.6, np.float32), 1.0, meta)

    def reset(self):
        self.resets += 1

    def close(self):
        self.closes += 1


def _floor_or_far(pitch=8.0):
    z = floor_depth(K, pitch)
    return np.where(z > 0, z, 80.0).astype(np.float32)


def test_anchored_holds_the_last_scale_through_a_short_miss_then_lets_go():
    img = np.zeros((480, 640, 3), np.uint8)
    wall = np.full((480, 640), 3.0, np.float32)
    a = dz.Anchored(Script([_floor_or_far() * 0.5, wall, wall, wall]), cam_h=1.2, hold=2)
    assert a.name == "script+anchor" and a.metric
    r0 = a.estimate(img, K)
    assert r0.meta["anchor"]["ok"] and r0.meta["anchor"]["scale"] == pytest.approx(2.0, rel=0.01)
    r1 = a.estimate(img, K)                              # the wall: no floor, the last scale is held
    assert not r1.meta["anchor"]["ok"] and r1.meta["anchor"]["scale"] == pytest.approx(2.0, rel=0.01)
    assert np.allclose(r1.depth_m, 3.0 * r1.meta["anchor"]["scale"], rtol=1e-4)
    assert float(r1.confidence.max()) == pytest.approx(0.3)          # and trusted half as much
    a.estimate(img, K)
    r3 = a.estimate(img, K)                              # past `hold`: the raw map, unscaled
    assert r3.meta["anchor"]["scale"] is None and np.allclose(r3.depth_m, 3.0)
    a.reset()
    a.close()
    assert a.last is None and a.miss == 0 and a.base.resets == 1 and a.base.closes == 1


def test_anchored_smooths_the_scale_in_log_space():
    img = np.zeros((480, 640, 3), np.uint8)
    floor = _floor_or_far()
    a = dz.Anchored(Script([floor * 0.5, floor * 0.25]), cam_h=1.2, alpha=0.5)
    assert a.estimate(img, K).meta["anchor"]["scale"] == pytest.approx(2.0, rel=0.01)
    s = a.estimate(img, K).meta["anchor"]["scale"]
    assert s == pytest.approx(math.sqrt(2.0 * 4.0), rel=0.02)          # geometric mean of 2 and 4


def test_anchored_relative_model_uses_its_disparity_and_hides_it():
    img = np.zeros((480, 640, 3), np.uint8)
    z = floor_depth(K, 10.0)
    q = np.where(z > 0, 1 / np.maximum(z, 1e-6) / 3.0, 0).astype(np.float32)
    base = Script([np.ones((480, 640), np.float32)], metric=False, disp=[q])
    r = dz.Anchored(base, cam_h=1.2).estimate(img, K)
    assert r.meta["anchor"]["ok"] and r.meta["anchor"]["scale"] == pytest.approx(3.0, rel=0.01)
    assert "disparity" not in r.meta and r.meta["x"] == 1
    m = z > 0
    assert np.median(np.abs(r.depth_m[m] - z[m]) / z[m]) < 0.01


def test_temporal_and_fused_pass_reset_and_close_to_what_they_wrap():
    img = np.zeros((48, 64, 3), np.uint8)
    base = Script([np.full((48, 64), 2.0, np.float32)])
    t = dz.Temporal(base, work_width=64)
    assert t.name == "script+temporal" and t.temporal
    t.estimate(img)
    assert t.prev is not None
    t.forward_m = 0.1                                    # walking: the warped prior is moved 10 cm nearer
    r = t.estimate(img)
    assert float(r.depth_m.mean()) < 2.0
    t.reset()
    t.close()
    assert t.prev is None and base.resets == 1 and base.closes == 1
    dense = Script([np.full((48, 64), 2.0, np.float32)])
    f = dz.Fused(dense, dz.MotionStereo(work_width=64, pose_width=64))
    f.corr = 1.3
    f.reset()
    f.close()
    assert f.corr == 1.0 and dense.resets == 1 and dense.closes == 1 and f.motion.state is None


def test_push_pull_fill_fills_a_hole_from_both_sides():
    z = np.full((64, 64), 2.0, np.float32)
    z[:, 32:] = 8.0
    w = np.ones_like(z)
    w[24:40, 24:40] = 0.0                                 # a hole straddling the 2 m / 8 m edge
    out = dz.push_pull_fill(z, w)
    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(2.0, rel=1e-3) and out[0, 63] == pytest.approx(8.0, rel=1e-3)
    assert 2.0 < out[32, 31] < 8.0 and 2.0 < out[32, 32] < 8.0
    assert out[32, 24] < 4.0 < out[32, 39]               # each side of the hole leans to its own neighbour


def test_pipeline_depth_uses_the_cli_focal_length_when_given():
    seen = []

    class Spy(dz.DepthBackend):
        name = "spy"

        def estimate(self, frame_bgr, intrinsics=None):
            seen.append(intrinsics)
            h, w = frame_bgr.shape[:2]
            return dz.DepthResult(np.full((h, w), 5.0, np.float32), np.ones((h, w), np.float32), 0.0)

    d = dz.PipelineDepth(Spy(), size=(224, 126), fx=700.0)(np.zeros((270, 480, 3), np.uint8))
    assert d.shape == (126, 224) and np.allclose(d, 5.0)
    assert seen[0].fx == pytest.approx(350.0) and (seen[0].width, seen[0].height) == (480, 270)
    dz.PipelineDepth(Spy())(np.zeros((270, 480, 3), np.uint8))
    assert seen[1] == dz.Intrinsics.from_fov(480, 270, 60.0)
