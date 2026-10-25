"""depth_zoo: the interface, floor anchoring, motion stereo, fusion, temporal smoothing and the benchmark's
scoring rules. No model and no download: fake backends and the numpy ray caster stand in."""

import math

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
