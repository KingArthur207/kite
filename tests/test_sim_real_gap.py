import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sim_real_gap as g  # noqa: E402


def _scene(seed=0):
    """A textured 960x720 frame: smooth gradient plus random rectangles (edges for sharpness and blocks)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((720, 960, 3), np.uint8)
    img[:] = np.linspace(40, 200, 960, dtype=np.uint8)[None, :, None]
    for _ in range(60):
        x, y = rng.integers(0, 900), rng.integers(0, 680)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(10, 80)), int(y + rng.integers(10, 60))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def test_sharpness_drops_when_the_frame_is_blurred():
    img = _scene()
    assert g.sharpness(cv2.GaussianBlur(img, (0, 0), 2)) < 0.5 * g.sharpness(img)


def test_noise_estimate_recovers_added_gaussian_noise():
    flat = np.full((720, 960, 3), 120, np.uint8)
    assert g.noise_sigma(flat) < 0.1
    n = np.random.default_rng(0).normal(0, 5, flat.shape[:2])[..., None]     # same in every channel: sigma 5 in luma
    noisy = np.clip(flat + n, 0, 255).astype(np.uint8)
    assert 4.0 < g.noise_sigma(noisy) < 6.0


def test_blockiness_is_about_one_for_clean_frames_and_higher_after_low_quality_jpeg():
    texture = np.random.default_rng(0).integers(0, 255, (720, 960, 3)).astype(np.uint8)
    img = cv2.GaussianBlur(texture, (0, 0), 3)
    assert abs(g.blockiness(img) - 1.0) < 0.1
    jpg = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 10])[1], cv2.IMREAD_COLOR)
    assert g.blockiness(jpg) > g.blockiness(img) + 0.2


def test_vignette_ratio_and_colour_ratios():
    flat = np.full((720, 960, 3), (100, 150, 200), np.uint8)    # BGR
    s = g.frame_stats(flat)
    assert abs(s["vignette"] - 1.0) < 1e-6 and abs(s["brightness"] - g.gray(flat).mean()) < 1e-3
    assert abs(s["r_over_g"] - 200 / 150) < 1e-3 and abs(s["b_over_g"] - 100 / 150) < 1e-3


def test_colour_hist_distance_is_zero_for_the_same_frames_and_large_for_different_ones():
    a, b = [_scene(1)], [np.full((720, 960, 3), 250, np.uint8)]
    assert g.hist_distance(g.colour_hist(a), g.colour_hist(a)) < 1e-6
    assert g.hist_distance(g.colour_hist(a), g.colour_hist(b)) > 0.5


def test_to_tello_size_crops_16_9_to_960x720_on_the_block_grid():
    wide = np.zeros((720, 1280, 3), np.uint8)
    wide[:, 160:1120] = 255
    out = g.to_tello_size(wide)
    assert out.shape == (720, 960, 3) and out.min() == 255
    assert g.to_tello_size(np.zeros((1080, 1440, 3), np.uint8)).shape == (720, 960, 3)


def test_detector_table_bins_by_distance():
    dets = [("person", 0.9, 5.0), ("person", 0.2, 15.0), ("car", 0.5, 30.0)]
    t = g.detector_table(dets, n_frames=2)
    assert t["person <10m"] == (0.5, 0.9, 1.0)
    assert t["person 10-20m"] == (0.5, 0.2, 0.0)
    assert t["car >20m"][1] == 0.5 and t["car <10m"] == (0.0, None, None)


def test_fit_moves_the_model_toward_the_target_stats():
    model = g.load_model()
    raw = [_scene(2), _scene(3)]
    target = g.summarise(g.apply_model(raw, g.set_param(model, ("exposure", "target_luma"), 160.0)))
    start = g.gap(g.summarise(g.apply_model(raw, model)), target)
    fitted, score = g.fit(raw, target, model, passes=2, log=lambda *_: None)
    assert score < start
    assert fitted["exposure"]["target_luma"] > model["exposure"]["target_luma"]
