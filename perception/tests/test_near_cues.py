"""perception.near: each cue's distance formula, looming, robust fusion, the Kalman filter, priorities."""

import math

import numpy as np
import pytest

from perception.near import cues, fusion, priority, synth
from perception.near import intrinsics as I
from perception.near.priors import PRIORS, prior_for

C920 = I.BUILTIN["c920"]


def _det_box(o, pitch=6.0, prof=C920):
    b = synth.box_of(prof, o, 1.2, pitch)
    return cues.Box(*b)


# ------------------------------------------------------------------ size
def test_size_height_formula():
    p = C920
    box = cues.Box(100, 100, 150, 100 + p.fy * 1.7 / 4.0)          # a 1.7 m person 4 m away, pitch 0
    c = cues.size_height(box, p, PRIORS["person"], 0.0)
    assert c.d == pytest.approx(4.0, rel=1e-6)
    assert c.sigma == pytest.approx(math.hypot(0.08, 2.0 / box.h))


def test_size_width_formula_and_cut_boxes():
    p = C920
    box = cues.Box(100, 100, 100 + p.fx * 0.5 / 2.5, 400)
    assert cues.size_width(box, p, PRIORS["person"]).d == pytest.approx(2.5)
    assert cues.size_width(cues.Box(0, 0, 200, 300, cut_left=True), p, PRIORS["person"]) is None
    assert cues.size_height(cues.Box(0, 0, 200, 300, cut_top=True), p, PRIORS["person"]) is None


def test_size_bound_only_for_cut_boxes():
    p = C920
    assert cues.size_bound(cues.Box(0, 0, 200, 1079, cut_top=True, cut_bottom=True), p, PRIORS["person"]) == \
        pytest.approx(p.fy * 1.7 * math.exp(0.16) / 1079)
    assert cues.size_bound(cues.Box(5, 5, 200, 300), p, PRIORS["person"]) is None


def test_size_cue_on_a_render_with_pitch():
    o = synth.Obj("person", 6.0, 0.2, 1.70, 0.5)
    c = cues.size_height(_det_box(o), C920, PRIORS["person"], 6.0)
    assert c.d == pytest.approx(6.0, rel=0.04)       # cos(pitch) and the box's perspective, within 4 %


# ------------------------------------------------------------------ ground
@pytest.mark.parametrize("F", [2.5, 5.0, 9.0])
def test_ground_cue_exact_on_flat_ground(F):
    o = synth.Obj("bollard", F, 0.0, 0.9, 0.25)
    c = cues.ground(_det_box(o, 10.0), C920, 1.2, 10.0, 1.0)
    assert c.d == pytest.approx(F, rel=1e-3)


def test_ground_cue_sigma_grows_with_range_and_none_when_cut():
    near = cues.ground(_det_box(synth.Obj("bollard", 2.5, 0.0, 0.9, 0.25), 10.0), C920, 1.2, 10.0, 1.0)
    far = cues.ground(_det_box(synth.Obj("bollard", 12.0, 0.0, 0.9, 0.25), 10.0), C920, 1.2, 10.0, 1.0)
    assert far.sigma > 2 * near.sigma
    assert cues.ground(cues.Box(0, 0, 10, 10, cut_bottom=True), C920, 1.2, 10.0, 1.0) is None
    assert cues.ground(cues.Box(0, 900, 10, 1000), C920, 1.2, 10.0, 1.0, elevated=True) is None


# ------------------------------------------------------------------ depth
def test_depth_cue_reads_the_object_not_the_background():
    prof = C920.scaled(960, 540)
    o = synth.Obj("pole", 4.0, 0.0, 3.0, 0.2)
    z = synth.depth_map(prof, [o], 1.2, 5.0)
    conf = np.ones_like(z, np.float32)
    box = _det_box(o, 5.0, prof)
    c = cues.depth(box, prof, z, conf, prof.resized(448, 252), 5.0, 0.05, True)
    assert c.d == pytest.approx(4.0, rel=0.05)
    filled = cues.depth(cues.Box(10, 10, 950, 530), prof, z, conf, prof.resized(448, 252), 5.0, 0.05, True)
    assert filled.sigma > c.sigma + 0.4                    # a box filling the frame barely counts


# ------------------------------------------------------------------ looming
def test_looming_tau_constant_closing_speed():
    # an object 1.7 m tall closing at 2 m/s from 10 m, seen at 10 Hz
    hist = [(t, 1000 * 1.7 / (10 - 2 * t), "a") for t in np.arange(0, 1.01, 0.1)]
    lm = cues.looming(hist)
    assert lm.tau == pytest.approx((10 - 2 * 1.0) / 2, rel=0.03)   # now: 8 m at 2 m/s = 4 s (lag-corrected)
    assert lm.rate > 0


def test_looming_needs_significance_and_one_measure():
    rng = np.random.default_rng(0)
    still = [(t, 300 * (1 + rng.normal(0, 0.02)), "a") for t in np.arange(0, 1.0, 0.1)]
    assert cues.looming(still).tau is None
    mixed = [(t, 300 + 50 * t, "a" if t < 0.5 else "w") for t in np.arange(0, 1.0, 0.1)]
    assert cues.looming(mixed).n < 6
    assert cues.looming([(0.0, 10.0, "a")]).tau is None


def test_tau_upper_is_cautious():
    lm = cues.Loom(2.0, 0.5, 0.1, 9)
    assert cues.tau_upper(lm) > 2.0
    assert cues.tau_upper(cues.Loom(2.0, 0.5, 0.6, 9)) == math.inf


def test_motion_cue_is_speed_times_tau():
    lm = cues.Loom(3.0, 1 / 3.0, 0.02, 9)
    c = cues.motion(lm, 1.2)
    assert c.d == pytest.approx(3.6)
    assert cues.motion(lm, 0.0) is None and cues.motion(cues.Loom(None, 0, 1, 3), 1.2) is None


def test_box_scale_measure():
    assert cues.box_scale(cues.Box(0, 0, 100, 400)) == pytest.approx(200.0)
    assert cues.box_scale(cues.Box(0, 0, 100, 400, cut_top=True)) == 100
    assert cues.box_scale(cues.Box(0, 0, 100, 400, cut_top=True, cut_left=True)) is None


# ------------------------------------------------------------------ fusion
def test_fuse_is_inverse_variance_in_log_space():
    f = fusion.fuse({"a": cues.Cue(4.0, 0.1), "b": cues.Cue(5.0, 0.2)})
    w = np.array([1 / 0.01, 1 / 0.04])
    want = math.exp((w[0] * math.log(4) + w[1] * math.log(5)) / w.sum())
    assert f.d == pytest.approx(want, rel=1e-6) and f.used == ["a", "b"]


def test_fuse_rejects_an_outlier_and_inflates_disagreement():
    f = fusion.fuse({"size": cues.Cue(3.0, 0.1), "ground": cues.Cue(3.2, 0.1), "depth": cues.Cue(12.0, 0.15)})
    assert f.rejected == ["depth"] and f.d == pytest.approx(math.sqrt(3.0 * 3.2), rel=1e-6)
    tight = fusion.fuse({"a": cues.Cue(3.0, 0.2), "b": cues.Cue(3.0, 0.2)})
    loose = fusion.fuse({"a": cues.Cue(2.4, 0.2), "b": cues.Cue(3.6, 0.2)})
    assert loose.sigma > tight.sigma


def test_fuse_upper_bound_and_empty():
    assert fusion.fuse({"a": cues.Cue(6.0, 0.1)}, upper=2.0).d == pytest.approx(2.0)
    assert fusion.fuse({"a": None}) is None
    assert fusion.fuse({}, upper=2.0).d == pytest.approx(1.6)


def test_kalman_learns_ttc_from_distances():
    t, d = 0.0, 10.0
    kf = fusion.LogKalman(math.log(d), 0.01, t)
    for _ in range(40):
        t += 0.1
        d -= 0.2                               # 2 m/s
        kf.predict(t)
        kf.update_distance(d, 0.05)
    ttc, _ = kf.ttc()
    assert ttc == pytest.approx(d / 2.0, rel=0.15)


def test_kalman_looming_alone_sets_the_rate_and_gates_outliers():
    kf = fusion.LogKalman(math.log(5.0), 0.01, 0.0)
    for i in range(1, 15):
        kf.predict(i * 0.1)
        kf.update_rate(0.5, 0.05)              # expanding 50 %/s: ttc 2 s
    assert kf.ttc()[0] == pytest.approx(2.0, rel=0.1)
    before = kf.d
    kf.update_distance(40.0, 0.05)             # one wild reading is gated out
    assert kf.d == pytest.approx(before, rel=0.05)
    kf.update_distance(40.0, 0.05)
    kf.update_distance(40.0, 0.05)             # three in a row: the track was wrong, reset
    assert kf.d == pytest.approx(40.0, rel=1e-6)


# ------------------------------------------------------------------ priorities
@pytest.mark.parametrize("d, p", [(0.5, 1), (1.199, 1), (1.2, 2), (2.99, 2), (3.0, 3), (30.0, 3), (None, None)])
def test_raw_priority_boundaries(d, p):
    assert priority.raw_priority(d) == p


def test_size_scales_thresholds():
    car, pole = prior_for("vehicle").size_scale(), prior_for("pole").size_scale()
    assert car > 1.25 and pole < 1.0
    assert priority.raw_priority(1.4, car) == 1 and priority.raw_priority(1.4, 1.0) == 2
    assert priority.raw_priority(1.15, pole) == 2


def test_hysteresis_escalates_after_n_and_deescalates_after_m():
    s = priority.PriorityState()
    assert s.update(1.8) is None and s.update(1.8) == 2           # N_UP = 2 frames
    assert s.update(0.9) == 2 and s.update(0.9) == 1
    for _ in range(10):
        assert s.update(1.26) == 1                               # inside the 10 % band: hold P1
    seq = [s.update(1.5) for _ in range(5)]
    assert seq[:4] == [1, 1, 1, 1] and seq[4] == 2                # M_DOWN = 5 frames
    for _ in range(4):
        assert s.update(None) == 2
    assert s.update(None) is None


def test_escalation_through_p2_to_p1_commits():
    s = priority.PriorityState()
    s.update(4.8)
    s.update(4.8)
    assert s.current == 3
    assert s.update(2.4) == 3 and s.update(0.9) == 2              # two frames more urgent: at least P2
    assert s.update(0.9) == 2 and s.update(0.84) == 1


def test_ttc_override_is_immediate_and_released_with_margin():
    s = priority.PriorityState()
    assert s.update(6.0, ttc=1.2) == 1                           # far, but closing fast: P1 at once
    assert s.update(5.8, ttc=1.7) == 1                           # held until ttc > 1.9 s
    assert s.update(5.8, ttc=2.5) == 1                           # then normal de-escalation (M frames)


def test_still_mode_decides_at_once():
    s = priority.PriorityState(still=True)
    assert s.update(1.0) == 1 and s.update(9.0) == 3
