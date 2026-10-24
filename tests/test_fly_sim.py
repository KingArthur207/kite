"""kite.fly.sim: the box-level follow simulator around kite's real WalkerTracker, FollowController and SafetyGovernor.

Determinism and common random numbers, the camera geometry at the kite spec (1.0 m behind, 0.30 m above), the person
box, the detector's timing and per-frame luck, the drone's dead time, lag and position-held hover, the walker's speed
cap, acceleration and step-back stop, the metrics on constructed sequences, the governor clamping a controller that
asks for more, the yaw_source hook (a source returning `fallback` is KITE-LAW exactly, a source returning 0 is clearly
worse), the collision rate of KITE-LAW on demo, and sane metrics on every profile. Seeds are from BASELINE_SEEDS."""

import math

import numpy as np
import pytest

from kite.camera import Camera
from kite.fly.interfaces import BASELINE_SEEDS, METRICS
from kite.fly.sim import run_episode
from kite.fly.sim.batch import parse_seeds, summarize
from kite.fly.sim.camera import H, DetectorParams, FakeDetector, head_in_view, project, visible_box
from kite.fly.sim.drone import DroneModel, DroneParams, noise_len
from kite.fly.sim.metrics import EpisodeMetrics
from kite.fly.sim.profiles import PROFILES, draw_params, walker_speed_cap
from kite.fly.sim.reward import TERMS, Reward, RewardCfg, e_z
from kite.fly.sim.walker import Walker, WalkerCfg
from kite.fly.sim.world import default_follow_cfg, start_pose
from kite.follow import FollowCfg, FollowController

SEEDS = BASELINE_SEEDS


class FallbackYaw:
    """Returns the law's own value: must reproduce KITE-LAW exactly."""

    def __init__(self):
        self.seeds, self.calls = [], 0

    def reset(self, seed=None):
        self.seeds.append(seed)

    def yaw(self, t, frame_t, bearing_deg, range_m, fallback):
        self.calls += 1
        return fallback


class ZeroYaw:
    def reset(self):
        pass

    def yaw(self, t, frame_t, bearing_deg, range_m, fallback):
        return 0.0


def walker_box(range_m, above=0.30, pitch=0.0, fx=920.0, right=0.0):
    """Head top `above` under the camera at range_m; head 0.23 m, 0.15 m wide; shoulders 0.45 m; camera 2.0 m up."""
    return project(range_m, right, -above, 0.23, 0.075, 0.225, 2.0, pitch, fx, fx)


# ------------------------------------------------------------------ determinism, common random numbers
def test_same_seed_and_profile_is_the_same_episode():
    a, b = run_episode(SEEDS[3], "train", record=True), run_episode(SEEDS[3], "train", record=True)
    assert a.ret == b.ret and a.metrics == b.metrics and a.terms == b.terms and a.n_ticks == b.n_ticks
    for k in ("x", "y", "z", "yaw", "fb", "state", "walker_x", "est_range_m", "t"):
        assert a.trace[k] == b.trace[k]
    assert run_episode(SEEDS[4], "train").ret != a.ret
    assert run_episode(SEEDS[3], "demo").ret != a.ret  # the profile is part of the world's seed


def test_the_world_is_the_same_for_every_arm():
    """Parameters, start, control clock and the walker's path come from the seed alone."""
    s = SEEDS[7]
    assert draw_params(s, "train") == draw_params(s, "train")
    law = run_episode(s, "demo", record=True)
    zero = run_episode(s, "demo", yaw_source=ZeroYaw(), record=True)
    meta_a, meta_b = law.trace["meta"][0], zero.trace["meta"][0]
    assert meta_a["params"] == meta_b["params"] and meta_a["segments"] == meta_b["segments"]
    assert meta_a["start"] == meta_b["start"] and meta_a["t_start"] == meta_b["t_start"]
    assert law.metrics["start_failed"] == zero.metrics["start_failed"] == 0.0
    n = min(len(law.trace["t"]), len(zero.trace["t"]))
    assert law.trace["t"][:n] == zero.trace["t"][:n]  # the jittered control clock
    if meta_a["step"][0] < 0:  # no step back toward the drone: the walker's path is identical
        assert law.trace["walker_x"][:n] == zero.trace["walker_x"][:n]
    assert law.trace["yaw"] != zero.trace["yaw"]


def test_profiles():
    demo = [draw_params(s, "demo") for s in SEEDS[:5]]
    assert all(p.fwd_gain_mps == 0.96 and p.fx_px == 920.0 and p.max_rc == 50 and p.det_rate_hz <= 15.0
               and p.video_latency_s == 0.10 and p.p_head == 0.55 for p in demo)
    assert len({p.head_top_m for p in demo}) == 5  # the walker still varies
    train = [draw_params(s, "train") for s in SEEDS[:50]]
    assert all(0.67 <= p.fwd_gain_mps <= 1.5 and 50 <= p.max_rc <= 100 and 880 <= p.fx_px <= 1100
               and 0.0 <= p.video_latency_s <= 0.25 and 0.03 <= p.det_time_s <= 0.10 and 0.3 <= p.p_head <= 0.8
               and 1.0 <= p.hold_tau_s <= 2.0 for p in train)
    stress = [draw_params(s, "stress") for s in SEEDS[:20]]
    assert {p.fx_px for p in stress} == {880.0, 1100.0} and all(p.dropout == 0.30 for p in stress)
    assert all(0.45 <= p.video_latency_s <= 0.65 and 0.3 <= p.p_head <= 0.8 for p in stress)
    assert all(p.max_rc == 100 and p.walk_speed_lo == 1.1 for p in (draw_params(s, "fast") for s in SEEDS[:5]))
    assert draw_params(SEEDS[0], "demo", {"fx_px": 1077, "p_head": 0.3}).p_head == 0.3
    assert walker_speed_cap(0.96, 50) == pytest.approx(0.432)
    assert parse_seeds("0-3,7,10-11") == [0, 1, 2, 3, 7, 10, 11] and parse_seeds("baseline") == list(BASELINE_SEEDS)
    assert default_follow_cfg() == FollowCfg(behind_m=1.0, above_m=0.30, max_yaw=70)


def test_detector_rate_never_exceeds_one_over_detector_time():
    ps = [draw_params(s, prof) for s in SEEDS for prof in ("train", "stress")]
    assert all(p.det_rate_hz <= 1.0 / p.det_time_s + 1e-9 for p in ps)
    assert any(p.det_rate_hz == pytest.approx(1.0 / p.det_time_s) for p in ps)  # the cap does bind sometimes
    assert draw_params(SEEDS[0], "demo", {"det_time_s": 0.1}).det_rate_hz == pytest.approx(10.0)


def test_starts_have_the_head_well_inside_the_frame():
    cfg = default_follow_cfg()
    for s in SEEDS:
        for profile in ("demo", "train"):
            p, sp = draw_params(s, profile), start_pose(s, profile, draw_params(s, profile), cfg)
            assert 360 + p.fx_px * (0.30 + sp.alt_offset_m) / sp.range_m <= 700 and sp.redraws < 64
            assert 0.8 <= sp.range_m <= 1.8 and abs(math.degrees(sp.bearing_rad)) <= 15


# ------------------------------------------------------------------ camera and detector
def test_head_top_row_at_the_kite_geometry():
    """1.0 m behind, 0.30 m above, fy 920, level: the head top is 16.7 deg down, row 636 of 720, as kite expects."""
    g = walker_box(1.0)
    assert g[0] == pytest.approx(480.0) and g[1] == pytest.approx(636.0, abs=0.5)
    assert FollowController(FollowCfg(behind_m=1.0, above_m=0.30), Camera()).target_row() == pytest.approx(g[1])
    # up close only the head is in frame (the shoulder line is 29 deg down), cut by the bottom edge
    x1, y1, x2, y2, torso = visible_box(g)
    assert not torso and y2 == H and x2 - x1 == pytest.approx(2 * 920 * 0.075)
    # 2 m back the shoulders are in frame and the box widens to them
    x1, y1, x2, y2, torso = visible_box(walker_box(2.0))
    assert torso and y2 == H and x2 - x1 == pytest.approx(920 * 0.45 / 2.0)
    # nose up (braking) moves the head down the frame: 5 deg pushes it out of the bottom
    assert walker_box(1.0, pitch=5.0)[1] > 716 and not head_in_view(walker_box(1.0, pitch=5.0))
    assert walker_box(1.0, pitch=-5.0)[1] < 636  # nose down moves it up
    assert walker_box(1.0, right=math.tan(math.radians(10)))[0] == pytest.approx(
        480 + 920 * math.tan(math.radians(10)))
    assert walker_box(1.0, fx=1077.0)[1] == pytest.approx(360 + 1077 * 0.3)


def _frames(det, geom, t1=6.0):
    """Run a detector over a constant geometry; every ready item (t_available, frame_t, dets, true_conf)."""
    out = []
    for i in range(int(t1 / 0.01)):
        det.capture(i * 0.01, (i + 1) * 0.01, geom, geom)
        out += det.ready((i + 1) * 0.01)
    return out


def test_detector_timing():
    p = DetectorParams(rate_hz=10.0, video_latency_s=0.2, det_time_s=0.05, noise_px=0.0, p_head=1.0)
    items = _frames(FakeDetector(p, np.random.default_rng(1), 0.0, 6.0), walker_box(1.0))
    frame_t = [it[1] for it in items]
    assert all(b - a == pytest.approx(0.1) for a, b in zip(frame_t, frame_t[1:]))  # rate_hz
    first_capture = frame_t[0] - 0.2
    assert 0.0 <= first_capture < 0.1  # a random phase, then the video latency
    for t_av, ft, dets, conf in items:  # boxes reach the tracker det_time after the frame reached the laptop
        assert t_av == pytest.approx(ft + 0.05)
    head_only = [it for it in items if it[2]]
    assert len(head_only) == len(items) and all(0.25 <= it[3] <= 0.6 for it in head_only)  # p_head 1, 84 px


def test_detector_luck_is_drawn_up_front():
    """Dropouts, bursts and false boxes depend on the seed only, not on what the camera sees (common random numbers)."""
    p = DetectorParams(rate_hz=15.0, dropout=0.3, burst_rate_hz=0.2, false_frac=0.05, p_head=0.55)
    a = _frames(FakeDetector(p, np.random.default_rng(7), 0.0, 20.0), walker_box(1.0), 20.0)
    b = _frames(FakeDetector(p, np.random.default_rng(7), 0.0, 20.0), walker_box(1.2, right=0.2), 20.0)
    assert [x[1] for x in a] == [x[1] for x in b]
    assert [x[3] > 0 for x in a] == [x[3] > 0 for x in b]  # the same frames carry a walker box
    assert [len(x[2]) - (x[3] > 0) for x in a] == [len(x[2]) - (x[3] > 0) for x in b]  # the same false boxes
    hit = sum(x[3] > 0 for x in a) / len(a)
    assert 0.2 < hit < 0.55  # p_head 0.55 x (1 - dropout 0.3) minus bursts


# ------------------------------------------------------------------ drone and walker
def test_drone_dead_time_then_first_order_lag():
    p = DroneParams(yaw_dead_s=0.20, tau_yaw_s=0.10, fwd_dead_s=0.50, tau_fwd_s=0.45, pitch_scale=1.0)
    d = DroneModel(p)
    d.reset(0.0, 0.0, 2.0, 0.0)
    rates = []
    for _ in range(200):
        d.step(0.0, 0.0, 0.0, 100.0)
        rates.append(d.yaw_rate_dps)
    assert max(rates[:20]) == 0.0 and rates[20] > 0.0  # 0.20 s dead time = 20 sub-steps of 0.01 s
    assert rates[29] == pytest.approx(55.0 * (1 - math.exp(-0.1 / 0.1)), rel=1e-6)  # then the first-order lag
    assert rates[-1] == pytest.approx(55.0, rel=1e-3) and d.heading > 0.0  # +yaw stick turns clockwise
    d = DroneModel(p)
    d.reset(0.0, 0.0, 2.0, 0.0)
    speeds, pitch = [], []
    for _ in range(400):
        d.step(0.0, 50.0, 0.0, 0.0)
        speeds.append(d.v_forward())
        pitch.append(d.pitch_deg)
    assert max(speeds[:50]) == 0.0 and speeds[50] > 0.0
    assert speeds[-1] == pytest.approx(0.48, rel=1e-3) and d.y > 0.0 and d.x == pytest.approx(0.0)  # north
    assert min(pitch) < -1.0 and pitch[-1] == pytest.approx(0.0, abs=0.05)  # nose down while accelerating


def test_the_hover_holds_position_and_sticks_let_the_wind_push():
    p = DroneParams(drift_sigma_mps=0.05, wind_sigma_mps=0.15, hold_tau_s=1.5)
    n = 6000  # 60 s
    held, pushed = [], []
    for s in range(20):
        noise = np.random.default_rng(s).standard_normal(noise_len(n)).tolist()
        for sticks, out in (((0, 0, 0, 0), held), ((0, 0, 0, 30), pushed)):  # yaw only: the hold stays on
            d = DroneModel(p, noise)
            d.reset(0.0, 0.0, 2.0, 0.0)
            for _ in range(n):
                d.step(*sticks)
            out.append(math.hypot(d.x, d.y))
    assert np.sqrt(np.mean(np.square(held))) < 0.35  # bounded wander
    assert np.allclose(held, pushed)  # a yaw stick does not release the position hold
    d = DroneModel(p, np.random.default_rng(0).standard_normal(noise_len(n)).tolist())
    d.reset(0.0, 0.0, 2.0, 0.0)
    far = []
    for _ in range(n):
        d.step(0, 1, 0, 0)  # a forward stick of 1 (1 cm/s): the hold is off, drift and wind integrate
        far.append(math.hypot(d.x, d.y - 0.0096 * d._n * 0.01))
    assert not d.holding and max(far) > 0.5


def test_walker_speed_cap_acceleration_and_step_back_stop():
    cap = walker_speed_cap(0.96, 50)
    for s in range(30):
        w = Walker(np.random.default_rng(s), 0.0, 1.0, 0.0, cap, (0.5, 1.5), 6000, WalkerCfg(step_prob=1.0))
        assert max(w.vs) <= cap + 1e-12
        assert max(abs(b - a) for a, b in zip(w.vs, w.vs[1:])) <= 1.0 * 0.01 + 1e-12  # 1 m/s^2
        kinds = [k for k, _, _ in w.segments]
        assert all(b in ("straight", "step") for a, b in zip(kinds, kinds[1:]) if a == "turn")  # never turn, turn
        first, last = w.step_ticks
        assert 0 < first < last
        # a drone hovering 0.5 m behind the walker where the step starts: the walker stops 0.8 m from it
        bx, by = w.xs[first], w.ys[first]
        assert all(w.vs[i] == 0.0 for i in range(first, last))  # the walker stands while it steps back
        drone = (bx, by - 1.2)
        gaps = [math.hypot(x - drone[0], y - drone[1]) for x, y in (w.position(i, *drone) for i in range(first, last))]
        stride = w.step.speed_mps * 0.01
        planned = w.step.speed_mps * (last - first) * 0.01  # the step's drawn length
        assert min(gaps) >= 0.8 - stride - 1e-9  # it never steps closer than 0.8 m
        assert gaps[-1] == pytest.approx(max(1.2 - planned, 0.8), abs=stride + 1e-9)  # full step or the stop


# ------------------------------------------------------------------ metrics and reward
def test_losses_corner_losses_and_pitch_exits_on_constructed_sequences():
    m = EpisodeMetrics(behind_m=1.0, min_range_m=0.7, t_metric=3.0, corners=[(10.0, 14.0)])
    gaps = [False] * 5 + [True] * 3 + [False] * 4 + [True] * 2 + [False] * 2  # control steps from t = 3.0
    last_true = [3.0] * 5 + [12.0] * 3 + [12.5] * 4 + [20.0] * 2 + [21.0] * 2
    for i, (g, lt) in enumerate(zip(gaps, last_true)):
        m.control(3.0 + 0.1 * i, 0.1, "following" if not g else "lost", 2.0 if i == 1 else 0.0, 5.0, g, lt)
    m.control(1.0, 0.1, "following", 50.0, 50.0, True, 0.0)  # before the window: ignored
    views = [(True, False, 1.0), (False, True, 1.0), (True, False, 1.0), (False, False, 1.0),  # bottom, side
             (True, False, 2.0), (False, True, 2.0)]  # bottom but far
    for i, (v, below, r) in enumerate(views):
        m.phys(5.0 + 0.01 * i, r, 0.0, v, below, 0.0, r)
    out = m.result(n_phys_full=6, n_ctrl_full=16)
    assert out["losses"] == 2 and out["corner_losses"] == 1 and out["time_lost_s"] == pytest.approx(0.5)
    assert out["pitch_exits"] == 1 and out["in_view"] == pytest.approx(3 / 6)
    assert out["following"] == pytest.approx(11 / 16) and out["stick_change"] == pytest.approx(2.0 / 11)


def test_a_collision_charges_the_rest_of_the_episode():
    m = EpisodeMetrics(behind_m=1.0, min_range_m=0.7, t_metric=3.0)
    for i in range(100):
        m.phys(3.01 + 0.01 * i, 1.0, 0.0, True, False, 0.0, 1.0)
    m.control(3.5, 0.1, "following", 0.0, 0.0, False, 3.4)
    m.collided(0.29, 0.4, ticks_left=300, window_s_left=3.0)
    out = m.result(n_phys_full=400, n_ctrl_full=40)
    assert out["collision"] == 1.0 and out["in_view"] == pytest.approx(0.25) and out["losses"] == 1
    assert out["rms_bearing_deg"] == pytest.approx(math.sqrt(0.75) * 90) and out["time_lost_s"] == 3.0
    assert out["min_range_m"] == 0.29 and out["min_dist3d_m"] == 0.4 and out["rms_range_err_m"] == 0.0
    r = Reward(RewardCfg(), 1.0, 0.7)
    tail = r.collision(10.0)
    assert tail == pytest.approx(-200.0 - 10.0 * (5.0 + 7.6 * (math.pi / 2) ** 2))
    assert r.sums["lost"] == pytest.approx(-50.0) and r.sums["e_x"] < -180.0


def test_reward_band_and_terms():
    c = RewardCfg()
    assert e_z(1.10, 1.0, 0.15) == 0.0 and e_z(0.80, 1.0, 0.15) == pytest.approx(1 / 3)
    assert e_z(1.30, 1.0, 0.15) > e_z(0.70, 1.0, 0.15)  # the far side costs more
    assert e_z(9.0, 1.0, 0.15) == c.e_z_cap
    r = run_episode(SEEDS[0], "demo")
    assert set(r.terms) == set(TERMS) == {"e_x", "jerk", "lost", "e_z", "too_close", "ev_loss", "ev_too_close",
                                          "ev_collision"}
    assert sum(r.terms.values()) == pytest.approx(r.ret)


def test_the_real_governor_clamps_a_controller_that_asks_for_more():
    r = run_episode(SEEDS[1], "demo", record=True, controller_max_rc=100)  # governor: SafetyCfg.max_rc 50
    tr = r.trace
    assert max(max(abs(v) for v in tr[k]) for k in ("lr", "fb", "ud", "yaw")) <= 50
    assert max(v for v in tr["cmd_fb"] if v is not None) > 50  # the controller did ask for more
    assert set(tr["action"]) == {"ok"} and not tr["meta"][0]["cancelled"]
    gaps = [b - a for a, b in zip(tr["t"], tr["t"][1:])]
    assert 0.0999 <= min(gaps) and max(gaps) <= 0.1121  # the jittered control period


# ------------------------------------------------------------------ the yaw_source hook
def test_a_yaw_source_returning_fallback_is_kite_law():
    for seed in SEEDS[:3]:
        law = run_episode(seed, "demo")
        src = FallbackYaw()
        hooked = run_episode(seed, "demo", yaw_source=src, arm="FALLBACK")
        assert hooked.ret == law.ret and hooked.metrics == law.metrics and hooked.terms == law.terms
        assert src.calls > 0 and seed in src.seeds  # reset with the episode seed after the controller's reset


def test_a_yaw_source_returning_zero_is_clearly_worse():
    seeds = SEEDS[:8]
    law = summarize([run_episode(s, "demo") for s in seeds])
    zero = summarize([run_episode(s, "demo", yaw_source=ZeroYaw()) for s in seeds])
    assert zero["rms_bearing_deg"] > law["rms_bearing_deg"] + 10.0
    assert zero["mean_abs_yaw"] == 0.0 and zero["in_view"] < law["in_view"]


# ------------------------------------------------------------------ safety and sanity
def test_kite_law_collision_rate_on_demo():
    res = [run_episode(s, "demo") for s in SEEDS]
    assert sum(r.metrics["collision"] for r in res) <= 3
    assert sum(r.metrics["start_failed"] for r in res) <= 3


@pytest.mark.parametrize("profile", PROFILES)
def test_metrics_are_finite_and_in_range(profile):
    for s in SEEDS[:3]:
        r = run_episode(s, profile, episode_s=30.0)
        assert set(METRICS) | {"min_dist3d_m", "start_failed"} == set(r.metrics)
        assert r.profile == profile and r.arm == "KITE-LAW"
        m = r.metrics
        assert all(math.isfinite(v) for v in m.values()) and math.isfinite(r.ret)
        for k in ("in_view", "following", "in_band"):
            assert 0.0 <= m[k] <= 1.0
        assert 0.0 <= m["rms_bearing_deg"] <= 180.0 and m["p95_bearing_deg"] <= 180.0
        assert 0.0 <= m["time_lost_s"] <= 30.0 and m["collision"] in (0.0, 1.0)
        assert 0 <= m["corner_losses"] <= m["losses"] and m["pitch_exits"] >= 0
        assert m["min_dist3d_m"] >= m["min_range_m"] >= 0.0 and m["rms_alt_err_m"] < 0.5
        assert 0.0 <= m["mean_abs_yaw"] <= 70.0 and m["start_failed"] in (0.0, 1.0)
        assert r.ret <= 0.0 and sum(r.terms.values()) == pytest.approx(r.ret)
        assert r.n_ticks == (600 if not m["collision"] else r.n_ticks) and r.wall_s > 0.0
