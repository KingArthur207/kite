"""kite.fly.steer as kite.follow's yaw sources (FlySteer, PredPD): the fly arms turn toward a walker steadily off
centre (the R8 sign check), the ticks follow the caller's clock (at most 3 per call, resync after gaps), the latency
filter predicts the walker to the tick time, no measurement for 0.5 s means no target, the geometry comes from the
controller's FollowCfg, determinism by seed, the fallback on any failure, and both inside a real FollowController."""

import json
import math

import numpy as np
import pytest

from kite.camera import Camera
from kite.fly import steer as steer_mod
from kite.fly.interfaces import BRAIN_TICK_MS, DT, OUTPUT_GROUPS, PREV_PARAMS
from kite.fly.decoder import soft_deadzone
from kite.fly.params import HAND_FILE, PD_SPACE, load_params
from kite.fly.steer import (MAX_TICKS_PER_CALL, WARMUP_S, FlySteer, PredPD, make_fly_steer, make_hand, make_pd,
                            make_prev)
from kite.follow import FollowCfg, FollowController, Track
from kite.types import Telemetry

FALLBACK = -77.0
CFG = FollowCfg(behind_m=1.0, above_m=0.30, max_yaw=70)  # DESIGN.md R3: what every arm flies


def drive(fly, bearing, seconds=3.0, range_m=1.0, t0=50.0, rate=10.0):
    """Open loop: a walker held at `bearing`, a fresh frame on every call (arrived 30 ms before it)."""
    out = []
    for i in range(int(round(seconds * rate))):
        t = t0 + i / rate
        out.append(fly.yaw(t, t - 0.03, bearing, range_m, FALLBACK))
    return np.array(out)


@pytest.fixture(scope="module")
def prev():
    return make_fly_steer("FLY-YAW-PREV", cfg=CFG, seed=0)


# FLY-YAW-HAND's ridge readout crosses zero at about -5.5 deg (it turns right at 0 deg and holds the walker left of
# centre), so its sign is wrong at -4 deg; FLY-YAW-PREV crosses within 0.5 deg of centre.
R8 = {"FLY-YAW-HAND": (-18.0, -10.0, 10.0, 18.0), "FLY-YAW-PREV": (-18.0, -10.0, -4.0, 4.0, 10.0, 18.0)}


@pytest.mark.parametrize("size_blind", [True, False])
@pytest.mark.parametrize("arm", list(R8))
def test_fly_arms_turn_toward_a_walker_off_centre(arm, size_blind):
    """R8: a walker held steadily off centre (at the standoff, so s = 1 in both size modes) -> the right sign."""
    fly = make_fly_steer(arm, cfg=CFG, size_blind=size_blind, seed=0)
    assert fly.arm == arm and fly.size_blind is size_blind
    for bearing in R8[arm]:
        fly.reset()
        y = drive(fly, bearing)
        assert math.copysign(1.0, bearing) * y[-10:].mean() > 1.0, (bearing, y[-10:])  # after 2 s of settling
        assert np.all(np.abs(y) <= 60.0) and FALLBACK not in y and fly.failed is None
        assert fly.stats["ticks"] == 2 * 30 - 1  # 2 per 0.1 s call, 1 on the first


def test_deterministic_for_a_seed_and_reset_replays():
    a = make_fly_steer("FLY-YAW-PREV", seed=5)
    b = FlySteer.from_file(PREV_PARAMS, seed=0)
    b.reset(seed=5)  # the simulator's call: the episode seed seeds the brain's Poisson input
    ya, yb = drive(a, 7.0, seconds=1.5), drive(b, 7.0, seconds=1.5)
    assert np.array_equal(ya, yb) and np.array_equal(a.last_rates, b.last_rates)
    a.reset()  # FollowController.reset() calls it without a seed: the same fly again
    assert np.array_equal(drive(a, 7.0, seconds=1.5), ya)
    a.reset(seed=6)
    assert a.seed == 6
    drive(a, 7.0, seconds=1.5)
    assert a.brain.last_counts.sum() > 0 and not np.array_equal(a.brain.last_counts, b.brain.last_counts)


def test_geometry_and_lag_come_from_the_controllers_follow_cfg():
    """How a trainer builds one per candidate: params and norm dicts, behind_m and the BELIEVED video lag from the
    same FollowCfg the controller flies; explicit values that disagree with it are refused."""
    pf = load_params(PREV_PARAMS)
    cfg = FollowCfg(behind_m=1.3, above_m=0.30, video_lag_s=0.3)
    fly = FlySteer(params=pf.params, norm=dict(zip(OUTPUT_GROUPS, pf.norm.tolist())), size_blind=False, seed=4, cfg=cfg)
    assert (fly.behind_m, fly.video_lag_s, fly.size_blind, fly.cfg) == (1.3, 0.3, False, cfg)
    ref = FlySteer.from_file(PREV_PARAMS, size_blind=False, seed=4, behind_m=1.3, video_lag_s=0.3)  # no cfg: explicit
    assert np.array_equal(drive(fly, -9.0, seconds=1.0, range_m=1.2), drive(ref, -9.0, seconds=1.0, range_m=1.2))
    assert fly.stats["s"] == pytest.approx(1.3 / 1.2, abs=1e-3)
    assert FlySteer.from_file(PREV_PARAMS, cfg=cfg, behind_m=1.3).behind_m == 1.3  # agreeing values are fine
    for bad in ({"behind_m": 1.0}, {"video_lag_s": 0.25}):
        with pytest.raises(ValueError):
            make_fly_steer("FLY-YAW-PREV", cfg=cfg, **bad)
        with pytest.raises(ValueError):
            PredPD(cfg=cfg, **bad)
    assert make_fly_steer("FLY-YAW-HAND", cfg=CFG).video_lag_s == FollowCfg.video_lag_s == 0.25
    with pytest.raises(KeyError):
        FlySteer(params={"enc_r_max_hz": 100.0}, norm=pf.norm)
    assert make_prev().arm == "FLY-YAW-PREV" and make_hand().arm == "FLY-YAW-HAND" and make_pd().arm == "PRED-PD"
    assert (make_prev().behind_m, make_prev().video_lag_s) == (1.0, 0.25)  # no cfg: the kite spec


def test_ticks_follow_the_callers_clock_and_resync_after_gaps(prev):
    prev.reset()
    t = 10.0
    prev.yaw(t, t, 3.0, 1.0, FALLBACK)
    assert prev.stats["ticks"] == 1
    for _ in range(4):  # 20 Hz calls: one tick each
        t += 0.05
        prev.yaw(t, t, 3.0, 1.0, FALLBACK)
    assert prev.stats["ticks"] == 5
    prev.yaw(t + 0.02, t, 3.0, 1.0, FALLBACK)  # not due yet
    assert prev.stats["ticks"] == 5
    t += 1.0  # the controller was coasting for a second: run the 3 most recent ticks, not 20
    prev.yaw(t, t, 3.0, 1.0, FALLBACK)
    assert prev.stats["ticks"] == 5 + MAX_TICKS_PER_CALL and prev.stats["resyncs"] == 1
    t += 0.1
    prev.yaw(t, t, 3.0, 1.0, FALLBACK)
    assert prev.stats["ticks"] == 5 + MAX_TICKS_PER_CALL + 2  # back on the 50 ms grid
    assert prev.brain.ticks == round(WARMUP_S / DT) + prev.stats["ticks"] and BRAIN_TICK_MS == 1000 * DT
    prev.yaw(t - 5.0, t - 5.0, 3.0, 1.0, FALLBACK)  # a clock that jumps back restarts the schedule
    assert prev.failed is None and prev.stats["ticks"] == 5 + MAX_TICKS_PER_CALL + 3


def test_latency_filter_predicts_the_walker_to_the_tick(prev):
    """The walker sweeps right at 10 deg/s; each frame shows it video_lag_s late. The fly sees where it is now."""
    prev.reset()
    t0, rate = 20.0, 10.0
    for i in range(30):
        t = t0 + i / rate
        frame_t = t - 0.03  # captured at frame_t - 0.25
        prev.yaw(t, frame_t, -5.0 + 10.0 * (frame_t - 0.25 - t0), 1.0, FALLBACK)
    t_tick = t0 + 29 / rate
    truth = -5.0 + 10.0 * (t_tick - t0)
    assert prev.stats["bearing_pred_deg"] == pytest.approx(truth, abs=0.3)
    assert prev.stats["bearing_pred_deg"] - (-5.0 + 10.0 * (t_tick - 0.28 - t0)) > 2.5  # not the late frame


def test_no_new_frame_for_half_a_second_means_no_target(prev):
    prev.reset()
    drive(prev, 15.0, seconds=1.0, t0=0.0)
    assert prev.stats["target_valid"] and prev.dn_rates["DNa02_R"] > 100
    frame_t = 0.9 - 0.03  # the same frame from now on
    for i in range(1, 15):
        t = 0.9 + 0.1 * i
        y = prev.yaw(t, frame_t, 15.0, 1.0, FALLBACK)
        assert prev.stats["target_valid"] == (t - frame_t <= 0.5 + 1e-9)
    assert prev.dn_rates["DNa02_R"] == 0.0 and y == 0.0  # the DNs relaxed, the stick went back to 0


def test_size_comes_from_the_range_unless_size_blind():
    seeing = FlySteer.from_file(HAND_FILE, size_blind=False, behind_m=1.0)
    blind = FlySteer.from_file(HAND_FILE, size_blind=True)
    for fly in (seeing, blind):
        drive(fly, 4.0, seconds=0.5, range_m=0.8)
    assert seeing.stats["s"] == pytest.approx(1.25, abs=0.01) and blind.stats["s"] == 1.0


def test_any_failure_falls_back_for_the_rest_of_the_episode(monkeypatch):
    msgs = []
    fly = make_fly_steer("FLY-YAW-PREV", log_fn=msgs.append)
    drive(fly, 5.0, seconds=0.5)
    real_tick = fly.brain.tick

    def broken(*a, **k):
        raise RuntimeError("injected")

    monkeypatch.setattr(fly.brain, "tick", broken)
    assert fly.yaw(60.0, 59.97, 5.0, 1.0, 12.5) == 12.5
    monkeypatch.setattr(fly.brain, "tick", real_tick)  # even once it would work again: fallback until reset
    assert [fly.yaw(60.0 + 0.1 * i, 60.0 + 0.1 * i, 5.0, 1.0, -3.0) for i in range(1, 4)] == [-3.0] * 3
    assert len(msgs) == 1 and "injected" in msgs[0] and "RuntimeError" in fly.stats["failed"]
    fly.reset()
    assert fly.failed is None and drive(fly, 5.0, seconds=0.3)[-1] != FALLBACK
    assert fly.yaw(80.0, 79.0, float("nan"), 1.0, 4.0) == 4.0  # a non-finite measurement is a failure too
    monkeypatch.setattr(fly.brain, "reset", broken)
    fly.reset()  # never raises, even when the brain cannot reset
    assert fly.failed and fly.yaw(90.0, 90.0, 5.0, 1.0, 9.0) == 9.0 and len(msgs) == 3


def test_fly_arm_factory(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        make_fly_steer("KITE-LAW")
    monkeypatch.setitem(steer_mod.ARM_FILES, "FLY-YAW", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        make_fly_steer("FLY-YAW")
    hand = make_fly_steer("FLY-YAW-HAND", video_lag_s=0.4, behind_m=1.2)
    assert hand.stats["source"] == str(HAND_FILE) and hand.video_lag_s == 0.4 and hand.behind_m == 1.2


@pytest.mark.parametrize("kind", ["FLY-YAW-PREV", "PRED-PD"])
def test_inside_a_real_follow_controller(kind):
    """The source decides yaw only while following, inside FollowController's clamps; coasting does not call it, and
    a lost -> following resume resets it."""
    cam = Camera()
    cfg = CFG
    src = PredPD(cfg=cfg) if kind == "PRED-PD" else make_fly_steer(kind, cfg=cfg)
    ctrl = FollowController(cfg, cam, max_rc=80, yaw_source=src)
    for bearing, sign in ((12.0, 1), (-12.0, -1)):
        ctrl.reset(2.0)
        assert src.stats["ticks"] == 0  # FollowController.reset() reset the source too
        y1 = cam.cy + cam.fy * cfg.above_m / 1.0
        xc = cam.cx + cam.fx * math.tan(math.radians(bearing))
        rcs = []
        for i in range(30):
            t = 100.0 + 0.1 * i
            track = Track((xc - 70, y1, xc + 70, cam.height), 0.8, t - 0.03, (cam.width, cam.height), hits=5)
            cmd = ctrl.update(track, Telemetry(t=t, height_m=2.0, pitch_deg=0.0), t)
            assert cmd.state == "following" and cmd.bearing_deg == pytest.approx(bearing, abs=0.1)
            rcs.append(cmd.rc)
        assert all(isinstance(v, int) for rc in rcs for v in rc)
        yaw = np.array([rc[3] for rc in rcs])
        assert np.all(np.abs(yaw) <= cfg.max_yaw) and np.all(np.abs(np.diff(yaw)) <= cfg.slew * 0.1 + 1)
        assert sign * yaw[-10:].mean() > 5.0 and src.failed is None and ctrl.yaw_source is src
        assert abs(rcs[-1][1]) < 30 and abs(rcs[-1][2]) <= 1  # at the standoff and the held height: little fb / ud
        ticks = src.stats["ticks"]
        stale = Track(track.box, 0.8, 102.9, (cam.width, cam.height), hits=5)
        assert ctrl.update(stale, Telemetry(t=103.3, height_m=2.0), 103.3).state == "coast"
        assert src.stats["ticks"] == ticks
        assert ctrl.update(stale, Telemetry(t=103.6, height_m=2.0), 103.6).state == "lost"
        back = Track(track.box, 0.8, 103.97, (cam.width, cam.height), hits=5)
        assert ctrl.update(back, Telemetry(t=104.0, height_m=2.0), 104.0).state == "following"
        assert src.stats["ticks"] == 1  # reset on the resume, then one tick


def test_pred_pd_shares_the_filter_and_conditioning(tmp_path):
    """PRED-PD: kp * deadband(theta_hat) + kd * dtheta_hat on exactly FlySteer's prediction, then the same stick
    conditioning; its 3 parameters live in PD_SPACE for CMA-ES."""
    assert PD_SPACE.names == ("kp", "kd", "deadband_deg") and PD_SPACE.dim == 3
    assert PD_SPACE.decode(PD_SPACE.default_x()) == pytest.approx({"kp": 1.5, "kd": 0.1, "deadband_deg": 2.0})
    assert PD_SPACE.decode(np.zeros(3)) == pytest.approx({"kp": 0.5, "kd": 0.0, "deadband_deg": 0.0})
    assert PD_SPACE.decode(np.ones(3)) == pytest.approx({"kp": 4.0, "kd": 1.0, "deadband_deg": 4.0})
    pd = PredPD.from_x(PD_SPACE.encode({"kp": 2.0, "kd": 0.0, "deadband_deg": 1.0}), cfg=CFG)
    for bearing in (-18.0, -10.0, 10.0, 18.0):
        pd.reset()
        y = drive(pd, bearing, seconds=1.0)
        assert y[-1] == pytest.approx(math.copysign(2.0 * (abs(bearing) - 1.0) - 4.0, bearing), abs=0.05)
    pd.reset()
    drive(pd, 4.0, seconds=1.0)
    assert pd.stats["yaw_raw"] == pytest.approx(6.0, abs=0.05)  # 2 stick after the deadband: under the hysteresis
    assert pd.stats["yaw"] == 0.0
    # the same calls give the fly and the PD the same prediction; kd acts on the predicted bearing rate (deg/s)
    fly, pd = make_fly_steer("FLY-YAW-PREV", cfg=CFG), PredPD({"kp": 1.0, "kd": 0.5, "deadband_deg": 0.0}, cfg=CFG)
    for i in range(20):
        t = 30.0 + 0.1 * i
        for src in (fly, pd):
            src.yaw(t, t - 0.03, 2.0 + 10.0 * (t - 0.28 - 30.0), 1.0, FALLBACK)
    assert pd.stats["bearing_pred_deg"] == fly.stats["bearing_pred_deg"] == pytest.approx(2.0 + 10.0 * 1.9, abs=0.3)
    assert pd.stats["yaw_raw"] == pytest.approx(pd.stats["bearing_pred_deg"] + 0.5 * 10.0, abs=0.3)
    assert pd.stats["ticks"] == fly.stats["ticks"] == 39
    path = tmp_path / "pd.json"
    path.write_text(json.dumps({"arm": "PRED-PD", "x": [0.5, 0.5, 0.5]}))
    assert PredPD.from_file(path).params == pytest.approx(PD_SPACE.decode(np.full(3, 0.5)))
    msgs = []
    bad = PredPD(cfg=CFG, log_fn=msgs.append)
    bad.kp = float("nan")
    assert bad.yaw(1.0, 0.97, 5.0, 1.0, 6.5) == 6.5 and bad.failed and len(msgs) == 1  # shared fail-safe


def test_closed_loop_moss_walk_with_the_fly_steering(monkeypatch):
    """test_follow's kinematic Moss St walk (straight, curb stop, left turn) with FLY-YAW-PREV deciding yaw: the walker
    stays in frame and the follow checks there all hold."""
    import test_follow

    cfg = FollowCfg(behind_m=1.0, above_m=0.30)
    fly = make_fly_steer("FLY-YAW-PREV", cfg=cfg)

    def with_fly(*a, **k):
        return FollowController(*a, yaw_source=fly, **k)

    monkeypatch.setattr(test_follow, "FollowController", with_fly)
    log = test_follow.run_sim(speed=0.5, max_rc=50, cfg=cfg, seed=0)
    test_follow.check_sim(log, cfg)
    bearing = np.array([r["bearing"] for r in log if r["t"] > 3.0])
    assert np.sqrt(np.mean(bearing ** 2)) < 6.0 and fly.failed is None and fly.stats["ticks"] > 500


def test_non_finite_values_never_read_as_a_zero_stick(monkeypatch, tmp_path):
    """NaN rates from the brain, NaN parameters, a NaN clock: the fallback (or a refusal at construction), never a
    silent 0 stick."""
    assert math.isnan(soft_deadzone(float("nan"), 2.0)) and soft_deadzone(1.0, 2.0) == 0.0
    msgs = []
    fly = make_fly_steer("FLY-YAW-PREV", cfg=CFG, log_fn=msgs.append)
    drive(fly, 6.0, seconds=0.5)
    monkeypatch.setattr(fly.brain, "tick", lambda *a, **k: np.full(10, np.nan))
    assert fly.yaw(51.0, 50.97, 6.0, 1.0, 8.5) == 8.5 and "FloatingPointError" in fly.failed and len(msgs) == 1
    pf = load_params(PREV_PARAMS)
    with pytest.raises(ValueError):
        FlySteer(pf.params | {"dec_g_yaw": float("nan")}, pf.norm, cfg=CFG)
    with pytest.raises(ValueError):
        FlySteer(pf.params, np.full(10, np.nan), cfg=CFG)
    with pytest.raises(ValueError):
        PredPD({"kp": float("nan"), "kd": 0.1, "deadband_deg": 2.0})
    with pytest.raises(ValueError):
        PD_SPACE.decode(np.array([0.5, np.nan, 0.5]))
    for bad in ({"params": pf.params | {"enc_r_max_hz": float("nan")}}, {"x": [float("nan")] + [0.5] * 33}):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad))  # json writes NaN
        with pytest.raises(ValueError):
            load_params(f)
    other = make_fly_steer("FLY-YAW-PREV", cfg=CFG, log_fn=msgs.append)
    assert other.yaw(float("nan"), 1.0, 5.0, 1.0, 3.5) == 3.5 and other.failed
    other.reset(seed="not a seed")  # never raises
    assert other.failed and other.yaw(2.0, 1.97, 5.0, 1.0, 2.5) == 2.5


def test_prediction_reaches_the_call_time_with_a_jittered_caller():
    """kite.record's loop period is 0.100 + U(0, 0.012) s: the ticks stay on their 50 ms grid, but the stick a call
    returns is computed on the walker predicted to the call time itself."""
    rng = np.random.default_rng(0)
    for src in (make_fly_steer("FLY-YAW-PREV", cfg=CFG), PredPD(cfg=CFG)):
        t, errs = 40.0, []
        for i in range(40):
            t += 0.100 + rng.uniform(0.0, 0.012)
            frame_t = t - 0.03
            src.yaw(t, frame_t, -20.0 + 10.0 * (frame_t - CFG.video_lag_s - 40.0), 1.0, FALLBACK)  # 10 deg/s sweep
            if i >= 20:
                errs.append(src.stats["bearing_pred_deg"] - (-20.0 + 10.0 * (t - 40.0)))
        assert np.max(np.abs(errs)) < 0.05, errs  # a tick-time prediction would lag t by 0-50 ms: 0-0.5 deg here
        assert src.stats["ticks"] > 2 * 40 * 0.95 and src.failed is None


def test_an_outlier_frame_still_refreshes_the_hold():
    fly = PredPD(cfg=CFG)
    drive(fly, 5.0, seconds=1.0, t0=0.0)  # last frame at 0.87
    fly.yaw(1.35, 1.32, 45.0, 1.0, FALLBACK)  # a frame 0.45 s later, gated as an outlier
    assert fly.stats["rejected"] == 1
    fly.yaw(1.65, 1.32, 45.0, 1.0, FALLBACK)  # 0.78 s after the last fused frame, 0.33 s after the outlier
    assert fly.stats["target_valid"] and fly.stats["bearing_pred_deg"] == pytest.approx(5.0, abs=0.5)
