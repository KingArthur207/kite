"""kite.fly.behind's simulator: the app's real FollowMission against a fake YOLO pose detector and a simulated Tello.

The camera geometry against kite.follow_behind's own conventions (the head-top row is CameraModel.range_to_height's
inverse; PersonEstimator reads 1.8 m from it; nose up moves the scene down and the telemetry pitch undoes it; seen
from behind the right shoulder is on the image right and facing() reads 0 / +90 / -90), the walker's segments
(stand, the turn around passing beside the drone, the step back stopping short of it, the demo stretch), the
profiles, determinism and common random numbers, the timed discrete moves, the fly brain from a candidate x (and the
bounded brain cache), the tuned follow at its defaults being the flown one exactly, the flown follow on demo, a
0.6 m/s person outwalking its stick-35 cap, and the governor's 2.3 m ceiling. Runs headless, no device.
"""

import math

import numpy as np
import pytest

from kite import follow_tuned as tuned
from kite.follow_behind import BehindCfg, CameraModel, PersonEstimator
from kite.fly.behind import SELECTION_SEEDS, TEST_SEEDS, profiles as pf
from kite.fly.behind.pose import SRC_PERSON, SRC_SECOND, Body, Camera3, FakePose, Snap, head_in_view, view
from kite.fly.behind.walker import EV_STEP_BACK, EV_STOP, EV_TURN_AROUND, Person, WalkCfg
from kite.fly.behind.world import SimTello, episode_cfg, run_behind_episode
from kite.fly.sim.drone import DroneModel, DroneParams

QUIET = {"box_noise_px": 0.0, "kp_noise_px": 0.0, "det_base_p": 1.0, "burst_rate_hz": 0.0, "false_box_p": 0.0,
         "swap_p": 0.0, "fx_px": 1077.0, "mount_pitch_deg": 0.0, "shoulder_m": 0.40, "person_h_m": 1.70}
STILL = {"speed_lo": 0.0, "speed_hi": 0.0, "stretch_mps": 0.0, "p_step_back": 0.0}  # the person never walks
CAM = CameraModel(960, 720, 1077.0, 1077.0, 480.0, 360.0)


def ranges(res) -> np.ndarray:
    ph = res.trace["phys"]
    return np.hypot(np.asarray(ph["px"]) - np.asarray(ph["x"]), np.asarray(ph["py"]) - np.asarray(ph["y"]))


def window(res, a: float = 3.0) -> np.ndarray:
    return np.asarray(res.trace["phys"]["t"]) >= a


def quiet_pose(n: int = 64, overrides: dict | None = None, q_body: Body | None = None) -> FakePose:
    bp = pf.draw(0, "demo", {**QUIET, **(overrides or {})})
    return FakePose(bp, pf.stream(0, "demo", pf.S_POSE), pf.stream(0, "demo", pf.S_SECOND),
                    [j / 30.0 for j in range(n)], Body(bp.person_h_m, bp.shoulder_m), q_body)


def snap(r: float = 1.8, z: float = 1.9, pitch: float = 0.0, facing_deg: float = 0.0, q=(math.nan, math.nan, 0.0)):
    """The drone at the origin looking north, the person r m ahead facing `facing_deg` (0 = away)."""
    return Snap(0.0, 0.0, z, 0.0, pitch, 0.0, r, math.radians(facing_deg), *q)


# ------------------------------------------------------------------ geometry
def test_head_top_row_is_range_to_height_inverse_and_head_elev_reads_the_range():
    cam = Camera3(snap(), 1077.0, 0.0)
    col, row, _ = cam.project(0.0, 1.8, 1.70)
    assert col == pytest.approx(480.0)
    assert row == pytest.approx(360.0 + 1077.0 * 0.2 / 1.8)
    assert CAM.range_to_height(col, row, 0.2) == pytest.approx(1.8, rel=1e-9)
    dets, srcs = quiet_pose().detect(0, snap())
    assert srcs == [SRC_PERSON]
    d = dets[0]
    assert d.bbox[1] == pytest.approx(row, abs=0.5) and d.bbox[3] == 720.0  # the body runs out of the frame bottom
    est = PersonEstimator(1.70, 0.40, 0.48, height_sd_m=0.02)
    assert est.ranges(d, CAM, 1.9, 0.0)["head_elev"] == pytest.approx(1.8, rel=0.01)
    assert est.estimate(d, CAM, 1.9, 0.0).range_m == pytest.approx(1.8, rel=0.02)


def test_nose_up_moves_the_scene_down_and_the_telemetry_pitch_undoes_it():
    pose = quiet_pose()
    level = pose.detect(0, snap())[0][0]
    up = pose.detect(1, snap(pitch=5.0))[0][0]
    assert up.bbox[1] - level.bbox[1] > 80.0  # ~fx tan(5 deg)
    est = PersonEstimator(1.70, 0.40, 0.48, height_sd_m=0.02)
    assert est.ranges(up, CAM, 1.9, 5.0)["head_elev"] == pytest.approx(1.8, rel=0.01)
    assert est.ranges(up, CAM, 1.9, 0.0)["head_elev"] < 1.2  # pitch not undone: far too close


@pytest.mark.parametrize("facing_deg, expect", [(0.0, 0.0), (90.0, 90.0), (-90.0, -90.0)])
def test_facing_from_behind_and_in_profile(facing_deg, expect):
    """Seen from behind the person's RIGHT shoulder is on the image right; facing() reads 0 for a back view, +90 for a
    person facing image-right (here: east, the drone looking north), -90 for image-left."""
    pose = quiet_pose(200)
    est = PersonEstimator(1.70, 0.40, 0.48, height_sd_m=0.02)
    got = []
    for j in range(200):
        dets, _ = pose.detect(j, snap(facing_deg=facing_deg))
        d = dets[0]
        if facing_deg == 0.0:
            assert d.keypoints[6, 0] > d.keypoints[5, 0]  # right shoulder (6) right of the left (5)
        f, conf = est.facing(d, CAM, 1.8)
        if f is not None and conf >= 0.5:
            got.append(f)
    assert len(got) > 100
    assert float(np.median(got)) == pytest.approx(expect, abs=12.0)


def test_detections_of_two_people_carry_their_sources_and_the_far_head_can_be_hidden():
    pose = quiet_pose(8, q_body=Body(1.85, 0.42))
    # second person 3.0 m ahead and 1 m to the right, facing west: both detected, sources kept per detection
    dets, srcs = pose.detect(0, snap(q=(1.0, 3.0, math.radians(-90.0))))
    assert sorted(srcs) == [SRC_PERSON, SRC_SECOND] and len(dets) == 2
    assert [d.conf for d in dets] == sorted((d.conf for d in dets), reverse=True)
    # a taller second person right between the drone and the person hides the person's head
    _, srcs = pose.detect(1, snap(r=2.5, q=(0.0, 1.0, math.radians(90.0))))
    assert srcs == [SRC_SECOND]
    # ... a shorter one does not: the head shows over theirs
    short = quiet_pose(8, q_body=Body(1.55, 0.40))
    _, srcs = short.detect(1, snap(r=2.5, q=(0.0, 1.5, math.radians(90.0))))
    assert sorted(srcs) == [SRC_PERSON, SRC_SECOND]
    # FakePose as Perception's detect callable: the frame load()ed last, its sources kept by seq
    pose.load(7, 2, snap())
    assert len(pose(np.zeros((720, 960, 3), np.uint8))) == 1 and pose.sources[7] == [SRC_PERSON]


def test_view_and_in_view_edges():
    cam = Camera3(snap(), 1077.0, 0.0)
    assert head_in_view(cam, 0.0, 1.8, 1.70)
    assert not head_in_view(cam, 0.0, 0.5, 1.70)  # under the camera: out of the bottom
    assert not head_in_view(cam, 0.0, -1.0, 1.70)  # behind
    v = view(cam, Body(1.70, 0.40), 0.0, 1.8, 0.0)
    assert v.visible and v.cut[3] and not v.cut[1]  # the head top in frame, the body cut by the bottom edge
    # the box across the arms is 1.2 x the shoulder width from behind (BehindCfg.body_width_m 0.48 for 0.40)
    w = v.bbox[2] - v.bbox[0]
    assert w == pytest.approx(1077.0 * 0.48 / 1.8, rel=0.12)


# ------------------------------------------------------------------ walker and profiles
def test_turn_around_walks_back_past_the_drone_beside_it():
    cfg = WalkCfg(p_straight=0.0, p_corner=0.0, p_stop=0.0, p_speed=0.0, p_turn_around=1.0)
    p = Person(np.random.default_rng(1), 0.0, 3.0, 0.0, 1100, 0.5, 0.5, 1.0, 0.0, None, cfg)
    closest, events = math.inf, set()
    for _ in range(1100):
        p.step(0.0, 0.0)  # the drone hovers at the origin
        closest = min(closest, math.hypot(p.x, p.y))
        events.add(p.event)
    assert EV_TURN_AROUND in events and p.drone_dependent
    assert 0.6 < closest < 1.5  # it passes 0.7-1.3 m beside the drone, never through it
    assert p.y < 0.0  # and on past it


def test_step_back_moves_toward_the_drone_and_stops_short():
    cfg = WalkCfg(step_window=(0.05, 0.06), step_m=(0.8, 0.8), step_speed_mps=(0.6, 0.6))
    p = Person(np.random.default_rng(2), 0.0, 1.2, 0.0, 400, 0.0, 0.0, 1.0, 1.0, None, cfg)  # standing still
    d0, dmin, seen = 1.2, math.inf, False
    for _ in range(400):
        p.step(0.0, 0.0)
        dmin = min(dmin, math.hypot(p.x, p.y))
        seen |= p.event == EV_STEP_BACK
    assert seen and dmin < d0 - 0.2
    assert dmin >= WalkCfg().step_stop_m - 0.01  # stops 0.8 m from the drone, short of the planned 0.8 m step
    assert math.degrees(p.facing) == pytest.approx(0.0, abs=1e-9)  # facing away throughout


def test_walk_plan_stand_speeds_and_the_demo_stretch():
    p = Person(np.random.default_rng(3), 0.0, 2.0, 0.0, 6000, 0.45, 0.45, 2.0, 0.0, (0.65, 0.3, 8.0))
    sp, ev = [], []
    for _ in range(6000):
        p.step(0.0, -50.0)
        sp.append(p.speed)
        ev.append(p.event)
    assert max(sp[:199]) == 0.0 and ev[0] == EV_STOP  # standing for stand_s
    walking = [s for s, e in zip(sp, ev) if e not in (EV_STOP, EV_TURN_AROUND, EV_STEP_BACK)]
    assert max(sp) == pytest.approx(0.65) and max(walking) <= 0.65 + 1e-9
    assert sum(abs(s - 0.65) < 1e-9 for s in sp) * 0.01 > 3.0  # one stretch at 0.65, seconds long
    assert all(abs(b - a) <= 0.8 * 0.01 + 1e-9 for a, b in zip(sp, sp[1:]))  # 0.8 m/s^2 at most


def test_profiles_are_deterministic_and_override_only_what_they_name():
    assert pf.PROFILES == ("train", "demo", "fast", "slow", "stress")
    a, b = pf.draw(11, "train"), pf.draw(11, "train")
    assert a == b and pf.draw(12, "train") != a
    o = pf.draw(11, "train", {"speed_lo": 0.6, "speed_hi": 0.6})
    assert (o.speed_lo, o.speed_hi) == (0.6, 0.6) and o.fx_px == a.fx_px and o.person_h_m == a.person_h_m
    d = pf.draw(5, "demo")
    assert (d.fwd_gain_mps, d.video_latency_s, d.speed_lo, d.stretch_mps, d.second_person) == (0.96, 0.25, 0.35, 0.6,
                                                                                               False)
    assert 0.5 <= pf.draw(5, "fast").speed_lo and pf.draw(5, "slow").speed_hi <= 0.35
    assert 0.35 <= pf.draw(5, "stress").video_latency_s <= 0.5
    with pytest.raises(ValueError):
        pf.draw(1, "train", {"no_such_field": 1.0})
    with pytest.raises(ValueError):
        pf.draw(1, "street")
    s0 = pf.stream(3, "train", pf.S_WALK).random()
    assert s0 == pf.stream(3, "train", pf.S_WALK).random() != pf.stream(3, "train", pf.S_WALK, attempt=1).random()
    assert not set(SELECTION_SEEDS) & set(TEST_SEEDS)


def test_episode_cfg_sets_the_operator_height_on_a_copy():
    cfg = BehindCfg(steering="pid")
    bp = pf.draw(4, "train", {"person_h_m": 1.87, "height_err_m": 0.0})
    c = episode_cfg(cfg, bp)
    assert (cfg.person_height_m, cfg.altitude_m) == (1.70, 1.9)  # untouched
    assert c.person_height_m == 1.87 and c.altitude_m >= 1.87 + 0.2 and c.altitude_m == pytest.approx(2.07)
    c.check()


# ------------------------------------------------------------------ the drone
def test_sim_tello_discrete_moves_are_timed_and_rc_is_ignored_meanwhile():
    sim, model = SimTello(), DroneModel(DroneParams(), (), 0.01)
    model.reset(0.0, 0.0, 1.5, 0.0)
    sim.move("up", 50)
    sim.rc(0, 30, 0, 0)
    assert sim.busy() and sim.cmd == (0, 0, 0, 0) and sim.last_result() is None
    n = 0
    while sim.busy():
        model.step(0.0, 0.0, 0.0, 0.0)
        sim.advance(model, 0.01)
        n += 1
    assert n == pytest.approx(100, abs=1) and model.z == pytest.approx(2.0, abs=0.01)  # 0.5 m at 0.5 m/s
    assert sim.last_result() == "ok"
    sim.rotate(30)
    n = 0
    while sim.busy():
        model.step(0.0, 0.0, 0.0, 0.0)
        sim.advance(model, 0.01)
        n += 1
    assert n == pytest.approx(50, abs=1) and math.degrees(model.heading) == pytest.approx(30.0, abs=0.5)


# ------------------------------------------------------------------ episodes
def test_same_seed_same_episode():
    cfg = BehindCfg(steering="pid")
    a = run_behind_episode(21, "train", cfg, episode_s=20.0, record=True)
    b = run_behind_episode(21, "train", cfg, episode_s=20.0, record=True)
    assert a.trace["phys"] == b.trace["phys"] and a.trace["ctrl"] == b.trace["ctrl"]
    assert a.metrics.keys() == b.metrics.keys()
    for k in a.metrics:
        assert a.metrics[k] == b.metrics[k] or (math.isnan(a.metrics[k]) and math.isnan(b.metrics[k])), k
    assert run_behind_episode(22, "train", cfg, episode_s=20.0, record=True).trace["phys"] != a.trace["phys"]


def test_common_random_numbers_same_walk_and_clock_for_two_follows():
    """Only the drone's path depends on the arm: the person's walk (no step back or turn around in this episode)
    and the control clock are the same for two different follows."""
    for seed in range(40):
        meta = run_behind_episode(seed, "train", BehindCfg(steering="pid"), episode_s=20.0, record=True)
        kinds = {k for k, _, _ in meta.trace["meta"]["segments"]}
        if not kinds & {"step back", "turn around"} and not meta.trace["info"]["collided"]:
            break
    else:
        pytest.fail("no episode without a drone-dependent walk segment")
    a = meta
    b = run_behind_episode(seed, "train", BehindCfg(steering="pid", dist_gain=80.0, yaw_gain=2.0), episode_s=20.0,
                           record=True)
    if b.trace["info"]["collided"]:
        pytest.skip("the other follow collided: traces end early")
    pa, pb = a.trace["phys"], b.trace["phys"]
    for k in ("t", "px", "py", "p_heading", "p_speed", "p_event"):
        assert pa[k] == pb[k], k
    assert a.trace["ctrl"]["t"] == b.trace["ctrl"]["t"]
    assert pa["x"] != pb["x"] or pa["heading"] != pb["heading"]  # the drones did fly differently


def test_brain_x_is_the_brain_and_is_reseeded_from_the_episode_seed():
    """The fly from smooth2's x through a candidate params file is smooth2 exactly (the brain's noise is reseeded
    from the episode seed at each follow start, whatever the FlySteer instance)."""
    import json

    from kite.follow_behind import FLY_PARAMS

    x = json.loads(FLY_PARAMS.read_text(encoding="utf-8"))["x"]
    a = run_behind_episode(31, "demo", BehindCfg(), episode_s=8.0, record=True)
    b = run_behind_episode(31, "demo", BehindCfg(), episode_s=8.0, record=True, brain_x=x)
    assert a.trace["ctrl"] == b.trace["ctrl"] and a.brain_s > 0.0 and b.brain_s > 0.0
    assert a.trace["meta"]["follow_starts"] == 1
    assert b.trace["meta"]["cfg"]["fly_params_path"] != str(FLY_PARAMS)


def test_the_brain_cache_stays_bounded():
    from kite import follow_behind as fb
    from kite.fly.behind.world import FLY_CACHE_MAX

    rng = np.random.default_rng(0)
    for _ in range(FLY_CACHE_MAX + 2):
        run_behind_episode(1, "demo", BehindCfg(), episode_s=0.5, brain_x=rng.random(34).tolist())
        assert len(fb._FLY_SHARED) <= FLY_CACHE_MAX
    with pytest.raises(ValueError):
        run_behind_episode(1, "demo", BehindCfg(), episode_s=0.5, brain_x=[0.5] * 3)


def test_tuned_follow_at_its_defaults_flies_exactly_as_the_flown_one():
    for seed in (0, 1):
        a = run_behind_episode(seed, "train", BehindCfg(steering="pid"), episode_s=30.0, record=True)
        b = run_behind_episode(seed, "train", tuned.TunedCfg(steering="pid"), episode_s=30.0, record=True)
        assert a.trace["ctrl"] == b.trace["ctrl"] and a.trace["phys"] == b.trace["phys"]
    with pytest.raises(TypeError):
        run_behind_episode(0, "train", object(), episode_s=1.0)


def test_flown_on_demo_keeps_the_person_in_view_without_a_collision():
    """FLOWN (FollowMission, BehindCfg defaults, smooth2) on demo seeds 0-9. It never collides, keeps the person in
    view and never sits on top of them. It does NOT hold 1.8 m here: demo walks 0.35 m/s (0.6 for one stretch) and
    the flown forward stick is capped at 35 (~0.34 m/s at 0.96 m/s per 100), so it falls behind; on top of that its
    approach-rate feed-forward fires on range-rate noise and backs off at -50 (see the next test)."""
    ranges_mean = []
    for seed in range(10):
        r = run_behind_episode(seed, "demo", BehindCfg(), record=True)
        m, w = r.metrics, window(r)
        assert not r.trace["info"]["collided"] and m["collided"] == 0.0, seed
        assert not r.trace["info"]["start_failed"], seed
        assert m["in_view"] > 0.8, (seed, m["in_view"])
        assert r.trace["meta"]["follow_starts"] >= 1 and r.brain_s > 0.0
        ranges_mean.append(float(ranges(r)[w].mean()))
    assert min(ranges_mean) >= 1.4, ranges_mean


def test_without_the_approach_feed_forward_the_flown_law_holds_the_standoff():
    """The simulator's geometry and noise let a follow hold 1.8 m: the flown laws with the approach-rate feed-forward
    switched off hold a standing person in the 1.5-2.1 m band and trail a 0.2 m/s walker by the P-law's lag. With it
    on, range-rate noise makes the flown follow back off at -50 and oscillate (the controller, not the sim)."""
    off = BehindCfg(steering="pid", approach_rate_mps=10.0)
    for seed in range(4):
        r = run_behind_episode(seed, "demo", off, overrides=STILL, record=True)
        mr = float(ranges(r)[window(r)].mean())
        assert 1.5 <= mr <= 2.2, (seed, mr)
        assert r.metrics["stick_fb"] < 2.0, (seed, r.metrics["stick_fb"])
        r = run_behind_episode(seed, "demo", off, overrides={**STILL, "speed_lo": 0.2, "speed_hi": 0.2}, record=True)
        mr = float(ranges(r)[window(r, 10.0)].mean())
        assert 1.6 <= mr <= 2.8, (seed, mr)


def test_a_person_walking_0_6_mps_outwalks_the_flown_cap():
    fast = {"speed_lo": 0.6, "speed_hi": 0.6, "stretch_mps": 0.0}
    for seed in (0, 1):
        r = run_behind_episode(seed, "demo", BehindCfg(steering="pid"), overrides=fast, record=True)
        assert r.metrics["far_s"] > 0.0
        assert ranges(r)[-1] > 3.0  # well behind by the end
        ph = r.trace["phys"]
        h = np.asarray(ph["heading"])
        forward = np.asarray(ph["vx"]) * np.sin(h) + np.asarray(ph["vy"]) * np.cos(h)
        assert float(np.percentile(forward, 99)) < 0.42  # stick 35 at 0.96 m/s per 100, plus drift


def test_the_governor_holds_the_2_3_m_ceiling():
    # started 0.6 m above the follow height (2.65 m): the governor pushes it down and says so
    over = {"person_h_m": 1.85, "height_err_m": 0.0, "alt_offset_m": 0.6}
    r = run_behind_episode(2, "train", BehindCfg(steering="pid"), overrides=over, episode_s=15.0, record=True)
    z, t = np.asarray(r.trace["phys"]["z"]), np.asarray(r.trace["phys"]["t"])
    assert z[0] > 2.6 and any(r.trace["ctrl"]["gov_event"][:5])
    assert z[t > 8.0].max() < 2.3 + 0.01
    # a follow that wants to fly at 2.2 m and climb 0.25 m above it on every back-off never goes through 2.3 m
    high = BehindCfg(steering="pid", altitude_m=2.3, person_height_m=1.70)
    for seed in (3, 4):
        r = run_behind_episode(seed, "train", high, overrides={"p_step_back": 1.0}, record=True)
        assert r.trace["info"]["altitude_m"] == pytest.approx(2.2)
        assert max(r.trace["phys"]["z"]) < 2.3 + 0.03


# ------------------------------------------------------------------ review fixes
def test_telemetry_pitch_error_is_an_ou_process_of_the_profiles_sd_and_correlation():
    """train draws the error's sd U(0, 0.5) deg and its correlation U(0.5, 3) s, demo 0.25 / 1.5, stress DESIGN's
    white 0.5 (one control tick); common random numbers: the normals and the tick periods are pre-drawn."""
    from kite.fly.behind.world import pitch_noise

    sds = [pf.draw(s, "train").pitch_noise_deg for s in range(200)]
    taus = [pf.draw(s, "train").pitch_noise_tau_s for s in range(200)]
    assert 0.0 <= min(sds) < 0.05 and 0.45 < max(sds) <= 0.5 and 0.5 <= min(taus) and max(taus) <= 3.0
    assert (pf.draw(3, "demo").pitch_noise_deg, pf.draw(3, "demo").pitch_noise_tau_s) == (0.25, 1.5)
    assert (pf.draw(3, "stress").pitch_noise_deg, pf.draw(3, "stress").pitch_noise_tau_s) == (0.5, 0.05)
    # appended last: every field drawn before it keeps its value
    a, b = pf.draw(11, "train"), pf.draw(11, "train", {"pitch_noise_tau_s": 2.0})
    assert a.person_h_m == b.person_h_m and a.stand_s == b.stand_s
    rng = np.random.default_rng(0)
    n = 200_000
    z, per = rng.standard_normal(n).tolist(), [0.055] * n
    slow = np.asarray(pitch_noise(pf.draw(3, "demo", {"pitch_noise_deg": 0.5, "pitch_noise_tau_s": 2.0}), z, per))
    white = np.asarray(pitch_noise(pf.draw(3, "stress"), z, per))
    assert slow.std() == pytest.approx(0.5, rel=0.1) and white.std() == pytest.approx(0.5, rel=0.02)
    lag = int(round(2.0 / 0.055))  # the correlation at one correlation time is 1/e
    assert np.corrcoef(slow[:-lag], slow[lag:])[0, 1] == pytest.approx(math.exp(-1.0), abs=0.08)
    assert abs(np.corrcoef(white[:-2], white[2:])[0, 1]) < 0.15
    assert not np.any(pitch_noise(pf.draw(3, "demo", {"pitch_noise_deg": 0.0}), z[:100], per[:100]))


def test_the_tof_reads_the_persons_shoulders_inside_its_cone():
    """The ToF's 25 deg cone below the drone sees the shoulders of a person 0.32 m off the head axis (outside the
    0.3 m collision radius), so the governor's no-descent over a person can act before a collision; farther out it
    reads the floor."""
    from kite.fly.behind.world import COLLIDE_H_M, tof_top

    body = Body(1.70, 0.40)
    z = 1.90
    top = tof_top(body, 0.32, 0.0, z)  # beside them, at shoulder width
    assert top is not None and 0.75 * 1.70 < top < 1.70 - 0.2 and 0.32 > COLLIDE_H_M
    assert tof_top(body, 0.0, 0.0, z) == pytest.approx(1.70, abs=0.03)  # right above the head
    legs = tof_top(body, 0.0, 0.45, z)  # 0.45 m behind them the cone (wider toward the floor) reaches the legs
    assert legs is not None and legs < 0.53 * 1.70
    assert tof_top(body, 0.6, 0.0, z) is None and tof_top(body, 0.0, 0.6, z) is None


def test_every_follow_start_sets_the_shared_brains_clamp():
    """kite.follow_behind.FlyYaw.load keys the shared FlySteer without fly_max_rc_yaw: two configs differing only in
    it would share the first one's clamp. The world sets the clamp at every follow start."""
    from kite import follow_behind as fbm

    for clamp in (20, 60, 20):
        run_behind_episode(5, "demo", tuned.TunedCfg(fly_max_rc_yaw=clamp, max_rc_yaw=clamp), episode_s=1.0)
        steers = [v for k, v in fbm._FLY_SHARED.items() if k[0] == BehindCfg().fly_params_path]
        assert steers and all(s.max_yaw == clamp for s in steers)


def test_dropping_the_back_away_does_not_pay_on_demo():
    """The objective review's regression: a follow that no longer backs away from a person coming closer
    (approach_rate_mps 0.5, the old top of its range) came closer to the person and scored BETTER than the flown
    follow while it made near passes. With every term but the distance at a fixed price, on demo seeds 0-19 it scores
    worse than the flown law it came from."""
    from kite.fly.behind.fitness import fitness

    base = [run_behind_episode(s, "demo", BehindCfg(steering="pid")).metrics for s in range(20)]
    noappr = [run_behind_episode(s, "demo", BehindCfg(steering="pid", approach_rate_mps=0.5)).metrics
              for s in range(20)]
    f = fitness(noappr, base)
    assert f["fitness"] > 1.0, f["scores"]
