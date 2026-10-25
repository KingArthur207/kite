"""The rendered follow sim (sim/follow): SimTello's Drone contract and safety clamps, the street scene's geometry,
renderer, oracle detector and delayed video link, and short headless runs of python -m sim.follow (oracle detector
only: no YOLO, no weights, no window, no wall-clock sleeps)."""

import csv
import json
import math
import types

import numpy as np
import pytest

import sim.follow.__main__ as sim_main
from kite.camera import Camera
from kite.fly.sim.camera import visible_box
from kite.follow import FollowCfg
from sim.follow import scene
from sim.follow.scene import (HAIR, SKY_TOP, DelayedVideo, Leg, OracleDetector, Pose, SimFrame, Walker, moss_walk,
                              render, to_drone, to_image, walker_geom)
from sim.follow.sim_tello import TAKEOFF_H_M, SimTello

CAM = Camera()  # 960 x 720, fx = fy = 920, centre (480, 360)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_tello(**kw):
    clock = Clock()
    return SimTello(clock, "the-source", **kw), clock


def run(tello, clock, secs, dt=0.01):
    for _ in range(int(round(secs / dt))):
        clock.t += dt
        tello.step(dt)


def until_idle(tello, clock, limit_s=30.0):
    """Step until the discrete command finishes; fail if it never does."""
    t_end = clock.t + limit_s
    while tello.busy() and clock.t < t_end:
        run(tello, clock, 0.1)
    assert not tello.busy(), f"still busy after {limit_s} s: {tello.log}"


def airborne(**kw):
    tello, clock = make_tello(**kw)
    tello.takeoff()
    until_idle(tello, clock)
    return tello, clock


# ------------------------------------------------------------------ SimTello
def test_takeoff_is_non_blocking_and_stops_at_the_tellos_takeoff_height():
    """takeoff() returns at once (busy, no result yet), climbs to ~1 m, then reports ok and flying."""
    tello, clock = make_tello()
    assert not tello.flying and tello.last_result() == "ok"
    tello.takeoff()
    assert tello.busy() and tello.last_result() is None and not tello.flying
    until_idle(tello, clock)
    assert tello.flying and tello.last_result() == "ok"
    assert TAKEOFF_H_M <= tello.model.z < TAKEOFF_H_M + 0.1
    assert [c for _, c in tello.log] == ["takeoff", "takeoff -> ok"]


def test_a_second_discrete_command_while_one_runs_is_refused():
    """Two discrete commands at once would abort the first on the real Tello: SimTello raises instead."""
    tello, _ = make_tello()
    tello.takeoff()
    with pytest.raises(RuntimeError, match="still running"):
        tello.rotate(90)


def test_rc_is_clamped_and_non_finite_sticks_hover():
    """Every stick is rounded and clamped to +-100; NaN / inf (a controller dividing by zero) becomes 0."""
    tello, _ = airborne()
    tello.rc(500, -500, float("nan"), float("inf"))
    assert tello._sticks == (100, -100, 0, 0)
    tello.rc(12.4, -12.6, 0, -float("inf"))
    assert tello._sticks == (12, -13, 0, 0)


def test_rc_is_ignored_while_a_discrete_command_runs():
    """A stick during takeoff / move would abort it on the Tello, so rc() must not touch the sticks then."""
    tello, _ = make_tello()
    tello.takeoff()
    tello.rc(50, 50, 50, 50)
    assert tello._sticks == (0, 0, 0, 0)


def test_sticks_on_the_ground_do_not_lift_the_drone():
    """Before takeoff the model is pinned to the ground whatever rc() says."""
    tello, clock = make_tello()
    tello.rc(0, 80, 100, 0)
    run(tello, clock, 2.0)
    assert tello.model.z == 0.0 and tello.model.vz == 0.0


def test_rc_in_flight_turns_clockwise_for_positive_yaw():
    """+yaw = clockwise (heading grows), as on the Tello."""
    tello, clock = airborne()
    h0 = tello.model.heading
    tello.rc(0, 0, 0, 100)
    run(tello, clock, 1.0)
    assert tello.model.heading > h0 + math.radians(20)


@pytest.mark.parametrize("direction,cm", [("sideways", 100), ("forward", 19), ("forward", 501), ("up", 0)])
def test_move_rejects_bad_commands_without_starting_anything(direction, cm):
    """kite.drone.base.check_move: unknown direction or a distance outside 20..500 cm never reaches the model."""
    tello, _ = airborne()
    with pytest.raises(ValueError):
        tello.move(direction, cm)
    assert not tello.busy() and tello.last_result() == "ok"


@pytest.mark.parametrize("deg", [0, 0.4, 361, -361])
def test_rotate_rejects_zero_and_more_than_a_full_turn(deg):
    """kite.drone.base.check_rotate: |deg| must round to 1..360."""
    tello, _ = airborne()
    with pytest.raises(ValueError):
        tello.rotate(deg)
    assert not tello.busy()


def test_move_forward_follows_the_body_frame():
    """'forward 100' facing east goes ~1 m east (world x), within the 5 cm completion tolerance."""
    tello, clock = airborne(heading_deg=90.0)
    x0, y0 = tello.model.x, tello.model.y
    tello.move("forward", 100)
    assert tello.busy()
    until_idle(tello, clock)
    assert tello.last_result() == "ok"
    assert tello.model.x - x0 == pytest.approx(1.0, abs=0.08)
    assert tello.model.y - y0 == pytest.approx(0.0, abs=0.08)
    assert tello.log[-2:][0][1] == "forward 100" and tello.log[-1][1] == "forward 100 -> ok"


def test_move_right_and_up():
    """'right' facing north is +x; 'up' climbs by the asked distance."""
    tello, clock = airborne()
    x0, z0 = tello.model.x, tello.model.z
    tello.move("right", 50)
    until_idle(tello, clock)
    assert tello.model.x - x0 == pytest.approx(0.5, abs=0.08)
    tello.move("up", 40)
    until_idle(tello, clock)
    assert tello.last_result() == "ok"
    assert tello.model.z - z0 == pytest.approx(0.4, abs=0.08)


def test_rotate_reaches_the_target_and_names_the_direction():
    """cw 90 then ccw 45 end within 3 deg of the target and are logged as the Tello commands."""
    tello, clock = airborne()
    tello.rotate(90)
    until_idle(tello, clock)
    assert math.degrees(tello.model.heading) == pytest.approx(90.0, abs=3.0)
    tello.rotate(-45)
    until_idle(tello, clock)
    assert math.degrees(tello.model.heading) == pytest.approx(45.0, abs=3.0)
    cmds = [c for _, c in tello.log]
    assert "cw 90" in cmds and "ccw 45 -> ok" in cmds


def test_a_move_that_overruns_its_timeout_reports_an_error():
    """A discrete command that cannot finish in time ends with 'error: timeout' and frees the drone."""
    tello, clock = airborne()
    tello.move("forward", 500)
    clock.t += 500 / 50 + 7.5 + 0.1  # past d / speed + 7.5 s
    tello.step(0.01)
    assert not tello.busy() and tello.last_result() == "error: timeout"


def test_stop_cancels_a_move_and_zeroes_the_sticks_but_not_takeoff_or_land():
    """stop() hovers now, except that takeoff and land are never cancelled (the Tello finishes them)."""
    tello, clock = airborne()
    tello.rc(0, 40, 0, 0)
    tello.stop()
    assert tello._sticks == (0, 0, 0, 0)
    tello.move("forward", 200)
    tello.stop()
    assert not tello.busy() and tello.last_result() == "ok"
    tello.land()
    tello.stop()
    assert tello.busy() and tello._pending["kind"] == "land"
    t2, _ = make_tello()
    t2.takeoff()
    t2.stop()
    assert t2.busy()


def test_land_overrides_a_move_and_repeated_land_is_one_command():
    """land() always wins over a running move; asking again while landing does not restart it."""
    tello, clock = airborne()
    tello.move("forward", 300)
    tello.land()
    tello.land()
    assert [c for _, c in tello.log].count("land") == 1
    until_idle(tello, clock)
    assert not tello.flying and tello.model.z == 0.0 and tello.last_result() == "ok"


def test_emergency_drops_the_drone_at_once():
    """emergency(): motors off, any pending command cleared, on the ground, and it reports it was sent."""
    tello, clock = airborne()
    tello.move("up", 50)
    assert tello.emergency() is True
    assert not tello.flying and not tello.busy() and tello.model.z == 0.0
    assert tello.log[-1][1] == "emergency"


def test_battery_drains_only_in_flight_at_about_ten_minutes_per_charge():
    """DRAIN_PCT_S: 100 % over ~600 s of flight; nothing on the ground."""
    tello, clock = make_tello()
    run(tello, clock, 30.0, dt=0.1)
    assert tello.battery == 100.0
    tello, clock = airborne()
    b0 = tello.battery
    run(tello, clock, 60.0, dt=0.1)
    assert b0 - tello.battery == pytest.approx(10.0, abs=0.05)


def test_telemetry_reads_like_the_tello_state_stream():
    """Whole wrapped degrees, cm height, 0.1 m/s speeds with forward as negative vx, int battery, sim clock."""
    tello, clock = make_tello(heading_deg=270.4)
    clock.t = 12.5
    m = tello.model
    m.z, m.vx, m.vy, m.vz, m.pitch_deg = 1.234, 0.0, 0.0, 0.26, -4.6
    tello.battery = 55.9
    # facing west (270 deg): moving west at 0.5 m/s is forward, moving north at 0.3 m/s is to the right
    m.vx, m.vy = -0.5, 0.3
    tel = tello.telemetry()
    assert tel.t == 12.5
    assert tel.yaw_deg == -90.0  # wrapped to -180..180, whole degrees
    assert tel.pitch_deg == -5.0 and tel.roll_deg == 0.0
    assert tel.height_m == 1.23 and tel.tof_m == 1.23
    assert tel.vx == pytest.approx(-0.5) and tel.vy == pytest.approx(0.3) and tel.vz == pytest.approx(0.3)
    assert tel.battery_pct == 55.0
    m.z = 0.05
    assert tello.telemetry().tof_m is None  # the ToF reads nothing below 10 cm


def test_frame_source_and_set_speed():
    """frame_source() is the source it was given; set_speed() takes whole cm/s."""
    tello, _ = make_tello()
    assert tello.frame_source() == "the-source"
    tello.set_speed(72.9)
    assert tello.move_speed_cm_s == 72


# ------------------------------------------------------------------ walker
def test_moss_walk_is_a_minute_and_ends_facing_north():
    """Stand, walk, curb stop, left turn, walk, stop, right turn, walk: 60 s, the two 90 deg turns cancel."""
    legs = moss_walk(0.5)
    assert sum(leg.secs for leg in legs) == pytest.approx(60.0)
    assert legs[0].v == 0.0 and legs[1].v == 0.5
    assert sum(leg.secs * leg.turn for leg in legs) == pytest.approx(0.0)
    assert [leg.secs * leg.turn for leg in legs if leg.turn] == [-90.0, 90.0]


def test_walker_accelerates_within_its_limit_and_stands_still_after_the_walk():
    """Unicycle: no motion while standing, |dv/dt| <= accel, heading north = +y, then stops for good."""
    w = Walker([Leg(1.0, 0.0), Leg(3.0, 1.2)], x=0.0, y=0.0, accel=1.0)
    for _ in range(100):
        w.step(0.01)
    assert (w.x, w.y, w.v) == (0.0, 0.0, 0.0)
    for _ in range(50):
        w.step(0.01)
    assert w.v == pytest.approx(0.5, abs=1e-9)  # 0.5 s at 1 m/s^2
    for _ in range(250):
        w.step(0.01)
    assert w.v == pytest.approx(1.2) and w.x == pytest.approx(0.0, abs=1e-9) and w.y > 2.0
    for _ in range(300):
        w.step(0.01)
    assert w.v == pytest.approx(0.0, abs=1e-9)
    assert w._leg().secs == math.inf


def test_walker_turns_follow_the_clockwise_heading_convention():
    """A -30 deg/s leg for 3 s is a 90 deg left turn: from north to west, i.e. moving toward -x."""
    w = Walker([Leg(3.0, 1.0, -30.0), Leg(2.0, 1.0)], x=0.0, y=0.0, accel=100.0)
    for _ in range(300):
        w.step(0.01)
    assert w.heading == pytest.approx(-90.0)
    x, y = w.x, w.y
    for _ in range(100):
        w.step(0.01)
    assert w.x - x == pytest.approx(-1.0, abs=0.02) and w.y - y == pytest.approx(0.0, abs=0.02)


def test_the_whole_moss_walk_turns_back_north():
    """After both corners the walker heads north again, displaced west by the cross-street leg."""
    w = Walker(moss_walk(0.4))
    for _ in range(int(w.duration * 100) + 200):
        w.step(0.01)
    assert w.heading == pytest.approx(0.0, abs=1e-6)
    assert w.x < -3.0 and w.y > 10.0 and w.v == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ camera geometry
def test_to_drone_maps_world_points_to_forward_right_up():
    """Heading 0 looks north (+y), right is east; heading 90 looks east, right is south."""
    pts = np.array([[0.0, 5.0, 1.0], [2.0, 0.0, 0.0]])
    north = to_drone(Pose(0.0, 0.0, 1.0, 0.0, 0.0), pts)
    assert north == pytest.approx(np.array([[5.0, 0.0, 0.0], [0.0, 2.0, -1.0]]))
    east = to_drone(Pose(0.0, 0.0, 0.0, 90.0, 0.0), np.array([[3.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    assert east == pytest.approx(np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def test_to_image_projects_like_the_tello_camera():
    """Straight ahead is the principal point; right is +u, below is +v; points behind are masked out."""
    pose = Pose(0.0, 0.0, 1.0, 0.0, 0.0)
    pts = np.array([[0.0, 4.0, 1.0], [1.0, 4.0, 1.0], [0.0, 4.0, 0.0], [0.0, -4.0, 1.0]])
    uv, ok = to_image(CAM, pose, pts)
    assert ok.tolist() == [True, True, True, False]
    assert uv[0] == pytest.approx([CAM.cx, CAM.cy])
    assert uv[1][0] == pytest.approx(CAM.cx + CAM.fx / 4.0)
    assert uv[2][1] == pytest.approx(CAM.cy + CAM.fy / 4.0)


def test_nose_up_pitch_moves_the_horizon_down():
    """+pitch = nose up: a level point straight ahead drops below the image centre."""
    uv, ok = to_image(CAM, Pose(0.0, 0.0, 1.0, 0.0, 10.0), np.array([[0.0, 50.0, 1.0]]))
    assert ok[0] and uv[0][1] == pytest.approx(CAM.cy + CAM.fy * math.tan(math.radians(10.0)), rel=1e-6)


def test_walker_geom_reads_the_follow_distance_from_the_head_top_row():
    """Camera above_m over the head top at behind_m: head-top row cy + fy * above / behind (590 px), centred."""
    cfg = FollowCfg()
    w = Walker(moss_walk(0.4), x=1.0, y=1.2)
    pose = Pose(1.0, 1.2 - cfg.behind_m, w.height_m + cfg.above_m, 0.0, 0.0)
    g = walker_geom(CAM, pose, w)
    assert g[0] == pytest.approx(CAM.cx)
    assert g[1] == pytest.approx(CAM.cy + CAM.fy * cfg.above_m / cfg.behind_m)
    assert g[1] == pytest.approx(590.0)
    assert visible_box(g) is not None
    right = walker_geom(CAM, Pose(0.5, pose.y, pose.z, 0.0, 0.0), w)  # drone 0.5 m left of the walker
    assert right[0] > CAM.cx
    assert walker_geom(CAM, Pose(1.0, 3.0, pose.z, 0.0, 0.0), w) is None  # walker behind the drone


# ------------------------------------------------------------------ rendering
def test_render_draws_sky_ground_and_the_walkers_head():
    """A 960 x 720 BGR frame: sky at the top row, the walker's hair at the projected head centre."""
    cfg = FollowCfg()
    w = Walker(moss_walk(0.4), x=1.0, y=1.2)
    pose = Pose(1.0, 1.2 - cfg.behind_m, w.height_m + cfg.above_m, 0.0, 0.0)
    img = render(CAM, pose, w)
    assert img.shape == (CAM.height, CAM.width, 3) and img.dtype == np.uint8
    assert tuple(img[0, CAM.width // 2]) == SKY_TOP
    uv, ok = to_image(CAM, pose, np.array([[w.x, w.y, w.height_m - scene.HEAD_M / 2]]))
    u, v = (int(round(c)) for c in uv[0])
    assert ok[0] and tuple(img[v, u]) == HAIR
    assert tuple(img[CAM.height - 1, CAM.width // 2]) != SKY_TOP  # ground below


def test_render_without_the_walker_in_view_has_no_hair():
    """Walker behind the drone: nothing of it is drawn (no wrap-around from behind the camera plane)."""
    w = Walker(moss_walk(0.4), x=1.0, y=1.2)
    img = render(CAM, Pose(1.0, 3.0, 2.0, 0.0, 0.0), w)
    assert not np.all(img == np.array(HAIR, np.uint8), axis=2).any()


# ------------------------------------------------------------------ oracle detector
def _frame(geom, seq=1):
    return SimFrame(np.zeros((CAM.height, CAM.width, 3), np.uint8), 0.0, seq, "sim", geom, 0.0)


def _behind_geom():
    cfg = FollowCfg()
    w = Walker(moss_walk(0.4), x=1.0, y=1.2)
    return walker_geom(CAM, Pose(1.0, 1.2 - cfg.behind_m, w.height_m + cfg.above_m, 0.0, 0.0), w)


def test_oracle_detector_returns_the_true_visible_box_without_noise():
    """noise 0, miss 0: exactly kite.fly.sim.camera.visible_box, inside the frame, confidence in the torso band."""
    g = _behind_geom()
    (d,) = OracleDetector(np.random.default_rng(0), noise_px=0.0, miss=0.0)(_frame(g))
    x1, y1, x2, y2, torso = visible_box(g)
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((x1, y1, x2, y2))
    assert 0.0 <= d.x1 < d.x2 <= CAM.width and 0.0 <= d.y1 < d.y2 <= CAM.height
    assert (0.4 <= d.conf <= 0.9) if torso else (0.25 <= d.conf <= 0.6)


def test_oracle_detector_noise_is_clipped_to_the_frame():
    """Big per-edge noise never puts a box edge outside the image."""
    det = OracleDetector(np.random.default_rng(1), noise_px=400.0, miss=0.0)
    for _ in range(50):
        (d,) = det(_frame(_behind_geom()))
        assert d.x1 >= 0.0 and d.y1 >= 0.0 and d.x2 <= CAM.width and d.y2 <= CAM.height


def test_oracle_detector_misses():
    """No geometry (walker not in view / on the ground) or a missed frame -> no detections."""
    assert OracleDetector(np.random.default_rng(0), miss=0.0)(_frame(None)) == []
    assert OracleDetector(np.random.default_rng(0), miss=1.0)(_frame(_behind_geom())) == []
    det = OracleDetector(np.random.default_rng(2), miss=0.5)
    hits = sum(bool(det(_frame(_behind_geom()))) for _ in range(400))
    assert 140 < hits < 260


# ------------------------------------------------------------------ delayed video
def test_delayed_video_releases_frames_after_the_lag_and_keeps_the_latest():
    """Frame.t is the arrival time (capture + lag); read() is the newest arrived frame, until a newer one lands."""
    video = DelayedVideo(0.25)
    img = np.zeros((4, 4, 3), np.uint8)
    video.push(img, 1.0, ("g",))
    assert not img.flags.writeable  # consumers must not draw on the shared frame
    video.advance(1.2)
    assert video.read() is None
    video.advance(1.25)
    f = video.read()
    assert (f.seq, f.t, f.t_capture, f.geom, f.source) == (1, 1.25, 1.0, ("g",), "sim")
    video.push(np.zeros((4, 4, 3), np.uint8), 1.1, None)
    video.push(np.zeros((4, 4, 3), np.uint8), 1.2, None)
    video.advance(1.30)
    assert video.read().seq == 1
    video.advance(2.0)
    assert video.read().seq == 3 and video.t_now == 2.0  # both arrived; the newest wins
    video.advance(5.0)
    assert video.read().seq == 3


# ------------------------------------------------------------------ headless runs of python -m sim.follow
BLANK = np.zeros((CAM.height, CAM.width, 3), np.uint8)


@pytest.fixture
def fast_sim(monkeypatch):
    """Skip the 30 fps rendering (the oracle reads the geometry, not the pixels) and keep handles on the walker
    and the drone the run builds."""
    made = {}

    class RecWalker(Walker):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made["walker"] = self

    class RecTello(SimTello):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made["tello"] = self

    monkeypatch.setattr(sim_main, "render", lambda cam, pose, walker: BLANK)
    monkeypatch.setattr(sim_main, "Walker", RecWalker)
    monkeypatch.setattr(sim_main, "SimTello", RecTello)
    monkeypatch.setattr(sim_main.cv2, "imshow", lambda *a: pytest.fail("a window was opened"))
    return made


def _summary(capsys):
    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):out.rindex("}") + 1])


def test_follow_run_takes_off_climbs_follows_behind_and_lands(fast_sim, capsys):
    """The whole stack (governor, SimTello, tracker, controller) keeps the drone behind the walking walker,
    never within collision range, and lands at the end: exit code 0."""
    assert sim_main.main(["--seconds", "26", "--seed", "1"]) == 0
    s = _summary(capsys)
    w, tello = fast_sim["walker"], fast_sim["tello"]
    log = s["drone_log"]
    assert log[:2] == ["takeoff", "takeoff -> ok"]
    assert log[2].startswith("up ") and log[3] == log[2] + " -> ok"
    assert log[-2:] == ["land", "land -> ok"]
    assert s["landed"] and s["collision_t"] is None
    assert w.y > 4.0  # the walker really walked north
    assert 0.8 < w.y - tello.model.y < 2.5 and abs(w.x - tello.model.x) < 0.5  # landed behind, not beside
    assert s["range_m"]["min"] > 0.8 and s["range_m"]["max"] < 2.5
    assert s["range_error_rms_m"] < 0.5
    assert s["walker_in_frame_pct"] > 90.0
    assert s["max_rc"] == 50 and s["detector"] == "oracle"


def test_follow_run_is_deterministic_for_a_seed(fast_sim, capsys):
    """Same seed, same run: sim results can be compared across code changes."""
    sim_main.main(["--seconds", "10", "--seed", "4", "--miss", "0.3"])
    a = _summary(capsys)
    sim_main.main(["--seconds", "10", "--seed", "4", "--miss", "0.3"])
    assert _summary(capsys) == a


def test_follow_run_writes_video_ticks_and_summary(fast_sim, capsys, tmp_path):
    """--out: follow.mp4, ticks.csv (one row per control tick while following) and summary.json as printed."""
    out = tmp_path / "run"
    assert sim_main.main(["--seconds", "9", "--out", str(out)]) == 0
    printed = _summary(capsys)
    assert json.loads((out / "summary.json").read_text()) == printed
    assert (out / "follow.mp4").stat().st_size > 0
    with open(out / "ticks.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["t", "walker_v", "range_true", "range_est", "state", "lr", "fb", "ud", "yaw", "alt", "gov"]
    assert len(rows) > 5 and all(len(r) == 11 for r in rows[1:])
    assert all(abs(int(v)) <= 50 for r in rows[1:] for v in r[5:9])  # sticks within --max-rc


@pytest.mark.filterwarnings("ignore:Mean of empty slice", "ignore:All-NaN slice")  # quit before any range
def test_preview_paces_on_sim_time_and_q_quits(fast_sim, capsys, monkeypatch):
    """--preview shows frames, sleeps to sim time (an injected clock: no real sleep) and q ends the run."""
    shown, sleeps, closed = [], [], []
    monkeypatch.setattr(sim_main.cv2, "imshow", lambda name, img: shown.append(img.shape))
    monkeypatch.setattr(sim_main.cv2, "waitKey", lambda ms: ord("q"))
    monkeypatch.setattr(sim_main.cv2, "destroyAllWindows", lambda: closed.append(True))
    monkeypatch.setattr(sim_main, "time", types.SimpleNamespace(monotonic=lambda: 0.0, sleep=sleeps.append))
    assert sim_main.main(["--preview"]) == 0
    assert shown == [(CAM.height, CAM.width, 3)] and closed == [True]
    assert sleeps and all(s > 0 for s in sleeps) and sleeps == sorted(sleeps)
    assert _summary(capsys)["sim_s"] < 1.0  # quit on the first frame shown


def test_yolo_detector_unavailable_exits_2_without_loading_it(monkeypatch, capsys):
    """--detector yolo with no ultralytics / weights reports why and exits 2; the detector is never built."""
    monkeypatch.setattr(sim_main, "detector_available", lambda: "no weights here")
    monkeypatch.setattr(sim_main, "YoloPersonDetector", lambda *a, **kw: pytest.fail("YOLO was constructed"))
    assert sim_main.main(["--detector", "yolo"]) == 2
    assert "no weights here" in capsys.readouterr().out


def test_a_collision_makes_the_exit_code_1(fast_sim, capsys, monkeypatch):
    """Exit code 1 when the drone came within COLLISION_M of the walker (here: a 10 m 'collision' radius)."""
    monkeypatch.setattr(sim_main, "COLLISION_M", 10.0)
    assert sim_main.main(["--seconds", "8"]) == 1
    assert _summary(capsys)["collision_t"] is not None


def test_bad_detector_choice_is_a_usage_error():
    """argparse rejects an unknown --detector."""
    with pytest.raises(SystemExit) as e:
        sim_main.main(["--detector", "magic"])
    assert e.value.code == 2
