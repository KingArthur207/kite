"""Follow-controller robustness without a drone: random-input properties of FollowController (limits, head guard,
braking, outpaced), its new terms (integral with anti-windup, lag and tilt compensation), the outpaced warning in
FollowMode, and closed-loop runs on the measured Tello plant of tests/follow_sim.py (dead time, tilt, saturation,
wind, lag jitter, dropped frames, noisy telemetry)."""

from __future__ import annotations

import math

import follow_sim as fs
import numpy as np
import pytest

import kite.follow as kf
from kite.camera import Camera
from kite.follow import FollowCfg, FollowController, FollowMode, Track
from kite.types import Telemetry

CAM = Camera()
W, H = CAM.width, CAM.height


def box_at(range_m, bearing_deg=0.0, cfg=FollowCfg(), half_w=70):
    y1 = CAM.cy + CAM.fy * cfg.above_m / range_m
    xc = CAM.cx + CAM.fx * math.tan(math.radians(bearing_deg))
    return (xc - half_w, y1, xc + half_w, H)


def ctrl(max_rc=50, **kw):
    c = FollowController(FollowCfg(**kw), CAM, max_rc=max_rc)
    c.reset(2.0)
    return c


# ------------------------------------------------------------------ properties on random inputs
@pytest.mark.parametrize("seed", range(12))
def test_random_tracks_never_break_the_limits_or_the_head_guard(seed):
    """Random ranges (0.4..6 m), bearings, missed / stale frames, noisy or missing telemetry: every command is an int
    inside the clamps; closer than min_range_m there is no forward stick and no descent; inside the standoff fb never
    goes up; forward stick rises at most slew_fb_up per second while braking is instant; lost is rc 0; outpaced has
    no fb / lr."""
    rng = np.random.default_rng(seed)
    max_rc = int(rng.choice([20, 50, 80]))
    c = ctrl(max_rc=max_rc, predict=bool(rng.random() < 0.3))
    cfg = c.cfg
    lim = np.minimum([cfg.max_lr, cfg.max_fb, cfg.max_ud, cfg.max_yaw], max_rc)
    rng_m, bearing, t, prev = 1.2, 0.0, 10.0, None
    tr = None
    for i in range(400):
        t += 0.1
        rng_m = float(np.clip(rng_m + rng.normal(0, 0.15), 0.4, 6.0))
        bearing = float(np.clip(bearing + rng.normal(0, 3), -25, 25))
        if rng.random() < 0.85:
            tr = Track(box_at(rng_m, bearing), 0.7, t, (W, H), hits=5)
        tel = None if rng.random() < 0.05 else Telemetry(
            t=t, height_m=float(rng.normal(2.0, 0.3)) if rng.random() < 0.9 else None,
            pitch_deg=float(rng.normal(0, 3)) if rng.random() < 0.8 else None)
        cmd = c.update(tr, tel, t)
        rc = np.array(cmd.rc)
        assert all(type(v) is int for v in cmd.rc)
        assert np.all(np.abs(rc) <= lim), (seed, i, cmd.rc, lim)
        if cmd.state == "lost":
            assert cmd.rc == (0, 0, 0, 0)
        if cmd.state == "outpaced":
            assert cmd.lr == 0 and cmd.fb == 0 and cmd.warning
        if cmd.range_m is not None:
            near = min(cmd.range_m, cmd.range_now_m)
            if near < cfg.min_range_m:
                assert cmd.fb <= 0 and cmd.ud >= 0, (seed, i, near, cmd.rc)
            if prev is not None and near < cfg.behind_m - cfg.db_range_m and cmd.fb > 0:
                assert cmd.fb <= prev.fb + 1, (seed, i, near, prev.fb, cmd.fb)  # +1: rounding
        if prev is not None and cmd.fb > max(prev.fb, 0):
            assert cmd.fb - max(prev.fb, 0) <= cfg.slew_fb_up * 0.1 + 1, (seed, i, prev.fb, cmd.fb)
        prev = cmd


def test_braking_is_instant_and_speeding_up_is_rate_limited():
    c = ctrl(max_rc=100, outpace_s=1e9, tilt_comp=0.0)  # synthetic boxes: the camera does not really tilt
    t = 10.0
    for i in range(40):  # walker far ahead: fb climbs at slew_fb_up
        t += 0.1
        cmd = c.update(Track(box_at(3.0), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=2.0), t)
        if i == 0:
            assert cmd.fb <= c.cfg.slew_fb_up * 0.1 + 1
    assert cmd.fb >= 50
    t += 0.1
    near = c.update(Track(box_at(0.8), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=2.0), t)
    assert near.fb <= 0  # from 50+ to <= 0 in one tick


def test_too_close_never_descends_toward_the_head():
    c = ctrl()
    c.reset(1.0)  # the altitude target is below the drone: ud would be negative
    t = 10.0
    for _ in range(10):
        t += 0.1
        cmd = c.update(Track(box_at(0.8), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=2.0), t)
    assert cmd.ud >= 0 and cmd.fb <= 0
    t += 0.1
    far = c.update(Track(box_at(1.6), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=2.0), t)
    assert far.ud < 0  # further back the altitude hold works as before


# ------------------------------------------------------------------ outpaced
def walker_receding(c, speed, seconds, t=10.0, r0=1.4, tel_h=2.0):
    """Open loop: the range grows at `speed` (the drone gets nowhere). Returns the commands and the end time."""
    out = []
    for i in range(int(seconds * 10)):
        t += 0.1
        r = r0 + speed * 0.1 * (i + 1)
        out.append(c.update(Track(box_at(min(r, 4.9), 5.0), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=tel_h), t))
    return out, t


def test_a_walker_the_drone_cannot_catch_puts_it_in_outpaced_hover_and_it_resumes():
    c = ctrl(max_rc=50)
    cmds, t = walker_receding(c, 0.6, 9.0)
    states = [x.state for x in cmds]
    assert "outpaced" in states
    first = states.index("outpaced")
    assert first * 0.1 <= 1.0 / 0.6 + c.cfg.outpace_s + 1.5  # ~1 m behind + outpace_s of no progress (+ smoothing)
    o = cmds[first]
    assert o.fb == 0 and o.lr == 0 and o.yaw > 0  # hovers in place, still turning to face the walker (bearing +5)
    assert "slow down" in o.warning and f"{c.top_speed():.2f}" in o.warning
    assert c.top_speed() == pytest.approx(50 * 0.0096)
    assert all(x.state == "outpaced" for x in cmds[first:])  # walker still pulling away: no resume
    for _ in range(30):  # stopped, but 5 m away (past max_follow_m): it keeps hovering and warning
        t += 0.1
        cmd = c.update(Track(box_at(4.9, 5.0), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=2.0), t)
        assert cmd.state == "outpaced"
    r_stop = 3.0  # the walker comes back within reach and waits
    for i in range(40):  # it resumes
        t += 0.1
        cmd = c.update(Track(box_at(min(r_stop, 4.9), 5.0), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=2.0), t)
        if cmd.state == "following":
            break
    assert cmd.state == "following" and cmd.warning is None


def test_a_walker_within_reach_is_never_outpaced():
    """0.3 m/s of range growth, open loop, fb below saturation at max_rc 100: no outpace (it is catching up)."""
    c = ctrl(max_rc=100)
    cmds, _ = walker_receding(c, 0.05, 15.0)
    assert all(x.state == "following" for x in cmds)


# ------------------------------------------------------------------ integral, lag and tilt compensation
def test_integral_takes_out_a_steady_lag_and_freezes_at_saturation():
    c = ctrl(ff=0.0, outpace_s=1e9, headway_s=0.0, tilt_comp=0.0, lag_comp=False)
    t = 10.0
    fbs = []
    for _ in range(120):  # a steady 0.4 m too far (wind holding it back): P alone would give 35 x 0.3
        t += 0.1
        fbs.append(c.update(Track(box_at(1.6), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=2.0), t).fb)
    assert fbs[-1] > fbs[10] + 10 and c._i_fb == pytest.approx(c.cfg.i_max)  # grows, then clamps
    sat = ctrl(max_rc=10, ff=0.0, outpace_s=1e9, headway_s=0.0, tilt_comp=0.0)
    for _ in range(40):  # saturated at max_rc 10: the integral must not wind up behind the clamp
        t += 0.1
        sat.update(Track(box_at(2.5), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=2.0), t)
    assert sat._i_fb < 2.0
    t += 1.0
    lost = sat.update(None, Telemetry(t=t, height_m=2.0), t)
    assert lost.state == "lost" and sat._i_fb == 0.0


def test_no_forward_integral_survives_inside_the_standoff():
    c = ctrl(outpace_s=1e9)
    c._i_fb = 15.0
    t = 10.1
    c.update(Track(box_at(1.0), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=2.0), t)
    assert c._i_fb <= 0.0


def test_lag_compensation_carries_a_receding_walker_forward_to_now():
    c = ctrl(outpace_s=1e9)
    t = 10.0
    for i in range(20):  # hovering drone, walker receding at 0.4 m/s
        t += 0.1
        cmd = c.update(Track(box_at(1.3 + 0.04 * i), 0.7, t, (W, H), hits=5), Telemetry(t=t, height_m=2.0), t)
    assert cmd.range_now_m > cmd.range_m + 0.02
    assert cmd.range_now_m - cmd.range_m <= 0.5  # clamped


@pytest.mark.parametrize("tilt", [-4.0, -1.5, 0.0, 2.0, 4.0])
def test_tilt_compensation_reads_the_true_range_from_a_tilted_camera(tilt):
    body = fs.walker_points(1.7)
    c = ctrl()
    for d in (1.0, 1.5, 2.5):
        det = fs.detect(np.array([0.0, -d, 2.0]), 0.0, tilt, np.zeros(3), 0.0, body, np.random.default_rng(0),
                        fs.Scenario(noise_px=0.0))[0]
        tr = Track(det.box, 0.6, 0.0, (W, H))
        assert c.measure(tr, None, tilt)[0] == pytest.approx(d, rel=0.03)
        if abs(tilt) >= 2:
            assert abs(c.measure(tr)[0] - d) > 0.1 * d  # uncompensated, it is badly off


# ------------------------------------------------------------------ FollowMode: the outpaced warning
class StillDrone:
    flying, land_requested, hovering_for_safety = True, False, False

    def busy(self):
        return False

    def telemetry(self):
        return Telemetry(t=0.0, height_m=2.0)


def test_follow_mode_logs_and_shows_outpaced(monkeypatch):
    logs = []
    fm = FollowMode(None, lambda: None, FollowCfg(), CAM, max_rc=50,
                    log=lambda msg, quiet=False, **kw: logs.append((msg, kw)))
    fm.active, fm.ready = True, True
    fm.ctrl.reset(2.0)
    clock = [100.0]
    monkeypatch.setattr(kf.time, "time", lambda: clock[0])
    for i in range(110):
        clock[0] += 0.1
        fm.track = Track(box_at(min(1.4 + 0.07 * i, 4.9), 3.0), 0.7, clock[0], (W, H), hits=5)
        rc = fm.tick(StillDrone())
        if fm.last.state == "outpaced":
            break
    msgs = [m for m, _ in logs]
    hit = next(m for m in msgs if m.startswith("follow: OUTPACED"))
    assert "slow down" in hit and rc[1] == 0 and rc[0] == 0
    assert "TOO FAST" in fm.status()
    kw = next(k for m, k in logs if m.startswith("follow: OUTPACED"))
    assert kw["top_speed_m_s"] == pytest.approx(0.48)


# ------------------------------------------------------------------ closed loop on the measured plant
@pytest.mark.parametrize("name", ["nominal 0.4 m/s, max_rc 50", "everything at once, max_rc 70",
                                  "0.6 m/s walker, max_rc 80"])
def test_closed_loop_on_the_measured_tello_plant(name):
    """Never loses the walker (stops at the curb and at the end included), never closer than the head guard's
    min_range_m allows, never faster than the stick limit, no false 'outpaced', altitude held."""
    sc = fs.SCENARIOS[name]
    m = fs.metrics(fs.run(sc), max_rc=sc.max_rc)
    assert m["lost_ticks"] == 0 and m["outpaced_ticks"] == 0, m
    assert m["min_dist"] > 0.85, m  # the head top leaves the frame below ~0.77 m
    assert m["max_speed"] <= sc.max_rc * sc.plant.stick_m_s + math.hypot(*sc.wind_m_s) + 3 * sc.gust_m_s + 0.02, m
    assert m["max_h_err"] < 0.15, m
    assert m["range_err_mean"] < 1.3, m


def test_closed_loop_a_fast_walker_is_outpaced_and_never_chased():
    """1.2 m/s against ~0.48 m/s: 'outpaced' within a few seconds, then it hovers (fb, lr 0) and warns, and it does
    not resume while the walker is beyond max_follow_m."""
    sc = fs.SCENARIOS["1.2 m/s walker, max_rc 50 (outpaced)"]
    log = fs.run(sc)
    m = fs.metrics(log, max_rc=sc.max_rc)
    assert m["first_outpaced_s"] is not None and m["first_outpaced_s"] < 6.0, m  # walking from t = 1 s
    out = [r for r in log if r["state"] == "outpaced"]
    assert all(r["rc"][0] == 0 and r["rc"][1] == 0 for r in out) and all("slow down" in r["warning"] for r in out)
    assert m["max_speed"] <= 50 * sc.plant.stick_m_s + 0.02 and m["lost_ticks"] == 0
    assert all(r["state"] == "outpaced" for r in log if r["t"] > m["first_outpaced_s"])


def test_closed_loop_outpaced_then_followed_again_once_the_walker_stops_within_reach():
    sc = fs.SCENARIOS["1.0 m/s walker stops after 4.2 m, max_rc 50 (outpaced, resumes)"]
    log = fs.run(sc)
    m = fs.metrics(log, max_rc=sc.max_rc)
    assert m["first_outpaced_s"] is not None and m["first_outpaced_s"] < 6.0, m
    assert log[-1]["state"] == "following" and m["lost_ticks"] == 0
    assert m["min_dist"] > 0.85
    assert abs(log[-1]["dist"] - 1.2) < 0.3  # closed back in on the stopped walker and holds the standoff


def test_closed_loop_a_headwind_it_cannot_beat_is_outpaced_not_chased():
    sc = fs.SCENARIOS["headwind drift 0.3 m/s, max_rc 50 (outpaced)"]
    m = fs.metrics(fs.run(sc), max_rc=sc.max_rc)
    assert m["outpaced_ticks"] > 0 and m["first_outpaced_s"] < 12.0, m
