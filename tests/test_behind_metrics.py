"""kite.fly.behind.metrics and .fitness on hand-built truth traces (no simulator).

Each test builds the phys / ctrl / info traces world.py hands over (a 60 s episode, 0.01 s physics, 0.05 s control)
for one situation whose numbers can be worked out by hand: a perfect follow at 1.8 m, a drift to 3 m, a stop the drone
overruns to 1.0 m, a turn around passing at 0.6 m, a ceiling excursion, the drone fast while the person is slow
(and the same speed while they are fast, or while the drone is catching up), jittery sticks, a collision, a failed
start, a lock on the second person. Then the fitness: FLOWN against itself is exactly 1.0, half the cost is 0.5, a
collision dominates, a faster-than-needed drone scores worse than a matched one. Some traces are not kinematically
consistent on purpose: the metrics read range from positions and speed from velocities, and each test isolates one."""

import math
import time

import numpy as np
import pytest

from kite.fly.behind.fitness import TERMS, FitnessCfg, cost_terms, episode_cost, episode_score, fitness
from kite.fly.behind.metrics import (BEHIND_EXTRA_METRICS, EV_SPEED, EV_STOP, EV_TURN_AROUND, LOCK_SECOND,
                                     METRICS_BEHIND, episode_metrics)

DT, CT, EP, D, ALT, H = 0.01, 0.05, 60.0, 1.8, 1.9, 1.7
N_WIN = 5701  # physics samples 3.00 .. 60.00


def base(speed: float = 0.5, ep: float = EP) -> tuple[dict, dict, dict]:
    """The person walks north at `speed` along x = 0; the drone holds 1.8 m behind, facing north, at ALT."""
    t = np.round(np.arange(int(round(ep / DT)) + 1) * DT, 10)
    n = t.size
    py = D + speed * t
    phys = {"t": t, "x": np.zeros(n), "y": py - D, "z": np.full(n, ALT), "heading": np.zeros(n),
            "pitch_deg": np.zeros(n), "vx": np.zeros(n), "vy": np.full(n, speed), "px": np.zeros(n), "py": py,
            "p_heading": np.zeros(n), "p_speed": np.full(n, speed), "p_event": np.zeros(n, dtype=int),
            "in_view": np.ones(n, dtype=bool), "qx": np.full(n, np.nan), "qy": np.full(n, np.nan)}
    tc = np.round(np.arange(int(round(ep / CT))) * CT, 10)
    m = tc.size
    ctrl = {"t": tc, "state": ["FOLLOW"] * m, "lr": np.zeros(m), "fb": np.full(m, 25.0), "ud": np.zeros(m),
            "yaw": np.zeros(m), "lock": np.ones(m, dtype=int), "seen": np.ones(m, dtype=bool),
            "gov_event": np.zeros(m, dtype=bool)}
    info = {"distance_m": D, "altitude_m": ALT, "person_height_m": H, "metric_start_s": 3.0, "episode_s": ep,
            "collided": False, "collision_t": None, "start_failed": False}
    return phys, ctrl, info


def with_gap(phys: dict, gap: np.ndarray) -> None:
    """Put the drone `gap` metres behind the person along north, with the matching velocity."""
    phys["y"] = phys["py"] - gap
    phys["vy"] = np.gradient(phys["y"], DT)


def walk(phys: dict, speed: np.ndarray) -> None:
    """The person's north speed profile (their position integrates it from 1.8 m)."""
    phys["p_speed"] = speed
    phys["py"] = D + np.concatenate(([0.0], np.cumsum(speed[:-1]) * DT))


def run(phys: dict, ctrl: dict, info: dict) -> dict:
    lists = lambda d: {k: (v.tolist() if isinstance(v, np.ndarray) else list(v)) for k, v in d.items()}
    return episode_metrics(lists(phys), lists(ctrl), dict(info))


def between(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return (t >= a - 1e-9) & (t < b - 1e-9)


# ---------------------------------------------------------------------------------------------------------- metrics


def test_perfect_follow():
    m = run(*base())
    assert set(m) == set(METRICS_BEHIND + BEHIND_EXTRA_METRICS)
    assert all(isinstance(v, float) for v in m.values())
    assert m["rms_range_err_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["in_band"] == pytest.approx(1.0) and m["in_view"] == pytest.approx(1.0)
    assert m["min_range_m"] == pytest.approx(D)
    for k in ("too_close_s", "very_close_s", "far_s", "overshoot_m", "losses", "time_lost_s", "wrong_lock_s",
              "stick_fb", "stick_yaw", "stick_ud", "stick_lr", "ceiling_s", "rms_alt_err_m", "speed_track_err",
              "excess_speed", "fast_unneeded_s", "governor_events", "collided", "start_failed", "overshoot_events"):
        assert m[k] == pytest.approx(0.0, abs=1e-9), k
    assert m["rms_bearing_deg"] == pytest.approx(0.0, abs=1e-6)
    assert m["accel_rms"] == pytest.approx(0.0, abs=1e-6)
    assert m["alt_max_m"] == pytest.approx(ALT)
    assert m["max_speed_mps"] == pytest.approx(0.5)
    assert m["following"] == pytest.approx(1.0, abs=1e-3)
    assert m["min_dist3d_m"] == pytest.approx(math.hypot(D, ALT - H))


def test_drift_to_3m():
    phys, ctrl, info = base()
    t = phys["t"]
    with_gap(phys, D + 1.2 * np.clip((t - 3.0) / 57.0, 0.0, 1.0))  # 1.8 m at 3 s to 3.0 m at 60 s, linearly
    m = run(phys, ctrl, info)
    assert m["rms_range_err_m"] == pytest.approx(1.2 / math.sqrt(3.0), rel=2e-3)
    assert m["far_s"] == pytest.approx(19.0, abs=0.03)  # > 2.6 m: the last third
    assert m["in_band"] == pytest.approx(0.25, abs=1e-3)  # <= 2.1 m: the first quarter
    assert m["too_close_s"] == 0.0 and m["overshoot_m"] == 0.0


def test_stop_overrun_to_1m():
    """The person stops 20-26 s; the drone closes from 1.8 to 1.0 m by 22 s and backs out to 1.8 m by 26 s."""
    phys, ctrl, info = base()
    t = phys["t"]
    stop = between(t, 20.0, 26.0)
    walk(phys, np.where(stop, 0.0, 0.5))
    phys["p_event"] = np.where(stop, EV_STOP, 0)
    gap = np.interp(t, [0.0, 20.0, 22.0, 26.0, 60.0], [D, D, 1.0, D, D])
    with_gap(phys, gap)
    m = run(phys, ctrl, info)
    assert m["overshoot_m"] == pytest.approx(0.5, abs=1e-6)
    assert m["overshoot_max_m"] == pytest.approx(0.5, abs=1e-6) and m["overshoot_events"] == 1.0
    assert m["min_range_m"] == pytest.approx(1.0, abs=1e-6)
    assert m["too_close_s"] == pytest.approx(1.5, abs=0.03)  # 21.5-22 s closing and 22-23 s backing out
    assert m["very_close_s"] == 0.0

    with_gap(phys, np.full(t.size, D))  # the same stop, the drone stops at 1.8 m: an event, no overshoot
    m = run(phys, ctrl, info)
    assert m["overshoot_m"] == 0.0 and m["overshoot_events"] == 1.0


def test_overshoot_counts_slow_downs_only():
    """A speed change is an event only when the new speed is lower; the drone holds 1.4 m through each."""
    for v0, v1, want in ((0.3, 0.6, 0.0), (0.6, 0.3, 0.1)):
        phys, ctrl, info = base()
        t = phys["t"]
        ramp = between(t, 20.0, 22.0)
        walk(phys, np.interp(t, [0.0, 20.0, 22.0, 60.0], [v0, v0, v1, v1]))
        phys["p_event"] = np.where(ramp, EV_SPEED, 0)
        with_gap(phys, np.interp(t, [0.0, 19.0, 20.0, 24.0, 25.0, 60.0], [D, D, 1.4, 1.4, D, D]))
        m = run(phys, ctrl, info)
        assert m["overshoot_m"] == pytest.approx(want, abs=1e-6), (v0, v1)


def test_turn_around_passing_at_0_6m():
    """The drone hovers at the origin facing north. The person waits at (0.6, 2.0), turns around 10-11 s and walks
    south at 0.5 m/s past the drone's side (closest 0.6 m at 15 s) to (0.6, -2.0) at 19 s."""
    phys, ctrl, info = base()
    t = phys["t"]
    n = t.size
    phys.update(x=np.zeros(n), y=np.zeros(n), vx=np.zeros(n), vy=np.zeros(n), px=np.full(n, 0.6))
    phys["py"] = np.interp(t, [0.0, 11.0, 19.0, 60.0], [2.0, 2.0, -2.0, -2.0])
    walking = between(t, 11.0, 19.0)
    phys["p_speed"] = np.where(walking, 0.5, 0.0)
    phys["p_heading"] = np.interp(t, [0.0, 10.0, 11.0, 60.0], [0.0, 0.0, math.pi, math.pi])
    phys["p_event"] = np.where(between(t, 10.0, 19.0), EV_TURN_AROUND, 0)
    m = run(phys, ctrl, info)
    assert m["min_range_m"] == pytest.approx(0.6, abs=1e-6)
    assert m["very_close_s"] == pytest.approx(2.0 * math.sqrt(0.8**2 - 0.36) / 0.5, abs=0.03)  # 2.12 s
    assert m["too_close_s"] == pytest.approx(2.0 * math.sqrt(1.2**2 - 0.36) / 0.5, abs=0.03)  # 4.16 s
    assert m["overshoot_m"] == pytest.approx(0.9, abs=1e-6)
    assert m["rms_bearing_deg"] > 30.0  # it never turned to them


def test_ceiling_excursion_and_governor():
    phys, ctrl, info = base()
    t, tc = phys["t"], ctrl["t"]
    up = between(t, 20.0, 22.0)
    phys["z"] = np.where(up, ALT + 0.4, ALT)
    ctrl["gov_event"] = between(tc, 20.0, 22.0)
    m = run(phys, ctrl, info)
    assert m["ceiling_s"] == pytest.approx(2.0, abs=0.011)
    assert m["alt_max_m"] == pytest.approx(ALT + 0.4)
    assert m["rms_alt_err_m"] == pytest.approx(math.sqrt(0.16 * 200 / N_WIN), rel=1e-6)
    assert m["governor_events"] == 40.0


def test_speed_only_when_needed():
    """The drone at 0.6 m/s 1.8 m behind a person walking 0.2 m/s: too fast; behind one walking 0.6 m/s: matched.
    0.6 m/s at a slow walker from 2.5 m back is not near (no excess, no tracking error) but still faster than needed
    (fast_unneeded_s counts at any range); catching up with a FAST walker from far behind is allowed."""
    phys, ctrl, info = base(speed=0.2)
    phys["vy"] = np.full(phys["t"].size, 0.6)
    m = run(phys, ctrl, info)
    assert m["excess_speed"] == pytest.approx(0.6 - 0.2 - 0.15, abs=1e-6)
    assert m["fast_unneeded_s"] == pytest.approx(57.0, abs=0.02)
    assert m["speed_track_err"] == pytest.approx(0.4, abs=1e-4)
    assert m["max_speed_mps"] == pytest.approx(0.6)

    m = run(*base(speed=0.6))
    assert m["excess_speed"] == pytest.approx(0.0, abs=1e-9) and m["fast_unneeded_s"] == 0.0
    assert m["speed_track_err"] == pytest.approx(0.0, abs=1e-9)

    phys, ctrl, info = base(speed=0.2)
    phys["y"] = phys["py"] - 2.5
    phys["vy"] = np.full(phys["t"].size, 0.6)
    m = run(phys, ctrl, info)
    assert m["excess_speed"] == 0.0 and m["speed_track_err"] == 0.0
    assert m["fast_unneeded_s"] == pytest.approx(57.0, abs=0.02)

    phys, ctrl, info = base(speed=0.6)
    phys["y"] = phys["py"] - 2.5
    phys["vy"] = np.full(phys["t"].size, 0.8)
    m = run(phys, ctrl, info)
    assert m["excess_speed"] == 0.0 and m["fast_unneeded_s"] == 0.0


def test_backing_away_and_braking_are_not_too_fast():
    """Speed is judged TOWARD the person: backing away from a standing person at 0.6 m/s, or side-stepping, is never
    excess speed; braking from 0.6 m/s after the person stops is charged only once p_recent (the most they walked
    in the last 1.5 s) is slow."""
    phys, ctrl, info = base(speed=0.0)
    phys["vy"] = np.full(phys["t"].size, -0.6)  # backing away (positions held: the metrics read velocity)
    m = run(phys, ctrl, info)
    assert m["excess_speed"] == 0.0 and m["fast_unneeded_s"] == 0.0 and m["max_speed_mps"] == pytest.approx(0.6)
    phys["vy"], phys["vx"] = np.zeros(phys["t"].size), np.full(phys["t"].size, 0.6)  # stepping sideways
    m = run(phys, ctrl, info)
    assert m["excess_speed"] == pytest.approx(0.0, abs=1e-9) and m["fast_unneeded_s"] == 0.0
    phys, ctrl, info = base(speed=0.6)
    t = phys["t"]
    walk(phys, np.where(t < 30.0, 0.6, 0.0))  # they stop at 30 s; the drone keeps 0.6 m/s until 31 s, then holds
    phys["vy"] = np.where(t < 31.0, 0.6, 0.0)
    phys["y"] = phys["py"] - D
    m = run(phys, ctrl, info)
    assert m["fast_unneeded_s"] == 0.0  # 1 s of braking: within FAST_RECENT_S of their last fast step
    phys["vy"] = np.where(t < 33.0, 0.6, 0.0)  # 3 s at 0.6 after the stop: 1.5 s of it is not needed
    m = run(phys, ctrl, info)
    assert m["fast_unneeded_s"] == pytest.approx(1.5, abs=0.03)


def test_jittery_vs_smooth_sticks():
    phys, ctrl, info = base()
    k = np.arange(ctrl["t"].size)
    ctrl["fb"] = np.where(k % 2 == 0, 20.0, 30.0)
    ctrl["yaw"] = np.where(k % 2 == 0, -5.0, 5.0)
    ctrl["ud"] = np.where(k % 2 == 0, 0.0, 2.0)
    ctrl["lr"] = np.where(k % 2 == 0, 0.0, 1.0)
    m = run(phys, ctrl, info)
    assert (m["stick_fb"], m["stick_yaw"], m["stick_ud"], m["stick_lr"]) == pytest.approx((10.0, 10.0, 2.0, 1.0))
    smooth = run(*base())
    assert smooth["stick_fb"] == smooth["stick_yaw"] == 0.0

    phys, ctrl, info = base()  # a jump into or out of a non-FOLLOW tick is not a FOLLOW stick change
    ctrl["state"] = ["REACQUIRE" if 400 <= i < 410 else "FOLLOW" for i in k]
    ctrl["fb"] = np.where((k >= 400) & (k < 410), 0.0, 25.0)
    m = run(phys, ctrl, info)
    assert m["stick_fb"] == 0.0
    assert m["following"] == pytest.approx(1.0 - 0.5 / 57.0, abs=1e-3)


def test_accel_rms_of_a_sinusoid():
    phys, ctrl, info = base()
    t = phys["t"]
    phys["vy"] = 0.5 + 0.2 * np.sin(2.0 * math.pi * t / 4.0)
    m = run(phys, ctrl, info)
    assert m["accel_rms"] == pytest.approx(0.2 * (2.0 * math.pi / 4.0) / math.sqrt(2.0), rel=0.01)


def test_losses_and_wrong_lock():
    phys, ctrl, info = base()
    tc = ctrl["t"]
    unseen = between(tc, 10.0, 10.5) | between(tc, 20.0, 21.5) | between(tc, 30.0, 31.5)
    ctrl["seen"] = ~unseen
    m = run(phys, ctrl, info)
    assert m["losses"] == 2.0  # the 0.5 s dropout is not a loss
    assert m["time_lost_s"] == pytest.approx(3.0, abs=1e-6)

    phys, ctrl, info = base()
    ctrl["lock"] = np.where(between(tc, 20.0, 25.0), LOCK_SECOND, 1)
    m = run(phys, ctrl, info)
    assert m["wrong_lock_s"] == pytest.approx(5.0, abs=1e-6)
    assert m["losses"] == 1.0 and m["time_lost_s"] == pytest.approx(5.0, abs=1e-6)

    phys, ctrl, info = base()  # a loss running into the window counts whole if it is > 1 s in all, clipped to it
    ctrl["seen"] = ~between(tc, 2.5, 3.6)
    m = run(phys, ctrl, info)
    assert m["losses"] == 1.0 and m["time_lost_s"] == pytest.approx(0.6, abs=1e-6)


def test_collision_charges_the_rest():
    """The drone closes from 1.8 m at 25 s to 0.25 m at 30 s: a collision; the traces stop there."""
    phys, ctrl, info = base()
    t = phys["t"]
    with_gap(phys, np.interp(t, [0.0, 25.0, 30.0, 60.0], [D, D, 0.25, 0.25]))
    keep = t <= 30.0 + 1e-9
    phys = {k: v[keep] for k, v in phys.items()}
    kc = ctrl["t"] < 30.0 - 1e-9
    ctrl = {k: (v[kc] if isinstance(v, np.ndarray) else [s for s, ok in zip(v, kc) if ok]) for k, v in ctrl.items()}
    info.update(collided=True, collision_t=30.0)
    m = run(phys, ctrl, info)
    assert m["collided"] == 1.0
    assert m["min_range_m"] == pytest.approx(0.25, abs=1e-6)
    assert m["time_lost_s"] == pytest.approx(30.0, abs=1e-6) and m["losses"] == 1.0
    n_ran, n_rest = 2701, 3000
    assert m["in_view"] == pytest.approx(n_ran / (n_ran + n_rest), rel=1e-6)
    assert m["in_band"] < n_ran / (n_ran + n_rest)
    assert m["rms_bearing_deg"] == pytest.approx(90.0 * math.sqrt(n_rest / (n_ran + n_rest)), rel=1e-6)
    assert m["following"] == pytest.approx(27.0 / 57.0, abs=1e-3)
    assert episode_score(m, run(*base())) == FitnessCfg().collision_score


def test_start_failed_passthrough():
    _, _, info = base()
    info["start_failed"] = True
    m = episode_metrics({k: [] for k in ("t", "x", "y")}, {"t": []}, info)
    assert m["start_failed"] == 1.0 and set(m) == set(METRICS_BEHIND + BEHIND_EXTRA_METRICS)
    phys, ctrl, info = base()
    info["start_failed"] = True
    m = run(phys, ctrl, info)
    assert m["start_failed"] == 1.0
    ok = run(*base())
    assert episode_score(ok, m) is None  # FLOWN did not start: the seed is unusable for everyone
    assert episode_score(m, ok) == FitnessCfg().max_score  # only the candidate failed: its own failure, charged


def test_episode_metrics_is_fast():
    phys, ctrl, info = base()
    k = np.arange(ctrl["t"].size)
    ctrl["fb"] = np.where(k % 2 == 0, 20.0, 30.0)
    phys["p_event"] = np.where((phys["t"] % 10.0) < 2.0, EV_STOP, 0)
    lists = lambda d: {key: (v.tolist() if isinstance(v, np.ndarray) else list(v)) for key, v in d.items()}
    p, c = lists(phys), lists(ctrl)
    episode_metrics(p, c, info)
    t0 = time.perf_counter()
    for _ in range(20):
        episode_metrics(p, c, info)
    assert (time.perf_counter() - t0) / 20 < 0.05  # ~3 ms here; an episode's own budget is well under 1 s


# ---------------------------------------------------------------------------------------------------------- fitness


def mediocre() -> dict:
    """Metrics of a mediocre FLOWN follow: every cost term near 1 (J ~ 12), nothing capped."""
    return {"collided": 0.0, "rms_range_err_m": 0.3, "rms_bearing_deg": 15.0, "time_lost_s": 4.0, "losses": 1.0,
            "stick_fb": 2.0, "stick_yaw": 1.5, "stick_ud": 1.0, "stick_lr": 0.5, "accel_rms": 0.3,
            "speed_track_err": 0.15, "excess_speed": 0.05, "fast_unneeded_s": 2.0, "very_close_s": 0.4,
            "too_close_s": 2.0, "overshoot_m": 0.2, "ceiling_s": 1.0, "wrong_lock_s": 0.5, "rms_alt_err_m": 0.05,
            "start_failed": 0.0}


def test_flown_against_itself_is_exactly_one():
    m = mediocre()
    assert episode_cost(m) > 0.5
    assert episode_score(m, m) == 1.0
    easy = {k: (0.0 if k not in ("rms_range_err_m", "rms_bearing_deg") else v) for k, v in m.items()}
    easy.update(rms_range_err_m=0.05, rms_bearing_deg=2.0)
    assert episode_cost(easy) < 0.5
    assert episode_score(easy, easy) == 1.0  # below the floor too
    better = dict(easy, rms_range_err_m=0.02)
    assert 0.0 <= episode_score(better, easy) < 1.0
    f = fitness([m, easy, m], [m, easy, m])
    assert f["fitness"] == 1.0 and f["n"] == 3 and f["excluded"] == 0 and f["collisions"] == 0
    assert set(f["parts"]) == set(TERMS)


def test_distance_is_scored_against_flown_and_everything_else_has_a_fixed_price():
    """Half FLOWN's distance cost scores 0.5; every other term costs J / q_scale on EVERY seed. The review's case:
    where FLOWN trails 7 m its J is ~130, and dividing all of J by it made a second under 0.8 m worth 0.008 there."""
    fc = FitnessCfg()
    m = mediocre()
    half = dict(m, rms_range_err_m=m["rms_range_err_m"] * math.sqrt(0.5))
    assert episode_score(half, m) == pytest.approx(0.5)
    assert fitness([half, half], [m, m])["fitness"] == pytest.approx(0.5)
    trailing = dict(m, rms_range_err_m=7.0)
    assert episode_cost(trailing) > 100.0
    for flown in (m, trailing):
        near_miss = dict(flown, very_close_s=flown["very_close_s"] + 1.0, too_close_s=flown["too_close_s"] + 1.0)
        assert episode_score(near_miss, flown) == pytest.approx(1.0 + (fc.w_very_close + fc.w_too_close) / fc.q_scale)
        fast = dict(flown, fast_unneeded_s=flown["fast_unneeded_s"] + 1.0)
        assert episode_score(fast, flown) == pytest.approx(1.0 + fc.w_fast / fc.q_scale)
    # NOAPPR's seed 13 (review): closer than a trailing FLOWN, but 5.6 s under 0.8 m and 1 m of overshoot. It scored
    # 0.995 under DESIGN's ratio; it must be clearly worse than FLOWN
    noappr = dict(trailing, rms_range_err_m=3.0, very_close_s=5.6, too_close_s=5.6, overshoot_m=1.0)
    assert episode_score(noappr, trailing) > 2.0
    assert cost_terms(m)["altitude"] == pytest.approx(fc.w_alt)  # 0.05 m RMS = 1 scale


def test_collisions_are_a_hard_constraint():
    """A collision scores 6, and any seed where the candidate collided and FLOWN did not lifts the fitness to at
    least 1 + collision_penalty: no gain elsewhere buys a collision (a flat 6 among scores of ~0.5 did)."""
    m = mediocre()
    great = dict(m, rms_range_err_m=0.1, very_close_s=0.0, too_close_s=0.0)
    crashed = dict(m, collided=1.0)
    assert episode_score(crashed, m) == 6.0
    assert episode_score(dict(m, rms_range_err_m=0.6, time_lost_s=20.0), m) < 6.0  # awful, but no collision
    f = fitness([great] * 31 + [crashed], [m] * 32)
    assert f["collisions"] == 1 and f["excess_collisions"] == 1
    assert f["mean_score"] < 1.0 and f["fitness"] == pytest.approx(2.0)
    assert fitness([m] * 32, [m] * 32)["fitness"] < f["fitness"]
    both = fitness([great] * 31 + [crashed], [m] * 31 + [crashed])  # FLOWN crashed on that seed too
    assert both["excess_collisions"] == 0 and both["fitness"] == both["mean_score"] < 1.0


def test_distance_term_goes_on_linearly_past_its_knee():
    """Square up to 0.9 m RMS, then its tangent (no plateau where the flown follow trails metres behind); "cap" is
    DESIGN's literal flat cap."""
    m = mediocre()
    d = lambda rms, **kw: cost_terms(dict(m, rms_range_err_m=rms), FitnessCfg(**kw))["distance"]  # noqa: E731
    assert d(0.9) == pytest.approx(9.0) and d(0.6) == pytest.approx(4.0)
    assert d(1.2) == pytest.approx(9.0 + 6.0 * (4.0 - 3.0))
    assert d(0.9 - 1e-6) < d(0.9) < d(0.9 + 1e-6)  # continuous through the knee, and still rising
    assert d(9.6) / d(7.6) > 1.25  # 2 m further behind is clearly worse
    assert d(9.6, range_tail="cap") == d(7.6, range_tail="cap") == 9.0
    with pytest.raises(ValueError):
        d(2.0, range_tail="log")


def test_faster_than_needed_scores_worse_than_matched():
    flown = run(*base(speed=0.2))  # matched: 0.2 m/s behind a 0.2 m/s walk
    phys, ctrl, info = base(speed=0.2)
    phys["vy"] = np.full(phys["t"].size, 0.6)
    fast = run(phys, ctrl, info)
    assert episode_cost(flown) < 0.5  # an easy seed: the floor applies
    assert episode_score(flown, flown) == 1.0
    assert episode_score(fast, flown) > 1.5
    assert cost_terms(fast)["speed_excess"] > 1.0 and cost_terms(fast)["speed_track"] > 1.0


def test_excluded_and_errors():
    m = mediocre()
    failed = dict(m, start_failed=1.0)
    err = {"error": 1.0}
    f = fitness([m, failed, err, m], [m, m, m, err])
    assert f["scores"] == [1.0, 6.0, 6.0, None]  # the candidate's own failed start is charged, FLOWN's error excluded
    assert f["n"] == 3 and f["excluded"] == 1 and f["errors"] == 1
    assert f["fitness"] == pytest.approx(13.0 / 3.0)
    assert fitness([m], [failed])["fitness"] == FitnessCfg().max_score  # nothing usable: never nan
    assert fitness([m], [failed])["excluded"] == 1
    with pytest.raises(ValueError):
        fitness([m], [m, m])
