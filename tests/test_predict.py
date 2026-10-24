"""kite.predict on its own: the walker filter on straight / stopping / turning tracks, the route prior, the planner."""

import math

import numpy as np
import pytest

from kite.predict import DronePlanner, WalkerFilter, load_route, route_turn_ahead, trail_point

SEEDS = range(6)


def feed(f, track, rng, hz=10.0, t0=0.0, noise=0.03):
    """track: [(duration s, speed m/s, turn deg/s)] at the control rate; the walker starts at 0,0 heading north.
    noise 0.03 m: 3 px detection noise at 1.2 m, as in the closed-loop sim (test_follow)."""
    R = np.diag([noise ** 2, noise ** 2])
    t, pos, h = t0, np.zeros(2), 0.0
    for dur, v, w in track:
        for _ in range(int(round(dur * hz))):
            t += 1 / hz
            h += w / hz
            pos = pos + v / hz * np.array([math.sin(math.radians(h)), math.cos(math.radians(h))])
            f.update(t, pos + rng.normal(0, noise, 2), R)
    return t, pos, h


def test_straight_walk_predicts_speed_heading_and_a_path_ahead():
    f = WalkerFilter()
    t, pos, _ = feed(f, [(4.0, 0.5, 0.0)], np.random.default_rng(0))
    p = f.predict(t, 1.5)
    assert p.speed == pytest.approx(0.5, abs=0.2) and abs(p.heading_deg) < 10 and p.p_stop < 0.2
    assert p.path[-1] - p.path[0] == pytest.approx([0, 0.75], abs=0.3)  # 1.5 s at 0.5 m/s, north
    assert np.all(np.diff(p.std) >= 0) and p.std[-1] < 0.8  # uncertainty grows along the path


def test_stop_at_the_curb_is_seen_within_half_a_second():
    f = WalkerFilter()
    t, pos, _ = feed(f, [(4.0, 0.5, 0.0), (0.5, 0.0, 0.0)], np.random.default_rng(1))
    p = f.predict(t, 1.5)
    assert p.speed < 0.15  # the stop shows in the velocity within 5 control ticks
    assert np.hypot(*(p.path[-1] - pos)) < 0.35  # it no longer predicts the walker walking on into the road
    assert abs(p.heading_deg) < 10  # still facing the way they walked


@pytest.mark.parametrize("seed", SEEDS)
def test_left_turn_bends_the_predicted_path_left(seed):
    f = WalkerFilter()
    t, pos, h = feed(f, [(3.0, 0.5, 0.0), (2.0, 0.5, -19.0)], np.random.default_rng(seed))
    p = f.predict(t, 1.5)
    assert p.turn_deg_s < -8
    d0, d1 = p.path[5] - p.path[0], p.path[-1] - p.path[5]
    assert d0[0] * d1[1] - d0[1] * d1[0] > 0  # counter-clockwise in x east / y north = turning left


def test_route_prior_finds_the_left_turn_at_the_moss_howard_corner():
    route = load_route()
    assert len(route) == 4 and np.hypot(*route[2]) < 15  # sim metres around the crossing node
    before_corner = route[1] + (route[0] - route[1]) / np.hypot(*(route[0] - route[1])) * 3.0
    assert route_turn_ahead(before_corner, route) == pytest.approx(-90, abs=25)  # NW -> SW: left
    far_back = route[0]
    assert route_turn_ahead(far_back, route) == 0.0
    assert route_turn_ahead(route[1] + [50.0, 50.0], route) is None  # GPS far off the route: no prior


@pytest.mark.parametrize("seed", SEEDS)
def test_route_prior_caps_the_predicted_turn_at_the_corner_angle(seed):
    f = WalkerFilter()
    t, _, _ = feed(f, [(3.0, 0.5, 0.0), (2.5, 0.5, -19.0)], np.random.default_rng(seed))
    assert f.turn < -8 and f.turned <= 0  # turning left (how far in lags: the rate is over a window)
    free, capped = f.predict(t, 1.5), f.predict(t, 1.5, route_turn_deg=f.turned - 5.0)
    end_heading = lambda p: math.degrees(math.atan2(*(p.path[-1] - p.path[-2])))  # noqa: E731
    # 5 deg of corner left: the capped path turns that much more, then goes straight
    assert end_heading(capped) - capped.heading_deg == pytest.approx(-5.0, abs=1.0)
    assert end_heading(free) < end_heading(capped) - 1
    assert f.predict(t, 1.5, route_turn_deg=0.0).path[-1] == pytest.approx(free.path[-1])  # no corner: no change
    assert f.predict(t, 1.5, route_turn_deg=90.0).path[-1] == pytest.approx(free.path[-1])  # other way: no change


def test_trail_point_walks_back_along_the_path():
    trail = np.array([[0, 0], [0, 1.0], [1.0, 1.0]])
    assert trail_point(trail, 0.5) == pytest.approx([0.5, 1.0])
    assert trail_point(trail, 1.5) == pytest.approx([0, 0.5])
    assert trail_point(trail, 9.0) == pytest.approx([0, 0])


def test_planner_flies_forward_behind_a_walker_and_brakes_when_too_close():
    f = WalkerFilter()
    t, pos, _ = feed(f, [(3.0, 0.5, 0.0)], np.random.default_rng(4))
    p = f.predict(t, 1.5)
    trail = np.array([[0, y] for y in np.arange(-1.5, pos[1], 0.1)])
    lim = np.array([20.0, 50.0, 30.0, 40.0])
    pl = DronePlanner()
    behind = pl.plan(p, trail, pos - [0, 1.6], 0.0, np.array([0.3, 0.0]), np.array([0, 30, 0, 0.0]), lim)
    assert behind.fb > 20 and abs(behind.yaw) < 5 and abs(behind.lr) <= 10
    assert behind.drone_path[-1][1] > behind.drone_path[0][1]  # the planned path moves on behind the walker
    close = pl.plan(p, trail, pos - [0, 0.6], 0.0, np.array([0.5, 0.0]), np.array([0, 50, 0, 0.0]), lim)
    assert close.fb < 0  # 0.6 m behind and closing: back off
    right = pl.plan(p, trail, pos - [0.5, 1.6], 0.0, np.zeros(2), np.zeros(4), lim)
    assert right.yaw > 0  # the walker is right of the drone's nose: yaw right
