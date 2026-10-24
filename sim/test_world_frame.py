"""Depth ground removal (synthetic depth maps), obstacle memory, the corner car watch and the scouting drone's
frame. No model needed.  python sim/test_world_frame.py  (or pytest sim/test_world_frame.py)"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import world_frame as wf  # noqa: E402
from live_walk import center_obstacle  # noqa: E402
from standin_detect import drone_obj_to_walker  # noqa: E402

H, W, F = 252, 336, 350.0       # depth map size and fx at 336 wide (fx 1000 at 960)
CAM = 1.2


def scene(pitch=0.0, boxes=(), walls=(), curb=None, scale=1.0, flat_walls=None, fy=F):
    """Ray-cast z-depth for a camera CAM m up, pitched down `pitch` deg. Ground at height 0 (or 0.15 from `curb`
    m ahead). boxes: (lat0, lat1, fwd0, fwd1, h0, h1) in metres, + lat = right. walls: lateral offset of an
    endless wall parallel to the path. `scale` mimics a depth model that misjudges metric scale, `flat_walls` one
    that flattens a wall seen edge-on (Depth Anything on Moss read the wall 0.5 m to the side as ~1 m ahead)."""
    p = math.radians(pitch)
    yc = ((np.arange(H) - H / 2) / fy)[:, None] * np.ones((1, W))
    xc = ((np.arange(W) - W / 2) / F)[None, :] * np.ones((H, 1))
    down, fwd, lat = yc * math.cos(p) + math.sin(p), math.cos(p) - yc * math.sin(p), xc
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(down > 1e-3, CAM / down, 60.0)
        if curb is not None:
            tc = np.where(down > 1e-3, (CAM - 0.15) / down, np.inf)
            t = np.where(tc * fwd >= curb, tc, np.where(t * fwd < curb, t, np.minimum(t, curb / fwd)))
        for x0, x1, f0, f1, h0, h1 in boxes:
            lo, hi = np.full_like(t, -np.inf), np.full_like(t, np.inf)
            for d, a, b in ((lat, x0, x1), (fwd, f0, f1), (-down, h0 - CAM, h1 - CAM)):
                ta, tb = a / d, b / d
                lo, hi = np.maximum(lo, np.minimum(ta, tb)), np.minimum(hi, np.maximum(ta, tb))
            t = np.where((hi >= lo) & (lo > 0) & (lo < t), lo, t)
        for x in walls:
            tw = x / lat
            hw = CAM - tw * down
            hit = (tw > 0) & (tw < t) & (hw >= 0) & (hw <= 4)
            t = np.where(hit, np.minimum(tw, flat_walls) if flat_walls else tw, t)
    return np.minimum(t, 60.0) * scale


# ------------------------------------------------------------------ depth: ground, curbs, scale, walls
def test_ground_and_curb_are_not_obstacles():
    for pitch in (0, 15):
        for scale in (0.6, 0.8, 1.0, 1.3, 1.5):          # sim frames: 0.76x to 1.48x
            assert center_obstacle(scene(pitch, scale=scale), F, CAM, pitch) is None, (pitch, scale)
            assert center_obstacle(scene(pitch, curb=1.0, scale=scale), F, CAM, pitch) is None   # 0.15 m step


def test_bin_ahead_is_found_at_its_distance_whatever_the_scale():
    bin_ = (-0.25, 0.25, 1.5, 2.0, 0.0, 1.0)
    for pitch in (0, 15):
        for scale in (0.8, 1.0, 1.3):
            o = center_obstacle(scene(pitch, [bin_], scale=scale), F, CAM, pitch)
            assert o and abs(o["distance"] - 1.5) < 0.2 and abs(o["bearing"]) < 8, (pitch, scale, o)


def test_head_height_sign_counts_and_high_things_do_not():
    sign = (-0.5, 0.5, 1.4, 1.7, 1.45, 1.85)
    o = center_obstacle(scene(0, [sign]), F, CAM, 0)
    assert o and abs(o["distance"] - 1.4) < 0.2, o
    awning = (-0.5, 0.5, 1.4, 1.7, 2.3, 2.6)            # over the walker's head
    assert center_obstacle(scene(0, [awning]), F, CAM, 0) is None


def test_wall_beside_the_path_is_not_ahead_but_a_bin_by_it_is():
    for x in (0.5, -0.5, 0.6):
        for flat in (None, 1.3, 1.0):
            assert center_obstacle(scene(0, walls=[x], flat_walls=flat), F, CAM, 0) is None, (x, flat)
    o = center_obstacle(scene(0, [(-0.25, 0.25, 1.2, 1.7, 0, 1.0)], walls=[0.5]), F, CAM, 0)
    assert o and abs(o["distance"] - 1.2) < 0.2, o


def test_live_camera_16_9_frames_squeezed_into_the_depth_map():
    # live walk: 960x540 at fx 700, 15 deg down, resized to 336x252 -> fx 245, fy 327 in depth pixels
    fx, fy = 700 * 336 / 960, 700 * 252 / 540
    floor = scene(15, fy=fy)
    assert center_obstacle(floor, fx, CAM, 15, fy=fy) is None
    o = center_obstacle(scene(15, [(-0.25, 0.25, 1.5, 2.0, 0, 1.0)], fy=fy), fx, CAM, 15, fy=fy)
    assert o and abs(o["distance"] - 1.5) < 0.2, o


def test_camera_tilted_other_than_configured():
    # configured 15 deg down (the live walk), really level or 25 deg down: the floor is fitted, not assumed
    bin_ = (-0.25, 0.25, 1.5, 2.0, 0, 1.0)
    for real in (0, 8, 25):
        assert center_obstacle(scene(real), F, CAM, 15) is None, real
        o = center_obstacle(scene(real, [bin_]), F, CAM, 15)
        assert o and abs(o["distance"] - 1.5) < 0.25, (real, o)


def test_something_off_the_corridor_is_ignored():
    assert center_obstacle(scene(0, [(0.8, 1.3, 1.5, 2.0, 0, 1.0)]), F, CAM, 0) is None


# ------------------------------------------------------------------ frames
def test_world_walker_round_trip():
    walker = (10.0, -5.0, 315.0)
    x, y = wf.to_world(walker, 20.0, 3.0)
    b, d = wf.to_walker(walker, x, y)
    assert math.isclose(b, 20.0, abs_tol=1e-9) and math.isclose(d, 3.0, abs_tol=1e-9)


def test_scouting_drone_objects_are_turned_into_the_walkers_frame():
    # walker at the origin facing north; the drone hovers 2 m north, 3 m up, looking east (scouting a corner)
    pose = {"x": 0.0, "y": 0.0, "heading": 0.0, "cam_x": 0.0, "cam_y": 2.0, "cam_heading": 90.0}
    b, d = drone_obj_to_walker({"bearing": 0.0, "distance": 4.0}, pose)   # 4 m east of the drone
    assert math.isclose(d, math.hypot(4, 2)) and math.isclose(b, math.degrees(math.atan2(4, 2)))
    b, d = drone_obj_to_walker({"bearing": 0.0, "distance": 5.0}, {"x": 0, "y": 0, "heading": 0})
    assert math.isclose(d, 4.0) and abs(b) < 1e-9                       # following: 1 m behind, same way


def test_dead_reckoning_walks_along_the_heading():
    dr = wf.DeadReckoner(1.2)
    dr.pose(0.0, 90.0)
    p = dr.pose(2.0, 90.0)
    assert math.isclose(p["x"], 2.4) and abs(p["y"]) < 1e-9


# ------------------------------------------------------------------ memory
def walk(n, t0=0.0, y0=0.0, v=1.2, dt=0.125):
    return [(t0 + i * dt, {"x": 0.0, "y": y0 + v * i * dt, "heading": 0.0}) for i in range(n)]


def bin_at(pose, y=3.0):
    b, d = wf.to_walker(wf.walker_of(pose), 0.0, y)
    return {"id": "depth", "kind": "obstacle", "bearing": b, "distance": d, "confidence": 0.6, "cut_off": False}


def test_memory_keeps_an_obstacle_after_it_leaves_the_view():
    mem, seq = wf.ObstacleMemory(), walk(14)
    for t, p in seq[:4]:                                 # seen from 3 m to 2.55 m, then out of view
        mem.step([bin_at(p)], p, t)
    dists = []
    for t, p in seq[4:]:
        out = mem.step([], p, t)
        assert len(out) == 1 and out[0]["id"].startswith("mem:")
        dists.append(out[0]["distance"])
    assert dists == sorted(dists, reverse=True) and math.isclose(dists[-1], 3.0 - 1.2 * 13 * 0.125, abs_tol=0.01)


def test_memory_forgets_once_passed_or_after_five_seconds_of_walking():
    mem = wf.ObstacleMemory()
    seq = walk(40)                                       # 0 .. 5.85 m
    mem.step([bin_at(seq[0][1])], seq[0][1], seq[0][0])
    kept = [bool(mem.step([], p, t)) for t, p in seq[1:]]
    passed_at = next(i for i, k in enumerate(kept) if not k) + 1
    assert seq[passed_at][1]["y"] > 3.0 + wf.PASSED_M - 0.2           # dropped just after walking past it
    mem = wf.ObstacleMemory()
    p0 = {"x": 0.0, "y": 0.0, "heading": 0.0}
    mem.step([bin_at(p0, y=50.0) | {"distance": 7.0, "bearing": 0.0}], p0, 0.0)   # 7 m ahead
    slow = walk(60, v=0.25)                              # walking slowly: still "moving", ages
    alive = [bool(mem.step([], p, t + 0.125)) for t, p in slow]
    assert alive[30] and not alive[-1]                   # gone after MEMORY_S of walking


def test_memory_keeps_the_nearer_of_two_readings():
    mem, p = wf.ObstacleMemory(), {"x": 0.0, "y": 0.0, "heading": 0.0}
    mem.step([{**bin_at(p, 2.0)}], p, 0.0)
    mem.step([{**bin_at(p, 2.5)}], p, 0.1)            # a later reading biased far (depth on a close bin)
    out = mem.step([], p, 0.2)
    assert len(out) == 1 and math.isclose(out[0]["distance"], 2.0, abs_tol=0.01)


def test_memory_does_not_age_while_the_walker_stands():
    mem, p = wf.ObstacleMemory(), {"x": 0.0, "y": 1.5, "heading": 0.0}
    mem.step([bin_at(p)], p, 0.0)
    assert all(mem.step([], p, 0.5 * i) for i in range(1, 40))         # 20 s standing in front of it


def test_memory_skips_depth_sightings_where_a_person_just_was():
    mem, p = wf.ObstacleMemory(), {"x": 0.0, "y": 0.0, "heading": 0.0}
    person = {"id": "3", "kind": "person", "bearing": 0.0, "distance": 2.0, "confidence": 0.9, "cut_off": False}
    mem.step([person], p, 0.0)
    mem.step([{**bin_at(p, 2.2), "bearing": 0.0}], p, 0.5)          # depth on the same spot, person missed
    assert mem.step([], p, 1.0) == []
    mem.step([{**bin_at(p, 2.2), "bearing": 0.0}], p, 3.0)          # the person left long ago: it is a thing
    assert len(mem.step([], p, 3.2)) == 1


def test_memory_ignores_movers_and_keeps_two_depth_sightings_apart():
    mem, p = wf.ObstacleMemory(), {"x": 0.0, "y": 0.0, "heading": 0.0}
    person = {"id": "3", "kind": "person", "bearing": 0.0, "distance": 2.0, "confidence": 0.9, "cut_off": False}
    mem.step([person], p, 0.0)
    assert mem.step([], p, 0.1) == []
    mem.step([bin_at(p, 3.0)], p, 2.2)                               # the person's spot is free again
    mem.step([{**bin_at(p, 3.0), "bearing": 40.0}], p, 2.3)         # same id "depth", somewhere else
    assert len(mem.step([], p, 2.4)) == 2


def test_depth_only_obstacles_stop_from_further_out_than_known_things():
    from live_walk import DEPTH_STOP_M, RangeFilter, chest_step
    for kind, stops in (("obstacle", True), ("person", False), ("bin", False)):
        o = {"id": "a", "kind": kind, "bearing": 0.0, "distance": 2.0, "confidence": 0.6, "cut_off": False}
        assert chest_step([o], RangeFilter(), 0.0, depth_stop_m=DEPTH_STOP_M)[0]["stop"] is stops, kind
        assert chest_step([o], RangeFilter(), 0.0)[0]["stop"] is False, kind      # live: 1.6 m for everything


# ------------------------------------------------------------------ corner watch
def car_run(watch, walker, start, step, n, dt=0.125, t0=0.0):
    out = []
    for i in range(n):
        x, y = start[0] + step[0] * i, start[1] + step[1] * i
        out.append(watch.step([("c", x, y)], walker, t0 + i * dt))
    return out


def test_car_closing_on_the_crossing_holds_the_walker_until_it_has_passed():
    walker = {"x": 6.0, "y": 0.0, "heading": 315.0}           # 6 m from the crossing at the origin
    watch = wf.CornerWatch((0.0, 0.0))
    evs = car_run(watch, walker, (21.2, 21.2), (-0.53, -0.53), 12)   # 30 m out, 6 m/s toward it
    first = next(i for i, e in enumerate(evs) if e)
    assert first <= 4 and evs[-1]["state"] == "wait" and evs[-1]["kind"] == "vehicle"
    assert evs[-1]["text"] == wf.CORNER_TEXT
    last_r = math.hypot(21.2 - 0.53 * 11, 21.2 - 0.53 * 11)
    t_last = 11 * 0.125
    eta = last_r / 6.0
    assert watch.step([], walker, t_last + eta) is not None              # out of view, not there yet: hold
    assert watch.step([], walker, t_last + eta + wf.PASS_MARGIN_S + 0.2) is None


def test_no_hold_for_cars_leaving_parked_or_when_the_walker_is_far():
    walker = {"x": 6.0, "y": 0.0, "heading": 315.0}
    assert not any(car_run(wf.CornerWatch((0.0, 0.0)), walker, (5.0, 5.0), (0.53, 0.53), 12))      # leaving
    assert not any(car_run(wf.CornerWatch((0.0, 0.0)), walker, (10.0, 12.0), (0.0, 0.0), 12))      # parked
    far = {"x": 40.0, "y": -40.0, "heading": 315.0}
    assert not any(car_run(wf.CornerWatch((0.0, 0.0)), far, (21.2, 21.2), (-0.53, -0.53), 12))
    assert not any(car_run(wf.CornerWatch(None), walker, (21.2, 21.2), (-0.53, -0.53), 12))      # no route


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("world frame ok")
