"""perception.near: ground-anchored depth, corridor occupancy, the whole system on synthetic renders."""

import math

import numpy as np
import pytest

from perception.near import depth as D
from perception.near import intrinsics as I
from perception.near import occupancy, synth
from perception.near.system import NearSystem

C270 = I.BUILTIN["c270"]
PD = C270.resized(448, 252)


def _scene_depth(objs, pitch=7.0, scale=1.0, kind="metric", noise=0.0, seed=0):
    z = synth.depth_map(C270, objs, 1.2, pitch)
    rng = np.random.default_rng(seed)
    return synth.model_depth(z, rng, scale, noise, kind) if (noise or scale != 1.0 or kind != "metric") else z


# ------------------------------------------------------------------ anchoring
@pytest.mark.parametrize("scale", [0.6, 1.0, 1.7])
def test_anchor_recovers_scale_and_pitch(scale):
    raw = _scene_depth([synth.Obj("person", 4.0, 0.1, 1.7, 0.5)], pitch=7.0, scale=scale, noise=0.03)
    a = D.anchor_ground(raw, PD, 1.2, 0.0)
    assert a.ok and a.pitch == pytest.approx(7.0, abs=0.6)
    assert a.scale == pytest.approx(1 / scale, rel=0.05)
    assert a.sigma_log < 0.1


def test_anchor_affine_and_inverse_models():
    z = _scene_depth([], pitch=7.0)
    rel = 0.5 * z + 0.8
    a = D.anchor_ground(rel, PD, 1.2, 7.0, kind="relative")
    assert a.ok and np.allclose(a.apply(rel)[200], z[200], rtol=0.03)
    disp = 0.7 / z + 0.02
    b = D.anchor_ground(disp, PD, 1.2, 7.0, kind="relative_inverse")
    assert b.ok and np.allclose(b.apply(disp)[200], z[200], rtol=0.03)


def test_anchor_fails_on_a_wall():
    wall = np.full((252, 448), 2.0)
    assert not D.anchor_ground(wall, PD, 1.2, 0.0).ok


def test_anchor_filter_holds_then_forgets():
    f = D.AnchorFilter(0.0, hold=2)
    good = D.Anchor(0.8, 0.0, 6.0, 0.7, 0.05, 900, True)
    f.update(good)
    bad = D.Anchor(ok=False)
    assert f.update(bad).ok and f.update(bad).ok
    assert not f.update(bad).ok


def test_confidence_is_low_at_depth_edges_and_far():
    z = np.full((50, 50), 3.0)
    z[:, 25:] = 9.0
    c = D.confidence_map(z)
    assert c[10, 24] < 0.2 and c[10, 5] > 0.8
    assert D.confidence_map(np.full((5, 5), 39.0)).max() < 0.1


def test_depth_processor_smooths_noise_over_frames():
    z = _scene_depth([], pitch=7.0)
    rng = np.random.default_rng(1)
    holder = {}
    dp = D.DepthProcessor(D.ArrayBackend(lambda f: holder["d"]), 1.2, 0.0)
    errs = []
    for i in range(8):
        holder["d"] = z * np.exp(rng.normal(0, 0.08, z.shape))
        df = dp.process(np.zeros((720, 1280, 3), np.uint8), i * 0.1, C270, (), v_ego=0.0)
        errs.append(float(np.median(np.abs(np.log(df.z[150:] / z[150:])))))
    assert errs[-1] < 0.6 * errs[0]


def test_make_backend_specs():
    assert D.make_backend("off") is None
    with pytest.raises(SystemExit):
        D.make_backend("lidar")


# ------------------------------------------------------------------ occupancy
def _df(objs, pitch=7.0, drop=None):
    z = synth.depth_map(C270, objs, 1.2, pitch)
    if drop is not None:                  # the ground falls `depth` m beyond `at` m: rays land farther
        at, dd = drop
        vv = np.arange(252)
        yn = (vv - PD.cy) / PD.fy
        p = math.radians(pitch)
        down = yn * math.cos(p) + math.sin(p)
        with np.errstate(divide="ignore"):
            fwd = np.where(down > 0, 1.2 / down * (math.cos(p) - yn * math.sin(p)), np.inf)
            lower = np.where(down > 0, (1.2 + dd) / down, 80.0)
        rows = fwd > at
        z = np.where(rows[:, None], np.minimum(lower[:, None], 80.0), z)
    a = D.Anchor(1.0, 0.0, pitch, 0.9, 0.04, 1000, True)
    return D.DepthFrame(z.astype(np.float32), D.confidence_map(z), a, PD, z)


def test_occupancy_finds_an_unnamed_obstacle():
    hits = occupancy.occupancy(_df([synth.Obj("x", 3.0, 0.1, 1.0, 0.4)]), 1.2, 7.0, 0.5)
    ob = [h for h in hits if h.kind in ("obstacle", "wall")]
    assert ob and ob[0].forward == pytest.approx(3.0, abs=0.3)


def test_occupancy_wall_and_clear_corridor():
    hits = occupancy.occupancy(_df([synth.Obj("x", 2.5, 0.0, 2.5, 4.0)]), 1.2, 7.0, 0.5)
    assert hits[0].kind == "wall" and hits[0].forward == pytest.approx(2.5, abs=0.3)
    assert occupancy.occupancy(_df([]), 1.2, 7.0, 0.5) == []
    side = occupancy.occupancy(_df([synth.Obj("x", 3.0, 1.5, 1.0, 0.4)]), 1.2, 7.0, 0.5)
    assert side == []


def test_occupancy_finds_a_drop_off():
    hits = occupancy.occupancy(_df([], drop=(4.0, 0.6)), 1.2, 7.0, 0.5)   # the C270 sees ground from 3.1 m
    d = [h for h in hits if h.kind in ("curb", "stairs_down")]
    assert d and d[0].kind == "stairs_down" and d[0].forward == pytest.approx(4.0, abs=0.4)


def test_box_mask_excludes_named_objects():
    m = occupancy.box_mask([(640, 360, 700, 500)], (252, 448), 1280, 720, grow=0.1)
    assert m[150, 235] and not m[10, 10]


# ------------------------------------------------------------------ whole system
def _run(objs, pitch=7.0, scale=1.3, still=True, ns=None, t=0.0, seed=0):
    r = synth.render(C270, objs, 1.2, pitch, np.random.default_rng(seed), depth_scale=scale)
    holder = {"d": r.depth}
    if ns is None:
        dp = D.DepthProcessor(D.ArrayBackend(lambda f: holder["d"]), 1.2, 0.0)
        ns = NearSystem(C270, depth=dp, cam_h=1.2, still=still)
    else:
        ns.depth.backend.fn = lambda f: holder["d"]
    return ns, ns.step(synth.image_for(C270), t, r.dets)


@pytest.mark.parametrize("F, p", [(1.0, 1), (2.0, 2), (5.0, 3)])
def test_person_ahead_gets_its_priority(F, p):
    _, nf = _run([synth.Obj("person", F, 0.1, 1.72, 0.5)])
    o = nf.objects[0]
    assert o["priority"] == p and o["corridor"] == "inside"
    assert o["distance_m"] == pytest.approx(F, rel=0.15)
    ev = nf.event()
    assert ev["priority"] == p and ev["kind"] == "person" and set(ev) >= {"t", "confidence", "cues", "objects"}


def test_person_off_to_the_side_is_ignored():
    _, nf = _run([synth.Obj("person", 3.0, 1.6, 1.72, 0.5)])
    assert nf.objects[0]["corridor"] == "out" and nf.top is None and nf.event()["priority"] is None


def test_unknown_obstacle_without_a_detection_is_prioritised():
    r = synth.render(C270, [synth.Obj("box", 2.0, 0.0, 1.0, 0.6)], 1.2, 7.0, np.random.default_rng(0), depth_scale=1.2)
    dp = D.DepthProcessor(D.ArrayBackend(lambda f: r.depth), 1.2, 0.0)
    nf = NearSystem(C270, depth=dp, still=True).step(synth.image_for(C270), 0.0, [])
    assert nf.top and nf.top["kind"] == "obstacle" and nf.top["priority"] == 2


def test_fast_approach_triggers_p1_by_time_to_contact_before_2_m():
    dp = D.DepthProcessor(D.ArrayBackend(lambda f: None), 1.2, 0.0)
    ns = NearSystem(C270, depth=dp, cam_h=1.2, v_ego=1.0)
    rng = np.random.default_rng(2)
    first_p1 = None
    for i in range(30):
        t = i * 0.1
        F = 20.0 - 6.5 * t                       # a bike at 6.5 m/s closing
        r = synth.render(C270, [synth.Obj("bike", F, 0.1, 1.05, 0.6)], 1.2, 8.0, rng, depth_scale=1.2)
        for d in r.dets:
            d.tid = "b"
        ns.depth.backend.fn = lambda f, z=r.depth: z
        nf = ns.step(synth.image_for(C270), t, r.dets)
        if nf.top and nf.top["priority"] == 1:
            first_p1 = F
            break
    assert first_p1 is not None and first_p1 > 5.0     # P1 from looming while still > 5 m away (< 1.5 s)


def test_hysteresis_keeps_a_jittery_p2_steady():
    dp = D.DepthProcessor(D.ArrayBackend(lambda f: None), 1.2, 0.0)
    ns = NearSystem(C270, depth=dp, cam_h=1.2, v_ego=0.0)
    seen = []
    for i in range(40):
        ns, nf = _run([synth.Obj("person", 2.76, 0.0, 1.70, 0.5)], ns=ns, t=i * 0.1, seed=i)
        seen.append(nf.top["priority"] if nf.top else None)
    assert seen[2:].count(2) >= 37


def test_far_pavement_under_a_small_pitch_error_is_not_a_drop():
    df = _df([], pitch=7.0)
    assert occupancy.occupancy(df, 1.2, 6.2, 0.5) == []            # fitted pitch 0.8 deg off


def test_no_occupancy_without_an_anchored_ground():
    r = synth.render(C270, [synth.Obj("box", 2.0, 0.0, 1.0, 0.6)], 1.2, 7.0, np.random.default_rng(0))
    wall = np.full((252, 448), 3.0)                                # nothing ground-like: anchoring fails
    dp = D.DepthProcessor(D.ArrayBackend(lambda f: wall), 1.2, 0.0)
    nf = NearSystem(C270, depth=dp, still=True).step(synth.image_for(C270), 0.0, [])
    assert nf.top is None and not nf.anchored and r.depth is not None


def test_collision_course_ignores_a_car_turning_across():
    from perception.near.corridor import CorridorHit
    from perception.near.system import collision_course, lateral_rate
    across = [(0.1 * i, -1.0 + 0.25 * i) for i in range(10)]       # 2.5 m/s sideways
    assert lateral_rate(across) == pytest.approx(2.5)
    hit = CorridorHit(True, False, across[-1][1], 1.0, 1.5, 0.0)
    assert not collision_course(across, hit, 1.2, 0.5)
    head_on = [(0.1 * i, 0.1) for i in range(10)]
    assert collision_course(head_on, CorridorHit(True, False, 0.1, -0.2, 0.4, 0.0), 1.2, 0.5)
    assert collision_course([], hit, None, 0.5)


def test_zoo_backend_adapter(monkeypatch):
    """perception/depth_zoo.py's interface (feat/depth-shootout): get(name).estimate(frame) -> DepthResult."""
    import sys
    import types
    from dataclasses import dataclass, field

    import perception

    @dataclass
    class Result:
        depth_m: np.ndarray
        confidence: np.ndarray
        latency_ms: float = 1.0
        meta: dict = field(default_factory=dict)

    class Rel:
        metric = False

        def estimate(self, frame, intrinsics=None):
            return Result(np.full((4, 4), 0.5, np.float32), np.ones((4, 4), np.float32), meta={"disparity": True})

    fake = types.ModuleType("perception.depth_zoo")
    fake.get = lambda name, device="cpu": Rel()
    monkeypatch.setitem(sys.modules, "perception.depth_zoo", fake)
    monkeypatch.setattr(perception, "depth_zoo", fake, raising=False)
    b = D.make_backend("zoo:da2-relative-s", "cpu")
    z, c = b.estimate(np.zeros((4, 4, 3), np.uint8))
    assert b.name == "zoo:da2-relative-s" and b.kind == "relative_inverse" and z.shape == c.shape == (4, 4)
