"""Vocabulary, geometry and depth fusion. No model, no download."""

import json
import math

import numpy as np
import pytest
from conftest import plane_depth

from earpiece.events import parse
from perception import fuse, vocab
from standin_detect import distance_m

W, H, FX = 960, 540, 700.0


# ------------------------------------------------------------------ vocabulary
def test_default_vocab_flattens_without_duplicates_and_maps_every_prompt_to_its_kind():
    prompts, kind_of = vocab.flatten(vocab.DEFAULT_VOCAB)
    assert len(prompts) == len(set(prompts)) == len(kind_of)
    assert kind_of["bollard"] == "bollard" and kind_of["electric scooter"] == "e_scooter"
    assert kind_of["street light"] == "pole" and kind_of["traffic cone"] == "construction"
    assert all(k in vocab.TYPICAL_H for k in vocab.DEFAULT_VOCAB)


def test_first_kind_wins_for_a_prompt_listed_twice_and_case_is_folded():
    prompts, kind_of = vocab.flatten({"a": ["Pole", "x"], "b": ["pole ", "y"]})
    assert prompts == ["pole", "x", "y"] and kind_of["pole"] == "a"


def test_command_line_prompts_extend_and_files_replace(tmp_path):
    v = vocab.load(extra=["construction=orange fence, jersey barrier", "puddle=puddle"])
    assert v["construction"][-2:] == ["orange fence", "jersey barrier"] and v["puddle"] == ["puddle"]
    f = tmp_path / "v.json"
    f.write_text(json.dumps({"pole": ["lamp post"]}))
    assert vocab.load(f) == {"pole": ["lamp post"]}
    with pytest.raises(ValueError):
        vocab.load(extra=["no-equals-sign"])
    f.write_text("[1, 2]")
    with pytest.raises(ValueError):
        vocab.load(f)


def test_spoken_names():
    assert vocab.spoken("e_scooter") == "e-scooter" and vocab.spoken("fire_hydrant") == "fire hydrant"
    assert vocab.spoken("pole") == "pole"


# ------------------------------------------------------------------ flat ground, same as the stand-in
def det(x1, y1, x2, y2, kind="pole", conf=0.6, **kw):
    return fuse.Det(x1, y1, x2, y2, kind=kind, conf=conf, **kw)


def test_without_depth_a_grounded_box_gets_the_stand_in_distance_and_bearing():
    d = det(430, 200, 530, 400, "bollard")
    o = fuse.object_from_det(d, 0, FX, fuse.Calib(0.0, 1.2), W, H)
    assert o["range"] == "ground" and o["kind"] == "bollard"
    assert math.isclose(o["distance"], round(distance_m(400, 200, 1.2, FX, H, 0.9), 2))
    assert o["bearing"] == 0.0
    assert o["h_bottom"] == pytest.approx(0.0, abs=0.01)


def test_cut_off_box_without_depth_falls_back_to_size_and_is_flagged():
    o = fuse.object_from_det(det(300, 100, 500, 540, "person"), 3, FX, fuse.Calib(0.0, 1.2), W, H)
    assert o["cut_off"] and o["range"] == "size" and o["id"] == "3"


# ------------------------------------------------------------------ ground fit
@pytest.mark.parametrize("pitch,scale", [(0.0, 1.0), (8.0, 1.0), (15.0, 0.5), (-4.0, 1.6)])
def test_fit_ground_recovers_pitch_and_the_depth_models_scale(pitch, scale):
    fx_d = FX * 448 / W
    c = fuse.fit_ground(plane_depth(fx=fx_d, pitch=pitch, scale=scale), fx_d, 1.2)
    assert c.source == "depth"
    assert c.pitch == pytest.approx(pitch, abs=0.3)
    assert c.scale == pytest.approx(1 / scale, rel=0.03)


def test_with_a_known_pitch_only_the_scale_is_fitted():
    fx_d = FX * 448 / W
    c = fuse.fit_ground(plane_depth(fx=fx_d, pitch=8.0, scale=0.5), fx_d, 1.2, given_pitch=8.0, fit_pitch=False)
    assert c.source == "depth" and c.pitch == pytest.approx(8.0) and c.scale == pytest.approx(2.0, rel=0.03)


def test_depth_takes_over_when_the_box_bottom_is_not_the_foot():
    # a box around a scooter's stem: its bottom row (the deck) reads 50 m on flat ground, depth says 3 m
    cal = fuse.Calib(6.8, 1.2, 1.0, "depth")
    box = (385, 55, 432, 202)
    depth = with_object(plane_depth(fx=FX * 448 / W, pitch=6.8), box, 3.0)
    o = fuse.object_from_det(det(*box, "pole"), 0, FX, cal, W, H, depth)
    assert o["range"] == "depth" and o["distance"] == pytest.approx(3.0, abs=0.1)


def test_a_wide_tree_box_is_aimed_at_its_trunk():
    fx_d = FX * 448 / W
    y = H / 2 + FX * 1.2 / 5.0                          # trunk base 5 m ahead
    box = (300, 40, 700, y)                             # the crown makes the box wide, centred at x = 500
    depth = with_object(plane_depth(fx=fx_d), (600, 40, 640, y), 5.0)    # the trunk is at x 600-640
    o = fuse.object_from_det(det(*box, "tree"), 0, FX, fuse.Calib(0.0, 1.2, 1.0, "depth"), W, H, depth)
    assert o["bearing"] == pytest.approx(math.degrees(math.atan((620 - W / 2) / FX)), abs=1.5)
    assert fuse.object_from_det(det(*box, "tree"), 0, FX, fuse.Calib(0.0, 1.2), W, H)["bearing"] == \
        pytest.approx(math.degrees(math.atan((500 - W / 2) / FX)), abs=0.1)


def test_fit_ground_falls_back_when_the_bottom_of_the_frame_is_not_ground():
    wall = np.full((252, 448), 1.5, np.float32)          # standing in front of a wall
    c = fuse.fit_ground(wall, 336.0, 1.2, given_pitch=5.0)
    assert c.source == "given" and c.pitch == 5.0 and c.scale == 1.0
    excl = np.ones((252, 448), bool)                     # everything covered by detections
    assert fuse.fit_ground(plane_depth(), 336.0, 1.2, excl).source == "given"


def test_calib_filter_smooths_and_forgets():
    f = fuse.CalibFilter(0.0, 1.2, alpha=0.5, hold=2)
    assert f.update(fuse.Calib(10.0, 1.2, 1.0, "depth")).pitch == 10.0
    assert f.update(fuse.Calib(20.0, 1.2, 2.0, "depth")).pitch == 15.0
    miss = fuse.Calib(0.0, 1.2)
    assert f.update(miss).pitch == 15.0 and f.update(miss).pitch == 15.0     # held
    assert f.update(miss).source == "given"


# ------------------------------------------------------------------ depth: overhead, stairs, range choice
def with_object(depth, box, z, w=W, h=H):
    d = depth.copy()
    sy, sx = d.shape[0] / h, d.shape[1] / w
    d[int(box[1] * sy):int(box[3] * sy), int(box[0] * sx):int(box[2] * sx)] = z
    return d


def test_a_sign_at_head_height_is_overhead_with_its_range_from_depth():
    # 3 m ahead, bottom edge at 1.6 m above ground, top at 2.0: rows from the camera at 1.2 m
    cal = fuse.Calib(0.0, 1.2)
    y_bot = H / 2 - FX * (1.6 - 1.2) / 3.0
    y_top = H / 2 - FX * (2.0 - 1.2) / 3.0
    box = (420, y_top, 540, y_bot)
    depth = with_object(plane_depth(fx=FX * 448 / W), box, 3.0)
    o = fuse.object_from_det(det(*box, "street_furniture"), 0, FX, cal, W, H, depth)
    assert o["kind"] == "overhead" and o["range"] == "depth"
    assert o["distance"] == pytest.approx(3.0, abs=0.05)
    assert o["h_bottom"] == pytest.approx(1.6, abs=0.1)
    # without depth the same box reads as something far away on the ground: the stand-in's failure
    flat = fuse.object_from_det(det(*box, "street_furniture"), 0, FX, cal, W, H)
    assert flat["kind"] == "street_furniture" and flat["distance"] > 6


def test_something_entirely_above_head_height_is_dropped():
    y_bot = H / 2 - FX * (2.3 - 1.2) / 5.0               # a sign 5 m ahead, bottom edge 2.3 m up
    box = (420, y_bot - 40, 540, y_bot)
    depth = with_object(plane_depth(fx=FX * 448 / W), box, 5.0)
    assert fuse.object_from_det(det(*box, "overhead"), 0, FX, fuse.Calib(0.0, 1.2), W, H, depth) is None


def test_a_person_never_becomes_overhead_and_far_edges_are_not_judged():
    y_bot = H / 2 - FX * (1.6 - 1.2) / 3.0
    box = (420, y_bot - 60, 540, y_bot)
    depth = with_object(plane_depth(fx=FX * 448 / W), box, 3.0)
    cal = fuse.Calib(0.0, 1.2)
    assert fuse.object_from_det(det(*box, "person"), 0, FX, cal, W, H, depth)["kind"] == "person"
    y_far = H / 2 - FX * (1.6 - 1.2) / 20.0              # same edge height, 20 m away
    box = (420, y_far - 10, 440, y_far)
    depth = with_object(plane_depth(fx=FX * 448 / W), box, 20.0)
    assert fuse.object_from_det(det(*box, "pole"), 0, FX, cal, W, H, depth)["kind"] == "pole"


def test_a_branch_lying_on_the_ground_is_just_an_obstacle():
    y_bot = H / 2 + FX * 1.2 / 4.0                       # ground row at 4 m
    box = (400, y_bot - 30, 560, y_bot)
    depth = with_object(plane_depth(fx=FX * 448 / W), box, 4.0)
    o = fuse.object_from_det(det(*box, "overhead"), 0, FX, fuse.Calib(0.0, 1.2), W, H, depth)
    assert o["kind"] == "obstacle"


def test_stairs_up_and_down_from_their_top_edge():
    y_bot = H / 2 + FX * 1.2 / 4.0
    cal = fuse.Calib(0.0, 1.2)
    up = fuse.object_from_det(det(400, H / 2 - 20, 560, y_bot, "stairs"), 0, FX, cal, W, H)
    down = fuse.object_from_det(det(400, y_bot - 15, 560, y_bot, "stairs"), 0, FX, cal, W, H)
    assert up["kind"] == "stairs_up" and down["kind"] == "stairs_down"


def test_depth_takes_over_when_the_ground_row_is_cut_off_and_scale_is_applied():
    box = (380, 150, 580, 540)                           # bottom cut off: a close object
    depth = with_object(plane_depth(fx=FX * 448 / W, scale=0.5), box, 0.6)   # model reads half
    cal = fuse.Calib(0.0, 1.2, scale=2.0, source="depth")
    o = fuse.object_from_det(det(*box, "pole"), 0, FX, cal, W, H, depth)
    assert o["range"] == "depth" and o["distance"] == pytest.approx(1.2, abs=0.05) and o["cut_off"]


def test_a_cut_off_person_keeps_the_nearer_of_the_stand_in_bound_and_depth():
    box = (300, 100, 520, 540)                           # feet below the frame
    cal = fuse.Calib(0.0, 1.2, 1.0, "depth")
    bound = fuse.object_from_det(det(*box, "person"), 0, FX, fuse.Calib(0.0, 1.2), W, H)["distance"]
    far = with_object(plane_depth(fx=FX * 448 / W), box, bound + 2.0)
    assert fuse.object_from_det(det(*box, "person"), 0, FX, cal, W, H, far)["distance"] == bound
    near = with_object(plane_depth(fx=FX * 448 / W), box, bound - 0.5)
    assert fuse.object_from_det(det(*box, "person"), 0, FX, cal, W, H, near)["distance"] == \
        pytest.approx(bound - 0.5, abs=0.02)


def test_box_depth_uses_the_mask_and_the_near_face():
    depth = np.full((10, 20), 9.0, np.float32)
    depth[:, 8:12] = 2.0                                 # a thin pole in front of a far wall
    mask = np.zeros((10, 20), bool)
    mask[:, 8:12] = True
    d = det(0, 0, 20, 10, mask=mask)
    assert fuse.box_depth(depth, d, 20, 10) == pytest.approx(2.0)
    assert fuse.box_depth(depth, det(0, 0, 20, 10), 20, 10) == pytest.approx(2.0)   # centre half of box
    assert fuse.box_depth(depth, det(0, 0, 4, 10), 20, 10) == pytest.approx(9.0)


# ------------------------------------------------------------------ corridor: rises and drops from depth
def _ray_depth(depth, fx_d, cam_h, pitch, rows, cols, height_fn):
    """Paint into a depth map the z where each pixel's ray meets a surface of height height_fn(forward)."""
    h, w = depth.shape
    p = math.radians(pitch)
    for v in rows:
        a = p + math.atan((v - h / 2) / fx_d)                  # ray angle below horizontal
        for f in np.linspace(0.3, 30, 3000):                   # march forward until the ray is below the surface
            if cam_h - f * math.tan(a) <= height_fn(f):
                z = f / math.cos(a) * math.cos(math.atan((v - h / 2) / fx_d))
                depth[v, cols] = z
                break


def test_corridor_finds_a_low_box_rising_in_the_path_and_ignores_flat_ground():
    fx_d, pitch = 336.0, 12.0
    flat = plane_depth(fx=fx_d, pitch=pitch)
    cal = fuse.fit_ground(flat, fx_d, 1.2)
    assert fuse.corridor_hazards(flat, fx_d, cal) == []
    d = flat.copy()
    _ray_depth(d, fx_d, 1.2, pitch, range(126, 252), slice(200, 248),
               lambda f: 0.6 if 2.5 <= f <= 2.9 else 0.0)        # a 0.6 m box 2.5 m ahead, in the path
    out = fuse.corridor_hazards(d, fx_d, cal)
    assert [o["kind"] for o in out] == ["obstacle"]
    assert out[0]["distance"] == pytest.approx(2.5, abs=0.25) and abs(out[0]["bearing"]) < 5


def test_corridor_finds_a_drop_and_needs_a_fitted_calibration():
    fx_d, pitch = 336.0, 12.0
    d = plane_depth(fx=fx_d, pitch=pitch)
    cal = fuse.fit_ground(d, fx_d, 1.2)
    _ray_depth(d, fx_d, 1.2, pitch, range(126, 252), slice(150, 298),
               lambda f: -0.5 if f >= 2.0 else 0.0)              # the ground falls 0.5 m from 2 m on
    out = fuse.corridor_hazards(d, fx_d, cal)
    assert [o["kind"] for o in out] == ["curb"] and out[0]["distance"] == pytest.approx(2.0, abs=0.3)
    assert fuse.corridor_hazards(d, fx_d, fuse.Calib(pitch, 1.2)) == []       # given, not fitted: no claim


# ------------------------------------------------------------------ NMS and the wire contract
def test_agnostic_nms_merges_two_prompts_on_one_thing_but_keeps_a_person_by_a_bus():
    scooter = det(400, 180, 600, 260, "e_scooter", 0.30)
    rack = det(402, 182, 598, 262, "street_furniture", 0.34)
    bus = det(100, 50, 700, 400, "vehicle", 0.9)
    person = det(300, 150, 360, 380, "person", 0.8)
    pole = det(500, 0, 520, 300, "pole", 0.5)
    pole_in_light = det(502, 100, 518, 300, "pole", 0.3)
    kept = fuse.nms_agnostic([scooter, rack, bus, person, pole, pole_in_light])
    assert rack in kept and scooter not in kept
    assert bus in kept and person in kept and pole in kept and pole_in_light not in kept


def test_event_objects_carry_only_contract_fields_and_parse_on_the_earpiece():
    o = fuse.object_from_det(det(430, 200, 530, 400, "fire_hydrant"), 0, FX, fuse.Calib(0.0, 1.2), W, H)
    o["stop"], o["closing_speed"] = False, None
    ev = fuse.to_event_objects([o])
    assert set(ev[0]) == {"id", "kind", "bearing", "distance", "confidence"}
    msg = parse({"type": "obstacles", "source": "perception-chest", "objects": ev})
    assert msg.objects[0].kind == "fire_hydrant" and msg.objects[0].distance == o["distance"]
