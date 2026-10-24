import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "video"))

import overlay  # noqa: E402
import seg_standin as ss  # noqa: E402

ROAD, SIDEWALK, PATH, PERSON, CAR, BUS, SKY = 6, 11, 52, 12, 20, 80, 2     # ADE20K ids


def _masks(labels):
    return ss.masks_from_labels(np.asarray(labels, np.uint8))


def test_masks_map_ade20k_ids_to_our_classes_and_resize_to_the_frame():
    labels = np.full((10, 10), SKY, np.uint8)
    labels[5:, :3], labels[5:, 3:5], labels[5:, 5:] = SIDEWALK, PATH, ROAD
    labels[0:3, 0:3] = PERSON
    labels[3:5, 0:4], labels[3:5, 6:] = CAR, BUS
    m = ss.masks_from_labels(labels, (20, 40))
    assert m["crosswalk"] is None
    assert all(m[k].shape == (20, 40) and m[k].dtype == bool for k in ("road", "sidewalk", "person", "vehicle"))
    assert m["sidewalk"][10:, :20].all() and not m["sidewalk"][10:, 20:].any()   # path counts as sidewalk
    assert m["road"][10:, 20:].all()
    assert m["person"][0:6, 0:12].all() and not m["person"][6:].any()
    assert m["vehicle"][6:10, 0:16].all() and m["vehicle"][6:10, 24:].all()     # car and bus
    assert not (m["sidewalk"] & m["road"]).any()


def test_specks_under_the_minimum_area_are_dropped():
    labels = np.full((100, 100), SKY, np.uint8)
    labels[50:, :] = SIDEWALK
    labels[10, 10] = SIDEWALK                     # 1 px: noise
    labels[20:24, 20:24] = PERSON                 # 16 px < 0.2% of 10 000
    labels[30:40, 30:40] = CAR                    # 100 px: kept
    m = _masks(labels)
    assert not m["sidewalk"][10, 10] and m["sidewalk"][50:].all()
    assert not m["person"].any() and m["vehicle"].sum() == 100


def test_corridor_is_centred_and_narrows_upwards():
    p = ss.corridor(720, 960)
    assert p[:, 1].max() == 719 and p[:, 1].min() == 360
    bottom = sorted(p[p[:, 1] == 719][:, 0])
    top = sorted(p[p[:, 1] == 360][:, 0])
    assert bottom[1] - bottom[0] > top[1] - top[0]
    assert abs((bottom[0] + bottom[1]) / 2 - 480) <= 1 and abs((top[0] + top[1]) / 2 - 480) <= 1


def test_walkable_ahead_full_sidewalk_road_only_and_narrowing():
    h, w = 720, 960
    full = np.full((h, w), SIDEWALK, np.uint8)
    wa = ss.walkable_ahead(_masks(full))
    assert wa == {"ahead": 1.0, "near": 1.0, "far": 1.0}
    assert ss.hint_text(wa) is None

    road = np.full((h, w), ROAD, np.uint8)
    wa = ss.walkable_ahead(_masks(road))
    assert wa["ahead"] == 0.0 and ss.hint_text(wa) == "No sidewalk ahead."

    narrows = road.copy()
    narrows[539:, :] = SIDEWALK                   # the near half of the corridor (rows 539-719) is walkable
    wa = ss.walkable_ahead(_masks(narrows))
    assert wa["near"] == 1.0 and wa["far"] == 0.0 and 0.4 < wa["ahead"] < 0.8
    assert ss.hint_text(wa) == "Sidewalk narrows ahead."


def test_walkable_ignores_sidewalk_outside_the_corridor_and_counts_crosswalk():
    h, w = 720, 960
    labels = np.full((h, w), ROAD, np.uint8)
    labels[:, :100] = SIDEWALK                    # sidewalk only at the far left edge
    m = _masks(labels)
    assert ss.walkable_ahead(m)["ahead"] == 0.0
    m["crosswalk"] = np.ones((h, w), bool)        # streetseg may provide one
    assert ss.walkable_ahead(m)["ahead"] == 1.0


def test_people_in_the_corridor_do_not_count_against_walkable():
    labels = np.full((720, 960), SIDEWALK, np.uint8)
    labels[400:719, 430:530] = PERSON             # the walker, cut off by the bottom edge
    assert ss.walkable_ahead(_masks(labels)) == {"ahead": 1.0, "near": 1.0, "far": 1.0}
    labels[400:719, 430:530] = CAR                # a car there is not walkable
    assert ss.walkable_ahead(_masks(labels))["ahead"] < 0.9


def test_hint_gate_needs_hold_frames_to_change_and_to_clear():
    g = ss.HintGate(hold=3)
    assert g.step("Sidewalk narrows ahead.") == (False, None)
    assert g.step(None) == (False, None)          # flicker resets the count
    assert g.step("Sidewalk narrows ahead.") == (False, None)
    assert g.step("Sidewalk narrows ahead.") == (False, None)
    assert g.step("Sidewalk narrows ahead.") == (True, "Sidewalk narrows ahead.")
    assert g.step("Sidewalk narrows ahead.") == (False, "Sidewalk narrows ahead.")
    assert g.step(None) == (False, "Sidewalk narrows ahead.")
    assert g.step(None) == (False, "Sidewalk narrows ahead.")
    assert g.step(None) == (True, None)


def test_process_emits_scene_only_never_a_stop():
    seg = ss.Segmenter.__new__(ss.Segmenter)      # no model: process() with a precomputed segmentation
    seg.gate = ss.HintGate(hold=2)
    road = {"masks": _masks(np.full((72, 96), ROAD, np.uint8))}
    road["walkable"] = ss.walkable_ahead(road["masks"])
    frame = np.zeros((72, 96, 3), np.uint8)
    assert seg.process(frame, seg=road) == []
    evs = seg.process(frame, seg=road)
    assert [e["type"] for e in evs] == ["scene"] and evs[0]["text"].startswith("No sidewalk ahead.")


def test_overlay_draw_seg_tints_masks_and_keeps_the_rest():
    img = np.full((72, 96, 3), 100, np.uint8)
    masks = _masks(np.full((72, 96), SKY, np.uint8))
    masks["sidewalk"][60:, :10] = True
    overlay.draw_seg(img, {"masks": masks}, 0.1)
    assert (img[65, 5] != 100).any() and img[65, 5][1] > img[65, 5][0]   # green tint on the sidewalk
    assert (img[5, 5] == 100).all()                                     # untouched outside masks and corridor
