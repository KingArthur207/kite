"""Pipeline, hazards and the hybrid detector with fake models: every event must parse on the earpiece."""

import numpy as np
import pytest
from conftest import plane_depth

from earpiece.events import parse
from perception import hazards, models
from perception.fuse import Det
from perception.pipeline import Pipeline

W, H, FX = 960, 540, 700.0
IMG = np.zeros((H, W, 3), np.uint8)


class FakeDetector:
    def __init__(self, frames):
        self.frames, self.n = frames, 0

    def __call__(self, img):
        out = self.frames[min(self.n, len(self.frames) - 1)]
        self.n += 1
        return list(out)


class FakeDepth:
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, img):
        return self.depth


def ground_row(d, cam_h=1.2, fx=FX, h=H):
    return h / 2 + fx * cam_h / d


def pole(d, x=470, tid="o1", cam_h=1.2):
    y = ground_row(d, cam_h)
    return Det(x, y - 300 / d, x + 20, y, kind="pole", conf=0.8, label="street light", tid=tid)


def person(d, x=440, tid="c1", cam_h=1.2):
    y = ground_row(d, cam_h)
    return Det(x, y - FX * 1.7 / d, x + 80, y, kind="person", conf=0.9, label="person", tid=tid)


def check_contract(events):
    for ev in events:
        parse(dict(ev))                     # raises BadMessage on anything the earpiece would refuse


# ------------------------------------------------------------------ chest
def test_chest_emits_obstacles_with_new_kinds_and_a_heartbeat_when_empty():
    p = Pipeline("chest", FakeDetector([[pole(4.0)], []]), fx=FX)
    r = p.step(IMG, 0.0)
    check_contract(r.events)
    assert [e["type"] for e in r.events] == ["obstacles"]
    obj = r.events[0]["objects"][0]
    assert obj["kind"] == "pole" and obj["distance"] == pytest.approx(4.0, abs=0.05) and obj["id"] == "o1"
    r = p.step(IMG, 0.1)
    assert [e["type"] for e in r.events] == ["obstacles", "heartbeat"] and r.events[0]["objects"] == []
    assert r.events[0]["seq"] == 2 and r.events[0]["source"] == "perception-chest"


def test_chest_stop_fires_for_a_pole_in_the_corridor_like_it_does_for_a_person():
    p = Pipeline("chest", FakeDetector([[pole(1.3)]]), fx=FX)
    r = p.step(IMG, 0.0)
    o = r.objects[0]
    assert o["stop"] and o["closing_speed"] > 0          # sent so the earpiece's sonar goes critical
    msg = parse(r.events[0])
    assert msg.objects[0].kind == "pole" and msg.objects[0].closing_speed > 0


def test_chest_with_depth_fits_the_pitch_and_ranges_by_it():
    fx_d = FX * 448 / W
    depth = FakeDepth(plane_depth(fx=fx_d, pitch=10.0, scale=0.7))
    # a pole 4 m ahead seen by a camera tilted 10 degrees down: the stand-in (pitch 0) would misread it
    import math
    y = H / 2 + FX * math.tan(math.atan(1.2 / 4.0) - math.radians(10.0))
    det = Det(470, y - 120, 490, y, kind="pole", conf=0.8, tid="o1")
    p = Pipeline("chest", FakeDetector([[det]]), depth=depth, fx=FX)
    for k in range(4):
        r = p.step(IMG, k * 0.1)
    assert r.calib.source == "depth" and r.calib.pitch == pytest.approx(10.0, abs=0.5)
    assert r.objects[0]["distance"] == pytest.approx(4.0, rel=0.08)
    flat = Pipeline("chest", FakeDetector([[det]]), fx=FX).step(IMG, 0.0)     # no depth: pitch 0 assumed
    assert flat.objects[0]["distance"] > 8


def test_depth_fallback_reports_an_unnamed_wall_once():
    fx_d = FX * 448 / W
    d = plane_depth(fx=fx_d)
    d[:, 150:300] = 1.0                                  # something 1 m ahead filling the centre band
    p = Pipeline("chest", FakeDetector([[]]), depth=FakeDepth(d), fx=FX, auto_pitch=False)
    r = p.step(IMG, 0.0)
    check_contract(r.events)
    assert r.depth_objects and r.events[0]["objects"][0]["kind"] == "obstacle"
    assert len(r.events) == 1                            # no heartbeat: something is there


# ------------------------------------------------------------------ drone
def drone_obj(kind, bearing, distance, cut=False):
    return {"id": kind, "kind": kind, "bearing": bearing, "distance": distance, "cut_off": cut}


def test_path_hazards_name_fixed_things_in_the_walkers_line_only():
    objs = [drone_obj("e_scooter", 0.0, 5.0), drone_obj("pole", 25.0, 5.0), drone_obj("construction", 2.0, 9.0),
            drone_obj("person", 0.0, 4.0), drone_obj("bollard", 0.0, 0.5)]
    hz = hazards.path_hazards(objs)
    assert set(hz) == {"obstacle", "construction"}       # pole off to the side, person not fixed,
    assert hz["obstacle"]["label"] == "e_scooter"        # bollard behind the walker (drone is 1 m back)
    assert hz["obstacle"]["distance"] == pytest.approx(4.0)


def test_gate_needs_three_frames_keeps_its_text_and_resends_every_half_ttl():
    g = hazards.Gate()
    c = {"obstacle": {"bearing": 0.0, "distance": 6.0, "label": "e_scooter"}}
    assert g.step(c, 0.0) == [] and g.step(c, 0.1) == []
    ev = g.step(c, 0.2)
    assert ev[0]["text"] == "E-scooter ahead, 6 meters." and ev[0]["kind"] == "obstacle"
    check_contract(ev)
    c2 = {"obstacle": {"bearing": 0.0, "distance": 5.0, "label": "e_scooter"}}
    assert g.step(c2, 0.3) == []                         # same direction, closer: no new words
    again = g.step(c2, 0.2 + hazards.HAZARD_TTL / 2)
    assert again[0]["text"] == "E-scooter ahead, 6 meters."   # refresh keeps the text (earpiece stays quiet)
    assert g.step({}, 3.0) == [] and g.live == {}


def test_drone_pipeline_sends_crowd_construction_and_a_scene():
    y = ground_row(6.0, cam_h=2.0)
    cone = Det(460, y - 40, 500, y, kind="construction", conf=0.6, label="traffic cone", tid="o9")
    people = [person(8.0, x=300 + 120 * i, tid=f"c{i}", cam_h=2.0) for i in range(3)]
    p = Pipeline("drone", FakeDetector([[cone] + people]), fx=FX, cam_h=2.0)
    evs = []
    for k in range(3):
        evs += p.step(IMG, k * 0.1).events
    check_contract(evs)
    kinds = {e.get("kind") for e in evs if e["type"] == "hazard"}
    assert {"crowd", "construction"} <= kinds
    scene = [e for e in evs if e["type"] == "scene"]
    assert len(scene) == 1 and "3 person" in scene[0]["text"] and "1 construction" in scene[0]["text"]


# ------------------------------------------------------------------ hybrid
def test_hybrid_keeps_people_from_coco_and_street_objects_from_open_vocab():
    coco = FakeDetector([[Det(100, 100, 200, 400, kind="person", conf=0.9, tid="c1")]])
    ov = FakeDetector([[Det(102, 98, 198, 402, kind="person", conf=0.3, tid="o1"),
                        Det(500, 50, 520, 400, kind="pole", conf=0.6, tid="o2")]])
    out = models.HybridDetector(coco, ov)(IMG)
    assert sorted((d.kind, d.tid) for d in out) == [("person", "c1"), ("pole", "o2")]


def test_hybrid_open_every_reuses_the_last_open_vocab_boxes():
    coco = FakeDetector([[]])
    ov = FakeDetector([[Det(500, 50, 520, 400, kind="pole", conf=0.6, tid="o2")], []])
    h = models.HybridDetector(coco, ov, open_every=2)
    assert len(h(IMG)) == 1 and len(h(IMG)) == 1 and ov.n == 1
    assert h(IMG) == [] and ov.n == 2


# ------------------------------------------------------------------ real models (optional)
@pytest.mark.slow
def test_real_models_name_the_scooter_scene():
    from pathlib import Path

    import cv2

    from perception import vocab
    root = Path(__file__).resolve().parents[2]
    img = cv2.resize(cv2.imread(str(root / "data/imagined/scooter.jpg")), (W, H))
    det = models.HybridDetector(models.CocoDetector("cpu", track=False),
                                models.OpenVocabDetector(vocab.load(), "cpu", track=False))
    kinds = {d.kind for d in det(img)}
    assert "vehicle" in kinds and ({"e_scooter", "pole"} & kinds)
