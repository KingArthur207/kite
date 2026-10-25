"""Detector boxes as obstacles: segmentation only, segmentation plus YOLO, and YOLO only."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "third_party", "videotesting")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from streetseg.fuse import assess  # noqa: E402
from streetseg.labels import NAMES  # noqa: E402
from streetseg.types import Box  # noqa: E402

H, W = 240, 360
Y0 = int(H * 0.55)  # top of the near band with the default FuseConfig
SIDEWALK, BUILDING, PERSON, CAR = 1, 2, 11, 13


def _box(label, x1, y1, x2, y2, signal=None):
    return Box(label, 0.9, (float(x1), float(y1), float(x2), float(y2)), signal=signal)


def _street():
    """Sidewalk across the whole near band, building above."""
    labels = np.full((H, W), BUILDING, np.int32)
    labels[Y0:] = SIDEWALK
    return labels


# A chair over 40% of the center column's width, the full near band tall.
CHAIR = _box("chair", 156, Y0, 204, H)


def _frames():
    street = _street()
    left_wall = street.copy()
    left_wall[Y0:, :120] = BUILDING
    crowd = street.copy()
    crowd[Y0:, 130:230] = PERSON
    blocked = np.full((H, W), BUILDING, np.int32)
    return [
        (street, []),
        (street, [CHAIR]),
        (left_wall, [_box("person", 250, 150, 290, 230)]),
        (crowd, [_box("person", 140, 140, 180, 235), _box("person", 190, 140, 220, 235)]),
        (street, [_box("car", 120, 130, 240, 240)]),
        (street, [_box("traffic light", 170, 10, 190, 60, signal="red")]),
        (blocked, []),
        (street, [_box("person", 130, 90, 230, H)]),
    ]


def test_seg_only_is_unchanged_when_obstacles_are_left_off():
    for labels, boxes in _frames():
        plain = assess(labels, NAMES, boxes)
        off = assess(labels, NAMES, boxes, obstacles=False)
        assert plain == off
        assert plain.to_dict() == off.to_dict()
        assert plain.reason != "obstacle"
        assert plain.obstacle == "" and plain.ground is True
        assert all(col.obstacle == 0.0 and col.obstacle_label == "" for col in plain.columns)


def test_yolo_only_takes_the_freer_side():
    bench = _box("bench", 0, Y0, 30, H)
    cue = assess(None, {}, [CHAIR, bench], obstacles=True, shape=(H, W))
    assert cue.direction == "right"
    assert cue.speech == "Bear right to go around the chair."
    assert cue.columns[0].obstacle_label == "bench"


def test_yolo_only_stops_when_every_column_is_blocked():
    boxes = [
        _box("bench", 0, Y0, 100, H),
        _box("chair", 125, Y0, 235, H),
        _box("suitcase", 260, Y0, W, H),
    ]
    cue = assess(None, {}, boxes, obstacles=True, shape=(H, W))
    assert cue.direction == "stop"
    assert cue.reason == "obstacle"
    assert cue.obstacle == "chair"
    assert cue.speech == "Stop. A chair is blocking the way."
    assert not any(col.selectable for col in cue.columns)


def test_yolo_only_walks_past_a_small_obstacle():
    pebble = _box("chair", 175, 220, 185, H)
    cue = assess(None, {}, [pebble], obstacles=True, shape=(H, W))
    assert cue.direction == "straight"
    assert cue.speech == "Continue straight. The way looks clear."


def test_seg_plus_yolo_bears_around_a_box_the_seg_missed():
    labels = _street()
    assert assess(labels, NAMES, [CHAIR]).direction == "straight"
    cue = assess(labels, NAMES, [CHAIR], obstacles=True)
    assert cue.ground is True
    assert cue.direction == "left"
    assert cue.reason == "obstacle"
    assert cue.speech == "Bear left to go around the chair."


def test_the_wearer_is_not_an_obstacle():
    wearer = _box("person", 130, 90, 230, H)
    cue = assess(None, {}, [wearer], obstacles=True, shape=(H, W))
    assert cue.wearer_excluded
    assert cue.direction == "straight"
    assert cue.reason == "clear"
    assert all(col.obstacle == 0.0 for col in cue.columns)


def test_a_red_light_ahead_still_wins_over_an_obstacle_stop():
    boxes = [
        _box("bench", 0, Y0, 100, H),
        _box("chair", 125, Y0, 235, H),
        _box("suitcase", 260, Y0, W, H),
        _box("traffic light", 170, 10, 190, 60, signal="red"),
    ]
    cue = assess(None, {}, boxes, obstacles=True, shape=(H, W))
    assert cue.reason == "signal"
    assert cue.speech == "Stop. The light looks red."
