"""--yolo on the drone: YOLO26n's boxes are shown on the camera and nothing else."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run"))

from obstacles import ObstacleView, overlaps  # noqa: E402

import ui  # noqa: E402


class Box:
    def __init__(self, label, xyxy):
        self.label, self.xyxy = label, xyxy


class Detector:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return [Box("person", (100.0, 50.0, 200.0, 300.0)), Box("bench", (300.0, 200.0, 400.0, 260.0))]


class Follow:
    def snapshot(self):
        return type("Shown", (), {"box": (102.0, 48.0, 198.0, 302.0), "t": time.time(), "confirmed": True})()


def test_the_view_detects_the_newest_frame_on_its_own_thread():
    detector = Detector()
    view = ObstacleView(detector).start()
    try:
        view.offer(np.zeros((72, 96, 3), np.uint8))
        deadline = time.time() + 2
        while not view.last and time.time() < deadline:
            time.sleep(0.01)
        assert [b.label for b in view.last] == ["person", "bench"]
        assert detector.calls == 1  # one frame offered, one detection
    finally:
        view.stop()


def test_the_followed_person_is_not_drawn_again():
    assert overlaps((100, 50, 200, 300), (102, 48, 198, 302))
    assert not overlaps((100, 50, 200, 300), (300, 200, 400, 260))
    view = type("View", (), {"last": Detector().predict(None), "last_t": time.time()})()
    img = np.zeros((360, 480, 3), np.uint8)
    ui.draw_obstacles(img, view, Follow())
    assert not img[50:300, 100].any()  # the walker's box: left to draw_people
    assert img[230, 300].any()  # the bench


def test_nothing_is_drawn_without_yolo_or_from_stale_boxes():
    img = np.zeros((360, 480, 3), np.uint8)
    ui.draw_obstacles(img, None, None)
    stale = type("View", (), {"last": Detector().predict(None), "last_t": time.time() - 5})()
    ui.draw_obstacles(img, stale, None)
    assert not img.any()
