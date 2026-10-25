"""One pipeline for every frame source.

``process(frame, camera, pose)`` returns earpiece events. The frame is BGR
at the size the source produced. It is not cropped and not stretched, and
``fx`` / ``fy`` are pixels in that frame. ``camera.role`` is ``drone`` or
``sim`` for the street view, and ``chest`` for close obstacles.

Stops (blocked ground, a red or yellow light, a stop sign, something large
and close) are said on the frame they appear. A new left, right, or straight
has to hold for a few frames. A person cut off by the bottom edge is the
walker, not a crowd.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from .camera import Camera, Pose, fit_frame, ground_near_frac, locate
from .events import Event, forward_events, obstacle_events
from .fuse import FuseConfig, assess, chest_view
from .mapillary import crosswalk_in_band
from .overlay import draw_chest, draw_overlay
from .signal import paint_signal
from .types import Box


@dataclass
class DirectionHold:
    """Commit the first direction. Later ones wait. Stop does not wait."""

    frames: int = 3
    committed: str | None = None
    pending: str | None = None
    count: int = 0
    committed_for: int = 0

    def update(self, direction: str) -> str:
        if direction == "stop":
            self.committed_for = 1 if self.committed != "stop" else self.committed_for + 1
            self.committed = "stop"
            self.pending = None
            self.count = 0
            return "stop"
        if self.committed is None or direction == self.committed:
            self.committed_for = 1 if self.committed != direction else self.committed_for + 1
            self.committed = direction
            self.pending = None
            self.count = 0
            return direction
        if direction != self.pending:
            self.pending = direction
            self.count = 1
        else:
            self.count += 1
        if self.count >= self.frames:
            self.committed = direction
            self.pending = None
            self.committed_for = self.frames
            self.count = 0
            return direction
        return self.committed


class Perception:
    def __init__(
        self,
        segmenter,
        detector,
        sign_detector=None,
        depth_estimator=None,
        lights=None,
        cfg: FuseConfig | None = None,
        hold_frames: int = 3,
        depth_stride: int = 4,
        parallel: bool = False,
    ):
        self.segmenter = segmenter
        self.detector = detector
        self.sign_detector = sign_detector
        self.depth_estimator = depth_estimator
        self.lights = lights
        self.parallel = parallel
        self._pool: ThreadPoolExecutor | None = None
        self.cfg = cfg or FuseConfig()
        self.hold = DirectionHold(max(1, hold_frames))
        self.depth_stride = max(1, depth_stride)
        self._last = None
        self._said: str | None = None
        self._chest_said: str | None = None
        self._depth = None
        self._depth_i = 0
        self.image: np.ndarray | None = None
        self.events: tuple[Event, ...] = ()
        self.cue = None
        self.boxes: list[Box] = []

    def process(self, frame: np.ndarray, camera: Camera, pose: Pose) -> tuple[Event, ...]:
        if camera.role not in ("drone", "sim", "chest"):
            raise ValueError("camera.role must be drone, sim, or chest")
        frame, camera = fit_frame(frame, camera)
        if camera.role == "chest":
            events = self._chest(frame, camera, pose)
        else:
            events = self._forward(frame, camera, pose)
        self.events = tuple(events)
        return self.events

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _forward(self, frame: np.ndarray, camera: Camera, pose: Pose) -> list[Event]:
        labels, detected, reading = self._views(frame)
        if labels.shape[:2] != frame.shape[:2]:
            raise ValueError("segmenter returned a different frame size")
        names = _names(self.segmenter)
        boxes = [paint_signal(box, frame) for box in detected]
        if self.sign_detector is not None and not self.parallel:
            boxes.extend(paint_signal(box, frame) for box in self.sign_detector.predict(frame))
        frac = ground_near_frac(frame.shape[0], camera, pose, self.cfg.near_distance_m)
        y0 = int(frame.shape[0] * frac)
        pedestrian = None if reading is None else reading.pedestrian
        raw = assess(
            labels,
            names,
            boxes,
            cfg=replace(self.cfg, near_frac=frac),
            pedestrian=pedestrian,
            crosswalk_pixels=crosswalk_in_band(labels, names, y0),
        )
        held = self.hold.update(raw.direction)
        if held != raw.direction and self._last is not None and self._last.direction == held:
            spoken = self._last
        else:
            spoken = raw
            self._last = raw
        pending = "" if raw.direction == held else raw.direction
        say = spoken.speech != self._said
        if say:
            self._said = spoken.speech
        self.cue = spoken
        drawn = list(boxes)
        extra = None
        if reading is not None:
            drawn.extend(reading.boxes)
            extra = {
                "pedestrian_light": reading.pedestrian or "",
                "traffic_lights": list(reading.labels),
            }
        self.boxes = drawn
        self.image = draw_overlay(frame, labels, names, drawn, spoken, role=camera.role)
        return forward_events(
            raw,
            spoken,
            camera.role,
            say=say,
            pending=pending,
            stable=self.hold.committed_for,
            extra=extra,
        )

    def _views(self, frame: np.ndarray):
        """Segment, detect, and classify lights. The first two share the frame."""
        if not self.parallel:
            labels = self.segmenter.predict(frame)
            boxes = self.detector.predict(frame)
            reading = None if self.lights is None else self.lights.read(frame, labels, _names(self.segmenter))
            return labels, boxes, reading
        pool = self._pool
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="streetseg")
            self._pool = pool
        segmented = pool.submit(self.segmenter.predict, frame)
        detected = pool.submit(self.detector.predict, frame)
        signed = None
        if self.sign_detector is not None:
            signed = pool.submit(self.sign_detector.predict, frame)
        labels = segmented.result()
        lit = None
        if self.lights is not None:
            lit = pool.submit(self.lights.read, frame, labels, _names(self.segmenter))
        boxes = list(detected.result())
        if signed is not None:
            boxes.extend(signed.result())
        reading = None if lit is None else lit.result()
        return labels, boxes, reading

    def _chest(self, frame: np.ndarray, camera: Camera, pose: Pose) -> list[Event]:
        del pose
        boxes = list(self.detector.predict(frame))
        found = chest_view(
            boxes,
            self._maybe_depth(frame),
            frame.shape[:2],
            camera,
            self.cfg,
        )
        self.boxes = boxes
        if found is None:
            self._chest_said = None
            self.image = draw_chest(frame, boxes, "Chest clear.", False)
            return []
        detail = {"hazards": list(found.hazards), "label": found.label or ""}
        if found.distance_m is not None:
            detail["distance_m"] = found.distance_m
        if found.lateral_m is not None:
            detail["lateral_m"] = found.lateral_m
        if found.box is not None:
            detail.update(locate(found.box, camera, frame.shape[:2]))
        say = found.speech != self._chest_said
        if say:
            self._chest_said = found.speech
        self.image = draw_chest(frame, boxes, found.speech, True)
        return obstacle_events(found.speech, found.reason, list(found.hazards), detail, say=say)

    def _maybe_depth(self, frame: np.ndarray):
        if self.depth_estimator is None:
            return None
        if self._depth is None or self._depth_i % self.depth_stride == 0:
            self._depth = self.depth_estimator.predict(frame)
        self._depth_i += 1
        return self._depth


def parallel_for_device(device: str | None) -> bool:
    """Overlap the segmenter and the detector.

    On CPU the two networks share the cores and the pair gets slower.
    CUDA and MPS keep the overlap. An empty device is CPU.
    """
    if not device:
        return False
    kind = str(device).strip().lower().split(":", 1)[0]
    return kind not in ("", "cpu")


def _names(segmenter) -> dict[int, str]:
    raw = segmenter.names
    if callable(raw):
        raw = raw()
    return {int(key): str(value) for key, value in dict(raw).items()}
