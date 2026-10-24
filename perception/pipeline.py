"""One camera's frames -> earpiece events. Models are passed in, so tests drive this with fakes.

chest -> `obstacles` every frame (+ `heartbeat` when empty), smoothed and stop-flagged by the stand-in's
         live_walk.chest_step, so "Stop." fires at the same range and in the same corridor as before.
drone -> `hazard` (crowd, vehicle, construction, overhead, curb, stairs_down, obstacle) + `scene` every 10 s.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import live_walk
from standin_detect import SCENE_EVERY, StillVehicles

from . import fuse, hazards


@dataclass
class FrameResult:
    events: list[dict]
    objects: list[dict]               # with the CLI-only fields (cut_off, range, h_bottom, stop, label)
    dets: list[fuse.Det]
    calib: fuse.Calib
    timings: dict[str, float] = field(default_factory=dict)   # ms per stage
    depth_objects: list[dict] = field(default_factory=list)   # class-agnostic, from depth alone


class Pipeline:
    def __init__(self, camera: str, detector, depth=None, fx: float = 1000.0, cam_h: float = 1.2,
                 pitch: float = 0.0, auto_pitch: bool = True, stop_m: float = live_walk.STOP_M,
                 source: str = "perception", ignore_bottom: float = 0.0):
        self.camera, self.detector, self.depth = camera, detector, depth
        self.fx, self.cam_h, self.pitch, self.auto_pitch, self.stop_m = fx, cam_h, pitch, auto_pitch, stop_m
        self.source = f"{source}-{camera}"
        self.ignore_bottom = ignore_bottom
        self.calib = fuse.CalibFilter(pitch, cam_h)
        self.rf = live_walk.RangeFilter()
        self.gate, self.lateral, self.still = hazards.Gate(), live_walk.RangeFilter(), StillVehicles()
        self.last_scene = -SCENE_EVERY
        self.seq = 0

    def step(self, img: np.ndarray, t: float, pose: dict | None = None) -> FrameResult:
        h, w = img.shape[:2]
        pose = pose or {}
        fx = pose.get("fx") or self.fx
        cam_h = pose.get("cam_height") or self.cam_h
        timings = {}
        ts = time.perf_counter()
        dets = self.detector(img)
        timings["detect"] = (time.perf_counter() - ts) * 1000
        depth = None
        calib = fuse.Calib(self.pitch, cam_h)
        if self.depth is not None:
            ts = time.perf_counter()
            depth = self.depth(img)
            timings["depth"] = (time.perf_counter() - ts) * 1000
            fx_d = fx * depth.shape[1] / w
            excl = _box_mask(dets, depth.shape, w, h)
            calib = self.calib.update(fuse.fit_ground(depth, fx_d, cam_h, excl, self.pitch,
                                                      fit_pitch=self.auto_pitch))
        objs = fuse.objects_from_dets(dets, fx, calib, w, h, depth)
        extra = self._depth_objects(depth, fx * depth.shape[1] / w, calib, objs) if depth is not None else []
        events: list[dict] = []
        if self.camera == "chest":
            objs = live_walk.chest_step(objs, self.rf, t, self.stop_m)
            extra = live_walk.chest_step(extra, self.rf, t, self.stop_m) if extra else []
            sent = objs + extra
            self.seq += 1
            events.append({"type": "obstacles", "source": self.source, "seq": self.seq,
                           "objects": fuse.to_event_objects(sent)})
            if not sent:
                events.append({"type": "heartbeat", "source": self.source})
        else:
            for ev in self.gate.step(hazards.all_hazards(objs + extra, self.lateral, t, pose, self.still), t):
                events.append({**ev, "source": self.source})
            if t - self.last_scene >= SCENE_EVERY:
                self.last_scene = t
                events.append({"type": "scene", "source": self.source, "text": hazards.scene_text(objs)})
        return FrameResult(events, objs, dets, calib, timings, extra)

    def _depth_objects(self, depth, fx_d, calib, objs) -> list[dict]:
        """Things no detector named, from depth alone. With a depth-fitted calibration: the corridor check
        (something rising out of the walking path, or the ground dropping away). Without one (fit failed,
        or --no-auto-pitch): the stand-in's centre-band check on the chest camera. A rise already covered by
        a named object (same direction, about the same range) is left out: one sound per thing."""
        if calib.source == "depth":
            found = fuse.corridor_hazards(depth, fx_d, calib, self.ignore_bottom)
        elif self.camera == "chest":
            o = live_walk.center_obstacle(depth * calib.scale, fx_d, calib.cam_h, calib.pitch)
            found = [{**o, "range": "depth", "label": "depth"}] if o else []
        else:
            found = []
        return [o for o in found if o["kind"] != "obstacle" or not any(
            abs(p["bearing"] - o["bearing"]) < 20 and p["distance"] < o["distance"] + 0.7 for p in objs)]


def _box_mask(dets, shape, w, h) -> np.ndarray:
    m = np.zeros(shape, bool)
    sy, sx = shape[0] / h, shape[1] / w
    for d in dets:
        m[int(d.y1 * sy):int(np.ceil(d.y2 * sy)), int(d.x1 * sx):int(np.ceil(d.x2 * sx))] = True
    return m
