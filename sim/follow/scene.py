"""The street, the walker, the rendered camera and the person detector of the follow sim.

World frame (as kite/fly/sim/drone.py): x east, y north, z up (m, ground = 0), headings clockwise from north. The
sidewalk runs north from the origin, 4 m wide (x -1..3), curb at x = 3, road beyond.

Camera: the Tello's, fixed to the body and pitched with it (+ = nose up), 960 x 720, kite.camera.Camera intrinsics.
A world point maps to (forward, right, up) of the drone, then depth z = f cos p + u sin p, column cx + fx r / z,
row cy + fy (f sin p - u cos p) / z: the same projection kite/fly/sim/camera.py uses for the box sim.

Detector "oracle": the walker's box from kite.fly.sim.camera's projection rules (visible part only, cut by the frame
edges), with per-edge noise, a miss probability and a low head-only confidence, as a head seen from above scores.
Detector "yolo": kite.follow.YoloPersonDetector on the rendered image (needs the `follow` extra and its weights; a
drawn person is not a photo, so expect it to miss more).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from kite.camera import Camera
from kite.fly.sim.camera import project, visible_box
from kite.follow import Detection
from kite.sources.base import FrameSource
from kite.types import Frame

SIDEWALK_X = (-1.0, 3.0)
HEAD_M, HEAD_HALF_W, TORSO_HALF_W = 0.23, 0.09, 0.21


# ------------------------------------------------------------------ walker
@dataclass
class Leg:
    """One part of the walk: `secs` long, at speed `v` (m/s), turning at `turn` deg/s (+ = right)."""
    secs: float
    v: float
    turn: float = 0.0


def moss_walk(v: float) -> list[Leg]:
    """Stand, walk the block, stop at the curb, turn left, walk on, stop."""
    return [Leg(3.0, 0.0), Leg(16.0, v), Leg(4.0, 0.0), Leg(3.0, v * 0.8, -30.0), Leg(16.0, v), Leg(5.0, 0.0),
            Leg(3.0, v * 0.8, 30.0), Leg(10.0, v)]


class Walker:
    """A unicycle along the legs with an acceleration limit; stands still after the last one."""

    def __init__(self, legs: list[Leg], x: float = 1.0, y: float = 1.2, heading_deg: float = 0.0,
                 height_m: float = 1.70, accel: float = 1.0):
        self.legs, self.accel, self.height_m = legs, accel, height_m
        self.x, self.y, self.heading, self.v = x, y, heading_deg, 0.0
        self.t = 0.0
        self.duration = sum(leg.secs for leg in legs)

    def _leg(self) -> Leg:
        t = self.t
        for leg in self.legs:
            if t < leg.secs:
                return leg
            t -= leg.secs
        return Leg(math.inf, 0.0)

    def step(self, dt: float) -> None:
        leg = self._leg()
        dv = leg.v - self.v
        self.v += max(-self.accel * dt, min(self.accel * dt, dv))
        self.heading += leg.turn * dt
        h = math.radians(self.heading)
        self.x += self.v * math.sin(h) * dt
        self.y += self.v * math.cos(h) * dt
        self.t += dt


# ------------------------------------------------------------------ camera
@dataclass
class Pose:
    x: float
    y: float
    z: float
    heading_deg: float
    pitch_deg: float


def to_drone(pose: Pose, pts: np.ndarray) -> np.ndarray:
    """World points (N, 3) -> (forward, right, up) of the drone, level frame."""
    h = math.radians(pose.heading_deg)
    s, c = math.sin(h), math.cos(h)
    d = pts - np.array([pose.x, pose.y, pose.z])
    return np.stack([d[:, 0] * s + d[:, 1] * c, d[:, 0] * c - d[:, 1] * s, d[:, 2]], axis=1)


def to_image(cam: Camera, pose: Pose, pts: np.ndarray, near: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """World points -> pixel (u, v) and a mask of the points in front of the camera."""
    fru = to_drone(pose, pts)
    p = math.radians(pose.pitch_deg)
    cp, sp = math.cos(p), math.sin(p)
    z = fru[:, 0] * cp + fru[:, 2] * sp
    ok = z > near
    zs = np.where(ok, z, 1.0)
    u = cam.cx + cam.fx * fru[:, 1] / zs
    v = cam.cy + cam.fy * (fru[:, 0] * sp - fru[:, 2] * cp) / zs
    return np.stack([u, v], axis=1), ok


def walker_geom(cam: Camera, pose: Pose, w: Walker):
    """kite.fly.sim.camera's walker geometry (head column, rows, widths) from this camera pose."""
    fru = to_drone(pose, np.array([[w.x, w.y, w.height_m]]))[0]
    return project(fru[0], fru[1], fru[2], HEAD_M, HEAD_HALF_W, TORSO_HALF_W, pose.z, pose.pitch_deg,
                   cam.fx, cam.fy, cam.cx, cam.cy)


# ------------------------------------------------------------------ rendering
SKY_TOP, SKY_LOW = (235, 200, 160), (250, 235, 215)  # BGR
SIDEWALK, ROAD, CURB, TILE = (175, 180, 185), (80, 80, 85), (215, 215, 220), (140, 145, 150)
SHIRT, PANTS, SKIN, HAIR = (160, 90, 40), (60, 50, 45), (140, 170, 205), (30, 30, 35)


def _poly(img, cam, pose, pts3, color, near: float = 0.2) -> None:
    """Fill a planar 3D polygon, clipped to the part in front of the camera (depth > near)."""
    fru = to_drone(pose, np.asarray(pts3, float))
    p = math.radians(pose.pitch_deg)
    cp, sp = math.cos(p), math.sin(p)
    depth = fru[:, 0] * cp + fru[:, 2] * sp
    keep = []
    n = len(fru)
    for i in range(n):  # Sutherland-Hodgman against the plane depth = near
        a, b, da, db = fru[i], fru[(i + 1) % n], depth[i], depth[(i + 1) % n]
        if da > near:
            keep.append(a)
        if (da > near) != (db > near):
            keep.append(a + (b - a) * (near - da) / (db - da))
    if len(keep) < 3:
        return
    k = np.array(keep)
    z = k[:, 0] * cp + k[:, 2] * sp
    uv = np.stack([cam.cx + cam.fx * k[:, 1] / z, cam.cy + cam.fy * (k[:, 0] * sp - k[:, 2] * cp) / z], axis=1)
    cv2.fillPoly(img, [np.round(np.clip(uv, -1e5, 1e5)).astype(np.int32)], color, cv2.LINE_AA)


def _ground_quad(img, cam, pose, x0, x1, y0, y1, color, step=1.0) -> None:
    for ya in np.arange(y0, y1, step):
        yb = min(ya + step, y1)
        _poly(img, cam, pose, [(x0, ya, 0), (x1, ya, 0), (x1, yb, 0), (x0, yb, 0)], color)


def render(cam: Camera, pose: Pose, walker: Walker) -> np.ndarray:
    W, H = cam.width, cam.height
    img = np.empty((H, W, 3), np.uint8)
    # sky above the horizon row (level camera: cy, moved by the pitch)
    hz = int(np.clip(cam.cy - cam.fy * math.tan(math.radians(pose.pitch_deg)), 0, H))
    t = np.linspace(0, 1, max(hz, 1))[:, None]
    img[:hz] = (np.array(SKY_TOP) * (1 - t) + np.array(SKY_LOW) * t)[:, None, :].astype(np.uint8)[:hz]
    img[hz:] = ROAD
    # ground within 25 m: paving west of the curb (so the walk can turn onto it), curb, road east of it
    r = 25.0
    x_lo, x_hi = pose.x - r, SIDEWALK_X[1]
    y_lo, y_hi = pose.y - r, pose.y + r
    if x_lo < x_hi:
        _ground_quad(img, cam, pose, x_lo, x_hi, y_lo, y_hi, SIDEWALK, 1.0)
        for yy in np.arange(math.floor(y_lo), y_hi, 1.0):  # tile joints every metre
            _ground_line(img, cam, pose, (x_lo, yy), (x_hi, yy))
        for xx in np.arange(math.floor(x_lo), x_hi, 1.0):
            _ground_line(img, cam, pose, (xx, y_lo), (xx, y_hi))
    _ground_quad(img, cam, pose, x_hi, x_hi + 0.2, y_lo, y_hi, CURB, 1.0)
    _draw_walker(img, cam, pose, walker)
    return img


def _ground_line(img, cam, pose, a, b, step=0.5) -> None:
    """A ground line drawn piecewise, so the part behind the camera is left out."""
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
    pts = np.stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n), np.zeros(n)], axis=1)
    uv, ok = to_image(cam, pose, pts)
    for p0, p1, o0, o1 in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
        if o0 and o1:
            cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), TILE, 1, cv2.LINE_AA)


def _draw_walker(img, cam, pose, w: Walker) -> None:
    """Legs, torso and arms as camera-facing quads, then the head as a disc: far parts first."""
    fru = to_drone(pose, np.array([[w.x, w.y, 0.0]]))[0]
    if fru[0] * math.cos(math.radians(pose.pitch_deg)) < 0.2:
        return
    # the quads face the camera: their width runs along the drone's right vector
    h = math.radians(pose.heading_deg)
    rx, ry = math.cos(h), -math.sin(h)
    hgt = w.height_m
    phase = math.sin(w.t * 6.0) * min(1.0, w.v / 0.5) * 0.12  # leg swing

    def quad(cx_off, half_w, z0, z1, color, lean=0.0):
        bx, by = w.x + rx * cx_off, w.y + ry * cx_off
        hx, hy = math.sin(math.radians(w.heading)) * lean, math.cos(math.radians(w.heading)) * lean
        pts = [(bx - rx * half_w, by - ry * half_w, z0), (bx + rx * half_w, by + ry * half_w, z0),
               (bx + rx * half_w + hx, by + ry * half_w + hy, z1), (bx - rx * half_w + hx, by - ry * half_w + hy, z1)]
        _poly(img, cam, pose, pts, color)

    quad(-0.09, 0.07, 0.0, 0.85, PANTS, phase)
    quad(0.09, 0.07, 0.0, 0.85, PANTS, -phase)
    quad(0.0, TORSO_HALF_W - 0.03, 0.85, hgt - 0.25, SHIRT)
    quad(-TORSO_HALF_W, 0.05, 0.85, hgt - 0.27, SHIRT, -phase)
    quad(TORSO_HALF_W, 0.05, 0.85, hgt - 0.27, SHIRT, phase)
    quad(0.0, 0.06, hgt - 0.27, hgt - HEAD_M + 0.02, SKIN)  # neck
    centre = np.array([[w.x, w.y, hgt - HEAD_M / 2]])
    uv, ok = to_image(cam, pose, centre)
    if ok[0]:
        z = to_drone(pose, centre)[0]
        depth = max(0.2, z[0] * math.cos(math.radians(pose.pitch_deg)) + z[2] * math.sin(math.radians(pose.pitch_deg)))
        r = int(round(cam.fx * HEAD_HALF_W / depth))
        c = tuple(int(round(v)) for v in uv[0])
        cv2.circle(img, c, max(r, 1), HAIR, -1, cv2.LINE_AA)  # seen from behind and above: hair


# ------------------------------------------------------------------ detector
class OracleDetector:
    """The walker's visible box, as kite/fly/sim/camera.py defines it, plus noise and misses."""

    def __init__(self, rng: np.random.Generator, noise_px: float = 2.0, miss: float = 0.05):
        self.rng, self.noise_px, self.miss = rng, noise_px, miss

    def __call__(self, frame: Frame) -> list[Detection]:
        g = getattr(frame, "geom", None)
        box = visible_box(g)
        if box is None or self.rng.random() < self.miss:
            return []
        x1, y1, x2, y2, torso = box
        n = self.rng.normal(0.0, self.noise_px, 4)
        conf = self.rng.uniform(0.4, 0.9) if torso else self.rng.uniform(0.25, 0.6)
        W, H = frame.width, frame.height
        return [Detection(max(0.0, x1 + n[0]), max(0.0, y1 + n[1]), min(W, x2 + n[2]), min(H, y2 + n[3]), conf)]


# ------------------------------------------------------------------ video link
@dataclass
class SimFrame(Frame):
    geom: tuple | None = None  # the walker's true geometry at capture (for the oracle detector)
    t_capture: float = 0.0


class DelayedVideo(FrameSource):
    """Frames captured at t_capture become readable at t_capture + lag_s (Frame.t = arrival, as TelloVideoSource)."""

    name = "sim"

    def __init__(self, lag_s: float):
        self.lag_s = lag_s
        self._queue: deque[SimFrame] = deque()
        self._latest: SimFrame | None = None
        self._seq = 0
        self.t_now = 0.0

    def push(self, image: np.ndarray, t_capture: float, geom) -> None:
        self._seq += 1
        image.flags.writeable = False
        self._queue.append(SimFrame(image, t_capture + self.lag_s, self._seq, self.name, geom, t_capture))

    def advance(self, t: float) -> None:
        self.t_now = t
        while self._queue and self._queue[0].t <= t:
            self._latest = self._queue.popleft()

    def read(self) -> SimFrame | None:
        return self._latest
