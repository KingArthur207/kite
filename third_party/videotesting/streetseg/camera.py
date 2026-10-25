"""Camera intrinsics and the camera's pose.

``process`` keeps the BGR frame at the size the source produced. ``fx`` and
``fy`` are in pixels of that frame. Pitch is degrees, positive looking down.
Height is meters above the ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .labels import norm_label
from .types import Box

FRAME_W = 960
FRAME_H = 720

# Typical real heights, used only to turn a box plus fy into meters.
REAL_HEIGHT_M = {
    "person": 1.70,
    "rider": 1.70,
    "pedestrian": 1.70,
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.20,
    "bus": 3.00,
    "truck": 2.50,
    "train": 3.50,
    "dog": 0.50,
    "chair": 0.90,
    "bench": 0.90,
    "backpack": 0.45,
    "suitcase": 0.70,
}


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    role: str = "drone"
    cx: float | None = None
    cy: float | None = None

    def with_role(self, role: str) -> Camera:
        return Camera(self.fx, self.fy, role, self.cx, self.cy)

    def centered(self, width: float, height: float) -> Camera:
        if self.cx is not None and self.cy is not None:
            return self
        return Camera(
            self.fx,
            self.fy,
            self.role,
            self.cx if self.cx is not None else width / 2.0,
            self.cy if self.cy is not None else height / 2.0,
        )

    def scaled(self, sx: float, sy: float, width: float, height: float) -> Camera:
        """Scale intrinsics from a ``width``×``height`` frame into the resized one."""
        cx = self.cx if self.cx is not None else width / 2.0
        cy = self.cy if self.cy is not None else height / 2.0
        return Camera(self.fx * sx, self.fy * sy, self.role, cx * sx, cy * sy)


@dataclass(frozen=True)
class Pose:
    height: float
    pitch: float


def focal_length(width: float, height: float, diagonal_fov_deg: float) -> float:
    """Square-pixel focal length from a diagonal field of view, in degrees."""
    diag = math.hypot(width, height)
    return (diag / 2.0) / math.tan(math.radians(diagonal_fov_deg / 2.0))


def focal_from_horizontal_fov(width: float, fov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))


# DJI Tello: 82.6° diagonal field of view, native 960×720.
TELLO_F = focal_length(FRAME_W, FRAME_H, 82.6)
TELLO_CAMERA = Camera(TELLO_F, TELLO_F, "drone")

# A laptop or phone chest camera, until someone calibrates it. 70° horizontal.
CHEST_F = focal_from_horizontal_fov(FRAME_W, 70.0)
CHEST_CAMERA = Camera(CHEST_F, CHEST_F, "chest")

# Height and pitch that put the next 5 m of ground near row 0.55 of a Tello frame.
# That is the band the synthetic street and the original scorer were tuned on.
DEMO_POSE = Pose(height=1.34, pitch=12.0)
DRONE_POSE = Pose(height=1.6, pitch=12.0)
CHEST_POSE = Pose(height=1.3, pitch=20.0)
NEAR_M = 5.0


def ground_row(frame_height: int, camera: Camera, pose: Pose, near_m: float) -> int:
    """Row where the ground ``near_m`` ahead projects.

    Rows below this are the ground the walker is about to step on. A level
    camera puts that strip at the bottom. Looking down pulls it up the frame.
    """
    if frame_height < 2 or camera.fy <= 0 or pose.height <= 0 or near_m <= 0:
        return int(frame_height * 0.55)
    cy = camera.cy if camera.cy is not None else frame_height / 2.0
    alpha = math.atan2(pose.height, near_m)
    beta = alpha - math.radians(pose.pitch)
    if abs(math.degrees(beta)) >= 89.0:
        return 0 if beta > 0 else frame_height - 2
    row = cy + camera.fy * math.tan(beta)
    return int(min(max(round(row), 0), frame_height - 2))


def ground_near_frac(frame_height: int, camera: Camera, pose: Pose, near_m: float) -> float:
    if frame_height <= 0:
        return 0.55
    return ground_row(frame_height, camera, pose, near_m) / float(frame_height)


def metric_distance(box: Box, camera: Camera | None, frame_height: int) -> float | None:
    """Meters to a known-height object, from its pixel height and ``fy``."""
    if camera is None or camera.fy <= 0 or frame_height <= 0:
        return None
    real = REAL_HEIGHT_M.get(norm_label(box.label))
    if real is None:
        return None
    pixel_h = float(box.xyxy[3] - box.xyxy[1])
    if pixel_h < 2:
        return None
    return camera.fy * real / pixel_h


def locate(box: Box, camera: Camera, shape: tuple[int, int]) -> dict[str, float]:
    """Distance and lateral offset in meters. Positive lateral is to the right."""
    dist = metric_distance(box, camera, shape[0])
    if dist is None:
        return {}
    out = {"distance_m": round(float(dist), 2)}
    if camera.fx > 0:
        cx = camera.cx if camera.cx is not None else shape[1] / 2.0
        lateral = (box.center[0] - cx) / camera.fx * dist
        out["lateral_m"] = round(float(lateral), 2)
    return out


def fit_frame(frame: np.ndarray, camera: Camera) -> tuple[np.ndarray, Camera]:
    """Return the whole frame and intrinsics for that frame.

    Nothing is cropped and nothing is stretched. A portrait phone frame stays
    portrait, so the sky, the signals, and a far crossing stay in view. A
    network that wants a fixed input size resizes on its own and paints the
    mask back onto this frame.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("frame is empty")
    return np.ascontiguousarray(frame), camera.centered(width, height)
