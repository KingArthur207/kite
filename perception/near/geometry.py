"""Rays, the ground plane and the walker's frame. Pure maths on undistorted pixels.

Camera frame: x right, y down, z along the optical axis. The camera is `cam_h` metres above flat ground and
pitched down by `pitch` degrees (no roll; a chest clip sits close to level side to side). The walker's frame:
forward (horizontal, along the camera's heading), lateral (+ right), height above the ground.

A pixel's ray direction per unit z is (xn, yn, 1) with xn = (u - cx)/fx, yn = (v - cy)/fy. Rotated into the
walker's frame it points `down = yn cos p + sin p` downward and `fwd = cos p - yn sin p` forward per unit z,
so it meets the ground at z = cam_h / down: forward = z * fwd, lateral = z * xn.
"""

from __future__ import annotations

import math

import numpy as np

from .intrinsics import CameraProfile

MIN_BELOW_HORIZON = math.radians(1.0)    # rays flatter than this hit the ground too far away to be useful


def norm_xy(prof: CameraProfile, u: float, v: float) -> tuple[float, float]:
    return (u - prof.cx) / prof.fx, (v - prof.cy) / prof.fy


def angle_below_horizon(prof: CameraProfile, v: float, pitch_deg: float) -> float:
    """Radians the ray through row v points below horizontal."""
    return math.radians(pitch_deg) + math.atan((v - prof.cy) / prof.fy)


def ground_point(prof: CameraProfile, u: float, v: float, cam_h: float, pitch_deg: float
                 ) -> tuple[float, float] | None:
    """(forward, lateral) metres where the ray through pixel (u, v) meets the ground, None above the horizon."""
    xn, yn = norm_xy(prof, u, v)
    p = math.radians(pitch_deg)
    down = yn * math.cos(p) + math.sin(p)
    if math.atan2(down, math.cos(p) - yn * math.sin(p)) < MIN_BELOW_HORIZON:
        return None
    z = cam_h / down
    return z * (math.cos(p) - yn * math.sin(p)), z * xn


def z_to_forward(z: float, yn: float, pitch_deg: float) -> float:
    """Depth along the optical axis at a pixel with normalised row yn -> horizontal forward distance."""
    p = math.radians(pitch_deg)
    return z * (math.cos(p) - yn * math.sin(p))


def forward_to_z(forward: float, yn: float, pitch_deg: float) -> float:
    p = math.radians(pitch_deg)
    return forward / max(1e-6, math.cos(p) - yn * math.sin(p))


def lateral_at(prof: CameraProfile, u: float, v: float, forward: float, pitch_deg: float) -> float:
    """Lateral offset (m) of the point seen at pixel (u, v) that is `forward` metres ahead."""
    xn, yn = norm_xy(prof, u, v)
    return forward_to_z(forward, yn, pitch_deg) * xn


def plane_depth(prof: CameraProfile, rows: np.ndarray, cam_h: float, pitch_deg: float) -> np.ndarray:
    """Optical-axis depth z of the ground seen at each image row (inf at or above the horizon)."""
    yn = (np.asarray(rows, float) - prof.cy) / prof.fy
    p = math.radians(pitch_deg)
    down = yn * math.cos(p) + math.sin(p)
    with np.errstate(divide="ignore"):
        z = np.where(down > 1e-4, cam_h / np.maximum(down, 1e-9), np.inf)
    return z


def horizon_row(prof: CameraProfile, pitch_deg: float) -> float:
    """Image row of the horizon (where the ground plane vanishes)."""
    return prof.cy - prof.fy * math.tan(math.radians(pitch_deg))


def points_from_depth(z: np.ndarray, prof: CameraProfile, cam_h: float, pitch_deg: float):
    """Depth map (optical-axis metres, at prof's size) -> per-pixel (forward, lateral, height above ground)."""
    h, w = z.shape
    vv, uu = np.mgrid[0:h, 0:w]
    xn = (uu - prof.cx) / prof.fx
    yn = (vv - prof.cy) / prof.fy
    p = math.radians(pitch_deg)
    fwd = z * (math.cos(p) - yn * math.sin(p))
    lat = z * xn
    height = cam_h - z * (yn * math.cos(p) + math.sin(p))
    return fwd, lat, height
