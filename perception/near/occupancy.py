"""Class-agnostic corridor occupancy: what the detector has no name for still gets a priority.

Every depth pixel, with the anchored scale and the fitted pitch, becomes a point (forward, lateral, height above
the ground). Inside the corridor, in 0.25 m slices from 0.5 m to 8 m:

  rise   points 0.18 m to 2.0 m above the ground (a step up to just over head height). A slice is occupied when the
         rising points cover at least MIN_AREA_M2 of surface (each pixel covers (z/fx) * (z/fy) m^2 facing the
         camera), so the test means the same thing at 1 m and at 7 m. The nearest occupied slice is the obstacle.
         If it spans most of the corridor's width and rises above 1.2 m it is reported as a wall (an open door
         seen edge-on is a narrow rise and stays an obstacle).
  drop   points more than DROP_M below the ground plane: the ray went past an edge and landed lower (a curb down,
         steps down, a loading dock). The edge is where the ray crossed height 0:
         f_edge = f * cam_h / (cam_h - height). A drop needs a run of such points beyond the edge, not a speckle.

Pixels inside named detection boxes are left out, so a person is ranged once (by the cue fusion), not twice.
This is the "depth-based collision" idea of DepthLab's localized-depth tools, run on the walker's swept volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .depth import DepthFrame
from .geometry import points_from_depth

RISE_M = (0.18, 2.0)
DROP_M = 0.12
NEAR_M, FAR_M, BIN_M = 0.5, 8.0, 0.25
MIN_AREA_M2 = 0.06
MIN_CONF = 0.25
PITCH_SLACK_DEG = 1.0
DROP_FAR_M = 6.0


@dataclass
class OccHit:
    kind: str           # obstacle | wall | curb | stairs_down
    forward: float
    lateral: float
    bearing: float
    area_m2: float
    confidence: float
    extent: float = 0.0     # lateral span (rise) or drop depth (m)


def occupancy(df: DepthFrame, cam_h: float, pitch: float, half: float, exclude: np.ndarray | None = None,
              ignore_bottom: float = 0.0, min_area: float = MIN_AREA_M2) -> list[OccHit]:
    z, conf, pd = df.z.astype(np.float64), df.conf, df.prof
    fwd, lat, hgt = points_from_depth(z, pd, cam_h, pitch)
    ok = np.isfinite(z) & (conf >= MIN_CONF) & (np.abs(lat) <= half)
    if exclude is not None:
        ok &= ~exclude
    if ignore_bottom > 0:
        ok[int(z.shape[0] * (1 - ignore_bottom)):] = False
    out: list[OccHit] = []
    pix_area = (z / pd.fx) * (z / pd.fy)
    rise = ok & (hgt >= RISE_M[0]) & (hgt <= RISE_M[1]) & (fwd >= NEAR_M) & (fwd <= FAR_M)
    if rise.any():
        f, a, c = fwd[rise], (pix_area * conf)[rise], conf[rise]
        bins = np.floor((f - NEAR_M) / BIN_M).astype(int)
        area = np.bincount(bins, weights=a, minlength=int((FAR_M - NEAR_M) / BIN_M) + 1)
        # two neighbouring slices together: an object straddling a slice boundary still counts
        both = area + np.concatenate([area[1:], [0.0]])
        hit = np.nonzero(both >= min_area)[0]
        if hit.size:
            b = int(hit[0])
            sel = (bins >= b) & (bins <= b + 1)
            near = float(np.percentile(f[sel], 10))
            la = lat[rise][sel]
            span = float(np.percentile(la, 95) - np.percentile(la, 5))
            top = float(np.percentile(hgt[rise][sel], 95))
            kind = "wall" if span >= 1.6 * half and top >= 1.2 else "obstacle"
            lc = float(np.median(la))
            out.append(OccHit(kind, near, lc, math.degrees(math.atan2(lc, near)), float(both[b]),
                              round(min(1.0, float(np.mean(c[sel])) * min(1.0, both[b] / (2 * min_area))), 2),
                              span))
    # a 1 degree pitch error tilts the fitted ground by f * tan(1 deg): far pavement would look like a drop
    drop_at = DROP_M + fwd * math.tan(math.radians(PITCH_SLACK_DEG))
    drop = ok & (hgt <= -drop_at) & (fwd >= NEAR_M) & (fwd <= DROP_FAR_M * 1.5)
    n_inside = int((ok & (fwd >= NEAR_M) & (fwd <= DROP_FAR_M * 1.5)).sum())
    if drop.sum() >= max(40, 0.03 * n_inside):
        h = hgt[drop]
        edge = fwd[drop] * cam_h / (cam_h - h)
        e = float(np.percentile(edge, 10))
        beyond = ok & (fwd >= e) & (fwd <= DROP_FAR_M * 1.5)
        frac = float(drop.sum()) / max(1, int(beyond.sum()))
        # and the pavement must be there, level, just before the edge: a drop is a break in the ground plane
        before = ok & (fwd >= max(NEAR_M, e - 1.0)) & (fwd < e)
        level = float((np.abs(hgt[before]) < 0.1).mean()) if before.sum() >= 20 else 0.0
        if frac >= 0.4 and level >= 0.6 and e <= DROP_FAR_M:
            depth_m = float(-np.median(h))
            lc = float(np.median(lat[drop]))
            out.append(OccHit("curb" if depth_m < 0.3 else "stairs_down", e, lc, math.degrees(math.atan2(lc, e)),
                              0.0, round(min(1.0, frac * float(np.mean(conf[drop]))), 2), depth_m))
    return out


def box_mask(boxes, shape: tuple[int, int], frame_w: int, frame_h: int, grow: float = 0.0) -> np.ndarray:
    """Depth-map pixels inside the boxes, each grown by `grow` of its size plus 2 px (box jitter, depth blur)."""
    m = np.zeros(shape, bool)
    sy, sx = shape[0] / frame_h, shape[1] / frame_w
    for b in boxes:
        gx, gy = grow * (b[2] - b[0]) * sx + 2, grow * (b[3] - b[1]) * sy + 2
        m[max(0, int(b[1] * sy - gy)):int(math.ceil(b[3] * sy + gy)),
          max(0, int(b[0] * sx - gx)):int(math.ceil(b[2] * sx + gx))] = True
    return m
