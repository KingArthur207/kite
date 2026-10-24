"""Detections (+ an optional metric depth map) -> earpiece objects. Pure numpy and geometry, no model.

Three things the stand-in cannot do, all from one depth map:

1. Self-calibrated pitch. Ground pixels in the lower middle of the frame lie on a plane. In camera
   coordinates (Y down, Z forward) a camera pitched down by p at height h sees the ground at
   Y = h / cos p - Z tan p, so a line fit of Y against Z over those pixels gives tan p from the slope
   (which does not depend on the depth model's scale), and the scale that makes the intercept equal
   the known camera height. A wrong pitch is the largest distance error the stand-in has (README:
   2 degrees doubles a 30 m estimate); this measures it every frame.
2. Range for things that are not standing on the ground: a branch or a sign at head height has its
   bottom row high in the image, which the flat-ground rule turns into a far distance. Depth inside
   the object's mask does not care.
3. Height above the ground of an object's bottom and top edges, which separates an overhead hazard
   (the cane passes under it) from a pole (the cane finds it) and stairs up from stairs down.

Geometry helpers (bearing, flat-ground distance, drone-to-walker) are the stand-in's, imported
read-only, so both detectors agree to the centimetre on the same box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from standin_detect import EDGE_PX, MAX_D, MIN_D, PERSON_W, bearing_deg, below_horizon, distance_m

from .vocab import MOVING, TYPICAL_H

# Heights above the ground (m) for the walker's body: a cane sweeps below KNEE_M, the head clears HEAD_M.
KNEE_M = 0.6
HEAD_M = 2.1
ELEVATED_M = 0.45          # an object whose bottom edge is this far above the ground is not standing on it
STAIRS_UP_M = 0.35         # stairs whose top edge is this high are steps up
HIDDEN_FOOT_RANGE = 10.0   # m; depth overrides a ground row that reads more than 1/0.6 times farther
OVERHEAD_RANGE = 8.0       # m; beyond this a 1-degree error moves an edge's height by > 0.14 m: don't judge it
PITCH_RANGE = (-30.0, 50.0)
GROUND_ROWS = (0.70, 0.98)  # fraction of the image height searched for ground pixels
GROUND_COLS = (0.25, 0.75)
MIN_GROUND_PX = 400


@dataclass
class Det:
    """One detection in image pixels. `mask` is an optional bool array at the frame's size."""
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str
    conf: float
    label: str = ""
    tid: str | None = None          # tracker id, prefixed by the model that made it ("c12", "o3")
    mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class Calib:
    """Per-frame camera estimate. `source` is "given" (flags) or "depth" (fitted this frame)."""
    pitch: float
    cam_h: float
    scale: float = 1.0
    source: str = "given"


# ------------------------------------------------------------------ ground fit
def fit_ground(depth: np.ndarray, fx: float, cam_h: float, exclude: np.ndarray | None = None,
               given_pitch: float = 0.0, fit_pitch: bool = True) -> Calib:
    """Metric depth (h, w, metres along the optical axis), fx in depth-map pixels -> Calib.

    Fits Y = a + b Z over ground candidates (lower middle of the frame, outside `exclude`), drops the
    worst residuals twice, and accepts the fit when it is tight and the pitch is plausible. With
    fit_pitch=False (the pitch is known, e.g. the sim's level cameras) only the scale is fitted, with
    b = -tan(given_pitch). On failure: the given pitch with scale 1, source "given"."""
    h, w = depth.shape
    r0, r1 = int(h * GROUND_ROWS[0]), int(h * GROUND_ROWS[1])
    c0, c1 = int(w * GROUND_COLS[0]), int(w * GROUND_COLS[1])
    vv, uu = np.mgrid[r0:r1, c0:c1]
    z = depth[r0:r1, c0:c1].astype(np.float64)
    ok = np.isfinite(z) & (z > 0.1)
    if exclude is not None:
        ok &= ~exclude[r0:r1, c0:c1]
    fallback = Calib(given_pitch, cam_h, 1.0, "given")
    if ok.sum() < MIN_GROUND_PX:
        return fallback
    z, v = z[ok], vv[ok].astype(np.float64)
    if np.ptp(z) < 0.3:                      # no depth gradient down the frame: a wall, not ground
        return fallback
    y = (v - h / 2) / fx * z

    def fit(m):
        if fit_pitch:
            return np.polyfit(z[m], y[m], 1)
        b = -math.tan(math.radians(given_pitch))
        return b, float(np.median(y[m] - b * z[m]))

    keep = np.ones(z.size, bool)
    for _ in range(3):
        if keep.sum() < MIN_GROUND_PX:
            return fallback
        b, a = fit(keep)
        res = np.abs(y - (a + b * z))
        keep = res <= max(np.percentile(res[keep], 80), 1e-6)
    b, a = fit(keep)
    pitch = math.degrees(math.atan(-b))
    rel = float(np.median(np.abs(y[keep] - (a + b * z[keep])))) / max(abs(a), 1e-6)
    if a <= 0 or rel > 0.08 or not PITCH_RANGE[0] <= pitch <= PITCH_RANGE[1]:
        return fallback
    scale = float(cam_h / (a * math.cos(math.radians(pitch))))
    if not 0.25 <= scale <= 4.0:
        return fallback
    return Calib(round(float(pitch), 2), cam_h, round(scale, 3), "depth")


class CalibFilter:
    """Smooths per-frame fits (exponential, alpha 0.3) and holds the last good one for `hold` frames
    before falling back to the given pitch."""

    def __init__(self, given_pitch: float, cam_h: float, alpha: float = 0.3, hold: int = 30):
        self.given, self.cam_h, self.alpha, self.hold = given_pitch, cam_h, alpha, hold
        self.cur: Calib | None = None
        self.miss = 0

    def update(self, c: Calib) -> Calib:
        if c.source == "depth":
            self.miss = 0
            if self.cur is None:
                self.cur = c
            else:
                a = self.alpha
                self.cur = Calib(round((1 - a) * self.cur.pitch + a * c.pitch, 2), self.cam_h,
                                 round((1 - a) * self.cur.scale + a * c.scale, 3), "depth")
        else:
            self.miss += 1
            if self.miss > self.hold:
                self.cur = None
        return self.cur or Calib(self.given, self.cam_h, 1.0, "given")


# ------------------------------------------------------------------ the walking corridor from depth
CORRIDOR_HALF_M = 0.5      # m either side of the walker's line
CORRIDOR_NEAR_M, CORRIDOR_FAR_M = 0.6, 4.0
RISE_M = (0.3, 2.0)        # height above the fitted ground that counts as something in the way
DROP_M = -0.3              # this far below the fitted ground is a drop (curb, steps down, open trench)
MIN_FRAC = 0.06            # share of the corridor's pixels that must agree


def ground_points(depth: np.ndarray, fx: float, calib: Calib):
    """Depth map (z, metres, model scale) -> per-pixel (forward m, lateral m, height above ground m),
    using the fitted pitch and scale. fx in depth-map pixels."""
    h, w = depth.shape
    z = depth.astype(np.float64) * calib.scale
    vv, uu = np.mgrid[0:h, 0:w]
    x = (uu - w / 2) / fx * z
    y = (vv - h / 2) / fx * z
    p = math.radians(calib.pitch)
    fwd = z * math.cos(p) - y * math.sin(p)
    height = calib.cam_h - (y * math.cos(p) + z * math.sin(p))
    return fwd, x, height


def corridor_hazards(depth: np.ndarray, fx: float, calib: Calib, ignore_bottom: float = 0.0) -> list[dict]:
    """Class-agnostic check of the next CORRIDOR_FAR_M metres of the walker's path: something rising out of
    the ground (`obstacle`) and the ground falling away (`curb`). Only with a depth-fitted calibration:
    with a guessed pitch the height of every pixel is a guess too. `ignore_bottom`: fraction of the rows at
    the bottom that show the wearer (the sim draws the walker's head and shoulders there). Returns earpiece
    objects."""
    if calib.source != "depth":
        return []
    fwd, lat, hgt = ground_points(depth, fx, calib)
    inside = (np.abs(lat) <= CORRIDOR_HALF_M) & (fwd >= CORRIDOR_NEAR_M) & (fwd <= CORRIDOR_FAR_M)
    if ignore_bottom > 0:
        inside[int(depth.shape[0] * (1 - ignore_bottom)):] = False
    n = int(inside.sum())
    if n < 200:
        return []
    out = []
    for kind, sel in (("obstacle", inside & (hgt >= RISE_M[0]) & (hgt <= RISE_M[1])),
                      ("curb", inside & (hgt <= DROP_M))):
        if sel.sum() < MIN_FRAC * n:
            continue
        f = fwd[sel]
        if kind == "curb":
            # the lower ground is seen past the edge that hides it: the ray over the edge (height 0) lands
            # at height h < 0 at forward f, so the edge is at f * cam_h / (cam_h - h)
            f = f * calib.cam_h / (calib.cam_h - hgt[sel])
        near = float(np.percentile(f, 10))
        b = math.degrees(math.atan2(float(np.median(lat[sel])), max(near, 0.1)))
        out.append({"id": f"corridor-{kind}", "kind": kind, "bearing": round(b, 1), "distance": round(near, 2),
                    "confidence": round(min(1.0, float(sel.sum()) / n * 3), 2), "cut_off": False,
                    "range": "depth", "label": f"corridor {kind}"})
    return out


# ------------------------------------------------------------------ per-object geometry
def height_at(y: float, d: float, cam_h: float, fx: float, height: int, pitch: float) -> float:
    """Height above the ground (m) of the point seen at image row y, at forward distance d."""
    return cam_h - d * math.tan(below_horizon(y, fx, height, pitch))


def box_depth(depth: np.ndarray, det: Det, img_w: int, img_h: int, pct: float = 25.0) -> float | None:
    """A low percentile of the depth inside the detection (its mask if any, else the middle of its box),
    so the near face of the object wins over the background seen around a thin pole."""
    dh, dw = depth.shape
    sx, sy = dw / img_w, dh / img_h
    x1, x2 = int(det.x1 * sx), max(int(det.x1 * sx) + 1, int(math.ceil(det.x2 * sx)))
    y1, y2 = int(det.y1 * sy), max(int(det.y1 * sy) + 1, int(math.ceil(det.y2 * sy)))
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(dw, x2), min(dh, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth[y1:y2, x1:x2]
    if det.mask is not None:
        m = _resize_bool(det.mask, (dh, dw))[y1:y2, x1:x2]
        vals = patch[m] if m.sum() >= 4 else patch
    elif (det.y2 - det.y1) >= 3 * (det.x2 - det.x1):
        vals, pct = patch, min(pct, 10.0)    # a thin upright box (pole, stem): the object is a sliver of it
    else:
        bw, bh = x2 - x1, y2 - y1
        vals = patch[bh // 4: max(bh // 4 + 1, 3 * bh // 4), bw // 4: max(bw // 4 + 1, 3 * bw // 4)]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.percentile(vals, pct)) if vals.size else None


def foot_x(depth: np.ndarray, det: Det, img_w: int, img_h: int) -> float | None:
    """Image x of where the object meets the ground: the nearest columns in the bottom quarter of its box.
    A street tree's box spans its whole crown, but the walker can only hit the trunk."""
    dh, dw = depth.shape
    sx, sy = dw / img_w, dh / img_h
    x1, x2 = max(0, int(det.x1 * sx)), min(dw, int(math.ceil(det.x2 * sx)))
    y2 = min(dh, int(math.ceil(det.y2 * sy)))
    y1 = max(0, min(y2 - 1, int(y2 - max(2, (det.y2 - det.y1) * sy / 4))))
    if x2 - x1 < 3 or y2 <= y1:
        return None
    col = np.nanmedian(np.where(np.isfinite(depth[y1:y2, x1:x2]), depth[y1:y2, x1:x2], np.nan), axis=0)
    near = np.isfinite(col) & (col <= 1.15 * np.nanmin(col)) if np.isfinite(col).any() \
        else None
    if near is None or not near.any():
        return None
    return (x1 + float(np.mean(np.nonzero(near)[0])) + 0.5) / sx


def _resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    rows = (np.arange(shape[0]) * h / shape[0]).astype(int)
    cols = (np.arange(shape[1]) * w / shape[1]).astype(int)
    return mask[rows][:, cols]


def object_from_det(det: Det, i: int, fx: float, calib: Calib, width: int, height: int,
                    depth: np.ndarray | None = None) -> dict | None:
    """One detection -> one earpiece object (plus fields the CLI uses and to_event_objects strips):
    cut_off, range (how distance was measured: ground, depth or size), h_bottom/h_top (m above ground)."""
    kind, pitch, cam_h = det.kind, calib.pitch, calib.cam_h
    real_h = TYPICAL_H.get(kind, 1.0)
    cut, top_cut = det.y2 >= height - EDGE_PX, det.y1 <= EDGE_PX
    d_ground = distance_m(det.y2, det.y2 - det.y1, cam_h, fx, height, real_h, pitch,
                          y_top=det.y1 if cut or top_cut else None)
    stable = below_horizon(det.y2, fx, height, pitch) > math.atan(12 / fx)
    how = "ground" if stable and not cut else "size"
    d = d_ground
    if cut and top_cut and kind == "person":
        d = max(MIN_D, min(d, PERSON_W * fx / max(det.x2 - det.x1, 1.0)))
    d_depth = None
    if depth is not None:
        z = box_depth(depth, det, width, height)
        if z is not None:
            d_depth = min(MAX_D, max(MIN_D, z * calib.scale))
    if d_depth is not None:
        # The ground row is the better range when the object stands on visible ground (that is what the
        # stand-in does, and the fitted pitch keeps it honest); depth takes over when the bottom edge is cut
        # off or at/above the horizon, or when depth says the bottom edge is in the air (a sign, a branch).
        floating = stable and not cut and d_depth <= OVERHEAD_RANGE and \
            height_at(det.y2, d_depth, cam_h, fx, height, pitch) > ELEVATED_M
        # the box's bottom row is not where it meets the ground (a box around part of something, or its foot
        # hidden): depth puts it much nearer than that row does
        hidden_foot = how == "ground" and d_depth <= HIDDEN_FOOT_RANGE and d_depth < 0.6 * d_ground
        if cut:
            # cut off by the bottom edge: the stand-in's rows only bound the range from above (it was tuned for
            # "Stop." on people close to a chest camera), so keep the nearer of the two
            if d_depth < d:
                d, how = d_depth, "depth"
        elif how == "size" or floating or hidden_foot:
            d, how = d_depth, "depth"
    h_bot = height_at(det.y2, d, cam_h, fx, height, pitch) if not cut else 0.0
    h_top = height_at(det.y1, d, cam_h, fx, height, pitch) if not top_cut else max(HEAD_M + 0.5, real_h)
    elevated = how == "depth" and not cut and d <= OVERHEAD_RANGE and h_bot > ELEVATED_M
    if elevated and kind not in MOVING:      # people and cars do not float: a bad depth reading, not a sign
        if h_bot >= HEAD_M:
            return None                      # entirely above head height: nothing to walk into
        kind = "overhead"
    elif kind == "overhead" and how == "ground":
        kind = "obstacle"                    # a "branch" standing on the ground is just an obstacle
    if kind == "stairs":
        kind = "stairs_up" if h_top > STAIRS_UP_M else "stairs_down"
    cx = (det.x1 + det.x2) / 2
    if depth is not None and det.kind not in MOVING and (det.x2 - det.x1) > 0.5 * (det.y2 - det.y1):
        cx = foot_x(depth, det, width, height) or cx      # wide boxes (a tree's crown): aim at the trunk
    return {"id": f"{det.tid if det.tid is not None else i}", "kind": kind,
            "bearing": round(bearing_deg(cx, fx, width), 1), "distance": round(d, 2),
            "confidence": round(float(det.conf), 2), "cut_off": cut, "range": how,
            "h_bottom": round(h_bot, 2), "h_top": round(h_top, 2), "label": det.label or det.kind}


def objects_from_dets(dets: list[Det], fx: float, calib: Calib, width: int, height: int,
                      depth: np.ndarray | None = None) -> list[dict]:
    out = []
    for i, det in enumerate(dets):
        o = object_from_det(det, i, fx, calib, width, height, depth)
        if o is not None:
            out.append(o)
    return out


EVENT_KEYS = ("id", "kind", "bearing", "distance", "closing_speed", "confidence")


def to_event_objects(objs: list[dict]) -> list[dict]:
    """Only the fields of the `obstacles` contract (earpiece/earpiece/events.py), None dropped."""
    return [{k: o[k] for k in EVENT_KEYS if o.get(k) is not None} for o in objs]


def nms_agnostic(dets: list[Det], iou: float = 0.6, contain: float = 0.85) -> list[Det]:
    """Class-agnostic NMS: two prompts for one thing ("bike rack" and "electric scooter" on the same
    scooter) keep the higher-scoring one. A box mostly inside a stronger box of the same kind goes too
    (a "pole" inside a "street light"); a person inside a bus's box stays."""
    keep: list[Det] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(_iou(d, k) < iou and not (d.kind == k.kind and _inside(d, k) >= contain) for k in keep):
            keep.append(d)
    return keep


def _inter(a: Det, b: Det) -> float:
    return max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))


def _area(a: Det) -> float:
    return max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)


def _iou(a: Det, b: Det) -> float:
    i = _inter(a, b)
    return i / max(_area(a) + _area(b) - i, 1e-6)


def _inside(a: Det, b: Det) -> float:
    """Fraction of a's box inside b's."""
    return _inter(a, b) / max(_area(a), 1e-6)
