"""Scores for the depth shoot-out: the standard ones, and the ones a blind walker's earpiece cares about.

Standard (over valid ground-truth pixels): AbsRel, RMSE, delta < 1.25.
Near field: the same errors over the walking corridor only (0.5 m either side of the walker's line, up to 5 m
ahead, any height), because a 30 % error on a facade 40 m away matters less than 10 % on a bin 1.5 m away.
Priority: the band of the nearest thing in the corridor (P1 under 2 m, P2 2 to 5 m, P3 5 to 10 m, clear),
computed the same way from the predicted and the true depth; "right" when the bands agree. P1 recall (true P1
frames called P1) and false P1 (a P1 call when the truth is P2 or farther) are the two numbers that matter
most: a missed P1 is a walk into a bin, a false one is a beep crying wolf.
"""

from __future__ import annotations

import math

import numpy as np

from ..depth_zoo import Intrinsics, down_vector

BANDS = ("P1", "P2", "P3", "clear")
P1_M, P2_M, P3_M = 2.0, 5.0, 10.0
CORRIDOR_HALF_M = 0.5
RISE_M = (0.25, 2.0)            # above the floor: a bin, a pole, a person; below 0.25 m is curb or kerb height
MAX_GT = 80.0                   # m; ground truth beyond this is not scored (sky, horizon)


def band(d: float | None) -> str:
    if d is None or not math.isfinite(d) or d > P3_M:
        return "clear"
    return "P1" if d < P1_M else "P2" if d < P2_M else "P3"


def ground_coords(depth: np.ndarray, K: Intrinsics, down: np.ndarray, cam_h: float):
    """Per pixel: forward (m, along the floor), lateral (m), height above the floor (m)."""
    if (K.width, K.height) != (depth.shape[1], depth.shape[0]):
        K = K.scaled(depth.shape[1], depth.shape[0])
    n = np.asarray(down, np.float64) / np.linalg.norm(down)
    f = np.array([0.0, 0.0, 1.0]) - n[2] * n
    f /= np.linalg.norm(f)
    lat = np.cross(n, f)
    P = K.rays() * depth[..., None].astype(np.float64)
    return P @ f, P @ lat, cam_h - P @ n


def nearest_obstacle(depth: np.ndarray, K: Intrinsics, down: np.ndarray, cam_h: float, valid: np.ndarray | None = None,
                     half: float = CORRIDOR_HALF_M, rise=RISE_M, max_range: float = P3_M,
                     min_frac: float = 0.0015) -> float | None:
    """Forward distance (m) of the nearest thing rising out of the floor in the corridor, or None. Robust to a few
    stray pixels: the distance at which `min_frac` of the frame's pixels have been passed."""
    fwd, lat, hgt = ground_coords(depth, K, down, cam_h)
    sel = (depth > 0) & np.isfinite(depth) & (fwd > 0.3) & (fwd <= max_range) & (np.abs(lat) <= half) & \
        (hgt >= rise[0]) & (hgt <= rise[1])
    if valid is not None:
        sel &= valid
    k = max(20, int(min_frac * depth.size))
    if sel.sum() < k:
        return None
    return float(np.partition(fwd[sel], k - 1)[k - 1])


def corridor_mask(gt: np.ndarray, K: Intrinsics, down: np.ndarray, cam_h: float, half: float = CORRIDOR_HALF_M,
                  far: float = P2_M) -> np.ndarray:
    fwd, lat, _ = ground_coords(gt, K, down, cam_h)
    return (gt > 0) & (fwd > 0) & (fwd <= far) & (np.abs(lat) <= half)


def dense(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    m = valid & (gt > 0) & (gt <= MAX_GT) & np.isfinite(pred) & (pred > 0)
    if m.sum() < 10:
        return {"absrel": None, "rmse": None, "d1": None, "n": int(m.sum())}
    p, g = pred[m].astype(np.float64), gt[m].astype(np.float64)
    r = np.maximum(p / g, g / p)
    return {"absrel": float(np.mean(np.abs(p - g) / g)), "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
            "d1": float(np.mean(r < 1.25)), "n": int(m.sum())}


def oracle_affine_disparity(disp: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Least-squares a, b with 1/gt = a q + b over valid pixels: the best a relative model could ever do."""
    m = valid & (gt > 0) & np.isfinite(disp)
    A = np.stack([disp[m], np.ones(m.sum())], 1).astype(np.float64)
    a, b = np.linalg.lstsq(A, 1.0 / gt[m].astype(np.float64), rcond=None)[0]
    w = a * disp + b
    return np.where(w > 1 / 80.0, 1 / np.maximum(w, 1e-9), 80.0).astype(np.float32)


# ------------------------------------------------------------------ box baselines
def scrollsafe_band(boxes: list[dict], width: int, height: int) -> str:
    """The ScrollSafe rule (github.com/MrAnitude/ScrollSafe, described in its README; reimplemented here from that
    description, no code copied: the repository has no licence). Only boxes whose centre lies in the middle 50 %
    of the frame count; the tallest one's height / frame height above 0.40 is DANGER (our P1), above 0.20 WARNING
    (P2), else SAFE but present (P3). No box in the corridor: clear."""
    best = 0.0
    for b in boxes:
        cx = (b["x1"] + b["x2"]) / 2
        if 0.25 * width <= cx <= 0.75 * width:
            best = max(best, (b["y2"] - b["y1"]) / height)
    if best == 0.0:
        return "clear"
    return "P1" if best > 0.40 else "P2" if best > 0.20 else "P3"


def flat_ground_box_distance(boxes: list[dict], K: Intrinsics, cam_h: float, pitch: float, roll: float = 0.0,
                             half: float = CORRIDOR_HALF_M) -> float | None:
    """The stand-in's flat-ground rule on boxes: a box's bottom-centre pixel is where it meets the floor, so its
    forward distance follows from the camera height and pitch. Nearest box whose foot is in the corridor."""
    n = down_vector(pitch, roll)
    best = None
    for b in boxes:
        u, v = (b["x1"] + b["x2"]) / 2, b["y2"]
        r = np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
        nr = float(r @ n)
        if nr <= 1e-6:
            continue
        P = r * (cam_h / nr)
        f = np.array([0.0, 0.0, 1.0]) - n[2] * n
        f /= np.linalg.norm(f)
        lat = float(P @ np.cross(n, f))
        fwd = float(P @ f)
        # a wide box can reach into the corridor with its foot centre outside: use its nearest edge
        half_w = (b["x2"] - b["x1"]) / 2 / K.fx * float(P[2])
        if abs(lat) - half_w <= half and fwd > 0.3 and (best is None or fwd < best):
            best = fwd
    return best


def priority_scores(pred: list[str], truth: list[str]) -> dict:
    n = len(truth)
    if not n:
        return {"acc": None, "p1_recall": None, "false_p1": None, "n": 0}
    acc = sum(p == t for p, t in zip(pred, truth)) / n
    p1 = [p for p, t in zip(pred, truth) if t == "P1"]
    not_p1 = [p for p, t in zip(pred, truth) if t != "P1"]
    return {"acc": acc, "p1_recall": (sum(p == "P1" for p in p1) / len(p1)) if p1 else None,
            "false_p1": (sum(p == "P1" for p in not_p1) / len(not_p1)) if not_p1 else None, "n": n,
            "n_p1": len(p1)}
