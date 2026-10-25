"""Traffic-light color from the pixels inside a detector box.

The Cityscapes segmenter knows *where* a light is, not whether it is red.
A vertical signal is a few saturated pixels, so this counts them instead of
trusting the average color of the whole box (which is mostly housing and sky).
"""

from __future__ import annotations

import numpy as np

from .types import Box


def classify_light(bgr: np.ndarray, xyxy: tuple[float, float, float, float]) -> str:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return "unknown"
    crop = bgr[y1:y2, x1:x2]
    blue = crop[:, :, 0].astype(np.int16)
    green = crop[:, :, 1].astype(np.int16)
    red = crop[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(red, green), blue)
    mn = np.minimum(np.minimum(red, green), blue)
    bright = (mx > 70) & ((mx - mn) > 35)
    is_red = bright & (red > 100) & (red > green + 25) & (red > blue)
    is_green = bright & (green > 80) & (green > red + 20) & (green > blue)
    is_yellow = bright & (red > 100) & (green > 100) & (blue < 90) & (np.abs(red - green) < 45)
    counts = (int(is_red.sum()), int(is_yellow.sum()), int(is_green.sum()))
    if max(counts) < 6:
        return "unknown"
    return ("red", "yellow", "green")[int(np.argmax(counts))]


def paint_signal(box: Box, bgr: np.ndarray) -> Box:
    from .labels import is_light

    if not is_light(box.label) or box.signal:
        return box
    return Box(
        box.label,
        box.conf,
        box.xyxy,
        box.track_id,
        classify_light(bgr, box.xyxy),
        box.source,
    )
