"""Class-map resize that keeps the existing nearest labels.

The live mask is built with ``linspace`` indexes, then fancy-indexed. OpenCV's
``resize(..., INTER_NEAREST)`` uses a different map, so it changes labels.
``remap`` with those same indexes is the fast equivalent.
"""

from __future__ import annotations

import threading

import numpy as np

_LOCK = threading.Lock()
_MAPS: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def nearest_labels(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize ``mask`` to ``shape`` (height, width). Same pixels as fancy indexing."""
    height, width = shape
    src_h, src_w = int(mask.shape[0]), int(mask.shape[1])
    if src_h == height and src_w == width:
        return np.array(mask, copy=True, order="C")
    if src_h < 1 or src_w < 1 or height < 1 or width < 1:
        raise ValueError("cannot resize an empty class map")
    import cv2

    src = np.ascontiguousarray(mask)
    map_x, map_y = _maps(src_h, src_w, height, width)
    try:
        return cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_NEAREST)
    except cv2.error:
        return _index(src, height, width)


def _maps(src_h: int, src_w: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    key = (src_h, src_w, height, width)
    cached = _MAPS.get(key)
    if cached is not None:
        return cached
    with _LOCK:
        cached = _MAPS.get(key)
        if cached is not None:
            return cached
        if len(_MAPS) >= 8:
            _MAPS.clear()
        # Float64 linspace, then truncate, matches the previous astype(int32).
        # Float32 coordinates of those integers are exact at frame sizes.
        ys = np.linspace(0, src_h - 1, height).astype(np.int32)
        xs = np.linspace(0, src_w - 1, width).astype(np.int32)
        map_x = np.empty((height, width), np.float32)
        map_y = np.empty((height, width), np.float32)
        map_x[:] = xs
        map_y[:] = ys[:, None]
        cached = (map_x, map_y)
        _MAPS[key] = cached
        return cached


def _index(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = np.linspace(0, mask.shape[0] - 1, height).astype(np.int32)
    xs = np.linspace(0, mask.shape[1] - 1, width).astype(np.int32)
    return mask[ys][:, xs]
