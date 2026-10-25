"""A forward street view with a known right answer, for the demo and tests.

Cars fill the lane ahead, several people stand on the right sidewalk, and
the left sidewalk is empty. A green pedestrian signal hangs over that left
side. The figure cut off by the bottom edge is the walker the drone follows.
"""

from __future__ import annotations

import numpy as np

from .labels import CITYSCAPES, PALETTE
from .types import Box


def synthetic_street(width: int = 960, height: int = 720):
    names = {i: name for i, name in enumerate(CITYSCAPES)}
    labels = np.full((height, width), CITYSCAPES.index("sky"), dtype=np.int32)
    horizon = int(height * 0.46)
    labels[horizon:, :] = CITYSCAPES.index("building")
    ground = int(height * 0.56)
    labels[ground:, :] = CITYSCAPES.index("road")

    third = width // 3
    labels[ground:, :third] = CITYSCAPES.index("sidewalk")
    labels[ground:, 2 * third:] = CITYSCAPES.index("sidewalk")
    labels[: int(height * 0.18), :] = CITYSCAPES.index("sky")
    labels[int(height * 0.18):horizon, :] = CITYSCAPES.index("building")

    def paint(name: str, x0: int, y0: int, x1: int, y1: int) -> None:
        labels[y0:y1, x0:x1] = CITYSCAPES.index(name)

    paint("car", third + 30, int(height * 0.62), third + 150, int(height * 0.84))
    paint("car", third + 170, int(height * 0.60), width - third - 24, int(height * 0.82))
    paint("vegetation", 0, horizon - 10, 70, ground + 30)

    people = []
    for i, x in enumerate((2 * third + 24, 2 * third + 90, 2 * third + 160, 2 * third + 230)):
        y0 = int(height * 0.60) + (i % 2) * 8
        y1 = int(height * 0.80)
        paint("person", x, y0, x + 36, y1)
        people.append(Box("person", 0.80, (x, y0, x + 36, y1)))

    wearer = Box("person", 0.90, (width * 0.42, height * 0.78, width * 0.58, height - 1))
    x1, y1, x2, y2 = (int(v) for v in wearer.xyxy)
    paint("person", x1, y1, x2, y2)

    light = Box("traffic light", 0.88, (70, 48, 102, 128), signal="green")
    lx1, ly1, lx2, ly2 = light.xyxy
    labels[ly1:ly2, lx1:lx2] = CITYSCAPES.index("traffic light")

    sign = Box("pedestrianCrossing", 0.86, (120, 56, 168, 112))
    image = _render(labels, names)
    image[ly1 + 8:ly2 - 8, lx1 + 8:lx2 - 8] = (40, 230, 40)
    boxes = [
        Box("car", 0.91, (third + 30, int(height * 0.62), third + 150, int(height * 0.84))),
        Box("car", 0.87, (third + 170, int(height * 0.60), width - third - 24, int(height * 0.82))),
        *people,
        wearer,
        light,
        sign,
    ]
    return image, labels, boxes


def _render(labels: np.ndarray, names: dict[int, str]) -> np.ndarray:
    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for idx, name in names.items():
        image[labels == idx] = PALETTE.get(name, (0, 0, 0))
    height = labels.shape[0]
    sky = labels == CITYSCAPES.index("sky")
    ramp = np.linspace(180, 240, height, dtype=np.float32)
    ys, xs = np.nonzero(sky)
    image[ys, xs, 0] = ramp[ys]
    image[ys, xs, 1] = ramp[ys] * 0.75
    image[ys, xs, 2] = ramp[ys] * 0.45
    return image
