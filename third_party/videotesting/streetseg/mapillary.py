"""Mapillary Vistas classes from SegFormer-B0 trained for 160k iterations.

The order and colors are the ones stored on the Flying Guide Dog checkpoint
(``segformer.b0.768x768.mapillary.160k``). Walkable ground includes the two
crosswalk classes that Cityscapes does not have: plain crossings and zebra
markings.
"""

from __future__ import annotations

import numpy as np

from .labels import norm_label

# Checkpoint meta CLASSES, index is the train id.
MAPILLARY: tuple[str, ...] = (
    "Bird",
    "Ground Animal",
    "Curb",
    "Fence",
    "Guard Rail",
    "Barrier",
    "Wall",
    "Bike Lane",
    "Crosswalk - Plain",
    "Curb Cut",
    "Parking",
    "Pedestrian Area",
    "Rail Track",
    "Road",
    "Service Lane",
    "Sidewalk",
    "Bridge",
    "Building",
    "Tunnel",
    "Person",
    "Bicyclist",
    "Motorcyclist",
    "Other Rider",
    "Lane Marking - Crosswalk",
    "Lane Marking - General",
    "Mountain",
    "Sand",
    "Sky",
    "Snow",
    "Terrain",
    "Vegetation",
    "Water",
    "Banner",
    "Bench",
    "Bike Rack",
    "Billboard",
    "Catch Basin",
    "CCTV Camera",
    "Fire Hydrant",
    "Junction Box",
    "Mailbox",
    "Manhole",
    "Phone Booth",
    "Pothole",
    "Street Light",
    "Pole",
    "Traffic Sign Frame",
    "Utility Pole",
    "Traffic Light",
    "Traffic Sign (Back)",
    "Traffic Sign (Front)",
    "Trash Can",
    "Bicycle",
    "Boat",
    "Bus",
    "Car",
    "Caravan",
    "Motorcycle",
    "On Rails",
    "Other Vehicle",
    "Trailer",
    "Truck",
    "Wheeled Slow",
    "Car Mount",
    "Ego Vehicle",
    "Unlabeled",
)

# RGB, same order as utils/colorize.py in EckoTan0804/flying-guide-dog.
_RGB: tuple[tuple[int, int, int], ...] = (
    (165, 42, 42),
    (0, 192, 0),
    (196, 196, 196),
    (190, 153, 153),
    (180, 165, 180),
    (90, 120, 150),
    (102, 102, 156),
    (128, 64, 255),
    (140, 140, 200),
    (170, 170, 170),
    (250, 170, 160),
    (96, 96, 96),
    (230, 150, 140),
    (128, 64, 128),
    (110, 110, 110),
    (244, 35, 232),
    (150, 100, 100),
    (70, 70, 70),
    (150, 120, 90),
    (220, 20, 60),
    (255, 0, 0),
    (255, 0, 100),
    (255, 0, 200),
    (200, 128, 128),
    (255, 255, 255),
    (64, 170, 64),
    (230, 160, 50),
    (70, 130, 180),
    (190, 255, 255),
    (152, 251, 152),
    (107, 142, 35),
    (0, 170, 30),
    (255, 255, 128),
    (250, 0, 30),
    (100, 140, 180),
    (220, 220, 220),
    (220, 128, 128),
    (222, 40, 40),
    (100, 170, 30),
    (40, 40, 40),
    (33, 33, 33),
    (100, 128, 160),
    (142, 0, 0),
    (70, 100, 150),
    (210, 170, 100),
    (153, 153, 153),
    (128, 128, 128),
    (0, 0, 80),
    (250, 170, 30),
    (192, 192, 192),
    (220, 220, 0),
    (140, 140, 20),
    (119, 11, 32),
    (150, 0, 255),
    (0, 60, 100),
    (0, 0, 142),
    (0, 0, 90),
    (0, 0, 230),
    (0, 80, 100),
    (128, 64, 64),
    (0, 0, 110),
    (0, 0, 70),
    (0, 0, 192),
    (32, 32, 32),
    (120, 10, 10),
    (0, 0, 0),
)

# BGR for OpenCV.
PALETTE_BGR: dict[str, tuple[int, int, int]] = {
    name: (rgb[2], rgb[1], rgb[0]) for name, rgb in zip(MAPILLARY, _RGB)
}

NAMES: dict[int, str] = {i: name for i, name in enumerate(MAPILLARY)}

# Cityscapes walkable names, plus the Mapillary surfaces a person can stand on.
# Crosswalk - Plain and Lane Marking - Crosswalk are the pedestrian crossings.
_WALK = frozenset({
    "road",
    "sidewalk",
    "terrain",
    "crosswalkplain",
    "lanemarkingcrosswalk",
    "pedestrianarea",
    "curbcut",
    "bikelane",
})
_PERSON = frozenset({
    "person",
    "rider",
    "pedestrian",
    "bicyclist",
    "motorcyclist",
    "otherrider",
})
_BLOCK = frozenset({
    "building",
    "wall",
    "fence",
    "pole",
    "vegetation",
    "guardrail",
    "barrier",
    "bridge",
    "tunnel",
    "trafficsignframe",
    "utilitypole",
    "pothole",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "caravan",
    "trailer",
    "othervehicle",
    "onrails",
    "wheeledslow",
    "boat",
})
CROSSWALK_CLASSES = frozenset({
    "crosswalkplain",
    "lanemarkingcrosswalk",
    "crosswalk",
})
TRAFFIC_LIGHT_CLASS = "trafficlight"


def ground_kind(name: str) -> str:
    """``walk``, ``person``, ``block``, or empty when the class is neither."""
    key = norm_label(name)
    if key in _WALK:
        return "walk"
    if key in _PERSON:
        return "person"
    if key in _BLOCK:
        return "block"
    return ""


def is_crosswalk_class(name: str) -> bool:
    return norm_label(name) in CROSSWALK_CLASSES


def is_traffic_light_class(name: str) -> bool:
    return norm_label(name) == TRAFFIC_LIGHT_CLASS


def crosswalk_in_band(labels: np.ndarray, names: dict[int, str], y0: int, minimum: int = 80) -> bool:
    """True when a crossing covers more than a speck of the ground band."""
    ids = [idx for idx, name in names.items() if is_crosswalk_class(name)]
    if not ids:
        return False
    band = labels[max(0, y0):]
    count = 0
    for idx in ids:
        count += int(np.count_nonzero(band == idx))
        if count >= minimum:
            return True
    return False


def paint_bgr(name: str) -> tuple[int, int, int] | None:
    from .labels import PALETTE

    if name in PALETTE:
        return PALETTE[name]
    return PALETTE_BGR.get(name)
