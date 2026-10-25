"""Street classes shared by the segmenter and the path scorer.

YOLO26n-sem and SegFormer-B0-Cityscapes both emit these 19 ids, in this order.
"""

from __future__ import annotations

CITYSCAPES: tuple[str, ...] = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

NAMES: dict[int, str] = {i: name for i, name in enumerate(CITYSCAPES)}

WALKABLE = frozenset({"road", "sidewalk", "terrain"})
PEOPLE = frozenset({"person", "rider"})
VEHICLES = frozenset({"car", "truck", "bus", "train", "motorcycle", "bicycle"})
BLOCKING = frozenset({"building", "wall", "fence", "pole", "vegetation"}) | VEHICLES

# BGR. Tuned so a blended frame still reads as a street, Waymo-demo style.
PALETTE: dict[str, tuple[int, int, int]] = {
    "road": (90, 90, 90),
    "sidewalk": (190, 150, 110),
    "building": (70, 70, 130),
    "wall": (60, 60, 100),
    "fence": (90, 130, 170),
    "pole": (190, 190, 190),
    "traffic light": (0, 220, 255),
    "traffic sign": (0, 180, 255),
    "vegetation": (50, 150, 50),
    "terrain": (50, 100, 150),
    "sky": (230, 170, 90),
    "person": (200, 70, 230),
    "rider": (220, 100, 240),
    "car": (50, 50, 220),
    "truck": (30, 30, 180),
    "bus": (20, 90, 210),
    "train": (20, 110, 170),
    "motorcycle": (180, 90, 40),
    "bicycle": (210, 150, 40),
}


def norm_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


LIGHT_LABELS = frozenset({"trafficlight", "trafficlightsignal"})
STOP_SIGN_LABELS = frozenset({"stopsign", "stop"})
CROSSWALK_LABELS = frozenset({
    "crosswalk",
    "crosswalksignal",
    "pedestriancrossing",
    "pedestriancrossingsign",
    "crossingsign",
})

# Body-camera classes worth saying out loud when they loom.
SUDDEN_LABELS = frozenset({
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "chair",
    "bench",
    "backpack",
    "suitcase",
    "pottedplant",
    "stopsign",
    "firehydrant",
    "parkingmeter",
})


def is_light(label: str) -> bool:
    return norm_label(label) in LIGHT_LABELS


def is_stop_sign(label: str) -> bool:
    return norm_label(label) in STOP_SIGN_LABELS


def is_crosswalk(label: str) -> bool:
    return norm_label(label) in CROSSWALK_LABELS


def is_person(label: str) -> bool:
    return norm_label(label) in {"person", "rider", "pedestrian"}


def is_vehicle(label: str) -> bool:
    return norm_label(label) in {norm_label(name) for name in VEHICLES}
