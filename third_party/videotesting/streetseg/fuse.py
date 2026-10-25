"""Turn a class map and detector boxes into one walking decision.

The forward camera looks ahead of the walker, so the bottom of the frame is
the ground they are about to step on. That band is split into left, center,
and right. The center is preferred when it is actually walkable, and a quieter
sidewalk wins when the scores are close.

How tall the band is comes from the camera pitch and height when the caller
sets ``FuseConfig.near_frac``. This module does not speak to glasses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import metric_distance
from .labels import (
    SUDDEN_LABELS,
    is_crosswalk,
    is_person,
    is_stop_sign,
    is_vehicle,
    norm_label,
)
from .mapillary import ground_kind
from .types import Box, ColumnScore, Cue

COLUMNS = ("left", "center", "right")
_ORDER = {"center": 0, "left": 1, "right": 2}
_WALK, _BLOCK, _PERSON = 1, 2, 3


@dataclass(frozen=True)
class FuseConfig:
    near_frac: float = 0.55
    near_distance_m: float = 5.0
    close_m: float = 2.0
    center_bias: float = 0.05
    crowd_window: float = 0.08
    min_walkable: float = 0.35
    max_blocked: float = 0.55
    min_score: float = 0.15
    person_count_penalty: float = 0.08
    person_pixel_penalty: float = 0.25
    vehicle_cover: float = 0.15
    exclude_wearer: bool = True
    depth_larger_is_closer: bool = True
    sudden_height: float = 0.38
    sudden_depth_percentile: float = 72.0


@dataclass(frozen=True)
class Obstacle:
    speech: str
    reason: str
    hazards: tuple[str, ...]
    side: str | None
    label: str | None
    box: Box | None
    distance_m: float | None
    lateral_m: float | None


def assess(
    labels: np.ndarray,
    names: dict[int, str],
    boxes: list[Box],
    body_boxes: list[Box] | None = None,
    depth: np.ndarray | None = None,
    body_shape: tuple[int, int] | None = None,
    cfg: FuseConfig | None = None,
    camera=None,
    pedestrian: str | None = None,
    crosswalk_pixels: bool = False,
) -> Cue:
    cfg = cfg or FuseConfig()
    if labels.ndim != 2:
        raise ValueError("labels must be an HxW class map")
    height, width = labels.shape
    columns, wearer = _score_columns(labels, names, boxes, cfg)
    pick, diverted = _pick(columns, cfg)
    crosswalk = any(is_crosswalk(box.label) for box in boxes) or crosswalk_pixels
    if pedestrian == "pedestrian-green":
        crosswalk = True
    reason = "crowd" if diverted else "clear"
    direction = {"left": "left", "right": "right", "center": "straight"}[pick.name]
    walk_direction = direction
    walk_reason = reason

    if not pick.selectable:
        direction = "stop"
        reason = "blocked"
        signal = _signal_for("straight", boxes, width)
        if signal in ("red", "yellow"):
            reason = "signal"
    else:
        signal = _signal_for(direction, boxes, width)
        if signal in ("red", "yellow"):
            direction = "stop"
            reason = "signal"
        elif _stop_sign_for(direction, boxes, width, height, cfg):
            direction = "stop"
            reason = "stop-sign"

    body_h, body_w = body_shape if body_shape is not None else (height, width)
    hazards, sudden_side, sudden_label, _box = _sudden(
        body_boxes or [], depth, body_w, body_h, cfg, camera
    )
    if sudden_label and (sudden_side == "center" or sudden_side == direction):
        direction = "stop"
        reason = "sudden"
    if pedestrian == "pedestrian-red":
        direction = "stop"
        signal = "red"
        if reason not in ("blocked", "sudden", "stop-sign"):
            reason = "signal"
    elif pedestrian == "pedestrian-green" and reason == "signal":
        signal = "green"
        if pick.selectable:
            direction = walk_direction
            reason = walk_reason
        else:
            direction = "stop"
            reason = "blocked"
    elif pedestrian == "pedestrian-green":
        signal = "green"

    crowd = _crowd(pick if direction != "stop" else pick)
    if direction in ("left", "right", "straight"):
        shown = _column_named(columns, "center" if direction == "straight" else direction)
        crowd = _crowd(shown)

    speech = _speech(
        direction, reason, crowd, signal, crosswalk, sudden_label, sudden_side,
        pedestrian=pedestrian,
    )
    shown_signal = signal if signal in ("red", "yellow", "green", "none") else signal
    return Cue(
        direction=direction,
        speech=speech,
        reason=reason,
        crowd=crowd,
        signal=shown_signal,
        crosswalk=crosswalk,
        hazards=tuple(hazards),
        columns=columns,
        wearer_excluded=wearer is not None,
    )


def chest_view(boxes, depth, shape, camera, cfg: FuseConfig | None = None) -> Obstacle | None:
    """Close obstacles in the chest camera. None when the way in front is clear."""
    cfg = cfg or FuseConfig()
    height, width = shape
    hazards, side, label, box = _sudden(boxes, depth, width, height, cfg, camera)
    if not label:
        return None
    from .camera import locate

    located = locate(box, camera, shape) if box is not None else {}
    speech = _speech("stop", "sudden", "clear", "none", False, label, side)
    return Obstacle(
        speech,
        "sudden",
        tuple(hazards),
        side,
        label,
        box,
        located.get("distance_m"),
        located.get("lateral_m"),
    )


def _score_columns(labels, names, boxes, cfg: FuseConfig):
    height, width = labels.shape
    y0 = int(height * cfg.near_frac)
    y0 = min(max(y0, 0), height - 1)
    table = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
    for idx, name in names.items():
        if idx < 0 or idx >= table.size:
            continue
        kind = ground_kind(name)
        if kind == "walk":
            table[idx] = _WALK
        elif kind == "person":
            table[idx] = _PERSON
        elif kind == "block":
            table[idx] = _BLOCK
    cats = table[np.clip(labels, 0, table.size - 1)]
    wearer = _wearer(boxes, width, height) if cfg.exclude_wearer else None
    if wearer is not None:
        x1, y1, x2, y2 = _clip_box(wearer, width, height)
        # The walker occludes the ground. Those pixels are the path under them,
        # not a crowd to steer away from.
        region = cats[y1:y2, x1:x2]
        region[region == _PERSON] = _WALK

    near = cats[y0:]
    edges = (0, width // 3, (2 * width) // 3, width)
    columns: list[ColumnScore] = []
    for name, x0, x1 in zip(COLUMNS, edges[:-1], edges[1:]):
        slab = near[:, x0:x1]
        total = max(1, slab.size)
        walkable = float(np.count_nonzero(slab == _WALK) / total)
        blocked = float(np.count_nonzero(slab == _BLOCK) / total)
        people_pixels = float(np.count_nonzero(slab == _PERSON) / total)
        people = _count(boxes, width, height, y0, x0, x1, is_person, wearer)
        vehicles = _count(boxes, width, height, y0, x0, x1, is_vehicle, None)
        cover = _vehicle_cover(boxes, width, height, y0, x0, x1)
        hard = cover >= cfg.vehicle_cover
        score = walkable * (1.0 - min(1.0, blocked))
        score -= cfg.person_count_penalty * people
        score -= cfg.person_pixel_penalty * people_pixels
        selectable = (
            not hard
            and walkable >= cfg.min_walkable
            and blocked <= cfg.max_blocked
            and score >= cfg.min_score
        )
        adjusted = score
        if name == "center" and selectable:
            adjusted += cfg.center_bias
        columns.append(ColumnScore(
            name, walkable, blocked, people_pixels, people, vehicles,
            score, adjusted, selectable,
        ))
    return tuple(columns), wearer


def _pick(columns: tuple[ColumnScore, ...], cfg: FuseConfig):
    selectable = [col for col in columns if col.selectable]
    if not selectable:
        center = columns[1]
        return center, False
    best = max(col.adjusted for col in selectable)
    pool = [col for col in selectable if col.adjusted >= best - cfg.crowd_window]
    pick = min(pool, key=lambda col: (col.people, -col.adjusted, _ORDER[col.name]))
    winner = max(selectable, key=lambda col: (col.adjusted, -_ORDER[col.name]))
    diverted = pick.people < winner.people
    return pick, diverted


def _wearer(boxes: list[Box], width: int, height: int) -> Box | None:
    """A person cut off by the bottom edge is the walker, not a crowd."""
    found: list[Box] = []
    for box in boxes:
        if not is_person(box.label):
            continue
        cx, _cy = box.center
        _x1, _y1, _x2, y2 = box.xyxy
        if y2 < height - 2:
            continue
        if not (0.15 * width <= cx <= 0.85 * width):
            continue
        found.append(box)
    if not found:
        return None
    return max(found, key=lambda box: box.area)


def _clip_box(box: Box, width: int, height: int):
    x1, y1, x2, y2 = box.xyxy
    x1 = int(np.clip(np.floor(x1), 0, width))
    x2 = int(np.clip(np.ceil(x2), 0, width))
    y1 = int(np.clip(np.floor(y1), 0, height))
    y2 = int(np.clip(np.ceil(y2), 0, height))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _count(boxes, width, height, y0, x0, x1, predicate, skip: Box | None) -> int:
    total = 0
    for box in boxes:
        if box is skip or not predicate(box.label):
            continue
        cx, cy = box.center
        if cy < y0:
            continue
        if x0 <= cx < x1:
            total += 1
    return total


def _vehicle_cover(boxes, width, height, y0, x0, x1) -> float:
    column_area = float((height - y0) * max(1, x1 - x0))
    covered = 0.0
    for box in boxes:
        if not is_vehicle(box.label):
            continue
        bx1, by1, bx2, by2 = box.xyxy
        ix1, iy1 = max(x0, bx1), max(y0, by1)
        ix2, iy2 = min(x1, bx2), min(height, by2)
        if ix2 > ix1 and iy2 > iy1:
            covered += (ix2 - ix1) * (iy2 - iy1)
    return covered / column_area


def _column_of(x: float, width: int) -> str:
    if x < width / 3.0:
        return "left"
    if x < 2.0 * width / 3.0:
        return "center"
    return "right"


def _signal_for(direction: str, boxes: list[Box], width: int) -> str:
    colors = []
    for box in boxes:
        if not box.signal:
            continue
        column = _column_of(box.center[0], width)
        if direction == "left" and column == "right":
            continue
        if direction == "right" and column == "left":
            continue
        if direction == "straight" and column != "center":
            continue
        colors.append(box.signal)
    if "red" in colors:
        return "red"
    if "yellow" in colors:
        return "yellow"
    if "green" in colors:
        return "green"
    return "none"


def _stop_sign_for(direction: str, boxes: list[Box], width: int, height: int, cfg: FuseConfig) -> bool:
    for box in boxes:
        if not is_stop_sign(box.label):
            continue
        _cx, cy = box.center
        if cy < height * cfg.near_frac and box.xyxy[3] < height * 0.70:
            continue
        column = _column_of(box.center[0], width)
        if direction == "straight" and column != "center":
            continue
        if direction == "left" and column == "right":
            continue
        if direction == "right" and column == "left":
            continue
        return True
    return False


def _sudden(body_boxes, depth, width, height, cfg: FuseConfig, camera=None):
    hazards: list[str] = []
    chosen_side = None
    chosen_label = None
    chosen_box = None
    chosen_area = 0.0
    for box in body_boxes:
        label = norm_label(box.label)
        x1, y1, x2, y2 = box.xyxy
        tall = (y2 - y1) / max(1, height) >= cfg.sudden_height
        huge = box.area >= 0.35 * width * height
        if label not in SUDDEN_LABELS and not huge:
            continue
        if y1 >= 0.75 * height:
            continue
        close = tall or huge
        if not close and camera is not None:
            dist = metric_distance(box, camera, height)
            close = dist is not None and dist <= cfg.close_m
        if depth is not None and not close:
            close = _close_depth(depth, box, cfg)
        if not close:
            continue
        side = _side(box.center[0], width)
        text = f"{box.label} close on the {side}" if side != "center" else f"{box.label} close in front"
        hazards.append(text)
        if box.area > chosen_area:
            chosen_area = box.area
            chosen_side = side
            chosen_label = box.label
            chosen_box = box
    return hazards, chosen_side, chosen_label, chosen_box


def _close_depth(depth: np.ndarray, box: Box, cfg: FuseConfig) -> bool:
    if depth.ndim != 2:
        return False
    height, width = depth.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, width, height)
    crop = depth[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    y0 = int(height * 0.45)
    ref = depth[y0:]
    if ref.size == 0:
        return False
    med = float(np.median(crop))
    if cfg.depth_larger_is_closer:
        return med >= float(np.percentile(ref, cfg.sudden_depth_percentile))
    return med <= float(np.percentile(ref, 100.0 - cfg.sudden_depth_percentile))


def _side(x: float, width: int) -> str:
    if x < 0.40 * width:
        return "left"
    if x > 0.60 * width:
        return "right"
    return "center"


def _crowd(column: ColumnScore) -> str:
    if column.people <= 0:
        return "clear"
    if column.people <= 2:
        return "light"
    return "crowded"


def _column_named(columns, name: str) -> ColumnScore:
    for column in columns:
        if column.name == name:
            return column
    return columns[1]


def _speech(
    direction, reason, crowd, signal, crosswalk, sudden_label, sudden_side, pedestrian: str | None = None,
) -> str:
    if reason == "sudden" and sudden_label:
        where = "in front of you" if sudden_side in (None, "center") else f"on your {sudden_side}"
        return f"Stop. A {sudden_label} is close {where}."
    if reason == "signal" and pedestrian == "pedestrian-red":
        return "Stop. The pedestrian light is red."
    if reason == "signal":
        color = signal if signal in ("red", "yellow") else "red"
        return f"Stop. The light looks {color}."
    if reason == "stop-sign":
        return "Stop. There is a stop sign ahead."
    if reason == "blocked" or direction == "stop":
        return "Stop. The path ahead looks blocked."

    way = {"left": "Bear left.", "right": "Bear right.", "straight": "Continue straight."}[direction]
    if reason == "crowd":
        clause = "Fewer people that way."
    elif crowd == "clear":
        clause = "The way looks clear."
    elif crowd == "light":
        clause = "A few people are that way."
    else:
        clause = "That way is crowded. It is still the clearest path."
    sentence = f"{way} {clause}"
    if signal == "green" and crosswalk:
        sentence += " The crossing light looks green."
    elif signal == "green":
        sentence += " A light ahead looks green."
    elif crosswalk:
        sentence += " Crosswalk ahead. I can't see the signal."
    return sentence
