"""Pedestrian and vehicle traffic lights, cropped from the segmentation mask.

The classifier is the Flying Guide Dog ResNet-18, trained on the PVTL set.
Class index order is theirs: others, pedestrian-green, pedestrian-red,
vehicle-green, vehicle-red. A pedestrian red stops on the frame it appears.
A pedestrian green has to be the majority of the last seven frames. Vehicle
lights are reported and do not move the walker.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .mapillary import is_traffic_light_class
from .types import Box

# Matches models/traffic_light_classification/classify.py.
PVTL_CLASSES: tuple[str, ...] = (
    "others",
    "pedestrian-green",
    "pedestrian-red",
    "vehicle-green",
    "vehicle-red",
)
PVTL_ID2NAME: dict[int, str] = {idx: name for idx, name in enumerate(PVTL_CLASSES)}
_PEDESTRIAN = frozenset({"pedestrian-red", "pedestrian-green"})
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LightReading:
    """What the light head said for this frame, after the seven-frame vote."""

    pedestrian: str | None
    labels: tuple[str, ...]
    boxes: tuple[Box, ...]


class LightVote:
    """Seven-frame vote. Red does not wait. Green waits until the window is full."""

    def __init__(self, frames: int = 7):
        self.frames = max(1, frames)
        self.buffer: list[str | None] = []

    def push(self, pedestrian: str | None) -> str | None:
        self.buffer.append(pedestrian)
        if len(self.buffer) > self.frames:
            self.buffer.pop(0)
        if pedestrian == "pedestrian-red":
            return "pedestrian-red"
        if len(self.buffer) < self.frames:
            return None
        counts = Counter(item for item in self.buffer if item in _PEDESTRIAN)
        if not counts:
            return None
        best = counts.most_common(1)[0][1]
        tied = [name for name, count in counts.items() if count == best]
        if "pedestrian-red" in tied:
            return "pedestrian-red"
        return tied[0]


class TrafficLightClassifier:
    """ResNet-18, five PVTL classes, loaded from disk."""

    def __init__(self, weights: str = "traffic_light_classification/resnet18-best.pth", device: str | None = None):
        self.weights = weights
        self.device = device
        self._model = None

    def load(self):
        return self._load()

    def _load(self):
        if self._model is None:
            import torch
            from torchvision.models import resnet18

            from .models import resolve_weights

            path = resolve_weights(self.weights)
            model = resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(PVTL_CLASSES))
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            if self.device:
                model.to(self.device)
            self._model = model
        return self._model

    def predict(self, crops_bgr: list[np.ndarray]) -> list[str]:
        if not crops_bgr:
            return []
        import torch

        batch = torch.stack([_prep(crop) for crop in crops_bgr])
        if self.device:
            batch = batch.to(self.device)
        with torch.inference_mode():
            logits = self._load()(batch)
        ids = logits.argmax(dim=1).detach().cpu().tolist()
        return [PVTL_ID2NAME[int(idx)] for idx in ids]


class TrafficLightReader:
    """Crop traffic-light pixels, classify them, and vote."""

    def __init__(self, classifier: TrafficLightClassifier | None = None, vote_frames: int = 7):
        self.classifier = classifier or TrafficLightClassifier()
        self.vote = LightVote(vote_frames)

    def load(self):
        self.classifier.load()
        return self

    def read(self, bgr: np.ndarray, labels: np.ndarray, names: dict[int, str]) -> LightReading:
        crops, boxes = crop_traffic_lights(bgr, labels, names)
        labels_out = tuple(self.classifier.predict(crops))
        pedestrian = next((name for name in labels_out if name in _PEDESTRIAN), None)
        decided = self.vote.push(pedestrian)
        painted = tuple(
            Box(name, 1.0, xy, signal=_signal_of(name), source="pvtl")
            for name, xy in zip(labels_out, boxes)
        )
        return LightReading(decided, labels_out, painted)


def crop_traffic_lights(
    bgr: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    *,
    min_side: int = 32,
    limit: int = 4,
) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]]]:
    """Largest traffic-light blobs. The guide-dog runtime keeps a box of at least 32px."""
    import cv2

    ids = [idx for idx, name in names.items() if is_traffic_light_class(name)]
    mask = np.zeros(labels.shape[:2], dtype=np.uint8)
    for idx in ids:
        mask[labels == idx] = 255
    if not ids or int(mask.max()) == 0:
        return [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    crops: list[np.ndarray] = []
    boxes: list[tuple[float, float, float, float]] = []
    height, width = bgr.shape[:2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_side or h < min_side:
            continue
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x or y2 <= y:
            continue
        crops.append(bgr[y:y2, x:x2])
        boxes.append((float(x), float(y), float(x2), float(y2)))
        if len(crops) >= limit:
            break
    return crops, boxes


def _signal_of(name: str) -> str | None:
    if name.endswith("-red"):
        return "red"
    if name.endswith("-green"):
        return "green"
    return None


def _prep(bgr: np.ndarray):
    import torch

    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std = torch.tensor(_STD).view(3, 1, 1)
    return (tensor - mean) / std
