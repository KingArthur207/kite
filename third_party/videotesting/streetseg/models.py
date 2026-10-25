"""Model backends. Weights are loaded from disk and never downloaded here.

Drone or sim frame:
  YOLO26n-sem  — Cityscapes semantic map.
  YOLO26n      — people, vehicles, traffic lights, stop signs.
  optional crossing-sign checkpoint (``--signs``).

Chest camera:
  the same detector, plus Depth Anything V2 Small for close obstacles.
  On that checkpoint larger values mean closer. If a hand over the lens
  looks far, set ``FuseConfig.depth_larger_is_closer = False``.

Fetch them first with ``python -m streetseg.download``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .labels import norm_label
from .sampling import nearest_labels
from .types import Box

_KEEP = frozenset({
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "trafficlight",
    "stopsign",
    "parkingmeter",
    "dog",
    "chair",
    "bench",
    "backpack",
    "suitcase",
})

_ROOT = Path(__file__).resolve().parent.parent / "weights"

_DIRS = {
    "depth-anything/Depth-Anything-V2-Small-hf": "depth-anything-v2-small",
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024": "segformer-b0-cityscapes",
}


def keep_detection(label: str) -> bool:
    name = norm_label(label)
    if name in _KEEP:
        return True
    return any(token in name for token in ("cross", "pedestrian", "yield", "stop", "traffic"))


def _missing(path: str | Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"Missing weights '{path}'. Run `python -m streetseg.download` before perception. "
        "Weights are not downloaded at runtime."
    )


def resolve_weights(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    # A relative path is looked up under weights/. An absolute missing path is an error.
    if not candidate.is_absolute():
        bundled = _ROOT / candidate
        if bundled.exists():
            return bundled.resolve()
    raise _missing(path)


def resolve_model_dir(model_id: str) -> Path:
    candidate = Path(model_id)
    if candidate.is_dir() and (candidate / "config.json").is_file():
        return candidate.resolve()
    name = _DIRS.get(model_id, candidate.name)
    bundled = _ROOT / name
    if bundled.is_dir() and (bundled / "config.json").is_file():
        return bundled.resolve()
    raise _missing(model_id)


def _nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return nearest_labels(mask, shape)


class YoloSemanticSegmenter:
    """Cityscapes semantic map from Ultralytics YOLO26 ``*-sem`` weights."""

    def __init__(self, weights: str = "yolo26n-sem.pt", imgsz: int = 640, device: str | None = None):
        self.weights = weights
        self.imgsz = imgsz
        self.device = device
        self._model = None

    def load(self):
        return self._load()

    def _load(self):
        if self._model is None:
            path = resolve_weights(self.weights)
            from ultralytics import YOLO

            self._model = YOLO(str(path))
        return self._model

    @property
    def names(self) -> dict[int, str]:
        raw = self._load().names
        return {int(key): str(value) for key, value in dict(raw).items()}

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        kwargs = {"imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        result = self._load().predict(bgr, **kwargs)[0]
        mask = getattr(result, "semantic_mask", None)
        data = None if mask is None else getattr(mask, "data", None)
        if data is None:
            raise RuntimeError(
                "Ultralytics did not return a semantic mask. The weights need to be "
                "a YOLO26 semantic checkpoint such as yolo26n-sem.pt."
            )
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        arr = np.squeeze(np.asarray(data))
        if arr.shape[:2] != bgr.shape[:2]:
            arr = _nearest(arr, bgr.shape[:2])
        return arr.astype(np.int32)


class YoloDetector:
    def __init__(
        self,
        weights: str = "yolo26n.pt",
        imgsz: int = 640,
        device: str | None = None,
        conf: float = 0.35,
        keep_all: bool = False,
        track: bool = False,
    ):
        self.weights = weights
        self.imgsz = imgsz
        self.device = device
        self.conf = conf
        self.keep_all = keep_all
        self.track = track
        self._model = None

    def load(self):
        return self._load()

    def _load(self):
        if self._model is None:
            path = resolve_weights(self.weights)
            from ultralytics import YOLO

            self._model = YOLO(str(path))
        return self._model

    def predict(self, bgr: np.ndarray) -> list[Box]:
        model = self._load()
        kwargs = {"imgsz": self.imgsz, "verbose": False, "conf": self.conf}
        if self.device:
            kwargs["device"] = self.device
        if self.track:
            kwargs["persist"] = True
            result = model.track(bgr, **kwargs)[0]
        else:
            result = model.predict(bgr, **kwargs)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        names = {int(key): str(value) for key, value in dict(model.names).items()}
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        ids = None if boxes.id is None else boxes.id.detach().cpu().numpy().astype(int)
        found: list[Box] = []
        for index, (xy, conf, cls_id) in enumerate(zip(xyxy, confs, classes)):
            label = names.get(int(cls_id), str(int(cls_id)))
            if not self.keep_all and not keep_detection(label):
                continue
            track_id = None if ids is None else int(ids[index])
            found.append(Box(
                label,
                float(conf),
                (float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])),
                track_id,
            ))
        return found


class SegFormerSegmenter:
    """Same Cityscapes ids, from a local Hugging Face snapshot."""

    def __init__(
        self,
        model_id: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    def load(self):
        return self._load()

    def _load(self):
        if self._model is None:
            path = resolve_model_dir(self.model_id)
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

            self._processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
            self._model = SegformerForSemanticSegmentation.from_pretrained(path, local_files_only=True)
            self._model.eval()
            if self.device:
                self._model.to(self.device)
        return self._model

    @property
    def names(self) -> dict[int, str]:
        config = self._load().config
        return {int(key): str(value) for key, value in config.id2label.items()}

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        model = self._load()
        image = Image.fromarray(bgr[:, :, ::-1])
        inputs = self._processor(images=image, return_tensors="pt")
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)


class DepthAnythingV2:
    """Relative depth for the chest camera. Larger values are treated as closer."""

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    def load(self):
        return self._load()

    def _load(self):
        if self._model is None:
            path = resolve_model_dir(self.model_id)
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self._processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
            self._model = AutoModelForDepthEstimation.from_pretrained(path, local_files_only=True)
            self._model.eval()
            if self.device:
                self._model.to(self.device)
        return self._model

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        model = self._load()
        image = Image.fromarray(bgr[:, :, ::-1])
        inputs = self._processor(images=image, return_tensors="pt")
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            predicted = model(**inputs).predicted_depth
        upsampled = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        return upsampled.squeeze().detach().cpu().numpy().astype(np.float32)
