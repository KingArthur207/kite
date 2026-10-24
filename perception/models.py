"""Model wrappers: the hybrid detector (COCO YOLO11n + open-vocabulary YOLO-World) and metric depth.

Why two detectors (perception/README.md has the numbers): the open-vocabulary models name poles,
bollards, trash cans and scooters that COCO has no class for, but on people they were weaker than
the COCO model the stand-in already uses (YOLOE-26s scored a clearly visible pedestrian 0.18 where
YOLO11n scored 0.93). So the COCO model keeps people, bikes, vehicles and dogs (the kinds that can
trigger "Stop.") exactly as before, and the open-vocabulary model only adds fixed street objects.

Nothing here downloads at runtime. `python -m perception --download` fetches every file into
weights/ (git-ignored) once, with internet. YOLO-World needs the CLIP text encoder only to embed the
prompts; the embedded model is saved per (model, prompt list) as weights/world-<model>-<hash>.pt, so
the walk itself never loads CLIP, and a changed vocabulary is embedded once on first use.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

import numpy as np

from standin_detect import KIND as COCO_KIND

from .fuse import Det, nms_agnostic
from .vocab import flatten

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
HF_CACHE = str(WEIGHTS / "hf")              # Hugging Face downloads stay in weights/ too
COCO_MODEL = "yolo11n.pt"
OPEN_MODEL = "yolov8s-worldv2.pt"
DEPTH_MODELS = {"outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
                "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"}
DEPTH_SIZE = (448, 252)                     # w, h; multiples of 14, 16:9 like the cameras
TRACKER = str(Path(__file__).with_name("bytetrack_open.yaml"))
COCO_OWNED = {"person", "bike", "vehicle", "dog"}   # kinds only the COCO model reports
# Per-kind minimum score for the open-vocabulary model (others use --conf). Its scores for rare street
# objects run low: "parked scooter" scored 0.16-0.42 on the scooter clip until the scooter was close.
OPEN_MIN_CONF = {"e_scooter": 0.15, "pole": 0.25, "construction": 0.2, "overhead": 0.3, "curb": 0.3,
                 "stairs": 0.3}


@contextlib.contextmanager
def _cwd(path: Path):
    """Ultralytics fetches text encoders into the current directory; keep them in weights/."""
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _weights(name: str) -> Path:
    p = Path(name) if Path(name).is_absolute() else WEIGHTS / name
    if not p.exists():
        raise FileNotFoundError(f"missing {p}; run: python -m perception --download")
    return p


def embedded_path(model: str, prompts: list[str]) -> Path:
    h = hashlib.sha1("\n".join(prompts).encode()).hexdigest()[:10]
    return WEIGHTS / f"world-{Path(model).stem}-{h}.pt"


def _boxes(r, prefix: str):
    b = r.boxes
    if b is None or len(b) == 0:
        return []
    ids = b.id.int().tolist() if b.id is not None else [None] * len(b)
    return [(xyxy, r.names[int(c)], float(cf), None if tid is None else f"{prefix}{tid}")
            for xyxy, c, cf, tid in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist(), ids)]


class CocoDetector:
    """The stand-in's model and class map (sim/standin_detect.py KIND), returning Det."""

    def __init__(self, device: str, conf: float = 0.35, imgsz: int = 640, track: bool = True,
                 weights: str = COCO_MODEL):
        from ultralytics import YOLO
        self.model = YOLO(str(_weights(weights)))
        self.device, self.conf, self.imgsz, self.track = device, conf, imgsz, track

    def __call__(self, img: np.ndarray) -> list[Det]:
        kw = dict(verbose=False, conf=self.conf, imgsz=self.imgsz, device=self.device)
        r = (self.model.track(img, persist=True, **kw) if self.track else self.model.predict(img, **kw))[0]
        return [Det(*xyxy, kind=COCO_KIND[name], conf=cf, label=name, tid=tid)
                for xyxy, name, cf, tid in _boxes(r, "c") if name in COCO_KIND]


class OpenVocabDetector:
    """YOLO-World v2 (or YOLOE, by file name) over the vocabulary's prompts -> Det with the prompt's kind."""

    def __init__(self, vocab: dict[str, list[str]], device: str, conf: float = 0.2, imgsz: int = 640,
                 track: bool = True, weights: str = OPEN_MODEL):
        self.prompts, self.kind_of = flatten(vocab)
        self.yoloe = "yoloe" in Path(weights).name
        self.model = self._load(weights)
        self.device, self.conf, self.imgsz, self.track = device, conf, imgsz, track

    def _load(self, weights: str):
        from ultralytics import YOLO, YOLOE
        src = _weights(weights)
        if self.yoloe:
            m = YOLOE(str(src))
            cache = src.with_name(f"prompts-{src.stem}-{embedded_path(src.name, self.prompts).stem[-10:]}.npz")
            if cache.exists():
                m.load_prompt_embeddings(cache)
            else:
                with _cwd(WEIGHTS):
                    m.set_classes(self.prompts)
                m.save_prompt_embeddings(cache)
            return m
        baked = embedded_path(src.name, self.prompts)
        if baked.exists():
            m = YOLO(str(baked))
            if list(m.names.values()) == self.prompts:
                return m
        m = YOLO(str(src))
        with _cwd(WEIGHTS):
            m.set_classes(self.prompts)          # CLIP ViT-B/32, fetched into weights/ the first time
        m.model.clip_model = None                # keep the saved file at the detector's own size
        m.save(str(baked))
        return YOLO(str(baked))

    def __call__(self, img: np.ndarray) -> list[Det]:
        floor = min([self.conf, *OPEN_MIN_CONF.values()])     # per-kind thresholds are applied below
        kw = dict(verbose=False, conf=floor, imgsz=self.imgsz, device=self.device)
        r = (self.model.track(img, persist=True, tracker=TRACKER, **kw) if self.track
             else self.model.predict(img, **kw))[0]
        masks = r.masks.data.cpu().numpy() > 0.5 if getattr(r, "masks", None) is not None else None
        h, w = img.shape[:2]
        out = []
        for j, (xyxy, label, cf, tid) in enumerate(_boxes(r, "o")):
            kind = self.kind_of.get(label, label)
            if cf < OPEN_MIN_CONF.get(kind, self.conf):
                continue
            m = crop_letterbox(masks[j], h, w) if masks is not None else None
            out.append(Det(*xyxy, kind=kind, conf=cf, label=label, tid=tid, mask=m))
        return out


class HybridDetector:
    """COCO for people, bikes, vehicles and dogs; open vocabulary for everything else; one agnostic NMS.
    `open_every` > 1 runs the open-vocabulary model on every n-th frame only and reuses its last boxes
    in between (fixed objects barely move between frames; people and cars are still tracked every frame)."""

    def __init__(self, coco, open_vocab, open_every: int = 1):
        self.coco, self.open, self.every = coco, open_vocab, max(1, open_every)
        self.n = 0
        self.last_open: list[Det] = []

    def __call__(self, img: np.ndarray) -> list[Det]:
        dets = list(self.coco(img)) if self.coco is not None else []
        if self.open is not None:
            if self.n % self.every == 0:
                self.last_open = [d for d in self.open(img) if d.kind not in COCO_OWNED or self.coco is None]
            dets += self.last_open
        self.n += 1
        return nms_agnostic(dets)

    def warm(self, shape=(540, 960, 3)) -> None:
        z = np.zeros(shape, np.uint8)
        for m in (self.coco, self.open):
            if m is not None:
                m.model.predict(z, verbose=False, imgsz=m.imgsz, device=m.device)


def crop_letterbox(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """A mask at the network's padded input size -> the part that covers the frame (still model-sized)."""
    mh, mw = mask.shape
    g = min(mh / h, mw / w)
    py, px = int(round((mh - h * g) / 2)), int(round((mw - w * g) / 2))
    return mask[py:mh - py or None, px:mw - px or None]


class MetricDepth:
    """Depth Anything V2 Small, metric (outdoor: fine-tuned on Virtual KITTI 2; indoor: Hypersim). Returns
    z-depth in metres at DEPTH_SIZE. Weights come from --download; never fetched at runtime."""

    def __init__(self, which: str = "outdoor", device: str = "cpu"):
        import torch
        from transformers import AutoModelForDepthEstimation
        name = DEPTH_MODELS.get(which, which)
        self.torch, self.device = torch, device
        self.model = AutoModelForDepthEstimation.from_pretrained(
            name, cache_dir=HF_CACHE, local_files_only=True).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import cv2
        x = cv2.resize(img, DEPTH_SIZE)[:, :, ::-1].copy()
        t = self.torch.from_numpy(x).to(self.device).permute(2, 0, 1)[None].float() / 255
        with self.torch.no_grad():
            d = self.model(pixel_values=(t - self.mean) / self.std).predicted_depth
        d = (d[0] if d.ndim == 3 else d[0, 0]).cpu().numpy()
        if d.shape != (DEPTH_SIZE[1], DEPTH_SIZE[0]):
            d = cv2.resize(d, DEPTH_SIZE, interpolation=cv2.INTER_LINEAR)
        return d


def download(vocab: dict[str, list[str]] | None = None, open_model: str = OPEN_MODEL, depth: bool = True) -> None:
    """Fetch YOLO11n, YOLO-World v2 and CLIP (to embed the prompts once), and the depth models, into weights/."""
    from ultralytics.utils.downloads import attempt_download_asset

    from .vocab import DEFAULT_VOCAB
    with _cwd(WEIGHTS):
        for name in (COCO_MODEL, open_model):
            print(attempt_download_asset(name))
    det = OpenVocabDetector(vocab or DEFAULT_VOCAB, "cpu", weights=open_model)
    print(f"embedded {len(det.prompts)} prompts into weights/")
    if depth:
        from transformers import AutoModelForDepthEstimation
        for name in DEPTH_MODELS.values():
            AutoModelForDepthEstimation.from_pretrained(name, cache_dir=HF_CACHE)
            print(f"cached {name}")
