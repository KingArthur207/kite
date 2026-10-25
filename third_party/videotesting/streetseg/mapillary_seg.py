"""SegFormer-B0 Mapillary, 160k iterations, without mmsegmentation.

The Flying Guide Dog checkpoint is an mmsegmentation state dict. This loads
it into the Hugging Face SegFormer and runs it on the frame. Weights are
read from disk.
"""

from __future__ import annotations

import re

import numpy as np

from .mapillary import NAMES
from .sampling import nearest_labels

_LONG_SIDE = 768
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _on_cpu(device: str | None) -> bool:
    if not device:
        return True
    return str(device).split(":", 1)[0].lower() == "cpu"


class MapillarySegFormer:
    """Street-view semantic map. ``names`` are the 66 Mapillary classes."""

    def __init__(
        self,
        weights: str = "segmentation/SegFormer_b0_Mapillary.pth",
        long_side: int = _LONG_SIDE,
        device: str | None = None,
    ):
        self.weights = weights
        self.long_side = long_side
        self.device = device
        self._model = None
        self.names = dict(NAMES)

    def load(self):
        return self._load()

    def _load(self):
        if self._model is None:
            import torch
            from transformers import SegformerConfig, SegformerForSemanticSegmentation

            from .models import resolve_weights

            path = resolve_weights(self.weights)
            blob = torch.load(path, map_location="cpu", weights_only=False)
            state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
            config = SegformerConfig(num_labels=len(NAMES))
            config.id2label = {int(idx): name for idx, name in NAMES.items()}
            config.label2id = {name: int(idx) for idx, name in NAMES.items()}
            model = SegformerForSemanticSegmentation(config)
            converted = convert_mmseg_state(state, model.state_dict())
            model.load_state_dict(converted, strict=True)
            model.eval()
            if self.device:
                model.to(self.device)
            # Channels-last matches argmax on real frames and is a bit faster on CPU.
            # GPU and MPS stay contiguous; that path was not retimed here.
            if _on_cpu(self.device):
                model = model.to(memory_format=torch.channels_last)
            self._model = model
        return self._model

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        import torch

        model = self._load()
        height, width = bgr.shape[:2]
        resized, _scale = _resize_multiple(bgr, self.long_side)
        pixel = _imagenet(resized)
        on_cpu = _on_cpu(self.device) and pixel.device.type == "cpu"
        if on_cpu:
            pixel = pixel.contiguous(memory_format=torch.channels_last)
        elif self.device:
            pixel = pixel.to(self.device)
        with torch.inference_mode():
            logits = model(pixel_values=pixel).logits
            if on_cpu and not logits.is_contiguous():
                logits = logits.contiguous()
            upsampled = torch.nn.functional.interpolate(
                logits, size=resized.shape[:2], mode="bilinear", align_corners=False
            )
            if on_cpu:
                labels = _argmax_cpu(upsampled)
            else:
                labels = upsampled.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)
        if labels.shape[:2] != (height, width):
            labels = nearest_labels(labels, (height, width))
        return labels


def convert_mmseg_state(src: dict, hf: dict) -> dict:
    """Rename an mmseg SegFormer-B0 checkpoint onto the Hugging Face module."""
    out: dict = {}
    used: set[str] = set()

    def put(dest: str, tensor) -> None:
        if dest not in hf:
            raise KeyError(dest)
        if tuple(hf[dest].shape) != tuple(tensor.shape):
            raise ValueError(f"{dest}: checkpoint {tuple(tensor.shape)} != model {tuple(hf[dest].shape)}")
        out[dest] = tensor

    for key, tensor in src.items():
        if key.endswith("num_batches_tracked"):
            dest = _rename(key)
            if dest in hf:
                put(dest, tensor)
            used.add(key)
            continue
        split = _kv(key)
        if split is not None:
            k_key, v_key = split
            dim = tensor.shape[0] // 2
            put(k_key, tensor[:dim])
            put(v_key, tensor[dim:])
            used.add(key)
            continue
        dest = _rename(key)
        if dest is None:
            continue
        put(dest, tensor)
        used.add(key)

    missing = [key for key in hf if key not in out and not key.endswith("num_batches_tracked")]
    if missing:
        raise RuntimeError("SegFormer checkpoint did not fill " + ", ".join(missing[:8]))
    for key, tensor in hf.items():
        if key not in out:
            out[key] = tensor
    unused = [
        key for key in src
        if key not in used and not key.startswith("decode_head.conv_seg.")
    ]
    if unused:
        raise RuntimeError("SegFormer checkpoint has unused keys " + ", ".join(unused[:8]))
    return out


def _kv(key: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"backbone\.block(\d+)\.(\d+)\.attn\.kv\.(weight|bias)", key)
    if not match:
        return None
    stage = int(match.group(1)) - 1
    block = match.group(2)
    kind = match.group(3)
    base = f"segformer.stages.{stage}.blocks.{block}.attention"
    return f"{base}.k_proj.{kind}", f"{base}.v_proj.{kind}"


def _rename(key: str) -> str | None:
    patterns = (
        (r"backbone\.patch_embed(\d+)\.proj\.(weight|bias)",
         lambda m: f"segformer.stages.{int(m.group(1)) - 1}.patch_embeddings.proj.{m.group(2)}"),
        (r"backbone\.patch_embed(\d+)\.norm\.(weight|bias)",
         lambda m: f"segformer.stages.{int(m.group(1)) - 1}.patch_embeddings.layer_norm.{m.group(2)}"),
        (r"backbone\.norm(\d+)\.(weight|bias)",
         lambda m: f"segformer.stages.{int(m.group(1)) - 1}.layer_norm.{m.group(2)}"),
        (r"backbone\.block(\d+)\.(\d+)\.norm1\.(weight|bias)",
         lambda m: f"segformer.stages.{int(m.group(1)) - 1}.blocks.{m.group(2)}.layernorm_before.{m.group(3)}"),
        (r"backbone\.block(\d+)\.(\d+)\.norm2\.(weight|bias)",
         lambda m: f"segformer.stages.{int(m.group(1)) - 1}.blocks.{m.group(2)}.layernorm_after.{m.group(3)}"),
        (r"backbone\.block(\d+)\.(\d+)\.attn\.q\.(weight|bias)",
         lambda m: f"segformer.stages.{int(m.group(1)) - 1}.blocks.{m.group(2)}.attention.q_proj.{m.group(3)}"),
        (r"backbone\.block(\d+)\.(\d+)\.attn\.proj\.(weight|bias)",
         lambda m: f"segformer.stages.{int(m.group(1)) - 1}.blocks.{m.group(2)}.attention.o_proj.{m.group(3)}"),
        (r"backbone\.block(\d+)\.(\d+)\.attn\.sr\.(weight|bias)",
         lambda m: (
             f"segformer.stages.{int(m.group(1)) - 1}.blocks.{m.group(2)}"
             f".attention.sequence_reduction.sequence_reduction.{m.group(3)}"
         )),
        (r"backbone\.block(\d+)\.(\d+)\.attn\.norm\.(weight|bias)",
         lambda m: (
             f"segformer.stages.{int(m.group(1)) - 1}.blocks.{m.group(2)}"
             f".attention.sequence_reduction.layer_norm.{m.group(3)}"
         )),
        (r"backbone\.block(\d+)\.(\d+)\.mlp\.fc1\.(weight|bias)",
         lambda m: f"segformer.stages.{int(m.group(1)) - 1}.blocks.{m.group(2)}.mlp.fc1.{m.group(3)}"),
        (r"backbone\.block(\d+)\.(\d+)\.mlp\.fc2\.(weight|bias)",
         lambda m: f"segformer.stages.{int(m.group(1)) - 1}.blocks.{m.group(2)}.mlp.fc2.{m.group(3)}"),
        (r"backbone\.block(\d+)\.(\d+)\.mlp\.dwconv\.dwconv\.(weight|bias)",
         lambda m: f"segformer.stages.{int(m.group(1)) - 1}.blocks.{m.group(2)}.mlp.dwconv.dwconv.{m.group(3)}"),
        (r"decode_head\.linear_c(\d+)\.proj\.(weight|bias)",
         lambda m: f"decode_head.linear_projections.{int(m.group(1)) - 1}.proj.{m.group(2)}"),
        (r"decode_head\.linear_fuse\.conv\.(weight|bias)",
         lambda m: f"decode_head.linear_fuse.{m.group(1)}"),
        (r"decode_head\.linear_fuse\.bn\.(.+)",
         lambda m: f"decode_head.batch_norm.{m.group(1)}"),
        (r"decode_head\.linear_pred\.(weight|bias)",
         lambda m: f"decode_head.classifier.{m.group(1)}"),
    )
    for pattern, build in patterns:
        match = re.fullmatch(pattern, key)
        if match:
            return build(match)
    return None


def _imagenet(bgr: np.ndarray):
    """RGB ImageNet norm as a contiguous NCHW batch. Same arithmetic as before."""
    import torch

    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    norm = (rgb - _MEAN) / _STD
    chw = np.ascontiguousarray(norm.transpose(2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def _argmax_cpu(upsampled) -> np.ndarray:
    """First-max class ids.

    Argmax over channels-last is faster than over NCHW here, and it matches
    torch's NCHW argmax. A row-wise thread pool also matched, then left extra
    OpenMP workers that slowed the next forward, so this stays one torch op.
    """
    labels = upsampled[0].permute(1, 2, 0).contiguous().argmax(dim=-1)
    return labels.numpy().astype(np.int32)


def _resize_multiple(bgr: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    import cv2

    height, width = bgr.shape[:2]
    scale = float(long_side) / float(max(height, width))
    resized_h = max(32, int(round(height * scale / 32.0)) * 32)
    resized_w = max(32, int(round(width * scale / 32.0)) * 32)
    if resized_h == height and resized_w == width:
        return bgr, 1.0
    return cv2.resize(bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR), scale
