"""STAND-IN until streetseg lands: walkable-ground segmentation for the drone view.

The `streetseg` package owns perception (spec 5.3). This file exists so the demo video's "Waymo-style"
overlay (walkable ground, people, vehicles) doesn't wait on it. It uses the model family streetseg picked
(Ultralytics YOLO26n semantic segmentation) and keeps a small interface so his model drops in:

    seg = Segmenter()                  # or streetseg's, with the same segment() and process()
    s = seg.segment(frame)             # BGR -> {"masks": {sidewalk, road, crosswalk, person, vehicle}, "walkable": {...}}
    events = seg.process(frame, camera, pose)   # -> earpiece `scene` events (context only, never a stop)

    python sim/seg_standin.py --download                    # once, fetches weights/yolo26n-sem-ade20k.pt
    python sim/seg_standin.py clips/street.mp4 [--jsonl out/seg_events.jsonl] [--seconds 20]   # fps + hints

Weights: the ADE20K checkpoint, not Cityscapes. Cityscapes is all dashcam footage, so the Cityscapes
model calls whatever is under the camera "road": on a stock clip following a walker down a sidewalk
it labelled 91 to 95% of the corridor road and 3% sidewalk. The ADE20K one labelled 87 to 97% sidewalk
on the same frames and still splits road from sidewalk on a Times Square clip.

Masks are bool arrays at the frame's size. ADE20K has no crosswalk class (zebra stripes come out as
`road`), so `crosswalk` is None here; streetseg can fill it. "Walkable ahead" is the fraction of
sidewalk (and crosswalk) pixels in the walker's corridor: a trapezoid up the middle of the drone view,
which flies behind and above the walker looking forward. People standing in it (the walker, mostly)
are left out of the count, since the ground under them can't be seen. It only feeds a `scene` hint such
as "Sidewalk narrows ahead."; deterministic stops (sonar, critical, dont_walk) never depend on it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "yolo26n-sem-ade20k.pt"
IMGSZ = 960                                    # the Tello's width

# ADE20K ids (the model's names) -> our classes. path counts as sidewalk; vehicle is car, bus, truck,
# van, minibike, bicycle.
CLASS_IDS = {"road": (6,), "sidewalk": (11, 52), "person": (12,), "vehicle": (20, 80, 83, 102, 116, 127)}
MIN_BLOB_FRAC = 0.002       # drop specks smaller than this share of the frame (flicker in the video)

# walker's corridor in the drone frame, as fractions of width/height (tune on the shoot footage)
CORRIDOR_BOTTOM_W, CORRIDOR_TOP_W, CORRIDOR_TOP_Y = 0.40, 0.12, 0.50
NARROW_NEAR, NARROW_FAR = 0.55, 0.30           # near half mostly walkable, far half not -> "narrows"
NO_SIDEWALK = 0.15                             # whole corridor below this -> "no sidewalk ahead"
HINT_HOLD = 5                                  # frames a new hint must hold before it is sent (spec 5.3)


# ------------------------------------------------------------------ post-processing (no model)
def masks_from_labels(labels: np.ndarray, shape: tuple[int, int] | None = None) -> dict[str, np.ndarray | None]:
    """ADE20K label map (H, W) -> bool masks per class, resized (nearest) to shape=(h, w), specks removed."""
    if shape and labels.shape[:2] != shape:
        labels = cv2.resize(labels.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    min_px = int(MIN_BLOB_FRAC * labels.size)
    out: dict[str, np.ndarray | None] = {k: drop_small(np.isin(labels, ids), min_px) for k, ids in CLASS_IDS.items()}
    out["crosswalk"] = None                    # ADE20K has no crosswalk class
    return out


def drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected blobs under min_px pixels."""
    if min_px <= 1 or not mask.any():
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]


def corridor(h: int, w: int, bottom_w: float = CORRIDOR_BOTTOM_W, top_w: float = CORRIDOR_TOP_W,
             top_y: float = CORRIDOR_TOP_Y) -> np.ndarray:
    """Points of the corridor trapezoid (int32, 4x2): bottom edge centred, narrowing up to row top_y*h."""
    cx, yt = w / 2, top_y * h
    return np.array([[cx - bottom_w * w / 2, h - 1], [cx - top_w * w / 2, yt],
                     [cx + top_w * w / 2, yt], [cx + bottom_w * w / 2, h - 1]], np.int32)


def walkable_ahead(masks: dict[str, np.ndarray | None], poly: np.ndarray | None = None) -> dict[str, float]:
    """Share of walkable pixels in the corridor, people left out: whole, near (lower) half and far (upper) half."""
    walk = masks["sidewalk"] if masks.get("crosswalk") is None else masks["sidewalk"] | masks["crosswalk"]
    h, w = walk.shape
    poly = corridor(h, w) if poly is None else poly
    region = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(region, poly, 1)
    region = region.astype(bool) & ~masks["person"]
    mid = int((poly[:, 1].min() + poly[:, 1].max()) / 2)

    def frac(sel: np.ndarray) -> float:
        n = int(sel.sum())
        return round(float((walk & sel).sum()) / n, 3) if n else 0.0

    far = region.copy()
    far[mid:] = False
    return {"ahead": frac(region), "near": frac(region & ~far), "far": frac(far)}


def hint_text(walk: dict[str, float]) -> str | None:
    if walk["ahead"] < NO_SIDEWALK:
        return "No sidewalk ahead."
    if walk["near"] >= NARROW_NEAR and walk["far"] < NARROW_FAR:
        return "Sidewalk narrows ahead."
    return None


class HintGate:
    """A hint (or its clearing) must hold HINT_HOLD frames before it changes; returns the new hint on change."""

    def __init__(self, hold: int = HINT_HOLD):
        self.hold, self.current, self.cand, self.n = hold, None, None, 0

    def step(self, hint: str | None) -> tuple[bool, str | None]:
        if hint == self.current:
            self.cand, self.n = None, 0
            return False, self.current
        self.n = self.n + 1 if hint == self.cand else 1
        self.cand = hint
        if self.n < self.hold:
            return False, self.current
        self.current, self.cand, self.n = hint, None, 0
        return True, hint


# ------------------------------------------------------------------ model
class Segmenter:
    """YOLO26n-sem (ADE20K) on mps. Weights must already be in weights/ (never fetched at runtime)."""

    def __init__(self, weights: Path = WEIGHTS, device: str | None = None):
        if not Path(weights).exists():
            sys.exit(f"missing {weights}; run: python sim/seg_standin.py --download")
        import torch
        from ultralytics import YOLO
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = YOLO(str(weights))
        self.gate = HintGate()

    def segment(self, frame: np.ndarray) -> dict:
        r = self.model.predict(frame, imgsz=IMGSZ, device=self.device, verbose=False)[0]
        masks = masks_from_labels(r.semantic_mask.data.cpu().numpy(), frame.shape[:2])
        return {"masks": masks, "walkable": walkable_ahead(masks)}

    def process(self, frame: np.ndarray, camera: dict | None = None, pose: dict | None = None,
                seg: dict | None = None) -> list[dict]:
        """Spec 5.3 shape: frame -> events. Only `scene` (context, not spoken), when the hint changes."""
        seg = seg or self.segment(frame)
        changed, hint = self.gate.step(hint_text(seg["walkable"]))
        if not changed:
            return []
        w = seg["walkable"]
        text = hint or "Sidewalk ahead looks clear."
        return [{"type": "scene", "source": "seg-standin", "text": f"{text} (walkable ahead {w['ahead']:.0%})"}]


def download() -> None:
    from ultralytics.utils.downloads import attempt_download_asset
    WEIGHTS.parent.mkdir(exist_ok=True)
    print(attempt_download_asset(str(WEIGHTS)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", nargs="?", help="video file to segment (prints fps and hints)")
    p.add_argument("--download", action="store_true", help="fetch weights/yolo26n-sem-ade20k.pt and exit")
    p.add_argument("--jsonl", help="write the scene events with t (seconds of video)")
    p.add_argument("--seconds", type=float, help="stop after this much video")
    p.add_argument("--device", help="mps / cpu (default: mps if available)")
    a = p.parse_args(argv)
    if a.download:
        download()
        return 0
    if not a.src:
        p.error("give a video file, or --download")
    cap = cv2.VideoCapture(a.src)
    if not cap.isOpened():
        sys.exit(f"{a.src}: cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seg = Segmenter(device=a.device)
    out = open(a.jsonl, "w") if a.jsonl else None
    i, times = 0, []
    while not a.seconds or i / fps < a.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != 960:
            frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
        ts = time.perf_counter()
        s = seg.segment(frame)
        times.append(time.perf_counter() - ts)
        for ev in seg.process(frame, seg=s):
            print(f"[seg] t={i / fps:.1f} {ev['text']}", flush=True)
            if out:
                out.write(json.dumps({"t": round(i / fps, 3), **ev}, separators=(",", ":")) + "\n")
        i += 1
    if out:
        out.close()
    if len(times) > 1:
        ms = float(np.median(times[1:])) * 1000
        print(f"[seg] {len(times)} frames at 960 wide on {seg.device}: {ms:.1f} ms median ({1000 / ms:.1f} fps), "
              f"first call {times[0] * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
