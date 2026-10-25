"""Ground-truth data for the depth shoot-out: a slice of DIODE (real, metric, laser-scanned) and the synthetic
chest-camera walks (synth.py).

DIODE (Vasiljevic et al. 2019, https://diode-dataset.org, MIT licence): validation split, 325 indoor and 446
outdoor 1024x768 frames from a FARO laser scanner, fx = 886.81 px, fy = 927.06 px. We take an evenly spaced slice and
resize to 640x480 (the C270's resolution), depth by nearest neighbour so no depth is invented at edges.
Everything lives under data/depth_bench/ (git-ignored).
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..depth_zoo import ROOT, Intrinsics

DATA = ROOT / "data" / "depth_bench"
DIODE_URL = "https://diode-dataset.s3.amazonaws.com/val.tar.gz"
DIODE_K = Intrinsics(886.81, 927.06, 512.0, 384.0, 1024, 768)   # as published with the dataset
SIZE = (640, 480)
MAX_GT = 80.0


@dataclass
class Still:
    name: str
    split: str                 # "indoors" or "outdoor"
    image: np.ndarray          # BGR uint8 (480, 640, 3)
    depth: np.ndarray          # float32 metres, 0 = no ground truth
    K: Intrinsics


def download_diode(dest: Path = DATA / "diode") -> Path:
    """Fetch and unpack DIODE val (2.6 GB) once."""
    root = dest / "val"
    if root.exists():
        return root
    dest.mkdir(parents=True, exist_ok=True)
    tgz = dest / "val.tar.gz"
    if not tgz.exists():
        print(f"[data] downloading {DIODE_URL} (2.6 GB) ...", flush=True)
        urllib.request.urlretrieve(DIODE_URL, tgz)     # noqa: S310  # nosec B310: fixed https dataset URL
    with tarfile.open(tgz) as t:
        t.extractall(dest, filter="data")
    return root


def diode(n_indoor: int = 120, n_outdoor: int = 160, root: Path | None = None) -> list[Still]:
    """An evenly spaced slice of DIODE val at 640x480, cached as one npz per split."""
    import cv2
    out = []
    for split, n in (("indoors", n_indoor), ("outdoor", n_outdoor)):
        if n <= 0:
            continue
        cache = DATA / f"diode_{split}_{n}_{SIZE[0]}x{SIZE[1]}.npz"
        if cache.exists():
            z = np.load(cache)
            names = [str(s) for s in z["names"]]
        else:
            base = (root or download_diode()) / split
            pngs = sorted(base.rglob("*.png"))
            pick = [pngs[int(i)] for i in np.linspace(0, len(pngs) - 1, min(n, len(pngs)))]
            imgs, deps, names = [], [], []
            for p in pick:
                img = cv2.imread(str(p))
                d = np.load(str(p).replace(".png", "_depth.npy"))[..., 0]
                m = np.load(str(p).replace(".png", "_depth_mask.npy")) > 0
                d = np.where(m & (d > 0) & (d <= MAX_GT), d, 0).astype(np.float32)
                imgs.append(cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA))
                deps.append(cv2.resize(d, SIZE, interpolation=cv2.INTER_NEAREST))
                names.append(p.stem)
            DATA.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, images=np.stack(imgs), depth=np.stack(deps), names=np.array(names))
            z = np.load(cache)
        K = DIODE_K.scaled(*SIZE)
        out += [Still(nm, split, z["images"][i], z["depth"][i], K) for i, nm in enumerate(names)]
    return out
