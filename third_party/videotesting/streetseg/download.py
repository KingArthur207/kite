"""Download perception weights ahead of time.

Runtime loading refuses to fetch. Run this once, on a network:

    python -m streetseg.download
    python -m streetseg.download --signs --depth
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "weights"

YOLO = {
    "yolo26n-sem.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem.pt",
        1_000_000,
    ),
    "yolo26n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        1_000_000,
    ),
}

SIGNS_REPO = "cvtechniques/TrafficSignDetection"
SIGNS_FILE = "best.pt"
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
SEGFORMER_REPO = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

# Flying Guide Dog release: SegFormer-B0 Mapillary 160k, and the PVTL ResNet-18.
# https://drive.google.com/drive/folders/1efvfGxh2f1nCppO9YaPn6SyUQjG--QkC
GUIDE_DOG = {
    "segmentation/SegFormer_b0_Mapillary.pth": ("1oEkJ7v4pPMVxilPmsZxIupt4zZE9WiB7", 30_000_000),
    "traffic_light_classification/resnet18-best.pth": ("1ILy6VpMcgIPb2INd6gk3JxCIRR9xpnR1", 30_000_000),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download street-perception weights. Not used at runtime.")
    parser.add_argument("--signs", action="store_true", help="crossing-sign checkpoint")
    parser.add_argument("--depth", action="store_true", help="Depth Anything V2 Small, for the chest camera")
    parser.add_argument("--segformer", action="store_true", help="SegFormer-B0 Cityscapes fallback")
    parser.add_argument("--mapillary", action="store_true", help="SegFormer-B0 Mapillary and the PVTL ResNet-18")
    parser.add_argument("--all", action="store_true", help="YOLO, Mapillary, PVTL, signs, depth, and Cityscapes SegFormer")
    args = parser.parse_args(argv)
    fetch(
        signs=args.signs or args.all,
        depth=args.depth or args.all,
        segformer=args.segformer or args.all,
        mapillary=args.mapillary or args.all or not (args.signs or args.depth or args.segformer),
    )
    return 0


def fetch(signs: bool = False, depth: bool = False, segformer: bool = False, mapillary: bool = False) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, (url, min_bytes) in YOLO.items():
        _http(url, ROOT / name, min_bytes)
    if mapillary:
        for name, (file_id, min_bytes) in GUIDE_DOG.items():
            _gdrive(file_id, ROOT / name, min_bytes)
    if signs:
        _hub_file(SIGNS_REPO, SIGNS_FILE, ROOT / "signs")
    if depth:
        _hub_snapshot(DEPTH_REPO, ROOT / "depth-anything-v2-small")
    if segformer:
        _hub_snapshot(SEGFORMER_REPO, ROOT / "segformer-b0-cityscapes")
    print(f"weights are in {ROOT}")


def _http(url: str, dest: Path, min_bytes: int) -> None:
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        print(f"have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {dest.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "kite-perception"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if tmp.stat().st_size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} downloaded smaller than expected")
    tmp.replace(dest)


def _gdrive(file_id: str, dest: Path, min_bytes: int) -> None:
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        print(f"have {dest.name}")
        return
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {dest.name}")
    gdown.download(id=file_id, output=str(dest), quiet=False)
    if not dest.is_file() or dest.stat().st_size < min_bytes:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} downloaded smaller than expected")


def _hub_file(repo: str, filename: str, dest: Path) -> None:
    target = dest / filename
    if target.is_file() and target.stat().st_size > 100_000:
        print(f"have {target}")
        return
    from huggingface_hub import hf_hub_download

    print(f"downloading {repo}/{filename}")
    dest.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo, filename, local_dir=dest)


def _hub_snapshot(repo: str, dest: Path) -> None:
    if (dest / "config.json").is_file():
        print(f"have {dest.name}")
        return
    from huggingface_hub import snapshot_download

    print(f"downloading {repo}")
    snapshot_download(repo, local_dir=dest)


if __name__ == "__main__":
    raise SystemExit(main())
