"""How far are the sim's drone frames from real Tello frames? Image statistics and detector behaviour, side by side.

    # sim frames, camera model OFF (this tool applies the model itself, so one capture serves every model version)
    cd sim && npx vite --port 5199 --strictPort &
    uv run --no-project --with playwright --with ultralytics --with opencv-python-headless \
        python sim/yolo_check.py --url "http://localhost:5199/?world=moss&cam=raw" --frames out/sim_raw --out out
    # the report: real | sim raw | sim + camera model
    uv run --no-project --with ultralytics --with opencv-python-headless \
        python scripts/sim_real_gap.py --real recordings/20261024_070312_tello --sim out/sim_raw
    # fit the camera model JSON to the real stats (writes --fit-out, default the model JSON itself)
    ... python scripts/sim_real_gap.py --real <take> --sim out/sim_raw --fit

--real: a kite.record take (its video.mkv or frames/), any video file, or a folder of images. Frames that are not
960x720 are scaled to 720 rows and centre-cropped to 960 columns. --sim: a folder of raw sim frames (--no-apply if
they already went through the model, e.g. saved from SimFrameSource).

Detector: YOLO person / car confidences, binned by a distance estimated from the box height (person 1.7 m, car
1.5 m tall, fy from data/tello_camera.json for real frames and the sim's 1000 px for sim frames), so real footage
needs no ground truth. --no-yolo skips it.

The stats depend on the scene as well as the camera, so the comparison means most on real footage of a street like
the sim's. Until the first Tello recording exists this was run on a stock clip (README); re-run it, and --fit, on
the first real Tello take.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from tello_cam import MODEL_PATH, TelloCam, load_model  # noqa: E402

W, H = 960, 720
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")
VIDEO_EXT = (".mkv", ".mp4", ".avi", ".mov", ".m4v", ".h264")
SIM_FY = 1000.0                      # sim/src/config.ts FX
OBJ_H = {"person": 1.7, "car": 1.5}  # m
COCO = {0: "person", 2: "car", 5: "car", 7: "car"}
BANDS = ((0, 10, "<10m"), (10, 20, "10-20m"), (20, 1e9, ">20m"))
HIST_BINS = 32


# ------------------------------------------------------------------ frames
def to_tello_size(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if (w, h) == (W, H):
        return bgr
    if h != H:
        bgr = cv2.resize(bgr, (round(w * H / h), H), interpolation=cv2.INTER_AREA)
        w = bgr.shape[1]
    if w < W:                        # narrower than 4:3: scale to the width instead and crop rows
        bgr = cv2.resize(bgr, (W, round(H * W / w)), interpolation=cv2.INTER_AREA)
        y = (bgr.shape[0] - H) // 2
        return bgr[y:y + H]
    x = (w - W) // 2 // 8 * 8        # keep the 8 px block grid where the source had one
    return bgr[:, x:x + W]


def load_frames(path: str | Path, n: int) -> list[np.ndarray]:
    p = Path(path)
    if p.is_dir():
        if (p / "video.mkv").is_file():          # kite.record take
            return load_frames(p / "video.mkv", n)
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise SystemExit(f"no images in {p}")
        pick = [files[i] for i in np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)]
        return [to_tello_size(cv2.imread(str(f))) for f in pick]
    if p.suffix.lower() not in VIDEO_EXT:
        raise SystemExit(f"{p}: not a folder or a video")
    cap = cv2.VideoCapture(str(p))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int) if total else range(n):
        if total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(to_tello_size(f))
    cap.release()
    if not out:
        raise SystemExit(f"could not read frames from {p}")
    return out


# ------------------------------------------------------------------ per-frame statistics
def gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian: high = crisp edges and fine detail."""
    return float(cv2.Laplacian(gray(bgr), cv2.CV_32F).var())


def noise_sigma(bgr: np.ndarray) -> float:
    """Immerkaer's fast noise estimate (sigma of additive Gaussian noise, 0-255 units)."""
    g = gray(bgr)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(g, cv2.CV_32F, k)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2) * np.abs(r).mean() / 6)


def blockiness(bgr: np.ndarray, block: int = 8) -> float:
    """Mean step across 8 px block boundaries / mean step elsewhere. 1.0 = no block grid."""
    g = gray(bgr)
    ratios = []
    for d in (np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0)).T):
        at = (np.arange(d.shape[1]) % block) == block - 1
        ratios.append(d[:, at].mean() / max(d[:, ~at].mean(), 1e-6))
    return float(np.mean(ratios))


def vignette_ratio(bgr: np.ndarray) -> float:
    """Mean luma of the four corner patches / the centre patch."""
    g = gray(bgr)
    h, w = g.shape
    ph, pw = h // 8, w // 8
    corners = np.mean([g[:ph, :pw].mean(), g[:ph, -pw:].mean(), g[-ph:, :pw].mean(), g[-ph:, -pw:].mean()])
    centre = g[h // 2 - ph // 2:h // 2 + ph // 2, w // 2 - pw // 2:w // 2 + pw // 2].mean()
    return float(corners / max(centre, 1e-6))


def frame_stats(bgr: np.ndarray) -> dict:
    g = gray(bgr)
    b, gr, r = (bgr[..., i].astype(np.float32).mean() for i in range(3))
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32).mean()
    return {
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
        "sharpness": sharpness(bgr),
        "noise": noise_sigma(bgr),
        "blockiness": blockiness(bgr),
        "saturation": float(sat),
        "r_over_g": float(r / max(gr, 1e-6)),
        "b_over_g": float(b / max(gr, 1e-6)),
        "vignette": vignette_ratio(bgr),
    }


def colour_hist(frames: list[np.ndarray]) -> np.ndarray:
    """Per-channel (B, G, R) luminance histograms over all frames, each normalised to sum 1. Shape (3, HIST_BINS)."""
    h = np.zeros((3, HIST_BINS), np.float64)
    for f in frames:
        for c in range(3):
            h[c] += np.bincount(f[..., c].ravel() // (256 // HIST_BINS), minlength=HIST_BINS)
    return h / h.sum(axis=1, keepdims=True)


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Bhattacharyya distance across channels: 0 = identical, 1 = disjoint."""
    bc = np.sqrt(a * b).sum(axis=1)
    return float(np.mean(np.sqrt(np.clip(1 - bc, 0, 1))))


def summarise(frames: list[np.ndarray]) -> dict:
    per = [frame_stats(f) for f in frames]
    return {k: float(np.median([p[k] for p in per])) for k in per[0]}


# ------------------------------------------------------------------ detector
def detections(frames: list[np.ndarray], fy: float, model) -> list[tuple[str, float, float]]:
    """(class, confidence, estimated distance m) for every person / car YOLO finds at conf >= 0.1."""
    out = []
    for f in frames:
        res = model(f, verbose=False, conf=0.1)[0]
        for box, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            k = COCO.get(int(cls))
            if k:
                out.append((k, float(conf), fy * OBJ_H[k] / max(box[3] - box[1], 1.0)))
    return out


def detector_table(dets: list, n_frames: int) -> dict:
    t = {}
    for k in OBJ_H:
        for lo, hi, band in BANDS:
            c = [conf for kk, conf, d in dets if kk == k and lo <= d < hi]
            t[f"{k} {band}"] = (len(c) / max(n_frames, 1), float(np.median(c)) if c else None,
                                sum(x >= 0.25 for x in c) / max(len(c), 1) if c else None)
    return t


# ------------------------------------------------------------------ the model and the fit
def apply_model(frames: list[np.ndarray], model: dict, seed: int = 0) -> list[np.ndarray]:
    """Each frame through a fresh (settled) TelloCam, seeded, so a fit sees the same noise every time."""
    return [TelloCam(model, seed=seed + i).process(f) for i, f in enumerate(frames)]


# stat -> scale it is compared on (differences are divided by this); sharpness on a log scale
STAT_SCALE = {"brightness": 10.0, "contrast": 8.0, "sharpness": 0.3, "noise": 0.5, "blockiness": 0.05,
              "saturation": 10.0, "r_over_g": 0.03, "b_over_g": 0.03, "vignette": 0.05}

# (path in the model JSON, low, high, initial step)
FIT_PARAMS = [
    (("exposure", "target_luma"), 60.0, 180.0, 15.0),
    (("color", "contrast"), 0.5, 1.4, 0.1),
    (("color", "black_lift"), 0.0, 40.0, 5.0),
    (("color", "saturation"), 0.3, 1.5, 0.1),
    (("color", "gains_rgb", 0), 0.7, 1.3, 0.04),
    (("color", "gains_rgb", 2), 0.7, 1.3, 0.04),
    (("lens", "softness_sigma_px"), 0.0, 3.0, 0.4),
    (("noise", "read"), 0.0, 10.0, 1.0),
    (("noise", "shot"), 0.0, 3.0, 0.3),
    (("compression", "jpeg_quality"), 10.0, 95.0, 10.0),
    (("vignette", "strength"), 0.0, 0.6, 0.08),
]


def get_param(model: dict, path: tuple):
    v = model
    for k in path:
        v = v[k]
    return v


def set_param(model: dict, path: tuple, value: float) -> dict:
    m = copy.deepcopy(model)
    v = m
    for k in path[:-1]:
        v = v[k]
    v[path[-1]] = round(float(value), 4)
    return m


def gap(a: dict, b: dict) -> float:
    """Squared distance between two stat summaries, each stat on its STAT_SCALE (sharpness in log units)."""
    total = 0.0
    for k, s in STAT_SCALE.items():
        x, y = (math.log(max(a[k], 1e-3)), math.log(max(b[k], 1e-3))) if k == "sharpness" else (a[k], b[k])
        total += ((x - y) / s) ** 2
    return total


def fit(sim_raw: list[np.ndarray], target: dict, model: dict, passes: int = 5, log=print) -> tuple[dict, float]:
    """Coordinate descent on FIT_PARAMS: try each parameter one step up and down, keep improvements, halve the steps."""
    score = lambda m: gap(summarise(apply_model(sim_raw, m)), target)  # noqa: E731
    best = score(model)
    steps = {p[0]: p[3] for p in FIT_PARAMS}
    for i in range(passes):
        for path, lo, hi, _ in FIT_PARAMS:
            for sign in (1, -1):
                cand = set_param(model, path, float(np.clip(get_param(model, path) + sign * steps[path], lo, hi)))
                s = score(cand)
                if s < best:
                    model, best = cand, s
                    break
        log(f"  fit pass {i + 1}: gap {best:.2f}")
        steps = {k: v / 2 for k, v in steps.items()}
    return model, best


# ------------------------------------------------------------------ report
def print_stats(cols: dict[str, dict], hists: dict[str, np.ndarray]) -> None:
    names = list(cols)
    print(f"{'stat (median per frame)':26s}" + "".join(f"{n:>16s}" for n in names))
    for k in STAT_SCALE:
        print(f"{k:26s}" + "".join(f"{cols[n][k]:16.3f}" for n in names))
    real = hists.get("real")
    if real is not None:
        print(f"{'colour hist dist to real':26s}" + "".join(f"{hist_distance(hists[n], real):16.3f}" for n in names))
        print(f"{'gap to real (fit score)':26s}" + "".join(f"{gap(cols[n], cols['real']):16.2f}" for n in names))


def print_detector(tables: dict[str, dict]) -> None:
    names = list(tables)
    print(f"\n{'YOLO: per frame | median conf | share >= 0.25':40s}")
    print(f"{'class, est. distance':18s}" + "".join(f"{n:>24s}" for n in names))
    for row in next(iter(tables.values())):
        cells = []
        for n in names:
            rate, med, ok = tables[n][row]
            cells.append(f"{rate:5.2f} | {med:.2f} | {ok:.0%}" if med is not None else f"{rate:5.2f} |   -  |  - ")
        print(f"{row:18s}" + "".join(f"{c:>24s}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real", required=True, help="kite.record take, video file or image folder")
    ap.add_argument("--sim", required=True, help="folder of sim drone frames (raw, unless --no-apply)")
    ap.add_argument("--n", type=int, default=40, help="frames sampled from each side")
    ap.add_argument("--model", default=str(MODEL_PATH), help="camera model JSON")
    ap.add_argument("--no-apply", action="store_true", help="--sim frames already went through the camera model")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolo11n.pt"))
    ap.add_argument("--real-fy", type=float, default=None, help="default: data/tello_camera.json (or 920)")
    ap.add_argument("--fit", action="store_true", help="tune the model JSON to the real stats")
    ap.add_argument("--fit-frames", type=int, default=8, help="sim frames the fit renders per try")
    ap.add_argument("--fit-out", default=None, help="where --fit writes the model (default: --model)")
    ap.add_argument("--json", default=None, help="also write the numbers here")
    a = ap.parse_args()

    real = load_frames(a.real, a.n)
    sim_raw = load_frames(a.sim, a.n)
    model = load_model(a.model)
    print(f"real: {len(real)} frames from {a.real}\nsim:  {len(sim_raw)} frames from {a.sim}\nmodel: {a.model}\n")
    sets = {"real": real, "sim raw" if not a.no_apply else "sim": sim_raw}
    if not a.no_apply:
        sets["sim + model"] = apply_model(sim_raw, model)

    fitted = None
    if a.fit:
        if a.no_apply:
            raise SystemExit("--fit needs raw sim frames (drop --no-apply)")
        sub = sim_raw[:: max(1, len(sim_raw) // a.fit_frames)][: a.fit_frames]
        print(f"fitting {len(FIT_PARAMS)} parameters on {len(sub)} sim frames ...")
        fitted, score = fit(sub, summarise(real), model)
        sets["sim + fitted"] = apply_model(sim_raw, fitted)
        fitted["_fit"] = {"real": str(a.real), "sim": str(a.sim), "gap": round(score, 3),
                          "note": "scripts/sim_real_gap.py --fit"}
        out = Path(a.fit_out or a.model)
        out.write_text(json.dumps(fitted, indent=2) + "\n")
        print(f"wrote {out}\n")

    cols = {n: summarise(f) for n, f in sets.items()}
    hists = {n: colour_hist(f) for n, f in sets.items()}
    print_stats(cols, hists)
    report = {"stats": cols, "hist_dist_to_real": {n: hist_distance(h, hists["real"]) for n, h in hists.items()}}

    if not a.no_yolo:
        from ultralytics import YOLO
        yolo = YOLO(a.weights)
        if a.real_fy is None:
            from kite.camera import load_camera
            a.real_fy = load_camera().fy
        tables = {n: detector_table(detections(f, a.real_fy if n == "real" else SIM_FY, yolo), len(f))
                  for n, f in sets.items()}
        print_detector(tables)
        report["detector"] = tables
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=float) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
