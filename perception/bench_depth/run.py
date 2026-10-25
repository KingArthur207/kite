"""The depth shoot-out: every depth source in perception/depth_zoo.py against metric ground truth.

    python -m perception.bench_depth --setup            # once, with internet: DIODE slice, weights, third-party code
    python -m perception.bench_depth                    # everything; writes perception/docs/depth_shootout.{md,json,png}
    python -m perception.bench_depth --quick            # 20 DIODE frames, short sequences, small models only

Each model runs once per frame; its variants (raw, floor-anchored, temporally smoothed, fused with motion
stereo) are computed from that one output by replaying it through the wrappers, so a variant costs only its own
overhead. One model is loaded at a time (this was written on an 8 GB M1).

Python env (torch is not in the earpiece venv): uv run --no-project --python 3.12 --with torch --with transformers
--with opencv-python-headless --with scipy --with ultralytics --with einops --with timm --with mmengine
--with easydict --with huggingface_hub python -m perception.bench_depth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import depth_zoo as dz
from . import data, metrics as M, synth

DOCS = dz.ROOT / "perception" / "docs"
THIRD_PARTY = {   # pinned commits of the code we run but do not vendor
    "UniDepth": ("https://github.com/lpiccinelli-eth/UniDepth", "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"),
    "Metric3D": ("https://github.com/YvanYin/Metric3D", "eb5b6fac0dc155e4e52f576e304fbf11655ff339"),
    "Video-Depth-Anything": ("https://github.com/DepthAnything/Video-Depth-Anything",
                             "4f5ae23172ba60fd7bc11ef671cca678842c7072"),
}
# name, run every k-th synthetic frame (1 = all, needed for temporal variants and fusion), max DIODE frames
MODELS = [
    ("da2-metric-outdoor-s", 1, None),
    ("da2-metric-indoor-s", 1, None),
    ("da2-relative-s", 1, None),
    ("da2-metric-outdoor-b", 1, None),
    ("da2-metric-indoor-b", 1, None),
    ("vda-metric-s", 1, 0),                 # a video model: sequences only
    ("unidepth-v2-s", 2, None),
    ("metric3d-v2-s", 3, None),
    ("depth-pro", 4, 60),
]
SEQS = ("sidewalk", "curb", "hallway", "standstill")
SHEET = {"diode-outdoor": 0, "diode-indoors": 0, "synth": ("sidewalk", 96)}


# ------------------------------------------------------------------ setup
def setup(models: list[str]) -> None:
    os.environ["KITE_DEPTH_DOWNLOAD"] = "1"
    data.diode()
    tp = dz.third_party()
    tp.mkdir(parents=True, exist_ok=True)
    for name, (url, rev) in THIRD_PARTY.items():
        d = tp / name
        if not d.exists():
            subprocess.run(["git", "clone", "-q", url, str(d)], check=True)
            subprocess.run(["git", "-C", str(d), "checkout", "-q", rev], check=True)
    for m in models:
        try:
            dz.get(m, "cpu").close()
            print(f"[setup] {m}: ok", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[setup] {m}: unavailable: {e}", flush=True)


# ------------------------------------------------------------------ replay: one model output, many variants
class Replay(dz.DepthBackend):
    def __init__(self, results: list, name: str, metric: bool):
        self.results, self.name, self.metric, self.i = results, name, metric, 0

    def reset(self):
        self.i = 0

    def estimate(self, frame_bgr, intrinsics=None):
        r = self.results[self.i]
        self.i += 1
        return dz.DepthResult(r.depth_m, r.confidence, r.latency_ms, dict(r.meta))


class DiskReplay(dz.DepthBackend):
    """Replay from _stash's files (float16 depth and confidence, memory-mapped)."""
    metric = True

    def __init__(self, stem: Path, name: str):
        self.name, self.stem, self.i = name, stem, 0
        self.depth = np.load(f"{stem}-depth.npy", mmap_mode="r")
        self.conf = np.load(f"{stem}-conf.npy", mmap_mode="r")
        self.lat = np.load(f"{stem}-lat.npy")
        self.sigma = np.load(f"{stem}-sigma.npy", mmap_mode="r") if Path(f"{stem}-sigma.npy").exists() else None

    def reset(self):
        self.i = 0

    def estimate(self, frame_bgr, intrinsics=None):
        i = self.i
        self.i += 1
        meta = {} if self.sigma is None else {"sigma_rel": np.where(self.sigma[i] >= 6e4, np.inf,
                                                                    self.sigma[i]).astype(np.float32)}
        return dz.DepthResult(self.depth[i].astype(np.float32), self.conf[i].astype(np.float32), float(self.lat[i]),
                              meta)


def _stash(name: str, sname: str, results: list) -> Path:
    d = data.DATA / "replay"
    d.mkdir(parents=True, exist_ok=True)
    stem = d / f"{name}-{sname}"
    np.save(f"{stem}-depth.npy", np.stack([r.depth_m for r in results]).astype(np.float16))
    np.save(f"{stem}-conf.npy", np.stack([r.confidence for r in results]).astype(np.float16))
    np.save(f"{stem}-lat.npy", np.array([r.latency_ms for r in results]))
    if "sigma_rel" in results[0].meta:
        np.save(f"{stem}-sigma.npy", np.stack([np.minimum(r.meta["sigma_rel"], 6e4) for r in results]).astype(np.float16))
    return stem


# ------------------------------------------------------------------ scoring
class Scores:
    """Per (variant, dataset): per-frame metrics, bands, pitch errors, latencies."""

    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = defaultdict(lambda: defaultdict(list))

    def add(self, variant: str, dataset: str, **kv):
        r = self.rows[(variant, dataset)]
        for k, v in kv.items():
            if v is not None:
                r[k].append(v)

    def summary(self) -> list[dict]:
        out = []
        for (variant, ds), r in sorted(self.rows.items()):
            def mean(k, r=r):
                v = [x for x in r.get(k, []) if x is not None and math.isfinite(x)]
                return round(float(np.mean(v)), 4) if v else None

            pr = M.priority_scores(r.get("band_pred", []), r.get("band_true", []))
            row = {"variant": variant, "dataset": ds, "frames": len(r.get("absrel", [])) or len(r.get("band_true", [])),
                   "absrel": mean("absrel"), "rmse": mean("rmse"), "d1": mean("d1"),
                   "near_absrel": mean("near_absrel"), "near_rmse": mean("near_rmse"),
                   "near_dist_err": mean("near_dist_err"),
                   "priority_acc": None if pr["acc"] is None else round(pr["acc"], 3),
                   "p1_recall": None if pr["p1_recall"] is None else round(pr["p1_recall"], 3),
                   "false_p1": None if pr["false_p1"] is None else round(pr["false_p1"], 3),
                   "priority_frames": pr["n"], "p1_frames": pr.get("n_p1", 0),
                   "pitch_err": mean("pitch_err"), "anchor_ok": mean("anchor_ok"),
                   "coverage": mean("coverage"), "conf_absrel": mean("conf_absrel"),
                   "latency_ms": round(float(np.median(r["latency"])), 1) if r.get("latency") else None}
            out.append(row)
        return out


_TRUTH: dict[int, dict] = {}


def _truth(gt: np.ndarray, K, geo) -> dict:
    """Ground-truth quantities shared by every variant scored on this frame (cached by array identity)."""
    t = _TRUTH.get(id(gt))
    if t is not None and t["gt"] is gt:
        return t
    valid = (gt > 0) & (gt <= M.MAX_GT)
    t = {"gt": gt, "valid": valid, "nvalid": max(int(valid.sum()), 1)}
    if geo is not None:
        down, h = geo
        t["corridor"] = M.corridor_mask(gt, K, down, h)
        t["nearest"] = M.nearest_obstacle(gt, K, down, h, valid)
        t["pitch"] = dz.angles_of(down)[0]
    if len(_TRUTH) > 4000:
        _TRUTH.clear()
    _TRUTH[id(gt)] = t
    return t


def score(sc: Scores, variant: str, ds, pred: np.ndarray, conf: np.ndarray | None, gt: np.ndarray, K, geo,
          latency: float | None = None, pitch: float | None = None, anchor_ok: bool | None = None):
    """geo: (down, cam_h) from the ground truth, or None when the true floor is not visible. ds: one dataset
    name or several (the same frame counts in the pooled and the per-sequence rows)."""
    tr = _truth(gt, K, geo)
    valid = tr["valid"]
    d = M.dense(pred, gt, valid)
    kv = {"absrel": d["absrel"], "rmse": d["rmse"], "d1": d["d1"], "latency": latency}
    if conf is not None:
        c = conf > 0.5
        kv["coverage"] = float((c & valid).sum() / tr["nvalid"])
        kv["conf_absrel"] = M.dense(pred, gt, valid & c)["absrel"]
    if geo is not None:
        down, h = geo
        if tr["corridor"].sum() >= 50:
            nd = M.dense(pred, gt, tr["corridor"])
            kv["near_absrel"], kv["near_rmse"] = nd["absrel"], nd["rmse"]
        t = tr["nearest"]
        p = M.nearest_obstacle(pred, K, down, h)
        kv["band_true"], kv["band_pred"] = M.band(t), M.band(p)
        if t is not None and p is not None and t <= M.P2_M:
            kv["near_dist_err"] = abs(p - t)
        if pitch is not None and math.isfinite(pitch):
            kv["pitch_err"] = abs(pitch - tr["pitch"])
    if anchor_ok is not None:
        kv["anchor_ok"] = float(anchor_ok)
    for name in ((ds,) if isinstance(ds, str) else ds):
        sc.add(variant, name, **kv)


def score_boxes(sc: Scores, ds: str, boxes, gt, K, geo):
    """The two box baselines: ScrollSafe's height rule and the stand-in's flat-ground range."""
    if geo is None:
        return
    down, h = geo
    t = M.band(_truth(gt, K, geo)["nearest"])
    pitch, roll = dz.angles_of(down)
    sc.add("baseline: ScrollSafe rule (boxes)", ds, band_true=t, band_pred=M.scrollsafe_band(boxes, K.width, K.height))
    sc.add("baseline: flat ground (boxes)", ds, band_true=t,
           band_pred=M.band(M.flat_ground_box_distance(boxes, K, h, pitch, roll)))


# ------------------------------------------------------------------ ground-truth geometry for real frames
def gt_geometry(gt: np.ndarray, K) -> tuple[np.ndarray, float] | None:
    """The true floor plane, when one is clearly in view: DIODE's scanner tilts up and down (its crops are named
    by azimuth and elevation) and many frames show only walls, ceilings or canopy. Kept: a floor seen like a
    chest or tripod camera sees it (pitch -5 to 35 degrees, roll within 12, height 0.8 to 3 m)."""
    fit = dz.fit_ground_plane(gt, K, 1.0, mask=gt > 0, pitch_range=(-5.0, 35.0), max_roll=12.0, min_ratio=0.2)
    if not fit.ok or not 0.8 <= fit.height <= 3.0:
        return None
    return fit.normal, float(fit.height)


def yolo_boxes(stills, weights: str | None) -> list[list[dict]] | None:
    """COCO boxes (YOLO11n, conf 0.35, like the stand-in) for the box baselines on DIODE; cached."""
    cache = data.DATA / f"yolo_boxes_{len(stills)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    w = weights or os.environ.get("KITE_YOLO_WEIGHTS") or str(dz.ROOT / "weights" / "yolo11n.pt")
    if not Path(w).exists():
        print(f"[bench] no {w}: box baselines skipped on DIODE", flush=True)
        return None
    from ultralytics import YOLO
    model = YOLO(w)
    out = []
    for s in stills:
        r = model.predict(s.image, verbose=False, conf=0.35, imgsz=640)[0]
        out.append([{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "name": r.names[int(c)]}
                    for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())])
    cache.write_text(json.dumps(out))
    return out


# ------------------------------------------------------------------ one model on DIODE
def run_diode(name: str, be, stills, geos, sc: Scores, sheet: dict, limit: int | None):
    by_split = defaultdict(list)
    for i, s in enumerate(stills):
        by_split[s.split].append(i)
    idx = []
    for split, ii in by_split.items():
        n = len(ii) if limit is None else min(limit // 2, len(ii))
        idx += [ii[int(k)] for k in np.linspace(0, len(ii) - 1, n)] if n else []
    heights = [g[1] for g in geos if g is not None]
    h_default = float(np.median(heights)) if heights else 1.5
    for j, i in enumerate(idx):
        s = stills[i]
        ds = f"diode-{s.split}"
        geo = geos[i]
        h = geo[1] if geo else h_default
        r = be.estimate(s.image, s.K)
        if j == 0:                         # the first call pays for kernels and caches: run it again, time that
            r = be.estimate(s.image, s.K)
        if be.metric:
            score(sc, name, ds, r.depth_m, r.confidence, s.depth, s.K, geo, r.latency_ms)
            fit = dz.fit_ground_plane(r.depth_m, s.K, h)
            z = r.depth_m * fit.scale if fit.ok else r.depth_m
            score(sc, f"{name}+anchor", ds, z, r.confidence, s.depth, s.K, geo, r.latency_ms,
                  fit.pitch if fit.ok else None, fit.ok)
        else:
            q = r.meta["disparity"]
            score(sc, f"{name} (oracle affine fit to GT)", ds, M.oracle_affine_disparity(q, s.depth, s.depth > 0),
                  None, s.depth, s.K, geo, r.latency_ms)
            fit = dz.fit_ground_disparity(q, s.K, h)
            z = dz.disparity_to_depth(q, fit.scale, fit.shift) if fit.ok else r.depth_m
            score(sc, f"{name}+anchor (shift 0)", ds, z, r.confidence, s.depth, s.K, geo, r.latency_ms,
                  fit.pitch if fit.ok else None, fit.ok)
            if geo is not None:
                fit = dz.fit_ground_disparity(q, s.K, h, pitch_prior=dz.angles_of(geo[0])[0])
                z = dz.disparity_to_depth(q, fit.scale, fit.shift) if fit.ok else r.depth_m
                score(sc, f"{name}+anchor (pitch prior)", ds, z, r.confidence, s.depth, s.K, geo, r.latency_ms,
                      None, fit.ok)
        if sheet.get(ds) == i:
            sheet.setdefault("maps", {}).setdefault(ds, {})[name] = _thumb(z)
        print(f"\r[bench] {name} diode {j + 1}/{len(idx)} {r.latency_ms:.0f} ms ", end="", flush=True)
    print(flush=True)


def _thumb(z: np.ndarray) -> list:
    import cv2
    return cv2.resize(z.astype(np.float32), (160, 120), interpolation=cv2.INTER_AREA).round(3).tolist()


# ------------------------------------------------------------------ one model on the synthetic walks
def run_synth(name: str, be, seqs: dict, sc: Scores, sheet: dict, every: int, keep: dict):
    for sname, seq in seqs.items():
        be.reset()
        frames = range(0, len(seq.frames), every)
        results = []
        for j, k in enumerate(frames):
            r = be.estimate(seq.frames[k], seq.K)
            if j == 0 and not be.temporal:
                r = be.estimate(seq.frames[k], seq.K)
            results.append(r)
            print(f"\r[bench] {name} {sname} {j + 1}/{len(frames)} {r.latency_ms:.0f} ms ", end="", flush=True)
        variants = _synth_variants(name, be.metric, results, every, synth.SEQS[sname][4])
        for vname, (wrapped, extra_ms) in variants.items():
            wrapped.reset()
            for j, k in enumerate(frames):
                rr = wrapped.estimate(seq.frames[k], seq.K)
                geo = (seq.down[k], float(seq.cam_h[k]))
                an = rr.meta.get("anchor") or {}
                score(sc, vname, ("synth", f"synth-{sname}"), rr.depth_m, rr.confidence, seq.depth[k], seq.K, geo,
                      rr.latency_ms, an.get("pitch") if an.get("ok") else None, an.get("ok") if an else None)
                if (sname, k) == SHEET["synth"]:
                    sheet.setdefault("maps", {}).setdefault("synth", {})[vname] = _thumb(rr.depth_m)
        if every == 1 and be.metric:               # for fusion with motion stereo, on disk (RAM is 8 GB)
            keep[(name, sname)] = _stash(name, sname, results)
        print(flush=True)


def _synth_variants(name, metric, results, every, mount_pitch):
    v = {}
    rp = Replay(results, name, metric)
    if metric:
        v[name] = (rp, 0)
    v[f"{name}+anchor"] = (dz.Anchored(Replay(results, name, metric), cam_h=dz.CHEST_H), 0)
    if not metric:      # a calibrated mounting pitch, as perception/calibrate_camera.py would give
        v[f"{name}+anchor (pitch prior)"] = (dz.Anchored(Replay(results, name, metric), cam_h=dz.CHEST_H,
                                                         pitch_prior=mount_pitch), 0)
    if every == 1:
        v[f"{name}+anchor+temporal"] = (dz.Temporal(dz.Anchored(Replay(results, name, metric), cam_h=dz.CHEST_H)), 0)
    return v


def run_motion(seqs, sc: Scores, sheet) -> dict:
    """Motion stereo alone on every walk; its outputs go to disk for the fusion pass."""
    stems = {}
    for sname, seq in seqs.items():
        ms = dz.MotionStereo(cam_h=dz.CHEST_H)
        res = []
        for k in range(len(seq.frames)):
            r = ms.estimate(seq.frames[k], seq.K)
            res.append(r)
            geo = (seq.down[k], float(seq.cam_h[k]))
            score(sc, "motion-stereo", ("synth", f"synth-{sname}"), r.depth_m, r.confidence, seq.depth[k], seq.K, geo,
                  r.latency_ms, r.meta.get("pitch") if r.meta.get("scale_ok") else None)
            # where it is confident only (ARCore's "raw depth"); everything else counts as unknown
            conf_only = np.where(r.confidence > 0.5, r.depth_m, 0)
            score(sc, "motion-stereo (confident px only)", ("synth", f"synth-{sname}"), conf_only, None, seq.depth[k],
                  seq.K, geo)
            if (sname, k) == SHEET["synth"]:
                sheet.setdefault("maps", {}).setdefault("synth", {})["motion-stereo"] = _thumb(r.depth_m)
            print(f"\r[bench] motion-stereo {sname} {k + 1}/{len(seq.frames)} {r.latency_ms:.0f} ms ", end="",
                  flush=True)
        stems[sname] = _stash("motion-stereo", sname, res)
        print(flush=True)
    return stems


def run_fusion(seqs, sc: Scores, sheet, keep: dict, ms_stems: dict):
    """Each dense model that ran on every frame, anchored, fused with motion stereo."""
    for (name, sname), stem in keep.items():
        seq = seqs[sname]
        fused = dz.Fused(dz.Anchored(DiskReplay(stem, name), cam_h=dz.CHEST_H),
                         DiskReplay(ms_stems[sname], "motion-stereo"))
        vname = f"fused: {name}+anchor + motion stereo"
        for k in range(len(seq.frames)):
            r = fused.estimate(seq.frames[k], seq.K)
            geo = (seq.down[k], float(seq.cam_h[k]))
            score(sc, vname, ("synth", f"synth-{sname}"), r.depth_m, r.confidence, seq.depth[k], seq.K, geo,
                  r.latency_ms)
            if (sname, k) == SHEET["synth"]:
                sheet.setdefault("maps", {}).setdefault("synth", {})[vname] = _thumb(r.depth_m)
        print(f"[bench] fused {name} {sname}", flush=True)


def run_flat(stills, geos, seqs, sc: Scores, boxes_diode):
    for i, s in enumerate(stills):
        geo = geos[i]
        if geo is None:
            continue
        p, r = dz.angles_of(geo[0])
        fg = dz.FlatGround(cam_h=geo[1], pitch=p, roll=r).estimate(s.image, s.K)
        score(sc, "baseline: flat ground (dense, true pitch)", f"diode-{s.split}", fg.depth_m, None, s.depth, s.K, geo,
              fg.latency_ms)
        if boxes_diode is not None:
            score_boxes(sc, f"diode-{s.split}", boxes_diode[i], s.depth, s.K, geo)
    for sname, seq in seqs.items():
        for k in range(len(seq.frames)):
            geo = (seq.down[k], float(seq.cam_h[k]))
            p, r = dz.angles_of(geo[0])
            fg = dz.FlatGround(cam_h=dz.CHEST_H, pitch=p, roll=r).estimate(seq.frames[k], seq.K)
            score(sc, "baseline: flat ground (dense, true pitch)", ("synth", f"synth-{sname}"), fg.depth_m, None,
                  seq.depth[k], seq.K, geo, fg.latency_ms)
            for ds in ("synth", f"synth-{sname}"):
                score_boxes(sc, ds, seq.boxes[k], seq.depth[k], seq.K, geo)


# ------------------------------------------------------------------ CPU latency
def cpu_latency(models: list[str], img: np.ndarray, K, frames: int = 3) -> dict:
    out = {}
    for m in models:
        try:
            be = dz.get(m, "cpu")
        except Exception as e:  # noqa: BLE001
            out[m] = f"unavailable: {str(e)[:120]}"
            continue
        n = 1 if m == "depth-pro" else frames
        ts = []
        for _ in range(n + 1):
            ts.append(be.estimate(img, K).latency_ms)
        be.close()
        out[m] = round(statistics.median(ts[1:]), 1)
        print(f"[bench] cpu {m}: {out[m]} ms", flush=True)
    ms = dz.MotionStereo()
    seq = synth.sequence("sidewalk", n=20, cache_dir=data.DATA / "synth")
    lat = [ms.estimate(f, seq.K).latency_ms for f in seq.frames]
    out["motion-stereo"] = round(statistics.median(lat[5:]), 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m perception.bench_depth", description=__doc__.split("\n")[0])
    ap.add_argument("--setup", action="store_true", help="download DIODE val, weights and third-party code, then exit")
    ap.add_argument("--quick", action="store_true", help="a smoke run: few frames, small models")
    ap.add_argument("--models", help="comma-separated subset of: " + ", ".join(m for m, _, _ in MODELS))
    ap.add_argument("--indoor", type=int, default=120)
    ap.add_argument("--outdoor", type=int, default=160)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU latency pass")
    ap.add_argument("--out", default=str(DOCS / "depth_shootout"), help="writes <out>.json (and the report)")
    a = ap.parse_args(argv)
    models = [m for m in MODELS if not a.models or m[0] in a.models.split(",")]
    if a.quick:
        models = [m for m in models if m[0] in ("da2-metric-outdoor-s", "da2-relative-s")]
        a.indoor, a.outdoor = 10, 10
    if a.setup:
        setup([m for m, _, _ in models])
        return
    import torch
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    stills = data.diode(a.indoor, a.outdoor)
    seqs = {s: synth.sequence(s, n=24 if a.quick else None, cache_dir=data.DATA / "synth") for s in SEQS}
    for q in seqs.values():
        q.depth = list(q.depth)                           # stable arrays: the scorer caches ground truth per frame
    geos = [gt_geometry(s.depth, s.K) for s in stills]
    print(f"[bench] {len(stills)} DIODE frames ({sum(g is not None for g in geos)} with a visible floor), "
          f"{sum(len(s.frames) for s in seqs.values())} synthetic frames, device {device}", flush=True)
    for split in ("indoors", "outdoor"):
        if f"diode-{split}" in SHEET:
            # the contact sheet's real frames: the first of each split with a visible floor
            SHEET[f"diode-{split}"] = next((i for i, s in enumerate(stills) if s.split == split and geos[i]), None)
    sc, sheet, keep, status = Scores(), dict(SHEET), {}, {}
    boxes = yolo_boxes(stills, None)
    run_flat(stills, geos, seqs, sc, boxes)
    ms_stems = run_motion(seqs, sc, sheet)
    for name, every, limit in models:
        try:
            be = dz.get(name, device)
        except Exception as e:  # noqa: BLE001
            status[name] = f"unavailable: {e}"
            print(f"[bench] {name}: {status[name]}", flush=True)
            continue
        try:
            if limit != 0:
                run_diode(name, be, stills, geos, sc, sheet, limit)
            run_synth(name, be, seqs, sc, sheet, every, keep)
            status[name] = "ok"
        except Exception as e:  # noqa: BLE001
            status[name] = f"failed: {type(e).__name__}: {e}"
            print(f"\n[bench] {name}: {status[name]}", flush=True)
        finally:
            be.close()
    run_fusion(seqs, sc, sheet, keep, ms_stems)
    cpu = {} if a.no_cpu else cpu_latency([m for m, _, _ in models], stills[-1].image, stills[-1].K)
    # contact-sheet inputs: RGB and GT thumbnails
    import cv2
    for ds, i in (("diode-outdoor", sheet.get("diode-outdoor")), ("diode-indoors", sheet.get("diode-indoors"))):
        if i is not None:
            sheet.setdefault("rgb", {})[ds] = cv2.resize(stills[i].image, (160, 120)).tolist()
            sheet.setdefault("gt", {})[ds] = _thumb(stills[i].depth)
    sname, k = SHEET["synth"]
    if k < len(seqs[sname].frames):
        sheet.setdefault("rgb", {})["synth"] = cv2.resize(seqs[sname].frames[k], (160, 120)).tolist()
        sheet.setdefault("gt", {})["synth"] = _thumb(seqs[sname].depth[k])
    out = {"rows": sc.summary(), "status": status, "cpu_latency_ms": cpu, "device": device,
           "machine": f"{platform.machine()} {platform.mac_ver()[0] or platform.platform()}",
           "chip": _chip(), "frames": {"diode": len(stills), "diode_with_floor": sum(g is not None for g in geos),
                                       "synth": {s: len(q.frames) for s, q in seqs.items()}},
           "licences": dz.describe(), "third_party": THIRD_PARTY, "minutes": round((time.time() - t0) / 60, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    (data.DATA / "sheet.json").write_text(json.dumps(sheet))
    from . import report
    report.write(Path(a.out + ".json"), data.DATA / "sheet.json", Path(a.out))
    print(f"[bench] done in {out['minutes']} min -> {a.out}.md / .json / .png", flush=True)


def _chip() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              check=False).stdout.strip()
    except OSError:
        return ""
