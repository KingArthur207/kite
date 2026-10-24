"""Benchmark: the stand-in (sim/standin_detect.py) vs perception/ on the same frames.

    python -m perception.bench                      # stills in perception/bench_labels.json + data/imagined/*.mp4
    python -m perception.bench --no-clips --overlays perception/docs

Stills: every hand-labelled street object (perception/bench_labels.json) is "named" by a detector when one
of its boxes contains the label's point, and "named right" when the kind matches too. Both detectors get
the same frame at 960 wide, the same fx and camera height, pitch 0 given (perception may fit its own).
Clips: frames at ~12 Hz through each detector's chest path (tracking, smoothing, stop zone), counting the
frames each kind was reported in, the nearest range, and the first "Stop.".
Latency is wall time per frame per stage, median over the run, on whatever machine runs it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import models, vocab
from .overlay import draw as draw_perception
from .pipeline import Pipeline

import live_walk  # noqa: E402  (path set by perception/__init__)
import standin_detect as SD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = Path(__file__).with_name("bench_labels.json")
CAM_H = {"chest": 1.2, "drone": 2.0}


def load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.resize(img, (960, round(img.shape[0] * 960 / img.shape[1])))


class Standin:
    """The stand-in's detector path, called in-process: YOLO11n + objects_from_boxes (+ chest_step)."""

    def __init__(self, device: str, track: bool):
        from ultralytics import YOLO
        self.model = YOLO(str(SD.WEIGHTS))
        self.device, self.track = device, track
        self.rf = live_walk.RangeFilter()

    def step(self, img, fx, cam_h, camera, t=0.0):
        ts = time.perf_counter()
        r = (self.model.track(img, persist=True, verbose=False, conf=0.35, imgsz=640, device=self.device)
             if self.track else self.model.predict(img, verbose=False, conf=0.35, imgsz=640, device=self.device))[0]
        b = r.boxes
        ids = b.id.int().tolist() if b.id is not None else [None] * len(b)
        boxes = [(*xyxy, r.names[int(c)], cf, tid) for xyxy, c, cf, tid in
                 zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist(), ids)]
        h, w = img.shape[:2]
        objs = SD.objects_from_boxes(boxes, fx, cam_h, w, h)
        if camera == "chest":
            objs = live_walk.chest_step(objs, self.rf, t)
        ms = (time.perf_counter() - ts) * 1000
        kept = [bx for bx in boxes if bx[4] in SD.KIND]
        return [(bx[:4], o) for bx, o in zip(kept, objs)], ms, boxes, objs


_STILL_DET = {}


def perception_pipe(camera, fx, device, track, depth, open_model=None, sim=False):
    """A fresh pipeline; still-image detectors (no tracker state) are loaded once and shared. sim: run as the
    CLI does on sim frames (level cameras: pitch given, only the depth scale fitted; walker's head ignored)."""
    kw = dict(fx=fx, cam_h=CAM_H[camera], auto_pitch=not sim, ignore_bottom=0.1 if sim else 0.0)
    if track or not _STILL_DET:
        coco = models.CocoDetector(device, track=track)
        ov = models.OpenVocabDetector(vocab.load(), device, track=track, weights=open_model or models.OPEN_MODEL)
        det = models.HybridDetector(coco, ov)
        if track:
            return Pipeline(camera, det, depth, **kw)
        _STILL_DET["det"] = det
    return Pipeline(camera, _STILL_DET["det"], depth, **kw)


def boxes_of(r):
    by_id = {o["id"]: o for o in r.objects}
    out = []
    for i, d in enumerate(r.dets):
        o = by_id.get(str(d.tid if d.tid is not None else i))
        if o is not None:
            out.append(((d.x1, d.y1, d.x2, d.y2), o))
    return out


def match(item, found):
    """found: [(box, obj)] -> (obj with the right kind, else any obj) whose box holds the point, or None."""
    x, y = item["at"]
    hits = [o for (x1, y1, x2, y2), o in found if x1 - 6 <= x <= x2 + 6 and y1 - 6 <= y <= y2 + 6]
    right = [o for o in hits if kind_ok(item["kind"], o["kind"])]
    pick = min(right or hits, key=lambda o: o["distance"]) if (right or hits) else None
    return pick, bool(right)


def kind_ok(want: str, got: str) -> bool:
    groups = {"curb": {"curb", "stairs_down"}, "pole": {"pole", "overhead"}, "tree": {"tree"}}
    return got in groups.get(want, {want})


def run_stills(labels, device, depth, overlays: Path | None, reps: int):
    rows, lat = [], defaultdict(list)
    sd = Standin(device, track=False)
    for name, spec in labels.items():
        if name.startswith("_"):
            continue
        img = load(ROOT / name)
        cam, fx = spec["camera"], float(spec["fx"])
        pipe = perception_pipe(cam, fx, device, False, depth, sim=name.startswith("sim/"))
        for k in range(reps):                         # first call per shape pays for kernels; time the rest
            s_found, s_ms, s_boxes, s_objs = sd.step(img, fx, CAM_H[cam], cam, k * 0.1)
            ts = time.perf_counter()
            r = pipe.step(img, k * 0.1)
            p_ms = (time.perf_counter() - ts) * 1000
            if k:
                lat["standin"].append(s_ms)
                lat["perception"].append(p_ms)
                for st, v in r.timings.items():
                    lat[f"perception.{st}"].append(v)
        p_found = boxes_of(r)
        for it in spec["items"]:
            so, s_right = match(it, s_found)
            po, p_right = match(it, p_found)
            rows.append({"scene": name, "camera": cam, "kind": it["kind"], "key": it.get("key", False),
                         "standin": so and {"kind": so["kind"], "distance": so["distance"], "right": s_right},
                         "perception": po and {"kind": po["kind"], "distance": po["distance"], "right": p_right,
                                               "range": po.get("range")},
                         "pitch": r.calib.pitch, "pitch_source": r.calib.source})
        unl = [o["kind"] for b, o in p_found if not any(match(it, [(b, o)])[0] for it in spec["items"])]
        rows.append({"scene": name, "extra_perception": unl,
                     "extra_standin": [o["kind"] for b, o in s_found
                                       if not any(match(it, [(b, o)])[0] for it in spec["items"])]})
        if overlays is not None:
            overlays.mkdir(parents=True, exist_ok=True)
            left = SD.draw(img, s_objs, s_boxes, "stand-in: YOLO11n (COCO)")
            right = draw_perception(img, r, cam)
            pair = cv2.hconcat([cv2.resize(left, (640, round(img.shape[0] * 640 / 960))),
                                cv2.resize(right, (640, round(img.shape[0] * 640 / 960)))])
            cv2.imwrite(str(overlays / f"bench_{Path(name).stem}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return rows, {k: statistics.median(v) for k, v in lat.items() if v}


def sonar_zone(o) -> bool:
    """Would the earpiece's sonar beep for it? Within its 6 m range and its corridor plus the soft margin
    (earpiece/earpiece/config.py SonarCfg: 0.55 + 0.35 m)."""
    b = math.radians(o["bearing"])
    return o["distance"] <= 6.0 and o["distance"] * math.cos(b) > 0 and abs(o["distance"] * math.sin(b)) <= 0.9


def run_clip(path: Path, device, depth, fx: float, every: int = 2):
    sd = Standin(device, track=True)
    pipe = perception_pipe("chest", fx, device, True, depth)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    res = {"standin": defaultdict(list), "perception": defaultdict(list)}
    stop = {"standin": None, "perception": None}
    lat = defaultdict(list)
    i = n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            img = cv2.resize(f, (960, round(f.shape[0] * 960 / f.shape[1])))
            t = i / fps
            _, s_ms, _, s_objs = sd.step(img, fx, 1.2, "chest", t)
            ts = time.perf_counter()
            r = pipe.step(img, t)
            p_ms = (time.perf_counter() - ts) * 1000
            if n:
                lat["standin"].append(s_ms)
                lat["perception"].append(p_ms)
                for st, v in r.timings.items():
                    lat[f"perception.{st}"].append(v)
            n += 1
            sent = r.objects + r.depth_objects
            for who, objs in (("standin", s_objs), ("perception", sent)):
                for k in {o["kind"] for o in objs}:
                    same = [o for o in objs if o["kind"] == k]
                    res[who][k].append((t, min(o["distance"] for o in same), any(sonar_zone(o) for o in same)))
                st = live_walk.stop_text(objs)
                if st and stop[who] is None:
                    stop[who] = f"{t:.1f}s {st}"
        i += 1
    out = {"frames": n, "latency_ms": {k: statistics.median(v) for k, v in lat.items() if v}}
    for who in res:
        out[who] = {k: {"frames": len(v), "first_s": round(min(t for t, _, _ in v), 1),
                        "nearest_m": round(min(d for _, d, _ in v), 1), "sonar_frames": sum(z for _, _, z in v)}
                    for k, v in sorted(res[who].items())}
        out[f"{who}_stop"] = stop[who]
    return out


def markdown(rows, lat, clips) -> str:
    def cell(x):
        if not x:
            return "missed"
        mark = "" if x["right"] else " (wrong kind)"
        return f"{x['kind']} {x['distance']:.1f} m{mark}"

    items = [r for r in rows if "kind" in r]
    md = ["| Scene | Object | Stand-in (YOLO11n) | perception/ (hybrid + depth) |", "|---|---|---|---|"]
    for r in items:
        rng = f" [{r['perception']['range']}]" if r["perception"] else ""
        md.append(f"| {Path(r['scene']).stem} | {r['kind']}{' (key)' if r['key'] else ''} | {cell(r['standin'])} | "
                  f"{cell(r['perception'])}{rng} |")
    tot = Counter()
    for r in items:
        for who in ("standin", "perception"):
            tot[(who, "any")] += bool(r[who])
            tot[(who, "right")] += bool(r[who] and r[who]["right"])
            if r["key"]:
                tot[(who, "key")] += bool(r[who] and r[who]["right"])
    nk = sum(r["key"] for r in items)
    summary = [f"Labelled objects: {len(items)} ({nk} key).",
               f"Named with the right kind: stand-in {tot[('standin', 'right')]}/{len(items)}, "
               f"perception {tot[('perception', 'right')]}/{len(items)}; key objects: stand-in "
               f"{tot[('standin', 'key')]}/{nk}, perception {tot[('perception', 'key')]}/{nk}.",
               f"Detected as anything: stand-in {tot[('standin', 'any')]}/{len(items)}, "
               f"perception {tot[('perception', 'any')]}/{len(items)}.",
               "Latency per still (ms, median): " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(lat.items()))]
    extra = [r for r in rows if "extra_perception" in r]
    ex = ["", "| Scene | Unlabelled boxes, stand-in | Unlabelled boxes, perception/ |", "|---|---|---|"]
    for r in extra:
        ex.append(f"| {Path(r['scene']).stem} | {', '.join(r['extra_standin']) or '-'} | "
                  f"{', '.join(r['extra_perception']) or '-'} |")
    cl = []
    if clips:
        cl = ["", "Per kind: frames it was reported in / frames it was in the sonar's zone (6 m, corridor), nearest m.",
              "", "| Clip | Frames | Stand-in | perception/ | First stop, stand-in | First stop, perception/ |",
              "|---|---|---|---|---|---|"]
        for name, c in clips.items():
            def kinds(d):
                return ", ".join(f"{k} {v['frames']}/{v['sonar_frames']}, {v['nearest_m']} m"
                                 for k, v in d.items()) or "nothing"
            cl.append(f"| {name} | {c['frames']} | {kinds(c['standin'])} | {kinds(c['perception'])} | "
                      f"{c['standin_stop'] or '-'} | {c['perception_stop'] or '-'} |")
        cl.append("")
        cl += [f"- {name} latency (ms, median): "
               + ", ".join(f"{k} {v:.0f}" for k, v in sorted(c["latency_ms"].items()))
               for name, c in clips.items()]
    return "\n".join(summary + [""] + md + ex + cl) + "\n"


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser(prog="python -m perception.bench", description=__doc__.split("\n")[0])
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--only", help="comma-separated substrings: run just these stills")
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--no-stills", action="store_true")
    ap.add_argument("--clips", nargs="*", help="video files (default data/imagined/*.mp4)")
    ap.add_argument("--clip-fx", type=float, default=700.0)
    ap.add_argument("--depth", choices=("outdoor", "indoor", "off"), default="outdoor")
    ap.add_argument("--reps", type=int, default=3, help="runs per still (the first is not timed)")
    ap.add_argument("--overlays", help="write side-by-side overlays here")
    ap.add_argument("--out", default="out/bench", help="results: <out>.json and <out>.md")
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    depth = None if a.depth == "off" else models.MetricDepth(a.depth, device)
    labels = json.loads(Path(a.labels).read_text())
    if a.only:
        keys = a.only.split(",")
        labels = {k: v for k, v in labels.items() if any(s in k for s in keys)}
    rows, lat = ([], {}) if a.no_stills else run_stills(labels, device, depth,
                                                          Path(a.overlays) if a.overlays else None, max(2, a.reps))
    clips = {}
    if not a.no_clips:
        for p in a.clips or sorted(str(p) for p in (ROOT / "data/imagined").glob("*.mp4")):
            clips[Path(p).stem] = run_clip(Path(p), device, depth, a.clip_fx)
            print(f"[bench] {Path(p).stem}: {json.dumps(clips[Path(p).stem])}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps({"stills": rows, "latency_ms": lat, "clips": clips,
                                                 "device": device}, indent=1))
    md = markdown(rows, lat, clips)
    Path(a.out + ".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
