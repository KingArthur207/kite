"""How good are the priorities? Synthetic renders with exact geometry, then real (generated and simulated) frames.

    python -m perception.near.evaluate --synthetic          # no models: per-cue error, confusion, sequences
    python -m perception.near.evaluate                      # + stills and clips with YOLO11n + Depth Anything V2
    python -m perception.near.evaluate --json perception/near/docs/eval.json --md perception/near/docs/eval.md

Variants compared on the same inputs (the "is the combination better" question):
    fused            every cue + looming + occupancy (what ships)
    depth raw        Depth Anything V2 Metric alone, at the model's own scale
    depth anchored   the same depth map, scale and pitch fitted to the ground (RANSAC)
    size only        known height / width priors
    ground only      flat-ground contact with the pitch prior (0 deg)
    scrollsafe       the rule of ScrollSafe (github.com/MrAnitude/ScrollSafe, no licence; reimplemented here from
                     its description, no code copied): an object is "in the walk path" when its box centre is in
                     the middle 50 % of the frame width; box height / frame height > 0.40 is DANGER (-> P1), > 0.20
                     is WARNING (-> P2), anything smaller in the path -> P3. It has no intrinsics, no corridor in
                     metres and no memory across frames.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from . import synth
from .depth import ArrayBackend, DepthProcessor
from .intrinsics import BUILTIN, CameraProfile
from .priority import raw_priority
from .priors import PRIORS
from .system import ALL_CUES, NearSystem

ROOT = Path(__file__).resolve().parents[2]
LABELS = Path(__file__).with_name("near_labels.json")
DATA = ROOT / "data" / "imagined"
MAIN_DATA = ROOT.parent.parent.parent / "data" / "imagined"     # a worktree's main checkout, if this is one
PRI = (1, 2, 3, None)

VARIANTS = {
    "fused": dict(use=ALL_CUES, occupancy_on=True, anchor=True, depth=True),
    "depth raw": dict(use=("depth",), occupancy_on=False, anchor=False, depth=True),
    "depth anchored": dict(use=("depth",), occupancy_on=False, anchor=True, depth=True),
    "size only": dict(use=("size_h", "size_w"), occupancy_on=False, anchor=False, depth=False),
    "ground only": dict(use=("ground",), occupancy_on=False, anchor=False, depth=False),
}


# ------------------------------------------------------------------ baseline
def scrollsafe(dets, w: int, h: int) -> list[int | None]:
    out = []
    for d in dets:
        cx = (d.x1 + d.x2) / 2
        if not 0.25 * w <= cx <= 0.75 * w:
            out.append(None)
            continue
        r = (d.y2 - d.y1) / h
        out.append(1 if r > 0.40 else 2 if r > 0.20 else 3)
    return out


def top_of(ps: list[int | None]) -> int | None:
    v = [p for p in ps if p is not None]
    return min(v) if v else None


# ------------------------------------------------------------------ scoring
class Score:
    def __init__(self):
        self.conf = {t: Counter() for t in PRI}      # single-answer labels: truth -> predicted counts
        self.ok = self.n = 0
        self.danger_miss = self.false_p1 = 0

    def add(self, pred, acceptable) -> None:
        acceptable = list(acceptable)
        self.n += 1
        self.ok += pred in acceptable
        if len(acceptable) == 1:
            self.conf[acceptable[0]][pred] += 1
        if 1 in acceptable and len(acceptable) == 1 and pred in (3, None):
            self.danger_miss += 1
        if pred == 1 and 1 not in acceptable:
            self.false_p1 += 1

    def as_dict(self) -> dict:
        return {"acc": round(self.ok / max(1, self.n), 3), "n": self.n, "danger_miss": self.danger_miss,
                "false_p1": self.false_p1,
                "confusion": {str(t): {str(p): self.conf[t][p] for p in PRI} for t in PRI}}


def _err_stats(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0}
    a = np.abs(np.array(errs))
    return {"n": len(errs), "median_pct": round(100 * (math.exp(float(np.median(a))) - 1), 1),
            "p90_pct": round(100 * (math.exp(float(np.percentile(a, 90))) - 1), 1),
            "bias_pct": round(100 * (math.exp(float(np.median(errs))) - 1), 1)}


# ------------------------------------------------------------------ synthetic stills
KIND_MIX = [("person", 0.40), ("pole", 0.15), ("vehicle", 0.15), ("trash_can", 0.10), ("chair", 0.10),
            ("bike", 0.10)]


def random_scene(rng: np.random.Generator):
    kinds, probs = zip(*KIND_MIX)
    kind = str(rng.choice(kinds, p=np.array(probs) / sum(probs)))
    pr = PRIORS[kind]
    F = float(math.exp(rng.uniform(math.log(0.7), math.log(14.0))))
    lat = float(rng.uniform(-1.6, 1.6))
    H = float(pr.h * math.exp(rng.normal(0, 0.8 * pr.sh)))
    W = float(pr.w * math.exp(rng.normal(0, 0.5 * pr.sw)))
    return synth.Obj(kind, F, lat, H, W), float(rng.uniform(0.0, 12.0)), float(rng.normal(1.2, 0.05)), \
        float(math.exp(rng.uniform(math.log(0.6), math.log(1.6))))


def truth_priority(o: synth.Obj, half: float = 0.5) -> int | None:
    if o.lateral + o.w / 2 < -half or o.lateral - o.w / 2 > half:
        return None
    return raw_priority(o.forward, PRIORS[o.kind].size_scale())


def synthetic_stills(n: int = 600, seed: int = 0, prof: CameraProfile | None = None) -> dict:
    prof = prof or BUILTIN["c270"]
    rng = np.random.default_rng(seed)
    scores = {k: Score() for k in [*VARIANTS, "scrollsafe"]}
    cue_err: dict[str, list[float]] = {}
    cue_seen: Counter = Counter()
    for _ in range(n):
        o, pitch, cam_h, dscale = random_scene(rng)
        r = synth.render(prof, [o], cam_h, pitch, rng, depth_scale=dscale)
        truth = truth_priority(o)
        if not r.dets:
            continue
        holder = r.depth
        for name, v in VARIANTS.items():
            dp = DepthProcessor(ArrayBackend(lambda f, d=holder: d), 1.2, 0.0, True, 0.0, v["anchor"]) \
                if v["depth"] else None
            ns = NearSystem(prof, depth=dp, cam_h=1.2, pitch=0.0, still=True, use=v["use"],
                            occupancy_on=v["occupancy_on"])
            nf = ns.step(synth.image_for(prof), 0.0, r.dets)
            obj = nf.objects[0]
            scores[name].add(obj["priority"], [truth])
            if name == "fused":
                cue_seen["frames"] += 1
                for k, d in obj["cues"].items():
                    cue_err.setdefault(k, []).append(math.log(d / o.forward))
                    cue_seen[k] += 1
                if obj["distance_m"]:
                    cue_err.setdefault("fused", []).append(math.log(obj["distance_m"] / o.forward))
        scores["scrollsafe"].add(scrollsafe(r.dets, prof.width, prof.height)[0], [truth])
    return {"camera": prof.name, "n": cue_seen["frames"],
            "variants": {k: s.as_dict() for k, s in scores.items()},
            "cues": {k: {**_err_stats(v), "coverage": round(len(v) / max(1, cue_seen["frames"]), 2)}
                     for k, v in sorted(cue_err.items())}}


# ------------------------------------------------------------------ synthetic sequences
def scenario(name: str):
    """-> (objects per frame t -> Obj, walker speed, duration). Positions relative to the walker."""
    if name == "walk to a pole":
        return (lambda t: synth.Obj("pole", 10.0 - 1.2 * t, 0.15, 3.2, 0.15)), 1.2, 7.8
    if name == "oncoming person":
        return (lambda t: synth.Obj("person", 14.0 - 2.6 * t, 0.10, 1.68, 0.48)), 1.2, 5.0
    if name == "fast cyclist head-on":
        return (lambda t: synth.Obj("bike", 20.0 - 6.5 * t, 0.20, 1.05, 0.6)), 1.0, 2.85
    if name == "person crossing at 3.5 m":
        return (lambda t: synth.Obj("person", 3.5, -3.0 + 1.4 * t, 1.72, 0.5)), 0.0, 4.3
    if name == "person standing at 4.6 m":
        return (lambda t: synth.Obj("person", 4.6, 0.0, 1.70, 0.5)), 0.0, 8.0
    if name == "pole standing at 1.8 m":
        return (lambda t: synth.Obj("pole", 1.8, -0.2, 3.0, 0.15)), 0.0, 8.0
    raise KeyError(name)


SCENARIOS = ("walk to a pole", "oncoming person", "fast cyclist head-on", "person crossing at 3.5 m",
             "person standing at 4.6 m", "pole standing at 1.8 m")


def run_sequence(name: str, variant: str, seed: int = 0, hz: float = 10.0, prof: CameraProfile | None = None
                 ) -> dict:
    prof = prof or BUILTIN["c270"]
    fn, v_ego, dur = scenario(name)
    rng = np.random.default_rng(seed)
    pitch = 8.0
    dscale = 1.3
    holder = {"d": None}
    use = ALL_CUES if variant != "no looming" else tuple(c for c in ALL_CUES if c not in ("loom", "motion"))
    kw = dict(n_up=1, m_down=1) if variant == "no hysteresis" else {}
    dp = DepthProcessor(ArrayBackend(lambda f: holder["d"]), 1.2, 0.0, True)
    ns = NearSystem(prof, depth=dp, cam_h=1.2, pitch=0.0, v_ego=v_ego, use=use, **kw)
    preds, truths, t_list, ttc_err = [], [], [], []
    t = 0.0
    while t <= dur + 1e-9:
        o = fn(t)
        r = synth.render(prof, [o], 1.2, pitch, rng, depth_scale=dscale * math.exp(rng.normal(0, 0.03)))
        holder["d"] = r.depth
        for d in r.dets:
            d.tid = "a"
        tp = truth_priority(o)
        closing = (fn(t).forward - fn(t + 0.1).forward) / 0.1
        true_ttc = o.forward / closing if closing > 0.05 else None
        if variant == "scrollsafe":
            p = top_of(scrollsafe(r.dets, prof.width, prof.height))
            ttc = None
        else:
            nf = ns.step(synth.image_for(prof), t, r.dets)
            obj = next((x for x in nf.objects if not x["id"].startswith("occ:")), None)
            p = obj["priority"] if obj else None
            ttc = obj["ttc_s"] if obj else None
            if true_ttc is not None and true_ttc < 4.0 and ttc is not None:
                ttc_err.append(math.log(ttc / true_ttc))
        if tp is not None and true_ttc is not None and true_ttc < 1.5:
            tp = 1                                     # the override is part of the spec: P1 under 1.5 s
        preds.append(p)
        truths.append(tp)
        t_list.append(t)
        t += 1.0 / hz
    acc = float(np.mean([p == q for p, q in zip(preds, truths)]))
    onset = lambda xs: next((tt for tt, x in zip(t_list, xs) if x == 1), None)
    t_true, t_pred = onset(truths), onset(preds)
    first = next((i for i, x in enumerate(preds) if x is not None), len(preds))
    flick = sum(1 for a, b in zip(preds[first:], preds[first + 1:]) if a != b)     # after the first report
    tflick = sum(1 for a, b in zip(truths[first:], truths[first + 1:]) if a != b)
    impact = None
    if t_pred is not None:
        o = fn(t_pred)
        closing = (o.forward - fn(t_pred + 0.1).forward) / 0.1
        impact = round(o.forward / closing, 2) if closing > 0.05 else None
    return {"acc": round(acc, 3), "p1_true_s": t_true, "p1_pred_s": None if t_pred is None else round(t_pred, 2),
            "p1_delay_s": None if (t_true is None or t_pred is None) else round(t_pred - t_true, 2),
            "time_to_impact_at_p1_s": impact, "changes": flick, "true_changes": tflick,
            "ttc_err": _err_stats(ttc_err)}


def synthetic_sequences(seeds: int = 5) -> dict:
    out = {}
    for name in SCENARIOS:
        out[name] = {}
        for variant in ("fused", "no looming", "no hysteresis", "scrollsafe"):
            runs = [run_sequence(name, variant, s) for s in range(seeds)]
            agg = {"acc": round(float(np.mean([r["acc"] for r in runs])), 3),
                   "changes": round(float(np.mean([r["changes"] for r in runs])), 1),
                   "true_changes": runs[0]["true_changes"], "p1_true_s": runs[0]["p1_true_s"]}
            d = [r["p1_delay_s"] for r in runs if r["p1_delay_s"] is not None]
            agg["p1_delay_s"] = round(float(np.median(d)), 2) if d else None
            imp = [r["time_to_impact_at_p1_s"] for r in runs if r["time_to_impact_at_p1_s"] is not None]
            agg["time_to_impact_at_p1_s"] = round(float(np.median(imp)), 2) if imp else None
            agg["p1_missed_runs"] = sum(1 for r in runs if r["p1_true_s"] is not None and r["p1_pred_s"] is None)
            tt = [r["ttc_err"]["median_pct"] for r in runs if r["ttc_err"].get("n")]
            agg["ttc_median_err_pct"] = round(float(np.median(tt)), 1) if tt else None
            out[name][variant] = agg
    return out


def distortion_check(n: int = 300, seed: int = 3, cam: str = "c270") -> dict:
    """A lens with real barrel distortion, ranged with the calibrated profile vs the spec one (no distortion)."""
    from dataclasses import replace
    true = replace(BUILTIN[cam], dist=[-0.12, 0.05, 0.0, 0.0, 0.0], source="calibrated: synthetic")
    rng = np.random.default_rng(seed)
    errs = {"calibrated (undistorts)": [], "spec (ignores distortion)": []}
    for _ in range(n):
        o, pitch, cam_h, _ = random_scene(rng)
        o.lateral = float(rng.uniform(-3.0, 3.0))      # distortion matters off-axis
        r = synth.render(true, [o], 1.2, pitch, rng, depth=False, jitter=0.0, jitter_px=0.0)
        if not r.dets:
            continue
        d = r.dets[0]
        if d.x1 < 12 or d.y1 < 12 or d.x2 > true.width - 12 or d.y2 > true.height - 12 or o.forward < 3:
            continue                                     # whole object in view: the ground cue's pure geometry
        for name, prof in (("calibrated (undistorts)", true), ("spec (ignores distortion)", BUILTIN[cam])):
            ns = NearSystem(prof, cam_h=cam_h, pitch=pitch, pitch_sigma=0.5, still=True, use=("ground",))
            obj = ns.step(None, 0.0, r.dets, (true.width, true.height)).objects[0]
            if obj["distance_m"]:
                errs[name].append(math.log(obj["distance_m"] / o.forward))
    return {f"{true.name}, {k}": _err_stats(v) for k, v in errs.items()}


# ------------------------------------------------------------------ real frames
def _profile(kind: str, w: int, h: int, labels: dict) -> tuple[CameraProfile, dict]:
    p = labels["profiles"][kind]
    f = p["fx_at_960"] * w / 960
    return CameraProfile(kind, w, h, f, f, w / 2, h / 2, [0.0] * 5, f"evaluation: fx {p['fx_at_960']} at 960"), p


def _load(path: Path):
    import cv2
    img = cv2.imread(str(path))
    return cv2.resize(img, (960, round(img.shape[0] * 960 / img.shape[1])))


def _variant_system(v: dict, prof, raw, p: dict, still: bool, v_ego: float = 1.2):
    dp = DepthProcessor(ArrayBackend(lambda f: raw() if callable(raw) else raw), p["cam_h"], p["pitch"],
                        p["fit_pitch"], 0.0 if still else 0.6, v["anchor"]) if v["depth"] else None
    return NearSystem(prof, depth=dp, cam_h=p["cam_h"], pitch=p["pitch"], v_ego=v_ego, still=still, use=v["use"],
                      occupancy_on=v["occupancy_on"], ignore_bottom=p.get("ignore_bottom", 0.0))


def _match(dets, objs, at, kind, prof=None) -> int | None:
    """The priority the system gave the labelled object: the smallest box containing the label's point, else a
    class-agnostic corridor hit in the same direction (within 8 degrees; a curb label takes a drop anywhere ahead)."""
    hits = [i for i, d in enumerate(dets) if d.x1 <= at[0] <= d.x2 and d.y1 <= at[1] <= d.y2]
    if hits:
        i = min(hits, key=lambda i: (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1))
        return objs[i]["priority"]
    occ = [o for o in objs if o["id"].startswith("occ:") and o["priority"] is not None]
    if kind == "curb":
        occ = [o for o in occ if o["kind"] in ("curb", "stairs_down")]
    elif prof is not None:
        b = math.degrees(math.atan((at[0] - prof.cx) / prof.fx))
        occ = [o for o in occ if o["kind"] not in ("curb", "stairs_down") and abs(o["bearing"] - b) <= 8.0]
    return top_of([o["priority"] for o in occ]) if occ else None


def real_stills(det, depth_model, labels: dict, root: Path) -> dict:
    scores = {k: Score() for k in [*VARIANTS, "scrollsafe"]}
    rows = []
    for rel, spec in labels["stills"].items():
        path = root / rel
        if not path.is_file():
            continue
        img = _load(path)
        h, w = img.shape[:2]
        prof, p = _profile(spec["profile"], w, h, labels)
        dets = det(img)
        raw = depth_model(img) if depth_model is not None else None
        ss = scrollsafe(dets, w, h)
        per = {}
        for name, v in VARIANTS.items():
            if v["depth"] and raw is None:
                continue
            ns = _variant_system(v, prof, raw, p, True)
            nf = ns.step(img, 0.0, dets)
            per[name] = nf.objects
        for it in spec["items"]:
            for name, objs in per.items():
                pred = _match(dets, objs, it["at"], it["kind"], prof)
                scores[name].add(pred, it["p"])
            hits = [i for i, d in enumerate(dets) if d.x1 <= it["at"][0] <= d.x2 and d.y1 <= it["at"][1] <= d.y2]
            ssp = ss[min(hits, key=lambda i: (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1))] if hits else None
            scores["scrollsafe"].add(ssp, it["p"])
            fo = per.get("fused", [])
            fobj = None
            if hits and fo:
                fobj = fo[min(hits, key=lambda i: (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1))]
            rows.append({"image": rel.split("/")[-1], "kind": it["kind"], "label": it["p"],
                         "fused": _match(dets, fo, it["at"], it["kind"], prof) if fo else None,
                         "distance_m": fobj["distance_m"] if fobj else None, "cues": fobj["cues"] if fobj else {},
                         "scrollsafe": ssp})
    return {"variants": {k: s.as_dict() for k, s in scores.items()}, "rows": rows}


def real_clips(make_det, depth_model, labels: dict, data: Path, hz: float = 12.0) -> dict:
    import cv2
    scores = {k: Score() for k in [*VARIANTS, "scrollsafe"]}
    per_clip = {}
    for name, spec in labels["clips"].items():
        path = data / f"{name}.mp4"
        if not path.is_file():
            continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        step = max(1, round(fps / hz))
        det = make_det()
        frames, i = [], 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                img = cv2.resize(img, (960, round(img.shape[0] * 960 / img.shape[1])))
                frames.append((i / fps, img, det(img), depth_model(img) if depth_model is not None else None))
            i += 1
        cap.release()
        h, w = frames[0][1].shape[:2]
        prof, p = _profile(spec["profile"], w, h, labels)
        clip = {}
        for vname, v in [*VARIANTS.items(), ("scrollsafe", None)]:
            if v is not None and v["depth"] and frames[0][3] is None:
                continue
            cur = {"raw": None}
            ns = _variant_system(v, prof, lambda c=cur: c["raw"], p, False, spec["walk"]) if v else None
            preds, oks = [], []
            for t, img, dets, raw in frames:
                acceptable = next((a for t0, t1, a in spec["p"] if t0 <= t < t1), None)
                if v is None:
                    pred = top_of(scrollsafe(dets, w, h))
                else:
                    cur["raw"] = raw
                    nf = ns.step(img, t, dets)
                    pred = nf.top["priority"] if nf.top else None
                preds.append((t, pred))
                if acceptable is not None:
                    scores[vname].add(pred, acceptable)
                    oks.append(pred in acceptable)
            onset_label = next((t0 for t0, t1, a in spec["p"] if a == [1]), None)
            onset = next((t for t, q in preds if q == 1), None)
            clip[vname] = {"acc": round(float(np.mean(oks)), 3) if oks else None,
                           "changes": sum(1 for a, b in zip(preds, preds[1:]) if a[1] != b[1]),
                           "p1_onset_s": None if onset is None else round(onset, 2), "p1_label_s": onset_label,
                           "timeline": "".join("-" if q is None else str(q) for _, q in preds)}
        per_clip[name] = clip
    return {"variants": {k: s.as_dict() for k, s in scores.items()}, "clips": per_clip}


def real(device: str = "cpu") -> dict:
    from .. import models
    labels = json.loads(LABELS.read_text())
    data = DATA if any(DATA.glob("*.mp4")) else MAIN_DATA
    root = ROOT if (ROOT / "data/imagined/doorway.jpg").is_file() else MAIN_DATA.parents[1]
    depth_model = models.MetricDepth("outdoor", device)
    still_det = models.HybridDetector(models.CocoDetector(device, 0.35, 640, track=False), None)
    stills = real_stills(still_det, depth_model, labels, root)
    clips = real_clips(lambda: models.HybridDetector(models.CocoDetector(device, 0.35, 640, track=True), None),
                       depth_model, labels, data)
    return {"stills": stills, "clips": clips}


# ------------------------------------------------------------------ report
def confusion_md(c: dict) -> str:
    head = "| truth \\ predicted | P1 | P2 | P3 | none |\n|---|---|---|---|---|\n"
    rows = []
    for t in ("1", "2", "3", "None"):
        r = c[t]
        if sum(r.values()) == 0:
            continue
        rows.append(f"| {'P' + t if t != 'None' else 'none'} | " + " | ".join(str(r[p]) for p in ("1", "2", "3",
                                                                                                 "None")) + " |")
    return head + "\n".join(rows)


def variants_md(v: dict) -> str:
    out = "| variant | accuracy | n | P1 missed (said P3/none) | false P1 |\n|---|---|---|---|---|\n"
    for k, s in v.items():
        out += f"| {k} | {100 * s['acc']:.1f} % | {s['n']} | {s['danger_miss']} | {s['false_p1']} |\n"
    return out


def markdown(res: dict) -> str:
    md = []
    s = res.get("synthetic_stills")
    if s:
        md.append(f"### Synthetic stills ({s['n']} renders, {s['camera']} spec intrinsics)\n")
        md.append(variants_md(s["variants"]))
        md.append("\nPer-cue error against the true distance (median and 90th percentile of |error|, bias = median "
                  "signed error, coverage = share of renders where the cue applied):\n")
        md.append("| cue | coverage | median | p90 | bias |\n|---|---|---|---|---|")
        for k, c in s["cues"].items():
            md.append(f"| {k} | {100 * c.get('coverage', 0):.0f} % | {c['median_pct']} % | {c['p90_pct']} % | "
                      f"{c['bias_pct']:+} % |")
        md.append("\nConfusion, fused:\n\n" + confusion_md(s["variants"]["fused"]["confusion"]))
        md.append("\nConfusion, ScrollSafe rule:\n\n" + confusion_md(s["variants"]["scrollsafe"]["confusion"]))
    q = res.get("synthetic_sequences")
    if q:
        md.append("\n### Synthetic sequences (10 Hz, 5 seeds each)\n")
        md.append("| scenario | variant | frame accuracy | P1 onset vs truth | time to impact at P1 | priority "
                  "changes (truth) | TTC error |\n|---|---|---|---|---|---|---|")
        for name, vs in q.items():
            for vname, a in vs.items():
                delay = "-" if a["p1_delay_s"] is None else f"{a['p1_delay_s']:+.1f} s"
                if a["p1_true_s"] is not None and a["p1_missed_runs"]:
                    delay += f" ({a['p1_missed_runs']}/5 missed)"
                imp = "-" if a["time_to_impact_at_p1_s"] is None else f"{a['time_to_impact_at_p1_s']:.1f} s"
                tt = "-" if a["ttc_median_err_pct"] is None else f"{a['ttc_median_err_pct']} %"
                md.append(f"| {name} | {vname} | {100 * a['acc']:.0f} % | {delay} | {imp} | {a['changes']} "
                          f"({a['true_changes']}) | {tt} |")
    d = res.get("distortion")
    if d:
        md.append("\n### Lens distortion (k1 -0.12, k2 0.05; ground cue, whole object in view, 3 m and beyond)\n")
        md.append("| intrinsics used | median error | p90 |\n|---|---|---|")
        for k, c in d.items():
            md.append(f"| {k} | {c['median_pct']} % | {c['p90_pct']} % |")
    r = res.get("real")
    if r:
        md.append("\n### Real stills (hand labels, perception/near/near_labels.json)\n")
        md.append(variants_md(r["stills"]["variants"]))
        md.append("\nConfusion, fused (labels with one acceptable answer):\n\n"
                  + confusion_md(r["stills"]["variants"]["fused"]["confusion"]))
        md.append("\n| image | object | label | fused | distance | cues | ScrollSafe |\n|---|---|---|---|---|---|---|")
        for row in r["stills"]["rows"]:
            lab = "/".join("none" if x is None else f"P{x}" for x in row["label"])
            f = "none" if row["fused"] is None else f"P{row['fused']}"
            sp = "none" if row["scrollsafe"] is None else f"P{row['scrollsafe']}"
            cues = ", ".join(f"{k} {v}" for k, v in row["cues"].items())
            dist = "-" if row["distance_m"] is None else f"{row['distance_m']} m"
            md.append(f"| {row['image']} | {row['kind']} | {lab} | {f} | {dist} | {cues} | {sp} |")
        md.append("\n### Real clips (12 Hz; most urgent priority per frame vs the key object's label)\n")
        md.append(variants_md(r["clips"]["variants"]))
        md.append("\n| clip | variant | frame accuracy | P1 onset (label) | changes | timeline (- = clear) "
                  "|\n|---|---|---|---|---|---|")
        for clip, vs in r["clips"]["clips"].items():
            for vname, a in vs.items():
                on = "-" if a["p1_onset_s"] is None else f"{a['p1_onset_s']} s"
                lab = "-" if a["p1_label_s"] is None else f"{a['p1_label_s']} s"
                acc = "-" if a["acc"] is None else f"{100 * a['acc']:.0f} %"
                md.append(f"| {clip} | {vname} | {acc} | {on} ({lab}) | {a['changes']} | `{a['timeline']}` |")
    return "\n".join(md) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m perception.near.evaluate", description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true", help="synthetic only (no models)")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    from . import priority
    priority.P1_M, priority.P2_M = json.loads(LABELS.read_text()).get("bands", [priority.P1_M, priority.P2_M])
    print(f"[evaluate] scoring at the labels' bands: P1 under {priority.P1_M:g} m, P2 under {priority.P2_M:g} m",
          flush=True)
    res = {"synthetic_stills": synthetic_stills(a.n), "synthetic_sequences": synthetic_sequences(),
           "distortion": {**distortion_check(cam="c270"), **distortion_check(cam="c920")}}
    if not a.synthetic:
        res["real"] = real(a.device)
    text = markdown(res)
    print(text)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(text)


if __name__ == "__main__":
    main()
