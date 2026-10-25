"""depth_shootout.json -> depth_shootout.md (tables + the hand-written recommendation) and a turbo contact sheet.

    python -m perception.bench_depth.report            # re-render from the last run without re-running models
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .. import depth_zoo as dz

HERE = Path(__file__).parent
RECOMMENDATION = HERE / "recommendation.md"


def _f(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{100 * x:.0f} %" if pct else f"{x:.{nd}f}"


def _rows(res, ds):
    return [r for r in res["rows"] if r["dataset"] == ds]


def _lic(variant: str) -> str:
    base = variant.split("+")[0].split(" ")[0]
    e = dz.REGISTRY.get(base)
    return e.licence if e else ""


def table_dense(res, ds, title) -> list[str]:
    rows = sorted(_rows(res, ds), key=lambda r: (r["absrel"] is None, r["absrel"] or 0))
    if not rows:
        return []
    md = [f"### {title}", "", "| Depth source | Frames | AbsRel | RMSE (m) | delta<1.25 | Corridor AbsRel (0-5 m) | "
          "Priority right | P1 recall | False P1 | Pitch error (deg) | Floor fit ok | ms/frame |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["absrel"] is None and r["priority_acc"] is None:
            continue
        md.append(f"| {r['variant']} | {r['frames']} | {_f(r['absrel'])} | {_f(r['rmse'], 2)} | {_f(r['d1'], 3)} | "
                  f"{_f(r['near_absrel'])} | {_f(r['priority_acc'], pct=True)} | {_f(r['p1_recall'], pct=True)} | "
                  f"{_f(r['false_p1'], pct=True)} | {_f(r['pitch_err'], 1)} | {_f(r['anchor_ok'], pct=True)} | "
                  f"{_f(r['latency_ms'], 0)} |")
    return md + [""]


def table_sequences(res) -> list[str]:
    seqs = sorted({r["dataset"][6:] for r in res["rows"] if r["dataset"].startswith("synth-")})
    variants = sorted({r["variant"] for r in res["rows"] if r["dataset"].startswith("synth-")})
    by = {(r["variant"], r["dataset"]): r for r in res["rows"]}
    md = ["### Priority band right, per synthetic walk", "",
          "| Depth source | " + " | ".join(seqs) + " |", "|---|" + "---|" * len(seqs)]
    for v in variants:
        cells = []
        for s in seqs:
            r = by.get((v, f"synth-{s}"))
            cells.append("-" if r is None else f"{_f(r['priority_acc'], pct=True)} ({_f(r['near_dist_err'], 2)} m)")
        md.append(f"| {v} | " + " | ".join(cells) + " |")
    md += ["", "In brackets: mean error of the nearest corridor obstacle's distance when it is within 5 m.", ""]
    return md


def table_latency(res) -> list[str]:
    mps = {}
    for r in res["rows"]:
        if r["dataset"] in ("synth", "diode-outdoor") and r["latency_ms"] is not None and "+" not in r["variant"] \
                and not r["variant"].startswith(("baseline", "fused")):
            mps.setdefault(r["variant"].split(" ")[0], r["latency_ms"])
    md = ["### Latency per 640x480 frame", "",
          f"Measured on {res.get('chip') or res.get('machine')} ({res['device']} for the models, numpy + OpenCV on the "
          "CPU for motion stereo and the wrappers), median over the run, first call excluded. CPU column: the same "
          "model forced onto the CPU.", "",
          "| Depth source | Licence | ms/frame (GPU) | ms/frame (CPU) | fps alone (GPU) |", "|---|---|---|---|---|"]
    names = sorted(set(mps) | set(res.get("cpu_latency_ms", {})))
    for n in names:
        g = mps.get(n)
        c = res.get("cpu_latency_ms", {}).get(n)
        md.append(f"| {n} | {_lic(n)} | {_f(g, 0)} | {c if isinstance(c, str) else _f(c, 0)} | "
                  f"{_f(1000 / g, 1) if g else '-'} |")
    return md + [""]


def write(json_path: Path, sheet_path: Path | None, out: Path) -> None:
    res = json.loads(json_path.read_text())
    md = ["# Depth shoot-out: which depth source should Kite run live?", ""]
    if RECOMMENDATION.exists():
        md += [RECOMMENDATION.read_text().strip(), ""]
    md += ["## Results", "",
           f"Ground truth: {res['frames']['diode']} DIODE validation frames ({res['frames']['diode_with_floor']} with "
           f"enough floor in view to define the walking corridor) and four synthetic chest-camera walks "
           f"({', '.join(f'{k} {v}' for k, v in res['frames']['synth'].items())} frames at 15 fps, 640x480, "
           "60 degree diagonal like a C270). Run time "
           f"{res['minutes']} min. Regenerate: `python -m perception.bench_depth` (see perception/README.md).", ""]
    if (out.parent / (out.name + ".png")).exists() or sheet_path:
        md += [f"![GT and each depth source, turbo colormap (near = red)]({out.name}.png)", ""]
    md += table_dense(res, "synth", "Synthetic walks (exact depth; motion, anchoring and fusion all apply)")
    md += table_sequences(res)
    md += table_dense(res, "diode-outdoor", "DIODE outdoor (laser-scanned, stills)")
    md += table_dense(res, "diode-indoors", "DIODE indoor (laser-scanned, stills)")
    md += table_latency(res)
    bad = {k: v for k, v in res.get("status", {}).items() if v != "ok"}
    if bad:
        md += ["### Not run", ""] + [f"- {k}: {v}" for k, v in bad.items()] + [""]
    md += ["### How to read the columns", "",
           "- **AbsRel**: mean |pred - true| / true over pixels with ground truth (sky and missing scans excluded). "
           "**RMSE** in metres. **delta<1.25**: share of pixels within 25 % of the truth.",
           "- **Corridor AbsRel**: the same over the walking corridor only (0.5 m either side, up to 5 m ahead).",
           "- **Priority right**: share of frames where the band of the nearest thing rising 0.25 to 2 m out of the "
           "floor in the corridor (P1 under 2 m, P2 2-5 m, P3 5-10 m, clear) matches the band the ground truth gives. "
           "Both use the true floor plane, so only the depth differs. **P1 recall**: true P1 frames called P1. "
           "**False P1**: P1 calls when the truth was P2 or farther.",
           "- **Pitch error**: anchored fits only, against the floor plane of the ground truth. **Floor fit ok**: "
           "frames where the RANSAC floor was accepted (otherwise the raw output is scored).",
           "- Box baselines use ground-truth boxes of every discrete object on the synthetic walks (a perfect "
           "detector) and YOLO11n COCO boxes on DIODE.", ""]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    if sheet_path and sheet_path.exists():
        contact_sheet(json.loads(sheet_path.read_text()), out.parent / (out.name + ".png"))


SHEET_COLS = ["da2-metric-outdoor-s+anchor", "da2-metric-indoor-s+anchor", "da2-relative-s+anchor (shift 0)",
              "da2-metric-outdoor-b+anchor", "unidepth-v2-s+anchor", "metric3d-v2-s+anchor", "depth-pro+anchor",
              "vda-metric-s+anchor", "motion-stereo", "fused: da2-metric-indoor-s+anchor + motion stereo"]
SHORT = {"da2-metric-outdoor-s+anchor": "DA2-S out", "da2-metric-indoor-s+anchor": "DA2-S in",
         "da2-relative-s+anchor (shift 0)": "DA2-S rel", "da2-relative-s+anchor": "DA2-S rel",
         "da2-metric-outdoor-b+anchor": "DA2-B out", "unidepth-v2-s+anchor": "UniDepth2-S",
         "metric3d-v2-s+anchor": "Metric3D2-S", "depth-pro+anchor": "Depth Pro", "vda-metric-s+anchor": "VDA-S",
         "motion-stereo": "motion stereo", "fused: da2-metric-outdoor-s+anchor + motion stereo": "fused",
         "fused: da2-metric-indoor-s+anchor + motion stereo": "fused"}


def contact_sheet(sheet: dict, path: Path) -> None:
    """RGB | GT | each source, one row per dataset; turbo on log depth (1 m red ... 30 m blue), shared per row."""
    import cv2
    rows = []
    maps = sheet.get("maps", {})
    for ds in ("synth", "diode-outdoor", "diode-indoors"):
        if ds not in sheet.get("gt", {}):
            continue
        tiles = [np.array(sheet["rgb"][ds], np.uint8), _turbo(np.array(sheet["gt"][ds], np.float32))]
        names = ["RGB", "ground truth"]
        m = maps.get(ds, {})
        for col in SHEET_COLS:
            key = col if col in m else col.replace(" (shift 0)", "")
            alt = col.replace("+anchor", "") if ds == "synth" and col.startswith("da2-relative") else None
            z = m.get(key) or (m.get(alt) if alt else None)
            if z is None and ds.startswith("diode"):   # run_diode keeps the anchored map under the bare model name
                z = m.get(col.replace(" (shift 0)", "").replace("+anchor", ""))
            if z is None:
                continue
            tiles.append(_turbo(np.array(z, np.float32), np.array(sheet["gt"][ds], np.float32) > 0))
            names.append(SHORT.get(col, col))
        row = []
        for t, n in zip(tiles, names):
            t = t.copy()
            cv2.rectangle(t, (0, 0), (160, 15), (20, 20, 20), -1)
            cv2.putText(t, n, (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1, cv2.LINE_AA)
            row.append(t)
        rows.append(row)
    if not rows:
        return
    width = max(len(r) for r in rows)
    blank = np.full((120, 160, 3), 24, np.uint8)
    img = np.vstack([np.hstack(r + [blank] * (width - len(r))) for r in rows])
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def _turbo(z: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    import cv2
    v = np.clip(1 - np.log(np.clip(z, 1.0, 30.0)) / math.log(30.0), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[z <= 0] = 0
    return img


if __name__ == "__main__":
    from ..depth_zoo import ROOT
    base = ROOT / "perception" / "docs" / "depth_shootout"
    write(base.with_suffix(".json"), ROOT / "data" / "depth_bench" / "sheet.json", base)
