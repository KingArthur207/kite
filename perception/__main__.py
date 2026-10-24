"""Open-vocabulary street detector for Kite: same events as sim/standin_detect.py, more things named.

    python -m perception --download                                   # once, with internet: YOLOE, prompts, depth
    python -m perception --source data/imagined/scooter.mp4 --jsonl out/s.jsonl --overlay out/s.mp4 --no-send
    python -m perception --source sim:chest                           # the browser sim's chest camera (relay up)
    python -m perception --source webcam:iPhone --camera chest        # live chest camera (kite.sources)
    python -m perception --source photo.jpg --overlay out/photo.jpg --no-send   # one still
    python -m perception --prompt "construction=orange fence,jersey barrier" ...   # add prompts
    python -m perception --vocab my_vocab.json ...                    # replace the vocabulary

chest camera -> `obstacles` every frame (~5-8 Hz on an M1) + `heartbeat` when empty
drone camera -> `hazard` (crowd, vehicle, construction, overhead, curb, stairs_down, obstacle) + `scene` every 10 s
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import vocab as V

DEFAULT_FX, LIVE_FX = 1000.0, 700.0            # same defaults as the stand-in
DEFAULT_HEIGHT = {"chest": 1.2, "drone": 2.0}
STILL = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m perception", description=__doc__.split("\n")[0])
    ap.add_argument("--download", action="store_true", help="fetch weights into weights/ and exit")
    ap.add_argument("--source", help="video file, still image, sim:chest|sim:drone, webcam:<index or name>, "
                                     "streetview:..., tello (kite.sources specs)")
    ap.add_argument("--camera", choices=("chest", "drone", "both"), default="chest")
    ap.add_argument("--open-model", default=None, help="open-vocabulary weights in weights/ "
                                                       "(default yolov8s-worldv2.pt; a yoloe-*.pt also works)")
    ap.add_argument("--open-every", type=int, default=1, help="run the open-vocabulary model every n-th frame")
    ap.add_argument("--no-open", action="store_true", help="COCO model only (the stand-in's classes)")
    ap.add_argument("--vocab", help="JSON {kind: [prompt, ...]} replacing the default vocabulary")
    ap.add_argument("--prompt", action="append", default=[], help="add prompts: kind=prompt[,prompt] (repeatable)")
    ap.add_argument("--print-vocab", action="store_true", help="print the vocabulary as JSON and exit")
    ap.add_argument("--conf", type=float, default=0.2, help="minimum open-vocabulary score (some kinds: see "
                                                            "models.OPEN_MIN_CONF)")
    ap.add_argument("--coco-conf", type=float, default=0.35, help="minimum COCO score (the stand-in's 0.35)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--depth", choices=("outdoor", "indoor", "off"), default="outdoor",
                    help="metric depth model (Depth Anything V2 Small); off = flat-ground ranges only")
    ap.add_argument("--auto-pitch", choices=("on", "off"), help="fit the camera pitch to the ground in the depth "
                                                             "map every frame (default on; off for sim:, whose "
                                                             "cameras are level). Off still fits the depth scale")
    ap.add_argument("--seg", action="store_true", help="drone: add walkable-ground `scene` hints from "
                                                       "sim/seg_standin.py (YOLO26n-sem, ADE20K)")
    ap.add_argument("--fx", type=float, help=f"focal length in px at 960 wide (default {DEFAULT_FX:.0f}, "
                                              f"{LIVE_FX:.0f} for a live camera or a still)")
    ap.add_argument("--cam-height", type=float, help="camera height (m); sim uses its pose")
    ap.add_argument("--pitch", type=float, default=0.0, help="camera tilt in degrees, + = down")
    ap.add_argument("--ignore-bottom", type=float, help="fraction of rows at the bottom that show the wearer, left "
                                                         "out of the depth corridor check (default 0.1 for sim:, "
                                                         "else 0)")
    ap.add_argument("--stop-m", type=float, default=None, help="chest: critical inside this range (default 1.6)")
    ap.add_argument("--device", default=None, help="mps / cpu (default: mps if available)")
    ap.add_argument("--jsonl", help="also write events with a t field (scripts/replay.py)")
    ap.add_argument("--overlay", help="write an mp4 (or .jpg for a still) with boxes and labels")
    ap.add_argument("--no-send", action="store_true", help="do not send to the earpiece")
    ap.add_argument("--ear-port", type=int, default=int(os.environ.get("KITE_EARPIECE_PORT", "5010")),
                    help="earpiece UDP port (default $KITE_EARPIECE_PORT or 5010)")
    ap.add_argument("--fast", action="store_true", help="video: do not pace to real time")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this long (0 = forever)")
    return ap.parse_args(argv)


def build(a, cams):
    import torch

    from . import models
    from .pipeline import Pipeline
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    vocab = V.load(a.vocab, a.prompt)
    depth = None if a.depth == "off" else models.MetricDepth(a.depth, device)
    pipes = {}
    for c in cams:                          # one detector pair per camera: each keeps its own tracks
        coco = models.CocoDetector(device, a.coco_conf, a.imgsz)
        ov = None if a.no_open else models.OpenVocabDetector(vocab, device, a.conf, a.imgsz,
                                                             weights=a.open_model or models.OPEN_MODEL)
        det = models.HybridDetector(coco, ov, a.open_every)
        kw = {} if a.stop_m is None else {"stop_m": a.stop_m}
        pipes[c] = Pipeline(c, det, depth, cam_h=a.cam_height or DEFAULT_HEIGHT[c], pitch=a.pitch,
                            auto_pitch=a.auto_pitch == "on", ignore_bottom=a.ignore_bottom or 0.0, **kw)
    return pipes, device


def main(argv=None) -> None:
    a = parse_args(argv)
    if a.print_vocab:
        print(json.dumps(V.load(a.vocab, a.prompt), indent=2))
        return
    if a.download:
        from . import models
        models.download(V.load(a.vocab, a.prompt), a.open_model or models.OPEN_MODEL)
        if a.seg:
            import seg_standin
            seg_standin.download()
        return
    if not a.source:
        sys.exit("--source is required (a file, sim:chest, webcam:<name>, ...)")
    import cv2
    import numpy as np

    from .overlay import draw
    still = a.source.lower().endswith(STILL)
    kind = "still" if still else None
    if not still:
        from kite.sources import parse_spec
        try:
            kind = parse_spec(a.source)[0]
        except ValueError as e:
            sys.exit(str(e))
    live = kind in ("webcam", "tello")
    if a.overlay and kind in ("webcam", "tello", "streetview"):
        sys.exit("--overlay is off for live cameras and Street View: frames are never stored (spec 5.3)")
    cams = ("chest", "drone") if a.camera == "both" else (a.camera,)
    if kind == "sim":
        cams = (a.source.split(":", 1)[1],)
    fx = a.fx or (LIVE_FX if live or still else DEFAULT_FX)
    a.auto_pitch = a.auto_pitch or ("off" if kind == "sim" else "on")
    if a.ignore_bottom is None:
        a.ignore_bottom = 0.1 if kind == "sim" else 0.0
    pipes, device = build(a, cams)
    seg = None
    if a.seg:                               # sim/seg_standin.py, read-only: YOLO26n-sem (ADE20K) walkable ground
        import seg_standin
        seg = seg_standin.Segmenter(device=device)
    for p in pipes.values():
        p.fx = fx
    ear = None
    if not a.no_send:
        from earpiece.client import EarpieceClient
        ear = EarpieceClient(host=os.environ.get("KITE_EARPIECE_HOST", "127.0.0.1"),
                             port=a.ear_port, source="perception")
    jsonl = open(a.jsonl, "w") if a.jsonl else None

    def emit(ev: dict, t: float) -> None:
        if ear:
            ear.send(dict(ev))
        if jsonl:
            jsonl.write(json.dumps({"t": round(t, 3), **ev}, separators=(",", ":")) + "\n")

    src = cap = None
    if still:
        img0 = cv2.imread(a.source)
        if img0 is None:
            sys.exit(f"cannot read {a.source}")
    elif kind == "file":
        cap = cv2.VideoCapture(a.source)
        if not cap.isOpened():
            sys.exit(f"cannot open {a.source}")
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step, fidx = max(1, round(vfps / 10)), 0
    else:
        from kite.sources import open_source
        try:
            src = open_source(a.source)
        except (LookupError, RuntimeError, OSError, ValueError) as e:
            sys.exit(f"cannot open {a.source}: {e}")
    for p in pipes.values():
        p.detector.warm()
    writer, n, stage, was_stop = None, 0, {}, {}
    t0 = time.monotonic()
    print(f"[perception] cams={cams} device={device} depth={a.depth} fx={fx:.0f} source={a.source}", flush=True)
    try:
        while True:
            if a.max_seconds and time.monotonic() - t0 > a.max_seconds:
                break
            pose = {}
            if still:
                if n:
                    break
                img, t = img0, 0.0
            elif cap is not None:
                want = fidx + step - 1
                if not a.fast:
                    want = max(want, int((time.monotonic() - t0) * vfps))
                while fidx < want and cap.grab():
                    fidx += 1
                ok, img = cap.read()
                if not ok:
                    break
                t = fidx / vfps
                fidx += 1
                if not a.fast and (lag := t - (time.monotonic() - t0)) > 0:
                    time.sleep(lag)
            else:
                ok, img = src.read()
                if not ok:
                    if kind in ("file", "recording", "streetview"):
                        break
                    continue
                pose = dict(getattr(src, "pose", None) or {})
                t = time.monotonic() - t0
            if img.shape[1] != 960:
                img = cv2.resize(img, (960, round(img.shape[0] * 960 / img.shape[1])))
            tiles = []
            for c, p in pipes.items():
                r = p.step(img, t, pose)
                n += 1
                for k, v in r.timings.items():
                    stage.setdefault(k, []).append(v)
                if seg is not None and c == "drone":
                    ts = time.perf_counter()
                    r.events += seg.process(img, None, pose)       # walkable-ground hints (scene, never a stop)
                    stage.setdefault("seg", []).append((time.perf_counter() - ts) * 1000)
                for ev in r.events:
                    emit(ev, t)
                    if ev["type"] == "hazard":
                        print(f"[perception] t={t:.1f} hazard {ev['kind']}: {ev['text']}", flush=True)
                stop = live_walk_stop(r.objects)
                if stop and not was_stop.get(c):
                    print(f"[perception] t={t:.1f} {stop}", flush=True)
                was_stop[c] = bool(stop)
                if a.overlay:
                    tiles.append(draw(img, r, c))
            if a.overlay and tiles:
                frame = tiles[0] if len(tiles) == 1 else cv2.hconcat(tiles)
                if still:
                    cv2.imwrite(a.overlay, frame)
                else:
                    if writer is None:
                        Path(a.overlay).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(a.overlay, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                                 (frame.shape[1], frame.shape[0]))
                    writer.write(frame)
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
        if jsonl:
            jsonl.close()
        if cap is not None:
            cap.release()
        if src is not None:
            src.release()
        if n:
            parts = ", ".join(f"{k} {float(np.median(v)):.0f} ms" for k, v in stage.items())
            print(f"[perception] {n} frames on {device}; median {parts}", flush=True)


def live_walk_stop(objs):
    import live_walk
    return live_walk.stop_text(objs)


if __name__ == "__main__":
    main()
