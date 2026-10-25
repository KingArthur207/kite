"""Camera-only "what is directly in front of me, and how urgent": priorities P1 / P2 / P3 and beeps.

    python -m perception.near --source webcam:Logitech                    # the chest C270 (Windows or macOS)
    python -m perception.near --source webcam:Logitech --radar            # live ASCII radar in the terminal
    python -m perception.near --source data/imagined/doorway.mp4 --no-send --hud out/doorway_hud.mp4 \\
        --wav out/doorway_beeps.wav --jsonl out/doorway_near.jsonl        # a file: review video, beeps, events
    python -m perception.near --source photo.jpg --no-send                # one still
    python -m perception.near --profile c920 ...                          # force a camera profile (or a JSON)
    python -m perception.near --list-profiles

Sends `obstacles` to the earpiece (UDP, $KITE_EARPIECE_PORT or 5010) with each priority mapped onto the
sonar's existing fields (perception/near/output.py), unless --no-send. Beeps are only ever written to --wav.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import intrinsics as I

STILL = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m perception.near", description=__doc__.split("\n")[0])
    ap.add_argument("--source", help="webcam:<name or index>, a video file, a still image, or a kite.sources spec")
    ap.add_argument("--profile", default="auto", help="auto (USB name; calibrated JSON in data/cameras/ wins), "
                                                      "c270, c310, c920, generic, or a profile JSON path")
    ap.add_argument("--list-profiles", action="store_true", help="print the built-in camera profiles and exit")
    ap.add_argument("--cam-height", type=float, default=1.2, help="camera height above the ground (m)")
    ap.add_argument("--pitch", type=float, default=0.0, help="camera pitch prior, degrees down (fitted from depth)")
    ap.add_argument("--no-fit-pitch", action="store_true", help="use --pitch as is (level sim cameras)")
    ap.add_argument("--walk-speed", type=float, default=1.2, help="walker's speed (m/s): motion cue, corridor width")
    ap.add_argument("--depth", default="outdoor", help="outdoor | indoor (Depth Anything V2 Metric Small) | off | "
                                                      "zoo:<backend> (perception/depth_zoo.py)")
    ap.add_argument("--detector", choices=("coco", "hybrid"), default="coco",
                    help="coco: YOLO11n (people, bikes, cars, chairs, ...); hybrid: + YOLO-World street prompts")
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (default: the best available)")
    ap.add_argument("--every", type=int, default=0, help="video files: process every n-th frame (default ~10 Hz)")
    ap.add_argument("--fast", action="store_true", help="video files: do not pace to real time")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--ignore-bottom", type=float, default=0.0, help="fraction of rows that show the wearer")
    ap.add_argument("--jsonl", help="write every near event (one JSON per line)")
    ap.add_argument("--wav", help="render the beeps the run would have made to this WAV (never played)")
    ap.add_argument("--hud", help="files only: write a review video (boxes | turbo depth | radar)")
    ap.add_argument("--radar", action="store_true", help="redraw an ASCII radar in the terminal (a TTY)")
    ap.add_argument("--no-send", action="store_true", help="do not send obstacles to the earpiece")
    ap.add_argument("--ear-port", type=int, default=int(os.environ.get("KITE_EARPIECE_PORT", "5010")))
    return ap.parse_args(argv)


def pick_device(want: str | None) -> str:
    if want:
        return want
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def build_detector(kind: str, device: str):
    from .. import models
    coco = models.CocoDetector(device, 0.35, 640)
    ov = None
    if kind == "hybrid":
        from .. import vocab as V
        ov = models.OpenVocabDetector(V.load(), device, 0.2, 640)
    return models.HybridDetector(coco, ov)


def open_live(spec: str):
    """A camera through kite.sources (macOS names, indices). On Windows, where kite.sources cannot list cameras by
    name, fall back to DirectShow: names from pygrabber if installed, else the first camera, labelled with the
    PnP name Windows reports (enough to pick the profile)."""
    from kite.sources import open_source
    try:
        src = open_source(spec)
        return src, src.name
    except (LookupError, RuntimeError, OSError, ValueError) as e:
        if sys.platform != "win32":
            raise SystemExit(f"cannot open {spec}: {e}")
        return _open_windows(spec.split(":", 1)[-1], e)


def _open_windows(query: str, why):
    import subprocess

    import cv2
    names: list[str] = []
    try:
        from pygrabber.dshow_graph import FilterGraph  # optional: DirectShow order == cv2.CAP_DSHOW indices
        names = FilterGraph().get_input_devices()
    except Exception:
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "Get-PnpDevice -Class Camera,Image -PresentOnly | Select -Expand FriendlyName"],
                                 capture_output=True, text=True, timeout=10).stdout
            names = [n.strip() for n in out.splitlines() if n.strip()]
        except Exception:
            names = []
    idx = int(query) if query.isdigit() else next((i for i, n in enumerate(names) if query.lower() in n.lower()), 0)
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {query!r} ({why}); cameras seen: {names or 'none'}")
    label = names[idx] if idx < len(names) else (next((n for n in names if query.lower() in n.lower()), query))
    return cap, f"webcam:{idx} {label}"


def main(argv=None) -> None:
    a = parse_args(argv)
    if a.list_profiles:
        for k, p in I.BUILTIN.items():
            print(f"{k:8s} {p.describe()}")
        return
    if not a.source:
        sys.exit("--source is required: webcam:Logitech, a video file or a still")
    import cv2
    import numpy as np

    from . import beeper, output
    from .depth import DepthProcessor, make_backend
    from .system import NearSystem

    still = a.source.lower().endswith(STILL)
    path = Path(a.source).expanduser()
    is_file = still or path.is_file()
    if a.hud and not is_file:
        sys.exit("--hud is for files: frames from a live camera are never stored")
    src = cap = None
    name = None
    if still:
        img0 = cv2.imread(str(path))
        if img0 is None:
            sys.exit(f"cannot read {path}")
        size = (img0.shape[1], img0.shape[0])
    elif is_file:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            sys.exit(f"cannot open {path}")
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = a.every or max(1, round(vfps / 10))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    else:
        src, name = open_live(a.source)
        size = None
    try:
        prof, why = I.resolve(a.profile, name, size)
    except ValueError as e:
        sys.exit(str(e))
    print(f"[near] {prof.describe()}", flush=True)
    print(f"[near] profile: {why}", flush=True)
    device = pick_device(a.device)
    det = build_detector(a.detector, device)
    backend = make_backend(a.depth, device)
    dp = DepthProcessor(backend, a.cam_height, a.pitch, not a.no_fit_pitch) if backend else None
    ns = NearSystem(prof, det, dp, a.cam_height, a.pitch, 3.0, a.walk_speed, still=still,
                    ignore_bottom=a.ignore_bottom)
    ear = None
    if not a.no_send:
        from earpiece.client import EarpieceClient
        ear = EarpieceClient(host=os.environ.get("KITE_EARPIECE_HOST", "127.0.0.1"), port=a.ear_port,
                             source=output.SOURCE)
    term = output.Terminal(a.radar)
    jsonl = open(a.jsonl, "w") if a.jsonl else None
    writer, events, n, seq, t0, fidx = None, [], 0, 0, time.monotonic(), 0
    stage: dict[str, list[float]] = {}
    print(f"[near] device={device} depth={a.depth} detector={a.detector} cam_h={a.cam_height} m "
          f"source={a.source}", flush=True)
    try:
        while True:
            if a.max_seconds and time.monotonic() - t0 > a.max_seconds:
                break
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
                if not ok or img is None:
                    continue
                if prof.width != img.shape[1] or prof.height != img.shape[0]:
                    prof = prof.scaled(img.shape[1], img.shape[0])
                    ns.prof = prof
                t = time.monotonic() - t0
            ts = time.perf_counter()
            dets = det(img)
            dt_det = (time.perf_counter() - ts) * 1000
            nf = ns.step(img, t, dets)
            nf.timings["detect"] = dt_det
            n += 1
            for k, v in nf.timings.items():
                stage.setdefault(k, []).append(v)
            ev = nf.event()
            events.append(ev)
            term.show(ev)
            if jsonl:
                jsonl.write(json.dumps(ev, separators=(",", ":")) + "\n")
            if ear:
                seq += 1
                ear.send(output.to_obstacles(ev, seq))
            if a.hud:
                from .hud import draw
                frame = draw(img, nf, prof, dets, a.cam_height)
                if still:
                    Path(a.hud).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(Path(a.hud).with_suffix(".jpg")), frame)
                else:
                    if writer is None:
                        Path(a.hud).parent.mkdir(parents=True, exist_ok=True)
                        fps = vfps / step if cap is not None else 10
                        writer = cv2.VideoWriter(a.hud, cv2.VideoWriter_fourcc(*"mp4v"), fps,
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
        if a.wav and events:
            p = beeper.write_wav(a.wav, beeper.render(beeper.timeline_from_events(events)))
            print(f"[near] wrote {p} (not played)", flush=True)
        if n:
            parts = ", ".join(f"{k} {float(np.median(v)):.0f} ms" for k, v in stage.items())
            counts = {p: sum(1 for e in events if e["priority"] == p) for p in (1, 2, 3, None)}
            print(f"[near] {n} frames; P1 {counts[1]}, P2 {counts[2]}, P3 {counts[3]}, clear {counts[None]}; "
                  f"median {parts}", flush=True)


if __name__ == "__main__":
    main()
