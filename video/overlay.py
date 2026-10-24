"""Burn Kite's cues onto footage for the demo video.

    python video/overlay.py recordings/20261025_070312_tello --events events.jsonl \\
        --wav out/heard.wav -o out/final.mp4 [--offset 2.5] [--blur-faces] [--badge "Drone view"]
    python video/overlay.py clips/street.mp4 --events data/sample_walk.jsonl -o out/test.mp4 --seconds 20

SRC is a kite.record take directory (its video.mkv; --wall-clock times it by frames.csv instead of
frame index / fps) or any video file. --events is a JSONL of earpiece events with a "t" field (a detector's --jsonl,
data/sample_walk.jsonl). Event t=0 is the first event's t, and --offset is the video time (s) where it
lands. The WAV from `python -m earpiece render --jsonl` shares that t=0, so the same offset aligns it,
and its <wav>.captions.jsonl (what the earpiece actually said) becomes the captions.

Drawn: captions (bottom), active hazards as a top banner with a bearing arrow and distance, a sonar
indicator (left/right dots that grow as the nearest obstacle on that side gets closer), a corner
badge. Needs opencv-python and numpy; ffmpeg (brew install ffmpeg) for H.264 and the audio track.

--seg puts translucent street segmentation under all of that (walkable ground green, road grey, people
orange, vehicles purple), the walker's corridor outline and a "walkable ahead" readout. It comes from
sim/seg_standin.py, a STAND-IN until the streetseg perception package lands (ultralytics + weights/yolo26n-sem-ade20k.pt).
--events is optional: without it there are no cues, which suits a --seg only render.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HAZARD_TTL_S = 20.0      # the earpiece's default when a hazard has no ttl
CAPTION_MIN_S = 2.5      # a caption stays up at least this long
OBSTACLE_STALE_S = 0.5   # an obstacles frame older than this no longer drives the sonar dots
SONAR_NEAR_M, SONAR_FAR_M = 0.5, 6.0
CENTER_DEG = 10.0        # objects this close to straight ahead light both dots
FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
RED, ORANGE, GREEN = (40, 40, 230), (0, 150, 255), (90, 200, 90)
SEG_COLORS = {"road": (70, 70, 70), "sidewalk": GREEN, "crosswalk": GREEN,      # painted in this order
              "vehicle": (200, 60, 170), "person": ORANGE}                      # vehicles purple
SEG_ALPHA = 0.4


# ------------------------------------------------------------------ inputs
def load_jsonl(path: str | Path) -> list[dict]:
    out = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{n}: bad JSON ({e.msg})")
    return out


def timed_events(events: list[dict], offset: float) -> list[tuple[float, dict]]:
    """(video time, event), with the first event's t at --offset (same t=0 as earpiece render)."""
    ts = [e["t"] for e in events if isinstance(e.get("t"), (int, float))]
    if not ts:
        sys.exit("--events: no line has a numeric 't'")
    t0 = ts[0]
    return sorted(((e["t"] - t0 + offset, e) for e in events if isinstance(e.get("t"), (int, float))),
                  key=lambda x: x[0])


def captions_from_events(evs: list[tuple[float, dict]]) -> list[tuple[float, float, str]]:
    """Fallback when there is no render: the lines the events ask the earpiece to say."""
    out = []
    for t, e in evs:
        text = e.get("text") if e.get("type") in ("say", "guidance") else None
        if e.get("type") == "hazard":
            text = hazard_text(e)
        if text:
            out.append((t, t + CAPTION_MIN_S, text))
    return out


def hazard_text(e: dict) -> str:
    if e.get("text"):
        return e["text"]
    if e.get("kind") == "crosswalk":
        return "Walk signal is on." if e.get("state") in ("walk", "go", "green") else "Don't walk."
    return (e.get("kind") or "hazard").replace("_", " ").capitalize()


class Timeline:
    """What is on screen at video time T. Call at() with non-decreasing T."""

    def __init__(self, evs: list[tuple[float, dict]], captions: list[tuple[float, float, str]]):
        self.evs, self.i = evs, 0
        self.captions = captions
        self.hazards: dict[str, tuple[float, dict]] = {}   # key -> (until, event)
        self.obstacles: tuple[float, list] | None = None

    def at(self, T: float) -> dict:
        while self.i < len(self.evs) and self.evs[self.i][0] <= T:
            t, e = self.evs[self.i]
            self.i += 1
            if e.get("type") == "hazard":
                self.hazards[e.get("key") or e.get("kind") or "hazard"] = (t + (e.get("ttl") or HAZARD_TTL_S), e)
            elif e.get("type") == "obstacles":
                self.obstacles = (t, e.get("objects") or [])
        self.hazards = {k: v for k, v in self.hazards.items() if v[0] > T}
        objs = self.obstacles[1] if self.obstacles and T - self.obstacles[0] < OBSTACLE_STALE_S else []
        caps = [c for c in self.captions if c[0] <= T < max(c[1], c[0] + CAPTION_MIN_S)]
        return {"hazards": [e for _, e in self.hazards.values()], "objects": objs,
                "caption": caps[-1][2] if caps else None}


def open_source(src: Path, wall_clock: bool):
    """(cv2.VideoCapture, fps, frame times or None for a plain file). Frame i is at i / fps, which is how a
    detector reading the file (sim/standin_detect.py --source video.mkv) stamps its events.
    wall_clock: time a kite.record take by frames.csv t_mono instead (events stamped on t_mono)."""
    if not src.is_dir():
        cap = cv2.VideoCapture(str(src))
        return cap, cap.get(cv2.CAP_PROP_FPS) or 30.0, None
    cap = cv2.VideoCapture(str(src / "video.mkv"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    with open(src / "frames.csv") as fh:             # Matroska MJPG has no reliable frame count
        mono = [float(r["t_mono"]) for r in csv.DictReader(fh)]
    if wall_clock and mono:
        return cap, fps, [t - mono[0] for t in mono]
    return cap, fps, [i / fps for i in range(len(mono))]


# ------------------------------------------------------------------ drawing
def _text(img, s: str, org, scale: float, color=WHITE, thick: int = 2) -> None:
    cv2.putText(img, s, org, FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _wrap(s: str, scale: float, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in s.split():
        nxt = f"{cur} {word}".strip()
        if cur and cv2.getTextSize(nxt, FONT, scale, 2)[0][0] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = nxt
    return lines + [cur] if cur else lines


def _panel(img, x1, y1, x2, y2, color=BLACK, alpha: float = 0.55) -> None:
    roi = img[max(0, y1):y2, max(0, x1):x2]
    roi[:] = (roi * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def draw_badge(img, label: str, k: float) -> int:
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.7 * k, 2)
    pad, r, x1 = int(10 * k), max(3, int(6 * k)), int(16 * k)
    x2, y2 = x1 + 3 * pad + 2 * r + tw, x1 + th + 2 * pad
    _panel(img, x1, x1, x2, y2, (30, 30, 30), 0.7)
    cv2.circle(img, (x1 + pad + r, (x1 + y2) // 2), r, RED, -1, cv2.LINE_AA)
    _text(img, label, (x1 + 2 * pad + 2 * r, y2 - pad), 0.7 * k)
    return x2


def draw_hazards(img, hazards: list[dict], x0: int, k: float) -> None:
    if not hazards:
        return
    w = img.shape[1]
    h = int(64 * k)
    y = int(16 * k)
    for e in hazards[:2]:
        _panel(img, x0, y, w - int(16 * k), y + h, RED, 0.75)
        cx, cy, r = x0 + h // 2 + int(6 * k), y + h // 2, int(h * 0.36)
        cv2.circle(img, (cx, cy), r, WHITE, max(1, int(2 * k)), cv2.LINE_AA)
        b = e.get("bearing")
        if isinstance(b, (int, float)):
            tip = (int(cx + r * 0.85 * math.sin(math.radians(b))), int(cy - r * 0.85 * math.cos(math.radians(b))))
            cv2.arrowedLine(img, (cx, cy), tip, WHITE, max(2, int(3 * k)), cv2.LINE_AA, tipLength=0.45)
        label = hazard_text(e).rstrip(".").upper()
        d = e.get("distance")
        if isinstance(d, (int, float)):
            label += f"   {d:.0f} m"
        _text(img, label, (cx + r + int(16 * k), cy + int(10 * k)), 0.85 * k)
        y += h + int(8 * k)


def _dot(img, center, d: float | None, k: float) -> None:
    if d is None:
        cv2.circle(img, center, int(6 * k), (160, 160, 160), max(1, int(2 * k)), cv2.LINE_AA)
        return
    near = min(1.0, max(0.0, (SONAR_FAR_M - d) / (SONAR_FAR_M - SONAR_NEAR_M)))
    color = RED if d < 1.5 else ORANGE if d < 3.0 else GREEN
    cv2.circle(img, center, int((8 + 26 * near) * k), color, -1, cv2.LINE_AA)


def draw_sonar(img, objects: list[dict], y_bottom: int, k: float) -> None:
    def nearest(side) -> float | None:
        ds = [o["distance"] for o in objects if isinstance(o.get("distance"), (int, float))
              and isinstance(o.get("bearing"), (int, float)) and side(o["bearing"])]
        return min(ds) if ds else None

    left = nearest(lambda b: b <= CENTER_DEG)
    right = nearest(lambda b: b >= -CENTER_DEG)
    pw, ph = int(190 * k), int(96 * k)
    x1, y1 = int(16 * k), y_bottom - ph
    _panel(img, x1, y1, x1 + pw, y_bottom, (30, 30, 30), 0.6)
    cx, cy = x1 + pw // 2, y1 + ph // 2 + int(8 * k)
    cv2.circle(img, (cx, cy), int(12 * k), WHITE, -1, cv2.LINE_AA)               # the walker's head
    _dot(img, (cx - int(56 * k), cy), left, k)
    _dot(img, (cx + int(56 * k), cy), right, k)
    grey = (200, 200, 200)
    _text(img, "SONAR", (x1 + int(8 * k), y1 + int(20 * k)), 0.5 * k, grey, 1)
    _text(img, "L", (x1 + int(8 * k), y_bottom - int(8 * k)), 0.5 * k, grey, 1)
    _text(img, "R", (x1 + pw - int(20 * k), y_bottom - int(8 * k)), 0.5 * k, grey, 1)


def draw_seg(img, seg: dict, k: float) -> None:
    """Translucent masks (seg_standin.Segmenter.segment output) and the walker's corridor outline."""
    from seg_standin import corridor
    tint = img.copy()
    for name, color in SEG_COLORS.items():
        m = seg["masks"].get(name)
        if m is not None:
            tint[m] = color
    cv2.addWeighted(tint, SEG_ALPHA, img, 1 - SEG_ALPHA, 0, dst=img)
    cv2.polylines(img, [corridor(*img.shape[:2])], True, WHITE, max(1, int(2 * k)), cv2.LINE_AA)


def draw_walkable(img, walk: dict, y_bottom: int, k: float) -> None:
    """Bottom right, mirroring the sonar panel: share of the walker's corridor that is sidewalk."""
    pct = walk["ahead"]
    color = GREEN if pct >= 0.5 else ORANGE if pct >= 0.2 else RED
    pw, ph = int(190 * k), int(96 * k)
    x2 = img.shape[1] - int(16 * k)
    x1, y1 = x2 - pw, y_bottom - ph
    _panel(img, x1, y1, x2, y_bottom, (30, 30, 30), 0.6)
    _text(img, "WALKABLE AHEAD", (x1 + int(8 * k), y1 + int(20 * k)), 0.5 * k, (200, 200, 200), 1)
    _text(img, f"{pct:.0%}", (x1 + int(8 * k), y_bottom - int(16 * k)), 1.4 * k, color, 3)


def draw_caption(img, text: str | None, k: float) -> int:
    """Draws the caption; returns the y of its top edge (the sonar panel sits above it)."""
    h, w = img.shape[:2]
    margin = int(20 * k)
    if not text:
        return h - margin
    scale = 0.95 * k
    lines = _wrap(f'"{text}"', scale, w - 4 * margin)[-2:]
    lh = int(42 * k)
    top = h - margin - lh * len(lines) - int(12 * k)
    _panel(img, margin, top, w - margin, h - margin, BLACK, 0.6)
    for i, s in enumerate(lines):
        tw = cv2.getTextSize(s, FONT, scale, 2)[0][0]
        _text(img, s, ((w - tw) // 2, top + lh * (i + 1) - int(6 * k)), scale)
    return top


# ------------------------------------------------------------------ face blur
class FaceBlur:
    """Pixelates heads. With ultralytics + YOLO weights: the top of every person box (works for
    profiles and backs of heads, same idea as the live walk's reveal screen). Otherwise OpenCV 4's
    bundled Haar face detectors (frontal + profile), which miss some faces: check the output."""

    def __init__(self, weights: str):
        self.model = None
        try:
            from ultralytics import YOLO
            if Path(weights).exists():
                self.model = YOLO(weights)
        except ImportError:
            pass
        if self.model is None:
            if not hasattr(cv2, "CascadeClassifier"):     # OpenCV 5 dropped the Haar detectors
                sys.exit(f"--blur-faces needs ultralytics and {weights} "
                         "(pip install ultralytics; python sim/standin_detect.py --download), or OpenCV 4")
            print(f"[overlay] face blur: Haar cascades (no ultralytics or no {weights})", flush=True)
            self.cascades = [cv2.CascadeClassifier(cv2.data.haarcascades + n)
                             for n in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")]

    def boxes(self, img) -> list[tuple[int, int, int, int]]:
        if self.model is not None:
            r = self.model.predict(img, classes=[0], conf=0.3, verbose=False)[0]
            out = []
            for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy().astype(int):
                side = int(min(x2 - x1, (y2 - y1) * 0.25))
                out.append((int((x1 + x2 - side) / 2), y1, side, side))
            return out
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = []
        for c in self.cascades:
            for g, flip in ((gray, False), (cv2.flip(gray, 1), True)):
                for x, y, w, h in c.detectMultiScale(g, 1.1, 5, minSize=(16, 16)):
                    found.append((gray.shape[1] - x - w if flip else x, y, w, h))
        return found

    def __call__(self, img):
        for x, y, w, h in self.boxes(img):
            pad = w // 5
            x, y = max(0, x - pad), max(0, y - pad)
            roi = img[y:y + h + 2 * pad, x:x + w + 2 * pad]
            if roi.size:
                roi[:] = cv2.resize(cv2.resize(roi, (6, 6)), roi.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        return img


# ------------------------------------------------------------------ main
def ffmpeg_writer(out: Path, w: int, h: int, fps: float, wav: str | None, offset: float, seconds: float):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps:.3f}", "-i", "-"]
    if wav:
        cmd += ["-itsoffset", f"{offset:.3f}"] if offset >= 0 else ["-ss", f"{-offset:.3f}"]
        cmd += ["-i", wav, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast",
            "-t", f"{seconds:.3f}", "-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="kite.record take directory, or a video file")
    p.add_argument("--events", help="JSONL of earpiece events with t (omit for a --seg only render)")
    p.add_argument("--offset", type=float, default=0.0, help="video time (s) of the first event (default 0)")
    p.add_argument("--wav", help="what the wearer heard (earpiece render --jsonl); muxed as the audio track")
    p.add_argument("--captions", help="captions JSONL (default: <wav>.captions.jsonl, else event text)")
    p.add_argument("--badge", default="Drone view", help='corner label, e.g. "Kite sees" (default "Drone view")')
    p.add_argument("--blur-faces", action="store_true", help="pixelate faces/heads (always for public footage)")
    p.add_argument("--yolo", default=str(Path(__file__).resolve().parents[1] / "weights" / "yolo11n.pt"),
                   help="person model for --blur-faces (falls back to Haar faces)")
    p.add_argument("--seg", action="store_true",
                   help="segmentation masks under the cues (stand-in: sim/seg_standin.py)")
    p.add_argument("--wall-clock", action="store_true",
                   help="take dir: place frames by frames.csv t_mono (for events stamped on t_mono)")
    p.add_argument("--seconds", type=float, help="stop after this much video")
    p.add_argument("-o", "--out", required=True, help="output .mp4")
    a = p.parse_args(argv)

    src = Path(a.src)
    if not src.exists():
        sys.exit(f"{src}: not found")
    cap, fps, times = open_source(src, a.wall_clock)
    if not cap.isOpened():
        sys.exit(f"{src}: cannot open video")
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (len(times) if times else 0)
    src_dur = times[-1] + 1 / fps if times else n_src / fps
    if src_dur <= 0:
        sys.exit(f"{src}: cannot tell the video's length (no frame count)")
    dur = min(src_dur, a.seconds) if a.seconds else src_dur

    evs = timed_events(load_jsonl(a.events), a.offset) if a.events else []
    cap_path = a.captions or (str(Path(a.wav).with_suffix(".captions.jsonl")) if a.wav else None)
    if cap_path and Path(cap_path).exists():
        captions = [(c["t"] + a.offset, c.get("end", c["t"]) + a.offset, c["text"]) for c in load_jsonl(cap_path)]
        print(f"[overlay] captions: {cap_path} ({len(captions)} lines)")
    else:
        captions = captions_from_events(evs)
        print(f"[overlay] captions: from event text ({len(captions)} lines)")
    tl = Timeline(evs, captions)
    blur = FaceBlur(a.yolo) if a.blur_faces else None
    segmenter = None
    if a.seg:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
        from seg_standin import Segmenter
        segmenter = Segmenter()

    ok, frame = cap.read()
    if not ok:
        sys.exit(f"{src}: no frames")
    h, w = frame.shape[:2]
    k = w / 1280
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        proc, writer = ffmpeg_writer(out, w, h, fps, a.wav, a.offset, dur), None
    else:
        print("[overlay] ffmpeg not found (brew install ffmpeg): writing mp4v video WITHOUT audio", flush=True)
        proc, writer = None, cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    si = 0                                        # index of `frame` in the source
    n_out = int(round(dur * fps))
    for j in range(n_out):
        T = j / fps
        if times:                                 # hold the latest source frame at or before T
            while si + 1 < len(times) and times[si + 1] <= T:
                ok, nxt = cap.read()
                if not ok:
                    break
                frame, si = nxt, si + 1
        elif j > 0:
            ok, nxt = cap.read()
            if not ok:
                break
            frame = nxt
        seg = segmenter.segment(frame) if segmenter else None   # on the raw frame, before the blur
        img = blur(frame.copy()) if blur else frame.copy()
        if seg:
            draw_seg(img, seg, k)                 # translucent, so blurred heads stay blurred
        st = tl.at(T)
        x_badge = draw_badge(img, a.badge, k)
        draw_hazards(img, st["hazards"], x_badge + int(16 * k), k)
        top = draw_caption(img, st["caption"], k)
        draw_sonar(img, st["objects"], top - int(12 * k), k)
        if seg:
            draw_walkable(img, seg["walkable"], top - int(12 * k), k)
        if proc:
            proc.stdin.write(img.tobytes())
        else:
            writer.write(img)
        if j % int(fps * 5) == 0:
            print(f"[overlay] {T:5.1f}/{dur:.1f}s", flush=True)
    cap.release()
    if proc:
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit("[overlay] ffmpeg failed")
    else:
        writer.release()
    print(f"[overlay] wrote {out} ({dur:.1f}s, {w}x{h} @ {fps:.2f} fps{', audio ' + a.wav if a.wav and proc else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
