"""Live walk bench: 40 walks with no judge and no camera (spec section 9, Sat 6 PM gate).

Each run plays one clip through the stand-in detector (chest settings, real-time pacing) into a
headless earpiece on its own ports, then scores what the earpiece said against the clip's truth.

    RUN="uv run --no-project --with ultralytics --with opencv-python-headless --with lap --with websockets --with python-dotenv --with numpy --with scipy"
    $RUN python scripts/live_walk_bench.py prepare          # once, with internet: clips/bench/
    $RUN python scripts/live_walk_bench.py run              # 40 runs: table on stdout, JSON in out/bench/
    $RUN python scripts/live_walk_bench.py run --runs 8 --only syn_walkup,syn_cross
    python scripts/live_walk_bench.py selftest              # the scoring rules, no clips needed

Clips (all local, gitignored): stock walks from Pexels and Mixkit (free licenses, never committed),
cut into approach-then-walk-away "ping-pong" clips, plus synthetic walks: a person cut out of a
stock frame and drawn into an empty corridor along a scripted path with the live camera's geometry
(fx 700, 960x540, 1.2 m up, 15 deg down), so their true distance is known on every frame.
Every run gets fresh jitter (speed, stop point, offset, brightness, chest bob).

Scoring, per run: a stop is expected when the truth (nearest person within 0.7 m of the path)
reaches 1.5 m. Latency runs from that frame to the start of "Stop. Person ahead.". Any stop while
the truth is beyond 2.5 m, or on a clip that never comes inside 1.5 m, is a false stop. "Path is
clear" must come after the person has left (beyond 2.5 m or off the path) and before the clip ends;
one that comes while they are still there is an early continue (counted as a failure).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH, OUT = ROOT / "clips" / "bench", ROOT / "out" / "bench"
EAR_PY = ROOT / "earpiece" / ".venv" / "bin" / "python"
UDP_PORT, WS_PORT = 6210, 6211
FPS, LEAD_S, SETTLE_S = 30, 1.5, 4.0     # written clips; static first frame (detector warm-up); gap between runs
CROSS_M, FALSE_M, FAR_M, CORRIDOR_M = 1.5, 2.2, 2.5, 0.7   # stop expected / a stop beyond this is false / gone
LATE_S = 1.0                             # a stop more than this after the 1.5 m frame is late (a fail)
W, H = 960, 540
SYN = {"fx": 700.0, "cam_height": 1.2, "pitch": 15.0}          # scripts/demo.sh live: an iPhone at 1280x720 -> 960x540
REAL = {"fx": 700.0, "cam_height": 1.2, "pitch": 0.0}          # stock clips: an iPhone-like 1x lens, level
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) kite-bench"}

# name, url, ffmpeg input options, filter before ping-pong, ping-pong (approach then walk away), license
STOCK = [
    ("approach", "https://www.pexels.com/download/video/2795407/", ["-ss", "9"], "reverse", True,
     "Pexels 2795407, woman walking in a building (Pexels license)"),
    ("corridor", "https://assets.mixkit.co/videos/33396/33396-720.mp4", [], "", True,
     "Mixkit 33396, man walking in a corridor (Mixkit free license)"),
    ("group", "https://assets.mixkit.co/videos/23809/23809-720.mp4", ["-t", "9"], "", False,
     "Mixkit 23809, four people walking toward the camera (Mixkit free license)"),
    ("hotel_empty", "https://assets.mixkit.co/videos/34613/34613-720.mp4", [], "", True,
     "Mixkit 34613, empty hotel corridor (Mixkit free license)"),
    ("datacenter_empty", "https://assets.mixkit.co/videos/23219/23219-720.mp4", ["-t", "9"], "", False,
     "Mixkit 23219, empty data-centre aisle (Mixkit free license)"),
]
SPRITE_SRC, BG_SRC = "corridor", "hotel_empty"


# ------------------------------------------------------------------ synthetic walks
def walkup(rng, v=1.2, aside=1.0, x0=0.0, z0=4.0):
    """Teammate walks from z0 to a stop 1.0-1.3 m out, waits 2 s, steps aside, stays there."""
    v *= rng.uniform(0.85, 1.15)
    stop_z, x0 = rng.uniform(1.0, 1.3), x0 + rng.uniform(-0.15, 0.15)
    t1 = (z0 - stop_z) / v
    side = rng.choice((-1, 1)) * 1.8
    return [(0, x0, z0), (t1, x0, stop_z), (t1 + 2.0, x0, stop_z), (t1 + 2.0 + 1.8 / aside, side, stop_z),
            (t1 + 9.0, side, stop_z)]


def stepin(rng):
    """The demo: the judge walks at ~1 m/s and a teammate steps in from the side 2.5 m ahead."""
    v, side = rng.uniform(0.9, 1.2), rng.choice((-1, 1))
    stop_z = rng.uniform(1.0, 1.3)
    t1 = 1.0 + (2.4 - stop_z) / v
    return [(0, side * 1.6, 3.4), (1.0, 0.0, 2.4), (t1, 0.0, stop_z), (t1 + 2.0, 0.0, stop_z),
            (t1 + 3.8, -side * 1.8, stop_z), (t1 + 10.0, -side * 1.8, stop_z)]


SCENARIOS = {        # name -> (people(rng) -> [keyframes...], occluder: None, "strap" or "table")
    "syn_walkup": (lambda r: [walkup(r)], None),
    "syn_stepin": (lambda r: [stepin(r)], None),
    "syn_slow": (lambda r: [walkup(r, v=0.6)], None),
    "syn_fast": (lambda r: [walkup(r, v=1.8)], None),
    "syn_occluded": (lambda r: [walkup(r)], "strap"),     # a bag strap / hand over the lower-left of the lens
    "syn_table": (lambda r: [walkup(r)], "table"),        # legs hidden behind a low table: known limit (see README)
    "syn_two": (lambda r: [walkup(r), [(0, -1.4, 2.6), (30, -1.4, 2.6)]], None),
    "syn_cross": (lambda r: [[(0, -2.6, z), (5.2 / v, 2.6, z), (5.2 / v + 4, 2.6, z)]
                             for z, v in [(r.uniform(2.3, 2.8), r.uniform(1.0, 1.5))]], None),
    "syn_empty": (lambda r: [], None),
}


def at(keys, t):
    """Piecewise-linear (x, z) at time t; holds the ends."""
    if t <= keys[0][0]:
        return keys[0][1:]
    for (ta, xa, za), (tb, xb, zb) in zip(keys, keys[1:]):
        if t <= tb:
            f = (t - ta) / max(tb - ta, 1e-9)
            return xa + f * (xb - xa), za + f * (zb - za)
    return keys[-1][1:]


def project(x, y, z, g=SYN, bob=0.0):
    """World point (lateral x, height y, forward z) -> pixel (u, v) for a camera g['cam_height'] up, pitched
    g['pitch'] + bob degrees down. None behind the camera."""
    p = math.radians(g["pitch"] + bob)
    ry = y - g["cam_height"]
    zc = -ry * math.sin(p) + z * math.cos(p)
    if zc <= 0.05:
        return None
    yc = -ry * math.cos(p) - z * math.sin(p)
    return W / 2 + g["fx"] * x / zc, H / 2 + g["fx"] * yc / zc


def truth_of(people_xz):
    """Nearest person on the path (|x| <= CORRIDOR_M, in front), as range in m, else None."""
    d = [math.hypot(x, z) for x, z in people_xz if abs(x) <= CORRIDOR_M and z > 0]
    return round(min(d), 3) if d else None


def render_synth(name, rng, path):
    import cv2
    import numpy as np
    people_fn, occl = SCENARIOS[name]
    people = people_fn(rng)
    sprites = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in sorted(BENCH.glob("sprite_*.png"))]
    bg = cv2.imread(str(BENCH / "bg.png"))
    end = max([k[-1][0] for k in people], default=8.0) + LEAD_S
    gain, bob_a, bob_f, phase = rng.uniform(0.75, 1.15), rng.uniform(0.0, 1.2), rng.uniform(1.6, 2.0), rng.random()
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt = []
    for i in range(int(end * FPS)):
        t = max(0.0, i / FPS - LEAD_S)
        bob = bob_a * math.sin(2 * math.pi * (bob_f * t + phase)) if i / FPS >= LEAD_S else 0.0
        img = bg.copy()
        xz = [at(k, t) for k in people]
        for j, (x, z) in sorted(enumerate(xz), key=lambda p: -p[1][1]):        # far first
            top, foot = project(x, 1.75, z, bob=bob), project(x, 0.0, z, bob=bob)
            if top is None or foot is None:
                continue
            spr = sprites[(int(t * 4) + j) % len(sprites)]
            h = foot[1] - top[1]
            w = h * spr.shape[1] / spr.shape[0]
            paste(img, spr, foot[0] - w / 2, top[1], w, h)
        if occl == "table":                         # a table edge across the lower third
            cv2.rectangle(img, (0, int(H * 0.68)), (W, H), (60, 70, 80), -1)
        elif occl == "strap":                       # something dark over the lower-left 45 % of the lens
            cv2.fillPoly(img, [np.array([(0, H // 3), (int(W * 0.45), H), (0, H)], np.int32)], (25, 25, 30))
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        out.write(img)
        gt.append(truth_of(xz))
    out.release()
    return {"fps": FPS, "geom": SYN, "gt": gt, "people": people}


def paste(img, spr, x, y, w, h):
    """Alpha-blend sprite (BGRA) scaled to w x h at (x, y), clipped to img."""
    import cv2
    import numpy as np
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    if w < 2 or h < 2:
        return
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = spr.shape[1] / w, spr.shape[0] / h            # crop in sprite space before scaling: a person
    crop = spr[int((y0 - y) * sy):max(int((y1 - y) * sy), int((y0 - y) * sy) + 1),    # at 1 m is 3x the frame
               int((x0 - x) * sx):max(int((x1 - x) * sx), int((x0 - x) * sx) + 1)]
    crop = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    a = crop[:, :, 3:4].astype(np.float32) / 255
    roi = img[y0:y1, x0:x1]
    roi[:] = (crop[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


# ------------------------------------------------------------------ stock clips and their truth
def prepare(args) -> None:
    import cv2
    import numpy as np
    src = BENCH / "src"
    src.mkdir(parents=True, exist_ok=True)
    for name, url, inopt, vf, pingpong, lic in STOCK:
        raw = src / f"{name}.mp4"
        if not raw.exists():
            print(f"download {name}: {lic}")
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                raw.write_bytes(r.read())
        clip = BENCH / f"{name}.mp4"
        if not clip.exists():
            base = f"[0:v]fps={FPS},scale={W}:-2" + (f",{vf}" if vf else "")
            fc = f"{base},split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[o]" if pingpong else f"{base}[o]"
            subprocess.run(["ffmpeg", "-v", "error", "-y", *inopt, "-i", str(raw), "-filter_complex", fc,
                           "-map", "[o]", "-an", str(clip)], check=True)
    sys.path.insert(0, str(ROOT / "sim"))
    import standin_detect as sd
    from ultralytics import YOLO
    model = YOLO(str(sd.WEIGHTS))
    for name, *_ in STOCK:
        meta = BENCH / f"{name}.json"
        if meta.exists() and not args.force:
            continue
        cap, gt = cv2.VideoCapture(str(BENCH / f"{name}.mp4")), []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            h, w = img.shape[:2]
            objs = sd.objects_from_boxes(sd.detect(model, img, track=False), REAL["fx"], REAL["cam_height"],
                                         w, h, REAL["pitch"])
            gt.append(truth_of([(o["distance"] * math.sin(math.radians(o["bearing"])),
                                 o["distance"] * math.cos(math.radians(o["bearing"])))
                                for o in objs if o["kind"] == "person"]))
        # centred median over 5 frames: one-frame misses and flickers are not the truth
        sm = []
        for i in range(len(gt)):
            win = [g for g in gt[max(0, i - 2):i + 3] if g is not None]
            sm.append(round(statistics.median(win), 3) if len(win) >= 3 else None)
        meta.write_text(json.dumps({"fps": FPS, "geom": REAL, "gt": sm}))
        near = [g for g in sm if g is not None]
        print(f"truth {name}: {len(sm)} frames, nearest {min(near) if near else '-'} m")
    # sprites: the corridor man at his largest, cut out with GrabCut; the background: the empty hotel corridor
    cap = cv2.VideoCapture(str(src / f"{SPRITE_SRC}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k, fi in enumerate(range(n - 40, n, 10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        boxes = [b for b in sd.detect(model, img, track=False) if b[4] == "person"]
        x1, y1, x2, y2 = (int(v) for v in max(boxes, key=lambda b: (b[3] - b[1]))[:4])
        pad = 12
        rect = (max(x1 - pad, 0), max(y1 - pad, 0), x2 - x1 + 2 * pad, y2 - y1 + 2 * pad)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.grabCut(img, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        spr = np.dstack([img, alpha])[y1:y2, x1:x2]
        cv2.imwrite(str(BENCH / f"sprite_{k}.png"), spr)
    cap = cv2.VideoCapture(str(src / f"{BG_SRC}.mp4"))
    ok, img = cap.read()
    cv2.imwrite(str(BENCH / "bg.png"), cv2.resize(img, (W, H)))
    print(f"ready: {BENCH}")


def variant(name, rng, path):
    """A jittered copy of a prepared stock clip: LEAD_S of the first frame, brightness, chest bob."""
    import cv2
    import numpy as np
    meta = json.loads((BENCH / f"{name}.json").read_text())
    cap = cv2.VideoCapture(str(BENCH / f"{name}.mp4"))
    gain, bob_px, bob_f = rng.uniform(0.75, 1.15), rng.uniform(0, 8), rng.uniform(1.6, 2.0)
    lead = int(LEAD_S * FPS)
    out, first, i = None, None, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (W, round(img.shape[0] * W / img.shape[1])))
        if out is None:
            out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, img.shape[1::-1])
            first = img
            for _ in range(lead):
                out.write(first)
        dy = int(round(bob_px * math.sin(2 * math.pi * bob_f * i / FPS)))
        img = np.roll(img, dy, axis=0)
        out.write(np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8))
        i += 1
    out.release()
    return {**meta, "gt": [meta["gt"][0]] * lead + meta["gt"]}


# ------------------------------------------------------------------ scoring
def score(gt, fps, t0, stops, continues):
    """gt: truth per frame (m or None). t0: wall time of frame 0. stops / continues: wall start times of
    "Stop..." and "Path is clear..." utterances. Returns the run's verdict and numbers."""
    def frame(t):
        return min(len(gt) - 1, max(0, int((t - t0) * fps)))

    end = t0 + len(gt) / fps
    cross = next((i for i, g in enumerate(gt) if g is not None and g <= CROSS_M), None)
    r = {"expect_stop": cross is not None, "stops": len(stops), "false_stops": 0, "stop_latency_s": None,
         "stop_truth_m": None, "continue": None, "early_continue": False, "missed_stop": False, "late": False}
    episodes = []
    for s in sorted(stops):                       # repeats while still stopped (every 4 s) are one episode
        if not episodes or any(episodes[-1] < c < s for c in continues):
            episodes.append(s)
    for s in episodes:
        g = gt[frame(s)]
        if g is None or g > FALSE_M:
            r["false_stops"] += 1
        elif r["stop_latency_s"] is None and cross is not None:
            r["stop_latency_s"] = round(s - (t0 + cross / fps), 3)
            r["stop_truth_m"] = g
    if cross is not None:
        r["missed_stop"] = r["stop_latency_s"] is None
        r["late"] = r["stop_latency_s"] is not None and r["stop_latency_s"] > LATE_S
        last_near = max(i for i, g in enumerate(gt) if g is not None and g <= FAR_M)
        clear = last_near + 1 if last_near + 1 < len(gt) else None
        first_stop = min(stops) if stops else None
        conts = [c for c in continues if first_stop is not None and c > first_stop and c <= end + 0.5]
        r["early_continue"] = any(c < t0 + (last_near + 1) / fps for c in conts)
        if clear is not None and end - (t0 + clear / fps) >= 3.0 and not r["missed_stop"]:
            ok = [c for c in conts if c >= t0 + clear / fps]
            r["continue"] = round(ok[0] - (t0 + clear / fps), 3) if ok else "missed"
    r["pass"] = (not r["missed_stop"] and not r["late"] and r["false_stops"] == 0 and not r["early_continue"]
                 and r["continue"] != "missed")
    return r


def selftest() -> None:
    gt = [None] * 10 + [3.0 - 0.1 * i for i in range(20)] + [1.1] * 20 + [None] * 60   # 10 fps
    # stop 0.2 s after the 1.5 m frame (index 25), continue 2 s after clearing (index 50)
    r = score(gt, 10, 100.0, [102.7], [107.0])
    assert r["pass"] and math.isclose(r["stop_latency_s"], 0.2) and math.isclose(r["continue"], 2.0), r
    r = score(gt, 10, 100.0, [102.7, 106.7], [107.0])          # a repeat "Stop" while stopped: one episode
    assert r["pass"] and r["false_stops"] == 0, r
    assert score(gt, 10, 100.0, [], [])["missed_stop"]
    assert score(gt, 10, 100.0, [101.0, 102.7], [107.0])["false_stops"] == 1   # at 2.9 m: beyond 2.0 m
    assert score(gt, 10, 100.0, [102.7], [104.0])["early_continue"]            # still standing there
    assert score(gt, 10, 100.0, [102.7], [])["continue"] == "missed"
    assert score(gt, 10, 100.0, [104.0], [107.0])["late"]
    none = [None] * 50 + [2.6] * 50
    assert score(none, 10, 0.0, [], [])["pass"] and not score(none, 10, 0.0, [6.0], [])["pass"]
    print("selftest ok")


# ------------------------------------------------------------------ processes
def port_free(port: int, kind) -> bool:
    s = socket.socket(socket.AF_INET, kind)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # as the earpiece's server does: TIME_WAIT is free
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


class Earpiece:
    """Headless earpiece on the bench's own ports, without Cartesia; collects `spoken` and `continue` over WS."""

    def __init__(self, log_path: Path):
        if not EAR_PY.exists():
            sys.exit(f"missing {EAR_PY}: scripts/check.sh earpiece builds it")
        if not (port_free(UDP_PORT, socket.SOCK_DGRAM) and port_free(WS_PORT, socket.SOCK_STREAM)):
            sys.exit(f"ports {UDP_PORT}/{WS_PORT} are taken; the bench never stops other people's processes")
        env = {**os.environ, "CARTESIA_API_KEY": "", "PYTHONUNBUFFERED": "1"}   # "" beats the repo .env
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen([str(EAR_PY), "-m", "earpiece", "run", "--null", "--no-stem", "--no-judgment",
                                      "--udp-port", str(UDP_PORT), "--ws-port", str(WS_PORT)],
                                     cwd=ROOT / "earpiece", env=env, stdin=subprocess.DEVNULL,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        self.spoken, self.lock = [], threading.Lock()
        threading.Thread(target=self._ws, daemon=True).start()
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.beat = True
        threading.Thread(target=self._heartbeat, daemon=True).start()
        t = time.time()
        while not self.spoken and time.time() - t < 90:     # first `state` means the WS is up
            if self.proc.poll() is not None:
                sys.exit(f"earpiece exited; see {log_path}")
            time.sleep(0.2)

    def _ws(self) -> None:
        from websockets.sync.client import connect
        while self.proc.poll() is None:
            try:
                with connect(f"ws://127.0.0.1:{WS_PORT}", open_timeout=2, max_size=None) as ws:
                    for raw in ws:
                        m = json.loads(raw)
                        with self.lock:
                            if m.get("type") in ("spoken", "continue") or not self.spoken:
                                self.spoken.append(m)
            except (OSError, Exception):
                time.sleep(0.3)

    def _heartbeat(self) -> None:
        """Between walks the detector restarts; a live camera on an empty aisle would keep sending heartbeats."""
        while self.proc.poll() is None:
            if self.beat:
                self.udp.sendto(b'{"type":"heartbeat","source":"standin-chest"}', ("127.0.0.1", UDP_PORT))
            time.sleep(0.1)

    def said(self, since: float):
        with self.lock:
            return [m for m in self.spoken if m.get("type") == "spoken" and m.get("t_start", 0) >= since]

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def walk(ear: Earpiece, clip: Path, meta: dict, log_path: Path, extra: list[str], detector: Path) -> dict:
    g = meta["geom"]
    cmd = [sys.executable, str(detector), "--source", str(clip), "--camera", "chest",
           "--fx", str(g["fx"]), "--cam-height", str(g["cam_height"]), "--pitch", str(g["pitch"]), *extra]
    env = {**os.environ, "KITE_EARPIECE_HOST": "127.0.0.1", "KITE_EARPIECE_PORT": str(UDP_PORT)}
    started = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t0, fps = None, None
        for line in p.stdout:
            log.write(line)
            if line.startswith("[standin] t0_wall="):
                t0 = float(line.split("=")[1])
                ear.beat = False
            elif " inferences, " in line:               # "[standin] 189 inferences, 202.9 ms each (4.9 fps) ..."
                fps = float(line.split("(")[1].split()[0])
                ear.beat = True
        p.wait()
    ear.beat = True
    time.sleep(SETTLE_S)                         # the last utterance is reported when it finishes
    if t0 is None:
        return {"error": f"detector gave no t0 (exit {p.returncode}); see {log_path}"}
    said = ear.said(started)
    stops = [m["t_start"] for m in said if m["text"].startswith("Stop")]
    conts = [m["t_start"] for m in said if m["text"].startswith("Path is clear")]
    r = score(meta["gt"], meta["fps"], t0, stops, conts)
    r["said"] = [{"t": round(m["t_start"] - t0, 2), "text": m["text"], "result": m.get("result")} for m in said]
    r["detector_fps"], r["gt"], r["fps"] = fps, meta["gt"], meta["fps"]       # gt + said: rescore without rerunning
    return r


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)] if xs else None


def table(rows: list[dict]) -> str:
    def line(name, rs):
        lat = [r["stop_latency_s"] for r in rs if r.get("stop_latency_s") is not None]
        exp = [r for r in rs if r.get("expect_stop")]
        f = lambda v: "-" if v is None else f"{v:+.2f}"       # noqa: E731
        return (f"{name:<18}{len(rs):>5}{sum(r.get('pass', False) for r in rs):>6}"
                f"{sum(r.get('missed_stop', False) or r.get('late', False) for r in exp):>8}"
                f"{sum(r.get('false_stops', 0) > 0 for r in rs):>7}{f(pct(lat, .5)):>8}{f(pct(lat, .95)):>8}"
                f"{sum(r.get('continue') == 'missed' for r in exp):>9}{sum(r.get('early_continue', False) for r in rs):>7}"
                f"{sum('error' in r for r in rs):>5}")
    head = f"{'clip':<18}{'runs':>5}{'pass':>6}{'missed':>8}{'false':>7}{'p50 s':>8}{'p95 s':>8}{'no-cont':>9}{'early':>7}{'err':>5}"
    out = [head, "-" * len(head)]
    for name in dict.fromkeys(r["clip"] for r in rows):
        out.append(line(name, [r for r in rows if r["clip"] == name]))
    out += ["-" * len(head), line("all", rows)]
    n = len(rows)
    ok = sum(r.get("pass", False) for r in rows)
    out.append(f"success {ok}/{n} = {ok / max(n, 1):.0%}   false-stop runs {sum(r.get('false_stops', 0) > 0 for r in rows)}/{n}"
               f"   missed or late stops {sum(r.get('missed_stop', False) or r.get('late', False) for r in rows)}"
               f"   missed continues {sum(r.get('continue') == 'missed' for r in rows)}"
               f"   detector fps p50 {pct([r['detector_fps'] for r in rows if r.get('detector_fps')], .5)}")
    return "\n".join(out)


def run(args) -> None:
    names = [n for n, *_ in STOCK] + list(SCENARIOS)
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    missing = [n for n in names if not n.startswith("syn_") and not (BENCH / f"{n}.json").exists()]
    if missing or not (BENCH / "bg.png").exists():
        sys.exit("no bench clips yet: run `scripts/live_walk_bench.py prepare` once (needs internet)")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = OUT / f"{args.tag}-{stamp}"
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    ear = Earpiece(work / "earpiece.log")
    rows = []
    try:
        for k in range(args.runs):
            name = names[k % len(names)]
            clip = work / f"{k:02d}-{name}.mp4"
            r_rng = random.Random(rng.random())
            meta = render_synth(name, r_rng, clip) if name.startswith("syn_") else variant(name, r_rng, clip)
            r = {"run": k, "clip": name, **walk(ear, clip, meta, work / f"{k:02d}-{name}.log", args.detector_args.split(),
                                                    Path(args.detector))}
            rows.append(r)
            verdict = "PASS" if r.get("pass") else "FAIL"
            said = "; ".join(f"{s['t']:.1f} {s['text']}" for s in r.get("said", []))
            print(f"{k + 1:>2}/{args.runs} {name:<17} {verdict}  stop {r.get('stop_latency_s')} s at "
                  f"{r.get('stop_truth_m')} m  continue {r.get('continue')}  false {r.get('false_stops')}  "
                  f"{r.get('error', '')}{said[:160]}", flush=True)
            if not args.keep:
                clip.unlink()
    finally:
        ear.close()
    report = {"tag": args.tag, "when": stamp, "seed": args.seed, "runs": rows, "table": table(rows),
              "settings": {"syn": SYN, "real": REAL, "cross_m": CROSS_M, "false_m": FALSE_M, "far_m": FAR_M, "late_s": LATE_S,
                           "detector_args": args.detector_args, "detector": args.detector}}
    path = OUT / f"{args.tag}-{stamp}.json"
    path.write_text(json.dumps(report, indent=1))
    print("\n" + report["table"] + f"\n\nreport: {path}\nlogs: {work}")


def rescore(args) -> None:
    """Score a saved report again with the current rules (truth is rebuilt from the seed if it wasn't saved)."""
    rep = json.loads(Path(args.report).read_text())
    rng, rows = random.Random(rep["seed"]), []
    for r in rep["runs"]:
        r_rng = random.Random(rng.random())
        if "said" not in r:
            rows.append(r)
            continue
        if "gt" not in r:
            tmp = OUT / "rescore.mp4"
            meta = render_synth(r["clip"], r_rng, tmp) if r["clip"].startswith("syn_") else variant(r["clip"], r_rng, tmp)
            r = {**r, "gt": meta["gt"], "fps": meta["fps"]}
        stops = [s["t"] for s in r["said"] if s["text"].startswith("Stop")]
        conts = [s["t"] for s in r["said"] if s["text"].startswith("Path is clear")]
        rows.append({**r, **score(r["gt"], r["fps"], 0.0, stops, conts)})
    print(table(rows))
    if args.write:
        Path(args.report).write_text(json.dumps({**rep, "runs": rows, "table": table(rows),
                                                 "rescored": time.strftime("%Y%m%d-%H%M%S")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="download stock clips, cut them, measure their truth, make sprites")
    p.add_argument("--force", action="store_true", help="recompute the truth files")
    r = sub.add_parser("run", help="N live walks through detector + earpiece")
    r.add_argument("--runs", type=int, default=40)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--only", help="comma-separated clip names")
    r.add_argument("--tag", default="bench", help="report name prefix, e.g. before / after")
    r.add_argument("--keep", action="store_true", help="keep the rendered clips in out/bench/")
    r.add_argument("--detector-args", default="", help="extra standin_detect.py flags, e.g. '--stop-m 1.8'")
    r.add_argument("--detector", default=str(ROOT / "sim" / "standin_detect.py"),
                   help="detector script, e.g. a copy of another branch's sim/ at the repo root, to compare")
    s = sub.add_parser("rescore", help="score a saved report again with the current rules")
    s.add_argument("report")
    s.add_argument("--write", action="store_true", help="save the new scores into the report")
    sub.add_parser("selftest", help="check the scoring rules")
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "rescore": rescore, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
