"""End to end on one computer: body camera (UNO Q) -> near priorities -> earpiece -> beeps. No board, no camera,
nothing played aloud, no paid API calls.

    python -m perception.near.e2e_bodycam                       # doorway, oncoming, car_turning clips, ~25 s
    python -m perception.near.e2e_bodycam --clips data/imagined/doorway.mp4 --seconds 12
    python -m perception.near.e2e_bodycam --board unoq            # the REAL board: walk toward its camera
    python -m perception.near.e2e_bodycam --board 10.31.226.210   # same, by IP (the token saved by kite_link.py)

Every piece is the real code, wired the way it runs on the street:
  1. board.py, the UNO Q's own server, in this process, with its camera swapped for a clip player (the board's
     frame hub, token check and /video.mjpg are untouched);
  2. `python -m perception.near --source unoq:127.0.0.1:<port>`: finds the board, streams its camera, runs
     YOLO11n + Depth Anything, and sends the earpiece its `obstacles` messages over UDP;
  3. a UDP tee between them that records every message with its arrival time;
  4. `python -m earpiece run --null` (no audio device, no Cartesia, Claude or Moss keys): its sonar must track
     what near sent, read live from its WebSocket state;
  5. `python -m earpiece render` of the recorded messages: the beeps the walker would have heard, as a WAV,
     measured (pitch and spacing per priority) instead of listened to.

Prints PASS / FAIL per check and writes everything to --out (default out/e2e_bodycam/).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINK_DIR = ROOT / "hardware" / "arduino" / "unoq_link"
EAR_PY = ROOT / "earpiece" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
NEAR_CMD = ["uv", "run", "-q", "--no-project", "--python", "3.12", "--with", "ultralytics", "--with", "lap",
            "--with", "transformers", "--with", "torch", "--with", "scipy", "--with", "opencv-python-headless",
            "python", "-m", "perception.near"]
DEFAULT_CLIPS = ["doorway", "oncoming", "car_turning"]
NO_KEYS = ("CARTESIA_API_KEY", "ANTHROPIC_API_KEY", "MOSS_API_KEY", "MOSS_PROJECT_ID", "MOSS_PROJECT_KEY",
           "XAI_API_KEY", "GOOGLE_MAPS_API_KEY", "GEMINI_API_KEY", "SPACETIME_TOKEN")


def free_port(kind=socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_clip(name: str) -> Path:
    for c in (Path(name), ROOT / "data" / "imagined" / f"{name}.mp4", ROOT.parents[2] / "data" / "imagined" / f"{name}.mp4"):
        if c.is_file():
            return c.resolve()
    raise SystemExit(f"clip not found: {name}")


# ------------------------------------------------------------------ 1. the board, playing clips as its camera
class ClipCamera:
    """Plays clips back to back at their own speed from the first frame asked for, JPEG like the board's camera."""

    def __init__(self, clips: list[Path], quality: int = 80) -> None:
        import cv2
        self.cv2, self.clips, self.quality = cv2, clips, quality
        self.name = "usb camera clip"              # anything but "synthetic": the board reports a real camera
        self.t0: float | None = None
        self.idx, self.cap, self.fidx, self.vfps, self.clip_t0 = -1, None, 0, 24.0, 0.0
        self.finished = threading.Event()
        self.size = (640, 480)

    def _next_clip(self, now: float) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.idx += 1
        if self.idx >= len(self.clips):
            self.cap = None
            self.finished.set()
            return False
        self.cap = self.cv2.VideoCapture(str(self.clips[self.idx]))
        self.vfps = self.cap.get(self.cv2.CAP_PROP_FPS) or 24.0
        self.fidx, self.clip_t0 = 0, now
        return True

    def read(self) -> bytes | None:
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
            self._next_clip(now)
        if self.cap is None:
            return None
        want = int((now - self.clip_t0) * self.vfps)
        img = None
        while self.fidx <= want:
            ok, frame = self.cap.read()
            if not ok:
                if not self._next_clip(now):
                    return None
                return self.read()
            img, self.fidx = frame, self.fidx + 1
        if img is None:
            return None
        ok, buf = self.cv2.imencode(".jpg", img, [self.cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else None

    def timeline(self) -> list[tuple[str, float]]:
        return [(c.stem, i * 6.0) for i, c in enumerate(self.clips)]

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def start_board(token: str, clips: list[Path]):
    sys.path.insert(0, str(LINK_DIR))
    import board  # noqa: PLC0415
    cfg = board.Config({"KITE_PORT": "0", "KITE_BIND": "127.0.0.1", "KITE_TOKEN": token, "KITE_CAMERA": "synthetic",
                        "KITE_CLOUD": "0", "KITE_VIDEO_FPS": "10", "KITE_SOURCE": "unoq"})
    b = board.Board(cfg)
    cam = ClipCamera(clips)
    b.hub.camera.close()
    b.hub.camera, b.hub.reason = cam, "ok"
    srv = board.start(b)
    return board, b, srv, cam


# ------------------------------------------------------------------ 3. the UDP tee
class Tee(threading.Thread):
    def __init__(self, listen: int, forward: int) -> None:
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", listen))
        self.sock.settimeout(0.2)
        self.fwd = ("127.0.0.1", forward)
        self.msgs: list[dict] = []
        self.stop = threading.Event()
        self.t0: float | None = None

    def run(self) -> None:
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while not self.stop.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                return
            now = time.monotonic()
            out.sendto(data, self.fwd)
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            if self.t0 is None:
                self.t0 = now
            self.msgs.append({"t": round(now - self.t0, 3), "arrived": now, **msg})


# ------------------------------------------------------------------ 4. the earpiece's live state
class StateWatch(threading.Thread):
    def __init__(self, port: int) -> None:
        super().__init__(daemon=True)
        self.port, self.states, self.stop, self.error = port, [], threading.Event(), None

    def run(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        import websockets
        deadline = time.monotonic() + 60
        while not self.stop.is_set():
            try:
                async with websockets.connect(f"ws://127.0.0.1:{self.port}", max_size=2**22) as ws:
                    while not self.stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), 0.5)
                        except TimeoutError:
                            continue
                        m = json.loads(raw)
                        if m.get("type") == "state":
                            self.states.append({"at": time.monotonic(), "tracks": m.get("tracks") or [],
                                                "stop": m.get("stop_in_force")})
            except (OSError, ConnectionError) as e:
                self.error = str(e)
                if time.monotonic() > deadline:
                    return
                await asyncio.sleep(0.5)
            except Exception as e:  # noqa: BLE001  (a closed socket at shutdown)
                self.error = f"{type(e).__name__}: {e}"
                await asyncio.sleep(0.5)


# ------------------------------------------------------------------ 5. measure the rendered WAV
def beeps(wav: Path) -> list[tuple[float, float, float]]:
    """(onset s, duration s, dominant Hz) for each beep in the render (any channel mix)."""
    import numpy as np
    from scipy.io import wavfile
    sr, x = wavfile.read(wav)
    x = x.astype(float)
    x = x.mean(axis=1) if x.ndim > 1 else x
    if not np.abs(x).max():
        return []
    w = max(1, int(sr * 0.004))
    env = np.convolve(np.abs(x), np.ones(w) / w, "same")
    on = env > 0.12 * env.max()
    edges = np.flatnonzero(np.diff(on.astype(np.int8)))
    if on[0]:
        edges = np.r_[0, edges]
    if on[-1]:
        edges = np.r_[edges, len(on) - 1]
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < sr * 0.01:
            continue
        seg = x[a:b] * np.hanning(b - a)
        spec = np.abs(np.fft.rfft(seg, n=max(8192, len(seg))))
        hz = np.fft.rfftfreq(max(8192, len(seg)), 1 / sr)
        band = (hz > 200) & (hz < 4000)
        out.append((a / sr, (b - a) / sr, float(hz[band][spec[band].argmax()])))
    return out


def priority_at(msgs: list[dict], t: float, lag: float = 0.15) -> int | None:
    """The top priority in force at render time t (render time 0 = the first message)."""
    p = None
    for m in msgs:
        if m["t"] > t - lag:
            break
        ps = [o.get("priority") for o in m.get("objects", []) if o.get("priority")]
        p = min(ps) if ps else None
    return p


# ------------------------------------------------------------------ the run
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--seconds", type=float, default=0, help="near's run time after its models load (default: "
                                                             "the clips' length + 4 s)")
    ap.add_argument("--depth", default="outdoor", help="passed to perception.near --depth")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "out" / "e2e_bodycam"))
    ap.add_argument("--board", help="test the real UNO Q instead of a local board.py: unoq (find it) or its HOST[:PORT]; "
                                    "uses the token saved by kite_link.py. Walk toward the camera while it runs.")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [] if a.board else [find_clip(c) for c in a.clips]
    seconds = a.seconds or (30.0 if a.board else 6.0 * len(clips) + 4.0)
    results: list[tuple[bool, str, str]] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        results.append((bool(ok), what, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {what}" + (f": {detail}" if detail else ""), flush=True)

    board = b = srv = cam = None
    if a.board:
        token = None
        source = a.board if a.board.lower().startswith("unoq") else f"unoq:{a.board}"
        print(f"[e2e] the real board ({source}). After the models load (~20 s), walk slowly toward its camera from "
              "about 6 m away, stop about 1 m from it, then step aside.", flush=True)
    else:
        token = secrets.token_urlsafe(24)
        board, b, srv, cam = start_board(token, clips)
        source = f"unoq:127.0.0.1:{srv.server_address[1]}"
        print(f"[e2e] board.py on {source[5:]}, camera = {', '.join(c.stem for c in clips)}", flush=True)

    udp_ear, ws_ear, udp_tee = free_port(socket.SOCK_DGRAM), free_port(), free_port(socket.SOCK_DGRAM)
    env = {**os.environ, "KITE_NO_DEVICE_TESTS": "1", **{k: "" for k in NO_KEYS}}
    ear_log = open(out / "earpiece.log", "w")   # noqa: SIM115  (closed in finally)
    ear = subprocess.Popen([str(EAR_PY), "-m", "earpiece", "run", "--null", "--no-judgment", "--no-prewarm",
                            "--no-stem", "--udp-port", str(udp_ear), "--ws-port", str(ws_ear)],
                           cwd=ROOT / "earpiece", env=env, stdout=ear_log, stderr=subprocess.STDOUT)
    tee = Tee(udp_tee, udp_ear)
    tee.start()
    watch = StateWatch(ws_ear)
    watch.start()

    near_jsonl = out / "near.jsonl"
    near_env = {**env, "KITE_EARPIECE_HOST": "127.0.0.1"}
    if token:
        near_env.update(KITE_TOKEN=token, KITE_NO_CLOUD="1")
    near_cmd = NEAR_CMD + ["--source", source, "--ear-port", str(udp_tee), "--depth", a.depth,
                           "--max-seconds", str(seconds), "--jsonl", str(near_jsonl)]
    if a.device != "auto":
        near_cmd += ["--device", a.device]
    print("[e2e] " + " ".join(near_cmd[-14:]), flush=True)
    t_near = time.monotonic()
    try:
        near = subprocess.run(near_cmd, cwd=ROOT, env=near_env, capture_output=True, text=True,
                              timeout=seconds + 600)
    finally:
        time.sleep(1.0)
        tee.stop.set()
        watch.stop.set()
        ear.terminate()
        try:
            ear.wait(10)
        except subprocess.TimeoutExpired:
            ear.kill()
        ear_log.close()
        if board is not None:
            board.shutdown(b, srv)
            cam.close()
    (out / "near.log").write_text(near.stdout + near.stderr)
    near_lines = [ln for ln in (near.stdout + near.stderr).splitlines() if ln.startswith(("[near]", "t="))]
    print("\n".join("    " + ln for ln in near_lines[-12:]), flush=True)

    # 2. near: reached the board, saw frames, produced priorities
    check(near.returncode == 0, "near ran to the end", f"exit {near.returncode}")
    check(any("[near] UNO Q at " in ln for ln in near_lines), "near found the board over HTTP with the token",
          next((ln[7:] for ln in near_lines if "[near] UNO Q at " in ln), ""))
    if a.board:
        check(not any("test pattern" in ln for ln in near_lines), "the board is sending its camera, not the test pattern")
    evs = [json.loads(ln) for ln in near_jsonl.read_text().splitlines() if ln.strip()] if near_jsonl.exists() else []
    run_s = max(1e-6, (evs[-1]["t"] - evs[0]["t"]) if len(evs) > 1 else 0)
    fps = (len(evs) - 1) / run_s if len(evs) > 1 else 0
    check(len(evs) >= 10, "near processed the board's frames", f"{len(evs)} frames, {fps:.1f} fps")
    seen = {e.get("priority") for e in evs}
    check(1 in seen, "near raised P1 (something under 2 m ahead)", f"priorities seen: {sorted(p for p in seen if p)}")

    # 3. the messages on the wire
    obs = [m for m in tee.msgs if m.get("type") == "obstacles"]
    seqs = [m.get("seq") for m in obs]
    check(bool(obs) and len(obs) >= len(evs) and len(set(seqs)) == len(seqs),
          "every near frame reached the earpiece as `obstacles` (plus steady repeats)",
          f"{len(obs)} messages for {len(evs)} frames, seq unique")
    gaps = [b2["arrived"] - b1["arrived"] for b1, b2 in zip(obs, obs[1:])]
    worst = max(gaps) if gaps else 0
    gaps.sort()
    p95 = gaps[int(0.95 * (len(gaps) - 1))] if gaps else 0
    check(p95 < 0.6, "updates arrive faster than the sonar's 0.6 s fade", f"p95 gap {p95:.2f} s, worst {worst:.2f} s")
    with open(out / "obstacles.jsonl", "w") as fh:
        for m in obs:
            fh.write(json.dumps({k: v for k, v in m.items() if k != "arrived"}) + "\n")
    by_p = {p: sum(1 for m in obs if any(o.get("priority") == p for o in m.get("objects", []))) for p in (1, 2, 3)}

    # 4. the earpiece live
    tracked = [s for s in watch.states if s["tracks"]]
    kinds = sorted({t["kind"] for s in tracked for t in s["tracks"]})
    crit = sum(1 for s in tracked if any(t.get("critical") for t in s["tracks"]))
    check(len(watch.states) > 5, "earpiece ran headless and published its state", f"{len(watch.states)} states")
    check(bool(tracked), "earpiece sonar tracked near's objects", f"{len(tracked)} states with tracks, kinds {kinds}, "
                                                                  f"{crit} with a critical track")
    ear_text = (out / "earpiece.log").read_text(errors="replace")
    check("Traceback" not in ear_text, "earpiece log has no errors")

    # 5. what the walker hears
    wav = out / "heard.wav"
    r = subprocess.run([str(EAR_PY), "-m", "earpiece", "render", "--jsonl", str(out / "obstacles.jsonl"), str(wav)],
                       cwd=ROOT / "earpiece", env=env, capture_output=True, text=True, timeout=300)
    check(r.returncode == 0 and wav.exists(), "earpiece rendered the walk to WAV", (r.stdout + r.stderr).strip()[-160:])
    found = beeps(wav) if wav.exists() else []
    per: dict[int, list[tuple[float, float]]] = {1: [], 2: [], 3: []}
    for onset, _dur, hz in found:
        p = priority_at(obs, onset)
        if p in per:
            per[p].append((onset, hz))
    summary = {}
    for p, items in per.items():
        if not items:
            continue
        hz = sorted(h for _, h in items)
        onsets = [t for t, _ in items]
        gaps_p = sorted(t2 - t1 for t1, t2 in zip(onsets, onsets[1:]) if t2 - t1 < 2.0)
        summary[p] = {"beeps": len(items), "median_hz": round(hz[len(hz) // 2]),
                      "median_gap_s": round(gaps_p[len(gaps_p) // 2], 2) if gaps_p else None,
                      "messages": by_p[p]}
    print(f"[e2e] beeps by priority: {json.dumps(summary)}", flush=True)
    check(1 in summary and summary[1]["beeps"] >= 3, "P1 frames produced beeps", json.dumps(summary.get(1)))
    # the pitch and rate each priority got, from the live earpiece's own track urgency (exact; a beep's loudest
    # FFT bin is not its pitch, every object kind has its own timbre)
    sys.path.insert(0, str(ROOT / "earpiece"))
    from earpiece.config import SonarCfg
    from earpiece.sonar import PRIORITY_URGENCY
    c = SonarCfg()
    live: dict[int, set[float]] = {1: set(), 2: set(), 3: set()}
    for st in watch.states:
        sent = [m for m in obs if m["arrived"] <= st["at"]]
        if not sent:
            continue
        prio = {o["id"]: o.get("priority") for o in sent[-1].get("objects", [])}
        for tr in st["tracks"]:
            pr = prio.get(str(tr["id"]).split(":", 1)[-1])
            if pr in live and tr.get("urgency") is not None:
                live[pr].add(round(float(tr["urgency"]), 3))
    tones = {}
    for pr, us in live.items():
        if us:
            u = max(us, key=lambda x: sum(1 for v in us if v == x))
            tones[pr] = {"urgencies": sorted(us), "hz": round(c.pitch_low_hz + (c.pitch_high_hz - c.pitch_low_hz) * u),
                         "every_s": round(c.interval_slow_s - (c.interval_slow_s - c.interval_fast_s) * u ** 0.8, 2)}
    print(f"[e2e] live sonar tones by priority: {json.dumps(tones)}", flush=True)
    for pr, tone in tones.items():
        check(tone["urgencies"] == [PRIORITY_URGENCY[pr]], f"P{pr} always gets the P{pr} tone in the live earpiece",
              f"{tone['hz']} Hz every {tone['every_s']} s (urgencies seen {tone['urgencies']})")
    hzs = [tones[p]["hz"] for p in (1, 2, 3) if p in tones]
    check(len(hzs) >= 2 and hzs == sorted(hzs, reverse=True) and len(set(hzs)) == len(hzs),
          "each priority has its own frequency, highest for P1", " > ".join(f"P{p} {tones[p]['hz']} Hz" for p in tones))
    if 1 in summary and 2 in summary and summary[1]["median_gap_s"] and summary[2]["median_gap_s"]:
        check(summary[1]["median_gap_s"] < summary[2]["median_gap_s"], "P1 beeps faster than P2",
              f"{summary[1]['median_gap_s']} s vs {summary[2]['median_gap_s']} s")
    caps = wav.with_suffix(".captions.jsonl")
    said = [json.loads(ln)["text"] for ln in caps.read_text().splitlines() if ln.strip()] if caps.exists() else []

    report = {"clips": [c.name for c in clips], "tones": tones, "near_frames": len(evs), "near_fps": round(fps, 2),
              "obstacles_messages": len(obs), "messages_by_priority": by_p, "gap_p95_s": round(p95, 3),
              "earpiece_states": len(watch.states), "earpiece_tracked_states": len(tracked), "kinds": kinds,
              "beeps": summary, "spoken": said, "near_wall_s": round(time.monotonic() - t_near, 1),
              "checks": [{"ok": ok, "what": w, "detail": d} for ok, w, d in results]}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    failed = [w for ok, w, _ in results if not ok]
    print(f"\n[e2e] {len(results) - len(failed)}/{len(results)} checks passed; spoken: {said[:6]}; "
          f"report {out / 'report.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
