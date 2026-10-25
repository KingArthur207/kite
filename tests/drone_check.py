"""Dry test and flight test of kite's drone layer: every command goes through TelloDrone + SafetyGovernor.

    python tests/drone_check.py                  # DRY TEST (default): camera window, link and stream rates, then every
                                                 # movement in dry run: nothing reaches the motors
    python tests/drone_check.py --fly            # FLIGHT TEST: takeoff, each stick axis both ways, 30 cm moves in all
                                                 # six directions, 45 deg turns both ways, land (~1.5 min)
    python tests/drone_check.py --fly --no-moves # flight test with the stick steps only (~45 s)

Prints a PASS / FAIL line per step and a summary. In the flight test each stick step also checks the drone moved the
right way in its telemetry (yaw angle, height, pitch / roll tilt), and every stick step and move is also judged by
the drone's own camera (optical flow: the scene expands flying forward, shifts left flying right, down climbing), so
the verdict does not depend on where the pilot stands; a DIRECTION SUMMARY ends the flight. Like tests/drone_test.py
(the raw djitellopy latency test) but through kite.drone, so it checks the layer the rest of kite flies with.
Safety: 3 x 3 m clear space, 2.5 m ceiling, prop guards, a spotter. Ctrl+C lands. Works on macOS, Windows and Linux.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the kite package next to this file

WIN = "kite drone check  -  q closes the window, Ctrl+C in the terminal lands"
# stick step: name, rc vector (lr, fb, ud, yaw), telemetry field that must change, sign it must change with.
# yaw and height are checked strictly. Tilt only has to be clearly there; its sign is compared with kite's convention
# (pitch + = nose up, so forward tilts nose down; roll + = right side down), which the real drone has yet to confirm.
STICK_STEPS = [
    ("yaw right", (0, 0, 0, 1), "yaw", +1), ("yaw left", (0, 0, 0, -1), "yaw", -1),
    ("up", (0, 0, 1, 0), "h", +1), ("down", (0, 0, -1, 0), "h", -1),
    ("forward", (0, 1, 0, 0), "pitch", -1), ("back", (0, -1, 0, 0), "pitch", +1),
    ("right", (1, 0, 0, 0), "roll", +1), ("left", (-1, 0, 0, 0), "roll", -1),
]
MOVES = ["forward", "back", "left", "right", "up", "down"]

# ---------------------------------------------------------------------------------------- camera direction check
# The drone's own evidence of which way it moved, whatever side the pilot stands on: optical flow in its camera.
# Seen from the camera, a motion shows as:  forward  the scene EXPANDS      back  it contracts
#                                           right    it shifts LEFT        left  it shifts right
#                                           up       it shifts DOWN        down  it shifts up
#                                           yaw right (cw) it also shifts left: steps are judged by their command type.
CAMERA_WHERE = ("the camera is the drone's front: stand BEHIND the drone, camera facing away from you, to judge "
                "directions by eye (facing the camera, forward / back and left / right look swapped)")
CAM_W = 160  # optical flow on small grey frames: cheap, and blind to H.264 block noise
GRAD_MIN = 4.0  # grey levels per pixel: a pixel with less gradient carries no motion information
MIN_TEXTURE = 0.004  # fraction of textured pixels a frame pair needs
CAM_BASELINE_S = 0.5  # hover watched before each step's command: the camera's noise level
FLOW_LEVELS, FLOW_WIN = 5, 21  # Farneback pyramid depth and window: a fast turn moves the scene ~20 px a frame
# axis -> (index in (tx, ty, s, w), sign turning it into "+ = the first word", words, smallest total that counts,
# what the number means). tx / ty: scene shift in px on a CAM_W-wide image; s: scale change (+ = expands).
AXES = {
    "fb": (2, +1, ("forward", "back"), 0.015, "scale"),
    "lr": (0, -1, ("right", "left"), 1.5, "shift"),
    "ud": (1, +1, ("up", "down"), 1.5, "shift"),
    "yaw": (0, -1, ("yaw right", "yaw left"), 1.5, "shift"),
}
STEP_AXIS = {"forward": ("fb", +1), "back": ("fb", -1), "right": ("lr", +1), "left": ("lr", -1),
             "up": ("ud", +1), "down": ("ud", -1), "yaw right": ("yaw", +1), "yaw left": ("yaw", -1)}
TEL_AXIS = {"fb": ("vx", "pitch_deg"), "lr": ("vy", "roll_deg"), "ud": ("vz", None), "yaw": (None, None)}


def flow_params(a, b):
    """Motion between two grey frames: dense Farneback flow, fitted with u = tx + s*dx - w*dy, v = ty + s*dy + w*dx
    (dx, dy from the image centre; w is a roll about the optical axis, there so a tilt does not read as a move).
    Each pixel's u counts with its horizontal gradient and v with its vertical one (a flat patch or the length of an
    edge says nothing). Returns ((tx, ty, s, w) or None when the pair has too little texture, textured fraction)."""
    import cv2
    import numpy as np

    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, FLOW_LEVELS, FLOW_WIN, 3, 5, 1.2, 0)
    gx = np.abs(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)) / 8.0
    gy = np.abs(cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    sel = np.hypot(gx, gy) > GRAD_MIN
    tex = float(sel.mean())
    if tex < MIN_TEXTURE:
        return None, tex
    h, w = a.shape
    ys, xs = np.nonzero(sel)
    dx, dy = xs - (w - 1) / 2.0, ys - (h - 1) / 2.0
    wx, wy = gx[sel], gy[sel]
    one, zero = np.ones_like(dx), np.zeros_like(dx)
    rows = np.concatenate([np.stack([one, zero, dx, -dy], 1) * wx[:, None],
                           np.stack([zero, one, dy, dx], 1) * wy[:, None]])
    rhs = np.concatenate([flow[..., 0][sel] * wx, flow[..., 1][sel] * wy])
    p, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    return tuple(float(x) for x in p), tex


class CameraMotion:
    """Frames of one step: hover frames (the noise baseline), mark() when the command goes out, then the motion."""

    def __init__(self):
        self.prev = None
        self.params: list = []  # (tx, ty, s, w) or None per consecutive frame pair
        self.start: int | None = None

    def add(self, image) -> None:
        import cv2

        h, w = image.shape[:2]
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        g = cv2.resize(g, (CAM_W, max(1, round(CAM_W * h / w))), interpolation=cv2.INTER_AREA)
        if self.prev is not None:
            self.params.append(flow_params(self.prev, g)[0])
        self.prev = g

    def mark(self) -> None:
        self.start = len(self.params)

    def verdict(self, axis: str, sign: int) -> dict:
        """measured: the direction word or None (inconclusive), conf: |motion| / what noise could explain."""
        import numpy as np

        k, ksign, words, floor, _ = AXES[axis]
        start = self.start or 0
        base = [p for p in self.params[:start] if p is not None]
        step = [p for p in self.params[start:] if p is not None]
        out = {"axis": axis, "commanded": words[0 if sign > 0 else 1], "measured": None, "conf": 0.0, "value": 0.0}
        if len(step) < 3 or len(step) < 0.5 * (len(self.params) - start):
            out["why"] = "too little texture in view" if len(self.params) - start >= 3 else "too few frames"
            return out
        total = ksign * float(sum(p[k] for p in step))
        thr = floor
        if len(base) >= 3:  # what hovering produced, scaled to the length of the step
            b = ksign * np.array([p[k] for p in base])
            thr = max(floor, 4.0 * float(b.std()) * len(step) ** 0.5 + abs(float(b.mean())) * len(step))
        out.update(value=total, conf=abs(total) / thr)
        if out["conf"] < 1.0:
            out["why"] = f"motion only x{out['conf']:.1f} of the noise"
            return out
        out["measured"] = words[0] if total > 0 else words[1]
        return out


def describe(v: dict) -> tuple[str, bool]:
    """'camera: forward (ok ...)' and whether it confidently says opposite."""
    unit = AXES[v["axis"]][4]
    amount = f"scale {100 * v['value']:+.1f}%" if unit == "scale" else f"{v['value']:+.1f} px/{CAM_W}"
    if v["measured"] is None:
        return f"camera: inconclusive ({v['why']})", False
    if v["measured"] == v["commanded"]:
        return f"camera: {v['measured']} (ok, {amount}, x{v['conf']:.1f} noise)", False
    if v["axis"] == "yaw":  # a fast turn can alias the flow; the yaw telemetry judges turns
        return f"camera: {v['measured']}? (disagrees, info only: turns are judged by the yaw telemetry; {amount})", False
    return f"camera: {v['measured'].upper()} - opposite of the command ({amount}, x{v['conf']:.1f} noise)", True


def direction_summary(rows: list[dict]) -> tuple[list[str], str]:
    """Per axis: commanded vs camera-measured directions over all steps, and the one-line verdict."""
    lines = ["DIRECTION SUMMARY (from the drone's own camera; telemetry peaks shown, their signs not assumed)"]
    opposite, mixed, unknown = [], [], []
    for axis in ("fb", "lr", "ud", "yaw"):
        mine = [r for r in rows if r["axis"] == axis]
        if not mine:
            continue
        ok = sum(r["measured"] == r["commanded"] for r in mine)
        bad = sum(r["measured"] is not None and r["measured"] != r["commanded"] for r in mine)
        pairs = ", ".join(f"{r['commanded']}->{r['measured'] or '?'}" for r in mine)
        lines.append(f"  {axis:<3} {ok}/{len(mine)} as commanded, {bad} opposite, {len(mine) - ok - bad} "
                     f"inconclusive   ({pairs})")
        vel, tilt = TEL_AXIS[axis]
        tel = "; ".join(f"{r['commanded']} {vel} {r['tel'].get(vel, float('nan')):+.2f}" for r in mine
                        if vel and r["tel"].get(vel) is not None)
        if tel:
            lines.append(f"        telemetry {vel} peaks: {tel}")
        if axis == "yaw":
            continue
        if bad > ok:
            opposite.append(axis)
        elif bad:
            mixed.append(axis)
        elif ok == 0:
            unknown.append(axis)
    if opposite:
        verdict = f"the drone really moves opposite on {', '.join(opposite)}"
    elif mixed:
        verdict = f"the camera saw mixed directions on {', '.join(mixed)}: repeat the test, judge from behind the drone"
    elif unknown:
        verdict = (f"camera inconclusive on {', '.join(unknown)} (too little texture or motion): judge from behind "
                   "the drone")
    else:
        verdict = "the drone moves as commanded; if it looked reversed, you were facing the camera"
    lines.append(f"  VERDICT: {verdict}")
    return lines, verdict


class Abort(Exception):
    """Ctrl+C or a safety landing: stop the sequence and land."""


class Check:
    """The sequence, run in a worker thread; the main thread only draws the window and forwards Ctrl+C."""

    def __init__(self, drone, src, args):
        self.d, self.src, self.args = drone, src, args
        self.abort = threading.Event()
        self.results: list[tuple[str, bool, str]] = []
        self.phase = "starting"
        self.last_frame_t: float | None = None
        self.last_seq = -1  # the reader's frame counter: counts frames even between our polls
        self.cam: CameraMotion | None = None  # frames of the step being watched
        self.peaks: dict | None = None  # signed telemetry peaks during that step
        self.directions: list[dict] = []  # one camera verdict per watched step, for the DIRECTION SUMMARY
        self.direction_verdict: str | None = None

    # ------------------------------------------------------------------ helpers
    def frame(self):
        f = self.src.read()
        if f is not None and f.seq != self.last_seq:
            self.last_frame_t, self.last_seq = f.t, f.seq
            if self.cam is not None:
                self.cam.add(f.image)
        return f

    def wait(self, seconds: float, rc=(0, 0, 0, 0)) -> None:
        """Safety check + rc at 20 Hz for `seconds` (the rc also keeps the Tello's 15 s auto-land away)."""
        t_end = time.time() + seconds
        while True:
            if self.abort.is_set():
                raise Abort("Ctrl+C")
            self.frame()
            if self.peaks is not None:
                self.sample_peaks()
            action = self.d.check(time.time(), self.last_frame_t)
            if action == "land":
                raise Abort("safety governor is landing (battery, flight time or a land request)")
            if self.d.flying and not self.d.busy():
                self.d.rc(*(rc if action == "ok" else (0, 0, 0, 0)))
            if time.time() >= t_end:
                return
            time.sleep(0.05)

    def until_done(self, timeout_s: float) -> str | None:
        """Wait for the running discrete command; its reply ('ok', 'error ...') or None on timeout."""
        t_end = time.time() + timeout_s
        while self.d.busy():
            if time.time() > t_end:
                return None
            self.wait(0.05)
        return self.d.last_result()

    def tel(self):
        try:
            return self.d.telemetry()
        except Exception:
            return None

    def sample_peaks(self) -> None:
        t = self.tel()
        for k in ("vx", "vy", "vz", "pitch_deg", "roll_deg"):
            v = getattr(t, k, None) if t is not None else None
            if v is not None and abs(v) > abs(self.peaks.get(k, 0.0)):
                self.peaks[k] = v

    def watch_begin(self) -> None:
        """Before a step's command: hover CAM_BASELINE_S with the camera on (its noise level), then watch the step."""
        if self.args.dry_run:
            return
        self.cam = CameraMotion()
        self.wait(CAM_BASELINE_S)
        self.cam.mark()
        self.peaks = {}

    def watch_end(self, step: str) -> tuple[str, bool]:
        """The camera's verdict on the step ('' in dry run) and whether it confidently says the opposite."""
        cam, peaks, self.cam, self.peaks = self.cam, self.peaks, None, None
        if cam is None:
            return "", False
        axis, sign = STEP_AXIS[step]
        v = cam.verdict(axis, sign)
        v["tel"] = peaks or {}
        self.directions.append(v)
        text, opposite = describe(v)
        p = v["tel"]
        tel = " ".join(f"{k.split('_')[0]} {p[k]:+.{2 if k[0] == 'v' else 0}f}" for k in
                       ("vx", "vy", "vz", "pitch_deg", "roll_deg") if k in p)
        return f" | {text}" + (f" | tel peaks {tel}" if tel else ""), opposite

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<22} {detail}", flush=True)

    # ------------------------------------------------------------------ steps
    def link_check(self) -> None:
        self.phase = "link check"
        tello = self.d.inner.tello
        rtts = []
        for _ in range(5):
            replies = tello.get_own_udp_object()["responses"]
            self.wait(0.15)
            replies.clear()
            t0 = time.perf_counter()  # not time.time(): on Windows (Python 3.12) it only ticks every ~16 ms
            tello.send_command_without_return("battery?")
            while not replies and time.perf_counter() - t0 < 3.0:
                time.sleep(0.001)
            if replies:
                rtts.append(time.perf_counter() - t0)
                replies.pop(0)
        self.record("command round trip", len(rtts) >= 4,
                    f"{1000 * sum(rtts) / len(rtts):.0f} ms over {len(rtts)}/5" if rtts else "no reply")
        self.frame()
        n0, t0 = self.last_seq, time.time()
        states, last = 0, None
        while time.time() - t0 < 3.0:
            self.wait(0.02)
            st = tello.get_current_state()
            if st is not last:
                states, last = states + 1, st
        fps, hz = (self.last_seq - n0) / 3.0 if n0 >= 0 else 0.0, states / 3.0
        self.record("video", fps >= 15, f"{fps:.1f} fps ({self.d.inner.video_backend} reader)")
        self.record("telemetry", hz >= 5, f"{hz:.1f} Hz")
        t = self.tel()
        self.record("battery", t is not None and t.battery_pct is not None and t.battery_pct >= self.args.min_battery,
                    f"{t.battery_pct if t else '?'}% (needs {self.args.min_battery}%)")

    def takeoff(self) -> None:
        self.phase = "takeoff"
        print(f"  NOTE  {CAMERA_WHERE}", flush=True)
        if self.last_frame_t is None or time.time() - self.last_frame_t > 1.0:
            self.record("takeoff", False, "no live video: the safety governor would only hover")
            raise Abort("no video")
        self.d.takeoff()
        reply = self.until_done(25.0)
        ok = reply is not None and reply.lower().startswith("ok") and self.d.flying
        t = self.tel()
        self.record("takeoff", ok, f"reply {reply!r}" + (f", h {t.height_m:.2f} m" if t and t.height_m is not None else ""))
        if not ok:
            raise Abort("takeoff not confirmed")
        self.wait(2.0)  # settle

    def stick_steps(self) -> None:
        mag, dur = self.args.mag, self.args.dur
        for name, vec, field, sign in STICK_STEPS:
            self.phase = f"stick {name}"
            self.watch_begin()
            before = self.tel()
            rc = tuple(mag * c for c in vec)
            peak = 0.0
            t_end = time.time() + dur
            while time.time() < t_end:
                self.wait(0.05, rc)
                now = self.tel()
                if field in ("pitch", "roll") and now is not None:  # tilt is the reaction while the stick is held
                    v = getattr(now, f"{field}_deg") or 0.0
                    peak = v if abs(v) > abs(peak) else peak
            after = self.tel()
            self.wait(self.args.settle)
            if self.args.dry_run:
                self.record(f"rc {name}", True, f"sticks {rc} accepted (dry run: logged, not sent)")
                continue
            if field == "yaw":
                delta = ((after.yaw_deg - before.yaw_deg + 180) % 360) - 180 if after and before else 0.0
                ok = delta * sign > 3.0
                detail = f"yaw {delta:+.0f} deg"
            elif field == "h":
                delta = (after.height_m or 0) - (before.height_m or 0) if after and before else 0.0
                ok = delta * sign > 0.05
                detail = f"height {delta:+.2f} m"
            else:
                ok = abs(peak) > 2.0
                if not ok:
                    detail = f"no {field} tilt seen (peak {peak:+.1f} deg): did it move {name}?"
                elif peak * sign > 0:
                    detail = f"peak {field} {peak:+.0f} deg, sign as kite expects"
                else:
                    detail = f"peak {field} {peak:+.0f} deg, sign OPPOSITE to kite's convention: check it moved {name}"
            cam, opposite = self.watch_end(name)
            self.record(f"rc {name}", ok and not opposite, detail + cam)

    def moves(self) -> None:
        cm = self.args.move_cm
        for direction in MOVES:
            self.phase = f"move {direction} {cm} cm"
            self.watch_begin()
            self.d.move(direction, cm)
            reply = self.until_done(15.0)
            ok = reply is not None and reply.lower().startswith("ok")
            detail = f"reply {reply!r}"
            if not ok and self.args.dry_run and reply and reply.startswith("error: safety"):
                ok, detail = True, f"refused by the safety governor, as it should be on the table ({reply})"
            self.wait(self.args.settle)
            cam, opposite = self.watch_end(direction)
            self.record(f"move {direction} {cm}", ok and not opposite, detail + cam)
        for deg in (45, -45):
            self.phase = f"rotate {deg:+d}"
            self.watch_begin()
            before = self.tel()
            self.d.rotate(deg)
            reply = self.until_done(15.0)
            after = self.tel()
            delta = ((after.yaw_deg - before.yaw_deg + 180) % 360) - 180 if after and before and not self.args.dry_run \
                else None
            ok = reply is not None and reply.lower().startswith("ok")
            if delta is not None:
                ok = ok and delta * deg > 20
            self.wait(self.args.settle)
            cam, _ = self.watch_end("yaw right" if deg > 0 else "yaw left")
            self.record(f"rotate {deg:+d}", ok, f"reply {reply!r}" + (f", yaw {delta:+.0f} deg" if delta is not None else "")
                        + cam)

    def land(self) -> bool:
        """Land and wait until it is down (the main thread ignores Ctrl+C meanwhile)."""
        self.phase = "landing"
        if not (self.d.flying or self.d.busy()):
            return True
        t_end, asked = time.time() + 25.0, -1e9
        while time.time() < t_end:
            busy = self.d.busy()
            if asked > 0 and not busy and not self.d.flying:
                return True
            if asked < 0 or (not busy and time.time() - asked > 2.0):
                self.d.land()  # first ask, or the last one failed while it still flies
                asked = time.time()
            self.d.check(time.time(), time.time())
            time.sleep(0.05)
        return False

    def run(self) -> None:
        try:
            self.link_check()
            if not self.args.link_only:
                self.takeoff()
                self.stick_steps()
                if not self.args.no_moves:
                    self.moves()
        except Abort as e:
            print(f"  stopping: {e}", flush=True)
            self.record("sequence", False, f"stopped before the end: {e}")  # an interrupted test never reads as passed
        except Exception as e:  # a bug here must still land the drone
            print(f"  error: {e!r}", flush=True)
            self.record("sequence", False, repr(e))
        finally:
            if self.d.flying or self.d.busy():
                landed = self.land()
                self.record("land", landed, "down" if landed else "NOT CONFIRMED: it lands itself ~15 s after the last command")
            self.cam = self.peaks = None
            if self.directions:
                lines, self.direction_verdict = direction_summary(self.directions)
                print("\n" + "\n".join(lines), flush=True)
            self.phase = "done"


def show(check: Check, done: threading.Event, preview: bool) -> None:
    """Main thread: the camera window (macOS only draws windows from the main thread) and Ctrl+C."""
    import cv2

    shown = False
    while not done.is_set():
        try:
            if preview and not shown:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                shown = True
            if shown:
                f = check.src.read()
                if f is not None:
                    img = f.image.copy()
                    t = check.tel()
                    label = f"{check.phase}   " + (f"bat {t.battery_pct}%  h {t.height_m:.2f} m" if t and t.height_m is not None else "")
                    cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow(WIN, img)
                if (cv2.waitKey(1) & 0xFF) == ord("q") or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    shown, preview = False, False
            done.wait(0.01)
        except KeyboardInterrupt:
            if not check.abort.is_set():
                print("\nCtrl+C: landing", flush=True)
                check.abort.set()
            else:
                print("  Ctrl+C ignored: still landing", flush=True)
    if shown:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def build_parser() -> argparse.ArgumentParser:
    from kite.config import VIDEO_BACKENDS

    p = argparse.ArgumentParser(description="Dry test (default) or flight test (--fly) of kite's drone layer.")
    p.add_argument("--fly", action="store_true", help="really fly (default: dry run, nothing reaches the motors)")
    p.add_argument("--link-only", action="store_true", help="only the link, video, telemetry and battery checks")
    p.add_argument("--no-moves", action="store_true", help="skip the discrete moves and turns (stick steps only)")
    p.add_argument("--mag", type=int, default=30, help="stick value for the stick steps, 10-50 (default 30)")
    p.add_argument("--dur", type=float, default=1.0, help="seconds each stick step is held, 0.5-2 (default 1.0)")
    p.add_argument("--settle", type=float, default=1.5, help="hover seconds between steps (default 1.5)")
    p.add_argument("--move-cm", type=int, default=30, help="discrete move distance, 20-100 cm (default 30)")
    p.add_argument("--min-battery", type=int, default=30, help="refuse to fly below this %% (default 30)")
    p.add_argument("--no-preview", action="store_true", help="no camera window")
    p.add_argument("--video-backend", choices=VIDEO_BACKENDS, default=None, help="video reader: pyav (default), "
                   "opencv, raw, auto (raw on macOS)")
    return p


def main(argv: list[str] | None = None) -> int:
    from kite.config import Config, resolve_video_backend
    from kite.drone import SafetyGovernor, make_tello

    args = build_parser().parse_args(argv)
    if not 10 <= args.mag <= 50 or not 0.5 <= args.dur <= 2.0 or not 20 <= args.move_cm <= 100:
        print("--mag must be 10-50, --dur 0.5-2.0 s, --move-cm 20-100")
        return 2
    args.dry_run = not args.fly
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # FFmpeg's decoder chatter before the first keyframe
    try:
        import av.logging

        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass
    from djitellopy import Tello

    Tello.LOGGER.setLevel(logging.CRITICAL)  # its 0xCC binary-reply errors are harmless (it retries)

    cfg = Config()
    if args.video_backend:
        cfg.drone.video_backend = resolve_video_backend(args.video_backend)
    drone = SafetyGovernor(make_tello(cfg, dry_run=args.dry_run), cfg.safety)
    print(f"{'DRY TEST: nothing reaches the motors' if args.dry_run else 'FLIGHT TEST: the drone will fly'} - "
          "connecting ...", flush=True)
    try:
        try:
            drone.connect()
        except Exception as e:
            print(f"cannot connect: {e}\nJoin the TELLO-xxxxxx Wi-Fi and close other Tello programs. Windows: allow "
                  "Python through the firewall on Private AND Public networks (the Tello's Wi-Fi is Public).")
            return 1
        src = drone.frame_source()
        src.wait_first(10.0)
        check = Check(drone, src, args)
        if args.fly:
            t = drone.telemetry()
            if t.battery_pct is not None and t.battery_pct < args.min_battery:
                print(f"battery {t.battery_pct}% is under --min-battery {args.min_battery}%: not flying")
                return 1
        done = threading.Event()

        def worker():
            try:
                check.run()
            finally:
                done.set()

        th = threading.Thread(target=worker, name="drone-check")
        th.start()
        try:
            show(check, done, preview=not args.no_preview)
        finally:  # whatever happened to the window: stop the sequence (it lands) before closing the drone
            if not done.is_set():
                check.abort.set()
            while th.is_alive():
                try:
                    th.join()
                except KeyboardInterrupt:
                    print("  Ctrl+C ignored: still landing", flush=True)
    finally:
        drone.close()  # lands if still flying (never in dry run), stops the video
    failed = [name for name, ok, _ in check.results if not ok]
    print(f"\n{'DRY' if args.dry_run else 'FLIGHT'} TEST: {len(check.results) - len(failed)}/{len(check.results)} passed"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
