"""Record the drone's video and telemetry to disk, with optional keyboard flight control.

    python -m kite.record --source clip.mp4 --out recordings      # dry run on a file: no drone, no telemetry
    python -m kite.record --source tello --out recordings         # record only: drone on the ground / in hand
    python -m kite.record --source tello --out recordings --fly --dry-run   # rehearse the keys, motion only logged
    python -m kite.record --source tello --out recordings --fly   # the shoot: fly with the keys and record

Only one program can hold the Tello ports (UDP 8889/8890/11111), so the program that records from the Tello is also
the only one that can fly it: --fly adds keyboard control in this same process, through kite's drone layer and
SafetyGovernor (stick clamp, altitude band, battery / flight-time landing, hover on stale video or telemetry).
Without --fly the drone is opened with dry_run=True and NO motion command can ever be sent.

Keys (with --fly, typed in this terminal; one key at a time; arrow / function keys are ignored):
    t takeoff     l land     space hover (all sticks to 0)
    w/s forward/back   a/d left/right   r/f up/down   q/e turn left/right
    A move key drives that stick at --speed for --pulse seconds; holding it (key repeat) keeps it going, so letting
    go stops the drone within --pulse seconds. Press ! twice within 1 s for EMERGENCY (motors off: the drone falls).
    Ctrl+C lands (with --fly) and stops recording; a second Ctrl+C while landing is ignored. Ctrl+Z, Ctrl+\\ and
    Ctrl+S do nothing while flying (they would suspend, kill or freeze the program with the drone in the air).

Output: <out>/<YYYYmmdd_HHMMSS>_<source>/
    video.mkv        MJPG in Matroska, BGR frames as received. Every frame is a JPEG and Matroska needs no index, so
                     a crash or a pulled cable leaves a playable file (losing at most the last few frames).
    frames.csv       one row per frame in video.mkv: index (frame number in the file), seq (source frame number:
                     gaps = frames the source or the writer dropped), t_mono, t_wall, src_t
    telemetry.jsonl  one line per Tello state packet: t_mono, t_wall and the raw state fields (units in meta.json).
                     Empty for a file source.
    events.jsonl     takeoff / land / stick changes / safety events with t_mono, t_wall
    meta.json        source, args, start time, git commit, camera placeholder; summary added at the end
t_mono (time.monotonic(), one clock for the whole process) is when the recorder saw a frame or packet: align video
and telemetry on it. src_t is the source's own arrival stamp for the frame (time.time(), compare with t_wall).
"""

from __future__ import annotations

import argparse
import codecs
import csv
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import Config
from .sources.base import FrameSource
from .sources.opencv_sources import SteppedVideoSource
from .types import Frame

ROOT = Path(__file__).resolve().parent.parent
TELLO_FPS = 30.0  # nominal stream rate, used as the container frame rate (true times are in frames.csv)
# raw Tello state fields (djitellopy parses them to int / float)
STATE_UNITS = {
    "pitch, roll, yaw": "deg", "vgx, vgy, vgz": "dm/s per kite-notes-02 (TelloDrone scales by 0.1 to m/s)",
    "templ, temph": "degC", "tof": "cm to whatever is below (6553 = out of range)", "h": "cm above the takeoff point",
    "bat": "%", "baro": "m (absolute, drifts)", "time": "s motors have run", "agx, agy, agz": "~milli-g",
}
AXIS_KEYS = {"a": (0, -1), "d": (0, 1), "w": (1, 1), "s": (1, -1), "r": (2, 1), "f": (2, -1), "q": (3, -1), "e": (3, 1)}
KEY_HELP = ("keys: t takeoff | l land | space hover | w/s fwd/back | a/d left/right | r/f up/down | q/e turn | "
            "!! emergency | Ctrl+C land + quit")


@dataclass
class RecordCfg:
    out_dir: str = "recordings"
    fly: bool = False  # keyboard flight control (needs a drone)
    speed: int = 30  # stick value a move key sends (the SafetyGovernor clamps it to SafetyCfg.max_rc)
    pulse_s: float = 1.0  # a move key drives its stick for this long (key repeat extends it)
    max_s: float = 0.0  # stop recording after this many seconds (0: until Ctrl+C / end of file)
    fps: float = TELLO_FPS  # container frame rate
    queue_frames: int = 90  # frames waiting for the disk writer (3 s at 30 fps); beyond that new frames are dropped
    control_hz: float = 10.0  # safety check + rc keep-alive rate (the Tello lands 15 s after the last command)
    land_wait_s: float = 25.0  # how long Ctrl+C waits for the landing to be confirmed


class FileReplay(FrameSource):
    """EVERY frame of a video file, in order, at the file's frame rate (SteppedVideoSource, paced like a camera).

    VideoFileSource serves only the newest frame, so a slow reader would skip some; here each read() advances by at
    most one frame, so the recording has exactly the file's frames. finished is set after the last one.
    """

    name = "file"

    def __init__(self, path: str | Path, fps: float | None = None):
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()
        self._src = SteppedVideoSource(path, self.fps)
        self._frame: Frame | None = None
        self._t0: float | None = None
        self.finished = False

    def read(self) -> Frame | None:
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        due = self._frame is None or now >= self._t0 + self._frame.seq / self.fps
        if due and not self.finished:
            f = self._src.read()
            if f is None:
                self.finished = True
            else:
                self._frame = Frame(f.image, time.time(), f.seq, self.name)
        return self._frame

    def stop(self) -> None:
        self._src.stop()


class TerminalKeys:
    """Keys typed in this terminal, without Enter (cbreak mode: Ctrl+C still raises KeyboardInterrupt).

    read() returns printable ASCII only. Escape sequences are dropped whole, also when split across reads: an arrow key
    is ESC [ A..D, and its last letter would otherwise fly the drone (Up -> 'a' = left, Left -> 'd' = right). Ctrl+Z
    (suspend), Ctrl+\\ (quit without landing) and Ctrl+S (output freeze, which blocks print() and so the rc keep-alive)
    are disabled while the keys are active.
    """

    ESC_TIMEOUT_S = 0.1  # an unfinished sequence older than this (a lone Escape key) does not swallow the next key

    def __init__(self, fd: int | None = None):
        self._fd = fd
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        self._esc = ""  # "" | "esc" (after ESC) | "csi" (after ESC [) | "ss3" (after ESC O)
        self._esc_t = 0.0

    def __enter__(self) -> TerminalKeys:
        import termios
        import tty

        if self._fd is None:
            self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        a = termios.tcgetattr(self._fd)
        off = os.fpathconf(self._fd, "PC_VDISABLE")
        for name in ("VSUSP", "VDSUSP", "VQUIT"):  # VDSUSP (Ctrl+Y, delayed suspend) exists on macOS only
            i = getattr(termios, name, None)
            if i is not None:
                a[6][i] = off if isinstance(a[6][i], int) else bytes([off])
        a[0] &= ~termios.IXON
        termios.tcsetattr(self._fd, termios.TCSANOW, a)
        return self

    def read(self) -> str:
        import select

        if not select.select([self._fd], [], [], 0)[0]:
            return ""
        out, now = [], time.monotonic()
        for ch in self._dec.decode(os.read(self._fd, 64)):
            if self._esc and now - self._esc_t > self.ESC_TIMEOUT_S:
                self._esc = ""  # the Escape key on its own: a terminal sends a whole sequence at once
            if ch == "\x1b":
                self._esc, self._esc_t = "esc", now
            elif self._esc == "esc":
                self._esc = {"[": "csi", "O": "ss3"}.get(ch, "")  # anything else: Meta/Option+key, dropped
            elif self._esc == "csi":
                if "\x40" <= ch <= "\x7e":
                    self._esc = ""  # final byte of ESC [ ... (A-D arrows, ~ for F5/PgUp/...)
            elif self._esc == "ss3":
                self._esc = ""  # ESC O A..D (application-mode arrows), ESC O P (F1)
            elif " " <= ch <= "~":
                out.append(ch)
        return "".join(out)

    def __exit__(self, *exc) -> None:
        import termios

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


def _raw_state_reader(drone):
    """djitellopy's state dict getter behind any wrapper (SafetyGovernor.inner -> TelloDrone.tello), else None."""
    d = drone
    while d is not None and hasattr(d, "inner"):
        d = d.inner
    tello = getattr(d, "tello", None)
    return None if tello is None else tello.get_current_state


def _clean(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def git_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        return {"commit": commit or None, "dirty": bool(dirty)}
    except Exception as e:  # no git on the shoot laptop must not stop a recording
        return {"commit": None, "error": str(e)}


class Recorder:
    """Records `frames` (and the state stream of `drone`, if any) until Ctrl+C, end of file or cfg.max_s.

    Threads: frame pickup (polls the source, queues new frames), disk writer (encodes video.mkv + frames.csv), and
    telemetry (polls the state dict at 500 Hz, stamps each new packet). The main thread handles keys, the safety
    check, the rc keep-alive and the live status line, so a slow disk never delays flight control.
    keys: an object with read() -> str (TerminalKeys, or a script in tests); only used with cfg.fly.
    """

    def __init__(self, frames: FrameSource, drone=None, cfg: RecordCfg | None = None, meta: dict | None = None,
                 keys=None, safety=None):
        self.frames = frames
        self.drone = drone
        self.cfg = cfg or RecordCfg()
        self.meta = dict(meta or {})
        self.keys = keys
        self.safety = safety or Config().safety
        if self.cfg.fly and drone is None:
            raise ValueError("--fly needs a drone")
        self._stop = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=self.cfg.queue_frames)
        self.dir: Path | None = None
        # counters (written by one thread each, read by the status line)
        self.seen = self.dropped = self.written = self.tel_lines = 0
        self.last_frame_t: float | None = None  # source stamp (time.time()) of the newest frame
        self.last_state: dict = {}
        self.writer_error: str | None = None
        self.end_reason = ""
        # flight control
        self._hold = [0, 0, 0, 0]
        self._until = [0.0] * 4
        self._sent_rc = (0, 0, 0, 0)
        self._cmd: str | None = None  # discrete command we are waiting on
        self._bang_t = -1e9
        self._n_gov_events = 0
        self._takeoff_t: float | None = None
        self._events = None

    # ------------------------------------------------------------------ files
    def _open_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = self.meta.get("source_kind", "rec")
        d = Path(self.cfg.out_dir) / f"{stamp}_{src}"
        n = 1
        while d.exists():
            n += 1
            d = Path(self.cfg.out_dir) / f"{stamp}_{src}_{n}"
        d.mkdir(parents=True)
        return d

    def _write_meta(self) -> None:
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=1, default=str))
        os.replace(tmp, self.dir / "meta.json")  # atomic: a crash never leaves half a meta.json

    def _event(self, msg: str, **extra) -> None:
        rec = {"t_mono": round(time.monotonic(), 6), "t_wall": round(time.time(), 6), "event": msg, **extra}
        if self._events is not None:
            try:  # a full disk must not raise into the control loop (that would end the flight)
                self._events.write(json.dumps(rec) + "\n")
                self._events.flush()
            except (OSError, ValueError) as e:
                self.writer_error = self.writer_error or f"events.jsonl: {e!r}"
        print(f"  >> {msg}" + (f" {extra}" if extra else ""), flush=True)

    # ------------------------------------------------------------------ threads
    def _pickup(self) -> None:
        last_seq = None
        while not self._stop.is_set():
            f = self.frames.read()
            if f is None or f.seq == last_seq:
                if getattr(self.frames, "finished", False):
                    self.end_reason = self.end_reason or "end of file"
                    return
                time.sleep(0.002)
                continue
            last_seq = f.seq
            self.seen += 1
            self.last_frame_t = f.t
            try:  # never block here: a stalled disk drops frames instead of stalling the pickup
                self._q.put_nowait((f.image, f.seq, f.t, time.monotonic(), time.time()))
            except queue.Full:
                self.dropped += 1

    def _writer(self) -> None:
        import cv2

        vw, size, jpg_dir = None, None, None
        with open(self.dir / "frames.csv", "w", newline="") as fh:
            out = csv.writer(fh)
            out.writerow(["index", "seq", "t_mono", "t_wall", "src_t"])
            try:
                while True:
                    item = self._q.get()
                    if item is None:
                        break
                    img, seq, src_t, t_mono, t_wall = item
                    if size is None:
                        size = (img.shape[1], img.shape[0])
                        vw = cv2.VideoWriter(str(self.dir / "video.mkv"), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"MJPG"),
                                             self.cfg.fps, size)
                        if not vw.isOpened():  # fall back to one JPEG per frame rather than recording nothing
                            vw = None
                            jpg_dir = self.dir / "frames"
                            jpg_dir.mkdir()
                            self.writer_error = "video.mkv writer unavailable: writing frames/NNNNNN.jpg instead"
                        self.meta["video"].update(width=size[0], height=size[1],
                                                  file="video.mkv" if vw is not None else "frames/*.jpg")
                    if (img.shape[1], img.shape[0]) != size:
                        img = cv2.resize(img, size)  # VideoWriter silently drops frames of another size
                    if vw is not None:
                        vw.write(img)
                    else:
                        cv2.imwrite(str(jpg_dir / f"{self.written:06d}.jpg"), img)
                    out.writerow([self.written, seq, f"{t_mono:.6f}", f"{t_wall:.6f}", f"{src_t:.6f}"])
                    self.written += 1
                    if self.written % 30 == 0:
                        fh.flush()
            except Exception as e:
                self.writer_error = f"VIDEO WRITER FAILED: {e!r}"
                traceback.print_exc()
            finally:
                if vw is not None:
                    vw.release()

    def _telemetry(self) -> None:
        get_state = _raw_state_reader(self.drone)
        last = None
        with open(self.dir / "telemetry.jsonl", "w") as fh:
            while get_state is not None and not self._stop.wait(0.002):
                try:
                    st = get_state()
                except Exception:
                    continue
                if not st or st is last:  # djitellopy stores a NEW dict per packet: identity = new packet
                    continue
                last = st
                rec = {"t_mono": round(time.monotonic(), 6), "t_wall": round(time.time(), 6)}
                rec.update({k: _clean(v) for k, v in st.items()})
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                self.last_state = st
                self.tel_lines += 1

    # ------------------------------------------------------------------ flight control (main thread)
    def _on_key(self, ch: str, now: float) -> None:
        d = self.drone
        k = ch.lower()
        if ch == "!":
            if now - self._bang_t < 1.0:
                if d.emergency() is False:  # the packet could not be sent: it is still flying
                    self._event("EMERGENCY NOT SENT (Wi-Fi down): press ! twice again")
                else:
                    self._cmd = None
                    self._event("EMERGENCY (motors off)")
            else:
                self._bang_t = now
                print("  >> press ! again within 1 s for EMERGENCY", flush=True)
        elif k == " ":
            self._hold, self._until = [0, 0, 0, 0], [0.0] * 4
        elif k in AXIS_KEYS:
            i, sign = AXIS_KEYS[k]
            self._hold[i] = sign * self.cfg.speed
            self._until[i] = now + self.cfg.pulse_s
        elif k == "t":
            self._try_takeoff()
        elif k == "l":
            if d.flying or d.busy():
                d.land()
                self._cmd = "land"
                self._event("land (key)")

    def _try_takeoff(self) -> None:
        d = self.drone
        bat = self.last_state.get("bat")
        why = None
        if d.flying or d.busy():
            why = "already flying / busy"
        elif getattr(d, "land_requested", False):
            why = "it already landed in this run: restart the recorder to fly again"
        elif self.last_frame_t is None or time.time() - self.last_frame_t > 1.0:
            why = "no live video (the safety governor would only hover)"
        elif time.time() - d.telemetry().t > self.safety.telemetry_timeout_s:
            why = "no live telemetry (the safety governor would only hover)"
        elif bat is None or float(bat) <= self.safety.min_battery_pct:
            why = f"battery {bat}% (lands at {self.safety.min_battery_pct:.0f}%)"
        if why:
            print(f"  >> takeoff refused: {why}", flush=True)
            return
        d.takeoff()
        self._cmd = "takeoff"
        self._event("takeoff", battery=bat)

    def _control_tick(self, now: float) -> str:
        d = self.drone
        action = d.check(time.time(), self.last_frame_t)
        busy = d.busy()  # also what completes takeoff / land: call it every tick
        if self._cmd is not None and not busy:
            res = d.last_result()
            self._event(f"{self._cmd} -> {res}", flying=d.flying)
            if self._cmd == "takeoff" and d.flying:
                self._takeoff_t = now
            self._cmd = None
        if action == "land" and self._cmd is None and busy:
            self._cmd = "land"  # the governor started a landing
        events = getattr(d, "events", [])
        for _, msg in events[self._n_gov_events:]:
            self._event(f"safety: {msg}")
        self._n_gov_events = len(events)
        if action != "ok":
            self._hold, self._until = [0, 0, 0, 0], [0.0] * 4  # never resume a stick after a safety hover
        vals = tuple(h if now < u else 0 for h, u in zip(self._hold, self._until))
        if d.flying and not busy:
            d.rc(*vals)  # sent every tick: also the keep-alive against the Tello's 15 s auto-land
            if vals != self._sent_rc:
                self._event("rc", lr=vals[0], fb=vals[1], ud=vals[2], yaw=vals[3])
                self._sent_rc = vals
        return action

    def _land_and_wait(self) -> None:
        d = self.drone
        if not (self.cfg.fly and d is not None) or not (d.flying or d.busy()):
            return
        print("\nLANDING - keep clear (Ctrl+C is ignored until it is down)", flush=True)
        t_end = time.time() + self.cfg.land_wait_s
        asked = -1e9
        while time.time() < t_end:
            try:
                busy = d.busy()
                if asked > 0 and not busy and not d.flying:
                    self._event("landed", result=d.last_result())
                    return
                if asked < 0 or (not busy and time.time() - asked > 2.0):
                    # first ask (also cuts a running takeoff short), or the last land failed / timed out while flying
                    d.land()  # no-op while a landing is already running
                    asked = time.time()
                    self._event("land (exit)")
                time.sleep(0.05)
            except KeyboardInterrupt:
                print("  >> Ctrl+C ignored: still landing", flush=True)
        self._event("LAND NOT CONFIRMED: the drone may still be airborne (it auto-lands ~15 s after the last command)")

    # ------------------------------------------------------------------ status
    def _status(self, t0: float, fps: float, action: str) -> str:
        line = f"[{time.monotonic() - t0:6.1f}s] REC {fps:4.1f} fps  frames {self.written}"
        if self.dropped:
            line += f"  DROPPED {self.dropped}"
        if self._q.qsize() > self.cfg.queue_frames // 3:
            line += f"  disk queue {self._q.qsize()}"
        st = self.last_state
        if self.drone is None:
            line += " | no telemetry (file source)"
        elif st:
            bat = st.get("bat")
            line += f" | bat {bat}%  h {st.get('h', 0) / 100:.2f} m  tof {st.get('tof', 0) / 100:.2f} m"
            if bat is not None and float(bat) < 30:
                line += "  LOW BATTERY"
        else:
            line += " | waiting for telemetry"
        if self.cfg.fly:
            d = self.drone
            state = "FLYING" if d.flying else ("busy" if d.busy() else "on ground")
            if self._takeoff_t is not None and d.flying:
                state += f" {time.monotonic() - self._takeoff_t:.0f}s"
            line += f" | {state} rc {' '.join(str(v) for v in self._sent_rc)} | {action}"
        link = getattr(getattr(self.drone, "inner", self.drone), "link_error", None)
        if link:
            line += f" | WI-FI DOWN ({link})"
        if self.writer_error:
            line += f" | {self.writer_error}"
        return line

    # ------------------------------------------------------------------ main loop
    def run(self) -> Path:
        self.dir = self._open_dir()
        t0, wall0 = time.monotonic(), time.time()
        self.meta.update(
            started_wall=wall0, started_iso=datetime.fromtimestamp(wall0).isoformat(timespec="seconds"),
            t_mono_at_start=t0, git=git_info(), record_cfg=asdict(self.cfg),
            video={"file": "video.mkv", "codec": "MJPG", "container": "matroska", "fps_nominal": self.cfg.fps,
                   "color": "BGR as received (OpenCV reads it back as BGR)"},
            camera={"fx_px": None, "fx_guess_px": 920, "note": "uncalibrated; published Tello stream calibrations give "
                    "fx ~920 px (the sim starts at 1000). Measure with a known-height object at a taped distance."},
            frames_csv="index = frame number in video.mkv, seq = source frame number (gaps = drops), t_mono = "
                       "time.monotonic() when recorded, t_wall = time.time() then, src_t = source arrival time.time()",
            telemetry_units=STATE_UNITS if self.drone is not None else None,
        )
        self.meta.setdefault("telemetry", "tello state stream" if self.drone is not None
                             else "none: a file source has no drone, telemetry.jsonl is empty")
        self._write_meta()
        self._events = open(self.dir / "events.jsonl", "w")
        print(f"recording to {self.dir}", flush=True)
        if self.cfg.fly:
            print(KEY_HELP, flush=True)

        threads = [threading.Thread(target=fn, daemon=True, name=fn.__name__)
                   for fn in (self._writer, self._telemetry, self._pickup)]
        for th in threads:
            th.start()

        action, next_tick, next_status = "ok", 0.0, time.monotonic() + 1.0
        last_count, last_count_t, fps = 0, time.monotonic(), 0.0
        try:
            try:
                while True:
                    now = time.monotonic()
                    if self.end_reason:
                        break
                    if self.cfg.max_s and now - t0 >= self.cfg.max_s:
                        self.end_reason = f"max {self.cfg.max_s:g} s"
                        break
                    if self.cfg.fly:
                        for ch in self.keys.read() if self.keys is not None else "":
                            self._on_key(ch, now)
                        if now >= next_tick:
                            next_tick = now + 1.0 / self.cfg.control_hz
                            action = self._control_tick(now)
                    if now >= next_status:
                        fps = (self.seen - last_count) / (now - last_count_t)
                        last_count, last_count_t = self.seen, now
                        next_status = now + 1.0
                        try:  # the status line is cosmetic: a bug in it must not end (and land) the flight
                            print(self._status(t0, fps, action), flush=True)
                        except Exception as e:
                            print(f"[status line error: {e!r}]", flush=True)
                    time.sleep(0.01)
            except KeyboardInterrupt:
                self.end_reason = "Ctrl+C"
                print("\nCtrl+C", flush=True)
            except Exception as e:  # a bug here must still land the drone and close the files
                self.end_reason = f"error: {e!r}"
                traceback.print_exc()
        finally:
            self._land_and_wait()  # keeps recording while it lands
            self._stop.set()
            threads[2].join(timeout=2)  # pickup first: no frame is queued after the end marker
            if threads[0].is_alive():  # a dead writer would never empty the queue
                try:
                    self._q.put(None, timeout=30)
                except queue.Full:
                    pass
            threads[0].join(timeout=30)
            threads[1].join(timeout=2)
            self._events.close()
            self.meta["summary"] = {
                "end_reason": self.end_reason or "stopped", "duration_s": round(time.monotonic() - t0, 3),
                "frames_written": self.written, "frames_seen": self.seen, "frames_dropped_by_writer": self.dropped,
                "telemetry_lines": self.tel_lines, "battery_end": self.last_state.get("bat"),
                "writer_error": self.writer_error,
            }
            self._write_meta()
            print(f"stopped ({self.meta['summary']['end_reason']}): {self.written} frames, {self.tel_lines} telemetry "
                  f"lines, {self.dropped} dropped -> {self.dir}", flush=True)
        return self.dir


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kite.record", description="Record drone video + telemetry to disk.")
    p.add_argument("--source", required=True, help="a video file (dry run, no drone) or 'tello'")
    p.add_argument("--out", default="recordings", help="output folder; each run gets a timestamped subfolder")
    p.add_argument("--fly", action="store_true", help="keyboard flight control in this process (tello only)")
    p.add_argument("--dry-run", action="store_true", help="with --fly: keys work but no motion is sent to the drone")
    p.add_argument("--speed", type=int, default=30, help="stick value for the move keys, 10-100 (clamped to "
                   "SafetyCfg.max_rc = 50; default 30)")
    p.add_argument("--pulse", type=float, default=1.0, help="seconds one key press drives a stick (default 1.0)")
    p.add_argument("--max-s", type=float, default=0.0, help="stop after this many seconds (default: Ctrl+C)")
    p.add_argument("--video-backend", choices=["pyav", "opencv"], default=None, help="Tello H.264 decoder")
    args = p.parse_args(argv)

    if not 10 <= args.speed <= 100:
        p.error("--speed must be 10-100")
    if not 0.2 <= args.pulse <= 2.0:
        p.error("--pulse must be 0.2-2.0 s")
    tello = args.source.lower() == "tello"
    if args.fly and not tello:
        p.error("--fly needs --source tello")
    if args.fly and not sys.stdin.isatty():
        p.error("--fly needs an interactive terminal for the keys")
    if not tello and not os.path.isfile(args.source):
        p.error(f"no such video file: {args.source}")
    os.makedirs(args.out, exist_ok=True)
    free_gb = shutil.disk_usage(args.out).free / 1e9
    if free_gb < 5:
        print(f"WARNING: only {free_gb:.1f} GB free in {args.out} (~0.7 GB per 5 min of video)")

    cfg = Config()
    if args.video_backend:
        cfg.drone.video_backend = args.video_backend
    rcfg = RecordCfg(out_dir=args.out, fly=args.fly, speed=args.speed, pulse_s=args.pulse, max_s=args.max_s)
    meta = {"source": args.source, "args": vars(args), "argv": sys.argv,
            "drone_cfg": {"yaw_sign": cfg.drone.yaw_sign, "video_backend": cfg.drone.video_backend}}

    if not tello:
        frames = FileReplay(args.source)
        rcfg.fps = frames.fps
        meta.update(source_kind="file", dry_run=True)
        print(f"dry run on {args.source} ({frames.fps:g} fps): no drone, no telemetry")
        try:
            Recorder(frames, None, rcfg, meta).run()
        finally:
            frames.stop()
        return 0

    from .drone import SafetyGovernor, make_tello

    dry = args.dry_run or not args.fly  # record-only never sends motion
    meta.update(source_kind="tello", dry_run=dry)
    drone = SafetyGovernor(make_tello(cfg, dry_run=dry), cfg.safety)
    print(f"connecting to the Tello ({'DRY RUN: no motion is sent' if dry else 'LIVE: keys fly the drone'}) ...")
    try:
        try:
            drone.connect()
        except Exception as e:
            print(f"cannot connect: {e}\nJoin the TELLO-xxxxxx Wi-Fi, close the phone app, and check that "
                  f"`lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` prints nothing.")
            return 1
        frames = drone.frame_source()
        print("waiting for the first video frame ...", flush=True)
        if frames.wait_first(15.0) is None:
            print("WARNING: no video yet; recording anyway (the reader keeps retrying, takeoff waits for video)")
        meta["battery_start"] = drone.telemetry().battery_pct
        if args.fly:
            with TerminalKeys() as keys:
                Recorder(frames, drone, rcfg, meta, keys=keys, safety=cfg.safety).run()
        else:
            Recorder(frames, drone, rcfg, meta).run()
    finally:
        print("closing the drone link ...", flush=True)
        drone.close()  # lands if still flying (not in dry run), stops video
    return 0


if __name__ == "__main__":
    sys.exit(main())
