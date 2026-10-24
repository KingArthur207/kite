"""Replay one kite.record take: every frame in order, with the camera pose at that frame (perception's input).

    from kite.sources.recording import RecordingSource
    src = RecordingSource("recordings/20261024_070312_tello")   # realtime=True paces reads by t_mono
    while True:
        ok, bgr = src.read()                     # EVERY frame of video.mkv, in order; (False, None) at the end
        if not ok:
            break
        events = process(bgr, src.camera, src.pose)

    python -m kite.sources.recording <dir>              # summary of the take (run it right after landing)
    python -m kite.sources.recording <dir> --preview    # the frames with a pose HUD

Same shape as sim/frames.py's SimFrameSource (read() -> (ok, bgr), .pose), so perception swaps one for the other.
.pose describes the frame just read, with the sim's keys:
    x, y        None: real footage has no position
    heading     Tello yaw, degrees clockwise in [0, 360) (DroneCfg.yaw_sign applied, as TelloDrone does). RELATIVE to
                the orientation the Tello zeroed its yaw at (power-on / takeoff), NOT a compass heading like the sim's.
                heading_ref says so: "tello_yaw" here, absent in the sim.
    cam_height  m above the floor: the downward ToF when it agrees with 'h' within 0.2 m, else 'h'
                (Telemetry.floor_altitude_m). 'h' is the height above the TAKEOFF POINT, so it is the height above
                the floor only if the drone took off from the floor and the ground under it is level with that spot.
                The camera sits a few cm above the ToF sensor: ignored.
    pitch       degrees, camera pitch = body pitch (the Tello camera is fixed to the body). + = nose UP, 0 = level,
                matching the sim, whose drone camera looks level and always reports pitch 0. A ray at image row y
                points at elevation -atan((y - cy) / fy) + pitch above the horizon.
and, beyond the sim's: roll (deg, as reported), t (the frame's t_mono), battery (%), flying (True between the
recorder's confirmed takeoff and landing events), tel_dt (frame t minus the nearest state packet's t, s).
Without telemetry near a frame (none within max_gap_s, or a file-source dry run with an empty telemetry.jsonl) the
pose fields are None and the frames still play.

Alignment: video and telemetry share the recorder's t_mono clock. Each frame gets its pose interpolated between the
state packets just before and after it when both are within max_gap_s, else the nearest packet within max_gap_s,
else None. Raw state fields are converted by TelloDrone.telemetry() itself, so units match the live drone layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from ..camera import Camera, load_camera
from ..types import Frame
from .base import FrameSource
from .pose import MAX_GAP_S, PITCH_SIGN, TelemetryTrack, empty_pose

SEEK_GRAB_MAX = 30  # seeks up to this many frames ahead decode their way there; longer ones jump (indexed file)


class _State:
    """Stands in for djitellopy.Tello so TelloDrone.telemetry() parses a recorded packet (the one unit conversion)."""

    state: dict = {}

    def get_current_state(self) -> dict:
        return self.state


def _parse_states(packets: list[dict], yaw_sign: int | None) -> list:
    from ..config import Config
    from ..drone.tello import TelloDrone  # deferred: kite.drone imports kite.sources

    holder = _State()
    sign = Config().drone.yaw_sign if yaw_sign is None else yaw_sign
    parser = TelloDrone(tello=holder, video=False, dry_run=True, yaw_sign=sign)
    out = []
    for p in packets:
        holder.state = {k: v for k, v in p.items() if k not in ("t_mono", "t_wall")}
        tel = parser.telemetry()
        tel.t = p["t_mono"]
        out.append(tel)
    return out


def _read_jsonl(path: Path) -> list[dict]:
    """Lines of a .jsonl file; a missing file or a torn last line (crash) is skipped, not fatal."""
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("t_mono"), (int, float)):
            out.append(rec)
    return out


def _read_frames_csv(path: Path) -> list[dict]:
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                out.append({"index": int(r["index"]), "seq": int(r["seq"]), "t_mono": float(r["t_mono"]),
                            "t_wall": float(r["t_wall"])})
            except (TypeError, ValueError, KeyError):
                continue  # torn last row after a crash
    return out


def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


class RecordingSource:
    """One kite.record output folder, replayed frame by frame with the pose aligned to each frame.

    read() -> (ok, bgr): the next frame (BGR uint8), every frame in order. Offline by default (as fast as the caller
    reads); realtime=True sleeps so frames come out at their recorded t_mono spacing (speed=2 plays twice as fast).
    After read(): .index (frame number in video.mkv), .t (its t_mono), .pose, .telemetry (the raw nearest state
    packet within max_gap_s, as recorded, or None). .camera: intrinsics (load_camera(), or meta.json's calibrated
    fx_px if it has one), scaled to the video size if that is not the camera's. len(), seek(index | t= | sec=).
    """

    def __init__(self, path: str | Path, realtime: bool = False, speed: float = 1.0, max_gap_s: float = MAX_GAP_S,
                 yaw_sign: int | None = None, pitch_sign: int = PITCH_SIGN, camera: Camera | None = None):
        self.dir = Path(path)
        if not (self.dir / "frames.csv").is_file():
            raise FileNotFoundError(f"{self.dir} is not a kite.record folder (no frames.csv)")
        self.realtime, self.speed, self.max_gap_s, self.pitch_sign = realtime, speed, max_gap_s, pitch_sign
        meta_p = self.dir / "meta.json"
        try:
            self.meta: dict = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
        except ValueError:
            self.meta = {}
        self.rows = _read_frames_csv(self.dir / "frames.csv")
        self.times = np.array([r["t_mono"] for r in self.rows], dtype=float)
        self.raw_telemetry = _read_jsonl(self.dir / "telemetry.jsonl")
        self.raw_telemetry.sort(key=lambda p: p["t_mono"])
        self.events = _read_jsonl(self.dir / "events.jsonl")
        if yaw_sign is None:  # the sign the recorder used, if it saved one
            yaw_sign = self.meta.get("drone_cfg", {}).get("yaw_sign")
        self.tel = _parse_states(self.raw_telemetry, yaw_sign) if self.raw_telemetry else []
        self.camera = camera or self._load_camera()
        self._poses = self._align()

        self.index = -1  # frame just read (-1: none yet)
        self.t: float | None = None
        self.pose: dict | None = None
        self.telemetry: dict | None = None
        self._cap = None
        self._jpgs = None
        self._next = 0  # frame the next read() returns
        self._pos = 0  # frame the open capture decodes next
        self._pace: tuple[float, float] | None = None  # (wall time, t_mono) of the pacing origin
        self._indexed = "summary" in self.meta  # the recorder finished: video.mkv was closed cleanly
        self.seek_grab_max = SEEK_GRAB_MAX
        self._open()

    # ------------------------------------------------------------------ setup
    def _load_camera(self) -> Camera:
        cam = load_camera()
        mcam = self.meta.get("camera") or {}
        fx = mcam.get("fx_px")
        if isinstance(fx, (int, float)) and fx > 0:  # recorded with a calibration: that one wins
            fy = mcam.get("fy_px")
            cam.fx = float(fx)
            cam.fy = float(fy) if isinstance(fy, (int, float)) and fy > 0 else float(fx)
            for k in ("cx", "cy"):
                v = mcam.get(f"{k}_px")
                if isinstance(v, (int, float)):
                    setattr(cam, k, float(v))
            cam.source = "meta.json"
        vid = self.meta.get("video") or {}
        w, h = vid.get("width"), vid.get("height")
        if w and h:  # a dry run on another file: keep the field of view
            cam = cam.scaled(int(w), int(h))
        return cam

    def _flying_at(self) -> list[tuple[float, bool]]:
        """(t_mono, flying) changes from events.jsonl: confirmed takeoff / landing, emergency."""
        out = []
        for e in self.events:
            msg = str(e.get("event", ""))
            if isinstance(e.get("flying"), bool):  # "takeoff -> ok", "land -> ok", ... carry the drone's state
                out.append((e["t_mono"], e["flying"]))
            elif msg == "landed" or msg.startswith("EMERGENCY"):
                out.append((e["t_mono"], False))
        return out

    def _align(self) -> list[dict]:
        """One pose per frames.csv row (no decoding needed: pose_at(i) works before any read)."""
        n = len(self.rows)
        track = TelemetryTrack(self.tel, self.pitch_sign)
        flips = self._flying_at()
        fl_t = np.array([f[0] for f in flips], dtype=float)
        # with telemetry but no flight events (record only, drone in hand) it never flew: only --fly can fly it
        fl_default = None if not self.tel else False
        poses = []
        for i in range(n):
            t = self.times[i]
            pose = empty_pose(float(t), heading_ref="tello_yaw")
            near = track.fill(pose, t, self.max_gap_s)
            if near is not None:
                pose["_tel"] = near
            k = int(np.searchsorted(fl_t, t, side="right")) - 1
            pose["flying"] = flips[k][1] if k >= 0 else fl_default
            poses.append(pose)
        return poses

    # ------------------------------------------------------------------ video
    def _open(self) -> None:
        import cv2

        if self._cap is not None:
            self._cap.release()
        self._cap = None
        mkv = self.dir / "video.mkv"
        if mkv.is_file():
            self._cap = cv2.VideoCapture(str(mkv))
            if not self._cap.isOpened():
                raise RuntimeError(f"cannot open {mkv}")
        else:  # the recorder's fallback when the video writer would not open
            self._jpgs = sorted((self.dir / "frames").glob("*.jpg"))
            if not self._jpgs:
                raise FileNotFoundError(f"{self.dir}: no video.mkv and no frames/*.jpg")
        self._pos = 0

    def _decode(self, i: int):
        import cv2

        if self._jpgs is not None:
            return cv2.imread(str(self._jpgs[i])) if i < len(self._jpgs) else None
        if i != self._pos:
            # every MJPG frame is a keyframe: a finished take (Matroska index written on close) seeks directly; a
            # crashed one has no index, so step forward (~6 ms per 960x720 frame) and reopen to go back
            if self._indexed and abs(i - self._pos) > self.seek_grab_max:
                if (self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        and int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) == i):
                    self._pos = i
                else:
                    self._open()
            if i < self._pos:
                self._open()
            while self._pos < i:
                if not self._cap.grab():
                    return None
                self._pos += 1
        ok, img = self._cap.read()
        if not ok:
            return None
        self._pos += 1
        return img

    # ------------------------------------------------------------------ SimFrameSource shape
    def read(self) -> tuple[bool, np.ndarray | None]:
        i = self._next
        if i >= len(self.rows):
            return False, None
        if self.realtime:
            now = time.monotonic()
            if self._pace is None:
                self._pace = (now, self.times[i])
            wait = self._pace[0] + (self.times[i] - self._pace[1]) / self.speed - now
            if wait > 0:
                time.sleep(wait)
        img = self._decode(i)
        if img is None:  # video.mkv shorter than frames.csv (a crash loses the last few frames)
            self._next = len(self.rows)
            return False, None
        self._next = i + 1
        self.index, self.t = i, float(self.times[i])
        self.pose = self.pose_at(i)
        k = self._poses[i].get("_tel")
        self.telemetry = None if k is None else self.raw_telemetry[k]
        return True, img

    def pose_at(self, i: int) -> dict:
        """Pose of frame i (a copy), without decoding it."""
        return {k: v for k, v in self._poses[i].items() if not k.startswith("_")}

    def seek(self, index: int | None = None, *, t: float | None = None, sec: float | None = None) -> int:
        """Make the next read() return frame `index`, or the first frame at or after t_mono `t`, or `sec` seconds
        after the first frame. Returns that frame's index (len(self) past the end)."""
        if sum(v is not None for v in (index, t, sec)) != 1:
            raise ValueError("seek(index), seek(t=t_mono) or seek(sec=seconds from the start)")
        if index is None:
            target = (self.times[0] + sec) if sec is not None else t
            index = int(np.searchsorted(self.times, target - 1e-9)) if len(self.times) else 0
        self._next = max(0, min(int(index), len(self.rows)))
        self._pace = None  # realtime pacing restarts from here
        return self._next

    def __len__(self) -> int:
        return len(self.rows)

    def isOpened(self) -> bool:
        return self._cap is not None or self._jpgs is not None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._jpgs = None

    def __enter__(self) -> RecordingSource:
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def frame_source(self) -> RecordingFrameSource:
        return RecordingFrameSource(self)

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        n = len(self.rows)
        seq = [r["seq"] for r in self.rows]
        dur = float(self.times[-1] - self.times[0]) if n > 1 else 0.0
        covered = [p for p in self._poses if p["tel_dt"] is not None]

        def rng(key):
            v = [p[key] for p in self._poses if p[key] is not None]
            return (min(v), max(v)) if v else None

        tel_t = [p["t_mono"] for p in self.raw_telemetry]
        bats = [t.battery_pct for t in self.tel if t.battery_pct is not None]
        gaps = np.diff(self.times) if n > 1 else np.array([])
        flights = [(e["t_mono"] - self.times[0] if n else 0.0, e["event"]) for e in self.events
                   if isinstance(e.get("flying"), bool) or e.get("event") in ("takeoff", "landed")
                   or str(e.get("event", "")).startswith(("EMERGENCY", "LAND NOT", "safety"))]
        return {
            "dir": str(self.dir), "frames": n, "duration_s": dur, "fps": (n - 1) / dur if dur > 0 else None,
            "longest_frame_gap_s": float(gaps.max()) if len(gaps) else None,
            "dropped_frames": sum(max(0, b - a - 1) for a, b in zip(seq, seq[1:])),
            "dropped_by_writer": (self.meta.get("summary") or {}).get("frames_dropped_by_writer"),
            "telemetry_packets": len(self.raw_telemetry),
            "telemetry_hz": (len(tel_t) - 1) / (tel_t[-1] - tel_t[0]) if len(tel_t) > 1 and tel_t[-1] > tel_t[0] else None,
            "telemetry_coverage": len(covered) / n if n else 0.0,
            "cam_height_m": rng("cam_height"), "pitch_deg": rng("pitch"), "roll_deg": rng("roll"),
            "battery_start": bats[0] if bats else self.meta.get("battery_start"),
            "battery_end": bats[-1] if bats else (self.meta.get("summary") or {}).get("battery_end"),
            "flying_frames": sum(1 for p in self._poses if p["flying"]),
            "flight_events": flights,
            "camera": f"fx {self.camera.fx:.0f} fy {self.camera.fy:.0f} px ({self.camera.source})",
            "end_reason": (self.meta.get("summary") or {}).get("end_reason", "no summary: the recorder did not finish"),
        }


class RecordingFrameSource(FrameSource):
    """A recording behind kite's FrameSource.read() -> Frame: each call returns the NEXT frame (every frame, like
    SteppedVideoSource; with realtime=True on the RecordingSource, paced by t_mono). Frame.t is the recorded
    t_wall, Frame.seq the source seq. After the last frame it keeps returning it and sets finished. .pose and
    .camera are the RecordingSource's."""

    name = "recording"

    def __init__(self, rec: RecordingSource):
        self.rec = rec
        self._frame: Frame | None = None
        self.finished = False

    @property
    def pose(self) -> dict | None:
        return self.rec.pose

    @property
    def camera(self) -> Camera:
        return self.rec.camera

    def read(self) -> Frame | None:
        if not self.finished:
            ok, img = self.rec.read()
            if ok:
                img.flags.writeable = False  # shared between consumers, like every kite frame
                r = self.rec.rows[self.rec.index]
                self._frame = Frame(img, r["t_wall"], r["seq"], self.name)
            else:
                self.finished = True
        return self._frame

    def stop(self) -> None:
        self.rec.release()


# ------------------------------------------------------------------ CLI
def _fmt_range(r, unit: str, nd: int = 2) -> str:
    return "none" if r is None else f"{r[0]:.{nd}f} .. {r[1]:.{nd}f} {unit}"


def print_summary(s: dict) -> None:
    print(f"{s['dir']}")
    fps = f"{s['fps']:.1f} fps" if s["fps"] else "? fps"
    print(f"  frames      {s['frames']} over {s['duration_s']:.1f} s ({fps}, longest gap "
          f"{(s['longest_frame_gap_s'] or 0) * 1000:.0f} ms)")
    print(f"  dropped     {s['dropped_frames']} (seq gaps)" + (f", {s['dropped_by_writer']} by the disk writer"
                                                              if s["dropped_by_writer"] else ""))
    hz = f"{s['telemetry_hz']:.1f} Hz" if s["telemetry_hz"] else "-"
    print(f"  telemetry   {s['telemetry_packets']} packets ({hz}), pose on {s['telemetry_coverage'] * 100:.0f}% of frames")
    print(f"  cam_height  {_fmt_range(s['cam_height_m'], 'm')}")
    print(f"  pitch       {_fmt_range(s['pitch_deg'], 'deg', 0)}   roll {_fmt_range(s['roll_deg'], 'deg', 0)}")
    print(f"  battery     {s['battery_start']} -> {s['battery_end']} %")
    print(f"  flying      {s['flying_frames']} frames")
    for t, ev in s["flight_events"]:
        print(f"    {t:7.1f} s  {ev}")
    print(f"  camera      {s['camera']}")
    print(f"  end         {s['end_reason']}")


def _hud(img: np.ndarray, src: RecordingSource) -> np.ndarray:
    import cv2

    p = src.pose or {}

    def v(k, fmt):
        return "-" if p.get(k) is None else format(p[k], fmt)

    t0 = src.times[0]
    lines = [f"#{src.index}/{len(src)}  {src.t - t0:6.2f} s  {'FLYING' if p.get('flying') else 'ground'}",
             f"h {v('cam_height', '.2f')} m  pitch {v('pitch', '+.0f')}  roll {v('roll', '+.0f')}  "
             f"hdg {v('heading', '.0f')}  bat {v('battery', '.0f')}%  dt {v('tel_dt', '+.2f')}"]
    out = img.copy()
    for i, line in enumerate(lines):
        y = 24 + 24 * i
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if p.get("pitch") is not None and src.camera.fy > 0:  # the horizon, where a level ray lands in the image
        h, w = out.shape[:2]
        sy = h / src.camera.height
        y = int(round((src.camera.cy + src.camera.fy * math.tan(math.radians(p["pitch"]))) * sy))
        if 0 <= y < h:
            cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
    return out


def preview(src: RecordingSource) -> None:
    import cv2

    print("preview: space pause, , . step back / forward while paused, q or Esc quit")
    paused = False
    win = "kite recording"
    while True:
        if not paused or src.index < 0:
            ok, img = src.read()
            if not ok:
                break
            cv2.imshow(win, _hud(img, src))
        k = cv2.waitKey(0 if paused else 1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused = not paused
            src.seek(src.index + 1)  # also restarts the realtime pacing
        elif paused and k in (ord(","), ord(".")):
            src.seek(max(0, src.index + (-1 if k == ord(",") else 1)))
            ok, img = src.read()
            if ok:
                cv2.imshow(win, _hud(img, src))
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kite.sources.recording", description="Check a kite.record take.")
    p.add_argument("dir", help="one kite.record output folder (recordings/<stamp>_<source>)")
    p.add_argument("--preview", action="store_true", help="play the frames with a pose HUD (realtime)")
    p.add_argument("--speed", type=float, default=1.0, help="preview playback speed (default 1)")
    p.add_argument("--start", type=float, default=0.0, help="preview from this many seconds in")
    args = p.parse_args(argv)
    try:
        src = RecordingSource(args.dir, realtime=args.preview, speed=args.speed)
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
        return 1
    with src:
        print_summary(src.summary())
        if args.preview:
            src.seek(sec=args.start)
            preview(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
