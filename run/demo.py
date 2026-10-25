"""The kite window, with SegFormer-B0 on the webcam.

    python run/demo.py

The camera pane shows that overlay. The map and the console are the same window as
``python run/app.py``. This script does not open the drone, and ``run/app.py`` is unchanged.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VENDOR = os.path.join(ROOT, "third_party", "videotesting")

# Names DirectShow uses for the team's Logitech. Tried when the device list is empty.
CANDIDATES = ("Logi C270 HD WebCam", "Logitech HD Webcam C270", "HD Webcam C270")
WEBCAM_HEIGHT_M = 1.2
WEBCAM_PITCH_DEG = 8.0  # positive looks down; a lid camera is nearly level


def dshow_video_names(listing: str) -> list[str]:
    """Video device names from ``ffmpeg -list_devices true -f dshow`` (stderr)."""
    return re.findall(r'"([^"\r\n]+)" \(video\)', listing)


def choose_camera(names: list[str]) -> str | None:
    """Prefer a Logitech C270, then any Logitech, then the first real camera."""
    real = [name for name in names if name and "virtual" not in name.lower()]
    if not real:
        return None

    def rank(name: str) -> tuple[int, int, int]:
        low = name.lower()
        logitech = any(token in low for token in ("logi", "c270", "c310", "c920", "c922"))
        return (0 if "c270" in low else 1, 0 if logitech else 1, real.index(name))

    return min(real, key=rank)


def camera_order(names: list[str], requested: str | None) -> list[str]:
    """Names to try, best first. A requested name is the only one."""
    if requested:
        return [requested]
    ordered: list[str] = []
    picked = choose_camera(names)
    if picked:
        ordered.append(picked)
    for name in names:
        if name not in ordered and "virtual" not in name.lower():
            ordered.append(name)
    for name in CANDIDATES:
        if name not in ordered:
            ordered.append(name)
    return ordered


def list_dshow_videos() -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    return dshow_video_names((proc.stderr or "") + (proc.stdout or ""))


def _vendor_path() -> None:
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)


def load_perception(device: str = ""):
    """SegFormer-B0 (Mapillary) plus the detector and light reader the overlay draws."""
    _vendor_path()
    from streetseg.__main__ import _models

    args = type("SegmentArgs", (), {})()
    args.segmenter = "mapillary"
    args.seg_weights = "segmentation/SegFormer_b0_Mapillary.pth"
    args.sem_weights = "yolo26n-sem.pt"
    args.det_weights = "yolo26n.pt"
    args.no_lights = False
    args.light_weights = "traffic_light_classification/resnet18-best.pth"
    args.signs = False
    args.sign_weights = ""
    args.depth = False
    args.device = device
    args.imgsz = 640
    return _models(args)


def console_lines(events) -> tuple[str, ...]:
    """Scene text and the spoken line, once each. Hints repeat the same words in a shorter form."""
    lines: list[str] = []
    for event in events:
        if getattr(event, "kind", "") not in ("scene", "say", "hazard"):
            continue
        text = getattr(event, "text", "")
        if text and text not in lines:
            lines.append(text)
    return tuple(lines)


def camera_for(shape: tuple[int, ...], fov_deg: float):
    """Intrinsics for this frame. Role ``drone`` is what paints the segmentation overlay."""
    _vendor_path()
    from streetseg.camera import Camera, focal_length

    height, width = shape[:2]
    focal = focal_length(width, height, fov_deg)
    return Camera(focal, focal, "drone", width / 2.0, height / 2.0)


def open_index(index: int):
    """Open a camera by the index from the DirectShow list.

    OpenCV 5's DirectShow backend rejects ``video=Name`` and an explicit Media Foundation open can
    sit forever, so this passes the list index with ``CAP_DSHOW``. That list matches the order
    Media Foundation reports (built-in first, then the C270, on this machine).
    """
    import cv2

    _vendor_path()
    from streetseg.sources import C270_FPS, C270_HEIGHT, C270_WIDTH, configure_capture

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open camera {index}")
    configure_capture(cap, C270_WIDTH, C270_HEIGHT, C270_FPS)
    return cap


class Shown:
    def __init__(self, image, seq: int):
        self.image = image
        self.seq = seq


class Snapshot:
    """The fields the kite window reads. ``meter`` labels the second fps number."""

    def __init__(self):
        self.state = "starting"
        self.battery = None
        self.height_m = None
        self.video_fps = 0.0
        self.det_fps = 0.0
        self.link_error = None
        self.follow = ""
        self.yaw = ""
        self.meter = "seg"


class Demo:
    """Webcam in, segmentation overlay out. Quacks like the drone pilot the window already drives."""

    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.snap = Snapshot()
        self.seen = None
        self.follow = None
        self.last_frame = None
        self.state = "starting"
        self.closed = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._segmented = False
        self._said: tuple[str, ...] = ()
        self._camera_name = ""
        self._size = ""
        self._cap = None
        self._perception = None
        self._video_times: deque[float] = deque()
        self._seg_times: deque[float] = deque()
        self._thread = threading.Thread(target=self._run, name="demo", daemon=True)
        self._grabber: threading.Thread | None = None

    def start(self) -> Demo:
        self._thread.start()
        return self

    def send(self, text: str) -> None:
        cmd = text.strip().lower()
        if not cmd:
            return
        self.log(text.strip(), "sent")
        if cmd in ("e", "emergency"):
            self._set_state("emergency")
            self.log("closing the camera", "warn")
            self._stop.set()
        elif cmd == "status":
            self.log(self._status_line(), "info")
        elif cmd == "quit":
            return
        elif cmd in ("takeoff", "land", "follow", "follow me", "hold", "resume", "stop", "come back", "cancel"):
            self.log("this window is the camera", "info")
        else:
            self.log("status, or quit", "warn")

    def request_land(self) -> None:
        return None

    def shutdown(self, land: bool = False) -> None:
        del land
        self._stop.set()
        self._thread.join(timeout=30)
        if self._grabber is not None:
            self._grabber.join(timeout=5)
        self.closed.set()

    def _set_state(self, state: str) -> None:
        self.state = state
        self.snap.state = state

    def _status_line(self) -> str:
        bits = [self._camera_name or "camera"]
        if self._size:
            bits.append(self._size)
        bits.append(f"video {self.snap.video_fps:.0f} fps")
        bits.append(f"seg {self.snap.det_fps:.0f} fps")
        return "  ".join(bits)

    def _rate(self, bucket: deque[float], now: float) -> float:
        cutoff = now - 1.0
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        bucket.append(now)
        return float(len(bucket))

    def _publish(self, image, segmented: bool) -> None:
        with self._lock:
            if segmented:
                self._segmented = True
            elif self._segmented:
                return
            self._seq += 1
            self.last_frame = Shown(image, self._seq)

    def _take_latest(self):
        with self._lock:
            frame = self._latest
            self._latest = None
        return frame

    def _grab(self) -> None:
        cap = self._cap
        try:
            while not self._stop.is_set() and cap is not None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._stop.wait(0.05)
                    continue
                frame = frame.copy()
                now = time.monotonic()
                with self._lock:
                    self._latest = frame
                    self.snap.video_fps = self._rate(self._video_times, now)
                    height, width = frame.shape[:2]
                    self._size = f"{width}x{height}"
                self._publish(frame, False)
        finally:
            if cap is not None:
                cap.release()

    def _open_camera(self):
        requested = (getattr(self.args, "camera", None) or "").strip() or None
        names = list_dshow_videos()
        errors = []
        for name in camera_order(names, requested):
            if name not in names:
                errors.append(f"{name}: not in the camera list")
                continue
            index = names.index(name)
            try:
                cap = open_index(index)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                continue
            self._camera_name = name
            self._cap = cap
            return cap
        detail = "; ".join(errors) if errors else "no camera listed"
        raise RuntimeError(detail)

    def _run(self) -> None:
        try:
            self._open_camera()
        except Exception as exc:
            self._set_state("offline")
            self.log(f"camera did not open: {exc}", "error")
            self.closed.set()
            return
        self._set_state("live")
        self.log(f"camera: {self._camera_name}", "info")
        self._grabber = threading.Thread(target=self._grab, name="webcam", daemon=True)
        self._grabber.start()
        try:
            self.log("loading SegFormer-B0 ...", "info")
            self._perception = load_perception(getattr(self.args, "device", "") or "")
        except Exception as exc:
            self.log(f"segmentation unavailable: {exc}", "error")
            while not self._stop.wait(0.2):
                pass
            return
        if self._stop.is_set():
            close = getattr(self._perception, "close", None)
            if close is not None:
                close()
            return
        self.log("segmentation ready", "info")
        _vendor_path()
        from streetseg.camera import Pose
        from streetseg.sources import C270_DIAGONAL_FOV

        pose = Pose(WEBCAM_HEIGHT_M, WEBCAM_PITCH_DEG)
        camera = None
        while not self._stop.is_set():
            frame = self._take_latest()
            if frame is None:
                self._stop.wait(0.01)
                continue
            if camera is None or camera.cx != frame.shape[1] / 2.0:
                camera = camera_for(frame.shape, C270_DIAGONAL_FOV)
            try:
                events = self._perception.process(frame, camera, pose)
            except Exception as exc:
                self.log(f"segmentation frame failed: {exc}", "warn")
                continue
            now = time.monotonic()
            self.snap.det_fps = self._rate(self._seg_times, now)
            if self._perception.image is not None:
                self._publish(self._perception.image, True)
            shown = console_lines(events)
            if shown and shown != self._said:
                self._said = shown
                for text in shown:
                    self.log(text, "info")
        close = getattr(self._perception, "close", None)
        if close is not None:
            close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python run/demo.py", description="kite window with SegFormer-B0 on the webcam")
    parser.add_argument("--camera", default="", help="DirectShow name to open (default: the Logitech C270 if it is connected)")
    parser.add_argument("--device", default="", help="torch device for the models (default: cpu)")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    from ui import App

    app = App(
        lambda log: Demo(args, log),
        title="kite",
        opening="starting: opening the camera and loading segmentation ...",
        hint="status · quit",
        close_note="closing the camera ...",
        shutdown_note="closing the camera ...",
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
