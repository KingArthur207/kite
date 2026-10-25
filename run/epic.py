"""kite: fly, and show the demo's camera processing on the pane.

    python run/epic.py                 # take off and follow; YOLO obstacle overlay on the picture
    python run/epic.py --dry-run --seg # on the table, with segmentation on as well

Takeoff, follow, land and e are run/app.py. The drone camera, the person lock and the yaw steering are the
pilot's, and nothing in the camera processing writes them. BODYCAM, DRONE VIEW, YOLO and Segmentation are the
picture: YOLO and segmentation run on a copy of whichever picture the pane is showing. The overlay and the
scene lines stay in the window. The body camera is the one the pilot already opens. DRONE VIEW is the frame
the pilot already has, not a second connection to the drone.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import build_parser as flight_parser  # noqa: E402
from app import check_flight, quiet_libraries  # noqa: E402
from demo import (DRONE_DIAGONAL_FOV, DRONE_HEIGHT_M, DRONE_PITCH_DEG, WEBCAM_HEIGHT_M,  # noqa: E402
                  WEBCAM_PITCH_DEG, camera_for, console_lines, load_perception, load_segmentation, meter_for,
                  switch_word)


class Picture:
    """A pane image. The body camera does not get the drone's person boxes drawn on top."""

    def __init__(self, image, seq: int, view: str):
        self.image = image
        self.seq = seq
        self.view = view
        self.overlay = view != "body"


class _PaneStatus:
    """The window's status line. The pane's model rate is shown here and is not written onto the pilot."""

    def __init__(self, epic: Epic):
        self._epic = epic

    def __getattr__(self, name):
        epic = self._epic
        if epic._pane_meter and name == "meter":
            return epic._meter
        if epic._pane_meter and name == "det_fps":
            return epic._det_fps
        return getattr(epic.pilot.snap, name)


class Epic:
    """The pilot flies. This runs the demo's YOLO and segmentation on a copy of the pane and stops there."""

    def __init__(self, args, log):
        from pilot import Pilot

        self.args = args
        self.log = log
        self.pilot = Pilot(args, log)
        self._seg_on = bool(getattr(args, "seg", False))
        self._yolo_on = not bool(getattr(args, "no_yolo", False))
        self._loaded = False
        self._seg_state = "loading"  # loading, ready, failed
        self._meter = "yolo"
        self._det_fps = 0.0
        self._pane_meter = False
        self._shown: Picture | None = None
        self._seq = 0
        self._raw_key = None
        self._mode = None
        self._said: tuple[str, ...] = ()
        self._perception = None
        self._times: deque[float] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pilot_started = False
        self._thread = threading.Thread(target=self._run, name="epic-view", daemon=True)

    def start(self) -> Epic:
        """The picture thread loads its models and then starts the pilot, so the two imports do not overlap."""
        self._thread.start()
        return self

    def send(self, text: str) -> None:
        """Flight words go to the pilot. YOLO and segmentation are the picture only."""
        cmd = text.strip().lower()
        if not cmd:
            return
        switch = switch_word(cmd)
        if switch is None:
            self.pilot.send(text)
            return
        self.log(text.strip(), "sent")
        key, on = switch
        self.toggle(key, on)

    def toggles(self) -> list[dict]:
        """The pilot's picture buttons, then the demo's two processing buttons."""
        loading = "" if self._loaded else "loading"
        seg_note = "unavailable" if self._seg_state == "failed" else loading
        return list(self.pilot.toggles()) + [
            {"key": "yolo", "label": "YOLO", "on": self._yolo_on, "note": loading if self._yolo_on else ""},
            {"key": "seg", "label": "Segmentation", "on": self._seg_on, "note": seg_note if self._seg_on else ""},
        ]

    def toggle(self, key: str, on: bool | None = None) -> None:
        """BODYCAM and DRONE VIEW are the pilot's. YOLO and segmentation never reach it."""
        if key in ("body", "drone"):
            self.pilot.toggle(key, on)
            return
        if key not in ("seg", "yolo"):
            return
        with self._lock:
            current = self._seg_on if key == "seg" else self._yolo_on
            now = (not current) if on is None else on
            if key == "seg":
                self._seg_on = now
            else:
                self._yolo_on = now
            seg, yolo = self._seg_on, self._yolo_on
            if not (seg or yolo):
                self._shown = None
                self._pane_meter = False
        name = "segmentation" if key == "seg" else "YOLO"
        self.log(f"{name} {'on' if now else 'off'}", "info")
        if not (seg or yolo):
            self.log("guidance off: camera only", "info")

    def display_frame(self):
        """The painted picture, or the pilot's raw one. Never the frame the follow reads."""
        with self._lock:
            shown = self._shown
            busy = self._seg_on or self._yolo_on
        if busy and shown is not None and shown.view == self.pilot._view:
            return shown
        return self.pilot.display_frame()

    @property
    def snap(self):
        return _PaneStatus(self)

    def shutdown(self, land: bool = True, timeout_s: float = 30.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        self._close_perception()
        if self._pilot_started:
            self.pilot.shutdown(land=land, timeout_s=timeout_s)

    def __getattr__(self, name):
        return getattr(self.pilot, name)

    def _raw_copy(self):
        """(view, image) for the picture the pane selected. The pilot's frame object is not the copy."""
        pilot = self.pilot
        view = pilot._view
        frame = pilot._body_frame if view == "body" else pilot.last_frame
        image = None if frame is None else getattr(frame, "image", None)
        if image is None:
            return None
        key = (view, getattr(frame, "seq", id(image)))
        if key == self._raw_key:
            return None
        self._raw_key = key
        copied = image.copy() if hasattr(image, "copy") else image
        return view, copied

    def _intrinsics(self, image, view: str):
        if view == "drone":
            fov, height, pitch = DRONE_DIAGONAL_FOV, DRONE_HEIGHT_M, DRONE_PITCH_DEG
        else:
            from streetseg.sources import C270_DIAGONAL_FOV

            fov, height, pitch = C270_DIAGONAL_FOV, WEBCAM_HEIGHT_M, WEBCAM_PITCH_DEG
        from streetseg.camera import Pose

        return camera_for(image.shape, fov), Pose(height, pitch)

    def _on_frame(self, view: str, image) -> None:
        """Paint one copied frame. The result is a picture and, at most, a console line."""
        with self._lock:
            seg = self._seg_on and self._seg_state == "ready"
            yolo = self._yolo_on
        if (seg, yolo) != self._mode:
            self._mode = (seg, yolo)
            self._perception.set_modes(seg, yolo)
            self._meter = meter_for(seg, yolo)
            self._times.clear()
            self._det_fps = 0.0
            self._said = ()
            self._pane_meter = bool(seg or yolo)
        if not (seg or yolo):
            with self._lock:
                self._shown = None
            return
        camera, pose = self._intrinsics(image, view)
        events = self._perception.process(image, camera, pose)
        now = time.monotonic()
        self._det_fps = self._rate(now)
        painted = getattr(self._perception, "image", None)
        if painted is not None:
            self._publish(painted.copy() if hasattr(painted, "copy") else painted, view)
        shown = console_lines(events)
        if shown and shown != self._said:
            self._said = shown
            for text in shown:
                self.log(text, "info")

    def _publish(self, image, view: str) -> None:
        with self._lock:
            if view != self.pilot._view:
                return
            if not (self._seg_on or self._yolo_on):
                self._shown = None
                return
            self._seq += 1
            self._shown = Picture(image, self._seq, view)

    def _rate(self, now: float) -> float:
        cutoff = now - 1.0
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
        self._times.append(now)
        return float(len(self._times))

    def _run(self) -> None:
        loaded = self._load()
        if self._stop.is_set():
            self._close_perception()
            return
        self.pilot.start()
        self._pilot_started = True
        # Pose YOLO is imported while the pilot is still "starting". Wait that out before a forward.
        while not self._stop.is_set() and self.pilot.state == "starting":
            self._stop.wait(0.05)
        if not loaded:
            while not self._stop.wait(0.2):
                pass
            self._close_perception()
            return
        while not self._stop.is_set():
            got = self._raw_copy()
            if got is None:
                self._stop.wait(0.01)
                continue
            view, image = got
            try:
                self._on_frame(view, image)
            except Exception as exc:
                self.log(f"guidance frame failed: {exc}", "warn")
        self._close_perception()

    def _load(self) -> bool:
        """The demo's two models. A failure here leaves the pilot flying with the plain picture."""
        device = getattr(self.args, "device", "") or ""
        self.log("loading YOLO26n and SegFormer-B0 for the picture ...", "info")
        try:
            self._perception = load_perception(device)
        except Exception as exc:
            self._seg_state = "failed"
            self.log(f"YOLO unavailable: {exc}", "error")
            return False
        try:
            self._perception.segmenter, self._perception.lights = load_segmentation(device)
            self._seg_state = "ready"
        except Exception as exc:
            self._seg_state = "failed"
            self.log(f"segmentation unavailable: {exc}", "error")
        self._loaded = True
        on = [name for name, used in (("YOLO", self._yolo_on), ("segmentation", self._seg_on)) if used]
        self.log(f"picture models ready; on: {', '.join(on) or 'nothing (camera only)'}", "info")
        return True

    def _close_perception(self) -> None:
        with self._lock:
            perception, self._perception = self._perception, None
        close = getattr(perception, "close", None)
        if close is not None:
            close()


def build_parser():
    parser = flight_parser()
    parser.prog = "python run/epic.py"
    parser.description = ("kite: follow the person in front. The pane can also run the demo's YOLO and "
                          "segmentation, and that picture never steers.")
    parser.add_argument("--seg", action="store_true",
                        help="start with segmentation drawn on the pane (both picture models load at start)")
    parser.add_argument("--no-yolo", action="store_true",
                        help="start with the pane's YOLO obstacle overlay off")
    parser.add_argument("--device", default="", help="torch device for the pane's models (default: cpu)")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    why = check_flight(args)
    if why:
        print(why)
        return 2
    quiet_libraries()

    from ui import App

    window = App(
        lambda log: Epic(args, log),
        title="kite" + ("  -  DRY RUN" if args.dry_run else ""),
        opening="starting: loading the picture models, then connecting to the drone ...",
        hint="bodycam · drone · yolo · seg · takeoff · land · e",
    )
    window.log(f"{'DRY RUN: nothing reaches the motors. ' if args.dry_run else ''}follow at {args.height:.2f} m, "
               f"{args.behind:.1f} m behind, steering: {args.yaw}", "info")
    window.log("camera processing is the picture only: it does not steer", "info")
    return window.run()


if __name__ == "__main__":
    sys.exit(main())
