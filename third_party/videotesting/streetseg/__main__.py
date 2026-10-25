"""Live street segmentation.

    python -m streetseg.download
    python -m streetseg --demo
    python -m streetseg --video clip.mp4
    python -m streetseg --webcam 0 --chest 1 --depth
    python -m streetseg --tello
    python -m streetseg --web
    python -m streetseg --web --webcam 0 --tello
    python -m streetseg --sim
    python -m streetseg --send clip.mp4

The window is the segmentation. q closes it. Events are JSON lines.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .camera import (
    CHEST_POSE,
    DEMO_POSE,
    DRONE_POSE,
    FRAME_H,
    FRAME_W,
    TELLO_CAMERA,
    Camera,
    Pose,
    focal_from_horizontal_fov,
    focal_length,
)
from .labels import NAMES
from .live import play
from .pipeline import Perception, parallel_for_device
from .scene import synthetic_street


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Segment a live view and emit earpiece events.")
    parser.add_argument("--demo", action="store_true", help="synthetic street, no weights")
    parser.add_argument("--download", action="store_true", help="fetch weights, then exit")
    parser.add_argument("--video", type=str, default="", help="mp4 or avi")
    parser.add_argument("--webcam", type=int, default=None, help="camera index. With --web, the Logitech C270 at 1280×720")
    parser.add_argument("--chest", type=int, default=None, help="chest camera index")
    parser.add_argument("--tello", action="store_true", help="Tello stream; does not take off")
    parser.add_argument("--web", action="store_true", help="browser page: C270 and Tello, input feed beside output feed")
    parser.add_argument("--droidcam", default="", help="optional phone IP or http://ip:4747/video")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--sim", action="store_true", help="listen for sim JPEG frames")
    parser.add_argument("--sim-port", type=int, default=8765)
    parser.add_argument("--sim-url", type=str, default="", help="connect out instead of listening")
    parser.add_argument("--send", type=str, default="", help="push this video to a sim socket")
    parser.add_argument("--send-url", type=str, default="ws://127.0.0.1:8765")
    parser.add_argument("--send-fps", type=float, default=8.0)
    parser.add_argument("--role", choices=("drone", "sim", "chest"), default="")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--fov", type=float, default=None, help="degrees, diagonal for a forward view, horizontal for chest")
    parser.add_argument("--height", type=float, default=None, help="camera height, meters")
    parser.add_argument("--pitch", type=float, default=None, help="degrees, positive looks down")
    parser.add_argument("--chest-height", type=float, default=CHEST_POSE.height)
    parser.add_argument("--chest-pitch", type=float, default=CHEST_POSE.pitch)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--segmenter", choices=("mapillary", "yolo", "segformer"), default="mapillary")
    parser.add_argument("--sem-weights", default="yolo26n-sem.pt")
    parser.add_argument("--seg-weights", default="segmentation/SegFormer_b0_Mapillary.pth")
    parser.add_argument("--det-weights", default="yolo26n.pt")
    parser.add_argument("--no-lights", action="store_true", help="skip the PVTL traffic-light classifier")
    parser.add_argument("--light-weights", default="traffic_light_classification/resnet18-best.pth")
    parser.add_argument("--signs", action="store_true")
    parser.add_argument("--sign-weights", default="")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--hold-ms", type=int, default=0, help="with --demo, keep the window up this long")
    args = parser.parse_args(argv)
    if (args.fx is None) != (args.fy is None):
        parser.error("--fx and --fy are set together")

    if args.web:
        return _web(args)
    if args.download:
        from .download import main as download_main

        extra = []
        if args.signs:
            extra.append("--signs")
        if args.depth:
            extra.append("--depth")
        return download_main(extra)
    if args.send:
        from .sources import send_jpegs

        send_jpegs(args.send, args.send_url, args.send_fps)
        return 0
    if args.demo or not _live_source(args):
        return _demo(args)
    return _live(args)


def _live_source(args) -> bool:
    return bool(args.video or args.tello or args.sim or args.sim_url or args.webcam is not None or args.chest is not None)


def _demo(args) -> int:
    image, labels, boxes = synthetic_street(FRAME_W, FRAME_H)
    perception = Perception(_FixedSeg(labels), _FixedDet(boxes), hold_frames=1)
    camera = TELLO_CAMERA.with_role("sim")
    events = perception.process(image, camera, DEMO_POSE)
    for event in events:
        print(json.dumps(event.to_dict()))
    out = args.out or "streetseg/examples/waymo_style_overlay.png"
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    _imwrite(path, perception.image)
    print(f"wrote {path}", file=sys.stderr)
    if not args.no_show:
        from .display import Display

        display = Display(True)
        display.show(perception.image, "Kite street")
        if args.hold_ms:
            display.wait(args.hold_ms)
        elif sys.stdin.isatty():
            print("q closes the window", file=sys.stderr)
            while not display.poll_quit():
                if display.wait(30) in (ord("q"), 27, -1):
                    break
        display.close()
    return 0


def _web(args) -> int:
    from .sources import C270_DIAGONAL_FOV, open_c270, open_file, open_tello
    from .webapp import LiveBoards, droidcam_url, serve

    role = args.role or "drone"

    def factory():
        return _models(args, need_segmenter=role != "chest")

    boards = LiveBoards(
        factory,
        _camera(args, role),
        _pose(args, role),
        camera_fns=None
        if args.fx
        else {
            "c270": _fov_camera(args, role, C270_DIAGONAL_FOV),
            "tello": _fov_camera(args, role, 82.6),
            "phone": _fov_camera(args, role, 82.6),
        },
        tello_factory=open_tello,
        c270_factory=open_c270,
    )
    if args.webcam is not None:
        boards.connect_c270(args.webcam)
    if args.tello:
        boards.connect_tello()
    if args.droidcam:
        boards.connect_url(droidcam_url(args.droidcam))
    elif args.video and args.webcam is None:
        boards.connect_source(lambda: open_file(args.video), "Video", "c270")
    host = args.web_host
    print(f"Open http://{host}:{args.web_port}", file=sys.stderr)
    if host in ("0.0.0.0", "::"):
        print(f"On this machine that is http://127.0.0.1:{args.web_port}", file=sys.stderr)
    try:
        serve(boards, host, args.web_port)
    finally:
        boards.close()
    return 0


def _live(args) -> int:
    from .display import Display
    from .sources import SimSocketSource, open_file, open_tello, open_webcam

    role = args.role or ("sim" if (args.sim or args.sim_url) else "chest" if _chest_only(args) else "drone")
    perception = _models(args, need_segmenter=role != "chest")
    source = _open_primary(args, open_file, open_webcam, open_tello, SimSocketSource)
    try:
        chest = None if args.chest is None or role == "chest" else open_webcam(args.chest)
    except Exception:
        source.close()
        raise
    camera = _camera(args, role)
    pose = _pose(args, role)
    chest_camera = _camera(args, "chest")
    chest_pose = Pose(args.chest_height, args.chest_pitch)
    display = Display(enabled=not args.no_show)
    writer = {"handle": None}

    def on_event(event) -> None:
        print(json.dumps(event.to_dict()), flush=True)

    def on_image(image) -> None:
        if not args.out:
            return
        if writer["handle"] is None:
            writer["handle"] = _writer(args.out, image.shape, _fps(source))
        writer["handle"].write(image)

    try:
        play(
            perception,
            source,
            camera,
            pose,
            display,
            chest=chest,
            chest_camera=chest_camera,
            chest_pose=chest_pose,
            max_frames=args.max_frames,
            on_event=on_event,
            on_image=on_image if args.out else None,
            camera_fn=None if args.fx else _fov_camera(args, role),
        )
    finally:
        if writer["handle"] is not None:
            writer["handle"].release()
    return 0


def _fov_camera(args, role: str, default_fov: float = 82.6):
    ready = {"yes": False}

    def apply(frame, camera: Camera) -> Camera:
        if ready["yes"]:
            return camera
        height, width = frame.shape[:2]
        fov = args.fov
        if role == "chest":
            focal = focal_from_horizontal_fov(width, fov if fov is not None else 70.0)
        else:
            focal = focal_length(width, height, fov if fov is not None else default_fov)
        ready["yes"] = True
        return Camera(focal, focal, camera.role)

    return apply


def _models(args, need_segmenter: bool = True) -> Perception:
    from .models import DepthAnythingV2, SegFormerSegmenter, YoloDetector, YoloSemanticSegmenter

    device = args.device or None
    segmenter = None
    lights = None
    if need_segmenter:
        if args.segmenter == "mapillary":
            from .mapillary_seg import MapillarySegFormer

            segmenter = MapillarySegFormer(args.seg_weights, device=device)
        elif args.segmenter == "segformer":
            segmenter = SegFormerSegmenter(device=device)
        else:
            segmenter = YoloSemanticSegmenter(args.sem_weights, imgsz=args.imgsz, device=device)
        segmenter.load()
        if args.segmenter == "mapillary" and not args.no_lights:
            from .lights import TrafficLightClassifier, TrafficLightReader

            lights = TrafficLightReader(TrafficLightClassifier(args.light_weights, device=device))
            lights.load()
    detector = YoloDetector(args.det_weights, imgsz=args.imgsz, device=device, track=True)
    detector.load()
    signs = None
    if args.signs:
        path = args.sign_weights or str(Path(__file__).resolve().parent.parent / "weights" / "signs" / "best.pt")
        signs = YoloDetector(path, imgsz=args.imgsz, device=device, keep_all=True)
        signs.load()
    depth = DepthAnythingV2(device=device) if args.depth else None
    if depth is not None:
        depth.load()
    if segmenter is None:
        segmenter = _UnusedSeg()
    return Perception(
        segmenter,
        detector,
        sign_detector=signs,
        depth_estimator=depth,
        lights=lights,
        parallel=need_segmenter and parallel_for_device(device),
    )


def _chest_only(args) -> bool:
    return args.chest is not None and not (args.video or args.tello or args.sim or args.sim_url or args.webcam is not None)


def _open_primary(args, open_file, open_webcam, open_tello, sim_source):
    if args.tello:
        return open_tello()
    if args.sim or args.sim_url:
        return sim_source(
            port=args.sim_port,
            url=args.sim_url or None,
        )
    if args.video:
        return open_file(args.video)
    if args.webcam is not None:
        return open_webcam(args.webcam)
    return open_webcam(args.chest)


def _camera(args, role: str) -> Camera:
    if args.fx and args.fy:
        return Camera(args.fx, args.fy, role)
    if role == "chest":
        from .camera import CHEST_CAMERA

        return CHEST_CAMERA.with_role("chest")
    return TELLO_CAMERA.with_role(role)


def _pose(args, role: str) -> Pose:
    if args.height is not None or args.pitch is not None:
        base = CHEST_POSE if role == "chest" else DRONE_POSE
        return Pose(
            args.height if args.height is not None else base.height,
            args.pitch if args.pitch is not None else base.pitch,
        )
    if role == "chest":
        return CHEST_POSE
    if role == "sim":
        return Pose(DEMO_POSE.height, DEMO_POSE.pitch)
    return DRONE_POSE


def _fps(source) -> float:
    return float(getattr(source, "fps", 8.0) or 8.0)


def _imwrite(path: Path, image) -> None:
    import cv2

    if image is None or not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write {path}")


def _writer(path: str, shape, fps: float):
    height, width = shape[:2]
    if path.lower().endswith(".mp4") and shutil.which("ffmpeg"):
        return _H264Writer(path, width, height, max(fps, 1.0))
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, max(fps, 1.0), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open {path}")
    return writer


class _H264Writer:
    """H.264 so a browser can play the file. OpenCV's mp4v cannot."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        self.width = width
        self.height = height
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "high",
                "-movflags",
                "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def isOpened(self) -> bool:
        return self._proc.poll() is None and self._proc.stdin is not None

    def write(self, image) -> None:
        if image.shape[0] != self.height or image.shape[1] != self.width or image.shape[2] != 3:
            raise ValueError("frame size does not match the writer")
        if not self.isOpened():
            raise RuntimeError("video writer closed")
        self._proc.stdin.write(np.ascontiguousarray(image).tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        err = self._proc.stderr.read().decode("utf-8", "replace") if self._proc.stderr else ""
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(err.strip() or f"ffmpeg exited {code}")


class _FixedSeg:
    def __init__(self, labels):
        self._labels = labels
        self.names = dict(NAMES)

    def predict(self, frame):
        if frame.shape[:2] != self._labels.shape[:2]:
            raise ValueError("demo frame size does not match the labels")
        return self._labels


class _FixedDet:
    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, _frame):
        return list(self._boxes)


class _UnusedSeg:
    names = NAMES

    def predict(self, _frame):
        raise RuntimeError("the chest camera does not run street segmentation")

    def load(self):
        return self


if __name__ == "__main__":
    raise SystemExit(main())
