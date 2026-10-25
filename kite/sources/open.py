"""One way to open any frame source for perception: open_source(spec) -> read() / .pose / .camera, like SimFrameSource.

    from kite.sources import open_source
    with open_source("webcam:iPhone") as src:      # or "tello", "sim:chest", "clip.mp4", "recordings/<take>", 0
        while True:
            ok, bgr = src.read()
            if not ok:
                break                              # end of a file / take, or no new live frame for `timeout` s
            events = process(bgr, src.camera, src.pose)

Specs (case-insensitive kinds):
    tello                   live Tello, dry run: video + telemetry only, motion is never sent. Newest frame only.
    tello:<backend>         the same with one video reader: pyav | opencv | raw | auto (default pyav)
    recording:<dir>, <dir>  a kite.record take (a folder with meta.json / frames.csv): RecordingSource, every frame.
    <file>, file:<file>     a video file: every frame in order (realtime=True paces it at the file's fps).
    webcam:<n>, <n>         camera index n (cv2.VideoCapture(n)). Newest frame only.
    webcam:<name>           the camera whose name contains <name>, e.g. webcam:iPhone (Continuity Camera).
    sim:drone, sim:chest    the browser sim's cameras through sim/frames.py (sim relay running).
    streetview:route        Google Street View along sim/public/osm/route.json (GOOGLE_MAPS_API_KEY; billed per image)
    streetview:<lat>,<lng>,<heading>   one Street View image. Options: kite/sources/streetview.py (pitch, hfov_deg,
                            step_m, cache_dir, ...)

Every source has: read() -> (ok, bgr) (BGR uint8, yours to draw on), .pose (dict for the frame just read, None before
the first; keys in kite.sources.pose.POSE_KEYS, None where unknown), .camera (kite.camera.Camera at the size of the
frames actually read), .name, release(), and works as a context manager.

Options (keyword arguments; an option a source does not take raises TypeError):
    camera=Camera(...) / fx=px / hfov_deg=deg   intrinsics instead of the source's default (all but recordings)
    cam_height=m, pitch=deg                     pose of a webcam (default 1.3 m, 0: a chest camera) or a video file
    realtime=True, speed=x                      pace a file or recording at its own frame rate
    timeout=s                                   live sources: how long read() waits for a new frame
    video_backend=pyav|opencv|raw|auto   tello: the H.264 reader (default DroneCfg's pyav; auto: raw on macOS)

    python -m kite.sources.open <spec> [--seconds 5] [--preview]    does this source work? fps, size, camera, poses
    python -m kite.sources.open --list-cameras [--probe]            this Mac's cameras with their indices
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..camera import Camera, load_camera
from ..config import VIDEO_BACKENDS
from .opencv_sources import WebcamSource, _ThreadedCapture
from .pose import MAX_GAP_S, PITCH_SIGN, POSE_KEYS, TelemetryTrack, empty_pose
from .recording import RecordingSource

ROOT = Path(__file__).resolve().parent.parent.parent
SIM_FRAMES = ROOT / "sim" / "frames.py"
SIM_CONFIG = ROOT / "sim" / "src" / "config.ts"
SIM_MODULE = "kite_sim_frames"  # sys.modules name sim/frames.py is loaded under
SIM_FX, SIM_SIZE = 1000.0, (960, 720)  # fallback if sim/src/config.ts cannot be read (spec 5.4)
CHEST_HEIGHT_M = 1.3  # default webcam pose: a chest camera on an adult, level
WEBCAM_HFOV_DEG = 68.0  # ASSUMED: rough horizontal FOV of a laptop webcam / iPhone main camera in 16:9 video
WEBCAM_SIZE = (1280, 720)  # requested; the camera picks its nearest mode and .camera follows the real size
FIRST_FRAME_S = 15.0  # live sources: the first frame may take this long (Tello keyframes, camera warm-up)

CameraFor = Callable[[int, int], Camera]  # frame width, height -> intrinsics


def _camera_for(default: CameraFor, camera: Camera | None, fx: float | None, hfov_deg: float | None) -> CameraFor:
    """Intrinsics at any frame size: an explicit Camera (scaled), fx (px at the frame size), a FOV, or the default."""
    if camera is not None:
        return camera.scaled
    if fx is not None:
        return lambda w, h: Camera(w, h, float(fx), float(fx), w / 2, h / 2, f"fx {fx:g} px from options")
    if hfov_deg is not None:
        return lambda w, h: Camera.from_hfov(w, h, hfov_deg, f"uncalibrated: {hfov_deg:g} deg horizontal FOV from options")
    return default


def tello_camera(width: int, height: int) -> Camera:
    """load_camera() (the Tello's intrinsics) at width x height. Another aspect ratio (a 16:9 stock clip) keeps the
    horizontal field of view with square pixels instead of stretching the 4:3 intrinsics."""
    cam = load_camera()
    if width * cam.height == height * cam.width:
        return cam.scaled(width, height)
    s = width / cam.width
    return Camera(width, height, cam.fx * s, cam.fy * s, width / 2, height / 2,
                  f"{cam.source}, scaled to {width}x{height} keeping the horizontal FOV")


class Source:
    """The SimFrameSource / RecordingSource shape. Subclasses implement _next() -> (bgr | None, pose)."""

    name = "source"

    def __init__(self, name: str, camera_for: CameraFor, size: tuple[int, int]):
        self.name = name
        self._camera_for = camera_for
        self.camera = camera_for(*size)  # nominal until the first frame, then the size actually read
        self.pose: dict | None = None
        self.frames = 0  # frames returned so far

    def _next(self) -> tuple[np.ndarray | None, dict | None]:
        raise NotImplementedError

    def read(self) -> tuple[bool, np.ndarray | None]:
        img, pose = self._next()
        if img is None:
            return False, None
        h, w = img.shape[:2]
        if (w, h) != (self.camera.width, self.camera.height):
            self.camera = self._camera_for(w, h)
        self.pose = pose
        self.frames += 1
        return True, img

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ------------------------------------------------------------------ video file
class FileSource(Source):
    """Every frame of a video file in order; realtime=True sleeps to its fps (speed=2: twice as fast). The pose has
    t = seconds into the file and cam_height / pitch only if given. Intrinsics: tello_camera() (the Tello's, scaled)."""

    def __init__(self, path: str | Path, realtime: bool = False, speed: float = 1.0, cam_height: float | None = None,
                 pitch: float | None = None, camera: Camera | None = None, fx: float | None = None,
                 hfov_deg: float | None = None):
        import cv2

        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {self.path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.realtime, self.speed, self.cam_height, self.pitch = realtime, speed, cam_height, pitch
        size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960, int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720)
        super().__init__(f"file:{self.path}", _camera_for(tello_camera, camera, fx, hfov_deg), size)
        self.index = -1
        self._t0: float | None = None

    def _next(self):
        if self._cap is None:
            return None, None
        i = self.index + 1
        if self.realtime:
            now = time.monotonic()
            if self._t0 is None:
                self._t0 = now - i / self.fps / self.speed
            wait = self._t0 + i / self.fps / self.speed - now
            if wait > 0:
                time.sleep(wait)
        ok, img = self._cap.read()
        if not ok:
            return None, None
        self.index = i
        return img, empty_pose(i / self.fps, cam_height=self.cam_height, pitch=self.pitch)

    def __len__(self) -> int:
        import cv2

        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap is not None else 0

    def isOpened(self) -> bool:
        return self._cap is not None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None


# ------------------------------------------------------------------ live sources (newest frame only)
class _LiveSource(Source):
    """read() waits (up to timeout, FIRST_FRAME_S for the first frame) for a frame newer than the last one returned
    and returns a copy of the newest; frames that arrived in between are skipped. (False, None) after a timeout."""

    def __init__(self, name, camera_for, size, timeout: float):
        super().__init__(name, camera_for, size)
        self.timeout = timeout
        self._frames = None  # a kite FrameSource
        self._seq = None

    def _newest(self):
        limit = max(self.timeout, FIRST_FRAME_S) if self._seq is None else self.timeout
        t_end = time.monotonic() + limit
        while True:
            f = self._frames.read() if self._frames is not None else None
            if f is not None and f.seq != self._seq:
                self._seq = f.seq
                return f
            if time.monotonic() >= t_end:
                return None
            time.sleep(0.003)

    def isOpened(self) -> bool:
        return self._frames is not None


class _PyAVCamera:
    """cv2.VideoCapture-like reader for an AVFoundation camera opened BY NAME through PyAV, for cameras OpenCV's
    device list does not have (macOS can hide Continuity Camera from it)."""

    def __init__(self, device: str, width: int, height: int, fps: int = 30):
        import av

        self._av = av
        last = None
        for opts in ({"framerate": str(fps), "video_size": f"{width}x{height}"}, {"framerate": str(fps)}, {}):
            try:
                self._c = av.open(f"{device}:none", format="avfoundation", options=opts)
                break
            except Exception as e:  # unsupported mode: try looser options
                last = e
        else:
            raise RuntimeError(f"cannot open camera {device!r} through PyAV: {last}")
        self._decode = self._c.decode(video=0)

    def read(self):
        try:
            return True, next(self._decode).to_ndarray(format="bgr24")
        except (StopIteration, self._av.FFmpegError):
            return False, None

    def set(self, *_):
        return False

    def release(self) -> None:
        self._c.close()


class _PyAVWebcam(_ThreadedCapture):
    def __init__(self, device: str, width: int, height: int):
        super().__init__("webcam")
        self.device, self.width, self.height = device, width, height

    def _open(self):
        return _PyAVCamera(self.device, self.width, self.height)


class CameraSource(_LiveSource):
    """A webcam or Continuity Camera: cv2 index, or a device name opened through PyAV (device=). Pose: cam_height /
    pitch as given (default a level chest camera at 1.3 m); t = arrival time (time.time()). Intrinsics: a rough
    default from WEBCAM_HFOV_DEG at the real frame size, labelled uncalibrated, unless camera / fx / hfov_deg."""

    def __init__(self, index: int | None = 0, device: str | None = None, label: str = "", width: int = WEBCAM_SIZE[0],
                 height: int = WEBCAM_SIZE[1], cam_height: float | None = CHEST_HEIGHT_M, pitch: float | None = 0.0,
                 camera: Camera | None = None, fx: float | None = None, hfov_deg: float | None = None,
                 timeout: float = 5.0):
        default = lambda w, h: Camera.from_hfov(  # noqa: E731
            w, h, WEBCAM_HFOV_DEG, f"uncalibrated: assumed {WEBCAM_HFOV_DEG:g} deg horizontal FOV")
        where = f"webcam:{index}" if device is None else f"webcam:{device!r} (PyAV)"
        super().__init__(f"{where} {label}".strip(), _camera_for(default, camera, fx, hfov_deg), (width, height), timeout)
        self.cam_height, self.pitch = cam_height, pitch
        src = WebcamSource(index, width, height) if device is None else _PyAVWebcam(device, width, height)
        self._frames = src.start()  # RuntimeError if it does not open

    def _next(self):
        f = self._newest()
        if f is None:
            return None, None
        return f.image.copy(), empty_pose(f.t, cam_height=self.cam_height, pitch=self.pitch)

    def release(self) -> None:
        if self._frames is not None:
            self._frames.stop()
        self._frames = None


class TelloSource(_LiveSource):
    """The live Tello in DRY RUN: make_tello(cfg, dry_run=True), so only 'command', 'streamon' / 'streamoff' and
    queries ever reach the drone (TelloDrone logs any motion instead of sending it, and nothing here calls one).
    Hold it powered on with motors off. Pose: telemetry around the frame's arrival time, the way RecordingSource
    aligns a take (kite.sources.pose.TelemetryTrack): heading = Tello yaw, cam_height, pitch, roll, battery.
    .telemetry: the newest converted state packet."""

    def __init__(self, cfg=None, timeout: float = 5.0, max_gap_s: float = MAX_GAP_S, pitch_sign: int = PITCH_SIGN,
                 camera: Camera | None = None, fx: float | None = None, hfov_deg: float | None = None,
                 poll_s: float = 0.02, video_backend: str | None = None):
        from ..config import Config, resolve_video_backend
        from ..drone import make_tello

        cfg = cfg or Config()
        if video_backend:
            resolve_video_backend(video_backend)  # ValueError before touching the drone
            cfg = replace(cfg, drone=replace(cfg.drone, video_backend=video_backend))  # never edit the caller's cfg
        backend = resolve_video_backend(cfg.drone.video_backend)
        super().__init__(f"tello (live, dry run, {backend} video)", _camera_for(tello_camera, camera, fx, hfov_deg),
                         (960, 720), timeout)
        self.max_gap_s, self.pitch_sign, self.poll_s = max_gap_s, pitch_sign, poll_s
        drone = make_tello(cfg, dry_run=True)
        if not getattr(drone, "dry_run", False):  # make_tello must honour dry_run=True; never go on without it
            raise RuntimeError("tello source: the drone is not in dry run, refusing to open it")
        self._drone = drone  # private on purpose: nothing outside this class should get at its motion methods
        self.telemetry = None
        self._tel: deque = deque(maxlen=50)  # ~5 s of state packets
        self._lock = threading.Lock()
        self._halt = threading.Event()
        try:
            drone.connect()  # 'command', then 'streamon' until video packets arrive
            self._frames = drone.frame_source()
        except Exception:
            drone.close()
            raise
        self._poller = threading.Thread(target=self._poll, name="tello-telemetry", daemon=True)
        self._poller.start()

    def _poll(self) -> None:
        last = None
        while not self._halt.wait(self.poll_s):
            tel = self._drone.telemetry()  # stamped when first seen: polling often keeps the stamp close to arrival
            if tel.t and tel.t != last:
                last = tel.t
                with self._lock:
                    self._tel.append(tel)
                    self.telemetry = tel

    def _next(self):
        f = self._newest()
        if f is None:
            return None, None
        pose = empty_pose(f.t, heading_ref="tello_yaw", flying=self._drone.flying)
        with self._lock:
            track = TelemetryTrack(list(self._tel), self.pitch_sign)
        track.fill(pose, f.t, self.max_gap_s)
        return f.image.copy(), pose

    def release(self) -> None:
        self._halt.set()
        if self._frames is not None:
            self._frames = None
            self._drone.close()  # stops the video reader and ends djitellopy (dry run: never lands, never flies)


# ------------------------------------------------------------------ sim
def sim_camera() -> Camera:
    """The sim's virtual camera (both drone and chest): FX, CAM_W, CAM_H from sim/src/config.ts."""
    fx, (w, h) = SIM_FX, SIM_SIZE
    try:
        text = SIM_CONFIG.read_text()
        vals = {k: float(v) for k, v in re.findall(r"export const (FX|CAM_W|CAM_H)\s*=\s*([0-9.]+)", text)}
        fx, w, h = vals.get("FX", fx), int(vals.get("CAM_W", w)), int(vals.get("CAM_H", h))
        src = "sim (sim/src/config.ts FX)"
    except OSError:
        src = "sim (default FX 1000)"
    return Camera(w, h, fx, fx, w / 2, h / 2, src)


def _sim_module():
    mod = sys.modules.get(SIM_MODULE)
    if mod is None:
        spec = importlib.util.spec_from_file_location(SIM_MODULE, SIM_FRAMES)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except ModuleNotFoundError as e:
            raise RuntimeError(f"sim sources need the {e.name!r} package (pip install websockets)") from e
        sys.modules[SIM_MODULE] = mod
    return mod


class SimSource(Source):
    """sim/frames.py's SimFrameSource ("drone" or "chest"), its pose passed through (x, y metres east / north,
    heading compass degrees, cam_height, pitch), the other pose keys None. read() blocks up to `timeout` for the next
    frame (the sim sends ~5 to 10 fps)."""

    def __init__(self, which: str = "drone", url: str | None = None, timeout: float = 2.0,
                 camera: Camera | None = None, fx: float | None = None, hfov_deg: float | None = None):
        cam = sim_camera()
        super().__init__(f"sim:{which}", _camera_for(cam.scaled, camera, fx, hfov_deg), (cam.width, cam.height))
        kw = {"url": url} if url else {}
        self._sim = _sim_module().SimFrameSource(which, timeout=timeout, **kw)

    def _next(self):
        ok, img = self._sim.read()
        if not ok or img is None:
            return None, None
        pose = empty_pose(time.time(), heading_ref="compass")
        pose.update({k: v for k, v in (self._sim.pose or {}).items() if k in POSE_KEYS})
        return img, pose

    def isOpened(self) -> bool:
        return self._sim is not None

    def release(self) -> None:
        if self._sim is not None:
            self._sim.release()
        self._sim = None


# ------------------------------------------------------------------ the factory
def parse_spec(spec) -> tuple[str, object]:
    """spec -> (kind, argument): ("tello", None), ("recording", Path), ("file", Path), ("webcam", int | name),
    ("sim", "drone" | "chest")."""
    if isinstance(spec, int):
        return "webcam", spec
    s = str(spec).strip()
    low = s.lower()
    if s.isdigit():
        return "webcam", int(s)
    if low == "tello":
        return "tello", None
    if low.startswith("tello:"):
        from ..config import resolve_video_backend

        backend = low.partition(":")[2].strip()
        resolve_video_backend(backend)  # ValueError for an unknown reader
        return "tello", backend
    if low == "webcam":
        return "webcam", 0
    if low == "sim":
        return "sim", "drone"
    kind, sep, rest = s.partition(":")
    kind = kind.lower()
    if sep and kind == "webcam":
        rest = rest.strip()
        return "webcam", int(rest) if rest.isdigit() else rest
    if sep and kind == "sim":
        which = rest.strip().lower() or "drone"
        if which not in ("drone", "chest"):
            raise ValueError(f"{spec!r}: use sim:drone or sim:chest")
        return "sim", which
    if kind == "streetview":
        return "streetview", rest
    if sep and kind == "recording":
        return "recording", Path(rest).expanduser()
    if sep and kind in ("file", "video"):
        return "file", Path(rest).expanduser()
    p = Path(s).expanduser()
    if p.is_dir():
        if (p / "meta.json").is_file() or (p / "frames.csv").is_file():
            return "recording", p
        raise ValueError(f"{p} is a folder but not a kite.record take (no meta.json / frames.csv)")
    if p.is_file():
        return "file", p
    raise ValueError(f"unknown source {spec!r}: use tello, sim:drone, sim:chest, webcam:<index or name>, <index>, "
                     "streetview:route, streetview:<lat>,<lng>,<heading>, "
                     "recording:<dir>, a take folder, or a video file (which must exist)")


def open_source(spec, **opts):
    """Open a frame source from a spec string (see this module's docstring). Returns an object with read() ->
    (ok, bgr), .pose, .camera, .name, release(), usable as a context manager."""
    kind, arg = parse_spec(spec)
    if kind == "tello":
        if arg:  # tello:<backend>
            opts.setdefault("video_backend", arg)
        return TelloSource(**opts)
    if kind == "recording":
        rec = RecordingSource(arg, **opts)
        rec.name = f"recording:{arg}"
        return rec
    if kind == "file":
        if not Path(arg).is_file():
            raise FileNotFoundError(f"no video file {arg}")
        return FileSource(arg, **opts)
    if kind == "sim":
        return SimSource(arg, **opts)
    if kind == "streetview":
        from .streetview import StreetViewSource, parse

        return StreetViewSource(**parse(arg), **opts)
    from .cameras import find_camera, opencv_cameras

    if isinstance(arg, int):
        label = next((n for i, n, _ in opencv_cameras() if i == arg), "")
        return CameraSource(arg, label=label, **opts)
    index, name = find_camera(arg)
    if index is None:  # only AVFoundation's discovery list has it: open by name through PyAV
        return CameraSource(None, device=name, **opts)
    return CameraSource(index, label=name, **opts)


# ------------------------------------------------------------------ CLI
def _fmt_pose(p: dict | None) -> str:
    if p is None:
        return "none"
    out = []
    for k in ("t", "cam_height", "pitch", "roll", "heading", "x", "y", "battery", "tel_dt"):
        v = p.get(k)
        if v is not None:
            out.append(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}")
    return ", ".join(out) or "all None"


def _fmt_camera(c: Camera) -> str:
    return (f"{c.width}x{c.height} fx {c.fx:.0f} fy {c.fy:.0f} cx {c.cx:.0f} cy {c.cy:.0f} px, "
            f"hfov {c.hfov_deg:.1f} deg ({c.source})")


def list_cameras(probe: bool = False) -> None:
    from .cameras import camera_permission, opencv_cameras, system_cameras

    perm = camera_permission()
    if perm:
        print(f"camera permission for this program: {perm}")
    cv = opencv_cameras()
    print("OpenCV cameras (webcam:<index>, or webcam:<part of the name>):")
    for i, name, uid in cv:
        size = ""
        if probe:  # opens each camera for one frame
            import cv2

            cap = cv2.VideoCapture(i)
            ok, img = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            size = f"  {img.shape[1]}x{img.shape[0]}" if ok else "  (no frame)"
        print(f"  {i}  {name}{size}   [{uid}]")
    if not cv:
        print("  none found")
    extra = [n for n in system_cameras() if n not in {name for _, name, _ in cv}]
    if extra:
        print("Only in AVFoundation's discovery list (webcam:<name> opens them through PyAV, by name):")
        for n in extra:
            print(f"  -  {n}")


def _hud(img: np.ndarray, src, fps: float) -> np.ndarray:
    import cv2

    out = img.copy()
    for i, line in enumerate([f"{src.name}  {fps:.1f} fps", _fmt_pose(src.pose)]):
        y = 24 + 24 * i
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def check(src, seconds: float, preview: bool = False, n_poses: int = 5) -> dict:
    """Read for `seconds` (or to the end) and print what a detector would get."""
    print(f"source  {src.name}")
    print(f"camera  {_fmt_camera(src.camera)}  (before the first frame)")
    t0 = time.monotonic()
    times, sizes, poses = [], {}, []
    reason = f"{seconds:g} s elapsed"
    while time.monotonic() - t0 < seconds:
        ok, img = src.read()
        now = time.monotonic()
        if not ok:
            reason = ("end of the file / take" if isinstance(src, (FileSource, RecordingSource))
                      else "no new frame within the timeout")
            break
        times.append(now)
        sz = f"{img.shape[1]}x{img.shape[0]}"
        sizes[sz] = sizes.get(sz, 0) + 1
        if len(poses) < n_poses and now - t0 >= len(poses) * seconds / n_poses:
            poses.append(src.pose)
        if preview:
            import cv2

            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
            cv2.imshow("kite source", _hud(img, src, fps))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                reason = "quit"
                break
    if preview:
        import cv2

        cv2.destroyAllWindows()
    n = len(times)
    fps = (n - 1) / (times[-1] - times[0]) if n > 1 and times[-1] > times[0] else None
    print(f"frames  {n} ({fps:.1f} fps)" if fps else f"frames  {n}", end="")
    print(f", first after {times[0] - t0:.2f} s" if times else "", end="")
    print(f", size {', '.join(f'{k} x{v}' for k, v in sizes.items())}" if sizes else "", f"; stopped: {reason}")
    if n:
        print(f"camera  {_fmt_camera(src.camera)}")
    for p in poses:
        print(f"pose    {_fmt_pose(p)}")
    return {"frames": n, "fps": fps, "sizes": sizes, "poses": poses, "reason": reason}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kite.sources.open", description="Check that a frame source works.")
    p.add_argument("spec", nargs="?", help="tello | tello:raw | sim:drone | sim:chest | webcam:<index or name> | <index> | "
                                           "recording:<dir> | <take dir> | <video file>")
    p.add_argument("--seconds", type=float, default=5.0, help="read for this long (default 5)")
    p.add_argument("--preview", action="store_true", help="show the frames")
    p.add_argument("--list-cameras", action="store_true", help="this Mac's cameras with their OpenCV indices")
    p.add_argument("--probe", action="store_true", help="with --list-cameras: grab one frame from each (frame size)")
    p.add_argument("--realtime", action="store_true", help="files and recordings: play at their own frame rate")
    p.add_argument("--cam-height", type=float, help="webcam / file: camera height above the ground, m")
    p.add_argument("--pitch", type=float, help="webcam / file: camera pitch, deg (+ = up)")
    p.add_argument("--fx", type=float, help="focal length in px at the frame size (instead of the default)")
    p.add_argument("--hfov", type=float, help="horizontal field of view in deg (instead of the default)")
    p.add_argument("--video-backend", choices=VIDEO_BACKENDS, default=None,
                   help="tello: the H.264 reader (default pyav; also opencv; opt-in: raw, auto = raw on macOS)")
    args = p.parse_args(argv)
    if args.list_cameras:
        list_cameras(args.probe)
        return 0
    if not args.spec:
        p.error("give a source spec, or --list-cameras")
    opts = {k: v for k, v in (("cam_height", args.cam_height), ("pitch", args.pitch), ("fx", args.fx),
                              ("hfov_deg", args.hfov), ("video_backend", args.video_backend)) if v is not None}
    try:
        if (args.realtime or args.preview) and parse_spec(args.spec)[0] in ("file", "recording"):
            opts["realtime"] = True
        src = open_source(args.spec, **opts)
    except TypeError as e:
        print(f"option not supported by this source: {e}")
        return 2
    except (ValueError, LookupError, FileNotFoundError, RuntimeError, OSError) as e:
        print(e)
        return 1
    with src:
        res = check(src, args.seconds, args.preview)
    return 0 if res["frames"] else 1


if __name__ == "__main__":
    sys.exit(main())
