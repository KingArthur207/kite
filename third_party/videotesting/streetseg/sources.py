"""Frame sources behind one ``read()``.

A video file, a webcam, and a live Tello all return a BGR frame or None.
Arthur's ``kite.sources`` readers are used when that package is installed;
these are the same contract when it is not.

Sim frames are JPEGs on a local WebSocket, about 5 to 10 fps. Perception
listens, and the sim pushes. A text message may carry intrinsics and pose::

    {"jpeg": "<base64>", "fx": 682, "fy": 682, "height": 1.6, "pitch": 12}

A binary message is the JPEG alone.
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
from pathlib import Path

import numpy as np

_KITE = {
    "file": ("FileSource", "VideoFileSource", "open_file"),
    "webcam": ("WebcamSource", "CameraSource", "open_webcam"),
    "tello": ("TelloSource", "open_tello"),
}

# Logitech C270: 720p UVC webcam. The published field of view is 60°.
C270_WIDTH = 1280
C270_HEIGHT = 720
C270_FPS = 30.0
C270_DIAGONAL_FOV = 60.0


class VideoFileSource:
    def __init__(self, path: str | Path):
        import cv2

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self._cv2 = cv2
        self.ended = False

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            self.ended = True
            return None
        return frame

    @property
    def fps(self) -> float:
        value = float(self.cap.get(self._cv2.CAP_PROP_FPS) or 0.0)
        return value if value > 1 else 20.0

    @property
    def size(self) -> tuple[int, int]:
        width = int(self.cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def close(self) -> None:
        self.ended = True
        self.cap.release()


def configure_capture(cap, width: int | None = None, height: int | None = None, fps: float | None = None) -> None:
    """Ask a UVC camera for a size. MJPEG lets the C270 deliver 720p."""
    import cv2

    def _set(prop, value) -> None:
        if prop is None:
            return
        try:
            cap.set(prop, value)
        except cv2.error:
            return

    _set(getattr(cv2, "CAP_PROP_BUFFERSIZE", None), 1)
    if width and height:
        _set(getattr(cv2, "CAP_PROP_FOURCC", None), cv2.VideoWriter_fourcc(*"MJPG"))
        _set(getattr(cv2, "CAP_PROP_FRAME_WIDTH", None), int(width))
        _set(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", None), int(height))
    if fps:
        _set(getattr(cv2, "CAP_PROP_FPS", None), float(fps))


class WebcamSource:
    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None, fps: float | None = None):
        import cv2

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {index}")
        configure_capture(self.cap, width, height, fps)
        self.ended = False
        self.failed_open = False

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            self.ended = True
            return None
        return frame

    def close(self) -> None:
        self.ended = True
        self.cap.release()


class TelloSource:
    """DJI Tello video via djitellopy. Connects and calls streamon only.

    The decoder starts on a black placeholder. Those frames are skipped until
    the drone camera sends a real picture. This path never takes off.
    """

    ended = False

    def __init__(self):
        try:
            from djitellopy import Tello
        except ImportError as exc:
            raise RuntimeError("Install djitellopy to pull the Tello camera.") from exc

        self._got_frame = False
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.reader = self.tello.get_frame_read()

    def read(self) -> np.ndarray | None:
        frame = self.reader.frame
        if frame is None:
            return None
        if not self._got_frame and int(np.max(frame)) == 0:
            return None
        self._got_frame = True
        return frame.copy()

    def close(self) -> None:
        self.ended = True
        try:
            self.tello.streamoff()
        finally:
            self.tello.end()


class AdoptedSource:
    """A ``kite.sources`` reader, exposed as ``read() -> BGR | None``."""

    def __init__(self, inner, end_on_none: bool):
        self.inner = inner
        self.end_on_none = end_on_none
        self.ended = False

    def read(self) -> np.ndarray | None:
        frame = as_frame(self.inner.read())
        if frame is None and self.end_on_none:
            self.ended = True
        return frame

    def close(self) -> None:
        self.ended = True
        close = getattr(self.inner, "close", None)
        if close:
            close()


def as_frame(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    for attr in ("image", "frame", "bgr"):
        found = getattr(value, attr, None)
        if isinstance(found, np.ndarray):
            return found
    raise TypeError("FrameSource.read() must return a BGR array or None")


def open_file(path: str | Path):
    adopted = _from_kite("file", path)
    if adopted is not None:
        return adopted
    return VideoFileSource(path)


def open_webcam(index: int = 0):
    adopted = _from_kite("webcam", index)
    if adopted is not None:
        return adopted
    return WebcamSource(index)


def open_c270(index: int = 0):
    """Logitech C270 at 1280×720. This reader always requests that size."""
    return WebcamSource(index, width=C270_WIDTH, height=C270_HEIGHT, fps=C270_FPS)


def open_tello():
    adopted = _from_kite("tello", None)
    if adopted is not None:
        return adopted
    return TelloSource()


def _from_kite(kind: str, target):
    try:
        import kite.sources as sources
    except ImportError:
        return None
    for name in _KITE[kind]:
        factory = getattr(sources, name, None)
        if factory is None:
            continue
        try:
            inner = factory() if target is None else factory(target)
        except TypeError:
            continue
        return AdoptedSource(inner, end_on_none=(kind == "file"))
    return None


def decode_jpeg_message(message: bytes | str) -> tuple[np.ndarray, dict]:
    import cv2

    meta: dict = {}
    if isinstance(message, str):
        meta = json.loads(message)
        if "jpeg" not in meta:
            raise ValueError("sim text frames need a jpeg field")
        data = base64.b64decode(meta["jpeg"])
    else:
        data = bytes(message)
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("sim frame is not a JPEG")
    return frame, meta


class SimSocketSource:
    """JPEG frames from a local WebSocket, kept to the latest one."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        url: str | None = None,
        timeout: float = 0.5,
    ):
        self.host = host
        self.port = port
        self.url = url
        self.timeout = timeout
        self.ended = False
        self.error: BaseException | None = None
        self.fx: float | None = None
        self.fy: float | None = None
        self.height_m: float | None = None
        self.pitch: float | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sim-frames", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            self.close()
            raise RuntimeError("sim websocket did not start")
        if self.error is not None:
            err = self.error
            self.close()
            raise RuntimeError(f"sim websocket failed: {err}") from err

    def read(self) -> np.ndarray | None:
        if self.ended and self._queue.empty():
            return None
        try:
            return self._queue.get(timeout=self.timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        self.ended = True
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:
            self.error = exc
            self.ended = True
            self._ready.set()

    async def _main(self) -> None:
        if self.url:
            await self._client()
        else:
            await self._serve()

    async def _serve(self) -> None:
        import websockets

        async def handler(ws) -> None:
            try:
                async for message in ws:
                    if self._stop.is_set():
                        break
                    self._take(message)
            except Exception:
                return

        server = await websockets.serve(
            handler,
            self.host,
            self.port,
            max_size=12 * 1024 * 1024,
            ping_interval=None,
        )
        sock = next(iter(server.sockets))
        self.port = sock.getsockname()[1]
        self._ready.set()
        while not self._stop.is_set():
            await asyncio.sleep(0.05)
        server.close()
        await server.wait_closed()

    async def _client(self) -> None:
        import websockets

        last: BaseException | None = None
        for _ in range(50):
            if self._stop.is_set():
                return
            try:
                async with websockets.connect(self.url, max_size=12 * 1024 * 1024) as ws:
                    self._ready.set()
                    async for message in ws:
                        if self._stop.is_set():
                            break
                        self._take(message)
                    self.ended = True
                    return
            except Exception as exc:
                last = exc
                await asyncio.sleep(0.2)
        self.error = last or RuntimeError("could not connect")
        self.ended = True
        self._ready.set()

    def _take(self, message) -> None:
        try:
            frame, meta = decode_jpeg_message(message)
        except ValueError:
            return
        if "fx" in meta and "fy" in meta:
            self.fx = float(meta["fx"])
            self.fy = float(meta["fy"])
        if "height" in meta and "pitch" in meta:
            self.height_m = float(meta["height"])
            self.pitch = float(meta["pitch"])
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass


def send_jpegs(path: str | Path, url: str, fps: float = 8.0) -> None:
    """Push a video file to a perception sim socket as JPEG frames."""
    import cv2
    import websockets

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    delay = 1.0 / max(1.0, fps)

    async def run() -> None:
        async with websockets.connect(url, max_size=12 * 1024 * 1024) as ws:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if not ok:
                    continue
                await ws.send(buf.tobytes())
                await asyncio.sleep(delay)

    try:
        asyncio.run(run())
    finally:
        cap.release()
