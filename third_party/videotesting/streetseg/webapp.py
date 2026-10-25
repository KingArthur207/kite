"""Browser view of the live street cameras.

The phone (DroidCam), the Logitech C270 (720p), and the DJI Tello each show
the raw input beside the segmented output. The Tello path pulls the drone
camera with streamon and does not take off. The phone and the Tello are
different Wi-Fi networks. A cloud machine cannot see either radio.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .camera import Camera, Pose
from .events import Event

_BOARDS = ("c270", "tello", "phone")

_IDLE = {
    "c270": "Plug in the Logitech C270 and press Connect. It opens at 1280×720.",
    "tello": "Power on the Tello, join its Wi-Fi from this computer, and press Connect. This pulls the drone camera. The drone stays on the ground.",
    "phone": "Open DroidCam and enter the IP it shows.",
}

_OPEN_ERRORS = {
    "Logitech C270": (
        "Could not open the Logitech C270. Plug it in, and change the camera number if it is not camera 0."
    ),
    "Tello": "Could not open the Tello. Join its Wi-Fi from this computer. The drone stays on the ground.",
    "Phone": (
        "Could not open the phone. Use the IP from the DroidCam app, and keep this computer on the same Wi-Fi."
    ),
}

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kite street</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 16px/1.45 ui-sans-serif, system-ui, sans-serif; background: #101316; color: #f2f4f6; }
  main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 22px; font-weight: 650; margin: 0 0 6px; }
  h2 { font-size: 18px; font-weight: 650; margin: 0; }
  p { margin: 0; }
  .lead { margin: 0 0 18px; color: #b7c0c8; max-width: 46rem; }
  .board { margin: 0 0 28px; }
  .board-head { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; margin: 0 0 10px; }
  .spec { color: #8e99a3; font-size: 14px; margin-top: 2px; }
  form { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  label { color: #d5dde3; font-size: 14px; }
  input { background: #1c2126; color: inherit; border: 1px solid #3a4450; border-radius: 8px; padding: 8px 10px; font: inherit; }
  input[name="index"] { width: 4.5rem; }
  input[name="ip"] { min-width: 12rem; }
  input[name="port"] { width: 5.5rem; }
  button { background: #e8eef3; color: #121416; border: 0; border-radius: 8px; padding: 8px 14px; font: inherit; font-weight: 650; cursor: pointer; }
  button.drone { background: #9fd0ff; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  figure { margin: 0; background: #000; border-radius: 12px; overflow: hidden; min-height: 220px; }
  figcaption { padding: 8px 12px 0; color: #d5dde3; font-size: 13px; }
  img { display: block; width: 100%; max-height: 42vh; object-fit: contain; background: #000; }
  .status { margin-top: 10px; }
  .meta { color: #8e99a3; font-size: 14px; min-height: 1.2em; }
  @media (max-width: 720px) { .pair { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <h1>Kite street</h1>
  <p class="lead">Each camera shows the input feed beside the segmented output. The phone row reads DroidCam. The DJI Tello row pulls the drone camera and leaves the drone on the ground. The picture updates as fast as the model finishes a frame.</p>

  <section class="board" id="board-c270">
    <div class="board-head">
      <div>
        <h2>Logitech C270</h2>
        <p class="spec">720p webcam · 1280×720 · 60° field of view</p>
      </div>
      <form id="c270-form" data-board="c270">
        <label>Camera <input name="index" value="0" inputmode="numeric" aria-label="C270 camera number"></label>
        <button type="submit">Connect</button>
      </form>
    </div>
    <div class="pair">
      <figure>
        <figcaption>Input feed</figcaption>
        <img id="c270-input" alt="Logitech C270 input feed" hidden>
      </figure>
      <figure>
        <figcaption>Output feed</figcaption>
        <img id="c270-output" alt="Logitech C270 segmentation" hidden>
      </figure>
    </div>
    <p class="status" id="c270-text">Plug in the Logitech C270 and press Connect. It opens at 1280×720.</p>
    <p class="meta" id="c270-meta"></p>
  </section>

  <section class="board" id="board-phone">
    <div class="board-head">
      <div>
        <h2>Phone</h2>
        <p class="spec">DroidCam on the same Wi-Fi · http://phone:4747/video</p>
      </div>
      <form id="phone-form" data-board="phone">
        <label for="ip">Phone</label>
        <input id="ip" name="ip" placeholder="192.168.0.20" autocomplete="off" inputmode="decimal">
        <input name="port" value="4747" inputmode="numeric" aria-label="port">
        <button type="submit">Connect phone</button>
      </form>
    </div>
    <div class="pair">
      <figure>
        <figcaption>Input feed</figcaption>
        <img id="phone-input" alt="Phone input feed" hidden>
      </figure>
      <figure>
        <figcaption>Output feed</figcaption>
        <img id="phone-output" alt="Phone segmentation" hidden>
      </figure>
    </div>
    <p class="status" id="phone-text">Open DroidCam and enter the IP it shows.</p>
    <p class="meta" id="phone-meta"></p>
  </section>

  <section class="board" id="board-tello">
    <div class="board-head">
      <div>
        <h2>DJI Tello</h2>
        <p class="spec">Pulls the drone camera · stream on · stays on the ground</p>
      </div>
      <div>
        <button class="drone" id="tello" type="button">Connect</button>
      </div>
    </div>
    <div class="pair">
      <figure>
        <figcaption>Input feed</figcaption>
        <img id="tello-input" alt="Tello input feed" hidden>
      </figure>
      <figure>
        <figcaption>Output feed</figcaption>
        <img id="tello-output" alt="Tello segmentation" hidden>
      </figure>
    </div>
    <p class="status" id="tello-text">Power on the Tello, join its Wi-Fi from this computer, and press Connect. This pulls the drone camera. The drone stays on the ground.</p>
    <p class="meta" id="tello-meta"></p>
  </section>
</main>
<script>
const boards = ["c270", "tello", "phone"];
function setText(board, message) {
  const node = document.querySelector("#" + board + "-text");
  if (node) node.textContent = message;
}
async function post(url, body) {
  const response = await fetch(url, { method: "POST", body });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Could not connect");
  return payload;
}
function bind(formId, url, pending) {
  const form = document.querySelector(formId);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setText(form.dataset.board, pending);
    try {
      await post(url, new FormData(form));
    } catch (error) {
      setText(form.dataset.board, error.message);
    }
  });
}
bind("#c270-form", "/c270", "Connecting to the Logitech C270…");
bind("#phone-form", "/connect", "Connecting to the phone…");
document.querySelector("#tello").addEventListener("click", async () => {
  setText("tello", "Connecting to the Tello…");
  try {
    await post("/tello");
  } catch (error) {
    setText("tello", error.message);
  }
});
async function tick() {
  try {
    const status = await fetch("/status").then((response) => response.json());
    for (const name of boards) {
      const row = status[name] || {};
      if (row.text) setText(name, row.text);
      const bits = [];
      if (row.source) bits.push(row.source);
      if (row.width && row.height) bits.push(row.width + "×" + row.height);
      if (row.fps) bits.push(row.fps.toFixed(1) + " fps");
      if (row.frames) bits.push(row.frames + " frames");
      document.querySelector("#" + name + "-meta").textContent = bits.join(" · ");
      if (!row.frames) continue;
      for (const kind of ["input", "output"]) {
        const image = document.querySelector("#" + name + "-" + kind);
        image.hidden = false;
        image.src = "/boards/" + name + "/" + kind + "?n=" + row.frames;
      }
    }
  } catch (error) {
    setText("c270", "Page lost the server.");
  }
}
setInterval(tick, 400);
tick();
</script>
</body>
</html>
"""


def droidcam_url(text: str, port: int = 4747) -> str:
    """Turn the IP shown in the DroidCam app into an MJPEG URL."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Enter the IP shown in the DroidCam app.")
    if "://" not in raw:
        if "/" in raw:
            raw = "http://" + raw
        elif raw.count(":") == 1 and raw.rsplit(":", 1)[1].isdigit():
            raw = f"http://{raw}/video"
        else:
            raw = f"http://{raw}:{int(port)}/video"
    parts = urllib.parse.urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise ValueError("The phone address has to be an http link.")
    if not parts.hostname:
        raise ValueError("Enter the IP shown in the DroidCam app.")
    if parts.path in ("", "/"):
        parts = parts._replace(path="/video")
        raw = urllib.parse.urlunsplit(parts)
    return raw


def camera_index(text: str) -> int:
    raw = (text or "0").strip() or "0"
    if not raw.isdigit():
        raise ValueError("The camera number starts at 0.")
    return int(raw)


class UrlSource:
    """OpenCV reader for a DroidCam MJPEG URL, or any other http video URL."""

    def __init__(self, url: str):
        import cv2

        self.ended = False
        self.cap = cv2.VideoCapture(url)
        for prop, value in (
            (getattr(cv2, "CAP_PROP_BUFFERSIZE", None), 1),
            (getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None), 8000),
            (getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None), 8000),
        ):
            if prop is None:
                continue
            try:
                self.cap.set(prop, value)
            except cv2.error:
                pass
        self.failed_open = False
        if not self.cap.isOpened():
            self.ended = True
            self.failed_open = True

    def read(self) -> np.ndarray | None:
        if self.ended:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.ended = True
            return None
        return frame

    def close(self) -> None:
        self.ended = True
        self.cap.release()


class SegmentFeed:
    """Run perception on one camera and keep the raw frame beside the segmentation."""

    def __init__(self, perception, camera: Camera, pose: Pose, camera_fn=None, tello_factory=None):
        self.perception = perception
        self.camera = camera
        self.pose = pose
        self.camera_fn = camera_fn
        self.tello_factory = tello_factory
        self._lock = threading.Lock()
        self._input_jpeg: bytes | None = None
        self._output_jpeg: bytes | None = None
        self._text = ""
        self._error = ""
        self._source_name = ""
        self._fps = 0.0
        self._frames = 0
        self._width = 0
        self._height = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect_url(self, url: str) -> None:
        self._swap("Phone", lambda: UrlSource(url))

    def connect_tello(self) -> None:
        factory = self.tello_factory
        if factory is None:
            from .sources import open_tello

            factory = open_tello
        self._swap("Tello", factory)

    def connect_source(self, factory, label: str) -> None:
        self._swap(label, factory)

    def status(self) -> dict:
        with self._lock:
            text = self._error or self._text
            return {
                "source": self._source_name,
                "text": text,
                "error": self._error,
                "fps": round(self._fps, 2),
                "frames": self._frames,
                "width": self._width,
                "height": self._height,
            }

    def frame(self, kind: str = "output") -> bytes | None:
        with self._lock:
            if kind == "input":
                return self._input_jpeg
            if kind == "output":
                return self._output_jpeg
        raise KeyError(kind)

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._thread = None
        close = getattr(self.perception, "close", None)
        if close:
            close()

    def _swap(self, label: str, factory) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._stop = threading.Event()
        with self._lock:
            self._source_name = label
            self._error = ""
            self._text = f"Connecting to the {label}…"
            self._fps = 0.0
        self._thread = threading.Thread(
            target=self._loop, args=(factory, label), name=f"feed-{label}", daemon=True
        )
        self._thread.start()

    def _loop(self, factory, label: str) -> None:
        try:
            source = factory()
        except Exception as exc:
            self._fail(f"Could not open the {label}. {exc}")
            return
        try:
            while not self._stop.is_set():
                frame = source.read()
                if frame is None:
                    if getattr(source, "failed_open", False):
                        self._fail(_OPEN_ERRORS.get(label, f"Could not open the {label}."))
                        break
                    if getattr(source, "ended", False):
                        self._fail(f"The {label} feed stopped.")
                        break
                    if label == "Tello" and self._frames == 0:
                        with self._lock:
                            if not self._error:
                                self._text = "The Tello camera is on. Waiting for the first picture."
                    time.sleep(0.02)
                    continue
                camera = self.camera
                if self.camera_fn is not None:
                    camera = self.camera_fn(frame, camera)
                    self.camera = camera
                started = time.perf_counter()
                try:
                    events = self.perception.process(frame, camera, self.pose)
                except Exception as exc:
                    self._fail(str(exc))
                    break
                elapsed = time.perf_counter() - started
                image = getattr(self.perception, "image", None)
                if image is None:
                    image = frame
                raw = np.ascontiguousarray(frame)
                try:
                    input_jpeg = _jpeg(raw)
                    output_jpeg = _jpeg(image)
                except Exception as exc:
                    self._fail(str(exc))
                    break
                text = _caption(events)
                height, width = int(raw.shape[0]), int(raw.shape[1])
                with self._lock:
                    self._input_jpeg = input_jpeg
                    self._output_jpeg = output_jpeg
                    self._frames += 1
                    self._width = width
                    self._height = height
                    self._fps = (1.0 / elapsed) if elapsed > 0 else 0.0
                    self._text = text
                    self._error = ""
        finally:
            close = getattr(source, "close", None)
            if close:
                close()

    def _fail(self, message: str) -> None:
        with self._lock:
            self._error = message
            self._text = message


class LiveBoards:
    """C270 and Tello boards. Each board is created the first time it connects."""

    def __init__(self, perception_factory, camera: Camera, pose: Pose, camera_fns=None, tello_factory=None, c270_factory=None):
        self._perception_factory = perception_factory
        self._camera = camera
        self._pose = pose
        self._camera_fns = dict(camera_fns or {})
        self._tello_factory = tello_factory
        self._c270_factory = c270_factory
        self._boards: dict[str, SegmentFeed] = {}
        self._notes: dict[str, str] = {}
        self._create_lock = threading.Lock()

    def connect_c270(self, index: int = 0) -> None:
        if index < 0:
            raise ValueError("The camera number starts at 0.")
        self._notes["c270"] = "Connecting to the Logitech C270…"
        factory = self._c270_factory
        if factory is None:
            from .sources import open_c270

            factory = open_c270
        self._ensure("c270").connect_source(lambda idx=index: factory(idx), "Logitech C270")

    def connect_tello(self) -> None:
        self._notes["tello"] = "Connecting to the Tello…"
        self._ensure("tello").connect_tello()

    def connect_url(self, url: str) -> None:
        self._notes["phone"] = "Connecting to the phone…"
        self._ensure("phone").connect_url(url)

    def connect_source(self, factory, label: str, board: str = "c270") -> None:
        self._notes[board] = f"Connecting to the {label}…"
        self._ensure(board).connect_source(factory, label)

    def status(self) -> dict:
        out = {}
        for name in _BOARDS:
            board = self._boards.get(name)
            if board is None:
                out[name] = {
                    "source": "",
                    "text": self._notes.get(name) or _IDLE[name],
                    "error": "",
                    "fps": 0,
                    "frames": 0,
                    "width": 0,
                    "height": 0,
                }
            else:
                out[name] = board.status()
        return out

    def frame(self, name: str, kind: str) -> bytes | None:
        board = self._boards.get(name)
        if board is None or name not in _BOARDS:
            return None
        return board.frame(kind)

    def close(self) -> None:
        for board in list(self._boards.values()):
            board.close()

    def _ensure(self, name: str) -> SegmentFeed:
        with self._create_lock:
            board = self._boards.get(name)
            if board is not None:
                return board
            perception = self._perception_factory()
            board = SegmentFeed(
                perception,
                self._camera,
                self._pose,
                camera_fn=self._camera_fns.get(name),
                tello_factory=self._tello_factory,
            )
            self._boards[name] = board
            return board


def _jpeg(image: np.ndarray) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("could not encode the frame")
    return encoded.tobytes()


def _caption(events) -> str:
    say = ""
    scene = ""
    for event in events or ():
        if not isinstance(event, Event):
            continue
        if event.kind == "say":
            say = event.text
        elif event.kind == "scene":
            scene = event.text
    return say or scene or "Segmenting."


def make_server(boards: LiveBoards, host: str, port: int) -> ThreadingHTTPServer:
    feed_box = boards

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path in ("/", "/index.html"):
                body = _PAGE.encode("utf-8")
                self._send(200, "text/html; charset=utf-8", body)
                return
            if path == "/status":
                body = json.dumps(feed_box.status()).encode("utf-8")
                self._send(200, "application/json", body)
                return
            parts = [part for part in path.split("/") if part]
            if len(parts) == 3 and parts[0] == "boards" and parts[1] in _BOARDS and parts[2] in ("input", "output"):
                jpeg = feed_box.frame(parts[1], parts[2])
                if jpeg is None:
                    self._send(204, "text/plain", b"")
                    return
                self._send(200, "image/jpeg", jpeg)
                return
            self._send(404, "text/plain", b"not found")

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            try:
                if path == "/c270":
                    fields = self._form()
                    index = camera_index(fields.get("index", "0"))
                    feed_box.connect_c270(index)
                    self._json(200, {"ok": True, "source": "Logitech C270", "index": index})
                    return
                if path == "/tello":
                    feed_box.connect_tello()
                    self._json(200, {"ok": True, "source": "Tello"})
                    return
                if path == "/connect":
                    fields = self._form()
                    port_text = (fields.get("port") or "4747").strip() or "4747"
                    url = droidcam_url(fields.get("ip") or fields.get("url") or "", int(port_text))
                    feed_box.connect_url(url)
                    self._json(200, {"ok": True, "url": url})
                    return
            except (ValueError, RuntimeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(400, {"error": str(exc)})
                return
            self._send(404, "text/plain", b"not found")

        def _form(self) -> dict[str, str]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8", "replace")
            content = self.headers.get("Content-Type", "")
            if "application/json" in content:
                payload = json.loads(raw or "{}")
                return {str(key): "" if value is None else str(value) for key, value in payload.items()}
            parsed = urllib.parse.parse_qs(raw, keep_blank_values=True)
            return {key: values[-1] for key, values in parsed.items()}

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, "application/json", json.dumps(payload).encode("utf-8"))

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            return

    class Server(ThreadingHTTPServer):
        allow_reuse_address = True

    return Server((host, port), Handler)


def serve(boards: LiveBoards, host: str, port: int) -> None:
    server = make_server(boards, host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        boards.close()
