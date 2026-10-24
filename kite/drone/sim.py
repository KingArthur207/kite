"""The browser sim's drone (sim/src/simDrone.ts, `?drone=sim`) behind the Drone interface, over the frame relay.

    drone = SafetyGovernor(SimDrone("ws://127.0.0.1:5020/sub"), cfg.safety)
    drone.connect()                        # waits for the page's first drone_state
    frames = drone.frame_source()          # SimVideoSource: the drone camera, newest frame first

One relay connection (/sub) carries everything. In: binary drone frames (camera id 0, JPEG, already held back by the
page's video latency) and `drone_state` JSON at 20 Hz with Tello field names (h, tof, bat, yaw, pitch, roll, vgx..,
time) plus sim truth (x, y, alt, heading, state, walker_*). Out: `{"type": "drone_cmd", "cmd": "rc"|"takeoff"|"land"|
"stop"|"emergency", ...}`, which the relay passes to the page. The page models the drone; limits (max_rc, altitude
band, battery) are the SafetyGovernor's job here, as with the Tello.

Not modelled: discrete move/rotate (they fail with an error result; follow and the keys use rc only).
dry_run=True streams video and state but never sends a motion command (like TelloDrone).
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np

from ..camera import Camera
from ..sources.base import FrameSource
from ..types import Frame, Telemetry
from .base import Drone, clamp_rc

DEFAULT_URL = "ws://127.0.0.1:5020/sub"
# The sim drone camera: 960x720, FX in sim/src/config.ts (keep in step with it).
SIM_CAMERA = Camera(960, 720, 1000.0, 1000.0, 480.0, 360.0, "sim (sim/src/config.ts FX)")
TIMEOUTS_S = {"takeoff": 10.0, "land": 15.0}
DONE_STATE = {"takeoff": "flying", "land": "ground"}


class SimVideoSource(FrameSource):
    """Drone frames from the sim, fed by SimDrone's relay thread."""

    name = "sim"

    def __init__(self):
        self._frame: Frame | None = None
        self._seq = 0

    def push(self, jpeg: bytes) -> None:
        import cv2

        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            self._seq += 1
            self._frame = Frame(img, time.time(), self._seq, self.name)

    def read(self) -> Frame | None:
        return self._frame


class SimDrone(Drone):
    name = "sim"

    def __init__(self, url: str = DEFAULT_URL, dry_run: bool = False, connect_timeout_s: float = 15.0):
        self.url, self.dry_run, self.connect_timeout_s = url, dry_run, connect_timeout_s
        self.video = SimVideoSource()
        self.link_error: str | None = None
        self._ws = None
        self._send_lock = threading.Lock()
        self._state: dict = {}
        self._state_t = 0.0
        self._pending: tuple[str, float] | None = None
        self._result: str | None = "ok"
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ link
    def connect(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sim-drone")
        self._thread.start()
        t_end = time.time() + self.connect_timeout_s
        while not self._state and time.time() < t_end:
            time.sleep(0.05)
        if not self._state:
            raise RuntimeError(f"no drone_state from the sim on {self.url}: is the relay running and the sim open "
                               f"with ?drone=sim (and ?relay= pointing at the same relay)?")

    def _loop(self) -> None:
        from websockets.sync.client import connect

        while self._running:
            try:
                with connect(self.url, max_size=8 * 1024 * 1024) as ws:
                    self._ws, self.link_error = ws, None
                    for msg in ws:
                        if not self._running:
                            return
                        if isinstance(msg, bytes):
                            if msg[:1] == b"\x00":
                                self.video.push(msg[1:])
                            continue
                        d = json.loads(msg)
                        if d.get("type") == "drone_state":
                            self._state, self._state_t = d, time.time()  # a NEW dict per packet (record.py relies on it)
            except Exception as e:  # relay not up yet, or restarted
                self._ws = None
                self.link_error = f"relay: {e!r}"
                time.sleep(0.5)

    def publish(self, msg: dict) -> bool:
        """Send one JSON message toward the page (the relay passes drone_cmd and walker_cmd types)."""
        ws = self._ws
        if ws is None:
            return False
        try:
            with self._send_lock:
                ws.send(json.dumps(msg))
            return True
        except Exception as e:
            self.link_error = f"send: {e!r}"
            return False

    def _send(self, cmd: str, **kw) -> bool:
        return self.publish({"type": "drone_cmd", "cmd": cmd, **kw})

    def close(self) -> None:
        if self.flying and not self.dry_run:
            self._send("land")
        self._running = False
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ state
    def get_current_state(self) -> dict:
        """The latest raw state packet (kite.record logs these to telemetry.jsonl, like djitellopy's)."""
        return self._state

    @property
    def flying(self) -> bool:
        return self._state.get("state") in ("takeoff", "flying", "landing")

    def telemetry(self) -> Telemetry:
        s = self._state

        def f(k, scale=1.0):
            v = s.get(k)
            return None if v is None else float(v) * scale

        return Telemetry(t=self._state_t, yaw_deg=f("yaw"), pitch_deg=f("pitch"), roll_deg=f("roll"),
                         height_m=f("h", 0.01), tof_m=f("tof", 0.01), vx=f("vgx", 0.1), vy=f("vgy", 0.1),
                         vz=f("vgz", 0.1), battery_pct=f("bat"), flight_time_s=f("time"))

    def frame_source(self) -> FrameSource:
        return self.video

    # ------------------------------------------------------------------ commands
    def _start(self, cmd: str) -> None:
        if self.dry_run:
            self._result = "ok"
            return
        self._pending = (cmd, time.time())
        self._result = None
        if not self._send(cmd):
            self._pending, self._result = None, "error: not sent"

    def takeoff(self) -> None:
        self._start("takeoff")

    def land(self) -> None:
        if self._pending and self._pending[0] == "land":
            return
        self._start("land")

    def emergency(self) -> bool | None:
        if self.dry_run:
            return True
        self._pending, self._result = None, "ok"
        return self._send("emergency")

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        if self.dry_run or self.busy():
            return
        self._send("rc", lr=clamp_rc(lr), fb=clamp_rc(fb), ud=clamp_rc(ud), yaw=clamp_rc(yaw))

    def move(self, direction: str, cm: int) -> None:
        self._result = "error: sim drone has no discrete moves (use rc)"

    def rotate(self, deg: int) -> None:
        self._result = "error: sim drone has no discrete rotations (use rc)"

    def stop(self) -> None:
        if not self.dry_run:
            self._send("stop")

    def busy(self) -> bool:
        if self._pending is None:
            return False
        cmd, t_sent = self._pending
        if self._state_t > t_sent and self._state.get("state") == DONE_STATE[cmd]:
            self._pending, self._result = None, "ok"
        elif time.time() - t_sent > TIMEOUTS_S[cmd]:
            self._pending, self._result = None, f"error: {cmd} timed out"
        return self._pending is not None

    def last_result(self) -> str | None:
        return None if self.busy() else self._result
