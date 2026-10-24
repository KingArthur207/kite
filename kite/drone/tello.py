"""DJI Tello adapter (djitellopy for commands + state, kite's TelloVideoSource for video: video_backend "auto" picks
kite's own UDP reader on macOS and PyAV elsewhere, see kite.config.resolve_video_backend).

Discrete commands are NON-BLOCKING: the command is sent, and busy() polls djitellopy's reply list
(filled by its UDP receiver thread) with a per-command timeout. After stop() the late reply of the
cancelled move is discarded so it cannot be credited to the next command.

dry_run=True: connects, streams video and reads telemetry, but NEVER sends motion commands (takeoff,
rc, moves, land are logged and "complete" after their expected duration). Use it to run the whole
mission with the drone sitting on a table / held in hand.

A Wi-Fi drop never raises out of a flight call: sendto's OSError (ENETUNREACH, ENOBUFS...) is logged once, kept in
link_error, and the call is retried by the next one (rc every tick; a discrete command fails at once with
'error: not sent' so it can be re-sent). Intervals use time.monotonic(), so a wall-clock step cannot stall the loop.
"""

from __future__ import annotations

import logging
import math
import socket
import time
from dataclasses import dataclass

from ..config import resolve_video_backend
from ..sources.opencv_sources import TelloVideoSource
from ..types import Telemetry
from .base import Drone, check_move, check_rotate, clamp_rc

log = logging.getLogger("kite.tello")

MIN_GAP_S = 0.1  # the Tello drops commands sent closer together than this (djitellopy's TIME_BTW_COMMANDS)
STOP_SETTLE_S = 1.0  # after 'stop', replies within this window belong to the stop / the cancelled command
STREAMON_TRIES, VIDEO_WAIT_S = 4, 3.0  # streamon is re-sent until video packets arrive
NO_VIDEO = ("the Tello answers commands but sends no video to UDP {port}. Close the Tello phone app and take the "
            "phone off the drone's Wi-Fi (while the app is connected the drone streams to the phone), then restart "
            "the drone. On Windows also allow Python through the firewall (inbound UDP 8890 and {port}); on macOS "
            "allow incoming connections for Python and give your terminal app Local Network access "
            "(System Settings > Privacy & Security).")


def video_packets_arrive(port: int, timeout_s: float) -> bool:
    """True once a UDP datagram reaches `port`. OSError if another program already holds the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", port))
        s.settimeout(timeout_s)
        try:
            s.recv(2048)
            return True
        except socket.timeout:
            return False


@dataclass
class _Pending:
    cmd: str
    t_sent: float
    timeout_s: float
    expected_s: float  # for dry-run completion


class TelloDrone(Drone):
    name = "tello"

    def __init__(self, tello=None, video: bool = True, dry_run: bool = False, move_speed_cm_s: int = 50,
                 yaw_sign: int = 1, velocity_scale: float = 0.1, video_fps: int = 60, video_backend: str = "auto", host: str = "192.168.10.1"):
        """tello: an existing djitellopy.Tello (or a test double); created on connect() if None.
        velocity_scale: state vgx/vgy/vgz -> m/s (the Tello reports dm/s; check with the latency test)."""
        self._tello = tello
        self._host = host
        self.video_enabled = video
        self.dry_run = dry_run
        self.move_speed_cm_s = move_speed_cm_s
        self.yaw_sign = yaw_sign
        self.velocity_scale = velocity_scale
        self.video_fps = video_fps
        self.video_backend = resolve_video_backend(video_backend)  # "auto" -> this platform's reader
        self._pending: _Pending | None = None
        self._result: str | None = "ok"
        self._flying = False
        self._last_send = 0.0
        self._ignore_until = 0.0
        self._source: TelloVideoSource | None = None
        self.link_error: str | None = None  # last send failure (Wi-Fi down); None once a send works again
        self.log: list[tuple[float, str]] = []
        self._last_state = None
        self._state_t = 0.0

    # ------------------------------------------------------------------ setup
    @property
    def tello(self):
        return self._tello

    def connect(self) -> None:
        if self._tello is None:
            from djitellopy import Tello

            Tello.LOGGER.setLevel(logging.WARNING)  # otherwise every rc packet is printed
            self._tello = Tello(self._host)
        self._tello.connect()
        log.info("connected, battery %s%%", self._state().get("bat"))
        if not self.dry_run:
            self._tello.send_control_command(f"speed {int(self.move_speed_cm_s)}")
        if self.video_enabled:
            self._start_video()

    def _start_video(self) -> None:
        url = self._tello.get_udp_video_address()
        port = int(url.rsplit(":", 1)[1])
        for i in range(STREAMON_TRIES):
            self._tello.streamon()  # re-sent: the drone may not stream after the first one
            try:
                if video_packets_arrive(port, VIDEO_WAIT_S):
                    break
            except OSError as e:
                raise RuntimeError(f"UDP port {port} is held by another program ({e}): close other Tello scripts "
                                   f"and tools (find it with: lsof -nP -iUDP:{port})") from None
            log.warning("no video packets yet (%d/%d), re-sending streamon", i + 1, STREAMON_TRIES)
        else:
            raise RuntimeError(NO_VIDEO.format(port=port))
        log.info("video packets arrive on UDP %d: reading them with the %s backend", port, self.video_backend)
        self._source = TelloVideoSource(url, fps=self.video_fps, backend=self.video_backend).start()

    def close(self) -> None:
        try:
            if self._flying and not self.dry_run:
                self._send("land")
                self._tello.is_flying = True  # djitellopy's end() lands again as a safety net
        finally:
            if self._source is not None:
                self._source.stop()
                self._source = None
            if self._tello is not None:
                try:
                    self._tello.end()
                except Exception as e:  # never let cleanup raise over the real error
                    log.warning("tello.end(): %s", e)

    # ------------------------------------------------------------------ low level
    def _state(self) -> dict:
        try:
            return self._tello.get_current_state() or {}
        except Exception:
            return {}

    def _replies(self) -> list:
        return self._tello.get_own_udp_object()["responses"]

    def _udp(self, fn, *args) -> bool:
        """One djitellopy send; False (logged once per outage) if the network refused it."""
        try:
            fn(*args)
        except OSError as e:
            if self.link_error is None:
                log.warning("Wi-Fi send failed (%s): retrying on every command", e)
            self.link_error = str(e)
            return False
        if self.link_error is not None:
            log.warning("Wi-Fi sends work again")
            self.link_error = None
        return True

    def _send(self, cmd: str) -> bool:
        wait = MIN_GAP_S - (time.monotonic() - self._last_send)
        if wait > 0:
            time.sleep(min(wait, MIN_GAP_S))
        self.log.append((time.time(), cmd))
        ok = True
        if self.dry_run and not cmd.endswith("?") and cmd not in ("command", "streamon", "streamoff"):
            log.info("[dry-run] %s", cmd)
        else:
            ok = self._udp(self._tello.send_command_without_return, cmd)
        self._last_send = time.monotonic()
        return ok

    def _start(self, cmd: str, timeout_s: float, expected_s: float) -> None:
        if self._pending is not None:
            raise RuntimeError(f"'{cmd}' while '{self._pending.cmd}' is still running")
        # Tello replies do not say which command they answer. Right after a stop(), the cancelled command
        # and the stop itself may still answer: wait that window out (hovering) so their late replies
        # cannot be taken as the answer to THIS command.
        wait = self._ignore_until - time.monotonic()
        if wait > 0 and not self.dry_run:
            time.sleep(min(wait, STOP_SETTLE_S))
        self._ignore_until = 0.0
        self._replies().clear()  # anything queued now is stale
        self._result = None
        self._pending = _Pending(cmd, time.monotonic(), timeout_s, expected_s)
        if not self._send(cmd):
            self._complete(f"error: not sent ({self.link_error})")  # never reached the drone: free to re-send

    # ------------------------------------------------------------------ Drone interface
    @property
    def flying(self) -> bool:
        return self._flying

    def takeoff(self) -> None:
        self._start("takeoff", 20.0, 5.0)

    def land(self) -> None:
        if self._pending is not None and self._pending.cmd == "land":
            return  # already landing: a 'stop' now would cancel the landing and leave it hovering
        if self._pending is not None and self._pending.cmd == "takeoff":
            self._pending = None  # land is accepted right after/during takeoff; drop the takeoff wait
            self._flying = True
        if self._pending is not None:
            self.stop()
        self._start("land", 15.0, 4.0)

    def emergency(self) -> bool:
        """False if the packet could not be sent (Wi-Fi down): nothing changed, the caller must say so."""
        self.log.append((time.time(), "emergency"))
        if not self.dry_run and not self._udp(self._tello.send_command_without_return, "emergency"):
            return False
        self._pending = None
        self._result = "ok"
        self._flying = False
        return True

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        if self._pending is not None:
            return  # an rc packet would abort the running discrete command
        vals = (clamp_rc(lr), clamp_rc(fb), clamp_rc(ud), clamp_rc(yaw))
        if self.dry_run:
            if any(vals):
                self.log.append((time.time(), "rc {} {} {} {}".format(*vals)))
            return
        if getattr(self._tello, "last_rc_control_timestamp", 0.0) > time.time():
            self._tello.last_rc_control_timestamp = 0.0  # wall clock stepped back: djitellopy would drop every rc
        self._udp(self._tello.send_rc_control, *vals)

    def move(self, direction: str, cm: int) -> None:
        cm = check_move(direction, cm)
        expected = cm / max(self.move_speed_cm_s, 10) + 1.5
        self._start(f"{direction} {cm}", expected + 6.0, expected)

    def rotate(self, deg: int) -> None:
        deg = check_rotate(deg)
        expected = abs(deg) / 60.0 + 1.0
        self._start(f"{'cw' if deg > 0 else 'ccw'} {abs(deg)}", expected + 5.0, expected)

    def stop(self) -> None:
        if self._pending is not None and self._pending.cmd in ("takeoff", "land"):
            return  # never interrupt a takeoff or a landing (the drone would hover, we would think it landed)
        self._pending = None
        self._result = "ok"
        self._send("stop")
        self._ignore_until = time.monotonic() + STOP_SETTLE_S  # 'stop' (and a cancelled command) will still answer

    def busy(self) -> bool:
        p = self._pending
        if p is None:
            return False
        now = time.monotonic()
        if self.dry_run:
            if now - p.t_sent >= p.expected_s:
                self._complete("ok")
            return self._pending is not None
        replies = self._replies()
        while replies:
            raw = replies.pop(0)
            text = raw.decode("utf-8", errors="ignore").strip() if isinstance(raw, (bytes, bytearray)) else str(raw).strip()
            if text.lower().startswith("ok"):
                self._complete("ok")
                return False
            if text.lower().startswith("error") or "out of range" in text.lower() or "motor stop" in text.lower():
                self._complete(f"error: {text}")
                return False
            # anything else (e.g. a number from a query) is not the answer to this command
        if now - p.t_sent > p.timeout_s:
            self._complete("error: timeout")
            return False
        return True

    def _complete(self, result: str) -> None:
        p = self._pending
        self._pending = None
        if p is not None and p.cmd == "land" and result != "ok" and not self.dry_run:
            h = self.telemetry().height_m
            if h is not None and h <= 0.2:
                result = "ok"  # the reply was lost / 'error' because it is already down: it has landed
        self._result = result
        if p is None:
            return
        if p.cmd == "takeoff" and result == "ok":
            self._flying = True
            if self._tello is not None:
                self._tello.is_flying = not self.dry_run
        elif p.cmd == "land" and result == "ok":
            self._flying = False
            if self._tello is not None:
                self._tello.is_flying = False

    def last_result(self) -> str | None:
        return self._result

    def set_speed(self, cm_s: int) -> None:
        self.move_speed_cm_s = int(cm_s)
        self._start(f"speed {int(cm_s)}", 5.0, 0.2)

    def telemetry(self) -> Telemetry:
        st = self._state()
        # djitellopy (2.5) does not timestamp state packets, but it stores a NEW dict per packet: stamp it on
        # first sight, so a frozen state stream (its receiver thread dies on one bad packet) shows up as stale
        if st is not self._last_state:
            self._last_state = st
            self._state_t = time.time() if st else 0.0
        t = self._state_t

        def num(k, scale=1.0):
            v = st.get(k)
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) * scale

        tof = num("tof", 0.01)
        if tof is not None and not 0.1 <= tof <= 8.0:  # 6553 (out of range) / 10 cm on the ground
            tof = None
        yaw = num("yaw")
        return Telemetry(
            t=t,
            yaw_deg=None if yaw is None else self.yaw_sign * yaw,
            pitch_deg=num("pitch"), roll_deg=num("roll"),
            height_m=num("h", 0.01), tof_m=tof,
            vx=num("vgx", self.velocity_scale), vy=num("vgy", self.velocity_scale), vz=num("vgz", self.velocity_scale),
            battery_pct=num("bat"), flight_time_s=num("time"),
        )

    def frame_source(self):
        if self._source is None:
            raise RuntimeError("video not started (connect() with video=True)")
        return self._source
