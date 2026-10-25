"""DJI Tello adapter: djitellopy for commands + state, TelloVideoSource for video (DroneCfg.video_backend picks the
reader: pyav by default, opencv, or kite's raw UDP reader; "auto" = raw on macOS).

jerkgt13's reachglass/drone/tello.py (github.com/nathanwuzhao/jerkgt13). kite's changes, each from a failure found
in tests or on the drone: Wi-Fi drops never raise (link_error), emergency() reports a packet it could not send,
close() lands a takeoff still in progress, set_speed() changes nothing while busy, intervals use time.monotonic(),
and the drone's address is a setting (host).

Discrete commands are NON-BLOCKING: the command is sent, and busy() polls djitellopy's reply list
(filled by its UDP receiver thread) with a per-command timeout. After stop() the late reply of the
cancelled move is discarded so it cannot be credited to the next command.

dry_run=True: connects, streams video and reads telemetry, but NEVER sends motion commands (takeoff,
rc, moves, land are logged and "complete" after their expected duration). Use it to run the whole
mission with the drone sitting on a table / held in hand.

A Wi-Fi drop never raises out of a flight call: sendto's OSError (ENETUNREACH, ENOBUFS...) is logged once, kept in
link_error, and the call is retried by the next one (rc every tick; a discrete command fails at once with
'error: not sent' so it can be re-sent). Intervals use time.monotonic(), so a wall-clock step cannot stall the loop.

rc robustness:
  * djitellopy 2.5 silently drops an rc sent within 1 ms of the previous one (TIME_BTW_RC_CONTROL_COMMANDS): a
    hover right after a stick would be lost and the drone would keep the stick. rc() waits that 1 ms out instead.
  * dead-man: a non-zero stick that is not refreshed for rc_hold_s (the control loop stalled: a GC pause, a
    blocked print, a hung detector) is followed by ONE rc 0 0 0 0 from a watchdog thread. Nothing is sent after
    that, so a dead program still lets the Tello's own auto-land (15 s without a command) happen. The SDK says rc
    gets no reply, and reports differ on whether a stick persists without updates: assume it does.
"""

from __future__ import annotations

import logging
import math
import socket
import threading
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
RC_GATE_S = 0.0015  # djitellopy drops an rc closer than 1 ms to the previous one: wait this long instead
RC_HOLD_S = 0.6  # dead-man: a non-zero stick not refreshed for this long gets one rc 0 0 0 0 (the loop runs at 10 Hz)
NO_VIDEO = ("the Tello answers commands but sends no video to UDP {port}. Make sure no other device or app is "
            "connected to the drone's Wi-Fi, then restart the drone. Windows: allow Python through the firewall on "
            "Private AND Public networks (inbound UDP 8890 and {port}). macOS: allow incoming connections for "
            "Python and give the terminal app Local Network access (System Settings > Privacy & Security).")


def video_link_commands(bitrate: str | None = None, resolution: str | None = None, fps: str | None = None) -> list[str]:
    """The Tello SDK commands for the requested video link settings, in sending order (None: not sent)."""
    out = []
    if bitrate is not None:
        out.append(f"setbitrate {0 if str(bitrate) == 'auto' else int(bitrate)}")
    if resolution is not None:
        out.append(f"setresolution {resolution}")
    if fps is not None:
        out.append(f"setfps {fps}")
    return out


def apply_video_link(send, bitrate: str | None = None, resolution: str | None = None, fps: str | None = None,
                     say=log.info) -> dict[str, str]:
    """Send the video link settings (call after `command`, before `streamon`). send(cmd) returns the drone's reply
    text (None or a timeout message if none came). Never raises: a firmware without the command (SDK < 2.0 answers
    "error" or "unknown command") gets one line from say() and keeps its default. Returns {command: reply}."""
    replies = {}
    for cmd in video_link_commands(bitrate, resolution, fps):
        try:
            reply = send(cmd)
        except Exception as e:  # Wi-Fi send failure: the stream still works at the drone's defaults
            reply = f"not sent ({e})"
        text = (reply.decode("utf-8", "ignore") if isinstance(reply, (bytes, bytearray)) else str(reply or "")).strip()
        replies[cmd] = text
        word = cmd.split()[0]
        if text.lower().startswith("ok"):
            say(f"{cmd} -> ok")
        elif text.lower().startswith(("error", "unknown")):
            say(f"this Tello's firmware doesn't support {word}; keeping its default (it answered {text!r})")
        else:
            say(f"{cmd}: no usable reply ({text or 'none'}); keeping the drone's default")
    return replies


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


def body_sticks(lr: float, fb: float, right_sign: int = 1, forward_sign: int = 1) -> tuple[float, float]:
    """kite: rc lr / fb as the drone must receive them (DroneCfg.right_sign / forward_sign; +1 = unchanged)."""
    return lr * (1 if right_sign >= 0 else -1), fb * (1 if forward_sign >= 0 else -1)


_OPPOSITE = {"forward": "back", "back": "forward", "left": "right", "right": "left"}


def body_direction(direction: str, right_sign: int = 1, forward_sign: int = 1) -> str:
    """kite: a discrete move's direction as the drone must receive it, with the same signs as body_sticks()."""
    if (direction in ("forward", "back") and forward_sign < 0) or (direction in ("left", "right") and right_sign < 0):
        return _OPPOSITE[direction]
    return direction


@dataclass
class _Pending:
    cmd: str
    t_sent: float
    timeout_s: float
    expected_s: float  # for dry-run completion


class TelloDrone(Drone):
    name = "tello"

    def __init__(self, tello=None, video: bool = True, dry_run: bool = False, move_speed_cm_s: int = 50,
                 yaw_sign: int = 1, velocity_scale: float = 0.1, video_fps: int = 60, video_backend: str | None = None, host: str = "192.168.10.1",
                 video_bitrate: str | None = None, video_resolution: str | None = None,
                 video_stream_fps: str | None = None, rc_hold_s: float | None = RC_HOLD_S,
                 forward_sign: int = 1, right_sign: int = 1):
        """tello: an existing djitellopy.Tello (or a test double); created on connect() if None.
        forward_sign / right_sign (DroneCfg): +1 = the SDK as documented; -1 reverses that axis for rc AND moves.
        velocity_scale: state vgx/vgy/vgz -> m/s (the Tello reports dm/s; check with the latency test).
        video_bitrate / video_resolution / video_stream_fps: sent before streamon (None: not sent), see DroneCfg.
        rc_hold_s: the dead-man delay (None: no watchdog)."""
        self._tello = tello
        self._host = host
        self.video_enabled = video
        self.dry_run = dry_run
        self.move_speed_cm_s = move_speed_cm_s
        self.yaw_sign = yaw_sign
        self.forward_sign, self.right_sign = (1 if forward_sign >= 0 else -1), (1 if right_sign >= 0 else -1)
        self.velocity_scale = velocity_scale
        self.video_fps = video_fps
        self.video_backend = resolve_video_backend(video_backend)  # None -> the default, "auto" -> this platform's reader
        self.video_link = {"bitrate": video_bitrate, "resolution": video_resolution, "fps": video_stream_fps}
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
        self.rc_hold_s = rc_hold_s
        self._rc_lock = threading.Lock()
        self._rc_live = False  # the last stick sent was not all zeros
        self._rc_call_t = 0.0  # monotonic time of the last rc() call (sent or not)
        self._rc_sent_t = 0.0  # monotonic time of the last rc packet
        self._watchdog: threading.Thread | None = None
        self._closing = threading.Event()
        self.watchdog_trips = 0

    # ------------------------------------------------------------------ setup
    @property
    def tello(self):
        return self._tello

    def connect(self) -> None:
        if self._tello is None:
            from djitellopy import Tello

            if Tello.LOGGER.level < logging.WARNING:  # otherwise every rc packet is printed (a quieter level set by
                Tello.LOGGER.setLevel(logging.WARNING)  # the caller stays)
            self._tello = Tello(self._host)
        self._tello.connect()
        log.info("connected, battery %s%%", self._state().get("bat"))
        if not self.dry_run:
            self._tello.send_control_command(f"speed {int(self.move_speed_cm_s)}")
        if self.video_enabled:
            apply_video_link(lambda c: self._tello.send_command_with_return(c, timeout=3), **self.video_link)
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
                raise RuntimeError(f"UDP port {port} is held by another program ({e}): close other Tello scripts and "
                                   f"tools (find it with: lsof -nP -iUDP:{port} on macOS/Linux, "
                                   f"netstat -ano -p udp | findstr {port} on Windows)") from None
            log.warning("no video packets yet (%d/%d), re-sending streamon", i + 1, STREAMON_TRIES)
        else:
            raise RuntimeError(NO_VIDEO.format(port=port))
        log.info("video packets arrive on UDP %d: reading them with the %s backend", port, self.video_backend)
        self._source = TelloVideoSource(url, fps=self.video_fps, backend=self.video_backend).start()

    def close(self) -> None:
        self._closing.set()
        try:
            taking_off = self._pending is not None and self._pending.cmd == "takeoff"  # airborne before its 'ok'
            if (self._flying or taking_off) and not self.dry_run:
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
            time.sleep(wait)
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
            time.sleep(wait)
        self._ignore_until = 0.0
        self._replies().clear()  # anything queued now is stale
        self._rc_live = False  # a discrete command replaces the stick
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
        if self._tello is not None:
            self._tello.is_flying = False  # as djitellopy's own emergency(): its end() must not send 'land' afterwards
        return True

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self._rc_call_t = time.monotonic()
        if self._pending is not None:
            return  # an rc packet would abort the running discrete command
        lr, fb = body_sticks(lr, fb, self.right_sign, self.forward_sign)
        vals = (clamp_rc(lr), clamp_rc(fb), clamp_rc(ud), clamp_rc(yaw))
        if self.dry_run:
            if any(vals):
                self.log.append((time.time(), "rc {} {} {} {}".format(*vals)))
            return
        self._send_rc(vals)
        if any(vals) and self.rc_hold_s is not None:
            self._start_watchdog()

    def _send_rc(self, vals: tuple[int, int, int, int]) -> bool:
        with self._rc_lock:
            last = getattr(self._tello, "last_rc_control_timestamp", 0.0)
            if not isinstance(last, (int, float)):
                last = 0.0
            if last > time.time():
                self._tello.last_rc_control_timestamp = 0.0  # wall clock stepped back: djitellopy would drop every rc
            else:
                wait = RC_GATE_S - (time.time() - last)
                if wait > 0:  # inside djitellopy's 1 ms gate it would drop this packet without a word
                    time.sleep(min(wait, RC_GATE_S))
            ok = self._udp(self._tello.send_rc_control, *vals)
            if ok:
                self._rc_live = any(vals)
                self._rc_sent_t = time.monotonic()
            return ok

    def _start_watchdog(self) -> None:
        if self._watchdog is None or not self._watchdog.is_alive():
            self._watchdog = threading.Thread(target=self._watch, daemon=True, name="tello-rc-deadman")
            self._watchdog.start()

    def _watch(self) -> None:
        """Dead-man: one rc 0 after a non-zero stick that the control loop stopped refreshing."""
        while not self._closing.wait(0.05):
            hold = self.rc_hold_s
            if hold is None:
                return
            if (self._rc_live and self._pending is None and self._flying
                    and time.monotonic() - self._rc_call_t > hold):
                if self._send_rc((0, 0, 0, 0)):
                    self.watchdog_trips += 1
                    self.log.append((time.time(), f"rc watchdog: no stick update for {hold:g} s -> rc 0 0 0 0"))
                    log.warning("no rc update for %.1f s (control loop stalled?): sent rc 0 0 0 0", hold)

    def move(self, direction: str, cm: int) -> None:
        cm = check_move(direction, cm)
        direction = body_direction(direction, self.right_sign, self.forward_sign)
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
        # kite: never judge by height when the command did not even leave the laptop (Wi-Fi down): nothing happened
        sent = not result.startswith("error: not sent")
        if p is not None and p.cmd == "land" and result != "ok" and sent and not self.dry_run:
            h = self.telemetry().height_m
            if h is not None and h <= 0.2:
                result = "ok"  # the reply was lost / 'error' because it is already down: it has landed
        if p is not None and p.cmd == "takeoff" and result == "error: timeout" and not self.dry_run:
            h = self.telemetry().height_m
            if h is not None and h >= 0.3:
                result = "ok"  # kite: the 'ok' was lost but it is up: count it as flying so every landing path applies
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
        self._start(f"speed {int(cm_s)}", 5.0, 0.2)  # raises while busy: change nothing then
        self.move_speed_cm_s = int(cm_s)

    def temperature_c(self) -> float | None:
        """Highest board temperature from the state stream ('temph', C), None if not reported."""
        v = self._state().get("temph")
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    def telemetry(self) -> Telemetry:
        st = self._state()
        # djitellopy (2.5) does not timestamp state packets, but it stores a NEW dict per packet: stamp it on
        # first sight, so a frozen state stream (its receiver thread dies on one bad packet) shows up as stale
        if st is not self._last_state:
            self._last_state = st
            self._state_t = time.time() if st else 0.0
        t = self._state_t

        def num(k, scale=1.0):  # missing, non-numeric or non-finite -> None (unknown), never an exception
            try:
                v = float(st.get(k)) * scale
            except (TypeError, ValueError, OverflowError):  # OverflowError: a 400-digit int
                return None
            return v if math.isfinite(v) else None

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
