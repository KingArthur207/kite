"""A fake DJI Tello that speaks the Tello SDK 2.0 over real UDP on loopback, for end-to-end tests.

    with TelloEmulator(time_scale=4) as emu, djitellopy_loopback(emu) as link:
        drone = TelloDrone(host=emu.host, video=False)     # kite's real adapter, real djitellopy, real sockets
        drone.connect()
        drone.takeoff(); ...
        emu.snapshot().h                                     # ground truth: what the "drone" really did
        emu.set_drop("cmd", 0.3); link.wifi_down(); ...      # faults

What runs where
  command port   text commands in, replies out ("ok", "error", "error Motor stop", "out of range", values).
                 Replies go back to the sender's address, like the real drone. `rc` gets no reply.
  state port     after `command`, one ASCII state line per packet at state_hz (real time), in the exact field
                 order of the standard Tello (pitch;roll;yaw;vgx;vgy;vgz;templ;temph;tof;h;bat;baro;time;agx;agy;agz;)
                 sent to (controller IP, state_port).
  video port     (video=True, needs ffmpeg) after `streamon`, H.264 Annex-B at 960x720 / 30 fps in 1460-byte UDP
                 packets, drawn from the emulated pose (coloured poles on the ground, a red one due north, the
                 horizon moving with pitch), relayed through the emulator so drop / pause / Wi-Fi faults apply.

Time: the flight model runs on a simulated clock, `time_scale` times faster than real time (takeoff 5 s -> 1.25 s
at 4x; also moves, landing, the 15 s auto-land, battery drain, temperature). Network-level things stay in real
time: state packet rate, reply latency, late replies after `stop`, injected delays. kite's own timeouts (1 s stale
telemetry, 1 s stop settle, per-command timeouts) are real time, so tests keep them meaningful.

The port problem (djitellopy binds its client to ('', 8889) and sends to (host, 8889); on macOS loopback the
emulator cannot also bind 127.0.0.1:8889): see djitellopy_loopback(). In short, djitellopy's two receiver sockets
are bound to free ports (its CONTROL/STATE_UDP_PORT class attributes are read only when the first Tello() starts
its threads), and kite's `from djitellopy import Tello` gets a thin subclass whose only change is the destination
address (127.0.0.1, emulator port) and the video port. kite's TelloDrone, djitellopy's sender, receiver threads and
state parser are the real ones.

Assumed physical numbers (TelloPhysics; sources: kite-notes-02.md, the team spec section 3, kite's own comments; all
"to measure" on the real drone are marked ASSUMED):
  forward / sideways speed  1.0 m/s at stick 100, linear (spec 3: ~0.96 m/s at full stick, from a stick-30 test)
  climb speed               1.0 m/s at stick 100 (ASSUMED)
  yaw rate                  90 deg/s at stick 100 (ASSUMED)
  stick -> velocity lag     first order, tau 0.4 s (ASSUMED; matches safety.py "keeps rising ~0.2 m after rc 0"
                            at stick 50)
  takeoff                   5 s to answer ok, hovers at 0.8 m (notes: "can take many seconds"; shoot doc ~1 m)
  land                      descends at 0.5 m/s, motors off 1 s after touchdown (kite expects ~4 s)
  discrete move             1 s + cm / speed (`speed`, default 100 cm/s; kite sends speed 50); rotate 60 deg/s + 0.5 s
  auto-land                 15 s after the last command it heard, rc included (notes 4.2)
  last stick is held        until the next rc / stop / auto-land (shoot doc: "it may hold its last stick")
  battery                   8 %/min flying (preflight's estimate), 0.5 %/min idle, +0.5 %/min streaming (ASSUMED);
                            takeoff refused below 10 %, lands itself at 5 % (ASSUMED thresholds)
  temperature               starts 52/55 C, +2 C/min on the ground, -2 C/min in flight down to 60 C (ASSUMED);
                            takeoff refused at temph >= 90 C (ASSUMED)
  tilt                      nose-down 8 deg per m/s + 3 deg per m/s^2 forward, same for roll; reported pitch is
                            + = nose UP (pose.PITCH_SIGN = +1, ASSUMED), roll + = right side down
  signs                     yaw grows with cw (DroneCfg.yaw_sign = +1, ASSUMED), vgx is NEGATIVE flying forward
                            (flight lag logs, Telemetry.vx docstring), vgy + right, vgz + up (ASSUMED), body frame
  units                     vg* in dm/s, h / tof in cm (tof >= 10 on the ground, 6553 beyond 8 m), baro m, time s
Protocol behaviour (ASSUMED where marked):
  replies                   "ok"; "error" (takeoff while flying / flat battery / overheated, land on the ground, a
                            discrete command while another runs); "error Motor stop" (a move with the motors off);
                            "out of range"; "unknown command: ..." (EDU-only commands); query values ("87", "801mm",
                            "8dm", "52~55C", "0s"). No reply to rc, nor before `command`.
  stop                      answers "ok" 0.3 s later; the move it cancels still answers 0.6 s after the stop (ASSUMED,
                            both configurable: kite ignores replies for 1 s after a stop)
  rc during a discrete move cancels it without a reply (notes 4.2); rc is ignored while taking off / landing
  land while landing        answered together with the first land, at touchdown (ASSUMED)
  one controller            the first address that sends `command` owns the drone; others are ignored (ASSUMED
                            silent); phone_app() models the phone holding the video
  100 ms rule               optional (min_command_gap_s), off by default: unverified folklore from djitellopy
"""

from __future__ import annotations

import contextlib
import errno
import heapq
import itertools
import math
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..sources.base import FrameSource
from ..types import Frame

LOCALHOST = "127.0.0.1"
G = 9.81
MOVE_WORDS = ("forward", "back", "left", "right", "up", "down")
EDU_ONLY = ("mon", "moff", "mdirection", "ap", "port", "setfps", "setbitrate", "setresolution", "downvision",
            "motoron", "motoroff", "throwfly", "keepalive", "ext", "reboot", "active?", "go", "curve", "jump")
PORTS = ("cmd", "state", "video")


@dataclass
class TelloPhysics:
    """The flight model's numbers. See the module docstring for where each comes from."""

    max_speed_m_s: float = 1.0  # fb / lr stick 100
    max_climb_m_s: float = 1.0  # ud stick 100
    max_yaw_rate_deg_s: float = 90.0  # yaw stick 100
    lag_tau_s: float = 0.4  # first-order lag from stick to velocity / yaw rate
    takeoff_height_m: float = 0.8
    takeoff_s: float = 5.0
    land_speed_m_s: float = 0.5
    land_settle_s: float = 1.0
    move_overhead_s: float = 1.0
    default_speed_cm_s: int = 100  # discrete-move speed until a `speed` command
    rotate_rate_deg_s: float = 60.0
    rotate_overhead_s: float = 0.5
    flip_s: float = 1.5
    min_height_m: float = 0.3  # rc / down moves cannot go lower in flight
    max_height_m: float = 10.0
    auto_land_s: float = 15.0
    tilt_deg_per_m_s: float = 8.0
    tilt_deg_per_m_s2: float = 3.0
    max_tilt_deg: float = 25.0
    drain_flying_pct_s: float = 8.0 / 60.0
    drain_idle_pct_s: float = 0.5 / 60.0
    drain_stream_pct_s: float = 0.5 / 60.0
    min_takeoff_battery_pct: float = 10.0
    auto_land_battery_pct: float = 5.0
    min_flip_battery_pct: float = 50.0
    temp_ground_rise_c_s: float = 2.0 / 60.0
    temp_flight_fall_c_s: float = 2.0 / 60.0
    temp_flight_floor_c: float = 60.0
    temp_max_c: float = 95.0
    overheat_c: float = 90.0
    tof_min_cm: int = 10
    tof_max_cm: int = 800
    tof_out_of_range: int = 6553
    # sign conventions of the state stream
    yaw_sign: int = 1  # +1: state yaw grows with cw
    pitch_sign: int = 1  # +1: state pitch + = nose up
    roll_sign: int = 1  # +1: state roll + = right side down
    vgx_forward_sign: int = -1
    vgy_right_sign: int = 1
    vgz_up_sign: int = 1
    velocity_frame: str = "body"  # body | world (world: vgx north, vgy east)


@dataclass
class Snapshot:
    """Ground truth of the emulated drone (SI units, heading in degrees clockwise from north)."""

    t_sim: float
    phase: str  # ground | takeoff | air | landing | falling
    maneuver: str | None  # the running discrete command ("forward 100", "cw 90", "takeoff", "land", ...)
    motors: bool
    north: float
    east: float
    h: float
    heading: float
    v_fwd: float
    v_right: float
    v_up: float
    yaw_rate: float
    tilt_fwd_deg: float  # + = nose down
    tilt_right_deg: float  # + = right side down
    sticks: tuple
    battery: float
    templ: float
    temph: float
    motor_time: float
    speed_cm_s: int
    sdk_mode: bool
    stream_on: bool
    last_cmd_age_sim: float

    @property
    def flying(self) -> bool:
        return self.phase in ("takeoff", "air", "landing")


@dataclass
class _Maneuver:
    kind: str  # takeoff | land | move | rotate | flip
    label: str
    t0: float
    duration: float
    start: tuple  # (north, east, h, heading)
    delta: tuple  # (d_north, d_east, d_h, d_heading)
    reply_to: list = field(default_factory=list)


def _smooth(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return p * p * (3.0 - 2.0 * p)


def _wrap(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _state_receiver_port(th: threading.Thread | None) -> int | None:
    """The port a djitellopy state receiver thread has bound (read from its frame: probing the port with a bind of our
    own could race its bind and kill it), None if it is not (yet / any more) listening."""
    if th is None or not th.is_alive():
        return None
    frame = sys._current_frames().get(th.ident)
    while frame is not None and frame.f_code.co_name != "udp_state_receiver":
        frame = frame.f_back
    sock = None if frame is None else frame.f_locals.get("state_socket")
    try:
        port = sock.getsockname()[1] if sock is not None else 0
    except OSError:
        return None
    return port or None


# ---------------------------------------------------------------------------------------------- scene / camera
_RNG = np.random.default_rng(7)
POLE_H_M = 2.0
RED = (0, 0, 255)  # BGR: the marker pole due north, 6 m from the takeoff point
POLES = [(6.0, 0.0, RED)] + [
    (float(r * math.cos(b)), float(r * math.sin(b)), tuple(int(c) for c in col))
    for r, b, col in zip(_RNG.uniform(3.0, 12.0, 18), _RNG.uniform(0.35, 2 * math.pi - 0.35, 18),
                         [(255, 120, 0), (0, 200, 0), (255, 0, 200), (0, 220, 220), (40, 40, 40), (255, 255, 255)] * 3)
]
FX_960 = 920.0  # published Tello stream calibration (notes 8)


def render_view(s: Snapshot, size=(960, 720)) -> np.ndarray:
    """What the forward camera would see: sky, ground, and the poles around the takeoff point (BGR uint8)."""
    import cv2

    w, hgt = size
    f = FX_960 * w / 960.0
    cx, cy = w / 2.0, hgt / 2.0
    shift = -f * math.tan(math.radians(s.tilt_fwd_deg))  # nose down: the world moves up in the image
    horizon = int(min(max(cy + shift, 0), hgt))
    img = np.empty((hgt, w, 3), np.uint8)
    img[:horizon] = (235, 206, 135)
    img[horizon:] = (80, 105, 85)
    psi = math.radians(s.heading)
    fn, fe = math.cos(psi), math.sin(psi)
    rn, re_ = -math.sin(psi), math.cos(psi)
    cam_h = max(s.h, 0.05)
    items = []
    for pn, pe, color in POLES:
        dn, de = pn - s.north, pe - s.east
        fwd, right = dn * fn + de * fe, dn * rn + de * re_
        if fwd < 0.3:
            continue
        u = cx + f * right / fwd
        half = max(1.0, f * 0.12 / fwd)
        v_top = cy + shift - f * (POLE_H_M - cam_h) / fwd
        v_bot = cy + shift + f * cam_h / fwd
        items.append((fwd, u, half, v_top, v_bot, color))
    for _, u, half, v_top, v_bot, color in sorted(items, key=lambda it: -it[0]):
        x0, x1 = int(max(u - half, -1)), int(min(u + half, w))
        y0, y1 = int(max(v_top, -1)), int(min(v_bot, hgt))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
    cv2.putText(img, f"h {s.h:.2f} hdg {s.heading:.0f}", (10, hgt - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * w / 960.0 + 0.3, (255, 255, 255), 1)
    return img


def red_pole_column(img: np.ndarray) -> float | None:
    """Mean x of the red marker pole's pixels (None if it is not in view): a test probe for the rendered video."""
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    mask = (r > 170) & (g < 80) & (b < 80)
    if mask.sum() < 20:
        return None
    return float(np.nonzero(mask)[1].mean())


# ---------------------------------------------------------------------------------------------- emulator
class TelloEmulator:
    """See the module docstring. All public methods are thread-safe."""

    def __init__(self, host: str = LOCALHOST, command_port: int = 0, state_port: int = 8890, video_port: int = 11111,
                 time_scale: float = 1.0, physics: TelloPhysics | None = None, battery_pct: float = 87.0,
                 video: bool = False, video_size=(960, 720), video_fps: float = 30.0, state_hz: float = 10.0,
                 reply_latency_s: float = 0.003, mission_pad_fields: bool = False, exclusive: bool = True,
                 stop_reply_s: float = 0.3, cancelled_reply: str | None = "ok", cancelled_reply_s: float = 0.6,
                 min_command_gap_s: float = 0.0, terrain=None, seed: int = 0):
        """time_scale: simulated seconds per real second. cancelled_reply: what a move cancelled by `stop` still
        answers (None: nothing), cancelled_reply_s real seconds after the stop; `stop` itself answers after
        stop_reply_s. min_command_gap_s > 0 drops a (non-rc) command arriving sooner than that after the previous
        one (notes 4.2: "commands sent closer than ~100 ms apart get dropped"; off by default, unverified).
        terrain(north, east) -> height (m) of whatever is under the drone (a table...): tof sees it, h does not."""
        self.host = host
        self.physics = physics or TelloPhysics()
        self.time_scale = float(time_scale)
        self.state_port = state_port
        self.video_port = video_port
        self.video = video
        self.video_size = tuple(video_size)
        self.video_fps = video_fps
        self.state_hz = state_hz
        self.reply_latency_s = reply_latency_s
        self.mission_pad_fields = mission_pad_fields
        self.exclusive = exclusive
        self.stop_reply_s = stop_reply_s
        self.cancelled_reply = cancelled_reply
        self.cancelled_reply_s = cancelled_reply_s
        self.min_command_gap_s = min_command_gap_s
        self.terrain = terrain
        self._rng = random.Random(seed)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_sock.bind((host, command_port))
        self._cmd_sock.settimeout(0.05)
        self.command_port = self._cmd_sock.getsockname()[1]
        self._out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # state + video go out from here
        self._out_sock.bind((host, 0))
        self._outbox: list = []  # heap of (t_real_due, n, port, payload, addr)
        self._seq = itertools.count()
        # logs (read them under no lock: appends are atomic)
        self.received: list[tuple[float, float, str, tuple, str]] = []  # (t_real, t_sim, text, addr, verdict)
        self.replies: list[tuple[float, str, str]] = []  # (t_real, text, "sent" | why dropped)
        self.events: list[tuple[float, str]] = []  # (t_sim, what the drone did on its own / notable)
        self.state_packets = 0
        self.video_packets = 0
        self.video_frames = 0  # frames handed to the encoder
        # drone state
        p = self.physics
        self._t_sim = 0.0
        self._sdk = False
        self._controller: tuple | None = None
        self._stream = False
        self._phase = "ground"
        self._motors = False
        self._man: _Maneuver | None = None
        self._n = self._e = self._h = 0.0
        self._heading = 0.0
        self._v_fwd = self._v_right = self._v_up = self._yaw_rate = 0.0
        self._a_fwd = self._a_right = self._a_up = 0.0
        self._tilt_fwd = self._tilt_right = 0.0
        self._sticks = (0, 0, 0, 0)
        self._battery = float(battery_pct)
        self._templ, self._temph = 52.0, 55.0
        self._temp_override = False
        self._motor_time = 0.0
        self._speed = p.default_speed_cm_s
        self._last_cmd_sim = 0.0
        self._last_text_cmd_real = -1e9
        # faults
        self._drop = {k: 0.0 for k in PORTS}
        self._garbage = {k: 0 for k in PORTS}
        self._garbage_replace = False
        self._reply_delay = 0.0
        self._silent = False
        self._state_paused = False
        self._video_paused = False
        self._wifi_down = False
        self._phone_app = False
        # video
        self._ffmpeg: subprocess.Popen | None = None
        self._relay_sock: socket.socket | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> TelloEmulator:
        for fn in (self._command_loop, self._tick_loop):
            th = threading.Thread(target=fn, daemon=True, name=f"tello-emulator-{fn.__name__}")
            th.start()
            self._threads.append(th)
        return self

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2)
        if self._ffmpeg is not None:
            with contextlib.suppress(Exception):
                self._ffmpeg.stdin.close()
            self._ffmpeg.kill()
            self._ffmpeg.wait(timeout=5)
            self._ffmpeg = None
        for s in (self._cmd_sock, self._out_sock, self._relay_sock):
            if s is not None:
                s.close()

    def __enter__(self) -> TelloEmulator:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ inspection
    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(
                t_sim=self._t_sim, phase=self._phase, maneuver=None if self._man is None else self._man.label,
                motors=self._motors, north=self._n, east=self._e, h=self._h, heading=_wrap(self._heading),
                v_fwd=self._v_fwd, v_right=self._v_right, v_up=self._v_up, yaw_rate=self._yaw_rate,
                tilt_fwd_deg=self._tilt_fwd, tilt_right_deg=self._tilt_right, sticks=self._sticks,
                battery=self._battery, templ=self._templ, temph=self._temph, motor_time=self._motor_time,
                speed_cm_s=self._speed, sdk_mode=self._sdk, stream_on=self._stream,
                last_cmd_age_sim=self._t_sim - self._last_cmd_sim)

    def commands(self, include_rc: bool = True, accepted_only: bool = True) -> list[str]:
        """Texts the drone received (in order), optionally without rc / with the ignored or dropped ones."""
        return [text for _, _, text, _, verdict in list(self.received)
                if (include_rc or not text.startswith("rc ")) and (not accepted_only or verdict == "ok")]

    def rc_received(self) -> list[tuple[int, int, int, int]]:
        out = []
        for text in self.commands():
            if text.startswith("rc "):
                with contextlib.suppress(ValueError):
                    out.append(tuple(int(v) for v in text.split()[1:5]))
        return out

    def wait_until(self, pred, timeout_s: float = 5.0, poll_s: float = 0.01) -> Snapshot:
        """Poll the ground truth until pred(snapshot) holds; AssertionError with the last snapshot on timeout."""
        t_end = time.monotonic() + timeout_s
        while True:
            s = self.snapshot()
            if pred(s):
                return s
            if time.monotonic() > t_end:
                raise AssertionError(f"emulator condition not met within {timeout_s} s: {s}")
            time.sleep(poll_s)

    def real_s(self, sim_s: float) -> float:
        """Real seconds for a span of simulated time."""
        return sim_s / self.time_scale

    # ------------------------------------------------------------------ scenario setters
    def set_battery(self, pct: float) -> None:
        with self._lock:
            self._battery = float(pct)

    def set_temperature(self, templ: float, temph: float, hold: bool = True) -> None:
        """hold: keep these values (no warming / cooling) until the next call with hold=False."""
        with self._lock:
            self._templ, self._temph, self._temp_override = float(templ), float(temph), hold

    def place(self, north: float | None = None, east: float | None = None, h: float | None = None,
              heading: float | None = None) -> None:
        """Teleport (tests only: e.g. a known height / heading to check telemetry units)."""
        with self._lock:
            if north is not None:
                self._n = float(north)
            if east is not None:
                self._e = float(east)
            if h is not None:
                self._h = float(h)
            if heading is not None:
                self._heading = float(heading)

    # ------------------------------------------------------------------ faults
    def set_drop(self, port: str, fraction: float) -> None:
        """Drop this fraction of packets on a port ('cmd': commands in AND replies out, 'state', 'video', 'all')."""
        with self._lock:
            for k in PORTS if port == "all" else (port,):
                self._drop[k] = float(fraction)

    def silence_replies(self, on: bool = True) -> None:
        """Commands are still executed, but no reply is ever sent (a lost reply)."""
        with self._lock:
            self._silent = on

    def delay_replies(self, seconds: float) -> None:
        """Extra real-time delay on every reply from now on (0 to clear)."""
        with self._lock:
            self._reply_delay = float(seconds)

    def garbage(self, port: str = "cmd", n: int = 1, replace: bool = False) -> None:
        """The next n packets on the port are preceded by (replace=False) or replaced by (replace=True) random
        non-ASCII bytes. On 'state' one such packet kills djitellopy's state receiver thread for good."""
        with self._lock:
            self._garbage[port] += n
            self._garbage_replace = replace

    def pause_state(self, on: bool = True) -> None:
        with self._lock:
            self._state_paused = on

    def pause_video(self, on: bool = True) -> None:
        with self._lock:
            self._video_paused = on

    def wifi(self, up: bool) -> None:
        """down: nothing gets in or out (commands, replies, state, video). Use the link's wifi_down() to also make
        the laptop's sendto fail like macOS does."""
        with self._lock:
            self._wifi_down = not up
            self.events.append((self._t_sim, "wifi up" if up else "wifi down"))

    def phone_app(self, connected: bool = True) -> None:
        """The Tello phone app is connected: the drone still answers us, but streams its video to the phone."""
        with self._lock:
            self._phone_app = connected

    def clear_faults(self) -> None:
        with self._lock:
            self._drop = {k: 0.0 for k in PORTS}
            self._garbage = {k: 0 for k in PORTS}
            self._reply_delay = 0.0
            self._silent = self._state_paused = self._video_paused = self._wifi_down = self._phone_app = False

    def send_raw_state(self, payload: bytes) -> None:
        """One raw datagram on the state port (e.g. a malformed line)."""
        with self._lock:
            addr = self._state_addr()
        if addr is not None:
            self._out_sock.sendto(payload, addr)

    # ------------------------------------------------------------------ network plumbing
    def _state_addr(self):
        return None if self._controller is None else (self._controller[0], self.state_port)

    def _lose(self, port: str) -> bool:
        return self._wifi_down or (self._drop[port] > 0 and self._rng.random() < self._drop[port])

    def _garbage_bytes(self) -> bytes:
        return bytes(self._rng.randrange(0x80, 0x100) for _ in range(self._rng.randrange(4, 24)))

    def _reply(self, addr, text: str, delay_s: float = 0.0) -> None:
        """Queue a reply (lock held). delay_s is real time."""
        due = time.monotonic() + self.reply_latency_s + self._reply_delay + delay_s
        heapq.heappush(self._outbox, (due, next(self._seq), "cmd", text, addr))

    def _flush_outbox(self) -> None:
        now = time.monotonic()
        while True:
            with self._lock:
                if not self._outbox or self._outbox[0][0] > now:
                    return
                _, _, port, text, addr = heapq.heappop(self._outbox)
                if self._silent:
                    self.replies.append((time.time(), text, "silenced"))
                    continue
                if self._lose(port):
                    self.replies.append((time.time(), text, "lost"))
                    continue
                packets = []
                if self._garbage[port] > 0:
                    self._garbage[port] -= 1
                    packets.append(self._garbage_bytes())
                    if self._garbage_replace:
                        self.replies.append((time.time(), text, "replaced by garbage"))
                        text = None
                if text is not None:
                    packets.append(text.encode())
                    self.replies.append((time.time(), text, "sent"))
            for data in packets:
                with contextlib.suppress(OSError):
                    self._cmd_sock.sendto(data, addr)

    def _command_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._cmd_sock.recvfrom(2048)
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            text = data.decode("utf-8", errors="replace").strip()
            with self._lock:
                self._on_datagram(text, addr)

    # ------------------------------------------------------------------ command handling (lock held)
    def _log(self, text, addr, verdict):
        self.received.append((time.time(), self._t_sim, text, addr, verdict))

    def _on_datagram(self, text: str, addr) -> None:
        if self._lose("cmd"):
            self._log(text, addr, "lost")
            return
        if self.exclusive and self._controller is not None and addr != self._controller:
            self._log(text, addr, "ignored: another controller")
            return
        is_rc = text.startswith("rc ")
        if not is_rc and self.min_command_gap_s > 0:
            now = time.monotonic()
            if now - self._last_text_cmd_real < self.min_command_gap_s:
                self._log(text, addr, "dropped: too soon after the previous command")
                return
            self._last_text_cmd_real = now
        if not self._sdk:
            if text == "command":
                self._sdk, self._controller = True, addr
                self._last_cmd_sim = self._t_sim
                self._log(text, addr, "ok")
                self._reply(addr, "ok")
            else:
                self._log(text, addr, "ignored: not in SDK mode")
            return
        self._log(text, addr, "ok")
        self._last_cmd_sim = self._t_sim  # anything heard, rc included, resets the auto-land timer
        words = text.split()
        if not words:
            return
        w, args = words[0].lower(), words[1:]
        if w == "rc":
            self._on_rc(args)
            return
        reply = self._dispatch(w, args, addr)
        if reply is not None:
            self._reply(addr, reply)

    def _on_rc(self, args) -> None:
        try:
            vals = tuple(int(a) for a in args)
        except ValueError:
            return
        if len(vals) != 4 or any(abs(v) > 100 for v in vals):
            return  # malformed rc: ignored, no reply (like every rc)
        if self._phase != "air":
            return  # motors off / taking off / landing: sticks do nothing
        if self._man is not None and self._man.kind in ("move", "rotate", "flip"):
            self.events.append((self._t_sim, f"rc cancels {self._man.label}"))
            self._man = None  # notes 4.2: an rc packet during a discrete move cancels the move (no reply)
        self._sticks = vals

    def _dispatch(self, w: str, args: list, addr) -> str | None:
        p = self.physics
        if w == "command":
            return "ok"
        if w in ("streamon", "streamoff"):
            self._stream = w == "streamon"
            if self._stream and self.video:
                self._ensure_video()
            return "ok"
        if w == "takeoff":
            if self._phase != "ground":
                return "error"
            if self._battery < p.min_takeoff_battery_pct:
                return "error"
            if self._temph >= p.overheat_c:
                return "error"
            self._motors, self._phase, self._sticks = True, "takeoff", (0, 0, 0, 0)
            self._man = _Maneuver("takeoff", "takeoff", self._t_sim, p.takeoff_s, self._pose(),
                                  (0.0, 0.0, p.takeoff_height_m, 0.0), [addr])
            self.events.append((self._t_sim, "takeoff"))
            return None
        if w == "land":
            if self._phase in ("ground", "falling"):
                return "error"
            if self._phase == "landing":
                self._man.reply_to.append(addr)  # answered together at touchdown
                return None
            self._start_landing("land", [addr])
            return None
        if w == "emergency":
            self._man = None
            self._motors, self._sticks = False, (0, 0, 0, 0)
            self._phase = "falling" if self._h > 0.0 else "ground"
            self.events.append((self._t_sim, "emergency: motors off"))
            return "ok"
        if w == "stop":
            if self._phase == "air":
                if self._man is not None and self._man.kind in ("move", "rotate", "flip"):
                    cancelled = self._man
                    self._man = None
                    self.events.append((self._t_sim, f"stop cancels {cancelled.label}"))
                    if self.cancelled_reply is not None:
                        for a in cancelled.reply_to:
                            self._reply(a, self.cancelled_reply, self.cancelled_reply_s)
                self._sticks = (0, 0, 0, 0)
            self._reply(addr, "ok", self.stop_reply_s)
            return None
        if w in MOVE_WORDS or w in ("cw", "ccw"):
            try:
                n = int(args[0]) if len(args) == 1 else None
            except ValueError:
                n = None
            if n is None:
                return "error"
            lo, hi = (20, 500) if w in MOVE_WORDS else (1, 360)
            if not lo <= n <= hi:
                return "out of range"
            if not self._motors:
                return "error Motor stop"
            if self._phase != "air" or self._man is not None:
                return "error"
            self._start_discrete(w, n, addr)
            return None
        if w == "flip":
            if len(args) != 1 or args[0] not in ("l", "r", "f", "b"):
                return "error"
            if not self._motors:
                return "error Motor stop"
            if self._phase != "air" or self._man is not None or self._battery < p.min_flip_battery_pct:
                return "error"
            self._man = _Maneuver("flip", f"flip {args[0]}", self._t_sim, p.flip_s, self._pose(), (0, 0, 0, 0), [addr])
            return None
        if w == "speed":
            try:
                v = int(float(args[0])) if len(args) == 1 else None
            except ValueError:
                v = None
            if v is None:
                return "error"
            if not 10 <= v <= 100:
                return "out of range"
            self._speed = v
            return "ok"
        if w.endswith("?"):
            return self._query(w)
        return f"unknown command: {' '.join([w, *args])}"  # EDU-only commands (EDU_ONLY) included

    def _query(self, w: str) -> str:
        if w == "battery?":
            return str(int(self._battery))
        if w == "speed?":
            return f"{float(self._speed):.1f}"
        if w == "time?":
            return f"{int(self._motor_time)}s"
        if w == "height?":
            return f"{int(round(self._h * 10))}dm"
        if w == "temp?":
            return f"{int(self._templ)}~{int(self._temph)}C"
        if w == "attitude?":
            pitch, roll, yaw = self._attitude()
            return f"pitch:{pitch};roll:{roll};yaw:{yaw};"
        if w == "baro?":
            return f"{self._baro():.2f}"
        if w == "tof?":
            return f"{self._tof_cm() * 10}mm"
        if w == "wifi?":
            return "90"
        if w == "sdk?":
            return "20"
        if w == "sn?":
            return "0TQZEMU000001"
        return f"unknown command: {w}"

    def _pose(self) -> tuple:
        return (self._n, self._e, self._h, self._heading)

    def _start_discrete(self, w: str, n: int, addr) -> None:
        p = self.physics
        if w in ("cw", "ccw"):
            d = n if w == "cw" else -n
            self._man = _Maneuver("rotate", f"{w} {n}", self._t_sim, p.rotate_overhead_s + n / p.rotate_rate_deg_s,
                                  self._pose(), (0.0, 0.0, 0.0, float(d)), [addr])
            return
        m = n / 100.0
        psi = math.radians(self._heading)
        fwd = (math.cos(psi), math.sin(psi))
        right = (-math.sin(psi), math.cos(psi))
        dn = de = dh = 0.0
        if w == "forward":
            dn, de = fwd[0] * m, fwd[1] * m
        elif w == "back":
            dn, de = -fwd[0] * m, -fwd[1] * m
        elif w == "right":
            dn, de = right[0] * m, right[1] * m
        elif w == "left":
            dn, de = -right[0] * m, -right[1] * m
        elif w == "up":
            dh = min(m, p.max_height_m - self._h)
        else:
            dh = -min(m, max(self._h - p.min_height_m, 0.0))
        duration = p.move_overhead_s + n / max(self._speed, 10)
        self._man = _Maneuver("move", f"{w} {n}", self._t_sim, duration, self._pose(), (dn, de, dh, 0.0), [addr])

    def _start_landing(self, why: str, reply_to: list) -> None:
        p = self.physics
        h = max(self._h - self._ground(), 0.0)
        self._man = _Maneuver("land", "land", self._t_sim, h / p.land_speed_m_s + p.land_settle_s, self._pose(),
                              (0.0, 0.0, -h, 0.0), list(reply_to))
        self._phase, self._sticks = "landing", (0, 0, 0, 0)
        self.events.append((self._t_sim, why))

    def _ground(self) -> float:
        return 0.0

    # ------------------------------------------------------------------ state line
    def _attitude(self) -> tuple[int, int, int]:
        p = self.physics
        pitch = int(round(p.pitch_sign * -self._tilt_fwd))
        roll = int(round(p.roll_sign * self._tilt_right))
        yaw = int(round(_wrap(p.yaw_sign * self._heading)))
        return pitch, roll, yaw

    def _baro(self) -> float:
        return 31.25 + self._h

    def _tof_cm(self) -> int:
        p = self.physics
        under = self.terrain(self._n, self._e) if self.terrain is not None else 0.0
        cm = int(round((self._h - under) * 100))
        if cm > p.tof_max_cm:
            return p.tof_out_of_range
        return max(p.tof_min_cm, cm)

    def state_line(self) -> str:
        """The state packet the drone would send now."""
        with self._lock:
            p = self.physics
            pitch, roll, yaw = self._attitude()
            if p.velocity_frame == "world":
                psi = math.radians(self._heading)
                vx = self._v_fwd * math.cos(psi) - self._v_right * math.sin(psi)
                vy = self._v_fwd * math.sin(psi) + self._v_right * math.cos(psi)
            else:
                vx, vy = self._v_fwd, self._v_right
            vgx = int(round(p.vgx_forward_sign * vx * 10))
            vgy = int(round(p.vgy_right_sign * vy * 10))
            vgz = int(round(p.vgz_up_sign * self._v_up * 10))
            free_fall = self._phase == "falling"
            agx = self._a_fwd / G * 1000.0
            agy = self._a_right / G * 1000.0
            agz = 0.0 if free_fall else -1000.0 - self._a_up / G * 1000.0
            line = (f"pitch:{pitch};roll:{roll};yaw:{yaw};vgx:{vgx};vgy:{vgy};vgz:{vgz};"
                    f"templ:{int(self._templ)};temph:{int(self._temph)};tof:{self._tof_cm()};"
                    f"h:{int(round(self._h * 100))};bat:{int(self._battery)};baro:{self._baro():.2f};"
                    f"time:{int(self._motor_time)};agx:{agx:.2f};agy:{agy:.2f};agz:{agz:.2f};\r\n")
            if self.mission_pad_fields:
                line = "mid:-1;x:0;y:0;z:0;mpry:0,0,0;" + line
            return line

    def _send_state(self) -> None:
        with self._lock:
            addr = self._state_addr()
            if addr is None or self._state_paused or self._lose("state"):
                return
            packets = []
            if self._garbage["state"] > 0:
                self._garbage["state"] -= 1
                packets.append(self._garbage_bytes())
                if not self._garbage_replace:
                    packets.append(self.state_line().encode())
            else:
                packets.append(self.state_line().encode())
            self.state_packets += 1
        for data in packets:
            with contextlib.suppress(OSError):
                self._out_sock.sendto(data, addr)

    # ------------------------------------------------------------------ flight model
    def _tick_loop(self) -> None:
        last = time.monotonic()
        next_state = last
        while not self._stop.wait(0.005):
            now = time.monotonic()
            dt = (now - last) * self.time_scale
            last = now
            with self._lock:
                self._step(dt)
            self._flush_outbox()
            if now >= next_state:
                next_state = max(next_state + 1.0 / self.state_hz, now)
                self._send_state()

    def _step(self, dt: float) -> None:
        if dt <= 0:
            return
        p = self.physics
        self._t_sim += dt
        v_prev = (self._v_fwd, self._v_right, self._v_up)
        if self._phase == "ground":
            self._v_fwd = self._v_right = self._v_up = self._yaw_rate = 0.0
        elif self._phase == "falling":
            self._v_fwd *= math.exp(-dt / 0.5)
            self._v_right *= math.exp(-dt / 0.5)
            self._v_up -= G * dt
            self._h += self._v_up * dt
            self._yaw_rate = 0.0
            if self._h <= 0.0:
                self._h, self._phase = 0.0, "ground"
                self._v_fwd = self._v_right = self._v_up = 0.0
                self.events.append((self._t_sim, "hit the ground"))
        elif self._man is not None:
            self._step_maneuver(dt)
        else:
            self._step_sticks(dt)
        # derived: accelerations and tilt
        for i, name in enumerate(("_a_fwd", "_a_right", "_a_up")):
            v_now = (self._v_fwd, self._v_right, self._v_up)[i]
            a = (v_now - v_prev[i]) / dt
            setattr(self, name, getattr(self, name) + (a - getattr(self, name)) * min(1.0, dt / 0.1))
        if self._phase in ("ground", "falling"):
            self._tilt_fwd = self._tilt_right = 0.0
        else:
            lim = p.max_tilt_deg
            self._tilt_fwd = max(-lim, min(lim, p.tilt_deg_per_m_s * self._v_fwd + p.tilt_deg_per_m_s2 * self._a_fwd))
            self._tilt_right = max(-lim, min(lim, p.tilt_deg_per_m_s * self._v_right
                                             + p.tilt_deg_per_m_s2 * self._a_right))
        # battery, temperature, motor time
        drain = p.drain_flying_pct_s if self._motors else p.drain_idle_pct_s
        if self._stream:
            drain += p.drain_stream_pct_s
        self._battery = max(0.0, self._battery - drain * dt)
        if self._motors:
            self._motor_time += dt
        if not self._temp_override:
            if self._motors:
                if self._temph > p.temp_flight_floor_c:
                    self._temph = max(p.temp_flight_floor_c, self._temph - p.temp_flight_fall_c_s * dt)
            else:
                self._temph = min(p.temp_max_c, self._temph + p.temp_ground_rise_c_s * dt)
            self._templ = self._temph - 3.0
        # the drone acting on its own
        if self._phase == "air":
            if self._t_sim - self._last_cmd_sim > p.auto_land_s:
                self._man = None
                self._start_landing(f"auto-land: no command for {p.auto_land_s:g} s", [])
            elif self._battery <= p.auto_land_battery_pct:
                self._man = None
                self._start_landing("auto-land: battery critical", [])

    def _step_sticks(self, dt: float) -> None:
        p = self.physics
        lr, fb, ud, yw = self._sticks
        k = 1.0 - math.exp(-dt / p.lag_tau_s)
        self._v_fwd += (fb / 100.0 * p.max_speed_m_s - self._v_fwd) * k
        self._v_right += (lr / 100.0 * p.max_speed_m_s - self._v_right) * k
        self._v_up += (ud / 100.0 * p.max_climb_m_s - self._v_up) * k
        self._yaw_rate += (yw / 100.0 * p.max_yaw_rate_deg_s - self._yaw_rate) * k
        psi = math.radians(self._heading)
        self._n += (self._v_fwd * math.cos(psi) - self._v_right * math.sin(psi)) * dt
        self._e += (self._v_fwd * math.sin(psi) + self._v_right * math.cos(psi)) * dt
        self._heading = _wrap(self._heading + self._yaw_rate * dt)
        h = self._h + self._v_up * dt
        lo, hi = p.min_height_m, p.max_height_m
        if h < lo and self._v_up < 0:
            h, self._v_up = max(h, min(self._h, lo)), 0.0
        elif h > hi and self._v_up > 0:
            h, self._v_up = hi, 0.0
        self._h = h

    def _step_maneuver(self, dt: float) -> None:
        m, p = self._man, self.physics
        t = self._t_sim - m.t0
        if m.kind == "takeoff":  # spin up, climb, settle
            prog = _smooth((t - 0.35 * m.duration) / (0.4 * m.duration))
        elif m.kind == "land":  # descend at land speed, then settle
            descent = m.duration - p.land_settle_s
            prog = 1.0 if descent <= 0 else min(1.0, t / descent)
        else:
            prog = _smooth(t / m.duration)
        n0, e0, h0, psi0 = m.start
        dn, de, dh, dpsi = m.delta
        n1, e1, h1 = n0 + dn * prog, e0 + de * prog, h0 + dh * prog
        psi1 = psi0 + dpsi * prog
        vn, ve = (n1 - self._n) / dt, (e1 - self._e) / dt
        psi = math.radians(self._heading)
        self._v_fwd = vn * math.cos(psi) + ve * math.sin(psi)
        self._v_right = -vn * math.sin(psi) + ve * math.cos(psi)
        self._v_up = (h1 - self._h) / dt
        self._yaw_rate = _wrap(psi1 - self._heading) / dt
        self._n, self._e, self._h = n1, e1, h1
        self._heading = _wrap(psi1)
        if t >= m.duration:
            self._man = None
            self._v_fwd = self._v_right = self._v_up = self._yaw_rate = 0.0
            if m.kind == "takeoff":
                self._phase = "air"
            elif m.kind == "land":
                self._phase, self._motors, self._h = "ground", False, 0.0
                self.events.append((self._t_sim, "landed"))
            for a in m.reply_to:
                self._reply(a, "ok")

    # ------------------------------------------------------------------ video (ffmpeg H.264 through a relay)
    def _ensure_video(self) -> None:
        if self._ffmpeg is not None or self._stop.is_set():
            return
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("TelloEmulator(video=True) needs ffmpeg on PATH")
        self._relay_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._relay_sock.bind((LOCALHOST, 0))
        self._relay_sock.settimeout(0.1)
        relay = self._relay_sock.getsockname()[1]
        w, h = self.video_size
        self._ffmpeg = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", f"{self.video_fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
             "-g", "15", "-pix_fmt", "yuv420p", "-f", "h264", f"udp://{LOCALHOST}:{relay}?pkt_size=1460"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for fn in (self._render_loop, self._relay_loop):
            th = threading.Thread(target=fn, daemon=True, name=f"tello-emulator-{fn.__name__}")
            th.start()
            self._threads.append(th)

    def _render_loop(self) -> None:
        period = 1.0 / self.video_fps
        next_t = time.monotonic()
        while not self._stop.is_set():
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
            with self._lock:
                streaming = self._stream
            if not streaming:
                continue
            img = render_view(self.snapshot(), self.video_size)
            try:
                self._ffmpeg.stdin.write(img.tobytes())
                self._ffmpeg.stdin.flush()
                self.video_frames += 1
            except (BrokenPipeError, OSError, ValueError, AttributeError):
                return

    def _relay_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._relay_sock.recv(65536)
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                return
            with self._lock:
                if (self._controller is None or not self._stream or self._video_paused or self._phone_app
                        or self._lose("video")):
                    continue
                addr = (self._controller[0], self.video_port)
                self.video_packets += 1
            with contextlib.suppress(OSError):
                self._out_sock.sendto(data, addr)


# ---------------------------------------------------------------------------------------------- djitellopy side
class _FlakySocket:
    """Stands in for djitellopy's module-level client socket: sendto raises like macOS when the Wi-Fi is down."""

    def __init__(self, real: socket.socket):
        self.real = real
        self.down_errno: int | None = None

    def sendto(self, data, addr):
        if self.down_errno is not None:
            raise OSError(self.down_errno, "Network is unreachable (emulated Wi-Fi drop)")
        return self.real.sendto(data, addr)

    def __getattr__(self, name):
        return getattr(self.real, name)


_receivers_lock = threading.Lock()
_state_thread: threading.Thread | None = None


def ensure_djitellopy_receivers() -> tuple[int, int]:
    """djitellopy's response and state receivers bound to free ports, never the real 8889 / 8890 (so a test can run
    next to a live kite process). Returns (client command port, state port). Idempotent; restarts the state
    receiver if it died (it does on one non-ASCII state packet)."""
    global _state_thread
    from djitellopy import tello as dt

    T = dt.Tello
    with _receivers_lock:
        if not dt.threads_initialized:
            orig_cmd = T.CONTROL_UDP_PORT
            T.CONTROL_UDP_PORT = 0  # the OS picks the client port: read only for this first bind
            T.STATE_UDP_PORT = free_udp_port()
            before = set(threading.enumerate())
            level = T.LOGGER.level
            T.LOGGER.setLevel("WARNING")
            boot = T("kite-emulator-bootstrap")  # starts the receiver threads; registers a host nothing comes from
            T.LOGGER.setLevel(level)
            T.CONTROL_UDP_PORT = orig_cmd
            boot.end()
            new = [th for th in threading.enumerate() if th not in before and "udp_state_receiver" in th.name]
            _state_thread = new[0] if new else None
        if _state_thread is None:  # someone else's djitellopy started the threads: find its state receiver
            _state_thread = next((th for th in threading.enumerate() if th.is_alive()
                                  and "udp_state_receiver" in th.name), None)
        for attempt in range(3):
            t_end = time.monotonic() + 2.0
            while time.monotonic() < t_end:
                port = _state_receiver_port(_state_thread)
                if port is not None or (_state_thread is not None and not _state_thread.is_alive()):
                    break
                time.sleep(0.005)
            if port is not None:
                T.STATE_UDP_PORT = port
                break
            # dead (a garbage packet, or its bind failed): start another on a fresh port
            T.STATE_UDP_PORT = free_udp_port()
            _state_thread = threading.Thread(target=T.udp_state_receiver, daemon=True,
                                             name="kite-emulator (udp_state_receiver)")
            _state_thread.start()
        else:
            raise RuntimeError("djitellopy's state receiver does not start")
        sock = dt.client_socket
        real = sock.real if isinstance(sock, _FlakySocket) else sock
        return real.getsockname()[1], T.STATE_UDP_PORT


class DjitellopyLoopback:
    """Points djitellopy (as kite imports it) at a TelloEmulator. Use as a context manager; see djitellopy_loopback.

    While active, `djitellopy.Tello` is a subclass that sends to (host, emulator.command_port) and asks for video on
    video_port; everything else is djitellopy's own code. Its end() only cleans up the drones-table slot it
    registered (djitellopy's __del__ would otherwise delete a newer Tello's slot for the same host later).
    """

    def __init__(self, emu: TelloEmulator, video_port: int | None = None):
        self.emu = emu
        self.video_port = video_port or free_udp_port()
        self.client_port = self.state_port = None
        self._saved = None
        self._sock: _FlakySocket | None = None

    def __enter__(self) -> DjitellopyLoopback:
        import djitellopy
        from djitellopy import tello as dt

        self.client_port, self.state_port = ensure_djitellopy_receivers()
        self.emu.state_port = self.state_port
        self.emu.video_port = self.video_port
        cmd_port, video_port = self.emu.command_port, self.video_port

        class LoopbackTello(dt.Tello):
            def __init__(self, host=dt.Tello.TELLO_IP, retry_count=dt.Tello.RETRY_COUNT, vs_udp=None):
                super().__init__(host, retry_count, video_port)
                self.address = (host, cmd_port)
                self._own_slot = dt.drones[host]
                self._ended = False

            def end(self):
                if getattr(self, "_ended", True):
                    return
                self._ended = True
                if dt.drones.get(self.address[0]) is self._own_slot:
                    super().end()

        self.tello_class = LoopbackTello
        self._saved = djitellopy.Tello
        djitellopy.Tello = LoopbackTello
        sock = dt.client_socket
        self._sock = sock if isinstance(sock, _FlakySocket) else _FlakySocket(sock)
        self._sock.down_errno = None
        dt.client_socket = self._sock
        return self

    def __exit__(self, *exc) -> None:
        import djitellopy
        from djitellopy import tello as dt

        djitellopy.Tello = self._saved
        if self._sock is not None:
            self._sock.down_errno = None
            dt.client_socket = self._sock.real

    def wifi_down(self, err: int = errno.ENETUNREACH) -> None:
        """The laptop's Wi-Fi drops: every djitellopy send raises OSError(err), and the drone hears nothing."""
        self._sock.down_errno = err
        self.emu.wifi(up=False)

    def wifi_up(self) -> None:
        self._sock.down_errno = None
        self.emu.wifi(up=True)

    def make_tello(self, host: str | None = None):
        """A djitellopy Tello for this emulator (for tests that drive djitellopy directly)."""
        return self.tello_class(host or self.emu.host)


def djitellopy_loopback(emu: TelloEmulator, video_port: int | None = None) -> DjitellopyLoopback:
    return DjitellopyLoopback(emu, video_port)


# ---------------------------------------------------------------------------------------------- in-process camera
class EmulatorCamera(FrameSource):
    """A FrameSource drawing the emulator's view in-process (no H.264, no UDP): for fast tests that need live frames
    (the recorder refuses takeoff without them). Frames are BGR, read-only, stamped on arrival."""

    name = "tello-emulator"

    def __init__(self, emu: TelloEmulator, size=(320, 240), fps: float = 30.0):
        self.emu, self.size, self.fps = emu, tuple(size), fps
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.paused = False

    def start(self) -> EmulatorCamera:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="tello-emulator-camera")
            self._thread.start()
        return self

    def _run(self) -> None:
        seq = 0
        while not self._stop.wait(1.0 / self.fps):
            if self.paused:
                continue
            img = render_view(self.emu.snapshot(), self.size)
            img.flags.writeable = False
            seq += 1
            with self._lock:
                self._frame = Frame(img, time.time(), seq, self.name)

    def read(self):
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


__all__ = ["TelloEmulator", "TelloPhysics", "Snapshot", "DjitellopyLoopback", "djitellopy_loopback",
           "ensure_djitellopy_receivers", "EmulatorCamera", "render_view", "red_pole_column", "free_udp_port",
           "POLES"]
