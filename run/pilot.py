"""The drone side of the app: one control thread that owns the Tello, the person lock and the follow loop.

Commands arrive from the console (`send()`, any thread). The control thread runs at 10 Hz through kite's drone layer:
SafetyGovernor.check(), then the sticks (every tick: also the rc keep-alive), exactly like kite.record's loop.

    idle --takeoff--> taking off --ok--> locking --person locked--> following
                                                  ^        (lost: back to locking, it re-locks by itself)
    any --land--> landing --> landed (idle)
    e (emergency()) cuts the motors at once from the caller's thread; the app then quits.

Locking: kite.follow's tracker acquires the person right in front (a box cut off by the bottom edge, nearest the
centre) once it has seen them a few frames; FollowMode.start() then locks onto that track and the follow controller
(steered by the retrained kite.fly yaw source, or the follow law) holds the drone behind them at --height.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

TICK_S = 0.1  # 10 Hz control loop
LOCK_TIMEOUT_S = 30.0  # locking this long without a person: keep hovering, say so again


@dataclass
class Snapshot:
    """What the UI shows, read without locks (plain attributes swapped atomically)."""

    state: str = "starting"
    battery: float | None = None
    height_m: float | None = None
    video_fps: float = 0.0
    det_fps: float = 0.0
    link_error: str | None = None
    follow: str = ""
    yaw: str = ""


class SeenDetector:
    """Wraps the YOLO person detector to remember the last frame's boxes for the camera overlay."""

    def __init__(self, detect):
        self._detect = detect
        self.last: list = []
        self.last_t = 0.0

    def __call__(self, image):
        dets = self._detect(image)
        self.last, self.last_t = dets, time.time()
        return dets


class Pilot:
    def __init__(self, args, log):
        """log(text, kind): kind is sent | recv | info | warn | error (the console colours them)."""
        self.args = args
        self.log = log
        self.snap = Snapshot()
        self.drone = None
        self.frames = None
        self.follow = None
        self.seen: SeenDetector | None = None
        self._cmds: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pilot", daemon=True)
        self.state = "starting"
        self._lock_t0 = 0.0
        self._lock_said = 0.0
        self._pending = None  # discrete command we wait on: "takeoff" | "land"
        self.last_frame = None  # newest kite Frame (BGR), for the UI
        self._fps_n, self._fps_t = 0, time.time()
        self.closed = threading.Event()

    # ------------------------------------------------------------------ public (any thread)
    def start(self) -> Pilot:
        self._thread.start()
        return self

    def send(self, text: str) -> None:
        """A console command. 'e' is handled right here (no queue): motors off now."""
        cmd = text.strip().lower()
        if not cmd:
            return
        self.log(text.strip(), "sent")
        if self.closed.is_set() and cmd not in ("quit",):
            self.log("not connected: restart the app", "error")
            return
        if cmd == "takeoff" and self.state == "starting":
            self.log("still starting: wait for 'ready' below", "warn")
            return
        if cmd in ("e", "emergency"):
            self.emergency()
        elif cmd in ("takeoff", "land", "status", "help", "quit"):
            self._cmds.put(cmd)
        else:
            self.log(f"unknown command {text.strip()!r}: takeoff | land | e (emergency) | status | quit", "warn")

    def emergency(self) -> None:
        """Motors off immediately, from the caller's thread: the drone falls if it is in the air."""
        d = self.drone
        if d is None:
            self._stop.set()
            self._set("emergency")
            self.log("EMERGENCY: no drone connected", "error")
            return
        if self.follow is not None:
            self.follow.cancel("emergency")
        if d.emergency() is False:  # never reached the drone: keep the control loop (and the governor) running
            self.log("EMERGENCY NOT SENT (Wi-Fi down): type e again", "error")
            return
        self._stop.set()
        self._set("emergency")
        self.log("EMERGENCY: motors off", "error")

    def request_land(self) -> None:
        self._cmds.put("land")

    def shutdown(self, land: bool = True, timeout_s: float = 30.0) -> None:
        """Stop the loop; land first unless told not to (after an emergency); close the drone."""
        if land and self.drone is not None and not self._stop.is_set():
            self._cmds.put("land")
            t_end = time.time() + timeout_s
            while time.time() < t_end and (self.drone.flying or self.drone.busy()):
                time.sleep(0.1)
        self._stop.set()
        self._thread.join(timeout=5)
        self._close()

    # ------------------------------------------------------------------ control thread
    def _set(self, state: str) -> None:
        self.state = state
        self.snap.state = state

    def _connect(self) -> bool:
        from kite.camera import load_camera
        from kite.config import Config, resolve_video_backend
        from kite.drone import SafetyGovernor, make_tello
        from kite.follow import FollowCfg, FollowMode, YoloPersonDetector, build_yaw_source, detector_available

        a = self.args
        why = detector_available(a.weights)
        if why:
            self.log(f"person detector unavailable: {why}", "error")
            return False
        cfg = Config()
        if a.video_backend:
            cfg.drone.video_backend = resolve_video_backend(a.video_backend)
        fcfg = FollowCfg(height_m=a.height, video_lag_s=cfg.drone.video_lag_s, behind_m=a.behind)
        try:
            yaw_source, line = build_yaw_source(a.yaw, fcfg)
        except Exception as e:
            self.log(f"steering '{a.yaw}' unavailable ({e}); using the follow law", "warn")
            yaw_source, line = build_yaw_source("law", fcfg)
        self.log(line, "info")
        self.snap.yaw = a.yaw if yaw_source is not None else "law"
        self.drone = SafetyGovernor(make_tello(cfg, dry_run=a.dry_run), cfg.safety)
        self.log(f"connecting to the Tello ({'DRY RUN: nothing reaches the motors' if a.dry_run else 'LIVE'}) ...",
                 "info")
        try:
            self.drone.connect()
        except Exception as e:
            self.log(f"cannot connect: {e}", "error")
            self.log("join the TELLO-xxxxxx Wi-Fi and close other Tello programs (Windows: allow Python through the "
                     "firewall on Private and Public networks)", "warn")
            self.drone = None
            return False
        tel = self.drone.telemetry()
        self.log(f"connected, battery {tel.battery_pct:.0f}%" if tel.battery_pct is not None else "connected", "recv")
        self.frames = self.drone.frame_source()
        try:
            self.seen = SeenDetector(YoloPersonDetector(a.weights))
        except Exception as e:
            self.log(f"person detector failed to load: {e}", "error")
            return False
        self.follow = FollowMode(self.frames, lambda: self.seen, fcfg, load_camera(), max_rc=cfg.safety.max_rc,
                                 yaw_source=yaw_source, log=self._follow_log).start_tracker()
        self.log(f"ready: video ({cfg.drone.video_backend} reader), person detector on {self.seen._detect.device}. "
                 "Stand in front of the drone, then type takeoff", "info")
        return True

    def _follow_log(self, msg: str, quiet: bool = False, **extra) -> None:
        if quiet or msg == "follow refused":  # refused every tick while locking: the pilot says why every 3 s
            return
        detail = ", ".join(f"{k} {v}" for k, v in extra.items() if k in ("reason", "box", "conf", "why"))
        line = msg if msg.lower().startswith("follow") else f"follow: {msg}"
        self.log(line + (f" ({detail})" if detail else ""), "info")

    def _run(self) -> None:
        try:
            if not self._connect():
                self._set("offline")
                return
            self._set("idle")
            next_tick = time.monotonic()
            while not self._stop.is_set():
                self._read_frame()
                while True:
                    try:
                        self._command(self._cmds.get_nowait())
                    except queue.Empty:
                        break
                now = time.monotonic()
                if now >= next_tick:
                    next_tick = now + TICK_S
                    self._tick()
                time.sleep(0.005)
        except Exception as e:  # a bug here must still land the drone
            self.log(f"control loop error: {e!r}", "error")
            logging.getLogger("kite.run").exception("pilot")
            if self.drone is not None and not self._stop.is_set():
                try:
                    self.drone.land()
                except Exception:
                    pass
        finally:
            self.closed.set()

    def _read_frame(self) -> None:
        f = self.frames.read() if self.frames is not None else None
        if f is not None and (self.last_frame is None or f.seq != self.last_frame.seq):
            self.last_frame = f
            self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self.snap.video_fps = self._fps_n / (now - self._fps_t)
            self._fps_n, self._fps_t = 0, now

    def _command(self, cmd: str) -> None:
        d = self.drone
        if cmd == "help":
            self.log("takeoff: take off and lock onto the person in front | land: land and stop following | "
                     "e: EMERGENCY motors off, quits | status | quit: land and close", "info")
        elif cmd == "status":
            s = self.snap
            bat = f"{s.battery:.0f}%" if s.battery is not None else "?"
            h = f"{s.height_m:.2f} m" if s.height_m is not None else "?"
            self.log(f"{self.state}, battery {bat}, height {h}, video {s.video_fps:.0f} fps, {s.follow}", "info")
        elif cmd == "takeoff":
            self._takeoff()
        elif cmd in ("land", "quit"):
            if self.follow is not None:
                self.follow.cancel("land command")
            if d.flying or d.busy():
                d.land()
                self._pending = "land"
                self._set("landing")
            elif cmd == "land":
                self.log("already on the ground", "info")

    def _takeoff(self) -> None:
        d, f = self.drone, self.follow
        why = None
        if self.state not in ("idle", "landed"):
            why = f"busy ({self.state})"
        elif not f.ready:
            why = f.error or "the person detector is still loading"
        elif self.last_frame is None or time.time() - self.last_frame.t > 1.0:
            why = "no live video"
        elif getattr(d, "land_requested", False):
            why = "it already landed in this run: restart the app to fly again"
        else:
            bat = d.telemetry().battery_pct
            if bat is not None and bat <= d.cfg.min_battery_pct:
                why = f"battery {bat:.0f}%"
        if why:
            self.log(f"takeoff refused: {why}", "warn")
            return
        d.takeoff()
        self._pending = "takeoff"
        self._set("taking off")

    def _tick(self) -> None:
        d, f = self.drone, self.follow
        tel = d.telemetry()
        self.snap.battery, self.snap.height_m = tel.battery_pct, tel.height_m
        self.snap.link_error = getattr(getattr(d, "inner", d), "link_error", None)
        self.snap.det_fps = f.det_fps
        self.snap.follow = f.status()
        last_t = self.last_frame.t if self.last_frame is not None else None
        action = d.check(time.time(), last_t)
        busy = d.busy()  # also what completes takeoff / land: call it every tick
        if self._pending is not None and not busy:
            res = d.last_result()
            self.log(f"drone: {self._pending} -> {res}", "recv" if res == "ok" else "warn")
            if self._pending == "takeoff":
                if d.flying:
                    self._set("locking")
                    self._lock_t0 = self._lock_said = time.monotonic()
                    self.log("looking for the person in front ...", "info")
                else:
                    self._set("idle")
            elif self._pending == "land" and not d.flying:
                self._set("landed")
                self.log("landed. Restart the app to fly again", "info")
            self._pending = None
        if self.state == "landing" and self._pending is None and not d.flying and not busy:
            self._set("landed")  # e.g. the first land was not sent (Wi-Fi) and the governor's re-send landed it
            self.log("landed. Restart the app to fly again", "info")
        if action == "land" and self.state not in ("landing", "landed"):
            f.cancel("safety landing")
            self._set("landing")
            self._pending = "land"
            self.log("safety governor is landing (battery, flight time, stale data or a land request)", "warn")
        if self.state == "locking" and d.flying and not busy:
            why = f.start(d)
            if why is None:
                self._set("following")
            elif time.monotonic() - self._lock_said > 3.0:
                self._lock_said = time.monotonic()
                waited = time.monotonic() - self._lock_t0
                self.log(f"no lock yet: {why}" + (" - stand about 1 m in front of the drone, facing away"
                                                   if waited > LOCK_TIMEOUT_S else ""), "warn")
        if self.state == "following" and not f.active:
            self._set("locking")  # lost or cancelled by a safety action: re-lock by itself
            self._lock_t0 = self._lock_said = time.monotonic()
        if d.flying and not busy:
            vals = f.tick(d, action, busy) if (self.state == "following" and f.active) else (0, 0, 0, 0)
            d.rc(*vals)  # every tick: the follow sticks, or hover; also the keep-alive against the 15 s auto-land

    def _close(self) -> None:
        if self.follow is not None:
            self.follow.close()
        if self.drone is not None:
            try:
                self.drone.close()  # lands if still flying (never in dry run), stops the video
            except Exception as e:
                self.log(f"closing the drone: {e}", "warn")
