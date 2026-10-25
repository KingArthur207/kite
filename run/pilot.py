"""The drone side of the app: one control thread that owns the Tello, the person perception and the follow mission.

Commands arrive from the console (`send()`, any thread). The control thread runs at 20 Hz through kite's drone layer:
the newest frame goes through perception (pose YOLO, tracker, person lock, range), then SafetyGovernor.check(), then
one mission step (kite.follow_behind.FollowMission), which sends the sticks or a discrete command.

    idle --takeoff--> taking off --ok--> climbing --at follow height--> following
                                         searching (person lost 20 s: down to look height, turn 30 deg at a time)
                                           --found--> following   --not found--> holding --follow--> searching
    any --land--> landing --> landed
    e (emergency()) cuts the motors at once from the caller's thread; the app then quits.

Following (kite/follow_behind.py): the largest person in view is locked; the drone holds --height above the floor,
turns toward them, holds --behind behind them (moving toward them only when every range cue says it is safe), slides
around behind them when they turn, and backs off when they come closer. Lost: hold, turn toward where they were.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

TICK_S = 0.05  # 20 Hz control loop
STATES = {"CLIMB": "climbing", "FOLLOW": "following", "REACQUIRE": "searching", "HOLD": "holding"}


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
    """Wraps the person detector to remember the last frame's boxes for the camera overlay."""

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
        self.follow = None  # kite.follow_behind.FollowMission
        self.seen: SeenDetector | None = None
        self._cmds: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pilot", daemon=True)
        self.state = "starting"
        self._pending = None  # discrete command we wait on: "takeoff" | "land"
        self._ready_line = ""
        self.last_frame = None  # newest kite Frame (BGR), for the UI
        self._fps_n, self._fps_t = 0, time.time()
        self._det_n, self._det_seq = 0, None
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
        elif cmd in ("takeoff", "land", "follow", "status", "help", "quit"):
            self._cmds.put(cmd)
        else:
            self.log(f"unknown command {text.strip()!r}: takeoff | land | follow | e (emergency) | status | quit",
                     "warn")

    def emergency(self) -> None:
        """Motors off immediately, from the caller's thread: the drone falls if it is in the air."""
        d = self.drone
        if d is None:
            self._stop.set()
            self._set("emergency")
            self.log("EMERGENCY: no drone connected", "error")
            return
        if d.emergency() is False:  # never reached the drone: keep the control loop (and the governor) running
            self.log("EMERGENCY NOT SENT (Wi-Fi down): type e again", "error")
            return
        self._stop.set()  # the control loop sends nothing more (a step already running finishes on its own)
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
        from kite.follow import FollowCfg, build_yaw_source
        from kite.follow_behind import BehindCfg, CameraModel, FollowMission, Perception, PoseDetector, pose_available

        a = self.args
        why = pose_available(a.weights)
        if why:
            self.log(f"person detector unavailable: {why}", "error")
            return False
        cfg = Config()
        if a.video_backend:
            cfg.drone.video_backend = resolve_video_backend(a.video_backend)
        bcfg = BehindCfg(distance_m=a.behind, altitude_m=a.height, person_height_m=a.walker_height,
                         min_altitude_m=cfg.safety.min_altitude_m, max_altitude_m=cfg.safety.max_altitude_m)
        try:
            bcfg.check()
        except ValueError as e:
            self.log(f"follow settings: {e}", "error")
            return False
        yaw_source = None
        if a.yaw != "law":
            try:
                yaw_source, line = build_yaw_source(a.yaw, FollowCfg(behind_m=a.behind,
                                                                     video_lag_s=cfg.drone.video_lag_s))
                self.log(line, "info")
            except Exception as e:
                self.log(f"steering '{a.yaw}' unavailable ({e}); using the follow law", "warn")
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
        self.log("loading the person detector (YOLO pose) ...", "info")
        try:
            self.seen = SeenDetector(PoseDetector(a.weights))
        except Exception as e:
            self.log(f"person detector failed to load: {e}", "error")
            return False
        perception = Perception(bcfg, self.seen, CameraModel.from_camera(load_camera()))
        self.follow = FollowMission(self.drone, perception, bcfg, log=self.log, yaw_source=yaw_source)
        self._ready_line = (f"ready: video ({cfg.drone.video_backend} reader), person detector on "
                            f"{self.seen._detect.device}. Stand 1-2 m in front of the drone facing away, then type "
                            "takeoff")
        self.log("waiting for the first video frame ...", "info")
        return True

    def _run(self) -> None:
        try:
            if not self._connect():
                self._set("offline")
                return
            self._set("idle")
            next_tick = time.monotonic()
            while not self._stop.is_set():
                while True:
                    try:
                        self._command(self._cmds.get_nowait())
                    except queue.Empty:
                        break
                now = time.monotonic()
                if now >= next_tick:
                    next_tick = now + TICK_S
                    self._tick()
                time.sleep(0.002)
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

    def _read_frame(self):
        f = self.frames.read() if self.frames is not None else None
        if f is not None and (self.last_frame is None or f.seq != self.last_frame.seq):
            if self.last_frame is None and self._ready_line:
                self.log(self._ready_line, "info")  # only now: a takeoff before the first frame is refused
            self.last_frame = f
            self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self.snap.video_fps = self._fps_n / (now - self._fps_t)
            self.snap.det_fps = self._det_n / (now - self._fps_t)
            self._fps_n, self._det_n, self._fps_t = 0, 0, now
        return f

    def _command(self, cmd: str) -> None:
        d, m = self.drone, self.follow
        if cmd == "help":
            self.log("takeoff: take off, climb, follow the person in front | land: stop and land | follow: look for "
                     "the person again | e: EMERGENCY motors off, quits | status | quit: land and close", "info")
        elif cmd == "status":
            s = self.snap
            bat = f"{s.battery:.0f}%" if s.battery is not None else "?"
            h = f"{s.height_m:.2f} m" if s.height_m is not None else "?"
            self.log(f"{self.state}, battery {bat}, height {h}, video {s.video_fps:.0f} fps, "
                     f"detector {s.det_fps:.0f}/s, {s.follow}", "info")
        elif cmd == "takeoff":
            self._takeoff()
        elif cmd == "follow":
            if not (m is not None and m.reacquire("asked to follow")):
                self.log("follow: only while hovering after 'person not found', or once the person has been out of "
                         "view for 2 s", "warn")
        elif cmd in ("land", "quit"):
            if m is not None:
                m.stop()
            if d is not None and (d.flying or d.busy()):
                d.land()
                self._pending = "land"
                self._set("landing")
            elif cmd == "land":
                self.log("already on the ground", "info")

    def _takeoff(self) -> None:
        d = self.drone
        why = None
        if self.state not in ("idle", "landed"):
            why = f"busy ({self.state})"
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
        d, m = self.drone, self.follow
        f = self._read_frame()
        m.sense(f, time.time())  # perception on every NEW frame, flying or not (the overlay shows the lock)
        res = m.ctx.res
        if res is not None and res.seq != self._det_seq:
            self._det_seq, self._det_n = res.seq, self._det_n + 1
        tel = m.ctx.tel
        self.snap.battery, self.snap.height_m = tel.battery_pct, tel.floor_altitude_m()
        self.snap.link_error = getattr(getattr(d, "inner", d), "link_error", None)
        self.snap.follow = m.status
        last_t = self.last_frame.t if self.last_frame is not None else None
        action = d.check(time.time(), last_t)
        if self._pending is not None:
            if d.busy():  # also what completes takeoff / land: polled every tick while waiting
                return
            res_ = d.last_result()
            self.log(f"drone: {self._pending} -> {res_}", "recv" if res_ == "ok" else "warn")
            if self._pending == "takeoff":
                if d.flying:
                    m.begin()  # climb to the follow height, then follow
                else:
                    self._set("idle")
            elif self._pending == "land" and not d.flying:
                self._set("landed")
                self.log("landed. Restart the app to fly again", "info")
            self._pending = None
        if self.state == "landing" and not d.flying and not d.busy():
            self._set("landed")  # e.g. the first land was not sent (Wi-Fi) and the governor's re-send landed it
            self.log("landed. Restart the app to fly again", "info")
        if action == "land" and self.state not in ("landing", "landed"):
            m.stop()
            self._set("landing")
            self._pending = "land"
            self.log("safety governor is landing (battery, flight time, stale data or a land request)", "warn")
            return
        if m.state in STATES and d.flying and not self._stop.is_set():
            m.step()  # the sticks every tick (also the keep-alive), or one discrete command it waits on
            self._set(STATES.get(m.state, self.state))

    def _close(self) -> None:
        if self.drone is not None:
            try:
                self.drone.close()  # lands if still flying (never in dry run), stops the video
            except Exception as e:
                self.log(f"closing the drone: {e}", "warn")
