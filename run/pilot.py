"""The drone side of the app: one control thread that owns the Tello, the person perception and the follow mission.

Commands arrive from the console (`send()`, any thread). The control thread runs at 20 Hz through kite's drone layer:
the newest frame goes through perception (pose YOLO, tracker, person lock, range), then SafetyGovernor.check(), then
one mission step (kite.follow_behind.FollowMission), which sends the sticks or a discrete command.

    idle --takeoff--> taking off --ok--> climbing --at follow height--> following
                                         searching (person lost 20 s: down to look height, turn 30 deg at a time)
                                           --found--> following   --not found--> holding --follow / stop--> searching
    following / searching --hold--> holding --resume--> searching (turning at the current height)
    any --land--> landing --> landed
    e (emergency()) cuts the motors at once from the caller's thread; the app then quits.

Following (kite/follow_behind.py): the largest person in view is locked; the drone holds --height above the floor,
turns toward them, holds --behind behind them (moving toward them only when every range cue says it is safe) and backs
off when they come closer (--orbit: also slides around behind them when they turn). Lost: hold, turn toward where they
were. Operator: follow, hold, resume, stop (the console), and what the drone says goes to the console (--speak: aloud).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

TICK_S = 0.05  # 20 Hz control loop


WIN_SAY = ("Add-Type -AssemblyName System.Speech; "
           "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:KITE_SAY)")


def speaker():
    """say(text) without blocking the control loop: macOS `say`, else Windows SAPI via PowerShell, else nothing."""
    import shutil
    import subprocess
    import sys

    if sys.platform == "darwin" and shutil.which("say"):
        return lambda text: subprocess.Popen(["say", text])
    if sys.platform == "win32":
        import os

        def win(text: str) -> None:
            # The text goes in an environment variable, never into the command: PowerShell also ends a quoted
            # string at the curly quotes, so escaping ' alone would let a line run code.
            subprocess.Popen(["powershell", "-NoProfile", "-Command", WIN_SAY],
                             env={**os.environ, "KITE_SAY": text}, creationflags=0x08000000)
        return win
    return None
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
        self.obstacles = None  # run.obstacles.ObstacleView with --yolo: boxes for the camera pane only
        self._cmds: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pilot", daemon=True)
        self.state = "starting"
        self._pending = None  # discrete command we wait on: "takeoff" | "land"
        self._ready_line = ""
        self.last_frame = None  # newest kite Frame (BGR), for the UI
        self._capture = None  # run.capture.SessionCapture: video + JPEGs, off the flight path
        self._capture_warned = False
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
        elif cmd in ("follow me", "come back"):
            self._cmds.put("follow")
        elif cmd == "cancel":
            self._cmds.put("stop")
        elif cmd in ("takeoff", "land", "follow", "stop", "hold", "resume", "status", "help", "quit"):
            self._cmds.put(cmd)
        else:
            self.log(f"unknown command {text.strip()!r}: takeoff | land | follow | stop | hold | resume | "
                     "e (emergency) | status | quit", "warn")

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
        from kite.camera import DEFAULT_PATH as CAMERA_FILE
        from kite.camera import load_camera
        from kite.config import Config, resolve_video_backend
        from kite.drone import SafetyGovernor, make_tello
        from kite.follow import FollowCfg, build_yaw_source
        from kite.follow_behind import (BehindCfg, CameraModel, FlyYaw, FollowMission, Perception, PoseDetector,
                                        pose_available)

        a = self.args
        why = pose_available(a.weights)
        if why:
            self.log(f"person detector unavailable: {why}", "error")
            return False
        cfg = Config()
        if a.video_backend:
            cfg.drone.video_backend = resolve_video_backend(a.video_backend)
        model = getattr(a, "model", None)
        mission_cls = FollowMission
        if model:  # opt in: a trained follow model (kite.follow_tuned): the whole follow's settings, its own steering
            from kite.follow_tuned import TunedFollowMission, app_follow_cfg, model_governor, resolve_model

            try:
                bcfg = app_follow_cfg(model, distance_m=a.behind, altitude_m=a.height, person_height_m=a.walker_height,
                                      min_altitude_m=cfg.safety.min_altitude_m,
                                      max_altitude_m=cfg.safety.max_altitude_m, orbit_gain=0.5 if a.orbit else 0.0,
                                      video_lag_s=cfg.drone.video_lag_s)
            except (OSError, ValueError, TypeError) as e:
                self.log(f"follow model: {e}", "error")
                return False
            mission_cls = TunedFollowMission
            need = model_governor(resolve_model(model)).get("needs_max_rc")
            self.log(f"follow model {bcfg.model_name} (simulation-tested, opt-in): steering {bcfg.steering}, its own "
                     f"distance, speed and height settings" + (f"; --yaw {a.yaw} is not used" if a.yaw != "fly" else "")
                     + (f"; it asks for stick up to {need}, the safety cap {cfg.safety.max_rc} limits it"
                        if need and need > cfg.safety.max_rc else ""), "info")
        else:
            bcfg = BehindCfg(distance_m=a.behind, altitude_m=a.height, person_height_m=a.walker_height,
                             min_altitude_m=cfg.safety.min_altitude_m, max_altitude_m=cfg.safety.max_altitude_m,
                             orbit_gain=0.5 if a.orbit else 0.0, video_lag_s=cfg.drone.video_lag_s,
                             steering="fly" if a.yaw == "fly" else "pid")
        try:
            bcfg.check()
        except ValueError as e:
            self.log(f"follow settings: {e}", "error")
            return False
        yaw_source = None
        if model:
            if bcfg.steering == "fly":  # the model's brain (read from its file): load it now, not in the air
                try:
                    steer = FlyYaw.load(bcfg)
                    self.log(f"steering: fruit fly brain ({steer.stats['arm']}, {Path(bcfg.fly_params_path).name})",
                             "info")
                except Exception as e:
                    bcfg.steering = "pid"
                    self.log(f"fly steering unavailable ({type(e).__name__}: {e}); using the yaw law", "warn")
        elif a.yaw in ("retrained", "pd"):  # kite.fly's own yaw sources (simulation-tested)
            try:
                yaw_source, line = build_yaw_source("fly" if a.yaw == "retrained" else "pd",
                                                    FollowCfg(behind_m=a.behind, video_lag_s=cfg.drone.video_lag_s))
                self.log(line, "info")
            except Exception as e:
                self.log(f"steering '{a.yaw}' unavailable ({e}); using the yaw law", "warn")
        elif a.yaw == "fly":  # the fruit fly brain it flew with: load it now, not on the first follow in the air
            try:
                steer = FlyYaw.load(bcfg)
                self.log(f"steering: fruit fly brain ({steer.stats['arm']}, {Path(bcfg.fly_params_path).name})",
                         "info")
            except Exception as e:
                bcfg.steering = "pid"
                self.log(f"fly steering unavailable ({type(e).__name__}: {e}); using the yaw law", "warn")
        if model:
            self.snap.yaw = f"model {bcfg.model_name}" + ("" if bcfg.steering == "fly" else " (law)")
        else:
            self.snap.yaw = a.yaw if (yaw_source is not None or bcfg.steering == "fly") else "law"
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
        if getattr(a, "yolo", False):
            self.log("loading the obstacle detector (YOLO26n, show only) ...", "info")
            try:
                from obstacles import ObstacleView, load_detector

                self.obstacles = ObstacleView(load_detector()).start()
            except Exception as e:  # the boxes are only shown: flying does not need them
                self.log(f"obstacle detector unavailable: {e}", "warn")
        # a calibration saved by kite.camera wins; else the focal length measured on the Tello (BehindCfg.camera_fx)
        cam = (CameraModel.from_camera(load_camera()) if CAMERA_FILE.is_file() else
               CameraModel(960, 720, bcfg.camera_fx, bcfg.camera_fx, 480.0, 360.0))
        perception = Perception(bcfg, self.seen, cam)
        self.follow = mission_cls(self.drone, perception, bcfg, log=self.log, yaw_source=yaw_source,
                                  say=speaker() if a.speak else None)
        self._ready_line = (f"ready: video ({cfg.drone.video_backend} reader), person detector on "
                            f"{self.seen._detect.device}, camera fx {cam.fx:.0f}. Stand about 2 m in front of the "
                            "drone facing away, then type takeoff")
        from capture import SessionCapture

        self._capture = SessionCapture(Path(__file__).resolve().parent.parent / "recordings").start()
        self.log(f"recording video and frames to {self._capture.dir}", "info")
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
            self._stop_capture()
            self.closed.set()

    def _read_frame(self):
        f = self.frames.read() if self.frames is not None else None
        if f is not None and (self.last_frame is None or f.seq != self.last_frame.seq):
            if self.last_frame is None and self._ready_line:
                self.log(self._ready_line, "info")  # only now: a takeoff before the first frame is refused
                self.follow.announce("Ready. Say or type 'takeoff' to start.")
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
            self.log("takeoff: take off, climb, follow the person in front | land | follow (or follow me): look for "
                     "the person again | hold: pause and hover | resume: look for the person, then follow | stop: "
                     "from hold, look for the person | e: EMERGENCY motors off, quits | status | quit: land, close",
                     "info")
        elif cmd == "status":
            s = self.snap
            bat = f"{s.battery:.0f}%" if s.battery is not None else "?"
            h = f"{s.height_m:.2f} m" if s.height_m is not None else "?"
            self.log(f"{self.state}, battery {bat}, height {h}, video {s.video_fps:.0f} fps, "
                     f"detector {s.det_fps:.0f}/s, {s.follow}", "info")
        elif cmd == "takeoff":
            self._takeoff()
        elif cmd in ("follow", "stop", "hold", "resume"):
            if m.state == "IDLE":
                m.announce("I'm still on the ground. Say 'takeoff' first." if not d.flying else "Give me a second, I'm still taking off.")
            elif cmd == "follow" and not m.follow_me():
                self.log("follow: only while holding, or once the person has been out of view for 2 s", "warn")
            elif cmd == "stop" and not m.cancel():
                self.log("stop: only while holding (it then looks for the person again)", "warn")
            elif cmd == "hold" and not m.hold():
                self.log("hold: only while following or searching", "warn")
            elif cmd == "resume" and not m.resume():
                self.log("resume: only while holding", "warn")
        elif cmd in ("land", "quit"):
            m.stop()
            if d is not None and (d.flying or d.busy()):
                d.land()
                m.announce("Landing.")
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
            if bat is not None and bat < d.cfg.min_battery_pct + 10:  # room for a whole follow before the 20 % land
                why = f"battery {bat:.0f}% (needs {d.cfg.min_battery_pct + 10:.0f}% to start)"
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
        if m.ctx.new_frame:
            self._capture_frame(f)
            if self.obstacles is not None and f is not None:
                self.obstacles.offer(f.image)
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
                    m.announce("Takeoff failed.")
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
            ev = getattr(d, "events", None)
            self.log("safety governor is landing" + (f": {ev[-1][1]}" if ev else ""), "warn")
            m.announce("Landing.")
            return
        if m.state in STATES and d.flying and not self._stop.is_set():
            m.step()  # the sticks every tick (also the keep-alive), or one discrete command it waits on
            self._set(STATES.get(m.state, self.state))

    def _capture_frame(self, frame) -> None:
        """Hand the frame to the writer. A disk problem is logged once and never changes takeoff."""
        cap = self._capture
        if cap is None or frame is None:
            return
        try:
            cap.offer(frame.image, frame.seq, self.seen, self.follow)
        except Exception as e:
            if not self._capture_warned:
                self._capture_warned = True
                self.log(f"recording paused: {e}", "warn")

    def _stop_capture(self) -> None:
        cap, self._capture = self._capture, None
        if cap is None:
            return
        try:
            summary = cap.close()
        except Exception as e:
            self.log(f"recording failed to close: {e}", "warn")
            return
        extra = f", {summary.dropped} dropped" if summary.dropped else ""
        err = f" ({summary.error})" if summary.error else ""
        self.log(f"saved {summary.written} frames{extra} to {summary.directory}{err}", "info")

    def _close(self) -> None:
        if self.obstacles is not None:
            self.obstacles.stop()
        if self.drone is not None:
            try:
                self.drone.close()  # lands if still flying (never in dry run), stops the video
            except Exception as e:
                self.log(f"closing the drone: {e}", "warn")
