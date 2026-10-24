"""The earpiece: wires the mixer, speech, sonar, beacon, voice input, controls and
network together, and owns the safety policy between them.

Safety policy in one place:
  * the sonar never stops for anything except stale perception (then it says so);
  * "pause" and "mute" silence guidance and gentle beeps, never critical warnings;
  * critical speech preempts everything, including an answer mid-sentence;
  * navigation speech that could not be said in time is dropped, not said late;
  * losing the AirPods, the internet, or the camera is announced once, and the
    system keeps working in a degraded mode instead of failing silently.
"""

from __future__ import annotations

import dataclasses
import re
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from . import devices, sounds, spatial
from .beacon import Beacon
from .brain import Actions, Brain
from .config import Cfg
from .controls import PAUSE, REPEAT, TALK, Gestures
from .events import PRIORITY_RANK, Msg, clock_phrase, distance_phrase
from .camera import BodyCamera
from .judgment import RECENT_SPEECH_S, Judgment
from .vision import Vision
from .listen import Listener
from .memory import MemoryStore, name_of, pick
from .mixer import Engine
from .output import DeviceOutput, NullOutput, RustDeviceOutput
from .route import RouteGuide
from . import sonar as sonar_mod
from .speech import Speaker
from .transport import Transport

# Inbound types handle() mirrors itself; _event() skips them to avoid double rows.
_MIRRORED_INBOUND = {"guidance", "hazard"}
# Internal bookkeeping events: shown to WebSocket clients, not mirrored (what was SAID about
# them is already mirrored as `spoken`, and the Spacetime module does not accept these types).
_LOCAL_ONLY = {"judgment", "continue", "read_text"}


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.repeat(mono[:, None], 2, axis=1)


class Earpiece:
    def __init__(self, cfg: Cfg | None = None, output: str = "device", record: bool = False,
                 network: bool = True, log: Callable[[str], None] = print, output_device: str | None = None,
                 memory: MemoryStore | None = None):
        self.cfg = cfg or Cfg()
        self.log = log
        a = self.cfg.audio
        from . import rust_engine
        use_rust = a.engine == "rust" or (a.engine == "auto" and rust_engine.available())
        if a.engine == "rust" and not rust_engine.available():
            raise RuntimeError("audio.engine='rust' but kite_audio is not built "
                               "(maturin develop --release -m native/Cargo.toml)")
        E = rust_engine.RustEngine if use_rust else Engine
        self.engine = E(a.sample_rate, a.master_gain, a.limiter_ceiling, a.cue_gain, a.speech_gain,
                        a.duck_cues_during_speech)
        log(f"[audio] engine: {'rust (kite_audio)' if use_rust else 'python'}")
        if output == "device" and use_rust:
            self.output = RustDeviceOutput(self.engine, a.output_substr, a.rust_block, a.device_poll_s,
                                           on_change=self._output_changed, log=log, device=output_device)
        elif output == "device":
            self.output = DeviceOutput(self.engine, a.output_substr, a.block, a.device_poll_s,
                                       on_change=self._output_changed, log=log, device=output_device)
        else:
            self.output = NullOutput(self.engine, a.block, realtime=(output == "null"), record=record)
        self.speaker = Speaker(self.engine, self.cfg.speech, self.cfg.cartesia_key, log=log)
        self.speaker.on_spoken = self._spoken
        self._lock = threading.RLock()
        self.sonar = sonar_mod.create(self.engine, self.cfg.sonar, speak=self.say, log=log)
        log(f"[sonar] engine: {'rust (kite_audio)' if getattr(self.sonar, 'native', False) else 'python'}")
        self.beacon = Beacon(self.engine, self.cfg.beacon, speak=self.say, log=log)
        self.listener = Listener(self.cfg.listen, self.cfg.cartesia_key, on_text=self.heard,
                                 cue=self.cue, on_listening=self._on_listening, say=self.say, log=log)
        self.memory = memory if memory is not None else MemoryStore(log=log)
        self.brain = Brain(self.context, Actions(self.navigate, self.control, self.alert_caregiver,
                                                 self.stop_speaking, self.ear_check,
                                                 remember=self.remember, recall=self.recall,
                                                 forget=self.forget, rename=self.rename_memory,
                                                 undo=self.undo_forget, clarify=self.clarify,
                                                 describe_front=lambda: self.describe_front(),
                                                 read_text=lambda: self.read_text()),
                           self.say, self.cfg.anthropic_key, log=log)
        self.listener.turns = self.speaker.turns          # end-of-speech -> reply latency, by stage
        self.listener.on_partial = self.brain.prefetch    # Moss recall starts while the wearer talks
        self.judgment = Judgment(self.context, self.say, self.upgrade_hazard, self._event,
                                 api_key=self.cfg.anthropic_key if self.cfg.judgment else None, log=log)
        self._noted_places: set[str] = set()
        self._asking: dict | None = None           # "which bench?": the candidates, until an answer
        self._undo: tuple[object, float] | None = None   # last forgotten memory, for "undo"
        self._walk_pending: dict[str, tuple[float, Msg]] = {}
        self.vision = Vision(self.cfg.anthropic_key, log=log)
        self._stop_since: float | None = None      # a "Stop."/wait is in force since this time
        self._clear_since: float | None = None     # the way has looked clear since this time
        self._last_scene_image = 0.0
        self._chest_frame: tuple[bytes, float] | None = None   # newest chest-camera JPEG
        self.camera = BodyCamera(log=log)
        self.gestures = Gestures(self.gesture, log=log)
        self.gestures.on_press = self.barge_in            # cut at the press, not after classifying it
        self.transport = Transport(self.handle, self.cfg.net.bind_host, self.cfg.net.udp_port,
                                   self.cfg.net.ws_port, log=log) if network else None
        if self.transport is not None:
            self.transport.state_provider = self.state
        self.bridges: list = []                    # extra sinks (e.g. SpacetimeDB) for outbound events
        self.route = RouteGuide(self.handle, event=self._event, log=log)   # cues re-enter handle()
        self.battery = devices.BatteryWatch(lambda t: self.say(t, "system", key="airpods-battery", ttl=30))
        self._stop = threading.Event()
        self.paused = False
        self.scene: tuple[str, float] | None = None
        self.position: dict | None = None
        self.drone: dict = {}
        self.hazards: dict[str, dict] = {}
        self.last_guidance: tuple[str, float] | None = None
        self.perception_sources: set[str] = set()
        self.last_perception = 0.0
        self.perception_ok: bool | None = None     # None = never seen yet
        self.events: deque = deque(maxlen=200)     # recent activity for the dashboard
        self._drone_batt_warned: set[int] = set()
        self.started_at = time.time()

    # ================================================================== lifecycle
    def start(self, prewarm: bool = True, announce: bool = True) -> None:
        self.output.start()
        self.speaker.start()
        if prewarm:
            self.speaker.prewarm()
            threading.Thread(target=self.prewarm_cues, name="cue-prewarm", daemon=True).start()
        if self.transport is not None:
            self.transport.start()
        self.memory.start()
        threading.Thread(target=self._control_loop, name="control", daemon=True).start()
        if isinstance(self.output, (DeviceOutput, RustDeviceOutput)):
            devices.on_device_change(lambda: None)    # keeps CoreAudio notifications flowing
            self.battery.start()
        if announce:
            self.engine.play(_stereo(sounds.ready()), bus="cue", gain=1.0)
            self.say("Kite is ready.", "system", ttl=20)
            if isinstance(self.output, (DeviceOutput, RustDeviceOutput)) and not self.output.is_airpods:
                self.say("Headphones are not connected. Using the laptop speaker.", "system", ttl=20)

    def prewarm_cues(self) -> int:
        """Render the critical sonar cues and every beacon tick at all 5-degree bearings
        ahead of time (batched in Rust when kite_audio is built), off the control loop."""
        from . import beacon as beacon_mod, sonar as sonar_mod
        t0 = time.perf_counter()
        try:
            n = sonar_mod.prewarm(self.cfg.sonar, self.engine.rate)
            n += beacon_mod.prewarm(self.cfg.beacon, self.engine.rate)
        except Exception as e:                 # a warm cache is an optimisation, never a failure
            self.log(f"[audio] cue pre-warm failed: {e!r}")
            return 0
        self.log(f"[audio] pre-rendered {n} spatial cues in {(time.perf_counter() - t0) * 1e3:.0f} ms "
                 f"({'rust' if spatial.NATIVE else 'python'} binaural)")
        return n

    def run_forever(self) -> None:
        """Blocks the main thread (it pumps the macOS run loop for AirPods stem presses)."""
        try:
            self.gestures.run_main_loop(self._stop)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        if self._stop.is_set() and getattr(self, "_stopped", False):
            return
        self._stop.set()
        self._stopped = True
        self.judgment.stop()
        self.listener.stop()
        self.battery.stop()
        self.speaker.cancel_all(keep_critical=False)
        time.sleep(0.1)
        self.speaker.stop()
        self.output.stop()
        if hasattr(self.engine, "shutdown"):
            self.engine.shutdown()

    # ================================================================== speaking
    def say(self, text: str, priority: str = "info", key: str | None = None, ttl: float | None = None,
            pre_sound: np.ndarray | None = None) -> str | None:
        if priority == "critical" and text.startswith("Stop"):
            self._begin_wait()
        return self.speaker.say(text, priority, key=key, ttl=ttl, pre_sound=pre_sound)

    # ================================================================== stop and wait
    CLEAR_HOLD_S = 1.5      # the way must look clear this long before "you can continue"
    WAIT_GIVE_UP_S = 60.0   # after this, stop tracking the episode (no late "continue")
    JUDGMENT_HOLD_S = 6.0   # no "you can continue" this soon after Claude warned about something sonar can't see

    def _begin_wait(self) -> None:
        self._stop_since = time.monotonic()
        self._clear_since = None

    def _way_blocked(self) -> bool:
        if self.judgment.last_said_age() < self.JUDGMENT_HOLD_S:     # Claude may veto a clear, never grant one
            return True
        with self._lock:
            tracks = list(self.sonar.tracks.values())
        if any(t.critical or t.urgency > 0.45 for t in tracks):
            return True
        return any(h.get("state") in ("wait", "blocked", "stop") and h.get("kind") != "crosswalk"
                   for h in self.hazards.values())

    def _check_continue(self, now: float) -> None:
        """Guide-dog behaviour from the spec: stop, wait until the way is clear, then say so.
        Deterministic code is the only thing allowed to clear a stop."""
        if self._stop_since is None:
            return
        if now - self._stop_since > self.WAIT_GIVE_UP_S or self.perception_ok is False:
            self._stop_since = self._clear_since = None     # never announce "clear" on stale data
            return
        if now - self._stop_since < 1.0 or self._way_blocked():
            self._clear_since = None
            return
        if self._clear_since is None:
            self._clear_since = now
        elif now - self._clear_since >= self.CLEAR_HOLD_S:
            self._stop_since = self._clear_since = None
            self.speaker.say("Path is clear. You can continue.", "hazard", key="continue", ttl=3)
            self._event("continue", {})

    def cue(self, name: str) -> None:
        snd = {"listening": sounds.listening, "heard": sounds.heard, "error": sounds.error}.get(name)
        if snd is not None:
            self.engine.play(_stereo(snd()), bus="alert", gain=0.8)

    def _spoken(self, rec: dict) -> None:
        self._event("spoken", rec)

    def _event(self, kind: str, data: dict) -> None:
        obj = {"type": kind, "t": time.time(), **data}
        self.events.append(obj)
        if self.transport is not None:
            self.transport.publish(obj)
        if kind not in _MIRRORED_INBOUND and kind not in _LOCAL_ONLY:   # inbound went out in _mirror()
            self._bridge(obj)

    def _bridge(self, obj: dict) -> None:
        for b in self.bridges:
            try:
                b.publish(obj)
            except Exception as e:
                self.log(f"[bridge] {type(e).__name__}: {e}")

    def _mirror(self, msg: Msg) -> None:
        """Every validated inbound event goes to the bridges (SpacetimeDB) as-is."""
        d = dataclasses.asdict(msg)
        d.update(d.pop("data"))                   # position/status fields back at the top level
        self._bridge({k: v for k, v in d.items() if v not in (None, "", [], {})})

    # ================================================================== inbound events
    def handle(self, msg: Msg) -> None:
        # caregiver rows are already in Spacetime; a raw JPEG stays local (its description is
        # mirrored as a `scene` event instead)
        if self.bridges and msg.source != "caregiver" and msg.type != "scene_image":
            self._mirror(msg)
        now = time.monotonic()
        t = msg.type
        if t in ("obstacles", "obstacle"):
            self.perception_sources.add(msg.source)
            self.last_perception = now
            with self._lock:
                if t == "obstacles":
                    self.sonar.update_frame(msg.source, msg.objects, msg.seq, now)
                else:
                    for o in msg.objects:
                        self.sonar.update_one(msg.source, o, msg.ttl, now)
        elif t == "heartbeat":
            if msg.source in self.perception_sources:
                self.last_perception = now
        elif t == "guidance":
            self.last_guidance = (msg.text, now)
            self.say(msg.text, msg.priority, key=msg.key, ttl=msg.ttl,
                     pre_sound=spatial.render(sounds.nav_chime(msg.text), msg.bearing or 0.0, self.engine.rate)
                     if msg.priority == "nav" else None)
            self._event("guidance", {"text": msg.text})
        elif t == "beacon":
            if msg.active:
                self.beacon.set(msg.bearing, msg.distance, msg.label, now)
            else:
                self.beacon.clear()
        elif t == "hazard":
            self._hazard(msg, now)
        elif t == "say":
            self.say(msg.text, msg.priority, key=msg.key, ttl=msg.ttl)
        elif t == "scene":
            fresh = self.scene is None or self.scene[0] != msg.text
            self.scene = (msg.text, now)
            if fresh:
                self.judgment.notify()
        elif t == "scene_image":
            self._scene_image(msg, now)
        elif t == "position":
            self.position = {k: v for k, v in msg.data.items() if v is not None}
            self.route.position(self.position, now)
            place = self.position.get("place")
            if place and place not in self._noted_places:
                self._noted_places.add(place)
                threading.Thread(target=self._place_notes, args=(place,), daemon=True).start()
        elif t == "memory":
            meta = {k: v for k, v in msg.data.items() if v is not None}
            m = self.memory.remember(msg.kind, msg.text, meta)
            self._event("memory", {"id": m.id, "kind": m.kind, "text": m.text, "source": msg.source})
        elif t == "status":
            self._status(msg)
        elif t == "control":
            out = self.control(msg.action)
            if out and msg.action not in ("repeat",):
                self.say(out, "system", ttl=5)

    _HAZARD_NAMES = {"crowd": "Crowd", "vehicle": "Car", "overhead": "Overhead obstacle",
                     "overhead_obstacle": "Overhead obstacle", "curb": "Curb", "construction": "Construction",
                     "stairs_down": "Step down", "person": "Person", "bike": "Bike"}

    WALK_CONFIRM_S = 0.6   # "walk" must hold this long before it is announced; "don't walk" is instant

    def _hazard(self, msg: Msg, now: float) -> None:
        key = msg.key or (msg.kind or "hazard")
        if msg.kind == "crosswalk":
            # Asymmetric on purpose: a one-frame "walk" misread must never send someone into
            # traffic, while a "don't walk" is safe to act on at once.
            if msg.state in ("walk", "go", "green"):
                prev = self.hazards.get(key)
                if not (prev and prev.get("state") in ("walk", "go", "green")):
                    p = self._walk_pending.get(key)
                    if p is None:
                        self._walk_pending[key] = (now, msg)
                    else:
                        self._walk_pending[key] = (p[0], msg)
                    return
            else:
                self._walk_pending.pop(key, None)
        self._apply_hazard(key, msg, now)

    def _confirm_walks(self, now: float) -> None:
        for key, (since, msg) in list(self._walk_pending.items()):
            if now - since >= self.WALK_CONFIRM_S:
                del self._walk_pending[key]
                self._apply_hazard(key, msg, now)

    def _apply_hazard(self, key: str, msg: Msg, now: float) -> None:
        prev = self.hazards.get(key)
        changed = prev is None or prev.get("state") != msg.state or prev.get("text") != msg.text
        ttl = msg.ttl or 20.0
        text = msg.text
        if msg.kind == "crosswalk":
            if msg.state in ("walk", "go", "green"):
                text = text or "Walk signal is on."
                snd = sounds.crossing_go()
            elif msg.state in ("dont_walk", "wait", "red", "stop"):
                text = text or "Don't walk."
                snd = sounds.crossing_wait()
            else:
                snd = None
            if changed and snd is not None:
                self.engine.play(spatial.render(snd, msg.bearing or 0.0, self.engine.rate), bus="cue", gain=0.9)
        elif not text:
            name = self._HAZARD_NAMES.get(msg.kind or "", (msg.kind or "hazard").replace("_", " ").capitalize())
            where = clock_phrase(msg.bearing) if msg.bearing is not None else "ahead"
            text = f"{name}, {where}."                        # same form as the cached sonar labels
            if msg.state in ("wait", "blocked", "stop"):
                text += " Wait."
        self.hazards[key] = {"kind": msg.kind, "state": msg.state, "text": text, "bearing": msg.bearing,
                             "distance": msg.distance, "until": now + ttl, "t": now, "priority": msg.priority}
        if changed and msg.state in ("wait", "blocked", "stop") and msg.kind != "crosswalk":
            self._begin_wait()
        if changed and text:
            self.say(text, msg.priority, key=f"hazard:{key}:{msg.state}", ttl=min(ttl, 8.0))
        self._event("hazard", {"key": key, "state": msg.state, "text": text})
        if changed:                                # spoken above first: Claude only ever adds to it
            self.judgment.notify()

    def upgrade_hazard(self, key: str, priority: str) -> bool:
        """Judgment layer hook: raise (never lower) an existing hazard's priority and say it again."""
        h = self.hazards.get(key)
        if h is None or priority not in PRIORITY_RANK or priority == "critical":
            return False
        if PRIORITY_RANK[priority] >= PRIORITY_RANK.get(h.get("priority") or "hazard", len(PRIORITY_RANK)):
            return False
        h["priority"] = priority
        if h.get("text"):
            self.say(h["text"], priority, key=f"hazard:{key}:upgrade", ttl=4.0)
        self._event("hazard", {"key": key, "state": h.get("state"), "text": h.get("text"),
                               "priority": priority, "upgraded_by": "judgment"})
        return True

    # ================================================================== vision (Claude)
    SCENE_IMAGE_EVERY_S = 3.0

    def _scene_image(self, msg: Msg, now: float) -> None:
        """A camera frame. Drone/sim frames: Claude describes them (off the hot path,
        rate-limited) into the shared `scene` context. Chest frames are only kept, to answer
        "what's in front of me?" on demand (the chest camera sees what the drone can't)."""
        if msg.kind == "chest":
            self._chest_frame = (msg.data["jpeg"], now)
            return
        if not self.vision.online or now - self._last_scene_image < self.SCENE_IMAGE_EVERY_S:
            return
        self._last_scene_image = now
        where = {"drone": "flying just above and behind the walker's head",
                 "chest": "worn on the walker's chest, pointing forward",
                 "sim": "flying just above and behind the walker's head (a simulation)"}.get(msg.kind, "near the walker")

        def work():
            text = self.vision.describe_image(msg.data["jpeg"], where)
            if text:
                self.scene = (text, time.monotonic())
                self._event("scene", {"text": text, "source": f"claude:{msg.kind}"})
                self.judgment.notify()
        threading.Thread(target=work, name="scene-image", daemon=True).start()

    CHEST_FRAME_MAX_AGE_S = 3.0

    def _body_frame(self) -> bytes | None:
        """Newest body-camera frame: one streamed by perception in the last few seconds, else a
        single frame grabbed from the local camera (indoor mode, no drone, no perception)."""
        f = self._chest_frame
        if f is not None and time.monotonic() - f[1] <= self.CHEST_FRAME_MAX_AGE_S:
            return f[0]
        return self.camera.capture_jpeg()

    def describe_front(self) -> str | None:
        """Claude looks at the body camera. None if there is no frame or Claude is unreachable
        (the caller then answers from the obstacle tracks)."""
        if not self.vision.online:
            return None
        jpeg = self._body_frame()
        if jpeg is None:
            return None
        return self.vision.describe_image(jpeg, "worn on the walker's chest, pointing forward", timeout=7.0)

    def read_text(self) -> str:
        """Read printed or on-screen text aloud: a menu, a sign, a label, a monitor. The body
        camera frames it, Claude transcribes it, Cartesia reads it sentence by sentence (so
        "stop" or any warning cuts in between sentences)."""
        if not self.vision.online:
            return "Reading text needs the internet right now."
        self.say("Hold it steady. Reading.", "answer", key="reading", ttl=10)
        jpeg = self._body_frame()
        if jpeg is None:
            return "I can't see through the camera. Check that it's connected and allowed."
        text = self.vision.read_text(jpeg)
        if text is None:
            return "Sorry, I couldn't read that."
        if not text:
            return "I can't see any text. Try holding it closer and a little steadier."
        sentences = [x.strip() for x in re.split(r"(?<=[.!?:;])\s+", text) if x.strip()]
        for sentence in sentences[:-1]:
            self.say(sentence, "answer", ttl=120)
        self._event("read_text", {"chars": len(text)})
        return sentences[-1] if sentences else text

    def _status(self, msg: Msg) -> None:
        prev_state = self.drone.get("state")
        self.drone.update({k: v for k, v in msg.data.items() if v is not None})
        self.drone["source"] = msg.source
        b = msg.data.get("battery")
        if b is not None:
            for thr in (15, 30):
                if b <= thr and thr not in self._drone_batt_warned:
                    self._drone_batt_warned.add(thr)
                    self.say(f"Drone battery is at {int(b)} percent.", "system", key="drone-batt", ttl=30)
                    break
        if msg.text and msg.data.get("state") != prev_state:
            self.say(msg.text, "system", key=f"status:{msg.source}", ttl=15)

    # ================================================================== control loop
    def _control_loop(self) -> None:
        period = 0.02
        nxt = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                with self._lock:
                    chosen = self.sonar.tick(now)
                self._watchdog(now)
                self._confirm_walks(now)
                self._check_continue(now)
                warn = any(t.urgency > 0.3 for t in chosen)
                self.beacon.duck = 0.35 if warn else 1.0
                self.beacon.tick(now, speech_playing=self.engine.speaking or self.listener.active)
                for k in [k for k, h in self.hazards.items() if h["until"] < now]:
                    del self.hazards[k]
            except Exception as e:
                self.log(f"[control] {type(e).__name__}: {e}")
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()

    def _watchdog(self, now: float) -> None:
        if not self.perception_sources:
            return
        ok = now - self.last_perception <= self.cfg.watchdog.perception_stale_s
        if ok == self.perception_ok:
            return
        first = self.perception_ok is None
        self.perception_ok = ok
        self.sonar.enabled = ok
        if not ok:
            with self._lock:
                self.sonar.clear()
            self.engine.play(_stereo(sounds.sensor_lost()), bus="alert", gain=0.9)
            self.speaker.drop_where(lambda u: u.key == "perception-up")
            self.say("Obstacle detection paused.", "hazard", key="perception-down", ttl=5)
            self.log("[watchdog] perception feed stale: sonar disabled")
        elif not first and self.cfg.watchdog.announce_recovery:
            self.engine.play(_stereo(sounds.sensor_back()), bus="cue", gain=0.9)
            self.speaker.drop_where(lambda u: u.key == "perception-down")   # stale news: never say it late
            self.say("Obstacle detection is back.", "hazard", key="perception-up", ttl=5)
            self.log("[watchdog] perception feed back")

    # ================================================================== wearer input
    def gesture(self, g: str) -> None:
        self._event("gesture", {"gesture": g})
        if g == TALK:
            self.listener.push_to_talk()
        elif g == REPEAT:
            self.repeat()
        elif g == PAUSE:
            self.control("resume" if self.paused else "pause", speak=True)

    def heard(self, text: str) -> None:
        self._event("heard", {"text": text})
        self.brain.handle(text)

    def _on_listening(self, on: bool) -> None:
        if on:
            self.brain.interrupt()
        self.speaker.set_listening(on)

    def barge_in(self) -> None:
        """The wearer pressed: stop non-safety speech and the answer still being written."""
        self.brain.interrupt()
        self.speaker.barge_in()

    # ================================================================== actions
    def repeat(self) -> None:
        if not self.speaker.repeat_last():
            if self.beacon.active and self.beacon.distance is not None:
                self.say(f"Next waypoint {distance_phrase(self.beacon.distance)}, "
                         f"{clock_phrase(self.beacon.bearing)}.", "nav", ttl=10)
            else:
                self.say("No directions yet.", "system", ttl=5)

    def navigate(self, destination: str) -> str:
        """Work out what the wearer means, then hand the navigation side something routable.
        Saved places (Moss) resolve instantly; relative targets ("the end of the street") go
        on as structured requests; nearby / named places are looked up with Google Places in
        the background while Kite acknowledges right away."""
        from . import destinations as dst
        destination = destination.strip().rstrip(".")
        res = self.memory.resolve(destination)
        if res.ambiguous:
            return self._ask("navigate", res, "Which one")
        if res.place is not None:
            return self._go_to_memory(res.place, destination)
        d = dst.classify(destination)
        if d.kind == "relative":
            self._event("voice_request", d.request())
            return f"Okay, I'll guide you to {d.spoken}."
        pos = self.position or {}
        lat, lng = pos.get("lat"), pos.get("lng")
        if d.kind == "nearby" and lat is None:
            self._event("voice_request", d.request())
            return f"I don't have your location yet, so I've asked for the nearest {d.query}."

        def lookup():
            try:
                r = dst.resolve(d, lat, lng, search=self.places_search)
            except Exception as e:
                self.log(f"[nav] place lookup failed ({type(e).__name__}: {e})")
                self._event("voice_request", d.request())
                self.say(f"I couldn't look that up, so I've asked the route planner for {destination}.",
                         "answer", ttl=15)
                return
            if r.name is None:
                self.say(f"I couldn't find {('a ' + d.query + ' nearby') if d.kind == 'nearby' else destination}.",
                         "answer", ttl=15)
                return
            self._event("voice_request", r.request())
            far = f", about {distance_phrase(r.distance_m)} away" if r.distance_m is not None else ""
            where = f" on {r.address.split(',')[0]}" if r.address else ""
            lead = f"The nearest {d.query} is" if d.kind == "nearby" else "Found"
            self.say(f"{lead} {r.name}{where}{far}. Starting directions.", "answer", ttl=20)
            if r.lat is not None and r.lng is not None:
                self.route.start((r.lat, r.lng), r.name)

        threading.Thread(target=lookup, name="places", daemon=True).start()
        return f"Looking for the nearest {d.query}." if d.kind == "nearby" else f"Looking up {d.query}."

    @staticmethod
    def places_search(query, lat, lng, nearby):
        from . import destinations as dst
        return dst.places_search(query, lat, lng, nearby)

    def _go_to_memory(self, place, said: str) -> str:
        name = name_of(place)
        self._event("voice_request", {"intent": "navigate", "destination": said,
                                      "target_type": "memory", "name": name,
                                      "address": place.meta.get("address"), "lat": place.meta.get("lat"),
                                      "lng": place.meta.get("lng"), "memory_id": place.id})
        if place.meta.get("lat") is not None and place.meta.get("lng") is not None:
            self.route.start((place.meta["lat"], place.meta["lng"]), name)
        return f"Okay, heading to {name}."

    ASK_TTL_S = 20.0        # a "which one?" waits this long for its answer
    UNDO_S = 120.0          # "undo" brings back a forgotten memory for this long

    def _ask(self, action: str, res, lead: str, arg: str | None = None) -> str:
        self._asking = {"action": action, "cands": res.candidates, "arg": arg,
                        "until": time.monotonic() + self.ASK_TTL_S}
        return res.question(lead)

    def clarify(self, text: str) -> str | None:
        """The answer to a pending "which one?", or None when no question is open or this
        doesn't answer it (then the utterance is handled as a fresh request)."""
        a, self._asking = self._asking, None
        if a is None or time.monotonic() > a["until"]:
            return None
        i = pick(a["cands"], text)
        if i is None:
            return None
        if i < 0:
            return "Okay, never mind."
        m = a["cands"][i]
        if a["action"] == "navigate":
            return self._go_to_memory(m, name_of(m))
        if a["action"] == "forget":
            return self._forget(m)
        return self._rename(m, a["arg"] or "")

    def remember(self, kind: str, text: str) -> str:
        text = text.strip().rstrip(".")
        if not text:
            return "What should I remember?"
        pos = self.position or {}
        if kind == "place":
            meta = {"name": text, "lat": pos.get("lat"), "lng": pos.get("lng"), "near": pos.get("place")}
            body = f"{text}" + (f", near {pos['place']}" if pos.get("place") else "")
            m, status = self.memory.save("place", body, meta)
            reply = {"updated": f"Updated. {text} is saved right here now.",
                     "same_name": f"Saved. You have another place called {text}, so I'll ask which one "
                                  f"when you go there."}.get(status, f"Saved. I'll remember this place as {text}.")
            if pos.get("lat") is None:
                reply += " I don't have your exact location yet, so I saved the name only."
        else:
            m, status = self.memory.save(kind, text, {"near": pos.get("place")})
            reply = ("I already had that. I've marked it as still true." if status == "duplicate"
                     else "Got it. I'll remember that.")
        self._event("memory", {"id": m.id, "kind": m.kind, "text": m.text, "source": "voice", "action": status})
        return reply

    def forget(self, what: str) -> str:
        what = what.strip().rstrip(".!?")
        res = self.memory.resolve(what, kinds=None)
        if res.ambiguous:
            return self._ask("forget", res, "Which one should I forget")
        if res.place is None:
            return f"I don't have anything saved called {what}."
        return self._forget(res.place)

    def _forget(self, m) -> str:
        self.memory.forget(m.id)
        self._undo = (m, time.monotonic())
        self._event("memory", {"id": m.id, "kind": m.kind, "text": m.text, "source": "voice",
                               "action": "forgotten"})
        return f"Okay, I've forgotten {name_of(m) if m.kind == 'place' else 'that note'}. Say undo to bring it back."

    def undo_forget(self) -> str:
        u, self._undo = self._undo, None
        if u is None or time.monotonic() - u[1] > self.UNDO_S:
            return "There's nothing to undo."
        m = self.memory.restore(u[0])
        self._event("memory", {"id": m.id, "kind": m.kind, "text": m.text, "source": "voice", "action": "restored"})
        return f"Okay, {name_of(m) if m.kind == 'place' else 'that note'} is back."

    def rename_memory(self, old: str, new: str) -> str:
        new = new.strip().rstrip(".!?")
        res = self.memory.resolve(old)
        if res.ambiguous:
            return self._ask("rename", res, f"Which one should I call {new}", arg=new)
        if res.place is None:
            return f"I don't have a place called {old.strip()}."
        return self._rename(res.place, new)

    def _rename(self, m, new: str) -> str:
        old = name_of(m)
        r = self.memory.rename(m.id, new)
        if r is None:
            return "I can only rename saved places."
        self._event("memory", {"id": r.id, "kind": r.kind, "text": r.text, "source": "voice", "action": "renamed"})
        return f"Okay, {old} is now called {new}."

    def recall(self, query: str) -> list[dict]:
        return [{"kind": h.kind, "text": h.text, "score": round(h.score, 3)}
                for h in self.memory.search(query, top_k=3)]

    def _place_notes(self, place: str) -> None:
        for h in self.memory.notes_for(place)[:1]:
            # a remembered note is context, not an imminent danger: live directions go first;
            # an old one is said more softly and gives up sooner
            text, ttl = self.memory.heads_up(h)
            self.say(text, "info", key=f"memory:{h.id}", ttl=ttl)

    def alert_caregiver(self, message: str) -> str:
        self._event("caregiver_alert", {"message": message, "position": self.position})
        return "I've alerted your caregiver."

    def stop_speaking(self) -> None:
        self.speaker.cancel_all(keep_critical=True)

    def control(self, action: str, speak: bool = False) -> str | None:
        out = None
        if action == "mute_cues":
            self.sonar.muted = True
            out = "Beeps muted. I'll still warn you about anything very close."
        elif action == "unmute_cues":
            self.sonar.muted = False
            out = "Beeps on."
        elif action == "pause":
            self.paused = True
            self.speaker.paused = True
            self.beacon.enabled = False
            self.speaker.drop_where(lambda u: u.priority in ("nav", "info"))
            out = "Guidance paused. Safety warnings stay on."
        elif action == "resume":
            self.paused = False
            self.speaker.paused = False
            self.beacon.enabled = True
            out = "Guidance resumed."
        elif action in ("volume_up", "volume_down"):
            g = self.engine.master_gain * (1.25 if action == "volume_up" else 0.8)
            self.engine.master_gain = float(min(1.0, max(0.15, g)))
            out = "Louder." if action == "volume_up" else "Quieter."
        elif action == "repeat":
            self.repeat()
        elif action == "beacon_on":
            self.beacon.enabled = True
            out = "Beacon on." if self.beacon.active else "Beacon on. There is no route yet."
        elif action == "beacon_off":
            self.beacon.enabled = False
            out = "Beacon off."
        elif action == "cancel_route":
            self.route.cancel()
            self.beacon.clear()
            self._event("voice_request", {"intent": "cancel_route"})
            out = "Route cancelled."
        elif action == "stop_speaking":
            self.stop_speaking()
        elif action in ("speak_faster", "speak_slower", "speak_normal"):
            out = self.speaker.adjust_speed({"speak_faster": 1, "speak_slower": -1}.get(action, 0))
        self._event("control", {"action": action})
        if speak and out:
            self.say(out, "system", ttl=5)
        return out

    def ear_check(self) -> None:
        """'Left ear' from the left, then 'right ear' from the right: catches swapped or
        half-inserted AirPods before a walk."""
        def run():
            self.speaker.wait_idle(5)
            for side, az in (("Left ear.", -90.0), ("Right ear.", 90.0)):
                pcm = self.speaker.cache.get(side)
                if pcm is None:
                    try:
                        from .speech import offline_tts
                        pcm = offline_tts(side, self.engine.rate, self.cfg.speech.offline_voice)
                    except Exception:
                        pcm = sounds.ear_check("left" if az < 0 else "right")
                st = spatial.render(pcm, az, self.engine.rate, extra_ild_db=20.0)
                self.engine.play(st, bus="alert", gain=1.0)
                time.sleep(st.shape[0] / self.engine.rate + 0.35)
        threading.Thread(target=run, name="ear-check", daemon=True).start()

    def _output_changed(self, name: str, is_airpods: bool, prev: str | None) -> None:
        self._event("output", {"device": name, "airpods": is_airpods})
        if prev is None:
            return
        ap = "airpods" in name.lower() or "airpods" in (prev or "").lower()
        if is_airpods:
            self.say("AirPods connected." if ap else "Headphones connected.", "system", key="output", ttl=10)
        else:
            self.say("AirPods disconnected. Using the laptop speaker." if ap else
                     "Headphones disconnected. Using the laptop speaker.", "hazard", key="output", ttl=10)

    # ================================================================== state
    def context(self) -> dict:
        """What Kite knows right now, for the voice brain (and the dashboard)."""
        now = time.monotonic()
        with self._lock:
            nearby = [{k: o[k] for k in ("kind", "bearing", "distance", "urgency")}
                      for o in self.sonar.summary()[:6]]
        b = self.beacon
        return {
            "nearby": nearby,
            "perception_ok": self.perception_ok is not False,
            "hazards": [{"key": key, **{k: v for k, v in h.items()
                                        if k in ("kind", "state", "text", "bearing", "distance", "priority")}}
                        for key, h in list(self.hazards.items())],
            "route": {"label": b.label, "bearing": b.bearing, "distance": b.distance,
                      "region": b.region} if b.active else None,
            "directions": self.route.summary(),
            "last_instruction": None if self.last_guidance is None else {
                "text": self.last_guidance[0], "seconds_ago": round(now - self.last_guidance[1])},
            "scene": self.scene[0] if self.scene and now - self.scene[1] < 90 else None,
            "recent_speech": [h["text"] for h in list(self.speaker.history)
                              if h.get("result") != "dropped" and time.time() - h["t"] < RECENT_SPEECH_S][-4:],
            "position": self.position,
            "drone": self.drone or None,
            "guidance_paused": self.paused,
            "beeps_muted": self.sonar.muted,
            "local_time": time.strftime("%H:%M"),
        }

    def state(self) -> dict:
        out = isinstance(self.output, (DeviceOutput, RustDeviceOutput))
        return {
            "context": self.context(),
            "tracks": self.sonar.summary()[:8],
            "audio": {**self.engine.snapshot_stats(), "device": self.output.device_name,
                      "airpods": self.output.is_airpods, "underflows": self.output.underflows,
                      "master_gain": round(self.engine.master_gain, 2), "speaking": self.engine.speaking},
            "speech": {"online": self.speaker.online, "history": list(self.speaker.history)[-5:],
                       "cartesia_chars_session": self.speaker.meter.session,
                       "cartesia_chars_total": self.speaker.meter.total,
                       **{k: v for k, v in self.speaker.stats.items() if k != "first_audio_ms"},
                       "first_audio_ms": (sorted(self.speaker.stats["first_audio_ms"])[
                           len(self.speaker.stats["first_audio_ms"]) // 2]
                           if self.speaker.stats["first_audio_ms"] else None)},
            "listen": {"active": self.listener.active, "available": self.listener.available,
                       "mic": self.listener.device_name, **self.listener.stats},
            "stop_in_force": self._stop_since is not None,
            "vision": {"online": self.vision.online, **self.vision.stats},
            "brain": {"online": self.brain.online, **{k: v for k, v in self.brain.stats.items()
                                                      if k != "first_sentence_ms"}},
            "judgment": {"online": self.judgment.online, **self.judgment.stats},
            "net": dict(self.transport.stats) if self.transport else None,
            "battery": self.battery.last if out else None,
            "memory": self.memory.summary(),
            "uptime_s": round(time.time() - self.started_at),
        }
