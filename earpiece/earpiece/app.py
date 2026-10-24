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
from .events import Msg, clock_phrase, distance_phrase
from .listen import Listener
from .memory import MemoryStore
from .mixer import Engine
from .output import DeviceOutput, NullOutput, RustDeviceOutput
from .route import RouteGuide
from .sonar import Sonar
from .speech import Speaker
from .transport import Transport

# Inbound types handle() mirrors itself; _event() skips them to avoid double rows.
_MIRRORED_INBOUND = {"guidance", "hazard"}


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
        self.sonar = Sonar(self.engine, self.cfg.sonar, speak=self.say, log=log)
        self.beacon = Beacon(self.engine, self.cfg.beacon, speak=self.say, log=log)
        self.listener = Listener(self.cfg.listen, self.cfg.cartesia_key, on_text=self.heard,
                                 cue=self.cue, on_listening=self._on_listening, say=self.say, log=log)
        self.memory = memory if memory is not None else MemoryStore(log=log)
        self.brain = Brain(self.context, Actions(self.navigate, self.control, self.alert_caregiver,
                                                 self.stop_speaking, self.ear_check,
                                                 remember=self.remember, recall=self.recall),
                           self.say, self.cfg.anthropic_key, log=log)
        self._noted_places: set[str] = set()
        self._walk_pending: dict[str, tuple[float, Msg]] = {}
        self.gestures = Gestures(self.gesture, log=log)
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
                self.say("AirPods are not connected. Using the laptop speaker.", "system", ttl=20)

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
        return self.speaker.say(text, priority, key=key, ttl=ttl, pre_sound=pre_sound)

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
        if kind not in _MIRRORED_INBOUND:          # those already went out in _mirror()
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
        if self.bridges:
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
                     pre_sound=spatial.render(sounds.nav_chime(), msg.bearing or 0.0, self.engine.rate)
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
            self.scene = (msg.text, now)
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
            where = f" {clock_phrase(msg.bearing)}" if msg.bearing is not None else ""
            dist = f", {distance_phrase(msg.distance)}" if msg.distance is not None else ""
            text = f"{(msg.kind or 'hazard').replace('_', ' ').capitalize()}{where}{dist}."
        self.hazards[key] = {"kind": msg.kind, "state": msg.state, "text": text, "bearing": msg.bearing,
                             "distance": msg.distance, "until": now + ttl, "t": now}
        if changed and text:
            self.say(text, msg.priority, key=f"hazard:{key}:{msg.state}", ttl=min(ttl, 8.0))
        self._event("hazard", {"key": key, "state": msg.state, "text": text})

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
        self.speaker.set_listening(on)

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
        place = self.memory.resolve_place(destination)
        if place is not None:
            name = place.meta.get("name") or place.text.split(":")[0]
            self._event("voice_request", {"intent": "navigate", "destination": destination,
                                          "target_type": "memory", "name": name,
                                          "address": place.meta.get("address"), "lat": place.meta.get("lat"),
                                          "lng": place.meta.get("lng"), "memory_id": place.id})
            if place.meta.get("lat") is not None and place.meta.get("lng") is not None:
                self.route.start((place.meta["lat"], place.meta["lng"]), name)
            return f"Okay, heading to {name}."
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

    def remember(self, kind: str, text: str) -> str:
        text = text.strip().rstrip(".")
        if not text:
            return "What should I remember?"
        meta = {}
        if kind == "place":
            pos = self.position or {}
            meta = {"name": text, "lat": pos.get("lat"), "lng": pos.get("lng"), "near": pos.get("place")}
            body = f"{text}" + (f", near {pos['place']}" if pos.get("place") else "")
            m = self.memory.remember("place", body, meta)
            reply = f"Saved. I'll remember this place as {text}."
            if pos.get("lat") is None:
                reply += " I don't have your exact location yet, so I saved the name only."
        else:
            m = self.memory.remember(kind, text, {"near": (self.position or {}).get("place")})
            reply = "Got it. I'll remember that."
        self._event("memory", {"id": m.id, "kind": m.kind, "text": m.text, "source": "voice"})
        return reply

    def recall(self, query: str) -> list[dict]:
        return [{"kind": h.kind, "text": h.text, "score": round(h.score, 3)}
                for h in self.memory.search(query, top_k=3)]

    def _place_notes(self, place: str) -> None:
        for h in self.memory.notes_for(place)[:1]:
            self.say(f"Heads up: {h.text}", "hazard", key=f"memory:{h.id}", ttl=20)

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
        if is_airpods:
            self.say("AirPods connected.", "system", key="output", ttl=10)
        else:
            self.say("AirPods disconnected. Using the laptop speaker.", "hazard", key="output", ttl=10)

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
            "hazards": [{k: v for k, v in h.items() if k in ("kind", "state", "text", "bearing", "distance")}
                        for h in self.hazards.values()],
            "route": {"label": b.label, "bearing": b.bearing, "distance": b.distance,
                      "region": b.region} if b.active else None,
            "directions": self.route.summary(),
            "last_instruction": None if self.last_guidance is None else {
                "text": self.last_guidance[0], "seconds_ago": round(now - self.last_guidance[1])},
            "scene": self.scene[0] if self.scene and now - self.scene[1] < 90 else None,
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
                       **{k: v for k, v in self.speaker.stats.items() if k != "first_audio_ms"},
                       "first_audio_ms": (sorted(self.speaker.stats["first_audio_ms"])[
                           len(self.speaker.stats["first_audio_ms"]) // 2]
                           if self.speaker.stats["first_audio_ms"] else None)},
            "listen": {"active": self.listener.active, "available": self.listener.available,
                       "mic": self.listener.device_name, **self.listener.stats},
            "brain": {"online": self.brain.online, **{k: v for k, v in self.brain.stats.items()
                                                      if k != "first_sentence_ms"}},
            "net": dict(self.transport.stats) if self.transport else None,
            "battery": self.battery.last if out else None,
            "memory": self.memory.summary(),
            "uptime_s": round(time.time() - self.started_at),
        }
