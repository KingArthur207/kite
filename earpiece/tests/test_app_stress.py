"""End-to-end: the whole earpiece over real UDP and WebSocket sockets, plus abuse."""

import json
import math
import os
import random
import socket
import threading
import time

import numpy as np
import pytest

from conftest import free_port, wait_for
from earpiece.app import Earpiece
from earpiece.client import EarpieceClient
from earpiece.config import Cfg


def client(app, source="webcam"):
    return EarpieceClient("127.0.0.1", app.cfg.net.udp_port, source=source)


def spoken(app):
    return [h["text"] for h in app.speaker.history]


def test_udp_frames_drive_the_sonar(app):
    c = client(app)
    for i in range(30):
        c.obstacles([{"id": 1, "kind": "person", "bearing": -10, "distance": 1.8}])
        time.sleep(1 / 15)
    assert app.sonar.stats["frames"] >= 25
    assert app.sonar.stats["beeps"] > 0
    assert app.perception_ok is True


def test_guidance_hazards_and_controls_over_udp(app):
    c = client(app, "nav")
    c.guidance("Turn left onto Bay Street.", key="s2")
    c.hazard("crosswalk", state="dont_walk", bearing=0, distance=8)
    assert wait_for(lambda: {"Turn left onto Bay Street.", "Don't walk."} <= set(spoken(app)), 5)
    c.hazard("crosswalk", state="dont_walk", bearing=0, distance=7)        # unchanged state: silent
    c.hazard("crosswalk", state="walk", bearing=0, distance=6)
    assert wait_for(lambda: "Walk signal is on." in spoken(app), 5)
    assert spoken(app).count("Don't walk.") == 1
    c.control("mute_cues")
    assert wait_for(lambda: app.sonar.muted, 2)


def test_perception_watchdog(app):
    c = client(app)
    for _ in range(10):
        c.obstacles([{"id": 1, "kind": "person", "bearing": 0, "distance": 3}])
        time.sleep(0.05)
    assert wait_for(lambda: app.perception_ok is True, 2)
    assert wait_for(lambda: app.perception_ok is False, 3)            # camera "unplugged"
    assert not app.sonar.enabled and app.sonar.tracks == {}
    assert wait_for(lambda: "Obstacle detection paused." in spoken(app), 3)
    for _ in range(5):
        c.obstacles([])
        time.sleep(0.05)
    assert wait_for(lambda: app.perception_ok is True, 2)
    assert wait_for(lambda: "Obstacle detection is back." in spoken(app), 3)


def test_heartbeats_keep_an_empty_scene_alive(app):
    c = client(app)
    c.obstacles([])
    for _ in range(40):
        c.heartbeat()
        time.sleep(0.05)
    assert app.perception_ok is True


def test_websocket_state_and_voice_requests(app):
    from websockets.sync.client import connect
    with connect(f"ws://127.0.0.1:{app.cfg.net.ws_port}") as ws:
        first = json.loads(ws.recv(timeout=2))
        assert first["type"] == "state" and "context" in first and "audio" in first
        app.heard("take me to Crissy Field")
        got = None
        t0 = time.time()
        while time.time() - t0 < 3:
            m = json.loads(ws.recv(timeout=3))
            if m["type"] == "voice_request":
                got = m
                break
        assert got == {**got, "intent": "navigate", "destination": "Crissy Field"}
        ws.send(json.dumps({"type": "say", "text": "Hello from the dashboard", "priority": "info"}))
        assert wait_for(lambda: "Hello from the dashboard" in spoken(app), 4)


def test_gestures(app):
    app.gesture("pause")
    assert app.paused and not app.beacon.enabled
    assert wait_for(lambda: "Guidance paused. Safety warnings stay on." in spoken(app), 3)
    app.gesture("pause")
    assert not app.paused
    app.speaker.say("Turn right.", "nav")
    assert wait_for(lambda: app.speaker.last_nav is not None, 3)
    app.gesture("repeat")
    assert wait_for(lambda: spoken(app).count("Turn right.") == 2, 3)


def test_paused_guidance_is_silent_but_critical_is_not(app):
    app.control("pause")
    c = client(app)
    c.guidance("Turn left now.", ttl=1)
    for i in range(25):
        c.obstacles([{"id": 1, "kind": "pole", "bearing": 0, "distance": 0.5}])
        time.sleep(1 / 15)
    assert wait_for(lambda: any(t.startswith("Stop") for t in spoken(app)), 3)
    time.sleep(1.2)
    assert "Turn left now." not in spoken(app)


def test_caregiver_alert_and_state_are_serialisable(app):
    app.alert_caregiver("I'm lost")
    s = json.dumps(app.state(), default=str)
    assert "context" in s
    assert any(e["type"] == "caregiver_alert" for e in app.events)


def test_ear_check_is_spatial(app):
    app.speaker.cache.put("Left ear.", (0.3 * np.sin(np.arange(9600) / 5)).astype(np.float32))
    app.speaker.cache.put("Right ear.", (0.3 * np.sin(np.arange(9600) / 5)).astype(np.float32))
    n0 = sum(f.shape[0] for f in app.output.frames)
    app.ear_check()
    time.sleep(1.6)
    audio = app.output.recorded()[n0:]
    env = np.abs(audio).max(axis=1)
    on = np.flatnonzero(env > 0.02)
    first = audio[on[0]:on[0] + 9600]
    later = audio[on[-1] - 9600:on[-1]]
    assert np.abs(first[:, 0]).sum() > 3 * np.abs(first[:, 1]).sum()
    assert np.abs(later[:, 1]).sum() > 3 * np.abs(later[:, 0]).sum()


# ---------------------------------------------------------------- stress
def test_flood_of_garbage_and_valid_frames(app):
    """20k datagrams, a third of them malformed, from 3 senders as fast as possible.
    The earpiece must stay up, rate-limit, keep rendering, and keep its audio clean."""
    socks = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(3)]
    addr = ("127.0.0.1", app.cfg.net.udp_port)
    rng = random.Random(1)
    garbage = [b"\x00\xff", b"{", b'{"type":"obstacles","objects":"x"}', b"[]",
               b'{"type":"obstacles","objects":[{"bearing":1e999,"distance":1}]}', os.urandom(300)]
    blocks0 = app.engine.stats["blocks"]
    t0 = time.time()
    for i in range(20000):
        s = socks[i % 3]
        if i % 3 == 0:
            s.sendto(rng.choice(garbage), addr)
        else:
            objs = [{"id": j, "kind": "person", "bearing": rng.uniform(-90, 90), "distance": rng.uniform(0.3, 8)}
                    for j in range(rng.randint(0, 64))]
            s.sendto(json.dumps({"type": "obstacles", "source": f"s{i % 3}", "seq": i, "objects": objs}).encode(), addr)
    time.sleep(1.0)
    st = app.transport.stats
    assert st["bad"] > 0 and (st["udp"] + st["bad"] + st["limited"]) > 1000
    blocks = app.engine.stats["blocks"] - blocks0
    elapsed = time.time() - t0
    assert blocks > 0.8 * elapsed * app.engine.rate / 1024          # audio clock never stalled
    audio = app.output.recorded()
    assert np.all(np.isfinite(audio)) and np.abs(audio).max() <= app.engine.ceiling + 1e-6
    assert len([t for t in app.sonar.tick() if t]) <= 2


def test_sustained_dense_perception_keeps_control_loop_realtime(app):
    """64 objects per frame at 30 Hz for 4 s: tick + render must keep up."""
    c = client(app)
    t_end = time.time() + 4
    seq = 0
    t_start = time.time()
    while time.time() < t_end:
        seq += 1
        objs = [{"id": j, "kind": ["person", "car", "pole"][j % 3], "bearing": -80 + j * 2.5,
                 "distance": 0.6 + (j % 10) * 0.5} for j in range(64)]
        c.obstacles(objs)
        time.sleep(1 / 30)
    time.sleep(0.3)
    st = app.engine.snapshot_stats()
    assert st["avg_render_ms"] < 3.0
    # every frame is a full snapshot, so dropping some under load is harmless; what matters
    # is that the newest data keeps arriving (>= 10 Hz, the spec's perception rate)
    assert app.sonar.stats["frames"] >= 10 * (time.time() - t_start - 0.3) * 0.9
    assert app.sonar.stats["out_of_order"] == 0


def test_speech_queue_storm(app):
    """500 say() calls from 5 threads at once: no deadlock, no duplicates spoken,
    queue drains, critical items all spoken."""
    def burst(k):
        for i in range(100):
            app.say(f"message {i % 7}", ["info", "nav", "hazard"][i % 3], ttl=2)
            if i % 25 == 0:
                app.say("Stop.", "critical")
    ths = [threading.Thread(target=burst, args=(k,)) for k in range(5)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert wait_for(lambda: not app.speaker.busy, 30)
    said = spoken(app)
    assert said.count("Stop.") >= 1
    non_crit = [s for s in said if s != "Stop."]
    assert len(non_crit) <= 7 * 2       # dedupe + TTL keep chatter bounded


def test_personal_destination_is_resolved_from_memory(app):
    from websockets.sync.client import connect
    with connect(f"ws://127.0.0.1:{app.cfg.net.ws_port}") as ws:
        ws.recv(timeout=2)
        app.heard("take me to my coffee shop")
        t0 = time.time()
        req = None
        while time.time() - t0 < 3:
            m = json.loads(ws.recv(timeout=3))
            if m["type"] == "voice_request":
                req = m
                break
    assert req["name"] == "Blue Bottle Coffee" and req["target_type"] == "memory" and req["lat"] is not None
    assert wait_for(lambda: "Okay, heading to Blue Bottle Coffee." in spoken(app), 3)


def test_remember_place_by_voice_then_navigate_there(app):
    c = client(app, "nav")
    c.position(lat=37.7990, lng=-122.4400, place="Lombard Street")
    time.sleep(0.2)
    app.heard("remember this place as the pharmacy")
    assert wait_for(lambda: any("the pharmacy" in t for t in spoken(app)), 3)
    p = app.memory.resolve_place("the pharmacy")
    assert p is not None and p.meta["lat"] == 37.799
    assert app.navigate("the pharmacy") == "Okay, heading to the pharmacy."


def test_arriving_somewhere_with_a_saved_hazard_warns_once(app):
    c = client(app, "nav")
    c.position(place="Lyon Street and Bay Street")
    c.position(place="Lyon Street and Bay Street")
    assert wait_for(lambda: any(t.startswith("Heads up: The curb at Lyon") for t in spoken(app)), 3)
    time.sleep(0.5)
    assert sum(t.startswith("Heads up") for t in spoken(app)) == 1


def test_caregiver_can_add_memories_over_the_wire(app):
    c = client(app, "dashboard")
    c.send({"type": "memory", "kind": "hazard", "text": "Broken paving outside the library on Union"})
    assert wait_for(lambda: any("library" in m.text for m in app.memory.docs.values()), 2)
    app.heard("remember that the elevator at the station is out of order")
    assert wait_for(lambda: any("elevator" in m.text for m in app.memory.docs.values()), 2)


def test_walk_signal_needs_to_hold_but_dont_walk_is_instant(app):
    c = client(app, "drone")
    c.hazard("crosswalk", state="dont_walk", bearing=0, distance=6)
    assert wait_for(lambda: "Don't walk." in spoken(app), 2)
    # a single-frame "walk" misread, contradicted 0.2 s later: never announced
    c.hazard("crosswalk", state="walk", bearing=0, distance=6)
    time.sleep(0.2)
    c.hazard("crosswalk", state="dont_walk", bearing=0, distance=6)
    time.sleep(1.0)
    assert "Walk signal is on." not in spoken(app)
    # a real change, reported steadily: announced after it holds
    t0 = time.time()
    for _ in range(10):
        c.hazard("crosswalk", state="walk", bearing=0, distance=6)
        time.sleep(0.1)
    assert wait_for(lambda: "Walk signal is on." in spoken(app), 3)
    assert spoken(app).count("Walk signal is on.") == 1


def test_cue_prewarm_renders_sonar_and_beacon_cues(app):
    n = app.prewarm_cues()
    assert n > 6 * 73                                   # critical sonar cues + beacon ticks
    assert any("pre-rendered" in line for line in app.logs)
