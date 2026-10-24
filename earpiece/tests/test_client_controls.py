"""The teammate client (UDP sender) and the AirPods stem gesture classifier, without devices."""

from __future__ import annotations

import json
import random
import socket
import threading

import pytest

from conftest import wait_for
from earpiece.client import MAX_DATAGRAM, EarpieceClient
from earpiece.controls import PAUSE, REPEAT, TALK, Gestures
from earpiece.events import MAX_OBJECTS, parse


def receiver():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(2)
    return s


def test_a_crowded_frame_is_trimmed_to_the_nearest_objects_instead_of_vanishing():
    """macOS refuses datagrams over 9216 bytes, and send() swallowed the error: a 150-person
    crowd (exactly when the walker needs the sonar) was silently never delivered."""
    rx = receiver()
    c = EarpieceClient("127.0.0.1", rx.getsockname()[1], source="cam")
    rng = random.Random(3)
    objs = [{"id": f"track-{i:04d}", "kind": "person", "bearing": round(rng.uniform(-90, 90), 1),
             "distance": round(rng.uniform(0.4, 30), 2), "confidence": 0.87, "cut_off": False}
            for i in range(150)]
    assert len(json.dumps({"type": "obstacles", "objects": objs})) > 9216
    c.obstacles(objs)
    data = rx.recv(65536)
    assert len(data) <= MAX_DATAGRAM and c.trimmed == 1 and c.dropped == 0
    m = parse(data)
    nearest = sorted(o["distance"] for o in objs)[:len(m.objects)]
    assert len(m.objects) >= 48 and [o.distance for o in m.objects] == nearest
    assert len(m.objects) <= MAX_OBJECTS


def test_small_frames_go_out_unchanged_and_a_missing_earpiece_never_raises():
    rx = receiver()
    c = EarpieceClient("127.0.0.1", rx.getsockname()[1])
    c.obstacles([{"id": 1, "kind": "pole", "bearing": 3, "distance": 2.0}])
    m = json.loads(rx.recv(65536))
    assert m["objects"] == [{"id": 1, "kind": "pole", "bearing": 3, "distance": 2.0}] and c.trimmed == 0
    rx.close()
    dead = EarpieceClient("127.0.0.1", 9)            # nothing listens there
    for _ in range(5):
        dead.say("hello")
    dead.send({"type": "say", "text": "x" * 70000})   # too big even for the OS: counted, not raised
    assert dead.dropped >= 1


def test_every_client_helper_sends_what_the_earpiece_accepts():
    """The client is the contract teammates code against: each helper's datagram must parse."""
    rx = receiver()
    c = EarpieceClient("127.0.0.1", rx.getsockname()[1], source="drone")
    c.heartbeat()
    c.guidance("In 20 meters, turn left onto Lyon Street", key="step-3", ttl=10, bearing=-30)
    c.beacon(bearing=-35, distance=42, label="Palace of Fine Arts")
    c.beacon(active=False)
    c.hazard("crosswalk", state="dont_walk", bearing=5, distance=9, ttl=5)
    c.say("Hello", priority="answer")
    c.scene("Wide sidewalk, light crowd near the corner")
    c.position(lat=37.8, lng=-122.44, heading=90, place="Lyon Street")
    c.status(battery=81, state="flying", text="ok")
    c.control("pause")
    c.obstacles([{"id": 1, "kind": "pole", "bearing": 3, "distance": 2.0}], source="chest")
    got = [parse(rx.recv(65536)) for _ in range(11)]
    rx.close()
    assert [m.type for m in got] == ["heartbeat", "guidance", "beacon", "beacon", "hazard", "say", "scene",
                                     "position", "status", "control", "obstacles"]
    assert all(m.source == "drone" for m in got[:-1]) and got[-1].source == "chest"


def test_odd_obstacle_distances_never_break_trimming():
    """Trimming sorts by distance; a missing, NaN, string or bool distance sorts last, never raises."""
    rx = receiver()
    c = EarpieceClient("127.0.0.1", rx.getsockname()[1])
    odd = [{"id": i, "kind": "person", "bearing": 0, "distance": d, "pad": "x" * 200}
           for i, d in enumerate([float("nan"), "near", True, None, float("inf")] * 20)]
    near = [{"id": 1000 + i, "kind": "pole", "bearing": 0, "distance": 1.0 + i, "pad": "x" * 200} for i in range(5)]
    c.obstacles(odd + near)
    m = json.loads(rx.recv(65536))
    rx.close()
    assert c.trimmed == 1 and len(json.dumps(m)) <= MAX_DATAGRAM
    assert [o["id"] for o in m["objects"][:5]] == [1000, 1001, 1002, 1003, 1004]   # real distances first


def test_requests_yields_only_the_asked_kinds_and_reconnects(monkeypatch):
    """client.requests(): the generator teammates use for voice requests. It skips other
    events and, when the earpiece is not there yet, retries instead of raising."""
    import time

    from earpiece import client
    from earpiece.transport import Transport
    t = Transport(lambda m: None, "127.0.0.1", 0, 0, log=lambda *_: None)
    t.start()
    try:
        url = f"ws://127.0.0.1:{t.ws_port}"
        assert wait_for(lambda: t.ws_port, 5)
        gen = client.requests(url)
        out: list = []
        reader = threading.Thread(target=lambda: out.extend([next(gen), next(gen)]), daemon=True)
        reader.start()
        assert wait_for(lambda: t.stats["clients"] == 1, 10)
        t.publish({"type": "spoken", "text": "not for the client"})
        t.publish({"type": "voice_request", "intent": "navigate", "destination": "Crissy Field"})
        t.publish({"type": "caregiver_alert", "message": "help"})
        reader.join(10)
        assert [m["type"] for m in out] == ["voice_request", "caregiver_alert"]
    finally:
        t.stop()
    # nothing listens: it backs off 1 s and retries, never raises. The third back-off ends the test.
    sleeps: list = []

    class Done(Exception):
        pass

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            raise Done
    monkeypatch.setattr(time, "sleep", fake_sleep)
    with pytest.raises(Done):
        next(client.requests("ws://127.0.0.1:9"))
    assert sleeps == [1.0, 1.0, 1.0]


def test_gestures_single_double_triple():
    got = []
    done = threading.Event()

    def on(g):
        got.append(g)
        if len(got) == 3:
            done.set()
    g = Gestures(on, debounce_s=0.05, log=lambda *_: None)
    try:
        g.raw("toggle")
        assert wait_for(lambda: got == [TALK], 2)
        g.raw("toggle")
        g.raw("next")                     # within the debounce window: one gesture, not two
        assert wait_for(lambda: got == [TALK, REPEAT], 2)
        g.raw("previous")
        assert done.wait(2) and got == [TALK, REPEAT, PAUSE]
    finally:
        g.stop()


def test_a_failing_gesture_handler_does_not_kill_the_classifier():
    got, logs = [], []

    def on(g):
        got.append(g)
        if len(got) == 1:
            raise RuntimeError("handler bug")
    g = Gestures(on, debounce_s=0.02, log=logs.append)
    try:
        g.raw("play")
        assert wait_for(lambda: len(got) == 1, 2)
        g.raw("play")
        assert wait_for(lambda: len(got) == 2, 2) and any("handler bug" in m for m in logs)
    finally:
        g.stop()
