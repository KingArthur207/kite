"""The teammate client (UDP sender) and the AirPods stem gesture classifier, without devices."""

from __future__ import annotations

import json
import random
import socket
import threading

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
