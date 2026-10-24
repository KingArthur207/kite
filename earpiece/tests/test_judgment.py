"""The judgment layer may only make Kite more careful. These tests try hard to make it do otherwise."""

import base64
import json
import time
import types

import pytest

from conftest import wait_for
from earpiece import judgment as J
from earpiece.events import parse


# ---------------------------------------------------------------- the box
@pytest.mark.parametrize("text", [
    "Go ahead.", "You can cross now.", "The path is clear.", "It's safe to cross.", "Continue walking.",
    "Keep walking straight.", "Walk now, the light is green.", "All clear!", "Turn left to avoid the crowd.",
    "Take the other sidewalk instead.", "Step off the curb.", "You can go.", "Proceed carefully.",
    "the way is now clear", "reroute through Bay Street",
])
def test_anything_that_moves_or_clears_is_blocked(text):
    assert not J.is_permitted(text)


@pytest.mark.parametrize("text", [
    "Wait, the crosswalk is crowded.", "Slow down, a cyclist is coming from the left.",
    "Careful, low branch at head height ahead.", "Wait for the people to pass.", "Stay where you are.",
])
def test_caution_is_allowed(text):
    assert J.is_permitted(text)


def test_sanitize_only_raises_existing_hazards():
    hazards = {"crowd": {"priority": "nav"}, "car": {"priority": "critical"}}
    d = J.sanitize({"say": "Go now.", "wait": False, "upgrade": [
        {"key": "crowd", "priority": "hazard"},       # raise: allowed
        {"key": "car", "priority": "hazard"},         # lower critical -> hazard: refused
        {"key": "ghost", "priority": "critical"},     # unknown hazard: refused
        {"key": "crowd", "priority": "info"}]}, hazards)
    assert d.say == "" and d.blocked
    assert d.upgrade == [("crowd", "hazard")]


def test_sanitize_handles_garbage():
    for raw in (None, [], "x", {"say": 5, "upgrade": "no"}, {"upgrade": [None, 3, {"key": 1}]}):
        d = J.sanitize(raw, {})
        assert d.say == "" and d.upgrade == []


# ---------------------------------------------------------------- with a fake Claude
class FakeClaude:
    """Stands in for anthropic.AsyncAnthropic: returns scripted replies, records requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        outer = self

        class Messages:
            async def create(self, **kw):
                outer.requests.append(kw)
                r = outer.replies.pop(0) if outer.replies else {"say": "", "wait": False, "upgrade": []}
                text = r if isinstance(r, str) else json.dumps(r)
                return types.SimpleNamespace(stop_reason="end_turn",
                                             content=[types.SimpleNamespace(type="text", text=text)])

        self.beta = types.SimpleNamespace(messages=Messages())


def test_judgment_speaks_waits_and_upgrades(app):
    fake = FakeClaude([{"say": "Wait, the crosswalk is crowded.", "wait": True,
                        "upgrade": [{"key": "crowd", "priority": "critical"}]}])
    app.judgment._client = fake
    app.judgment.min_interval_s = 0
    app.handle(parse({"type": "hazard", "kind": "crowd", "key": "crowd", "bearing": 5, "distance": 8,
                      "priority": "nav"}))
    said = lambda: [h["text"] for h in app.speaker.history]
    assert wait_for(lambda: "Wait, the crosswalk is crowded." in said(), 8)
    assert app.hazards["crowd"]["priority"] == "critical"
    assert app._stop_since is not None
    req = fake.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["output_config"]["effort"] == "low"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in req                                         # it cannot act, only advise
    snap = json.loads(req["messages"][0]["content"])
    assert snap["hazards"][0]["key"] == "crowd"


def test_judgment_cannot_clear_a_stop(app):
    fake = FakeClaude([{"say": "The path is clear, you can continue.", "wait": False, "upgrade": []}])
    app.judgment._client = fake
    app.judgment.min_interval_s = 0
    app.handle(parse({"type": "hazard", "kind": "vehicle", "key": "car", "state": "wait", "bearing": 0}))
    assert wait_for(lambda: app.judgment.stats["blocked"] == 1, 8)
    time.sleep(0.5)
    texts = [h["text"] for h in app.speaker.history]
    assert "The path is clear, you can continue." not in texts
    assert "Car, ahead. Wait." in texts


def test_judgment_failure_changes_nothing(app):
    class Boom:
        class beta:
            class messages:
                @staticmethod
                async def create(**kw):
                    raise RuntimeError("network down")
    app.judgment._client = Boom()
    app.judgment.min_interval_s = 0
    app.handle(parse({"type": "hazard", "kind": "crowd", "key": "c2", "bearing": 20}))
    assert wait_for(lambda: app.judgment.stats["errors"] >= 1, 8)
    assert wait_for(lambda: "Crowd, slightly right." in [h["text"] for h in app.speaker.history], 4)


def test_judgment_is_rate_limited_and_coalesced(app):
    fake = FakeClaude([])
    app.judgment._client = fake
    app.judgment.min_interval_s = 1.0
    for i in range(20):
        app.handle(parse({"type": "hazard", "kind": "crowd", "key": f"k{i}", "bearing": i}))
    time.sleep(2.6)
    assert 1 <= len(fake.requests) <= 3


# ---------------------------------------------------------------- stop, wait, continue
def test_stop_wait_then_continue(app):
    from earpiece.client import EarpieceClient
    c = EarpieceClient("127.0.0.1", app.cfg.net.udp_port, source="webcam")
    for _ in range(25):                                # someone right in front: "Stop."
        c.obstacles([{"id": 1, "kind": "person", "bearing": 0, "distance": 0.5}])
        time.sleep(1 / 15)
    said = lambda: [h["text"] for h in app.speaker.history]
    assert wait_for(lambda: any(t.startswith("Stop") for t in said()), 3)
    assert "Path is clear. You can continue." not in said()
    for _ in range(50):                                # they step aside; perception keeps running
        c.obstacles([])
        time.sleep(1 / 15)
    assert wait_for(lambda: "Path is clear. You can continue." in said(), 4)
    assert said().count("Path is clear. You can continue.") == 1


def test_no_continue_when_camera_dies(app):
    from earpiece.client import EarpieceClient
    c = EarpieceClient("127.0.0.1", app.cfg.net.udp_port, source="webcam")
    for _ in range(25):
        c.obstacles([{"id": 1, "kind": "pole", "bearing": 0, "distance": 0.5}])
        time.sleep(1 / 15)
    time.sleep(4.0)                                     # camera stops sending: never say "clear"
    assert "Path is clear. You can continue." not in [h["text"] for h in app.speaker.history]


# ---------------------------------------------------------------- scene images
def test_scene_image_is_validated():
    from earpiece.events import BadMessage
    with pytest.raises(BadMessage):
        parse({"type": "scene_image", "jpeg": base64.b64encode(b"not a jpeg").decode()})
    with pytest.raises(BadMessage):
        parse({"type": "scene_image", "jpeg": "!!!"})
    m = parse({"type": "scene_image", "camera": "chest", "jpeg": base64.b64encode(b"\xff\xd8xx").decode()})
    assert m.kind == "chest" and m.data["jpeg"].startswith(b"\xff\xd8")


def test_scene_image_becomes_scene_context(app):
    fake = FakeClaude(["Narrow sidewalk, two people ahead on the left. Go straight, it's clear."])
    app.judgment._client = fake
    app.handle(parse({"type": "scene_image", "camera": "drone",
                      "jpeg": base64.b64encode(b"\xff\xd8" + b"0" * 100).decode()}))
    assert wait_for(lambda: app.scene is not None, 8)
    assert app.scene[0] == "Narrow sidewalk, two people ahead on the left."   # the instruction is stripped
    req = fake.requests[0]
    assert req["messages"][0]["content"][0]["type"] == "image"
