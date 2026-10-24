"""Claude's eyes (vision.py): scene descriptions, "what's in front of me?", reading text
aloud, and the stop-and-wait / continue behaviour around them. Claude is faked throughout."""

import base64
import json
import time
import types

import pytest

from conftest import wait_for
from earpiece import vision as V
from earpiece.events import parse


# ---------------------------------------------------------------- descriptions never instruct
@pytest.mark.parametrize("text", [
    "Go ahead.", "You can cross now.", "The path is clear.", "It's safe to cross.", "Continue walking.",
    "Keep walking straight.", "Walk now, the light is green.", "All clear!", "Turn left to avoid the crowd.",
    "Take the other sidewalk instead.", "Step off the curb.", "You can go.", "Proceed carefully.",
    "the way is now clear", "reroute through Bay Street",
])
def test_go_sentences_are_recognised(text):
    assert V._GO.search(text)


@pytest.mark.parametrize("text", [
    "Narrow sidewalk, two people ahead on the left.", "A bench about one meter ahead.",
    "Wait, the crosswalk is crowded.", "A cyclist is coming from the left.", "Chicago Street sign.",
])
def test_plain_descriptions_pass(text):
    assert not V._GO.search(text)


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
    app.vision._client = fake
    app.handle(parse({"type": "scene_image", "camera": "drone",
                      "jpeg": base64.b64encode(b"\xff\xd8" + b"0" * 100).decode()}))
    assert wait_for(lambda: app.scene is not None, 8)
    assert app.scene[0] == "Narrow sidewalk, two people ahead on the left."   # the instruction is stripped
    req = fake.requests[0]
    assert req["messages"][0]["content"][0]["type"] == "image"


# ---------------------------------------------------------------- "what's in front of me?"
def test_front_question_uses_the_chest_camera(app):
    fake = FakeClaude(["A wooden bench about one meter ahead, slightly right."])
    app.vision._client = fake
    app.handle(parse({"type": "scene_image", "camera": "chest",
                      "jpeg": base64.b64encode(b"\xff\xd8chest").decode()}))
    assert app._chest_frame is not None and fake.requests == []      # chest frames are not auto-described
    app.heard("what's in front of me")
    assert wait_for(lambda: "A wooden bench about one meter ahead, slightly right." in
                    [h["text"] for h in app.speaker.history], 10)
    assert "worn on the walker's chest" in fake.requests[0]["messages"][0]["content"][1]["text"]


def test_front_question_without_a_fresh_frame_answers_locally(app):
    app.heard("what's in front of me")
    assert wait_for(lambda: any("Nothing close" in h["text"] or "not running" in h["text"]
                                for h in app.speaker.history), 6)


def test_judgment_events_are_not_bridged(app):
    sent = []
    app.bridges.append(types.SimpleNamespace(publish=lambda o: sent.append(o["type"])))
    app._event("judgment", {"say": "x"})
    app._event("continue", {})
    app._event("spoken", {"text": "x"})
    assert sent == ["spoken"]


# ---------------------------------------------------------------- reading text aloud (OCR)
def test_read_this_reads_the_text_sentence_by_sentence(app):
    fake = FakeClaude(["**Lunch menu.** Margherita pizza, 14 dollars. Caesar salad, 11 dollars."])
    app.vision._client = fake
    app.handle(parse({"type": "scene_image", "camera": "chest", "jpeg": base64.b64encode(b"\xff\xd8menu").decode()}))
    app.heard("read this to me")
    said = lambda: [h["text"] for h in app.speaker.history]
    assert wait_for(lambda: "Caesar salad, 11 dollars." in said(), 15)
    assert said()[:4] == ["Hold it steady. Reading.", "Lunch menu.", "Margherita pizza, 14 dollars.",
                          "Caesar salad, 11 dollars."]                       # markdown stripped, in order
    assert "Transcribe the readable text exactly" in fake.requests[0]["messages"][0]["content"][1]["text"]


def test_read_with_no_text_or_no_camera(app):
    fake = FakeClaude(["NO_TEXT"])
    app.vision._client = fake
    app.heard("what does this say")                     # no frame and the camera is off in tests
    assert wait_for(lambda: any("can't see through the camera" in h["text"] for h in app.speaker.history), 8)
    app.handle(parse({"type": "scene_image", "camera": "chest", "jpeg": base64.b64encode(b"\xff\xd8blank").decode()}))
    app.heard("read this")
    assert wait_for(lambda: any("can't see any text" in h["text"] for h in app.speaker.history), 10)


def test_warnings_cut_into_a_long_reading(app):
    long = " ".join(f"Item {i} costs {i} dollars." for i in range(1, 12))
    app.vision._client = FakeClaude([long])
    app.handle(parse({"type": "scene_image", "camera": "chest", "jpeg": base64.b64encode(b"\xff\xd8m").decode()}))
    app.heard("read the menu")
    assert wait_for(lambda: any(h["text"].startswith("Item 1") for h in app.speaker.history), 10)
    app.say("Stop.", "critical")
    assert wait_for(lambda: "Stop." in [h["text"] for h in app.speaker.history], 5)
    order = [h["text"] for h in app.speaker.history]
    assert order.index("Stop.") < len(order) - 1 or app.speaker.busy    # the stop did not wait for the menu


def test_vision_failure_is_counted_and_answered(app):
    class Boom:
        class beta:
            class messages:
                @staticmethod
                async def create(**kw):
                    raise RuntimeError("network down")
    app.vision._client = Boom()
    app.handle(parse({"type": "scene_image", "camera": "chest", "jpeg": base64.b64encode(b"\xff\xd8x").decode()}))
    app.heard("read this")
    assert wait_for(lambda: any("couldn't read that" in h["text"] for h in app.speaker.history), 10)
    assert app.vision.stats["errors"] == 1
    assert app.state()["vision"]["errors"] == 1
