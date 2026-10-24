"""scene_describe self-check, no network: python sim/test_scene_describe.py (or pytest)."""

import base64
import json
import os
import socket
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_describe as sd  # noqa: E402
from earpiece.client import EarpieceClient  # noqa: E402


class FakeClient:
    """Stands in for anthropic.Anthropic(); records each request, optionally blocks until released."""

    def __init__(self, text="Sidewalk ahead, a person 3 meters left.", gate=None):
        self.text, self.gate, self.calls, self.opts = text, gate, [], []
        self.messages = self

    def with_options(self, **kw):
        self.opts.append(kw)
        return self

    def create(self, **kw):
        self.calls.append(kw)
        if self.gate:
            self.gate.wait(5)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=self.text)])


def ear_pair(source):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    return rx, EarpieceClient("127.0.0.1", rx.getsockname()[1], source=source)


IMG = np.full((720, 960, 3), 120, np.uint8)


def test_event_shape_and_request():
    rx, ear = ear_pair("scene-drone")
    c = FakeClient()
    d = sd.SceneDescriber(c, ear, log=lambda *_: None)
    assert d.tick(IMG, 100.0, wait=True)
    msg = json.loads(rx.recv(65536))
    assert msg == {"type": "scene", "text": "Sidewalk ahead, a person 3 meters left.", "source": "scene-drone"}
    req = c.calls[0]
    assert req["model"] == sd.MODEL and c.opts[0]["timeout"] == 4.0
    img = req["messages"][0]["content"][0]
    assert img["source"]["media_type"] == "image/jpeg"
    jpg = cv2.imdecode(np.frombuffer(base64.b64decode(img["source"]["data"]), np.uint8), cv2.IMREAD_COLOR)
    assert jpg.shape[1] == 768


def test_rate_limit_and_skip_while_busy():
    gate = threading.Event()
    d = sd.SceneDescriber(FakeClient(gate=gate), ear_pair("scene-chest")[1], interval=6.0, log=lambda *_: None)
    assert not d.tick(None, 0.0)                 # no frame yet
    assert d.tick(IMG, 0.0)                      # first call starts
    assert not d.tick(IMG, 10.0, force=True)     # still running: skipped, even when asked
    gate.set()
    d._busy.acquire(timeout=2)                   # wait for the call to finish
    d._busy.release()
    assert not d.tick(IMG, 3.0)                  # periodic: too soon
    assert d.tick(IMG, 3.0, force=True, wait=True)  # a question bypasses the interval
    assert not d.tick(IMG, 8.0)
    assert d.tick(IMG, 9.1, wait=True)


def test_no_disk_writes():
    def boom(*a, **k):
        raise AssertionError("frame written to disk")
    old = cv2.imwrite
    cv2.imwrite = boom
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            d = sd.SceneDescriber(FakeClient(), ear_pair("scene-chest")[1], log=lambda *_: None)
            assert d.tick(IMG, 0.0, wait=True)
            assert os.listdir(tmp) == []
        finally:
            os.chdir(cwd)
            cv2.imwrite = old


def test_prompt_rules():
    p = sd.SYSTEM_PROMPT.lower()
    assert "never make a safety judgment" in p and "safe" in p and "cross" in p
    assert "faces" in p and "identity" in p
    assert "only describe what is visible" in p


def test_question_trigger():
    assert sd.asks_whats_ahead("Kite, what's in front of me?")
    assert sd.asks_whats_ahead("what is around me")
    assert sd.asks_whats_ahead("describe the street")
    assert not sd.asks_whats_ahead("take me to Blue Bottle")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
