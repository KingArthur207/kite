import os
import socket
import time

import numpy as np
import pytest

from earpiece.app import Earpiece
from earpiece.config import Cfg

# Tests never touch the network voices or real devices unless a test asks for it.
LIVE_ANTHROPIC_KEY = os.environ.pop("ANTHROPIC_API_KEY", None)   # kept only for opt-in live tests


@pytest.fixture
def offline_speech(monkeypatch, tmp_path):
    """Fast fake synthesis: a 0.15 s tone per utterance, no network, no `say`."""
    from earpiece import speech

    def fake(text, rate, voice="", wpm=210):
        # length tracks the text (like real speech) so "long" lines really are long
        n = int(max(0.15, 0.03 * len(text)) * rate)
        t = np.arange(n) / rate
        return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    monkeypatch.setattr(speech, "offline_tts", fake)
    monkeypatch.setenv("CARTESIA_API_KEY", "")
    return tmp_path


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_for(pred, timeout=5.0, step=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(step)
    return pred()


@pytest.fixture
def app(offline_speech):
    cfg = Cfg()
    cfg.speech.cache_dir = offline_speech / "cache"
    cfg.net.udp_port, cfg.net.ws_port = free_port(), free_port()
    logs = []
    from earpiece.memory import MemoryStore
    mem = MemoryStore(project_id="", project_key="", store=offline_speech / "memory.json", log=logs.append)
    a = Earpiece(cfg, output="null", record=True, log=logs.append, memory=mem)
    a.logs = logs
    a.start(prewarm=False, announce=False)
    yield a
    a.stop()


