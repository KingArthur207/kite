import os
import socket
import time

import numpy as np
import pytest

# Tests never touch the network voices or real devices unless a test asks for it.
os.environ.pop("ANTHROPIC_API_KEY", None)


@pytest.fixture
def offline_speech(monkeypatch, tmp_path):
    """Fast fake synthesis: a 0.15 s tone per utterance, no network, no `say`."""
    from earpiece import speech

    def fake(text, rate, voice="", wpm=210):
        n = int(0.15 * rate)
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
