"""Live voice loop: real Cartesia Ink STT and Sonic TTS, with recorded speech instead of a
microphone and a null audio output (nothing is played out loud).

Opt-in and cheap: KITE_LIVE_TESTS=1 and CARTESIA_API_KEY. Two short utterances, one STT
socket and one TTS context at a time (the plan allows two), about 60 TTS characters and
about 6 s of STT audio per run. Prints the end-of-speech -> first-audio breakdown:

    KITE_LIVE_TESTS=1 python -m pytest tests/test_voice_live.py -s
"""

import os
import shutil
import time

import numpy as np
import pytest

from conftest import free_port, wait_for

pytestmark = pytest.mark.skipif(
    not (os.environ.get("KITE_LIVE_TESTS") and os.environ.get("CARTESIA_API_KEY") and shutil.which("say")),
    reason="live Cartesia voice loop: set KITE_LIVE_TESTS=1 and CARTESIA_API_KEY (macOS `say` renders the speech)")

MAX_TTS_CHARS = 120          # hard stop for this test's Cartesia spend


class SpokenSource:
    """`say`-rendered speech delivered at real-time pace, like a microphone would."""

    def __init__(self, pcm: np.ndarray, rate: int = 16000, lead_s: float = 0.4, tail_s: float = 2.5):
        rng = np.random.default_rng(1)
        room = lambda s: (0.0015 * rng.standard_normal(int(s * rate))).astype(np.float32)   # noqa: E731
        self.audio = np.concatenate([room(lead_s), pcm.astype(np.float32), room(tail_s)])
        self.rate, self.block, self.pos, self.t0 = rate, int(rate * 0.05), 0, None

    def __enter__(self):
        self.t0 = time.monotonic()
        return self

    def __exit__(self, *exc):
        pass

    def get(self, timeout):
        if self.pos >= self.audio.size:
            time.sleep(min(timeout, 0.05))
            return None
        due = self.t0 + (self.pos + self.block) / self.rate
        time.sleep(max(0.0, due - time.monotonic()))
        b = self.audio[self.pos:self.pos + self.block]
        self.pos += self.block
        return b


@pytest.fixture
def live_app(tmp_path):
    from earpiece.app import Earpiece
    from earpiece.config import Cfg
    from earpiece.memory import MemoryStore
    cfg = Cfg()
    cfg.speech.cache_dir = tmp_path / "cache"          # nothing cached: every reply is synthesised
    cfg.net.udp_port, cfg.net.ws_port = free_port(), free_port()
    cfg.judgment = False
    logs = []
    mem = MemoryStore(project_id="", project_key="", store=tmp_path / "memory.json", log=logs.append)
    a = Earpiece(cfg, output="null", record=False, log=logs.append, memory=mem)
    a.logs = logs
    a.start(prewarm=False, announce=False)
    yield a
    a.stop()


def _turn(app, phrase: str, reply_contains: str):
    from earpiece.speech import offline_tts
    pcm = offline_tts(phrase, 16000, "Samantha", wpm=180)
    app.listener._open_source = lambda: SpokenSource(pcm)
    before = len(app.speaker.turns.turns)
    app.listener.push_to_talk()
    assert wait_for(lambda: len(app.speaker.turns.turns) > before, 15), app.logs[-15:]
    assert wait_for(lambda: not app.speaker.busy and not app.listener.active, 10)
    rec = app.speaker.turns.turns[-1]
    print(f"\n  {phrase!r} -> heard {rec['text']!r}, early end {rec['early_end']}, reply via {rec['source']}")
    print("  " + "  ".join(f"{k} {rec[k]:.0f} ms" for k in ("endpoint_ms", "stt_ms", "brain_ms", "tts_ms", "total_ms")))
    assert any(reply_contains in h["text"] for h in app.speaker.history)
    return rec


def test_live_voice_loop_breakdown(live_app):
    app = live_app
    quick = _turn(app, "Louder, please.", "Louder.")
    assert quick["early_end"] is True and quick["source"] == "cartesia"
    assert app.speaker.meter.session <= MAX_TTS_CHARS
    full = _turn(app, "What's around me?", "Nothing close to you")
    assert full["early_end"] is False and full["source"] == "cartesia"
    assert app.speaker.meter.session <= MAX_TTS_CHARS
    lat = app.speaker.latency()
    print(f"  Cartesia TTS first byte (median) {lat['tts_first_byte_ms']} ms; "
          f"TTS characters this run {app.speaker.meter.session}; STT audio {app.listener.stats['stt_audio_s']:.1f} s")
    assert quick["total_ms"] < full["total_ms"]          # the short command really is faster
    assert full["total_ms"] < 3000
