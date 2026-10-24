"""The Cartesia streaming path against a local fake of its TTS WebSocket (no network, no credits,
null audio). What the transcript says decides how the fake behaves."""

from __future__ import annotations

import asyncio
import base64
import json
import threading

import numpy as np
import pytest
import websockets

from conftest import wait_for
from earpiece import speech
from earpiece.config import SpeechCfg
from earpiece.mixer import Engine
from earpiece.output import NullOutput
from earpiece.speech import Speaker


class FakeCartesia:
    """Behaviour by transcript: "busy" -> one concurrency error first; "hang" -> silence;
    "drop" -> one chunk then the socket closes; "junk" -> non-object frames before the audio."""

    def __init__(self):
        self.requests: list[dict] = []
        self.keys: list[str] = []
        self.busy_left = 1
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        assert self._ready.wait(5)

    def _run(self):
        asyncio.set_event_loop(self._loop)

        async def main():
            self.server = await websockets.serve(self._handler, "127.0.0.1", 0)
            self.port = self.server.sockets[0].getsockname()[1]
            self._ready.set()
            await asyncio.Future()
        try:
            self._loop.run_until_complete(main())
        except RuntimeError:
            pass

    async def _handler(self, ws):
        self.keys.append(ws.request.headers.get("X-API-Key"))
        async for raw in ws:
            m = json.loads(raw)
            if m.get("cancel"):
                continue
            self.requests.append(m)
            ctx, text = m["context_id"], m["transcript"]
            chunk = base64.b64encode((0.1 * np.ones(2400, np.float32)).tobytes()).decode()
            if "hang" in text:
                continue
            if "busy" in text and self.busy_left > 0:
                self.busy_left -= 1
                await ws.send(json.dumps({"type": "error", "context_id": ctx, "message": "Concurrency limit reached"}))
                continue
            if "junk" in text:
                for j in ("not json", "[1, 2]", "7", json.dumps({"type": "chunk"}), json.dumps(["chunk"])):
                    await ws.send(j)
            await ws.send(json.dumps({"type": "chunk", "context_id": ctx, "data": chunk}))
            if "drop" in text:
                await ws.close()
                return
            await ws.send(json.dumps({"type": "chunk", "context_id": ctx, "data": chunk}))
            await ws.send(json.dumps({"type": "done", "context_id": ctx}))

    def stop(self):
        self._loop.call_soon_threadsafe(self.server.close)


@pytest.fixture
def cartesia(monkeypatch):
    f = FakeCartesia()
    monkeypatch.setattr(speech.CartesiaStream, "URL", f"ws://127.0.0.1:{f.port}/tts?cartesia_version={{v}}")
    yield f
    f.stop()


@pytest.fixture
def spk(offline_speech, cartesia):
    offline: list[str] = []
    real_fake = speech.offline_tts

    def offline_tts(text, rate, voice="", wpm=210):
        offline.append(text)
        return real_fake(text, rate, voice, wpm)
    speech.offline_tts = offline_tts            # the conftest fake, recorded (monkeypatch undoes both)
    eng = Engine()
    out = NullOutput(eng, 1024, realtime=True)
    out.start()
    cfg = SpeechCfg(cache_dir=offline_speech / "cache", first_chunk_timeout_s=0.5, char_budget=0)
    s = Speaker(eng, cfg, key="test-key", log=lambda *_: None)
    s.offline = offline
    s.start()
    yield s
    s.stop()
    out.stop()
    speech.offline_tts = real_fake


def results(s):
    return [(h["text"], h["result"]) for h in s.history]


def test_neural_speech_streams_then_repeats_from_the_cache(spk, cartesia):
    spk.say("Construction on the left sidewalk.", "info")
    assert wait_for(lambda: results(spk) == [("Construction on the left sidewalk.", "done")], 5)
    assert spk.stats["neural"] == 1 and spk.online is True and cartesia.keys == ["test-key"]
    assert cartesia.requests[0]["output_format"]["sample_rate"] == spk.engine.rate
    spk._recent.clear()
    spk.say("Construction on the left sidewalk.", "info")
    assert wait_for(lambda: len(spk.history) == 2, 5)
    assert spk.stats["cache_hits"] == 1 and len(cartesia.requests) == 1     # repeats cost nothing
    assert spk.meter.session == len("Construction on the left sidewalk.")


def test_concurrency_limit_is_retried_not_sent_to_the_offline_voice(spk, cartesia):
    spk.say("busy line for the retry", "answer")
    assert wait_for(lambda: results(spk) == [("busy line for the retry", "done")], 5)
    assert spk.stats.get("concurrency_retries") == 1 and len(cartesia.requests) == 2
    assert spk.offline == [] and spk.tts.down_until == 0.0


def test_silence_from_cartesia_falls_back_to_the_offline_voice_and_backs_off(spk, cartesia):
    spk.say("hang forever please", "hazard", ttl=10)
    assert wait_for(lambda: results(spk) == [("hang forever please", "done")], 8)
    assert spk.offline == ["hang forever please"] and spk.online is False
    assert not spk.tts.available                      # backed off: the next line won't wait 0.5 s again


def test_a_dropped_connection_mid_line_is_not_restarted(spk, cartesia):
    spk.say("drop after the first chunk", "nav", ttl=10)
    assert wait_for(lambda: len(spk.history) == 1, 8)
    assert spk.history[0]["result"] == "done" and spk.offline == []   # what arrived was played
    assert len(cartesia.requests) == 1


def test_junk_frames_from_cartesia_do_not_kill_the_connection(spk, cartesia):
    spk.say("junk first then audio", "answer")
    assert wait_for(lambda: results(spk) == [("junk first then audio", "done")], 5)
    spk.say("and the next line too", "answer")
    assert wait_for(lambda: len(spk.history) == 2 and spk.history[1]["result"] == "done", 5)
    assert len(cartesia.keys) == 1 and spk.offline == []                 # same socket, still neural


def test_budget_keeps_safety_lines_neural(spk, cartesia):
    spk.meter.budget, spk.meter.total = 10, 10
    spk.say("a chatty info line", "info")
    spk.say("Scooter coming fast on your left.", "hazard")
    assert wait_for(lambda: len(spk.history) == 2, 8)
    assert [r["transcript"] for r in cartesia.requests] == ["Scooter coming fast on your left."]
    assert "a chatty info line" in spk.offline       # (a hazard may also render an offline backup)
