"""The voice loop: barge-in, speaking rate, pronunciation, latency breakdown, early end of
short commands, Moss prefetch, and how the voice behaves when Cartesia is slow or gone.
Everything here runs on fakes: no network, no microphone, no audio device."""

import asyncio
import base64
import json
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import wait_for

try:                                  # the brain imports it on the first question: pay that at collection
    import anthropic  # noqa: F401
    HAVE_ANTHROPIC = True
except ImportError:
    HAVE_ANTHROPIC = False
needs_anthropic = pytest.mark.skipif(not HAVE_ANTHROPIC, reason="anthropic SDK not installed")
from earpiece import speech
from earpiece.brain import TOOLS, Actions, Brain, classify, quick_command
from earpiece.config import ListenCfg, SpeechCfg
from earpiece.controls import REPEAT, TALK, Gestures
from earpiece.listen import FLUSH, Listener
from earpiece.mixer import Engine
from earpiece.output import NullOutput
from earpiece.speech import (FIXED_PHRASES, VOICE_BACK, VOICE_OFFLINE, CartesiaStream, ConcurrencyLimited,
                             Speaker, VoiceTurns, pronounce)


# ---------------------------------------------------------------- fakes
class FakeTTSSocket:
    """Stands in for the Cartesia TTS WebSocket: answers each request with a few chunks of
    silence after `ttfb` seconds, or with an error."""

    def __init__(self, cs: CartesiaStream, ttfb: float = 0.02, chunks: int = 4, chunk_s: float = 0.05):
        self.cs, self.ttfb, self.chunks, self.chunk_s = cs, ttfb, chunks, chunk_s
        self.error: dict | None = None
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        m = json.loads(raw)
        self.sent.append(m)
        if not m.get("cancel"):
            asyncio.get_running_loop().create_task(self._respond(m))

    async def _respond(self, m: dict) -> None:
        await asyncio.sleep(self.ttfb)
        q = self.cs._ctx.get(m["context_id"])
        if q is None:
            return
        if self.error is not None:
            q.put_nowait({"type": "error", "context_id": m["context_id"], **self.error})
            return
        data = base64.b64encode(np.zeros(int(self.cs.rate * self.chunk_s), np.float32).tobytes()).decode()
        for _ in range(self.chunks):
            q.put_nowait({"type": "chunk", "data": data, "context_id": m["context_id"]})
            await asyncio.sleep(0.002)
        q.put_nowait({"type": "done", "context_id": m["context_id"]})

    async def close(self) -> None:
        pass


def neural_speaker(tmp_path, monkeypatch, **sock):
    """A Speaker whose Cartesia is a FakeTTSSocket (and whose connect can be made to fail)."""
    monkeypatch.setattr(Speaker, "PROBE_EVERY_S", 0.1)
    monkeypatch.setattr(Speaker, "ANNOUNCE_GAP_S", 0.0)
    eng = Engine()
    out = NullOutput(eng, 1024, realtime=True)
    out.start()
    s = Speaker(eng, SpeechCfg(cache_dir=tmp_path / "cache", char_budget=0), key="test-key", log=lambda *_: None)
    fake = FakeTTSSocket(s.tts, **sock)
    state = {"down": False, "connect_s": 0.0}

    async def connect():
        if state["down"]:
            await asyncio.sleep(state["connect_s"])       # a dead network: the handshake hangs, then fails
            raise OSError("network is unreachable")
        s.tts._ws = fake
        return fake

    s.tts._connect = connect
    s.start()
    return s, fake, state, out


def results(s):
    return [(h["text"], h["result"]) for h in list(s.history)]


@pytest.fixture
def spk(offline_speech):
    eng = Engine()
    out = NullOutput(eng, 1024, realtime=True)
    out.start()
    s = Speaker(eng, SpeechCfg(cache_dir=offline_speech / "cache"), key=None, log=lambda *_: None)
    s.start()
    yield s
    s.stop()
    out.stop()


# ---------------------------------------------------------------- barge-in
def test_barge_in_cuts_an_answer_fast_and_forgets_queued_replies(spk):
    spk.say("This is the first sentence of a long answer about the park.", "answer")
    spk.say("And this is the second sentence of it.", "answer")
    spk.say("Crowd ahead.", "info")
    assert wait_for(lambda: spk._current is not None and spk._current.priority == "answer", 2)
    time.sleep(0.1)
    t0 = time.monotonic()
    assert spk.barge_in() is True
    assert wait_for(lambda: spk.history, 2)
    cut_s = time.monotonic() - t0
    assert results(spk)[0] == ("This is the first sentence of a long answer about the park.", "cut")
    assert cut_s < 0.25, f"barge-in took {cut_s * 1000:.0f} ms"
    time.sleep(0.4)
    assert len(spk.history) == 1 and not spk.busy           # the rest of the old reply is gone
    assert spk.stats["barge_ins"] == 1


def test_barge_in_never_cuts_a_warning(spk):
    spk.say("Construction ahead on the left, keep right.", "hazard")
    assert wait_for(lambda: spk._current is not None, 2)
    spk.set_listening(True)                          # the wearer pressed to talk
    assert wait_for(lambda: spk.history, 5)
    spk.set_listening(False)
    assert results(spk) == [("Construction ahead on the left, keep right.", "done")]
    assert spk.stats["barge_ins"] == 0


def test_barge_in_requeues_fresh_navigation(spk):
    spk.say("In thirty meters, turn left onto Bay Street.", "nav", ttl=10)
    assert wait_for(lambda: spk._current is not None, 2)
    spk.set_listening(True)
    assert wait_for(lambda: spk.history, 2)
    assert results(spk)[0][1] == "cut"
    spk.set_listening(False)
    assert wait_for(lambda: len(spk.history) == 2, 5)
    assert results(spk)[1] == ("In thirty meters, turn left onto Bay Street.", "done")


def test_critical_speech_still_plays_while_the_wearer_talks(spk):
    spk.set_listening(True)
    spk.say("Stop.", "critical")
    spk.say("Here is an answer.", "answer")
    assert wait_for(lambda: spk.history, 3)
    time.sleep(0.3)
    assert [h["text"] for h in list(spk.history)] == ["Stop."]
    spk.set_listening(False)


def test_first_press_cuts_before_the_gesture_is_classified():
    pressed, gestures = [], []
    g = Gestures(gestures.append, debounce_s=0.3, log=lambda *_: None)
    g.on_press = lambda: pressed.append(time.monotonic())
    t0 = time.monotonic()
    g.raw("toggle")
    assert wait_for(lambda: pressed, 1)
    assert pressed[0] - t0 < 0.05                       # not after the 0.3 s classification window
    assert not gestures
    assert wait_for(lambda: gestures == [TALK], 2)
    g.raw("pause")
    g.raw("next")                                        # a double press: one cut, one REPEAT
    assert wait_for(lambda: gestures == [TALK, REPEAT], 2)
    assert len(pressed) == 2 and max(g.press_to_cut_ms) < 50


# ---------------------------------------------------------------- speaking rate
def test_speaking_rate_steps_clamps_and_persists(spk, offline_speech):
    assert spk.speed == 1.1
    assert spk.adjust_speed(+1) == "Faster." and spk.speed == 1.2
    for _ in range(6):
        spk.adjust_speed(+1)
    assert spk.speed == speech.SPEED_MAX
    assert spk.adjust_speed(+1) == "That's as fast as I go."
    assert spk.adjust_speed(0) == "Normal speed." and spk.speed == 1.1
    spk.adjust_speed(-1)
    again = Speaker(Engine(), SpeechCfg(cache_dir=offline_speech / "cache"), key=None, log=lambda *_: None)
    assert again.speed == 1.0                            # remembered across runs
    assert json.loads((offline_speech / "voice_prefs.json").read_text()) == {"speed": 1.0}
    for _ in range(10):
        again.adjust_speed(-1)
    assert again.speed == speech.SPEED_MIN and again.adjust_speed(-1) == "That's as slow as I go."


def test_rate_change_never_makes_a_cached_warning_wait(spk):
    """'Stop.' recorded at the default rate keeps playing instantly at any new rate."""
    spk.cache.put("Stop.", np.zeros(4800, np.float32))
    spk.set_speed(1.4)
    pcm, kind = spk.cache.lookup("Stop.")
    assert pcm is not None and kind == "stale"
    spk.say("Stop.", "critical")
    assert wait_for(lambda: spk.history, 2)
    assert spk.stats["cache_hits"] == 1 and spk.stats["stale_rate_hits"] == 1 and spk.stats["offline"] == 0


def test_rate_reaches_cartesia_and_the_backup_voice(tmp_path, monkeypatch, offline_speech):
    wpms = []
    real = speech.offline_tts
    monkeypatch.setattr(speech, "offline_tts", lambda t, r, v="", wpm=210: (wpms.append(wpm), real(t, r, v, wpm))[1])
    s, fake, state, out = neural_speaker(tmp_path, monkeypatch)
    try:
        s.set_speed(1.3)
        s.say("Here is a fresh answer.", "answer")
        assert wait_for(lambda: any(m.get("transcript") == "Here is a fresh answer." for m in fake.sent), 3)
        req = next(m for m in fake.sent if m.get("transcript") == "Here is a fresh answer.")
        assert req["generation_config"] == {"speed": 1.3}
        assert speech.offline_wpm(1.3) == 247 and speech.offline_wpm(1.1) == 209
    finally:
        s.stop()
        out.stop()


# ---------------------------------------------------------------- pronunciation
@pytest.mark.parametrize("text,wire", [
    ("Head toward Lyon Street, slightly left.", "Head toward Lion Street, slightly left."),
    ("Turn right onto Gough Street.", "Turn right onto Goff Street."),
    ("Continue on Kearny.", "Continue on Kerny."),
    ("Lyon is a city in France.", "Lyon is a city in France."),      # only the street
    ("Stop.", "Stop."),
])
def test_street_names_are_respelled_for_the_synthesiser(text, wire):
    assert pronounce(text) == wire
    assert pronounce(pronounce(text)) == wire                         # idempotent (the cache keys on it)


def test_pronunciation_changes_the_audio_not_the_record(spk, monkeypatch):
    heard = []
    real = speech.offline_tts
    monkeypatch.setattr(speech, "offline_tts", lambda t, r, v="", wpm=210: (heard.append(t), real(t, r, v, wpm))[1])
    spk.say("Head toward Lyon Street.", "nav")
    assert wait_for(lambda: spk.history, 3)
    assert heard == ["Head toward Lion Street."]
    assert spk.history[0]["text"] == "Head toward Lyon Street."


# ---------------------------------------------------------------- latency breakdown
def test_voice_turns_split_the_time_by_stage():
    t = VoiceTurns()
    t.begin(t_voice_end=10.0, t_endpoint=10.7)
    t.transcript(10.8, "louder")
    t.reply_queued("u1", 10.81)
    assert t.audio("other", 10.9, "cartesia") is None              # not the reply: ignored
    rec = t.audio("u1", 10.95, "cartesia")
    assert rec["endpoint_ms"] == 700.0 and rec["stt_ms"] == 100.0
    assert rec["brain_ms"] == 10.0 and rec["tts_ms"] == 140.0 and rec["total_ms"] == 950.0
    t.begin(20.0, 20.7)                                              # "stop": no spoken reply
    t.begin(30.0, 30.4)
    s = t.summary()
    assert s["turns"] == 1 and s["abandoned"] == 1 and s["median"]["total_ms"] == 950.0


class FakeSource:
    """Recorded audio in real time: `lead` s of room noise, `voiced` s of speech, then quiet."""

    def __init__(self, rate=16000, lead=0.3, voiced=0.6, tail=3.0, block_s=0.05):
        self.rate, self.block = rate, int(rate * block_s)
        rng = np.random.default_rng(0)
        t = np.arange(int(voiced * rate)) / rate
        parts = [0.001 * rng.standard_normal(int(lead * rate)), 0.2 * np.sin(2 * np.pi * 220 * t),
                 0.001 * rng.standard_normal(int(tail * rate))]
        self.audio = np.concatenate(parts).astype(np.float32)
        self.voice_end = lead + voiced
        self.pos = 0
        self.t0 = None

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


class FakeSTT:
    """Ink-like (timings measured live): a final segment arrives `seg_delay` s after the
    audio goes quiet on its own, or `finalize_s` after a (speculative) finalize."""

    def __init__(self, text, rate=16000, seg_delay=0.4, finalize_s=0.15):
        self.text, self.rate, self.seg_delay, self.finalize_s = text, rate, seg_delay, finalize_s
        self.finalized = False
        self.flushes = 0

    async def transcribe_stream(self, chunks, timeout_s=4.0, on_segment=None):
        voiced, quiet, sent = False, 0.0, 0.0
        segmented = False
        pending: list = []                   # hold flush tasks: the loop keeps only weak references

        async def flushed():
            nonlocal segmented
            await asyncio.sleep(self.finalize_s)
            if voiced and not segmented and on_segment is not None:
                segmented = True
                on_segment(self.text, sent)

        while True:
            c = await chunks.get()
            if c is None:
                break
            if c == FLUSH:
                self.flushes += 1
                pending.append(asyncio.ensure_future(flushed()))
                continue
            a = np.frombuffer(c, dtype="<i2").astype(np.float32) / 32767
            sent += a.size / self.rate
            if np.sqrt(np.mean(a ** 2)) > 0.02:
                voiced, quiet = True, 0.0
            elif voiced:
                quiet += a.size / self.rate
                if quiet >= self.seg_delay and not segmented and on_segment is not None:
                    segmented = True
                    on_segment(self.text, sent)
        self.finalized = True
        await asyncio.sleep(self.finalize_s)
        return self.text


def make_listener(text, monkeypatch, **src):
    got, cues, listening = [], [], []
    lst = Listener(ListenCfg(), key=None, on_text=got.append, cue=cues.append, on_listening=listening.append,
                   say=lambda *a, **k: got.append(("say", a[0])), log=lambda *_: None)
    lst.stt = FakeSTT(text)
    source = FakeSource(**src)
    lst._open_source = lambda: source
    lst.turns = VoiceTurns()
    return lst, got, cues, listening, source


def test_short_command_ends_early_without_waiting_for_finalize(monkeypatch):
    lst, got, cues, listening, src = make_listener("Louder.", monkeypatch)
    partials = []
    lst.on_partial = partials.append
    lst.push_to_talk()
    assert wait_for(lambda: got, 5)
    assert got == ["Louder."] and partials == ["Louder."]
    assert lst.stats["early_ends"] == 1
    o = lst.turns._open
    assert o["early"] is True and lst.stt.flushes == 1
    assert 300 <= (o["t_end"] - o["t_voice"]) * 1000 < 650     # ~0.5 s of quiet, not 0.7 s
    assert (o["t_text"] - o["t_end"]) * 1000 < 50               # the segment was already in hand
    assert listening == [True, False]                           # released before the socket closed


def test_prefetch_is_asked_without_the_wake_word(monkeypatch):
    lst, got, cues, listening, src = make_listener("Kite, how do I get home?", monkeypatch)
    partials = []
    lst.on_partial = partials.append
    lst.push_to_talk()
    assert wait_for(lambda: got, 6)
    assert got == ["how do i get home?"] and partials == ["how do i get home?"]   # same words, same Moss query


def test_open_question_waits_for_the_full_silence(monkeypatch):
    lst, got, cues, listening, src = make_listener("What is the weather like today?", monkeypatch)
    lst.push_to_talk()
    assert wait_for(lambda: got, 6)
    assert got == ["What is the weather like today?"] and lst.stats["early_ends"] == 0
    o = lst.turns._open
    assert (o["t_end"] - o["t_voice"]) * 1000 >= 650             # the normal 0.7 s end-of-speech
    assert lst.stt.finalized


def test_hands_free_wake_word_barges_in_before_the_sentence_ends(monkeypatch):
    lst, got, cues, listening, src = make_listener("Kite, what's around me?", monkeypatch)
    lst.stt.seg_delay = 0.1
    t_listen = []
    lst.on_listening = lambda on: (listening.append(on), t_listen.append(time.monotonic()))
    t0 = time.monotonic()
    lst._active.set()
    asyncio.run(lst._hf_once())
    assert listening == [True, False] and got == ["what's around me?"]
    assert lst.stats["voice_barge_ins"] == 1
    assert t_listen[0] - t0 < src.voice_end + 0.4           # cut while the endpointer still waited
    assert t_listen[1] - t_listen[0] > 0.3


def test_full_voice_loop_latency_shows_up_in_state(app):
    """Wearer says 'louder' -> early end -> local intent -> 'Louder.' -> state() breakdown."""
    app.listener.stt = FakeSTT("Kite, louder.")
    app.listener._open_source = lambda: FakeSource()
    app.gesture(TALK)
    assert wait_for(lambda: "Louder." in [h["text"] for h in list(app.speaker.history)], 8)
    loop = app.state()["speech"]["voice_loop"]
    last = loop["last"]
    assert loop["turns"] == 1 and last["early_end"] is True and last["source"] == "offline"
    for stage in VoiceTurns.STAGES:
        assert last[stage] is not None and last[stage] >= 0
    assert last["total_ms"] == pytest.approx(last["endpoint_ms"] + last["stt_ms"] + last["brain_ms"]
                                             + last["tts_ms"], abs=1.0)
    assert last["brain_ms"] < 200
    assert app.state()["listen"]["early_ends"] == 1


def test_talk_faster_end_to_end(app):
    app.heard("talk a bit faster")
    assert wait_for(lambda: "Faster." in [h["text"] for h in list(app.speaker.history)], 5)
    assert app.speaker.speed == 1.2 and app.state()["speech"]["speed"] == 1.2
    assert app.control("speak_normal") == "Normal speed." and app.speaker.speed == 1.1


@needs_anthropic
def test_pressing_to_talk_stops_the_answer_being_written(app):
    said = []
    app.brain.speak = lambda t, p, **k: said.append(t)
    app.brain._client = FakeClaude(["One. ", "Two. ", "Three. ", "Four."], delay=0.15)
    app.brain.handle("tell me about this park")
    assert wait_for(lambda: said, 3)
    app._on_listening(True)                                  # what a press does
    time.sleep(0.6)
    app._on_listening(False)
    assert said == ["One."]


# ---------------------------------------------------------------- brain
class FakeClaude:
    """Streams text parts with a delay, like Messages streaming, and remembers if closed."""

    def __init__(self, parts, delay=0.05):
        self.parts, self.delay, self.calls, self.closed = parts, delay, 0, 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kw):
        self.calls += 1
        outer = self

        class S:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                outer.closed += 1

            def __aiter__(self):
                async def gen():
                    for p in outer.parts:
                        await asyncio.sleep(outer.delay)
                        yield SimpleNamespace(type="text", text=p)
                return gen()

            async def get_final_message(self):
                return SimpleNamespace(stop_reason="end_turn",
                                       content=[SimpleNamespace(type="text", text="".join(outer.parts))])
        return S()


def llm_brain(parts, delay=0.05, recall=None):
    said, recalls = [], []

    def rec(q):
        recalls.append(q)
        return (recall or (lambda q: [{"kind": "place", "text": "Home: 3601 Lyon St", "score": 0.9}]))(q)

    a = Actions(navigate=lambda d: "ok", control=lambda x: x, alert_caregiver=lambda m: "sent",
                stop_speaking=lambda: None, ear_check=lambda: None, recall=rec)
    b = Brain(lambda: {}, a, lambda t, p, **k: said.append(t), None, log=lambda *_: None)
    b._client = FakeClaude(parts, delay)
    return b, said, recalls


@needs_anthropic
def test_barge_in_supersedes_the_streaming_answer():
    b, said, _ = llm_brain(["First sentence. ", "Second sentence. ", "Third sentence."], delay=0.12)
    b.handle("tell me a story")
    assert wait_for(lambda: said, 3)
    b.interrupt()
    time.sleep(0.6)
    assert said == ["First sentence."]
    assert b.stats["interrupted"] == 1 and b._client.closed == 1
    assert b.history == []                                   # an abandoned answer is not "said"


@needs_anthropic
def test_a_new_question_supersedes_the_old_answer():
    b, said, _ = llm_brain(["Old one. ", "Old two. ", "Old three."], delay=0.12)
    b.handle("first question")
    assert wait_for(lambda: said, 3)
    assert b.handle_sync("louder") == "local:volume_up"
    time.sleep(0.6)
    assert "Old two." not in said and "Old three." not in said


@needs_anthropic
def test_memories_prefetched_while_talking_are_reused():
    b, said, recalls = llm_brain(["Your home is on Lyon Street."])
    b.prefetch("how do I get home")
    assert wait_for(lambda: recalls, 2)
    assert b.handle_sync("How do I get home?") == "llm"
    assert recalls == ["how do I get home"]                 # one Moss query, done while talking
    assert b.stats["recall_prefetch_hits"] == 1 and b.stats["recall_ms"] is not None


@needs_anthropic
def test_slow_memory_never_holds_the_answer():
    b, said, _ = llm_brain(["Here you go."], recall=lambda q: (time.sleep(1.5), [])[1])
    t0 = time.monotonic()
    assert b.handle_sync("tell me about my home") == "llm"
    assert time.monotonic() - t0 < 1.0 and said == ["Here you go."]
    assert b.stats["recall_timeouts"] == 1


@pytest.mark.parametrize("q,intent", [
    ("talk faster", "speak_faster"), ("speak a little slower", "speak_slower"), ("slow down", "speak_slower"),
    ("faster please", "speak_faster"), ("you're talking too fast", "speak_slower"), ("normal speed", "speak_normal"),
    ("read slower", "speak_slower"), ("speak up", "volume_up"), ("stop", "stop_speaking"),
])
def test_speaking_rate_intents(q, intent):
    assert classify(q)[0] == intent


@pytest.mark.parametrize("text,quick", [
    ("Stop.", True), ("Kite, louder.", True), ("talk faster", True), ("Repeat that.", True),
    ("Pause.", True), ("take me to the library", False), ("what's around me", False),
    ("stop the music", False), ("remember that the cafe has a step", False), ("", False),
    ("what is the meaning of life and everything else in it", False),
])
def test_quick_commands(text, quick):
    assert quick_command(text) is quick


def test_claude_can_change_the_speaking_rate():
    enum = next(t for t in TOOLS if t["name"] == "control")["input_schema"]["properties"]["action"]["enum"]
    assert {"speak_faster", "speak_slower", "speak_normal"} <= set(enum)
    b, _, _ = llm_brain([])
    assert b._run_tool("control", {"action": "speak_slower"}) == ("speak_slower", False)


# ---------------------------------------------------------------- Cartesia edge cases
def test_offline_voice_never_inherits_stdin(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        raise subprocess.TimeoutExpired(cmd, 15)

    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/say")
    with pytest.raises(subprocess.TimeoutExpired):
        speech.offline_tts("Hello.", 48000)
    assert seen["stdin"] is subprocess.DEVNULL


def test_cartesia_error_codes(tmp_path, monkeypatch):
    async def run(error):
        cs = CartesiaStream(SpeechCfg(cache_dir=tmp_path), "k", 48000)
        fake = FakeTTSSocket(cs)
        fake.error = error

        async def connect():
            return fake
        cs._connect = connect
        try:
            async for _ in cs.stream("Hello.", "ctx"):
                pass
        except ConcurrencyLimited:
            return cs, "concurrency"
        except RuntimeError:
            return cs, "error"
        return cs, "ok"

    cs, how = asyncio.run(run({"error_code": "concurrency_limited", "status_code": 429,
                               "title": "Too many concurrent requests"}))
    assert how == "concurrency" and cs.available                      # transient: no backoff
    cs, how = asyncio.run(run({"error_code": "quota_exceeded", "message": "Out of credits"}))
    assert how == "error" and cs.quota_exhausted and cs.down_until - time.monotonic() > 200


def test_connect_failure_backs_off_instead_of_retrying_every_line(tmp_path):
    async def run():
        cs = CartesiaStream(SpeechCfg(cache_dir=tmp_path), "k", 48000)

        async def connect():
            raise OSError("no route to host")
        cs._connect = connect
        with pytest.raises(RuntimeError):
            async for _ in cs.stream("Hello.", "ctx"):
                pass
        return cs
    cs = asyncio.run(run())
    assert not cs.available and cs.failures == 1 and "connect" in cs.last_error


def test_request_carries_rate_and_pronunciation_dictionary(tmp_path):
    cs = CartesiaStream(SpeechCfg(cache_dir=tmp_path), "k", 48000, pronunciation_dict="pdict_123")
    r = cs.request("Hi.", "c1", speed=1.25)
    assert r["generation_config"] == {"speed": 1.25} and r["pronunciation_dict_id"] == "pdict_123"
    assert r["model_id"] == "sonic-3.6" and r["output_format"]["encoding"] == "pcm_f32le"
    assert "pronunciation_dict_id" not in CartesiaStream(SpeechCfg(cache_dir=tmp_path), "k", 48000).request("Hi.", "c")


def test_outage_switches_voice_once_and_recovers_by_probing(tmp_path, monkeypatch, offline_speech):
    s, fake, state, out = neural_speaker(tmp_path, monkeypatch)
    try:
        s.say("First answer, over the network.", "answer")
        assert wait_for(lambda: s.online is True and not s.busy, 3)
        assert s.stats["tts_first_byte_ms"] is not None and s.latency()["first_audio_ms"] is not None
        state["down"] = True                                  # the internet goes away
        s.tts._ws = None
        s.say("Second answer, the network just failed.", "answer")
        assert wait_for(lambda: VOICE_OFFLINE in [h["text"] for h in list(s.history)], 5)
        assert s.online is False and s.stats["offline"] >= 1
        sent = len(fake.sent)
        state["down"] = False                                 # and comes back
        assert wait_for(lambda: VOICE_BACK in [h["text"] for h in list(s.history)], 5)   # found by the probe
        assert fake.sent[sent]["transcript"] == VOICE_BACK    # the probe itself synthesised nothing
        assert s.online is True and s.tts.available
        texts = [h["text"] for h in list(s.history)]
        assert texts.count(VOICE_OFFLINE) == 1 and texts.count(VOICE_BACK) == 1
    finally:
        s.stop()
        out.stop()


def test_dead_network_costs_no_connect_timeout_after_the_first_failure(tmp_path, monkeypatch, offline_speech):
    s, fake, state, out = neural_speaker(tmp_path, monkeypatch)
    try:
        state.update(down=True, connect_s=1.0)
        t0 = time.monotonic()
        s.say("An answer while offline.", "answer")
        assert wait_for(lambda: s.history, 6)
        first_s = time.monotonic() - t0
        assert not s.tts.available and s.online is False
        t0 = time.monotonic()
        s.say("Another answer while offline.", "answer")
        assert wait_for(lambda: len(s.history) >= 2, 6)
        second_s = time.monotonic() - t0
        # the first line paid the hung handshake once; the next goes straight to the backup voice
        assert first_s > 1.0 and second_s < first_s - 0.6, (first_s, second_s)
        assert [h["result"] for h in list(s.history)] == ["done", "done"]
    finally:
        s.stop()
        out.stop()


def test_probe_lifts_a_connect_backoff_as_soon_as_the_network_is_back(tmp_path, monkeypatch, offline_speech):
    s, fake, state, out = neural_speaker(tmp_path, monkeypatch)
    try:
        state["down"] = True
        s.tts._ws = None
        s.prewarm(["Kite is ready."])                  # fails during warm-up: nothing was said yet
        assert wait_for(lambda: not s.tts.available, 3)
        assert s.tts.down_until - time.monotonic() > 20  # a long backoff ...
        state["down"] = False
        assert wait_for(lambda: s.tts.available, 2)     # ... lifted by the free handshake
    finally:
        s.stop()
        out.stop()


def test_uncached_warning_uses_the_backup_voice_when_cartesia_is_slow(tmp_path, monkeypatch, offline_speech):
    s, fake, state, out = neural_speaker(tmp_path, monkeypatch, ttfb=3.0)
    try:
        t0 = time.monotonic()
        s.say("Scaffolding ahead on the right.", "hazard")
        assert wait_for(lambda: s.history, 4)
        assert results(s) == [("Scaffolding ahead on the right.", "done")]
        assert s.stats["urgent_backup"] == 1 and s.stats["offline"] == 1
        assert time.monotonic() - t0 < 2.8               # ~1 s deadline + the line itself, not 3 s + line
        assert s.online is not False                      # one slow reply is not an outage
        assert s.tts.available
        assert any(m.get("cancel") for m in fake.sent)
    finally:
        s.stop()
        out.stop()


def test_rate_change_rerecords_the_fixed_phrases_in_the_background(tmp_path, monkeypatch, offline_speech):
    s, fake, state, out = neural_speaker(tmp_path, monkeypatch, ttfb=0.0, chunks=1, chunk_s=0.01)
    try:
        s.set_speed(1.3)
        assert wait_for(lambda: all(s.cache.has_neural(p) for p in map(pronounce, FIXED_PHRASES)), 15)
        warm = [m for m in fake.sent if m.get("context_id", "").startswith("warm-")]
        assert {m["generation_config"]["speed"] for m in warm} == {1.3}
    finally:
        s.stop()
        out.stop()


def test_an_outage_that_ends_unheard_is_never_announced(tmp_path, monkeypatch, offline_speech):
    s, fake, state, out = neural_speaker(tmp_path, monkeypatch)
    try:
        s.say("First answer, over the network.", "answer")
        assert wait_for(lambda: s.online is True and not s.busy, 3)
        s.paused = True                                     # hold the system notice in the queue
        s._set_online(False)
        assert any(q.key == "voice-offline" for q in s._queue)
        s._set_online(True)                                 # back before the wearer heard about it
        assert not any(q.key in ("voice-offline", "voice-back") for q in s._queue)
    finally:
        s.stop()
        out.stop()


class FakeInkSocket:
    """Ink's STT WebSocket: each `finalize` flushes the audio received since the last one
    as a final segment, then answers flush_done; `close` answers done."""

    def __init__(self, words):
        self.words, self.sent, self.inbox = list(words), [], asyncio.Queue()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def send(self, m):
        self.sent.append(m if isinstance(m, str) else "audio")
        if m == "finalize":
            await asyncio.sleep(0.05)
            if self.words:
                await self.inbox.put({"type": "transcript", "is_final": True, "text": " " + self.words.pop(0),
                                      "duration": 0.5})
            await self.inbox.put({"type": "flush_done"})
        elif m == "close":
            await self.inbox.put({"type": "done"})

    def __aiter__(self):
        return self

    async def __anext__(self):
        return json.dumps(await self.inbox.get())


def test_speculative_flush_keeps_the_stream_open(monkeypatch):
    import websockets
    from earpiece.listen import CartesiaSTT
    sock = FakeInkSocket(["Take me", "to the library."])
    monkeypatch.setattr(websockets, "connect", lambda *a, **k: sock)
    segments = []

    async def run():
        q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(CartesiaSTT("k", "ink-whisper", 16000).transcribe_stream(
            q, on_segment=lambda t, c: segments.append((t, c))))
        await q.put(b"\0\0" * 1600)
        await q.put(FLUSH)                       # the wearer paused after "take me"
        await q.put(b"\0\0" * 1600)
        await q.put(None)                        # ... then finished
        return await asyncio.wait_for(task, 3)

    assert asyncio.run(run()) == "Take me to the library."
    assert segments == [("Take me", 0.5), ("Take me to the library.", 1.0)]
    assert sock.sent == ["audio", "finalize", "audio", "finalize", "close"]
