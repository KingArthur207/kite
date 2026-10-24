import time

import numpy as np
import pytest

from conftest import wait_for
from earpiece.brain import Actions, Brain, strip_wake
from earpiece.config import SpeechCfg
from earpiece.mixer import Engine
from earpiece.output import NullOutput
from earpiece.speech import PhraseCache, Speaker, safety_phrases


@pytest.fixture
def spk(offline_speech):
    eng = Engine()
    out = NullOutput(eng, 1024, realtime=True)
    out.start()
    cfg = SpeechCfg(cache_dir=offline_speech / "cache")
    s = Speaker(eng, cfg, key=None, log=lambda *_: None)
    s.start()
    yield s
    s.stop()
    out.stop()


def results(s):
    return [(h["text"], h["result"]) for h in s.history]


def test_priority_order(spk):
    spk.paused = True                 # hold the queue while we fill it
    spk.say("info one", "info")
    spk.say("nav one", "nav")
    spk.say("hazard one", "hazard")
    spk.paused = False
    spk._kick()
    assert wait_for(lambda: len(spk.history) == 3, 5)
    # paused lets hazards through first; then nav before info
    assert [h["text"] for h in spk.history] == ["hazard one", "nav one", "info one"]


def test_critical_preempts_and_nav_is_requeued(spk):
    spk.say("this is a long navigation instruction", "nav", ttl=10)
    assert wait_for(lambda: spk._current is not None, 2)
    time.sleep(0.05)
    spk.say("Stop.", "critical")
    assert wait_for(lambda: len(spk.history) >= 3, 5)
    r = results(spk)
    assert r[0] == ("this is a long navigation instruction", "cut")
    assert r[1] == ("Stop.", "done")
    assert r[2] == ("this is a long navigation instruction", "done")
    assert spk.stats["preempted"] == 1


def test_answer_is_not_requeued_after_preempt(spk):
    spk.say("a long answer", "answer")
    assert wait_for(lambda: spk._current is not None, 2)
    spk.say("Careful.", "hazard")
    assert wait_for(lambda: len(spk.history) >= 2, 5)
    time.sleep(0.4)
    assert results(spk) == [("a long answer", "cut"), ("Careful.", "done")]


def test_expired_items_are_dropped_not_spoken_late(spk):
    spk.listening = True               # wearer talking: everything non-critical waits
    spk.say("turn left now", "nav", ttl=0.3)
    spk.say("Stop.", "critical")
    time.sleep(0.6)
    spk.listening = False
    spk._kick()
    assert wait_for(lambda: spk.stats["dropped_ttl"] == 1, 3)
    time.sleep(0.3)
    assert [h["text"] for h in spk.history] == ["Stop."]


def test_dedupe_and_key_replacement(spk):
    spk.paused = True
    assert spk.say("Crowd ahead.", "nav", key="crowd") is not None
    spk.say("Crowd ahead.", "nav")                                  # same text: replaces, never doubles
    assert len(spk._queue) == 1
    spk.say("Crowd ahead on the left.", "nav", key="crowd")         # newer version replaces
    spk.paused = False
    spk._kick()
    assert wait_for(lambda: not spk.busy and spk.history, 5)
    assert [h["text"] for h in spk.history] == ["Crowd ahead on the left."]
    assert spk.say("Crowd ahead on the left.", "nav") is None         # just said: deduped


def test_critical_is_never_deduped(spk):
    for _ in range(3):
        spk.say("Stop.", "critical")
        assert wait_for(lambda: not spk.busy, 3)
    assert [h["text"] for h in spk.history] == ["Stop."] * 3


def test_cancel_all_keeps_critical(spk):
    spk.paused = True
    spk.say("info", "info")
    spk.say("Stop.", "critical")
    spk.cancel_all(keep_critical=True)
    spk.paused = False
    spk._kick()
    assert wait_for(lambda: spk.history, 3)
    assert [h["text"] for h in spk.history] == ["Stop."]


def test_repeat_last_nav(spk):
    spk.say("Turn right onto Bay Street.", "nav")
    assert wait_for(lambda: spk.last_nav is not None, 3)
    assert spk.repeat_last()
    assert wait_for(lambda: len(spk.history) == 2, 3)


def test_phrase_cache_roundtrip(tmp_path):
    c = PhraseCache(SpeechCfg(cache_dir=tmp_path), 48000)
    pcm = np.linspace(-0.5, 0.5, 1000).astype(np.float32)
    c.put("Stop.", pcm)
    c2 = PhraseCache(SpeechCfg(cache_dir=tmp_path), 48000)
    assert np.array_equal(c2.get("Stop."), pcm)
    assert c2.get("unknown") is None
    c2.put("Offline only.", pcm, offline=True)
    assert c2.get("Offline only.") is not None and not c2.has_neural("Offline only.")


def test_safety_phrases_cover_every_sonar_label():
    from earpiece.events import clock_phrase
    from earpiece.sonar import _SPOKEN_KIND
    ph = set(safety_phrases())
    for name in set(_SPOKEN_KIND.values()) | {"Obstacle"}:
        for b in range(-180, 181, 3):
            assert f"{name}, {clock_phrase(b)}." in ph
    for s in ("Stop.", "Stop. Person ahead.", "Stop. Obstacle ahead.", "Stop. Car."):
        assert s in ph


def test_cached_phrase_plays_without_synthesis(spk):
    spk.cache.put("Stop.", np.zeros(4800, np.float32))
    spk.say("Stop.", "critical")
    assert wait_for(lambda: spk.history, 2)
    assert spk.stats["cache_hits"] == 1 and spk.stats["offline"] == 0


# ---------------------------------------------------------------- brain
def make_brain(ctx=None):
    said, acts = [], []
    a = Actions(navigate=lambda d: (acts.append(("nav", d)), f"Okay, finding a route to {d}.")[1],
                control=lambda x: (acts.append(x), x)[1], alert_caregiver=lambda m: (acts.append(("help", m)), "sent")[1],
                stop_speaking=lambda: acts.append("stop"), ear_check=lambda: acts.append("ears"))
    b = Brain(ctx or (lambda: {}), a, lambda t, p, **k: said.append(t), None, log=lambda *_: None)
    return b, said, acts


@pytest.mark.parametrize("q,intent", [
    ("Take me to Crissy Field", "navigate"), ("navigate to 3601 Lyon Street", "navigate"),
    ("stop", "stop_speaking"), ("be quiet", "stop_speaking"), ("say that again", "repeat"),
    ("pause", "pause"), ("resume guidance", "resume"), ("mute the beeps", "mute_cues"),
    ("turn the beeps back on", "unmute_cues"), ("louder", "volume_up"), ("too loud", "volume_down"),
    ("turn off the beacon", "beacon_off"), ("cancel the route", "cancel_route"), ("check my ears", "ear_check"),
    ("what's around me", "around"), ("is anyone in front of me", "around"), ("where am I", "where"),
    ("how far is it", "howfar"),
])
def test_local_intents(q, intent):
    b, said, acts = make_brain()
    assert b.handle_sync(q) == f"local:{intent}"


def test_navigation_keeps_destination_casing():
    b, said, acts = make_brain()
    b.handle_sync("Kite take me to the Palace of Fine Arts.")
    assert ("nav", "Palace of Fine Arts") in acts


def test_offline_description_uses_context():
    ctx = lambda: {"nearby": [{"kind": "person", "bearing": -20, "distance": 2.2, "urgency": 0.5}],
                   "hazards": [{"text": "Don't walk."}], "route": None}
    b, said, _ = make_brain(ctx)
    b.handle_sync("what's around me")
    assert said == ["A person slightly left, 2 meters; Don't walk."]


def test_offline_description_when_camera_down():
    b, said, _ = make_brain(lambda: {"nearby": [], "perception_ok": False})
    b.handle_sync("what's ahead")
    assert "not running" in said[0]


def test_unknown_question_offline_is_honest():
    b, said, _ = make_brain()
    assert b.handle_sync("what is the meaning of life") == "fallback"
    assert "internet" in said[0]


def test_tool_validation():
    b, said, acts = make_brain()
    assert b._run_tool("control", {"action": "self_destruct"})[1] is True
    assert b._run_tool("navigate_to", {"destination": ""})[1] is True
    assert b._run_tool("navigate_to", "oops")[1] is True
    assert b._run_tool("alert_caregiver", {"message": "I'm lost"}) == ("sent", False)
    assert b._run_tool("nope", {})[1] is True


@pytest.mark.parametrize("text,woke,rest", [
    ("Kite, take me home", True, "take me home"), ("Hey Kite what's ahead", True, "what's ahead"),
    ("Kyte. stop", True, "stop"), ("ok kite louder", True, "louder"), ("take me home", False, "take me home"),
    ("kitchen is that way", False, "kitchen is that way"),
])
def test_wake_word(text, woke, rest):
    assert strip_wake(text, ()) == (woke, rest)
