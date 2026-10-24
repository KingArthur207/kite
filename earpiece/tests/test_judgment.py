"""Claude judgment layer: guardrails with a fake client (no network), plus one optional live call."""

import json
import time
from types import SimpleNamespace

import pytest

from conftest import LIVE_ANTHROPIC_KEY, wait_for
from earpiece.client import EarpieceClient
from earpiece.judgment import Judgment, brief, parse_reply, validate


class FakeClient:
    """Stands in for anthropic.Anthropic: replies with a scripted decision as Claude's one-line text
    ("-" for silence, "!key ..." for an upgrade)."""

    def __init__(self, decision=None, delay=0.0, error=None, stop_reason="end_turn"):
        self.decision = decision or {}
        self.delay = delay
        self.error = error
        self.stop_reason = stop_reason
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        d = self.decision
        line = " ".join(x for x in (f"!{d['upgrade_key']}" if d.get("upgrade_key") else "", d.get("say", "")) if x)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=line or "-")])


HAZ = {"hazards": [{"key": "crowd", "kind": "crowd", "text": "Crowd ahead.", "priority": "nav"}],
       "scene": "A busy crosswalk."}


def make(decision=None, snap=None, **kw):
    said, events, ups = [], [], []
    state = {"snap": snap or HAZ}
    j = Judgment(lambda: state["snap"], lambda *a, **k: said.append((a, k)),
                 lambda key, p: ups.append((key, p)) or True, lambda t, d: events.append((t, d)),
                 client=FakeClient(decision, **{k: v for k, v in kw.items() if k in ("delay", "error", "stop_reason")}),
                 log=lambda *_: None, **{k: v for k, v in kw.items() if k in ("min_gap_s", "timeout_s")})
    j.stop()                                   # drive step() by hand
    return j, said, events, ups, state


@pytest.mark.parametrize("text", [
    "It is safe to cross now.", "Go ahead, the light is green.", "You can cross.", "The path is clear.",
    "Keep walking.", "Proceed to the corner.", "You don't need to stop.", "Take a detour to the left.",
    "All clear.", "It's clear to go."])
def test_permission_and_reroutes_are_never_spoken(text):
    j, said, events, _, _ = make({"say": text})
    j.step()
    assert said == [] and events == []
    assert j.stats["rejected"] == 1


def test_a_warning_is_spoken_at_hazard_priority_and_mirrored_as_say():
    j, said, events, _, _ = make({"say": "Wait, the crosswalk is crowded.", "warning": True})
    assert j.step() == "said"
    (args, kw), = said
    assert args == ("Wait, the crosswalk is crowded.", "hazard") and kw["ttl"] <= 5
    (kind, d), = events
    assert kind == "say" and d["source"] == "judgment" and d["priority"] == "hazard"


def test_reply_line_parsing():
    assert parse_reply("-")["say"] == "" and parse_reply("")["say"] == ""
    assert parse_reply('"Wait. Car turning in from your right."\n')["say"] == "Wait. Car turning in from your right."
    d = parse_reply("!crowd Crowd on the crossing. Hold here.\nextra line")
    assert (d["say"], d["upgrade_key"], d["upgrade_priority"]) == ("Crowd on the crossing. Hold here.", "crowd", "hazard")
    assert parse_reply("!crowd")["say"] == "" and parse_reply("!crowd")["upgrade_key"] == "crowd"
    assert validate(parse_reply("Sign ahead says use other side."), {})[0] is None   # echoing a sign is a reroute


def test_a_reply_cut_off_at_max_tokens_is_never_spoken():
    j, said, events, *_ = make({"say": "Wait. Car turning in", "warning": True}, stop_reason="max_tokens")
    assert j.step() == "none" and said == [] and events == []


def test_upgrade_does_not_trigger_another_call():
    j, _, _, _, state = make({"upgrade_key": "crowd"})
    j.step()
    state["snap"] = {**HAZ, "hazards": [{**HAZ["hazards"][0], "priority": "hazard"}]}
    assert j.step() == "unchanged" and len(j.client.calls) == 1


def test_claude_sees_a_brief_snapshot_with_recent_speech():
    snap = {"hazards": [{"key": "vehicle", "kind": "vehicle", "text": "Car at 2 o'clock, 9 meters.", "bearing": 55.04,
                         "distance": 9.0, "state": None, "priority": "hazard"}],
            "nearby": [{"kind": "bike", "bearing": 3.7, "distance": 6.23, "urgency": 0.1}], "scene": "A car turns in.",
            "route": {"label": "Howard Street", "bearing": 0.0, "distance": 26.4, "region": "ahead"},
            "last_instruction": {"text": "Cross Moss Street.", "seconds_ago": 2}, "drone": {"battery": 60},
            "position": {"lat": 37.7, "lng": -122.4, "place": "Moss St at Howard St"}, "local_time": "14:02",
            "recent_speech": ["Stop."], "guidance_paused": False}
    b = brief(snap)
    assert b == {"hazards": [{"key": "vehicle", "kind": "vehicle", "text": "Car at 2 o'clock, 9 meters.", "bearing": 55,
                              "distance": 9, "priority": "hazard"}],
                 "nearby": [{"kind": "bike", "bearing": 4, "distance": 6.2}], "scene": "A car turns in.",
                 "route": {"to": "Howard Street", "distance": 26, "last_instruction": "Cross Moss Street."},
                 "place": "Moss St at Howard St", "recent_speech": ["Stop."]}
    j, *_ = make(snap=snap)
    j.step()
    assert json.loads(j.client.calls[0]["messages"][0]["content"]) == b


def test_a_slow_answer_is_raced_by_a_second_request():
    class Slow(FakeClient):
        def create(self, **kw):
            self.delay = 0.6 if not self.calls else 0.0   # first request stuck in the slow tail
            return super().create(**kw)

    said = []
    j = Judgment(lambda: HAZ, lambda *a, **k: said.append(a), lambda *a: True, lambda *a: None,
                 client=Slow({"say": "Crowd on the crossing. Hold here."}), hedge_s=0.1, timeout_s=0.5,
                 log=lambda *_: None)
    j.stop()
    t0 = time.monotonic()
    assert j.step() == "said" and time.monotonic() - t0 < 0.4
    assert j.stats["hedged"] == 1 and len(j.client.calls) == 2


def test_the_same_line_is_not_said_twice_in_a_row():
    j, said, _, _, state = make({"say": "Listen for turning cars first."})
    j.step()
    state["snap"] = {**HAZ, "scene": "Still a busy crosswalk."}
    j.step()
    assert len(said) == 1 and j.stats["repeats"] == 1
    assert "Listen for turning cars first." in json.loads(j.client.calls[1]["messages"][0]["content"])["recent_speech"]


def test_app_context_lists_what_was_just_said(app):
    app.say("Stop.", "critical")
    assert wait_for(lambda: "Stop." in app.context()["recent_speech"], 3)


def test_too_long_is_rejected():
    j, said, *_ = make({"say": "Wait. " * 30, "warning": True})
    j.step()
    assert said == []


def test_downgrades_unknown_keys_and_critical_are_rejected():
    hz = {"crowd": "hazard"}
    assert validate({"upgrade_key": "crowd", "upgrade_priority": "info"}, hz)[1] is None        # down
    assert validate({"upgrade_key": "crowd", "upgrade_priority": "hazard"}, hz)[1] is None      # same
    assert validate({"upgrade_key": "ghost", "upgrade_priority": "hazard"}, hz)[1] is None      # no key
    assert validate({"upgrade_key": "crowd", "upgrade_priority": "critical"}, hz)[1] is None    # sonar only
    assert validate({"upgrade_key": "crowd", "upgrade_priority": "hazard"}, {"crowd": "nav"})[1] == ("crowd", "hazard")


def test_upgrade_goes_through_the_app_hook():
    j, _, _, ups, _ = make({"upgrade_key": "crowd", "upgrade_priority": "hazard"})
    assert j.step() == "upgraded"
    assert ups == [("crowd", "hazard")]


def test_no_call_when_nothing_changed():
    j, _, _, _, state = make({"say": "Wait, a car is turning in from your right.", "warning": True})
    j.step()
    j.step()
    j.step()
    assert len(j.client.calls) == 1
    state["snap"] = {**HAZ, "scene": "A car turning in from the right."}
    j.step()
    assert len(j.client.calls) == 2
    j2, *_ = make(snap={"hazards": [], "scene": None})
    j2.step()
    assert j2.client.calls == []


def test_slow_or_failing_claude_is_dropped_silently():
    j, said, events, *_ = make({"say": "Wait.", "warning": True}, delay=0.3, timeout_s=0.1)
    assert j.step() == "late" and said == [] and events == []
    j2, said2, *_ = make(error=TimeoutError("read timeout"))
    assert j2.step() == "dropped" and said2 == []


def test_rate_limit_coalesces_a_burst_of_changes():
    said = []
    n = {"i": 0}

    def snap():
        n["i"] += 1
        return {"hazards": [{"key": "crowd", "text": f"Crowd {n['i']}", "priority": "nav"}], "scene": None}

    fake = FakeClient({"say": "Wait.", "warning": True})
    j = Judgment(snap, lambda *a, **k: said.append(a), lambda *a: True, lambda *a: None,
                 client=fake, min_gap_s=0.5, log=lambda *_: None)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.2:
            j.notify()
            time.sleep(0.02)
        time.sleep(0.1)
        assert 2 <= len(fake.calls) <= 3       # ~60 notifications, at most one call per 0.5 s
    finally:
        j.stop()


def test_dont_walk_is_spoken_even_while_claude_hangs(app):
    app.judgment.stop()
    fake = FakeClient({"say": "It's safe to cross.", "warning": False}, delay=3.0)
    app.judgment = Judgment(app.context, app.say, app.upgrade_hazard, app._event,
                            client=fake, min_gap_s=0.1, timeout_s=2.5, log=app.log)
    c = EarpieceClient("127.0.0.1", app.cfg.net.udp_port, source="drone")
    c.hazard("crosswalk", state="dont_walk", bearing=0, distance=6)
    assert wait_for(lambda: "Don't walk." in [h["text"] for h in app.speaker.history], 2)
    assert wait_for(lambda: fake.calls, 2)          # Claude was asked, and is still thinking
    assert app.judgment.stats["said"] == 0
    time.sleep(3.2)
    assert app.judgment.stats["dropped"] == 1        # answered late: dropped, never spoken
    assert "It's safe to cross." not in [h["text"] for h in app.speaker.history]
    app.judgment.stop()


def test_app_upgrade_hook_only_raises(app):
    c = EarpieceClient("127.0.0.1", app.cfg.net.udp_port, source="drone")
    c.send({"type": "hazard", "kind": "crowd", "text": "Small crowd ahead.", "priority": "nav", "bearing": 20})
    assert wait_for(lambda: "crowd" in app.hazards, 2)
    assert not app.upgrade_hazard("crowd", "info")
    assert not app.upgrade_hazard("crowd", "critical")
    assert not app.upgrade_hazard("nope", "hazard")
    assert app.upgrade_hazard("crowd", "hazard")
    assert app.hazards["crowd"]["priority"] == "hazard"
    assert {"key": "crowd", "priority": "hazard"}.items() <= next(
        h for h in app.context()["hazards"] if h["key"] == "crowd").items()


def test_judgment_off_without_key_or_by_flag(monkeypatch):
    from earpiece.__main__ import main  # noqa: F401  (flag is parsed there)
    from earpiece.config import Cfg
    monkeypatch.setenv("KITE_JUDGMENT", "0")
    assert Cfg().judgment is False
    monkeypatch.delenv("KITE_JUDGMENT")
    assert Cfg().judgment is True
    j = Judgment(lambda: HAZ, print, print, print, api_key=None, log=lambda *_: None)
    assert not j.online


@pytest.mark.skipif(not LIVE_ANTHROPIC_KEY, reason="live Claude call: needs ANTHROPIC_API_KEY")
def test_live_decision():
    said = []
    j = Judgment(lambda: {"hazards": [{"key": "crosswalk", "kind": "crosswalk", "state": "walk",
                                       "text": "Walk signal is on.", "priority": "hazard"}],
                          "scene": "A white car is turning right across the crosswalk from the walker's right.",
                          "position": {"place": "Lyon Street and Bay Street"}},
                 lambda *a, **k: said.append(a), lambda *a: True, lambda *a: None,
                 api_key=LIVE_ANTHROPIC_KEY, timeout_s=8.0, log=print)
    j.stop()
    out = j.step()
    print("live:", out, said)
    assert out in ("said", "quiet", "upgraded", "none")
    for text, _ in said:
        assert len(text) <= 120
