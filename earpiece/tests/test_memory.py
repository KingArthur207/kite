import os
import time

import pytest

from earpiece.memory import MemoryStore


@pytest.fixture
def mem(tmp_path):
    return MemoryStore(project_id="", project_key="", store=tmp_path / "m.json", log=lambda *_: None)


def test_seeded_places_resolve_offline(mem):
    assert mem.resolve_place("my coffee shop").meta["name"] == "Blue Bottle Coffee"
    assert mem.resolve_place("home").id == "place-home"
    assert mem.resolve_place("the bench").id == "place-bench"
    assert mem.resolve_place("the library") is None          # never invent a destination
    assert mem.resolve_place("Golden Gate Park") is None


def test_remember_persists_and_forget_removes(mem, tmp_path):
    m = mem.remember("place", "the pharmacy on Lombard", {"name": "the pharmacy", "lat": 37.79})
    again = MemoryStore(project_id="", project_key="", store=tmp_path / "m.json", log=lambda *_: None)
    assert again.resolve_place("the pharmacy").id == m.id
    assert again.forget(m.id) and again.resolve_place("the pharmacy") is None
    assert not again.forget("nope")


def test_notes_need_a_shared_place_word(mem):
    assert [h.id for h in mem.notes_for("Lyon Street and Bay Street")] == ["hazard-lyon-bay"]
    assert mem.notes_for("Union Street") == []
    assert [h.id for h in mem.notes_for("Marina Boulevard")] == ["note-crosswalk"]


def test_bad_kind_becomes_note_and_text_is_capped(mem):
    m = mem.remember("weird", "x " * 400)
    assert m.kind == "note" and len(m.text) <= 300


def test_brain_memory_intents():
    from earpiece.brain import Actions, Brain
    saved, said = [], []
    a = Actions(lambda d: d, lambda x: x, lambda m: "sent", lambda: None, lambda: None,
                remember=lambda k, t: (saved.append((k, t)), "ok")[1],
                recall=lambda q: [{"kind": "place", "text": "Blue Bottle on Chestnut", "score": 1.0}])
    b = Brain(lambda: {}, a, lambda t, p, **k: said.append(t), None, log=lambda *_: None)
    assert b.handle_sync("Remember this place as the bench") == "local:remember_place"
    assert b.handle_sync("remember that the curb on Union is broken") == "local:remember_note"
    assert b.handle_sync("note that I like quiet streets") == "local:remember_note"
    assert b.handle_sync("what do you remember about coffee") == "local:recall"
    assert saved == [("place", "the bench"), ("hazard", "the curb on Union is broken"),
                     ("note", "I like quiet streets")]
    assert said[-1] == "I remember: Blue Bottle on Chestnut"


@pytest.mark.skipif(not os.environ.get("KITE_LIVE_TESTS"), reason="set KITE_LIVE_TESTS=1 to hit Moss")
def test_live_moss_resolution():
    m = MemoryStore(log=lambda *_: None)
    m.start()
    t0 = time.time()
    while m.status != "moss" and time.time() - t0 < 90:
        time.sleep(0.3)
    assert m.status == "moss"
    p = m.resolve_place("my usual coffee place")
    assert p is not None and p.id == "place-coffee"
    assert m.stats["last_ms"] < 50
