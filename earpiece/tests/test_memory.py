import asyncio
import json
import os
import random
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import wait_for
from earpiece import memory as M
from earpiece.memory import MemoryStore, Resolution, pick

DAY = 86400.0
SILENT = dict(project_id="", project_key="", log=lambda *_: None)


@pytest.fixture
def mem(tmp_path):
    return MemoryStore(store=tmp_path / "m.json", **SILENT)


@pytest.fixture
def world(mem):
    """The seed plus a few memories a real wearer would have after a few weeks."""
    now = time.time()
    mem.save("place", "the bench by the park, near Moscone Recreation Center",
             {"name": "the park bench", "lat": 37.8024, "lng": -122.4332}, now=now - 20 * DAY)
    mem.save("hazard", "I fell on the uneven paving outside the library on Union", now=now - 6 * DAY)
    mem.save("note", "The bakery on Chestnut opens at seven", now=now - 2 * DAY)
    return mem


# ====================================================================== the seed
def test_seeded_places_resolve_offline(mem):
    assert mem.resolve_place("my coffee shop").meta["name"] == "Blue Bottle Coffee"
    assert mem.resolve_place("home").id == "place-home"
    assert mem.resolve_place("the bench").id == "place-bench"
    assert mem.resolve_place("the library") is None          # never invent a destination
    assert mem.resolve_place("Golden Gate Park") is None


def test_demo_home_is_a_public_stand_in_with_no_address(mem):
    home = mem.docs["place-home"]
    assert "demo" in home.text.lower() and "address" not in home.meta
    assert not any(ch.isdigit() for ch in home.text)              # no house number anywhere
    # a public lawn on Marina Green, at most 4 decimals (about 11 m): nobody's front door
    assert (home.meta["lat"], home.meta["lng"]) == (37.8066, -122.4391)
    for v in (home.meta["lat"], home.meta["lng"]):
        assert len(str(v).split(".")[1]) <= 4


def test_remember_persists_and_forget_removes(mem, tmp_path):
    m = mem.remember("place", "the pharmacy on Lombard", {"name": "the pharmacy", "lat": 37.79})
    again = MemoryStore(store=tmp_path / "m.json", **SILENT)
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


# ====================================================================== recall precision
@pytest.mark.parametrize("said,expected", [
    ("my usual coffee place", "place-coffee"),
    ("my usual cafe", "place-coffee"),
    ("the coffee shop", "place-coffee"),
    ("Blue Bottle", "place-coffee"),
    ("my place", "place-home"),
    ("home", "place-home"),
    ("the bench by the bus stop", "place-bench"),
    ("the palace", "place-palace"),
    ("Palace of Fine Arts", "place-palace"),
    ("the beach path", "place-crissy"),
    ("where I like to walk on weekends", "place-crissy"),
])
def test_natural_phrasing_finds_the_right_place(world, said, expected):
    r = world.resolve(said)
    assert r.place is not None and r.place.id == expected, (said, r)


def test_the_bench_by_the_park_is_the_park_bench_not_the_bus_stop_bench(world):
    assert M.name_of(world.resolve_place("the bench by the park")) == "the park bench"
    assert world.resolve_place("the park bench").meta["lat"] == 37.8024


@pytest.mark.parametrize("said", ["the library", "Golden Gate Park", "Union Street", "Marina Green",
                                  "the hospital", "my sister's house on Lombard", "", "the", "my place on"])
def test_unknown_places_are_never_invented(world, said):
    if said == "my place on":
        assert world.resolve_place(said).id == "place-home"      # "my place" is home, "on" is noise
        return
    assert world.resolve_place(said) is None, said


@pytest.mark.parametrize("asked", ["where I fell last week", "where did I trip", "the place I fell",
                                   "the uneven paving"])
def test_recall_of_a_fall_by_any_wording(world, asked):
    top = world.search(asked, top_k=1)
    assert top and "fell" in top[0].text, (asked, top)


def test_two_equally_good_places_ask_instead_of_guessing(world):
    r = world.resolve("the bench")
    assert r.ambiguous and r.place is None
    assert {c.id for c in r.candidates} == {"place-bench", world.resolve_place("the park bench").id}
    q = r.question()
    assert q.startswith("Which one: ") and q.endswith("?") and ", or " in q
    assert "bus stop" in q and "park" in q                       # options a listener can tell apart
    assert world.resolve_place("the bench") is None              # the simple API never guesses


def test_the_same_place_saved_twice_is_not_ambiguous(mem):
    a, s1 = mem.save("place", "the library", {"name": "the library", "lat": 37.80, "lng": -122.44})
    b, s2 = mem.save("place", "the library", {"name": "the library", "lat": 37.80005, "lng": -122.44})
    assert (s1, s2) == ("new", "updated") and a.id == b.id
    assert mem.resolve("the library").place.id == a.id
    assert sum(m.kind == "place" and "library" in m.text for m in mem.docs.values()) == 1


def test_same_name_far_away_is_kept_and_asked_about(mem):
    mem.save("place", "the library", {"name": "the library", "lat": 37.80, "lng": -122.44})
    _, status = mem.save("place", "the library", {"name": "the library", "lat": 37.78, "lng": -122.41})
    assert status == "same_name"
    assert mem.resolve("the library").ambiguous


def test_re_saving_a_place_by_name_only_keeps_the_fuller_description(mem):
    _, status = mem.save("place", "Blue Bottle Coffee", {"name": "Blue Bottle Coffee", "lat": 37.80031,
                                                         "lng": -122.4410})
    assert status == "updated"
    assert mem.docs["place-coffee"].text.startswith("My usual coffee shop")


def test_repeating_a_hazard_confirms_it_instead_of_copying_it(mem):
    a, s1 = mem.save("hazard", "the curb on Union is broken", now=1000.0)
    b, s2 = mem.save("hazard", "The curb on Union is broken!", now=5000.0)
    c, s3 = mem.save("hazard", "the curb on Union is very tall", now=6000.0)
    assert (s1, s2, s3) == ("new", "duplicate", "new")
    assert a.id == b.id != c.id and b.meta["t"] == 5000 and b.meta["confirmed"] == 2


# ====================================================================== which one?
def _cands():
    return [M.Memory("a", "place", "the bench by the 30 bus stop", {"name": "the bus stop bench"}),
            M.Memory("b", "place", "the bench by the park", {"name": "the park bench"})]


@pytest.mark.parametrize("answer,idx", [
    ("the first one", 0), ("first", 0), ("the second", 1), ("number two", 1), ("the last one", 1),
    ("the park one", 1), ("by the bus stop", 0), ("Bus stop.", 0), ("the 30", 0),
    ("never mind", -1), ("neither", -1), ("no", -1), ("none of them", -1),
    ("hmm", None), ("the bench", None), ("what time is it", None), ("stop", -1), ("louder", None),
])
def test_pick_understands_short_answers(answer, idx):
    assert pick(_cands(), answer) == idx


def test_question_wording_uses_names_when_they_differ_and_descriptions_otherwise():
    assert Resolution(None, _cands()).question() == "Which one: the bus stop bench, or the park bench?"
    same = [M.Memory("a", "place", "the library, near Union Street", {"name": "the library"}),
            M.Memory("b", "place", "The library by the 30 bus stop on Chestnut and Divisadero",
                     {"name": "the library"})]
    q = Resolution(None, same).question("Which one should I forget")
    assert q == ("Which one should I forget: the library, near Union Street, "
                 "or the library by the 30 bus stop on Chestnut?")


# ====================================================================== forget, rename, undo
def test_rename_keeps_the_old_name_as_an_alias(world):
    bench = world.resolve_place("the bus stop bench")
    r = world.rename(bench.id, "my reading bench")
    assert r.meta["name"] == "my reading bench" and "the bus stop bench" in r.meta["aka"]
    assert world.resolve_place("my reading bench").id == bench.id
    assert world.resolve_place("the bus stop bench").id == bench.id
    assert world.rename("hazard-lyon-bay", "x") is None            # only places have names
    world.rename(bench.id, "the blue bench")                       # twice: both old names still work
    assert world.resolve_place("my reading bench").id == bench.id


def test_forget_and_restore_round_trip(mem, tmp_path):
    m = mem.docs["place-bench"]
    assert mem.forget(m.id)
    assert mem.resolve_place("the bench") is None
    assert "place-bench" not in {d["id"] for d in json.loads((tmp_path / "m.json").read_text())}
    mem.restore(m)
    assert mem.resolve_place("the bench").id == "place-bench"


def test_voice_forget_rename_undo_and_which_one(app):
    b = app.brain
    app.memory.save("place", "the bench by the park", {"name": "the park bench", "lat": 37.8024, "lng": -122.4332})
    said = []
    app.say = lambda t, *a, **k: said.append(t)
    b.speak = lambda t, *a, **k: said.append(t)

    assert b.handle_sync("forget the bus stop bench") == "local:forget"
    assert said[-1] == "Okay, I've forgotten the bus stop bench. Say undo to bring it back."
    assert app.memory.resolve_place("the bus stop bench") is None
    assert b.handle_sync("undo") == "local:undo"
    assert said[-1] == "Okay, the bus stop bench is back."
    assert b.handle_sync("undo") == "local:undo" and said[-1] == "There's nothing to undo."

    assert b.handle_sync("forget the bench") == "local:forget"   # two benches: which one?
    assert said[-1].startswith("Which one should I forget: ")
    assert b.handle_sync("the park one") == "local:clarify"
    assert said[-1].startswith("Okay, I've forgotten the park bench.")
    assert app.memory.resolve_place("the bench").id == "place-bench"

    assert b.handle_sync("rename the bench to my reading bench") == "local:rename"
    assert said[-1] == "Okay, the bus stop bench is now called my reading bench."
    assert app.memory.resolve_place("my reading bench").id == "place-bench"

    assert b.handle_sync("forget the opera house") == "local:forget"
    assert said[-1] == "I don't have anything saved called the opera house."
    for not_memory in ("forget it", "forget that", "remove the beacon"):
        assert (b.local_intent(not_memory, not_memory) or ("",))[0] != "forget", not_memory


def test_voice_navigation_asks_which_one_and_routes_the_answer(app):
    app.memory.save("place", "the bench by the park", {"name": "the park bench", "lat": 37.8024, "lng": -122.4332})
    said = []
    app.brain.speak = lambda t, *a, **k: said.append(t)
    assert app.brain.handle_sync("take me to the bench") == "local:navigate"
    assert said[-1].startswith("Which one: ") and not app.route.active
    assert app.brain.handle_sync("the second one") == "local:clarify"
    chosen = said[-1]
    assert chosen.startswith("Okay, heading to the ") and app.route.active
    assert app.route.label in chosen
    # an answer that isn't one goes through as a normal request, and the question is dropped
    app.route.cancel()
    app.brain.handle_sync("take me to the bench")
    assert app.brain.handle_sync("louder") == "local:volume_up"
    assert app.clarify("the first one") is None


def test_take_me_home_is_local(app):
    said = []
    app.brain.speak = lambda t, *a, **k: said.append(t)
    assert app.brain.handle_sync("take me home") == "local:navigate"
    assert said[-1] == "Okay, heading to home."
    assert app.route.destination == (37.8066, -122.4391)


def test_voice_save_reports_updates_and_duplicates(app):
    app.position ={"lat": 37.80, "lng": -122.44, "place": "Union Street"}
    assert app.remember("place", "the library") == "Saved. I'll remember this place as the library."
    assert app.remember("place", "the library") == "Updated. the library is saved right here now."
    app.position = {"lat": 37.78, "lng": -122.41}
    assert app.remember("place", "the library").startswith("Saved. You have another place called the library")
    assert app.remember("hazard", "the curb on Union is broken") == "Got it. I'll remember that."
    assert app.remember("hazard", "the curb on union is broken") == "I already had that. I've marked it as still true."


# ====================================================================== time
def test_old_hazards_are_said_more_softly(mem):
    now = 1_800_000_000.0
    h = mem.docs["hazard-lyon-bay"]
    assert mem.heads_up(h, now) == (f"Heads up: {h.text}", 30.0)       # undated seed: plain
    h.meta["t"] = now - 3 * DAY
    assert mem.heads_up(h, now)[0].startswith("Heads up: ")
    h.meta["t"] = now - 40 * DAY
    assert mem.heads_up(h, now) == (f"From last month: {h.text}", 20.0)
    h.meta["t"] = now - 95 * DAY
    assert mem.heads_up(h, now)[0] == f"From 3 months ago: {h.text}"
    h.meta["t"] = now - 400 * DAY
    text, ttl = mem.heads_up(h, now)
    assert text.startswith("An old note, it may have changed: ") and ttl == 15.0


def test_fresher_hazards_come_first(mem):
    now = time.time()
    old = mem.save("hazard", "Roots lifting the sidewalk on Pierce Street near the school", now=now - 300 * DAY)[0]
    new = mem.save("hazard", "Parked scooters block the sidewalk on Pierce Street", now=now - 1 * DAY)[0]
    assert [h.id for h in mem.notes_for("Pierce Street", now=now)][:2] == [new.id, old.id]
    assert M.freshness(old, now) < 0.5 < M.freshness(new, now) == pytest.approx(1.0, abs=0.01)


def test_app_says_an_old_hazard_softly(app):
    h = app.memory.docs["hazard-lyon-bay"]
    h.meta["t"] = time.time() - 200 * DAY
    from earpiece.events import parse
    app.handle(parse({"type": "position", "place": "Lyon Street and Bay Street"}))
    spoken = lambda: [r["text"] for r in app.speaker.history]   # noqa: E731
    assert wait_for(lambda: any(t.startswith("An old note, it may have changed: The curb") for t in spoken()), 5)


# ====================================================================== privacy and the local mirror
def test_memory_module_talks_to_nothing_but_moss():
    src = Path(M.__file__).read_text()
    imports = set(re.findall(r"^\s*(?:from|import)\s+([\w.]+)", src, re.M))
    assert not imports & {"httpx", "requests", "urllib", "urllib.request", "socket", "http", "aiohttp"}, imports
    assert {i for i in imports if i.startswith("inferedge")} <= {"inferedge_moss"}


def test_mirror_is_private_to_the_user(mem, tmp_path):
    mem.remember("place", "my sister's flat", {"name": "my sister's flat", "lat": 37.79, "lng": -122.43})
    assert (tmp_path / "m.json").stat().st_mode & 0o077 == 0


def test_offline_with_the_network_cut(monkeypatch, tmp_path):
    import socket

    def no_network(*a, **k):
        raise AssertionError("memory tried to open a network connection")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    m = MemoryStore(store=tmp_path / "m.json", **SILENT)
    m.start()                                                    # no keys: local only, no network
    assert m.status == "local"
    m.remember("place", "the gym", {"name": "the gym", "lat": 37.79, "lng": -122.43})
    assert m.resolve_place("the gym") is not None and m.search("gym")[0].text == "the gym"


def test_corrupt_mirror_is_kept_aside_and_the_seed_reloaded(tmp_path):
    (tmp_path / "m.json").write_text("{not json")
    m = MemoryStore(store=tmp_path / "m.json", **SILENT)
    assert m.resolve_place("home").id == "place-home"
    assert (tmp_path / "m.corrupt.json").read_text() == "{not json"


# ====================================================================== Moss, faked
class FakeMoss:
    """The slice of inferedge_moss.MossClient that memory.py uses, in memory. `noise` makes
    its scores relative and a little arbitrary, the way a hybrid semantic index's are."""

    def __init__(self, docs=None, noise=0.0, fail=None, hang=0.0, seed=0):
        self.docs = {d.id: d for d in (docs or [])}
        self.created = False
        self.calls: list[str] = []
        self.texts_seen: list[str] = []
        self.noise, self.fail, self.hang = noise, fail, hang
        self.rng = random.Random(seed)

    async def list_indexes(self):
        self.calls.append("list")
        if self.fail == "connect":
            raise ConnectionError("moss unreachable")
        return [SimpleNamespace(name="kite-memory")] if self.created or self.docs else []

    async def create_index(self, name, docs):
        self.calls.append("create")
        self.created = True
        for d in docs:
            self.docs[d.id] = d
            self.texts_seen.append(d.text)

    async def get_docs(self, name):
        return list(self.docs.values())

    async def add_docs(self, name, docs, opts=None):
        self.calls.append("add")
        for d in docs:
            self.docs[d.id] = d
            self.texts_seen.append(d.text)

    async def delete_docs(self, name, ids):
        self.calls.append("delete")
        for i in ids:
            self.docs.pop(i, None)

    async def get_index(self, name):
        return SimpleNamespace(status="Ready")

    async def load_index(self, name):
        self.calls.append("load")

    async def query(self, name, query, opts=None):
        self.calls.append("query")
        if self.hang:
            await asyncio.sleep(self.hang)
        if self.fail == "query":
            raise RuntimeError("query failed")
        q = M._tokens(query)
        out = []
        for d in self.docs.values():
            t = M._tokens(d.text)
            s = 0.55 + 0.4 * (len(q & t) / max(1, len(q | t))) + self.rng.uniform(-self.noise, self.noise)
            out.append(SimpleNamespace(id=d.id, text=d.text, metadata=d.metadata, score=min(1.0, s)))
        out.sort(key=lambda d: -d.score)
        return SimpleNamespace(docs=out[: getattr(opts, "top_k", 5) or 5])


def moss_store(tmp_path, fake, name="m.json"):
    m = MemoryStore(store=tmp_path / name, client=fake, **SILENT)
    m.SYNC_SETTLE_S = 0.0
    errors = m.stats["errors"]
    m.start()
    assert wait_for(lambda: m.status == "moss" or m.stats["errors"] > errors, 5)
    return m


PHRASES = ["my usual coffee place", "home", "my place", "the bench", "the bench by the bus stop", "the palace",
           "Crissy Field", "the library", "Golden Gate Park", "Union Street", "the beach", "Blue Bottle",
           "my favourite walk", "the hospital", "Marina Green", "where I like to walk on weekends"]


@pytest.mark.parametrize("noise", [0.0, 0.2])
def test_resolution_is_the_same_with_moss_and_offline(tmp_path, noise):
    offline = MemoryStore(store=tmp_path / "off.json", **SILENT)
    online = moss_store(tmp_path, FakeMoss(noise=noise, seed=7), "on.json")
    assert online.status == "moss" and online.ready
    for p in PHRASES:
        a, b = offline.resolve(p), online.resolve(p)
        assert (a.place and a.place.id) == (b.place and b.place.id), p
        assert [c.id for c in a.candidates] == [c.id for c in b.candidates], p
    for place in ["Lyon Street and Bay Street", "Union Street", "Marina Boulevard", "Chestnut and Divisadero"]:
        assert {h.id for h in offline.notes_for(place)} == {h.id for h in online.notes_for(place)}, place
    assert online.stats["moss_queries"] > 0                      # notes really went through Moss


def test_moss_errors_fall_back_and_a_failing_moss_is_benched(tmp_path):
    fake = FakeMoss(fail="query")
    m = moss_store(tmp_path, fake)
    assert m.status == "moss"
    for _ in range(M.MOSS_FAILS):
        assert m.search("coffee")[0].id == "place-coffee"         # keyword answers every time
    n = fake.calls.count("query")
    for _ in range(10):
        m.search("coffee")
    assert fake.calls.count("query") == n                       # benched: no more slow failures
    assert m.resolve_place("my coffee shop").id == "place-coffee"


def test_a_hanging_moss_costs_at_most_the_timeout(tmp_path):
    m = moss_store(tmp_path, FakeMoss(hang=2.0))
    t0 = time.perf_counter()
    hits = m.search("coffee", timeout=0.1)
    assert time.perf_counter() - t0 < 0.5 and hits[0].id == "place-coffee"


def test_moss_unreachable_at_start_means_local_memory(tmp_path):
    m = moss_store(tmp_path, FakeMoss(fail="connect"))
    assert wait_for(lambda: m.status == "local", 3) and not m.ready
    assert m.resolve_place("home").id == "place-home"


def test_forgotten_offline_stays_forgotten_when_moss_comes_back(tmp_path):
    fake = FakeMoss()
    m = moss_store(tmp_path, fake)
    assert "place-bench" in fake.docs
    fake.fail = "connect"
    m._loop = None                                               # Moss gone: the delete can't be sent
    m.forget("place-bench")
    assert "place-bench" in fake.docs and "place-bench" in json.loads((tmp_path / "m.forgotten.json").read_text())
    fake.fail = None
    again = moss_store(tmp_path, fake)                           # next start: the delete goes through
    assert wait_for(lambda: "place-bench" not in fake.docs, 3)
    assert "place-bench" not in again.docs and again.resolve_place("the bus stop bench") is None
    assert all(h.id != "place-bench" for h in again.search("bench by the bus stop", top_k=5))


def test_memories_from_another_device_are_pulled_in_and_edits_pushed_out(tmp_path):
    fake = FakeMoss()
    m = moss_store(tmp_path, fake)
    fake.docs["place-gym"] = SimpleNamespace(id="place-gym", text="The gym on Fillmore",
                                             metadata={"kind": "place", "name": "the gym", "lat": "37.79",
                                                       "lng": "-122.43"})
    fake.docs["place-home"] = SimpleNamespace(id="place-home", text="Home: an old street address",
                                              metadata={"kind": "place"})
    again = moss_store(tmp_path, fake)
    gym = again.resolve_place("the gym")
    assert gym is not None and gym.meta["lat"] == 37.79
    # the mirror wins for content it already has: the scrubbed demo home replaces the old copy
    assert wait_for(lambda: fake.docs["place-home"].text == m.docs["place-home"].text, 3)


def test_only_moss_sees_memory_text(tmp_path):
    fake = FakeMoss()
    m = moss_store(tmp_path, fake)
    m.remember("note", "my door code is not something to share")
    assert wait_for(lambda: any("door code" in t for t in fake.texts_seen), 3)


# ====================================================================== latency
def _bench(fn, phrases, n=400):
    for p in phrases[:20]:
        fn(p)                                                    # warm caches
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(phrases[i % len(phrases)])
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"p50": round(statistics.median(ts), 3), "p95": round(ts[int(0.95 * len(ts))], 3),
            "max": round(ts[-1], 3)}


@pytest.mark.slow
def test_local_lookup_latency_stays_in_budget(tmp_path, capsys):
    """The offline mirror must be as cheap as Moss claims to be: with 500 memories, a lookup
    on every utterance stays under 12 ms at the 95th percentile."""
    m = MemoryStore(store=tmp_path / "m.json", **SILENT)
    rng = random.Random(1)
    words = ["bench", "curb", "pole", "cafe", "bakery", "stairs", "ramp", "crossing", "bus", "stop", "park",
             "gate", "library", "market", "school", "fountain", "corner", "door", "elevator", "pharmacy"]
    streets = ["Chestnut", "Lombard", "Union", "Fillmore", "Pierce", "Scott", "Divisadero", "Broderick"]
    for i in range(500):
        kind = rng.choice(["place", "hazard", "note"])
        text = f"The {rng.choice(words)} {rng.choice(['by', 'near', 'on'])} {rng.choice(streets)} " \
               f"{' '.join(rng.sample(words, 3))} {i}"
        m.save(kind, text, {"name": text[:20], "lat": 37.79 + rng.random() / 100, "lng": -122.44 + rng.random() / 100})
    resolve = _bench(m.resolve, PHRASES)
    search = _bench(m.search, PHRASES)
    with capsys.disabled():
        print(f"\n[memory latency, 509 memories, offline] resolve {resolve}  search {search} (ms)")
    assert resolve["p95"] < 12 and search["p95"] < 12


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("KITE_LIVE_TESTS"), reason="set KITE_LIVE_TESTS=1 to hit Moss")
def test_live_moss_latency(tmp_path, capsys):
    """Real Moss, index loaded into the local runtime: how long one lookup takes."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    m = MemoryStore(store=tmp_path / "m.json", log=lambda *_: None)
    m.start()
    assert wait_for(lambda: m.status == "moss" or m.stats["errors"] > 0, 120, 0.3)
    assert m.status == "moss", m.status
    p = m.resolve_place("my usual coffee place")
    assert p is not None and p.id == "place-coffee"
    query = lambda q: asyncio.run_coroutine_threadsafe(   # noqa: E731
        m._client.query(m.index, q, M._moss_types()[2](top_k=7, alpha=0.7)), m._loop).result(5)
    raw = _bench(query, PHRASES, n=100)
    full = _bench(m.search, PHRASES, n=100)
    with capsys.disabled():
        print(f"\n[memory latency, live Moss local index] query {raw}  search {full} (ms)")
    assert raw["p50"] < 12
