"""The Rust sonar must make exactly the decisions the Python one makes.

`sonar.Sonar` is the reference; `rust_sonar.RustSonar` runs the same decision core in
`kite_audio`. Every test here drives BOTH with an identical input stream (frames, single
objects, mute/label toggles, watchdog disable + clear, engine clock) and, after every
tick, requires:

  * the same beeps: start frame (exactly), bus, tag, cue (kind, pitch, bearing), gain;
  * the same speech: text, priority, key, ttl, in the same order;
  * the same chosen tracks, and the same state of every track (urgency, TTC, closing
    speed, smoothed bearing/distance, history, beep schedule, label timers);
  * the same stats.

Float state is compared within 1e-9 (in practice it is bit-identical; the worst
deviation is printed with `-s`). These tests call RustSonar directly, so parity is
checked whichever engine KITE_SONAR_ENGINE selects for the rest of the suite.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import pytest

from earpiece import events, rust_sonar, sim
from earpiece import sonar as sonar_mod
from earpiece.config import SonarCfg
from earpiece.events import Obstacle
from earpiece.mixer import Engine

pytestmark = pytest.mark.skipif(not rust_sonar.available(), reason="kite_audio built without the sonar")

RATE = 48000
BLOCK = 512                       # the engine clock advances a device block at a time
REPO = Path(__file__).resolve().parents[2]
TOL = 1e-9
FLOAT_FIELDS = ("bearing", "distance", "confidence", "given_speed", "est_speed", "last_seen", "ttl",
                "urgency", "ttc", "last_label_t", "last_stop_t", "first_seen", "closing_speed")
EXACT_FIELDS = ("id", "kind", "source", "active", "critical", "next_beep", "last_beep")
WORST = {"float": 0.0, "where": None, "gain": 0.0, "ticks": 0, "beeps": 0, "says": 0}


class RecEngine:
    """The engine surface the sonar uses, with a settable clock; records every play."""

    def __init__(self, rate: int = RATE):
        self.rate = rate
        self.now = 0
        self.plays: list[tuple] = []

    def samples(self, seconds: float) -> int:
        return int(round(seconds * self.rate))

    def play(self, stereo, at=None, bus="cue", gain=1.0, tag=None):
        self.plays.append((at, bus, gain, tag, stereo))


@pytest.fixture(autouse=True)
def cue_keys(monkeypatch):
    """Record which cue each beep asked for instead of rendering it (both engines look the
    cue up through sonar._cue, so they see the same stand-in)."""
    monkeypatch.setattr(sonar_mod, "_cue", lambda kind, pq, bq, rate: ("cue", kind, pq, bq, rate))


def _dev(a, b) -> float:
    if a is None or b is None:
        assert a is None and b is None, (a, b)
        return 0.0
    if math.isinf(a) or math.isinf(b):
        assert a == b, (a, b)
        return 0.0
    return abs(a - b)


class Pair:
    """A Python sonar and a Rust sonar fed the same inputs, compared after every tick."""

    def __init__(self, cfg: SonarCfg | None = None, speak: bool = True, rate: int = RATE):
        self.cfg = cfg or SonarCfg()
        self.ep, self.er = RecEngine(rate), RecEngine(rate)
        self.said_p: list[tuple] = []
        self.said_r: list[tuple] = []
        sp = (lambda *a, **k: self.said_p.append((a, k))) if speak else None
        sr = (lambda *a, **k: self.said_r.append((a, k))) if speak else None
        self.py = sonar_mod.Sonar(self.ep, self.cfg, speak=sp)
        self.rs = rust_sonar.RustSonar(self.er, self.cfg, speak=sr)
        self.n_plays = self.n_said = 0
        self.ticks = 0

    def both(self, fn) -> None:
        fn(self.py)
        fn(self.rs)

    def frame(self, source, objs, seq, now):
        self.both(lambda s: s.update_frame(source, list(objs), seq=seq, now=now))

    def one(self, source, o, ttl, now):
        self.both(lambda s: s.update_one(source, o, ttl, now=now))

    def set(self, name, value):
        self.both(lambda s: setattr(s, name, value))

    def clear(self, source=None):
        self.both(lambda s: s.clear(source))

    def tick(self, now: float, engine_now: int) -> list:
        self.ep.now = self.er.now = engine_now
        cp = self.py.tick(now=now)
        cr = self.rs.tick(now=now)
        self.ticks += 1
        ctx = f"tick {self.ticks} t={now:.4f}"
        assert [t.id for t in cp] == [t.id for t in cr], ctx
        new_p, new_r = self.ep.plays[self.n_plays:], self.er.plays[self.n_plays:]
        assert len(new_p) == len(new_r), (ctx, new_p, new_r)
        for a, b in zip(new_p, new_r):
            assert (a[0], a[1], a[3], a[4]) == (b[0], b[1], b[3], b[4]), (ctx, a, b)
            g = abs(a[2] - b[2])
            WORST["gain"] = max(WORST["gain"], g)
            assert g <= TOL, (ctx, a, b)
        self.n_plays = len(self.ep.plays)
        assert len(self.ep.plays) == len(self.er.plays), ctx
        assert self.said_p[self.n_said:] == self.said_r[self.n_said:], ctx
        assert len(self.said_p) == len(self.said_r), ctx
        self.n_said = len(self.said_p)
        self.compare_state(ctx)
        return cp

    def compare_state(self, ctx: str = "") -> None:
        tp, tr = self.py.tracks, self.rs.tracks
        assert list(tp) == list(tr), ctx                    # same ids, same (insertion) order
        for k, a in tp.items():
            b = tr[k]
            for f in EXACT_FIELDS:
                assert getattr(a, f) == getattr(b, f), (ctx, k, f, getattr(a, f), getattr(b, f))
            for f in FLOAT_FIELDS:
                d = _dev(getattr(a, f), getattr(b, f))
                if d > WORST["float"]:
                    WORST["float"], WORST["where"] = d, (f, getattr(a, f), getattr(b, f))
                assert d <= TOL, (ctx, k, f, getattr(a, f), getattr(b, f))
            ha, hb = list(a.hist), list(b.hist)
            assert len(ha) == len(hb), (ctx, k)
            for (t1, d1), (t2, d2) in zip(ha, hb):
                assert abs(t1 - t2) <= TOL and abs(d1 - d2) <= TOL, (ctx, k)
        assert self.py.stats == self.rs.stats, ctx
        assert (self.py.enabled, self.py.muted, self.py.labels) == (self.rs.enabled, self.rs.muted, self.rs.labels)
        assert self.py.summary() == self.rs.summary(), ctx

    def finish(self) -> dict:
        WORST["ticks"] += self.ticks
        WORST["beeps"] += len(self.ep.plays)
        WORST["says"] += len(self.said_p)
        return {"ticks": self.ticks, "beeps": len(self.ep.plays), "says": len(self.said_p),
                "criticals": self.py.stats["criticals"]}


def replay(pair: Pair, timeline, seconds: float, *, tick_s: float = 0.02, jitter: float = 0.0,
           stalls=(), watchdog: float | None = None, seed: int = 0) -> dict:
    """Play `timeline` [(t, fn(pair))] against both sonars with control ticks every `tick_s`
    (optionally jittered, optionally stalled), the engine clock advancing in device blocks,
    and optionally the app's stale-perception watchdog. Inputs at time t are applied
    before the tick at t, as in the app (frames arrive on another thread)."""
    rng = random.Random(seed)
    ticks, t = [], 0.0
    while t < seconds:
        ticks.append(t)
        t += tick_s + (rng.uniform(-jitter, jitter) if jitter else 0.0)
        for a, b in stalls:                       # the control loop froze between a and b
            if a <= t < b:
                t = b
    timeline = sorted(timeline, key=lambda x: x[0])
    i, last_perception, ok = 0, None, None
    for tt in ticks:
        while i < len(timeline) and timeline[i][0] <= tt:
            ft, fn = timeline[i]
            if fn(pair) is not False:             # an input counts as perception
                last_perception = ft
            i += 1
        if watchdog is not None and last_perception is not None:
            now_ok = tt - last_perception <= watchdog
            if now_ok != ok:
                ok = now_ok
                pair.set("enabled", ok)           # app._watchdog
                if not ok:
                    pair.clear()
        pair.tick(tt, int(tt * RATE) // BLOCK * BLOCK)
    return pair.finish()


def frames(t0, t1, fn, fps=10.0, source="cam", seq0=1, jitter=0.0, rng=None):
    """[(t, update)] for a frame stream; fn(t) -> objects."""
    out, n = [], 0
    while t0 + n / fps < t1:
        t = t0 + n / fps + (rng.uniform(0, jitter) if jitter else 0.0)
        objs = fn(t)
        out.append((t, lambda p, objs=objs, seq=seq0 + n, t=t: p.frame(source, objs, seq, t)))
        n += 1
    return out


def do(t, fn):
    """A control action at time t that is not perception (mute, labels, ...)."""
    return (t, lambda p: (fn(p), False)[1])


# ---------------------------------------------------------------- scripted sequences
def test_approaching_person():
    rng = random.Random(1)
    p = Pair()
    tl = frames(0, 6, lambda t: [
        Obstacle("p", "person", rng.gauss(0, 3), max(0.35, 6.0 - 1.3 * t + rng.gauss(0, 0.05)), None, 0.9),
        Obstacle("pole", "pole", 35, 3.0, None, 0.7)], fps=10, rng=rng, jitter=0.01)
    r = replay(p, tl, 7.0, jitter=0.004)
    assert r["beeps"] > 20 and r["criticals"] >= 1
    assert any(a[0][0] == "Stop. Person ahead." for a in p.said_p)
    print(f"\n[parity] approaching person: {r}")


def test_crossing_cyclist_and_wraparound_behind():
    p = Pair()

    def objs(t):
        b = -80 + 40 * t                                     # left to right across the path
        behind = 170 + 8 * t                                 # crosses +-180 behind the wearer
        return [Obstacle("bike", "bike", b, 3.5 - 0.2 * t, 1.5 + 0.5 * math.sin(t), 1.0),
                Obstacle("b2", "bike", spatial_wrap(behind), 0.7, None, 0.8)]
    r = replay(p, frames(0, 4.2, objs, fps=9), 5.0, jitter=0.003, seed=2)
    assert r["beeps"] > 5
    print(f"\n[parity] crossing cyclist: {r}")


def spatial_wrap(a):
    from earpiece.spatial import wrap_deg
    return wrap_deg(a)


def test_out_of_order_and_restarted_frames():
    p = Pair()
    tl, t = [], 0.0
    seqs = [1, 2, 3, 5, 4, 6, 6, 7, 2, 8, 9, 400, 401, 3, 4, 5, 6, 1, 2]   # late, duplicate, restart
    for i, s in enumerate(seqs):
        t = i * 0.1
        d = 4.0 - 0.15 * i if s < 100 else 9.0 - 0.1 * i                # late frames carry stale data
        tl.append((t, lambda pp, s=s, d=d, t=t: pp.frame("cam", [Obstacle("p", "person", 2, d, None, 0.9)], s, t)))
    tl.append((3.0, lambda pp: pp.frame("cam", [Obstacle("p", "person", 2, 1.0, None, 0.9)], 1, 3.0)))   # after a pause
    tl.append((3.05, lambda pp: pp.frame("cam", [Obstacle("p", "person", 2, 0.9, None, 0.9)], 2 ** 62, 3.05)))
    tl.append((3.1, lambda pp: pp.frame("cam", [Obstacle("p", "person", 2, 0.8, None, 0.9)], -5, 3.1)))
    r = replay(p, tl, 4.0)
    assert p.py.stats["out_of_order"] >= 4
    print(f"\n[parity] out-of-order/restart: {r}, out_of_order {p.py.stats['out_of_order']}")


def test_stale_perception_watchdog_disables_and_clears():
    p = Pair()
    near = lambda t: [Obstacle("p", "pole", 0, 1.4, None, 1.0), Obstacle("c", "vehicle", -15, 4.0, 2.5, 0.9)]
    tl = frames(0, 2.0, near) + frames(4.5, 7.0, near, seq0=100)     # 2.5 s with no frames at all
    r = replay(p, tl, 8.0, watchdog=1.5)
    assert p.py.enabled and r["beeps"] > 0
    print(f"\n[parity] stale perception: {r}")


def test_two_source_cap_ranking_and_ties():
    p = Pair()

    def crowd(t):
        out = [Obstacle(f"c{i}", "person", -50 + i * 9, 1.0 + 0.25 * i - 0.1 * t, None, 0.8) for i in range(12)]
        out += [Obstacle("tl", "pole", -10, 2.0, None, 1.0), Obstacle("tr", "pole", 10, 2.0, None, 1.0)]  # exact tie
        if t > 1.5:
            out.append(Obstacle("car", "vehicle", 5, 6 - 3 * (t - 1.5), 3.0, 1.0))
        return out
    seen = []
    orig = p.tick
    p.tick = lambda now, en: seen.append(len(orig(now, en))) or []
    r = replay(p, frames(0, 3.5, crowd, fps=12), 4.0)
    assert max(seen) == 2
    assert len({a[3] for a in p.ep.plays}) >= 3
    print(f"\n[parity] two-source cap: {r}")


def test_critical_stop_variants_mute_and_labels():
    p = Pair()
    tl = frames(0, 2, lambda t: [Obstacle("c", "vehicle", 0, 8 - 4 * t, 4.0, 1.0)], source="drone")
    tl += frames(2.5, 4, lambda t: [Obstacle("p", "person", 40, 0.5, None, 1.0)], seq0=50)        # "Stop."
    tl += frames(4.5, 6, lambda t: [Obstacle("o", "pole", 3, 0.6, None, 1.0)], seq0=80)           # obstacle ahead
    tl += frames(6.5, 8, lambda t: [Obstacle("k", "curb", -5, 0.7, None, 1.0)], seq0=120)         # "Stop."
    tl += frames(8.5, 11, lambda t: [Obstacle("m", "person", 0, 3.2 - 0.9 * (t - 8.5), None, 1.0)], seq0=200)
    tl += [do(9.0, lambda pp: pp.set("muted", True)), do(10.2, lambda pp: pp.set("muted", False)),
           do(10.4, lambda pp: pp.set("labels", False))]
    r = replay(p, tl, 12.0, jitter=0.002, seed=5)
    said = [a[0][0] for a in p.said_p]
    assert {"Stop. Car.", "Stop.", "Stop. Obstacle ahead."} <= set(said), said
    print(f"\n[parity] critical stop: {r} said={said}")


def test_no_speak_callback_means_no_speech_decisions():
    p = Pair(speak=False)
    r = replay(p, frames(0, 3, lambda t: [Obstacle("p", "person", 0, 3 - t, None, 1.0)]), 3.5)
    assert r["beeps"] > 0 and r["says"] == 0
    p.set("speak", lambda *a, **k: None)          # a callback attached later starts speaking
    p.frame("cam", [Obstacle("p", "person", 0, 0.5, None, 1.0)], 999, 3.5)
    p.tick(3.5, int(3.5 * RATE))


def test_single_objects_with_ttl_and_clear_by_source():
    p = Pair()
    tl = frames(0, 5, lambda t: [Obstacle("a", "person", -20, 2.5, None, 1.0)], source="cam")
    for i, ttl in enumerate([None, 0, 0.3, 2.0, 0.5, None]):
        t = 0.4 + i * 0.7
        tl.append((t, lambda pp, t=t, ttl=ttl, i=i: pp.one("drone", Obstacle(f"x{i % 3}", "stairs_down", 5 * i, 1.0 + 0.3 * i, None, 0.8), ttl, t)))
    tl += [do(2.6, lambda pp: pp.clear("cam")), do(3.9, lambda pp: pp.clear("nobody"))]
    tl.append((4.2, lambda pp: pp.frame("drone", [], 7, 4.2)))       # a frame never fades ttl objects
    r = replay(p, tl, 6.0)
    print(f"\n[parity] ttl objects: {r}")


def test_control_loop_stalls_catch_up_identically():
    """GIL stalls of 150-500 ms: the beat is kept, so both sonars schedule the same catch-up."""
    p = Pair()
    tl = frames(0, 6, lambda t: [Obstacle("p", "pole", 0, max(0.5, 3 - 0.6 * t), None, 1.0),
                                 Obstacle("q", "person", 30, 2.0, None, 1.0)])
    r = replay(p, tl, 6.5, stalls=[(1.0, 1.15), (2.0, 2.5), (3.3, 3.62), (5.0, 5.2)])
    print(f"\n[parity] stalls: {r}")


def test_other_rates_and_config():
    cfg = SonarCfg(max_sources=3, interval_fast_s=0.15, pitch_high_hz=1400.0, hysteresis=0.25,
                   corridor_half_width_m=0.7, stale_track_s=0.9)
    for rate in (44100, 16000):
        p = Pair(cfg, rate=rate)
        tl = frames(0, 4, lambda t: [Obstacle(f"o{i}", "obstacle", -30 + 15 * i, 3 - 0.5 * t + 0.2 * i, None, 0.9)
                                     for i in range(5)])
        ticks = []
        t = 0.0
        while t < 4.5:
            ticks.append(t)
            t += 0.02
        i = 0
        for tt in ticks:
            while i < len(tl) and tl[i][0] <= tt:
                tl[i][1](p)
                i += 1
            p.tick(tt, int(tt * rate) // BLOCK * BLOCK)
        p.finish()


@pytest.mark.parametrize("seed", range(24))
def test_randomised_multi_source_streams(seed):
    """Seeded fuzz: three sources, objects appearing/vanishing, every kind, bearings across
    +-180, closing speeds present or not, out-of-order seqs, restarts, mute/labels toggles,
    single-object events, watchdog dropouts and jittered, stalled ticks."""
    rng = random.Random(seed)
    p = Pair(speak=seed % 5 != 0)
    kinds = ["person", "vehicle", "bike", "curb", "stairs_down", "obstacle", "pole", "dog"]
    tl = []
    for src in ("webcam", "drone", "sim")[: rng.randint(1, 3)]:
        objs = {}
        seq = rng.randint(0, 1000)
        t = rng.uniform(0, 0.3)
        while t < 10:
            for oid in list(objs):
                if rng.random() < 0.03:
                    del objs[oid]
            if len(objs) < 6 and rng.random() < 0.15:
                objs[f"{src[0]}{rng.randint(0, 30)}"] = [rng.choice(kinds), rng.uniform(-180, 180),
                                                          rng.uniform(0.2, 7), rng.uniform(-1, 3)]
            frame = []
            for oid, o in objs.items():
                o[1] = spatial_wrap(o[1] + rng.gauss(0, 4))
                o[2] = max(0.0, o[2] - o[3] * 0.1 + rng.gauss(0, 0.04))
                if rng.random() < 0.9:                                  # dropped for a frame
                    cs = round(o[3] + rng.gauss(0, 0.2), 3) if rng.random() < 0.4 else None
                    frame.append(Obstacle(oid, o[0], o[1], o[2], cs, rng.choice([0.3, 0.6, 1.0])))
            seq += 1
            s = seq
            r = rng.random()
            if r < 0.06:
                s = seq - rng.randint(1, 5)                             # late datagram
            elif r < 0.07:
                s = rng.randint(0, 5)                                   # sender restart
                seq = s
            tl.append((t, lambda pp, src=src, fr=frame, s=s, t=t: pp.frame(src, fr, s, t)))
            if rng.random() < 0.05:
                o = Obstacle(f"one{rng.randint(0, 3)}", rng.choice(kinds), rng.uniform(-60, 60),
                             rng.uniform(0.3, 4), None, 1.0)
                ttl = rng.choice([None, 0, 0.2, 0.8, 1.5])
                tl.append((t + 0.001, lambda pp, o=o, ttl=ttl, t=t: pp.one(src, o, ttl, t + 0.001)))
            t += rng.uniform(0.06, 0.14)
            if rng.random() < 0.01:
                t += rng.uniform(1.0, 2.5)                              # perception dropout
    for _ in range(rng.randint(0, 4)):
        what = rng.choice(["muted", "labels"])
        tl.append(do(rng.uniform(0, 10), lambda pp, w=what, v=rng.random() < 0.5: pp.set(w, v)))
    stalls = [(a, a + rng.uniform(0.1, 0.6)) for a in sorted(rng.uniform(0, 10) for _ in range(rng.randint(0, 3)))]
    r = replay(p, tl, 11.0, jitter=0.006, stalls=stalls, watchdog=1.5, seed=seed)
    assert r["ticks"] > 300


# ---------------------------------------------------------------- recorded / scripted replays
def _replay_events(timeline, seconds, watchdog=1.5) -> dict:
    """Feed parsed wire events to both sonars exactly as app.handle does."""
    p = Pair()
    tl, sources = [], set()
    for t, raw in timeline:
        try:
            msg = events.parse(raw)
        except events.BadMessage:
            continue
        if msg.type == "obstacles":
            sources.add(msg.source)
            tl.append((t, lambda pp, m=msg, t=t: pp.frame(m.source, m.objects, m.seq, t)))
        elif msg.type == "obstacle":
            sources.add(msg.source)
            tl.append((t, lambda pp, m=msg, t=t: [pp.one(m.source, o, m.ttl, t) for o in m.objects]))
        elif msg.type == "heartbeat" and msg.source in sources:
            tl.append((t, lambda pp: None))
    if not tl:
        return {"ticks": 0}
    return replay(p, tl, seconds, jitter=0.003, watchdog=watchdog)


@pytest.mark.parametrize("path", ["data/sample_walk.jsonl", "data/sample_track.jsonl",
                                  *sorted(str(x.relative_to(REPO)) for x in (REPO / "data/scenarios").glob("*.jsonl"))])
def test_recorded_jsonl_replay(path):
    timeline = sim.jsonl_timeline(REPO / path)
    r = _replay_events(timeline, max(t for t, _ in timeline) + 2.0)
    print(f"\n[parity] {path}: {r}")
    if path == "data/sample_walk.jsonl":
        assert r["ticks"] > 2000 and r["beeps"] > 20 and r["criticals"] >= 1


@pytest.mark.parametrize("name", sorted(sim.SCENARIOS))
def test_scripted_sim_scenarios(name):
    r = _replay_events(sim.timeline(name), sim.duration(name))
    print(f"\n[parity] sim {name}: {r}")
    assert r["ticks"] > 100


def test_worst_deviation_report():
    """Runs last in this module: prints the totals and the worst deviation seen."""
    print(f"\n[parity] totals: {WORST['ticks']} ticks, {WORST['beeps']} beeps, {WORST['says']} utterances; "
          f"worst float deviation {WORST['float']:.3g} {WORST['where'] or ''}, "
          f"worst gain deviation {WORST['gain']:.3g}")
    assert WORST["float"] <= TOL and WORST["gain"] <= TOL


# ---------------------------------------------------------------- the real audio path
def test_rendered_audio_is_identical(monkeypatch):
    """No stand-ins: real cues, real mixer, both sonars; the wearer hears the same samples."""
    monkeypatch.undo()                                          # real sonar._cue
    outs = []
    for make in (sonar_mod.Sonar, rust_sonar.RustSonar):
        eng = Engine()
        s = make(eng, SonarCfg(), speak=None)
        audio, seq, t = [], 0, 0.0
        while t < 4.0:
            seq += 1
            d = max(0.4, 5 - 1.2 * t)
            s.update_frame("cam", [Obstacle("p", "person", -20 + 10 * t, d, None, 0.9),
                                   Obstacle("c", "vehicle", 30, 6 - t, 1.0, 1.0)], seq=seq, now=t)
            s.tick(now=t)
            audio.append(eng.render(1024))
            t += 1024 / 48000
        outs.append(np.concatenate(audio))
    assert np.abs(outs[0]).max() > 0.05
    assert np.array_equal(outs[0], outs[1])


# ---------------------------------------------------------------- engine selection
def test_engine_selection(monkeypatch):
    eng = RecEngine()
    assert isinstance(sonar_mod.create(eng, SonarCfg(engine="python")), sonar_mod.Sonar)
    assert isinstance(sonar_mod.create(eng, SonarCfg(engine="rust")), rust_sonar.RustSonar)
    assert isinstance(sonar_mod.create(eng, SonarCfg(engine="auto")), rust_sonar.RustSonar)
    assert isinstance(sonar_mod.create(eng, SonarCfg(engine=" Rust ")), rust_sonar.RustSonar)
    with pytest.raises(ValueError):
        sonar_mod.create(eng, SonarCfg(engine="fortran"))
    monkeypatch.setenv("KITE_SONAR_ENGINE", "python")
    assert SonarCfg().engine == "python"
    monkeypatch.setattr(rust_sonar, "_ka", None)               # extension not built
    assert isinstance(sonar_mod.create(eng, SonarCfg(engine="auto")), sonar_mod.Sonar)
    with pytest.raises(RuntimeError):
        sonar_mod.create(eng, SonarCfg(engine="rust"))


def test_app_uses_the_selected_engine(offline_speech, monkeypatch):
    from earpiece.app import Earpiece
    from earpiece.config import Cfg
    from earpiece.memory import MemoryStore
    for choice, cls in (("python", sonar_mod.Sonar), ("rust", rust_sonar.RustSonar)):
        monkeypatch.setenv("KITE_SONAR_ENGINE", choice)
        cfg = Cfg()
        cfg.speech.cache_dir = offline_speech / "cache"
        logs = []
        mem = MemoryStore(project_id="", project_key="", store=offline_speech / "memory.json", log=logs.append)
        a = Earpiece(cfg, output="none", network=False, log=logs.append, memory=mem)
        assert isinstance(a.sonar, cls)
        assert f"[sonar] engine: {'rust (kite_audio)' if choice == 'rust' else 'python'}" in logs


def test_rust_tick_is_not_slower_than_python():
    """A 20-object crowd at 50 Hz: the native decisions cost less than the Python ones."""
    cost = {}
    objs = [Obstacle(f"c{i}", "person", -40 + i * 4, 1.2 + 0.1 * i, None, 0.9) for i in range(20)]
    for name, make in (("python", sonar_mod.Sonar), ("rust", rust_sonar.RustSonar)):
        eng = RecEngine()
        s = make(eng, SonarCfg())
        best = math.inf
        for rep in range(3):
            t0 = time.perf_counter()
            for i in range(300):
                now = rep * 10 + i * 0.02
                if i % 5 == 0:
                    s.update_frame("cam", objs, seq=rep * 1000 + i, now=now)
                eng.now = int(now * RATE)
                s.tick(now=now)
            best = min(best, time.perf_counter() - t0)
        cost[name] = best / 300 * 1e6
    print(f"\n[perf] sonar frame+tick, 20 objects: python {cost['python']:.1f} us, rust {cost['rust']:.1f} us")
    assert cost["rust"] < cost["python"]
