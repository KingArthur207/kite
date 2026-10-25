"""The native sonar thread (KITE_SONAR_THREAD=native).

1. Same cue stream: one native step (tick, watchdog, Rust synthesis and placement,
   scheduling into the mixer) must produce exactly the audio and speech of the
   Python-ticked path (RustSonar.tick + sonar._cue + engine.play + the app's watchdog),
   given the same tick times. Checked sample for sample over every recorded and
   simulated timeline.
2. Python stalls: with the interpreter frozen for 500 ms (the GIL held by a busy loop),
   native beeps stay on their beat to within one 512-frame block, while the
   Python-ticked path misses them. Real time, rendered by the native null output.
3. The stale-perception watchdog works natively: beeps stop on stale data and the
   sonar switches off while Python is frozen.
4. Fallbacks when the native path is unavailable.

Nothing is played aloud: the native null output renders into memory.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from earpiece import events, rust_engine, rust_sonar, sim, sounds
from earpiece import sonar as sonar_mod
from earpiece.config import Cfg, SonarCfg
from earpiece.events import Obstacle

pytestmark = pytest.mark.skipif(not (rust_sonar.native_available() and rust_engine.available()),
                                reason="kite_audio built without the native sonar thread")

RATE = 48000
BLOCK = 512
REPO = Path(__file__).resolve().parents[2]
STALE_S = 1.5                          # WatchdogCfg.perception_stale_s


# ---------------------------------------------------------------- 1. same cue stream
def timelines():
    out = [(f"sim:{n}", sim.timeline(n)) for n in sim.SCENARIOS]
    for p in sorted((REPO / "data").glob("*.jsonl")) + sorted((REPO / "data" / "scenarios").glob("*.jsonl")):
        out.append((p.name, sim.jsonl_timeline(p)))
    return out


def perception_msgs(tl):
    """(t, Msg) for every obstacle message, parsed exactly as the transport would."""
    out = []
    for t, m in tl:
        if m.get("type") in ("obstacles", "obstacle"):
            try:
                out.append((t, events.parse(m)))
            except ValueError:
                pass
    return out


class PyTicked:
    """The default path: RustSonar ticked from Python, cues played by Python, the app's watchdog."""

    def __init__(self, cfg):
        self.eng = rust_engine.RustEngine(RATE)
        self.said: list[tuple] = []
        self.transitions: list[tuple] = []
        self.s = rust_sonar.RustSonar(self.eng, cfg, speak=lambda *a, **k: self.said.append((a, k)))
        self.last = None
        self.ok = None

    def feed(self, msg, now):
        self.last = now
        if msg.type == "obstacles":
            self.s.update_frame(msg.source, msg.objects, msg.seq, now)
        else:
            for o in msg.objects:
                self.s.update_one(msg.source, o, msg.ttl, now)

    def step(self, now):
        self.s.tick(now)
        if self.last is None:                       # Earpiece._watchdog
            return
        ok = now - self.last <= STALE_S
        if ok != self.ok:
            self.transitions.append((ok, self.ok is None))
            self.ok = ok
            self.s.enabled = ok
            if not ok:
                self.s.clear()


class NativeStepped:
    """The native path, one native step per tick on this thread (deterministic)."""

    def __init__(self, cfg):
        self.eng = rust_engine.RustEngine(RATE)
        self.said: list[tuple] = []
        self.s = rust_sonar.RustSonar(self.eng, cfg, speak=lambda *a, **k: self.said.append((a, k)))
        self.transitions: list[tuple] = []

    def feed(self, msg, now):
        self.s.note_perception(now)
        if msg.type == "obstacles":
            self.s.update_frame(msg.source, msg.objects, msg.seq, now)
        else:
            for o in msg.objects:
                self.s.update_one(msg.source, o, msg.ttl, now)

    def step(self, now):
        self.s.native_step(now)
        self.s.poll()
        self.transitions += self.s.take_perception_events()


def run_pair(tl, cfg=None, t0=5000.0, tick=0.02, frames=960, tail_s=2.5):
    cfg = cfg or SonarCfg()
    a, b = PyTicked(cfg), NativeStepped(cfg)
    msgs = perception_msgs(tl)
    end = (msgs[-1][0] if msgs else 0.0) + tail_s
    i, t, n_audio = 0, 0.0, 0
    out_a, out_b = [], []
    while t <= end:
        while i < len(msgs) and msgs[i][0] <= t:
            for side in (a, b):
                side.feed(msgs[i][1], t0 + msgs[i][0])
            i += 1
        for side in (a, b):
            side.step(t0 + t)
        out_a.append(a.eng.render(frames))
        out_b.append(b.eng.render(frames))
        n_audio += frames
        t += tick
    a.eng.shutdown()
    b.eng.shutdown()
    return a, b, np.concatenate(out_a), np.concatenate(out_b)


@pytest.mark.parametrize("name,tl", timelines(), ids=[n for n, _ in timelines()])
def test_native_steps_match_the_python_ticked_path_sample_for_sample(name, tl):
    a, b, wa, wb = run_pair(tl)
    assert a.said == b.said, name
    assert a.transitions == b.transitions, name
    assert a.s.stats == b.s.stats, name
    assert wa.shape == wb.shape
    assert np.array_equal(wa, wb), f"{name}: {int(np.sum(wa != wb))} samples differ"
    if a.s.stats["beeps"]:
        assert np.abs(wa).max() > 0.01            # the scenario really beeped


def test_random_multi_source_streams_match():
    rng = random.Random(11)
    for seed in range(6):
        tl, t = [], 0.0
        ids = [f"o{k}" for k in range(rng.randint(1, 6))]
        kinds = {k: rng.choice(["person", "vehicle", "bike", "pole", "curb", "stairs_down"]) for k in ids}
        while t < 6.0:
            objs = [{"id": k, "kind": kinds[k], "bearing": rng.uniform(-120, 120),
                     "distance": rng.uniform(0.3, 6.0), "confidence": rng.uniform(0.3, 1.0)}
                    for k in ids if rng.random() < 0.85]
            tl.append((t, {"type": "obstacles", "source": "cam", "seq": int(t * 15) + 1, "objects": objs}))
            if rng.random() < 0.05:
                t += rng.uniform(0.5, 2.5)         # dropouts: fades, prunes, the watchdog
            t += 1 / 15
        a, b, wa, wb = run_pair(tl)
        assert a.said == b.said and a.transitions == b.transitions, seed
        assert np.array_equal(wa, wb), seed


def test_gentle_profile_and_profile_switch_match():
    tl = sim.timeline("car") + [(t + 4.5, m) for t, m in sim.timeline("crowd")]
    sounds.set_profile("gentle")
    try:
        a, b, wa, wb = run_pair(tl)
    finally:
        sounds.set_profile("standard")
    assert a.said == b.said and np.array_equal(wa, wb)
    # and back: a fresh standard run differs from the gentle one but still matches itself
    a2, b2, wa2, wb2 = run_pair(tl)
    assert np.array_equal(wa2, wb2) and not np.array_equal(wa, wa2)


# ---------------------------------------------------------------- 2. stalls (real time)
def onsets(audio: np.ndarray, thr: float = 0.02, quiet_s: float = 0.03) -> np.ndarray:
    """Frames where a beep starts: above `thr` after at least `quiet_s` below it."""
    idx = np.flatnonzero(np.abs(audio).max(axis=1) > thr)
    if idx.size == 0:
        return idx
    return idx[np.r_[True, np.diff(idx) > int(quiet_s * RATE)]]


def hold_the_gil(seconds: float) -> None:
    """Freeze every other Python thread: a busy loop that never lets the GIL go."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(30.0)
    try:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            pass
    finally:
        sys.setswitchinterval(old)


class Heartbeat(threading.Thread):
    """Measures how long Python threads were really kept from running."""

    def __init__(self):
        super().__init__(daemon=True)
        self.max_gap = 0.0
        self.halt = threading.Event()

    def run(self):
        last = time.perf_counter()
        while not self.halt.is_set():
            time.sleep(0.005)
            now = time.perf_counter()
            self.max_gap = max(self.max_gap, now - last)
            last = now


def live_run(native: bool, stall_s: float, cfg: SonarCfg, before_s=1.5, after_s=1.2, distance=0.9):
    eng = rust_engine.RustEngine(RATE)
    eng._e.open_null(BLOCK, before_s + stall_s + after_s + 2.0)
    said: list = []
    s = rust_sonar.RustSonar(eng, cfg, speak=lambda *a, **k: said.append(a), thread=native,
                             perception_stale_s=STALE_S)
    halt = threading.Event()
    marks: dict[str, int] = {}

    def camera():                                  # the perception feed, 15 Hz, from Python
        seq = 0
        while not halt.is_set():
            seq += 1
            now = time.monotonic()
            s.note_perception(now)
            s.update_frame("cam", [Obstacle("p", "pole", 0.0, distance)], seq, now)
            marks["last_frame"] = eng.now
            time.sleep(1 / 15)

    def control():                                 # the app's 50 Hz control loop (Python path)
        nxt = time.monotonic()
        while not halt.is_set():
            s.tick(time.monotonic())
            nxt += 0.02
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()

    hb = Heartbeat()
    threads = [threading.Thread(target=camera, daemon=True), hb]
    if not native:
        threads.append(threading.Thread(target=control, daemon=True))
    for th in threads:
        th.start()
    time.sleep(before_s)
    marks["stall_start"] = eng.now
    marks["frame_before_stall"] = marks.get("last_frame", 0)
    hold_the_gil(stall_s)
    marks["stall_end"] = eng.now
    time.sleep(after_s)
    halt.set()
    hb.halt.set()
    for th in threads:
        th.join(2)
    stats = s.native_stats() if native else {}
    s.close()
    eng.close()
    audio = np.frombuffer(eng._e.take_recording(), dtype="<f4").reshape(-1, 2)
    eng.shutdown()
    return audio, marks, hb.max_gap, stats, s


def grid_error_frames(on: np.ndarray, first: int = 3) -> tuple[int, int]:
    """(step, worst distance of an onset from the beat grid) after the first beeps settle."""
    on = on[first:]
    step = int(np.median(np.diff(on)))
    k = np.round((on - on[0]) / step)
    return step, int(np.max(np.abs(on - (on[0] + k * step))))


def beats_due_in_stall(on: np.ndarray, step: int, lo: int, hi: int) -> tuple[int, int]:
    """(beats the pre-stall beat grid puts inside [lo, hi), how many of them sounded on time)."""
    ref = on[on < lo][-1]
    due = [ref + k * step for k in range(1, 100) if lo <= ref + k * step < hi]
    hit = sum(1 for d in due if np.any(np.abs(on - d) <= BLOCK))
    return len(due), hit


def stall_window(marks: dict) -> tuple[int, int]:
    # beats due in the first 60 ms of the stall were already scheduled (the lookahead)
    return marks["stall_start"] + int(0.07 * RATE), marks["stall_end"]


def test_a_500_ms_python_stall_does_not_move_native_beeps():
    cfg = SonarCfg()
    cfg.stale_track_s = 1.0     # keep the track alive across the stall: this test is about timing,
    #                             the stale-data policy has its own test below
    audio, marks, gap, stats, s = live_run(True, 0.5, cfg)
    assert gap >= 0.45, f"the GIL was not really held (longest Python gap {gap * 1e3:.0f} ms)"
    on = onsets(audio)
    step, worst = grid_error_frames(on)
    due, hit = beats_due_in_stall(on, step, *stall_window(marks))
    print(f"\nnative: {len(on)} beeps, beat {step / RATE * 1e3:.0f} ms, worst onset off the beat "
          f"{worst} frames ({worst / RATE * 1e3:.2f} ms), {hit}/{due} beats due inside the "
          f"{(marks['stall_end'] - marks['stall_start']) / RATE * 1e3:.0f} ms stall sounded on time; "
          f"longest Python gap {gap * 1e3:.0f} ms; native step max {stats['max_step_ms']:.2f} ms, "
          f"late wake max {stats['max_late_ms']:.2f} ms")
    assert worst <= BLOCK, worst
    assert due >= 1 and hit == due                  # the beat went on through the stall
    assert stats["dropped_beeps"] == 0


def test_the_same_stall_delays_beeps_on_the_python_ticked_path():
    """The reason for the native thread: a Python-ticked sonar misses the beat."""
    cfg = SonarCfg()
    cfg.stale_track_s = 1.0
    audio, marks, gap, _, _ = live_run(False, 0.5, cfg)
    assert gap >= 0.45
    on = onsets(audio)
    step = int(np.median(np.diff(on[3:])))
    due, hit = beats_due_in_stall(on, step, *stall_window(marks))
    late = on[on >= marks["stall_end"]][0] - marks["stall_start"]
    print(f"\npython-ticked: beat {step / RATE * 1e3:.0f} ms, {hit}/{due} beats due inside the stall "
          f"sounded on time; first beep after the stall began came {late / RATE * 1e3:.0f} ms in")
    assert due >= 1 and hit == 0


# ---------------------------------------------------------------- 3. watchdog
def test_native_watchdog_silences_stale_perception_while_python_is_frozen():
    cfg = SonarCfg()                               # default fade: stale_track_s 0.6
    eng = rust_engine.RustEngine(RATE)
    eng._e.open_null(BLOCK, 8.0)
    s = rust_sonar.RustSonar(eng, cfg, speak=lambda *a, **k: None, thread=True, perception_stale_s=STALE_S)
    seq = 0

    def frames(seconds):
        """Feed frames for `seconds`; returns (engine frame, monotonic time) of the last one."""
        nonlocal seq
        end = time.monotonic() + seconds
        last = (0, 0.0)
        while time.monotonic() < end:
            seq += 1
            now = time.monotonic()
            s.note_perception(now)
            s.update_frame("cam", [Obstacle("p", "pole", 0.0, 1.2)], seq, now)
            last = (eng.now, now)
            time.sleep(1 / 15)
        return last

    try:
        (last_frame, last_t) = frames(1.2)
        assert s.perception_ok is True
        hold_the_gil(2.2)                          # nobody in Python can run, let alone feed frames
        frozen_end = eng.now
        assert s.perception_ok is False            # decided natively, during the freeze
        assert not s.enabled and not s.tracks
        _, ev = s._s.poll()
        ev = [e for e in ev if e[0] == "perception"]
        assert [e[:3] for e in ev] == [("perception", True, True), ("perception", False, False)]
        lost_after = ev[1][3] - last_t
        frames(1.0)
        assert s.perception_ok is True and s.enabled
        s.poll()
        assert s.take_perception_events() == [(True, False)]
    finally:
        s.close()
        eng.close()
    audio = np.frombuffer(eng._e.take_recording(), dtype="<f4").reshape(-1, 2)
    eng.shutdown()
    on = onsets(audio)
    # the track is pruned stale_track_s after its last frame; a beep scheduled by the tick
    # before that may still start up to the 60 ms lookahead (+ one 20 ms period) later
    quiet_from = last_frame + int((cfg.stale_track_s + 0.08) * RATE)
    stale = on[(on > quiet_from) & (on < frozen_end)]
    last_beep = (on[on < frozen_end][-1] - last_frame) / RATE
    print(f"\nwatchdog: last beep {last_beep * 1e3:.0f} ms after the last frame, sonar off "
          f"{lost_after * 1e3:.0f} ms after it, Python frozen meanwhile")
    assert stale.size == 0, f"beeped on stale data at {stale / RATE}"
    assert STALE_S <= lost_after <= STALE_S + 0.1
    assert on[on > frozen_end].size > 0            # and resumed once frames came back


# ---------------------------------------------------------------- 4. lifecycle and fallbacks
def test_lifecycle_and_argument_checks():
    eng = rust_engine.RustEngine(RATE)
    s = rust_sonar.RustSonar(eng, SonarCfg(), thread=True)
    try:
        assert s.thread and s._s.native_running
        with pytest.raises(RuntimeError):
            s._s.start_native(eng._e, time.monotonic())
        with pytest.raises(RuntimeError):
            s.native_step(time.monotonic())
    finally:
        s.close()
        s.close()
    assert not s._s.native_running
    with pytest.raises(ValueError):
        s._s.start_native(eng._e, time.monotonic(), 0.2)          # period past the 60 ms lookahead
    with pytest.raises(ValueError):
        s._s.start_native(eng._e, time.monotonic(), 0.02, 1.5, "loud")
    other = rust_engine.RustEngine(44100)
    with pytest.raises(ValueError):
        rust_sonar.RustSonar(other, SonarCfg())._s.start_native(other._e, time.monotonic())
    eng.shutdown()
    other.shutdown()


def test_selection_and_fallbacks(monkeypatch):
    from earpiece.mixer import Engine as PyEngine
    logs = []
    cfg = SonarCfg()
    cfg.engine = "auto"                            # whatever KITE_SONAR_ENGINE says
    cfg.thread = "native"
    eng = rust_engine.RustEngine(RATE)
    s = sonar_mod.create(eng, cfg, log=logs.append)
    assert isinstance(s, rust_sonar.RustSonar) and s.thread
    s.close()
    # the Python audio engine: Rust decisions, ticked from Python
    s = sonar_mod.create(PyEngine(RATE), cfg, log=logs.append)
    assert isinstance(s, rust_sonar.RustSonar) and not s.thread
    assert "audio engine is the Python one" in logs[-1]
    # another rate: same fallback
    s = sonar_mod.create(rust_engine.RustEngine(44100), cfg, log=logs.append)
    assert not s.thread and "48000" in logs[-1]
    # extension missing: the Python sonar
    monkeypatch.setattr(rust_sonar, "_ka", None)
    s = sonar_mod.create(eng, cfg, log=logs.append)
    assert isinstance(s, sonar_mod.Sonar)
    assert "not selected or not built" in logs[-1]
    monkeypatch.undo()
    cfg.thread = "sideways"
    with pytest.raises(ValueError):
        sonar_mod.create(eng, cfg)
    cfg.thread = "python"
    s = sonar_mod.create(eng, cfg, log=logs.append)
    assert isinstance(s, rust_sonar.RustSonar) and not s.thread
    eng.shutdown()


def test_profile_change_reaches_the_native_cache():
    eng = rust_engine.RustEngine(RATE)
    s = rust_sonar.RustSonar(eng, SonarCfg(), thread=True)
    try:
        sounds.set_profile("gentle")
        time.sleep(0.1)
        assert s._s.native_running
    finally:
        sounds.set_profile("standard")
        s.close()
        eng.shutdown()
    assert s._s.set_profile not in sounds._profile_listeners


# ---------------------------------------------------------------- the whole app
def test_app_with_the_native_thread(offline_speech, monkeypatch):
    from earpiece.app import Earpiece
    from earpiece.output import RustNullOutput
    cfg = Cfg()
    cfg.audio.engine = cfg.sonar.engine = "rust"   # whatever the environment selects
    cfg.sonar.thread = "native"
    cfg.speech.cache_dir = offline_speech / "cache"
    logs: list[str] = []
    app = Earpiece(cfg, output="null", record=True, network=False, log=logs.append)
    said: list[str] = []
    chained = app.speaker.on_spoken
    app.speaker.on_spoken = lambda rec: (said.append(rec["text"]), chained and chained(rec))
    assert isinstance(app.output, RustNullOutput) and app.sonar.thread
    assert any("native thread" in line for line in logs)
    app.start(prewarm=True, announce=False)
    try:
        t0 = time.monotonic()
        tl = sim.timeline("car") + [(t + 6.0, m) for t, m in sim.timeline("dropout")]
        for t, m in tl:
            d = t0 + t - time.monotonic()
            if d > 0:
                time.sleep(d)
            app.handle(events.parse(m))
        time.sleep(1.0)
        app.speaker.wait_idle(5)
    finally:
        app.stop()
    st = app.sonar.native_stats()
    assert st["beeps"] > 10 and st["dropped_beeps"] == 0
    assert any(s.startswith("Stop. Car.") for s in said), said
    assert "Obstacle detection paused." in said and "Obstacle detection is back." in said
    audio = app.output.recorded()
    assert audio.shape[0] > 10 * RATE and np.abs(audio).max() > 0.05
    assert not app.sonar._s.native_running
