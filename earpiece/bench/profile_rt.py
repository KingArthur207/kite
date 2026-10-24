"""Where the earpiece spends CPU, and where a Python stall would move a warning beep.

Headless and silent: null audio output, fake offline voice, no network voices, no camera.
Run from earpiece/:

    python bench/profile_rt.py                 # everything (about a minute of real-time replay)
    python bench/profile_rt.py micro stalls    # only some sections: micro, replay, stalls, beepsim

Sections
  micro    cost per call of everything on the warning path, Python and Rust side by side
  replay   data/sample_walk.jsonl and the sim scenarios through the whole app over UDP, in real
           time: CPU per stage, control-loop period jitter, beep scheduling headroom, cue cache misses
  stalls   how long common Python operations hold the interpreter (full GC, a big scene_image)
  beepsim  what a stall of N ms on the Python control loop does to beep onsets (deterministic)
"""

from __future__ import annotations

import base64
import collections
import gc
import json
import math
import os
import resource
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
REPO = HERE.parents[1]

os.environ["CARTESIA_API_KEY"] = ""          # never spend credits; the fake offline voice below speaks
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("KITE_SPACETIME", None)
os.environ["KITE_CAMERA"] = "off"
os.environ["KITE_JUDGMENT"] = "0"

import numpy as np  # noqa: E402

from earpiece import events, sim, sonar as sonar_mod, sounds, spatial, speech  # noqa: E402
from earpiece.config import Cfg, SonarCfg  # noqa: E402


def _fake_tts(text, rate, voice="", wpm=210):
    n = int(max(0.15, 0.03 * len(text)) * rate)
    return (0.2 * np.sin(2 * np.pi * 440 * np.arange(n) / rate)).astype(np.float32)


speech.offline_tts = _fake_tts


def timeit(fn, n: int = 200, warm: int = 5) -> float:
    """Median microseconds per call."""
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


# ---------------------------------------------------------------- micro
def micro() -> None:
    print("\n== micro: cost per call (median) ==")
    frame = lambda n: json.dumps({"type": "obstacles", "source": "webcam", "seq": 12, "objects": [
        {"id": str(i), "kind": "person", "bearing": -20 + 3 * i, "distance": 2.4 + 0.1 * i,
         "closing_speed": 0.4, "confidence": 0.8} for i in range(n)]}).encode()
    for n in (0, 3, 20):
        raw = frame(n)
        print(f"  events.parse, obstacles frame with {n:2d} objects ({len(raw):4d} B): "
              f"{timeit(lambda: events.parse(raw), 2000):7.1f} us")

    from earpiece import rust_sonar

    class Eng:
        rate, now = 48000, 0

        def samples(self, s):
            return int(round(s * self.rate))

        def play(self, *a, **k):
            pass

    for n in (3, 20):
        objs = [events.Obstacle(str(i), "person", -20 + 3 * i, 2.4 + 0.1 * i, 0.4, 0.8) for i in range(n)]
        for name, cls in (("python", sonar_mod.Sonar), ("rust", getattr(rust_sonar, "RustSonar", None))):
            if cls is None or (name == "rust" and not rust_sonar.available()):
                continue
            s = cls(Eng(), SonarCfg())
            st = {"t": 0.0, "seq": 0}

            def step(s=s, st=st, objs=objs):
                st["t"] += 0.02
                st["seq"] += 1
                s.update_frame("cam", objs, st["seq"], st["t"])
                s.tick(st["t"])
            print(f"  sonar frame + tick, {n:2d} objects, {name:6s}: {timeit(step, 500):7.1f} us")

    def py(fn):                       # sounds.py's own numpy/scipy code, whatever is built
        def run():
            with sounds.reference():
                return fn()
        return run

    engines = [("python", py)] + ([("rust", lambda fn: fn)] if sounds.NATIVE else [])
    print("  cue synthesis, mono, per kind:")
    for kind in ("person", "vehicle", "obstacle", "curb"):
        print(f"    {kind:9s} " + ", ".join(
            f"{name} {timeit(wrap(lambda: sounds.proximity_cue(kind, 900.0)), 100):6.0f} us"
            for name, wrap in engines))
    mono = sounds.proximity_cue("vehicle", 900.0)
    print(f"  binaural placement of one cue ({'rust' if spatial.NATIVE else 'python'}): "
          f"{timeit(lambda: spatial.render(mono, 35.0, 48000), 200):7.0f} us")
    for name, wrap in engines:
        miss = wrap(lambda: sonar_mod._cue_render.__wrapped__('vehicle', 900, 35, 48000))
        print(f"  sonar cue cache miss (synth + placement, what tick() pays inline), {name} synth: "
              f"{timeit(miss, 100):7.0f} us")
        print(f"  beacon cue synthesis ('near'), {name}: "
              f"{timeit(wrap(lambda: sounds.beacon_tick('near', 30.0)), 100):7.0f} us")
    combos = len(range(525, 1251, 25)) * len(spatial.BEARINGS_5DEG) * 4
    print(f"  distinct sonar cues (4 timbres x 30 pitches x 73 bearings): {combos}; "
          f"prewarm covers the {4 * len(spatial.BEARINGS_5DEG)} critical ones, lru_cache holds 2048")

    from earpiece.mixer import Engine
    from earpiece import rust_engine
    for name, E in (("python", Engine), ("rust", rust_engine.RustEngine if rust_engine.available() else None)):
        if E is None:
            continue
        e = E(48000)
        st = spatial.render(mono, 0.0, 48000)
        for i in range(8):
            e.play(st, at=i * 3000, bus="cue")
        print(f"  mixer render, 512-frame block with ~8 voices, {name:6s}: {timeit(lambda: e.render(512), 300):7.1f} us")
        if hasattr(e, "shutdown"):
            e.shutdown()


# ---------------------------------------------------------------- replay
class Meter:
    """Wrap callables and accumulate wall time per stage (calls from any thread)."""

    def __init__(self):
        self.t = collections.defaultdict(float)
        self.n = collections.Counter()
        self.lock = threading.Lock()

    def wrap(self, owner, attr: str, name: str):
        fn = getattr(owner, attr)

        def w(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                with self.lock:
                    self.t[name] += dt
                    self.n[name] += 1
        setattr(owner, attr, w)
        return fn


def replay_one(name: str, timeline: list, dur: float) -> dict:
    from earpiece.app import Earpiece
    from earpiece import transport as transport_mod
    import tempfile
    m = Meter()
    orig_parse = m.wrap(transport_mod, "parse", "events.parse (per datagram)")
    sonar_mod._cue_render.cache_clear()
    miss_t: list[float] = []
    orig_mono = sonar_mod._mono_cue

    def mono_timed(*a):
        t0 = time.perf_counter()
        try:
            return orig_mono(*a)
        finally:
            miss_t.append(time.perf_counter() - t0)
    sonar_mod._mono_cue = mono_timed
    cfg = Cfg()
    cfg.speech.cache_dir = Path(tempfile.mkdtemp()) / "tts"
    cfg.net.udp_port = cfg.net.ws_port = 0
    app = Earpiece(cfg, output="null", record=False, log=lambda s: None)
    m.wrap(app.sonar, "update_frame", "sonar.update_frame")
    m.wrap(app.beacon, "tick", "beacon.tick")
    ticks: list[float] = []
    tick_cost: list[float] = []
    orig_tick = app.sonar.tick

    def tick(now=None):
        t0 = time.perf_counter()
        ticks.append(t0)
        try:
            return orig_tick(now)
        finally:
            tick_cost.append(time.perf_counter() - t0)
    app.sonar.tick = tick
    headroom: list[int] = []
    orig_play = app.engine.play

    def play(stereo, at=None, bus="cue", gain=1.0, tag=None):
        if tag and tag.startswith("sonar:") and at is not None:
            headroom.append(at - app.engine.now)
        return orig_play(stereo, at=at, bus=bus, gain=gain, tag=tag)
    app.engine.play = play
    t_pw = time.perf_counter()
    app.start(prewarm=True, announce=False)
    for th in threading.enumerate():                # let the cue pre-warm finish first
        if th.name == "cue-prewarm":
            th.join()
    prewarm_ms = (time.perf_counter() - t_pw) * 1e3
    gcs: list[tuple[int, float]] = []
    gc_t0 = [0.0]

    def on_gc(phase, info):                         # automatic collections stop every Python thread
        if phase == "start":
            gc_t0[0] = time.perf_counter()
        else:
            gcs.append((info["generation"], (time.perf_counter() - gc_t0[0]) * 1e3))
    gc.callbacks.append(on_gc)
    port = app.transport.udp_port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ru0, w0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    t0 = time.monotonic()
    for t, msg in timeline:
        d = t0 + t - time.monotonic()
        if d > 0:
            time.sleep(d)
        sock.sendto(json.dumps(msg).encode(), ("127.0.0.1", port))
    while time.monotonic() - t0 < dur:
        time.sleep(0.05)
    ru1, wall = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter() - w0
    gc.callbacks.remove(on_gc)
    cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    beeps = app.sonar.stats["beeps"]
    app.stop()
    native = app.sonar.native_stats() if getattr(app.sonar, "native_stats", None) and app.sonar._s.native_stats()["steps"] else None
    sonar_mod._mono_cue = orig_mono
    transport_mod.parse = orig_parse
    gaps = [(b - a) * 1e3 for a, b in zip(ticks, ticks[1:])]
    hr_ms = [h / 48.0 for h in headroom]
    return {"name": name, "wall": wall, "cpu": cpu, "stages": dict(m.t), "calls": dict(m.n),
            "tick_us_p50": pct(tick_cost, 50) * 1e6, "tick_us_p99": pct(tick_cost, 99) * 1e6,
            "tick_us_max": max(tick_cost) * 1e6 if tick_cost else 0,
            "gap_p50": pct(gaps, 50), "gap_p99": pct(gaps, 99), "gap_max": max(gaps) if gaps else 0,
            "headroom_min_ms": min(hr_ms) if hr_ms else float("nan"), "late_beeps": sum(h < 0 for h in hr_ms),
            "beeps": beeps, "misses": len(miss_t), "miss_ms_total": sum(miss_t) * 1e3,
            "miss_ms_max": max(miss_t) * 1e3 if miss_t else 0.0, "ticks": len(ticks), "prewarm_ms": prewarm_ms,
            "gc": {g: (sum(1 for gg, _ in gcs if gg == g), max((d for gg, d in gcs if gg == g), default=0.0))
                   for g in (0, 1, 2)}, "native": native}


def replay() -> None:
    print("\n== replay: whole app over UDP, real time, null output ==")
    runs = [("sample_walk.jsonl", sim.jsonl_timeline(REPO / "data" / "sample_walk.jsonl"))]
    runs += [(f"sim:{n}", sim.timeline(n)) for n in ("car", "crowd")]
    for name, tl in runs:
        r = replay_one(name, tl, tl[-1][0] + 1.5)
        print(f"\n  {r['name']}: {r['wall']:.1f} s wall, process CPU {r['cpu']:.2f} s "
              f"({100 * r['cpu'] / r['wall']:.1f} % of one core), {r['beeps']} beeps")
        for k, v in sorted(r["stages"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:30s} {v * 1e3:8.1f} ms total, {r['calls'][k]:6d} calls, "
                  f"{v / max(1, r['calls'][k]) * 1e6:7.1f} us/call, {100 * v / r['wall']:.2f} % of wall")
        print(f"    startup cue pre-warm (a Python thread competing for the GIL): {r['prewarm_ms']:.0f} ms")
        print(f"    sonar.tick: p50 {r['tick_us_p50']:.0f} us, p99 {r['tick_us_p99']:.0f} us, "
              f"max {r['tick_us_max'] / 1e3:.1f} ms over {r['ticks']} ticks")
        print(f"    control loop period (target 20 ms): p50 {r['gap_p50']:.1f}, p99 {r['gap_p99']:.1f}, "
              f"max {r['gap_max']:.1f} ms")
        print(f"    sonar cue cache misses synthesised on the control loop: {r['misses']} "
              f"({r['miss_ms_total']:.0f} ms total, worst {r['miss_ms_max']:.1f} ms)")
        print("    automatic GC: " + ", ".join(f"gen{g} x{c} (worst {mx:.1f} ms)" for g, (c, mx) in r["gc"].items()))
        if r["native"]:
            n = r["native"]
            print(f"    native sonar thread: {n['steps']:.0f} steps, {n['beeps']:.0f} beeps scheduled, step max "
                  f"{n['max_step_ms']:.2f} ms, wake-up lateness max {n['max_late_ms']:.1f} ms (beeps are "
                  f"scheduled 60 ms ahead), {n['cue_misses']:.0f} cue misses (worst {n['max_miss_ms']:.2f} ms, "
                  f"off the GIL)")
        else:
            print(f"    beep scheduling headroom (at - engine.now): min {r['headroom_min_ms']:.1f} ms, "
                  f"late beeps {r['late_beeps']}")


# ---------------------------------------------------------------- stalls
def stalls() -> None:
    print("\n== stalls: how long ordinary Python work holds the interpreter ==")
    from earpiece.app import Earpiece  # noqa: F401  (load the whole app's module graph)
    import anthropic  # noqa: F401
    for gen in (0, 1, 2):
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            gc.collect(gen)
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"  gc.collect({gen}) with the app's modules loaded: median {statistics.median(ts):.2f} ms, "
              f"max {max(ts):.2f} ms ({len(gc.get_objects())} tracked objects)")
    blob = base64.b64encode(b"\xff\xd8\xff\xe0" + os.urandom(175_000) + b"\xff\xd9").decode()
    raw = json.dumps({"type": "scene_image", "camera": "drone", "jpeg": blob}).encode()
    print(f"  events.parse of a 180 KB scene_image ({len(raw) // 1024} KB JSON): "
          f"{timeit(lambda: events.parse(raw), 50) / 1e3:.2f} ms")
    def cold():
        with sounds.reference():
            return sonar_mod._cue_render.__wrapped__('person', 650, -15, 48000)
    print(f"  a cold sonar cue (numpy synth + place) on the control loop: {timeit(cold, 50) / 1e3:.2f} ms")
    print(f"  sys.getswitchinterval() = {sys.getswitchinterval() * 1e3:.0f} ms: a busy Python thread keeps "
          f"the GIL this long before another thread may run")


# ---------------------------------------------------------------- beepsim
def beepsim() -> None:
    """The Python-ticked sonar against a free-running audio clock: one control-loop stall of N ms."""
    print("\n== beepsim: a pole 0.9 m ahead, one control-loop stall, beep onset error ==")
    class Clock:
        def __init__(self):
            self.rate, self.now, self.plays = 48000, 0, []

        def samples(self, s):
            return int(round(s * self.rate))

        def play(self, stereo, at=None, bus="cue", gain=1.0, tag=None):
            self.plays.append((at, self.now))       # requested frame, clock when the request arrived

    for stall_ms in (0, 30, 50, 100, 200, 500):
        clk = Clock()
        s = sonar_mod.Sonar(clk, SonarCfg())
        t, stalled = 0.0, False
        while t < 3.0:
            if 1.0 <= t and not stalled and stall_ms:
                t += stall_ms / 1e3                   # Python was busy: no frame, no tick
                stalled = True
            clk.now = int(t * 48000)
            s.update_frame("cam", [events.Obstacle("p", "pole", 0.0, 0.9)], None, t)
            s.tick(t)
            t += 0.02
        # the mixer plays a late request at once (never chops the attack): actual onset = max(at, arrival)
        onsets = [max(at, arr) for at, arr in clk.plays]
        late = [(max(at, arr) - at) / 48.0 for at, arr in clk.plays]
        same = collections.Counter(onsets)
        print(f"  stall {stall_ms:3d} ms: {len(onsets)} beeps, worst onset {max(late):6.1f} ms late, "
              f"{sum(v - 1 for v in same.values() if v > 1)} beeps stacked on another's onset")


if __name__ == "__main__":
    want = sys.argv[1:] or ["micro", "stalls", "beepsim", "replay"]
    for sec in want:
        {"micro": micro, "replay": replay, "stalls": stalls, "beepsim": beepsim}[sec]()
