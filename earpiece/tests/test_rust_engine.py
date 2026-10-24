"""The Rust engine must sound exactly like the Python one, and survive what Python can't."""

import threading
import time

import numpy as np
import pytest

from earpiece import rust_engine, sounds, spatial
from earpiece.mixer import Engine

pytestmark = pytest.mark.skipif(not rust_engine.available(), reason="kite_audio not built")
RATE = 48000


def script(e, render):
    """A busy minute of audio compressed: overlapping cues on both buses, speech that is
    cut and replaced, a fade-out, a mute, gain changes, and a limiter pile-up."""
    out = []
    pip = spatial.render(sounds.proximity_cue("person", 800), -30, RATE)
    buzz = spatial.render(sounds.proximity_cue("vehicle", 900), 60, RATE)
    e.play(pip, at=100, bus="cue", gain=0.8, tag="a")
    e.play(buzz, at=700, bus="alert", gain=1.0, tag="b")
    out.append(render(1024))
    e.speech_begin("s1")
    e.speech_feed("s1", (0.3 * np.sin(np.arange(20000) / 7)).astype(np.float32))
    e.speech_end("s1")
    for i in range(6):
        e.play(pip, at=e.now + 300 * i, bus="cue", gain=0.5 + 0.1 * i)
        out.append(render(1024))
    e.speech_cancel("s1", fade_s=0.02)
    out.append(render(1024))
    e.speech_begin("s2")
    e.speech_feed("s2", np.full(3000, 0.2, np.float32))
    e.speech_feed("s2", np.full(3000, -0.2, np.float32))
    e.speech_end("s2")
    e.play(np.ones((30000, 2), np.float32) * 0.6, bus="cue", tag="long")
    out.append(render(1024))
    e.stop("long", fade_s=0.01)
    e.master_gain = 0.7
    out.append(render(1024))
    e.cues_muted = True
    e.play(pip, bus="cue")
    e.play(buzz, bus="alert")
    out.append(render(1024))
    e.cues_muted = False
    for _ in range(40):
        e.play(np.ones((4000, 2), np.float32), bus="alert")
    for _ in range(8):
        out.append(render(1024))
    return np.concatenate(out)


def test_parity_with_python_engine():
    py = Engine(RATE)
    rs = rust_engine.RustEngine(RATE)
    a = script(py, py.render)
    b = script(rs, rs.render)
    assert a.shape == b.shape
    assert np.abs(a - b).max() < 2e-5, f"max diff {np.abs(a - b).max()}"
    assert np.abs(b).max() <= rs.ceiling + 1e-6
    rs.shutdown()


def test_speech_events_flow_back_as_strings():
    rs = rust_engine.RustEngine(RATE)
    rs.speech_begin("hello-1")
    rs.speech_feed("hello-1", np.full(2000, 0.1, np.float32))
    rs.speech_end("hello-1")
    for _ in range(4):
        rs.render(1024)
    evs = []
    t0 = time.time()
    while len(evs) < 2 and time.time() - t0 < 1:
        try:
            evs.append(rs.speech_events.get(timeout=0.1)[:2])
        except Exception:
            pass
    assert evs == [("start", "hello-1"), ("done", "hello-1")]
    assert not rs.speaking
    rs.shutdown()


def test_render_releases_the_gil_and_is_fast():
    rs = rust_engine.RustEngine(RATE)
    for i in range(60):
        rs.play(np.ones((4800, 2), np.float32) * 0.01, at=i * 200)
    t0 = time.perf_counter()
    for _ in range(200):
        rs.render(1024)
    per_block_ms = (time.perf_counter() - t0) * 1000 / 200
    assert per_block_ms < 1.0
    rs.shutdown()


@pytest.mark.skipif(not rust_engine._ka or not rust_engine._ka.Engine.outputs(), reason="no audio device")
def test_real_device_survives_python_abuse():
    """Open the real output, then hammer the GIL from several threads (pure-Python CPU
    burn + thousands of play() calls). The Rust callback must never be starved."""
    rs = rust_engine.RustEngine(RATE)
    name, ch, buf = rs.open("", 512)
    assert name
    time.sleep(0.3)
    stop = threading.Event()

    def burn():
        x = 0
        while not stop.is_set():
            for i in range(100000):
                x += i * i % 7

    def spam():
        quiet = np.zeros((480, 2), np.float32)
        while not stop.is_set():
            rs.play(quiet, bus="cue")

    ths = [threading.Thread(target=burn) for _ in range(3)] + [threading.Thread(target=spam) for _ in range(2)]
    for t in ths:
        t.start()
    b0 = rs.snapshot_stats()["blocks"]
    time.sleep(3.0)
    stop.set()
    for t in ths:
        t.join()
    st = rs.snapshot_stats()
    expected = 3.0 * RATE / (buf or 512)
    assert st["blocks"] - b0 > 0.85 * expected, st
    assert st["glitches"] == 0, st
    assert st["max_render_ms"] < 5.0, st
    assert rs.callback_age_ms() < 200
    rs.shutdown()
