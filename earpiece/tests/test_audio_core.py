import math

import numpy as np
import pytest

from earpiece import sounds, spatial
from earpiece.mixer import Engine

RATE = 48000


def noise(n=4800, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 0.3).astype(np.float32)


# ---------------------------------------------------------------- spatial
@pytest.mark.parametrize("az,sign", [(30, 1), (90, 1), (135, 1), (-30, -1), (-90, -1), (-150, -1)])
def test_level_and_time_cues_point_the_right_way(az, sign):
    st = spatial.render(noise(), az, RATE)
    assert sign * spatial.interaural_level_db(st) > 3.0
    # positive delay = left ear first; a source on the right reaches the right ear first
    assert -sign * spatial.interaural_delay_s(st, RATE) > 100e-6


def test_center_and_behind_are_symmetric():
    for az in (0, 180):
        st = spatial.render(noise(), az, RATE)
        assert abs(spatial.interaural_level_db(st)) < 0.2
        assert abs(spatial.interaural_delay_s(st, RATE)) < 30e-6


def test_lateralisation_grows_toward_the_side():
    ilds = [spatial.interaural_level_db(spatial.render(noise(), az, RATE)) for az in (0, 20, 45, 90)]
    assert ilds == sorted(ilds)


def test_max_itd_is_physical():
    itd = abs(spatial.interaural_delay_s(spatial.render(noise(), 90, RATE), RATE))
    assert 500e-6 < itd < 800e-6          # spherical head: (a/c)(pi/2 + 1) ~ 655 us


def test_loudness_constant_across_directions():
    x = noise()
    e0 = float(np.sum(x.astype(float) ** 2))
    for az in range(-180, 181, 15):
        st = spatial.render(x, az, RATE)
        assert abs(float(np.sum(st.astype(float) ** 2)) / e0 - 2.0) < 0.02


def test_wrap():
    assert spatial.wrap_deg(270) == -90
    assert spatial.wrap_deg(-270) == 90
    assert spatial.wrap_deg(180) == 180
    assert spatial.wrap_deg(-180) == 180
    assert spatial.wrap_deg(720.5) == pytest.approx(0.5)


def test_behind_is_duller_than_front():
    x = noise()

    def hf(st):
        spec = np.abs(np.fft.rfft(st.mean(axis=1)))
        f = np.fft.rfftfreq(st.shape[0], 1 / RATE)
        return spec[f > 4000].sum() / spec.sum()
    assert hf(spatial.render(x, 160, RATE)) < hf(spatial.render(x, 20, RATE)) * 0.8


# ---------------------------------------------------------------- sounds
@pytest.mark.parametrize("fn", [sounds.ready, sounds.arrived, sounds.listening, sounds.heard, sounds.error,
                                sounds.sensor_lost, sounds.sensor_back, sounds.critical_alarm,
                                sounds.crossing_go, sounds.crossing_wait, sounds.beacon_on_course,
                                sounds.beacon_off_course, sounds.beacon_behind, sounds.nav_chime])
def test_cues_are_clean(fn):
    x = fn()
    assert x.dtype == np.float32 and x.ndim == 1 and x.size > 100
    assert np.all(np.isfinite(x))
    assert 0.2 < np.abs(x).max() <= 0.61
    assert abs(x[0]) < 0.05 and abs(x[-1]) < 0.05       # no clicks at the edges


@pytest.mark.parametrize("kind", ["person", "vehicle", "obstacle", "curb", "bike", "pole"])
def test_each_class_has_a_cue(kind):
    x = sounds.proximity_cue(kind, 800)
    assert 0.03 < x.size / RATE < 0.2


def test_class_timbres_differ():
    a, b = sounds.proximity_cue("person", 800), sounds.proximity_cue("vehicle", 800)
    n = min(a.size, b.size)
    c = np.corrcoef(a[:n], b[:n])[0, 1]
    assert abs(c) < 0.5


# ---------------------------------------------------------------- mixer
def test_scheduled_cue_lands_on_the_exact_sample():
    e = Engine(RATE)
    click = np.zeros((10, 2), np.float32)
    click[0] = 0.5
    e.play(click, at=1234)
    out = np.concatenate([e.render(512) for _ in range(4)])
    assert np.flatnonzero(np.abs(out[:, 0]) > 0.1).tolist() == [1234]


def test_late_cue_plays_whole_not_chopped():
    e = Engine(RATE)
    e.render(4096)
    beep = np.ones((100, 2), np.float32) * 0.3
    e.play(beep, at=10)                      # already in the past
    out = e.render(512)
    assert np.isclose(out[:100, 0].sum(), 0.3 * 100 * e.master_gain * e.cue_gain, rtol=0.05)


def test_limiter_holds_the_ceiling_under_pileup():
    e = Engine(RATE)
    loud = np.ones((4800, 2), np.float32)
    for _ in range(60):
        e.play(loud, bus="alert", gain=1.0)
    out = np.concatenate([e.render(512) for _ in range(12)])
    assert np.abs(out).max() <= e.ceiling + 1e-6
    assert e.stats["limited_blocks"] > 0


def test_speech_stream_and_events():
    e = Engine(RATE)
    e.speech_begin("u1")
    e.speech_feed("u1", np.full(1000, 0.25, np.float32))
    e.speech_feed("u1", np.full(1000, 0.25, np.float32))
    e.speech_end("u1")
    out = np.concatenate([e.render(512) for _ in range(6)])
    assert np.isclose(np.abs(out[:2000, 0]).mean(), 0.25 * e.master_gain, rtol=0.02)
    evs = []
    while not e.speech_events.empty():
        evs.append(e.speech_events.get()[:2])
    assert evs == [("start", "u1"), ("done", "u1")]
    assert not e.speaking


def test_speech_cancel_fades_without_click():
    e = Engine(RATE)
    e.speech_begin("u1")
    e.speech_feed("u1", np.full(48000, 0.4, np.float32))
    e.speech_end("u1")
    a = e.render(512)
    e.speech_cancel("u1", fade_s=0.02)
    b = np.concatenate([e.render(512) for _ in range(4)])
    seg = np.concatenate([a[:, 0], b[:, 0]])
    assert np.abs(np.diff(seg)).max() < 0.05          # no step discontinuity
    assert np.abs(b[-512:]).max() == 0
    kinds = []
    while not e.speech_events.empty():
        kinds.append(e.speech_events.get()[0])
    assert kinds[-1] == "cut"


def test_new_utterance_cuts_the_old_one():
    e = Engine(RATE)
    e.speech_begin("a")
    e.speech_feed("a", np.full(48000, 0.4, np.float32))
    e.render(512)
    e.speech_begin("b")
    e.speech_feed("b", np.full(512, 0.1, np.float32))
    e.speech_end("b")
    e.render(512)
    e.render(512)
    evs = []
    while not e.speech_events.empty():
        evs.append(e.speech_events.get()[:2])
    assert ("cut", "a") in evs and ("done", "b") in evs


def test_cues_duck_under_speech_but_alerts_do_not():
    e = Engine(RATE)
    tone = np.full((48000, 2), 0.2, np.float32)
    e.play(tone, bus="cue")
    base = np.abs(np.concatenate([e.render(512) for _ in range(10)])).mean()
    e.speech_begin("s")
    e.speech_feed("s", np.zeros(48000, np.float32) + 1e-4)
    ducked = np.abs(np.concatenate([e.render(512) for _ in range(20)])[-512:]).mean()
    assert ducked < base * 0.6
    e2 = Engine(RATE)
    e2.play(tone, bus="alert")
    a0 = np.abs(np.concatenate([e2.render(512) for _ in range(10)])).mean()
    e2.speech_begin("s")
    e2.speech_feed("s", np.zeros(48000, np.float32) + 1e-4)
    a1 = np.abs(np.concatenate([e2.render(512) for _ in range(20)])[-512:]).mean()
    assert a1 > a0 * 0.95


def test_render_is_fast():
    e = Engine(RATE)
    for i in range(40):
        e.play(np.ones((2400, 2), np.float32) * 0.01, at=i * 300)
    e.speech_begin("s")
    e.speech_feed("s", np.zeros(48000, np.float32))
    for _ in range(200):
        e.render(1024)
    assert e.stats["avg_render_ms"] < 2.0
