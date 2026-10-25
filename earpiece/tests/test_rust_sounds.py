"""The Rust cue synthesiser must produce exactly the samples sounds.py produces.

`sounds.py` (numpy/scipy, run inside `sounds.reference()`) is the reference; `kite_audio`
synthesises the same cues in native/src/sounds.rs. Every test compares float32 arrays for
exact equality: same length, same bits. Covered: every object kind at every quantised sonar
pitch (and off-grid pitches), both profiles, every beacon region at every 5-degree bearing
(and off-grid bearings), and every status earcon and building block.

These call kite_audio directly, so parity is checked whichever engine KITE_AUDIO_ENGINE
selects for the rest of the suite.
"""

from __future__ import annotations

import random
import time

import numpy as np
import pytest

from earpiece import sounds, spatial

ka = pytest.importorskip("kite_audio")
pytestmark = pytest.mark.skipif(not hasattr(ka, "synth"), reason="kite_audio built without the cue synthesiser")

KINDS = sounds.PERSON_KINDS + sounds.VEHICLE_KINDS + sounds.DROP_KINDS + ("pole", "obstacle", "bench", "")
PITCHES = tuple(range(525, 1251, 25))            # every pitch_q the sonar can ask for (SonarCfg defaults)
REGIONS = ("on", "near", "side", "behind")
STATS = {"cues": 0, "samples": 0}


def native(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f4")


def same(ref: np.ndarray, got: np.ndarray, what: str) -> None:
    assert ref.dtype == np.float32, what
    assert ref.shape == got.shape, (what, ref.shape, got.shape)
    if not np.array_equal(ref, got):
        bad = np.flatnonzero(ref != got)
        raise AssertionError(f"{what}: {bad.size} samples differ, first at {bad[0]}, "
                             f"max |diff| {np.max(np.abs(ref.astype(np.float64) - got)):.3g}")
    STATS["cues"] += 1
    STATS["samples"] += ref.size


@pytest.fixture(params=sounds.PROFILES)
def profile(request):
    old = sounds.profile()
    sounds.set_profile(request.param)
    try:
        yield request.param
    finally:
        sounds.set_profile(old)


def reference(fn, *args, **kw) -> np.ndarray:
    with sounds.reference():
        return fn(*args, **kw)


# ---------------------------------------------------------------- sonar kinds
@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_at_every_quantised_pitch(kind, profile):
    for p in PITCHES:
        same(reference(sounds.proximity_cue, kind, float(p)),
             native(ka.synth_proximity(kind, float(p), profile)), f"{kind} {p} Hz {profile}")


@pytest.mark.parametrize("name,fn", [("person", sounds.person_cue), ("vehicle", sounds.vehicle_cue),
                                     ("obstacle", sounds.obstacle_cue), ("drop", sounds.drop_cue)])
def test_timbres_by_name_and_off_grid_pitches(name, fn, profile):
    rng = random.Random(f"{name}-{profile}")
    pitches = [rng.uniform(60.0, 3000.0) for _ in range(12)] + [520.0, 1250.0, 1637.5, 99.99]
    for p in pitches:
        same(reference(fn, p), native(ka.synth(name, [p], profile)), f"{name} {p:.3f} Hz {profile}")


def test_gentle_shaping_fades_out_at_the_critical_pitch():
    # the same samples in both profiles at full urgency, in Rust as in Python
    for kind in ("person", "vehicle", "pole", "curb"):
        std = native(ka.synth_proximity(kind, 1250.0, "standard"))
        gen = native(ka.synth_proximity(kind, 1250.0, "gentle"))
        assert np.array_equal(std, gen), kind
        assert native(ka.synth_proximity(kind, 600.0, "gentle")).size < native(
            ka.synth_proximity(kind, 600.0, "standard")).size


# ---------------------------------------------------------------- beacon
@pytest.mark.parametrize("region", REGIONS)
def test_every_beacon_region_at_every_bearing(region, profile):
    for bq in spatial.BEARINGS_5DEG:
        same(reference(sounds.beacon_tick, region, float(bq)),
             native(ka.synth_beacon(region, float(bq), profile)), f"beacon {region} {bq} {profile}")
    rng = random.Random(region)
    for b in [rng.uniform(-200, 200) for _ in range(10)] + [0.0, -0.0, 54.999, 55.0, -55.0001]:
        same(reference(sounds.beacon_tick, region, b), native(ka.synth_beacon(region, b, profile)),
             f"beacon {region} {b} {profile}")


def test_beacon_shortcuts_follow_the_dispatcher(profile):
    with sounds.reference():
        ref = [sounds.beacon_on_course(), sounds.beacon_off_course(-30.0), sounds.beacon_off_course(80.0),
               sounds.beacon_behind()]
    got = [native(ka.synth_beacon("on", 0.0, profile)), native(ka.synth_beacon("near", -30.0, profile)),
           native(ka.synth_beacon("side", 80.0, profile)), native(ka.synth("beacon_behind", [], profile))]
    for i, (r, g) in enumerate(zip(ref, got)):
        same(r, g, f"beacon shortcut {i} {profile}")


# ---------------------------------------------------------------- status cues
STATUS = ("arrived", "ready", "listening", "heard", "error", "sensor_lost", "sensor_back", "critical_alarm",
          "crossing_go", "crossing_wait")


@pytest.mark.parametrize("name", STATUS)
def test_status_earcons(name):
    same(reference(getattr(sounds, name)), native(ka.synth(name)), name)


def test_ear_check_and_nav_chime():
    for side in ("left", "right"):
        same(reference(sounds.ear_check, side), native(ka.synth(f"ear_check_{side}")), f"ear {side}")
    for text, side in (("Turn left onto Bay Street.", -1), ("In 30 meters, turn right", 1),
                       ("Head north on Lyon Street.", 0), ("", 0)):
        assert sounds.turn_side(text) == side
        same(reference(sounds.nav_chime, text), native(ka.synth("nav_chime", [side])), f"nav {text!r}")


def test_building_blocks_with_arguments():
    rng = random.Random(7)
    for _ in range(8):
        f, d, pk = rng.uniform(150, 3000), rng.uniform(0.02, 0.3), rng.uniform(0.1, 0.9)
        same(reference(sounds.sine_pip, f, d, pk), native(ka.synth("sine_pip", [f, d, pk])), "sine_pip")
        same(reference(sounds.knock, f, d, pk), native(ka.synth("knock", [f, d, pk])), "knock")
        same(reference(sounds.buzz, f, d, pk), native(ka.synth("buzz", [f, d, pk])), "buzz")
        f1 = rng.uniform(150, 3000)
        same(reference(sounds.glide, f, f1, d, pk), native(ka.synth("glide", [f, f1, d, pk])), "glide")
        fs = [rng.uniform(300, 2000) for _ in range(rng.randint(1, 4))]
        same(reference(sounds.chime, *fs, step_s=d, peak=pk), native(ka.synth("chime", [d, pk, *fs])), "chime")


# ---------------------------------------------------------------- dispatch and errors
def test_sounds_dispatches_to_rust_when_built():
    if not sounds.NATIVE:
        pytest.skip("KITE_AUDIO_ENGINE=python: sounds.py runs its own code")
    for kind in ("person", "vehicle", "pole", "curb"):
        same(reference(sounds.proximity_cue, kind, 900.0), sounds.proximity_cue(kind, 900.0), f"dispatch {kind}")
    out = sounds.proximity_cue("person", 900.0)
    out[:4] = 0.0                                  # callers own (and may modify) what they get back
    with pytest.raises(KeyError):
        sounds.beacon_tick("nowhere", 0.0)


def test_bad_arguments_raise_instead_of_crashing():
    for call in (lambda: ka.synth_proximity("person", float("nan")), lambda: ka.synth_proximity("person", -5.0),
                 lambda: ka.synth_proximity("person", 900.0, "loud"), lambda: ka.synth_beacon("nowhere", 0.0),
                 lambda: ka.synth("nope"), lambda: ka.synth("sine_pip", [440.0]),
                 lambda: ka.synth("glide", [float("inf"), 1.0, 0.1, 0.5]), lambda: ka.synth("chime", [0.1])):
        with pytest.raises(ValueError):
            call()


def test_rust_synthesis_is_faster():
    def best(fn, n=15):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return min(ts)
    py = best(lambda: reference(sounds.proximity_cue, "vehicle", 900.0))
    rs = best(lambda: ka.synth_proximity("vehicle", 900.0))
    print(f"\nvehicle cue: python {py * 1e3:.2f} ms, rust {rs * 1e3:.3f} ms ({py / rs:.0f}x); "
          f"parity so far: {STATS['cues']} cues, {STATS['samples']} samples, max deviation 0")
    assert rs < py
