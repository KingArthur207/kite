"""Measured properties of Kite's cue sounds: every number here comes from rendering the
cue (or a WAV file of it) and analysing it, never from playing it."""

import math

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import find_peaks, hilbert

from earpiece import sounds, spatial
from earpiece.beacon import BEHIND_MIN_PERIOD_S, Beacon
from earpiece.config import BeaconCfg
from earpiece.mixer import Engine

RATE = sounds.RATE
KINDS = ("person", "vehicle", "obstacle", "curb")
PITCHES = (525, 900, 1250)          # calm, mid, critical (SonarCfg maps urgency to 520-1250 Hz)


def spectrum(x):
    X = np.abs(np.fft.rfft(np.asarray(x, np.float64), 1 << 15)) ** 2
    return np.fft.rfftfreq(1 << 15, 1 / RATE), X


def centroid(x):
    f, X = spectrum(x)
    return float((f * X).sum() / X.sum())


def band_fraction(x, lo, hi):
    f, X = spectrum(x)
    return float(X[(f >= lo) & (f < hi)].sum() / X.sum())


def band_energy_db(x, lo, hi):
    f, X = spectrum(x)
    return 10 * math.log10(float(X[(f >= lo) & (f < hi)].sum()) + 1e-20)


def components_1_4k(x, floor_db=-20.0):
    f, X = spectrum(x)
    d = 10 * np.log10(X / X.max() + 1e-20)
    pk, _ = find_peaks(d, height=floor_db, distance=int(80 / (RATE / (1 << 15))))
    return int(np.sum((f[pk] > 1000) & (f[pk] < 4000)))


def envelope_itd_s(st):
    e = np.abs(hilbert(st.astype(np.float64), axis=0))
    return spatial.interaural_delay_s(e - e.mean(axis=0), RATE)


@pytest.fixture
def gentle():
    sounds.set_profile("gentle")
    try:
        yield
    finally:
        sounds.set_profile("standard")


# ---------------------------------------------------------------- loudness meter
def test_loudness_meter_matches_bs1770_reference():
    # BS.1770: a 0 dBFS 997 Hz sine in one channel reads -3.01 LKFS (over a 400 ms block)
    t = np.arange(int(0.4 * RATE)) / RATE
    assert sounds.loudness_lu(np.sin(2 * math.pi * 997 * t)) == pytest.approx(-3.01, abs=0.1)
    # half the energy -> -3 LU; K-weighting lifts 4 kHz over 100 Hz
    assert sounds.loudness_lu(np.sin(2 * math.pi * 997 * t[:t.size // 2])) == pytest.approx(-6.02, abs=0.1)
    assert sounds.loudness_lu(np.sin(2 * math.pi * 4000 * t)) > sounds.loudness_lu(np.sin(2 * math.pi * 100 * t)) + 3


# ---------------------------------------------------------------- sonar kind cues
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("pitch", PITCHES)
def test_kind_cue_duration_peak_loudness_and_edges(kind, pitch):
    x = sounds.proximity_cue(kind, pitch)
    assert x.dtype == np.float32 and x.ndim == 1 and np.all(np.isfinite(x))
    assert 0.06 <= x.size / RATE <= 0.13
    assert 0.2 < float(np.abs(x).max()) <= sounds.MAX_PEAK + 1e-6
    assert sounds.loudness_lu(x) == pytest.approx(sounds.SONAR_TARGET_LU, abs=1.0)
    assert x[0] == 0.0 and abs(x[-1]) < 1e-3                   # click-free edges


@pytest.mark.parametrize("pitch", PITCHES)
def test_loudness_matched_across_kinds(pitch):
    # before: 7 LU apart at equal peak (buzz -18, knock -25)
    lus = [sounds.loudness_lu(sounds.proximity_cue(k, pitch)) for k in KINDS]
    assert max(lus) - min(lus) <= 1.5


@pytest.mark.parametrize("kind", KINDS)
def test_cues_are_deterministic(kind):
    assert np.array_equal(sounds.proximity_cue(kind, 900), sounds.proximity_cue(kind, 900))


@pytest.mark.parametrize("kind,lo,hi", [("person", 700, 1100), ("vehicle", 1100, 1500),
                                        ("obstacle", 1000, 1300), ("curb", 1100, 1500)])
def test_spectral_centroid_per_kind(kind, lo, hi):
    assert lo < centroid(sounds.proximity_cue(kind, 900)) < hi


@pytest.mark.parametrize("kind", KINDS)
def test_brightness_rises_with_urgency(kind):
    cs = [centroid(sounds.proximity_cue(kind, p)) for p in PITCHES]
    assert cs == sorted(cs) and cs[-1] > 1.5 * cs[0]


@pytest.mark.parametrize("kind", KINDS)
def test_cues_are_rich_but_band_limited(kind):
    for p in PITCHES:
        x = sounds.proximity_cue(kind, p)
        assert components_1_4k(x) >= (1 if kind == "person" and p < 1000 else 2)
        assert band_fraction(x, 7500, RATE / 2) < 1e-3       # nothing up where AAC is weakest


def test_harmonic_kinds_have_several_components_between_1_and_4_khz():
    # Patterson (1982) wants >= 4 for a warning; the old pip and buzz had 1-2
    assert components_1_4k(sounds.proximity_cue("vehicle", 900)) >= 4
    assert components_1_4k(sounds.proximity_cue("obstacle", 525)) >= 3
    assert components_1_4k(sounds.proximity_cue("obstacle", 900)) >= 3


def _halves(x):
    e = np.asarray(x, np.float64) ** 2
    h = e.size // 2
    return e[:h].sum(), e[h:].sum()


def _modulation_fraction(x, lo=30.0, hi=150.0):
    env = np.abs(hilbert(np.asarray(x, np.float64)))
    env = env - np.linspace(env[0], env[-1], env.size)
    M = np.abs(np.fft.rfft(env * np.hanning(env.size), 1 << 15)) ** 2
    f = np.fft.rfftfreq(1 << 15, 1 / RATE)
    return float(M[(f >= lo) & (f < hi)].sum() / M[(f >= 1) & (f < 400)].sum())


def test_each_kind_has_its_own_envelope_or_modulation():
    person, vehicle, obstacle, curb = (sounds.proximity_cue(k, 900) for k in KINDS)
    a, b = _halves(person)
    assert a / (a + b) > 0.8                                   # knock: percussive, front-loaded
    a, b = _halves(obstacle)
    assert 0.4 < a / (a + b) < 0.65                            # pip: flat sustain
    rough = {k: _modulation_fraction(sounds.proximity_cue(k, 900)) for k in KINDS}
    assert rough["vehicle"] > 2 * max(v for k, v in rough.items() if k != "vehicle")   # only the car is rough
    h = curb.size // 2
    assert centroid(curb[:h]) > 1.3 * centroid(curb[h:])       # glide falls


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("az", [30, 60, 90])
def test_left_right_energy_and_onset_itd_point_the_right_way(kind, az):
    x = sounds.proximity_cue(kind, 900)
    right, left = spatial.render(x, az, RATE), spatial.render(x, -az, RATE)
    ild_r, ild_l = spatial.interaural_level_db(right), spatial.interaural_level_db(left)
    assert ild_r > (5.0 if az >= 60 else 3.0)
    assert ild_r == pytest.approx(-ild_l, abs=0.05)
    assert envelope_itd_s(right) < -150e-6 and envelope_itd_s(left) > 150e-6   # right ear first


def test_wav_round_trip_measures_the_same(tmp_path):
    """The listening check, numerically: write the cue as a 16-bit WAV, read it back, measure."""
    for kind in KINDS:
        st = spatial.render(sounds.proximity_cue(kind, 900), 60, RATE)
        path = tmp_path / f"{kind}.wav"
        wavfile.write(path, RATE, (np.clip(st, -1, 1) * 32767).astype(np.int16))
        rate, back = wavfile.read(path)
        back = back.astype(np.float64) / 32767
        assert rate == RATE and back.shape == st.shape
        assert spatial.interaural_level_db(back) == pytest.approx(spatial.interaural_level_db(st), abs=0.05)
        assert sounds.loudness_lu(back) == pytest.approx(sounds.loudness_lu(st), abs=0.05)


# ---------------------------------------------------------------- gentle profile
def test_profile_defaults_to_standard_and_rejects_unknown():
    assert sounds.profile() == "standard"
    with pytest.raises(ValueError):
        sounds.set_profile("loud")


@pytest.mark.parametrize("kind", KINDS)
def test_gentle_is_quieter_shorter_and_clears_the_traffic_band(kind, gentle):
    g = sounds.proximity_cue(kind, 600)
    sounds.set_profile("standard")
    s = sounds.proximity_cue(kind, 600)
    sounds.set_profile("gentle")
    assert sounds.loudness_lu(g) == pytest.approx(sounds.loudness_lu(s) + sounds.GENTLE_DB, abs=1.0)
    assert g.size <= 0.65 * s.size
    # tyre noise sits at ~0.8-1.6 kHz: the gentle cue leaves far less energy there
    # (the knock's 2.76x bar partial lands at ~1.3 kHz, so it clears less than the harmonic kinds)
    assert band_energy_db(g, 500, 1600) < band_energy_db(s, 500, 1600) - (3.0 if kind == "person" else 8.0)
    assert band_fraction(g, 0, 1000) < band_fraction(s, 0, 1000)


@pytest.mark.parametrize("kind", KINDS)
def test_gentle_never_softens_a_critical_cue(kind, gentle):
    g = sounds.proximity_cue(kind, 1250)
    sounds.set_profile("standard")
    assert np.array_equal(g, sounds.proximity_cue(kind, 1250))


def test_gentle_amount_fades_out_with_urgency(gentle):
    a = [sounds.gentle_amount(p) for p in (520, 900, 958, 1100, 1250)]
    assert a[0] == a[1] == 1.0 and a[-1] == 0.0 and a == sorted(a, reverse=True)


def test_set_profile_drops_cached_renderings():
    from earpiece import beacon as beacon_mod
    from earpiece import sonar as sonar_mod
    from earpiece.config import SonarCfg
    sonar_mod.prewarm(SonarCfg(), RATE, kinds=("person",))
    beacon_mod.prewarm(BeaconCfg(), RATE)
    assert sonar_mod._WARM and beacon_mod._WARM
    sounds.set_profile("standard")
    assert not sonar_mod._WARM and not beacon_mod._WARM


# ---------------------------------------------------------------- beacon
def test_brightness_curve_is_a_flat_topped_raised_cosine():
    b = [sounds.beacon_brightness(a) for a in range(0, 91, 5)]
    assert b[0] == 1.0 and b[1] > 0.97                         # compass jitter on axis changes nothing
    assert b == sorted(b, reverse=True)
    assert sounds.beacon_brightness(27.5) == pytest.approx(0.5)
    assert all(v == 0.0 for v in b[11:])                       # 55 deg and beyond: dull
    assert sounds.beacon_brightness(-30) == sounds.beacon_brightness(30)


def test_beacon_gets_brighter_as_you_turn_toward_it():
    cfg = BeaconCfg()
    from earpiece.beacon import region_of
    cs = [centroid(sounds.beacon_tick(region_of(a, cfg), a)) for a in range(0, 125, 5)]
    assert all(x >= y - 1.0 for x, y in zip(cs, cs[1:]))       # never brighter further off axis
    assert cs[0] > 2.5 * cs[-1]
    on, side = sounds.beacon_tick("on", 0), sounds.beacon_tick("side", 90)
    assert band_fraction(on, *sounds.FRONT_BAND_HZ) > 0.2      # Blauert's front band
    assert band_fraction(side, *sounds.FRONT_BAND_HZ) < 0.01


def test_behind_cue_lives_in_the_back_band_and_falls():
    x = sounds.beacon_behind()
    assert band_fraction(x, *sounds.FRONT_BAND_HZ) < 0.005
    assert band_fraction(x, *sounds.BACK_BAND_HZ) > 0.5
    assert centroid(sounds.beacon_on_course()) > 2.5 * centroid(x)
    nz = np.flatnonzero(np.abs(x) > 1e-4)
    gap = np.flatnonzero(np.diff(nz) > int(0.03 * RATE))
    assert gap.size == 1                                       # exactly two notes
    first, second = x[:nz[gap[0]] + 1], x[nz[gap[0] + 1]:]
    assert centroid(first) > centroid(second)                  # falling: 'turn around'
    assert 0.2 < x.size / RATE < 0.3


def test_beacon_levels_sit_under_the_sonar():
    for region, a in (("near", 20), ("near", 50), ("side", 90)):
        assert sounds.loudness_lu(sounds.beacon_tick(region, a)) == pytest.approx(sounds.BEACON_TARGET_LU, abs=0.5)
    assert sounds.BEACON_TARGET_LU <= sounds.SONAR_TARGET_LU - 2


def _second_partial_ratio(x):
    f, X = spectrum(x)
    pk, _ = find_peaks(X, distance=int(150 / (RATE / (1 << 15))))
    top = pk[np.argsort(X[pk])[::-1][:2]]
    lo, hi = sorted(f[top])
    return hi / lo


def test_beacon_timbre_is_not_a_sonar_timbre():
    # the old beacon was the person knock an octave up. The ping is harmonic (2nd partial at
    # 2x), the knock inharmonic (2.76x), and neither the ping nor the knock is rough.
    ping = sounds.beacon_tick("near", 35)
    assert _second_partial_ratio(ping) == pytest.approx(2.0, abs=0.03)
    assert abs(_second_partial_ratio(sounds.proximity_cue("person", 900)) - 2.0) > 0.3
    assert _modulation_fraction(ping) < 0.5 * _modulation_fraction(sounds.proximity_cue("vehicle", 900))


@pytest.mark.parametrize("bearing,sign", [(40, 1), (-40, -1), (100, 1), (-100, -1), (150, 1), (-150, -1)])
def test_beacon_comes_from_the_waypoint_side(bearing, sign):
    from earpiece import beacon as beacon_mod
    st = beacon_mod._beacon_render(beacon_mod.region_of(bearing, BeaconCfg()), bearing, RATE)
    assert sign * spatial.interaural_level_db(st) > 3.0


def test_behind_cue_does_not_nag():
    eng = Engine()
    b = Beacon(eng, BeaconCfg())
    b.set(180, 5, "x", now=0)               # close: the normal period would be 0.45 s
    times = []
    orig = eng.play

    def spy(st, at=None, bus="cue", gain=1.0, tag=None):
        if tag == "beacon":
            times.append(at / eng.rate)
        orig(st, at, bus, gain, tag)
    eng.play = spy
    t = 0.0
    while t < 4.0:
        b.tick(now=t)
        eng.render(1024)
        t += 1024 / eng.rate
    assert len(times) >= 3
    assert np.diff(times).min() >= BEHIND_MIN_PERIOD_S - 0.01


def test_beacon_mono_depends_only_on_region_and_abs_bearing():
    from earpiece import beacon as beacon_mod
    assert np.array_equal(beacon_mod._mono_sound("near", 30, RATE), beacon_mod._mono_sound("near", -30, RATE))
    assert not np.array_equal(beacon_mod._mono_sound("near", 20, RATE), beacon_mod._mono_sound("near", 50, RATE))
