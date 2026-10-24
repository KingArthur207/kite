"""The Rust binaural renderer must place every cue exactly where the Python one does.

`spatial._render_py` is the numpy/scipy reference; `kite_audio.render_binaural` is
its native port. These tests call the extension directly (not through the
KITE_AUDIO_ENGINE dispatch) so parity is checked whichever engine the suite runs
with, then check the dispatch itself, the batch API, the opt-in windowed-sinc
delay, and that the native path is actually faster.
"""

import math
import os
import time

import numpy as np
import pytest

from earpiece import sounds, spatial

ka = pytest.importorskip("kite_audio")
if not hasattr(ka, "render_binaural"):
    pytest.skip("kite_audio built without the binaural renderer", allow_module_level=True)

A_C = spatial.HEAD_RADIUS_M / spatial.SPEED_OF_SOUND


def native(mono, az, rate, gain=1.0, extra_ild_db=6.0, sinc_taps=0):
    raw = ka.render_binaural(np.ascontiguousarray(mono, dtype="<f4").tobytes(), float(az), rate,
                             gain, extra_ild_db, sinc_taps)
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2)


def delay(x, d, out_len, taps):
    raw = ka.fractional_delay(np.ascontiguousarray(x, dtype="<f4").tobytes(), float(d), out_len, taps)
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def signals(rate):
    t = np.arange(int(0.07 * rate)) / rate
    rng = np.random.default_rng(rate)
    return {
        "tone": (0.4 * np.sin(2 * np.pi * 700 * t) * np.hanning(t.size)).astype(np.float32),
        "noise": (rng.standard_normal(int(0.05 * rate)) * 0.3).astype(np.float32),
        "knock": sounds.proximity_cue("person", 800),
    }


def rel_err(a, b):
    return float(np.abs(a.astype(np.float64) - b).max() / max(np.abs(a).max(), 1e-30))


# ---------------------------------------------------------------- parity
@pytest.mark.parametrize("rate", [48000, 44100, 16000])
@pytest.mark.parametrize("gain,ild", [(1.0, 6.0), (0.35, 0.0), (1.7, 20.0)])
def test_parity_with_python_reference(rate, gain, ild):
    worst, exact, n = 0.0, 0, 0
    for name, x in signals(rate).items():
        for az in range(-180, 181, 7):
            ref = spatial._render_py(x, az, rate, gain, ild)
            got = native(x, az, rate, gain, ild)
            assert got.shape == ref.shape, (name, az)
            e = rel_err(ref, got)
            assert e < 1e-5, f"{name} az={az}: rel err {e:.2e}"
            worst = max(worst, e)
            exact += int(np.array_equal(ref, got))
            n += 1
    print(f"\n[parity] rate {rate} gain {gain} ild {ild}: max rel err {worst:.1e}, "
          f"{exact}/{n} bit-identical")


@pytest.mark.parametrize("taps", [2, 8, 16, 32])
def test_sinc_mode_parity_with_python_reference(taps):
    for rate in (48000, 16000):
        for name, x in signals(rate).items():
            for az in range(-180, 181, 23):
                ref = spatial._render_py(x, az, rate, sinc_taps=taps)
                got = native(x, az, rate, sinc_taps=taps)
                assert got.shape == ref.shape == (spatial.out_frames(x.size, rate, taps), 2)
                assert rel_err(ref, got) < 1e-5, (name, az, taps)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 31])
def test_short_and_empty_buffers_match(n):
    x = (np.random.default_rng(n).standard_normal(n) * 0.3).astype(np.float32)
    for taps in (0, 16):
        for az in (-135, -90, 0, 45, 170):
            ref = spatial._render_py(x, az, 48000, sinc_taps=taps)
            got = native(x, az, 48000, sinc_taps=taps)
            assert got.shape == ref.shape == (spatial.out_frames(n, 48000, taps), 2)
            assert np.all(np.isfinite(got))
            if n:
                assert rel_err(ref, got) < 1e-5
            else:
                assert not got.any() and not ref.any()


def test_silence_stays_silent_and_finite():
    for taps in (0, 16):
        for az in (0, 100, -170):
            got = native(np.zeros(500, np.float32), az, 48000, sinc_taps=taps)
            assert np.all(np.isfinite(got)) and not got.any()


def test_loud_and_tiny_inputs_are_finite():
    rng = np.random.default_rng(3)
    for scale in (1e-30, 1e-6, 1.0, 1e6):
        x = (rng.standard_normal(1000) * scale).astype(np.float32)
        got = native(x, 120, 48000)
        assert np.all(np.isfinite(got))
        assert rel_err(spatial._render_py(x, 120, 48000), got) < 1e-5


@pytest.mark.parametrize("az,same_as", [(720, 0), (-540, 180), (450, 90), (-450, -90), (360.5, 0.5),
                                        (-180, 180)])
def test_azimuth_wraps_like_python(az, same_as):
    x = signals(48000)["knock"]
    assert np.array_equal(native(x, az, 48000), native(x, same_as, 48000))
    assert rel_err(spatial._render_py(x, az, 48000), native(x, az, 48000)) < 1e-5


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_azimuth_is_rejected_by_both(bad):
    x = np.ones(10, np.float32)
    with pytest.raises(ValueError):
        spatial._render_py(x, bad, 48000)
    with pytest.raises(ValueError):
        native(x, bad, 48000)


def test_bad_arguments_are_rejected():
    x = np.ones(10, np.float32)
    for taps in (1, 3, 66, 100):
        with pytest.raises(ValueError):
            native(x, 0, 48000, sinc_taps=taps)
        with pytest.raises(ValueError):
            spatial._render_py(x, 0, 48000, sinc_taps=taps)
    with pytest.raises(ValueError):
        ka.render_binaural(b"\x00" * 7, 0.0, 48000)
    with pytest.raises(ValueError):
        ka.render_binaural(x.tobytes(), 0.0, 0)


# ---------------------------------------------------------------- batch + dispatch
def test_batch_equals_single_renders_bit_for_bit():
    x = sounds.proximity_cue("vehicle", 900)
    azs = [float(a) for a in spatial.BEARINGS_5DEG] + [720.0, -540.0, 12.3]
    for taps in (0, 16):
        raws = ka.render_binaural_batch(x.tobytes(), azs, 48000, 0.8, 6.0, taps)
        assert len(raws) == len(azs)
        for raw, az in zip(raws, azs):
            assert raw == ka.render_binaural(x.tobytes(), az, 48000, 0.8, 6.0, taps), az
    assert ka.render_binaural_batch(x.tobytes(), [], 48000) == []


def test_render_many_matches_render():
    x = sounds.beacon_behind()
    many = spatial.render_many(x, spatial.BEARINGS_5DEG, 48000)
    assert len(many) == len(spatial.BEARINGS_5DEG)
    for st, az in zip(many, spatial.BEARINGS_5DEG):
        assert np.array_equal(st, spatial.render(x, az, 48000))
        assert st.dtype == np.float32 and st.flags.writeable and st.flags.c_contiguous


def test_dispatch_follows_the_engine_setting():
    python_only = os.environ.get("KITE_AUDIO_ENGINE", "auto").strip().lower() == "python"
    assert spatial.NATIVE is (not python_only)
    x = signals(48000)["knock"]
    st = spatial.render(x, 30, 48000)
    assert st.dtype == np.float32 and st.shape == (spatial.out_frames(x.size, 48000), 2)
    assert st.flags.writeable and st.flags.c_contiguous    # callers may scale it in place
    assert rel_err(spatial._render_py(x, 30, 48000), st) < 1e-5


def test_float64_input_is_placed_as_float32():
    x = signals(48000)["tone"].astype(np.float64)
    assert rel_err(spatial._render_py(x, -60, 48000), native(x, -60, 48000)) < 1e-6


# ---------------------------------------------------------------- windowed-sinc delay
def measured_delay_s(a, b, rate, f_max_frac=0.35):
    """b-minus-a delay in seconds: integer lag from the cross-correlation peak, then the
    residual from a magnitude-weighted fit of the cross-spectrum phase slope."""
    n = 1 << int(math.ceil(math.log2(2 * a.size)))
    xs = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    lag = int(np.argmax(np.fft.irfft(xs, n)))
    lag = lag - n if lag > n // 2 else lag
    f = np.fft.rfftfreq(n, 1 / rate)
    band = (f > 100) & (f < f_max_frac * rate)
    ph = np.angle(xs[band] * np.exp(2j * np.pi * f[band] * lag / rate))
    w = np.abs(xs[band])
    resid = -np.sum(w * f[band] * ph) / (2 * np.pi * np.sum(w * f[band] ** 2))
    return lag / rate + resid


def woodworth_itd_s(az_deg):
    th = math.radians(abs(spatial.wrap_deg(az_deg)))
    th = math.pi - th if th > math.pi / 2 else th
    return A_C * (th + math.sin(th))


@pytest.mark.parametrize("rate", [48000, 44100])
def test_sinc_delay_reproduces_woodworth_itd_within_10us(rate):
    x = (np.random.default_rng(7).standard_normal(rate // 2) * 0.3).astype(np.float32)
    out_len = spatial.out_frames(x.size, rate, 16)
    errs = {0: [], 16: []}
    for az in (5, 10, 20, 30, 45, 60, 75, 90, 110, 135, 160):
        dl = spatial.ear_delay_s(spatial._ear_angle(az, -1)) * rate
        dr = spatial.ear_delay_s(spatial._ear_angle(az, +1)) * rate
        assert abs((dl - dr) / rate - woodworth_itd_s(az)) < 1e-9     # the model itself
        for taps in errs:
            itd = measured_delay_s(delay(x, dr, out_len, taps), delay(x, dl, out_len, taps), rate)
            errs[taps].append(abs(itd - woodworth_itd_s(az)))
    print(f"\n[itd] rate {rate}: max |error| sinc16 {max(errs[16]) * 1e9:.1f} ns, "
          f"linear {max(errs[0]) * 1e9:.0f} ns")
    assert max(errs[16]) < 10e-6


def test_sinc_delay_keeps_the_top_octave_where_linear_dulls():
    rate = 48000
    x = (np.random.default_rng(11).standard_normal(rate // 4) * 0.3).astype(np.float32)
    n = 1 << 14
    f = np.fft.rfftfreq(n, 1 / rate)
    hi = (f > 0.35 * rate) & (f < 0.4 * rate)
    ref = np.abs(np.fft.rfft(x.astype(np.float64), n))[hi]

    def hf_db(taps, frac):
        y = delay(x, 3 + frac, x.size + 64, taps)
        return 20 * math.log10(np.mean(np.abs(np.fft.rfft(y, n))[hi]) / np.mean(ref))
    for frac in (0.0, 0.25, 0.5, 0.75):
        assert abs(hf_db(16, frac)) < 1.0, frac
    assert hf_db(0, 0.5) < -6.0                       # linear: |cos(pi f / fs)|, -7..-10 dB here


def test_sinc_render_keeps_the_spatial_cues():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(4800) * 0.3).astype(np.float32)
    e0 = float(np.sum(x.astype(np.float64) ** 2))
    for az, sign in ((30, 1), (90, 1), (-45, -1), (-150, -1)):
        st = native(x, az, 48000, sinc_taps=16)
        assert sign * spatial.interaural_level_db(st) > 3.0
        assert -sign * spatial.interaural_delay_s(st, 48000) > 100e-6
        assert abs(float(np.sum(st.astype(np.float64) ** 2)) / e0 - 2.0) < 0.02


# ---------------------------------------------------------------- speed
def _best_us(fn, *args, rounds=7, calls=40):
    best = math.inf
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(calls):
            fn(*args)
        best = min(best, (time.perf_counter() - t0) / calls)
    return best * 1e6


def test_native_render_is_faster_than_python():
    cue = sounds.sine_pip(800, 0.07)                  # a typical 70 ms sonar pip
    assert cue.size == 3360
    for az in (30, 150):                              # front, and behind (extra low-pass)
        py_us = _best_us(spatial._render_py, cue, az, 48000)
        rs_us = _best_us(native, cue, az, 48000)
        print(f"\n[speed] 70 ms cue at {az} deg: python {py_us:.0f} us, rust {rs_us:.0f} us, "
              f"{py_us / rs_us:.1f}x")
        assert rs_us < py_us
    azs = [float(a) for a in spatial.BEARINGS_5DEG]
    py_ms = _best_us(lambda: [spatial._render_py(cue, a, 48000) for a in azs], rounds=3, calls=1) / 1e3
    rs_ms = _best_us(lambda: ka.render_binaural_batch(cue.tobytes(), azs, 48000), rounds=3, calls=1) / 1e3
    print(f"[speed] 73-bearing pre-warm: python {py_ms:.1f} ms, rust batch {rs_ms:.1f} ms, "
          f"{py_ms / rs_ms:.1f}x")
    assert rs_ms < py_ms
