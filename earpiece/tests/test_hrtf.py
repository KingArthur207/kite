"""The optional KEMAR HRTF path. Skipped unless the SOFA file is already cached
(`python -m earpiece.hrtf fetch`) and sofar is installed: tests never download."""

import numpy as np
import pytest

from earpiece import hrtf, sounds, spatial
from earpiece.config import BeaconCfg

RATE = 48000

needs_sofa = pytest.mark.skipif(not hrtf.available(), reason="KEMAR SOFA file not cached or sofar not installed")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("KITE_HRTF", raising=False)
    assert not hrtf.enabled()
    monkeypatch.setenv("KITE_HRTF", "kemar")
    assert hrtf.enabled()


def test_missing_file_is_unavailable_and_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("KITE_HRTF_SOFA", str(tmp_path / "none.sofa"))
    assert not hrtf.available()
    assert hrtf.ensure(44100 + 1, download=False, log=lambda *_: None) is False


def test_bad_checksum_is_rejected_before_use(tmp_path, monkeypatch):
    bad = tmp_path / "x.sofa"
    bad.write_bytes(b"not a sofa file")
    assert hrtf._sha256(bad) != hrtf.SHA256

    class Boom(Exception):
        pass

    def no_network(*a, **k):
        raise Boom()
    import httpx
    monkeypatch.setattr(httpx, "stream", no_network)
    with pytest.raises(Boom):            # a wrong file triggers a re-download, never a load
        hrtf.fetch(bad)


@needs_sofa
def test_cached_file_matches_the_pinned_checksum():
    assert hrtf._sha256(hrtf.sofa_path()) == hrtf.SHA256


@needs_sofa
def test_kemar_itd_matches_brown_duda_and_points_the_right_way():
    rows = hrtf.compare(RATE)
    for r in rows:
        assert abs(r["kemar_itd_us"] - r["bd_itd_us"]) <= 100, r
        assert np.sign(r["kemar_ild_db"]) == np.sign(r["bd_ild_db"]) or abs(r["bd_ild_db"]) < 0.5, r
    by = {r["az"]: r for r in rows}
    assert by[90]["kemar_ild_db"] > 10 and by[-90]["kemar_ild_db"] < -10      # +90 = right in Kite's frame
    assert by[90]["kemar_itd_us"] < -500                                       # right ear first


@needs_sofa
def test_kemar_render_keeps_loudness_constant_across_directions():
    x = (np.random.default_rng(1).standard_normal(4800) * 0.2).astype(np.float32)
    e0 = float(np.sum(x.astype(float) ** 2))
    for az in range(-180, 181, 30):
        st = hrtf.render(x, az, RATE)
        assert st.dtype == np.float32 and st.shape[1] == 2
        assert float(np.sum(st.astype(float) ** 2)) / e0 == pytest.approx(2.0, rel=1e-3)


@needs_sofa
def test_kemar_front_back_contrast_is_at_least_brown_duda():
    """The offline comparison behind keeping KEMAR optional: on a beacon cue its Blauert-band
    front/back contrast is no worse than the sphere's (it is a little better)."""
    rows = hrtf.compare(RATE, cue=sounds.beacon_tick("near", 35))
    m = hrtf.mirror_contrast(rows)
    assert m["kemar"] >= m["bd"] - 0.5


@needs_sofa
def test_beacon_uses_kemar_only_when_asked(monkeypatch):
    from earpiece import beacon as beacon_mod
    monkeypatch.setenv("KITE_HRTF", "kemar")
    try:
        beacon_mod._WARM.clear()
        beacon_mod._beacon_render.cache_clear()
        n = beacon_mod.prewarm(BeaconCfg(), RATE, log=lambda *_: None)
        assert 73 <= n < 2 * 73
        st = beacon_mod._beacon_sound("near", 40, RATE)
        assert np.array_equal(st, hrtf.render(beacon_mod._mono_sound("near", 40, RATE), 40, RATE))
        assert spatial.interaural_level_db(st) > 3.0
        monkeypatch.delenv("KITE_HRTF")
        beacon_mod._WARM.clear()
        beacon_mod._beacon_render.cache_clear()
        bd = beacon_mod._beacon_sound("near", 40, RATE)
        assert np.array_equal(bd, spatial.render(beacon_mod._mono_sound("near", 40, RATE), 40.0, RATE))
    finally:
        beacon_mod._WARM.clear()
        beacon_mod._beacon_render.cache_clear()
