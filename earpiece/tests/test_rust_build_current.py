"""The Rust extension must match its sources, or the earpiece must notice and fall back.

Not in test_rust_sonar.py on purpose: that file skips itself when the Rust sonar is unavailable, which is exactly
what a stale build looks like, so it could never catch one.
"""

from __future__ import annotations

import pytest

from earpiece import rust_sonar
from earpiece import sonar as sonar_mod
from earpiece.config import SonarCfg


def test_a_stale_build_falls_back_to_the_python_sonar(monkeypatch, capsys):
    """A kite_audio built before the sources gained a method (note_perception, once) used to be picked by "auto"
    and then raise on every obstacle message: the earpiece went silent. Now "auto" picks Python and says why."""
    class OldSonar:   # every required method but one
        pass
    for m in rust_sonar.REQUIRED:
        if m != "note_perception":
            setattr(OldSonar, m, lambda *a, **k: None)
    fake = type("kite_audio", (), {"Sonar": OldSonar, "SONAR_ABI": rust_sonar.SONAR_ABI})
    monkeypatch.setattr(rust_sonar, "_ka", fake)
    monkeypatch.setattr(sonar_mod, "_warned_stale", False)
    assert rust_sonar.stale_build() == ["note_perception"]
    assert not rust_sonar.available()
    assert sonar_mod.use_rust(SonarCfg(engine="auto")) is False
    assert "missing note_perception" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="older than its sources"):
        sonar_mod.use_rust(SonarCfg(engine="rust"))


def test_the_built_extension_is_current():
    """The kite_audio this suite runs against has every method the wrapper calls (check.sh rebuilds it)."""
    if rust_sonar._ka is None or not hasattr(rust_sonar._ka, "Sonar"):
        pytest.skip("kite_audio not built")
    assert rust_sonar.stale_build() == []


def test_an_old_obstacle_signature_counts_as_stale(monkeypatch):
    """A build from before obstacles carried a priority takes 6-tuples and would reject every frame."""
    class Sonar:
        pass
    for m in rust_sonar.REQUIRED:
        setattr(Sonar, m, lambda *a, **k: None)
    monkeypatch.setattr(rust_sonar, "_ka", type("kite_audio", (), {"Sonar": Sonar}))   # no SONAR_ABI = 1
    assert rust_sonar.stale_build() == [f"SONAR_ABI {rust_sonar.SONAR_ABI}"]
    assert not rust_sonar.available()
