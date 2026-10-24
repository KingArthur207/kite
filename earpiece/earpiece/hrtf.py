"""Optional measured-HRTF renderer (MIT KEMAR, SOFA), OFF by default.

The default binaural renderer is the Brown-Duda spherical head in spatial.py and
it stays that way. This module is an experiment you switch on with

    KITE_HRTF=kemar             # the beacon renders through KEMAR instead
    python -m earpiece.hrtf fetch       # one-off download (~1.2 MB) into earpiece/.cache/hrtf
    python -m earpiece.hrtf compare     # offline numbers: KEMAR vs Brown-Duda

Why it is only an option: a dummy-head HRTF adds pinna spectral cues (the
thing the sphere lacks, and the main cue for front/back), but non-individual
HRTFs still give frequent front/back reversals without head tracking
(Wenzel et al., JASA 94(1) 1993; Wightman & Kistler, JASA 105(5) 1999 showed
head movement is what resolves them), and they colour every cue. Kite already
says "behind" with a distinct sound, so the measured set has to earn its keep.

Data: Gardner & Martin (1995), "HRTF measurements of a KEMAR", JASA 97(6):3907,
SOFA conversion by the Acoustics Research Institute (sofacoustics.org),
"provided without any usage restrictions". 710 directions, 44.1 kHz, 512 taps;
the horizontal plane is sampled every 5 degrees, exactly Kite's cue grid.
SOFA azimuth runs counter-clockwise (+90 = LEFT); Kite's runs clockwise.

Processing: horizontal-plane HRIRs resampled to the engine rate, then
diffuse-field equalised (divided by the 1/3-octave-smoothed RMS spectrum over
all 710 directions, limited to +-12 dB) so the dummy head's ear-canal
resonance is not printed onto every cue. Energy is normalised exactly like
spatial.render: a cue is equally loud from every direction.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import spatial

URL = "https://sofacoustics.org/data/database/mit/mit_kemar_normal_pinna.sofa"
SHA256 = "e7035994f5fd754058424c061380ee92b1d5ed58fccef2887a4266916616acdf"
CACHE = Path(__file__).resolve().parents[1] / ".cache" / "hrtf" / "mit_kemar_normal_pinna.sofa"
_ON = ("1", "kemar", "mit", "on", "true", "yes")
_SETS: dict[int, dict[int, np.ndarray]] = {}       # rate -> {kite azimuth (5 deg) -> (n, 2) HRIR}


def enabled() -> bool:
    """True when KITE_HRTF asks for the KEMAR renderer (default: off)."""
    return os.environ.get("KITE_HRTF", "").strip().lower() in _ON


def sofa_path() -> Path:
    return Path(os.environ.get("KITE_HRTF_SOFA") or CACHE)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(path: Path | None = None, url: str = URL, sha256: str = SHA256, timeout_s: float = 60.0) -> Path:
    """Download the SOFA file into the git-ignored cache and verify its SHA-256.
    A file already there with the right checksum is reused; a bad one is replaced."""
    path = Path(path or sofa_path())
    if path.exists() and _sha256(path) == sha256:
        return path
    import httpx
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with httpx.stream("GET", url, timeout=timeout_s, follow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    got = _sha256(tmp)
    if got != sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"HRTF download checksum mismatch: got {got}, expected {sha256}")
    tmp.replace(path)
    return path


def available(path: Path | None = None) -> bool:
    """The SOFA file is on disk and sofar is importable (no download, no parsing)."""
    try:
        import sofar  # noqa: F401
    except ImportError:
        return False
    return Path(path or sofa_path()).exists()


def _smooth_third_octave(f: np.ndarray, mag: np.ndarray) -> np.ndarray:
    out = np.empty_like(mag)
    for i, fc in enumerate(f):
        lo, hi = fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6)
        m = (f >= lo) & (f <= hi)
        out[i] = np.sqrt(np.mean(mag[m] ** 2)) if m.any() else mag[i]
    return out


def _diffuse_field_eq(irs: np.ndarray, rate: int, taps: int = 255) -> np.ndarray:
    """Linear-phase FIR that divides out the RMS spectrum over every direction and ear."""
    from scipy.signal import firwin2
    nfft = 2048
    spec = np.abs(np.fft.rfft(irs.reshape(-1, irs.shape[-1]), nfft, axis=-1))
    f = np.fft.rfftfreq(nfft, 1 / rate)
    df = _smooth_third_octave(f, np.sqrt(np.mean(spec ** 2, axis=0)))
    ref = np.sqrt(np.mean(df[(f > 300) & (f < 3000)] ** 2))       # keep mid-band gain at ~0 dB
    inv = np.clip(ref / np.maximum(df, 1e-9), 10 ** (-12 / 20), 10 ** (12 / 20))
    inv[f < 200] = inv[np.searchsorted(f, 200)]                     # KEMAR data is unreliable down there
    return firwin2(taps, f / (rate / 2), inv)


def _load(path: Path, rate: int) -> dict[int, np.ndarray]:
    import sofar
    from scipy.signal import fftconvolve, resample_poly
    sofa = sofar.read_sofa(str(path), verbose=False)
    if sofa.GLOBAL_SOFAConventions != "SimpleFreeFieldHRIR":
        raise ValueError(f"expected SimpleFreeFieldHRIR, got {sofa.GLOBAL_SOFAConventions}")
    if str(sofa.SourcePosition_Type).lower() != "spherical":
        raise ValueError("expected spherical source positions")
    irs = np.asarray(sofa.Data_IR, dtype=np.float64)              # (M, R=2, N); R0 = left (+y)
    pos = np.asarray(sofa.SourcePosition, dtype=np.float64)
    fs = int(round(float(np.asarray(sofa.Data_SamplingRate).ravel()[0])))
    if fs != rate:
        g = math.gcd(rate, fs)
        irs = resample_poly(irs, rate // g, fs // g, axis=-1)
    eq = _diffuse_field_eq(irs, rate)
    delay = (eq.size - 1) // 2
    horiz = np.flatnonzero(np.abs(pos[:, 1]) < 1e-6)
    out: dict[int, np.ndarray] = {}
    for m in horiz:
        kite_az = int(round(spatial.wrap_deg(-pos[m, 0])))         # SOFA ccw -> Kite cw
        if kite_az % 5:
            continue
        pair = np.stack([fftconvolve(irs[m, ch], eq)[delay:delay + irs.shape[-1]] for ch in (0, 1)], axis=1)
        out[180 if kite_az == -180 else kite_az] = pair
    missing = [a for a in range(-175, 181, 5) if a not in out]
    if missing:
        raise ValueError(f"HRTF set lacks horizontal-plane azimuths {missing[:5]}...")
    return out


def ensure(rate: int, download: bool = True, log=print) -> bool:
    """Load (and if allowed, first download) the HRIRs for `rate`. False on any failure, in
    which case callers keep the Brown-Duda renderer. Never call this from the audio tick."""
    if rate in _SETS:
        return True
    try:
        path = sofa_path()
        if not path.exists():
            if not download:
                return False
            fetch(path)
        _SETS[rate] = _load(path, rate)
        return True
    except Exception as e:                                          # offline, no sofar, bad file
        log(f"[hrtf] KEMAR unavailable ({e}); using the Brown-Duda renderer")
        return False


def ready(rate: int) -> bool:
    return rate in _SETS


def hrir(azimuth_deg: float, rate: int) -> np.ndarray:
    """The (n, 2) left/right HRIR nearest to a Kite bearing (5-degree grid)."""
    if rate not in _SETS and not ensure(rate, download=False):
        raise RuntimeError("HRTF set not loaded: run `python -m earpiece.hrtf fetch` first")
    a = int(round(spatial.wrap_deg(float(azimuth_deg)) / 5.0) * 5)
    return _SETS[rate][180 if a == -180 else a]


def render(mono: np.ndarray, azimuth_deg: float, rate: int, gain: float = 1.0) -> np.ndarray:
    """Mono -> stereo through the KEMAR HRIR at azimuth_deg (n + taps - 1, 2) float32, with
    the same energy rule as spatial.render (total energy = 2 x the mono energy)."""
    from scipy.signal import oaconvolve
    h = hrir(azimuth_deg, rate)
    x = np.asarray(mono, dtype=np.float64)
    st = np.stack([oaconvolve(x, h[:, 0]), oaconvolve(x, h[:, 1])], axis=1)
    e_in = float(np.sum(x ** 2))
    e_out = float(np.sum(st ** 2))
    if e_in > 0 and e_out > 0:
        st *= math.sqrt(2.0 * e_in / e_out)
    return (st * gain).astype(np.float32)


# ---------------------------------------------------------------- offline comparison
def _band_db(st: np.ndarray, rate: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(st.astype(np.float64), 1 << 15, axis=0)) ** 2
    f = np.fft.rfftfreq(1 << 15, 1 / rate)
    return 10 * math.log10(float(spec[(f >= lo) & (f < hi)].sum()) + 1e-20)


def front_back_contrast_db(st: np.ndarray, rate: int) -> float:
    """Blauert directional-band balance: level in the 2.9-5.8 kHz 'front' band minus the
    0.72-1.8 kHz 'back' band, both ears summed. Higher = more 'in front' by spectrum."""
    return _band_db(st, rate, 2900, 5800) - _band_db(st, rate, 720, 1800)


def compare(rate: int = 48000, seed: int = 0, cue: np.ndarray | None = None) -> list[dict]:
    """KEMAR vs Brown-Duda on a 100 ms noise burst (or `cue`): ITD, ILD and the Blauert
    front/back band balance per azimuth. Needs the SOFA file (no download here)."""
    if not ensure(rate, download=False, log=lambda *_: None):
        raise RuntimeError("HRTF set not available: run `python -m earpiece.hrtf fetch` first")
    x = cue if cue is not None else np.random.default_rng(seed).standard_normal(int(0.1 * rate)) * 0.2
    x = np.asarray(x, dtype=np.float32)
    rows = []
    for az in (0, 30, 60, 90, 120, 150, 180, -30, -90, -150):
        bd = spatial._render_py(x, az, rate)
        km = render(x, az, rate)
        rows.append({
            "az": az,
            "bd_itd_us": round(1e6 * spatial.interaural_delay_s(bd, rate)),
            "kemar_itd_us": round(1e6 * spatial.interaural_delay_s(km, rate)),
            "bd_ild_db": round(spatial.interaural_level_db(bd), 1),
            "kemar_ild_db": round(spatial.interaural_level_db(km), 1),
            "bd_fb_db": round(front_back_contrast_db(bd, rate), 1),
            "kemar_fb_db": round(front_back_contrast_db(km, rate), 1),
        })
    return rows


def mirror_contrast(rows: list[dict]) -> dict[str, float]:
    """Mean front-minus-back band balance over mirrored pairs (30/150, 0/180, -30/-150):
    how much more 'front-like' the spectrum of a front source is than its mirror image."""
    by = {r["az"]: r for r in rows}
    pairs = [(0, 180), (30, 150), (-30, -150)]
    out = {}
    for k in ("bd", "kemar"):
        out[k] = round(float(np.mean([by[f][f"{k}_fb_db"] - by[b][f"{k}_fb_db"] for f, b in pairs])), 1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m earpiece.hrtf", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the MIT KEMAR SOFA file into the cache (checksummed)")
    cp = sub.add_parser("compare", help="print KEMAR vs Brown-Duda ITD / ILD / front-back numbers")
    cp.add_argument("--rate", type=int, default=48000)
    cp.add_argument("--wav", type=Path, help="also write each rendering to this folder as WAV")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        print(fetch())
        return 0
    rows = compare(a.rate)
    keys = list(rows[0])
    print("  ".join(f"{k:>12}" for k in keys))
    for r in rows:
        print("  ".join(f"{r[k]:>12}" for k in keys))
    print("mean front-vs-mirror band contrast (dB):", mirror_contrast(rows))
    if a.wav:
        from scipy.io import wavfile
        a.wav.mkdir(parents=True, exist_ok=True)
        x = (np.random.default_rng(0).standard_normal(int(0.1 * a.rate)) * 0.2).astype(np.float32)
        for az in (0, 90, 150, 180):
            wavfile.write(a.wav / f"bd_{az}.wav", a.rate, spatial._render_py(x, az, a.rate))
            wavfile.write(a.wav / f"kemar_{az}.wav", a.rate, render(x, az, a.rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
