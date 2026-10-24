"""kite.fly brain, encoder, readout, parameter space and hand calibration: the connectome file and its groups, LIF
determinism by seed, the steering sign the circuit is known for (spot right -> DNa02_R fires, DNa02_L falls silent),
tick speed, and the parameter files (the previously trained values and FLY-YAW-HAND)."""

import json
import math

import numpy as np
import pytest

from kite.fly.brain import LIFParams, PursuitBrain, load_wiring
from kite.fly.calibrate import calibrate
from kite.fly.decoder import NORM_FLOOR_HZ, YAW_SCALE, YawReadout
from kite.fly.encoder import CH_AROUSAL, CH_LC9, CH_LC11, LC_CAP_HZ, RATE_CAP_HZ, TargetEncoder, bin_centres_deg
from kite.fly.interfaces import (CHANNELS, INPUT_GROUPS, N_AZIMUTH_BINS, N_CHANNELS, NO_TARGET, OUTPUT_GROUPS,
                                 PREV_CALIBRATION, PREV_PARAMS, REFERENCE_DIR, TargetFeatures)
from kite.fly.params import HAND_FILE, NAMES, SPACE, load_norm, load_params, params_record

HAND = SPACE.defaults()  # the hand-set encoder (and readout shape) that calibration uses
DNA02_L, DNA02_R = OUTPUT_GROUPS.index("DNa02_L"), OUTPUT_GROUPS.index("DNa02_R")


def spot(theta_deg, s=1.0, dtheta=0.0):
    return TargetFeatures(True, math.radians(theta_deg), s, dtheta, 0.0)


def mean_rates(brain, rates, settle=4, n=20):
    for _ in range(settle):
        brain.tick(rates)
    return np.mean([brain.tick(rates) for _ in range(n)], axis=0)


@pytest.fixture(scope="module")
def brain():
    return PursuitBrain(seed=0)


def test_brain_file_groups_and_input_layout():
    w = load_wiring()
    assert load_wiring() is w  # loaded once per process
    assert w.n == 1446 and w.n_connections == 40586 and w.name == "pursuit-core1"
    sizes = {"LC10a_L": 135, "LC10a_R": 140, "LC9_L": 104, "LC9_R": 115, "LC11_L": 68, "LC11_R": 75,
             "AROUSAL_L": 43, "AROUSAL_R": 43}
    assert {g: w.group_sizes[g] for g in INPUT_GROUPS} == sizes
    assert all(w.group_sizes[g] == 1 for g in OUTPUT_GROUPS)  # one steering cell per side
    # input neurons come first, in channel order; LC10a split into the file's 8 azimuth bins per side
    assert w.n_in == sum(sizes.values()) == 723
    per_chan = np.bincount(w.in_chan, minlength=N_CHANNELS)
    bins = w.meta["azimuth_bins"]
    assert per_chan[:N_AZIMUTH_BINS].tolist() == [len(b) for b in bins["LC10a_L"]]
    assert per_chan[N_AZIMUTH_BINS:2 * N_AZIMUTH_BINS].tolist() == [len(b) for b in bins["LC10a_R"]]
    assert per_chan[CHANNELS.index("LC9_R")] == 115 and per_chan[CHANNELS.index("AROUSAL_L")] == 43
    assert np.array_equal(np.sort(w.perm[:w.n_in]), np.sort(np.concatenate([w.groups[g] for g in INPUT_GROUPS])))
    k = CHANNELS.index("LC10a_R_b3")
    assert np.array_equal(np.sort(w.perm[:w.n_in][w.in_chan == k]), np.sort(bins["LC10a_R"][3]))
    # the graph is presynaptic-major: AOTU019_R (file index 0) inhibits DNa02_L
    row = slice(w.indptr[w.inv[0]], w.indptr[w.inv[0] + 1])
    to_dna02_l = w.weights_mv[row][w.indices[row] == w.inv[w.groups["DNa02_L"][0]]]
    assert to_dna02_l.size == 1 and to_dna02_l[0] == pytest.approx(-297 * 0.275)


def test_lif_parameters_shiu_2024():
    p = load_wiring().p
    assert (p.v_rest, p.v_reset, p.v_th, p.tau_m, p.tau_syn, p.t_ref, p.delay, p.w_syn, p.poisson_factor, p.dt) == (
        -52.0, -52.0, -45.0, 20.0, 5.0, 2.2, 1.8, 0.275, 250.0, 0.5)
    assert p.delay_steps == 4 and p.ref_steps == 4
    with pytest.raises(ValueError):
        LIFParams.from_meta({"noise_mv": 1.0})


def test_no_input_stays_at_rest(brain):
    b = brain.copy(seed=3)
    rates = np.mean([b.tick(np.zeros(N_CHANNELS)) for _ in range(5)], axis=0)
    assert not rates.any() and np.all(b.v == np.float32(-52.0)) and b.last_counts.sum() == 0
    with pytest.raises(ValueError):
        b.tick(np.zeros(5))


def test_deterministic_for_a_seed(brain):
    ch = TargetEncoder(HAND).channels(spot(8.0))
    a, b, c = brain.copy(seed=7), brain.copy(seed=7), brain.copy(seed=8)
    ra = [a.tick(ch) for _ in range(6)]
    rb = [b.tick(ch) for _ in range(6)]
    for _ in range(6):
        c.tick(ch)
    assert np.array_equal(ra, rb) and np.array_equal(a.last_counts, b.last_counts)
    assert not np.array_equal(a.last_counts, c.last_counts)  # another fly: same wiring, its own noise
    a.reset()  # same seed again: the same fly
    assert np.array_equal([a.tick(ch) for _ in range(6)], ra)
    assert a.w is brain.w  # copies share the wiring


def test_spot_right_drives_dna02_r_left_drives_dna02_l(brain):
    """The known steering sign (ipsiversive DNa02): +30 deg -> DNa02_R fires, DNa02_L is silent; -30 the mirror."""
    enc = TargetEncoder(HAND)
    right = mean_rates(brain.copy(seed=0), enc.channels(spot(+30.0)))
    left = mean_rates(brain.copy(seed=0), enc.channels(spot(-30.0)))
    assert right[DNA02_R] > 250 and right[DNA02_L] < 20
    assert left[DNA02_L] > 250 and left[DNA02_L] > 2 * left[DNA02_R]


def test_tick_speed(brain):
    b = brain.copy(seed=1)
    ch = TargetEncoder(HAND).channels(spot(10.0))
    for _ in range(3):
        b.tick(ch)
    b.reset()
    for _ in range(20):
        b.tick(ch)
    assert 1000.0 * b.brain_s / b.ticks < 10.0  # about 1.7 ms on an M5 laptop


def test_encoder_channels():
    enc = TargetEncoder(HAND)
    assert not enc.channels(NO_TARGET).any() and enc.channels().shape == (N_CHANNELS,)  # P1 tonic 0, nothing else
    c = enc.channels(spot(20.0))
    right, left = c[N_AZIMUTH_BINS:2 * N_AZIMUTH_BINS], c[:N_AZIMUTH_BINS]
    assert right.max() > 100 and left.max() < 0.15 * right.max() and np.argmax(left) == 0  # overlap: L0 sees a bit
    centres = bin_centres_deg(HAND["enc_overlap_deg"])
    assert np.argmax(right) == np.argmin(np.abs(centres[N_AZIMUTH_BINS:] - 20.0))
    assert centres[N_AZIMUTH_BINS] < 0 < centres[N_AZIMUTH_BINS + 1] and centres[-1] < 40  # bin 0 frontal
    assert np.allclose(centres[:N_AZIMUTH_BINS], -centres[N_AZIMUTH_BINS:])  # the left side mirrors the right
    assert c[CH_LC9 + 1] > c[CH_LC9] and c[CH_LC11 + 1] > 0 and not c[CH_AROUSAL:].any()
    hot = TargetEncoder(HAND | {"enc_lc9_hz": 10.0, "enc_r_max_hz": 400.0, "enc_bin_gain_R2": 2.0})
    h = hot.channels(spot(5.0, s=5.0, dtheta=3.0))
    assert h.max() <= RATE_CAP_HZ and h[CH_LC9:CH_LC9 + 4].max() <= LC_CAP_HZ
    mirror = enc.channels(spot(-20.0))
    assert np.allclose(mirror[:N_AZIMUTH_BINS], right) and np.allclose(mirror[N_AZIMUTH_BINS:2 * N_AZIMUTH_BINS], left)


def test_readout_sign_filter_and_deadzone():
    p = HAND | {"dec_w_yaw_DNa02": 1.0, "dec_g_yaw": 1.0, "dec_deadzone": 2.0, "dec_tau_yaw_ms": 500.0}
    ro = YawReadout(p, np.full(10, 100.0))
    assert ro.tau_s == 0.15  # tau is capped at 150 ms
    r = np.zeros(10)
    r[DNA02_R] = 100.0
    assert ro.update(r, 0.05) == pytest.approx(YAW_SCALE * math.tanh(1.0) - 2.0)  # the first sample initializes
    ro.reset()
    r[DNA02_R], r[DNA02_L] = 0.0, 100.0
    assert ro.update(r, 0.05) < 0
    assert ro.update(np.zeros(10), 0.05) < 0  # low-passed: the left turn decays, it does not vanish
    small = YawReadout(HAND | {"dec_w_yaw_DNa02": 0.01, "dec_deadzone": 2.0}, np.full(10, 100.0))
    assert small.update(r, 0.05) == 0.0  # inside the soft deadzone
    assert YawReadout(HAND, np.zeros(10)).norm.min() == NORM_FLOOR_HZ


def test_param_space_round_trip_and_bounds():
    assert SPACE.dim == 34 and SPACE.names == NAMES and len(set(NAMES)) == 34
    x = np.random.default_rng(0).random(34)
    assert np.allclose(SPACE.encode(SPACE.decode(x)), x, atol=1e-12)
    d = SPACE.decode(np.full(34, 0.5))
    assert d["enc_r_max_hz"] == pytest.approx(math.sqrt(20 * 400))  # log scale
    assert d["dec_w_yaw_DNg13"] == pytest.approx(0.0) and d["enc_overlap_deg"] == pytest.approx(10.0)  # linear
    assert SPACE.decode(np.full(34, 2.0))["dec_tau_yaw_ms"] == pytest.approx(150.0)  # x is clipped to [0, 1]
    assert SPACE.encode(HAND | {"dec_g_yaw": 7.0})[NAMES.index("dec_g_yaw")] == 1.0  # values clipped to bounds
    assert np.allclose(SPACE.decode(SPACE.default_x())["enc_width_deg"], 8.0)
    with pytest.raises(KeyError):
        SPACE.encode({"enc_r_max_hz": 100.0})


def test_reference_parameters_load(tmp_path):
    for path in (PREV_PARAMS, REFERENCE_DIR / "fly_yaw_prev.json"):
        rec = json.loads(path.read_text())
        pf = load_params(path)
        assert np.allclose(pf.x, rec["x"], atol=1e-9)
        assert all(pf.params[k] == pytest.approx(v, rel=1e-9) for k, v in rec["params"].items())
        assert pf.norm_from == PREV_CALIBRATION and np.array_equal(pf.norm, load_norm())
        assert pf.params["dec_w_yaw_DNa02"] > 0  # turns toward the walker
    assert load_norm()[OUTPUT_GROUPS.index("DNa02_R")] == pytest.approx(241.01, abs=0.01)
    # x alone loads too, and a written record reads back
    only_x = tmp_path / "x.json"
    only_x.write_text(json.dumps({"x": json.loads(PREV_PARAMS.read_text())["x"], "size_blind": True}))
    from_x = load_params(only_x)
    assert from_x.params == pytest.approx(load_params(PREV_PARAMS).params) and from_x.size_blind is True
    rt = tmp_path / "rt.json"
    rt.write_text(json.dumps(params_record(HAND, np.arange(10) + 5.0, arm="TEST")))
    back = load_params(rt)
    assert back.arm == "TEST" and back.norm.tolist() == (np.arange(10) + 5.0).tolist() and back.norm_from == rt
    (tmp_path / "bad.json").write_text(json.dumps({"x": [0.5] * 10}))
    with pytest.raises(ValueError):
        load_params(tmp_path / "bad.json")


def test_hand_calibration_file():
    """kite/fly/data/init/fly_yaw_hand.json, written by python -m kite.fly.calibrate."""
    rec = json.loads(HAND_FILE.read_text())
    pf = load_params(HAND_FILE)
    assert rec["arm"] == "FLY-YAW-HAND" and pf.norm_from == HAND_FILE and set(rec["norm"]) == set(OUTPUT_GROUPS)
    assert rec["fit"]["r2"] > 0.7 and rec["fit"]["r2_holdout"] > 0.7 and rec["fit"]["at_bound"] == 0
    assert pf.params["dec_w_yaw_DNa02"] > 0 and pf.params["dec_w_yaw_DNg13"] > 0
    assert all(pf.params[k] == pytest.approx(HAND[k]) for k in NAMES if k.startswith("enc_"))  # encoder untouched
    z = rec["fit"]["z_by_bearing"]
    assert z["-30"] < z["-10"] < 0 < z["+10"] < z["+18"]


def test_calibration_runs_quick():
    rec = calibrate(out=None, repeats=1, settle_ms=100.0, measure_ms=300.0)
    assert rec["fit"]["r2"] > 0.6 and rec["params"]["dec_w_yaw_DNa02"] > 0 and len(rec["x"]) == 34
    # speed is test_tick_speed's job; here only that it was measured (a timing bound in a
    # correctness test made this fail whenever the machine was busy)
    assert len(rec["rates"]["X"]) == 27 and rec["brain_ms_per_tick"] > 0
