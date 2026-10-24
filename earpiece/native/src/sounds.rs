//! Synthesised cue sounds. A line-for-line port of `earpiece/sounds.py`: the four
//! sonar timbres (knock, buzz, pip, glide) with their loudness matching and the
//! gentle profile, the waypoint beacon (pings, turn pairs, the "behind" pulses) and
//! the status earcons. The design rationale lives in the Python module's docstring.
//!
//! Parity with Python is kept down to the arithmetic, so the Python test suite can
//! require identical samples:
//!   * every product and sum runs in numpy's order (scalar chains like
//!     `2 * math.pi * f * t` fold the scalars first, then touch the array);
//!   * `np.linspace`, `np.hanning`, `np.cumsum` and `np.sum` (pairwise) are
//!     reproduced exactly;
//!   * `scipy.signal.lfilter` runs its transposed direct form II with the same
//!     contraction into fused multiply-adds that clang applies to scipy's C on
//!     arm64 (plain multiply-add elsewhere, like the x86-64 wheels);
//!   * the Butterworth designs are scipy's `butter` coefficients, written out
//!     (the Python tests check them against scipy);
//!   * the knock's mallet noise is `np.random.default_rng(seed).standard_normal`
//!     (see `numpy_rng`);
//!   * float32 steps happen where numpy's happen (`_norm`, `_finish`, the
//!     concatenation of notes, and a float32 cue scaled by a Python float).

use crate::numpy_rng;
use crate::spatial::pairwise_sum;
use std::f64::consts::PI;

pub const RATE: u32 = 48000;
const RATE_F: f64 = 48000.0;

pub const SONAR_TARGET_LU: f64 = -21.0;
pub const BEACON_TARGET_LU: f64 = -24.0;
pub const BEHIND_TARGET_LU: f64 = -23.0;
pub const GENTLE_DB: f64 = -6.0;
pub const MAX_PEAK: f64 = 0.8;
pub const PARTIAL_MAX_HZ: f64 = 7000.0;
pub const URGENCY_PITCH_HZ: (f64, f64) = (520.0, 1250.0);
pub const TURN_INTERVAL: f64 = 4.0 / 3.0;
pub const BRIGHT_SPAN_DEG: f64 = 55.0;

/// `sounds.PROFILES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    Standard,
    Gentle,
}

impl Profile {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "standard" => Some(Profile::Standard),
            "gentle" => Some(Profile::Gentle),
            _ => None,
        }
    }
}

/// The four sonar timbres; `proximity_cue` maps object kinds onto them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CueClass {
    Person,
    Vehicle,
    Obstacle,
    Drop,
}

/// `sounds.PERSON_KINDS` / `VEHICLE_KINDS` / `DROP_KINDS`, everything else an obstacle.
pub fn cue_class(kind: &str) -> CueClass {
    match kind {
        "person" | "people" | "pedestrian" | "crowd" => CueClass::Person,
        "vehicle" | "car" | "bike" | "bicycle" | "scooter" | "motorcycle" | "bus" | "truck" => {
            CueClass::Vehicle
        }
        "stairs_down" | "curb" | "drop" | "stairs" | "step" => CueClass::Drop,
        _ => CueClass::Obstacle,
    }
}

// ---------------------------------------------------------------- numpy helpers
/// Python's `max(a, b)`: `b` only if strictly greater.
#[inline]
fn pmax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's `min(a, b)`: `b` only if strictly smaller.
#[inline]
fn pmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `int(x)` of a non-negative float: truncation.
#[inline]
fn trunc(x: f64) -> usize {
    if x > 0.0 {
        x as usize
    } else {
        0
    }
}

/// `np.linspace(start, stop, num)` (endpoint=True).
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    let delta = stop - start;
    let div = num - 1;
    let mut y: Vec<f64> = (0..num).map(|i| i as f64).collect();
    if div > 0 {
        let step = delta / div as f64;
        if step == 0.0 {
            for v in y.iter_mut() {
                *v = *v / div as f64 * delta;
            }
        } else {
            for v in y.iter_mut() {
                *v *= step;
            }
        }
    } else {
        for v in y.iter_mut() {
            *v *= delta;
        }
    }
    for v in y.iter_mut() {
        *v += start;
    }
    if num > 1 {
        y[num - 1] = stop;
    }
    y
}

/// `np.hanning(m)`.
pub fn hanning(m: usize) -> Vec<f64> {
    match m {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => {
            let mf = m as f64;
            (0..m)
                .map(|i| {
                    let n = (1.0 - mf) + 2.0 * i as f64;
                    0.5 + 0.5 * (PI * n / (mf - 1.0)).cos()
                })
                .collect()
        }
    }
}

/// `np.cumsum`: a running sum, left to right.
fn cumsum(x: &[f64]) -> Vec<f64> {
    let mut s = 0.0;
    x.iter()
        .map(|&v| {
            s += v;
            s
        })
        .collect()
}

fn max_abs(x: &[f64]) -> f64 {
    x.iter()
        .fold(0.0f64, |m, &v| if v.abs() > m { v.abs() } else { m })
}

/// `scipy.signal.lfilter(b, a, x)` from rest: transposed direct form II, as scipy's
/// `_linear_filter` C loop computes it on this platform (see the module docs).
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let n = b.len();
    debug_assert!(n == a.len() && n >= 2);
    let a0 = a[0];
    let b: Vec<f64> = b.iter().map(|v| v / a0).collect();
    let a: Vec<f64> = a.iter().map(|v| v / a0).collect();
    let mut z = vec![0.0f64; n - 1];
    let fused = cfg!(target_arch = "aarch64");
    x.iter()
        .map(|&xn| {
            let yn = if fused {
                b[0].mul_add(xn, z[0])
            } else {
                z[0] + b[0] * xn
            };
            for i in 0..n - 2 {
                z[i] = if fused {
                    (-yn).mul_add(a[i + 1], xn.mul_add(b[i + 1], z[i + 1]))
                } else {
                    z[i + 1] + xn * b[i + 1] - yn * a[i + 1]
                };
            }
            z[n - 2] = if fused {
                xn.mul_add(b[n - 1], -(yn * a[n - 1]))
            } else {
                xn * b[n - 1] - yn * a[n - 1]
            };
            yn
        })
        .collect()
}

// The scipy.signal.butter(..., fs=48000) designs sounds.py uses, printed with repr(). A
// wrong digit would break the sample parity in tests/test_rust_sounds.py (gentle cues use the
// high-pass, the knock's click the band-pass, the "behind" pulses the low-pass).
const GENTLE_HP: ([f64; 5], [f64; 5]) = (
    // butter(4, 1500, btype="highpass")
    [
        0.7733467891606218,
        -3.0933871566424873,
        4.640080734963731,
        -3.0933871566424873,
        0.7733467891606218,
    ],
    [
        1.0,
        -3.4873077415499,
        4.589291232078407,
        -2.6988843913407545,
        0.5980652616008872,
    ],
);
const CLICK_BP: ([f64; 5], [f64; 5]) = (
    // butter(2, (1500, 6000), btype="bandpass")
    [
        0.060498507630940564,
        0.0,
        -0.12099701526188113,
        0.0,
        0.060498507630940564,
    ],
    [
        1.0,
        -2.9435274178551247,
        3.4275681738020976,
        -1.9037859794941834,
        0.4359073982443404,
    ],
);
const BEHIND_LP: ([f64; 3], [f64; 3]) = (
    // butter(2, 1800, btype="lowpass")
    [
        0.011857682643241161,
        0.023715365286482323,
        0.011857682643241161,
    ],
    [1.0, -1.6692031429311929, 0.7166338735041575],
);
// ITU-R BS.1770-4 K-weighting at 48 kHz (sounds._K_SHELF, sounds._K_HPF)
const K_SHELF: ([f64; 3], [f64; 3]) = (
    [1.53512485958697, -2.69169618940638, 1.19839281085285],
    [1.0, -1.69065929318241, 0.73248077421585],
);
const K_HPF: ([f64; 3], [f64; 3]) = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]);

// ---------------------------------------------------------------- building blocks
/// `_t(dur_s)`: sample times of a `dur_s` cue.
fn t_axis(dur_s: f64) -> Vec<f64> {
    let n = (dur_s * RATE_F).round_ties_even();
    let n = if n > 0.0 { n as usize } else { 0 };
    (0..n).map(|i| i as f64 / RATE_F).collect()
}

/// `_env(n, attack_s, decay_tau_s, release_s)`.
fn env(n: usize, attack_s: f64, decay_tau_s: Option<f64>, release_s: f64) -> Vec<f64> {
    let mut e = vec![1.0f64; n];
    let a = trunc(attack_s * RATE_F).max(1);
    for (i, v) in linspace(0.0, 1.0, a).into_iter().enumerate().take(n) {
        e[i] = v * v;
    }
    if let Some(tau) = decay_tau_s.filter(|&d| d != 0.0) {
        for (i, v) in e.iter_mut().enumerate() {
            let t = i as f64 / RATE_F;
            let d = t - attack_s;
            let m = if d > 0.0 || d.is_nan() { d } else { 0.0 };
            *v *= (-m / tau).exp();
        }
    }
    let r = (n / 2).min(trunc(release_s * RATE_F)).max(1);
    if n > 0 {
        let r = r.min(n);
        for (v, g) in e[n - r..].iter_mut().zip(linspace(1.0, 0.0, r)) {
            *v *= g;
        }
    }
    e
}

/// `_norm(x, peak)`: scale to `peak`, then float32.
fn norm(x: &[f64], peak: f64) -> Vec<f32> {
    let m = max_abs(x);
    let s = if m > 0.0 { peak / m } else { 0.0 };
    x.iter().map(|&v| (v * s) as f32).collect()
}

fn gap(s: f64) -> Vec<f32> {
    vec![0.0f32; trunc(s * RATE_F)]
}

fn seq(parts: &[Vec<f32>]) -> Vec<f32> {
    parts.concat()
}

/// `loudness_lu`: BS.1770 K-weighted energy of one cue over a 400 ms window.
pub fn loudness_lu(x: &[f64]) -> f64 {
    let tail = trunc(0.05 * RATE_F);
    let mut y = Vec::with_capacity(x.len() + tail);
    y.extend_from_slice(x);
    y.resize(x.len() + tail, 0.0);
    let y = lfilter(&K_SHELF.0, &K_SHELF.1, &y);
    let y = lfilter(&K_HPF.0, &K_HPF.1, &y);
    let ms = pairwise_sum(&y, |v| v * v) / (0.4 * RATE_F);
    if ms > 0.0 {
        -0.691 + 10.0 * ms.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// `_to_loudness` of a float64 cue: the gain is applied in float64, then float32.
fn to_loudness(x: &[f64], target_lu: f64, peak_max: f64) -> Vec<f32> {
    let lu = loudness_lu(x);
    if !lu.is_finite() {
        return x.iter().map(|&v| v as f32).collect();
    }
    let mut g = 10f64.powf((target_lu - lu) / 20.0);
    let m = max_abs(x);
    if m * g > peak_max {
        g = peak_max / m;
    }
    x.iter().map(|&v| (v * g) as f32).collect()
}

/// `_to_loudness` of a float32 cue: numpy scales a float32 array by a Python float in
/// float32 (NEP 50), so the gain is rounded to float32 first.
fn to_loudness_f32(x: &[f32], target_lu: f64, peak_max: f64) -> Vec<f32> {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let lu = loudness_lu(&x64);
    if !lu.is_finite() {
        return x.to_vec();
    }
    let mut g = 10f64.powf((target_lu - lu) / 20.0);
    let m = max_abs(&x64);
    if m * g > peak_max {
        g = peak_max / m;
    }
    let g = g as f32;
    x.iter().map(|&v| v * g).collect()
}

fn urgency(pitch_hz: f64) -> f64 {
    let (lo, hi) = URGENCY_PITCH_HZ;
    pmin(1.0, pmax(0.0, (pitch_hz - lo) / (hi - lo)))
}

/// `gentle_amount(pitch_hz)`: 0 in the standard profile; in gentle, 1 for calm cues,
/// fading to 0 (smoothstep) between 60 % and 100 % urgency.
pub fn gentle_amount(pitch_hz: Option<f64>, profile: Profile) -> f64 {
    if profile != Profile::Gentle {
        return 0.0;
    }
    let Some(p) = pitch_hz else {
        return 1.0;
    };
    let u = (urgency(p) - 0.6) / 0.4;
    let u = pmin(1.0, pmax(0.0, u));
    1.0 - u * u * (3.0 - 2.0 * u)
}

fn dur(base_s: f64, amount: f64) -> f64 {
    pmax(0.04, base_s * (1.0 - 0.4 * amount))
}

fn gentle_filter(x: Vec<f64>, amount: f64) -> Vec<f64> {
    if amount <= 0.0 {
        return x;
    }
    let h = lfilter(&GENTLE_HP.0, &GENTLE_HP.1, &x);
    x.iter()
        .zip(&h)
        .map(|(&v, &hv)| (1.0 - amount) * v + amount * hv)
        .collect()
}

/// `_finish`: gentle filter, click-free edges, loudness match (-6 dB in gentle).
fn finish(x: Vec<f64>, target_lu: f64, amount: f64, fade_s: f64) -> Vec<f32> {
    let mut x = gentle_filter(x, amount);
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let r = (n / 4).min(trunc(fade_s * RATE_F)).max(1).min(n);
    for (v, g) in x[n - r..].iter_mut().zip(linspace(1.0, 0.0, r)) {
        *v *= g;
    }
    x[0] = 0.0;
    to_loudness(&x, target_lu + GENTLE_DB * amount, MAX_PEAK)
}

/// `_harmonics(ph, f_top, amps)`: harmonics while k * f_top stays <= PARTIAL_MAX_HZ.
fn harmonics(ph: &[f64], f_top: f64, amps: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0f64; ph.len()];
    for (i, &a) in amps.iter().enumerate() {
        let k = (i + 1) as f64;
        if k * f_top > PARTIAL_MAX_HZ {
            break;
        }
        for (xv, &p) in x.iter_mut().zip(ph) {
            *xv += a * (k * p).sin();
        }
    }
    x
}

/// `_click(n, seed)`: 4 ms of band-passed (1.5-6 kHz), Hann-windowed seeded noise.
fn click(n: usize, seed: u64) -> Vec<f64> {
    let m = trunc(0.004 * RATE_F);
    let noise = numpy_rng::standard_normal(seed, m);
    let burst = lfilter(&CLICK_BP.0, &CLICK_BP.1, &noise);
    let win = hanning(m);
    let mut out = vec![0.0f64; n];
    for i in 0..n.min(m) {
        out[i] = burst[i] * win[i];
    }
    out
}

/// `2 * math.pi * f * t` for a scalar chain `w = 2 * math.pi * f ...` already folded.
fn phase(w: f64, t: &[f64]) -> Vec<f64> {
    t.iter().map(|&v| w * v).collect()
}

/// `2 * math.pi * np.cumsum(f) / RATE`.
fn phase_of(freqs: &[f64]) -> Vec<f64> {
    cumsum(freqs)
        .into_iter()
        .map(|c| 2.0 * PI * c / RATE_F)
        .collect()
}

// ---------------------------------------------------------------- sonar kinds
pub fn person_cue(freq: f64, profile: Profile) -> Vec<f32> {
    let amount = gentle_amount(Some(freq), profile);
    let f = freq * 0.8;
    let t = t_axis(dur(0.08, amount));
    let w = 2.0 * PI * f;
    let mut x: Vec<f64> = t
        .iter()
        .map(|&tv| (w * tv).sin() * (-tv / 0.045).exp())
        .collect();
    for (ratio, amp, tau) in [(2.76, 0.5, 0.025), (3.9, 0.35, 0.015), (5.4, 0.25, 0.01)] {
        if ratio * f <= PARTIAL_MAX_HZ {
            let w = 2.0 * PI * ratio * f;
            for (xv, &tv) in x.iter_mut().zip(&t) {
                *xv += amp * (w * tv).sin() * (-tv / tau).exp();
            }
        }
    }
    let c = click(t.len(), trunc(f) as u64);
    for (xv, cv) in x.iter_mut().zip(&c) {
        *xv += 0.25 * cv;
    }
    for (xv, ev) in x.iter_mut().zip(env(t.len(), 0.002, None, 0.008)) {
        *xv *= ev;
    }
    finish(x, SONAR_TARGET_LU, amount, 0.004)
}

pub fn vehicle_cue(freq: f64, profile: Profile) -> Vec<f32> {
    let amount = gentle_amount(Some(freq), profile);
    let f0 = freq * 0.75;
    let t = t_axis(dur(0.09, amount));
    let f: Vec<f64> = t
        .iter()
        .map(|&tv| f0 * (1.0 + 0.06 * (tv / 0.03).min(1.0)))
        .collect();
    let ph = phase_of(&f);
    let amps: Vec<f64> = (1..40).map(|k| 1.0 / k as f64).collect();
    let mut x = harmonics(&ph, f0 * 1.06, &amps);
    let w = 2.0 * PI * 60.0;
    for (xv, &tv) in x.iter_mut().zip(&t) {
        *xv *= 1.0 - 0.45 * (0.5 + 0.5 * (w * tv).cos());
    }
    let last = *t.last().unwrap_or(&0.0);
    for (xv, ev) in x
        .iter_mut()
        .zip(env(t.len(), 0.003, Some(last * 1.2), 0.01))
    {
        *xv *= ev;
    }
    finish(x, SONAR_TARGET_LU, amount, 0.004)
}

pub fn obstacle_cue(freq: f64, profile: Profile) -> Vec<f32> {
    let amount = gentle_amount(Some(freq), profile);
    let t = t_axis(dur(0.07, amount));
    let ph = phase(2.0 * PI * freq, &t);
    let amps: Vec<f64> = (1..7).map(|k| (k as f64).powf(-1.5)).collect();
    let mut x = harmonics(&ph, freq, &amps);
    for (xv, ev) in x.iter_mut().zip(env(t.len(), 0.004, None, 0.015)) {
        *xv *= ev;
    }
    finish(x, SONAR_TARGET_LU, amount, 0.004)
}

pub fn drop_cue(freq: f64, profile: Profile) -> Vec<f32> {
    let amount = gentle_amount(Some(freq), profile);
    let t = t_axis(dur(0.12, amount));
    let (f0, f1) = (freq * 1.2, freq * 0.6);
    let last = *t.last().unwrap_or(&0.0);
    let ratio = f1 / f0;
    let f: Vec<f64> = t.iter().map(|&tv| f0 * ratio.powf(tv / last)).collect();
    let ph = phase_of(&f);
    let mut x = harmonics(&ph, f0, &[1.0, 0.5, 0.33, 0.25]);
    for (xv, ev) in x
        .iter_mut()
        .zip(env(t.len(), 0.004, Some(last * 0.8), 0.012))
    {
        *xv *= ev;
    }
    finish(x, SONAR_TARGET_LU, amount, 0.004)
}

/// `proximity_cue(kind, freq)`: the repeated sonar pip of one object class.
pub fn proximity_cue(kind: &str, freq: f64, profile: Profile) -> Vec<f32> {
    class_cue(cue_class(kind), freq, profile)
}

pub fn class_cue(class: CueClass, freq: f64, profile: Profile) -> Vec<f32> {
    match class {
        CueClass::Person => person_cue(freq, profile),
        CueClass::Vehicle => vehicle_cue(freq, profile),
        CueClass::Drop => drop_cue(freq, profile),
        CueClass::Obstacle => obstacle_cue(freq, profile),
    }
}

// ---------------------------------------------------------------- beacon
pub fn beacon_pitch(region: &str) -> Option<f64> {
    match region {
        "on" => Some(2093.0),
        "near" => Some(1175.0),
        "side" => Some(880.0),
        "behind" => Some(784.0),
        _ => None,
    }
}

pub fn beacon_brightness(bearing_deg: f64) -> f64 {
    let a = pmin(bearing_deg.abs(), BRIGHT_SPAN_DEG);
    0.5 * (1.0 + (PI * a / BRIGHT_SPAN_DEG).cos())
}

fn beacon_ping(freq: f64, brightness: f64, dur_s: f64) -> Vec<f64> {
    let t = t_axis(dur_s);
    let slope = 2.6 - 1.8 * pmin(1.0, pmax(0.0, brightness));
    let amps: Vec<f64> = (1..8).map(|k| (k as f64).powf(-slope)).collect();
    let mut x = harmonics(&phase(2.0 * PI * freq, &t), freq, &amps);
    for (xv, ev) in x.iter_mut().zip(env(t.len(), 0.004, Some(0.035), 0.012)) {
        *xv *= ev;
    }
    x
}

/// `_notes`: notes matched to each other, then the whole cue to `target_lu`.
fn notes(notes: Vec<Vec<f64>>, gap_s: f64, target_lu: f64, amount: f64, fade_s: f64) -> Vec<f32> {
    let mut parts: Vec<Vec<f32>> = Vec::new();
    let k = notes.len();
    for (i, n) in notes.into_iter().enumerate() {
        parts.push(finish(n, target_lu, amount, fade_s));
        if i + 1 < k {
            parts.push(gap(gap_s));
        }
    }
    to_loudness_f32(&seq(&parts), target_lu + GENTLE_DB * amount, MAX_PEAK)
}

pub fn turn_pitches(f_low: f64, bearing_deg: f64) -> (f64, f64) {
    let hi = f_low * TURN_INTERVAL;
    if bearing_deg >= 0.0 {
        (f_low, hi)
    } else {
        (hi, f_low)
    }
}

/// `beacon_tick(region, bearing)`; None for an unknown region.
pub fn beacon_tick(region: &str, bearing_deg: f64, profile: Profile) -> Option<Vec<f32>> {
    if region == "behind" {
        return Some(beacon_behind());
    }
    let f = beacon_pitch(region)?;
    let amount = gentle_amount(None, profile);
    let b = if region != "side" {
        beacon_brightness(bearing_deg)
    } else {
        0.0
    };
    let (f1, f2) = if region == "on" {
        (f, f)
    } else {
        turn_pitches(f, bearing_deg)
    };
    let ns = vec![
        beacon_ping(f1, b, dur(0.07, amount)),
        beacon_ping(f2, b, dur(0.09, amount)),
    ];
    Some(notes(ns, 0.03, BEACON_TARGET_LU, amount, 0.004))
}

pub fn beacon_behind() -> Vec<f32> {
    let f = 784.0;
    let ns = (0..3)
        .map(|_| {
            let t = t_axis(0.075);
            let x = harmonics(&phase(2.0 * PI * f, &t), f, &[1.0, 0.6, 0.15]);
            let e = env(t.len(), 0.008, Some(0.06), 0.02);
            let xe: Vec<f64> = x.iter().zip(&e).map(|(a, b)| a * b).collect();
            lfilter(&BEHIND_LP.0, &BEHIND_LP.1, &xe)
        })
        .collect();
    notes(ns, 0.06, BEHIND_TARGET_LU, 0.0, 0.01)
}

// ---------------------------------------------------------------- status cues
pub fn sine_pip(freq: f64, dur_s: f64, peak: f64) -> Vec<f32> {
    let t = t_axis(dur_s);
    let (w1, w2) = (2.0 * PI * freq, 2.0 * PI * 2.0 * freq);
    let e = env(t.len(), 0.003, Some(dur_s * 0.6), 0.006);
    let x: Vec<f64> = t
        .iter()
        .zip(&e)
        .map(|(&tv, &ev)| ((w1 * tv).sin() + 0.25 * (w2 * tv).sin()) * ev)
        .collect();
    norm(&x, peak)
}

pub fn knock(freq: f64, dur_s: f64, peak: f64) -> Vec<f32> {
    let t = t_axis(dur_s);
    let (w1, w2, w3) = (
        2.0 * PI * freq,
        2.0 * PI * 3.9 * freq,
        2.0 * PI * 9.2 * freq,
    );
    let e = env(t.len(), 0.002, None, 0.006);
    let x: Vec<f64> = t
        .iter()
        .zip(&e)
        .map(|(&tv, &ev)| {
            ((w1 * tv).sin() * (-tv / (dur_s * 0.5)).exp()
                + 0.5 * (w2 * tv).sin() * (-tv / 0.012).exp()
                + 0.25 * (w3 * tv).sin() * (-tv / 0.005).exp())
                * ev
        })
        .collect();
    norm(&x, peak)
}

pub fn buzz(freq: f64, dur_s: f64, peak: f64) -> Vec<f32> {
    let t = t_axis(dur_s);
    let f: Vec<f64> = t
        .iter()
        .map(|&tv| freq * (1.0 + 0.06 * (tv / 0.03).min(1.0)))
        .collect();
    let ph = phase_of(&f);
    let e = env(t.len(), 0.003, Some(dur_s * 0.8), 0.006);
    let x: Vec<f64> = ph
        .iter()
        .zip(&e)
        .map(|(&p, &ev)| {
            let mut s = 0.0;
            for k in [1.0f64, 3.0, 5.0, 7.0] {
                s += (k * p).sin() / k;
            }
            s * ev
        })
        .collect();
    norm(&x, peak)
}

pub fn glide(f0: f64, f1: f64, dur_s: f64, peak: f64) -> Vec<f32> {
    let t = t_axis(dur_s);
    let last = *t.last().unwrap_or(&0.0);
    let ratio = f1 / f0;
    let f: Vec<f64> = t.iter().map(|&tv| f0 * ratio.powf(tv / last)).collect();
    let ph = phase_of(&f);
    let e = env(t.len(), 0.003, Some(dur_s * 0.7), 0.006);
    let x: Vec<f64> = ph
        .iter()
        .zip(&e)
        .map(|(&p, &ev)| (p.sin() + 0.3 * (2.0 * p).sin()) * ev)
        .collect();
    norm(&x, peak)
}

pub fn chime(freqs: &[f64], step_s: f64, peak: f64) -> Vec<f32> {
    let parts: Vec<Vec<f32>> = freqs
        .iter()
        .enumerate()
        .map(|(i, &f)| knock(f, if i == freqs.len() - 1 { 0.16 } else { step_s }, peak))
        .collect();
    seq(&parts)
}

/// The named status earcons of `sounds.py` (`nav_chime` takes the instruction's side,
/// -1 left, 0 none, +1 right, instead of its text).
pub fn status(name: &str, args: &[f64]) -> Option<Vec<f32>> {
    let a = |i: usize| args.get(i).copied();
    Some(match name {
        "arrived" => chime(&[784.0, 988.0, 1175.0, 1568.0], 0.09, 0.42),
        "ready" => chime(&[659.0, 988.0], 0.09, 0.42),
        "listening" => glide(600.0, 1300.0, 0.16, 0.4),
        "heard" => seq(&[
            sine_pip(1319.0, 0.05, 0.38),
            gap(0.035),
            sine_pip(1319.0, 0.06, 0.38),
        ]),
        "error" => buzz(196.0, 0.22, 0.4),
        "sensor_lost" => seq(&[
            sine_pip(784.0, 0.1, 0.45),
            gap(0.03),
            sine_pip(659.0, 0.14, 0.45),
            gap(0.12),
            sine_pip(784.0, 0.1, 0.45),
            gap(0.03),
            sine_pip(659.0, 0.14, 0.45),
        ]),
        "sensor_back" => seq(&[
            sine_pip(659.0, 0.08, 0.42),
            gap(0.03),
            sine_pip(784.0, 0.12, 0.42),
        ]),
        "critical_alarm" => {
            let b = buzz(1320.0, 0.06, 0.6);
            seq(&[b.clone(), gap(0.035), b.clone(), gap(0.035), b])
        }
        "crossing_go" => {
            let mut parts = Vec::new();
            for _ in 0..4 {
                parts.push(knock(2000.0, 0.03, 0.4));
                parts.push(gap(0.07));
            }
            seq(&parts)
        }
        "crossing_wait" => seq(&[
            sine_pip(440.0, 0.05, 0.4),
            gap(0.4),
            sine_pip(440.0, 0.05, 0.4),
        ]),
        "ear_check_left" => knock(880.0, 0.12, 0.5),
        "ear_check_right" => knock(1320.0, 0.12, 0.5),
        "nav_chime" => {
            let side = a(0)?;
            if side == 0.0 {
                chime(&[1175.0], 0.09, 0.32)
            } else {
                let (f1, f2) = turn_pitches(988.0, side);
                chime(&[f1, f2], 0.07, 0.32)
            }
        }
        "beacon_behind" => beacon_behind(),
        "sine_pip" => sine_pip(a(0)?, a(1)?, a(2)?),
        "knock" => knock(a(0)?, a(1)?, a(2)?),
        "buzz" => buzz(a(0)?, a(1)?, a(2)?),
        "glide" => glide(a(0)?, a(1)?, a(2)?, a(3)?),
        "chime" => {
            // args: step_s, peak, *freqs
            if args.len() < 3 {
                return None;
            }
            chime(&args[2..], args[0], args[1])
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_numpy() {
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspace(1.0, 0.0, 1), vec![1.0]);
        assert_eq!(linspace(0.0, 1.0, 1), vec![0.0]);
        assert!(linspace(0.0, 1.0, 0).is_empty());
        // np.linspace(1.0, 0.0, 3) == [1.0, 0.5, 0.0]
        assert_eq!(linspace(1.0, 0.0, 3), vec![1.0, 0.5, 0.0]);
    }

    #[test]
    fn hanning_is_symmetric_and_zero_at_the_edges() {
        let w = hanning(192);
        assert_eq!(w.len(), 192);
        assert!(w[0].abs() < 1e-15 && w[191].abs() < 1e-15);
        for i in 0..96 {
            assert_eq!(w[i], w[191 - i]);
        }
        assert_eq!(hanning(1), vec![1.0]);
    }

    #[test]
    fn loudness_of_a_full_scale_997_hz_sine_is_minus_3_lufs() {
        let x: Vec<f64> = (0..19200)
            .map(|i| (2.0 * PI * 997.0 * i as f64 / RATE_F).sin())
            .collect();
        assert!((loudness_lu(&x) + 3.01).abs() < 0.1, "{}", loudness_lu(&x));
        assert_eq!(loudness_lu(&[0.0; 100]), f64::NEG_INFINITY);
    }

    #[test]
    fn kind_cues_hit_the_loudness_target_with_clean_edges() {
        for class in [
            CueClass::Person,
            CueClass::Vehicle,
            CueClass::Obstacle,
            CueClass::Drop,
        ] {
            for f in [525.0, 900.0, 1250.0] {
                let x = class_cue(class, f, Profile::Standard);
                let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
                assert!(x.len() as f64 / RATE_F >= 0.06 && x.len() as f64 / RATE_F <= 0.13);
                assert_eq!(x[0], 0.0);
                assert!(x.last().unwrap().abs() < 1e-3);
                assert!(max_abs(&x64) <= MAX_PEAK + 1e-6);
                assert!(
                    (loudness_lu(&x64) - SONAR_TARGET_LU).abs() < 1.0,
                    "{class:?} {f}"
                );
            }
        }
    }

    #[test]
    fn gentle_is_shorter_and_quieter_until_critical() {
        for class in [CueClass::Person, CueClass::Obstacle] {
            let s = class_cue(class, 600.0, Profile::Standard);
            let g = class_cue(class, 600.0, Profile::Gentle);
            assert!(g.len() < s.len());
            let crit_s = class_cue(class, 1250.0, Profile::Standard);
            let crit_g = class_cue(class, 1250.0, Profile::Gentle);
            assert_eq!(crit_s, crit_g, "critical cues are untouched by gentle");
        }
        assert_eq!(gentle_amount(Some(600.0), Profile::Standard), 0.0);
        assert_eq!(gentle_amount(None, Profile::Gentle), 1.0);
        assert_eq!(gentle_amount(Some(1250.0), Profile::Gentle), 0.0);
    }

    #[test]
    fn kinds_map_onto_four_timbres() {
        assert_eq!(cue_class("crowd"), CueClass::Person);
        assert_eq!(cue_class("bike"), CueClass::Vehicle);
        assert_eq!(cue_class("curb"), CueClass::Drop);
        assert_eq!(cue_class("pole"), CueClass::Obstacle);
        assert_eq!(cue_class("anything"), CueClass::Obstacle);
    }

    #[test]
    fn beacon_regions_and_status_cues_render() {
        for r in ["on", "near", "side", "behind"] {
            for b in [-40.0, 0.0, 40.0] {
                let x = beacon_tick(r, b, Profile::Standard).unwrap();
                assert!(!x.is_empty() && x.iter().all(|v| v.is_finite()));
            }
        }
        assert!(beacon_tick("nowhere", 0.0, Profile::Standard).is_none());
        for n in [
            "arrived",
            "ready",
            "listening",
            "heard",
            "error",
            "sensor_lost",
            "sensor_back",
            "critical_alarm",
            "crossing_go",
            "crossing_wait",
            "ear_check_left",
            "ear_check_right",
            "beacon_behind",
        ] {
            assert!(status(n, &[]).is_some_and(|x| !x.is_empty()), "{n}");
        }
        assert!(status("nav_chime", &[1.0]).is_some());
        assert!(status("nav_chime", &[]).is_none());
        assert!(status("nope", &[]).is_none());
    }

    #[test]
    fn synthesis_is_deterministic() {
        assert_eq!(
            person_cue(900.0, Profile::Standard),
            person_cue(900.0, Profile::Standard)
        );
    }
}
