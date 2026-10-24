//! Binaural placement. A line-for-line port of `earpiece/spatial.py` (Brown & Duda
//! 1998 spherical head: per-ear head-shadow filter, per-ear Woodworth delay, the
//! "behind" low-pass, the extra constant-power ILD and the energy normalisation),
//! so a cue placed by Rust is the cue placed by Python; the Python test suite
//! checks that parity sample by sample.
//!
//! Parity is kept down to the arithmetic: the same f64 filter recursion as
//! scipy's `lfilter`, the same f32 rounding where numpy works in float32 (the
//! shaded signal and the linear-interpolation delay), the same alpha rounding to
//! three decimals that keys the Python coefficient cache, the same evaluation
//! order for every product.
//!
//! On top of the reference there are two things Python does not have:
//!   * `render_batch` places one mono cue at many azimuths, sharing the
//!     direction-independent work (f64 conversion, input energy, the behind
//!     low-pass) and spreading azimuths over threads; used to pre-warm the
//!     sonar and beacon caches at startup.
//!   * `Delay::Sinc`, an opt-in Kaiser-windowed-sinc fractional delay. Linear
//!     interpolation is a comb that dulls the top octave by up to ~10 dB when the
//!     delay lands half-way between samples, so the timbre of a cue wobbles with
//!     azimuth; the sinc keeps it flat and the ITD sub-sample exact. The default
//!     stays linear so both engines produce identical samples.

use std::f64::consts::PI;

pub const HEAD_RADIUS_M: f64 = 0.0875;
pub const SPEED_OF_SOUND: f64 = 343.0;
pub const ALPHA_MIN: f64 = 0.1;
pub const THETA_MIN_DEG: f64 = 150.0;
const BEHIND_FC_HZ: f64 = 2500.0;
const BEHIND_GAIN: f64 = 0.85;
/// Kaiser beta of the windowed-sinc delay: ~-60 dB sidelobes, flat to ~0.4 fs at 16 taps.
const KAISER_BETA: f64 = 6.0;
pub const MAX_SINC_TAPS: usize = 64;

/// Any angle -> (-180, 180]. `%` on f64 is C `fmod`, like `math.fmod`.
pub fn wrap_deg(a: f64) -> f64 {
    let mut a = a % 360.0;
    if a <= -180.0 {
        a += 360.0;
    } else if a > 180.0 {
        a -= 360.0;
    }
    a
}

/// Angle in degrees [0, 180] between the source and an ear's axis
/// (ear = +1 right, axis at +90 deg; -1 left, axis at -90 deg).
pub fn ear_angle(azimuth_deg: f64, ear: f64) -> f64 {
    wrap_deg(azimuth_deg - 90.0 * ear).abs()
}

/// Brown-Duda a(th): 2.0 facing the ear, 0.1 at 150 deg, back up slightly at 180.
pub fn head_shadow_alpha(ear_angle_deg: f64) -> f64 {
    (1.0 + ALPHA_MIN / 2.0)
        + (1.0 - ALPHA_MIN / 2.0)
            * (ear_angle_deg.to_radians() / THETA_MIN_DEG.to_radians() * PI).cos()
}

/// Brown-Duda per-ear delay in seconds (shifted so the earliest arrival is 0).
pub fn ear_delay_s(ear_angle_deg: f64) -> f64 {
    let th = ear_angle_deg.to_radians();
    let a_c = HEAD_RADIUS_M / SPEED_OF_SOUND;
    let t = if th < PI / 2.0 {
        -a_c * th.cos()
    } else {
        a_c * (th - PI / 2.0)
    };
    t + a_c
}

/// Python's `round(x, 3)`: the exact binary value correctly rounded to three
/// decimals, ties to even. Rust's `{:.3}` formats the exact value the same way
/// and parsing is correctly rounded, so this is bit-identical (`x * 1000` is not).
pub fn round3(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.3}").parse().unwrap_or(x)
}

/// Tail room `spatial.py` gives a cue: 2a/c (the textbook ITD) plus two samples.
/// The far ear's delay goes up to (a/c)(pi/2 + 1), a few samples more, so the
/// linear path (like Python) drops up to ~5 trailing samples of the far ear for a
/// source at +-90 deg. Cues end in a release, so that tail is silence; kept for
/// parity. The sinc path, which has no parity constraint, sizes itself fully.
pub fn pad_samples(rate: u32) -> usize {
    (2.0 * HEAD_RADIUS_M / SPEED_OF_SOUND * rate as f64).ceil() as usize + 2
}

/// Output frames for an `n`-sample cue in each delay mode.
pub fn out_frames(n: usize, rate: u32, mode: Delay) -> usize {
    match mode {
        Delay::Linear => n + pad_samples(rate),
        Delay::Sinc(taps) => n + (ear_delay_s(180.0) * rate as f64).ceil() as usize + 2 + taps,
    }
}

/// A first-order IIR `y = b0 x + b1 x[-1] - a1 y[-1]`, run exactly as scipy's
/// `lfilter` runs it (transposed direct form II, f64).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnePole {
    pub b0: f64,
    pub b1: f64,
    pub a1: f64,
}

impl OnePole {
    /// Bilinear transform of H(s) = (beta + alpha s) / (beta + s), beta = 2 c / r.
    pub fn shadow(alpha: f64, rate: u32) -> Self {
        let beta = 2.0 * SPEED_OF_SOUND / HEAD_RADIUS_M;
        let k = 2.0 * rate as f64;
        OnePole {
            b0: (beta + alpha * k) / (beta + k),
            b1: (beta - alpha * k) / (beta + k),
            a1: (beta - k) / (beta + k),
        }
    }

    /// One-pole low-pass at 2.5 kHz: the pinna's "behind you" dulling.
    pub fn behind(rate: u32) -> Self {
        let x = (-2.0 * PI * BEHIND_FC_HZ / rate as f64).exp();
        OnePole {
            b0: 1.0 - x,
            b1: 0.0,
            a1: -x,
        }
    }

    /// One sample through the filter with state `z`.
    #[inline(always)]
    pub fn tick(&self, z: &mut f64, x: f64) -> f64 {
        let y = *z + self.b0 * x;
        *z = x * self.b1 - y * self.a1;
        y
    }

    /// Stream `x` through the filter from rest, handing each output to `sink`.
    pub fn run(&self, x: &[f64], mut sink: impl FnMut(usize, f64)) {
        let mut z = 0.0f64;
        for (i, &xn) in x.iter().enumerate() {
            sink(i, self.tick(&mut z, xn));
        }
    }
}

/// How the per-ear delay handles the fractional part of a sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delay {
    /// Linear interpolation in float32: identical to `spatial.py`.
    Linear,
    /// Kaiser-windowed sinc with this many taps (even, 2..=64). Adds taps/2
    /// samples of latency to both ears, so the interaural delay is unchanged.
    Sinc(usize),
}

impl Delay {
    pub fn from_taps(taps: usize) -> Result<Self, String> {
        match taps {
            0 => Ok(Delay::Linear),
            t if t.is_multiple_of(2) && (2..=MAX_SINC_TAPS).contains(&t) => Ok(Delay::Sinc(t)),
            t => Err(format!(
                "sinc_taps must be 0 (linear) or an even number in 2..={MAX_SINC_TAPS}, got {t}"
            )),
        }
    }
}

/// Zeroth-order modified Bessel function of the first kind (power series).
fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let (mut sum, mut term, mut k) = (1.0f64, 1.0f64, 1.0f64);
    while term > sum * 1e-17 {
        term *= q / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

/// Windowed-sinc weights for a delay of `frac` in [0, 1) on top of a bulk
/// `taps / 2`: `w[j]` multiplies `x[n - whole - j]`, j = 0..=taps. Normalised
/// to unit DC gain; an integer delay (`frac == 0`) is a pure shift.
pub fn sinc_weights(frac: f64, taps: usize) -> Vec<f64> {
    let half = (taps / 2) as f64;
    let i0b = bessel_i0(KAISER_BETA);
    let mut w: Vec<f64> = (0..=taps)
        .map(|j| {
            let t = j as f64 - half - frac;
            if t.abs() >= half {
                return 0.0;
            }
            let u = t / half;
            let win = bessel_i0(KAISER_BETA * (1.0 - u * u).sqrt()) / i0b;
            let sinc = if t == 0.0 {
                1.0
            } else if t.fract() == 0.0 {
                0.0 // sin(pi k) is not exactly 0 in floating point
            } else {
                (PI * t).sin() / (PI * t)
            };
            sinc * win
        })
        .collect();
    let s: f64 = w.iter().sum();
    if s != 0.0 {
        w.iter_mut().for_each(|v| *v /= s);
    }
    w
}

/// numpy's pairwise summation (what `np.sum` does on a contiguous float64
/// array), applied to `f(a[i])`: 8 accumulators per block of up to 128, larger
/// spans split in halves on multiples of 8. Same order as numpy, so the
/// energies, and with them the normalisation gain, match bit for bit; the 8
/// independent accumulators also keep the FPU pipelines full.
fn pairwise_sum(a: &[f64], f: impl Fn(f64) -> f64 + Copy) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += f(v);
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        for (rj, &v) in r.iter_mut().zip(&a[..8]) {
            *rj = f(v);
        }
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += f(a[i + j]);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[i..] {
            res += f(v);
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2], f) + pairwise_sum(&a[n2..], f)
    }
}

/// A linear-interpolation delay as `spatial._frac_delay` sets it up: integer
/// part, the two weights rounded to f32 (numpy casts the Python float to the
/// float32 array's dtype) and how many input samples fit.
struct LinearTap {
    whole: usize,
    a: f32,
    b: f32,
    n: usize,
}

impl LinearTap {
    fn new(delay_samples: f64, len: usize, out_len: usize) -> Option<Self> {
        let whole = delay_samples.floor();
        let frac = delay_samples - whole;
        let whole = whole as isize;
        let n = (len as isize).min(out_len as isize - whole - 1);
        if n <= 0 || whole < 0 {
            return None; // nothing fits: silence, like Python
        }
        Some(LinearTap {
            whole: whole as usize,
            a: (1.0 - frac) as f32,
            b: frac as f32,
            n: n as usize,
        })
    }

    /// Delay `x` into `out` (zeroed) in numpy's two slice passes, so every
    /// sample gets the same two f32 products and one f32 add; both passes vectorise.
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        let (w, n) = (self.whole, self.n);
        for (o, &v) in out[w..w + n].iter_mut().zip(&x[..n]) {
            *o += self.a * v;
        }
        for (o, &v) in out[w + 1..w + 1 + n].iter_mut().zip(&x[..n]) {
            *o += self.b * v;
        }
    }
}

/// Windowed-sinc delay of `x` into `out` (zeroed), like `spatial._frac_delay_sinc`.
fn sinc_into(x: &[f64], delay_samples: f64, taps: usize, out: &mut [f64]) {
    let whole = delay_samples.floor();
    let k = sinc_weights(delay_samples - whole, taps);
    let whole = whole as usize;
    for (i, &v) in x.iter().enumerate() {
        let base = whole + i;
        if base >= out.len() {
            break;
        }
        let span = (out.len() - base).min(k.len());
        for (o, &kj) in out[base..base + span].iter_mut().zip(&k[..span]) {
            *o += kj * v;
        }
    }
}

/// The per-ear delay stage on its own (diagnostics and the ITD test): delays a
/// mono signal by `delay_samples` (>= 0), returns `out_len` samples.
pub fn fractional_delay(x: &[f32], delay_samples: f64, out_len: usize, mode: Delay) -> Vec<f32> {
    match mode {
        Delay::Linear => {
            let mut out = vec![0.0f32; out_len];
            if let Some(t) = LinearTap::new(delay_samples, x.len(), out_len) {
                t.apply(x, &mut out);
            }
            out
        }
        Delay::Sinc(taps) => {
            let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let mut out = vec![0.0f64; out_len];
            sinc_into(&x64, delay_samples, taps, &mut out);
            out.iter().map(|&v| v as f32).collect()
        }
    }
}

/// The direction-independent half of a render: shared by every azimuth of a batch.
pub struct Source {
    src: Vec<f64>,
    behind: Option<Vec<f64>>,
    e_in: f64,
    rate: u32,
}

impl Source {
    /// `share_behind` precomputes the behind low-pass once (batches); otherwise
    /// a render that needs it runs it inline, in the same loop as the ears.
    pub fn new(mono: &[f32], rate: u32, share_behind: bool) -> Self {
        let src: Vec<f64> = mono.iter().map(|&v| v as f64).collect();
        let e_in = pairwise_sum(&src, |v| v * v);
        let behind = share_behind.then(|| {
            let mut lp = vec![0.0f64; src.len()];
            OnePole::behind(rate).run(&src, |i, y| lp[i] = y);
            lp
        });
        Source {
            src,
            behind,
            e_in,
            rate,
        }
    }

    /// Run the source (dulled from 90 to 135 deg if behind) through both ears'
    /// head-shadow filters in one pass. The recursions are serial in time, so
    /// interleaving the independent ones (left, right, behind) is what makes
    /// this fast without changing a single rounding.
    fn shade(&self, az: f64, fl: OnePole, fr: OnePole, mut sink: impl FnMut(usize, f64, f64)) {
        let (mut zl, mut zr) = (0.0f64, 0.0f64);
        let mut ears = |i: usize, x: f64| {
            let yl = fl.tick(&mut zl, x);
            let yr = fr.tick(&mut zr, x);
            sink(i, yl, yr);
        };
        if az.abs() > 90.0 {
            // fade the dulling in from 90 to 135 deg so it does not jump at the boundary
            let w = ((az.abs() - 90.0) / 45.0).min(1.0);
            match &self.behind {
                Some(lp) => {
                    for (i, (&s, &l)) in self.src.iter().zip(lp).enumerate() {
                        ears(i, (1.0 - w) * s + w * l * BEHIND_GAIN);
                    }
                }
                None => {
                    let lpf = OnePole::behind(self.rate);
                    let mut z = 0.0f64;
                    for (i, &s) in self.src.iter().enumerate() {
                        let l = lpf.tick(&mut z, s);
                        ears(i, (1.0 - w) * s + w * l * BEHIND_GAIN);
                    }
                }
            }
        } else {
            for (i, &s) in self.src.iter().enumerate() {
                ears(i, s);
            }
        }
    }

    /// Mono -> interleaved stereo placed at `azimuth_deg` (see `spatial.render`).
    /// Length is `out_frames(n, rate, mode) * 2`.
    pub fn render(&self, azimuth_deg: f64, gain: f64, extra_ild_db: f64, mode: Delay) -> Vec<f32> {
        let rate = self.rate;
        let az = wrap_deg(azimuth_deg);
        let n = self.src.len();
        let out_len = out_frames(n, rate, mode);
        let (thl, thr) = (ear_angle(az, -1.0), ear_angle(az, 1.0));
        let fl = OnePole::shadow(round3(head_shadow_alpha(thl)), rate);
        let fr = OnePole::shadow(round3(head_shadow_alpha(thr)), rate);
        let (dl, dr) = (
            ear_delay_s(thl) * rate as f64,
            ear_delay_s(thr) * rate as f64,
        );

        // the extra ILD, applied (like numpy's in-place *=) only when asked for
        let ild = (extra_ild_db != 0.0).then(|| {
            let d = extra_ild_db * az.to_radians().sin() / 20.0; // -1 left .. +1 right
            (10f64.powf(-d / 2.0), 10f64.powf(d / 2.0))
        });
        // st is numpy's (out_len, 2) float64 array: row-major, so interleaved
        let st: Vec<f64> = match mode {
            Delay::Linear => {
                // shaded signals in f32, as numpy's .astype(np.float32)
                let (mut sl, mut sr) = (vec![0.0f32; n], vec![0.0f32; n]);
                self.shade(az, fl, fr, |i, yl, yr| {
                    sl[i] = yl as f32;
                    sr[i] = yr as f32;
                });
                let (mut l, mut r) = (vec![0.0f32; out_len], vec![0.0f32; out_len]);
                if let Some(t) = LinearTap::new(dl, n, out_len) {
                    t.apply(&sl, &mut l);
                }
                if let Some(t) = LinearTap::new(dr, n, out_len) {
                    t.apply(&sr, &mut r);
                }
                interleave(&l, &r, ild, |v| v as f64)
            }
            Delay::Sinc(taps) => {
                // shaded signals rounded to f32 like the Python path, then delayed in f64
                let (mut sl, mut sr) = (vec![0.0f64; n], vec![0.0f64; n]);
                self.shade(az, fl, fr, |i, yl, yr| {
                    sl[i] = yl as f32 as f64;
                    sr[i] = yr as f32 as f64;
                });
                let (mut l, mut r) = (vec![0.0f64; out_len], vec![0.0f64; out_len]);
                sinc_into(&sl, dl, taps, &mut l);
                sinc_into(&sr, dr, taps, &mut r);
                interleave(&l, &r, ild, |v| v)
            }
        };
        let e_out = pairwise_sum(&st, |v| v * v);
        if e_out > 0.0 && self.e_in > 0.0 {
            let s = (2.0 * self.e_in / e_out).sqrt();
            st.iter().map(|&v| (v * s * gain) as f32).collect()
        } else {
            st.iter().map(|&v| (v * gain) as f32).collect()
        }
    }
}

/// Two channels -> interleaved f64, scaled per channel when `gains` is given.
fn interleave<T: Copy>(
    l: &[T],
    r: &[T],
    gains: Option<(f64, f64)>,
    to: impl Fn(T) -> f64,
) -> Vec<f64> {
    let mut st = Vec::with_capacity(l.len() * 2);
    match gains {
        Some((gl, gr)) => {
            for (&a, &b) in l.iter().zip(r) {
                st.push(to(a) * gl);
                st.push(to(b) * gr);
            }
        }
        None => {
            for (&a, &b) in l.iter().zip(r) {
                st.push(to(a));
                st.push(to(b));
            }
        }
    }
    st
}

/// One render: `spatial.render(mono, azimuth_deg, rate, gain, extra_ild_db)`.
pub fn render(
    mono: &[f32],
    azimuth_deg: f64,
    rate: u32,
    gain: f64,
    extra_ild_db: f64,
    mode: Delay,
) -> Vec<f32> {
    Source::new(mono, rate, false).render(azimuth_deg, gain, extra_ild_db, mode)
}

/// One mono cue at many azimuths. Each result is bit-identical to `render` at
/// that azimuth; the shared work is done once and azimuths run on scoped threads.
pub fn render_batch(
    mono: &[f32],
    azimuths: &[f64],
    rate: u32,
    gain: f64,
    extra_ild_db: f64,
    mode: Delay,
) -> Vec<Vec<f32>> {
    let behind = azimuths.iter().any(|&a| wrap_deg(a).abs() > 90.0);
    let source = Source::new(mono, rate, behind);
    let work = azimuths.len() * mono.len().max(1);
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(8);
    if threads <= 1 || azimuths.len() < 4 || work < 200_000 {
        return azimuths
            .iter()
            .map(|&a| source.render(a, gain, extra_ild_db, mode))
            .collect();
    }
    let chunk = azimuths.len().div_ceil(threads);
    let source = &source;
    std::thread::scope(|s| {
        let handles: Vec<_> = azimuths
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|&a| source.render(a, gain, extra_ild_db, mode))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("binaural render thread panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const A_C: f64 = HEAD_RADIUS_M / SPEED_OF_SOUND;

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        // xorshift: deterministic, no dependency
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * 0.3
            })
            .collect()
    }

    fn energy(x: &[f32]) -> f64 {
        x.iter().map(|&v| v as f64 * v as f64).sum()
    }

    #[test]
    fn wrap_matches_python() {
        assert_eq!(wrap_deg(270.0), -90.0);
        assert_eq!(wrap_deg(-270.0), 90.0);
        assert_eq!(wrap_deg(180.0), 180.0);
        assert_eq!(wrap_deg(-180.0), 180.0);
        assert_eq!(wrap_deg(720.0), 0.0);
        assert_eq!(wrap_deg(-540.0), 180.0);
        assert!((wrap_deg(720.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ear_angles_cover_both_sides() {
        assert_eq!(ear_angle(90.0, 1.0), 0.0);
        assert_eq!(ear_angle(90.0, -1.0), 180.0);
        assert_eq!(ear_angle(0.0, 1.0), 90.0);
        assert_eq!(ear_angle(-30.0, -1.0), 60.0);
        assert_eq!(ear_angle(180.0, 1.0), 90.0);
    }

    #[test]
    fn head_shadow_alpha_follows_brown_duda() {
        assert!((head_shadow_alpha(0.0) - 2.0).abs() < 1e-12);
        assert!((head_shadow_alpha(150.0) - ALPHA_MIN).abs() < 1e-12);
        assert!((head_shadow_alpha(90.0) - (1.05 + 0.95 * (0.6 * PI).cos())).abs() < 1e-12);
        let a180 = head_shadow_alpha(180.0);
        assert!(
            a180 > ALPHA_MIN && a180 < 0.5,
            "slight recovery behind the head: {a180}"
        );
    }

    #[test]
    fn round3_is_pythons_round() {
        // values whose decimal expansion is an exact tie: Python rounds half to even
        assert_eq!(round3(0.0625), 0.062);
        assert_eq!(round3(0.1875), 0.188);
        assert_eq!(round3(1.0625), 1.062);
        assert_eq!(round3(1.3125), 1.312);
        assert_eq!(round3(2.0), 2.0);
        assert_eq!(round3(0.1004999), 0.1);
        assert_eq!(round3(1.2345678), 1.235);
        // 1.0005 is really 1.000499999...: Python gives 1.0, naive x*1000 rounding gives 1.001
        assert_eq!(round3(1.0005), 1.0);
    }

    #[test]
    fn shadow_filter_is_unity_at_dc_and_alpha_at_nyquist() {
        for &rate in &[16000u32, 44100, 48000] {
            for &alpha in &[0.1, 0.5, 1.0, 1.5, 2.0] {
                let f = OnePole::shadow(alpha, rate);
                let dc = (f.b0 + f.b1) / (1.0 + f.a1);
                let ny = (f.b0 - f.b1) / (1.0 - f.a1);
                assert!((dc - 1.0).abs() < 1e-12, "dc {dc}");
                assert!((ny - alpha).abs() < 1e-12, "nyquist {ny} vs {alpha}");
                assert!(f.a1.abs() < 1.0, "stable");
            }
        }
        let id = OnePole::shadow(1.0, 48000);
        assert!(
            (id.b0 - 1.0).abs() < 1e-15 && (id.b1 - id.a1).abs() < 1e-15,
            "alpha 1 is a wire"
        );
    }

    #[test]
    fn shadow_coefficients_match_the_bilinear_formula() {
        let (beta, k) = (2.0 * 343.0 / 0.0875, 96000.0);
        let f = OnePole::shadow(2.0, 48000);
        assert_eq!(f.b0, (beta + 2.0 * k) / (beta + k));
        assert_eq!(f.b1, (beta - 2.0 * k) / (beta + k));
        assert_eq!(f.a1, (beta - k) / (beta + k));
        let lp = OnePole::behind(48000);
        let x = (-2.0 * PI * 2500.0 / 48000.0f64).exp();
        assert_eq!((lp.b0, lp.b1, lp.a1), (1.0 - x, 0.0, -x));
    }

    #[test]
    fn one_pole_impulse_response() {
        let f = OnePole {
            b0: 0.5,
            b1: 0.25,
            a1: -0.5,
        };
        let mut y = vec![0.0; 4];
        f.run(&[1.0, 0.0, 0.0, 0.0], |i, v| y[i] = v);
        // y0 = .5, y1 = .25 + .5*.5 = .5, y2 = .25, y3 = .125
        assert_eq!(y, vec![0.5, 0.5, 0.25, 0.125]);
    }

    #[test]
    fn ear_delay_is_the_woodworth_formula() {
        assert_eq!(ear_delay_s(0.0), 0.0);
        assert!((ear_delay_s(90.0) - A_C).abs() < 1e-15);
        assert!((ear_delay_s(180.0) - A_C * (PI / 2.0 + 1.0)).abs() < 1e-15);
        // the near/far ear pair reproduces Woodworth's ITD = (a/c)(th + sin th)
        for az in [0.0f64, 10.0, 30.0, 45.0, 60.0, 89.0, 90.0] {
            let itd = ear_delay_s(ear_angle(az, -1.0)) - ear_delay_s(ear_angle(az, 1.0));
            let th = az.to_radians();
            assert!(
                (itd - A_C * (th + th.sin())).abs() < 1e-12,
                "az {az}: {itd}"
            );
        }
        // and a source behind mirrors the one in front
        let f = ear_delay_s(ear_angle(30.0, -1.0)) - ear_delay_s(ear_angle(30.0, 1.0));
        let b = ear_delay_s(ear_angle(150.0, -1.0)) - ear_delay_s(ear_angle(150.0, 1.0));
        assert!((f - b).abs() < 1e-12);
    }

    #[test]
    fn output_lengths_match_python_and_fit_the_sinc_tail() {
        assert_eq!(pad_samples(48000), 27);
        for rate in [16000u32, 44100, 48000, 96000] {
            // the sinc path holds every sample of the far ear's delayed tail
            let worst = (ear_delay_s(180.0) * rate as f64).floor() as usize;
            let n = 100;
            assert!(
                out_frames(n, rate, Delay::Sinc(16)) > n - 1 + worst + 16,
                "rate {rate}"
            );
            assert_eq!(out_frames(n, rate, Delay::Linear), n + pad_samples(rate));
        }
    }

    #[test]
    fn render_preserves_energy_and_length() {
        let x = noise(4800, 1);
        let e0 = energy(&x);
        for az in (-180..=180).step_by(15) {
            let st = render(&x, az as f64, 48000, 1.0, 6.0, Delay::Linear);
            assert_eq!(st.len(), (4800 + 27) * 2);
            assert!((energy(&st) / e0 - 2.0).abs() < 1e-4, "az {az}");
        }
    }

    #[test]
    fn render_is_mirror_symmetric() {
        let x = noise(2000, 2);
        for az in [10.0, 45.0, 90.0, 120.0, 170.0] {
            let r = render(&x, az, 48000, 1.0, 6.0, Delay::Linear);
            let l = render(&x, -az, 48000, 1.0, 6.0, Delay::Linear);
            for f in 0..r.len() / 2 {
                assert!((r[2 * f] - l[2 * f + 1]).abs() < 1e-6);
                assert!((r[2 * f + 1] - l[2 * f]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn empty_and_silent_inputs_give_silent_padding() {
        let st = render(&[], 45.0, 48000, 1.0, 6.0, Delay::Linear);
        assert_eq!(st.len(), 27 * 2);
        assert!(st.iter().all(|&v| v == 0.0));
        let st = render(&[0.0; 100], 135.0, 48000, 1.0, 6.0, Delay::Sinc(16));
        assert_eq!(st.len(), out_frames(100, 48000, Delay::Sinc(16)) * 2);
        assert!(st.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn batch_equals_single_renders_bit_for_bit() {
        let x = noise(3360, 3);
        let azs: Vec<f64> = (-180..=180).step_by(5).map(|a| a as f64).collect();
        for mode in [Delay::Linear, Delay::Sinc(16)] {
            let batch = render_batch(&x, &azs, 48000, 0.7, 6.0, mode);
            assert_eq!(batch.len(), azs.len());
            for (b, &az) in batch.iter().zip(&azs) {
                assert_eq!(*b, render(&x, az, 48000, 0.7, 6.0, mode), "az {az}");
            }
        }
    }

    #[test]
    fn linear_delay_interpolates_between_samples() {
        let y = fractional_delay(&[1.0], 2.25, 6, Delay::Linear);
        assert_eq!(y, vec![0.0, 0.0, 0.75, 0.25, 0.0, 0.0]);
        // too short for the delay: silence, like Python
        assert!(fractional_delay(&[1.0, 1.0], 5.0, 4, Delay::Linear)
            .iter()
            .all(|&v| v == 0.0));
    }

    #[test]
    fn sinc_weights_are_normalised_and_exact_on_integers() {
        for taps in [4usize, 8, 16, 32] {
            for frac in [0.0, 0.1, 0.5, 0.9] {
                let w = sinc_weights(frac, taps);
                assert_eq!(w.len(), taps + 1);
                assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            }
            let w = sinc_weights(0.0, taps);
            for (j, &v) in w.iter().enumerate() {
                assert_eq!(
                    v,
                    if j == taps / 2 { 1.0 } else { 0.0 },
                    "taps {taps} j {j}"
                );
            }
        }
        // integer delay in sinc mode is a pure shift by delay + taps/2
        let y = fractional_delay(&[1.0, -0.5], 3.0, 30, Delay::Sinc(16));
        assert_eq!(y[11], 1.0);
        assert_eq!(y[12], -0.5);
        assert_eq!(y.iter().filter(|v| **v != 0.0).count(), 2);
    }

    #[test]
    fn sinc_delay_is_flat_where_linear_dulls() {
        // magnitude at 0.4 fs of a half-sample delay: linear is cos(0.4 pi) ~ -10 dB
        let gain_at = |w: &[f64], f: f64| {
            let (mut re, mut im) = (0.0, 0.0);
            for (j, &v) in w.iter().enumerate() {
                re += v * (2.0 * PI * f * j as f64).cos();
                im -= v * (2.0 * PI * f * j as f64).sin();
            }
            (re * re + im * im).sqrt()
        };
        let lin = gain_at(&[0.5, 0.5], 0.4);
        let sinc = gain_at(&sinc_weights(0.5, 16), 0.4);
        assert!(20.0 * lin.log10() < -9.0, "linear {lin}");
        assert!((20.0 * sinc.log10()).abs() < 1.0, "sinc {sinc}");
    }

    #[test]
    fn taps_are_validated() {
        assert_eq!(Delay::from_taps(0), Ok(Delay::Linear));
        assert_eq!(Delay::from_taps(16), Ok(Delay::Sinc(16)));
        assert!(Delay::from_taps(15).is_err());
        assert!(Delay::from_taps(66).is_err());
    }

    #[test]
    fn pairwise_sum_is_numpys_summation_order() {
        // np.sum((np.arange(n) * 0.001 + 0.1) ** 2); naive left-to-right differs for n >= 100
        let cases = [
            (5usize, 0.05203000000000001),
            (100, 2.31835),
            (1000, 442.73349999999994),
            (6774, 108245.845399),
        ];
        for (n, want) in cases {
            let a: Vec<f64> = (0..n).map(|i| i as f64 * 0.001 + 0.1).collect();
            assert_eq!(pairwise_sum(&a, |v| v * v), want, "n {n}");
        }
        assert_eq!(pairwise_sum(&[], |v| v), 0.0);
    }

    #[test]
    fn bessel_i0_reference_values() {
        assert_eq!(bessel_i0(0.0), 1.0);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-14);
        assert!((bessel_i0(6.0) - 67.234_406_976_477_97).abs() < 1e-10);
    }
}
