//! Proximity sonar decision core. A line-for-line port of the decision logic in
//! `earpiece/sonar.py`: per-object tracks with bearing smoothing and a least-squares
//! closing-speed estimate, corridor weighting, time-to-contact and distance urgency
//! with hysteresis, the critical decision, the two-source cap, beep scheduling in
//! engine samples, and the "Stop." / "person, left" speech decisions.
//!
//! It decides *what* to sound and *when*; it does not render audio. `tick` returns
//! an ordered list of [`Cue`]s (beeps at absolute engine frames, utterances) that the
//! caller plays exactly where `Sonar.tick` in Python would have called
//! `engine.play` / `speak`.
//!
//! Parity with Python is kept down to the arithmetic, so the Python test suite can
//! assert identical cue streams:
//!   * every product and sum is evaluated in the same order as the Python source;
//!   * `round()` is round-half-to-even, like Python's;
//!   * `min` / `max` follow Python's builtins (first argument wins ties), so even
//!     the sign of a zero matches;
//!   * the closing-speed fit sums with numpy's pairwise order (`np.sum`);
//!   * tracks keep dict insertion order and the ranking sort is stable, like
//!     `sorted` over `dict.values()`.

use crate::spatial::{pairwise_sum, wrap_deg};
use std::collections::{HashMap, VecDeque};

/// Matches `deque(maxlen=12)` in `Track.hist`.
const HIST_LEN: usize = 12;

/// The tunables of `config.SonarCfg` that the decision core reads.
#[derive(Clone, Debug, PartialEq)]
pub struct SonarCfg {
    pub walk_speed_mps: f64,
    pub corridor_half_width_m: f64,
    pub corridor_margin_m: f64,
    pub max_range_m: f64,
    pub ttc_start_s: f64,
    pub ttc_critical_s: f64,
    pub critical_dist_m: f64,
    pub hysteresis: f64,
    pub interval_slow_s: f64,
    pub interval_fast_s: f64,
    pub pitch_low_hz: f64,
    pub pitch_high_hz: f64,
    pub max_sources: usize,
    pub second_source_gain: f64,
    pub stale_track_s: f64,
    pub speak_label_cooldown_s: f64,
    pub speak_global_gap_s: f64,
}

impl Default for SonarCfg {
    /// The defaults of `config.SonarCfg`.
    fn default() -> Self {
        SonarCfg {
            walk_speed_mps: 1.2,
            corridor_half_width_m: 0.55,
            corridor_margin_m: 0.35,
            max_range_m: 6.0,
            ttc_start_s: 3.5,
            ttc_critical_s: 1.0,
            critical_dist_m: 0.75,
            hysteresis: 0.15,
            interval_slow_s: 1.0,
            interval_fast_s: 0.11,
            pitch_low_hz: 520.0,
            pitch_high_hz: 1250.0,
            max_sources: 2,
            second_source_gain: 0.55,
            stale_track_s: 0.6,
            speak_label_cooldown_s: 6.0,
            speak_global_gap_s: 2.5,
        }
    }
}

/// Urgency for a sender-ranked object, as `sonar.PRIORITY_URGENCY`: P1 ~1228 Hz every ~0.14 s,
/// P2 ~812 Hz every ~0.57 s, P3 ~630 Hz every ~0.81 s with the default pitch and interval range.
pub fn priority_urgency(p: u8) -> f64 {
    match p {
        1 => 0.97,
        2 => 0.40,
        _ => 0.15,
    }
}

/// How a sender-ranked object sounds, as `sonar.PRIORITY_TONE`: (pitch Hz, seconds between beeps, gain).
pub fn priority_tone(p: u8) -> (f64, f64, f64) {
    match p {
        1 => (1250.0, 0.12, 1.0),
        2 => (875.0, 0.40, 0.9),
        _ => (600.0, 0.90, 0.8),
    }
}

/// One detected object, as `events.Obstacle`.
#[derive(Clone, Debug, PartialEq)]
pub struct Obstacle {
    pub id: String,
    pub kind: String,
    pub bearing: f64,
    pub distance: f64,
    pub closing_speed: Option<f64>,
    pub confidence: f64,
    /// The sender's priority, 1 (most urgent) to 3, when it ranks objects itself (perception/near).
    pub priority: Option<u8>,
}

/// One tracked object, as `sonar.Track`.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub id: String,
    pub kind: String,
    pub source: String,
    pub bearing: f64,
    pub distance: f64,
    pub confidence: f64,
    pub given_speed: Option<f64>,
    pub est_speed: Option<f64>,
    pub last_seen: f64,
    pub ttl: Option<f64>,
    pub hist: VecDeque<(f64, f64)>,
    pub urgency: f64,
    pub ttc: f64,
    pub active: bool,
    pub critical: bool,
    /// Engine frame of the next scheduled beep.
    pub next_beep: Option<i64>,
    /// Engine frame of the last scheduled beep.
    pub last_beep: Option<i64>,
    pub last_label_t: f64,
    pub last_stop_t: f64,
    pub first_seen: f64,
    pub priority: Option<u8>,
}

impl Track {
    /// The sender's closing speed when it supplied one, else the estimate.
    pub fn closing_speed(&self) -> Option<f64> {
        self.given_speed.or(self.est_speed)
    }
}

/// A beep to schedule: `engine.play(cue(kind, pitch_q, bearing_q), at=at,
/// bus="alert" if critical else "cue", gain=gain, tag=f"sonar:{track}")`.
#[derive(Clone, Debug, PartialEq)]
pub struct Beep {
    pub at: i64,
    pub kind: String,
    pub pitch_q: i64,
    pub bearing_q: i64,
    pub critical: bool,
    pub gain: f64,
    pub track: String,
}

/// A line to speak: `speak(text, priority, key=key, ttl=ttl)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Utterance {
    pub text: String,
    pub priority: &'static str,
    pub key: String,
    pub ttl: f64,
}

/// What one tick asks the caller to do, in the order Python would do it.
#[derive(Clone, Debug, PartialEq)]
pub enum Cue {
    Beep(Beep),
    Say(Utterance),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub frames: u64,
    pub objects: u64,
    pub out_of_order: u64,
    pub beeps: u64,
    pub criticals: u64,
}

/// Python's `max(a, b)`: `b` only if it is strictly greater (ties keep `a`).
#[inline]
fn pmax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's `min(a, b)`: `b` only if it is strictly smaller (ties keep `a`).
#[inline]
fn pmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python's `round(x)` on a float: nearest integer, ties to even.
#[inline]
fn pround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// `spatial.distance_gain` with its defaults (ref 1 m, floor 0.35).
pub fn distance_gain(distance_m: f64) -> f64 {
    if !distance_m.is_finite() {
        return 1.0;
    }
    pmax(0.35, pmin(1.0, (1.0 / pmax(distance_m, 0.1)).powf(0.5)))
}

/// `events.clock_phrase`: bearing -> spoken direction.
pub fn clock_phrase(bearing: f64) -> String {
    let b = wrap_deg(bearing);
    let a = b.abs();
    if a <= 12.0 {
        return "ahead".into();
    }
    if a >= 160.0 {
        return "behind you".into();
    }
    let side = if b > 0.0 { "right" } else { "left" };
    if a <= 35.0 {
        return format!("slightly {side}");
    }
    if (75.0..=105.0).contains(&a) {
        return format!("on your {side}");
    }
    let hour = match pround(b / 30.0).rem_euclid(12) {
        0 => 12,
        h => h,
    };
    format!("at {hour} o'clock")
}

/// `sonar._SPOKEN_KIND`, defaulting to "Obstacle".
pub fn spoken_kind(kind: &str) -> &'static str {
    match kind {
        "person" => "Person",
        "vehicle" => "Car",
        "bike" => "Bike",
        "curb" => "Curb",
        "stairs_down" => "Step down",
        _ => "Obstacle",
    }
}

/// Least-squares slope of distance over the last ~0.8 s: positive = approaching.
/// `Sonar._estimate_speed`, with numpy's summation order.
pub fn estimate_speed(hist: &VecDeque<(f64, f64)>) -> Option<f64> {
    let last_t = hist.back()?.0;
    let pts: Vec<(f64, f64)> = hist
        .iter()
        .copied()
        .filter(|&(tt, _)| last_t - tt <= 0.8)
        .collect();
    if pts.len() < 3 || pts[pts.len() - 1].0 - pts[0].0 < 0.2 {
        return None;
    }
    let n = pts.len() as f64;
    let mut ts: Vec<f64> = pts.iter().map(|p| p.0).collect();
    let ds: Vec<f64> = pts.iter().map(|p| p.1).collect();
    let t_mean = pairwise_sum(&ts, |x| x) / n;
    for t in ts.iter_mut() {
        *t -= t_mean;
    }
    let tt: Vec<f64> = ts.iter().map(|t| t * t).collect();
    let den = pairwise_sum(&tt, |x| x);
    if den <= 0.0 {
        return None;
    }
    let d_mean = pairwise_sum(&ds, |x| x) / n;
    let prod: Vec<f64> = ts.iter().zip(&ds).map(|(t, d)| t * (d - d_mean)).collect();
    let slope = pairwise_sum(&prod, |x| x) / den;
    Some(pmax(-10.0, pmin(15.0, -slope)))
}

/// The sonar's decision state. Not thread-aware; the Python binding wraps it in a mutex.
#[derive(Clone, Debug)]
pub struct Sonar {
    pub cfg: SonarCfg,
    pub rate: u32,
    /// In insertion order, like the Python dict.
    pub tracks: Vec<Track>,
    /// The watchdog switches this off when perception is stale.
    pub enabled: bool,
    /// The wearer asked for quiet (critical still sounds).
    pub muted: bool,
    /// Speak "person, left" on new urgent objects.
    pub labels: bool,
    /// False = no speak callback: no speech decisions at all (Python's `speak=None`).
    pub speak: bool,
    pub stats: Stats,
    last_global_label: f64,
    last_seq: HashMap<String, i64>,
    last_seq_t: HashMap<String, f64>,
}

impl Sonar {
    pub fn new(cfg: SonarCfg, rate: u32) -> Self {
        Sonar {
            cfg,
            rate,
            tracks: Vec::new(),
            enabled: true,
            muted: false,
            labels: true,
            speak: true,
            stats: Stats::default(),
            last_global_label: -1e9,
            last_seq: HashMap::new(),
            last_seq_t: HashMap::new(),
        }
    }

    /// `sonar.critical_pitch_q(cfg)`: the quantised pitch of every critical beep.
    pub fn critical_pitch_q(&self) -> i64 {
        let c = &self.cfg;
        pround((c.pitch_low_hz + (c.pitch_high_hz - c.pitch_low_hz) * 1.0) / 25.0) * 25
    }

    /// `Engine.samples`: seconds -> frames, `int(round(seconds * rate))`.
    pub fn samples(&self, seconds: f64) -> i64 {
        pround(seconds * self.rate as f64)
    }

    pub fn track(&self, id: &str) -> Option<&Track> {
        self.tracks.iter().find(|t| t.id == id)
    }

    // ------------------------------------------------------------------ input
    /// A full perception frame from `source`: ids missing from it are gone.
    /// Returns false when the frame was dropped as a late (out-of-order) datagram.
    pub fn update_frame(
        &mut self,
        source: &str,
        objects: &[Obstacle],
        seq: Option<i64>,
        now: f64,
    ) -> bool {
        if let Some(seq) = seq {
            let last = self.last_seq.get(source).copied();
            let last_t = self.last_seq_t.get(source).copied().unwrap_or(-1e9);
            // A late UDP datagram (a few frames old) must never overwrite newer data. A big
            // backward jump, or any seq after a pause, means the sender restarted: accept it.
            if let Some(last) = last {
                let (s, l) = (seq as i128, last as i128);
                if s <= l && l - s <= 5 && now - last_t < 0.5 {
                    self.stats.out_of_order += 1;
                    return false;
                }
            }
            self.last_seq.insert(source.to_string(), seq);
            self.last_seq_t.insert(source.to_string(), now);
        }
        self.stats.frames += 1;
        self.stats.objects += objects.len() as u64;
        let mut seen: Vec<String> = Vec::with_capacity(objects.len());
        for o in objects {
            let key = format!("{source}:{}", o.id);
            self.upsert(&key, source, o, now, None);
            seen.push(key);
        }
        for t in self.tracks.iter_mut() {
            if t.source == source && t.ttl.is_none() && !seen.contains(&t.id) {
                // gone from this frame: let it fade over stale_track_s instead of
                // vanishing (detectors drop objects for a frame all the time)
                t.last_seen = pmin(t.last_seen, now - 1e-3);
            }
        }
        true
    }

    /// A single object with its own time-to-live (`ttl` None or 0 = 0.5 s).
    pub fn update_one(&mut self, source: &str, o: &Obstacle, ttl: Option<f64>, now: f64) {
        let ttl = match ttl {
            Some(x) if x != 0.0 => x,
            _ => 0.5,
        };
        let key = format!("{source}:{}", o.id);
        self.upsert(&key, source, o, now, Some(ttl));
    }

    fn upsert(&mut self, key: &str, source: &str, o: &Obstacle, now: f64, ttl: Option<f64>) {
        let i = match self.tracks.iter().position(|t| t.id == key) {
            Some(i) => {
                let t = &mut self.tracks[i];
                // wrap-aware bearing smoothing; distance mostly trusted (it is what matters)
                t.bearing = wrap_deg(t.bearing + 0.6 * wrap_deg(o.bearing - t.bearing));
                t.distance = 0.3 * t.distance + 0.7 * o.distance;
                t.kind.clone_from(&o.kind);
                t.confidence = o.confidence;
                t.given_speed = o.closing_speed;
                t.last_seen = now;
                t.ttl = ttl;
                t.priority = o.priority;
                i
            }
            None => {
                self.tracks.push(Track {
                    id: key.to_string(),
                    kind: o.kind.clone(),
                    source: source.to_string(),
                    bearing: o.bearing,
                    distance: o.distance,
                    confidence: o.confidence,
                    given_speed: o.closing_speed,
                    est_speed: None,
                    last_seen: now,
                    ttl,
                    hist: VecDeque::with_capacity(HIST_LEN),
                    urgency: 0.0,
                    ttc: f64::INFINITY,
                    active: false,
                    critical: false,
                    next_beep: None,
                    last_beep: None,
                    last_label_t: -1e9,
                    last_stop_t: -1e9,
                    first_seen: now,
                    priority: o.priority,
                });
                self.tracks.len() - 1
            }
        };
        let t = &mut self.tracks[i];
        if t.hist.len() == HIST_LEN {
            t.hist.pop_front();
        }
        t.hist.push_back((now, o.distance));
        t.est_speed = estimate_speed(&t.hist);
    }

    /// Forget every track (or those of one source).
    pub fn clear(&mut self, source: Option<&str>) {
        self.tracks
            .retain(|t| source.is_some_and(|s| t.source != s));
    }

    // ------------------------------------------------------------------ urgency
    fn score(cfg: &SonarCfg, stats: &mut Stats, t: &mut Track) {
        let c = cfg;
        let b = t.bearing.to_radians();
        let mut fwd = t.distance * b.cos();
        // Python calls libm's sin and cos separately. Given both of the same argument,
        // LLVM fuses them into one sincos call (`__sincos_stret` on macOS), which can
        // differ in the last bit, and on the corridor's soft edge that bit is amplified
        // into the urgency. The opaque copy keeps the two calls separate.
        let lat = (t.distance * std::hint::black_box(b).sin()).abs();
        let half = c.corridor_half_width_m + if t.distance < 1.5 { 0.25 } else { 0.0 }; // close = wider
        let mut corridor = if lat <= half {
            1.0
        } else {
            pmax(0.0, 1.0 - (lat - half) / c.corridor_margin_m)
        };
        if fwd <= 0.0 {
            // beside or behind: only matters when very close (someone right next to you)
            corridor = if t.distance < 0.8 { 1.0 } else { 0.0 };
            fwd = t.distance;
        }
        if t.priority.is_some() {
            corridor = 1.0; // the sender already put it in the walking corridor
        }
        let v = t.closing_speed();
        let approach = match v {
            None => c.walk_speed_mps,
            Some(v) => pmax(v, 0.0),
        };
        let approach = pmax(approach, 0.25);
        let ttc = fwd / approach;
        let warn_range = pmin(c.max_range_m, c.walk_speed_mps * c.ttc_start_s + 0.5);
        let span_t = pmax(1e-3, c.ttc_start_s - c.ttc_critical_s);
        let u_ttc = pmin(1.0, pmax(0.0, (c.ttc_start_s - ttc) / span_t));
        let u_dist = pmin(
            1.0,
            pmax(
                0.0,
                (warn_range - t.distance) / pmax(1e-3, warn_range - c.critical_dist_m),
            ),
        );
        let conf = 0.5 + 0.5 * t.confidence;
        let mut u = pmax(u_ttc, 0.85 * u_dist) * corridor * conf;
        if t.kind == "vehicle" {
            u = pmin(1.0, u * 1.25);
        }
        if let Some(p) = t.priority {
            u = priority_urgency(p); // one fixed tone and rate per priority, whatever the kind or bearing
        }
        let h = if t.critical { 1.0 - c.hysteresis } else { 1.0 };
        let critical = corridor > 0.5
            && (t.distance <= c.critical_dist_m / h
                || (v.is_some_and(|v| v > 0.3) && ttc <= c.ttc_critical_s / h));
        let enter = 0.08;
        let exit = 0.08 * (1.0 - c.hysteresis) * 0.5;
        t.active = u > enter || (t.active && u > exit);
        t.urgency = u;
        t.ttc = ttc;
        if critical && !t.critical {
            stats.criticals += 1;
        }
        t.critical = critical;
    }

    // ------------------------------------------------------------------ tick
    /// Call ~50 Hz with the wall clock `now` and the engine clock `engine_now`
    /// (frames). Updates urgency, prunes stale tracks, schedules beeps. Returns the
    /// indices (into `tracks`) of the sonified tracks and the cues to play, in order.
    pub fn tick(&mut self, now: f64, engine_now: i64) -> (Vec<usize>, Vec<Cue>) {
        let stale = self.cfg.stale_track_s;
        self.tracks.retain(|t| {
            let age = now - t.last_seen;
            let limit = t.ttl.unwrap_or(stale);
            age <= limit
        });
        for t in self.tracks.iter_mut() {
            Self::score(&self.cfg, &mut self.stats, t);
        }
        let mut cues = Vec::new();
        if !self.enabled {
            for t in self.tracks.iter_mut() {
                t.next_beep = None;
            }
            return (Vec::new(), cues);
        }
        let mut ranked: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| self.tracks[i].active)
            .collect();
        // sorted(key=(not critical, -urgency)): stable, critical first, then most urgent
        ranked.sort_by(|&a, &b| {
            let (ta, tb) = (&self.tracks[a], &self.tracks[b]);
            (!ta.critical).cmp(&!tb.critical).then_with(|| {
                (-ta.urgency)
                    .partial_cmp(&-tb.urgency)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        ranked.truncate(self.cfg.max_sources);
        let chosen = ranked;
        for (i, t) in self.tracks.iter_mut().enumerate() {
            if !chosen.contains(&i) {
                t.next_beep = None;
            }
        }
        let lookahead = engine_now + self.samples(0.06);
        let start_gap = self.samples(0.02);
        for (rank, &i) in chosen.iter().enumerate() {
            let c = &self.cfg;
            if self.muted && !self.tracks[i].critical {
                self.tracks[i].next_beep = None;
                continue;
            }
            let t = &self.tracks[i];
            let u = if t.critical { 1.0 } else { t.urgency };
            let mut interval =
                c.interval_slow_s - (c.interval_slow_s - c.interval_fast_s) * u.powf(0.8);
            let mut pitch = c.pitch_low_hz + (c.pitch_high_hz - c.pitch_low_hz) * u;
            let mut level = distance_gain(t.distance);
            if let (Some(p), false) = (t.priority, t.critical) {
                (pitch, interval, level) = priority_tone(p);
            }
            let pitch_q = pround(pitch / 25.0) * 25;
            let bearing_q = pround(t.bearing / 5.0) * 5;
            let fade = pmin(1.0, pmax(0.0, 1.0 - (now - t.last_seen) / c.stale_track_s));
            let gain = if rank == 0 { 1.0 } else { c.second_source_gain } * level * fade;
            let step = self.samples(interval);
            let t = &mut self.tracks[i];
            match (t.next_beep, t.last_beep) {
                (None, last) => {
                    let mut nb = engine_now + start_gap;
                    if let Some(lb) = last {
                        // resuming: keep at least one gap after the last beep
                        nb = nb.max(lb + step);
                    }
                    t.next_beep = Some(nb);
                }
                (Some(nb), Some(lb)) if nb > lb + step => {
                    // speeding up: pull the beat in
                    t.next_beep = Some(engine_now.max(lb + step));
                }
                _ => {}
            }
            while let Some(nb) = t.next_beep.filter(|&nb| nb <= lookahead) {
                cues.push(Cue::Beep(Beep {
                    at: nb,
                    kind: t.kind.clone(),
                    pitch_q,
                    bearing_q,
                    critical: t.critical,
                    gain,
                    track: t.id.clone(),
                }));
                self.stats.beeps += 1;
                t.last_beep = Some(nb);
                t.next_beep = Some(nb + step);
            }
            if let Some(u) = self.maybe_speak(i, now) {
                cues.push(Cue::Say(u));
            }
        }
        (chosen, cues)
    }

    fn maybe_speak(&mut self, i: usize, now: f64) -> Option<Utterance> {
        if !self.speak {
            return None;
        }
        let c = &self.cfg;
        let t = &mut self.tracks[i];
        let name = spoken_kind(&t.kind);
        if t.critical && now - t.last_stop_t > 4.0 {
            t.last_stop_t = now;
            t.last_label_t = now;
            self.last_global_label = now;
            let ahead = t.bearing.abs() <= 25.0;
            let text = if ahead && (name == "Person" || name == "Obstacle") {
                format!("Stop. {name} ahead.")
            } else if name == "Car" {
                "Stop. Car.".into()
            } else {
                "Stop.".into()
            };
            return Some(Utterance {
                text,
                priority: "critical",
                key: format!("stop:{}", t.id),
                ttl: 1.0,
            });
        }
        if !self.labels || self.muted {
            return None;
        }
        // Label early (urgency 0.3) so the words finish before any "Stop"; skip the label
        // entirely when contact is so near that "Stop" would cut it off mid-word.
        let imminent = t
            .closing_speed()
            .is_some_and(|v| v > 0.3 && t.ttc < c.ttc_critical_s + 1.2);
        if t.urgency >= 0.3
            && !imminent
            && now - t.last_label_t > c.speak_label_cooldown_s
            && now - self.last_global_label > c.speak_global_gap_s
        {
            t.last_label_t = now;
            self.last_global_label = now;
            return Some(Utterance {
                text: format!("{name}, {}.", clock_phrase(t.bearing)),
                priority: "hazard",
                key: format!("label:{}", t.id),
                ttl: 1.5,
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48000;

    fn ob(id: &str, kind: &str, bearing: f64, distance: f64, v: Option<f64>) -> Obstacle {
        Obstacle {
            id: id.into(),
            kind: kind.into(),
            bearing,
            distance,
            closing_speed: v,
            confidence: 1.0,
            priority: None,
        }
    }

    fn beeps(cues: &[Cue]) -> Vec<&Beep> {
        cues.iter()
            .filter_map(|c| match c {
                Cue::Beep(b) => Some(b),
                _ => None,
            })
            .collect()
    }

    fn says(cues: &[Cue]) -> Vec<&Utterance> {
        cues.iter()
            .filter_map(|c| match c {
                Cue::Say(u) => Some(u),
                _ => None,
            })
            .collect()
    }

    /// Frames at `fps`, ticks every 20 ms, engine clock = wall clock in frames.
    fn run(
        s: &mut Sonar,
        t0: f64,
        seconds: f64,
        fps: f64,
        mut frame: impl FnMut(f64) -> Vec<Obstacle>,
    ) -> Vec<Cue> {
        let mut out = Vec::new();
        let (mut t, mut fr, mut seq) = (0.0, 0u32, 0i64);
        while t < seconds {
            if t * fps >= fr as f64 {
                fr += 1;
                seq += 1;
                s.update_frame("cam", &frame(t), Some(seq), t0 + t);
            }
            let now = t0 + t;
            out.extend(s.tick(now, (now * RATE as f64) as i64).1);
            t += 0.02;
        }
        out
    }

    #[test]
    fn python_rounding_and_minmax() {
        assert_eq!(pround(0.5), 0);
        assert_eq!(pround(1.5), 2);
        assert_eq!(pround(2.5), 2);
        assert_eq!(pround(-2.5), -2);
        assert_eq!(pround(-0.4), 0);
        assert!(pmax(-0.0, 0.0).is_sign_negative()); // max(-0.0, 0.0) keeps the first
        assert!(pmin(0.0, -0.0).is_sign_positive());
        assert_eq!(pmax(1.0, 2.0), 2.0);
        assert_eq!(pmin(1.0, 2.0), 1.0);
    }

    #[test]
    fn clock_phrases_match_events_py() {
        let cases = [
            (0.0, "ahead"),
            (10.0, "ahead"),
            (-20.0, "slightly left"),
            (30.0, "slightly right"),
            (90.0, "on your right"),
            (-85.0, "on your left"),
            (60.0, "at 2 o'clock"),
            (-60.0, "at 10 o'clock"),
            (130.0, "at 4 o'clock"),
            (175.0, "behind you"),
            (-45.0, "at 10 o'clock"), // round(-1.5) = -2 -> 10
            (45.0, "at 2 o'clock"),   // round(1.5) = 2
            (370.0, "ahead"),
        ];
        for (b, want) in cases {
            assert_eq!(clock_phrase(b), want, "bearing {b}");
        }
    }

    #[test]
    fn distance_gain_matches_spatial_py() {
        assert_eq!(distance_gain(1.0), 1.0);
        assert_eq!(distance_gain(0.05), 1.0);
        assert_eq!(distance_gain(100.0), 0.35);
        assert!((distance_gain(4.0) - 0.5).abs() < 1e-15);
        assert_eq!(distance_gain(f64::INFINITY), 1.0);
    }

    #[test]
    fn closing_speed_is_estimated() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        for i in 0..10 {
            let t = i as f64 / 15.0;
            s.update_frame(
                "cam",
                &[ob("p", "person", 0.0, 6.0 - 2.0 * t, None)],
                Some(i + 1),
                t,
            );
        }
        let v = s.track("cam:p").unwrap().est_speed.unwrap();
        assert!((v - 2.0).abs() < 1e-9, "{v}");
    }

    #[test]
    fn speed_needs_three_points_over_200ms() {
        let mut h = VecDeque::new();
        h.push_back((0.0, 5.0));
        h.push_back((0.1, 4.9));
        assert_eq!(estimate_speed(&h), None);
        h.push_back((0.15, 4.8));
        assert_eq!(estimate_speed(&h), None); // spans only 150 ms
        h.push_back((0.3, 4.6));
        assert!(estimate_speed(&h).unwrap() > 0.0);
        let flat: VecDeque<_> = [(1.0, 2.0); 4].into_iter().collect();
        assert_eq!(estimate_speed(&flat), None);
    }

    #[test]
    fn approaching_person_speeds_up_then_goes_critical() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        let cues = run(&mut s, 0.0, 4.5, 15.0, |t| {
            vec![ob("p", "person", 0.0, (5.0 - 1.2 * t).max(0.4), None)]
        });
        let b = beeps(&cues);
        assert!(b.len() > 10);
        let gaps: Vec<i64> = b.windows(2).map(|w| w[1].at - w[0].at).collect();
        assert!(gaps[0] as f64 / RATE as f64 > 0.4);
        assert!((*gaps.last().unwrap() as f64 / RATE as f64) < 0.13);
        assert!(b.iter().any(|x| x.critical));
        assert!(b.windows(2).all(|w| w[1].pitch_q >= w[0].pitch_q));
        let said = says(&cues);
        assert!(said
            .iter()
            .any(|u| u.text == "Stop. Person ahead." && u.priority == "critical"));
        assert!(s.stats.criticals >= 1);
    }

    #[test]
    fn beside_the_path_is_quiet_but_right_beside_you_is_not() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        let cues = run(&mut s, 0.0, 3.0, 15.0, |_| {
            vec![ob("p", "person", 60.0, 3.0, None)]
        });
        assert!(beeps(&cues).is_empty());
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        let cues = run(&mut s, 0.0, 2.0, 15.0, |_| {
            vec![ob("p", "person", 95.0, 0.6, None)]
        });
        assert!(!beeps(&cues).is_empty());
    }

    #[test]
    fn crowd_sonifies_at_most_two() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        let mut most = 0;
        let (mut t, mut seq) = (0.0, 0);
        while t < 2.0 {
            seq += 1;
            let objs: Vec<_> = (0..20)
                .map(|i| {
                    ob(
                        &format!("c{i}"),
                        "person",
                        -40.0 + i as f64 * 4.0,
                        1.2 + 0.1 * i as f64,
                        None,
                    )
                })
                .collect();
            s.update_frame("cam", &objs, Some(seq), t);
            let (chosen, _) = s.tick(t, (t * RATE as f64) as i64);
            most = most.max(chosen.len());
            t += 0.05;
        }
        assert_eq!(most, 2);
    }

    #[test]
    fn out_of_order_and_restart() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        assert!(s.update_frame("cam", &[ob("p", "person", 0.0, 1.0, None)], Some(10), 0.0));
        assert!(!s.update_frame("cam", &[ob("p", "person", 0.0, 5.0, None)], Some(9), 0.01));
        assert_eq!(s.track("cam:p").unwrap().distance, 1.0);
        assert_eq!(s.stats.out_of_order, 1);
        // a big backward jump is a restarted sender
        assert!(s.update_frame("cam", &[ob("p", "person", 0.0, 1.0, None)], Some(1), 0.05));
        // the same seq after a pause is accepted too
        assert!(s.update_frame("cam", &[ob("p", "person", 0.0, 3.0, None)], Some(1), 2.0));
        assert_eq!(s.stats.out_of_order, 1);
        // extreme seqs never overflow
        assert!(s.update_frame("cam", &[], Some(i64::MIN), 2.01));
        assert!(s.update_frame("cam", &[], Some(i64::MAX), 2.02));
    }

    #[test]
    fn stale_tracks_fade_and_go() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        let cues = run(&mut s, 0.0, 3.0, 15.0, |t| {
            if t < 1.0 {
                vec![ob("p", "pole", 0.0, 1.5, None)]
            } else {
                vec![]
            }
        });
        let last = beeps(&cues).last().unwrap().at as f64 / RATE as f64;
        assert!(last < 1.0 + 0.6 + 0.1);
        assert!(s.tracks.is_empty());
    }

    #[test]
    fn ttl_objects_live_for_their_ttl() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        s.update_one("drone", &ob("x", "curb", 0.0, 2.0, None), None, 0.0);
        assert_eq!(s.track("drone:x").unwrap().ttl, Some(0.5));
        s.update_one("drone", &ob("y", "curb", 0.0, 2.0, None), Some(2.0), 0.0);
        s.update_frame("drone", &[], None, 0.1); // a frame never fades ttl objects
        s.tick(1.0, 48000);
        assert!(s.track("drone:x").is_none());
        assert!(s.track("drone:y").is_some());
        s.tick(2.01, 96480);
        assert!(s.tracks.is_empty());
    }

    #[test]
    fn fast_car_is_critical_before_it_is_close() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        s.update_frame(
            "cam",
            &[ob("c", "vehicle", 0.0, 3.0, Some(4.0))],
            Some(1),
            0.0,
        );
        let (_, cues) = s.tick(0.0, 0);
        assert!(s.track("cam:c").unwrap().critical);
        let b = beeps(&cues);
        assert!(!b.is_empty() && b.iter().all(|x| x.critical && x.pitch_q == 1250));
        assert_eq!(says(&cues)[0].text, "Stop. Car.");
    }

    #[test]
    fn muted_still_warns_when_critical_and_disabled_is_silent() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        s.muted = true;
        let cues = run(&mut s, 0.0, 1.5, 15.0, |_| {
            vec![ob("p", "pole", 0.0, 3.0, None)]
        });
        assert!(cues.is_empty());
        let cues = run(&mut s, 1.5, 1.0, 15.0, |_| {
            vec![ob("p", "pole", 0.0, 0.5, None)]
        });
        assert!(!cues.is_empty() && beeps(&cues).iter().all(|b| b.critical));
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        s.enabled = false;
        let cues = run(&mut s, 0.0, 1.0, 15.0, |_| {
            vec![ob("p", "pole", 0.0, 0.5, None)]
        });
        assert!(cues.is_empty());
        assert!(s.tracks.iter().all(|t| t.next_beep.is_none()));
    }

    #[test]
    fn labels_are_rate_limited_and_optional() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        let cues = run(&mut s, 0.0, 4.0, 15.0, |_| {
            vec![
                ob("a", "person", -10.0, 2.2, None),
                ob("b", "pole", 10.0, 2.4, None),
            ]
        });
        let labels: Vec<_> = says(&cues)
            .into_iter()
            .filter(|u| !u.text.starts_with("Stop"))
            .collect();
        assert!((1..=2).contains(&labels.len()), "{labels:?}");
        assert_eq!(labels[0].text, "Person, ahead.");
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        s.speak = false;
        let cues = run(&mut s, 0.0, 2.0, 15.0, |_| {
            vec![ob("p", "person", 0.0, 0.5, None)]
        });
        assert!(says(&cues).is_empty() && !beeps(&cues).is_empty());
    }

    #[test]
    fn clear_by_source_keeps_the_others_in_order() {
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        s.update_frame("a", &[ob("1", "pole", 0.0, 2.0, None)], None, 0.0);
        s.update_frame("b", &[ob("2", "pole", 0.0, 2.0, None)], None, 0.0);
        s.update_frame("a", &[ob("3", "pole", 0.0, 2.0, None)], None, 0.0);
        s.clear(Some("a"));
        assert_eq!(s.tracks.len(), 1);
        assert_eq!(s.tracks[0].id, "b:2");
        s.clear(None);
        assert!(s.tracks.is_empty());
    }

    #[test]
    fn a_stall_catches_up_on_missed_beeps() {
        // Same as Python: the beat is kept, so a late tick schedules every missed beep.
        let mut s = Sonar::new(SonarCfg::default(), RATE);
        s.update_frame("cam", &[ob("p", "pole", 0.0, 0.5, None)], None, 0.0);
        let (_, first) = s.tick(0.0, 0);
        assert_eq!(beeps(&first).len(), 1);
        s.update_frame("cam", &[ob("p", "pole", 0.0, 0.5, None)], None, 0.3);
        let (_, late) = s.tick(0.3, 14400);
        let step = s.samples(0.11);
        let b = beeps(&late);
        assert!(b.len() >= 2);
        assert!(b.windows(2).all(|w| w[1].at - w[0].at == step));
    }
}
