//! Python binding of the sonar decision core (`kite_audio.Sonar`).
//!
//! The Python side (`earpiece/rust_sonar.py`) feeds frames in and, in the default
//! mode, plays the cues `tick` hands back; this file only converts. In native mode
//! (`start_native`) a Rust thread ticks instead and Python only `poll`s for speech
//! requests and watchdog transitions (see `sonar_thread`). The core sits behind a
//! mutex and the lock is never held while touching Python objects, so the
//! perception thread (frames), the control loop and the native thread cannot
//! deadlock against the GIL.

use crate::sonar::{Cue, Obstacle, Sonar, SonarCfg, Track};
use crate::sonar_thread::{self, Clock, Event, Shared};
use crate::sounds::Profile;
use crate::Engine;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashMap;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// `(id, kind, bearing, distance, closing_speed | None, confidence)`.
type ObTuple = (String, String, f64, f64, Option<f64>, f64);

fn obstacle(o: ObTuple) -> Obstacle {
    Obstacle {
        id: o.0,
        kind: o.1,
        bearing: o.2,
        distance: o.3,
        closing_speed: o.4,
        confidence: o.5,
    }
}

/// A read-only snapshot of one track, with the attributes of `sonar.Track`.
#[pyclass(frozen, get_all, name = "TrackState", module = "kite_audio")]
#[derive(Clone)]
pub struct TrackState {
    id: String,
    kind: String,
    source: String,
    bearing: f64,
    distance: f64,
    confidence: f64,
    given_speed: Option<f64>,
    est_speed: Option<f64>,
    last_seen: f64,
    ttl: Option<f64>,
    hist: Vec<(f64, f64)>,
    urgency: f64,
    ttc: f64,
    active: bool,
    critical: bool,
    next_beep: Option<i64>,
    last_beep: Option<i64>,
    last_label_t: f64,
    last_stop_t: f64,
    first_seen: f64,
}

impl From<&Track> for TrackState {
    fn from(t: &Track) -> Self {
        TrackState {
            id: t.id.clone(),
            kind: t.kind.clone(),
            source: t.source.clone(),
            bearing: t.bearing,
            distance: t.distance,
            confidence: t.confidence,
            given_speed: t.given_speed,
            est_speed: t.est_speed,
            last_seen: t.last_seen,
            ttl: t.ttl,
            hist: t.hist.iter().copied().collect(),
            urgency: t.urgency,
            ttc: t.ttc,
            active: t.active,
            critical: t.critical,
            next_beep: t.next_beep,
            last_beep: t.last_beep,
            last_label_t: t.last_label_t,
            last_stop_t: t.last_stop_t,
            first_seen: t.first_seen,
        }
    }
}

#[pymethods]
impl TrackState {
    /// The sender's closing speed when it supplied one, else the estimate.
    #[getter]
    fn closing_speed(&self) -> Option<f64> {
        self.given_speed.or(self.est_speed)
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackState(id={:?}, kind={:?}, bearing={:.1}, distance={:.2}, urgency={:.2}, critical={})",
            self.id, self.kind, self.bearing, self.distance, self.urgency, self.critical
        )
    }
}

fn cfg_from(obj: &Bound<'_, PyAny>) -> PyResult<SonarCfg> {
    let d = SonarCfg::default();
    let f = |name: &str, dflt: f64| -> PyResult<f64> {
        match obj.getattr(name) {
            Ok(v) => v.extract::<f64>(),
            Err(_) => Ok(dflt),
        }
    };
    let max_sources = match obj.getattr("max_sources") {
        Ok(v) => v.extract::<usize>()?,
        Err(_) => d.max_sources,
    };
    Ok(SonarCfg {
        walk_speed_mps: f("walk_speed_mps", d.walk_speed_mps)?,
        corridor_half_width_m: f("corridor_half_width_m", d.corridor_half_width_m)?,
        corridor_margin_m: f("corridor_margin_m", d.corridor_margin_m)?,
        max_range_m: f("max_range_m", d.max_range_m)?,
        ttc_start_s: f("ttc_start_s", d.ttc_start_s)?,
        ttc_critical_s: f("ttc_critical_s", d.ttc_critical_s)?,
        critical_dist_m: f("critical_dist_m", d.critical_dist_m)?,
        hysteresis: f("hysteresis", d.hysteresis)?,
        interval_slow_s: f("interval_slow_s", d.interval_slow_s)?,
        interval_fast_s: f("interval_fast_s", d.interval_fast_s)?,
        pitch_low_hz: f("pitch_low_hz", d.pitch_low_hz)?,
        pitch_high_hz: f("pitch_high_hz", d.pitch_high_hz)?,
        max_sources,
        second_source_gain: f("second_source_gain", d.second_source_gain)?,
        stale_track_s: f("stale_track_s", d.stale_track_s)?,
        speak_label_cooldown_s: f("speak_label_cooldown_s", d.speak_label_cooldown_s)?,
        speak_global_gap_s: f("speak_global_gap_s", d.speak_global_gap_s)?,
    })
}

/// The sonar decision core. `cfg` is any object with `SonarCfg`'s attributes
/// (missing ones take the defaults).
#[pyclass(frozen, name = "Sonar", module = "kite_audio")]
pub struct PySonar {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl PySonar {
    fn with<R>(&self, f: impl FnOnce(&mut Sonar) -> R) -> R {
        // A panic while holding the lock would poison it; the state is still
        // consistent field by field, so keep going rather than fail every call.
        f(&mut self.shared.core().sonar)
    }

    fn running(&self) -> bool {
        self.thread
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    fn stop_thread(&self) -> Option<JoinHandle<()>> {
        let h = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take();
        if h.is_some() {
            self.shared.stop.store(true, Relaxed);
        }
        h
    }
}

impl Drop for PySonar {
    fn drop(&mut self) {
        if let Some(h) = self.stop_thread() {
            let _ = h.join();
        }
    }
}

fn event_tuple<'py>(py: Python<'py>, e: Event) -> PyResult<Bound<'py, PyTuple>> {
    match e {
        Event::Say(u) => ("say", u.text, u.priority, u.key, u.ttl).into_pyobject(py),
        Event::Perception { ok, first, at } => ("perception", ok, first, at).into_pyobject(py),
    }
}

#[pymethods]
impl PySonar {
    #[new]
    #[pyo3(signature = (rate, cfg=None))]
    fn new(rate: u32, cfg: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if rate == 0 {
            return Err(PyValueError::new_err("rate must be positive"));
        }
        let cfg = match cfg {
            Some(c) => cfg_from(c)?,
            None => SonarCfg::default(),
        };
        Ok(PySonar {
            shared: Arc::new(Shared::new(Sonar::new(cfg, rate), 1.5)),
            thread: Mutex::new(None),
        })
    }

    // ---------------------------------------------------------------- native mode
    /// Tick on a native thread every `period_s`, synthesising, placing and scheduling
    /// every beep into `engine` (a `kite_audio.Engine` at 48 kHz) without the GIL.
    /// `py_now` is `time.monotonic()` now, so frames stamped by Python and the native
    /// clock agree. The stale-perception watchdog (`stale_s`) runs natively too.
    #[pyo3(signature = (engine, py_now, period_s=0.02, stale_s=1.5, profile="standard", prewarm=true))]
    fn start_native(
        &self,
        engine: PyRef<'_, Engine>,
        py_now: f64,
        period_s: f64,
        stale_s: f64,
        profile: &str,
        prewarm: bool,
    ) -> PyResult<()> {
        let link = engine.link();
        if link.rate != crate::sounds::RATE || link.rate != self.with(|s| s.rate) {
            return Err(PyValueError::new_err(format!(
                "the native sonar renders at {} Hz; engine and sonar must use that rate",
                crate::sounds::RATE
            )));
        }
        if !(period_s.is_finite() && period_s > 0.0 && period_s <= 0.05) {
            return Err(PyValueError::new_err(
                "period_s must be in (0, 0.05]: beeps are scheduled 60 ms ahead",
            ));
        }
        if !py_now.is_finite() || !stale_s.is_finite() {
            return Err(PyValueError::new_err("py_now and stale_s must be finite"));
        }
        let p = Profile::parse(profile)
            .ok_or_else(|| PyValueError::new_err(format!("unknown sound profile {profile:?}")))?;
        let mut slot = self.thread.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_some() {
            return Err(PyRuntimeError::new_err(
                "the native sonar thread is already running",
            ));
        }
        self.shared.set_profile(p);
        self.shared.core().watch.stale_s = stale_s;
        self.shared.stop.store(false, Relaxed);
        let (sh, clock) = (self.shared.clone(), Clock::new(py_now));
        let period = Duration::from_secs_f64(period_s);
        let h = std::thread::Builder::new()
            .name("kite-sonar".into())
            .spawn(move || sonar_thread::run(sh, link, clock, period, prewarm))
            .map_err(|e| PyRuntimeError::new_err(format!("spawn: {e}")))?;
        *slot = Some(h);
        Ok(())
    }

    /// Stop the native thread (no-op if it is not running).
    fn stop_native(&self, py: Python<'_>) {
        if let Some(h) = self.stop_thread() {
            py.detach(|| {
                let _ = h.join();
            });
        }
    }

    #[getter]
    fn native_running(&self) -> bool {
        self.running()
    }

    /// One native step at wall time `now`, on the calling thread: what the native
    /// thread does every period (tick, watchdog, render, schedule into `engine`).
    /// Deterministic, for tests; not allowed while the thread runs.
    fn native_step(&self, py: Python<'_>, engine: PyRef<'_, Engine>, now: f64) -> PyResult<()> {
        if self.running() {
            return Err(PyRuntimeError::new_err(
                "the native sonar thread is running",
            ));
        }
        let link = engine.link();
        if link.rate != crate::sounds::RATE {
            return Err(PyValueError::new_err("the native sonar renders at 48 kHz"));
        }
        let sh = self.shared.clone();
        py.detach(move || sh.step(&link, now));
        Ok(())
    }

    /// Native mode: the tracks sonified at the last native step, and every queued
    /// event since the last poll, in order: `("say", text, priority, key, ttl)` and
    /// `("perception", ok, first, at)`.
    fn poll<'py>(&self, py: Python<'py>) -> PyResult<(Vec<TrackState>, Vec<Bound<'py, PyTuple>>)> {
        let chosen: Vec<TrackState> = {
            let g = self.shared.core();
            g.chosen
                .iter()
                .filter_map(|id| g.sonar.tracks.iter().find(|t| &t.id == id))
                .map(TrackState::from)
                .collect()
        };
        let events = self.shared.drain_events();
        let out = events
            .into_iter()
            .map(|e| event_tuple(py, e))
            .collect::<PyResult<Vec<_>>>()?;
        Ok((chosen, out))
    }

    /// A perception frame or heartbeat arrived at `now` (feeds the native watchdog).
    fn note_perception(&self, now: f64) {
        self.shared.core().watch.note(now);
    }

    /// The native watchdog's view: None until the first frame, then fresh or stale.
    #[getter]
    fn perception_ok(&self) -> Option<bool> {
        self.shared.core().watch.ok
    }

    /// Follow `sounds.set_profile`: the native cue cache re-renders in `profile`.
    fn set_profile(&self, profile: &str) -> PyResult<()> {
        let p = Profile::parse(profile)
            .ok_or_else(|| PyValueError::new_err(format!("unknown sound profile {profile:?}")))?;
        self.shared.set_profile(p);
        Ok(())
    }

    /// Counters of the native path (steps, lateness, cue cache misses, beeps scheduled).
    fn native_stats(&self) -> HashMap<&'static str, f64> {
        let s = &self.shared.stats;
        HashMap::from([
            ("steps", s.steps.load(Relaxed) as f64),
            ("late_steps", s.late_steps.load(Relaxed) as f64),
            ("max_step_ms", s.max_step_ns.load(Relaxed) as f64 / 1e6),
            ("max_late_ms", s.max_late_ns.load(Relaxed) as f64 / 1e6),
            ("cue_misses", s.cue_misses.load(Relaxed) as f64),
            ("max_miss_ms", s.max_miss_ns.load(Relaxed) as f64 / 1e6),
            ("beeps", s.beeps.load(Relaxed) as f64),
            ("dropped_beeps", s.dropped_beeps.load(Relaxed) as f64),
            ("dropped_events", s.dropped_events.load(Relaxed) as f64),
        ])
    }

    /// A full perception frame from `source`: objects are
    /// `(id, kind, bearing, distance, closing_speed | None, confidence)`.
    /// Returns False when the frame was dropped as out of order.
    #[pyo3(signature = (source, objects, seq, now))]
    fn update_frame(
        &self,
        source: &str,
        objects: Vec<ObTuple>,
        seq: Option<i64>,
        now: f64,
    ) -> bool {
        let objs: Vec<Obstacle> = objects.into_iter().map(obstacle).collect();
        self.with(|s| s.update_frame(source, &objs, seq, now))
    }

    #[pyo3(signature = (source, obj, ttl, now))]
    fn update_one(&self, source: &str, obj: ObTuple, ttl: Option<f64>, now: f64) {
        let o = obstacle(obj);
        self.with(|s| s.update_one(source, &o, ttl, now))
    }

    #[pyo3(signature = (source=None))]
    fn clear(&self, source: Option<&str>) {
        self.with(|s| s.clear(source))
    }

    /// One control tick at wall time `now` with the engine clock at `engine_now`
    /// frames. Returns `(chosen tracks, cues)`; cues are, in order,
    /// `("beep", at, kind, pitch_q, bearing_q, critical, gain, track_id)` and
    /// `("say", text, priority, key, ttl)`.
    fn tick<'py>(
        &self,
        py: Python<'py>,
        now: f64,
        engine_now: i64,
    ) -> PyResult<(Vec<TrackState>, Vec<Bound<'py, PyTuple>>)> {
        let (chosen, cues) = self.with(|s| {
            let (idx, cues) = s.tick(now, engine_now);
            let chosen: Vec<TrackState> = idx.iter().map(|&i| (&s.tracks[i]).into()).collect();
            (chosen, cues)
        });
        let mut out = Vec::with_capacity(cues.len());
        for c in cues {
            out.push(match c {
                Cue::Beep(b) => (
                    "beep",
                    b.at,
                    b.kind,
                    b.pitch_q,
                    b.bearing_q,
                    b.critical,
                    b.gain,
                    b.track,
                )
                    .into_pyobject(py)?,
                Cue::Say(u) => ("say", u.text, u.priority, u.key, u.ttl).into_pyobject(py)?,
            });
        }
        Ok((chosen, out))
    }

    /// Snapshots of every track, in insertion order.
    fn tracks(&self) -> Vec<TrackState> {
        self.with(|s| s.tracks.iter().map(TrackState::from).collect())
    }

    fn stats(&self) -> HashMap<&'static str, u64> {
        let st = self.with(|s| s.stats);
        HashMap::from([
            ("frames", st.frames),
            ("objects", st.objects),
            ("out_of_order", st.out_of_order),
            ("beeps", st.beeps),
            ("criticals", st.criticals),
        ])
    }

    #[getter]
    fn rate(&self) -> u32 {
        self.with(|s| s.rate)
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.with(|s| s.enabled)
    }

    #[setter]
    fn set_enabled(&self, v: bool) {
        self.with(|s| s.enabled = v)
    }

    #[getter]
    fn muted(&self) -> bool {
        self.with(|s| s.muted)
    }

    #[setter]
    fn set_muted(&self, v: bool) {
        self.with(|s| s.muted = v)
    }

    #[getter]
    fn labels(&self) -> bool {
        self.with(|s| s.labels)
    }

    #[setter]
    fn set_labels(&self, v: bool) {
        self.with(|s| s.labels = v)
    }

    /// False = no speech decisions at all (the Python sonar with `speak=None`).
    #[getter]
    fn speak(&self) -> bool {
        self.with(|s| s.speak)
    }

    #[setter]
    fn set_speak(&self, v: bool) {
        self.with(|s| s.speak = v)
    }
}
