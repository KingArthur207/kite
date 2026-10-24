//! Python binding of the sonar decision core (`kite_audio.Sonar`).
//!
//! The Python side (`earpiece/rust_sonar.py`) feeds frames in and plays the cues
//! `tick` hands back; this file only converts. The core sits behind a mutex and the
//! lock is never held while touching Python objects, so the perception thread
//! (frames) and the control loop (ticks) cannot deadlock against the GIL.

use crate::sonar::{Cue, Obstacle, Sonar, SonarCfg, Track};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashMap;
use std::sync::Mutex;

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
    core: Mutex<Sonar>,
}

impl PySonar {
    fn with<R>(&self, f: impl FnOnce(&mut Sonar) -> R) -> R {
        // A panic while holding the lock would poison it; the state is still
        // consistent field by field, so keep going rather than fail every call.
        let mut g = self.core.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut g)
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
            core: Mutex::new(Sonar::new(cfg, rate)),
        })
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
