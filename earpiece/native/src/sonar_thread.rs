//! The sonar's native mode: the decision core ticks on its own thread, synthesises
//! and places every beep in Rust and schedules it straight into the mixer, so the
//! warning beeps keep their timing however long Python is stalled (garbage
//! collection, a busy GIL, a slow import).
//!
//! What crosses to Python: obstacle frames come in through `PySonar` (a short
//! mutex hold, no GIL needed by this thread), and "speak" requests ("Stop.",
//! "Person, slightly left.") plus watchdog transitions go out through a lock-free
//! ring that Python's control loop polls. Speech itself is still Python's.
//!
//! One step here is exactly what the Python-ticked path does per control-loop
//! iteration: `Sonar::tick`, then the app's stale-perception watchdog, then each
//! beep is played as `sonar._cue(kind, pitch_q, bearing_q)` would render it
//! (`sounds::proximity_cue` placed by `spatial::render` with its defaults), so with
//! the same tick times both paths produce the same samples.

use crate::mixer::{Voice, BUS_ALERT, BUS_CUE};
use crate::sonar::{Cue, Sonar, Utterance};
use crate::sounds::{self, CueClass, Profile};
use crate::spatial::{self, Delay};
use crate::EngineLink;
use rtrb::{Consumer, Producer, RingBuffer};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

const EVENT_RING: usize = 1024;
/// Like `sonar._cue_render`'s `lru_cache(maxsize=2048)`; pre-warmed cues are never evicted.
const CACHE_MAX: usize = 2048;

/// The app's stale-perception watchdog (`Earpiece._watchdog`), natively.
#[derive(Clone, Debug)]
pub struct Watchdog {
    pub stale_s: f64,
    pub last: f64,
    pub seen: bool,
    pub ok: Option<bool>,
}

impl Watchdog {
    pub fn new(stale_s: f64) -> Self {
        Watchdog {
            stale_s,
            last: 0.0,
            seen: false,
            ok: None,
        }
    }

    /// A frame or heartbeat arrived at `now`.
    pub fn note(&mut self, now: f64) {
        self.seen = true;
        self.last = now;
    }

    /// After a tick: switch the sonar off (and forget every track) when perception has
    /// gone quiet for `stale_s`, back on when it returns. Returns the transition.
    fn check(&mut self, now: f64, sonar: &mut Sonar) -> Option<Event> {
        if !self.seen {
            return None;
        }
        let ok = now - self.last <= self.stale_s;
        if self.ok == Some(ok) {
            return None;
        }
        let first = self.ok.is_none();
        self.ok = Some(ok);
        sonar.enabled = ok;
        if !ok {
            sonar.clear(None);
        }
        Some(Event::Perception { ok, first, at: now })
    }
}

/// Everything Python's control loop picks up with `poll()`.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Say(Utterance),
    Perception { ok: bool, first: bool, at: f64 },
}

/// The state Python's methods and the native step share, behind one mutex.
pub struct Core {
    pub sonar: Sonar,
    pub watch: Watchdog,
    /// Ids of the tracks the last native step sonified.
    pub chosen: Vec<String>,
}

#[derive(Default)]
pub struct NativeStats {
    pub steps: AtomicU64,
    pub late_steps: AtomicU64,
    pub max_step_ns: AtomicU64,
    pub max_late_ns: AtomicU64,
    pub cue_misses: AtomicU64,
    pub max_miss_ns: AtomicU64,
    pub beeps: AtomicU64,
    pub dropped_beeps: AtomicU64,
    pub dropped_events: AtomicU64,
}

pub struct Shared {
    pub core: Mutex<Core>,
    events_tx: Mutex<Producer<Event>>,
    pub events_rx: Mutex<Consumer<Event>>,
    cache: Mutex<CueCache>,
    profile: AtomicU8,
    profile_gen: AtomicU64,
    pub stop: AtomicBool,
    pub stats: NativeStats,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // a panic while holding a lock leaves the state consistent field by field: keep going
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn profile_code(p: Profile) -> u8 {
    match p {
        Profile::Standard => 0,
        Profile::Gentle => 1,
    }
}

fn profile_of(c: u8) -> Profile {
    if c == 1 {
        Profile::Gentle
    } else {
        Profile::Standard
    }
}

impl Shared {
    pub fn new(sonar: Sonar, stale_s: f64) -> Self {
        let (tx, rx) = RingBuffer::new(EVENT_RING);
        Shared {
            core: Mutex::new(Core {
                sonar,
                watch: Watchdog::new(stale_s),
                chosen: Vec::new(),
            }),
            events_tx: Mutex::new(tx),
            events_rx: Mutex::new(rx),
            cache: Mutex::new(CueCache::new(Profile::Standard)),
            profile: AtomicU8::new(0),
            profile_gen: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            stats: NativeStats::default(),
        }
    }

    pub fn core(&self) -> MutexGuard<'_, Core> {
        lock(&self.core)
    }

    /// `sounds.set_profile` happened: re-render every cue in the new profile.
    pub fn set_profile(&self, p: Profile) {
        if self.profile.swap(profile_code(p), Relaxed) != profile_code(p) {
            self.profile_gen.fetch_add(1, Relaxed);
        }
    }

    pub fn profile(&self) -> Profile {
        profile_of(self.profile.load(Relaxed))
    }

    fn emit(&self, e: Event) {
        if lock(&self.events_tx).push(e).is_err() {
            self.stats.dropped_events.fetch_add(1, Relaxed);
        }
    }

    pub fn drain_events(&self) -> Vec<Event> {
        let mut rx = lock(&self.events_rx);
        std::iter::from_fn(|| rx.pop().ok()).collect()
    }

    /// One control-loop iteration at wall time `now`: tick, watchdog, then synthesise,
    /// place and schedule each beep, and queue each utterance for Python. The core lock
    /// is released before any cue is rendered.
    pub fn step(&self, link: &EngineLink, now: f64) {
        let engine_now = link.now();
        let (cues, watch) = {
            let mut g = self.core();
            let Core {
                sonar,
                watch,
                chosen,
            } = &mut *g;
            let (idx, cues) = sonar.tick(now, engine_now);
            *chosen = idx.iter().map(|&i| sonar.tracks[i].id.clone()).collect();
            let w = watch.check(now, sonar);
            (cues, w)
        };
        let mut cache = lock(&self.cache);
        cache.sync(self.profile(), self.profile_gen.load(Relaxed));
        for c in cues {
            match c {
                Cue::Beep(b) => {
                    let t0 = Instant::now();
                    let (data, miss) = cache.get(
                        sounds::cue_class(&b.kind),
                        b.pitch_q,
                        b.bearing_q,
                        link.rate,
                    );
                    if miss {
                        self.stats.cue_misses.fetch_add(1, Relaxed);
                        self.stats
                            .max_miss_ns
                            .fetch_max(t0.elapsed().as_nanos() as u64, Relaxed);
                    }
                    let bus = if b.critical { BUS_ALERT } else { BUS_CUE };
                    if link.play(Voice::new(
                        data.as_ref().clone(),
                        b.at,
                        b.gain as f32,
                        bus,
                        0,
                    )) {
                        self.stats.beeps.fetch_add(1, Relaxed);
                    } else {
                        self.stats.dropped_beeps.fetch_add(1, Relaxed);
                    }
                }
                Cue::Say(u) => self.emit(Event::Say(u)),
            }
        }
        drop(cache);
        if let Some(e) = watch {
            self.emit(e);
        }
        link.drain_garbage();
    }

    /// Render every critical cue (urgency 1) at every 5-degree bearing ahead of time,
    /// like `sonar.prewarm`, so the first beep of a sudden danger never waits on synthesis.
    pub fn prewarm(&self, rate: u32) -> usize {
        let pitch_q = self.core().sonar.critical_pitch_q();
        let mut cache = lock(&self.cache);
        cache.sync(self.profile(), self.profile_gen.load(Relaxed));
        cache.prewarm(pitch_q, rate)
    }
}

type CueKey = (CueClass, i64, i64);

/// Rendered stereo sonar cues by (timbre, quantised pitch, 5-degree bearing).
struct CueCache {
    profile: Profile,
    gen: u64,
    map: HashMap<CueKey, (Arc<Vec<f32>>, u64, bool)>, // data, last use, warm
    clock: u64,
}

const CLASSES: [CueClass; 4] = [
    CueClass::Person,
    CueClass::Vehicle,
    CueClass::Obstacle,
    CueClass::Drop,
];

impl CueCache {
    fn new(profile: Profile) -> Self {
        CueCache {
            profile,
            gen: 0,
            map: HashMap::new(),
            clock: 0,
        }
    }

    fn sync(&mut self, profile: Profile, gen: u64) {
        if gen != self.gen || profile != self.profile {
            self.map.clear();
            self.profile = profile;
            self.gen = gen;
        }
    }

    fn render(&self, class: CueClass, pitch_q: i64, bearing_q: i64, rate: u32) -> Vec<f32> {
        let mono = sounds::class_cue(class, pitch_q as f64, self.profile);
        spatial::render(&mono, bearing_q as f64, rate, 1.0, 6.0, Delay::Linear)
    }

    fn get(
        &mut self,
        class: CueClass,
        pitch_q: i64,
        bearing_q: i64,
        rate: u32,
    ) -> (Arc<Vec<f32>>, bool) {
        self.clock += 1;
        let key = (class, pitch_q, bearing_q);
        if let Some(e) = self.map.get_mut(&key) {
            e.1 = self.clock;
            return (e.0.clone(), false);
        }
        let data = Arc::new(self.render(class, pitch_q, bearing_q, rate));
        let cold = self.map.values().filter(|e| !e.2).count();
        if cold >= CACHE_MAX {
            if let Some(k) = self
                .map
                .iter()
                .filter(|(_, e)| !e.2)
                .min_by_key(|(_, e)| e.1)
                .map(|(k, _)| *k)
            {
                self.map.remove(&k);
            }
        }
        self.map.insert(key, (data.clone(), self.clock, false));
        (data, true)
    }

    fn prewarm(&mut self, pitch_q: i64, rate: u32) -> usize {
        let bearings: Vec<f64> = (-180..=180).step_by(5).map(|b| b as f64).collect();
        let mut n = 0;
        for class in CLASSES {
            let mono = sounds::class_cue(class, pitch_q as f64, self.profile);
            let placed = spatial::render_batch(&mono, &bearings, rate, 1.0, 6.0, Delay::Linear);
            for (&b, st) in bearings.iter().zip(placed) {
                self.map
                    .insert((class, pitch_q, b as i64), (Arc::new(st), 0, true));
                n += 1;
            }
        }
        n
    }
}

/// `time.monotonic()` on the native side: Python's reading at start plus the native
/// monotonic clock's elapsed time (both are the uptime clock on macOS and Linux).
#[derive(Clone, Copy)]
pub struct Clock {
    py_t0: f64,
    epoch: Instant,
}

impl Clock {
    pub fn new(py_now: f64) -> Self {
        Clock {
            py_t0: py_now,
            epoch: Instant::now(),
        }
    }

    pub fn now(&self) -> f64 {
        self.py_t0 + self.epoch.elapsed().as_secs_f64()
    }
}

/// Put this thread in macOS's real-time band, like an audio thread (period 20 ms, up to
/// 3 ms of work, done within 10 ms), falling back to the user-interactive QoS class. Beeps
/// are scheduled only 60 ms ahead, and on a loaded laptop (load average 10 to 20) a
/// default thread woke up to 51 ms late, a user-interactive one 42 ms, a real-time one 19 ms.
#[cfg(target_os = "macos")]
fn raise_priority() {
    #[repr(C)]
    struct TimeConstraint {
        period: u32,
        computation: u32,
        constraint: u32,
        preemptible: i32,
    }
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        fn pthread_mach_thread_np(thread: usize) -> u32;
        fn pthread_self() -> usize;
        fn mach_timebase_info(info: *mut Timebase) -> i32;
        fn thread_policy_set(
            thread: u32,
            flavor: u32,
            info: *mut TimeConstraint,
            count: u32,
        ) -> i32;
    }
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    const THREAD_TIME_CONSTRAINT_POLICY: u32 = 2;
    let mut tb = Timebase { numer: 0, denom: 0 };
    // SAFETY: mach_timebase_info writes two u32s into the struct we own; the policy struct
    // is the documented 4-field layout and lives for the call; pthread_mach_thread_np of
    // pthread_self() names the calling thread without creating a port right to release.
    unsafe {
        if mach_timebase_info(&mut tb) != 0 || tb.numer == 0 {
            return;
        }
        let abs = |ns: u64| (ns * tb.denom as u64 / tb.numer as u64) as u32;
        let mut p = TimeConstraint {
            period: abs(20_000_000),
            computation: abs(3_000_000),
            constraint: abs(10_000_000),
            preemptible: 1,
        };
        let me = pthread_mach_thread_np(pthread_self());
        if thread_policy_set(me, THREAD_TIME_CONSTRAINT_POLICY, &mut p, 4) != 0 {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn raise_priority() {}

/// The native thread: a step every `period`, on an absolute schedule like the Python
/// control loop (`nxt += period`; when it falls behind it starts again from now).
pub fn run(shared: Arc<Shared>, link: EngineLink, clock: Clock, period: Duration, prewarm: bool) {
    raise_priority();
    let mut warm_gen = u64::MAX;
    let mut next = Instant::now();
    while !shared.stop.load(Relaxed) {
        if prewarm && warm_gen != shared.profile_gen.load(Relaxed) {
            warm_gen = shared.profile_gen.load(Relaxed);
            shared.prewarm(link.rate);
        }
        let t0 = Instant::now();
        let late = t0.saturating_duration_since(next);
        shared.step(&link, clock.now());
        let st = &shared.stats;
        st.steps.fetch_add(1, Relaxed);
        st.max_step_ns
            .fetch_max(t0.elapsed().as_nanos() as u64, Relaxed);
        st.max_late_ns.fetch_max(late.as_nanos() as u64, Relaxed);
        if late > period {
            st.late_steps.fetch_add(1, Relaxed);
        }
        next += period;
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        } else {
            next = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixer::{Cmd, Mixer};
    use crate::sonar::{Obstacle, SonarCfg};
    use crate::Stats;

    fn rig() -> (EngineLink, Mixer, Consumer<Cmd>) {
        let (cp, cc) = RingBuffer::new(4096);
        let (gp, gc) = RingBuffer::new(4096);
        let (ep, _ec) = RingBuffer::new(64);
        std::mem::forget(_ec);
        let link = EngineLink {
            producer: Arc::new(Mutex::new(cp)),
            garbage: Arc::new(Mutex::new(gc)),
            stats: Arc::new(Stats::default()),
            rate: 48000,
        };
        (
            link,
            Mixer::new(48000, 0.9, 0.89, 0.55, 1.0, 0.45, ep, gp),
            cc,
        )
    }

    fn pole(d: f64) -> Obstacle {
        Obstacle {
            id: "p".into(),
            kind: "pole".into(),
            bearing: 0.0,
            distance: d,
            closing_speed: None,
            confidence: 1.0,
            priority: None,
        }
    }

    #[test]
    fn steps_schedule_rendered_beeps_into_the_mixer() {
        let (link, mut mixer, mut cmds) = rig();
        let sh = Shared::new(Sonar::new(SonarCfg::default(), 48000), 1.5);
        let mut out = vec![0.0f32; 960 * 2];
        let mut heard = 0usize;
        for i in 0..100 {
            let now = i as f64 * 0.02;
            {
                let mut g = sh.core();
                g.sonar.update_frame("cam", &[pole(0.5)], None, now);
                g.watch.note(now);
            }
            sh.step(&link, now);
            mixer.render(&mut out, &mut cmds);
            link.stats.now.store(mixer.now, Relaxed);
            heard += out.iter().filter(|v| v.abs() > 1e-3).count();
        }
        assert!(sh.stats.beeps.load(Relaxed) > 10);
        assert!(heard > 0);
        let ev = sh.drain_events();
        assert!(ev
            .iter()
            .any(|e| matches!(e, Event::Say(u) if u.text == "Stop. Obstacle ahead.")));
        assert!(ev.iter().any(|e| matches!(
            e,
            Event::Perception {
                ok: true,
                first: true,
                ..
            }
        )));
    }

    #[test]
    fn watchdog_silences_stale_perception_and_recovers() {
        let (link, _m, _c) = rig();
        let sh = Shared::new(Sonar::new(SonarCfg::default(), 48000), 1.5);
        {
            let mut g = sh.core();
            g.sonar.update_frame("cam", &[pole(0.5)], None, 0.0);
            g.watch.note(0.0);
        }
        sh.step(&link, 0.0);
        sh.step(&link, 1.6);
        {
            let g = sh.core();
            assert!(!g.sonar.enabled && g.sonar.tracks.is_empty());
            assert_eq!(g.watch.ok, Some(false));
        }
        let ev = sh.drain_events();
        assert!(ev.contains(&Event::Perception {
            ok: false,
            first: false,
            at: 1.6
        }));
        sh.core().watch.note(2.0);
        sh.step(&link, 2.0);
        assert!(sh.core().sonar.enabled);
    }

    #[test]
    fn cache_reuses_and_follows_the_profile() {
        let mut c = CueCache::new(Profile::Standard);
        let (a, miss) = c.get(CueClass::Person, 900, 15, 48000);
        assert!(miss);
        let (b, miss) = c.get(CueClass::Person, 900, 15, 48000);
        assert!(!miss && Arc::ptr_eq(&a, &b));
        c.sync(Profile::Gentle, 1);
        let (g, miss) = c.get(CueClass::Person, 900, 15, 48000);
        assert!(miss && *g != *a);
        assert_eq!(c.prewarm(1250, 48000), 4 * 73);
    }

    #[test]
    fn clock_advances_from_the_python_reading() {
        let c = Clock::new(1000.0);
        let t = c.now();
        assert!((1000.0..1001.0).contains(&t));
    }
}
