//! kite_audio: the earpiece's real-time audio engine.
//!
//! Python decides *what* to play; this crate owns *when* every sample goes out.
//! The device callback runs on CoreAudio's thread inside Rust, never touches the
//! Python interpreter, and talks to Python only through lock-free rings. So no
//! amount of Python work (Claude streaming, Moss lookups, perception floods) can
//! make the wearer's audio stutter.
//!
//! It also owns cue *placement*: `render_binaural` is the Brown-Duda renderer of
//! `spatial.py` in Rust, so the whole audible path (placement, mixing, output)
//! is native and runs without the GIL.
//!
//! And it owns the sonar's *decisions*: `Sonar` is the per-tick logic of
//! `sonar.py` (tracks, time-to-contact, urgency, beep scheduling, "Stop."), so
//! when each warning beep is due no longer depends on Python bytecode.
//!
//! And the cue *sounds*: `sounds` is `sounds.py` sample for sample, so a beep can
//! be synthesised without the interpreter.

mod mixer;
pub mod numpy_rng;
pub mod sonar;
mod sonar_py;
pub mod sounds;
pub mod spatial;
mod ziggurat;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use mixer::{Cmd, Mixer, Voice};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

struct Core {
    mixer: Mixer,
    cmds: Consumer<Cmd>,
}

#[derive(Default)]
struct Stats {
    blocks: AtomicU64,
    limited_blocks: AtomicU64,
    max_render_ns: AtomicU64,
    ema_render_ns: AtomicU64,
    peak_bits: AtomicU64,
    voices: AtomicU64,
    speaking: AtomicBool,
    contended: AtomicU64,
    glitches: AtomicU64,
    last_cb_ns: AtomicU64,
    stream_errors: AtomicU64,
    dropped_cmds: AtomicU64,
    now: AtomicU64,
}

struct DeviceHandle {
    stop: mpsc::Sender<()>,
    thread: Option<JoinHandle<()>>,
    name: String,
    channels: u16,
    buffer: u32,
}

#[derive(Clone, Copy)]
struct Params {
    master: f32,
    cue: f32,
    speech: f32,
    duck: f32,
    muted: bool,
}

#[pyclass(frozen)]
struct Engine {
    core: Arc<Mutex<Core>>,
    producer: Mutex<Producer<Cmd>>,
    events: Mutex<Consumer<(u8, u64)>>,
    garbage: Mutex<Consumer<Vec<f32>>>,
    stats: Arc<Stats>,
    device: Mutex<Option<DeviceHandle>>,
    params: Mutex<Params>,
    epoch: Instant,
    rate: u32,
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn record(
    stats: &Stats,
    t0: Instant,
    epoch: Instant,
    expected_ns: u64,
    b: mixer::BlockStats,
    now_frames: u64,
) {
    stats.now.store(now_frames, Relaxed);
    let dt = t0.elapsed().as_nanos() as u64;
    stats.blocks.fetch_add(1, Relaxed);
    stats.max_render_ns.fetch_max(dt, Relaxed);
    let ema = stats.ema_render_ns.load(Relaxed);
    stats.ema_render_ns.store(
        if ema == 0 {
            dt
        } else {
            ema - ema / 50 + dt / 50
        },
        Relaxed,
    );
    if b.limited {
        stats.limited_blocks.fetch_add(1, Relaxed);
    }
    stats.voices.store(b.voices as u64, Relaxed);
    stats.speaking.store(b.speaking, Relaxed);
    let prev_peak = f32::from_bits(stats.peak_bits.load(Relaxed) as u32);
    stats
        .peak_bits
        .store((prev_peak * 0.95).max(b.peak).to_bits() as u64, Relaxed);
    let now = epoch.elapsed().as_nanos() as u64;
    let last = stats.last_cb_ns.swap(now, Relaxed);
    if expected_ns > 0 && last > 0 && now - last > expected_ns * 2 + 5_000_000 {
        stats.glitches.fetch_add(1, Relaxed); // callback arrived late: the device starved
    }
}

impl Engine {
    fn push(&self, cmd: Cmd) -> bool {
        let mut p = self.producer.lock().unwrap();
        match p.push(cmd) {
            Ok(()) => true,
            Err(_) => {
                self.stats.dropped_cmds.fetch_add(1, Relaxed);
                false
            }
        }
    }

    fn send_params(&self) {
        let p = *self.params.lock().unwrap();
        self.push(Cmd::Params {
            master: p.master,
            cue: p.cue,
            speech: p.speech,
            duck: p.duck,
            muted: p.muted,
        });
    }
}

fn pick_device(host: &cpal::Host, substr: &str) -> Result<(cpal::Device, String), String> {
    let want = substr.to_lowercase();
    if !want.is_empty() {
        if let Ok(devs) = host.output_devices() {
            for d in devs {
                let name = device_name(&d);
                if name.to_lowercase().contains(&want) {
                    return Ok((d, name));
                }
            }
        }
    }
    let d = host.default_output_device().ok_or("no output device")?;
    let name = device_name(&d);
    Ok((d, name))
}

fn device_name(d: &cpal::Device) -> String {
    d.description()
        .map(|x| x.name().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (rate=48000, master_gain=0.9, ceiling=0.89, cue_gain=0.55, speech_gain=1.0, duck=0.45))]
    fn new(
        rate: u32,
        master_gain: f32,
        ceiling: f32,
        cue_gain: f32,
        speech_gain: f32,
        duck: f32,
    ) -> Self {
        let (cp, cc) = RingBuffer::<Cmd>::new(8192);
        let (ep, ec) = RingBuffer::<(u8, u64)>::new(4096);
        let (gp, gc) = RingBuffer::<Vec<f32>>::new(16384);
        let mixer = Mixer::new(
            rate,
            master_gain,
            ceiling,
            cue_gain,
            speech_gain,
            duck,
            ep,
            gp,
        );
        Engine {
            core: Arc::new(Mutex::new(Core { mixer, cmds: cc })),
            producer: Mutex::new(cp),
            events: Mutex::new(ec),
            garbage: Mutex::new(gc),
            stats: Arc::new(Stats::default()),
            device: Mutex::new(None),
            params: Mutex::new(Params {
                master: master_gain,
                cue: cue_gain,
                speech: speech_gain,
                duck,
                muted: false,
            }),
            epoch: Instant::now(),
            rate,
        }
    }

    // ---------------------------------------------------------------- control
    /// Schedule interleaved-stereo float32 bytes at absolute frame `at` (-1 = ASAP).
    #[pyo3(signature = (data, at=-1, bus=0, gain=1.0, tag=0))]
    fn play(&self, data: &[u8], at: i64, bus: u8, gain: f32, tag: u64) -> bool {
        let v = bytes_to_f32(data);
        if v.len() < 2 {
            return false;
        }
        self.push(Cmd::Play(Voice::new(v, at, gain, bus, tag)))
    }

    #[pyo3(signature = (tag, fade=1440))]
    fn stop(&self, tag: u64, fade: u64) -> bool {
        self.push(Cmd::Stop { tag, fade })
    }

    #[pyo3(signature = (uid, gain=1.0))]
    fn speech_begin(&self, uid: u64, gain: f32) -> bool {
        self.push(Cmd::SpBegin { uid, gain })
    }

    fn speech_feed(&self, uid: u64, data: &[u8]) -> bool {
        let v = bytes_to_f32(data);
        if v.is_empty() {
            return true;
        }
        self.push(Cmd::SpFeed { uid, data: v })
    }

    fn speech_end(&self, uid: u64) -> bool {
        self.push(Cmd::SpEnd { uid })
    }

    #[pyo3(signature = (uid=0, fade=1920))]
    fn speech_cancel(&self, uid: u64, fade: u64) -> bool {
        self.push(Cmd::SpCancel { uid, fade })
    }

    #[pyo3(signature = (master_gain=None, cue_gain=None, speech_gain=None, duck=None, cues_muted=None))]
    fn set_params(
        &self,
        master_gain: Option<f32>,
        cue_gain: Option<f32>,
        speech_gain: Option<f32>,
        duck: Option<f32>,
        cues_muted: Option<bool>,
    ) {
        {
            let mut p = self.params.lock().unwrap();
            if let Some(x) = master_gain {
                p.master = x;
            }
            if let Some(x) = cue_gain {
                p.cue = x;
            }
            if let Some(x) = speech_gain {
                p.speech = x;
            }
            if let Some(x) = duck {
                p.duck = x;
            }
            if let Some(x) = cues_muted {
                p.muted = x;
            }
        }
        self.send_params();
    }

    /// Drain events [(kind, uid)] (0 start, 1 done, 2 cut) and free returned buffers.
    fn poll_events(&self) -> Vec<(u8, u64)> {
        {
            let mut g = self.garbage.lock().unwrap();
            while g.pop().is_ok() {}
        }
        let mut e = self.events.lock().unwrap();
        std::iter::from_fn(|| e.pop().ok()).collect()
    }

    // ---------------------------------------------------------------- offline render
    /// Render frames without a device (tests, simulation, WAV export). Returns
    /// interleaved stereo float32 bytes. Releases the GIL while mixing.
    fn render<'py>(&self, py: Python<'py>, frames: usize) -> Bound<'py, PyBytes> {
        let core = self.core.clone();
        let stats = self.stats.clone();
        let epoch = self.epoch;
        let out: Vec<f32> = py.detach(move || {
            let mut out = vec![0.0f32; frames * 2];
            let t0 = Instant::now();
            let b = {
                let mut g = core.lock().unwrap();
                let Core { mixer, cmds } = &mut *g;
                let b = mixer.render(&mut out, cmds);
                (b, mixer.now)
            };
            record(&stats, t0, epoch, 0, b.0, b.1);
            out
        });
        let bytes: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
        PyBytes::new(py, &bytes)
    }

    // ---------------------------------------------------------------- device
    /// Open the output (first device whose name contains `substr`, else the default).
    /// The cpal stream lives on its own thread; returns (device name, channels, buffer).
    #[pyo3(signature = (substr="", block=512))]
    fn open(&self, py: Python<'_>, substr: &str, block: u32) -> PyResult<(String, u16, u32)> {
        self.close(py);
        let core = self.core.clone();
        let stats = self.stats.clone();
        let epoch = self.epoch;
        let rate = self.rate;
        let substr = substr.to_string();
        let (res_tx, res_rx) = mpsc::channel::<Result<(String, u16, u32), String>>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("kite-audio-device".into())
            .spawn(move || {
                let host = cpal::default_host();
                let (dev, name) = match pick_device(&host, &substr) {
                    Ok(x) => x,
                    Err(e) => {
                        let _ = res_tx.send(Err(e));
                        return;
                    }
                };
                let channels: u16 = match dev.default_output_config() {
                    Ok(c) => c.channels().min(2).max(1),
                    Err(_) => 2,
                };
                let expected_ns = (block as u64) * 1_000_000_000 / rate as u64;
                let make = |buf: cpal::BufferSize| {
                    let core = core.clone();
                    let stats = stats.clone();
                    let stats_err = stats.clone();
                    let mut scratch = vec![0.0f32; 16384 * 2];
                    let cfg = cpal::StreamConfig {
                        channels,
                        sample_rate: rate,
                        buffer_size: buf,
                    };
                    dev.build_output_stream(
                        cfg,
                        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                            let t0 = Instant::now();
                            let frames = data.len() / channels as usize;
                            match core.try_lock() {
                                Ok(mut g) => {
                                    let Core { mixer, cmds } = &mut *g;
                                    let b = if channels == 2 {
                                        mixer.render(data, cmds)
                                    } else {
                                        if scratch.len() < frames * 2 {
                                            scratch.resize(frames * 2, 0.0);
                                        }
                                        let b = mixer.render(&mut scratch[..frames * 2], cmds);
                                        for f in 0..frames {
                                            data[f] = 0.5 * (scratch[f * 2] + scratch[f * 2 + 1]);
                                        }
                                        b
                                    };
                                    let now_frames = mixer.now;
                                    drop(g);
                                    record(&stats, t0, epoch, expected_ns, b, now_frames);
                                }
                                Err(_) => {
                                    data.fill(0.0); // an offline render holds the lock: never block here
                                    stats.contended.fetch_add(1, Relaxed);
                                }
                            }
                        },
                        move |_err| {
                            stats_err.stream_errors.fetch_add(1, Relaxed);
                        },
                        None,
                    )
                };
                let (stream, used) = match make(cpal::BufferSize::Fixed(block)) {
                    Ok(s) => (s, block),
                    Err(_) => match make(cpal::BufferSize::Default) {
                        Ok(s) => (s, 0),
                        Err(e) => {
                            let _ = res_tx.send(Err(format!("build stream: {e}")));
                            return;
                        }
                    },
                };
                if let Err(e) = stream.play() {
                    let _ = res_tx.send(Err(format!("play: {e}")));
                    return;
                }
                let _ = res_tx.send(Ok((name, channels, used)));
                let _ = stop_rx.recv(); // park until close() (or the Engine is dropped)
                drop(stream);
            })
            .map_err(|e| PyRuntimeError::new_err(format!("spawn: {e}")))?;
        let res = py.detach(move || res_rx.recv_timeout(std::time::Duration::from_secs(5)));
        match res {
            Ok(Ok((name, ch, buf))) => {
                self.stats.last_cb_ns.store(0, Relaxed);
                *self.device.lock().unwrap() = Some(DeviceHandle {
                    stop: stop_tx,
                    thread: Some(thread),
                    name: name.clone(),
                    channels: ch,
                    buffer: buf,
                });
                Ok((name, ch, buf))
            }
            Ok(Err(e)) => Err(PyRuntimeError::new_err(e)),
            Err(_) => Err(PyRuntimeError::new_err(
                "timed out opening the audio device",
            )),
        }
    }

    fn close(&self, py: Python<'_>) {
        let h = self.device.lock().unwrap().take();
        if let Some(mut h) = h {
            let _ = h.stop.send(());
            if let Some(t) = h.thread.take() {
                py.detach(|| {
                    let _ = t.join();
                });
            }
        }
    }

    fn device(&self) -> Option<(String, u16, u32)> {
        self.device
            .lock()
            .unwrap()
            .as_ref()
            .map(|h| (h.name.clone(), h.channels, h.buffer))
    }

    /// Milliseconds since the device last asked for audio (None if no device / never).
    fn callback_age_ms(&self) -> Option<f64> {
        let last = self.stats.last_cb_ns.load(Relaxed);
        if self.device.lock().unwrap().is_none() || last == 0 {
            return None;
        }
        Some((self.epoch.elapsed().as_nanos() as u64).saturating_sub(last) as f64 / 1e6)
    }

    #[staticmethod]
    fn outputs() -> Vec<(String, bool)> {
        let host = cpal::default_host();
        let dflt = host.default_output_device().map(|d| device_name(&d));
        host.output_devices()
            .map(|it| {
                it.map(|d| {
                    let n = device_name(&d);
                    let is = Some(&n) == dflt.as_ref();
                    (n, is)
                })
                .collect()
            })
            .unwrap_or_default()
    }

    // ---------------------------------------------------------------- state
    /// The mixer clock in frames (published by the audio thread after every block).
    #[getter]
    fn now(&self) -> u64 {
        self.stats.now.load(Relaxed)
    }

    #[getter]
    fn speaking(&self) -> bool {
        self.stats.speaking.load(Relaxed)
    }

    #[getter]
    fn rate(&self) -> u32 {
        self.rate
    }

    fn stats(&self) -> std::collections::HashMap<String, f64> {
        let s = &self.stats;
        let mut m = std::collections::HashMap::new();
        m.insert("blocks".into(), s.blocks.load(Relaxed) as f64);
        m.insert(
            "limited_blocks".into(),
            s.limited_blocks.load(Relaxed) as f64,
        );
        m.insert(
            "max_render_ms".into(),
            s.max_render_ns.load(Relaxed) as f64 / 1e6,
        );
        m.insert(
            "avg_render_ms".into(),
            s.ema_render_ns.load(Relaxed) as f64 / 1e6,
        );
        m.insert(
            "peak".into(),
            f32::from_bits(s.peak_bits.load(Relaxed) as u32) as f64,
        );
        m.insert("voices".into(), s.voices.load(Relaxed) as f64);
        m.insert("contended".into(), s.contended.load(Relaxed) as f64);
        m.insert("glitches".into(), s.glitches.load(Relaxed) as f64);
        m.insert("stream_errors".into(), s.stream_errors.load(Relaxed) as f64);
        m.insert("dropped_cmds".into(), s.dropped_cmds.load(Relaxed) as f64);
        m
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(mut h) = self.device.lock().unwrap().take() {
            let _ = h.stop.send(());
            if let Some(t) = h.thread.take() {
                let _ = t.join();
            }
        }
    }
}

// -------------------------------------------------------------------- binaural
fn f32_le(b: &[u8], what: &str) -> PyResult<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        return Err(PyValueError::new_err(format!(
            "{what} must be float32 little-endian bytes (length a multiple of 4, got {})",
            b.len()
        )));
    }
    Ok(bytes_to_f32(b))
}

fn f32_bytes<'py>(py: Python<'py>, v: &[f32]) -> PyResult<Bound<'py, PyBytes>> {
    PyBytes::new_with(py, v.len() * 4, |buf| {
        for (o, x) in buf.chunks_exact_mut(4).zip(v) {
            o.copy_from_slice(&x.to_le_bytes());
        }
        Ok(())
    })
}

fn check_render_args(azimuths: &[f64], rate: u32, sinc_taps: usize) -> PyResult<spatial::Delay> {
    if rate == 0 {
        return Err(PyValueError::new_err("rate must be positive"));
    }
    if let Some(a) = azimuths.iter().find(|a| !a.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "azimuth must be finite, got {a}"
        )));
    }
    spatial::Delay::from_taps(sinc_taps).map_err(PyValueError::new_err)
}

/// Place a mono cue at `azimuth_deg` (0 ahead, +90 right): float32 LE bytes in,
/// interleaved stereo float32 LE bytes out, `n + pad (+ sinc_taps)` frames.
/// Bit-for-bit the reference `spatial._render_py` when `sinc_taps == 0`; an even
/// `sinc_taps` (2..=64) selects the windowed-sinc fractional delay instead.
/// Releases the GIL while rendering.
#[pyfunction]
#[pyo3(signature = (mono, azimuth_deg, rate, gain=1.0, extra_ild_db=6.0, sinc_taps=0))]
fn render_binaural<'py>(
    py: Python<'py>,
    mono: &[u8],
    azimuth_deg: f64,
    rate: u32,
    gain: f64,
    extra_ild_db: f64,
    sinc_taps: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let mode = check_render_args(&[azimuth_deg], rate, sinc_taps)?;
    let x = f32_le(mono, "mono")?;
    let out = py.detach(move || spatial::render(&x, azimuth_deg, rate, gain, extra_ild_db, mode));
    f32_bytes(py, &out)
}

/// The same cue at many azimuths in one call (cache pre-warming): one bytes
/// object per azimuth, each identical to `render_binaural` at that azimuth.
/// Shared work is done once and azimuths are spread over threads, GIL released.
#[pyfunction]
#[pyo3(signature = (mono, azimuths, rate, gain=1.0, extra_ild_db=6.0, sinc_taps=0))]
fn render_binaural_batch<'py>(
    py: Python<'py>,
    mono: &[u8],
    azimuths: Vec<f64>,
    rate: u32,
    gain: f64,
    extra_ild_db: f64,
    sinc_taps: usize,
) -> PyResult<Vec<Bound<'py, PyBytes>>> {
    let mode = check_render_args(&azimuths, rate, sinc_taps)?;
    let x = f32_le(mono, "mono")?;
    let outs =
        py.detach(move || spatial::render_batch(&x, &azimuths, rate, gain, extra_ild_db, mode));
    outs.iter().map(|o| f32_bytes(py, o)).collect()
}

/// The per-ear delay stage on its own (diagnostics, the ITD accuracy test):
/// delays mono float32 bytes by `delay_samples`, returns `out_len` samples.
#[pyfunction]
#[pyo3(signature = (mono, delay_samples, out_len, sinc_taps=0))]
fn fractional_delay<'py>(
    py: Python<'py>,
    mono: &[u8],
    delay_samples: f64,
    out_len: usize,
    sinc_taps: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let mode = spatial::Delay::from_taps(sinc_taps).map_err(PyValueError::new_err)?;
    if !delay_samples.is_finite() || delay_samples < 0.0 {
        return Err(PyValueError::new_err(
            "delay_samples must be finite and >= 0",
        ));
    }
    let x = f32_le(mono, "mono")?;
    let out = py.detach(move || spatial::fractional_delay(&x, delay_samples, out_len, mode));
    f32_bytes(py, &out)
}

// -------------------------------------------------------------------- cue synthesis
fn profile_arg(profile: &str) -> PyResult<sounds::Profile> {
    sounds::Profile::parse(profile).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown sound profile {profile:?}; expected 'standard' or 'gentle'"
        ))
    })
}

fn finite(name: &str, v: f64) -> PyResult<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be finite, got {v}"
        )))
    }
}

/// `sounds.proximity_cue(kind, freq)` under `profile`: mono float32 LE bytes at 48 kHz,
/// sample-identical to the numpy/scipy reference. Releases the GIL.
#[pyfunction]
#[pyo3(signature = (kind, freq, profile="standard"))]
fn synth_proximity<'py>(
    py: Python<'py>,
    kind: &str,
    freq: f64,
    profile: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let p = profile_arg(profile)?;
    let freq = finite("freq", freq)?;
    if freq <= 0.0 {
        return Err(PyValueError::new_err("freq must be positive"));
    }
    let class = sounds::cue_class(kind);
    let out = py.detach(move || sounds::class_cue(class, freq, p));
    f32_bytes(py, &out)
}

/// `sounds.beacon_tick(region, bearing)` under `profile` (mono float32 LE bytes).
#[pyfunction]
#[pyo3(signature = (region, bearing, profile="standard"))]
fn synth_beacon<'py>(
    py: Python<'py>,
    region: &str,
    bearing: f64,
    profile: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let p = profile_arg(profile)?;
    let bearing = finite("bearing", bearing)?;
    let region = region.to_string();
    let out = py
        .detach(move || sounds::beacon_tick(&region, bearing, p))
        .ok_or_else(|| PyValueError::new_err("region must be on, near, side or behind"))?;
    f32_bytes(py, &out)
}

/// Any other cue of `sounds.py` by name: `"person"`, `"vehicle"`, `"obstacle"` and
/// `"drop"` take `[freq]` and the profile; the status cues and building blocks take
/// their numeric arguments in order (see `sounds::status`). Mono float32 LE bytes.
#[pyfunction]
#[pyo3(signature = (name, args=Vec::new(), profile="standard"))]
fn synth<'py>(
    py: Python<'py>,
    name: &str,
    args: Vec<f64>,
    profile: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let p = profile_arg(profile)?;
    for &a in &args {
        finite("argument", a)?;
    }
    let name = name.to_string();
    let out = py.detach(move || {
        let class = match name.as_str() {
            "person" => Some(sounds::CueClass::Person),
            "vehicle" => Some(sounds::CueClass::Vehicle),
            "obstacle" => Some(sounds::CueClass::Obstacle),
            "drop" => Some(sounds::CueClass::Drop),
            _ => None,
        };
        match class {
            Some(c) => args
                .first()
                .filter(|&&f| f > 0.0)
                .map(|&f| sounds::class_cue(c, f, p)),
            None => sounds::status(&name, &args),
        }
    });
    let out = out.ok_or_else(|| PyValueError::new_err("unknown cue name or wrong arguments"))?;
    f32_bytes(py, &out)
}

#[pymodule]
fn kite_audio(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<sonar_py::PySonar>()?;
    m.add_class::<sonar_py::TrackState>()?;
    m.add_function(wrap_pyfunction!(render_binaural, m)?)?;
    m.add_function(wrap_pyfunction!(render_binaural_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fractional_delay, m)?)?;
    m.add_function(wrap_pyfunction!(synth_proximity, m)?)?;
    m.add_function(wrap_pyfunction!(synth_beacon, m)?)?;
    m.add_function(wrap_pyfunction!(synth, m)?)?;
    m.add("SOUND_RATE", sounds::RATE)?;
    m.add("SINC_TAPS_MAX", spatial::MAX_SINC_TAPS)?;
    m.add("BUS_CUE", mixer::BUS_CUE)?;
    m.add("BUS_ALERT", mixer::BUS_ALERT)?;
    Ok(())
}
