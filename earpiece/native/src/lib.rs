//! kite_audio: the earpiece's real-time audio engine.
//!
//! Python decides *what* to play; this crate owns *when* every sample goes out.
//! The device callback runs on CoreAudio's thread inside Rust, never touches the
//! Python interpreter, and talks to Python only through lock-free rings. So no
//! amount of Python work (Claude streaming, Moss lookups, perception floods) can
//! make the wearer's audio stutter.

mod mixer;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use mixer::{Cmd, Mixer, Voice};
use pyo3::exceptions::PyRuntimeError;
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

#[pymodule]
fn kite_audio(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add("BUS_CUE", mixer::BUS_CUE)?;
    m.add("BUS_ALERT", mixer::BUS_ALERT)?;
    Ok(())
}
