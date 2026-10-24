//! The real-time mixer. A line-for-line port of `earpiece/mixer.py` (same buses,
//! ducking, fades and limiter), so both engines produce the same samples; the
//! Python test suite checks that parity.
//!
//! Nothing in `render` allocates, locks or blocks: commands arrive through a
//! lock-free SPSC ring, finished buffers are handed back through another ring so
//! they are freed on a non-real-time thread, and events go out through a third.

use rtrb::{Consumer, Producer};
use std::collections::VecDeque;

pub const BUS_CUE: u8 = 0;
pub const BUS_ALERT: u8 = 1;

pub const EV_START: u8 = 0;
pub const EV_DONE: u8 = 1;
pub const EV_CUT: u8 = 2;

pub struct Voice {
    pub data: Vec<f32>, // interleaved stereo
    pub start: i64,     // absolute frame of data[0]; < 0 means "as soon as possible"
    pub gain: f32,
    pub bus: u8,
    pub tag: u64,
    fade_left: Option<u64>,
    fade_len: u64,
}

impl Voice {
    pub fn new(data: Vec<f32>, start: i64, gain: f32, bus: u8, tag: u64) -> Self {
        Voice {
            data,
            start,
            gain,
            bus,
            tag,
            fade_left: None,
            fade_len: 1,
        }
    }
    fn frames(&self) -> u64 {
        (self.data.len() / 2) as u64
    }
}

struct Speech {
    uid: u64,
    chunks: VecDeque<Vec<f32>>, // mono
    offset: usize,
    ended: bool,
    started: bool,
    fade_left: Option<u64>,
    fade_len: u64,
    gain: f32,
    pending: i64,
}

pub enum Cmd {
    Play(Voice),
    Stop {
        tag: u64,
        fade: u64,
    },
    SpBegin {
        uid: u64,
        gain: f32,
    },
    SpFeed {
        uid: u64,
        data: Vec<f32>,
    },
    SpEnd {
        uid: u64,
    },
    SpCancel {
        uid: u64,
        fade: u64,
    }, // uid 0 = whatever is playing
    Params {
        master: f32,
        cue: f32,
        speech: f32,
        duck: f32,
        muted: bool,
    },
}

#[derive(Default, Clone, Copy)]
pub struct BlockStats {
    pub limited: bool,
    pub peak: f32,
    pub voices: usize,
    pub speaking: bool,
}

pub struct Mixer {
    #[allow(dead_code)]
    pub rate: u32,
    pub now: u64,
    pub master: f32,
    pub ceiling: f32,
    pub cue_gain: f32,
    pub speech_gain: f32,
    pub duck: f32,
    pub muted: bool,
    voices: Vec<Voice>,
    speech: Option<Speech>,
    duck_env: f32,
    lim_env: f32,
    release: f64,
    cue: Vec<f32>,
    alert: Vec<f32>,
    sp: Vec<f32>,
    events: Producer<(u8, u64)>,
    garbage: Producer<Vec<f32>>,
    pub speaking: bool,
}

impl Mixer {
    pub fn new(
        rate: u32,
        master: f32,
        ceiling: f32,
        cue_gain: f32,
        speech_gain: f32,
        duck: f32,
        events: Producer<(u8, u64)>,
        garbage: Producer<Vec<f32>>,
    ) -> Self {
        let cap = 16384;
        Mixer {
            rate,
            now: 0,
            master,
            ceiling,
            cue_gain,
            speech_gain,
            duck,
            muted: false,
            voices: Vec::with_capacity(512),
            speech: None,
            duck_env: 1.0,
            lim_env: 1.0,
            release: (-1.0 / (0.25 * rate as f64)).exp(),
            cue: vec![0.0; cap],
            alert: vec![0.0; cap],
            sp: vec![0.0; cap / 2],
            events,
            garbage,
            speaking: false,
        }
    }

    fn recycle(&mut self, v: Vec<f32>) {
        // hand the buffer back to a non-RT thread to free; if that ring is full, drop here
        let _ = self.garbage.push(v);
    }

    fn emit(&mut self, kind: u8, uid: u64) {
        let _ = self.events.push((kind, uid));
    }

    pub fn apply(&mut self, cmds: &mut Consumer<Cmd>) {
        while let Ok(cmd) = cmds.pop() {
            match cmd {
                Cmd::Play(mut v) => {
                    if v.start < self.now as i64 {
                        v.start = self.now as i64; // late or ASAP: play whole, never chop the attack
                    }
                    if self.voices.len() < self.voices.capacity() {
                        self.voices.push(v);
                    } else {
                        self.recycle(v.data);
                    }
                }
                Cmd::Stop { tag, fade } => {
                    for v in self.voices.iter_mut() {
                        if v.tag == tag && v.fade_left.is_none() {
                            v.fade_left = Some(fade.max(1));
                            v.fade_len = fade.max(1);
                        }
                    }
                }
                Cmd::SpBegin { uid, gain } => {
                    if self.speech.is_some() && !self.speech_finished() {
                        self.finish_speech(EV_CUT);
                    } else if let Some(old) = self.speech.take() {
                        for c in old.chunks {
                            self.recycle(c);
                        }
                    }
                    self.speech = Some(Speech {
                        uid,
                        chunks: VecDeque::with_capacity(64),
                        offset: 0,
                        ended: false,
                        started: false,
                        fade_left: None,
                        fade_len: 1,
                        gain,
                        pending: 0,
                    });
                }
                Cmd::SpFeed { uid, data } => {
                    let ok =
                        matches!(&self.speech, Some(sp) if sp.uid == uid && sp.fade_left.is_none());
                    if ok {
                        let sp = self.speech.as_mut().unwrap();
                        sp.pending += data.len() as i64;
                        sp.chunks.push_back(data);
                    } else {
                        self.recycle(data);
                    }
                }
                Cmd::SpEnd { uid } => {
                    if let Some(sp) = self.speech.as_mut() {
                        if sp.uid == uid {
                            sp.ended = true;
                        }
                    }
                }
                Cmd::SpCancel { uid, fade } => {
                    let hit = matches!(&self.speech, Some(sp) if (uid == 0 || sp.uid == uid) && sp.fade_left.is_none());
                    if hit {
                        let started = self.speech.as_ref().unwrap().started;
                        if !started {
                            self.finish_speech(EV_CUT);
                        } else {
                            let sp = self.speech.as_mut().unwrap();
                            sp.fade_left = Some(fade.max(1));
                            sp.fade_len = fade.max(1);
                        }
                    }
                }
                Cmd::Params {
                    master,
                    cue,
                    speech,
                    duck,
                    muted,
                } => {
                    self.master = master;
                    self.cue_gain = cue;
                    self.speech_gain = speech;
                    self.duck = duck;
                    self.muted = muted;
                }
            }
        }
    }

    fn speech_finished(&self) -> bool {
        match &self.speech {
            None => true,
            Some(sp) => sp.ended && sp.pending <= 0,
        }
    }

    fn finish_speech(&mut self, how: u8) {
        if let Some(sp) = self.speech.take() {
            let uid = sp.uid;
            for c in sp.chunks {
                self.recycle(c);
            }
            self.emit(how, uid);
        }
    }

    /// Mono speech for this block into self.sp[..frames]; returns whether any was produced.
    fn render_speech(&mut self, frames: usize) -> bool {
        if self.speech.is_none() {
            return false;
        }
        for x in self.sp[..frames].iter_mut() {
            *x = 0.0;
        }
        let mut filled = 0usize;
        {
            // disjoint field borrows: finished chunks go straight to the garbage ring (no alloc)
            let garbage = &mut self.garbage;
            let buf = &mut self.sp;
            let sp = self.speech.as_mut().unwrap();
            while filled < frames && !sp.chunks.is_empty() {
                let c = &sp.chunks[0];
                let take = (frames - filled).min(c.len() - sp.offset);
                buf[filled..filled + take].copy_from_slice(&c[sp.offset..sp.offset + take]);
                filled += take;
                sp.offset += take;
                sp.pending -= take as i64;
                if sp.offset >= sp.chunks[0].len() {
                    if let Some(done) = sp.chunks.pop_front() {
                        let _ = garbage.push(done);
                    }
                    sp.offset = 0;
                }
            }
        }
        let (started_now, uid) = {
            let sp = self.speech.as_mut().unwrap();
            let s = filled > 0 && !sp.started;
            if s {
                sp.started = true;
            }
            (s, sp.uid)
        };
        if started_now {
            self.emit(EV_START, uid);
        }
        let gain = self.speech.as_ref().unwrap().gain;
        if let Some(fl) = self.speech.as_ref().unwrap().fade_left {
            let len = self.speech.as_ref().unwrap().fade_len as f64;
            let n = (frames as u64).min(fl) as usize;
            for i in 0..n {
                let r = (fl as f64 - i as f64) / len; // linspace(fl/len, (fl-n)/len, n, endpoint=False)
                self.sp[i] *= r as f32;
            }
            for x in self.sp[n..frames].iter_mut() {
                *x = 0.0;
            }
            let left = fl - n as u64;
            self.speech.as_mut().unwrap().fade_left = Some(left);
            if left == 0 {
                self.finish_speech(EV_CUT);
            }
            for x in self.sp[..frames].iter_mut() {
                *x *= gain;
            }
            return true;
        }
        if self.speech_finished() {
            self.finish_speech(EV_DONE);
        }
        if filled > 0 {
            for x in self.sp[..frames].iter_mut() {
                *x *= gain;
            }
            true
        } else {
            false
        }
    }

    /// Render `out.len() / 2` stereo frames (interleaved).
    pub fn render(&mut self, out: &mut [f32], cmds: &mut Consumer<Cmd>) -> BlockStats {
        let frames = out.len() / 2;
        if frames * 2 > self.cue.len() {
            self.cue.resize(frames * 2, 0.0);
            self.alert.resize(frames * 2, 0.0);
            self.sp.resize(frames, 0.0);
        }
        self.apply(cmds);
        for x in self.cue[..frames * 2].iter_mut() {
            *x = 0.0;
        }
        for x in self.alert[..frames * 2].iter_mut() {
            *x = 0.0;
        }
        let now = self.now as i64;
        let end = now + frames as i64;
        let mut i = 0;
        while i < self.voices.len() {
            let n = self.voices[i].frames() as i64;
            let start = self.voices[i].start;
            if start >= end {
                i += 1;
                continue;
            }
            let a = now.max(start);
            let b = end.min(start + n);
            if b <= a {
                let v = self.voices.swap_remove(i);
                self.recycle(v.data);
                continue;
            }
            let v = &mut self.voices[i];
            let dst = if v.bus == BUS_ALERT {
                &mut self.alert
            } else {
                &mut self.cue
            };
            let k_total = (b - a) as usize;
            let mut drop_now = false;
            let (fade_n, fl, flen) = match v.fade_left {
                Some(fl) => ((k_total as u64).min(fl) as usize, fl, v.fade_len as f64),
                None => (usize::MAX, 0, 1.0),
            };
            for j in 0..k_total {
                let src = ((a - start) as usize + j) * 2;
                let o = ((a - now) as usize + j) * 2;
                let mut g = v.gain;
                if v.fade_left.is_some() {
                    if j < fade_n {
                        g *= ((fl as f64 - j as f64) / flen) as f32;
                    } else {
                        g = 0.0;
                    }
                }
                dst[o] += v.data[src] * g;
                dst[o + 1] += v.data[src + 1] * g;
            }
            if let Some(fl) = v.fade_left {
                let left = fl - fade_n as u64;
                v.fade_left = Some(left);
                if left == 0 {
                    drop_now = true;
                }
            }
            if drop_now || start + n <= end {
                let v = self.voices.swap_remove(i);
                self.recycle(v.data);
                continue;
            }
            i += 1;
        }
        // swap_remove reorders voices; mixing is a sum, so order does not matter

        let has_speech = self.render_speech(frames);
        self.speaking = has_speech || (self.speech.is_some() && !self.speech_finished());
        let target = if self.speaking { self.duck } else { 1.0 };
        let d0 = self.duck_env;
        let d1 = d0 + (target - d0) * if target < d0 { 0.35 } else { 0.08 };
        self.duck_env = d1;
        let cue_gain = if self.muted { 0.0 } else { self.cue_gain };
        let step = if frames > 1 {
            (d1 - d0) as f64 / (frames - 1) as f64
        } else {
            0.0
        };
        let mut peak = 0.0f32;
        for f in 0..frames {
            let duck = (d0 as f64 + step * f as f64) as f32; // linspace(d0, d1, frames)
            let s = if has_speech {
                self.sp[f] * self.speech_gain
            } else {
                0.0
            };
            for c in 0..2 {
                let x = (self.cue[f * 2 + c] * (duck * cue_gain) + self.alert[f * 2 + c] + s)
                    * self.master;
                out[f * 2 + c] = x;
                peak = peak.max(x.abs());
            }
        }
        // limiter: instant attack ramped across the block, ~250 ms release
        let want = if peak > 0.0 {
            (self.ceiling / peak).min(1.0)
        } else {
            1.0
        };
        let g0 = self.lim_env;
        let limited;
        let g1 = if want < g0 {
            limited = true;
            want
        } else {
            limited = false;
            let r = 1.0 - (1.0 - g0 as f64) * self.release.powi(frames as i32);
            (r as f32).min(want)
        };
        self.lim_env = g1;
        let mut out_peak = 0.0f32;
        let lstep = if frames > 1 {
            (g1 - g0) as f64 / (frames - 1) as f64
        } else {
            0.0
        };
        let apply = g0 != 1.0 || g1 != 1.0;
        for f in 0..frames {
            let g = if apply {
                (g0 as f64 + lstep * f as f64) as f32
            } else {
                1.0
            };
            for c in 0..2 {
                let x = (out[f * 2 + c] * g).clamp(-self.ceiling, self.ceiling);
                out[f * 2 + c] = x;
                out_peak = out_peak.max(x.abs());
            }
        }
        self.now = end as u64;
        BlockStats {
            limited,
            peak: out_peak,
            voices: self.voices.len(),
            speaking: self.speaking,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rig() -> (Mixer, Producer<Cmd>, Consumer<Cmd>, Consumer<(u8, u64)>) {
        let (ep, ec) = rtrb::RingBuffer::new(1024);
        let (gp, _gc) = rtrb::RingBuffer::new(1024);
        let (cp, cc) = rtrb::RingBuffer::new(1024);
        std::mem::forget(_gc);
        (
            Mixer::new(48000, 0.9, 0.89, 0.55, 1.0, 0.45, ep, gp),
            cp,
            cc,
            ec,
        )
    }

    #[test]
    fn scheduled_click_lands_on_exact_frame() {
        let (mut m, mut cp, mut cc, _) = rig();
        let mut d = vec![0.0f32; 20];
        d[0] = 0.5;
        d[1] = 0.5;
        cp.push(Cmd::Play(Voice::new(d, 1234, 1.0, BUS_ALERT, 1)))
            .ok();
        let mut out = vec![0.0f32; 4096 * 2];
        m.render(&mut out, &mut cc);
        let hits: Vec<usize> = (0..4096).filter(|&f| out[f * 2].abs() > 0.1).collect();
        assert_eq!(hits, vec![1234]);
    }

    #[test]
    fn limiter_holds_ceiling() {
        let (mut m, mut cp, mut cc, _) = rig();
        for t in 0..60 {
            cp.push(Cmd::Play(Voice::new(
                vec![1.0; 9600],
                -1,
                1.0,
                BUS_ALERT,
                t,
            )))
            .ok();
        }
        let mut out = vec![0.0f32; 1024];
        for _ in 0..10 {
            m.render(&mut out, &mut cc);
            assert!(out.iter().all(|x| x.abs() <= 0.89 + 1e-6));
        }
    }

    #[test]
    fn speech_events_start_then_done() {
        let (mut m, mut cp, mut cc, mut ec) = rig();
        cp.push(Cmd::SpBegin { uid: 7, gain: 1.0 }).ok();
        cp.push(Cmd::SpFeed {
            uid: 7,
            data: vec![0.2; 1000],
        })
        .ok();
        cp.push(Cmd::SpEnd { uid: 7 }).ok();
        let mut out = vec![0.0f32; 1024];
        for _ in 0..4 {
            m.render(&mut out, &mut cc);
        }
        let evs: Vec<(u8, u64)> = std::iter::from_fn(|| ec.pop().ok()).collect();
        assert_eq!(evs, vec![(EV_START, 7), (EV_DONE, 7)]);
        assert!(!m.speaking);
    }
}
