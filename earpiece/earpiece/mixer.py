"""Real-time mixer. One instance renders everything the wearer hears.

Threading model
  * render() runs on the audio device thread (PortAudio callback) or, in tests and
    offline rendering, wherever it is called. It must never block: it only mixes
    buffers that were fully prepared elsewhere.
  * Every other thread talks to it through a lock-free command deque
    (deque.append / popleft are atomic under the GIL).

Buses
  * cue     : sonar pips, beacon ticks, status earcons. Ducked while speech plays.
  * alert   : critical alarms. Never ducked, never limited below audibility.
  * speech  : one utterance at a time, streamed in as chunks (mono, centred).

Sample-accurate scheduling: callers pass `at` in absolute samples (engine.now
is the clock), so beep rhythm is exact even though the control threads are
jittery. A master peak limiter keeps the output under the ceiling.
"""

from __future__ import annotations

import collections
import math
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Voice:
    data: np.ndarray            # (n, 2) float32
    start: int                  # absolute sample index of data[0]
    gain: float
    bus: str
    tag: str | None
    fade_left: int | None = None    # samples remaining in a fade-out, None = not fading
    fade_len: int = 1


@dataclass
class _Speech:
    uid: str
    chunks: collections.deque = field(default_factory=collections.deque)
    offset: int = 0             # read position inside chunks[0]
    ended: bool = False         # producer finished
    started_at: int | None = None
    fade_left: int | None = None
    fade_len: int = 1
    gain: float = 1.0
    pending: int = 0            # samples queued but not yet played


class Engine:
    def __init__(self, rate: int = 48000, master_gain: float = 0.9, ceiling: float = 0.89,
                 cue_gain: float = 0.55, speech_gain: float = 1.0, duck: float = 0.45):
        self.rate = rate
        self.now = 0                                   # samples rendered so far
        self.master_gain = master_gain
        self.ceiling = ceiling
        self.cue_gain = cue_gain
        self.speech_gain = speech_gain
        self.duck = duck
        self.cues_muted = False
        self._cmds: collections.deque = collections.deque()
        self._voices: list[_Voice] = []
        self._speech: _Speech | None = None
        self._duck_env = 1.0
        self._lim_env = 1.0
        self._release = math.exp(-1.0 / (0.25 * rate))   # limiter recovers over ~250 ms
        self.speech_events: queue.SimpleQueue = queue.SimpleQueue()   # ("start"|"done"|"cut", uid, t)
        self._speaking = False
        self._lock = threading.Lock()                     # only guards stats snapshots
        self.stats = {"blocks": 0, "max_render_ms": 0.0, "avg_render_ms": 0.0,
                      "voices": 0, "limited_blocks": 0, "peak": 0.0}

    # ------------------------------------------------------------------ control API
    def play(self, stereo: np.ndarray, at: int | None = None, bus: str = "cue",
             gain: float = 1.0, tag: str | None = None) -> None:
        """Schedule a prepared stereo buffer. at=None means 'as soon as possible'."""
        if stereo.ndim == 1:
            stereo = np.repeat(stereo[:, None], 2, axis=1)
        self._cmds.append(("play", _Voice(np.ascontiguousarray(stereo, dtype=np.float32),
                                          -1 if at is None else int(at), float(gain), bus, tag)))

    def stop(self, tag: str, fade_s: float = 0.03) -> None:
        self._cmds.append(("stop", tag, max(1, int(fade_s * self.rate))))

    def speech_begin(self, uid: str, gain: float = 1.0) -> None:
        """Start a new utterance. Any utterance still playing is cut with a short fade."""
        self._cmds.append(("sp_begin", uid, gain))

    def speech_feed(self, uid: str, mono: np.ndarray) -> None:
        if mono.size:
            self._cmds.append(("sp_feed", uid, np.ascontiguousarray(mono, dtype=np.float32)))

    def speech_end(self, uid: str) -> None:
        self._cmds.append(("sp_end", uid))

    def speech_cancel(self, uid: str | None = None, fade_s: float = 0.04) -> None:
        """Cut the current utterance (if uid matches, or any if uid is None)."""
        self._cmds.append(("sp_cancel", uid, max(1, int(fade_s * self.rate))))

    @property
    def speaking(self) -> bool:
        return self._speaking

    def seconds(self, samples: int) -> float:
        return samples / self.rate

    def samples(self, seconds: float) -> int:
        return int(round(seconds * self.rate))

    # ------------------------------------------------------------------ render
    def _apply_commands(self) -> None:
        while True:
            try:
                cmd = self._cmds.popleft()
            except IndexError:
                return
            kind = cmd[0]
            if kind == "play":
                v = cmd[1]
                if v.start < self.now:      # ASAP, or scheduled time already passed: play now,
                    v.start = self.now      # from the beginning (never chop a cue's attack)
                self._voices.append(v)
            elif kind == "stop":
                for v in self._voices:
                    if v.tag == cmd[1] and v.fade_left is None:
                        v.fade_left = v.fade_len = cmd[2]
            elif kind == "sp_begin":
                if self._speech is not None and not self._speech_finished():
                    self._finish_speech("cut")
                self._speech = _Speech(cmd[1], gain=cmd[2])
            elif kind == "sp_feed":
                sp = self._speech
                if sp is not None and sp.uid == cmd[1] and sp.fade_left is None:
                    sp.chunks.append(cmd[2])
                    sp.pending += cmd[2].size
            elif kind == "sp_end":
                sp = self._speech
                if sp is not None and sp.uid == cmd[1]:
                    sp.ended = True
            elif kind == "sp_cancel":
                sp = self._speech
                if sp is not None and (cmd[1] is None or sp.uid == cmd[1]) and sp.fade_left is None:
                    if sp.started_at is None:
                        self._finish_speech("cut")
                    else:
                        sp.fade_left = sp.fade_len = cmd[2]

    def _speech_finished(self) -> bool:
        sp = self._speech
        return sp is None or (sp.ended and sp.pending <= 0)

    def _finish_speech(self, how: str) -> None:
        sp = self._speech
        if sp is not None:
            self.speech_events.put((how, sp.uid, time.monotonic()))
        self._speech = None

    def _render_speech(self, frames: int) -> np.ndarray | None:
        sp = self._speech
        if sp is None:
            return None
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        while filled < frames and sp.chunks:
            c = sp.chunks[0]
            take = min(frames - filled, c.size - sp.offset)
            out[filled:filled + take] = c[sp.offset:sp.offset + take]
            filled += take
            sp.offset += take
            sp.pending -= take
            if sp.offset >= c.size:
                sp.chunks.popleft()
                sp.offset = 0
        if filled and sp.started_at is None:
            sp.started_at = self.now
            self.speech_events.put(("start", sp.uid, time.monotonic()))
        if sp.fade_left is not None:
            n = min(frames, sp.fade_left)
            ramp = np.linspace(sp.fade_left / sp.fade_len, (sp.fade_left - n) / sp.fade_len,
                               n, endpoint=False, dtype=np.float32)
            out[:n] *= ramp
            out[n:] = 0.0
            sp.fade_left -= n
            if sp.fade_left <= 0:
                self._finish_speech("cut")
            return out * sp.gain
        if self._speech_finished():
            self._finish_speech("done")
        return out * sp.gain if filled else None

    def render(self, frames: int) -> np.ndarray:
        t0 = time.perf_counter()
        self._apply_commands()
        cue = np.zeros((frames, 2), dtype=np.float32)
        alert = np.zeros((frames, 2), dtype=np.float32)
        end = self.now + frames
        keep: list[_Voice] = []
        for v in self._voices:
            n = v.data.shape[0]
            if v.start >= end:
                keep.append(v)
                continue
            a = max(self.now, v.start)              # absolute range this block covers
            b = min(end, v.start + n)
            if b <= a:
                continue                            # finished (or started in the past and over)
            seg = v.data[a - v.start:b - v.start] * v.gain
            if v.fade_left is not None:
                k = min(b - a, v.fade_left)
                ramp = np.linspace(v.fade_left / v.fade_len, (v.fade_left - k) / v.fade_len,
                                   k, endpoint=False, dtype=np.float32)[:, None]
                seg = seg.copy()
                seg[:k] *= ramp
                seg[k:] = 0.0
                v.fade_left -= k
                if v.fade_left <= 0:
                    (alert if v.bus == "alert" else cue)[a - self.now:b - self.now] += seg
                    continue
            (alert if v.bus == "alert" else cue)[a - self.now:b - self.now] += seg
            if v.start + n > end:
                keep.append(v)
        self._voices = keep

        speech = self._render_speech(frames)
        self._speaking = speech is not None or (self._speech is not None and not self._speech_finished())
        # duck the cue bus smoothly while speech is audible
        target = self.duck if self._speaking else 1.0
        d0 = self._duck_env
        d1 = d0 + (target - d0) * (0.35 if target < d0 else 0.08)
        self._duck_env = d1
        duck = np.linspace(d0, d1, frames, dtype=np.float32)[:, None]
        cue_gain = 0.0 if self.cues_muted else self.cue_gain
        mix = cue * (duck * cue_gain) + alert
        if speech is not None:
            mix += (speech * self.speech_gain)[:, None]
        mix *= self.master_gain

        # peak limiter: instant attack (ramped across the block), ~250 ms release
        peak = float(np.max(np.abs(mix))) if frames else 0.0
        want = min(1.0, self.ceiling / peak) if peak > 0 else 1.0
        g0 = self._lim_env
        if want < g0:
            g1 = want
            with self._lock:
                self.stats["limited_blocks"] += 1
        else:
            g1 = 1.0 - (1.0 - g0) * (self._release ** frames)
            g1 = min(g1, want)
        self._lim_env = g1
        if g0 != 1.0 or g1 != 1.0:
            mix *= np.linspace(g0, g1, frames, dtype=np.float32)[:, None]
        np.clip(mix, -self.ceiling, self.ceiling, out=mix)   # hard safety net (ramp start)

        self.now = end
        dt = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            s = self.stats
            s["blocks"] += 1
            s["max_render_ms"] = max(s["max_render_ms"], dt)
            s["avg_render_ms"] += (dt - s["avg_render_ms"]) * 0.02
            s["voices"] = len(self._voices)
            s["peak"] = max(s["peak"] * 0.95, float(np.max(np.abs(mix))) if frames else 0.0)
        return mix

    def snapshot_stats(self) -> dict:
        with self._lock:
            return dict(self.stats)
