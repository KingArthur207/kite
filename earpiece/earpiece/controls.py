"""Hands-on controls: AirPods stem presses (macOS) and a keyboard fallback.

AirPods decode presses on the bud and send Bluetooth media commands:
    press once  -> play/pause      -> TALK      (push-to-talk; press again to finish early)
    press twice -> next track      -> REPEAT    (last instruction)
    press 3x    -> previous track  -> PAUSE     (toggle guidance; safety warnings stay on)
On macOS these reach the "now playing" app through MPRemoteCommandCenter. We
claim now-playing and register handlers; verified on this Mac with synthetic
media keys from a plain Python script (no .app bundle), provided the MAIN
thread pumps the Cocoa run loop, which run_main_loop() does. A double press
can arrive as PAUSE then NEXT a few ms apart, so presses inside a short window
are collapsed and the strongest wins (previous > next > play/pause).
Another media app that starts playing can take now-playing away; we re-assert
it every few seconds.

Barge-in does not wait for that window: the FIRST press of any burst calls
`on_press` straight away (the app cuts non-safety speech there), and the
gesture itself is classified 0.3 s later. Whatever the wearer meant (talk,
repeat, pause), they wanted Kite to stop talking first.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Callable

TALK, REPEAT, PAUSE = "talk", "repeat", "pause"


class Gestures:
    def __init__(self, on_gesture: Callable[[str], None], debounce_s: float = 0.3, log=print):
        self.on_gesture = on_gesture
        self.debounce_s = debounce_s
        self.log = log
        self._raw: queue.Queue = queue.Queue()
        self.stem_ok = False
        self._targets = []
        self.on_press: Callable[[], None] | None = None    # immediate, before classification
        self.press_to_cut_ms: list[float] = []              # raw press -> on_press returned
        threading.Thread(target=self._classify, name="gestures", daemon=True).start()

    # -------------------------------------------------------------- raw input
    def raw(self, kind: str) -> None:
        """kind: 'toggle'|'play'|'pause'|'next'|'previous'. Thread-safe."""
        self._raw.put((kind, time.monotonic()))

    def stop(self) -> None:
        self._raw.put((None, 0.0))

    def _classify(self) -> None:
        while True:
            kind, t_press = self._raw.get()
            if kind is None:                       # stop()
                return
            if self.on_press is not None:
                try:
                    self.on_press()
                except Exception as e:
                    self.log(f"[controls] press handler failed: {type(e).__name__}: {e}")
                self.press_to_cut_ms.append((time.monotonic() - t_press) * 1000)
                del self.press_to_cut_ms[:-20]
            kinds = {kind}
            t_end = time.monotonic() + self.debounce_s
            while True:
                left = t_end - time.monotonic()
                if left <= 0:
                    break
                try:
                    k = self._raw.get(timeout=left)[0]
                except queue.Empty:
                    break
                if k is None:
                    return
                kinds.add(k)
            if "previous" in kinds:
                g = PAUSE
            elif "next" in kinds:
                g = REPEAT
            else:
                g = TALK
            try:
                self.on_gesture(g)
            except Exception as e:
                self.log(f"[controls] gesture handler failed: {type(e).__name__}: {e}")

    # -------------------------------------------------------------- macOS stems
    def enable_stem(self) -> bool:
        """Register for AirPods / media-key commands. Must be called on the main thread."""
        if sys.platform != "darwin":
            return False
        try:
            from MediaPlayer import (MPMediaItemPropertyArtist, MPMediaItemPropertyPlaybackDuration,
                                     MPMediaItemPropertyTitle, MPNowPlayingInfoCenter,
                                     MPNowPlayingInfoPropertyElapsedPlaybackTime,
                                     MPNowPlayingInfoPropertyPlaybackRate, MPNowPlayingPlaybackStatePlaying,
                                     MPRemoteCommandCenter, MPRemoteCommandHandlerStatusSuccess)
        except ImportError:
            self.log("[controls] pyobjc MediaPlayer not installed: stem presses disabled "
                     "(pip install 'kite-earpiece[mac]')")
            return False
        cc = MPRemoteCommandCenter.sharedCommandCenter()

        def handler(kind):
            def h(_event):
                self.raw(kind)
                return MPRemoteCommandHandlerStatusSuccess
            return h

        for kind, cmd in (("toggle", cc.togglePlayPauseCommand()), ("play", cc.playCommand()),
                          ("pause", cc.pauseCommand()), ("next", cc.nextTrackCommand()),
                          ("previous", cc.previousTrackCommand())):
            cmd.setEnabled_(True)
            self._targets.append(cmd.addTargetWithHandler_(handler(kind)))
        info = MPNowPlayingInfoCenter.defaultCenter()
        self._now_playing = (info, {
            MPMediaItemPropertyTitle: "Kite guidance", MPMediaItemPropertyArtist: "Kite",
            MPMediaItemPropertyPlaybackDuration: 36000.0, MPNowPlayingInfoPropertyElapsedPlaybackTime: 0.0,
            MPNowPlayingInfoPropertyPlaybackRate: 1.0}, MPNowPlayingPlaybackStatePlaying)
        self._assert_now_playing()
        self.stem_ok = True
        self.log("[controls] AirPods stem presses: once = talk, twice = repeat, three times = pause")
        return True

    def _assert_now_playing(self) -> None:
        info, meta, state = self._now_playing
        info.setNowPlayingInfo_(meta)
        info.setPlaybackState_(state)

    def run_main_loop(self, stop: threading.Event) -> None:
        """Block the main thread: pump the Cocoa run loop (needed for stem events) until stop."""
        if not self.stem_ok:
            while not stop.wait(0.2):
                pass
            return
        from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop
        last = time.monotonic()
        rl = NSRunLoop.currentRunLoop()
        while not stop.is_set():
            rl.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05))
            if time.monotonic() - last > 3.0:       # keep now-playing (other apps can steal it)
                last = time.monotonic()
                try:
                    self._assert_now_playing()
                except Exception:
                    pass

    # -------------------------------------------------------------- keyboard
    def enable_keyboard(self, on_key: Callable[[str], None]) -> bool:
        """Line-based stdin: works in any terminal, needs no permissions."""
        if not sys.stdin or not sys.stdin.isatty():
            return False

        def read():
            for line in sys.stdin:
                k = line.strip().lower()
                if k == "":
                    self.raw("toggle")
                elif k == "r":
                    self.raw("next")
                elif k == "p":
                    self.raw("previous")
                else:
                    try:
                        on_key(k)
                    except Exception as e:
                        self.log(f"[controls] key {k!r} failed: {e}")

        threading.Thread(target=read, name="keyboard", daemon=True).start()
        return True


def post_media_key(kind: str) -> None:
    """Simulate a media key (= an AirPods stem press) system-wide. For self-tests.
    Needs Accessibility permission for the terminal on macOS."""
    import AppKit
    import Quartz
    key = {"play": 16, "next": 17, "previous": 18}[kind]
    for down in (True, False):
        flags = 0xA00 if down else 0xB00
        data1 = (key << 16) | ((0xA if down else 0xB) << 8)
        ev = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            AppKit.NSEventTypeSystemDefined, (0, 0), flags, 0, 0, None, 8, data1, -1)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev.CGEvent())
