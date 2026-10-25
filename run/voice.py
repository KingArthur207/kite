"""Voice commands for the kite window (run/epic.py --voice): "kite, take off" is typed into the console for you.

Off unless --voice is given: without it nothing here is imported. With it, the AirPods microphone streams to
Cartesia Ink (speech to text) through the earpiece's hands-free Listener, starting once the drone is ready (its first
video frame). A sentence acts only if it contains the wake word; the words after it become one console command,
which is then submitted exactly as if it had been typed, so it shows in the console and goes through the same checks:

    kite take off                      -> takeoff
    kite land / come down / emergency  -> land         (cutting the motors stays on the keyboard: e)
    kite hold / stop / wait / hover    -> hold
    kite follow me / come back         -> follow me
    kite resume / continue             -> resume
    kite status / battery              -> status
    kite take me to [place]            -> destination: [place]   ("the nearest X" works too)
    kite one / number two ...          -> 1, 2 ...     (answers "Which one?")
    kite body cam / drone view         -> bodycam, drone
    kite yolo on / segmentation off    -> yolo on, seg off

"land" and "hold" act on Ink's first words (about half a second after you stop talking), the rest at the end of
the sentence. Nothing is spoken back. Needs CARTESIA_API_KEY (environment or the repo .env) and
pip install -e ".[voice]".
"""

from __future__ import annotations

import importlib.util
import os
import queue
import re
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The wake word as Ink writes it. Anywhere in the sentence only the close spellings count: chatter before "kite
# land" must not hide it, and "kit" or "Kate" in the middle of someone else's sentence must not wake it.
_WAKE_START = re.compile(r"^\W*(?:(?:hey|okay|ok|hi)\W+)?(?:kite|kyte|kight|kate|kit)\b\W*", re.I)
_WAKE_ANY = re.compile(r"\b(?:kite|kyte|kight)\b\W*", re.I)
_GO_TO = re.compile(r"^(?:please\s+)?(?:(?:take|bring|walk|guide|get)\s+me\s+(?:to|towards?)|(?:go|navigate|head)\s+"
                    r"(?:to|towards?))\s+(?P<place>.+?)[\s.!?,]*$", re.I)

_WORDS = {
    "takeoff": ("take off", "takeoff", "lift off", "liftoff"),
    "land": ("land", "lands", "landing", "land now", "come down", "emergency", "emergency land",
             "emergency landing"),
    "hold": ("hold", "hold on", "hold position", "stop", "wait", "hover", "pause", "stay"),
    "follow me": ("follow", "follow me", "come back"),
    "resume": ("resume", "continue", "keep going"),
    "status": ("status", "battery"),
    "bodycam": ("bodycam", "body cam", "body camera"),
    "drone": ("drone", "drone view", "drone camera"),
    "yolo on": ("yolo on",),
    "yolo off": ("yolo off",),
    "seg on": ("seg on", "segmentation on"),
    "seg off": ("seg off", "segmentation off"),
}
_PHRASES = {phrase: cmd for cmd, phrases in _WORDS.items() for phrase in phrases}
_NUMBERS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5"}
QUICK = ("land", "hold")  # acted on from Ink's first words: both are safe to send twice
EARLY_S = 12.0  # longer than any sentence (the listener ends one at 8 s): an older early command is forgotten


def _command(rest: str) -> str | None:
    """The console command for the words after the wake word, or None."""
    m = _GO_TO.match(rest.strip())
    if m:
        place = m.group("place").strip()
        return f"destination: {place}" if place else None
    words = re.sub(r"[^\w\s]", " ", rest.lower().replace("take-off", "takeoff")).split()
    while words and words[0] in ("please", "and", "now"):
        words = words[1:]
    while words and words[-1] in ("please", "now"):
        words = words[:-1]
    text = " ".join(words)
    if text in _PHRASES:
        return _PHRASES[text]
    num = text.removeprefix("number ").removeprefix("the ").removesuffix(" one")
    if num.isdigit() and 0 < int(num) < 10:
        return num
    return _NUMBERS.get(num)


def parse_command(text: str) -> str | None:
    """'Kite, take off.' -> 'takeoff'. None without the wake word or without a command after it.

    Every wake word is tried, the last one first (the newest thing said), then the one that starts the sentence."""
    if not isinstance(text, str) or not text.strip():
        return None
    starts = [m.end() for m in _WAKE_ANY.finditer(text)]
    head = _WAKE_START.match(text)
    if head:
        starts.insert(0, head.end())
    for at in reversed(starts):
        cmd = _command(text[at:])
        if cmd is not None:
            return cmd
    return None


def missing() -> str | None:
    """Why voice cannot start on this machine, or None."""
    need = [m for m in ("sounddevice", "websockets", "numpy") if importlib.util.find_spec(m) is None]
    if need:
        return f"missing {', '.join(need)}: pip install -e \".[voice]\""
    return None


def _earpiece_listen():
    """The earpiece's Listener, from the repo even when kite-earpiece is not installed."""
    path = os.path.join(ROOT, "earpiece")
    if path not in sys.path:
        sys.path.insert(0, path)
    from earpiece import config, listen

    return config, listen


class VoiceCommands:
    """Hands-free Cartesia Ink on the AirPods microphone. on_command(text, heard) gets each console command once."""

    def __init__(self, on_command, log, mic: str = "airpods"):
        self.on_command = on_command
        self.log = log
        self.mic = mic
        self.listener = None
        self._last_err = 0.0

    def start(self) -> bool:
        why = missing()
        if why:
            self.log(f"voice off: {why}", "error")
            return False
        try:
            config, listen = _earpiece_listen()
        except Exception as e:
            self.log(f"voice off: cannot load the earpiece listener ({type(e).__name__}: {e})", "error")
            return False
        cfg = config.Cfg()
        key = cfg.cartesia_key
        if not key:
            self.log("voice off: no CARTESIA_API_KEY (set it in the repo .env, and pip install python-dotenv)",
                     "error")
            return False
        lcfg = cfg.listen
        lcfg.input_substr = self.mic
        lcfg.hands_free = True
        try:
            _idx, name = listen.pick_input(self.mic)
        except Exception as e:
            self.log(f"voice off: no microphone ({type(e).__name__}: {e})", "error")
            return False
        if self.mic and self.mic.lower() not in (name or "").lower():
            self.log(f"voice: no microphone named like {self.mic!r}; using {name}", "warn")
        self.listener = _make_listener(listen)(self, lcfg, key)
        self.listener.start_hands_free(lambda: False)  # output is not spoken here: the mic never needs gating
        self.log(f"voice: listening on {name} for 'kite ...' (Cartesia {lcfg.stt_model})", "info")
        return True

    def stop(self) -> None:
        """Stop listening and let the microphone close (the listener checks every 0.3 s), at most 2 s."""
        if self.listener is not None:
            self.listener.stop()
            thread = getattr(self.listener, "_hf_thread", None)
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)

    def _listen_log(self, text: str) -> None:
        """The listener's own lines. A dead network retries every second: show that at most every 10 s."""
        if "error" in text or "failed" in text:
            now = time.monotonic()
            if now - self._last_err < 10.0:
                return
            self._last_err = now
            self.log(f"voice: {text.removeprefix('[listen] ')}", "warn")


def _make_listener(listen):
    """A Listener that hands commands back instead of asking the earpiece brain. Built on first use, so importing
    this module never imports the earpiece."""

    class CommandListener(listen.Listener):
        def __init__(self, owner: VoiceCommands, cfg, key: str):
            super().__init__(cfg, key, on_text=lambda _t: None, cue=lambda _c: None,
                             on_listening=lambda _on: None, say=lambda *_a, **_k: None, log=owner._listen_log)
            self.owner = owner
            self._early: tuple[str, float] | None = None  # land / hold already sent for the sentence being said

        def _partial(self, text: str) -> None:
            """Ink's words so far, when the sentence starts with the wake word: land and hold go now."""
            cmd = parse_command(text)
            now = time.monotonic()
            if cmd in QUICK and (self._early is None or now - self._early[1] > EARLY_S):
                self._early = (cmd, now)
                self.owner.on_command(cmd, text)

        def _deliver(self, text: str, require_wake: bool) -> None:
            early, self._early = self._early, None
            if early is not None and time.monotonic() - early[1] > EARLY_S:  # from a sentence that never ended
                early = None
            early = early[0] if early is not None else None
            cmd = parse_command(text)
            if cmd is None:
                if text.strip() and (_WAKE_START.match(text) or _WAKE_ANY.search(text)):
                    self.owner.log(f"voice: heard {text.strip()!r}, no command in it", "warn")
                return
            if cmd == early:  # already sent from the first words
                return
            self.owner.on_command(cmd, text)

    return CommandListener


def attach(window, mic: str = "airpods") -> VoiceCommands:
    """Listen once the drone is ready, and type each command into the window's console on the Tk thread."""
    heard: queue.Queue = queue.Queue()
    voice = VoiceCommands(lambda cmd, text: heard.put((cmd, text)), window.log, mic)
    state = {"started": False}

    def tick():
        if getattr(window, "_closing", False):
            return
        try:
            if not state["started"] and getattr(window.pilot, "last_frame", None) is not None:
                state["started"] = True
                voice.start()
            while True:
                try:
                    cmd, text = heard.get_nowait()
                except queue.Empty:
                    break
                window.log(f"voice: heard {text.strip()!r}", "info")
                type_command(window, cmd)
        except Exception as e:  # voice must never stop the window
            window.log(f"voice: {type(e).__name__}: {e}", "warn")
        window.root.after(50, tick)

    window.log("voice: on. Listening starts once the drone is ready", "info")
    window.root.after(50, tick)
    return voice


def type_command(window, cmd: str) -> None:
    """Submit cmd from the console's input box, as Enter would, and give back what was being typed there."""
    entry = window.entry
    typed = entry.get()
    entry.delete(0, "end")
    entry.insert(0, cmd)
    window._submit()
    entry.delete(0, "end")
    entry.insert(0, typed)


def _main(argv: list[str] | None = None) -> int:
    """python run/voice.py: listen now (no drone, no window) and print what each sentence would type."""
    import argparse

    p = argparse.ArgumentParser(prog="python run/voice.py", description="Try kite's voice commands without "
                                "the drone: prints what each sentence would type into the console.")
    p.add_argument("--voice-mic", default="airpods", help="microphone name to look for (default: airpods)")
    p.add_argument("--list", action="store_true", help="list the microphones and exit")
    a = p.parse_args(argv)
    if a.list:
        import sounddevice as sd

        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"{i:3d}  {d['name']}  ({int(d['default_samplerate'])} Hz)")
        return 0

    def log(text: str, kind: str = "info") -> None:
        print(f"{time.strftime('%H:%M:%S')} {kind:5s} {text}", flush=True)

    voice = VoiceCommands(lambda cmd, text: log(f"heard {text.strip()!r} -> would type: {cmd}", "sent"), log,
                          a.voice_mic)
    if not voice.start():
        return 1
    log("say 'kite take off', 'kite land', 'kite take me to the nearest coffee shop' ... (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()
    return 0


if __name__ == "__main__":
    code = _main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # never wait on an audio or network thread at exit: Ctrl+C always ends this script
