"""kite-earpiece: the AirPods layer of Kite.

    python -m earpiece run                 # the real thing (AirPods, stem presses, network)
    python -m earpiece run --sim walk      # same, fed by a scripted street scenario
    python -m earpiece render walk out.wav # render exactly what the wearer would hear
    python -m earpiece selftest            # check audio, voices, STT, controls, network
    python -m earpiece send '{"type":"say","text":"hello"}'   # poke a running earpiece
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import wave

import numpy as np

from .config import Cfg


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"{_ts()} {msg}", flush=True)


def _feed(app, name: str, stop: threading.Event, loop: bool = False) -> threading.Thread:
    from . import sim
    from .events import parse

    def run():
        while not stop.is_set():
            t0 = time.monotonic()
            for t, m in sim.timeline(name):
                d = t0 + t - time.monotonic()
                if d > 0 and stop.wait(d):
                    return
                app.handle(parse(m))
            if not loop:
                return
            time.sleep(3)

    th = threading.Thread(target=run, name="sim", daemon=True)
    th.start()
    return th


def cmd_run(a) -> int:
    from .app import Earpiece
    cfg = Cfg()
    cfg.net.bind_host = a.bind
    if a.mic:
        cfg.listen.input_substr = a.mic
    cfg.listen.hands_free = a.hands_free
    app = Earpiece(cfg, output="null" if a.null else "device", log=_log, output_device=a.output)
    if a.spacetime or cfg.net.spacetime_enabled:
        from .spacetime import SpacetimeBridge
        app.bridges.append(SpacetimeBridge(source="earpiece", log=_log))
    app.start()
    if not a.no_stem:
        app.gestures.enable_stem()

    def on_key(k: str) -> None:
        if k in ("q", "quit", "exit"):
            app._stop.set()
        elif k == "m":
            app.control("unmute_cues" if app.sonar.muted else "mute_cues", speak=True)
        elif k == "b":
            app.control("beacon_off" if app.beacon.enabled else "beacon_on", speak=True)
        elif k == "d":
            app.heard("what's around me")
        elif k == "e":
            app.ear_check()
        elif k == "s":
            print(json.dumps(app.state(), indent=1, default=str)[:4000])
        elif k.startswith("t "):
            app.heard(k[2:])
        else:
            print("keys: Enter=talk  r=repeat  p=pause  m=beeps  b=beacon  d=describe  "
                  "e=ear check  s=status  t <words>=type a command  q=quit")

    if app.gestures.enable_keyboard(on_key):
        _log("[keys] Enter=talk  r=repeat  p=pause  m=beeps  b=beacon  d=describe  e=ear check  "
             "s=status  t <words>=type a command  q=quit")
    if cfg.listen.hands_free:
        app.listener.start_hands_free(lambda: app.engine.speaking and not app.output.is_airpods)
    if a.sim:
        _feed(app, a.sim, app._stop, loop=a.loop)
    app.run_forever()
    return 0


def cmd_render(a) -> int:
    """Run a scenario through the full app with a recording null output, write a WAV."""
    from . import sim
    from .app import Earpiece
    cfg = Cfg()
    app = Earpiece(cfg, output="null", record=True, network=False, log=_log)
    app.start(prewarm=False, announce=False)
    stop = threading.Event()
    _feed(app, a.scenario, stop)
    dur = a.seconds or sim.duration(a.scenario)
    t0 = time.monotonic()
    while time.monotonic() - t0 < dur:
        time.sleep(0.2)
    stop.set()
    app.speaker.wait_idle(8)
    audio = app.output.recorded()
    app.stop()
    write_wav(a.out, audio, app.engine.rate)
    st = app.state()
    _log(f"[render] {a.out}: {audio.shape[0] / app.engine.rate:.1f}s, peak {np.abs(audio).max():.2f}, "
         f"beeps {app.sonar.stats['beeps']}, criticals {app.sonar.stats['criticals']}, "
         f"spoken {st['speech']['spoken']}, dropped(ttl) {st['speech']['dropped_ttl']}")
    for h in app.speaker.history:
        _log(f"   said [{h['priority']}/{h['result']}] {h['text']}")
    return 0


def write_wav(path: str, stereo: np.ndarray, rate: int) -> None:
    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def cmd_send(a) -> int:
    data = a.json.encode()
    json.loads(data)
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(data, (a.host, a.port))
    return 0


def cmd_selftest(a) -> int:
    from .selftest import run
    return run(play=not a.silent, log=_log)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kite-earpiece", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="run the earpiece")
    r.add_argument("--sim", choices=["walk", "car", "dropout", "crowd"], help="feed a scripted scenario")
    r.add_argument("--loop", action="store_true", help="repeat the scenario")
    r.add_argument("--null", action="store_true", help="no audio device (headless)")
    r.add_argument("--output", help="output device name substring (default: AirPods, else system default)")
    r.add_argument("--mic", help="input device substring ('airpods' to use the AirPods mic)")
    r.add_argument("--hands-free", action="store_true", help="always listen for 'Kite, ...'")
    r.add_argument("--no-stem", action="store_true", help="do not capture media keys / stem presses")
    r.add_argument("--bind", default="127.0.0.1", help="0.0.0.0 to accept events from the LAN")
    r.add_argument("--spacetime", action="store_true", help="mirror events to SpacetimeDB (SPACETIME_HOST)")
    rr = sub.add_parser("render", help="render a scenario to a WAV file")
    rr.add_argument("scenario", choices=["walk", "car", "dropout", "crowd"])
    rr.add_argument("out")
    rr.add_argument("--seconds", type=float)
    s = sub.add_parser("send", help="send one JSON event to a running earpiece")
    s.add_argument("json")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5010)
    t = sub.add_parser("selftest", help="check audio, voices, STT, controls and network")
    t.add_argument("--silent", action="store_true", help="do not play sound")
    a = p.parse_args(argv)
    if a.cmd is None:
        a = p.parse_args(["run", *(argv or sys.argv[1:])])
    return {"run": cmd_run, "render": cmd_render, "send": cmd_send, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
