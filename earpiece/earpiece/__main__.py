"""kite-earpiece: the AirPods layer of Kite.

    python -m earpiece run                 # the real thing (AirPods, stem presses, network)
    python -m earpiece run --sim walk      # same, fed by a scripted street scenario
    python -m earpiece render walk out.wav # render exactly what the wearer would hear
    python -m earpiece render --jsonl events.jsonl out.wav   # same, from a detector/recorded timeline
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


def _feed(app, timeline: list, stop: threading.Event, loop: bool = False) -> threading.Thread:
    from .events import parse

    def run():
        while not stop.is_set():
            t0 = time.monotonic()
            for t, m in timeline:
                d = t0 + t - time.monotonic()
                if d > 0 and stop.wait(d):
                    return
                try:
                    app.handle(parse(m))
                except ValueError as e:          # BadMessage: skip the line, as the UDP transport does
                    _log(f"[feed] skipped {m.get('type')!r} at {t:.2f}s: {e}")
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
    cfg.net.udp_port, cfg.net.ws_port = a.udp_port, a.ws_port
    if a.mic:
        cfg.listen.input_substr = a.mic
    cfg.listen.hands_free = a.hands_free
    if a.no_judgment:
        cfg.judgment = False
    app = Earpiece(cfg, output="null" if a.null else "device", log=_log, output_device=a.output)
    if a.spacetime or cfg.net.spacetime_enabled:
        from .spacetime import CaregiverInbox, SpacetimeBridge
        app.bridges.append(SpacetimeBridge(cfg.net.spacetime_host, cfg.net.spacetime_db,
                                           source="earpiece", log=_log))
    app.start()
    if a.spacetime or cfg.net.spacetime_enabled:
        CaregiverInbox(app.handle, cfg.net.spacetime_host, cfg.net.spacetime_db, log=_log)
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
        from . import sim
        _feed(app, sim.timeline(a.sim), app._stop, loop=a.loop)
    app.run_forever()
    return 0


def cmd_render(a) -> int:
    """Run a scenario or a JSONL timeline through the full app with a recording null output, write a WAV.

    The WAV is the earpiece's real output (sonar, beacon and speech, mixed binaurally) with t=0 at the
    first event. Spoken lines go to <out>.captions.jsonl ({"t", "end", "text", "priority"}, seconds
    from t=0) for video/overlay.py."""
    from . import sim
    from .app import Earpiece
    if bool(a.scenario) == bool(a.jsonl):
        sys.exit("render: give a scenario name or --jsonl FILE (not both)")
    if a.jsonl:
        try:
            timeline = sim.jsonl_timeline(a.jsonl)
        except (OSError, ValueError) as e:
            sys.exit(f"render: {e}")
        if not timeline:
            sys.exit(f"render: {a.jsonl} has no events")
        dur = a.seconds or timeline[-1][0] + 2.0
    else:
        timeline = sim.timeline(a.scenario)
        dur = a.seconds or sim.duration(a.scenario)
    app = Earpiece(Cfg(), output="null", record=True, network=False, log=_log)
    spoken: list[dict] = []
    chained = app.speaker.on_spoken

    def on_spoken(rec: dict) -> None:
        spoken.append(rec)
        if chained is not None:
            chained(rec)

    app.speaker.on_spoken = on_spoken
    app.start(prewarm=False, announce=False)
    stop = threading.Event()
    n0 = sum(len(f) for f in app.output.frames)    # audio rendered before the first event is trimmed
    w0 = time.time()
    _feed(app, timeline, stop)
    t0 = time.monotonic()
    while time.monotonic() - t0 < dur:
        time.sleep(0.2)
    stop.set()
    app.speaker.wait_idle(8)
    audio = app.output.recorded()[n0:]
    app.stop()
    write_wav(a.out, audio, app.engine.rate)
    cap_path = os.path.splitext(a.out)[0] + ".captions.jsonl"
    with open(cap_path, "w") as fh:
        for r in spoken:
            if r["result"] != "dropped":
                fh.write(json.dumps({"t": round(r.get("t_start", r["t"]) - w0, 3), "end": round(r["t"] - w0, 3),
                                     "text": r["text"], "priority": r["priority"]}) + "\n")
    st = app.state()
    _log(f"[render] {a.out}: {audio.shape[0] / app.engine.rate:.1f}s, peak {np.abs(audio).max():.2f}, "
         f"beeps {app.sonar.stats['beeps']}, criticals {app.sonar.stats['criticals']}, "
         f"spoken {st['speech']['spoken']}, dropped(ttl) {st['speech']['dropped_ttl']}, captions {cap_path}")
    for h in spoken:
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
    r.add_argument("--udp-port", type=int, default=5010)
    r.add_argument("--ws-port", type=int, default=5011)
    r.add_argument("--no-judgment", action="store_true", help="turn off the Claude judgment layer (also KITE_JUDGMENT=0)")
    r.add_argument("--spacetime", action="store_true", help="mirror events to SpacetimeDB (SPACETIME_HOST) and speak caregiver messages from it")
    rr = sub.add_parser("render", help="render a scenario or a JSONL timeline to a WAV file")
    rr.add_argument("scenario", nargs="?", choices=["walk", "car", "dropout", "crowd"])
    rr.add_argument("out")
    rr.add_argument("--jsonl", help="event timeline, one event per line with 't' in seconds (instead of a scenario)")
    rr.add_argument("--seconds", type=float, help="length to render (default: last event + 2 s)")
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
