"""Pre-demo self-test: run this on the demo laptop with the AirPods in.

Checks, in order, and prints PASS / WARN / FAIL with the measured numbers:
  1. output device (are the AirPods the output, in high-quality stereo?)
  2. audio stream health (callbacks running, no underflows over 2 s)
  3. spatial audio (plays a knock on the left then the right and checks the rendering)
  4. neural voice (Cartesia WebSocket: time to first audio)
  5. offline voice (macOS `say`)
  6. speech recognition (Cartesia STT on a synthesised phrase: word accuracy + latency)
  7. Claude (one short answer, if ANTHROPIC_API_KEY is set)
  8. microphone (opens it for 1 s and checks it is not silent / blocked by permissions)
  9. AirPods stem presses (posts a simulated media key and checks it is received)
 10. network ports (UDP + WebSocket loopback)
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import numpy as np

from .config import Cfg


def run(play: bool = True, log=print) -> int:
    cfg = Cfg()
    results: list[tuple[str, str, str]] = []

    def rep(name: str, status: str, detail: str) -> None:
        results.append((name, status, detail))
        log(f"[selftest] {status:4} {name}: {detail}")

    from . import devices, sounds, spatial
    from .mixer import Engine

    # 1. output device
    outs = devices.outputs()
    dflt = next((d for d in outs if d["default"]), None)
    ap = [d for d in outs if "airpods" in d["name"].lower()]
    if ap:
        rate = ap[0]["rate"]
        rep("output", "PASS" if rate >= 44100 else "WARN",
            f"{ap[0]['name']} at {rate:.0f} Hz" + ("" if rate >= 44100 else
                                                  " (call mode: something is using the AirPods mic)"))
    else:
        rep("output", "WARN", f"no AirPods found; default output is {dflt['name'] if dflt else 'unknown'}")

    # 2. stream health
    engine = Engine(cfg.audio.sample_rate)
    from .output import DeviceOutput
    out = DeviceOutput(engine, cfg.audio.output_substr, cfg.audio.block, log=lambda *_: None)
    try:
        out.start()
        b0 = engine.stats["blocks"]
        time.sleep(2.0)
        blocks = engine.stats["blocks"] - b0
        expect = 2.0 * engine.rate / cfg.audio.block
        st = engine.snapshot_stats()
        ok = blocks > 0.8 * expect and out.underflows == 0
        rep("stream", "PASS" if ok else "WARN",
            f"{out.device_name}: {blocks}/{expect:.0f} callbacks, {out.underflows} underflows, "
            f"render avg {st['avg_render_ms']:.2f} ms max {st['max_render_ms']:.2f} ms")
    except Exception as e:
        rep("stream", "FAIL", f"{type(e).__name__}: {e}")
        out = None

    # 3. spatial
    left = spatial.render(sounds.ear_check("left"), -90, engine.rate)
    right = spatial.render(sounds.ear_check("right"), 90, engine.rate)
    ild_l, ild_r = spatial.interaural_level_db(left), spatial.interaural_level_db(right)
    burst = (np.random.default_rng(0).standard_normal(4800) * 0.3).astype(np.float32)
    itd_l = spatial.interaural_delay_s(spatial.render(burst, -90, engine.rate), engine.rate) * 1e6
    ok = ild_l < -6 and ild_r > 6 and 450 < itd_l < 800
    rep("spatial", "PASS" if ok else "FAIL",
        f"left cue {ild_l:+.1f} dB / ITD {itd_l:+.0f} us, right cue {ild_r:+.1f} dB")
    if play and out is not None:
        engine.play(left, bus="alert")
        time.sleep(0.6)
        engine.play(right, bus="alert")
        time.sleep(0.6)

    # 4. neural voice
    from .speech import CartesiaStream, offline_tts
    if cfg.cartesia_key:
        async def tts_probe():
            cs = CartesiaStream(cfg.speech, cfg.cartesia_key, engine.rate)
            t0 = time.monotonic()
            first, n = None, 0
            chunks = []
            async for c in cs.stream("Kite self test. Person, slightly left.", "selftest"):
                first = first or time.monotonic() - t0
                n += c.size
                chunks.append(c)
            await cs.close()
            return first, n, np.concatenate(chunks)
        try:
            first, n, pcm = asyncio.run(tts_probe())
            rep("voice", "PASS" if first < 1.0 else "WARN",
                f"Cartesia first audio {first * 1000:.0f} ms (incl. connect), {n / engine.rate:.1f}s of speech")
            if play and out is not None:
                engine.speech_begin("st")
                engine.speech_feed("st", pcm)
                engine.speech_end("st")
                time.sleep(n / engine.rate + 0.3)
        except Exception as e:
            rep("voice", "FAIL", f"Cartesia: {type(e).__name__}: {e}")
    else:
        rep("voice", "WARN", "CARTESIA_API_KEY not set: offline voice only")

    # 5. offline voice
    try:
        t0 = time.monotonic()
        pcm = offline_tts("Backup voice check.", engine.rate, cfg.speech.offline_voice)
        rep("offline", "PASS", f"macOS say {1000 * (time.monotonic() - t0):.0f} ms, {pcm.size / engine.rate:.1f}s")
    except Exception as e:
        rep("offline", "WARN", f"{type(e).__name__}: {e}")

    # 6. STT round trip on synthesised speech
    if cfg.cartesia_key:
        try:
            import httpx
            phrase = "Kite, take me to the Palace of Fine Arts"
            r = httpx.post("https://api.cartesia.ai/tts/bytes", timeout=20, headers={
                "Authorization": f"Bearer {cfg.cartesia_key}", "Cartesia-Version": cfg.speech.cartesia_version},
                json={"model_id": cfg.speech.cartesia_model, "transcript": phrase,
                      "voice": "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
                      "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}})
            r.raise_for_status()
            from .listen import CartesiaSTT

            async def stt_probe():
                stt = CartesiaSTT(cfg.cartesia_key, cfg.listen.stt_model, 16000)
                q: asyncio.Queue = asyncio.Queue()
                task = asyncio.create_task(stt.transcribe_stream(q))
                pcm16 = r.content
                for i in range(0, len(pcm16), 3200):
                    await q.put(pcm16[i:i + 3200])
                    await asyncio.sleep(0.1)
                t0 = time.monotonic()
                await q.put(None)
                text = await task
                return text, time.monotonic() - t0

            text, lat = asyncio.run(stt_probe())
            want = set(phrase.lower().replace(",", "").split())
            got = set(text.lower().replace(",", "").replace(".", "").split())
            acc = len(want & got) / len(want)
            rep("stt", "PASS" if acc >= 0.8 else "WARN", f"{text!r} ({acc:.0%} words, {lat * 1000:.0f} ms after end)")
        except Exception as e:
            rep("stt", "FAIL", f"{type(e).__name__}: {e}")

    # 7. Claude
    if cfg.anthropic_key:
        from .brain import Actions, Brain
        said = []
        b = Brain(lambda: {"nearby": [{"kind": "person", "bearing": -20, "distance": 2.5, "urgency": 0.4}]},
                  Actions(lambda d: "ok", lambda a: "ok", lambda m: "ok", lambda: None, lambda: None),
                  lambda t, p, **k: said.append(t), cfg.anthropic_key, log=lambda *_: None)
        t0 = time.monotonic()
        try:
            how = b.handle_sync("Is anyone near me?", timeout=25)
            first = b.stats["first_sentence_ms"][0] if b.stats["first_sentence_ms"] else None
            rep("claude", "PASS" if how == "llm" and said else "FAIL",
                f"{how}: {' '.join(said)!r} (first sentence {first and round(first)} ms, "
                f"total {1000 * (time.monotonic() - t0):.0f} ms)")
        except Exception as e:
            rep("claude", "FAIL", f"{type(e).__name__}: {e}")
    else:
        rep("claude", "WARN", "ANTHROPIC_API_KEY not set: offline answers only")

    # 8. microphone
    try:
        import sounddevice as sd
        from .listen import pick_input
        idx, name = pick_input(cfg.listen.input_substr)
        rec = sd.rec(int(1.0 * 16000), samplerate=16000, channels=1, dtype="float32", device=idx)
        sd.wait()
        rms = float(np.sqrt(np.mean(rec ** 2)))
        if rms == 0.0:
            rep("mic", "FAIL", f"{name}: pure silence (grant Microphone permission to this terminal)")
        else:
            rep("mic", "PASS", f"{name}: room level {20 * np.log10(rms + 1e-9):.0f} dBFS"
                + ("" if "airpods" not in name.lower() else " (AirPods mic: output drops to call quality)"))
    except Exception as e:
        rep("mic", "FAIL", f"{type(e).__name__}: {e}")

    # 9. stems via simulated media keys
    try:
        from .controls import Gestures, post_media_key
        got: list[str] = []
        g = Gestures(lambda x: got.append(x), log=lambda *_: None)
        if g.enable_stem():
            from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop
            for k in ("play", "next", "previous"):
                post_media_key(k)
                t_end = time.monotonic() + 0.8
                while time.monotonic() < t_end:
                    NSRunLoop.currentRunLoop().runMode_beforeDate_(
                        NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05))
            ok = got == ["talk", "repeat", "pause"]
            rep("stems", "PASS" if ok else "WARN", f"simulated presses -> {got}"
                + ("" if ok else " (posting keys needs Accessibility permission; real AirPods may still work)"))
        else:
            rep("stems", "WARN", "media-key capture unavailable on this system")
    except Exception as e:
        rep("stems", "WARN", f"{type(e).__name__}: {e}")

    # 10. network loopback
    try:
        from .transport import Transport
        got_msgs = []
        tr = Transport(lambda m: got_msgs.append(m.type), "127.0.0.1", 0, 0, log=lambda *_: None)
        tr.udp_port, tr.ws_port = _free_port(), _free_port()
        tr.start()
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(
            json.dumps({"type": "heartbeat", "source": "selftest"}).encode(), ("127.0.0.1", tr.udp_port))
        from websockets.sync.client import connect
        with connect(f"ws://127.0.0.1:{tr.ws_port}") as ws:
            ws.send(json.dumps({"type": "scene", "text": "self test"}))
            time.sleep(0.3)
        rep("network", "PASS" if got_msgs == ["heartbeat", "scene"] else "FAIL", f"received {got_msgs}")
    except Exception as e:
        rep("network", "FAIL", f"{type(e).__name__}: {e}")

    if out is not None:
        out.stop()
    fails = [r for r in results if r[1] == "FAIL"]
    warns = [r for r in results if r[1] == "WARN"]
    log(f"[selftest] {len(results) - len(fails) - len(warns)} pass, {len(warns)} warn, {len(fails)} fail")
    return 1 if fails else 0


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p
