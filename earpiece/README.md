# Kite earpiece

The AirPods layer of Kite: spatial obstacle sonar, a waypoint beacon, and a voice assistant (Cartesia voice, Claude reasoning, Moss memory) for blind pedestrians.

## Run

```bash
cd earpiece
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[mac,brain,memory,dev]"
maturin develop --release -m native/Cargo.toml   # optional: Rust audio engine, binaural renderer, cue sounds, sonar decisions and thread
python -m earpiece selftest                      # check audio, voices, STT, controls, network
python -m earpiece run                           # the real thing (AirPods + stem presses)
python -m earpiece run --sim walk                # with a scripted street scenario
python -m earpiece render walk out/walk.wav      # render what the wearer would hear
python -m earpiece render --jsonl events.jsonl out/heard.wav   # same, from a JSONL timeline (docs/internal/video-runbook.md)
python -m pytest                                 # 3,300+ tests, 2,376 of them safety evals (also with KITE_AUDIO_ENGINE=python / KITE_SONAR_ENGINE=python / KITE_SONAR_THREAD=native, or all at once with bench/matrix.sh; KITE_LIVE_TESTS=1 adds live API checks)
```

Keys come from the repo-root `.env` (`CARTESIA_API_KEY`, `ANTHROPIC_API_KEY`, `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY`, and `GOOGLE_MAPS_API_KEY`). Everything degrades gracefully without them.

## Send it events

UDP JSON on `127.0.0.1:5010`, or WebSocket on `:5011` (also streams state and voice requests back). Use `earpiece.client.EarpieceClient`:

```python
from earpiece.client import EarpieceClient
ear = EarpieceClient(source="webcam")
ear.obstacles([{"id": 7, "kind": "person", "bearing": -20, "distance": 2.4}])   # every frame
ear.guidance("In 20 meters, turn left onto Lyon Street", key="step-3")
ear.beacon(bearing=-35, distance=42, label="Palace of Fine Arts")
ear.hazard("crosswalk", state="dont_walk", bearing=5, distance=9)
```

Voice requests come back over the WebSocket, already resolved:

| Wearer says | `voice_request` |
|---|---|
| "take me to my coffee shop" | `target_type: "memory"`, name, address, lat/lng (from Moss) |
| "bring me to the end of the street" | `target_type: "relative"`, `relation: "end_of_street"` (also `next_intersection`, `nearest_crosswalk`, `cross_street`, `back`, `nearest_entrance`) |
| "the nearest pharmacy" | `target_type: "nearby"`, name, address, lat/lng, `distance_m` (Google Places, ranked by distance) |
| "Palace of Fine Arts" | `target_type: "named"`, name, address, lat/lng |

The full message contract is in `earpiece/events.py`. Bearings are degrees in the wearer's frame (0 ahead, +90 right); distances are metres.

## Two ways to wear it

- **Indoors, or wherever a drone can't fly:** the body camera alone. Obstacle beeps, "what's in front of me?", and "read this" (menus, signs, books, screens) read aloud.
- **Outdoors:** the drone above and behind the head adds the route, crowds, crossings and the waypoint beacon.

With no perception stream, the earpiece grabs one frame from the local camera only when asked (`KITE_CAMERA=<index>`, or `off`; install the `camera` extra). macOS asks for Camera permission for the terminal the first time.

## Voice commands

| Say | What happens |
|---|---|
| "Read this" / "what does the sign say" | Claude transcribes the text in front of the body camera; Cartesia reads it sentence by sentence |
| "What's in front of me?" | Claude describes the body camera frame; offline, the obstacle tracks answer |
| "Take me to my coffee shop" / "the nearest pharmacy" / "the end of the street" | Resolved through Moss, Google Places, or sent as a relative target |
| "Remember this place as the bench" / "remember that the curb is tall" | Saved to Moss memory |
| "Forget the bench" / "rename the bench to the reading bench" / "undo" | Edits a saved memory; "undo" brings back the last one forgotten |
| "Take me home" | Routes to the saved home |
| "Stop" / "repeat" / "pause" / "louder" / "mute the beeps" | Handled locally, offline |
| "Talk faster" / "slower" / "normal speed" | Speaking rate in steps of 0.1x between 0.7x and 1.5x (Sonic 3.6's `generation_config.speed`). The setting is kept in `.cache/voice_prefs.json`, and the backup voice follows it too |

## The voice loop

- **Barge-in.** The first stem press cuts non-safety speech at once, with a 40 ms fade and without waiting for the 0.3 s gesture window. It also drops queued replies to the old question and stops Claude's answer mid-stream. Warnings are never cut. A navigation line that was cut comes back afterwards if it is still fresh. In hands-free mode with headphones, saying "Kite" cuts speech the same way, because the wake word arrives from Ink as its own segment while you are still talking.
- **Short commands end early.** At the first 0.3 s pause, Kite sends Ink a speculative `finalize`, and the stream stays open. If the words so far already form a complete command ("louder", "stop", "talk faster"), the utterance ends there without waiting for the full 0.7 s of silence.
- **Moss is fetched while you talk.** Each Ink segment starts the memory recall, so the memories are ready when the question ends. A recall slower than 0.3 s is skipped rather than allowed to delay the answer.
- **Measured latency.** The time from the end of speech to the first audio of the reply is broken down by stage in `state()["speech"]["voice_loop"]`: endpoint, STT, brain, TTS and total. `selftest` prints an estimate from its measured parts. `tests/test_voice_live.py` runs the real Ink and Sonic round trip on recorded speech:

  | Run on Oct 24, null output | endpoint | STT | brain | TTS | total |
  |---|---|---|---|---|---|
  | "Louder, please." (early end) | 596 ms | 2 ms | 6 ms | 204 ms | **808 ms** |
  | "What's around me?" (full silence) | 695 ms | 95 ms | 3 ms | 98 ms | **892 ms** |

  The TTS number is for replies that are not cached. Cached replies such as "Louder." start in about 0 ms, and AirPods add 150 to 250 ms of Bluetooth latency on top.
- **Street names.** `speech.PRONUNCIATIONS` respells the street names the voices get wrong, in the synthesised text only. For example, "Lyon Street" becomes "Lion Street" because San Francisco says LIE-un. Set `KITE_CARTESIA_PRONUNCIATION_DICT` to apply a Cartesia pronunciation dictionary as well.
- **Outages.** A failed connection backs off. A background handshake, which costs no credits, notices when the network is back, so no sentence pays a connect timeout to find out. The handshake also keeps the socket warm: replies on a cold socket took about 300 ms, against about 100 ms on a warm one. The wearer hears "No internet. Using the backup voice." at most once a minute. They hear "Internet is back." only if they were told about the outage. An uncached warning gives Cartesia 1 s and then plays the backup voice, which has been rendering in parallel. `say` runs with stdin closed: when it inherits a pipe, it hangs for its full 15 s timeout.

## Safety rules in code

Claude's judgment layer (`judgment.py`) may only add caution: a filter discards any reply that would clear a stop, move the walker or reroute. After a "Stop.", only deterministic code says "Path is clear. You can continue.", and only once the way has stayed clear for 1.5 s with perception alive.

## The warning path in Rust

When `kite_audio` is built, three parts of the warning path run in Rust.

- **Sonar decisions** (`native/src/sonar.rs`, used through `rust_sonar.py`): the object tracks, the closing-speed estimate, time-to-contact, corridor weighting, urgency, the two-source cap, when the next beep is due and from which bearing at which pitch, and whether to say "Stop.". One frame plus one tick with 20 tracked objects takes about 20 µs in Rust and about 350 µs in Python. Choose them with `KITE_SONAR_ENGINE=rust` or `python`; the default is Rust when the extension is built. `tests/test_rust_sonar.py` feeds both implementations identical input (scripted sequences, 24 seeded random streams, and every timeline in `data/` and `sim.py`) and checks after every tick that beeps, speech, chosen tracks and track state are the same. Over about 22,000 ticks they are bit-identical.
- **Cue sounds** (`native/src/sounds.rs`): every cue in `sounds.py`, which uses it automatically. The output is bit-identical to the numpy and scipy code, down to scipy's filter arithmetic and numpy's seeded noise in the knock. `tests/test_rust_sounds.py` compares 2,160 cues (12.7 million samples) exactly: every object kind at every sonar pitch in both profiles, every beacon region at every 5-degree bearing in both profiles, off-grid pitches and bearings, and every status earcon. A cue takes 0.2 to 0.6 ms in Rust, 1.5 to 2.6 times faster than Python, and the GIL is released while it renders.
- **The native sonar thread** (`native/src/sonar_thread.rs`, `KITE_SONAR_THREAD=native`, off by default until it has been tried on the street). A Rust thread ticks the sonar every 20 ms. It synthesises and places each beep in Rust, schedules the beep straight into the mixer, and runs the stale-perception watchdog itself. Python only feeds obstacle frames, and its control loop picks up the "speak" requests (such as "Stop.") and watchdog changes from a lock-free queue. The thread runs in macOS's real-time band. Under a load average of 6 to 21 it woke at most 19 ms late; the default priority woke up to 51 ms late, and beeps are scheduled 60 ms ahead. Given the same tick times, one native step produces exactly the audio and speech of the Python-ticked path, and `tests/test_sonar_thread.py` checks this sample by sample on every timeline. The same test file freezes Python by holding the GIL for 500 ms. With the native thread, both beats due during the freeze played on time, and no onset was more than 0.85 ms off the beat. With the Python-ticked path, neither beat played, and the next beep came 503 ms after the freeze began. With Python frozen for 2.2 s, the native thread played no beep later than 0.6 s after the last frame (the usual fade) and switched the sonar off 1.5 s after it. It needs the Rust audio engine at 48 kHz; otherwise the earpiece logs why and ticks from Python. With `run --null`, the mixer then also runs on a native thread (`RustNullOutput`), as a sound device would.

What is still in Python, and why: speech, including "Stop." (the native thread queues it, and Python's speech scheduler says it); the network transport and `events.parse`, so during a long stall the native sonar works from the last frame it got; and the beacon's ticks (guidance, not warnings; its cues are synthesised in Rust).

**Why these parts.** `bench/profile_rt.py` replays `data/sample_walk.jsonl` and the sim scenarios through the whole app over UDP, in real time with the null output, and times every stage. The earpiece uses 2 to 4 % of one core. `events.parse` is 0.15 to 0.6 % of wall time (22 µs for a 3-object frame, 108 µs for 20 objects), so porting it would not change much, and the transport around it would still need the GIL. The time that mattered was what could delay a beep. The 50 Hz control loop's period had a p99 of 25 to 30 ms but a maximum of 86 to 137 ms. Each walk synthesised about 24 cold sonar cues on that loop, typically 0.6 to 1.6 ms each, with the worst at 14 ms and one at 130 ms. One or two beeps per replay were scheduled after their start time. A full garbage collection with the app loaded took 80 to 310 ms (none ran during the replays), and the startup cue pre-warm kept a Python thread busy for 1.8 s (now 0.1 to 0.16 s). Beeps are scheduled 60 ms ahead, so any stall of the Python loop longer than about 40 ms moves them. In the profiler's model, a 100 ms stall makes a beep 23 ms late, and a 500 ms stall makes one 423 ms late and stacks two beeps on top of each other. `bench/matrix.sh` runs the whole suite under each combination of audio engine, sonar engine and thread mode.

## Credits and limits

Characters sent to Cartesia are counted across runs in `earpiece/.cache/cartesia_usage.json`; past `KITE_CARTESIA_CHAR_BUDGET` (default 150,000) only urgent speech uses the neural voice. Phrase warm-up runs one request at a time (our plan allows two in flight). A `quota_exceeded` error switches to the backup voice for 5 minutes instead of retrying every line. After a rate change, only the fixed phrases (about 900 characters) are re-recorded; sonar labels recorded at the old rate keep playing instantly and are refreshed when they are next used. The live voice test costs about 56 TTS characters and 4 s of STT audio per run.

## Controls

AirPods (or any Bluetooth headphones' play button): once = talk, twice = repeat, three times = pause guidance (safety warnings stay on). Any press first stops Kite talking (barge-in). Keyboard fallback: Enter / `r` / `p`, plus `t <words>` to type a command.

## Layout

| Module | Job |
|---|---|
| `mixer.py` / `native/` (Rust) | real-time mixing, ducking, limiter, device output; native binaural placement |
| `spatial.py`, `sounds.py` | binaural rendering (Brown-Duda head model) and synthesised cues; both run in Rust when built, sample-identical |
| `sonar.py`, `rust_sonar.py`, `beacon.py` | obstacle warnings by time-to-contact (decisions in Rust when built, identical cue stream; optionally ticked on a native thread); Soundscape-style waypoint beacon |
| `speech.py`, `listen.py` | Cartesia TTS/STT, priorities, preemption, time-to-live, offline fallback |
| `brain.py`, `memory.py` | local voice commands, Claude with tools, Moss memory of places and hazards |
| `route.py`, `geo.py` | Google Routes v2 walking directions turned into spoken turns and the beacon; lat/lng and sim meters |
| `judgment.py` | Claude judgment calls on hazards and scene ("Wait, the crosswalk is crowded."), with code guardrails |
| `vision.py`, `camera.py` | Claude vision: scene descriptions, "what's in front of me?" and reading text aloud from the body camera |
| `app.py`, `transport.py`, `controls.py` | wiring, UDP/WebSocket, AirPods stem presses |
| `spacetime.py` | SpacetimeDB mirror (feeds the demo dashboard) |

## Memory (Moss)

Kite remembers the wearer's places, hazards and notes in a Moss index (`kite-memory`), loaded into Moss's local runtime so a lookup runs in-process on every utterance. Measured on a busy laptop over four runs of 100 queries: a live Moss query took 4 to 9 ms at the median and 12 to 15 ms at the 95th percentile, with one run under heavy CPU contention reaching 100 ms at the tail (the query embeds the text on the CPU). The offline mirror answers in under 0.2 ms at the median with 500 memories.

- **Finding places.** "My usual coffee place", "the cafe", "the bench by the park" and "where I fell last week" all find the right memory: plurals and common synonyms are folded together, rare words count for more, and the main noun counts for more than the words after it. A destination always has to share a real word with what was said, so Kite never invents somewhere to walk to.
- **Asking, not guessing.** When two saved places fit equally well, Kite asks ("Which one: the bus stop bench, or the park bench?") and understands short answers: "the second one", "the park one", "never mind".
- **No pile of copies.** Saving the same place again (same name, within 30 m) moves it; saying the same hazard again marks it as still true. Renamed places still answer to their old name.
- **Older is softer.** A hazard heads-up from last week is "Heads up: ..."; from a few months ago it is "From 3 months ago: ..."; older than six months, "An old note, it may have changed: ...".
- **Offline and privacy.** Every write goes to a local mirror (`earpiece/.cache/memory.json`, readable by the user only) and then to Moss; with no keys or no network the mirror answers on its own with the same matching. A memory forgotten while offline is deleted from Moss on the next start, so it cannot come back. `memory.py` talks to Moss and nothing else. Two other paths carry memory text off the device, both by design: the few memories relevant to a question go to Claude with that question, and memory events go to the demo dashboard when the SpacetimeDB bridge is on.
- The seeded demo home is a stand-in point on the public Marina Green lawn, not anyone's address.

Tests: `tests/test_memory.py` (fake Moss for parity and failure modes; `-m slow` for the latency benchmark; `KITE_LIVE_TESTS=1` for one live Moss check).

## Walking directions

When the wearer asks for a saved, nearby or named place, the earpiece routes there itself (Google Routes API v2, walking) and follows every `position` event:

- the first instruction at the start, said in the wearer's frame when a heading is known ("Head toward Lyon Street, slightly left");
- a heads-up about 30 m before each turn ("In 30 meters, turn left onto Bay Street") and "Now turn left onto Bay Street" at about 6 m, each exactly once, however much the GPS wanders;
- the beacon aimed at the next corner (never across it), then about 15 m along the route, with the distance left to the destination;
- "You're walking away from the route. Turn around." once, when the wearer heads back the way they came (or the wrong way from the start) for about 12 m; standing still with GPS wander never triggers it;
- "You're off the route. Finding a new one." after 8 s more than 25 m away (not repeated within a minute), then a new route from where the wearer is. New routes are requested at most every 20 s and 5 times in 10 minutes; while one waits, the beacon points straight at the destination;
- arrival within 8 m, or at the end of the route.

When Google can't help, the wearer hears one short line per destination ("No directions right now. The beacon points to Blue Bottle Coffee.") and the beacon points straight at the place. Timeouts, lost network and server errors retry after 15 s, quota errors after 60 s, and a bad key or "no walking route" are not retried. Distances are always spoken in meters, rounded the way a person would ("in 25 meters"), and directions are given in the wearer's frame ("slightly left", "at 2 o'clock") when a heading is known. Route cues are `nav` priority with short expiry, so a hazard always goes first and a turn that can't be said in time is dropped rather than said late.

Tests: `tests/test_route.py` and `tests/test_route_hardening.py` (random routes and random walks with jitter, drift, stops and U-turns; a fake Google for every failure).

`position.heading` is a compass heading (0 north, 90 east). Without one, the walking direction over the last few fixes is used, and no beacon is sent while the wearer stands still. The sim and real GPS share one mapping, `geo.LocalFrame`: anchor it with `LocalFrame.from_landmarks([((lat, lng), (x, y)), ...])` on the same two or three landmarks that line up the splat, then `to_latlng(x, y)` for positions and `heading_of(dx, dy)` for headings. Route state goes out as `route` events (`started` with the polyline, `fallback`, `failed` with `retry_s`, `rerouting` with `wait_s`, `holding`, `arrived`, `cancelled`) for the dashboard map.
