# Kite earpiece

The AirPods layer of Kite: spatial obstacle sonar, a waypoint beacon, and a voice assistant (Cartesia voice, Claude reasoning, Moss memory) for blind pedestrians.

## Run

```bash
cd earpiece
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[mac,brain,memory,dev]"
maturin develop --release -m native/Cargo.toml   # optional: Rust audio engine, binaural renderer, sonar decisions
python -m earpiece selftest                      # check audio, voices, STT, controls, network
python -m earpiece run                           # the real thing (AirPods + stem presses)
python -m earpiece run --sim walk                # with a scripted street scenario
python -m earpiece render walk out/walk.wav      # render what the wearer would hear
python -m earpiece render --jsonl events.jsonl out/heard.wav   # same, from a JSONL timeline (docs/internal/video-runbook.md)
python -m pytest                                 # 400+ tests (also with KITE_AUDIO_ENGINE=python / KITE_SONAR_ENGINE=python; KITE_LIVE_TESTS=1 adds live API checks)
```

Keys come from the repo-root `.env` (`CARTESIA_API_KEY`, `ANTHROPIC_API_KEY`, `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY`, `GOOGLE_MAPS_API_KEY`, and `XAI_API_KEY` for the caregiver line). Everything degrades gracefully without them.

## Send it events

UDP JSON on `127.0.0.1:5010`, or WebSocket on `:5011` (also streams state, voice requests and caregiver alerts back). Use `earpiece.client.EarpieceClient`:

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

## Sonar decisions in Rust

When `kite_audio` is built, the sonar's per-tick decisions run in Rust (`native/src/sonar.rs`, used through `rust_sonar.py`). That covers the object tracks, the closing-speed estimate, time-to-contact, corridor weighting, urgency, the two-source cap, when the next beep is due and from which bearing at which pitch, and whether to say "Stop.". The point is that warning timing should not depend on Python bytecode competing for the GIL with network, JSON and Claude work. One frame plus one tick with 20 tracked objects takes about 5 µs in Rust and about 110 µs in Python. Python still plays what Rust decided (a cached, pre-rendered cue goes to the audio engine) and still drives the 50 Hz control loop, so a long stall of that loop still delays the tick. Moving the tick onto its own native thread would also need cue synthesis in Rust, and that is not done yet. Choose the engine with `KITE_SONAR_ENGINE=rust` or `python`; the default is Rust when the extension is built, otherwise Python. It works with either audio engine. `tests/test_rust_sonar.py` feeds both implementations identical input: scripted sequences (an approaching person, a crossing cyclist, out-of-order and restarted frames, stale perception with the watchdog, the two-source cap, critical stops, mute and labels, control-loop stalls), 24 seeded random multi-source streams, and every recorded and simulated timeline in `data/` and `sim.py`. After every tick it checks that both produce the same beeps, speech, chosen tracks and track state. Over about 22,000 ticks the two are bit-identical, and rendered audio is sample-identical.

## Credits and limits

Characters sent to Cartesia are counted across runs in `earpiece/.cache/cartesia_usage.json`; past `KITE_CARTESIA_CHAR_BUDGET` (default 150,000) only urgent speech uses the neural voice. Phrase warm-up runs one request at a time (our plan allows two in flight). A `quota_exceeded` error switches to the backup voice for 5 minutes instead of retrying every line. After a rate change, only the fixed phrases (about 900 characters) are re-recorded; sonar labels recorded at the old rate keep playing instantly and are refreshed when they are next used. The live voice test costs about 56 TTS characters and 4 s of STT audio per run.

## Controls

AirPods (or any Bluetooth headphones' play button): once = talk, twice = repeat, three times = pause guidance (safety warnings stay on). Any press first stops Kite talking (barge-in). Keyboard fallback: Enter / `r` / `p`, plus `t <words>` to type a command.

## Caregiver line (Grok Voice)

A caregiver who is driving, cooking, or blind or low vision can talk to Kite instead of watching the dashboard. On the caregiver's laptop:

```bash
python -m earpiece caregiver                 # talk through the laptop mic and speakers
python -m earpiece caregiver --text          # type instead (add --mute for no audio)
python -m earpiece caregiver --dry-run       # no Grok session: print what the tools return, send nothing
python -m earpiece caregiver --input-wav q1.wav --input-wav q2.wav --output-wav reply.wav --transcript t.json
                                             # headless: recordings as the mic, Grok's voice to files
```

It opens one realtime speech-to-speech session with xAI's Grok Voice Agent API (`XAI_API_KEY` in the repo-root `.env`; `--voice`, default `eve`; `--db`, default `kite-sync-pl6rp`) and gives Grok three tools over the shared SpacetimeDB state:

| Caregiver says | Tool | What it does |
|---|---|---|
| "How's the walk going?" / "Where is she?" | `walk_status` | position, route, recent hazards, what Kite last said, which devices are reporting, each with its age |
| "Tell her I'm running five minutes late" | `message_walker` | pushes a caregiver `say` at priority info, the same row the dashboard's message box writes; the walker's earpiece speaks it |
| "Is she okay? Did she ask for help?" | `recent_alerts` | the walker's "tell my caregiver" alerts and recent stop warnings |

While the line is open, a new walker alert is announced right away without being asked. Grok only knows what the tools return; when the newest update is more than 90 s old, the tool result says so and Grok says how old the information is. Rows from earlier walks (more than 15 minutes before the newest update) are left out. In code, messages are cut to 200 characters, a go-ahead to cross is never relayed, and a repeated message is not sent twice. Reads are cached for 3 s and share the Spacetime token bucket. The mic is paused while Kite talks so the laptop speaker cannot interrupt it (`--barge-in` keeps it open; use headphones). The call hangs up after `--max-minutes` (default 30).

If the walker's earpiece has `KITE_CAREGIVER_TOKEN` set, set the same value on the caregiver's laptop: the line signs its messages exactly like the dashboard does, and the earpiece drops unsigned ones.

Headless mode opens no sound device at all. Each `--input-wav` file (16-bit WAV, any rate, mixed down to mono 24 kHz) is one caregiver turn, streamed in 100 ms chunks at real-time pace and followed by a second of silence so Grok's turn detection hears the end. `--output-wav` writes the whole reply plus `reply.turnN.wav` per turn, and `--transcript` writes the transcript, every tool call with its result, and each turn's timing as JSON.

### Headless end-to-end test

```bash
scripts/caregiver_e2e.sh                     # about a minute; prints PASS/FAIL per check
KITE_LIVE_TESTS=1 python -m pytest tests/test_caregiver_e2e_live.py   # the same run as a test
```

Nothing is played aloud, and no microphone or camera is opened. The script starts a throwaway SpacetimeDB on a free `127.0.0.1` port, with its own data directory, JWT keys and CLI config, so Maincloud and `~/.config/spacetime` are never touched. It publishes `dashboard/spacetimedb` there and seeds a walk (a position on Lyon Street, the Palace of Fine Arts waypoint, a crowd hazard, a spoken line). The walker's earpiece runs as `run --null --spacetime --no-stem --no-judgment --no-prewarm` with `KITE_CAMERA=off` and logs every line it says as `[said] ...`.

The caregiver's two questions, "How's the walk going?" and "Tell her I'm running five minutes late.", are rendered with `say -o`, which writes a file and does not play, and are sent to Grok with `--input-wav`. Both sides share a random `KITE_CAREGIVER_TOKEN`, so the signed path is what gets tested.

The script checks five things:

- Grok calls `walk_status`, and its reply names the seeded walk.
- Grok calls `message_walker`, and the signed caregiver `say` row lands in the local database.
- The silent earpiece says "Message from your caregiver: ...late..." within 10 s.
- The earpiece mirrors that line back as a `spoken` row.
- The caregiver side opens no audio device.

Audio, transcripts and logs stay in `earpiece/.cache/caregiver-e2e/<time>/`: the questions, `grok-reply.turnN.wav`, `transcript.json`, `caregiver.log`, `walker.log` and `summary.json`.

Live cost is one Grok session with two short turns, plus at most one new Cartesia line, the caregiver message. Phrase warm-up is skipped. `KITE_TTS_CACHE` points the earpiece at an existing phrase cache, and the script uses the main checkout's cache when it runs from a git worktree. Keys come from the repo-root `.env`, or from the main checkout's `.env` in a worktree. The script also needs the `spacetime` CLI, `openssl`, and node_modules for `dashboard/spacetimedb`, either installed or taken from the main checkout.

## Layout

| Module | Job |
|---|---|
| `mixer.py` / `native/` (Rust) | real-time mixing, ducking, limiter, device output; native binaural placement |
| `spatial.py`, `sounds.py` | binaural rendering (Brown-Duda head model; runs in Rust when built, sample-identical) and synthesised cues |
| `sonar.py`, `rust_sonar.py`, `beacon.py` | obstacle warnings by time-to-contact (decisions in Rust when built, identical cue stream); Soundscape-style waypoint beacon |
| `speech.py`, `listen.py` | Cartesia TTS/STT, priorities, preemption, time-to-live, offline fallback |
| `brain.py`, `memory.py` | local voice commands, Claude with tools, Moss memory of places and hazards |
| `route.py`, `geo.py` | Google Routes v2 walking directions turned into spoken turns and the beacon; lat/lng and sim meters |
| `judgment.py` | Claude judgment calls on hazards and scene ("Wait, the crosswalk is crowded."), with code guardrails |
| `vision.py`, `camera.py` | Claude vision: scene descriptions, "what's in front of me?" and reading text aloud from the body camera |
| `app.py`, `transport.py`, `controls.py` | wiring, UDP/WebSocket, AirPods stem presses |
| `spacetime.py`, `caregiver_voice.py` | SpacetimeDB mirror and caregiver inbox; the caregiver's Grok Voice line |

## Memory (Moss)

Kite remembers the wearer's places, hazards and notes in a Moss index (`kite-memory`), loaded into Moss's local runtime so a lookup runs in-process on every utterance. Measured on a busy laptop over four runs of 100 queries: a live Moss query took 4 to 9 ms at the median and 12 to 15 ms at the 95th percentile, with one run under heavy CPU contention reaching 100 ms at the tail (the query embeds the text on the CPU). The offline mirror answers in under 0.2 ms at the median with 500 memories.

- **Finding places.** "My usual coffee place", "the cafe", "the bench by the park" and "where I fell last week" all find the right memory: plurals and common synonyms are folded together, rare words count for more, and the main noun counts for more than the words after it. A destination always has to share a real word with what was said, so Kite never invents somewhere to walk to.
- **Asking, not guessing.** When two saved places fit equally well, Kite asks ("Which one: the bus stop bench, or the park bench?") and understands short answers: "the second one", "the park one", "never mind".
- **No pile of copies.** Saving the same place again (same name, within 30 m) moves it; saying the same hazard again marks it as still true. Renamed places still answer to their old name.
- **Older is softer.** A hazard heads-up from last week is "Heads up: ..."; from a few months ago it is "From 3 months ago: ..."; older than six months, "An old note, it may have changed: ...".
- **Offline and privacy.** Every write goes to a local mirror (`earpiece/.cache/memory.json`, readable by the user only) and then to Moss; with no keys or no network the mirror answers on its own with the same matching. A memory forgotten while offline is deleted from Moss on the next start, so it cannot come back. `memory.py` talks to Moss and nothing else. Two other paths carry memory text off the device, both by design: the few memories relevant to a question go to Claude with that question, and memory events go to the caregiver dashboard when the SpacetimeDB bridge is on.
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
