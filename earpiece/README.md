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
| "Stop" / "repeat" / "pause" / "louder" / "mute the beeps" | Handled locally, offline |

## Safety rules in code

Claude's judgment layer (`judgment.py`) may only add caution: a filter discards any reply that would clear a stop, move the walker or reroute. After a "Stop.", only deterministic code says "Path is clear. You can continue.", and only once the way has stayed clear for 1.5 s with perception alive.

## Sonar decisions in Rust

When `kite_audio` is built, the sonar's per-tick decisions run in Rust (`native/src/sonar.rs`, used through `rust_sonar.py`). That covers the object tracks, the closing-speed estimate, time-to-contact, corridor weighting, urgency, the two-source cap, when the next beep is due and from which bearing at which pitch, and whether to say "Stop.". The point is that warning timing should not depend on Python bytecode competing for the GIL with network, JSON and Claude work. One frame plus one tick with 20 tracked objects takes about 5 µs in Rust and about 110 µs in Python. Python still plays what Rust decided (a cached, pre-rendered cue goes to the audio engine) and still drives the 50 Hz control loop, so a long stall of that loop still delays the tick. Moving the tick onto its own native thread would also need cue synthesis in Rust, and that is not done yet. Choose the engine with `KITE_SONAR_ENGINE=rust` or `python`; the default is Rust when the extension is built, otherwise Python. It works with either audio engine. `tests/test_rust_sonar.py` feeds both implementations identical input: scripted sequences (an approaching person, a crossing cyclist, out-of-order and restarted frames, stale perception with the watchdog, the two-source cap, critical stops, mute and labels, control-loop stalls), 24 seeded random multi-source streams, and every recorded and simulated timeline in `data/` and `sim.py`. After every tick it checks that both produce the same beeps, speech, chosen tracks and track state. Over about 22,000 ticks the two are bit-identical, and rendered audio is sample-identical.

## Credits and limits

Characters sent to Cartesia are counted across runs in `earpiece/.cache/cartesia_usage.json`; past `KITE_CARTESIA_CHAR_BUDGET` (default 150,000) only urgent speech uses the neural voice. Phrase warm-up runs one request at a time (our plan allows two in flight).

## Controls

AirPods (or any Bluetooth headphones' play button): once = talk, twice = repeat, three times = pause guidance (safety warnings stay on). Keyboard fallback: Enter / `r` / `p`, plus `t <words>` to type a command.

## Caregiver line (Grok Voice)

A caregiver who is driving, cooking, or blind or low vision can talk to Kite instead of watching the dashboard. On the caregiver's laptop:

```bash
python -m earpiece caregiver                 # talk through the laptop mic and speakers
python -m earpiece caregiver --text          # type instead (add --mute for no audio)
python -m earpiece caregiver --dry-run       # no Grok session: print what the tools return, send nothing
```

It opens one realtime speech-to-speech session with xAI's Grok Voice Agent API (`XAI_API_KEY` in the repo-root `.env`; `--voice`, default `eve`; `--db`, default `kite-sync-pl6rp`) and gives Grok three tools over the shared SpacetimeDB state:

| Caregiver says | Tool | What it does |
|---|---|---|
| "How's the walk going?" / "Where is she?" | `walk_status` | position, route, recent hazards, what Kite last said, which devices are reporting, each with its age |
| "Tell her I'm running five minutes late" | `message_walker` | pushes a caregiver `say` at priority info, the same row the dashboard's message box writes; the walker's earpiece speaks it |
| "Is she okay? Did she ask for help?" | `recent_alerts` | the walker's "tell my caregiver" alerts and recent stop warnings |

While the line is open, a new walker alert is announced right away without being asked. Grok only knows what the tools return; when the newest update is more than 90 s old, the tool result says so and Grok says how old the information is. Rows from earlier walks (more than 15 minutes before the newest update) are left out. In code, messages are cut to 200 characters, a go-ahead to cross is never relayed, and a repeated message is not sent twice. Reads are cached for 3 s and share the Spacetime token bucket. The mic is paused while Kite talks so the laptop speaker cannot interrupt it (`--barge-in` keeps it open; use headphones). The call hangs up after `--max-minutes` (default 30).

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

## Walking directions

When the wearer asks for a saved, nearby or named place, the earpiece routes there itself (Google Routes API v2, walking) and follows every `position` event:

- the first instruction at the start, said in the wearer's frame when a heading is known ("Head toward Lyon Street, slightly left");
- a heads-up about 30 m before each turn ("In 30 meters, turn left onto Bay Street") and "Now turn left onto Bay Street" at about 6 m, each exactly once, however much the GPS wanders;
- the beacon aimed at the next corner (never across it), then about 15 m along the route, with the distance left to the destination;
- "You're off the route. Finding a new one." after 8 s more than 25 m away, then a new route from where the wearer is;
- arrival within 8 m. With no key or no network, the beacon points straight at the place and real routing is retried.

`position.heading` is a compass heading (0 north, 90 east). Without one, the walking direction over the last few fixes is used, and no beacon is sent while the wearer stands still. The sim and real GPS share one mapping, `geo.LocalFrame`: anchor it with `LocalFrame.from_landmarks([((lat, lng), (x, y)), ...])` on the same two or three landmarks that line up the splat, then `to_latlng(x, y)` for positions and `heading_of(dx, dy)` for headings. Route state goes out as `route` events (`started` with the polyline, `fallback`, `failed`, `rerouting`, `arrived`, `cancelled`) for the dashboard map.
