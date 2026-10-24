# Kite earpiece

The AirPods layer of Kite: spatial obstacle sonar, a waypoint beacon, and a voice assistant (Cartesia voice, Claude reasoning, Moss memory) for blind pedestrians.

## Run

```bash
cd earpiece
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[mac,brain,memory,dev]"
maturin develop --release -m native/Cargo.toml   # optional: Rust audio engine + binaural renderer
python -m earpiece selftest                      # check audio, voices, STT, controls, network
python -m earpiece run                           # the real thing (AirPods + stem presses)
python -m earpiece run --sim walk                # with a scripted street scenario
python -m earpiece render walk out/walk.wav      # render what the wearer would hear
python -m earpiece render --jsonl events.jsonl out/heard.wav   # same, from a JSONL timeline (docs/internal/video-runbook.md)
python -m pytest                                 # 330+ tests (also with KITE_AUDIO_ENGINE=python; KITE_LIVE_TESTS=1 adds live API checks)
```

Keys come from the repo-root `.env` (`CARTESIA_API_KEY`, `ANTHROPIC_API_KEY`, `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY`, `GOOGLE_MAPS_API_KEY`). Everything degrades gracefully without them.

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

## Credits and limits

Characters sent to Cartesia are counted across runs in `earpiece/.cache/cartesia_usage.json`; past `KITE_CARTESIA_CHAR_BUDGET` (default 150,000) only urgent speech uses the neural voice. Phrase warm-up runs one request at a time (our plan allows two in flight).

## Controls

AirPods (or any Bluetooth headphones' play button): once = talk, twice = repeat, three times = pause guidance (safety warnings stay on). Keyboard fallback: Enter / `r` / `p`, plus `t <words>` to type a command.

## Layout

| Module | Job |
|---|---|
| `mixer.py` / `native/` (Rust) | real-time mixing, ducking, limiter, device output; native binaural placement |
| `spatial.py`, `sounds.py` | binaural rendering (Brown-Duda head model; runs in Rust when built, sample-identical) and synthesised cues |
| `sonar.py`, `beacon.py` | obstacle warnings by time-to-contact; Soundscape-style waypoint beacon |
| `speech.py`, `listen.py` | Cartesia TTS/STT, priorities, preemption, time-to-live, offline fallback |
| `brain.py`, `memory.py` | local voice commands, Claude with tools, Moss memory of places and hazards |
| `route.py`, `geo.py` | Google Routes v2 walking directions turned into spoken turns and the beacon; lat/lng and sim meters |
| `judgment.py` | Claude judgment calls on hazards and scene ("Wait, the crosswalk is crowded."), with code guardrails |
| `vision.py`, `camera.py` | Claude vision: scene descriptions, "what's in front of me?" and reading text aloud from the body camera |
| `app.py`, `transport.py`, `controls.py` | wiring, UDP/WebSocket, AirPods stem presses |

## Walking directions

When the wearer asks for a saved, nearby or named place, the earpiece routes there itself (Google Routes API v2, walking) and follows every `position` event:

- the first instruction at the start, said in the wearer's frame when a heading is known ("Head toward Lyon Street, slightly left");
- a heads-up about 30 m before each turn ("In 30 meters, turn left onto Bay Street") and "Now turn left onto Bay Street" at about 6 m, each exactly once, however much the GPS wanders;
- the beacon aimed at the next corner (never across it), then about 15 m along the route, with the distance left to the destination;
- "You're off the route. Finding a new one." after 8 s more than 25 m away, then a new route from where the wearer is;
- arrival within 8 m. With no key or no network, the beacon points straight at the place and real routing is retried.

`position.heading` is a compass heading (0 north, 90 east). Without one, the walking direction over the last few fixes is used, and no beacon is sent while the wearer stands still. The sim and real GPS share one mapping, `geo.LocalFrame`: anchor it with `LocalFrame.from_landmarks([((lat, lng), (x, y)), ...])` on the same two or three landmarks that line up the splat, then `to_latlng(x, y)` for positions and `heading_of(dx, dy)` for headings. Route state goes out as `route` events (`started` with the polyline, `fallback`, `failed`, `rerouting`, `arrived`, `cancelled`) for the dashboard map.
