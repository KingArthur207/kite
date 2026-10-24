# Kite earpiece

The AirPods layer of Kite: spatial obstacle sonar, a waypoint beacon, and a voice assistant (Cartesia voice, Claude reasoning, Moss memory) for blind pedestrians.

## Run

```bash
cd earpiece
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[mac,brain,memory,dev]"
maturin develop --release -m native/Cargo.toml   # optional: Rust real-time audio engine
python -m earpiece selftest                      # check audio, voices, STT, controls, network
python -m earpiece run                           # the real thing (AirPods + stem presses)
python -m earpiece run --sim walk                # with a scripted street scenario
python -m earpiece render walk out/walk.wav      # render what the wearer would hear
python -m pytest                                 # 166 tests
```

Keys come from the repo-root `.env` (`CARTESIA_API_KEY`, `ANTHROPIC_API_KEY`, `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY`). Everything degrades gracefully without them.

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

## Controls

AirPods stem: once = talk, twice = repeat, three times = pause guidance (safety warnings stay on). Keyboard fallback: Enter / `r` / `p`, plus `t <words>` to type a command.

## Layout

| Module | Job |
|---|---|
| `mixer.py` / `native/` (Rust) | real-time mixing, ducking, limiter, device output |
| `spatial.py`, `sounds.py` | binaural rendering (Brown-Duda head model) and synthesised cues |
| `sonar.py`, `beacon.py` | obstacle warnings by time-to-contact; Soundscape-style waypoint beacon |
| `speech.py`, `listen.py` | Cartesia TTS/STT, priorities, preemption, time-to-live, offline fallback |
| `brain.py`, `memory.py` | local voice commands, Claude with tools, Moss memory of places and hazards |
| `app.py`, `transport.py`, `controls.py` | wiring, UDP/WebSocket, AirPods stem presses |
