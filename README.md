# Kite

Kite is a guide dog for blind pedestrians that flies. A DJI Tello hovers about 30 cm above and 1 m behind the walker's head, where it can see the route ahead, crowds and crossings farther out than a cane reaches. A webcam worn on the chest covers the ground near the walker's feet, which the drone's camera can't see. Everything reaches the walker through AirPods: directional beeps that speed up and rise in pitch as something gets closer, and Cartesia voice for directions and answers.

Each cue maps to a job a guide dog does. Kite stops the walker at hazards and waits until the way is clear instead of steering around them, and it never tells anyone it is safe to cross. Those decisions stay in deterministic code. Claude answers questions and can warn or tell the walker to wait, but it can't clear a stop.

Built at Cal Hacks 13.0 by Arthur, Eddy, Karthik and Rex.

## How it works

```
 sim (browser) --JPEG frames--> relay :5020 --> perception (streetseg, coming)
 chest webcam ------------------------------------> perception        |
 recorded Tello video -----------------------------> perception        |
                                                                       v  earpiece events (UDP 5010 / WS 5011)
 sim walker pose, route, replayed JSONL --------------------------> earpiece --> AirPods
                                                                       |
                                       every event it receives, mirrored over HTTP
                                                                       v
                                       SpacetimeDB (Maincloud, kite-sync-pl6rp) --> caregiver dashboard
```

The safety path is local. Producers send JSON events straight to the earpiece over UDP or WebSocket, and the earpiece turns them into sound. SpacetimeDB holds the shared world state that the dashboard and other machines watch, so a dropped internet connection never silences a warning.

What is real and what is simulated, as of this commit:

| Part | State |
|---|---|
| Earpiece: spatial sonar, waypoint beacon, speech queue, Cartesia voice in and out, Claude assistant, offline fallbacks | Real, runs on a laptop with any headphones or AirPods |
| SpacetimeDB event tables and the caregiver dashboard | Real, live on Maincloud |
| Sim street corner | Arrow-key walker, virtual drone and chest cameras matched to the Tello, collision counter. The world is a placeholder of boxes and capsules today; an OpenStreetMap model of the shoot corner (Moss St at Howard, SoMa) with realistic car and pedestrian models is in progress |
| Hazards in the sim | Will come from the real detector running on sim frames. Until perception lands, the only sim hazard is a debug key (`H`) |
| Perception (`streetseg`) | Coming; Rex is moving it into this repo |
| Drone layer (`kite/`) | Written and tested, on the `feat/drone-layer` branch, not yet on `main` |
| Real Tello footage | From the Saturday 7 AM shoot on the corner [TODO: confirm the take worked, and whether it was piloted or simple follow]. The drone does not fly at the venue (drones are banned there) |
| Collision numbers (sim metrics mode) | The harness runs. Scripted blind walks hit 4.6 obstacles per walk (seed 1, 20 walks). The number with Kite needs perception running and hasn't been measured |

## Repo map

| Path | What it is |
|---|---|
| `earpiece/` | The AirPods layer (Python, optional Rust audio engine in `earpiece/native/`). Event contract in `earpiece/earpiece/events.py`. See `earpiece/README.md` |
| `dashboard/` | SpacetimeDB module (`spacetimedb/src/index.ts`) and the React caregiver dashboard. See `dashboard/README.md` |
| `sim/` | three.js street corner, frame relay (`relay.py`) and `SimFrameSource` for perception (`frames.py`). See `sim/README.md` |
| `scripts/replay.py` | Replays a JSONL of events into the earpiece, and optionally into SpacetimeDB |
| `scripts/audit.py` | Pre-submission check: secrets, `.env`, repo visibility, README, large files, model weights |
| `scripts/netcheck.py` | Checks this laptop's setup: the earpiece over UDP and WebSocket, SpacetimeDB, internet to Claude, Cartesia and Google, and which keys are set |
| `scripts/offline.sh` | Switches to a local SpacetimeDB server when the internet is gone (`scripts/offline.sh off` undoes it) |
| `scripts/sim_metrics.py` | Runs sim metrics mode headless and prints collision counts |
| `data/sample_walk.jsonl` | A recorded 40-second walk in the earpiece event format |
| `data/sample_track.jsonl` | A moving walker track with hazards and a beacon, for testing the dashboard map |
| `tests/drone_test.py` | Tello latency and ground-check script |
| `docs/` | Team spec (`docs/superpowers/specs/`), Devpost draft, demo runbook |
| `kite/` | Drone layer. Coming to `main` from `feat/drone-layer` |
| `streetseg/` | Street segmentation and obstacle detection. Coming |

## Quick start on one laptop

You need macOS on Apple Silicon, Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 20.19 or later, and the [SpacetimeDB CLI](https://spacetimedb.com/install) for the offline fallback. Copy the env file and fill in what you have. Everything degrades without keys: no Cartesia key means macOS `say`, no Anthropic key means local voice commands only.

```bash
cp .env.example .env    # CARTESIA_API_KEY, ANTHROPIC_API_KEY, GOOGLE_MAPS_API_KEY, ...
```

1. Earpiece (terminal 1):

   ```bash
   cd earpiece
   uv venv --python 3.12 .venv && source .venv/bin/activate
   uv pip install -e ".[mac,brain,memory,dev]"
   python -m earpiece selftest            # audio, voices, STT, controls, network
   python -m earpiece run --spacetime     # listens on 127.0.0.1:5010 (UDP) and :5011 (WS)
   ```

   `python -m earpiece run --sim walk` plays a scripted walk with no other processes. `--output "<device>"` picks an output device; by default it looks for AirPods.

2. Dashboard (terminal 2):

   ```bash
   cd dashboard && npm install && npm run dev     # connects to Maincloud
   ```

3. Sim (terminal 3) and its frame relay (terminal 4):

   ```bash
   cd sim && npm install && npm run dev
   earpiece/.venv/bin/python sim/relay.py         # from the repo root; ws://127.0.0.1:5020
   ```

   Vite gives the second app 5174 if 5173 is taken; use the URL each one prints. In the sim, the arrow keys walk, `B` blacks out the screen, `V` reveals it.

4. Replay the sample walk (terminal 5, from the repo root):

   ```bash
   earpiece/.venv/bin/python scripts/replay.py data/sample_walk.jsonl --spacetime
   ```

   You should hear beeps and speech from the earpiece and see the walker, hazards and spoken lines appear on the dashboard. `--speed 2` and `--loop` also work.

Tests: `cd earpiece && python -m pytest`, and `earpiece/.venv/bin/python sim/test_relay.py` for the relay.

## Two laptops over a Thunderbolt bridge

The wearable laptop runs the earpiece and drives the AirPods. The drone laptop runs perception or a replay. Connect them with a Thunderbolt cable; macOS brings up a Thunderbolt Bridge interface with a self-assigned address.

On the wearable laptop:

```bash
ipconfig getifaddr bridge0                       # note this address
cd earpiece && source .venv/bin/activate
python -m earpiece run --bind 0.0.0.0 --spacetime
```

On the drone laptop, set the earpiece address in the repo-root `.env` and send events as usual:

```bash
KITE_EARPIECE_HOST=169.254.x.y                   # the address from bridge0 above
earpiece/.venv/bin/python scripts/replay.py data/sample_walk.jsonl
```

Run `python3 scripts/netcheck.py` on both laptops to confirm the link; `docs/network.md` has the full setup. `EarpieceClient` and `scripts/replay.py` both read `KITE_EARPIECE_HOST`. The sim's browser page still connects to the earpiece at `ws://127.0.0.1:5011`, so run the sim on the wearable laptop. If sends fail with `No route to host`, allow your terminal under System Settings > Privacy & Security > Local Network.

## Cartesia features used

All of this is in `earpiece/earpiece/speech.py`, `listen.py` and `config.py`.

- Text to speech with Sonic (`sonic-3.6`, API version `2026-08-14`), voice "Daniel" (`47c38ca4-5f35-497b-b1a3-415245fb35e1`) at speed 1.1 through `generation_config`, English.
- Streaming over one persistent TTS WebSocket. Each utterance is its own `context_id`, raw `pcm_f32le` audio is fed into the mixer as it arrives, and a lower-priority utterance is cancelled (`"cancel": true`) when a hazard preempts it. The code comments report about 100 ms to first sound.
- A phrase cache. At startup the earpiece renders every fixed safety phrase ("Stop.", "Don't walk.", "Crowd ahead.", "Person, at 2 o'clock." and the rest) with Cartesia and keeps the PCM on disk in `earpiece/.cache/tts`, so warnings play instantly and work offline. Only changing sentences, like street names and Claude's answers, are streamed.
- Speech to text with `ink-whisper` over the streaming STT WebSocket (`pcm_s16le`). Audio is sent while the wearer is still talking, and the code comments report 87 ms from end of speech to final transcript. Push-to-talk by AirPods stem press or Enter; optional hands-free mode with the wake word "Kite".

Limitations: Cartesia needs internet. Without it, cached phrases still play in the Cartesia voice, but new sentences fall back to macOS `say` (voice "Samantha", about 1.5 s slower), and speech input stops working. If the first audio chunk takes more than 2.5 s, that utterance also falls back to `say`. We record from the laptop mic, because opening the AirPods mic switches them to mono call audio and the spatial cues collapse.

## Claude

`earpiece/earpiece/brain.py` handles anything the wearer says that isn't a local command (stop, repeat, pause, louder, beeps off and so on, which work offline). It sends `claude-opus-5-5` the question with a JSON snapshot of what Kite currently knows: nearby objects, hazards, the route, the latest drone scene text, position and the wearer's saved places. The reply streams back and is spoken sentence by sentence, so the first words play before the answer is finished. The system prompt is cached, and Claude has four tools: `navigate_to`, `control`, `remember` and `alert_caregiver`.

The prompt tells Claude to describe only what is in the context and never to say a street is safe to cross. Sonar warnings, `critical` obstacles and `dont_walk` hazards are handled before Claude sees anything, so they keep working when Claude is slow or unreachable.

Still to build (spec section 5.6): the judgment layer that has Claude decide "wait" or "continue" from the current hazards, and Claude describing drone frames for the `scene` text.

## SpacetimeDB

The module in `dashboard/spacetimedb/src/index.ts` has two tables. `event` keeps the newest 500 events (`id`, `type`, `source`, `priority`, `payload` as JSON, `t`). `latest` holds the newest event per type and source, so a dashboard that opens mid-walk sees the current state right away. Reducers are `push` and `clear`.

Python writes with plain HTTP calls to the `push` reducer (`earpiece/earpiece/spacetime.py`, no SDK), on a background queue that drops events instead of blocking the audio loop. High-rate `obstacles` and `heartbeat` events are throttled to two per second per source. The earpiece mirrors every event it receives when run with `--spacetime`, and `scripts/replay.py --spacetime` does the same. The dashboard subscribes over WebSocket and draws the walker on a live map (Leaflet over OpenStreetMap) with a heading arrow, trail, the active hazards placed around the walker by bearing and distance, and the next waypoint. Beside the map it lists hazards, the last spoken line, each device with how long since it last sent, and a live feed. It shows no camera video. Its "Clear run" button calls `clear` before a new run.

Offline, the same module runs on a local server: `scripts/offline.sh` starts it, publishes the module and points the dashboard at it. See `docs/demo-runbook.md`.

## Known limitations

- Perception isn't on `main` yet, so the sim produces no real hazards and nothing runs on webcam or Tello frames in this repo today.
- The sim world is a placeholder until the OpenStreetMap model of Moss St at Howard and the car and pedestrian models land. We decided against a Gaussian splat: the judge walks with the screen black, and the real Tello footage of the corner is what shows it.
- Nothing in this repo makes the drone follow the walker on its own. [TODO: state whether the shoot footage was piloted.] The drone never flies at the venue.
- The Tello tops out around 0.96 m/s at full stick (extrapolated from a stick-30 test), slower than the 1.2 to 1.3 m/s people usually walk, so the walker in our footage walks slowly. One battery gives about 6 to 8 minutes with video on.
- From 30 cm above and 1 m behind, the Tello's fixed camera can't see the ground for the first few meters ahead. That is the chest webcam's job.
- A pedestrian signal 20 m away is about 14 px tall in the Tello stream, too small to read reliably. Crossing locations come from maps data, and signal handling is shown in the sim.
- Kite stops and waits; it never plans a detour around an obstacle.
- Voice output and input, Claude and SpacetimeDB need internet. The fallbacks are cached phrases plus macOS `say`, local voice commands, and a local SpacetimeDB server.
- Sim metrics mode has a blind baseline (4.6 collisions per walk) and an oracle upper bound that reacts to ground truth (0.45). The oracle is not Kite. The number with Kite's real cues needs perception running.
- In metrics mode the scripted walker steps 1.5 m to the side when something blocks it for a few seconds, the way a cane user would. Kite itself never suggests the sidestep.

## Credits

- Stock street footage used for testing: [TODO: clip titles, authors and Pexels or Pixabay links]
- 3D car and pedestrian models in the sim: [TODO: model names, authors, licenses]
- Map tiles and street data: © OpenStreetMap contributors, ODbL
- Perception models: [TODO: final list from Rex, with licenses]
- Open-source libraries: djitellopy, three.js, SpacetimeDB, React, Leaflet, react-leaflet, Vite, NumPy, SciPy, sounddevice, websockets, httpx, the Anthropic Python SDK, Moss (`inferedge-moss`)
- Head model for binaural rendering: Brown and Duda's spherical-head model

## Team

| Name | Built |
|---|---|
| Arthur | Drone layer, flight and recording |
| Eddy | SpacetimeDB tables, dashboard and map, sim and metrics mode, replay, network tools, submission |
| Karthik | Earpiece: audio, Cartesia voice, Claude assistant, destinations |
| Rex | Perception (`streetseg`), the interview, the shoot |
