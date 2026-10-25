<h1 align="center">
  <img src="site/public/assets/kite-mark.svg" alt="" width="54" valign="middle">&nbsp;Kite
</h1>

<p align="center"><b>A guide dog that flies.</b></p>

<p align="center"><img src="docs/img/devpost/kite-banner-wide.png" alt="Kite" width="100%"></p>

Kite is our Cal Hacks 13.0 project. The full write-up is on [Devpost](https://devpost.com/software/kite-od2vmu).

## Why we built it

Over a million Americans are blind (JAMA Ophthalmology, 2021), and fewer than 10,000 guide dogs work in the US, enough for less than 1% of them. A guide dog can cost over $100,000 to raise, train and support over its life, and the wait for one is about a year. So we built a guide dog that flies, detects obstacles, and helps blind people get around through computer vision and real-time speech AI.

## What it does

Kite watches the street from two angles. A DJI Tello flies 1.8 m behind you and 20 cm above your head, looking down the street. A Logitech C270 webcam on your chest, run by an Arduino UNO Q, watches the area right in front of you. Together they scan your surroundings at up to 18 frames per second, and Kite talks you through what they see in your AirPods.

When something is in your way, Kite says "Stop. Person ahead." in Cartesia's voice about 0.1 s after it decides, or tells you to veer left or right, and then guides you forward again. The drone recognizes you when it starts, holds its place behind you, and if it loses you it pings you and turns to search until it finds you again. You can ask "What's in front of me?", "Find me the nearest shopping centre" or "Take me to my usual coffee shop", and Kite answers out loud and walks you there.

## How we built it

- Hardware: DJI Tello, Arduino UNO Q with a Logitech webcam, an iPhone for GPS, AirPods, a MacBook, and a Windows laptop that flies the drone.
- Perception: Python, GPS, YOLO11n and YOLO-World with a street vocabulary, Depth Anything V2, and our own near-field distance estimator.
- Voice and AI: Cartesia Sonic and Ink, Claude Haiku 5.5 and Claude Opus 5.5, Google Routes and Places for walking directions, Grok Imagine for test scenes.
- Shared state: a SpacetimeDB module on Maincloud and a React dashboard.
- Drone control: Python on the Tello SDK, with a 1,446-neuron circuit steering the yaw.
- Testing: many nights walking around Berkeley, where we found out the vision model works surprisingly well in the dark.

## SpacetimeDB

SpacetimeDB holds the live state of every walk for five devices: the drone laptop, the earpiece, the UNO Q body camera, the walker's phone and the dashboard. We run no sync server of our own.

The module has two public tables. `event` keeps the walk's recent history as a 500-row ring, and `latest` keeps the newest state for each device, so a dashboard opened mid-walk is current immediately. The `push` reducer checks every write against an allowed list of event types, a source-name pattern, a priority and a 16 KB limit, and `clear` resets a run with a 60-second cooldown.

The earpiece mirrors every event through a bounded queue with a token bucket and backoff, so a dead network slows the mirror down without stalling the audio. The phone's GPS reaches the database through the UNO Q. The UNO Q also uses SpacetimeDB to be found: every two seconds it posts an HMAC-signed heartbeat that the laptop reads to locate the body camera on campus Wi-Fi, so nobody types an IP address, and stale or replayed heartbeats are rejected. The dashboard reads only through a live subscription.

Code: [`dashboard/spacetimedb/src/index.ts`](dashboard/spacetimedb/src/index.ts), [`earpiece/earpiece/spacetime.py`](earpiece/earpiece/spacetime.py).

## Cartesia

The walker can't look at a screen, so Kite's interface is voice. Everything the walker hears is Sonic, and everything they say is heard by Ink.

Sonic `sonic-3.6` streams over one persistent WebSocket with a separate context for each utterance, which gets the first audio out in about 0.1 s with no handshake per sentence. When "Stop." has to cut off a direction, Kite cancels that context, fades the audio out in 40 ms, and brings the direction back later if it still applies. Audio arrives as 48 kHz float PCM, the AirPods' native rate, and goes straight into our mixer. Speech plays at 1.1x, because many blind listeners prefer it faster. Every warning phrase is rendered with Sonic at startup and cached on disk, so "Stop." plays instantly in the same voice even with no signal.

Ink `ink-whisper` streams the walker's speech while they talk. After 0.7 s of silence Kite sends `finalize`, and the transcript is back 90 to 190 ms later. We listen through the MacBook mic, because opening the AirPods mic drops them into mono and would flatten the directional beeps.

### Known limitations

- New sentences need a connection. Offline, Kite switches to the macOS voice.
- Ink has no offline fallback, so Kite says "Voice commands are offline" and the AirPod buttons keep working.
- English only for now.

Code: [`earpiece/earpiece/speech.py`](earpiece/earpiece/speech.py), [`listen.py`](earpiece/earpiece/listen.py).

## Grok Imagine

The moments Kite exists for are the ones nobody can stage: a car turning into the crosswalk, a cyclist cutting across, a man stepping out of a doorway, people walking straight at you, a scooter lying across the sidewalk. Grok Imagine generated all five from chest height. The stills are 5 of the 16 labelled frames in our detector benchmark, and our Claude tests paint fake "cross now" signs onto them to check that Kite never reads a sign as an instruction. Rex used Cursor as his coding agent for the UNO Q code that sends the phone's GPS to SpacetimeDB, the Mapbox walking preview and the drone recorder on Windows. The team used Grok for research and planning.

Code: [`scripts/imagine_scenes.py`](scripts/imagine_scenes.py), [`data/imagined/`](data/imagined/).

## Claude

Kite is for blind and low-vision adults who walk city streets on their own. Claude handles the parts that need judgment. Claude Haiku 5.5 reads the current hazards and scene and can add one short sentence, such as "Wait, the crosswalk is crowded," through a single strict tool and within 2.5 s. Claude Opus 5.5 describes the camera view, reads signs and menus aloud, and answers questions with tools for directions and settings.

Claude doesn't detect obstacles, so a bad answer can't hide one. A filter in code drops any reply that sounds like permission to go, judged on what a listener would hear after undoing lookalike letters and hidden characters, and text read off a sign is spoken as a quote. We tested that filter against 662 hand-written ways to say "go ahead" plus 650 disguised variants.

Code: [`earpiece/earpiece/judgment.py`](earpiece/earpiece/judgment.py), [`vision.py`](earpiece/earpiece/vision.py), [`earpiece/evals/`](earpiece/evals/).

## Hardware

Four off-the-shelf devices work as one guide: a DJI Tello, an Arduino UNO Q with a Logitech webcam, an iPhone and a pair of AirPods.

The Tello flies follow mode. It locks onto the walker once they show up in three of five detections, holds 1.8 m behind and 20 cm above head height, uses its downward time-of-flight sensor for altitude, and turns toward where the walker was when a frame drops. If it loses them, it searches in 30° turns. A safety governor clamps every stick command, keeps altitude between 0.5 and 2.3 m, hovers after one second without video, and lands at 20% battery, after seven minutes of flight or past a 100 m geofence. We fought its 2.4 GHz video link all weekend and got it working with band planning, a lighter stream and our own packet-loss-tolerant reader.

The UNO Q is a body camera that sets itself up. At boot a systemd unit switches its USB port into host mode, it picks up any webcam plugged in later, and it streams MJPEG to the laptop. It finds the laptop through a signed heartbeat in SpacetimeDB, and it only sends its access token to Tailscale addresses.

The iPhone runs our Kite app, which sends GPS over Bluetooth LE and starts with one press of the Action button. Kite picks the AirPods by name, plays "Left ear" and "Right ear" so you can check they're the right way round, warns when the battery is low, and turns stem presses into talk, repeat and pause.

Code: [`kite/`](kite/), [`hardware/arduino/unoq_link/`](hardware/arduino/unoq_link/), [`phone/`](phone/), [`phone-app/`](phone-app/).

## Under the hood

Kite runs a drone controller, two vision pipelines, a spatial audio engine and a speech scheduler that all have to agree within a fraction of a second.

Distance from one webcam. For every object in the walker's path, the near-field estimator computes up to five distances in log space, each with its own uncertainty. Two come from the pinhole model using the object's known height and width, with an upper bound when the frame edge cuts the box off. One is a ray to the point where the object touches the ground. One is Depth Anything V2, rescaled so the floor sits at the camera's known height. The last is looming, a least-squares fit of how fast the object grows over 0.9 s. A weighted-median gate throws out outliers, the rest are combined by inverse variance, and a Kalman filter per track follows log-distance and its rate of change, which is minus one over the time to contact. A time to contact under 1.5 s jumps straight to the top priority. When the detector can't name something, an occupancy check over 0.25 m slices of the path still catches the obstacle, along with walls, curbs and steps down. Code: [`perception/near/`](perception/near/).

Audio that keeps time. The audio engine is Rust (cpal and rtrb, bound to Python with PyO3). Its device callback never blocks on Python: commands arrive through lock-free single-producer rings, and finished sample buffers go back through a third ring to be freed off the audio thread. A test plays on the real output device while five threads load the CPU and the mixer, and requires zero glitches. A native sonar thread runs in macOS's real-time scheduling band and places every beep at an absolute sample frame. When Python held its lock for 500 ms in a test, every beep still landed within one 512-frame block. To keep the Rust and Python paths identical, we reimplemented numpy's PCG64 generator and ziggurat sampler in Rust, so even the seeded noise inside a cue matches sample for sample. Code: [`earpiece/native/`](earpiece/native/).

3D sound and a speech scheduler. Spatial audio uses interaural time differences and a head-shadow filter from a spherical head model (Brown and Duda, 1998), with a low-pass that marks sounds behind you, so no HRTF dataset is needed. Speech runs through a scheduler with six priority levels, mid-word preemption with a 40 ms fade, and an expiry time on directions so they're never spoken late.

A fruit fly flies the drone. Follow mode estimates the walker's range from four cues, including the angle to their head computed from the drone's own pitch telemetry. Yaw is steered by a 1,446-neuron leaky integrate-and-fire pursuit circuit from the fruit-fly connectome, built on the open-source FlyDrones network, with a predictor that covers the video's 0.25 s lag. We flew it on the real Tello. Code: [`kite/follow_behind.py`](kite/follow_behind.py), [`kite/flybrain/`](kite/flybrain/), [`kite/fly/`](kite/fly/).

Pytest collects about 6,100 tests across the stack, including Hypothesis property tests on the drone's safety governor.

## Built for the walker

A cane can't feel a mirror or a branch at head height. Kite's chest camera covers that height, and the drone looks past the next step to the crosswalk. The walker stays in charge: Kite says "Stop." and waits instead of pulling you around obstacles, holds back "walk" until the signal has been steady for 0.6 s, and drops a direction that would arrive too late. Urgency comes through as one of three tones, which you can react to faster than a sentence, and you can pause guidance at any time. It all runs on consumer hardware, for a fraction of what one guide dog costs.

## Challenges

Our drone's video fell apart. We were testing indoors in an apartment packed with Wi-Fi networks, and the Tello's 2.4 GHz video turned into blocky frames because packets kept dropping. We took a train out to Berkeley's open fields to test, and bought a $30 USB antenna for the Windows laptop. The drone itself also fell apart, over and over, and we kept refitting the bumpers.

We tried to reset our own Arduino with a staple. The UNO Q wouldn't take our password and we were sure we'd locked ourselves out, so we tried to force the chip into its emergency download (EDL) mode and reflash it by shorting two tiny test pads with a bent staple. It never worked. When we finally flashed through Arduino App Lab, it told us the password had never been set.

By Sunday morning the drone, the Arduino, the phone and the AirPods were working together as one guide.

## What we learned

- A crowded 2.4 GHz band is a fight, and a drone's video loses it.
- Distance from one camera comes from several weak clues, and most of the accuracy comes from knowing when to stop trusting each one.
- Anything that has to happen on time can't wait on Python, which is why Kite's audio runs in Rust.
- Before you reach for a staple, check whether the password exists.

## What's next

First, we want to put Kite in the hands of the people it's built for. We'll walk real routes with blind travelers and the orientation and mobility instructors who teach them, and ask how Kite compares to a white cane or a guide dog. Their answers will decide how early Kite warns, when it beeps instead of speaking, and which sounds are impossible to mix up.

After that:

- Landing zones in pilot areas like college campuses, where drones recharge between walks. A walker asks for a guide through their AirPods, and Kite sends a drone to their phone's location.
- Perception on the phone, so a walker carries only a phone, a chest camera and headphones.
- A segmentation model for curbs, steps and doorways.
- A faster drone with longer flights.

## Team

- Karthik Subramanian built the earpiece that every other part feeds: the spatial sonar, the beacon toward the next turn, the Cartesia voice in and out, and stop and wait. He moved beep synthesis and the sonar thread into Rust, built the closeness detector that turns one webcam into three priority tones, and benchmarked depth models for it. He also built the Arduino UNO Q link for the body camera, the SpacetimeDB database, the Claude safety evals and the site.
- Arthur Armaing owned the drone. He wrote the Tello layer, camera calibration, replay, preflight checks and the fix for Tello video on macOS, then built follow mode, tested it in a follow simulator, flew it on the real Tello, and built the fruit-fly yaw controller. He leads the demo and the pitch.
- Eddy Zhao built the phone link: an iPhone app that sends GPS over Bluetooth to receivers on the laptop and the UNO Q, which keep the newest position in SpacetimeDB. He added "take me to", which turns a spoken destination into a walking route, and built the Moss St simulator, the demo scripts and the video tools. He wrote the team spec and owns the demo video and the Devpost.
- Rex Heng built the map view that routes the walker's live phone GPS to a destination on Mapbox, and the UNO Q code that publishes that GPS to SpacetimeDB. He fixed the drone recorder's controls on Windows, and his laptop ran as Kite's main computer during integration.

## Run it

You need macOS on Apple Silicon, Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 20.19 or later, and the [SpacetimeDB CLI](https://spacetimedb.com/install) for the offline fallback. Put the Cartesia, Anthropic, SpacetimeDB, Google Maps and xAI keys in `.env` (see `.env.example`) to run everything.

Install once (the demo script tells you if any of these is missing):

```bash
cp .env.example .env    # CARTESIA_API_KEY, ANTHROPIC_API_KEY, GOOGLE_MAPS_API_KEY, ...
cd earpiece && uv venv --python 3.12 .venv && uv pip install --python .venv -e ".[mac,brain,memory,dev]" && cd ..
(cd dashboard && npm install) && (cd sim && npm install)
uv run --no-project --with ultralytics --with opencv-python-headless --with lap --with websockets \
  --with python-dotenv --with numpy --with scipy python sim/standin_detect.py --download   # detector weights
```

Then one command runs the whole demo in the background:

```bash
scripts/demo.sh sim                    # earpiece, relay, sim, dashboard, detector on the sim's cameras
scripts/demo.sh live iPhone            # earpiece, dashboard, detector on that camera (index or name) with the reveal window
scripts/demo.sh replay data/sample_walk.jsonl
scripts/demo.sh bodycam                # the UNO Q body camera -> P1 / P2 / P3 -> beeps (finds the board on the network)
scripts/demo.sh status                 # what is up, recent log lines, then netcheck
scripts/demo.sh stop
```

`--output "<device>"` picks the headphones (by default the earpiece looks for AirPods), `KITE_OFFLINE=1` switches to a local SpacetimeDB server, and `KITE_DETECT_ARGS="--depth"` passes extra flags to the detector.

<details>
<summary>Start each piece by hand</summary>

1. Earpiece (terminal 1):

   ```bash
   cd earpiece && source .venv/bin/activate
   python -m earpiece selftest            # audio, voices, STT, controls, network
   python -m earpiece run --spacetime     # listens on 127.0.0.1:5010 (UDP) and :5011 (WS)
   ```

   `python -m earpiece run --sim walk` plays a scripted walk with no other processes.

2. Dashboard (terminal 2): `cd dashboard && npm run dev` (connects to Maincloud).

3. Sim (terminal 3) and its frame relay (terminal 4, from the repo root):

   ```bash
   cd sim && npm run dev
   earpiece/.venv/bin/python sim/relay.py         # ws://127.0.0.1:5020
   ```

   Vite gives the second app 5174 if 5173 is taken; use the URL each one prints. In the sim, the arrow keys walk, `B` blacks out the screen and `V` reveals it.

4. Replay the sample walk (terminal 5, from the repo root):

   ```bash
   earpiece/.venv/bin/python scripts/replay.py data/sample_walk.jsonl --spacetime
   ```

   You should hear beeps and speech and see the walker, hazards and spoken lines appear on the dashboard. `--speed 2` and `--loop` also work.

</details>

**Tests.** `cd earpiece && python -m pytest`, and `earpiece/.venv/bin/python sim/test_relay.py` for the relay. Before a merge, `scripts/check.sh quick` (about 3 minutes) or `scripts/check.sh` for everything: both audio engines, Rust, the drone layer, the dashboard build and the security scans. The landing site deploys with `scripts/deploy-site.sh`.

Two laptops over a Thunderbolt bridge, flying and recording the Tello, and walker GPS are in [`docs/operations.md`](docs/operations.md).

## Repo map

| Path | What it is |
|---|---|
| `scripts/demo.sh` | Starts and stops the whole demo (`sim`, `live`, `replay`, `bodycam`, `status`, `stop`) |
| `earpiece/` | The AirPods layer in Python, with the Rust audio engine in `earpiece/native/`. See [`earpiece/README.md`](earpiece/README.md) |
| `dashboard/` | The SpacetimeDB module and the React dashboard. See [`dashboard/README.md`](dashboard/README.md) |
| `perception/` | Detector benchmarks and the near-field distance estimator (`perception/near/`) |
| `kite/` | Drone layer: Tello control, safety governor, recorder, calibration, preflight, follow mode and the fruit-fly yaw controller |
| `hardware/arduino/unoq_link/` | The UNO Q body camera |
| `phone/`, `phone-app/` | The iPhone GPS link and app |
| `sim/` | The three.js Moss St simulator and its detector. See [`sim/README.md`](sim/README.md) |
| `scripts/imagine_scenes.py`, `data/imagined/` | Grok Imagine test scenes |
| `site/` | The landing page at trykite.pages.dev |
| `docs/` | Operations guides and images. Index in [`docs/README.md`](docs/README.md) |

## Credits

- Stock street footage used for testing: Mixkit "A man being followed walking in a city" and "Times Square during a sunny day" (Mixkit free license), and Pexels "Woman walking inside a building with camera" (video 2795407, Pexels license). The segmentation frames in `docs/img/seg-*.png` are from Mixkit "Sidewalk in a city with a couple of people walking" (35810) and "Times Square during a sunny day" (4442), Mixkit free license. None of the footage itself is committed.
- 3D car and pedestrian models in the sim: Quaternius (CC0), and "Range Rover by IvOfficial and Bus by Poly by Google, via Poly Pizza, licensed CC-BY 3.0". The full list is in [`sim/public/models/CREDITS.md`](sim/public/models/CREDITS.md).
- Map tiles and street data: © OpenStreetMap contributors, ODbL. Google Photorealistic 3D Tiles and Street View on Moss St, with Google's attribution on screen.
- Perception models: Ultralytics YOLO11n (AGPL-3.0) and Depth Anything V2 Small (metric, indoor; Apache-2.0). In `perception/`: Ultralytics YOLO-World v2-S (AGPL-3.0; YOLO-World by Tencent AI Lab), OpenAI CLIP ViT-B/32 to embed its prompts (MIT), and Depth Anything V2 Small metric outdoor (Apache-2.0). The depth shoot-out scores against DIODE (Vasiljevic et al., MIT). Ideas, not code: Google's ARCore Depth API and DepthLab (Apache-2.0) for depth from motion, depth confidence and collision checks, and ScrollSafe by MrAnitude, whose box-height rule is the near-field baseline.
- Fruit-fly pursuit circuit in [`kite/fly/`](kite/fly/): a subgraph of the MaleCNS v1.0 connectome (Janelia FlyEM et al., CC-BY 4.0), run with the leaky integrate-and-fire model and parameters of Shiu et al., Nature 2024, and tuned with pycma (CMA-ES).
- Video segmentation ([`sim/seg_standin.py`](sim/seg_standin.py)): Ultralytics YOLO26n-sem trained on ADE20K (`yolo26n-sem-ade20k.pt`), AGPL-3.0 like the rest of Ultralytics. ADE20K is from MIT CSAIL (Zhou et al., "Scene Parsing through ADE20K Dataset", CVPR 2017).
- Open-source libraries: djitellopy, three.js, 3d-tiles-renderer, SpacetimeDB, React, Leaflet, react-leaflet, Vite, NumPy, SciPy, sounddevice, websockets, httpx, the Anthropic Python SDK, OpenCV, PyAV, Ultralytics, Hugging Face Transformers, Playwright, cpal, rtrb, PyO3. Services: Google Routes API v2, Places API (New), Map Tiles API and Street View Static API
- Head model for binaural rendering: Brown and Duda's spherical-head model
- Built with Claude Code: Claude worked alongside the team on the earpiece, the Rust audio engine, perception, the safety evals and this site.
- Built with Cursor: Rex used Cursor's agent for the UNO Q GPS publisher, the Mapbox walking preview and the Windows drone recorder.

## License

[MIT](LICENSE)
