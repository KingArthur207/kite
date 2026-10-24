# Kite sim

A browser street corner with an arrow-key walker and two virtual cameras. A drone camera sits 0.3 m above the walker's head and 1 m behind; a chest camera sits at 1.2 m. Both render 960x720 with the Tello's vertical FOV, `2*atan(360/FX)`. Frames go to perception through a local relay. Walker position and status go to the earpiece.

## World

The default world is Moss Street in SoMa, San Francisco, running from Howard St at its NW end, where the demo crossing is, to Folsom. It is built from OpenStreetMap: extruded building footprints, roads at estimated widths, sidewalks raised 15 cm, the zebra across Moss at Howard, and Howard's cycle track. Building faces and curbs block the walker, and each marked crossing has a gap in the curb. `?world=placeholder` loads the old box world.

- `scripts/osm_corner.py` fetches OSM and writes `public/osm/moss.json` in metres (x east, y north), with the origin at the Moss zebra, OSM node 12144593726. Run it again to refresh. `python scripts/osm_corner.py raw.json` reuses a saved Overpass response.
- `moss.json` also has `lanes`: car lane polylines in travel order with `speed_mps`. Moss flows SE (Howard to Folsom) and Howard flows SW. `src/osmWorld.ts` exports them as `lanes`, and the traffic drives on them (see Cars and people).
- The demo walk is `route` in `moss.json`, and the same points as lat/lng are in `public/osm/route.json`. It starts on the NE sidewalk of Moss 70 m before the zebra, facing NW, turns left across the zebra, and ends 20 m SW along Howard's south sidewalk, 98 m in all. The sim draws it faintly on screen but never in the camera frames.
- The numbers are under `measurements` in `moss.json`. OSM has no width tag for Moss, so the 6.1 m roadway is an assumption. Tape-measure it on site.

## Run

```
cd sim && npm install && npm run dev        # http://localhost:5173
python sim/relay.py                         # frames, ws://127.0.0.1:5020 (needs websockets)
cd earpiece && python -m earpiece run       # earpiece, ws://127.0.0.1:5011
python sim/test_relay.py                    # relay self-check
```

Start them in any order. The page reconnects to the relay and the earpiece every 2 s, and the HUD shows which ones are connected.

## Keys

| Key | Does |
|---|---|
| Up / Down | walk forward / back at 1.2 m/s |
| Left / Right | turn |
| B | black screen (HUD stays) |
| V | reveal the view |
| H | DEBUG: send one fake `crowd` hazard to the earpiece, only to prove sim to earpiece to beep for the skeleton gate |

The HUD counts collisions with buildings, cars, pedestrians and the bin.

## Metrics mode

`?metrics=1&mode=blind|kite|oracle&trials=20&seed=1` runs an autopilot with no keyboard input, always in the Moss world. A walker at 1.2 m/s follows the demo `route`: up Moss's NE sidewalk, left across the zebra, then along Howard's south sidewalk. On the way it meets a pedestrian crossing Moss into a doorway, a bin, a head-height sign, a car turning in from Howard's left lane through the zebra without yielding, and a pedestrian stepping out of a doorway on Howard. Their timing is seeded, so each trial is different and every run with the same seed is identical. The street's own traffic and people are not spawned in this mode. Building faces and curbs still block the walker, and touching one counts as a collision.

- `blind`: the walker never reacts. Runs time-accelerated (`speed`, default 300).
- `kite`: the walker reacts only to the earpiece on `ws://127.0.0.1:5011`. It stops 0.7 s after a `spoken` event with priority `critical` or `hazard`, a `state` with a critical sonar track, or a hazard in `dont_walk`/`wait`/`red`/`stop`. It walks on after 1.5 s with no cue and never detours. Runs in real time, since the pipeline is real time.
- `oracle`: an upper bound that reacts to the true obstacles in a 5 m corridor ahead, looking up to 3 s into the future. It is not Kite and must never be presented as Kite.

In every mode, a walker that stays blocked for 1 s or waits 8 s feels around the obstacle with a sidestep, the way a cane user would: 1 m toward the buildings on Moss's narrow sidewalk, 1.5 m elsewhere. A sidestep that runs into a wall or curb stops there. Each obstacle counts as one collision per walk at most; walls count every new contact. Results show in the HUD and in `window.kiteMetrics`, get logged as one `KITE_METRICS {...}` console line, and can be downloaded as JSON. Blind and oracle open no sockets, so they never feed a shared earpiece or relay.

Headless (starts its own vite on port 5287; pass `--url` to reuse one):

```
uv run --with playwright playwright install chromium        # first time
uv run --with playwright python scripts/sim_metrics.py --mode blind
uv run --with playwright python scripts/sim_metrics.py --mode oracle
```

Kite mode: start the relay, the detector on sim frames and the earpiece, then open `?metrics=1&mode=kite` or run `--mode kite`. That takes about 90 s a walk. Run it on a machine where nothing else is sending to the earpiece.

Results on Moss, seed 1, 20 walks: blind 61 collisions (3.05 a walk, every walk hit the bin, the sign and the car; one also hit the second pedestrian), oracle 0 collisions and 40 near misses (the bin and sign on the 1 m sidestep). Kite mode has not been run on Moss yet.

## Cars and people

`src/actors.ts` loads glTF models from `public/models/` (credits and licenses in `public/models/CREDITS.md`) and scales them to real size: cars are 4.3 to 5.3 m long, the bus 12 m, and people 1.65 to 1.82 m tall.

Moss world (`src/mossActors.ts`): cars drive the OSM `lanes`. Five cars and the bus run Howard's three SW lanes at about 10 m/s. Four cars come down Howard's left lane, turn left into Moss across the zebra at 4 m/s, and drive Moss SE at about 5 m/s; one passes the zebra every 15 to 20 s. Cars brake smoothly for slower segments, keep a 3 m gap to the car ahead, stop for anyone within 1.5 m of their path, and wait at the stop line while anyone (a pedestrian or the walker) is on or within 0.6 m of a zebra's stripes. At the end of its lane a car starts over once the start is clear. Five people walk back and forth on Howard's south sidewalk (across the zebra), the demo route (across the zebra), and both Moss sidewalks, waiting if a car is right in front of them. One more stands on Howard's south sidewalk NE of the corner. Collision boxes: a car is a chain of axis-aligned squares along its body, sized to stay inside its width at any heading; people get a 0.5 m box.

Placeholder world (`?world=placeholder`): three cars and the bus loop the main road at 8 to 11 m/s. Two cars loop the cross street at 7 m/s and wait at a stop line while someone is on the crosswalk or main-road traffic is within 18 m. Faster cars hold back behind slower ones in the same lane. Five people walk a loop on the sidewalk at 1.25 to 1.5 m/s with their walk animation, crossing the cross street on the crosswalk both ways. One more stands near the corner. Collision boxes come from each car model's bounds; people get a 0.5 m box.

YOLO detects them. `python sim/yolo_check.py` (usage at the top of the file) drives the sim in headless Chromium with a frozen clock, mocks the relay and earpiece sockets, and scores YOLO against the actors' projected boxes. In the placeholder world (`--url 'http://localhost:5199/?world=placeholder'`), on 168 frames (83 drone, 85 chest), counting instances at least 30 px tall:

| Model | Drone car | Drone person | Chest car | Chest person |
|---|---|---|---|---|
| yolo11n | 91% | 99% | 85% | 96% |
| yolo26n | 90% | 99% | 93% | 97% |

The ground truth ignores buildings, so a car hidden behind one counts as a miss and these numbers are a floor. The pickup crossing at a distance is the weakest model (about 70%). Examples: `docs/yolo_drone.jpg`, `docs/yolo_chest.jpg`, `docs/yolo_drone_road.jpg` (thick boxes are detections, thin white boxes are found ground truth, thin red ones are misses).

In the Moss world the check walks up Moss's NE sidewalk to Howard, turns to face the zebra and waits. With yolo11n on 651 frames: drone car 62%, drone person 34%, chest car 63%, chest person 37%. At 10 to 20 m it finds 98% of cars and 90% of people. The low totals are mostly occlusion the ground truth doesn't model: from Moss, the people on Howard's sidewalk and much of Howard's traffic are behind the corner buildings (the standing man 0/167, the two Howard walkers 2/231, the bus 4/100), and cars crossing right in front at the curb are cut off by the frame edge. Examples: `docs/yolo_moss_drone.jpg`, `docs/yolo_moss_chest.jpg`.

## What the sim sends

To the earpiece (`ws://127.0.0.1:5011`):
- `position` at 5 Hz: `heading` (compass degrees, 0 = north), `lat`, `lng`, `x`/`y` (meters east/north of the origin), `place` (`"Moss St at Howard St"`).
- `status` when the walker starts or stops: `state` is `walking` or `stopped`.

The sim emits no hazards from its own object list. Hazards come from the real detector running on the sim's frames (team decision).

To the relay (`ws://127.0.0.1:5020/pub`):
- Binary frames, 8 fps per camera: 1 byte camera id (0 drone, 1 chest), then JPEG bytes.
- Pose as JSON text at 5 Hz, once per camera: `{"type":"pose","camera":"drone","x":..,"y":..,"heading":..,"cam_height":..,"pitch":0}`.

Subscribers on `/sub` get the same messages. On connect they first get the latest frame and pose for each camera.

## Frames in Python

```python
import sys; sys.path.insert(0, "sim")
from frames import SimFrameSource

src = SimFrameSource("drone")      # or "chest"; url defaults to ws://127.0.0.1:5020/sub
ok, bgr = src.read()               # like cv2.VideoCapture: 960x720 BGR, blocks up to 2 s for a new frame
pose = src.pose                    # latest pose for this camera, or None
```

Needs `websockets`, `numpy` and `opencv-python`.

## Scene descriptions (Claude)

`sim/scene_describe.py` sends one frame to Claude every 6 s and forwards the reply to the earpiece as a `scene` event, with source `scene-chest` or `scene-drone`. The earpiece doesn't speak scenes. When the wearer asks "Kite, what's in front of me?", the brain answers from the latest scene under 90 s old. Run it from the repo root:

```
uv run --no-project --with anthropic --with opencv-python-headless --with websockets \
  --with python-dotenv --with numpy python sim/scene_describe.py --source sim-drone
```

`--source` takes `sim-chest`, `sim-drone`, a camera index (`0`), a video file, or a `kite.record` folder. Other flags are `--interval 6`, `--model` (default `claude-haiku-5-5`), `--camera chest|drone` to rename the source, and `--host`, which defaults to `KITE_EARPIECE_HOST`. `ANTHROPIC_API_KEY` is read from the repo-root `.env`.

Each request carries one JPEG, 768 px wide at quality 70, with a 4 s timeout and no retries. If the previous call is still running, the next one is skipped. The system prompt asks for one or two sentences about the path, people, vehicles, obstacles, crossings and readable signs. It forbids safety judgments like "safe to cross", guesses beyond the image, and descriptions of faces or identity. Frames stay in memory and are never written to disk.

The script also listens on the earpiece WebSocket (`ws://HOST:5011`). A `heard` event that matches the brain's "what's around/ahead/in front" intent, or says "describe" or "see", starts a description right away. The brain answers that question at once, offline, from the scene it already has, so the fresh description is ready for the next question. The 6 s interval keeps the first answer at most about 6 s old.

`python sim/test_scene_describe.py` runs the self-check with a fake client and no network.

## Stand-in detector (until streetseg lands)

`sim/standin_detect.py` runs YOLO11n on the chest and drone frames and sends the events perception owes the earpiece (spec 5.1), so the sim to AirPods path works before Rex's `streetseg` exists. It reads frames through `SimFrameSource`, or any video file with `--source`. Replace it with streetseg; nothing downstream changes.

- Chest camera: `obstacles` for every frame (kind, bearing from the box centre, distance from the box bottom on flat ground at the pose's camera height, falling back to a typical object height), plus `heartbeat` when nothing is in view.
- Drone camera: a `hazard` for `crowd` (3 or more people 1 to 15 m ahead of the walker) or `vehicle` (a car within 12 m and 3 m of the walker's path). A person cut off by the bottom edge is the walker and never counts. A hazard, or a new direction for one, must hold 3 frames before it is spoken. It is refreshed every 2 s with the same text and expires on its 4 s ttl, with no clear message. A `scene` count goes out every 10 s.

From the repo root:

```
uv run --no-project --with ultralytics --with opencv-python-headless --with lap --with websockets --with python-dotenv --with numpy --with scipy python sim/standin_detect.py
```

That runs on sim chest + drone frames and sends to `KITE_EARPIECE_HOST` (`.env`, default 127.0.0.1). Append flags to the same command:

```
--download                                    # once: fetches weights/yolo11n.pt (gitignored); never fetched at runtime
--source clips/mixkit-134.mp4 --camera drone --jsonl out/clip.jsonl --overlay out/clip.mp4
```

`python3 sim/test_standin_geometry.py` checks the geometry and the hazard gate without a model. `--jsonl` writes the same events with a `t` field for `scripts/replay.py`. `--overlay` writes an mp4 with boxes and bearing/distance labels. Other flags: `--fx` (default 1000), `--cam-height`, `--device`, `--fast` (don't pace a video to real time), `--no-send`, `--max-seconds`. It uses `mps` when available.

On Moss, with the walker stopped 4 m short of the corner facing NW, every car that turned in from Howard during a 60 s wait drew a drone `hazard` as it came down Moss toward the walker ("Car slightly left, 7 meters" when it was 8 to 9 m away at about -25 degrees). Two gaps showed up. On the walk up Moss, oncoming Moss cars also draw "Car slightly left, 6 meters": the 3 m lane test can't tell that the walker is on the sidewalk of a 6 m alley. Once the walker turns to face the zebra (SW), a turning car comes from behind-right and is only in frame when it is about 3 m away and cut off by the edge, and the stand-in gave no hazard for it.

On an M-series Mac, YOLO11n with tracking takes about 25 ms a frame on `mps` with the machine idle, and 75 to 90 ms with headless Chrome rendering the sim alongside. The sim's placeholder boxes and capsules are rarely detected; the glTF cars and people are.

Test clips come from Mixkit (free license, mixkit.co): "A man being followed walking in a city" (`https://assets.mixkit.co/videos/134/134-720.mp4`) is close to the drone's view, and "Times Square during a sunny day" (`.../4442/4442-720.mp4`) has lots of cars. Download them into the gitignored `clips/`.

## Live walk (spec section 4)

The judge wears a phone on their chest as the camera, headphones and a sleep mask, and walks 3 to 4 m. A teammate steps in front. Kite says "Stop. Person ahead." Then the reveal window replays what the chest camera saw, with boxes and the STOP banner.

The live path is the stand-in detector with a camera index as `--source`. `sim/live_walk.py` holds the chest-only parts, and they run on sim chest frames too:

- Distances are a median over 3 frames, and closing speed is a least-squares fit over the last second.
- Anything inside 1.6 m (`--stop-m`) and within 0.7 m of the walker's path is a stop. So is anything under 1 s from contact. A stop goes out with a closing speed that puts the earpiece's time to contact at 0.8 s. Its sonar then goes critical and says "Stop. Person ahead." at about 1.5 m instead of its own 0.75 m.
- When a box is cut off by the bottom edge (a person close to a chest camera), the distance comes from the top of the head, assumed 1.7 m (`--person-height`).
- `--pitch` is the camera's downward tilt in degrees. It goes into the ground-plane distance, and a phone clipped to a shirt usually points 10 to 20 degrees down.
- `--depth` adds Depth Anything V2 Small (metric, indoor) at 336x252 as a fallback for things YOLO has no class for, such as walls, doors and boxes. It reports something closer than 2.5 m that fills the centre of the frame and isn't floor. It takes about 18 ms a frame on `mps`.
- `--show` opens the reveal window: live view, labels (kind, bearing, distance, closing speed), and a red banner such as `STOP - person 1.4 m` that stays up 1.5 s. Press `r` to replay the last 20 s, `l` for live, `q` to quit. Heads are pixelated (`--no-blur` turns that off). The 20 s buffer is JPEG bytes in RAM and is never written to disk, and `--overlay` is refused for a live camera (spec 5.3).

Commands, from the repo root. `--show` needs `opencv-python` (with GUI), not the headless build:

```
RUN="uv run --no-project --with ultralytics --with opencv-python --with lap --with websockets --with python-dotenv --with numpy --with scipy --with transformers"
$RUN python sim/standin_detect.py --download --depth          # once, with internet: YOLO weights and the depth model
$RUN python sim/standin_detect.py --list-cameras              # which indices open, at what resolution
$RUN python sim/standin_detect.py --source 1 --show --cam-height 1.2 --pitch 15 --depth
```

Frames are resized to 960 wide. A live camera defaults to `--fx 700`, which fits an iPhone at 1x or a MacBook camera at 960 wide. Calibrate on the day: stand someone on a tape mark 2 m from the judge and change `--pitch` (then `--fx`) until the label reads 2.0 m.

To use the iPhone as the camera, sign it into the same Apple ID as the Mac, with Wi-Fi and Bluetooth on, and lock it in landscape near the Mac. Continuity Camera then shows it as another index in `--list-cameras`, usually 1 when the built-in camera is 0. Turn off Center Stage in Control Center > Video Effects so the field of view doesn't change mid-walk. To check you have the right index, run with `--show` and cover the phone's lens.

macOS asks for Camera access the first time; grant it to the terminal app you run from. If `--list-cameras` finds nothing, check System Settings > Privacy & Security > Camera.

Measured on an M5 MacBook with `--show` and nothing else running, on a stock clip: about 22 fps end to end with YOLO alone, and about 15 fps with `--depth`. The STOP line printed at 1.3 m in `clips/approach.mp4` (below), and the earpiece said "Stop. Person ahead." at critical priority.

Table checklist:

- [ ] Earpiece running with the headphones as output: `cd earpiece && .venv/bin/python -m earpiece run --output "<device name>"` (default is any AirPods)
- [ ] `--list-cameras`, then the detector running with `--show` on the phone's index
- [ ] Phone clipped flat to the judge's chest, about 1.2 m up, landscape, lens clear of jacket and lanyard; `--cam-height` and `--pitch` match
- [ ] Calibrated on the 2 m tape mark
- [ ] 4 m of clear floor taped out, with nothing at head height
- [ ] One spotter walks beside the judge, within arm's reach, the whole time
- [ ] One teammate steps in at about 2 m and stays put until Kite says "Stop"
- [ ] Sleep mask on after the headphones are in; take it off for the reveal
- [ ] Reveal: press `r` on the laptop, and turn the screen to the judge

Test clip for the live path without a second person: Pexels "Woman walking inside a building with camera" (`https://www.pexels.com/download/video/2795407/`, free under the Pexels license), with its last 9 s reversed so she walks toward a waist-height camera:

```
ffmpeg -ss 9 -i clips/pexels-2795407.mp4 -vf "fps=30,scale=960:-2,reverse" -an clips/approach.mp4
$RUN python sim/standin_detect.py --source clips/approach.mp4 --camera chest --show
```

## Files and placeholders

- `src/world.ts` is the placeholder box world, with its glTF traffic from `src/actors.ts`. `src/osmWorld.ts` is Moss St, with its traffic from `src/mossActors.ts`. Both export `scene`, `update` and `solids`. The Moss world also exports `spawn`, `walls`, `overlay` and `place` (see `WorldExtras`).
- `ORIGIN` in `src/config.ts` is the Moss zebra (37.778352, -122.408768) and maps to sim (0, 0), so the route and the dashboard use the same points.
- `FX` in `src/config.ts` starts at 1000. Arthur's Tello calibration replaces it.
