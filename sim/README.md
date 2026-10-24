# Kite sim

A browser street corner with an arrow-key walker and two virtual cameras. A drone camera sits 0.3 m above the walker's head and 1 m behind; a chest camera sits at 1.2 m. Both render 960x720 with the Tello's vertical FOV, `2*atan(360/FX)`. Frames go to perception through a local relay. Walker position and status go to the earpiece.

## World

The default world is Moss Street in SoMa, San Francisco, running from Howard St at its NW end, where the demo crossing is, to Folsom. It is built from OpenStreetMap: extruded building footprints, roads at estimated widths, sidewalks raised 15 cm, the zebra across Moss at Howard, and Howard's cycle track. Building faces and curbs block the walker, and each marked crossing has a gap in the curb. `?world=placeholder` loads the old box world.

- `scripts/osm_corner.py` fetches OSM and writes `public/osm/moss.json` in metres (x east, y north), with the origin at the Moss zebra, OSM node 12144593726. Run it again to refresh. `python scripts/osm_corner.py raw.json` reuses a saved Overpass response.
- `moss.json` also has `lanes`: car lane polylines in travel order with `speed_mps`, for whoever drives traffic. Moss flows SE (Howard to Folsom) and Howard flows SW. `src/osmWorld.ts` exports them as `lanes`.
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

`?metrics=1&mode=blind|kite|oracle&trials=20&seed=1` runs an autopilot with no keyboard input. A walker at 1.2 m/s follows a fixed route: up the sidewalk, across the crosswalk, then west around the corner. On the way it meets a pedestrian crossing, a bin, a head-height sign, a car through the crosswalk and a second pedestrian. Their timing is seeded, so each trial is different and every run with the same seed is identical. The ambient cars and people are hidden in this mode.

- `blind`: the walker never reacts. Runs time-accelerated (`speed`, default 300).
- `kite`: the walker reacts only to the earpiece on `ws://127.0.0.1:5011`. It stops 0.7 s after a `spoken` event with priority `critical` or `hazard`, a `state` with a critical sonar track, or a hazard in `dont_walk`/`wait`/`red`/`stop`. It walks on after 1.5 s with no cue and never detours. Runs in real time, since the pipeline is real time.
- `oracle`: an upper bound that reacts to the true obstacles in a 5 m corridor ahead. It is not Kite and must never be presented as Kite.

In every mode, a walker that stays blocked for 1 s or waits 5 s feels around the obstacle with a 1.5 m sidestep, the way a cane user would. Results show in the HUD and in `window.kiteMetrics`, get logged as one `KITE_METRICS {...}` console line, and can be downloaded as JSON. Blind and oracle open no sockets, so they never feed a shared earpiece or relay.

Headless (starts its own vite on port 5287; pass `--url` to reuse one):

```
uv run --with playwright playwright install chromium        # first time
uv run --with playwright python scripts/sim_metrics.py --mode blind
uv run --with playwright python scripts/sim_metrics.py --mode oracle
```

Kite mode: start the relay, the detector on sim frames and the earpiece, then open `?metrics=1&mode=kite` or run `--mode kite`. That takes about 50 s a walk. Run it on a machine where nothing else is sending to the earpiece.
## Cars and people

`src/actors.ts` loads glTF models from `public/models/` (credits and licenses in `public/models/CREDITS.md`) and scales them to real size: cars are 4.3 to 5.3 m long, the bus 12 m, and people 1.65 to 1.82 m tall. Three cars and the bus loop the main road at 8 to 11 m/s. Two cars loop the cross street at 7 m/s and wait at a stop line while someone is on the crosswalk or main-road traffic is within 18 m. Faster cars hold back behind slower ones in the same lane. Five people walk a loop on the sidewalk at 1.25 to 1.5 m/s with their walk animation, crossing the cross street on the crosswalk both ways. One more stands near the corner. Collision boxes come from each car model's bounds; people get a 0.5 m box.

YOLO detects them. `python sim/yolo_check.py` (usage at the top of the file) drives the sim in headless Chromium with a frozen clock, mocks the relay and earpiece sockets, and scores YOLO against the actors' projected boxes. On 168 frames (83 drone, 85 chest), counting instances at least 30 px tall:

| Model | Drone car | Drone person | Chest car | Chest person |
|---|---|---|---|---|
| yolo11n | 91% | 99% | 85% | 96% |
| yolo26n | 90% | 99% | 93% | 97% |

The ground truth ignores buildings, so a car hidden behind one counts as a miss and these numbers are a floor. The pickup crossing at a distance is the weakest model (about 70%). Examples: `docs/yolo_drone.jpg`, `docs/yolo_chest.jpg`, `docs/yolo_drone_road.jpg` (thick boxes are detections, thin white boxes are found ground truth, thin red ones are misses).

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

On an M-series Mac, YOLO11n with tracking takes about 25 ms a frame on `mps` with the machine idle, and 75 to 90 ms with headless Chrome rendering the sim alongside. The sim's placeholder boxes and capsules are rarely detected; the glTF cars and people are.

Test clips come from Mixkit (free license, mixkit.co): "A man being followed walking in a city" (`https://assets.mixkit.co/videos/134/134-720.mp4`) is close to the drone's view, and "Times Square during a sunny day" (`.../4442/4442-720.mp4`) has lots of cars. Download them into the gitignored `clips/`.

## Placeholder parts that get replaced

- `src/world.ts` is the placeholder box world and `src/osmWorld.ts` is Moss St. Both export `scene`, `update` and `solids`. The Moss world also exports `spawn`, `walls`, `overlay` and `place` (see `WorldExtras`).
- Cars and people become realistic glTF models so YOLO detects them in the camera frames. Their collision boxes stay.
- `src/world.ts` is the static world: boxes for buildings, the bin and flat ground. The Gaussian splat replaces it and keeps the same exports (`scene`, `update`, `solids`). The glTF cars and people from `src/actors.ts` stay, composited into the splat.
- The splat has no collision, so walls and curbs will come from a hand-placed layer lined up with the splat.
- `ORIGIN` in `src/config.ts` is the Moss zebra (37.778352, -122.408768) and maps to sim (0, 0). The splat gets lined up to the OSM layer, so the route and the dashboard use the same points.
- `FX` in `src/config.ts` starts at 1000. Arthur's Tello calibration replaces it.
