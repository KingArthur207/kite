# Kite sim

A browser street corner with an arrow-key walker and two virtual cameras. A drone camera sits 0.3 m above the walker's head and 1 m behind; a chest camera sits at 1.2 m. Both render 960x720 with the Tello's vertical FOV, `2*atan(360/FX)`. Frames go to perception through a local relay. Walker position and status go to the earpiece.

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

## What the sim sends

To the earpiece (`ws://127.0.0.1:5011`):
- `position` at 5 Hz: `heading` (compass degrees, 0 = north), `lat`, `lng`, `x`/`y` (meters east/north of the origin), `place: "sim corner"`.
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

- `src/world.ts` is the whole world: boxes for buildings and cars, capsules for people. The Gaussian splat replaces it and keeps the same exports (`scene`, `update`, `solids`).
- Cars and people become realistic glTF models so YOLO detects them in the camera frames. Their collision boxes stay.
- The splat has no collision, so walls and curbs will come from a hand-placed layer lined up with the splat.
- `ORIGIN` in `src/config.ts` (37.8013, -122.4477) maps to sim (0, 0). It gets re-anchored to splat landmarks so Karthik's route and the dashboard line up with the sim.
- `FX` in `src/config.ts` starts at 1000. Arthur's Tello calibration replaces it.
