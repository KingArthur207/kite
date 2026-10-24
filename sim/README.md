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

## Placeholder parts that get replaced

- `src/world.ts` is the whole world: boxes for buildings and cars, capsules for people. The Gaussian splat replaces it and keeps the same exports (`scene`, `update`, `solids`).
- Cars and people become realistic glTF models so YOLO detects them in the camera frames. Their collision boxes stay.
- The splat has no collision, so walls and curbs will come from a hand-placed layer lined up with the splat.
- `ORIGIN` in `src/config.ts` (37.8013, -122.4477) maps to sim (0, 0). It gets re-anchored to splat landmarks so Karthik's route and the dashboard line up with the sim.
- `FX` in `src/config.ts` starts at 1000. Arthur's Tello calibration replaces it.
