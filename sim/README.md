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

## Placeholder parts that get replaced

- `src/world.ts` is the static world: boxes for buildings, the bin and flat ground. The Gaussian splat replaces it and keeps the same exports (`scene`, `update`, `solids`). The glTF cars and people from `src/actors.ts` stay, composited into the splat.
- The splat has no collision, so walls and curbs will come from a hand-placed layer lined up with the splat.
- `ORIGIN` in `src/config.ts` (37.8013, -122.4477) maps to sim (0, 0). It gets re-anchored to splat landmarks so Karthik's route and the dashboard line up with the sim.
- `FX` in `src/config.ts` starts at 1000. Arthur's Tello calibration replaces it.
