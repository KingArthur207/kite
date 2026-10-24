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

## Placeholder parts that get replaced

- `src/world.ts` is the placeholder box world and `src/osmWorld.ts` is Moss St. Both export `scene`, `update` and `solids`. The Moss world also exports `spawn`, `walls`, `overlay` and `place` (see `WorldExtras`).
- Cars and people become realistic glTF models so YOLO detects them in the camera frames. Their collision boxes stay.
- The splat has no collision, so walls and curbs will come from a hand-placed layer lined up with the splat.
- `ORIGIN` in `src/config.ts` is the Moss zebra (37.778352, -122.408768) and maps to sim (0, 0). The splat gets lined up to the OSM layer, so the route and the dashboard use the same points.
- `FX` in `src/config.ts` starts at 1000. Arthur's Tello calibration replaces it.
