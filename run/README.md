# run/

The main scripts. Each runs from the repo root, installed or not (`kite` is imported from the folder next to this one):

```bash
pip install -r requirements.txt          # once per laptop
python run/app.py --dry-run              # on the table: video, YOLO and commands, nothing reaches the motors
python run/app.py                        # fly
python run/demo.py                       # same window, webcam, YOLO26n obstacles (add --seg for SegFormer-B0)
```

## demo.py: the same window, on a webcam

`python run/demo.py` opens the kite window and guides the walker from the webcam with two switches, the
**YOLO** and **Segmentation** buttons on the camera pane (or `yolo` and `seg`, with `on`/`off`, typed in the console):

| On | Runs | Says |
|---|---|---|
| YOLO (the default) | YOLO26n, about 25 ms a frame on a laptop CPU | "Bear right to go around the bicycle.", "Stop. A chair is blocking the way." |
| Segmentation | SegFormer-B0 (Mapillary), the light reader and YOLO26n's people and vehicle counts, about 0.4 s a frame | "Bear left. Fewer people that way.", "Stop. The light looks red." |
| Both | all of the above, one decision from the ground and the objects together | either, whichever applies |

Both use the same lanes: the ground within 5 m is split into left, center and right, and the walker is sent down the
clearest. With YOLO on, every detected object (except the walker and signals) blocks the ground under its box, and
an object that covers less than 12 % of a lane is walked past rather than steered around. With segmentation off
the ground is taken as walkable except under a box. Segmentation loads the first time it is turned on (`--seg` turns it
on at launch, `--no-yolo` starts with YOLO off). With both off the pane shows the plain camera. A connected Logitech C270 is chosen over the built-in camera. The checkpoints are in `third_party/videotesting/weights/` and come down with a pull. `python run/app.py` is unchanged and still talks only to the drone.

## app.py: take off, lock onto the person in front, follow them

A dark window opens: the map (placeholder) on the left, the camera with YOLO's person boxes on the right, and a
console underneath that lists every command sent and every reply received. The same picture is saved while the
window is open: `recordings/<YYYYmmdd_HHMMSS>_app/video.mkv` and one JPEG per frame in that folder's `frames/`.
The console prints the folder when recording starts and the frame count when the window closes.

| Command | Does |
|---|---|
| `takeoff` | take off, climb to `--height`, follow the person in front of the drone |
| `land` | stop following and land |
| `follow` (or `follow me`) | look for the person again (while holding, or once they have been out of view for 2 s) |
| `hold` / `resume` | pause and hover / turn until the person is seen, then follow |
| `stop` | while holding: look for the person again |
| `e` | EMERGENCY: motors off at once (the drone drops if it is in the air), then the app quits |
| `status` | battery, height, video and follow state |
| `quit` | land, then close (closing the window does the same) |

Before `takeoff`, stand about 2 m in front of the drone, facing away from it, and give your real height with shoes
(`--walker-height`, default 1.7 m, to +-2 cm): the distance is read from it, so 5 cm off is 10-20 % distance error.
The defaults are the settings the follow flew well with: 1.9 m high (walker height + 0.2), 1.8 m behind, forward speed
capped at 35, orbit off (`--orbit` turns it on).

1. **Climb.** Right after takeoff it climbs to `--height` above the floor in one move.
2. **Follow.** It locks onto the largest person in view (you) and holds `--behind` behind you: it turns toward you,
   moves toward you only when every distance cue says it is safe, and backs off when you come closer. It holds its
   height above the floor the whole time: the downward sensor while it agrees with the takeoff height, else that.
3. **Lost.** Missing for a moment (a dropped detection), it keeps looking where you were, creeping forward slowly
   in case it is falling behind, and never switches to someone else. Then it turns toward where it last saw you.
   After 20 s it drops to just below head height and turns 30° at a time to find you; not found after a full turn,
   it says so and hovers until you type `follow`.

What the drone says ("Ready.", "Landing.", "I can't see you. ...") shows in the console; `--speak` also says it out
loud.

Steering (`--yaw`): `fly` (default) is the fruit fly brain it flew with: a 1,446-neuron pursuit circuit from the
fruit fly connectome, trained then fine-tuned for a smooth stick (`kite/flybrain`, `FLY-YAW_smooth2_best.json`). It
decides only the yaw stick; distance, height and safety stay as above. `law` turns toward the person by bearing;
`retrained` and `pd` are kite.fly's own yaw sources (simulation-tested only). Keep a hand on `e` and a spotter next
to the walker.

The YOLO weights are in `data/models/` and come down with a pull (`yolo11n-pose.pt` for the follow, `yolo11n.pt` for detection). To fetch the pose file again:

```bash
mkdir -p data/models && curl -L -o data/models/yolo11n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
```

Everything flies through kite's SafetyGovernor: stick cap, 0.5 to 2.3 m altitude band, lands on low battery, hovers
when video or telemetry stall. The follow itself is `kite/follow_behind.py`.
