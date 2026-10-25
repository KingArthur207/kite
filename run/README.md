# run/

The main scripts. Each runs from the repo root, installed or not (`kite` is imported from the folder next to this one):

```bash
pip install -r requirements.txt          # once per laptop
python run/app.py --dry-run              # on the table: video, YOLO and commands, nothing reaches the motors
python run/app.py                        # fly
```

## app.py: take off, lock onto the person in front, follow them

A dark window opens: the map (placeholder) on the left, the camera with YOLO's person boxes on the right, and a
console underneath that lists every command sent and every reply received.

| Command | Does |
|---|---|
| `takeoff` | take off, find the person right in front of the drone, lock on, follow them |
| `land` | stop following and land |
| `e` | EMERGENCY: motors off at once (the drone drops if it is in the air), then the app quits |
| `status` | battery, height, video and follow state |
| `quit` | land, then close (closing the window does the same) |

Before `takeoff`, stand about 1 m in front of the drone, facing away from it. The drone flies behind and above your
head: `--height` above the takeoff point (default your `--walker-height` + 0.3 m) and `--behind` behind your head
(1.0 m with the retrained steering). If the person walks out of view the drone hovers and re-locks when they are back
in front of it.

Steering (`--yaw`): `fly`, the retrained kite.fly brain (default), or `law`, the follow controller's own yaw law.
The fly brain has only been tested in simulation so far: keep a hand on `e` and a spotter next to the walker.

Needs the YOLO weights once (`data/models/yolo11n.pt`, see `kite/follow.py`). Everything flies through kite's
SafetyGovernor: stick cap, 0.5 to 2.3 m altitude band, lands on low battery, hovers when video or telemetry stall.
