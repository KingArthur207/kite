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

Before `takeoff`, stand about 1 m in front of the drone, facing away from it, and give your real height
(`--walker-height`, default 1.7 m): the distance is read from the angle to the top of your head, so 5 cm off is
10-20 % distance error. Right after takeoff it locks onto the largest person in view (you), then climbs to `--height`
(default walker height + 0.3 m) and holds `--behind` (default 1.0 m) behind you. It moves toward you only when every
distance cue says it is safe, backs off when you come closer, and holds its height the whole time. If you walk out of
view it holds, turns toward where you were for 20 s, then hovers, and re-locks once you are back in front of it.

Steering (`--yaw`): `law` turns toward the person (default); `fly`, the retrained kite.fly brain, has only been tested
in simulation. Keep a hand on `e` and a spotter next to the walker.

Needs the YOLO weights once (`data/models/yolo11n.pt`, see `kite/follow.py`). Everything flies through kite's
SafetyGovernor: stick cap, 0.5 to 2.3 m altitude band, lands on low battery, hovers when video or telemetry stall.
The follow itself is `kite/follow_behind.py`.
