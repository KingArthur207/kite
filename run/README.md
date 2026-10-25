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
| `takeoff` | take off, climb to `--height`, follow the person in front of the drone |
| `land` | stop following and land |
| `follow` | look for the person again (after "person not found") |
| `e` | EMERGENCY: motors off at once (the drone drops if it is in the air), then the app quits |
| `status` | battery, height, video and follow state |
| `quit` | land, then close (closing the window does the same) |

Before `takeoff`, stand 1 to 2 m in front of the drone, facing away from it, and give your real height
(`--walker-height`, default 1.7 m): the distance is read from the angle down to the top of your head, so 5 cm off is
10-20 % distance error.

1. **Climb.** Right after takeoff it climbs to `--height` above the floor (default walker height + 0.3 m, at least
   walker height + 0.2 m).
2. **Follow.** It locks onto the largest person in view (you) and holds `--behind` (default 1.0 m) behind you: it
   turns toward you, moves toward you only when every distance cue says it is safe, backs off when you come closer,
   and slides around behind you when you turn (from the pose model's shoulders, once they are in view). It holds its
   height above the floor the whole time: the downward sensor while it agrees with the takeoff height, else that.
3. **Lost.** It holds, then turns toward where it last saw you. After 20 s it drops to just below head height and turns
   30° at a time to find you; not found after a full turn, it hovers until you type `follow`.

Steering (`--yaw`): `law` turns toward the person (default); `fly`, the retrained kite.fly brain, has only been tested
in simulation. Keep a hand on `e` and a spotter next to the walker.

Needs the YOLO pose weights once (`data/models/yolo11n-pose.pt`):

```bash
mkdir -p data/models && curl -L -o data/models/yolo11n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
```

Everything flies through kite's SafetyGovernor: stick cap, 0.5 to 2.3 m altitude band, lands on low battery, hovers
when video or telemetry stall. The follow itself is `kite/follow_behind.py`.
