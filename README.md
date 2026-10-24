# kite

A DJI Tello (standard model) flies about 30 cm above and behind a visually impaired walker's head and talks them
through the route in their AirPods; a body-worn webcam catches sudden obstacles near their feet.

## Setup (once per laptop, on normal internet: the Tello's Wi-Fi has none)

```bash
uv venv --python 3.12 --seed .venv      # or: python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -m "not slow"          # ~15 s, no drone needed
python -m pytest                        # + the H.264-over-UDP reader tests (needs ffmpeg, ~75 s more)
```

## Drone layer (`kite/`)

| Module | What it does |
|---|---|
| `drone/tello.py` | `TelloDrone`: djitellopy for commands + state. Discrete moves are non-blocking (`busy()` / `last_result()`), late replies after `stop` are discarded, `streamon` is re-sent until video arrives, `dry_run=True` never sends motion. |
| `drone/safety.py` | `SafetyGovernor`: wraps any drone; clamps rc, altitude ceiling/floor, lands on low battery or flight time, hovers on stale video/telemetry. Call `check()` every loop. |
| `sources/` | Newest-frame video readers: `TelloVideoSource` (PyAV or the team's OpenCV `VideoStream`, auto-reconnect), `WebcamSource`, file replay. |
| `config.py` | `DroneCfg` / `SafetyCfg` defaults. |

```python
from kite.config import Config
from kite.drone import SafetyGovernor, make_tello

drone = SafetyGovernor(make_tello(cfg := Config(), dry_run=True), cfg.safety)
drone.connect()                          # join the TELLO-xxxxxx Wi-Fi first
frames = drone.frame_source()
```

## Before flying

- Latency / stream check: `python tests/drone_test.py --ground --preview`, then the flight version (3 x 3 m clear,
  prop guards, battery >= 30 %). Ctrl+C lands.
- Only one program can hold the Tello ports: `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` should print nothing.
- macOS: allow incoming connections for Python, and Local Network access for your terminal app
  (System Settings > Privacy & Security) if sends fail with "No route to host".

## Recording

`python -m kite.record` saves the video and every Tello state packet, both stamped with one monotonic clock. Only one
program can hold the Tello ports, so the recorder is also the flight controller: `--fly` adds keyboard control through
`SafetyGovernor` in the same process. Without `--fly` the drone is opened in dry run and never gets a motion command.

```bash
python -m kite.record --source clip.mp4 --out recordings               # dry run on a file: no drone, no telemetry
python -m kite.record --source tello --out recordings                  # record only, drone on the ground / in hand
python -m kite.record --source tello --out recordings --fly --dry-run  # rehearse the keys: motion is logged, not sent
python -m kite.record --source tello --out recordings --fly            # the shoot
```

- Keys (`--fly`, typed in the recorder's terminal, one at a time): `t` takeoff, `l` land, `space` hover, `w`/`s`
  forward/back, `a`/`d` left/right, `r`/`f` up/down, `q`/`e` turn. A move key drives its stick at `--speed` (30)
  for `--pulse` (1 s); hold it (key repeat) to keep going, let go and the drone stops. `!!` = emergency (motors off).
- **Ctrl+C lands and stops** (it keeps recording until the drone is down). Takeoff is refused without live video or
  at <= 20 % battery; the governor lands by itself at 20 % or after 7 min.
- The terminal prints fps, battery and height every second.
- Output: `<out>/<YYYYmmdd_HHMMSS>_<source>/` with `video.mkv` (MJPG, playable after a crash), `frames.csv`
  (frame index -> `t_mono`), `telemetry.jsonl` (raw state + `t_mono`), `events.jsonl`, `meta.json`. Align video and
  telemetry on `t_mono`. QuickTime can't open `.mkv`: VLC can, or `ffmpeg -i video.mkv -pix_fmt yuv420p video.mp4`.
- About 0.7 GB per 5 minutes. Don't commit `recordings/`.
