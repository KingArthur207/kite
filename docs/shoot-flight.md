# Shoot: flight checklist (Arthur)

The pilot's half of spec section 8. One battery, one take of about 5 minutes on Moss St at Howard, landing at 20 to 30%. Rex runs the shoot; this page is only what the pilot does and types.

## Tonight, on normal internet

- `git pull`, then `source .venv/bin/activate && pip install -e ".[dev]" && python -m pytest -m "not slow"` passes.
- Laptop charged, 5 GB or more free (a 5 minute take is about 0.7 GB).
- macOS: Local Network access for your terminal app and incoming connections for Python allowed (System Settings > Privacy & Security), so the first connect at the corner doesn't stall on a prompt.
- Terminal key repeat on (System Settings > Keyboard): holding `w` must keep the drone moving. Test it in the dry-run rehearsal below.
- Don't power the Tello tonight except for the one short ground check below. Streaming drains the battery.

## Ground check (drone on the ground, about 1 minute of power)

1. Close the Tello phone app. `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` prints nothing.
2. Power on the Tello, join its `TELLO-xxxxxx` Wi-Fi.
3. `python -m kite.record --source tello --out recordings --fly --dry-run --max-s 30`
   - The status line shows about 30 fps, a battery figure and `on ground`.
   - Press `t`, `w`, `r`, space, `l`: each shows in the status line as `rc ...` or an event, and nothing moves (dry run).
   - Hold `w` for 2 s: the `rc` value should stay at 30 the whole time, then drop to 0 within 1 s of letting go.
   - Pick the drone up and tilt the nose down for 2 s, then put it back.
4. Power off. Note the battery figure: it is the whole budget for the morning.
5. Back on normal Wi-Fi: `python -m kite.sources.recording recordings/<dir>`. The pitch range should go negative during the nose-down tilt. If it goes positive instead, the replay's pitch sign is backwards: change `PITCH_SIGN` in `kite/sources/recording.py` and tell Rex.

Camera calibration runs in this same power-on window, right after step 3, not separately:

- Before powering on: `python -m kite.calibrate --board-png board.png`, open it full screen at max brightness on a phone or a second laptop.
- `python -m kite.calibrate --source tello`. Hold the board 0.3 to 1.5 m from the camera; move it through the centre, corners and edges, tilted 20 to 45° in different directions, pausing until each beep. It stops itself at 15 views or 45 s and writes `data/tello_camera.json` (commit it; perception and the sim read fx from it).
- Power off as soon as "closing the drone link" prints. If it says BAD or "only N good views", rerun on the saved frames with `--source recordings/calib_<stamp>`: no drone needed.
- If the board is never detected (screen glare), use one object of known width at a taped distance: `--source tello --known-object 0.60 --distance-m 2.0`, then click its two edges.

## At the corner, 7:00 AM

Positions: the walker starts on the NE sidewalk of Moss about 70 m before Howard. Take off from the sidewalk about 1 m behind the walker, on flat paving, away from cars. The pilot walks 3 to 5 m behind the drone, laptop open on one forearm, with someone at the pilot's elbow watching the ground (the pilot is watching the screen).

1. Rehearse the route on foot with the drone off (Rex calls it).
2. Phone Tello app closed, `lsof` check empty, join the Tello Wi-Fi.
3. `python -m kite.record --source tello --out recordings --fly --speed 50`
   - Wait for the status line to show about 30 fps and the battery.
4. `t` to take off. It climbs to about 1 m.
5. Hold `r` until `h` in the status line reads the walker's height plus 0.3 m (about 2.0 m for a 1.7 m walker). The safety ceiling is 2.3 m.
6. Rex starts the walk. Hold `w` to follow at the walker's pace; the drone tops out around 0.5 m/s at `--speed 50`, so the walker goes slowly (0.5 to 0.7 m/s). Small `a`/`d` to stay behind their head, `q`/`e` to turn with them at the corner.
7. Space hovers at once. `l` lands. Ctrl+C lands and stops recording; keep clear until it is down.

Keys: `t` takeoff, `l` land, space hover, `w`/`s` forward/back, `a`/`d` left/right, `r`/`f` up/down, `q`/`e` turn, `!!` emergency (motors off, it falls: only if it is about to hit someone).

The drone also acts on its own: it lands at 20% battery or after 7 minutes in the air, and it hovers if video or telemetry stalls for 1 s (the status line says so). After a safety hover, press the move key again.

Land by 30% if the take is done, or when Rex calls it. To fly again in the same morning, restart the recorder.

## Right after landing

- Stay in the same folder; the recording is under `recordings/<date>_<time>_tello/`.
- Check the take: `python -m kite.sources.recording recordings/<dir>` prints frames, duration, dropped frames and the height range (`--preview` to watch it).
- Copy the folder to a second disk or a teammate before anything else.
- Hand the folder to Rex for perception. The video is MJPG in `video.mkv`; `ffmpeg -i video.mkv -pix_fmt yuv420p video.mp4` for Eddy's edit.
- Write down for the pitch: piloted by hand (not autonomous follow), the battery at landing, and anything that went wrong.

## Follow mode (only after the footage is safe)

Code: `kite/follow.py`. Tested in a simulation only; it has never flown. Set up once, on normal internet:

    pip install -e ".[follow]"
    mkdir -p data/models && curl -L -o data/models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

1. Rehearse on the morning's take, no drone: `python -m kite.follow --source recordings/<dir> --overlay`. It writes `follow_dry_run.csv` (the rc that follow would have sent on each frame) and `follow_dry_run.mp4` (the walker box, the target row, the sticks) into the take's folder. Check that the box stays on the walker, `yaw` has the same sign as the walker's offset (right is +), and `fb` goes up when the walker pulls away.
2. Fly: `python -m kite.record --source tello --out recordings --fly --follow`. Take off, climb to the walker's height plus 0.3 m, and get behind them by hand. The status line shows `follow off (walker conf .. box ..)` once it sees a person cut off by the bottom edge. Then press `g`.
3. Any key cancels follow first: space hovers, `w`/`a`/... take over, `l` lands. If it loses the walker for 0.5 s it hovers (rc 0), beeps and waits. It never searches. It picks the walker up again once they're back at the bottom edge.
4. Knobs: `--follow-behind 1.5` shows more of the head and shoulders, which helps if the detector misses a bare head. `--follow-height` sets the altitude to hold (default: the height when you press `g`). The drone can't go faster than `SafetyCfg.max_rc` (50 is about 0.5 m/s), so the walker keeps to 0.5 m/s.
