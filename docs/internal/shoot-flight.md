# Shoot: flight checklist (Arthur)

The pilot's half of spec section 8. One battery, one take of about 5 minutes on Moss St at Howard, landing at 20 to 30%. Rex runs the shoot; this page is only what the pilot does and types.

## Tonight, on normal internet

- `git pull`, then `source .venv/bin/activate && pip install -e ".[dev]" && python -m pytest -m "not slow"` passes.
- Laptop charged, 5 GB or more free (a 5 minute take is about 0.7 GB). `python -m kite.preflight --quick` checks both, plus the deps, ffmpeg and the ports (on normal Wi-Fi the Tello Wi-Fi line only warns).
- macOS: Local Network access for your terminal app and incoming connections for Python allowed (System Settings > Privacy & Security), so the first connect at the corner doesn't stall on a prompt.
- Terminal key repeat on (System Settings > Keyboard): holding `w` must keep the drone moving. Test it in the dry-run rehearsal below.
- Don't power the Tello tonight except for the one short ground check below. Streaming drains the battery.

## Ground check (drone on the ground, about 1 minute of power)

1. Close the Tello phone app. Power on the Tello, join its `TELLO-xxxxxx` Wi-Fi.
2. `python -m kite.preflight`. It checks the laptop (ports, disk, battery, Wi-Fi, terminal), then opens the drone in dry run (no motion is ever sent) for about 6 s: battery, temperature, first frame, fps, frame size, telemetry rate and command latency. It closes the link itself and ends with GO or NO-GO; every NO-GO line says what to fix. Fallback if it can't run: `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` must print nothing.
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

### If the video shows no image on a Mac

`--video-backend socket` is an opt-in decoder (the default stays `pyav` until socket has run against a real Tello): kite binds UDP 11111 itself with the largest receive buffer the OS allows (8 MB on our Macs) and feeds PyAV's parser, with no FFmpeg network code involved. The other readers are `--video-backend pyav` (default) and `opencv`; all three work in `kite.record`, `kite.preflight` and `kite.calibrate` all accept).

1. Check the permissions first, because they block every decoder the same way:
   - System Settings > Privacy & Security > Local Network: your terminal app (Terminal, iTerm, VS Code) is on.
   - System Settings > Network > Firewall: off, or python is allowed to accept incoming connections.
   - The Tello phone app is closed, and so is any other Tello script.
   - `lsof -nP -iUDP:11111` prints nothing before you start.
2. `python -m kite.preflight --video-backend socket`. When there is no frame, the NO-GO line says which case you are in: no packets at all (permissions, firewall, the port is held, or the drone isn't streaming), packets but no keyframe yet, or keyframes that never decode cleanly (packet loss).
3. If the socket reader also shows nothing, try `--video-backend pyav`, then run `python -m kite.video_doctor` (Arthur's tool, once `fix/mac-video` is merged). It records the raw packets and decodes them offline, so send the folder it writes to whoever is debugging the video.

## At the corner, 7:00 AM

Positions: the walker starts on the NE sidewalk of Moss about 70 m before Howard. Take off from the sidewalk about 1 m behind the walker, on flat paving, away from cars. The pilot walks 3 to 5 m behind the drone, laptop open on one forearm, with someone at the pilot's elbow watching the ground (the pilot is watching the screen).

1. Rehearse the route on foot with the drone off (Rex calls it).
2. Phone Tello app closed, join the Tello Wi-Fi, then `python -m kite.preflight`. Fly only on GO: NO-GO below 50% drone battery, busy ports, no video, or the wrong Wi-Fi. On GO it prints the next command. (Fallback: `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` prints nothing.)
3. `python -m kite.record --source tello --out recordings --fly --speed 50`
   - Wait for the status line to show about 30 fps and the battery.
4. `t` to take off. It climbs to about 1 m.
5. Hold `r` until `h` in the status line reads the walker's height plus 0.3 m (about 2.0 m for a 1.7 m walker). The safety ceiling is 2.3 m.
6. Rex starts the walk. Hold `w` to follow at the walker's pace; the drone tops out around 0.5 m/s at `--speed 50`, so the walker goes slowly (0.5 to 0.7 m/s). Small `a`/`d` to stay behind their head, `q`/`e` to turn with them at the corner.
   Corner scout at Howard (team decision): about 3 m before the corner, fly ahead 2 m past the corner, climb to about 3 m (`up`), and turn (`q`) to look down Howard for cars. Stay over the sidewalk, never over the roadway. Hold while any car passes and the walker waits, then come back down to head height plus 0.3 m behind the walker and follow them across the zebra. The spotter watches the walker the whole time, since the drone isn't. Follow mode doesn't do this on its own yet; fly it by hand.
7. Space hovers at once. `l` lands. Ctrl+C lands and stops recording; keep clear until it is down.
   - Letting go of a move key keeps the stick on for up to `--pulse` (1 s), about 0.5 m at speed 50. Tap space the moment the walker stops.

Keys: `t` takeoff, `l` land, space hover, `w`/`s` forward/back, `a`/`d` left/right, `r`/`f` up/down, `q`/`e` turn, `!!` emergency (motors off, it falls: only if it is about to hit someone).
Arrow keys, function keys, Ctrl+Z, Ctrl+\\ and Ctrl+S do nothing while flying (they used to move or stop the program). Capitals work like lowercase.

The drone also acts on its own: it lands at 20% battery or after 7 minutes in the air, and it hovers if video or telemetry stalls for 1 s (the status line says so). After a safety hover, press the move key again.

If the status line says `WI-FI DOWN`, the drone keeps hovering (it may hold its last stick until it hears from the laptop, and lands itself after 15 s of silence). Stop walking, keep within 3 to 5 m, don't hold a move key, and wait for it to clear; then press the key again. Before flying, turn off Auto-Join for other known Wi-Fi networks so macOS rejoins the Tello. If `telemetry stale -> hover` stays while the Wi-Fi is fine, press `l` and restart the recorder.

`t` is refused without live video or telemetry. If the drone is visibly up but the status isn't `FLYING` about 8 s after `t`, press `l`. A land that fails is re-sent every 2 s on its own.

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
3. Any key cancels follow first: space hovers, `w`/`a`/... take over, `l` lands. If it loses the walker for 0.5 s it hovers (rc 0), beeps and waits. It never searches. It picks the walker up again once they are back at the bottom edge. A safety hover or `WI-FI DOWN` switches follow off for good: press `g` again once it clears.
4. Knobs: `--follow-behind 1.5` shows more of the head and shoulders, which helps if the detector misses a bare head. `--follow-height` sets the altitude to hold (default: the height when you press `g`). The drone can't go faster than `SafetyCfg.max_rc` (50 is about 0.5 m/s), so the walker keeps to 0.5 m/s.
   - `--follow-yaw pd|fly` (a kite.fly yaw source: a predictive PD or the fruit-fly circuit) is experimental and has only run in simulation, where neither passed the ship gate. Not for the shoot: leave it at the default, `law`.

### Follow with prediction (`--follow-predict`)

`kite/predict.py` estimates the walker's velocity and turn rate (a Kalman filter that also knows "standing still"), predicts their next 1.5 s, and plans the drone's sticks over that 1.5 s so it stays about 1.2 m back along the path the walker already walked. Yaw turns toward where the walker is about to be, so it starts turning a little before the corner does. Lost walker (hover after 0.5 s), coast, altitude hold, the stick clamps, the safety governor and every refusal are the same as plain follow. Until the filter has 5 detections and is sure of the walker's position to 0.8 m, it sends the plain follow command. This mode has only flown in the simulation (`python scripts/follow_eval.py` prints reactive vs predict on the Moss walk).

Rehearse first, no drone: `python -m kite.follow --source recordings/<dir> --overlay --predict`. The overlay draws the predicted walker path in green on the image (at head height) and a top-down inset in the bottom right: the walker's trail in grey, the predicted path in green with circles for its uncertainty, the planned drone path in blue, the drone at the bottom facing up, 1 m grid lines. If the take has `gps/frames_gps.csv` (from `python -m kite.gps ... --recording <dir>`), the route in `sim/public/osm/route.json` caps how far a predicted turn goes at the corner.

Testing it for real, once the battery is charged:

- Fly in an open area: a park lawn or an empty lot, at least 20 m clear in front, with no cars, no road and no bystanders within 10 m. Never at the venue (spec section 3), and never on Moss St.
- Same setup as plain follow: `python -m kite.record --source tello --out recordings --fly --follow --follow-predict`, take off, climb to the walker's height plus 0.3 m, get behind them by hand, then `g`.
- The walker does three things, about 20 s each: walk straight at 0.5 m/s, stop dead and stand for 5 s, then turn left (a slow 90 degrees, like the Moss/Howard corner) and walk on.
- The pilot keeps a thumb over space (hover) or `l` (land) the whole time, walks 3 m behind and to one side of the drone, and has a spotter at the walker's elbow.
- What to watch: at the stop, the drone should brake and settle at about 1.2 m without drifting closer than about 1 m; if it closes in, press space. In the turn, the drone should yaw a little early, keeping the walker's head near the middle of the frame. The status line should not flick between `FOLLOWING` and `LOST`. Back on normal Wi-Fi, `events.jsonl` has every decision, with `planner` saying whether predict or the plain controller steered.
- If it drifts sideways, swings in yaw or brakes late, drop `--follow-predict`: plain follow is the fallback and needs no other change.
