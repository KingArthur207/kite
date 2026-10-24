# Shoot day: Moss St at Howard, Sat Oct 24, 7:00 AM

Print section 1 for everyone. The appendix has the positions, the scripted beats, the backups and every command. Spec section 8 says why the shoot looks like this; Arthur's flight steps are in `docs/shoot-flight.md`.

## 1. One page for the crew

Crew: Arthur flies from his laptop. Rex runs the shoot: he calls start, stop and cut, keeps time, and walks at Arthur's elbow watching the ground. Eddy films the side angle, then takes the facade photos. The walker is blindfolded and logs GPS on their phone. The spotter stays at arm's length from the walker the whole time. If we can get a sixth and seventh person, one plays the pedestrian and one drives the car (appendix C).

Light: sunrise is about 7:27, so at 7:00 it is still twilight, and Moss is a narrow alley between three storey buildings. The hover test decides whether we fly at 7:00 or wait for more light.

| Time | What | Who |
|---|---|---|
| 6:15 | Leave for the corner with the gear (appendix A). | all |
| 6:30 | Gear check, Tello still off. Measure the walker's height in shoes, to 2 cm. Check the wind on a weather app. | Rex |
| 6:35 | Walk the route on foot with the drone off: chalk the start 70 m before Howard, set the props, rehearse every beat, including the wait at the corner and the zebra crossing. | Rex, walker, spotter |
| 6:50 | Everyone to their start position (appendix A). Walker starts the GPS log. Driver waits on Russ St. | all |
| 6:55 | Tello app closed, Tello on, laptop on its Wi-Fi, `python -m kite.preflight`. Fly only on GO. | Arthur |
| 6:57 | `python -m kite.record --source tello --out recordings --fly --speed 50`. Walker shows time.is on the phone to the Tello camera for 5 s, then to Eddy's camera. Eddy rolls. | Arthur, walker, Eddy |
| 6:58 | `t` to take off, hover test at 1 m, then `r` to walker height plus 0.3 m. | Arthur |
| 7:00 | The take, about 5 minutes with the beats in appendix C. Land at 20 to 30%, or as soon as Rex calls cut. | all |
| 7:06 | Check the take and copy it to a second disk before anything else. Walker stops the GPS log and exports the GPX. | Arthur, walker |
| 7:30 | Facade photos for the sim, about 5 minutes, after sunrise (appendix B). | Eddy |
| 7:40 | Buy a USB-C to Micro-USB cable if nobody brought one. | Rex |
| 7:50 | Charge the battery in the drone (about 90 minutes). Run the after steps (appendix E). | Arthur, Eddy |
| 8:30 | Hand-off done. A second take after charging only if the first one failed. | Rex |

GO only if all of these hold:

- `kite.preflight` says GO: drone battery 50% or more, the Tello ports free, video arriving, the laptop on the Tello Wi-Fi.
- Hover test: at 1 m with no keys pressed, the drone stays within about half a metre for 5 s. If it drifts (wind, or too dark for its downward camera), land, wait 10 minutes for light and try once more. Then go to appendix D.
- No car moving on Moss at the start, and nobody near the takeoff spot.
- The clock shown to both cameras, then the walker blindfolded with the spotter in place, GPS logging, Eddy rolling.

Rules for everyone:

- The spotter is at arm's length from the walker at every moment, on the road side: the walker's left on Moss, their right on Howard.
- Nobody crosses Howard. The walk ends on Howard's south sidewalk.
- Anyone can say "stop". The walker stops and Arthur taps space. When Rex calls cut, Arthur lands with `l`.
- At the corner the walker waits on the curb until the spotter says the zebra is clear. Nobody steps in front of a turning car.
- `!!` cuts the motors and drops the drone. Use it only if the drone is about to hit someone.

<div style="page-break-after: always"></div>

## Appendix A: Route and positions

Gear: the Tello with its one battery, Arthur's laptop charged, Eddy's laptop charged as the backup recorder, every phone charged, the blindfold, chalk, a tape measure, the props from appendix C, a USB stick for the second copy, and a USB-C to Micro-USB cable if anyone owns one (old Android phones, Kindles and game controllers used them).

The points come from `sim/public/osm/route.json` (OpenStreetMap, our sim's route). House numbers are from OSM building tags.

| Point | lat, lng | Where |
|---|---|---|
| Start | 37.7779299, -122.4081767 | NE sidewalk of Moss, 70 m SE of the zebra, between the buildings numbered 37-39 and 47-49 Moss St. In the sim it is under a street tree next to a building. |
| Corner | 37.7783781, -122.4087325 | NE side of the zebra. The corner building is 1073 Howard St. |
| Zebra (sim origin) | 37.778352, -122.408768 | Marked crossing across Moss, no signal, tactile paving, 6.1 m curb to curb. |
| Across the zebra | 37.7783268, -122.4087976 | SW corner of Moss and Howard, at 1077 Howard St. |
| Stop | 37.7781999, -122.4089586 | Howard's south sidewalk, 20 m SW of the zebra, between 1077 and 1097 Howard, under a large tree. |

To find the start, stand at the zebra on the NE corner and walk 70 m SE along the NE sidewalk (about 90 steps), then check it against the coordinates on a phone map and chalk it.

The street: Moss is one way from Howard toward Folsom, so every car on Moss has just turned in from Howard and comes toward the walker. The roadway is 6.1 m, the sidewalks about 2.5 m, and the NE sidewalk narrows to 1.5 m in one spot. Building to building is 11 m. Howard is three lanes one way (SW) at 25 mph. Its bike track is on the north side, so on the south sidewalk traffic runs right beside the curb.

Positions in each phase:

| Phase | Walker and spotter | Drone | Arthur and Rex | Eddy |
|---|---|---|---|---|
| Start | At the chalk mark facing NW. Spotter on the walker's left. | On flat paving 1 m behind the walker, clear of the tree. | 4 to 6 m behind the walker on the same sidewalk. Rex at Arthur's elbow. | Across Moss on the SW sidewalk, back to the buildings, 5 to 10 m ahead. |
| Up Moss | Walk at 0.5 to 0.7 m/s. | 1 m behind, at walker height plus 0.3 m. | 3 to 5 m behind the drone. | Walks NW on the SW sidewalk, a little ahead of the walker, filming across. Stays off the roadway. |
| Corner wait | Stop 2 to 4 m short of the corner, still facing NW. | Hover. | Stop. | At the SW corner by 1077 Howard, 3 m along Howard from the zebra, back to the wall. Frames the walker, the drone and the turning car. |
| Crossing | Turn left (SW) and cross on the zebra. Spotter on the walker's right, the side cars turn in from. | Follows. | Cross the zebra behind the drone. | Holds. |
| Howard | Walker on the building side, spotter on the curb side. | Follows. | Behind the drone. | Moves SW ahead along the same sidewalk to about 30 m from the zebra and films them coming. |
| Land | Stop at 20 m along Howard. | Lands on the sidewalk behind the walker, away from the curb. | Stop. | Keeps rolling until the drone is down. |

The walker waits facing NW because that is where the drone's camera can see a car turn in. The sim showed that once the walker faces the zebra, a turning car comes from behind and to the right and is only in the drone's frame when it is about 3 m away (sim/README.md).

## Appendix B: Who records what

Tello video and telemetry (Arthur's laptop). Steps 2 to 7 of "At the corner" in `docs/shoot-flight.md`:

```bash
source .venv/bin/activate
python -m kite.preflight
python -m kite.record --source tello --out recordings --fly --speed 50
```

The take lands in `recordings/<YYYYMMDD_HHMMSS>_tello/`: `video.mkv`, `frames.csv`, `telemetry.jsonl`, `events.jsonl` and `meta.json`.

Walker phone GPS. Install tonight: Open GPX Tracker on an iPhone, or GPSLogger (by mendhak) on Android. Both are free and export GPX. Set the highest accuracy, a fix every second and no minimum distance. Start logging at 6:50 standing at the start mark, so the fix has a few minutes to settle. Carry the phone in a hand or an outer pocket, not a bag. Stop logging after the landing, export the GPX and AirDrop it to Arthur and Eddy as `walk.gpx`. GPX times are UTC, so `kite.gps` needs no `--tz`. A CSV export in local time needs `--tz America/Los_Angeles`.

The clock offset. On the Tello Wi-Fi the laptop has no internet, so its clock can drift by seconds against the phone's GPS times. Right after the recorder starts and before takeoff, the walker opens time.is on the phone (over cellular) and holds it about 30 cm in front of the Tello camera for 5 s, then in front of Eddy's camera. Later, step through the take to the first frame where the seconds digit changes:

```bash
python -m kite.sources.recording $TAKE --preview    # space pauses, , and . step one frame; the HUD shows #<frame>
```

Then put that frame number and the time on the phone into this, which prints the flag for `kite.gps` (it assumes the laptop is set to Pacific time, as time.is is in SF):

```bash
python -c "import csv,sys,datetime as d; n,clk,f=int(sys.argv[1]),sys.argv[2],sys.argv[3]; tw=next(float(r['t_wall']) for r in csv.DictReader(open(f)) if int(r['index'])==n); c=d.datetime.combine(d.datetime.fromtimestamp(tw).date(),d.time.fromisoformat(clk)).timestamp(); print(f'--offset-s {c-tw:+.2f}')" 412 06:57:41 $TAKE/frames.csv
```

Eddy's side angle. Phone in landscape, 4K at 30 fps, horizon level, no zoom past 2x. One continuous clip from the clock shot to the landing. Positions are in appendix A. This is the opening shot of the video (0:00 to 0:08 in `docs/video-runbook.md`).

Facade photos for the sim (new, about 5 minutes, after sunrise). Eddy, or Eddy and the spotter in parallel. Two passes on Moss and one on Howard:

- The NE side of Moss from the start to Howard, shot from the SW sidewalk. Then the SW side from the NE sidewalk.
- Howard: from the south sidewalk, the buildings across Howard that the walker faces at the end of Moss (1078 to 1088 Howard in OSM), then 1073 and 1077 Howard at an angle from the curb. Nobody crosses Howard for this either.
- Stand back against the buildings on the opposite sidewalk. Hold the phone level and use 0.5x if the roofline is cut off, so the verticals stay straight. Portrait. Take one photo every 5 steps (about 4 m), which overlaps each shot with the last by more than a third. One pass per side, walking toward Howard.
- Camera location tagging on, so each photo keeps its position and heading.
- Wait for people to pass. Blur or delete any photo with a face before it goes into the repo (spec 5.3).

These photos are ours, so they can be committed and shown with no license terms attached. They replace any Mapillary or Google Street View imagery in the sim. Google's permission (`docs/google-permission.md`) doesn't let us store their imagery anyway.

## Appendix C: Props and scripted beats

The walker hears the spotter during the take, not Kite. The earpiece audio is rendered afterwards from the detector running on the footage (appendix E), and the pitch says so.

| Walk time | Where | Beat |
|---|---|---|
| 0:00 | Start | Rex says "walking". The walker sets off at about 0.6 m/s. |
| about 0:25 | 55 m before Howard | A bin or a bag on the walker's line, at a stretch where the sidewalk is full width (not the 1.5 m pinch). The spotter says stop, the walker stops, Arthur taps space, and the spotter guides the walker round it. Arthur moves the drone round with `a` or `d`. |
| about 0:50 | 35 m before Howard | The pedestrian steps out from a doorway, crosses the walker's path 2 to 3 m ahead and walks on. The walker stops until the spotter says go. |
| about 1:15 | 20 m before Howard | Something at head height that a cane would miss: a flattened cardboard box or a jacket at 1.6 to 1.8 m, hanging from an awning, sign or doorway on the building side so it sticks out about half a metre over the walker's line. The walker stops, then steps out toward the curb to pass it, and Arthur moves the drone out with them before going forward. Use only soft, light props, and run no string across the sidewalk above 2 m, where the drone flies. |
| about 2:00 | 2 to 4 m short of the corner | The key moment. The walker stops facing NW. A car turns into Moss from Howard across the zebra and drives on toward Folsom. The walker waits until it has passed and the spotter says clear. |
| about 2:45 | The zebra | The walker turns left and crosses. |
| about 3:00 to 3:35 | Howard south sidewalk | 20 m SW, then stop. Arthur lands. |

The car. If someone can bring a car, the driver waits on Russ St just short of Howard (Russ is two way, 60 m NE of Moss), on a call with Rex on speaker. When the walker stops at the corner, Rex says "go". The driver turns left onto Howard and left again into Moss, slowly, stopping for the zebra like any driver, and drives on down Moss. For a retake, the loop is four left turns: Moss, Folsom, Russ, Howard. Without a car we wait for natural traffic. The walker holds at the corner for up to 60 s, and if nothing turns in, Rex calls the crossing and the take goes on.

What the stand-in detector will pick up from the drone footage (`sim/standin_detect.py`): the car, as a vehicle hazard once it comes down Moss toward the walker (in the sim it said "Car slightly left, 7 meters"), and people as sonar dots. A crowd hazard needs 3 or more people 1 to 15 m ahead, so a single pedestrian doesn't make one. If we have three spare people, they can stand together at the SW corner during the wait. YOLO has no class for bins or cardboard, so those two props show up in Eddy's side angle and for streetseg, but get no cue from the stand-in.

People: with five of us, Rex can't play the pedestrian because he is at Arthur's elbow, so that beat needs a sixth person or gets dropped. The driver is a seventh, or natural traffic.

## Appendix D: Backups

Eddy's laptop as the recorder. The main checkout on Eddy's laptop has a `.venv` at the repo root. Tonight, on normal internet:

```bash
cd ~/"Hackathon Destruction"/kite
git pull && source .venv/bin/activate && pip install -e ".[dev]"
python -m kite.preflight --quick
```

Give his terminal app Local Network access and let Python accept incoming connections (System Settings > Privacy & Security), and turn key repeat on, as in `docs/shoot-flight.md`. At the corner it is the same as on Arthur's laptop:

```bash
source .venv/bin/activate
python -m kite.preflight
python -m kite.record --source tello --out recordings --fly --speed 50
```

If the ports are busy (the Tello app or another script holds 8889, 8890 or 11111): `kite.preflight` names the program and its pid, and `kill <pid>` frees it on the laptop. If the Tello app is open on a phone, force quit it, turn that phone's Wi-Fi off, power cycle the Tello, rejoin from the laptop and run `python -m kite.preflight` again. The fallback check is `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111`, which must print nothing.

If the battery is low: `kite.preflight` is NO-GO under 50% and prints the estimated flight minutes before the 20% auto-land (about 8% a minute). From 50 to 60%, start 50 m before Howard and skip the bin, so the corner beat still happens. Under 50%, don't fly. Either charge and fly at about 9:30, or record handheld now: `python -m kite.record --source tello --out recordings` without `--fly` never sends motion, so Arthur can hold the Tello above and behind the walker's head at about 2 m and walk the route. The pitch then says it was handheld.

If it's windy or too dark: the hover test decides. As a rule of thumb (ours, not DJI's), gusts over about 10 mph on the weather app mean the hover test will probably fail. Wait for the light or the wind to change if the morning allows, otherwise do the handheld recording above.

If the laptop dies or the Wi-Fi drops mid-flight: the drone hovers and lands itself after 15 s without hearing from the laptop. Stop walking, keep everyone 3 to 5 m away, and let it come down.

## Appendix E: After the take

From the repo root, with `.venv` active, on normal internet. The folder name is the one `kite.record` printed when it stopped:

```bash
TAKE=recordings/20261024_065800_tello

# Check the take, then make the second copy before anything else.
python -m kite.sources.recording $TAKE
cp -R $TAKE /Volumes/<usb stick>/

# MJPG to mp4 for the edit.
ffmpeg -i $TAKE/video.mkv -pix_fmt yuv420p $TAKE/video.mp4

# Walker GPS onto the video frames, with the offset from appendix B. Writes $TAKE/gps/.
python -m kite.gps walk.gpx --recording $TAKE --offset-s <offset>
```

`kite.gps` prints the overlap between the GPS and the video. A whole hour off means a time zone mistake, and "NO OVERLAP" means the offset is wrong.

Then the video runbook's pipeline (`docs/video-runbook.md`). The stand-in detector runs in a throwaway uv environment, as in `sim/README.md`. Set `--cam-height` to the height the status line showed in flight:

```bash
DETECT="uv run --no-project --with ultralytics --with opencv-python-headless --with lap --with websockets --with python-dotenv --with numpy --with scipy"
$DETECT python sim/standin_detect.py --download        # once, with internet: weights/yolo11n.pt
mkdir -p out

# 1. Detector on the footage, events with t in seconds of video.
$DETECT python sim/standin_detect.py --source $TAKE/video.mkv --camera both --cam-height 2.0 \
    --jsonl out/events.jsonl --no-send --fast

# 2. What the walker would have heard (real time: a 5 minute take takes 5 minutes).
cd earpiece && .venv/bin/python -m earpiece render --jsonl ../out/events.jsonl ../out/heard.wav && cd ..

# 3. Overlay plus audio, faces blurred.
$DETECT python video/overlay.py $TAKE --events out/events.jsonl --wav out/heard.wav \
    --blur-faces --badge "Drone view" -o out/drone_cues.mp4
ffprobe -v error -show_entries format=duration:stream=codec_type -of compact out/drone_cues.mp4
```

The last line should list one video and one audio stream, as long as the take. To show the walk on the dashboard map, replay the GPS events: `earpiece/.venv/bin/python scripts/replay.py $TAKE/gps/positions.jsonl --spacetime`.

Hand-off. Everything also goes into one `shoot/` folder on Eddy's laptop: the take folder, `eddy_side/`, `walk.gpx` and `facades/` (with `moss_ne/`, `moss_sw/` and `howard/` inside).

| Who | Gets | Does |
|---|---|---|
| Arthur | The master take folder on his laptop. | Copies it to the USB stick and to Eddy, runs the summary, the mp4 and `kite.gps`. Writes down for the pitch: flown by hand, the battery at landing, anything that went wrong. |
| Rex | A copy of the take folder. | Runs streetseg on `video.mkv` if it's ready. Its JSONL can replace `out/events.jsonl`. |
| Eddy | The take folder with `gps/`, his side angle, `walk.gpx`, the facade photos. | Runs the detector, the render and the overlay, cuts the video to the runbook's shot list, and puts the facade photos into the sim. |
| Walker | Nothing to keep. | Sends `walk.gpx` to Arthur and Eddy, then deletes the log if they want. |
