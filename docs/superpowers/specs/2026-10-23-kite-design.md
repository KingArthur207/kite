# Kite: team spec

Cal Hacks 13.0, written Fri Oct 23 2026, 11:40 PM PT; updated Sat Oct 24, 12:30 AM. Team: Arthur, Eddy, Karthik, Rex.
Submission deadline Sun Oct 25 9:00 AM PT. Live table judging 10:30 AM to 12:30 PM at the Palace of Fine Arts.

This is the shared contract for all four of us and our Claude sessions. If your work disagrees with it, change the spec in a commit first, then the code.

## 1. What Kite is

Kite is a guide dog you don't wait two years for. A DJI Tello flies about 30 cm above and behind a blind walker's head. From there it sees what the walker can't: the route ahead, crowds, crossings and obstacles farther out. A webcam worn by the walker covers the ground near their feet, which the drone's camera can't see, and catches anything sudden. Everything reaches the walker through AirPods: directional beeps whose pitch rises as something gets closer, plus Cartesia voice for directions.

The guide dog comparison shapes the features. A guide dog leads around obstacles, stops at curbs and crossings, watches overhead clearance, and refuses "forward" when a car is coming (intelligent disobedience). Each Kite cue should map to one of those jobs.

Why it matters: commonly cited figures put a trained guide dog at around $50,000 with waitlists of years, and only about 2% of blind people work with one. Rex verifies these before they go on a slide.

## 2. Goals, non-goals, anti-goals

Goals
- A judge can walk a model of the real corner outside Aneesh's apartment (Moss St at Howard, built from OpenStreetMap) with a black screen and AirPods, guided only by Kite, then see the view.
- The same detection code runs on real Tello footage, on the sim's virtual drone camera, and on the live webcam. That is our main honesty claim.
- Real Tello footage of a walker on that corner, with Kite's cues overlaid, in the demo video.
- One real conversation with a blind person or an orientation and mobility instructor, quoted in the pitch.

Non-goals
- Flying at the venue. Drones are banned there.
- Running the full system live on the drone during the 7am shoot.
- Autonomous outdoor following as a requirement. It's a stretch for Arthur, and the pitch says honestly whether the footage was piloted.
- Reading pedestrian signal lights from the drone as a headline feature (see section 4).

Anti-goals (failure even if it technically works)
- Judges remember "a webcam on someone's chest." Blind-navigation wearables are a saturated hackathon category, and Nova Vision lost Cal Hacks 12 with one. The drone's view from above has to carry the demo.
- Claude, or any model, can tell the walker to go into traffic. Safety stays in deterministic code.
- A video-only demo. The guide says the pitch "will require a live demo."
- Work judges never see: tests, CI and refactors beyond what protects the demo path.
- Four codebases merged at the end. Everything lives in this repo on `main`.

## 3. Constraints

- Cal Hacks rules: no copying pre-made team code into the repo. VIGIL (HackGT 13) ideas can be rewritten, not pasted. Open-source libraries are fine.
- Team cap is 4. Aneesh hosts and can advise but not build, and he won't be at the apartment in the morning.
- Laptops are macOS on Apple Silicon. No NVIDIA GPU, no GX10.
- The Tello is the standard model: its own Wi-Fi access point, no internet on that link, no GPS, one controller at a time.
- Measured or published Tello limits (details in `kite-notes-02.md`):
  - Forward speed is about 0.96 m/s at full stick, extrapolated from a stick-30 test. Walkers do 1.2 to 1.3 m/s. `SafetyCfg.max_rc` is 50 (about 0.5 m/s).
  - Battery: about 6 to 8 minutes of flight with video on.
  - Video stream: 960×720 at about 30 fps, cropped from the photo spec. fx is about 920 px (published) to 1077 px (VIGIL measured), which is about 48° to 55° wide.
  - From about 2 m up and 1 m behind, only the walker's head is in frame. The camera can't see the ground for the first few meters ahead.
  - The drone lands itself 15 s after the last command it hears.
- Hotspots are banned at the venue. That's fine, because nothing at the venue needs the drone.

## 4. What judges see

The table kit: one laptop running the sim and the earpiece, AirPods, the body webcam rig, the Tello on the table (not flying), and the demo video ready to play.

The two-minute pitch, demo first:
1. 0:00 to 0:10. "Kite is a guide dog for blind pedestrians that flies." Hand the judge the AirPods.
2. 0:10 to 0:50. The judge walks the sim corner with the screen black, guided by beeps and voice. A car or a crowd appears; Kite stops them, waits until the way is clear, then says to continue. Kite does not detour around obstacles (team decision); "stop and wait" is the guide dog behavior we show.
3. 0:50 to 1:05. Reveal the view: the drone's camera with detections drawn on it.
4. 1:05 to 1:25. The real Tello footage from the 7am shoot, same corner, same cues. "Same code."
5. 1:25 to 1:45. Hold an obstacle in front of the webcam; the judge hears the beeps speed up. Then the quote from the interview and one measured number (collisions with and without Kite in the sim).
6. 1:45 to 2:00. What's real and what isn't, then questions.

The presenter is whoever wins a 30-second timed tryout Saturday night.

Crossing signals: a signal head 20 m away is about 14 px tall in the Tello stream, so reading it from the drone is marginal. Take the crossing location from Maps, show signal handling in the sim, and attempt it on real footage only if time allows.

## 5. System

```
 7am shoot (offline)                          At the table / during the build
 Tello -> drone laptop: record video +        sim (browser) --frames--> detector (Python) --+
          telemetry to disk                   webcam -------frames--> detector (Python) ---+
 walker phone: GPS log                        recorded Tello video -> detector (Python) ---+
                                                                                           v
                    route (Karthik) -----------------------------------------------> SpacetimeDB
                    Claude judgment (Karthik) <------------------------------------> (Maincloud
                    sim walker pose (sim)  -----------------------------------------> kite-sync-pl6rp)
                                                                                           |
                                         earpiece (Karthik, wearable laptop) <------------+
                                         caregiver dashboard (browser) <-------------------+
```

### 5.1 The contract: Karthik's earpiece events

We use the event types already defined in `earpiece/earpiece/events.py` (validated by `events.parse()`; summary in `earpiece/README.md` and `kite-notes-02.md` section 7). Producers send them with `earpiece.client.EarpieceClient` (`.obstacles()`, `.hazard()`, `.guidance()`, `.beacon()`, ...). Nobody invents a parallel format.

- Frame: bearings in degrees in the walker's frame, 0 straight ahead, +90 right, −90 left, ±180 behind. Distances in meters.
- Types: `obstacles`, `hazard`, `guidance`, `beacon`, `position`, `status`, `scene`, `say`, `heartbeat`, `control`.
- Priorities, most important first: `critical`, `hazard`, `nav`, `answer`, `info`, `system`.
- Who emits what:

  | Producer | Owner | Emits |
  |---|---|---|
  | Chest camera perception | Rex | `obstacles` (full frame of current objects, about 10 Hz), `heartbeat` |
  | Drone or sim perception | Rex | `hazard` (crowd, crosswalk, vehicle ahead, overhead obstacle), `scene` |
  | Sim | Eddy | `position` (walker pose), `status` (source `sim`) |
  | Route | Karthik | `guidance`, `beacon` |
  | Judgment layer | Karthik | `say`, and may upgrade a `hazard` priority; never downgrades one |

- Converting a drone or sim bearing to the walker's frame needs the walker's heading. The sim knows it exactly. On real footage, use the walking direction over the last second or so.

### 5.2 Transport: UDP to the earpiece, SpacetimeDB for shared state

What exists on `main`: the earpiece takes events on UDP 5010 and WebSocket 5011, on localhost by default and on the LAN with `--bind 0.0.0.0`. It can mirror outbound events to SpacetimeDB, but it does not read events from SpacetimeDB.

So the live path is direct: every producer uses `EarpieceClient`, pointed at `127.0.0.1` on one laptop or at the wearable laptop's Thunderbolt bridge address from the drone laptop (earpiece run with `--bind 0.0.0.0`). The address comes from `.env` (`KITE_EARPIECE_HOST`). This keeps the safety path off the internet and means nobody waits on Eddy's tables to test.

SpacetimeDB holds the shared state everyone else watches: the earpiece mirrors every event it receives and speaks, and the sim, drone laptop and dashboard subscribe to it. Default design, Eddy's to change: one `event` table (`id`, `type`, `source`, `priority`, `payload` as JSON, `t`) plus a `latest` table keyed by `(type, source)`. Python writes over HTTP (`POST /v1/database/kite-sync-pl6rp/call/<reducer>`, see `dashboard/README.md`). The dashboard shows the walker on a map, the route, active hazards and status, read only from Spacetime. For the Spacetime track, the pitch line is: "every device on the walk, and the caregiver, shares one live world state."

Networks: we build at Aneesh's apartment on its Wi-Fi. Maincloud, Claude and Cartesia need internet, not just a shared network; with Maincloud, two machines don't even need to be on the same network, only online. At judging the whole demo (sim, detector, earpiece, dashboard) runs on one laptop, so the only network question there is internet for those three cloud services, from the venue Wi-Fi ("Cal Hacks 13.0", per the hacker guide). Fallbacks, tested by Saturday freeze: cached Cartesia phrases plus macOS `say` for speech, a local `spacetime start --listen-addr 127.0.0.1:3000` instance with the same module, and the deterministic cues (sonar, `critical`, `dont_walk`) working with Claude unreachable. (Spec author's call.)

The dashboard shows the walker on a map, the current route, active hazards and drone or sim status. It shows no camera video (privacy).

### 5.3 Perception (Rex's `streetseg`)

One pipeline, three frame sources: `process(frame, camera, pose) -> events`, where `frame` is BGR 960×720, `camera` holds fx and fy, and `pose` is the camera's height and pitch.

- The drone or sim view gets street segmentation, Waymo-demo style: walkable ground, people, vehicles, signals. Rex's first version (`rexheng/videotesting` PR #2) uses YOLO26n-sem (Cityscapes) for segmentation, YOLO26n for boxes, an optional crossing-sign checkpoint, and Depth Anything V2 Small for the chest camera's close obstacles. Model choice is Rex's, including Roboflow RF-DETR if it does better.
- Output is earpiece events (section 5.1): `hazard` and `scene` from the drone or sim view, `obstacles` from the chest camera. Not the VIGIL glasses' `turn` of -1, 0 or +1: Kite has no glasses. A suggested direction ("bear left") goes out as a `say` with priority `nav`, or as a hint the judgment layer can use.
- Stops are immediate (blocked ground, red light, something large and close). A new direction has to hold for a few frames. A person cut off by the bottom edge is the walker, not a crowd.
- Frame sources: a video file, sim frames, the webcam, the live Tello. Arthur's `kite/sources/` readers cover file, webcam and Tello behind one `FrameSource.read()`.
- Sim frames reach perception over a local WebSocket as JPEG at about 5 to 10 fps.
- Weights are downloaded ahead of time and never at runtime.
- Faces in any frame we store get blurred. We don't store frames during the live demo.

### 5.4 Sim

- Eddy's call on the stack: three.js in the browser, because judges can open it anywhere and SpacetimeDB has a TypeScript client.
- The walker is driven with the arrow keys. A black-screen mode hides the view; a key reveals it.
- The virtual drone camera matches the Tello: 960×720, fx from Arthur's calibration (start at 1000), mounted 0.3 m above the walker's head and about 1 m behind, pointing forward, no tilt.
- Sim hazards come from the real detector, not from the sim's own object list (team decision). So moving cars and pedestrians are realistic glTF models (find free ones early and check YOLO detects them in a test render), with collision boxes.
- A second virtual camera at chest height stands in for the body webcam. Its frames go through the same detector and produce `obstacles`, so close-range cues in the sim come from the same code as the real webcam. Walls and curbs are collision geometry from the same OpenStreetMap model.
- Metrics mode runs scripted walks with and without Kite's cues and counts collisions. That number goes in the pitch (Microsoft's "measurable usefulness" and our own credibility).
- The world is Moss St and Howard built from OpenStreetMap: extruded buildings, sidewalks, curbs, the zebra crossing. No Gaussian splat (see section 11).

### 5.5 Route

- Real world: Google Routes API v2 and Places API (New); both are enabled on our key, the legacy APIs are not.
- Karthik turns the route into `guidance` events (spoken turn instructions with a `ttl`) and `beacon` events (bearing and distance to the next waypoint).
- In the sim, the route runs over the same corner. Karthik and Eddy agree on one mapping between latitude/longitude and sim meters, anchored on the zebra crossing node (section 8).

### 5.6 Judgment layer (Claude)

- Input: current hazards, the route, the latest `scene` text, walker pose. Output: one short spoken decision ("Wait, the crosswalk is crowded.") and an optional priority upgrade.
- Claude can tell the walker to wait or warn them. It never reroutes (no detours, team decision), and it cannot clear a stop. Sonar, `critical` obstacles and `dont_walk` hazards are handled in deterministic code before Claude sees anything.
- Use Claude for the scene description too (it reads images), instead of adding Gemini, so Claude is load-bearing for the Anthropic track.
- Fixed phrases ("stop", "curb ahead", "bear left") are generated once with Cartesia and played from a cache. Only changing sentences, like street names and Claude's decisions, are streamed. Navigation speech older than its `ttl` is dropped.

## 6. Build everything without hardware first

Every task below can be done by one person driving Claude (or Cursor) agents on a laptop with no drone, no webcam rig and no AirPods. Hardware plugs in afterward by changing one setting, not code. These seams make that possible:

| Seam | Stand-in while building | Real hardware | How it switches |
|---|---|---|---|
| Frames | Video files: daytime stock street clips (below), then sim frames and the 7am footage | Tello stream, body webcam | `FrameSource`: `VideoFileSource` vs `TelloVideoSource` / `WebcamSource` (already in `kite/sources/`) |
| Drone commands | `make_tello(cfg, dry_run=True)`: connects, streams, logs motion instead of sending it | Tello | `dry_run=False` |
| Events | The earpiece's scripted scenarios (`python -m earpiece run --sim walk`) or a JSONL file replayed through `EarpieceClient` | Live producers | Same `EarpieceClient` calls; `replay` is a 20-line script |
| Network | Everything on one laptop, `127.0.0.1` and Maincloud | Two laptops over the Thunderbolt bridge | Host addresses in `.env` (`KITE_EARPIECE_HOST`, `SPACETIME_HOST`), never hardcoded |
| Audio | Laptop speakers or any headphones | AirPods | Earpiece output device setting (it already matches "AirPods" by name) |
| Position | Sim pose, or a GPS track replayed from a file | Walker's phone GPS | Same `position` event either way |

Test data, since it's night and our own footage waits for the 7am shoot:
- Daytime street video from a free stock site (Pexels or Pixabay, search "walking sidewalk POV" or "crosswalk pedestrians"). Their licenses allow free use; credit them in the README. Pick one clip filmed from above head height and one from chest height. Don't rip YouTube walking tours (against its terms).
- The sim's own frames once Eddy's placeholder world runs.
- The earpiece's scripted scenarios for anything audio-side (no video needed).
- A quick night clip of the block is still fine for plumbing checks (does the pipeline run end to end), just not for judging quality.
The real corner footage comes from the 7am shoot and replaces all of this.

Each owner's done check below runs without hardware.

## 7. Owners

Each person owns a deliverable, not a layer. One owner per job.

| Owner | Owns | Build with Claude, no hardware (done check) | Plug in hardware afterward |
|---|---|---|---|
| Arthur | The drone: flight and recording | Push the drone layer (`kite/`, `pyproject.toml`, tests) to `main`. A `record` command that saves video and telemetry to disk, tested in dry run against a video file. | The 7am shoot on the one battery (section 8). Camera fx calibration if time allows (the sim starts at 1000). |
| Rex | Perception: `streetseg` | Move `streetseg` from `rexheng/videotesting` into this repo (rewritten or re-committed by Rex, per the rules), emitting earpiece events instead of the glasses' `turn`. Done check: `python -m streetseg --source <stock clip>` writes `hazard`, `scene` and `obstacles` events to a JSONL file and an overlay video that looks like a Waymo demo. | Point the source at Tello footage, the live webcam, and sim frames. Same command, different `--source`. |
| Karthik | The voice: earpiece, Google Maps directions, Cartesia, Claude | Earpiece and destinations are merged (PRs #2 and #3). Done check: `python -m earpiece run --sim walk` and a replayed JSONL from Rex both play the right beeps and speech on laptop speakers, with the Claude judgment layer and cached phrases in, and every event mirrored to Spacetime. | AirPods as the output device. Walker's phone GPS as the `position` source. |
| Eddy | Infrastructure, then the sim, the video and the submission | Replace the starter `person` table with the event tables; dashboard reads them. The `replay` script. `KITE_EARPIECE_HOST` in `.env`. Done check: a JSONL replayed from a second process plays in the earpiece and shows on the dashboard, on one laptop. Then the sim (section 5.4) with an arrow-key walker, the virtual drone and chest cameras, a placeholder world, frames to `streetseg`. | The Thunderbolt bridge: earpiece with `--bind 0.0.0.0`, `KITE_EARPIECE_HOST` set to the bridge address on the drone laptop, rerun the same check across two laptops. |

Everyone works in this repo, on short branches merged to `main` often. Nothing lives in personal repos. Python for perception and the drone; TypeScript for the sim, the Spacetime module and the dashboard.

Also owned: Rex runs the interview and verifies the guide dog figures. Eddy owns the clock, the demo video and the submission audit. The presenter comes from a timed tryout.

### 7.1 Kickoff prompts

Each owner pulls `main` and starts their agent with one of these. They assume the agent can read the repo.

- Arthur: "Read docs/superpowers/specs/2026-10-23-kite-design.md sections 6 to 8 and kite-notes-02.md. Commit the drone layer in kite/ with pyproject.toml and its tests. Add `python -m kite.record --source <file|tello> --out <dir>` that saves BGR video and timestamped telemetry, working in dry run against a video file."
- Rex: "Read the spec sections 5.1, 5.3, 6 and 7, and earpiece/earpiece/events.py. Bring streetseg into this repo as a top-level package. Replace its glasses `turn` output with earpiece events sent through earpiece.client.EarpieceClient, plus a `--jsonl` option that writes the same events to a file. Done when `python -m streetseg --source <stock clip> --jsonl out.jsonl --overlay out.mp4` works."
- Karthik: "Read the spec sections 5.1, 5.2, 5.6 and 7. Add the Claude judgment layer to the earpiece under the safety rule in 5.6, cache the fixed Cartesia phrases, and mirror every received and spoken event to SpacetimeDB through the existing bridge hook. Done when `python -m earpiece run --sim walk` and a replayed JSONL both sound right on laptop speakers."
- Eddy: "Read the spec sections 5.2, 5.4, 6 and 7. Replace the SpacetimeDB starter module in dashboard/ with the event and latest tables, make the dashboard show them, and add a replay script that sends a JSONL through EarpieceClient to KITE_EARPIECE_HOST. Then start the sim in sim/."

## 8. The shoot (Sat Oct 24, corner outside Aneesh's apartment)

Location: Moss Street at Howard Street, SoMa (OSM way 8916291). Moss is a one-way alley, 192 m from Folsom to Howard, with sidewalks on both sides. Where Moss meets Howard there is a marked zebra crosswalk across Moss with no signal and tactile paving (OSM node at 37.778352, -122.408768); cars turn into Moss from Howard across it. Howard is three lanes one-way at 25 mph with a protected bike track.

Route: start on the NE sidewalk of Moss about 70 m before Howard, walk NW to the corner, turn left, cross Moss on the zebra, and stop about 20 m along Howard's south sidewalk. We never cross Howard. The whole of Moss would take over 5 minutes at 0.6 m/s, which is the entire battery, so we skip the first two thirds. The crossing is uncontrolled, so there is no signal to read: the drone's job there is spotting a car turning in from Howard, and Kite says to stop and wait. That fits section 4. The sim's origin is the zebra crossing node, so sim and real lat/lng line up.

We have one charged Tello battery and no charger until someone buys a USB-C to Micro-USB cable in the morning (the Tello charges its battery in the drone over Micro-USB). Aneesh won't be at the apartment in the morning, so confirm tonight that we can still reach the corner and our gear.

- Don't fly before the shoot. Ground checks only, kept short, because streaming video drains the battery too. Skip the stick 60 and 100 speed test; the walker just walks slowly.
- 7:00 AM: one scripted take on the one battery, about 5 minutes, landing at 20 to 30%. It has to count: rehearse the route on foot first, with the drone still off.
- Rex runs it: timing, props, the walker's safety and the shot list. The blindfolded walker always has a spotter at arm's length and never walks near moving cars.
- Arthur flies, recording video and telemetry to disk. Manual or simple follow; we say which in the pitch.
- Eddy films the side-angle video of the take.
- The walker logs GPS on their phone, walks at about 0.5 to 0.7 m/s, and has their height measured with shoes to within 2 cm.
- Props: something at head height a cane would miss, a bag or bin on the sidewalk, a teammate as a "crowd" near the crossing.
- After the shoot: buy the cable, charge (expect about 90 minutes per battery), and do a second take later in the morning only if the first one failed.
- Before flying: phone Tello app closed, `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` empty, a short `drone_test.py --ground --preview` passes.

## 9. Gates (from `hackclock.py`)

| When (PT) | Phase | Rule |
|---|---|---|
| Now to Sat 3:45 AM | Skeleton | A hardcoded path end to end: sim walker moves, a fake hazard goes into Spacetime, a beep plays in the AirPods. Nothing else starts first. |
| Sat 7:00 AM | Shoot | Section 8. One battery, one take. |
| Sat 3:45 AM to 8:00 PM | Build | Every feature must make the demo better; timebox 2 hours. Ask "can we demo right now?" every 2 hours. |
| Sat 3:00 PM | Cut point | If the full path (real detector on sim frames to beep in AirPods) isn't working, cut "walk me to my friend" and signal reading. |
| Sat 8:00 PM | Freeze | No new features. Repo public. README with setup, Cartesia features used, known limitations. Devpost draft in. |
| Sun 7:00 AM | Record | Backup video of the real run. Rehearse the pitch. |
| Sun 8:00 AM | Submit | Disqualifier audit (repo public, teammates listed, tracks ticked, no keys in the repo), then submit and confirm. |
| Sun 10:30 AM | Judging | Whole team at the table for the full window. |

## 10. Tracks

Enter: Best Hardware Hack, Best People-First Hack, Cartesia, Spacetime, Anthropic. Overall, Most Technically Complex and Hacker's Choice come automatically.

Add if cheap: Microsoft "Built for the Real World" (check the live site for an Azure requirement first; the interview and the collision numbers cover most of its criteria), Meta "Bringing People Closer Together" (needs "walk me to my friend").

Skip: everything else, including Moss and Bitdeer. At Cal Hacks 12, 90 of 98 sponsor winners took a single sponsor prize, so extra opt-ins don't stack.

What each entered track needs from us:

| Track | Needs | Owner |
|---|---|---|
| Cartesia | Voice core to the flow; responsive (cached phrases, streaming); README lists Cartesia features and limitations; 2-minute demo-first pitch | Karthik |
| Spacetime | Spacetime is the real transport between machines and the dashboard's source | Karthik |
| Anthropic | Named community (blind pedestrians in SF), the interview, Claude making the judgment calls, and a clear answer on where it could go wrong | Rex, Karthik |
| Hardware, People-First | The table kit works; the drone footage is real | Arthur, Eddy, Rex |

## 11. Calls already made

Made by the team: the drone flies behind and above, not ahead (fix the README, which says "ahead"). The drone is for the demo only. The 7am shoot plus a sim of the same corner. Beeps with direction and pitch plus verbal directions. Cartesia and Spacetime are in.

Also made by the team (Oct 23, 11:50 PM): sim hazards come from the real detector, with realistic models and a chest-height virtual camera; Kite stops and waits instead of detouring.

Also made by the team (Oct 24, 1:30 AM): no Gaussian splat. The judge walks with the screen black, so the world only shows at the reveal, and the real Tello footage of the corner (pitch step 4) is what makes it look real. The sim uses the OpenStreetMap model of Moss and Howard with glTF cars and people. The reveal shows the judge's own walk in the sim with detections drawn, then cuts to the real footage.

Made in this spec, change them by editing it: webcam `obstacles` go to the earpiece over localhost, not through the cloud. Venue Wi-Fi is the primary link, with offline fallbacks. Karthik's earpiece events are the contract. Spacetime as the cross-machine transport with an `event` table. Claude instead of Gemini for scene description. Signal reading demoted. The owner split in section 6. Recording at the shoot instead of running live.

## 12. Left to each owner

- Arthur: manual or follow flight at the shoot, the recorder's file format.
- Rex: segmentation and detection models (YOLO26, RF-DETR or others), thresholds, how walker heading is estimated on real footage.
- Eddy: exact Spacetime schema and pruning (Karthik consumes it), network layout, sim stack and renderer, collision geometry, the video edit.
- Karthik: route-to-cue wording, Claude prompt and call rate, how the earpiece consumes Spacetime.

## 13. Open items

- Rex moves `streetseg` into this repo and switches its output to earpiece events.
- Arthur pushes the drone layer, which is still only on his laptop.
- Who walks at the shoot, and whether we can reach the corner and gear at 7 AM without Aneesh.
- Where to buy a USB-C to Micro-USB cable early Saturday.
- fx for our Tello (Arthur, if battery allows). The sim uses 1000 until then.
- Whether Microsoft's track requires Azure (Rex checks the live site).
