# Kite: team spec

Cal Hacks 13.0, written Fri Oct 23 2026, 11:40 PM PT. Team: Arthur, Eddy, Karthik, Rex.
Submission deadline Sun Oct 25 9:00 AM PT. Live table judging 10:30 AM to 12:30 PM at the Palace of Fine Arts.

This is the shared contract for all four of us and our Claude sessions. If your work disagrees with it, change the spec in a commit first, then the code.

## 1. What Kite is

Kite is a guide dog you don't wait two years for. A DJI Tello flies about 30 cm above and behind a blind walker's head. From there it sees what the walker can't: the route ahead, crowds, crossings and obstacles farther out. A webcam worn by the walker covers the ground near their feet, which the drone's camera can't see, and catches anything sudden. Everything reaches the walker through AirPods: directional beeps whose pitch rises as something gets closer, plus Cartesia voice for directions.

The guide dog comparison shapes the features. A guide dog leads around obstacles, stops at curbs and crossings, watches overhead clearance, and refuses "forward" when a car is coming (intelligent disobedience). Each Kite cue should map to one of those jobs.

Why it matters: commonly cited figures put a trained guide dog at around $50,000 with waitlists of years, and only about 2% of blind people work with one. Rex verifies these before they go on a slide.

## 2. Goals, non-goals, anti-goals

Goals
- A judge can walk a real street corner (a Gaussian splat of the corner outside Aneesh's apartment) with a black screen and AirPods, guided only by Kite, then see the view.
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
2. 0:10 to 0:50. The judge walks the splat corner with the screen black, guided by beeps and voice. A car or a crowd appears; Kite stops them, waits until the way is clear, then says to continue. Kite does not detour around obstacles (team decision); "stop and wait" is the guide dog behavior we show.
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

We use the event types already defined on `feat/airpods-audio` in `earpiece/` (summarised in `kite-notes-02.md` section 7). Nobody invents a parallel format.

- Frame: bearings in degrees in the walker's frame, 0 straight ahead, +90 right, −90 left, ±180 behind. Distances in meters.
- Types: `obstacles`, `hazard`, `guidance`, `beacon`, `position`, `status`, `scene`, `say`, `heartbeat`, `control`.
- Priorities, most important first: `critical`, `hazard`, `nav`, `answer`, `info`, `system`.
- Who emits what:

  | Producer | Owner | Emits |
  |---|---|---|
  | Webcam detector | Arthur | `obstacles` (full frame of current objects, about 10 Hz), `heartbeat` |
  | Drone or sim detector | Arthur | `hazard` (crowd, crosswalk, vehicle ahead, overhead obstacle), `scene` |
  | Sim | sim owner | `position` (walker pose), `status` (source `sim`) |
  | Route | Karthik | `guidance`, `beacon` |
  | Judgment layer | Karthik | `say`, and may upgrade a `hazard` priority; never downgrades one |

- Converting a drone or sim bearing to the walker's frame needs the walker's heading. The sim knows it exactly. On real footage, use the walking direction over the last second or so.

### 5.2 SpacetimeDB is the transport between machines

The earpiece binds to 127.0.0.1, so events from another process or machine arrive through SpacetimeDB. That also makes Spacetime the real backbone, which is what its track rewards.

Default design, Karthik's to change: one `event` table (`id`, `type`, `source`, `priority`, `payload` as the event JSON, `t`) that every producer writes through a reducer, plus a `latest` table keyed by `(type, source)` for state the dashboard shows (walker position, drone or sim status). The earpiece subscribes and passes each payload through its existing validator. Python producers call reducers over HTTP (`POST /v1/database/kite-sync-pl6rp/call/<reducer>` with a bearer token; see `dashboard/README.md`). Old events get pruned so the table stays small.

Exception for the safety path: the webcam detector and the earpiece run on the same laptop, so webcam `obstacles` and `heartbeat` go straight to the earpiece over localhost UDP 5010. They are mirrored to Spacetime at about 1 Hz for the dashboard only. The sonar never waits on the internet. (Spec author's call.)

Networks: we build at Aneesh's apartment on its Wi-Fi. Maincloud, Claude and Cartesia need internet, not just a shared network; with Maincloud, two machines don't even need to be on the same network, only online. At judging the whole demo (sim, detector, earpiece, dashboard) runs on one laptop, so the only network question there is internet for those three cloud services, from the venue Wi-Fi ("Cal Hacks 13.0", per the hacker guide). Fallbacks, tested by Saturday freeze: cached Cartesia phrases plus macOS `say` for speech, a local `spacetime start --listen-addr 127.0.0.1:3000` instance with the same module, and the deterministic cues (sonar, `critical`, `dont_walk`) working with Claude unreachable. (Spec author's call.)

The dashboard shows the walker on a map, the current route, active hazards and drone or sim status. It shows no camera video (privacy).

### 5.3 Detector

One function, three frame sources:

`detect(frame, camera, pose) -> list of hazards/obstacles` where `frame` is BGR 960×720, `camera` holds fx, fy and mount pose, and `pose` is the camera's height and pitch.

- Frame sources: the recorded Tello video file, sim frames, the webcam. Arthur's `kite/sources/` readers already cover file and webcam.
- Sim frames reach the detector over a local WebSocket as JPEG at about 5 to 10 fps. The detector process then writes events to Spacetime.
- Street classes: person, bicycle, car, bus, truck, traffic light, stop sign, bench, plus anything at head height. Weights are downloaded ahead of time and never at runtime.
- Faces in any frame we store get blurred. We don't store frames during the live demo.

### 5.4 Sim

- The sim owner's call on the stack (Rex, if he takes it). Default is three.js with a Gaussian splat renderer in the browser, because judges can open it anywhere and SpacetimeDB has a TypeScript client.
- The walker is driven with the arrow keys. A black-screen mode hides the view; a key reveals it.
- The virtual drone camera matches the Tello: 960×720, fx from Arthur's calibration (start at 1000), mounted 0.3 m above the walker's head and about 1 m behind, pointing forward, no tilt.
- Sim hazards come from the real detector, not from the sim's own object list (team decision). So moving cars and pedestrians are realistic glTF models (find free ones early and check YOLO detects them in a test render), composited into the splat, with collision boxes.
- A second virtual camera at chest height stands in for the body webcam. Its frames go through the same detector and produce `obstacles`, so close-range cues in the sim come from the same code as the real webcam. The splat has no collision, so walls and curbs come from a hand-placed or OSM-derived layer lined up with the splat using two or three landmarks.
- Metrics mode runs scripted walks with and without Kite's cues and counts collisions. That number goes in the pitch (Microsoft's "measurable usefulness" and our own credibility).
- Until the splat exists, the world is a placeholder (boxes from OSM building footprints is fine).

### 5.5 Route

- Real world: Google Routes API v2 and Places API (New); both are enabled on our key, the legacy APIs are not.
- Karthik turns the route into `guidance` events (spoken turn instructions with a `ttl`) and `beacon` events (bearing and distance to the next waypoint).
- In the sim, the route runs over the same corner. Karthik and the sim owner agree on one mapping between latitude/longitude and sim meters, anchored on the same landmarks used to line up the splat.

### 5.6 Judgment layer (Claude)

- Input: current hazards, the route, the latest `scene` text, walker pose. Output: one short spoken decision ("Wait, the crosswalk is crowded.") and an optional priority upgrade.
- Claude can tell the walker to wait or warn them. It never reroutes (no detours, team decision), and it cannot clear a stop. Sonar, `critical` obstacles and `dont_walk` hazards are handled in deterministic code before Claude sees anything.
- Use Claude for the scene description too (it reads images), instead of adding Gemini, so Claude is load-bearing for the Anthropic track.
- Fixed phrases ("stop", "curb ahead", "bear left") are generated once with Cartesia and played from a cache. Only changing sentences, like street names and Claude's decisions, are streamed. Navigation speech older than its `ttl` is dropped.

## 6. Owners

Each person owns a deliverable, not a layer. One owner per job.

| Owner | Owns | By Sat 3:45 AM (skeleton) | Saturday |
|---|---|---|---|
| Arthur | The eyes: drone, splat, detection | Push the drone layer (`kite/`, `pyproject.toml`, tests) to `main`. Run `tests/drone_test.py --ground`, then `--mag 60` and `--mag 100` for real top speed. A recorder that saves Tello video and telemetry to disk. `detect()` running on any video file. | Flies the 7am shoot. Trains the splat (his call on the tool without an NVIDIA GPU). Detector on sim frames, Tello footage and webcam. Crowds and overhead obstacles. Signal reading only if time allows. |
| Eddy | Infrastructure, then the video and submission | Thunderbolt bridge between the drone laptop and the wearable laptop (Internet Sharing; Thunderbolt Bridge above Wi-Fi in the drone laptop's service order). Replace the starter `person` table with the event tables. Route events from any producer to the earpiece. A hardcoded `hazard` written from the drone laptop plays a beep in the AirPods. | Demo laptop setup and the offline fallbacks (local Spacetime, cached speech). Then the demo video and the submission audit. |
| Karthik | The voice: AirPods, Google Maps directions, Cartesia, Claude | Earpiece and destinations are merged (PRs #2 and #3). Earpiece consumes events from Spacetime. Routes API to `guidance` and `beacon`. | Judgment layer with the safety rule. Cached phrases and streaming. One Devpost paragraph per sponsor rubric (Cartesia, Spacetime, Anthropic). |
| Rex | Not settled yet (he is building something now). Recommended: the sim, plus running the 7am shoot and the interview | If he takes the sim: sim page with an arrow-key walker, the virtual drone camera, a placeholder world, frames to the detector, `position` to Spacetime; glTF car and person models confirmed visible to the detector. | Splat into the sim, moving cars and people, chest-height webcam camera, black-screen mode, collision metrics. The interview and the guide dog figures. Marble fallback if splat training stalls. |

Everyone works in this repo, on short branches merged to `main` often. Python for perception and drone; TypeScript for the sim, the Spacetime module and the dashboard.

## 7. The 7am shoot (Sat Oct 24, corner outside Aneesh's apartment)

About 90 minutes, all four of us. Nothing runs live; everything is recorded and processed afterward.

- Rex runs it: timing, props, the walker's safety, and the shot list. The blindfolded walker always has a spotter at arm's length and never walks near moving cars.
- Arthur flies, from the laptop on the Tello's Wi-Fi, recording video and telemetry to disk. Manual or simple follow; we say which in the pitch.
- Eddy films the splat capture: a slow loop of the corner with the phone held overhead at about 2 m, since splats look worst from angles nobody filmed. Then the side-angle video of each walk.
- The walker logs GPS on their phone and has their height measured with shoes, to within 2 cm (distance-from-head estimates depend on it).
- The walker moves slowly, about 0.5 to 0.7 m/s, so the drone can keep up.
- Every battery is charged tonight. Each pack gets one scripted run of about 5 minutes, landing at 20 to 30%.
- Props for the shoot: something at head height the cane would miss, a bag or bin on the sidewalk, and a teammate as a "crowd" near the crossing.
- Before flying: phone Tello app closed, `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` empty, `drone_test.py --ground --preview` passes. Install everything while you still have internet.

## 8. Gates (from `hackclock.py`)

| When (PT) | Phase | Rule |
|---|---|---|
| Now to Sat 3:45 AM | Skeleton | A hardcoded path end to end: sim walker moves, a fake hazard goes into Spacetime, a beep plays in the AirPods. Nothing else starts first. |
| Sat 7:00 to 8:30 AM | Shoot | Section 7. |
| Sat 3:45 AM to 8:00 PM | Build | Every feature must make the demo better; timebox 2 hours. Ask "can we demo right now?" every 2 hours. |
| Sat 3:00 PM | Cut point | If the full path (real detector on sim frames to beep in AirPods) isn't working, cut "walk me to my friend" and signal reading. |
| Sat 8:00 PM | Freeze | No new features. Repo public. README with setup, Cartesia features used, known limitations. Devpost draft in. |
| Sun 7:00 AM | Record | Backup video of the real run. Rehearse the pitch. |
| Sun 8:00 AM | Submit | Disqualifier audit (repo public, teammates listed, tracks ticked, no keys in the repo), then submit and confirm. |
| Sun 10:30 AM | Judging | Whole team at the table for the full window. |

## 9. Tracks

Enter: Best Hardware Hack, Best People-First Hack, Cartesia, Spacetime, Anthropic. Overall, Most Technically Complex and Hacker's Choice come automatically.

Add if cheap: Microsoft "Built for the Real World" (check the live site for an Azure requirement first; the interview and the collision numbers cover most of its criteria), Meta "Bringing People Closer Together" (needs "walk me to my friend").

Skip: everything else, including Moss and Bitdeer. At Cal Hacks 12, 90 of 98 sponsor winners took a single sponsor prize, so extra opt-ins don't stack.

What each entered track needs from us:

| Track | Needs | Owner |
|---|---|---|
| Cartesia | Voice core to the flow; responsive (cached phrases, streaming); README lists Cartesia features and limitations; 2-minute demo-first pitch | Karthik |
| Spacetime | Spacetime is the real transport between machines and the dashboard's source | Karthik |
| Anthropic | Named community (blind pedestrians in SF), the interview, Claude making the judgment calls, and a clear answer on where it could go wrong | Rex, Karthik |
| Hardware, People-First | The table kit works; the drone footage is real | Arthur, Eddy |

## 10. Calls already made

Made by the team: the drone flies behind and above, not ahead (fix the README, which says "ahead"). The drone is for the demo only. The 7am shoot plus a splat sim of the same corner. Beeps with direction and pitch plus verbal directions. Cartesia and Spacetime are in.

Also made by the team (Oct 23, 11:50 PM): sim hazards come from the real detector, with realistic models and a chest-height virtual camera; Kite stops and waits instead of detouring.

Made in this spec, change them by editing it: webcam `obstacles` go to the earpiece over localhost, not through the cloud. Venue Wi-Fi is the primary link, with offline fallbacks. Karthik's earpiece events are the contract. Spacetime as the cross-machine transport with an `event` table. Claude instead of Gemini for scene description. Signal reading demoted. The owner split in section 6. Recording at the shoot instead of running live.

## 11. Left to each owner

- Arthur: detector model and thresholds, splat tool, whether the shoot uses manual or follow flight, how to estimate walker heading on real footage.
- Eddy: network layout and bridge setup, the video edit.
- Sim owner: sim stack and renderer, how collision geometry is built.
- Eddy and Karthik: exact Spacetime schema and pruning (Eddy builds it, Karthik consumes it). Karthik: Claude prompt and call rate.
- Karthik: route-to-cue wording.
- Rex: interview format, Devpost structure.

## 12. Open items

- Real Tello top speed at stick 60 and 100 (Arthur, tonight). If it can't reach 0.6 m/s, the walker goes slower still, or the drone flies a pre-planned path along the route rather than chasing.
- fx for our Tello: an object of known height at taped distances (Arthur). Until then the sim uses 1000.
- Who walks at the shoot.
- Whether Microsoft's track requires Azure (Rex checks the live site).
- How many Tello batteries we have.
