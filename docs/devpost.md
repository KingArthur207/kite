# Kite: Devpost draft

Tagline: A guide dog for blind pedestrians that flies.

Paste the sections below into Devpost. Everything in [TODO: ...] has to be filled or cut before submitting. Don't replace a TODO with a number or quote we don't have.

## Inspiration

Guiding Eyes for the Blind puts the cost of raising and training one guide dog at $50,000, paid by donors, and at Guide Dogs for the Blind the wait after applying was 7 to 10 months in 2022. About 1 million Americans are blind (JAMA Ophthalmology, 2021), and fewer than 10,000 guide dogs work in the US (IGDF, 2018). [TODO: Rex to sign off; sources and caveats in `docs/research/facts.md`. The "2% of blind people" figure dates from around 1990 and was dropped.] A cane finds what is a step or so ahead, and it can't find a sign or branch above waist height. It also can't tell you the crossing ahead is crowded or that a car is turning in across it.

We wanted to give a blind pedestrian some of what a guide dog gives them: something that looks farther ahead than they can, stops them at curbs and crossings, watches for overhead obstacles, and refuses to go forward when going forward isn't safe.

We didn't get to talk to a blind traveler this weekend. We built from published research: guide dog schools' own figures and the KIT Flying Guide Dog user study of blindfolded walkers. Testing with orientation and mobility instructors is our next step.

## Prior art

In 2021, researchers at KIT built Flying Guide Dog: a DJI Tello flies in front of the walker, segments the street to find the sidewalk, and the walker holds a string tied to it and follows the pull, with voice prompts in bone conduction headphones. Their user study had six sighted participants under blindfolds, who rated it on a questionnaire; it kept them centered on the sidewalk and was easy to learn, and traffic lights still needed work. Kite drops the leash. Our drone flies behind and above the walker, a chest camera covers the ground at their feet, and everything reaches them as sound. Our judges at the table are sighted and blindfolded too.

## What it does

The judge does the demo. They put on a sleep mask and headphones, and we clip a phone to their chest as the camera. They walk a few meters along the aisle with a spotter beside them. When a teammate steps into their path, the beeps come from the teammate's side and speed up, and at about 1.5 m Kite says "Stop. Person ahead." Kite then waits. Once the way has stayed clear for 1.5 seconds with the camera still sending frames, it says "Path is clear. You can continue." If the camera goes quiet, Kite never says the path is clear. The judge can also ask "Kite, what's in front of me?" and Claude answers from the newest camera frame.

Then the mask comes off. A reveal window replays the last 20 seconds of what the chest camera saw, with its boxes, distances and the STOP banner, faces pixelated and nothing written to disk. Next to it, the caregiver dashboard shows the walk as a timeline: each stop, hazard, Claude judgment and caregiver message, labeled by who acted (Kite's code, Claude or the caregiver).

Outdoors, a small drone flies about 30 cm above and 1 m behind the walker's head, where it can see the route ahead, crowds and crossings. Kite turns what the cameras see into sound:

Directional beeps mark nearby obstacles. They come from the direction of the obstacle and get faster and higher as it gets closer, and people, vehicles, poles and curbs each have their own sound. A ticking beacon points toward the next waypoint, and spoken warnings and walking directions come in a Cartesia voice ("Stop. Person ahead.", "Don't walk.", "In 20 meters, turn left").

The walker presses an AirPods stem to talk. Common commands (stop, repeat, pause, louder, "take me to ...") are handled on the laptop with no network. Anything else goes to Claude, which answers from what Kite currently knows. "Read this" reads a menu, sign or screen aloud, sentence by sentence, so a warning can cut in.

Kite never plans detours, and nothing in Kite ever says a street is safe to cross. A caregiver can follow the walk on a live map with hazards and what Kite last said, but never camera video, and can type a message that the walker hears in their headphones.

If the camera path fails at the table, the backup is a browser sim of Moss Street at Howard Street in San Francisco, which the judge walks with the screen black, guided by the same earpiece. We also show Tello footage of a walker on the same corner with Kite's cues on it. [TODO: confirm the shoot footage exists and say whether the drone was piloted or in follow mode.]

## How we built it

Earpiece (Python, with a Rust audio engine). Binaural rendering with Brown and Duda's spherical-head model, in Python and ported to Rust, with matching output. The Rust engine runs the mixer on CoreAudio's thread through `cpal`, with lock-free rings between it and Python, so a slow network call or Claude can't make the audio stutter. Obstacle sonar warns by time to contact. Speech has six priorities; a hazard cuts into a navigation sentence, and navigation speech older than its time to live is dropped. Producers send JSON events over UDP or WebSocket on localhost, so the safety path never touches the internet.

Stop and wait. "Stop." comes from deterministic code. Only that code can end the wait, after 1.5 s of clear way with perception still alive. A "walk" signal has to hold for 0.6 s before Kite trusts it; "Don't walk." is instant.

Voice (Cartesia). Sonic TTS streamed over one persistent WebSocket, one context per utterance, with cancellation when a warning cuts in. Every fixed safety phrase is pre-rendered and cached on disk so it plays instantly and offline. Speech input is Cartesia `ink-whisper` streaming STT, with audio sent while the wearer is still talking. Phrase warm-up runs one request at a time so it never takes the slot live speech needs.

Claude, in three jobs. A judgment layer sends `claude-haiku-5-5` the current hazards, obstacles, route, scene text and position whenever they change. Claude answers through one strict `decide` tool: at most one short sentence ("Wait, the crosswalk is crowded.") and an optional priority upgrade for a hazard already on the list. Code checks every answer. It drops any sentence that sounds like permission, a reroute or clearing a stop ("safe to cross", "go ahead", "keep walking", "detour" and more), anything over 120 characters, any upgrade to `critical`, any downgrade, and any call slower than 2.5 s. Vision uses Claude to describe drone, chest and sim frames as the shared scene text, to answer "what's in front of me?" and to read text aloud. Sentences in a description that sound like an instruction to move are stripped before they are stored. The voice assistant answers open questions with tools to start navigation, change settings, remember a place and alert the caregiver, and those tools are checked against an allow-list.

Shared state (SpacetimeDB). An `event` table and a `latest` table on Maincloud. The earpiece mirrors every event it receives and speaks, throttled per type (a position once a second per source) so walker updates don't crowd everything else out of the 500-row table. It also polls SpacetimeDB for caregiver messages, the one path from the shared state back into the earpiece. Those are clamped to `info` priority and can't clear a stop. The React dashboard subscribes and draws a Leaflet map of the walker and hazards, the walk timeline, and a presentation mode (press P) for the reveal.

Perception. Until Rex's `streetseg` lands, a stand-in detector runs Ultralytics YOLO11n on chest, drone and sim frames, with distance from the ground plane at the camera's height and pitch. On the chest camera, a 3-frame median and a closing-speed fit decide the stop zone (inside 1.6 m and within 0.7 m of the path, or under 1 s from contact). Depth Anything V2 Small catches walls, doors and boxes that YOLO has no class for. The drone view raises `crowd` and `vehicle` hazards that must hold for 3 frames. The same code reads the phone camera, recorded clips and the sim's frames. [TODO (Rex): replace with `streetseg` if it lands, and what it detects.]

Sim (three.js). Moss Street from Folsom to Howard, built from OpenStreetMap: buildings, sidewalks, curbs and the zebra crossing where cars turn in from Howard. Cars and people are glTF models driving and walking on the street's real lanes and sidewalks. With permission from Google's representatives at Cal Hacks, the screen shows Google Photorealistic 3D Tiles of the same street, geofenced in code to that one block: tile requests outside it are masked and counted, and the walker is kept inside. The virtual drone camera matches the Tello (960×720, 0.3 m above the head, 1 m behind), and a second camera sits at chest height. Both stream frames through a local relay to the detector, so sim hazards come from the detector, never from the sim's own object list. A metrics mode runs scripted walks through the same obstacles and counts collisions.

Drone. A layer over djitellopy that avoids its traps (late replies, moves sent twice, growing video lag), plus a safety governor with stick limits, an altitude band, landing on low battery or flight time, and a hover when video or telemetry goes stale. Around it: a recorder that is also the flight controller, so nothing else fights for the Tello's ports, with Wi-Fi drops and stray key sequences handled; chessboard calibration; a one-command GO / NO-GO preflight; GPS alignment that lines up the walker's phone track with the footage and replays it on the dashboard map; and a follow mode, behind a `--follow` flag, that keeps the walker centered from behind with YOLO11n and hovers the moment it loses them. Follow mode is tested in simulation and has not flown.

Routing. Google Routes API v2 walking directions become spoken turns and the beacon's bearing, with rerouting when the walker strays. Places API (New) turns "the nearest pharmacy" or "the end of the street" into a destination.

Demo video. `python -m earpiece render --jsonl` renders exactly what the wearer heard for a recorded timeline, and `video/overlay.py` draws captions, hazard banners and sonar on the footage, with heads blurred.

Checks. `scripts/check.sh` runs the earpiece tests on both audio engines, Rust fmt, clippy and tests, the drone and sim tests, the dashboard build, gitleaks over the full git history, and pip-audit, bandit, npm audit and cargo audit. `scripts/audit.py` checks the submission for secrets, `.env`, large files and model weights.

## Challenges we ran into

- The Tello tops out around 0.96 m/s at full stick, extrapolated from a stick-30 test, which is slower than people walk (1.2 to 1.3 m/s). We had one battery for the shoot and roughly 6 to 8 minutes of flight with video on.
- From above and behind, the drone's fixed camera can't see the ground for the first few meters ahead. That's why the chest camera exists.
- A pedestrian signal 20 m away is about 14 px tall in the Tello's video. We couldn't read signals reliably from the drone, so crossing locations come from maps data.
- Drones are banned at the venue, so the drone footage is recorded and the live walk uses a chest camera.
- The AirPods mic forces the AirPods into mono call audio, which destroys the spatial cues. We record from the laptop mic instead.
- Facing the zebra in the sim, a car turning in from Howard comes from behind and to the right, and it only enters the cameras' view about 3 m away, cut off by the frame edge. The stand-in detector gave no hazard there.
- Google's Map Tiles policy lists object detection as a prohibited use. We asked Google's representatives at Cal Hacks, who allowed it for this one street only, so we built a geofence into the sim and kept the OpenStreetMap world as the detector's fallback.
- Our Cartesia plan allows two requests in flight for the whole team, and one earpiece warming up its phrase cache could use both and break live speech.
- GitHub Actions stopped starting jobs on our repo partway through Saturday, so we moved every check into a local script.
- [TODO: anything else that actually happened during the build.]

## Accomplishments that we're proud of

- The safety cues are deterministic and work with no internet: cached Cartesia phrases, local voice commands, sonar and stops that don't wait on any model.
- Claude can only make Kite more careful. Every judgment is checked in code before it is spoken, and only deterministic code can end a stop.
- One detector reads the phone camera, recorded clips, Street View stills of Moss St and the sim's frames through the same code. It has run on clips, stills and sim frames. [TODO: confirm on the phone camera at the rehearsal.]
- [TODO: collision numbers from the sim with Kite, from metrics mode. On Moss St the blind baseline is 3.05 collisions per walk (seed 1, 20 walks); the oracle's 0 is an upper bound that sees ground truth, not Kite, and must not be quoted as Kite. Leave this out if we don't measure Kite.]

## What we learned

[TODO: write this as a team. Candidates: what the published research taught us, guide dog behavior (stop and wait, intelligent disobedience), Tello limits, latency budgets for speech.]

## What's next for Kite

- Flying follow mode outdoors, behind and above the walker, under the safety governor already in the drone layer.
- Reading pedestrian signals from closer range or from the chest camera.
- Testing with orientation and mobility instructors and blind walkers, outdoors, with a spotter. We did not interview anyone this weekend, so this is the first thing we would do.

## Built with

Python, Rust (cpal, rtrb, PyO3), TypeScript, React, Vite, three.js, 3d-tiles-renderer, Leaflet, OpenStreetMap, Google Photorealistic 3D Tiles, SpacetimeDB, Cartesia (Sonic TTS, ink-whisper STT), Claude (Anthropic API), Google Routes API, Google Places API, Google Street View Static API, DJI Tello, djitellopy, Ultralytics YOLO11n, Depth Anything V2, OpenCV, PyAV, NumPy, SciPy, sounddevice, websockets, httpx, Moss, Playwright, FFmpeg, Cloudflare Pages. [TODO: perception models and libraries from Rex, if `streetseg` lands.]

## Tracks and evidence

| Track | What we show | Where in the repo |
|---|---|---|
| Best Hardware Hack | A DJI Tello and a phone chest camera feeding the same detector. Drone layer with its own command handling, a safety governor, a recorder, calibration, preflight, GPS alignment and follow mode. AirPods with stem-press controls. A real-time Rust audio engine. Table kit: laptop, headphones, phone chest clip, Tello. | `kite/`, `tests/`, `sim/standin_detect.py`, `sim/live_walk.py`, `earpiece/native/`, `earpiece/earpiece/controls.py`, `earpiece/earpiece/devices.py`. Footage: [TODO: shoot video link] |
| Best People-First Hack | Built around how guide dogs work: stop and wait, then continue; never "safe to cross"; warnings that work offline. Caregiver dashboard without camera video. Faces pixelated in the reveal, and no frames stored during the live walk. | `earpiece/earpiece/app.py`, `sonar.py`, `speech.py` (priorities, preemption, TTL), `judgment.py`, `dashboard/`. Research: guide dog schools' figures and the KIT study (`docs/research/facts.md`) |
| Cartesia | Voice is the whole interface. Streaming Sonic TTS on one persistent WebSocket with per-utterance contexts and cancellation, on-disk phrase cache for safety phrases, long-form reading, `ink-whisper` streaming STT, offline fallback. The README lists features and limitations. | `earpiece/earpiece/speech.py`, `listen.py`, `config.py`; README "Cartesia: the voice, both ways" |
| Spacetime | Every event the earpiece receives goes into one live world state that the dashboard reads. Caregiver messages come back through it to the walker's headphones. Walk timeline and presentation mode built only from the shared tables. Local server fallback with the same module. | `dashboard/spacetimedb/src/index.ts`, `earpiece/earpiece/spacetime.py`, `dashboard/src/App.tsx`, `scripts/replay.py --spacetime`, `scripts/offline.sh` |
| Microsoft | Customer understanding from published research (no interviews this weekend), a measured number from the sim, cost against a $50,000 guide dog, deterministic safety and no stored video, and the live walk end to end on one laptop. | README "Microsoft" section, `scripts/sim_metrics.py`, `earpiece/tests/test_app_stress.py`. Research: `docs/research/facts.md` |
| Anthropic | Community: blind pedestrians in San Francisco, known through published research, not interviews, and we say so plainly. Claude makes the judgment calls at hazards, describes what the cameras see, reads text aloud and answers questions with tools. Where it could go wrong: Claude could invent an obstacle or say it's safe to cross, so it only describes what is in its context, code filters every judgment, and stops are decided in deterministic code before Claude sees anything. | `earpiece/earpiece/judgment.py`, `vision.py`, `brain.py`, `earpiece/tests/test_judgment.py`, `sim/scene_describe.py`. Research: `docs/research/facts.md` |

## Before submitting

- [ ] Every [TODO] above filled or cut
- [ ] Guide dog figures verified by Rex, with sources
- [ ] The pitch and this page say plainly that we did not interview anyone
- [ ] Demo video link: [TODO]
- [ ] All four teammates added on Devpost: Arthur, Eddy, Karthik, Rex
- [ ] Tracks ticked: Best Hardware Hack, Best People-First Hack, Cartesia, Spacetime, Microsoft, Anthropic (spec section 10)
- [ ] `python3 scripts/audit.py` passes
