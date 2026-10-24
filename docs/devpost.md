# Kite: Devpost draft

Tagline: A guide dog for blind pedestrians that flies.

Paste the sections below into Devpost. Everything in [TODO: ...] has to be filled or cut before submitting. Don't replace a TODO with a number or quote we don't have.

## Inspiration

A trained guide dog costs around $50,000, the waitlist can run for years, and only about 2% of blind people work with one. [TODO: verify (Rex), with sources, before these go on a slide or here.] A cane finds what is within a meter or so of the walker. It can't tell you the crosswalk ahead is crowded or that a car is pulling across the sidewalk.

We wanted to give a blind pedestrian some of what a guide dog gives them: something that looks farther ahead than they can, stops them at curbs and crossings, watches for overhead obstacles, and refuses to go forward when going forward isn't safe.

[TODO: one or two sentences from the interview with a blind person or an O&M instructor, quoted exactly, with their permission and how they want to be named.]

## What it does

A small drone flies about 30 cm above and 1 m behind the walker's head. From there it can see the route ahead, crowds and crossings. A webcam on the walker's chest watches the ground near their feet, which the drone can't see. Kite turns what both cameras see into sound in the walker's AirPods:

- Directional beeps for nearby obstacles. They come from the direction of the obstacle, and get faster and higher as it gets closer. People, vehicles, obstacles and drops each have their own sound.
- A ticking beacon pointing toward the next waypoint.
- Spoken warnings and directions in a Cartesia voice ("Stop. Person ahead.", "Don't walk.", "In 20 meters, turn left onto Lyon Street").
- A voice assistant. The walker presses an AirPods stem and asks "what's around me?" or "take me to the nearest pharmacy", and Claude answers from what Kite currently knows.

When something blocks the way, Kite tells the walker to stop and waits until it clears, the way a guide dog does. It doesn't plan detours, and nothing in Kite ever says a street is safe to cross.

A caregiver dashboard shows the walker's position, active hazards, what Kite last said and device status, live, without any camera video.

At the table, judges walk a simulated street corner with the screen black, guided only by Kite, and then see the view. [TODO: once the splat exists, say the corner is a Gaussian splat of a real San Francisco corner.] We also show real Tello footage of a walker on the same corner with Kite's cues on it. [TODO: confirm the shoot footage exists and say whether the drone was piloted.]

## How we built it

Earpiece (Python, with a Rust audio engine). Binaural rendering with Brown and Duda's spherical-head model, a real-time mixer with separate cue, alert and speech buses, ducking and a limiter. Obstacle sonar warns by time to contact, assuming a 1.2 m/s walk and a 0.55 m half-width corridor, and goes critical at 1 s or 0.75 m. Speech has six priorities; a hazard preempts a navigation sentence, and navigation speech older than its time-to-live is dropped. Producers send JSON events over UDP or WebSocket on localhost, so the safety path never touches the internet.

Voice (Cartesia). Sonic TTS streamed over one persistent WebSocket, with every fixed safety phrase pre-rendered and cached on disk so warnings play instantly and offline. Speech input is Cartesia `ink-whisper` streaming STT, with audio sent while the wearer is still talking.

Reasoning (Claude). Questions the earpiece can't answer locally go to Claude with a JSON snapshot of objects, hazards, route, drone scene text, position and the wearer's saved places. The answer streams back and is spoken sentence by sentence. Claude has tools to start navigation, change settings, remember a place and alert the caregiver. [TODO: describe the judgment layer (wait or continue at hazards) and Claude scene description of drone frames if they ship.]

Shared state (SpacetimeDB). An `event` table and a `latest` table on Maincloud. The earpiece mirrors every event it receives into it over HTTP, and the React dashboard subscribes and re-renders live.

Sim (three.js). A browser street corner with an arrow-key walker, a virtual drone camera matched to the Tello (960×720, 0.3 m above the head and 1 m behind) and a chest-height camera. Both stream JPEG frames to a local relay for perception. Sim hazards come from the same detector that runs on real footage, not from the sim's own object list. [TODO: splat pipeline and tool, once Arthur has it.]

Perception. [TODO (Rex): `streetseg` models and what it detects; the same code runs on Tello footage, sim frames and the webcam.]

Drone. A drone layer over djitellopy that avoids its traps (unmatched replies, retried moves flown twice, growing video lag), plus a safety governor with stick limits, an altitude band, battery and flight-time landing, and a hover when video or telemetry goes stale. [TODO: merge state of `kite/` and the `record` command.]

Routing. Google Routes API v2 and Places API (New) turn "the nearest pharmacy" or "the end of the street" into a destination.

## Challenges we ran into

- The Tello tops out around 0.96 m/s at full stick, extrapolated from a stick-30 test, which is slower than people walk (1.2 to 1.3 m/s). We had one battery for the shoot and roughly 6 to 8 minutes of flight with video on.
- From above and behind, the drone's fixed camera can't see the ground for the first few meters ahead. That's why the chest webcam exists.
- A pedestrian signal 20 m away is about 14 px tall in the Tello's video. We couldn't read signals reliably from the drone, so crossing locations come from maps data.
- Drones are banned at the venue, so the live demo is a simulation of the corner and the drone footage is recorded.
- The AirPods mic forces the AirPods into mono call audio, which destroys the spatial cues. We record from the laptop mic instead.
- [TODO: anything else that actually happened during the build.]

## Accomplishments that we're proud of

- The safety cues are deterministic and work with no internet: cached Cartesia phrases, local voice commands, sonar that doesn't wait on any model.
- [TODO: the same detector running on real Tello footage, sim frames and the live webcam, if it does by submission.]
- [TODO: collision numbers from the sim, with and without Kite, from metrics mode. Leave this out if we don't measure them.]
- [TODO: the interview, and what changed in Kite because of it.]

## What we learned

[TODO: write this as a team. Candidates: what the interview taught us, guide dog behavior (stop and wait, intelligent disobedience), Tello limits, latency budgets for speech.]

## What's next for Kite

- Autonomous following on the drone, behind and above the walker, using the safety governor already in the drone layer.
- Reading pedestrian signals from closer range or from the chest camera.
- Testing with blind walkers and O&M instructors outdoors, with a spotter.
- [TODO: anything the interview suggested.]

## Built with

Python, Rust, TypeScript, React, Vite, three.js, SpacetimeDB, Cartesia (Sonic TTS, ink-whisper STT), Claude (Anthropic API), Google Routes API, Google Places API, DJI Tello, djitellopy, NumPy, SciPy, sounddevice, websockets, httpx, Moss. [TODO: perception models and libraries from Rex; splat tool from Arthur.]

## Tracks and evidence

| Track | What we show | Where in the repo |
|---|---|---|
| Best Hardware Hack | A real DJI Tello and a chest webcam feeding the same pipeline. Drone layer with its own command handling and a safety governor. AirPods with stem-press controls. Table kit: laptop, AirPods, webcam rig, Tello. | `kite/` on `feat/drone-layer` [TODO: on `main`], `tests/drone_test.py`, `earpiece/earpiece/controls.py`, `earpiece/earpiece/devices.py`. Footage: [TODO: shoot video link] |
| Best People-First Hack | Built around how guide dogs work: stop and wait, never "safe to cross", warnings that work offline. Caregiver dashboard without camera video. | `earpiece/earpiece/sonar.py`, `speech.py` (priorities, preemption, TTL), `brain.py` (system prompt rules), `dashboard/`. Interview: [TODO: quote and who] |
| Cartesia | Voice is the whole interface. Streaming Sonic TTS on one persistent WebSocket with per-utterance contexts and cancellation, on-disk phrase cache for safety phrases, `ink-whisper` streaming STT, offline fallback. The README lists features and limitations. | `earpiece/earpiece/speech.py`, `listen.py`, `config.py`; README "Cartesia features used" |
| Spacetime | Every event the earpiece receives goes into one live world state that the dashboard and other machines subscribe to. Local server fallback with the same module. | `dashboard/spacetimedb/src/index.ts`, `earpiece/earpiece/spacetime.py`, `dashboard/src/App.tsx`, `scripts/replay.py --spacetime` |
| Anthropic | Community: blind pedestrians in San Francisco. Claude answers the walker's questions from live context, with tools for navigation, settings, memory and alerting a caregiver. Where it could go wrong: Claude could invent an obstacle or say it's safe to cross, so it only describes what is in its context, and stops are decided in deterministic code before Claude sees anything. | `earpiece/earpiece/brain.py`. Judgment layer: [TODO: path once built]. Interview: [TODO] |

## Before submitting

- [ ] Every [TODO] above filled or cut
- [ ] Guide dog figures verified by Rex, with sources
- [ ] Interview quote approved by the person quoted
- [ ] Demo video link: [TODO]
- [ ] All four teammates added on Devpost: Arthur, Eddy, Karthik, Rex
- [ ] Tracks ticked: Best Hardware Hack, Best People-First Hack, Cartesia, Spacetime, Anthropic
- [ ] `python3 scripts/audit.py` passes
