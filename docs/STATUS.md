# Kite status

Last updated Sat Oct 24, 4:15 AM PT. Submission Sun 9:00 AM; freeze Sat 8:00 PM.

## Try it now (on Eddy's laptop)

```bash
scripts/demo.sh live iPhone     # the table demo: earpiece, dashboard, detector on the phone camera with the reveal window
scripts/demo.sh sim             # the backup: earpiece, relay, sim, dashboard, detector on the sim's cameras
scripts/demo.sh status          # what is up, recent log lines, netcheck
scripts/demo.sh stop
```

- Dashboard: http://localhost:5174 (live map, walk timeline, "Message the walker" box; `P` for presentation mode)
- Sim: http://localhost:5173 (arrow keys walk, B blacks out the screen, V reveals)
- Reveal window (live mode): `r` replays the last 20 s, `l` goes back to live, `q` quits
- Logs: `logs/<piece>.log`

## Done and on main

| Area | What works | PR |
|---|---|---|
| Earpiece | Spatial sonar, beacon, Cartesia voice in and out, Claude assistant, offline fallbacks | #2, #3 |
| Earpiece | Stop and wait, then "Path is clear. You can continue." after 1.5 s of clear way; Rust binaural renderer; Google Routes walking directions with rerouting | #38 |
| Earpiece | Phrase warm-up one Cartesia request at a time, so live speech keeps a slot | #35 |
| Claude | Judgment layer: short "wait" calls on hazards; code drops any "safe to cross", reroute or cleared stop | #18 |
| Claude | Scene descriptions from the chest, drone and sim cameras; "what's in front of me?"; reading text aloud | #17, #38 |
| Live walk | Phone chest camera to "Stop. Person ahead." at about 1.5 m, depth fallback, reveal window with heads pixelated | #14 |
| Spacetime | Event and latest tables; every earpiece event mirrored, throttled per type; caregiver messages back to the walker | #1, #8, #30 |
| Dashboard | Live map, walk timeline with Kite, Claude and Caregiver labels, shared-state panel, presentation mode | #29 |
| Sim | Moss St from OSM, glTF traffic on real lanes, stand-in detector, metrics mode (blind 3.05 collisions per walk on Moss) | #32 |
| Sim | Google 3D Tiles on screen, geofenced to Moss St; OSM world as the detector fallback (`?detector=osm`) | #40, #41 |
| Drone | Drone layer, recorder, calibration, recording replay, flight checklist | #4, #9 |
| Drone | Flight safety fixes, preflight GO / NO-GO, `open_source()` for any camera, walker GPS alignment | #21, #22, #23, #25 |
| Drone | Follow mode behind `--follow`, tested in simulation only, never flown | #27 |
| Drone | Street View frame source, limited to Moss St | #34, #40 |
| Video | Real earpiece audio from an event log; captions, hazard banners and sonar over footage | #19 |
| Tooling | `scripts/demo.sh`, replay, netcheck, offline fallback, disqualifier audit | #5, #11, #26, #31 |
| Tooling | `scripts/check.sh` replaces GitHub Actions; `scripts/deploy-site.sh` for the landing site | #42 |
| Docs | Spec, sourced facts and prior art, interview guide (a future plan; no interviews this weekend), Google permission, runbooks, Devpost draft | #6, #10, #24, #26, #40 |

Karthik built the Claude judgment layer integration, the camera descriptions ("what's in front of me?" and reading text aloud), the cached safety phrases, stop and wait, walking routes, SpacetimeDB mirroring, and support for any stereo Bluetooth headphones.

## Not done, said plainly

- The Tello is untested. It is still in the box.
- The live walk has never run on a live camera, only on recorded clips (macOS has not yet given the terminal camera access).
- Follow mode has never flown.
- There is no real footage yet.
- The sim is not yet close to the real street.
- The demo video edit, the rehearsal and the final Devpost have not started.
- Rex's `streetseg` is still in his personal repo. Stand-ins (`sim/standin_detect.py`, `sim/live_walk.py`, `sim/seg_standin.py`) cover the demo.

## In progress

| What | Who | State |
|---|---|---|
| Kite-mode collision number | agent | running |
| 40-walk reliability bench | agent | running |
| Blender street with real facades (local only) | agent | running |
| Splat pipeline from the 7 AM phone walkthrough | agent | planned, needs the walkthrough |
| Tello camera model for sim frames | agent | running |
| Follow-mode path prediction | agent | running |
| Judgment layer tuning (#56) | agent | needs Karthik's review |

## Needs a person

| Who | What | When |
|---|---|---|
| Eddy | Ground check and calibration on Arthur's laptop; live walk on the iPhone | Now |
| Eddy, Arthur, Rex | The 7:30 AM shoot (`docs/shoot-day.md`, `docs/shoot-flight.md`), starting with `python -m kite.preflight`. Eddy films the side angle, splat walkthrough, facade photos and the Moss crossing width. | Sat 7:30 AM |
| Arthur | Charge, then a follow-mode test flight in an open area away from cars and the venue | After the shoot |
| Arthur, Eddy | Actions off in Settings; decide repo visibility | Sat |
| Rex | Bring `streetseg` in; sign off on `docs/research/facts.md`; shoot logistics | Sat |
| Karthik | Review #8 and #56; decide the Moss track row in the README with Eddy | Sat |
| Eddy | Footage into `recordings/`, cable, demo video edit | After the shoot |
| Eddy | Google representative's name and role in `docs/google-permission.md` | Sat |
| Eddy or Rex | Micro-USB cable, phone chest clip, sleep masks, wipes, over-ear headphones (Target, if needed) | About 8 AM |
| Everyone | 3 PM cut (live walk reliable, or the sim walk becomes the demo); 6 PM rehearsal, 40 walks; 8 PM freeze | Sat |
| Eddy | Devpost draft | Sat night |
| Eddy | Audit 8 AM, submit by 9 AM | Sun |

There is no CI. Run `scripts/check.sh quick` before every merge.
