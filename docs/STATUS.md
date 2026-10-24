# Kite status

Last updated Sat Oct 24, 3:25 AM PT by Eddy's Claude session. Submission Sun 9:00 AM; freeze Sat 8:00 PM.

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
| Drone | Follow mode behind `--follow`, tested in simulation, not flown | #27 |
| Drone | Street View frame source, limited to Moss St | #34, #40 |
| Video | Real earpiece audio from an event log; captions, hazard banners and sonar over footage | #19 |
| Tooling | `scripts/demo.sh`, replay, netcheck, offline fallback, disqualifier audit | #5, #11, #26, #31 |
| Tooling | `scripts/check.sh` replaces GitHub Actions; `scripts/deploy-site.sh` for the landing site | #42 |
| Docs | Spec, sourced facts and prior art, interview guide, Google permission, runbooks, Devpost draft | #6, #10, #24, #26, #40 |

## In progress

| What | Who | State |
|---|---|---|
| OSM building facades in the sim | agent | running |
| Judgment layer tuning | agent | running |

## Needs a person

- Eddy: the Google representative's name and role, and any written confirmation, in `docs/google-permission.md`.
- Rex or Eddy: send the interview emails (drafts in `docs/research/interview-guide.md`), then run the interview.
- Arthur: the 7 AM shoot (`docs/shoot-flight.md`), starting with `python -m kite.preflight`.
- Optional: a Mapillary token.
- Arthur (repo owner): GitHub Actions billing. The private repo ran out of Actions minutes, so hosted CI can't start; until then, run `scripts/check.sh quick` before every merge.
- Eddy or Rex: Micro-USB cable, phone chest clip, sleep masks, wipes, over-ear headphones (Target around 8 AM).
- Rex: `streetseg` into the repo, or confirm the stand-in detector is what we demo.

## Next gates

Sat 7:00 AM shoot. Sat 3:00 PM cut point (live walk reliable, or the sim walk becomes the live demo). Sat 6:00 PM rehearsal, 40 walks. Sat 8:00 PM freeze.
