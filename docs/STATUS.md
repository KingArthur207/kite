# Kite status

Last updated Sat Oct 24, 1:45 AM PT by Eddy's Claude session. Submission Sun 9:00 AM; freeze Sat 8:00 PM.

## Try it now (on Eddy's laptop)

- Dashboard: http://localhost:5174 (live map, hazards, devices, "Message the walker" box)
- Sim: http://localhost:5173 (arrow keys walk, B blacks out the screen, V reveals, H sends a test hazard you hear on the laptop speakers)
- Earpiece logs: `tail -f /tmp/kite-live/earpiece.log`

`scripts/demo.sh` (coming) will start and stop all of this with one command.

## Done and on main

| Area | What works | PR |
|---|---|---|
| Earpiece | Spatial sonar, beacon, Cartesia voice, Claude assistant, offline fallbacks | #2, #3 |
| Claude | Judgment layer: short "wait" calls on hazards, code blocks any "safe to cross" or cleared stop | #18 |
| Claude | Scene descriptions from the chest and drone cameras | #17 |
| Spacetime | Event and latest tables; every earpiece event mirrored; caregiver messages back to the walker | #1, #8 |
| Dashboard | Live map with walker, trail, hazards, next waypoint, device freshness | merged on main |
| Drone | Drone layer, recorder, calibration, recording replay, flight checklist, drone CI | #4, #9, #16 |
| Sim | three.js corner, Tello-matched drone and chest cameras, metrics mode (blind 4.6 collisions per walk) | merged on main |
| Video | Real earpiece audio from an event log; captions, hazard banners and sonar over footage | #19 |
| Tooling | replay, netcheck, offline fallback, disqualifier audit, CI and security scans | #5, #11, #26 |
| Docs | Spec after the meeting, sourced facts, interview guide, Devpost draft, runbook, meeting notes | #6, #10, #24, #26 |

## In progress

| What | Who | State |
|---|---|---|
| Sim on Moss St with real-lane traffic, glTF cars and people, stand-in YOLO detector | agent | running |
| Live walk on a real camera, stop at about 1.5 m, reveal screen | agent | PR #14, merges after the sim PR |
| Dashboard walk timeline and presentation mode | agent | running |
| One-command launcher `scripts/demo.sh` | agent | running |
| GPS alignment, flight safety fixes, preflight GO/NO-GO, open any camera | Arthur's agents | PRs #21, #22, #23, #25, waiting on CI |

## Needs a person

- Eddy: copy the team `.env` to the repo root so Claude and Cartesia can be tried live.
- Eddy or Rex: Micro-USB cable (Rex said he'd bring one; else Target around 8 AM), phone chest clip, sleep masks, wipes, over-ear headphones.
- Eddy: ask #0-ask-directors whether a powered-on Tello with motors off may stream at our table (optional).
- Arthur: ground check and calibration, then the 7 AM shoot (`docs/shoot-flight.md`). This laptop can run it as a backup.
- Rex: `streetseg` into the repo, chest camera obstacles by 3 PM; the interview (`docs/research/interview-guide.md`).
- Karthik: review the judgment layer (#18) and caregiver messages (#8), which were built while he slept.

## Next gates

Sat 7:00 AM shoot. Sat 3:00 PM cut point (live walk reliable, or the sim walk becomes the live demo). Sat 6:00 PM rehearsal, 40 walks. Sat 8:00 PM freeze.
