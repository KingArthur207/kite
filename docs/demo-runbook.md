# Demo runbook: table judging

Sun Oct 25, 10:30 AM to 12:30 PM PT, Palace of Fine Arts. The whole team stays at the table for the full window.

## Table kit

- One laptop running everything below, charged, charger plugged in
- AirPods, charged, paired to that laptop
- The chest webcam rig, plugged into the laptop
- The Tello on the table, not flying (drones are banned at the venue)
- The demo video, downloaded and open in a player, not streamed
- [TODO: a printed card with the repo URL and the Devpost link]

## Before judging starts (by 10:00 AM)

Do all of this on venue Wi-Fi ("Cal Hacks 13.0"), so the phrase cache and installs are warm in case the internet drops later.

1. `git pull` on `main`. Installs already done: `earpiece/.venv`, `npm install` in `dashboard/` and `sim/`.
2. `.env` at the repo root has `CARTESIA_API_KEY` and `ANTHROPIC_API_KEY`, and `KITE_EARPIECE_HOST=127.0.0.1`.
3. `cd earpiece && source .venv/bin/activate && python -m earpiece selftest`. Every line should pass, or you should know why it doesn't.
4. Once the earpiece is running (start order step 1), `python3 scripts/netcheck.py` should end with no failures.
5. Set Do Not Disturb on the laptop so notification sounds don't play in the judge's ears.

## Start and stop

One command from the repo root brings everything up in the background:

```bash
scripts/demo.sh sim                    # earpiece, relay, sim, dashboard, stand-in detector; opens the sim and dashboard
scripts/demo.sh live iPhone            # earpiece, dashboard, stand-in detector on that camera (index or name) with the reveal window
scripts/demo.sh replay data/sample_walk.jsonl   # earpiece, dashboard, and the replay
scripts/demo.sh status                 # what is up, the last 3 log lines of each piece, then netcheck
scripts/demo.sh stop                   # stops what the script started and nothing else
```

Add `--output "<device>"` (or set `KITE_OUTPUT`) to pick the headphones; by default the earpiece looks for AirPods. `KITE_SPACETIME=0` turns off mirroring to SpacetimeDB. `KITE_OFFLINE=1` runs `scripts/offline.sh` first and points the earpiece at the local server, so you don't need to edit `.env`. `KITE_DETECT_ARGS="--depth"` passes extra flags to the detector. For `live`, give a camera index or part of its name; `--list-cameras` lists them (sim/README.md).

Before it starts anything, the script checks for `earpiece/.venv`, `node_modules` in `sim/` and `dashboard/`, and the detector weights, and prints the command that fixes each missing one. If a port it needs (5010, 5011, 5020, 5173, 5174) is already taken, it names the process holding it and refuses to start. Logs go to `logs/<piece>.log`. If the stand-in detector isn't on `main` yet, the script says so and starts the rest.

The earpiece runs in the background, so you can't type `t <words>` into it. If you need typed questions, run the earpiece by hand from the table below instead.

## Start order by hand

Use this if the script fails. Each in its own terminal, from the repo root unless noted.

| # | What | Command | Check |
|---|---|---|---|
| 1 | Earpiece | `cd earpiece && .venv/bin/python -m earpiece run --spacetime` | Says "Kite is ready." in the AirPods. The log shows the phrase cache count |
| 2 | Frame relay | `earpiece/.venv/bin/python sim/relay.py` | Prints `[relay] ws://127.0.0.1:5020/pub -> /sub` |
| 3 | Perception | [TODO: Rex's `python -m streetseg` command for sim frames and the webcam] | Hazards show on the dashboard when a car or crowd is in view |
| 4 | Sim | `cd sim && npm run dev`, open the URL it prints | HUD shows relay and earpiece connected |
| 5 | Dashboard | `cd dashboard && npm run dev`, open the URL it prints (5174 if the sim took 5173) | Connected, and the sim walker's position appears when you walk |

Put the sim in full screen on the laptop. Keep the dashboard in a second window for the reveal.

## Running one judge

1. Hand over the AirPods. Press `B` in the sim to black out the screen.
2. The judge walks with the arrow keys (up and down walk, left and right turn), guided only by sound.
3. When Kite says stop, they wait until it says to continue.
4. Press `V` to reveal the view, then show the dashboard.
5. Play the Tello footage clip.
6. Hold an obstacle in front of the chest webcam so they hear the beeps speed up.

## Reset between judges

1. Dashboard: press "Clear run" and confirm. This wipes the event log and latest state.
2. Sim: reload the browser tab. The walker goes back to the start and the collision counter resets to 0. The page reconnects to the relay and the earpiece within 2 s.
3. Earpiece: leave it running. If it is still talking, send `stop_speaking`:
   `earpiece/.venv/bin/python -m earpiece send '{"type":"control","action":"stop_speaking"}'`
4. Wipe the AirPods.

## When something fails

Cartesia is down or slow. Nothing to do. Cached safety phrases still play in the Cartesia voice, and new sentences fall back to macOS `say` by themselves (the earpiece log shows `[speech] neural voice offline`). Voice questions stop working, because speech input needs Cartesia; type them instead with `t <words>` in the earpiece terminal.

Claude is down. Nothing to do. Sonar, stops and `dont_walk` don't depend on Claude, and local commands (stop, repeat, pause, louder, quieter, beeps on or off) still work. Open questions get a short offline answer.

Maincloud is down, or the internet is gone. With the script, run `scripts/demo.sh stop` and then `KITE_OFFLINE=1 scripts/demo.sh sim` (or `live`). By hand, switch SpacetimeDB to a local server on the same laptop:

```bash
scripts/offline.sh           # starts a local server, publishes the module, points the dashboard at it
cd dashboard && npm run dev  # restart the dashboard
```

Then set `SPACETIME_HOST=http://127.0.0.1:3000` in the repo-root `.env` (the script prints the line) and restart the earpiece (step 1 of the start order). Undo afterwards with `scripts/offline.sh off` and reset `SPACETIME_HOST` to `https://maincloud.spacetimedb.com`.

Wi-Fi is gone entirely. Phone hotspots are banned at the venue, so run fully offline: the local SpacetimeDB server above, cached phrases plus `say`, and local voice commands. The sim, relay, perception and earpiece all run on localhost and need no network.

AirPods drop. The earpiece switches to the laptop speaker and says "AirPods disconnected. Using the laptop speaker." Re-pair, and it says "AirPods connected."

The sim shows relay or earpiece disconnected. Check that terminal is still running and restart it. The page reconnects every 2 s.

Something else breaks. Play the backup video of the real run (recorded Sun 7:00 AM) and talk over it.

## The 2-minute pitch

Demo first. The presenter is whoever won Saturday night's timed tryout: [TODO: name].

| Time | What happens |
|---|---|
| 0:00 to 0:10 | "Kite is a guide dog for blind pedestrians that flies." Hand the judge the AirPods. |
| 0:10 to 0:50 | The judge walks the corner with the screen black, guided by beeps and voice. A car or crowd appears; Kite stops them, waits until the way is clear, then says to continue. |
| 0:50 to 1:05 | Reveal the view: the drone camera with detections drawn on it. |
| 1:05 to 1:25 | The real Tello footage from the 7 AM shoot, same corner, same cues. "Same code." |
| 1:25 to 1:45 | Hold an obstacle in front of the webcam; the judge hears the beeps speed up. Then the interview quote [TODO] and one measured number, collisions with and without Kite in the sim [TODO]. |
| 1:45 to 2:00 | What's real and what isn't, then questions. |

What to say at 1:45 when asked what's real: the earpiece, voice, Claude assistant, SpacetimeDB and dashboard run live on this laptop. The corner is a simulation of Moss St at Howard, built from OpenStreetMap, not a 3D scan. The drone footage is real and was recorded before the event [TODO: piloted or follow]. The drone never flew here.
