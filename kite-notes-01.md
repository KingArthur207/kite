# Kite notes 01

Everything known about the kite project as of **2026-10-23, late evening (hackathon day 1)**. Sources:
- this repository: the local checkout and `origin`;
- the team's previous project, vigil / ReachGlass, read in full at `~/vigil`;
- earlier session notes.

Labels used below:
- **(measured)**: measured on a real Tello.
- **(sim only)**: validated only in vigil's simulator.
- **(unverified)**: a design value nobody has checked.

Contents
1. [The project](#1-the-project)
2. [Repository state](#2-repository-state)
3. [The Tello: facts, measured numbers, gotchas](#3-the-tello-facts-measured-numbers-gotchas)
4. [vigil, the previous project](#4-vigil-the-previous-project)
5. [vigil perception in detail](#5-vigil-perception-in-detail)
6. [vigil autonomy in detail](#6-vigil-autonomy-in-detail)
7. [Never merged: obstacle avoidance while following](#7-never-merged-obstacle-avoidance-while-following-followavoid--looming)
8. [Fruit-fly steering and the fruitfly-training runtime](#8-fruit-fly-steering-and-the-fruitfly-training-runtime)
9. [vigil voice and glasses hardware](#9-vigil-voice-and-glasses-hardware)
10. [Lessons and pitfalls](#10-lessons-and-pitfalls)
11. [What this means for kite](#11-what-this-means-for-kite)
12. [Where things are](#12-where-things-are)

---

## 1. The project

### 1.1 What kite is
- A **DJI Tello (standard model, not EDU)** helps a **visually impaired walker** get around.
- **Directions:** the walker hears spoken directions in their **AirPods**. Goals are the shortest path, plus spotting **crowds, crossings and signs** so the route is efficient and avoids crowds.
- **Obstacles:** a **webcam worn by the walker** detects sudden obstacles close by.
- **Two framings exist; decide which one the demo is:**
  - Early notes say the drone **follows** the person.
  - The current local `README.md` (uncommitted) says the drone **"flies ahead of a visually impaired walker and talks them through the route"**.
  - The two lead to different control and safety designs (see 11.2).
- **Outdoors, in a city.** The teammate's audio code uses San Francisco places in its examples ("Lyon Street", "Palace of Fine Arts"), and the API keys include Google Maps routing (see 2.6).

### 1.2 Event and team
- **Event:** a ~**36 h hackathon** starting **2026-10-23**. The first commit message is "first commit to calchakcs", most likely Cal Hacks.
- **Repo:** `git@github.com:KingArthur207/kite.git`.
- **People seen in the history:**
  - **Arthur Armaing**: first commits, the drone layer port.
  - **Karthik Subramanian**: API key template, SpacetimeDB sync + dashboard, the earpiece audio WIP.
- **Reference project:** the team's previous project, vigil (ReachGlass, HackGT 13, 3rd place, Sept 2026), is the code base to reuse (section 4).

### 1.3 System shape (as far as the repo shows today)
- **Drone laptop:**
  - joined to the Tello's Wi-Fi access point, which has **no internet**;
  - runs the `kite/` Python package (drone control, video, later perception and autonomy).
- **Wearable side:** a webcam on the walker, plus a laptop that processes it.
  - The dashboard README talks about "the drone laptop, the wearable laptop and the caregiver dashboard".
- **Earpiece process (WIP):**
  - takes JSON events (obstacles, guidance, beacon, hazard, say, ...) and renders speech and spatial cue sounds to the AirPods;
  - voice comes from Cartesia.
- **Shared state:**
  - a **SpacetimeDB** database on Maincloud, `kite-sync-pl6rp`;
  - the README says it keeps the two laptops and the caregiver dashboard in sync, "even though the two cameras are on different WiFi networks";
  - only the template "person" table exists so far.
- **Cloud APIs** (from `.env.example`):

  | Key | Used for |
  |---|---|
  | `GOOGLE_MAPS_API_KEY` | Routes API v2 + Places API (New). The legacy Directions/Places APIs are not enabled on the key. |
  | `ANTHROPIC_API_KEY` | Claude: route + scene reasoning |
  | `GEMINI_API_KEY` | Gemini: drone video scene description |
  | `CARTESIA_API_KEY` | Cartesia: voice to the AirPods |
  | `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY` | Moss: route/landmark memory retrieval, only if the team enters the Moss track |

### 1.4 Platform
- The team's laptops are **macOS**. The default system Python is **3.14**; the kite venv uses **3.12.14** (built with `uv`).
- vigil's voice app and glasses tooling were Windows-specific, so the audio side has to be redone for macOS. Karthik's earpiece branch does that with PortAudio.

---

## 2. Repository state

### 2.1 Remote (`origin`, fetched 2026-10-23)

| Ref | Commit | What |
|---|---|---|
| `main` | `966c989` | Arthur's `kite-notes-02.md`, on top of the merge of PR #1 (`setup/sync-and-keys`) by Karthik |
| `docs/kite-notes-01` | (this file) | `kite-notes-01.md` on top of `main` |
| `setup/sync-and-keys` | `7da9520` | Already merged |
| `feat/airpods-audio` | `7672abf` | WIP: earpiece audio core (2.7) |

History of `main`:
1. `e932aea` "first commit to calchakcs"
2. `3f65a48` "added drone test" (`tests/drone_test.py`)
3. `f7267a0` "Ignore local env files and document required API keys"
   - adds `.env.example`;
   - **creates** a 3-line `.gitignore`: `.env`, `.env.*`, `!.env.example`.
4. `7da9520` "Add SpacetimeDB sync module and live dashboard" (`dashboard/`)
5. `91a7d7f` merge of PR #1
6. `966c989` "Add kite-notes-02: Tello, djitellopy, vigil lessons and kite plan" (Arthur)
   - a companion to this file;
   - goes deeper on Tello commands, the state stream, djitellopy internals and a pre-flight checklist.

### 2.2 Arthur's local checkout (uncommitted at the time of writing)
- **Branch:** local `main` was fast-forwarded to `origin/main` (`966c989`) on 2026-10-23.
- **The `.gitignore` collision is resolved.** Upstream's tracked `.gitignore` (`.env`, `.env.*`, `!.env.example`) now also carries the local lines (`.venv/ __pycache__/ *.pyc *.egg-info/ .pytest_cache/ .DS_Store`), as an uncommitted change.
- **Uncommitted work:**
  - `M .gitignore`: the merged lines above;
  - `M README.md`: rewritten with setup, drone-layer overview and pre-flight notes;
  - `M tests/drone_test.py`: now imports `kite.sources.video_stream`; run as `python tests/drone_test.py`;
  - untracked `kite/`, `pyproject.toml`, `tests/test_drone.py`, `tests/test_sources.py`.
- **Git rule worth remembering:** Git refuses to overwrite an untracked file on pull/merge even when the content is identical. Move it aside first.
- **`sims/`** exists locally and is empty.

### 2.3 The `kite/` package (ported from vigil, verbatim apart from log names and a macOS hint)

`kite/types.py`
- **`wrap_deg(a)`** wraps an angle to (-180, 180].
- **`Frame`:** `image` (BGR uint8, read-only), `t` (laptop time it became available), `seq` (monotonic per source), `source`.
- **`Telemetry`:** yaw, pitch, roll in degrees; `height_m` ('h', above the takeoff point); `tof_m` (downward ToF); `vx`, `vy`, `vz` in m/s (comment: vigil's lag log read forward as **negative** `vgx`); `battery_pct`; `flight_time_s`.
  - `altitude_m()`: clearance above whatever is below. Prefers the ToF when it reads 0.1-3 m.
  - `floor_altitude_m()`: height above the floor. Uses the ToF only when it agrees with 'h' within 0.2 m, otherwise 'h'.
- **Conventions:** heading in degrees, clockwise positive (like `cw`); metres; `time.time()` seconds.

`kite/config.py`: plain dataclasses, no YAML loader yet.

| Section | Field | Default | Note |
|---|---|---|---|
| `SafetyCfg` | `max_rc` | 50 | clamp on every rc channel. Comment: stick 50 ~ 0.5 m/s, raise it to keep up with a walk |
| | `min_altitude_m` | 0.5 | |
| | `max_altitude_m` | 2.3 | |
| | `min_battery_pct` | 20 | |
| | `max_flight_s` | 420 | |
| | `frame_timeout_s` | 1.0 | |
| | `telemetry_timeout_s` | 1.0 | |
| `DroneCfg` | `kind` | `tello` | `tello` or `dry_run` |
| | `move_speed_cm_s` | 50 | |
| | `yaw_sign` | 1 | |
| | `video_fps` | 60 | reader cap; must sit above the ~30 fps stream |
| | `video_backend` | `pyav` | or `opencv` |
| | `video_lag_s` | 0.25 | |

`kite/drone/base.py`: the `Drone` interface.
- **Two ways to move, never mixed:**
  - `rc(lr, fb, ud, yaw)`: sticks -100..100 (+ = right / forward / up / clockwise), sent at 10-20 Hz.
  - Discrete `move(dir, 20..500 cm)` / `rotate(±1..360°)`: **non-blocking**, poll `busy()` and `last_result()`.
- While `busy()`, `rc()` is ignored, because an rc packet would abort the Tello's move.
- `stop()` hovers now. Helpers: `check_move`, `check_rotate`, `clamp_rc`.

`kite/drone/tello.py`: `TelloDrone`, built on djitellopy 2.5.0.
- **Commands:** `send_command_without_return`. Replies are polled from djitellopy's reply list with a per-command timeout:
  - takeoff 20 s, land 15 s;
  - move: expected (cm / speed + 1.5 s) + 6 s;
  - rotate: expected (|deg| / 60 + 1 s) + 5 s.
- **Reply hygiene:**
  - Replies don't say which command they answer, so the reply list is cleared before each new command.
  - After `stop()`, a 1.0 s window passes before the next command, so the late reply of the cancelled move cannot be credited to it.
  - At least 0.1 s between sends (the Tello drops commands sent closer together).
- **Takeoff and landing are protected:**
  - `stop()` never interrupts a takeoff or a landing.
  - `land()` is idempotent and allowed during takeoff.
  - A failed or lost land reply with `h ≤ 0.2 m` counts as landed.
- **Video start:** `streamon` is **re-sent up to 4 times** until UDP packets actually reach port 11111 (3 s wait each). Otherwise it raises a clear diagnosis:
  - the phone app is holding the stream;
  - the firewall or macOS Local Network permission is blocking it;
  - another program holds the port (`lsof -nP -iUDP:11111`).
- **Dry run:** `dry_run=True` connects, streams and reads telemetry, but **never sends motion**. Commands complete after their expected duration.
- **Telemetry:**
  - converted to SI units; ToF outside 0.1-8 m becomes `None`; yaw times `yaw_sign`; velocity scale 0.1 (the Tello reports dm/s);
  - djitellopy does not timestamp state, so each new state dict is stamped on first sight. A frozen state stream (its receiver thread can die on one bad packet) therefore shows up as stale.
- **`close()`:** lands if flying, stops the video, calls `tello.end()`.

`kite/drone/safety.py`: `SafetyGovernor(inner, SafetyCfg)` wraps any drone.
- **Every loop**, `check(now, last_frame_t)` returns `'ok'`, `'hover'` or `'land'`:
  - lands on low battery or when flight time is over (re-sends land every 2 s if it was lost);
  - hovers with rc 0 when video or telemetry is stale.
- **rc:**
  - every channel is clamped to ±`max_rc`;
  - above the ceiling it forces `ud ≤ -20`;
  - climb is tapered to 0 over the last 0.4 m (the drone keeps rising ~0.2 m after rc 0);
  - below the floor, descent is cut (tapered the same way);
  - rc is ignored while landing or while hovering for safety.
- **Discrete moves:** up/down moves that would cross a limit are refused with `error: safety (...)`.

`kite/sources/`: newest-frame readers.
- **`FrameSource`:** `read()` is non-blocking and returns the newest `Frame` or `None`. Also `start/stop`, `wait_first(timeout)`, and use as a context manager.
- **`TelloVideoSource(url="udp://@0.0.0.0:11111", fps=60, backend="pyav"|"opencv")`:**
  - one thread opens the stream and retries until it decodes;
  - it reopens after 20 s without frames, and `seq` stays monotonic across reopens;
  - FFmpeg open/read timeouts are 15 s, because nothing decodes before a keyframe and the Tello sends them seconds apart.
- **`PyAVStream`:** djitellopy's decoder. Corrupt packets are skipped instead of ending the stream.
- **`VideoStream`:** the team's OpenCV/FFmpeg reader.
  - `nobuffer|low_delay|framedrop`, grab-drain with retrieve capped at `fps`;
  - it never releases the capture while the thread is inside `grab()`, because that segfaults on a stalled stream.
- **Others:** `WebcamSource(index, 960x720)`, `VideoFileSource` (real-time replay, optional loop), `SteppedVideoSource` (deterministic: next frame per read), `StaticImageSource`.
- **All frames are BGR and read-only.**
- The `SOURCES` registry from vigil was dropped.

`kite/drone/__init__.py`: `make_tello(cfg, dry_run)` builds a `TelloDrone` from `DroneCfg`.

### 2.4 Tests
- **Fast suite:** `python -m pytest -m "not slow"` → **31 passed, 4 deselected in ~13 s** (run 2026-10-23 on Arthur's laptop). The slow tests stream real H.264 over local UDP and need ffmpeg.
- **`tests/test_drone.py`** (fake djitellopy):
  - streamon re-send; no-video diagnosis; port held by another program;
  - non-blocking takeoff; rc blocked while busy; rc sent as ints;
  - error reply and timeout; stop discards the late reply; stale replies cleared; min gap between commands;
  - telemetry conversion; dry run sends no motion;
  - land idempotent; stop never cancels takeoff or land; lost land reply on the ground; close lands;
  - dry run from config; frozen state stream detected as stale;
  - governor: clamp, ceiling/floor, refused moves, low-battery land with re-send, max flight time, hover on stale video/telemetry and recovery.
- **`tests/test_sources.py`:**
  - stepped source order; file real-time pacing and newest-frame serving; loop and restart; read-only frames;
  - team VideoStream via `TelloVideoSource`;
  - before start; default cap; `wait_first` timeout;
  - slow: full-rate reader with a safe stop on a stalled UDP stream; a stream that starts after the open timeout.
- **`tests/drone_test.py`:** a **flight script, not a pytest module**. It is the Tello response-lag test.
  - **Measures:** command round trip (`battery?` ×10); takeoff and land time; state and video rates, jitter and worst gap.
  - **Per axis** (yaw, throttle, pitch, roll; rc steps both ways, `--trials` repeats): lag to the first reaction in telemetry, speed-up to 90 %, stop time, command→video, and the video delay (video minus telemetry lag). Also discrete forward/back moves.
  - **Options:** `--mag 30 --dur 1.5 --settle 2 --trials 3 --forward 50 --hover 3 --min-battery 30 --preview --preview-scale --video-fps 30 --ground --analyze JSON --out results`.
  - **Safety:** 3 × 3 m clear space, prop guards, battery ≥ 30 %. Ctrl+C lands.

### 2.5 Environment and packaging
- **`pyproject.toml`:**
  - project `kite` 0.1.0, `requires-python >= 3.12`;
  - deps: numpy ≥ 1.26, opencv-python ≥ 4.9, av ≥ 12, djitellopy ≥ 2.5; `[dev]` adds pytest ≥ 8;
  - pytest: `testpaths = tests`, `python_files = test_*.py`, marker `slow`.
  - djitellopy 2.5.0 on pip has no `tello.stop()`, so kite sends "stop" itself.
- **Installed in `.venv`:** Python 3.12.14, av 19.0.2, djitellopy 2.5.0, numpy 2.5.4, opencv-python 5.0.0.93, pytest 9.1.1.
  - **No ultralytics or torch yet:** they are needed before any detector is ported.
- **Install everything while on normal internet.** The Tello Wi-Fi has none, and YOLO weights must be downloaded beforehand.

### 2.6 `dashboard/` on `main` (Karthik)
- **What it is:** a SpacetimeDB 2.13 module (`dashboard/spacetimedb/src/index.ts`) plus a React 18 / Vite 7 dashboard (`dashboard/src/`).
- **It is still the starter template:**
  - one public table `person { name: string }`;
  - reducers `add(name)` and `sayHello`;
  - `init`, `onConnect`, `onDisconnect` hooks;
  - the dashboard lists people and adds names.
- **Database:** `kite-sync-pl6rp` on Maincloud. Scripts: `npm run dev`, `spacetime:publish` (needs `spacetime login` as owner), `spacetime:generate`, `spacetime:publish:local`.
- **Writing from Python needs no SDK:**
  1. `POST https://maincloud.spacetimedb.com/v1/identity` returns a token;
  2. `POST .../v1/database/kite-sync-pl6rp/call/<reducer>` with `Authorization: Bearer <token>` and a JSON body of arguments.
- **Offline option:** `spacetime start --listen-addr 127.0.0.1:3000`, then `VITE_SPACETIMEDB_HOST=ws://localhost:3000`.

### 2.7 `earpiece/` on `feat/airpods-audio` (Karthik, WIP, not on main)
Python package `earpiece/earpiece/`. I read the module headers and the first config sections.

- **Frame:** bearings are in the **wearer's frame**: 0 = ahead, +90 = right, -90 = left, ±180 = behind. Metres. Keys are loaded from the repo `.env`.

- **`config.AudioCfg`:**
  - output: 48 kHz (AirPods A2DP), block 512 (~10.7 ms; Bluetooth adds 150-250 ms), device matched by "AirPods";
  - gains: master 0.9, limiter ceiling 0.89 (-1 dBFS), speech 1.0, cues 0.55, cues ducked by 0.45 during speech;
  - checks for AirPods every 2 s.

- **`config.SonarCfg`** (proximity warnings):

  | Setting | Value |
  |---|---|
  | assumed walking speed | 1.2 m/s |
  | corridor | half-width 0.55 m, soft margin 0.35 m |
  | max range | 6 m |
  | start beeping | time-to-contact < 4 s |
  | critical | time-to-contact < 1 s, or closer than 0.75 m |
  | hysteresis | 15 % |
  | beep interval | 0.85 s down to 0.11 s |
  | pitch | 520-1250 Hz |
  | sources | at most 2 at once (second at gain 0.55) |
  | stale track | 0.6 s |
  | spoken label cooldown | 6 s per track, 2.5 s between any two labels |

- **`events.py`: the wire contract** (one JSON object per message, with `type`):

  | Type | Example / meaning |
  |---|---|
  | `obstacles` | `{"source":"webcam","seq":12,"objects":[{"id":"7","kind":"person","bearing":-20,"distance":2.4,"closing_speed":0.4,"confidence":0.8}]}`; missing ids = gone |
  | `guidance` | `{"text":"In 20 meters, turn left onto Lyon Street","key":"step-3","ttl":8}` |
  | `beacon` | `{"bearing":-35,"distance":42,"label":"Palace of Fine Arts"}`, or `{"active":false}` (Soundscape-style) |
  | `hazard` | `{"kind":"crosswalk","state":"dont_walk","bearing":5,"distance":9}` |
  | `say` | `{"text":"...","priority":"info"}` |
  | `scene` | context for questions, not spoken |
  | `position` | `{"lat":...,"lng":...,"heading":270,"place":"Lyon St"}` |
  | `status` | `{"source":"drone","battery":34,"state":"following"}` |
  | `heartbeat` | `{"source":"webcam"}` |
  | `control` | `mute_cues`, `unmute_cues`, `pause`, `resume`, `volume_up`, `volume_down`, `repeat`, `beacon_on`, `beacon_off`, `stop_speaking` |

  - Priorities, most important first: `critical`, `hazard`, `nav`, `answer`, `info`, `system`.
  - Kind aliases: pedestrian → person; car/truck/bus/motorcycle → vehicle; bicycle/cyclist/scooter → bike; stairs → stairs_down; step/drop → curb.
  - Limits: 64 objects, 400 characters of text, 500 m distance. Everything is validated and clamped.

- **`mixer.Engine`:**
  - renders in the PortAudio callback and never blocks; other threads talk to it through a lock-free deque;
  - three buses: `cue` (ducked under speech), `alert` (never ducked), `speech` (one streamed utterance at a time);
  - sample-accurate scheduling; a master peak limiter.

- **`sounds.py`:** synthesised, no asset files. Attack of 2-4 ms with exponential decay; energy kept under ~6 kHz (where Bluetooth AAC is clean). One timbre per class:
  - person: marimba-like knock;
  - vehicle: buzzy two-tone;
  - obstacle: sine pip;
  - drop: falling glide.

- **`spatial.py`:** binaural rendering with the Brown & Duda (1998) spherical-head model (time delay between the ears plus a head-shadow filter, head radius 8.75 cm). Sources behind the wearer get an extra low-pass, because front/back confusion is the model's weak point. Needs scipy.

---

## 3. The Tello: facts, measured numbers, gotchas

### 3.1 Platform facts
- **Network:**
  - the standard Tello is **access-point only**: the laptop joins `TELLO-XXXXXX` and the drone is at 192.168.10.1;
  - no station mode and no mission pads (those need the EDU); SDK 2.0;
  - **the Tello's Wi-Fi has no internet.**
- **Ports:**
  - commands: UDP 8889 (text);
  - state: UDP 8890, ~**10 Hz** (9.8 Hz measured);
  - video: UDP 11111, H.264 960×720 at ~30 fps.
  - **Only one program can hold these ports.** Check with `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111`.
- **Commands used:** `command`, `streamon`, `takeoff`, `land`, `emergency`, `stop`, `up/down/left/right/forward/back <20-500 cm>`, `cw/ccw <1-360>`, `go x y z speed`, `speed <10-100>`, `rc a b c d`.
- **Behaviour:**
  - it **auto-lands after 15 s without any command**, so send `rc 0 0 0 0` keep-alives while hovering;
  - takeoff ends ~0.8-1.0 m up;
  - discrete moves are closed-loop on the drone's own **optical-flow position hold**. That needs a textured, well-lit floor; vigil's plan estimated ±10 cm per metre and ±3-5° per turn **(unverified)**;
  - the downward ToF makes it **climb by a table's height when it crosses a table**, so never fly over furniture.
- **Camera:**
  - fixed, facing forward, no tilt;
  - the 960×720 **stream is cropped** relative to the 82.6° stills spec: published calibrations give f ≈ 920 px, about **55° × 43°**;
  - vigil measured **fx = fy = 1077 px** on site **(measured)**, which is narrower than that.
- **Battery:**
  - ~6-8 min of flight with video; safety config caps flight at 420 s and lands below 20 %;
  - vigil used a fresh battery per run.
- **Telemetry fields:** `pitch roll yaw vgx vgy vgz tof h bat time` (plus others). ToF reads 6553 when out of range and ~10 cm on the ground.

### 3.2 Measured rc response
From vigil's lag test on 2026-09-26, **rc stick 30 only**; larger sticks are assumed linear **(unverified)**. Source: `~/vigil/integration/fruitfly-training/archive/rl-pipeline/data/lag_test/fit_20260926_052637.json`.

| Axis | Dead time | Time constant | Gain per 100 stick | Command → visible in video (median) |
|---|---|---|---|---|
| yaw | 0.12-0.20 s (≈0.17) | 0.02 s (near instant) | ≈ 55 °/s | 0.39 s |
| forward (pitch) | 0.44-0.54 s (≈0.48) | 0.39-0.47 s (≈0.44) | ≈ 0.96 m/s | 1.17 s |
| lateral (roll) | 0.44-0.54 s (≈0.49) | 0.40-0.50 s (≈0.44) | ≈ 0.97 m/s | 0.82 s |
| throttle | 0.37-0.45 s (≈0.41) | 0.02 s | ≈ 0.32 m/s | 0.61 s |

- **Video delay** behind telemetry: 6 ± 78 ms for yaw and 21 ± 40 ms for a forward move, worst 85 ms. vigil used `drone.video_lag_s = 0.25` as the "frames show the new view" margin after a discrete command.
- **Rules of thumb vigil used:**
  - rc 100 ≈ 1 m/s for dead reckoning;
  - forward control needs low gains plus feed-forward, because the visual loop on forward motion is over 1 s.
- **Open question for kite:** if rc 100 really is only ~1 m/s, the Tello **cannot outpace a 1.2-1.3 m/s walker** (the earpiece config assumes 1.2 m/s). Measure the forward speed at rc 60 and rc 100 early: `tests/drone_test.py --mag 60`, then `--mag 100`.

### 3.3 Gotchas already hit (most of them are handled in `kite/`)
- **Phone app:** while the Tello phone app is connected, the drone streams video to the phone. Close the app and take the phone off the drone's Wi-Fi, then restart the drone.
- **`streamon`:** sometimes the first one is ignored, so re-send it until packets arrive (handled).
- **macOS permissions:** allow incoming connections for Python, and give the terminal app **Local Network** access (System Settings > Privacy & Security). Otherwise sends fail with "No route to host".
- **Reply hygiene:** replies don't name the command they answer. Clear stale ones, and after `stop` ignore 1 s of replies (handled).
- **Command spacing:** commands sent closer together than 0.1 s are dropped (handled).
- **Frozen state stream:** djitellopy's state receiver thread can die and leave state frozen. kite stamps each new dict, so a freeze shows up as stale (handled).
- **Colour order:** **djitellopy's `frame_read.frame` is RGB**, while kite's readers give BGR. The colour checks (blue bottle) silently fail on RGB.
- **Reader cap:** the decode/retrieve cap must sit well above 30 fps. A cap equal to the stream rate drops ~40 % of frames because of arrival jitter.
- **Keyframes:** nothing decodes before a keyframe, and the Tello sends them seconds apart. Short FFmpeg timeouts abort the stream on a weak link, which is why they are 15 s.
- **Yaw sign:** telemetry yaw may run opposite to `cw`. vigil checked it automatically on the first clean 20-135° rotation.
- **Pitch sign:** unverified. vigil's site config set `pitch_sign: 1`.

---

## 4. vigil, the previous project

### 4.1 What and where
- **ReachGlass ("vigil"):**
  - the Tello hovers behind and above the wearer's head;
  - on "find my water bottle" it explores the room, finds the team's **blue Hydro Flask**, flies to it, then **guides the wearer** there;
  - guidance uses haptic pads on 3D-printed glasses and spoken cues in AirPods;
  - also: "find arthur" (face ID).
- **Results:** HackGT 13, 3rd place. Built on a Windows laptop (Intel Core Ultra 7 258V).
- **Local clones:**
  - `~/vigil` = `github.com/parthm667/vigil`, 35 commits, newest (2026-09-29, "rearranged bc sim is kinda broken lwk");
  - `~/Desktop/jerkgt13` = `github.com/nathanwuzhao/jerkgt13`, same `reachglass/` code, 2 days older.
- **Layout:**

  | Path | Contents |
  |---|---|
  | `reachglass/` | the stack (~8.5k lines) |
  | `tests/` | ~200-230 tests |
  | `tools/` | latency test, video diagnostics, face enrolment, model download |
  | `voice/` | Windows AirPods app |
  | `firmware/` | ESP32 glasses |
  | `integration/fruitfly-training/` | the fruit-fly RL team's repo copy, including the two patches |
  | `integration/yolo/`, `rewrite/yolo/` | stand-alone copies of the detector |
  | `PLAN.md` | the 28 h plan |
  | `WORKFLOW.md` | demo runbook |
  | `site.yaml` | on-site values |

### 4.2 Conventions (vigil `types.py`; worth keeping in kite)
- **Mission/world frame:**
  - origin = the drone when the mission starts; x = its heading then; **y = to the RIGHT**;
  - headings in degrees, **clockwise positive**;
  - a point at distance d along heading h is `(d cos h, d sin h)`.
- **Camera:** image x right, y down. Bearing = horizontal angle from the optical axis, **+ right**. Elevation **+ up**. The world heading of a seen object = drone heading + bearing.
- **Units:** `time.time()` seconds, metres, degrees (names ending in `_rad` are radians).
- **Frame clash:** the fruitfly runtime used **y = left**, and kite's earpiece uses the **wearer's** frame. Convert at the boundary.

### 4.3 Config system (`reachglass/config.py`)
- Nested dataclasses with defaults, overridable from YAML (`--config site.yaml`) or a dict.
- **Unknown keys raise**, so a YAML typo fails loudly.
- `ComponentSpec(kind, params)` picks a pluggable implementation through a registry (`DETECTORS.build(spec)`). Naming a new `kind` resets the params (a trained YOLO must not inherit the YOLO-World prompts).
- Target presets: `--target yolo-world` (default) or `--target color`.
- **`validate()` cross-field rules:**
  - `follow.altitude_m ≥ person_height_m + 0.2`;
  - `follow.min_range_m < distance_m`;
  - `follow.altitude_m ≤ safety.max_altitude_m`;
  - steering must be `pid` or `fly`;
  - travel altitude inside the safety band;
  - `tracking.max_age_s ≥ lost_after_s`;
  - face-ID reject threshold < match threshold.

### 4.4 Main program (`reachglass/app.py`)
- **Modes:** `python -m reachglass sim|tello [--dry-run] [--config] [--target] [--no-takeoff (dry run only)] [--autostart] [--headless] [--record x.mp4] [--query ... --at s] [--hz 20] [--udp 5005] [--announce-udp 5006] [--log]`.
- **Loop at 20 Hz:**
  1. `sense()`: newest frame → perception, telemetry → odometry;
  2. `drone.check()`: safety;
  3. poll inboxes (stdin + UDP 5005) → `mission.query`;
  4. `mission.step()`;
  5. JSONL run log, dashboard window.
- **Start-up:**
  - wait for `takeoff` (typed or the `t` key) unless `--autostart`;
  - refuse to start below min battery + 10 %;
  - wait up to 30 s for the first decoded frame.
- **Keys:** `t` takeoff, `SPACE` hold/resume, `l` land, `e` EMERGENCY (motors off), `q` land and quit. Ctrl+C lands.
- **Announcements** are printed and sent to UDP 5006 for the voice app's text-to-speech.

---

## 5. vigil perception in detail

### 5.1 Data types (`types.py`)
- **`Detection`:** `cls`, `conf`, `bbox` (x1, y1, x2, y2 px), `keypoints` (K, 3) in COCO-17 order for people, `track_id`, `source`; properties `cx cy w h area`. Plus `iou(a, b)`.
- **`Pose2D(x, y, heading_deg)`:** `point_at(d, bearing)`, `bearing_to(x, y)`, `distance_to`, `moved(forward, right, turn)`.
- **`ObjectObs`:** `det`, `bearing_deg`, `elevation_deg`, `range_m` (horizontal), `range_src`.
- **`PersonObs(ObjectObs)`:**
  - `facing_deg`: relative to the camera→person ray, clockwise. **0 = back to us**, +90 = facing image-right, ±180 = facing the camera;
  - `facing_conf`;
  - `range_lo_m`: the most cautious cue, used before moving toward them;
  - `name`, `name_sim` (face ID).
- **`TargetObs(ObjectObs)`:** adds `confirmed`.
- **`PerceptionResult`:**
  - `t`, `seq`, `latency_ms`, `image_size`, `detections`;
  - `persons` and `person` (the locked one); `targets` and `target` (the locked one); `context` (furniture);
  - `ran` (role → did it run this frame); `target_ran`; `person_unseen_s`; `target_unseen_s`.

### 5.2 Camera geometry (`geometry.py`, `CameraModel`)
- **Pinhole model:** `width height fx fy cx cy pitch_deg`. `from_fov(...)`, and `scaled()` / `for_frame(w, h)` for other resolutions.
- **Angles:**
  - `bearing_deg(x) = atan((x − cx)/fx)`;
  - `elevation_deg(y, body_pitch, x)` is exact off-axis via `ray()`, including camera + body pitch. The Tello camera is fixed, so it tilts when the drone pitches to accelerate.
- **Ranges:**
  - `range_from_height(px_h, real_h, x) = fy·H/h / cos(bearing)`, and `range_from_width` likewise (horizontal distance);
  - `range_to_height(x, y, Δh)`: horizontal distance to a point known to be Δh below the camera. This is the **head-elevation** and **top-edge** cue. It returns None below 1.5° of depression;
  - `ground_distance(y, cam_h)` for floor points.
- **Defaults:** f = 920 at 960×720, overridden by `camera.fx/fy` (vigil measured 1077).

### 5.3 Detectors
- **`UltralyticsDetector` (`detect/yolo.py`):**
  - takes `weights`, `classes`, `conf`, `imgsz`, `device` ("auto" = cuda → mps → cpu), `rename`, `iou`, `max_det`, `half`, `warmup`, `prompts` (YOLO-World open vocabulary), `agnostic_nms`, `require_color`;
  - `require_color` keeps only boxes whose middle 60 % (by width) is at least `min_frac` that HSV colour;
  - pose models return 17 keypoints;
  - bare weight names resolve into `models/`;
  - it warms up at construction.
- **Presets:**

  | Role | Model | Settings |
  |---|---|---|
  | Person | `yolo11n-pose.pt` | class person, conf 0.4, imgsz 640 |
  | Context | `yolo11n.pt` | conf 0.35, imgsz 640; chair, couch, bed, dining table, tv, refrigerator, sink, oven, microwave, potted plant, laptop, bench, toilet, suitcase, backpack |
  | Target (blue bottle) | `yolov8s-worldv2.pt` | see below |

- **Target (blue bottle) details:**
  - prompts "blue water bottle" and "hydro flask water bottle", both renamed to `bottle`;
  - conf 0.2, imgsz 960, agnostic NMS;
  - blue check HSV `[95,60,40,130,255,255]`, `min_frac 0.3`;
  - lock `confirm_conf` 0.3; size prior 0.24 m tall × 0.09 m wide;
  - **(measured on team photos):** found in 95-100 % of views at 1.3-4 m and 60 % at 6 m, with no false detections. 13 ms/frame on a Mac GPU, ~100 ms on a CPU. The bottle scores ≥ 80 % blue, other colours ≤ 24 %.
- **`ColorBlobDetector`:**
  - HSV ranges (default blue `[100,100,70,122,255,255]`), blur, open/close morphology;
  - contour filters: min area fraction 0.0003, fill ≥ 0.35, aspect ≤ 6;
  - confidence from fill and size;
  - with the colour preset, height 0.19 m (the blue body only) and `confirm_conf` 0.55;
  - tool `reachglass.tools.hsv_picker` widens the range from clicks in each lighting.
- **`rewrite/yolo/`:** a newer stand-alone copy (`person_detector()`, `bottle_detector()`).
  - `bottle-world.pt` = YOLO-World with the prompts **baked in**, so no CLIP model is needed at runtime (26 MB file);
  - **never downloads at runtime** (`python -m yolo.prepare` once, with internet);
  - `detect()` is not thread-safe;
  - ~15 ms per 960×720 frame on MPS.

### 5.4 The perception pipeline (`perception.py`)
- **Modes:** `follow`, `search`, `approach`, `guide`, `idle`. Each role runs every Nth frame (0 = never):

  | Mode | person | target | context |
  |---|---|---|---|
  | follow | 1 | 0 | 0 |
  | search | 5 | 3 | 2 |
  | approach | 3 | 2 | 0 |
  | guide | 1 | 0 | 3 |
  | idle | 0 | 0 | 0 |

  - On the first frame of a new mode every detector runs.
  - In search and approach, whichever detector "owns" the target class runs at least at the target rate.
  - A frame already processed (same `seq` and `t`) returns the cached result, so repeats never count as new sightings.
- **Filters on target detections:**
  - boxes touching the frame border (2 px) are dropped: they can't be sized and are typically a slice of jeans;
  - boxes ≥ 50 % inside a person box from the last 1.5 s are dropped (blue jeans are not the bottle);
  - the context detector never reports `person`.
- **`object_range()`:**
  - height prior when the box is fully in frame vertically;
  - width prior when fully in frame horizontally **and ≥ 25 px wide** (blur inflates thin boxes);
  - the width wins when the height-based range is more than 1.3× the width-based one (top/bottom hidden);
  - for furniture seen from above with the bottom cut off: the elevation of the box's top edge plus the class height;
  - **class heights (m):**

    | bottle | cup | backpack | chair | dining table | couch | bed | tv | laptop | refrigerator | potted plant | bench | sink | oven | microwave | toilet | suitcase |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    | 0.24 | 0.10 | 0.45 | 0.85 | 0.75 | 0.85 | 0.60 | 0.55 | 0.25 | 1.70 | 0.60 | 0.45 | 0.20 | 0.85 | 0.30 | 0.75 | 0.60 |

  - widths only for round objects: bottle 0.09, cup 0.08.
- **Single-frame plausibility** (`_plausible_target`):
  - the long/short side ratio must be within 2× of the prior's;
  - the range must be inside `target_range_m` (0.2-8 m).
- **Vocabulary:** every class of the target and context detectors (except "person") plus enrolled face names. The query parser may only pick from it.
- **Name targets ("find arthur"):**
  - the target lock is bypassed;
  - a face match on a person crop makes that person the confirmed target directly.

### 5.5 Multi-object tracker (`track/tracker.py`, `SimpleTracker`)
- **Per-class greedy matching**, best pairs first:
  - IoU ≥ 0.2 scores `1 + IoU`;
  - otherwise, **only for a track seen on the immediately previous frame**: centre distance < 0.15 × image diagonal and area ratio > 0.35. This handles fast yaws that shift the box more than its width.
- Unmatched detections start new tracks; tracks expire after `max_age_s` 1.5 s.
- Each track keeps `seen_frames` (tracker frame indices, the last 50).
- Swap for ByteTrack or BoT-SORT behind the same `update()` if needed.

### 5.6 Target lock (`track/lock.py`, `TargetLock`)
One object of a class, confirmed, sticky, with loss detection. Used for both the person (`select = largest`, no single-frame confirmation) and the target (`select = most_confident` plus the plausibility check).

- **Candidates:** detections of the locked class.
- **Before confirmation:** re-pick every frame, preferring tracks with repeated hits. A one-frame false blob can't block the real target.
- **Confirmed when:**
  - the track was matched in ≥ `confirm_hits` (3) of the detector's last `confirm_window` (5) **runs**, counting detector runs rather than camera frames (at stride 3, counting frames would make 3-of-5 impossible);
  - **or** one detection has conf ≥ `confirm_conf` and passes `plausible()`.
- **Once confirmed:** follows that `track_id` only and **never jumps to a twin**.
  - If the id suddenly lands more than 2 box-sizes away (the id glued onto something else), confirmation is dropped and must be earned again with repeated hits.
- **Track missing while candidates exist:**
  - re-attach to the nearest candidate within 3 box-sizes; after a real loss, within 0.25 × the diagonal;
  - if the locked track was brief (fewer hits than `confirm_hits`) and another candidate keeps being detected, switch to that one;
  - a re-attached lock needs repeated hits again.
- **Lost / released:** lost after `lost_after_s` (1.5 s) unseen; released after 3× that (4.5 s), when any candidate may be locked.
- **Snapshots:** `update()` returns a copy.

### 5.7 Person geometry (`person/estimator.py`, `PersonEstimator`)
- **Range cues** (each only when it applies):

  | Cue | When | Notes |
  |---|---|---|
  | `full_height` | whole body in frame | height prior; relative error σ 0.07 |
  | `head_elev` | top of the head visible and altitude known | `range_to_height(cx, y1, altitude − person_height)`. **The only cue that doesn't change when the person turns**, so it's the one that matters when following from above and behind. Needs ≥ 0.15 m height difference. σ from the height prior (0.05 for strangers, **0.02 once the wearer is measured**), altimeter 0.03 m, and pitch uncertainty (1° with telemetry pitch, 3° without) divided by sin·cos of the depression angle |
  | `shoulders` | both shoulders visible and facing trusted (|cos facing| > 0.6) | shoulder width 0.40 m corrected by |cos facing|; σ 0.10 + 0.5 (1 − |cos f|) |
  | `bbox_width` | box not cut at the sides or bottom | body width 0.48 m (or facing-corrected with body depth 0.28 m); σ 0.2 (facing known) or 0.35 |

  - **Fusion:** inverse-variance weighted mean **in log space** (the errors are multiplicative). `range_src` lists the cues, most trusted first. Valid range 0.3-15 m.
- **Two passes:**
  1. the turn-independent cues (height, head) give a range hint;
  2. facing is computed with that hint;
  3. the shoulder and width cues are corrected by facing only if facing confidence ≥ 0.5;
  4. fuse.
- **`range_lo_m`:** the smallest cue, and also the box width against the narrowest a person can look (0.28 m, or 85 % of the facing-corrected width).
- **Facing:**
  - **Magnitude:** from the signed shoulder vector `dx = x_rshoulder − x_lshoulder`. dx > 0 means a back view; it is normalised by the shoulder width expected at the hinted range (or 83 % of the box width), and the magnitude is `acos`.
  - **Sign:** from the nose offset relative to the shoulder midpoint when the face is visible, otherwise from which ear is more visible.
  - **Confidence:**
    - 0.8 near pure back or front views, 0.3 in between, 0.5-1 when the sign is known;
    - **capped at 0.25 when the face visibility disagrees with front/back** (a left/right keypoint swap would otherwise flip it 180° with full confidence);
    - capped at 0.3 without a range hint.
- **Geometry warning:** at 2.0 m high and 1.0 m behind only the **head** is in frame, because the shoulders are ~29° below the camera. Facing and orbit can't work there, and detection relies on YOLO seeing a head from above. Shoulders come into view from ~1.6 m behind.

### 5.8 Face identification (`person/identify.py`)
- **"find <name>":** one photo per person in `people/` (`arthur.jpg` → "arthur"). `tools/enroll_faces.py` computes embeddings plus a pairwise confusability check; `tools/face_id_webcam.py` is the go/no-go check.
- **Backends:**
  - insightface `buffalo_l` (SCRFD + ArcFace), match 0.40, reject 0.20;
  - zero-install fallback `opencv` (YuNet + SFace, threshold ~0.363).
- **How it runs:**
  - verifies on person-box crops, upscaling small faces;
  - the identity sticks to the track between checks and is inherited across tracker resets;
  - faces are readable to ~3.5-4 m through the Tello camera.
- **The drone never approaches a person target.** It stops 1.5 m short and announces where they are.

### 5.9 Free space (`mapping/freespace.py`)
- **Default `null`:** nothing metric. Free space only comes from evidence: an object seen at distance d proves the line of sight to it is clear.
- **`depth` (Depth-Anything-V2-Small): EXPERIMENTAL, NOT RELIABLE on the Tello.**
  - At 1.2 m altitude with the 43° vertical FOV, the floor only appears beyond ~3.1 m, so the scale fit fails.
  - The model's "lower = closer" prior makes walls look like floor: a wall 1.5 m away scored as open as a 6 m room.

---

## 6. vigil autonomy in detail

### 6.1 Behaviour framework (`behaviors/base.py`, `loop.py`)
- **A behaviour** is `start(ctx)` once, then `step(ctx)` every loop, returning `running`, `success` or `failure`.
- **`Ctx`** is the only integration surface:
  - `cfg`, `drone`, `perception`, `odom`, `grid`, `memory`, `freespace`;
  - `now`, `frame`, `new_frame`, `res`, `tel`, `say`;
  - vantage bookkeeping; `person_origin` and `person_heading` (where the wearer stood and faced when the query came); `target_cls`; `notes`; `cmd_done_t`; `last_person`.
  - `altitude` = floor altitude; `clearance` = ToF.
- **`frame_after_cmd()`:** true only when the frame arrived ≥ `video_lag_s` after the last discrete command finished, i.e. it really shows the new view.
- **`Discrete(kind, value, direction)`:** issues one move or rotate, waits for the reply, then:
  - resets the tracks (boxes jump after a discrete command);
  - updates odometry: a rotation uses yaw before/after for the sign check; a move uses the commanded distance;
  - a move whose reply timed out is still credited (the Tello most likely flew it) but reported as failure.
- **`sense(ctx, source, now)`:** telemetry → newest frame → perception only on **new** frames → remember the last seen person → odometry heading.

### 6.2 Odometry, grid, semantic memory (`mapping/`)
- **`Odometry`:**
  - heading = `yaw_sign × (telemetry yaw − yaw at mission start)`, falling back to the sum of commanded rotations;
  - **the yaw sign is auto-checked once**: on a clean 20-135° rotation, it flips if the measurement disagrees by > 60° and agrees within 20° when flipped;
  - position advances by the **commanded** distance of completed moves × `move_scale` (the Tello closes that loop itself);
  - keeps `path`;
  - the unmerged follow-avoid patch adds `on_velocity(fwd, right, dt)` for rc flight.
- **`Grid`** (0.25 m cells, 16 × 16 m), per cell:
  - `seen`: dwell frames that looked at it;
  - `free`: evidence of emptiness at flight level, from lines of sight to detected objects or metric free-space rays;
  - `blocked`: object footprints, ends of free-space rays;
  - `top`: highest obstacle top, inf = unknown.
  - **Methods:**
    - `mark_view(pose, hfov, range, free)`;
    - `mark_free_ray`;
    - `mark_blocked(x, y, radius, top_m)`;
    - `blocks(c, altitude)`: true when top + 0.2 m > altitude;
    - `novelty()`: never-seen fraction along a ray;
    - `clear_distance(pose, heading, max, unknown_ok 0.5 m, margin 0.35 m, altitude)`: how far through free-evidenced cells, at most `unknown_ok` beyond the evidence.
  - Footprint radii: dining table 0.6, couch 0.8, bed 0.9, chair 0.3, person 0.5, others 0.3.
- **`SemanticMemory`:**
  - a sighting joins an existing object of the same class within 0.6 m (small things) or 1.0 m (furniture), plus 0.15 × range;
  - position = weighted mean with weight `conf / range²`;
  - an object is confirmed when any sighting was lock-confirmed, or it has ≥ 3 sightings;
  - `best(cls)` picks confirmed first, then by weight.

### 6.3 Person following (`behaviors/follow.py`, `FollowBehind`)
Continuous rc on every **fresh** frame with the locked person. Default values; the demo used `site.yaml` overrides, shown in brackets.

| Item | Setting |
|---|---|
| Geometry | `distance_m` 1.0 [1.8] horizontally behind, `altitude_m` 2.0 [1.9] above the floor; must be ≥ wearer height + 0.2 m |
| **Yaw** | `clamp(deadband(bearing, 4°) × 1.2, 40)`. Or the fruit fly's yaw with `steering: fly` |
| **Distance** | median of range estimates from the last 0.7 s → error = range − distance (deadband 0.1 m) → `× 40` rc/m, capped at 35 forward (~0.35 m/s, a slow walk) |
| Approach gate | moves **toward** the person only if `range_lo_m` > `min_range_m` (0.55) and the box is narrower than 75 % of the frame |
| **Feed-forward back-off** | range rate from a gap-free linear fit over ~1 s, plus the drone's own commanded speed from 0.3-1.3 s ago (rc 100 ≈ 1 m/s). Closing faster than 0.2 m/s → back away at their speed right away (`min(fwd, 100 × closing − 10)`, ≥ −50). The head leaves the frame at ~0.6 m |
| **Too close** | `range_lo` < 0.55 m or box > 75 % of the width → back off at rc −50 and climb a little |
| **Altitude** | `clamp(deadband(target − alt, 0.05) × 60, 30)` |
| **Orbit** | circular mean of ≥ 3 recent confident facings → `lateral = −0.5 × deadband(facing, 25°)`, capped at 30. Facing +90 means their back is to the image left, so move left. **The demo set `orbit_gain: 0`**: noisy facing caused drift |
| Lost while close | the last view was close, approaching, or head at the bottom 12 % of the frame. If unseen for 0.2-2.5 s they're walking under the drone: back off sideways away from their side and climb |
| Lost otherwise | < 0.35 s: keep the last command; < 1 s: hold; < 20 s: yaw toward the last-seen side at rc 20; then hover |

- It never finishes by itself; the mission switches behaviour when a query arrives.
- Status string: `person 1.8 m (lo 1.6) bearing +5 facing 3`.

### 6.4 Reacquire (`behaviors/reacquire.py`, plus the mission's REACQUIRE state)
1. Go to "look height", `max(min_alt + 0.2, person_height − 0.15)`. From follow height, a person closer than ~1.5 m is below the frame.
2. Turn first toward where the person stood when the query came.
3. Then turn in 30° steps, dwelling 0.8 s each, up to 13 steps.
4. Seen → FOLLOW. Not found → HOLD, saying "I can't see you. Say 'follow me'...".

### 6.5 Exploration: scan-score-hop (`behaviors/explore.py`)
- **`Scan`:**
  - rotate 8 × 45°;
  - after each turn, keep-alive rc 0 and wait `dwell_s` 0.8 s;
  - then call `observe()` on frames after the command (video lag passed) until the target detector has seen `frames_per_dwell` (2) frames **and** every active detector has run at least once, or a 3 s timeout;
  - stops early on a confirmed target.
- **`observe()`:**
  - targets → memory plus a free ray up to range − 0.3;
  - furniture → memory plus a blocked footprint, with the top height from its box's top edge, plus a free ray;
  - people → memory (cautious range if needed) plus a free ray;
  - mark the view (hfov, 3.5 m view range).
- **`choose_hop()`:** 24 headings, every 15°.

  ```
  dist     = grid.clear_distance(heading, hop_max 1.5, unknown_ok 0.5, altitude)   (skip if < 0.5)
  open     = dist / hop_max
  novelty  = never-seen fraction along the heading beyond the hop's end, within 3.5 m
  semantic = 1 + 2·Σ weights of remembered objects within 20° and 6 m
  cand     = 4 if an unconfirmed target sighting lies within 15°
  people   = 0 toward a recent (10 s) person within clearance (1.5 m) + dist and 30°, or toward where the wearer stood
  revisit  = 0.2 if the hop ends within 0.8 m of a previous vantage point
  score    = open · (0.3 + novelty) · (semantic + cand) · people · revisit       (none ≥ 0.05 → give up)
  ```

  - Semantic weights:
    - bottle: dining table 1.0, desk 1.0, refrigerator 0.6, sink 0.6, couch 0.3, chair 0.3, tv 0.2;
    - cup, backpack and laptop have their own tables;
    - `_person` for name targets: chair 0.8, couch 0.8, bench 0.5, dining table 0.5, bed 0.4.
  - For a person target, unidentified people become candidates to close in on, stopping 1.5 m short.
- **`Hop`:** rotate to the heading (if ≥ 5°), then move forward.
- **Budgets:** 150 s and 5 vantage points.
- **Before searching**, the mission descends to `scan_altitude_m` 1.3.

### 6.6 Approach (`behaviors/approach.py`)
- **Flies to where the map puts the target** and does not re-measure:
  1. level at `travel_altitude_m` 1.5;
  2. turn to the target;
  3. fly distance + `overshoot_m` 0.2 in moves of ≤ 4 m, re-levelling before each;
  4. end `over_object_m` 0.75 above the measured top.
- **Person target:** no climb and no fly-over; stop 1.5 m short.
- **Someone near the straight path** (within 1.0 m, including where the wearer stood) → stay put and say so.
- **Failures:**
  - a refused turn or forward move stops it there (success);
  - timeouts and refused height changes continue with the plan;
  - the mission retries once, then falls back to the memory position.
- Helpers `person_near_path` and `person_free_distance` are reusable for any "is someone in my way" check.

### 6.7 Guide the wearer (`behaviors/guide.py`, `mapping/walkpath.py`)
1. **Lower:** optional and off by default.
2. **Look back:**
   - hover 1.5 s;
   - turn toward where the wearer stood, then ±45°, in steps ≤ 90° with 0.8 s pauses;
   - each view runs the person detector ×3 and furniture ×2;
   - the wearer is the cluster of sightings nearest where they stood (within 2.5 m); keep turning if nobody is there;
   - heading comes from pose facing.
3. **Plan:**
   - cost map on the grid: blocked 50, margin 10 (inflated 0.4 m), proven-clear 1, unknown 1.5; cells the drone flew through count as proven;
   - Dijkstra over 8 neighbours;
   - goal = the walkable cell nearest the target (slack 0.3 m, the cheapest of those);
   - string-pulled into the fewest straight legs;
   - a raised target (top > 0.4 m, i.e. on a table) ends the walk 1.0 m short.
4. **Guide:**
   - **measuring the wearer:** the person nearest their track (gate 0.8 m + 1.2 m/s × time unseen); position smoothed with α 0.5;
   - **their heading:** walking direction over 1.2 s if they moved ≥ 0.3 m, else pose facing;
   - **cue:** aim 0.8 m ahead along the path, then:
     - heading error ≤ 15° → **0 forward**, kept until > 30°;
     - otherwise **−1 / +1**, kept near 180° so noise can't flip it;
   - the cue is repeated at 4 Hz and on every change;
   - **drone:** turns to keep the wearer within 18°; re-plans when they're > 0.8 m off the path or a new obstacle lands on it;
   - **arrival:** **cue 2** after 2 consecutive measurements within reach (0.5 m), or when they were last within reach + 0.7 m and are now unmeasurable for 0.5 s (too close to see).
   - **Glasses handoff (vigil only):** within 2.5 m of the target, facing it within 30° (inside 5 m), or unseen for 1 s → the glasses camera's own target bearing takes over.
   - **Limits:** wearer lost 30 s → failure; 300 s cap.

### 6.8 Mission state machine (`mission/mission.py`)
```
IDLE --takeoff--> TAKEOFF -> CLIMB -> FOLLOW --"find X"--> DESCEND -> EXPLORE -> APPROACH -> ARRIVED -> GUIDE -> DONE
any: "land" -> LAND -> LANDED (land re-sent every 2 s while it still flies);  "stop"/"cancel" -> REACQUIRE -> FOLLOW
EXPLORE fails -> REACQUIRE;  GUIDE fails -> HOLD (SPACE resumes);  operator SPACE -> HOLD / resume
```
- **On "find X":**
  - the mission frame is reset at the drone (new odometry, grid and memory);
  - **the wearer's position and facing are recorded** from the last 2 s of sightings, using the follow behaviour's smoothed facing;
  - a "find" heard during takeoff is deferred;
  - a repeat while searching says "Still looking";
  - a repeat while ARRIVED refreshes the guidance from the wearer's current pose.
- **ARRIVED:** descend to `arrived_altitude_m` 1.5, announce the guidance, start GUIDE.
- **FOLLOW lost for 20 s → REACQUIRE.** This actually fires ~4.5 s after a loss once FOLLOW has run 20 s, because the lock release makes `person_unseen_s` inf.
- **Leaving GUIDE** always releases the haptic pads and closes the glasses stream.

### 6.9 Guidance text, query parsing, inboxes
- **`compute_guidance`:**
  - target vs the person's recorded position and heading → distance, turn, clock face;
  - text such as "I found the bottle: about 4 meters away, ahead to your right (2 o'clock).";
  - direction words: < 12° "straight ahead", < 35° "slightly to your…", < 70° "ahead to your…", < 110° "to your…", < 160° "behind you on your…", else "behind you".
- **`KeywordQueryParser`:**
  - **intents:** find, cancel, land, follow, describe, unknown. Priority: land > find (with a detectable object) > cancel > describe > follow;
  - synonyms map to COCO classes ("water bottle", "flask", "thermos", "drink", ... → bottle);
  - purposes map to classes ("thirsty" → bottle or cup, "sit" → chair or couch);
  - **the target is always inside the detector vocabulary**; anything else gets "I can't look for keys yet".
- **`LLMQueryParser(complete)`:** any language model behind the same interface; answers outside the vocabulary are rejected.
- **Inboxes:** stdin, UDP 5005, scripted (tests), multi.

### 6.10 Safety and run logging
- **SafetyGovernor:** identical to kite's (2.3).
- **`RunLog`:** JSONL every 0.1 s plus events. Each line has state, status, pose, telemetry (alt, ToF, yaw, pitch, battery), seq, latency, person (range, lo, bearing, facing, conf, src), target, detection count. State changes and spoken lines are logged as events.

---

## 7. Never merged: obstacle avoidance while following (FollowAvoid + looming)
- **Where:** only as a patch, `~/vigil/integration/fruitfly-training/docs/integration/reachglass_follow_avoid.patch`. Write-ups: `FOLLOW_AVOID.md` next to it, and `wip/FOLLOW_AVOID_WIP.md` (a later version with go-round, bystanders and ducking).
- **Status:** made against jerkgt13 HEAD `2302928`; `reachglass/` has changed since (face ID etc.), so **it won't apply cleanly**. Every result below is **(sim only)**.

### 7.1 Looming detector (`detect/looming.py`, `DivergenceLooming`)
"Is something ahead getting closer?" from the expansion of the forward view. No model; ~0.9 ms at 960×720 on an M-series CPU. Adapted from the FlyDrones retina (MIT).

1. Grey, downsampled to 96×72, blurred.
2. Compare with the newest frame ≥ 0.12 s old (at most 0.4 s).
3. **Remove the drone's rotation:** estimate the global image shift with coarse-to-fine Lucas-Kanade and undo it, so turning isn't looming.
4. Per cell of a 3×4 grid over the central 70 % of the frame, plus the whole region as one extra cell (a doorway with an edge on each side is measurable as a whole):
   - fit brightness constancy with a local similarity motion `u = a + s·x − w·y, v = b + s·y + w·x`;
   - **s/dt = 1/TTC** of that patch. A surface at distance Z approached at speed V gives s/dt = V/Z;
   - sideways motion (parallax) only shifts a cell, so it doesn't count.
5. Pixels in the locked person's box (+15 % margin) get no weight.
6. A cell reports only when its standard error is below 0.12 /s.
7. Per-cell exponential smoothing 0.5. Nothing counts as looming below a rate of 0.05 /s (TTC > 20 s).

- **Output `LoomingObs`:** rate, `ttc_s`, `x_frac` (where the looming patch is), `score`, per-cell rates, `dt_s`, and **`texture`** (the fraction of cells that can measure). Texture 0 means blind, which is **not** the same as clear.
- **Forward motion is not removed.** That is exactly what makes a static obstacle loom; the behaviour turns rate into distance with its own speed: `distance ≈ v × TTC`.

### 7.2 `FollowAvoid(FollowBehind)`
`FollowBehind` computes its rc as usual; an `RcTap` captures it, and these checks run before sending.

- **Map while following:**
  - context detector every 5th frame (strides `follow: {person 1, target 3, context 5, looming 1}`) → `observe()` at most every 0.5 s, plus a 3 s list of recent obstacles;
  - the pose is dead-reckoned from the rc actually sent (rc 100 = 1 m/s through a 0.35 s lag).
- **Brake (forward capped at 0)** when either holds:
  - an obstacle reaching flight height (top + 0.2 m > altitude) is in the **1.0 m × 0.6 m forward corridor**. It can come from recent detections or the grid; other people count with a 1 m keep-out; the followed person doesn't;
  - **looming on two frames in a row**: flying, TTC < 2.5 s and v·TTC < 0.9 m; hovering, TTC < 1.2 s. Looming is ignored while |yaw rc| > 20 (derotation leftovers), and the patch is remembered as an obstacle for 3 s.
  - It reverses (rc 15) only from **seen** obstacles closer than 0.45 m.
- **Sidestep 0.6 m:** after 1.5 s blocked (max 2 per episode), toward where the person was, needing ≥ 0.5 m of room within 0.8 m, and the whole slide must keep 1 m from every person.
- **Rear rule:** never back into something mapped behind. Hovering next to a mapped obstacle drifts away at rc 10.
- **Occlusion recovery:** the person vanished while braking or sidestepping, or within 2 s after.
  - For up to 15 s: yaw holds on their last spot, moved on along their walking direction by ≤ 1 m;
  - creep there at rc 20, only when not turning hard and looming has texture, ≤ 1.5 m in all, stopping 1 m short;
  - the mission's REACQUIRE is held off via `reacquire_wait`.
- **WIP extras:**
  - go-round: slide until the corridor is clear, ≤ 1.5 m for a seen obstacle, then pass and drift back;
  - bystanders: 1.1 m keep-out, plus a "not the person" list;
  - **lock the NEAREST person by range, not the largest box**;
  - opt-in "duck" under door headers.
- **Config kill switch:** `follow.avoid.enabled: false` gives exactly the old FollowBehind.

### 7.3 Results (vigil simulator, 50 s runs, 3 seeds, PID yaw)

| Scenario | Mode | Collisions | Min distance to obstacle | User in view | Time lost | Ends in |
|---|---|---|---|---|---|---|
| doorway (header 2.03 m) | off | **3/3** (pinned on the header) | 0.12 m | 0.86 | 0 s | FOLLOW, stuck |
| | on | 0/3 | 0.33 m | 0.39 | 23.4 s | REACQUIRE, HOLD, HOLD |
| | on + recovery | 0/3 | 0.34 m | 0.50 | 18.5 s | FOLLOW, REACQUIRE, REACQUIRE |
| tall box in path | off | 0/3 | **0.25 m** (near miss) | 0.87 | 0 s | FOLLOW ×3 |
| | on | 0/3 | 0.90 m | 0.29 | 28.1 s | HOLD ×3 |
| | on + recovery | 0/3 | 0.40 m | 0.44 | 19.2 s | FOLLOW, FOLLOW, REACQUIRE |
| table + bookcase | off | 0/3 | **0.34 m** | 0.86 | 0 s | FOLLOW ×3 |
| | on | 0/3 | 1.03 m | 0.36 | 24.7 s | HOLD, HOLD, REACQUIRE |
| | on + recovery | 0/3 | 0.49 m | 0.42 | 21.9 s | REACQUIRE, REACQUIRE, FOLLOW |

- **The win:** avoidance removes the doorway crash and the 25-35 cm near misses.
- **The price:** the user is out of view much more often. Recovery brings them back in 5 of 9 runs (versus 0 of 9 without it), and its creep cuts clearance to 0.3-0.5 m.
- **Limits:**
  - forward camera only: no rear or side sensing, and walls aren't mapped;
  - looming is blind on blank walls and in dark rooms, and over-reads on door jambs (braked ~2 m early in 2 of 3 doorway runs);
  - **never verified on real Tello video.**

---

## 8. Fruit-fly steering and the fruitfly-training runtime

- **What it is:** a subgraph of the male fruit-fly connectome (MaleCNS):
  - 1,446 spiking LIF neurons and 40,586 connections;
  - LC10a → AOTU relays → the DNa02/DNa01 steering neurons (a push-pull);
  - **the wiring is never trained**: only a 25-parameter encoder (box → spikes) and a 9-parameter readout (DN rates → yaw), with CMA-ES in simulation on Modal (~$100 in total).
- **What it does:** **only yaw**. Distance, altitude and safety stay classical. `fly.forward` must stay false: the untrained forward readout held 8-10 m away and surged in a real flight on 2026-09-27.
- **Results (sim only):**
  - trained fly: 6.6° RMS bearing error, against 8.9° untrained, 11.0° with shuffled wiring, 5.5° with no brain, 5.4° for a CMA-tuned PID, 6.3° for the hand PID;
  - **honest summary: it steers about as well as classical control**;
  - in vigil's simulator the shipped setting (`FLY-YAW_smooth_best.json`, deadband 4, hysteresis 3) gave 6.1° RMS against the PID's 8.5°, but moves the stick about 2× as much.
- **Plugging in:** in reachglass via `behaviors/fly_steer.py` (`FlyYaw.for_behavior`). It falls back to the PID yaw law if `flyfollow` is missing or raises. Each 50 ms brain tick costs ~2-3 ms of CPU.
- **Reusable runtime pieces** (`integration/fruitfly-training/flyfollow/`):
  - **`pilot/box_filter.py` `BoxFilter`:**
    - constant-velocity Kalman filter on the target box (cx, cy, h), three scalar 1D filters;
    - **latency-compensated**: a box is placed at capture time (`t_decoded − latency_s`) and `output(t)` predicts it to now;
    - outlier gate 5σ, re-initialised after 3 rejects, hold 0.5 s;
    - a clean way to fight the ~0.4-1.2 s visual delay.
  - **`pilot/governor.py` `PersonGovernor`:**
    - stick clamps;
    - **slew limits** (moves toward zero are instant, a sign change goes through zero);
    - lost target → turn toward the last bearing → hover → land request after a timeout (10 s in FOLLOW/GUIDE);
    - min-distance back-off;
    - altitude floor and ceiling; battery land; a brain watchdog (0.5 s).
  - **`runtime/geometry.py`:**
    - `head_box_from_person`: the head box from a person box. With the full body, `h × head_size / user_height`; with the feet cut off, `w × head_size / 0.45`. About 20-30 % error;
    - `floor_visible_min_range(alt)`: the closest visible floor is 1.28 / 2.55 / 3.06 m at 0.5 / 1.0 / 1.2 m altitude;
    - attitude-corrected bearing/elevation; class size priors with aliases.
  - **`runtime/controller_runner.py`:**
    - a 20 Hz control process on a ZeroMQ bus, with message contracts in `messages.py`;
    - modes FOLLOW / APPROACH / GUIDE;
    - target selection that tolerates detector id changes: locked id → association by position and size → "the only person in view" after 3 s.

---

## 9. vigil voice and glasses hardware
- **Voice (`voice/`, Windows only):**
  - AirPods Pro 2 stem presses captured as Bluetooth AVRCP media commands through a pinned Windows media session: 1 press = push-to-talk, 2 = repeat guidance, 3 = "stop";
  - the mic is opened only while recording, because an open mic locks the AirPods into low-quality HFP; a `--hands-free` mode has adaptive VAD and drains the mic queue every pass;
  - STT: local faster-whisper (`tiny.en` default); TTS: edge-tts with SAPI fallback, routed to the AirPods' live endpoint as wav;
  - queries in on UDP 5005, announcements out on UDP 5006.
  - **kite needs a macOS replacement.** Karthik's earpiece branch covers the output side.
- **Glasses (`firmware/`):**
  - an ESP32-CAM hosts Wi-Fi "rover" (password `rover1234`) and serves MJPEG at `http://192.168.4.1/stream`, mounted rotated 90°;
  - an Arduino Nano ESP32 at 192.168.4.50, UDP 4210, runs two VL53L0X ToF sensors and two servo haptic pads;
  - **the sides are crossed**: +1 "go right" presses the LEFT pad;
  - pads auto-release after 600 ms without commands.
  - Not part of kite, which uses a body webcam plus audio. The "one camera at a time, hand off near the goal" idea may carry over.

---

## 10. Lessons and pitfalls
1. **Follow geometry:**
   - from 2 m up and 1 m behind, only the head is visible;
   - range then rests on the head-elevation cue, so **measure the person's height with shoes to ±2 cm**: 5 cm of error is up to 25 % range error at that geometry;
   - facing and orbit need ≥ 1.6 m behind.
2. **Lock the nearest person, not the largest box.** From above, a bystander's full body beats the wearer's head-only box.
3. **Noisy facing makes orbiting drift** (turned off in the demo).
4. **The "lost for 20 s" rule fires after ~4.5 s:** `person_unseen_s` becomes inf when the lock releases. Use an explicit flag or timer.
5. **Visual delay dominates forward control** (~1.2 s to see a forward change):
   - low forward gains, feed-forward from the commanded speed, and medians over 0.7 s;
   - never trust a range rate across a detection gap.
6. **Discrete commands are odometry:** after each one, reset the tracks and wait `video_lag_s` before using frames.
7. **Count detector runs, not frames,** when a detector skips frames (confirmation windows).
8. **Gate single-frame confirmations by size, shape and distance.**
9. **Things the target detector must reject:** boxes touching the border, and colour hits inside people.
10. **djitellopy frames are RGB** (3.3).
11. **No internet on the Tello Wi-Fi:** download weights and pip packages beforehand, and never download at runtime.
12. **Looming and depth are unproven on real video:** looming is blind on texture-less surfaces, and the depth model mistook walls for open floor.
13. **The fruit-fly brain is a nice story, not a performance gain**; yaw only.
14. **Tests went stale against behaviour changes.** 3 vigil tests were failing at the end: follow distance, the full mission (the drone now flies over the bottle), and the floor bottle. The simulator needs remaking.
15. **Dry-run first, every session:** `--dry-run` streams video and telemetry with no motion. Hold the drone where it would fly and watch what it *would* send.

---

## 11. What this means for kite

### 11.1 Reuse map

| vigil piece | For kite | Notes |
|---|---|---|
| `drone/`, `sources/` | **already ported** | keep `SafetyGovernor` in front of everything |
| `types.py` (Detection, Pose2D, Obs, PerceptionResult) | port | keep the frame conventions (4.2) |
| `geometry.py` CameraModel | port | set fx from a calibration, not the 920 default |
| `detect/yolo.py` (or `rewrite/yolo`) | port | swap classes for street use: person, bicycle, car, bus, truck, traffic light, stop sign, bench, ... Baked weights; no runtime download |
| `track/tracker.py` + `lock.py` | port | the person lock should select the nearest by range |
| `person/estimator.py` | port | the walker's height to ±2 cm; facing is useful if the drone faces the walker |
| `perception.py` (strides, filters) | port, re-scope | kite modes are not search/approach |
| `behaviors/follow.py` | port and adapt | the core of person following; re-tune for walking speed and distance |
| `config.py` YAML loader + `validate()` | port | unknown keys raise |
| BoxFilter + PersonGovernor (fruitfly) | consider | latency-compensated box; slew limits |
| looming + FollowAvoid (patch) | consider, verify on real video first | forward-only, sim-validated |
| explore / approach / guide / walkpath | mostly skip | indoor room search; the walkpath and cue hysteresis ideas transfer |
| query parser, guidance phrasing | adapt | clock-face and direction words fit the earpiece's `guidance`/`say` |
| sim | skip | broken |

### 11.2 Design questions to settle early
- **Ahead or behind?**
  - **Behind (vigil's FollowBehind):** proven code, but the camera faces the walker's back, so it can't see what's ahead of them beyond the head. Street hazards ahead are visible over their head from 2 m up.
  - **Ahead, facing the walker:** the drone flies backwards into space it cannot see, because **the Tello has no rear sensing**. It sees the walker's face (facing estimates get easier), but not the path.
  - **Ahead, facing forward:** it sees the path and hazards, but cannot keep the walker in view.
  - A middle ground: fly beside or above, facing forward, and lean on the walker's webcam plus GPS/route position for where they are. Pick deliberately; it changes perception, control and safety.
- **Speed:**
  - walkers go 1.2-1.3 m/s;
  - the Tello's measured response is ~0.96 m/s per 100 stick at stick 30 (extrapolated);
  - kite's `SafetyCfg.max_rc` is 50 (~0.5 m/s) and vigil's follow capped forward at 35.
  - Measure rc 60 and 100 now; raise `max_rc` only after that.
- **Outdoors:**
  - wind; sunlight on the camera;
  - the downward vision/ToF hold needs textured, lit ground;
  - no GPS on the Tello: position for routing must come from the phone or wearable;
  - the battery (~6-8 min) means planned swaps;
  - the Wi-Fi range ties the drone laptop to the walker or a carrier.
- **Networking:**
  - the drone laptop is on the Tello Wi-Fi, which has no internet;
  - the cloud APIs (Maps, Claude, Gemini, Cartesia) and the SpacetimeDB sync need internet;
  - vigil used a **USB Wi-Fi dongle + phone hotspot** on the drone laptop. Test all links together in the first hour.
- **Frames:**
  - the earpiece speaks in the **wearer's** frame (0 = ahead, + right);
  - drone perception gives camera bearings and mission-frame positions;
  - `wearer_bearing = wrap(heading_of_object_in_world − wearer_heading)`, which needs the wearer's heading (from the walking direction or pose facing, as vigil's Guide did, or from the phone compass).
- **Obstacle split:**
  - the body webcam (close, sudden obstacles) feeds `obstacles` events → earpiece sonar (time-to-contact thresholds 4 s / 1 s);
  - the drone (route-level: crowds, crossings, signs) feeds `hazard` and `scene` events.
- **What goes in SpacetimeDB:**
  - still the template table;
  - define the shared tables (drone status, walker position, hazards, route) before both sides start writing.

### 11.3 Suggested next steps (none started yet)
1. **Commit the drone layer port** (`kite/`, `pyproject.toml`, tests, README) and the merged `.gitignore`. Local `main` is already in sync with `origin/main`.
2. **Install ultralytics and pre-download weights:** `yolo11n-pose.pt`, plus a street-class model if needed.
3. **Calibrate:**
   - run `tests/drone_test.py` at `--mag 30/60/100` and record dead times, gains and video delay;
   - camera fx with an object of known height at taped distances (`f = px_h × d / h`);
   - check the yaw and pitch signs.
4. **Port perception** (types, geometry, detector, tracker, lock, person estimator, perception) with tests.
5. **Port `FollowBehind`** with nearest-person locking and the lost-timer fix, then:
   - dry-run while holding the drone;
   - then fly in an open area with a spotter and prop guards.
6. **Bridge to the earpiece and dashboard:** drone status, hazards, and person position in the wearer's frame.

---

## 12. Where things are

| What | Path |
|---|---|
| kite repo | `~/Desktop/kite` (`origin` = `KingArthur207/kite`) |
| kite drone layer | `kite/drone/{base,tello,safety}.py` |
| kite video readers | `kite/sources/` |
| Tello lag test (flight) | `tests/drone_test.py` |
| API key template | `.env.example` (on `origin/main`) |
| Sync + dashboard | `dashboard/` (on `origin/main`) |
| Earpiece audio (WIP) | `earpiece/` (on `origin/feat/airpods-audio`) |
| vigil (newest) | `~/vigil` (`parthm667/vigil`) |
| vigil (older copy) | `~/Desktop/jerkgt13` (`nathanwuzhao/jerkgt13`) |
| vigil stack | `~/vigil/reachglass/` |
| vigil plan / demo runbook / site config | `~/vigil/PLAN.md`, `WORKFLOW.md`, `site.yaml` |
| Stand-alone YOLO module | `~/vigil/rewrite/yolo/` |
| Follow-avoid patch + docs | `~/vigil/integration/fruitfly-training/docs/integration/` |
| Fruit-fly runtime (BoxFilter, governor, geometry) | `~/vigil/integration/fruitfly-training/flyfollow/` |
| Measured Tello lag data | `~/vigil/integration/fruitfly-training/archive/rl-pipeline/data/lag_test/` |
| Tello tools | `~/vigil/tools/tello_latency_test.py`, `tello_video_diag.py` |
