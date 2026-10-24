# Kite notes 02: the drone, the code we have, and what it means for the build

Written 2026-10-24, about 7 hours into the 36 h hackathon. Arthur and Claude wrote it after studying
[djitellopy](https://github.com/damiafuentes/djitellopy), the team's previous project
[vigil / ReachGlass](https://github.com/parthm667/vigil) and its saved flight logs, and the kite repo as it stood.

Each number says where it came from. **Measured** means vigil's real flight on 2026-09-26 (standard Tello, 87 %
battery, stick 30). **Spec / docs** means the SDK or a library docstring. **Estimate** means not verified on our
drone.

---

## 0. TL;DR

- The Tello is **its own Wi-Fi network**. The computer that flies it gets **no internet** on that adapter. It
  takes text commands on UDP 8889, sends state at ~10 Hz on UDP 8890 and sends H.264 video to UDP 11111.
- **It reacts slowly.** Forward/sideways motion starts about **0.5 s** after a stick command and takes another
  ~0.45 s to reach speed. Video adds ~0.25 s. Treat the drone as a look-ahead scout (crowds, crossings,
  route), not a reflex sensor. The body-worn webcam handles sudden obstacles.
- **Its top speed is about walking speed.** Stick 100 is about **1 m/s** forward, extrapolated from stick 30
  and never measured above it. People walk at 1.2–1.4 m/s. Measure stick 60 and 100 before planning around
  following at full walking speed.
- **Batteries are short.** One pack lasts about **6–8 min of flight** (measured ~9–11 %/min).
- **The camera is narrow and fixed.** It can't tilt and sees ~48°×37° (measured fx ≈ 1077 px at 960×720).
- **djitellopy has traps:**
  - replies aren't matched to the commands they answer;
  - timed-out commands retry 3×, so a move can be flown twice;
  - numpy values crash `send_rc_control`;
  - its frames are RGB;
  - its video reader lags more and more over a flight.

  Use kite's ported drone layer instead (§9).
- **Not pushed yet.** The drone layer is ported from vigil into `kite/` with 35 passing tests, but only in
  Arthur's local checkout. It isn't committed (§2).

---

## 1. The project

**Goal.** A DJI Tello follows a visually impaired person and gives directions through their AirPods:
- the shortest walking route;
- spotting people, crowds, crosswalks and signals, so the route avoids crowds and waits for the walk signal.

A webcam worn by the person runs object detection for sudden, close obstacles.

**Hardware.**

| Item | Notes |
|---|---|
| DJI Tello, **standard model (not EDU)** | The only drone we have. 80 g, prop guards. Has no GPS; holds position with a downward camera and a ToF (time-of-flight) range sensor. |
| Tello batteries | Each one lasts ~6–8 min of flight. Charging takes much longer than a flight, so rotate packs. |
| Laptops | Arthur's Mac (macOS 26, Apple Silicon). The dashboard README describes a **drone laptop** and a **wearable laptop**. |
| AirPods | Audio out. The earpiece module prefers the MacBook mic, because an open AirPods mic forces mono "call" quality. |
| Webcam | Worn by the person, plugged into the wearable laptop. |

**APIs planned** (`.env.example` on main, copy to `.env`, never commit it):
- **Google Maps:** Routes API v2 and Places API (New). The legacy Directions and Places APIs are *not* enabled on
  the key.
- **Anthropic (Claude):** route and scene reasoning.
- **Gemini:** describing the scene in the drone's video.
- **Cartesia:** text-to-speech in the AirPods (`sonic-3.6`) and speech-to-text (`ink-whisper`).
- **Moss:** route and landmark memory, only if we enter the Moss track.

---

## 2. Repository state (2026-10-24)

**On `origin/main`:**
- `README.md`, plus `tests/drone_test.py`: Arthur's Tello latency test (§6). On main it still imports a
  `video_stream` module that isn't in the repo. The import error is caught, so it **runs without video** and
  only prints "video unavailable". The local fix is not pushed yet.
- `dashboard/` (Karthik, PR #1). A SpacetimeDB module and a React + Vite dashboard; the hosted database is
  `kite-sync-pl6rp` on Maincloud. For now the module is the starter template (one `person` table and the
  `add` / `sayHello` reducers).
  - Purpose: the drone laptop, the wearable laptop and a caregiver dashboard all read and write one database,
    so they stay in sync "even though the two cameras are on different WiFi networks".
  - From Python, a reducer is a plain HTTP POST to
    `https://maincloud.spacetimedb.com/v1/database/kite-sync-pl6rp/call/<reducer>` with a bearer token
    (see `dashboard/README.md`).
- `.env.example` and `.gitignore` (ignores `.env` and `.env.*`).

**Branch `feat/airpods-audio`** (Karthik, work in progress). `earpiece/` holds the audio core: binaural
rendering (Brown–Duda spherical-head model), a real-time mixer with cue, alert and speech buses, synthesised
cue sounds, and the JSON event contract. Summary in §10.

**Arthur's local checkout, uncommitted** (done in this session, not pushed):
- `.venv/`: Python 3.12.14 (made with `uv`), djitellopy 2.5.0, PyAV 19.0.2, opencv-python 5.0, numpy 2.5, pytest.
- `pyproject.toml`: install with `pip install -e ".[dev]"`. Pytest only collects `test_*.py`, so
  `drone_test.py` (a flight script) is never run as a test.
- `kite/`: the drone layer and video readers ported from vigil (§9).
- `tests/test_drone.py` and `tests/test_sources.py`: 35 tests, all passing.
- `tests/drone_test.py`: now imports `kite.sources.video_stream` and stops before takeoff if it can't. Its
  docstring now shows the right run commands.
- `README.md` with setup steps, and a local `.gitignore`. That `.gitignore` will **conflict** with main's when
  pulling: merge the two by hand.

---

## 3. How the pieces talk (current plan, as far as the code shows)

```
   Tello  ──Wi-Fi "TELLO-xxxxxx" (no internet)──  DRONE LAPTOP
   192.168.10.1                                   UDP 8889 cmds / 8890 state / 11111 video
                                                  kite/ drone layer + perception on the drone video
                                                         │
                                       SpacetimeDB "kite-sync-pl6rp" (Maincloud, needs INTERNET)
                                                         │
   webcam ──USB──  WEARABLE LAPTOP  ── earpiece: UDP 5010 / WS 5011 on 127.0.0.1 ──► AirPods (Bluetooth)
                   obstacle detection, Maps routing, Claude / Gemini / Cartesia
                                                         │
                                               caregiver dashboard (browser)
```

Two things to settle early:
1. **Internet for the drone laptop.** Its Wi-Fi belongs to the Tello, so SpacetimeDB, Gemini and Claude are
   unreachable from it unless it gets a second link: iPhone USB tethering, Ethernet, or a USB Wi-Fi dongle
   (vigil used a dongle). Alternatively, the drone laptop only streams to the wearable laptop over a second
   link, and that laptop does the cloud calls.
2. **The earpiece binds to `127.0.0.1`** (`NetCfg.bind_host`). Only processes on the wearable laptop can send
   it UDP or WebSocket events. Drone events must come through SpacetimeDB, a relay on the wearable laptop, or a
   changed bind host.

---

## 4. The Tello: how it works

### 4.1 Network
- **Network.** The standard Tello is always an access point (`TELLO-xxxxxx`), drone at `192.168.10.1`,
  laptop gets an address like `192.168.10.2`. It **cannot join another network**: `ap` (station mode) is
  EDU-only.
- **One controller.** Only one controller works at a time. **Close the Tello phone app and take the phone off
  the drone's Wi-Fi.** While the app is connected, the drone streams video to the phone and our code gets
  none.
- **UDP ports, all on the laptop side, plain-text protocol:**
  - **8889:** we send commands from it. The drone replies to the same port with `ok`, `error …`, a value
    (`87`, `801mm`) or `out of range`.
  - **8890:** we receive the state stream: one ASCII line per packet, `key:value;` pairs, ~10 Hz.
  - **11111:** we receive raw H.264 (Annex-B) after `streamon`. 960×720 at ~30 fps.
- **Range.** Wi-Fi range is on the order of tens of metres in practice (spec claims ~100 m in open air).
  Stay close: video lag grows with a weak link (fruitfly checklist: keep the laptop within ~5 m, away from
  other 2.4 GHz networks).

### 4.2 Commands (text, UDP 8889)
**SDK mode and power**
- `command`: enter SDK mode (`connect()` sends it).
- `takeoff`: climbs to ~0.8–1.0 m. Measured: `ok` arrives **8.7 s** after sending.
- `land`: measured `ok` after 2.8 s.
- `emergency`: motors off immediately, the drone falls.

**Video:** `streamon` / `streamoff`.

**Discrete moves** (the drone flies them itself on its optical-flow hold and answers `ok` only when done):
- `forward|back|left|right|up|down <20–500 cm>`.
  - Measured: a 50 cm move answers after **2.25 s**, and motion starts ~0.2–0.25 s after the command.
- `cw|ccw <1–360 °>`.
- `go x y z speed`: each of x, y, z in −500..500 cm, speed 10–100 cm/s. x, y and z can't all be inside −20..20.
- `curve x1 y1 z1 x2 y2 z2 speed`: speed 10–60, arc radius 0.5–10 m.
- `flip l|r|f|b`.
- `speed <10–100>`: speed for discrete moves (vigil used 50).
- `stop`: hover now and cancel a running move. vigil relies on it. Pip djitellopy 2.5.0 has no `tello.stop()`
  method, so send the raw `"stop"` instead.
- Accuracy estimate from vigil's plan, unmeasured: ±10 cm per metre and ±3–5° per turn on a textured floor.

**Continuous control:** `rc lr fb ud yaw`, each −100..100. Positive means right / forward / up / clockwise.
- No reply. Send it at 10–20 Hz.
- An `rc` packet sent during a discrete move **cancels the move**.

**Read commands:** `battery?`, `speed?`, `time?`, `height?`, `temp?`, `attitude?`, `baro?`, `tof?` (`801mm`),
`wifi?` (SNR). The `get_*` equivalents that read the state stream are faster and send nothing.

**EDU / SDK 3.0 only, per the docs (expect `unknown command` on ours, never tried):**
- mission pads: `mon`, `moff`, `mdirection`, `go/curve/jump … m<id>`;
- `ap`;
- `port`, `setfps`, `setbitrate`, `setresolution`, `downvision` (downward camera);
- `motoron`, `motoroff`, `throwfly`, `keepalive`, `EXT`, `reboot`, `active?`.

**Behaviour to remember**
- **Auto-land 15 s after the last command it hears.** `rc` counts. Streaming `rc 0 0 0 0` is the keep-alive.
  If the laptop crashes, the drone lands by itself.
- **Commands sent closer than ~100 ms apart get dropped** (`TIME_BTW_COMMANDS = 0.1`).
- **Replies carry no ID.** You can't tell which command a reply answers.
- **Flying over a table makes it climb by the table height**, because the downward range sensor reads the
  tabletop. vigil treated tables as obstacles.
- **Position hold needs light and a textured floor.** Glossy, uniform or dark floors make it drift; bring a
  patterned mat for takeoff. Outdoors, the 80 g airframe is pushed around by wind.

### 4.3 State stream (UDP 8890), units as we actually see them

| Field | Unit / meaning | Notes |
|---|---|---|
| `pitch`, `roll`, `yaw` | degrees, integers | yaw grows with `cw` (vigil checks the sign on the first 20–135° turn) |
| `vgx`, `vgy`, `vgz` | **dm/s** (not cm/s), integers | in vigil's log, moving forward read as **negative** `vgx`. Steps of 0.1 m/s, so slow speeds round to 0 |
| `h` | cm above the **takeoff point**, in 10 cm steps | 0 on the ground, ~100 after takeoff |
| `tof` | cm to whatever is **below** | 10 on the ground, **6553 = out of range**, hover noise ~2.4 cm |
| `bat` | % | drains ~9–11 %/min in flight |
| `templ`, `temph` | °C | heats up on the ground with video on; fruitfly refused takeoff at `temph` ≥ 90 |
| `baro` | m (float) | absolute, drifts |
| `agx`, `agy`, `agz` | acceleration, ≈ milli-g | |
| `time` | s the motors have run | |
| `mid`, `x`, `y`, `z`, `mpry` | mission pads | EDU only |

---

## 5. djitellopy (pip 2.5.0) internals and traps

Source read: `djitellopy/tello.py` (1129 lines) and `enforce_types.py`.

**How it's built**
- **Shared sockets.** One module-level UDP socket is bound to port **8889** on all interfaces, plus a state
  socket on **8890**, each with a daemon thread. They're created by the first `Tello()` and kept for the life
  of the process. Only **one Tello program at a time**; otherwise you get `Errno 48 Address already in use`.
  Check with `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111`.
- **`drones` dict.** Keyed by drone IP. Packets from any other IP are ignored.
- **Replies.** Raw bytes are appended to `drones[ip]['responses']`, and each command pops the **oldest** one.
  **A late reply gets credited to the next command**: a late `ok` from a move makes `battery?` return "ok", and
  `int("ok")` crashes.
- **`send_command_with_return`:**
  - checks for the reply every **100 ms**, adding up to 100 ms of latency to every blocking command;
  - on timeout it *returns* an "Aborting command…" string instead of raising;
  - the minimum-gap code is buggy: it sleeps `diff` instead of `0.1 − diff`.
- **`send_control_command` retries 3×.** Any reply without "ok" counts as a failure, timeouts included. So:
  - a `forward 100` whose `ok` is late can be **flown twice**;
  - `takeoff()` (20 s timeout) can block for **60 s**;
  - `land()` (7 s timeout) can block for 21 s.
- **`@enforce_types`** checks every argument with `isinstance`. `send_rc_control(np.int64(30), …)` or a float
  raises a **TypeError mid-flight**. Cast with `int(round(v))`.
- **`send_rc_control`:**
  - clamps to ±100;
  - **silently drops** a call made within 1 ms of the previous one;
  - logs every packet at INFO. Set `Tello.LOGGER.setLevel(logging.WARNING)`.
- **State:**
  - the whole dict is replaced on each packet, and pip 2.5.0 doesn't timestamp it (master adds
    `received_at`). Time-stamp it yourself on first sight: a frozen dict means the receiver thread died;
  - `get_*()` raises `TelloException` until the first packet arrives;
  - `parse_state` turns known fields into int or float.
- **`connect()`** sends `command` (with the 3× retry), then waits only **1 s** for a state packet before raising
  "Did not receive a state packet". That error usually means the firewall is blocking inbound UDP 8890.
- **`get_frame_read()` → `BackgroundFrameRead`:**
  - opens `udp://@0.0.0.0:11111` with PyAV (5 s timeout) and decodes every frame;
  - frames come out as **RGB** (via `frame.to_image()`), not BGR. vigil's blue-bottle colour check silently
    failed because of this;
  - frames have no timestamps and old ones are never dropped, so **latency grows over a flight**;
  - a decode error kills the thread silently;
  - `with_queue=True` gives a 32-frame deque.
- **`end()`, and `__del__` which calls it:**
  - **lands if `is_flying`**, so a garbage-collected `Tello` object lands the drone;
  - turns the stream off and removes the drone from `drones`.
- **pip vs GitHub.** pip 2.5.0 is from 2023-06. GitHub master adds `stop()`, `received_at` and a fix for
  changing the video port. Those 2024 changes were never released.

**What kite does instead** (vigil's fixes, now in `kite/drone/tello.py`):
- sends everything with `send_command_without_return` and reads the reply list itself, never blocking;
- clears the reply list before each discrete command, and ignores replies for 1 s after `stop`;
- spaces sends at least 0.1 s apart;
- times out each command itself, never retrying;
- casts every `rc` value to a plain `int`;
- stamps state packets on arrival;
- reads video with its own newest-frame BGR readers.

---

## 6. Measured numbers (vigil, 2026-09-26, standard Tello, stick 30)

Sources:
- vigil's `integration/fruitfly-training/archive/rl-pipeline/data/lag_test/`: `drone_fulllogs_…json` (raw),
  `dronetestresults_…json`, `fit_…json` (dead time + first-order fit);
- the fruitfly docs (`drone_io_sim_r0.md`, `configs/env.yaml`).

`tests/drone_test.py --analyze` on the raw log reproduces the report.

| Quantity | Value |
|---|---|
| Read-command round trip (`battery?`, 1 ms polling) | **52 ± 2 ms** (48–55) |
| `takeoff` → `ok` / `land` → `ok` | **8.7 s** / 2.8 s |
| State stream | **9.8 Hz** (102 ms), jitter 18 ms, worst gap 299 ms |
| Video stream | **30.3 fps**, jitter 10 ms, worst gap 148 ms |
| **Yaw** | dead time 0.12–0.20 s, time constant ~0.02 s, 16.5°/s at stick 30, **~55°/s per 100**; ~0.3 s to stop |
| **Forward/back** | dead time **0.44–0.54 s**, time constant **0.39–0.47 s**, 0.29 m/s at 30, **~0.96 m/s per 100**; **~1.2 s** to drop below 10 % after rc 0 |
| **Sideways** | dead time 0.44–0.54 s, time constant 0.40–0.50 s, **~0.97 m/s per 100**; ~1.2 s to stop |
| **Vertical** | dead time 0.37–0.45 s, ~0.1 m/s at 30, **~0.32 m/s per 100** (±20 %: speed steps are 0.1 m/s) |
| Discrete `forward 50` | `ok` after 2.25 s; motion starts 0.2–0.25 s after the command |
| First visible change in video after a command (median) | yaw 0.39 s, vertical 0.61 s, sideways 0.82 s, forward 1.17 s (includes the motion dead time) |
| Video behind telemetry | ~0–60 ms for yaw and moves (worst ~85 ms) |
| Camera → decoded frame on the laptop | 0.15–0.45 s (fruitfly range); vigil used `video_lag_s: 0.25` |
| Battery | 87 → 67 % in a 135 s session; sim model 0.18 %/s, so **~6–8 min of flight per pack** |

**Caveats**
- Only **stick 30** was measured; everything "per 100" is linear extrapolation. The Tello's stick response may
  not be linear. Fruitfly's `flyfollow.tools.stick_response --send --axes yaw,fb --sticks 30,60,100` measures
  it in one ~3 min flight.
- The pitch/roll "lag" lines in `drone_test.py`'s report (~1.5–2.3 s) are a measurement artefact: it waits
  for a 2° angle change, and the angles are reported as whole degrees. Trust the fit (dead time ≈ 0.47 s).

**What this means for control.** From a stick command to seeing the result in video is ~0.7–1.2 s. vigil
coped by:
- keeping gains low and using deadbands;
- adding feed-forward (backing away at the walker's speed as soon as the measured range starts closing);
- using discrete moves plus a wait for the video to catch up before trusting frames
  (`video_lag_s` + 0.5–0.8 s of settle time).

---

## 7. Camera and geometry

- **Stream.** 960×720 H.264, ~30 fps. Forward camera, **fixed to the body, no tilt**. When the drone pitches
  nose-down to accelerate, the image tilts with it, so correct vertical angles with telemetry pitch (vigil's
  `pitch_sign` config).
- **Focal length:**
  - **measured on site by vigil: fx = fy ≈ 1077 px**, i.e. HFOV ≈ 48°, VFOV ≈ 37°;
  - published stream calibrations (tello_ros): fx ≈ 920 px, i.e. ~55°×43° (vigil's default);
  - the 82.6° diagonal spec is for stills and doesn't apply to the stream;
  - vigil's early plan guessed 680 px, which is wrong;
  - re-measure ours: an object of known height H at a taped distance d gives f = pixel_height · d / H.
- **Formulas:**
  - bearing = atan((x − 480) / fx), positive to the right;
  - elevation = −atan((y − 360) / fy);
  - range from a known height = fy · H / pixel_height (divide by cos(bearing) for off-axis objects);
  - ground distance of an image row from height h: d = h / tan(depression angle).
- **Pixel sizes at fx ≈ 1077:**

  | Object | Height in pixels |
  |---|---|
  | 1.7 m person at 20 m | ~92 px (fine for YOLO) |
  | 1.7 m person at 50 m | ~37 px |
  | 0.3 m pedestrian-signal head at 20 m | **~16 px** (reading its state from across a street is marginal) |

  Fly closer, or let the webcam and Maps data say where the crossing is.
- **Following from behind and above** (vigil): at 2.0 m high and 1.0 m behind, only the person's **head** is in
  frame (shoulders are 29° below the camera axis). Shoulders appear from ~1.6 m behind. vigil flew
  `follow: altitude 1.9 m, distance 1.8 m` and refused any altitude below the person's height + 0.2 m.
- **The camera can't see the ground right under or in front of the drone**, which is another reason sudden
  obstacles belong to the webcam.

---

## 8. vigil / ReachGlass (the team's previous Tello project)

**What it was**
- HackGT 13, Sept 2026, **3rd overall of 268** (demo: https://youtu.be/AqV3zwgGGPI).
- A standard Tello hovers behind and above the wearer. On "find my water bottle" it:
  - descends to 1.3 m;
  - explores (8 × 45° scans, scores directions, hops ≤ 1.5 m);
  - finds the bottle with YOLO-World ("blue water bottle" plus a blue-colour check);
  - flies to it;
  - guides the wearer there with spoken cues and haptic pads on glasses.
- It also ran "find arthur" with face ID (insightface).
- It ran on a **Windows** laptop.

**Architecture.** One 20 Hz loop, `reachglass/app.py`:
1. `sense()` reads the newest frame and telemetry and runs perception;
2. `SafetyGovernor.check()`;
3. inbox queries;
4. `mission.step()`, which runs the current behaviour;
5. the dashboard.

| Module | Role |
|---|---|
| `drone/` | `TelloDrone`, `SafetyGovernor`, `Drone` interface (**ported to kite**) |
| `sources/` | newest-frame video readers, webcam, file replay (**ported to kite**) |
| `types.py`, `geometry.py` | Frame/Detection/PersonObs/Pose2D, pinhole `CameraModel` (types partly ported) |
| `detect/`, `track/`, `person/`, `perception.py` | YOLO / YOLO-World / pose detectors, IoU tracker + target lock, person range (head elevation / height / shoulders / width fusion) and facing; detectors run every Nth frame depending on mode |
| `behaviors/` | `FollowBehind` (rc servo), `Scan`, `Explore`, `Approach`, `Guide`, `ReacquirePerson`, `Discrete` (one move + odometry update) |
| `mapping/` | odometry (IMU yaw + commanded moves), coverage/obstacle grid, semantic memory, `walkpath.py` (Dijkstra walking path) |
| `mission/` | state machine IDLE → TAKEOFF → FOLLOW → DESCEND → EXPLORE → APPROACH → ARRIVED → GUIDE → DONE/LANDED (+HOLD), guidance relative to the wearer |
| `voice/` | AirPods push-to-talk / hands-free (faster-whisper STT, edge-tts/SAPI) — **Windows only** (stem presses via Windows media session); UDP 5005 queries in, 5006 announcements out |
| `firmware/` | glasses: ESP32-CAM AP "rover" with MJPEG; Nano ESP32 at 192.168.4.50 UDP 4210 drives 2 ToF + 2 servo haptic pads (**sides crossed**: +1 = right presses the LEFT pad) |
| `integration/fruitfly-training/` | 1,446-neuron fruit-fly connectome yaw controller, RL/CMA training, `tello_io.py` runtime, R0 measurement tools (`stick_response`, `latency_test`, camera calibration), the lag-test data |
| `tools/` | `tello_latency_test.py` (= our `drone_test.py` minus the OpenCV reader), `tello_video_diag.py` (finds where the camera chain breaks: ports, firewall, link, state, raw packets, each decoder) |

**FollowBehind control law** (`behaviors/follow.py`, defaults):
- **Yaw:** 1.2 × bearing error, 4° deadband, max 40.
- **Forward:** 40 × (range − set distance), 0.1 m deadband, max 35 (~0.35 m/s, "a slow walk").
  - It only moves *toward* the person if even the most conservative range estimate is above 0.55 m and the
    person's box is narrower than 75 % of the frame.
  - Feed-forward: if the range is closing faster than 0.2 m/s (after subtracting our own commanded speed), it
    backs off at the person's speed, up to stick 50.
- **Altitude:** 60 × height error, 5 cm deadband, max 30.
- **Orbit-behind:** sideways motion from the person's facing. It was **switched off** for the demo
  (`orbit_gain 0`) because noisy facing estimates made the drone drift sideways.
- **Losing the person:**
  - keeps the last command for 0.35 s, then holds;
  - after 1 s, turns toward where the person was last seen (yaw 20), for up to 20 s;
  - if lost while close, backs off sideways and climbs, because the person is probably walking under the drone.

**Lessons learned** (from vigil's code, docs and a review of its follow/tracking code):
1. The person lock picks the **largest** box (`track/lock.py`). From above, a bystander 3 m away can beat the
   wearer's head-only box. The unmerged follow-avoid patch switched to nearest-by-range and a
   "not the person" list.
2. The mission's "person lost for 20 s" rule (`mission/mission.py`) fires within a few seconds of a loss once
   FOLLOW has run 20 s, not after 20 s: releasing the lock makes `person_unseen_s` infinite. This comes from a
   separate review of vigil and wasn't re-checked here.
3. Frames must be **BGR**; djitellopy's are RGB.
4. **Video reader:**
   - djitellopy's PyAV reader lags more and more;
   - the team's OpenCV reader uses FFmpeg's `nobuffer|low_delay|framedrop` and grabs and discards frames to
     stay current;
   - its rate cap must be **60**, because a cap of 30 drops ~40 % of frames from timing jitter;
   - open and read timeouts must be ~15 s, because the Tello sends keyframes seconds apart and nothing decodes
     before one;
   - never release the capture while `grab()` is blocked, or the process segfaults.
5. **`streamon` sometimes needs re-sending** before video packets arrive. "Answers commands but no video" almost
   always means the phone app or the firewall.
6. **Land replies get lost.** Re-send `land` every 2 s until it's down. A lost or `error` reply with `h` ≤ 0.2 m
   counts as landed. Never `stop` during takeoff or landing: the drone would hover while the code thinks it
   landed.
7. **Take off from the floor.** The altitude limits use `h`, which is measured from the takeoff surface.
8. **Fruit-fly steering:** only yaw is trained. `fly.forward` must stay false: the untrained forward control
   held 8–10 m and surged in flight. The plain PID steers as well.
9. **Looming detection** (optic-flow expansion) is blind on blank walls and dark rooms and over-reads on door
   frames. It was only validated in simulation. Obstacle avoidance while following exists only as an unmerged
   patch (`integration/fruitfly-training/docs/integration/reachglass_follow_avoid.patch` + `FOLLOW_AVOID.md`).
10. **Simulator and tests:** vigil's sim is self-declared broken, and 3 of its tests were already failing at the
   end (follow distance, full mission, floor bottle).
11. **Ports on a shared laptop:** on one team Mac an unrelated process (`gs-stand`) held UDP 8889. Always run
    `lsof` before a flight.

---

## 9. kite's drone layer (ported from vigil, local and uncommitted on 2026-10-24)

```
kite/
  config.py            DroneCfg, SafetyCfg, Config (plain dataclasses, no YAML loader yet)
  types.py             Frame (BGR, read-only, t = arrival time, seq), Telemetry (SI units), wrap_deg
  drone/base.py        Drone interface, check_move (20-500 cm), check_rotate (1-360), clamp_rc
  drone/tello.py       TelloDrone (vigil code; logger "kite.tello", macOS hint in the no-video error)
  drone/safety.py      SafetyGovernor (vigil code, unchanged)
  drone/__init__.py    make_tello(cfg, dry_run=False)
  sources/             FrameSource; TelloVideoSource (backend "pyav" default or "opencv"), WebcamSource,
                       VideoFileSource, SteppedVideoSource, StaticImageSource; video_stream.VideoStream
                       (the team's OpenCV reader), pyav_stream.PyAVStream
```

**Usage**
```python
from kite.config import Config
from kite.drone import SafetyGovernor, make_tello

cfg = Config()
drone = SafetyGovernor(make_tello(cfg, dry_run=True), cfg.safety)  # dry_run: never sends motion
drone.connect()                     # 'command', 'speed 50', streamon (re-sent until packets arrive)
src = drone.frame_source()          # newest-frame BGR reader; src.read() -> Frame | None
# every loop (10-20 Hz):
#   f = src.read(); action = drone.check(time.time(), f.t if f else None)   # 'ok' | 'hover' | 'land'
#   drone.rc(lr, fb, ud, yaw)       # ignored while a discrete command runs, or while landing / hovering for safety
#   drone.move("forward", 100); then poll drone.busy() / drone.last_result()  ('ok' | 'error: ...' | 'error: timeout')
#   drone.rotate(-90) (+ = cw); drone.stop(); drone.land(); drone.emergency()
#   drone.telemetry() -> Telemetry(yaw_deg, pitch_deg, roll_deg, height_m, tof_m, vx, vy, vz, battery_pct, ...)
drone.close()                       # lands if flying, stops video, ends djitellopy
```

**`TelloDrone` behaviour**
- **Discrete commands never block.** A second discrete command while one is running raises `RuntimeError`.
  That's a programming error: check `busy()` first.
- **Per-command timeouts:**

  | Command | Timeout |
  |---|---|
  | `takeoff` | 20 s |
  | `land` | 15 s |
  | move | cm / speed + 7.5 s |
  | rotate | deg / 60 + 6 s |

- **After `stop()`,** the next discrete command waits out a 1 s settle window, so late replies aren't credited
  to it.
- **`land()`:**
  - calling it twice does nothing the second time;
  - during takeoff it lands straight away;
  - during a move it sends `stop` first;
  - `stop()` never interrupts a takeoff or a landing.
- **Telemetry:**
  - speeds = dm/s × 0.1, giving m/s;
  - `tof` outside 0.1–8 m becomes `None`;
  - yaw is multiplied by `yaw_sign`;
  - `t` is when the packet was first seen, so a frozen state stream shows up as stale.
- **Dry run:** connects, streams video and reads telemetry, but logs motion commands instead of sending them.
  They "complete" after the time they'd normally take.

**`SafetyGovernor` defaults** (`SafetyCfg`):
- **Sticks:** every rc channel clamped to **±50** (~0.5 m/s forward). **Raise it to follow at walking speed.**
- **Altitude band 0.5–2.3 m:**
  - climbing slows to zero over the 0.4 m below the ceiling;
  - above the ceiling it's pushed down at stick −20;
  - descending is cut below the floor;
  - `up` / `down` moves that would cross a limit are refused with `error: safety (...)`.
- **Battery and flight time:** lands below 20 % battery or after 420 s of flight, re-sending `land` every 2 s if
  needed.
- **Stale data:** hovers (rc 0) if no new frame for 1 s, or no new telemetry for 1 s.
- `DroneCfg` defaults: `move_speed_cm_s 50`, `video_fps 60`, `video_backend "pyav"`, `video_lag_s 0.25`.

**Setup and tests**
```bash
uv venv --python 3.12 --seed .venv && source .venv/bin/activate   # or python3.12 -m venv .venv
pip install -e ".[dev]"
python -m pytest -m "not slow"      # 31 tests, ~13 s, no drone
python -m pytest                    # + 4 slow tests (ffmpeg as a fake Tello), ~75 s more
```
- **`tests/test_drone.py` (24 tests):**
  - `TelloDrone` against a fake djitellopy whose replies arrive asynchronously;
  - `SafetyGovernor` against a fake drone;
  - includes a check that `rc` values reach djitellopy as plain `int`.
- **`tests/test_sources.py` (11 tests):** file sources and read-only frames. Four slow tests run ffmpeg as a
  stand-in Tello, streaming H.264 at 960×720 / 30 fps over local UDP, with djitellopy imported first as in the
  real app. Results, for both the PyAV and OpenCV backends:
  - ~31 fps delivered;
  - ~15 ms frame age when read;
  - recovers after a 5 s outage without reopening the stream;
  - `stop()` takes ~2 s while the stream is stalled;
  - waits correctly for a stream that starts late.
- **Known noise:** every run prints macOS `objc: Class AVF… is implemented in both …cv2… and …av…` warnings,
  because OpenCV 5.0 and PyAV each bundle their own FFmpeg. Nothing broke in the tests. If an app crashes with
  a webcam open, suspect this first.
- **Not done yet:** nothing in kite has run against the real drone. First check:
  `python tests/drone_test.py --ground --preview` on the Tello's Wi-Fi.

**`tests/drone_test.py` (the flight script)**
- **Ground mode (`--ground`):**
  - 10 × `battery?` round trips;
  - state and video stream rates;
  - optional live preview window (`--preview`) showing frame age.
- **Flight (~2 min, 3 × 3 m space, battery ≥ 30 %):**
  - takes off, hovers;
  - for yaw, throttle, pitch and roll: `--trials` × (+mag / −mag) rc steps (default mag 30 for 1.5 s, settle
    2 s);
  - then `forward/back 50`.
- **Output:** for each step, lag, speed-up time, stop time, video onset and video delay.
- **Saved to** `results/tello_latency_<stamp>.json` and `…_summary.json`; `--analyze <json>` re-runs the
  analysis.
- **Argument limits** (`check_args`): mag 10–100, duration 0.5–3 s, mag × duration ≤ 90, forward 20–500 cm.
- **Ctrl+C lands,** retried 4 times; if it never lands, the 15 s auto-land takes over.

---

## 10. Earpiece event contract (branch `feat/airpods-audio`, WIP, Karthik)

**Transport:** JSON on **UDP 5010** (fire-and-forget) or **WebSocket 5011** (both ways), bound to `127.0.0.1`.
Optional SpacetimeDB link (`spacetime_enabled: false` by default).

**Conventions:**
- bearings are degrees in the **wearer's** frame: 0 ahead, +90 right, −90 left, ±180 behind;
- distances in metres;
- every message is validated and clamped, so a bad sender can't crash the audio loop.

| type | example | drone-side use |
|---|---|---|
| `obstacles` | `{"type":"obstacles","source":"webcam","seq":12,"objects":[{"id":"7","kind":"person","bearing":-20,"distance":2.4,"closing_speed":0.4,"confidence":0.8}]}`: a full frame (ids not listed are gone) | the webcam's job. The drone could send far-away people/crowds, but bearings must be converted to the wearer's frame |
| `guidance` | `{"type":"guidance","text":"In 20 meters, turn left onto Lyon Street","key":"step-3","ttl":8}` | route instructions (Maps) |
| `beacon` | `{"type":"beacon","bearing":-35,"distance":42,"label":"Palace of Fine Arts"}` / `{"type":"beacon","active":false}` | direction to the next waypoint (Soundscape-style) |
| `hazard` | `{"type":"hazard","kind":"crosswalk","state":"dont_walk","bearing":5,"distance":9}` | **main drone output**: crosswalk signal state, crowds, construction ahead |
| `say` | `{"type":"say","text":"...","priority":"info"}` | free text |
| `scene` | `{"type":"scene","text":"Wide sidewalk..."}` (not spoken; context for questions) | Gemini's description of the drone video |
| `position` | `{"type":"position","lat":37.80,"lng":-122.45,"heading":270,"place":"Lyon St"}` | phone/GPS position |
| `status` | `{"type":"status","source":"drone","battery":34,"state":"following"}` | drone battery + state |
| `heartbeat` | `{"type":"heartbeat","source":"webcam"}` | watchdog (no obstacle frames for 1.5 s = "detection is down") |
| `control` | `mute_cues`, `unmute_cues`, `pause`, `resume`, `volume_up`, `volume_down`, `repeat`, `beacon_on`, `beacon_off`, `stop_speaking` | |

- **Priorities**, most important first: `critical`, `hazard`, `nav`, `answer`, `info`, `system`.
- **Kind aliases:** `pedestrian`/`people` → `person`; `car`/`truck`/`bus` → `vehicle`; `bicycle`/`scooter` →
  `bike`; `stairs` → `stairs_down`; `step`/`drop` → `curb`.
- **Spoken directions:** "ahead" within 12°, "slightly left/right" up to 35°, "on your left/right" at 75–105°,
  clock positions in between, "behind you" past 160°.
- **Proximity sonar:**
  - assumes 1.2 m/s walking speed and a 0.55 m half-width walking corridor;
  - starts beeping at 4 s to contact; critical at 1 s or 0.75 m;
  - at most 2 sounds at once;
  - each object type has its own sound: person = knock, vehicle = buzz, obstacle = pip, drop = falling glide.
- **Speech:**
  - Cartesia voice "Daniel" at 1.1× speed;
  - falls back to macOS `say` ("Samantha") without internet;
  - navigation speech older than its `ttl` is dropped, never spoken late.
- **Microphone:** wake words "kite" / "hey kite". It uses the system default mic; the config recommends the
  MacBook mic, because an open AirPods mic drops them into mono "call" quality.

---

## 11. Risks for kite and what to do about them

| Risk | Why | Mitigation |
|---|---|---|
| Drone can't keep up with a walk | ~1 m/s at stick 100 (extrapolated), vigil capped forward at 35 and the governor at 50 | measure sticks 60 and 100 (`stick_response`); raise `SafetyCfg.max_rc`; fly ahead/above along the planned route rather than chasing; have the walker go slowly in the demo |
| Control loop lag ~0.7–1.2 s | dead time 0.5 s + rise 0.45 s + video 0.25 s | low gains, deadbands, feed-forward on range rate; drone = look-ahead, webcam = reflexes |
| No internet on the drone laptop | Tello AP only | iPhone USB tethering / Ethernet / USB Wi-Fi dongle on the drone laptop, or route everything through the wearable laptop |
| Earpiece only listens on localhost | `bind_host 127.0.0.1` | send drone events via SpacetimeDB or a relay on the wearable laptop, or change the bind host |
| Outdoors | 80 g, no GPS, downward position hold needs texture + light, wind | test outside early; have an indoor / hallway / campus-path fallback demo |
| Battery | 6–8 min per pack | short scripted demo runs, a battery log, start each run ≥ 50 %, land at 20–30 % |
| Wi-Fi range / interference | 2.4 GHz, lag grows with distance | laptop near the drone (carried by the walker or a teammate), few nearby networks |
| Narrow fixed camera | ~48°×37°, no tilt | yaw to scan; distant signal heads are only ~16 px at 20 m |
| djitellopy traps | §5 | always go through `kite.drone` (never raw djitellopy calls in app code) |
| macOS networking | firewall + Local Network privacy | allow Python incoming connections; if sends fail with `Errno 65 No route to host`, allow the terminal / IDE under System Settings → Privacy & Security → Local Network |
| Port clashes | one process per port | `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` before every run; quit other Tello tools |
| Safety around people and roads | a drone near a blind walker, bystanders, traffic | prop guards, a spotter who calls "LAND", stay away from road traffic, keep hard altitude/distance limits, emergency key ready |

---

## 12. Before every flight

1. **Install while you still have normal internet.** pip and npm downloads won't work once you're on the
   Tello's Wi-Fi.
2. Charged packs. Run `battery?` (or `--ground`) first: ≥ 50 % for a run, never fly below 30 % for tests.
3. **Phone app closed**, phone off the Tello Wi-Fi. The laptop joins `TELLO-xxxxxx`.
4. `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` prints nothing.
5. `python tests/drone_test.py --ground --preview`. Pass if:
   - state is ~10 Hz;
   - video is ~30 fps;
   - frame age stays low and doesn't grow.
6. **Space:**
   - at least 3 × 3 m for the latency test, 6 × 4 m for following;
   - ceiling above 2.5 m;
   - textured floor;
   - take off **from the floor**.
7. Roles: operator at the laptop, spotter watching only the drone, the walker. Ctrl+C / `l` lands, `e` is the
   emergency motor stop (the drone falls).
8. **First run of any new behaviour in dry-run** (`make_tello(cfg, dry_run=True)`), with the drone held by hand.

---

## 13. Open questions and next steps

- [ ] Commit and push the local drone layer (`kite/`, `pyproject.toml`, tests, README, merged `.gitignore`).
- [ ] First real check on our drone: `drone_test.py --ground`, then a flight run. Update `video_lag_s` and
      confirm yaw and `vgx` signs.
- [ ] Measure stick 60 and 100 forward/yaw response (fruitfly `stick_response`) to see whether following at
      walking speed is possible.
- [ ] Calibrate fx on our Tello (known object at 1.5 / 2 / 3 m).
- [ ] Decide the network layout: which laptop flies, how it gets internet, how drone events reach the earpiece.
- [ ] Define drone → earpiece messages: `hazard` (crosswalk / crowd), `status`, `scene`. Bearings must be in
      the wearer's frame, so the drone needs the wearer's position relative to itself (person detection +
      range from vigil's person estimator).
- [ ] Port what we need from vigil's perception: YOLO person detection, tracker + lock (fix the
      largest-box choice), person range and facing, `geometry.py` (`CameraModel`), and `FollowBehind`.
- [ ] SpacetimeDB schema for kite (replace the starter `person` table): drone status, hazards, route, wearer
      position.
- [ ] Outdoor test of position hold and wind tolerance at the demo site.
- [ ] Battery plan for the demo (packs, charging rotation).
