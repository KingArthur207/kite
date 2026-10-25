# Kite notes 02: the Tello, djitellopy and the camera

Written 2026-10-24, about 7 hours into the 36 h hackathon. Arthur and Claude wrote it after studying the DJI
Tello, the [djitellopy](https://github.com/damiafuentes/djitellopy) library and the kite repo.

Each fact says where it came from:
- **source** means djitellopy's code (pip 2.5.0 / GitHub master);
- **spec / docs** means DJI's Tello SDK documentation as reflected in djitellopy;
- **kite** means this repo;
- **to measure** means we haven't checked it on our drone yet.

## 4. The Tello: how it works

### 4.1 Network
- **Network.** The standard Tello is always an access point (`TELLO-xxxxxx`), drone at `192.168.10.1`.
  It **cannot join another network**: `ap` (station mode) is EDU-only.
- **One controller.** Only one controller works at a time. **Close the Tello phone app and take the phone off
  the drone's Wi-Fi.** While the app is connected, the drone streams video to the phone and our code gets
  none.
- **UDP ports, all on the laptop side, plain-text protocol:**
  - **8889:** we send commands from it. The drone replies to the same port with `ok`, `error …`, a value
    (`87`, `801mm`) or `out of range`.
  - **8890:** we receive the state stream: one ASCII line per packet, `key:value;` pairs.
  - **11111:** we receive raw H.264 (Annex-B) after `streamon`. 960×720 at ~30 fps.
- **Range.** DJI's spec says ~100 m in open air; expect much less in practice. Keep the laptop close and away
  from other 2.4 GHz networks: video lag grows on a weak link.

### 4.2 Commands (text, UDP 8889)
**SDK mode and power**
- `command`: enter SDK mode (`connect()` sends it).
- `takeoff`: can take many seconds to answer `ok`. djitellopy allows 20 s.
- `land`.
- `emergency`: motors off immediately, the drone falls.

**Video:** `streamon` / `streamoff`.

**Discrete moves** (the drone flies them itself on its optical-flow hold and answers `ok` only when done):
- `forward|back|left|right|up|down <20–500 cm>`.
- `cw|ccw <1–360 °>`.
- `go x y z speed`: each of x, y, z in −500..500 cm, speed 10–100 cm/s. x, y and z can't all be inside −20..20.
- `curve x1 y1 z1 x2 y2 z2 speed`: speed 10–60, arc radius 0.5–10 m.
- `flip l|r|f|b`.
- `speed <10–100>`: speed for discrete moves.
- `stop`: hover now and cancel a running move. Pip djitellopy 2.5.0 has no `tello.stop()` method, so send the
  raw `"stop"` instead.

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
- **Height comes from the downward range sensor.** Flying over a table or bench, it will rise by that object's
  height (**to confirm**). Keep away from furniture.
- **Position hold needs light and a textured floor.** Glossy, uniform or dark floors make it drift. Outdoors,
  the 80 g airframe is pushed around by wind.

### 4.3 State stream (UDP 8890)

| Field | Meaning | Notes |
|---|---|---|
| `pitch`, `roll`, `yaw` | degrees, integers | check which way yaw grows with `cw` (**to measure**) |
| `vgx`, `vgy`, `vgz` | speeds, integers | commonly reported as **dm/s**, not cm/s; sign convention **to measure** |
| `h` | cm above the **takeoff point** | take off from the floor, so this is height above the floor |
| `tof` | cm to whatever is **below** | very small on the ground; a huge value (e.g. 6553) means out of range |
| `bat` | % | |
| `templ`, `temph` | °C | the drone warms up on the ground with video on |
| `baro` | m (float) | absolute, drifts |
| `agx`, `agy`, `agz` | acceleration, ≈ milli-g | |
| `time` | s the motors have run | |
| `mid`, `x`, `y`, `z`, `mpry` | mission pads | EDU only |

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
  - frames come out as **RGB** (via `frame.to_image()`), not the BGR OpenCV expects, so colours are swapped
    and colour checks silently fail;
  - frames have no timestamps and old ones are never dropped, so **latency grows over a flight**;
  - a decode error kills the thread silently;
  - `with_queue=True` gives a 32-frame deque.
- **`end()`, and `__del__` which calls it:**
  - **lands if `is_flying`**, so a garbage-collected `Tello` object lands the drone;
  - turns the stream off and removes the drone from `drones`.
- **pip vs GitHub.** pip 2.5.0 is from 2023-06. GitHub master adds `stop()`, `received_at` and a fix for
  changing the video port. Those 2024 changes were never released.

## 8. Camera and geometry

- **Stream.** 960×720 H.264, ~30 fps. Forward camera, **fixed to the body, no tilt**. When the drone pitches
  nose-down to accelerate, the image tilts with it, so correct vertical angles with telemetry pitch.
- **Focal length:**
  - DJI's 82.6° diagonal field of view is for still photos; the video stream is cropped;
  - published stream calibrations (tello_ros) give fx ≈ 920 px, i.e. ~55°×43°;
  - **measure ours**: an object of known height H at a taped distance d gives f = pixel_height · d / H.
- **Formulas:**
  - bearing = atan((x − 480) / fx), positive to the right;
  - elevation = −atan((y − 360) / fy);
  - range from a known height = fy · H / pixel_height (divide by cos(bearing) for off-axis objects);
  - ground distance of an image row from height h: d = h / tan(depression angle).
- **Pixel sizes at fx ≈ 920:**

  | Object | Height in pixels |
  |---|---|
  | 1.7 m person at 20 m | ~78 px (fine for person detection) |
  | 1.7 m person at 50 m | ~31 px |
  | 0.3 m pedestrian-signal head at 20 m | **~14 px** (reading its state from across a street is marginal) |

  Fly closer, or let the webcam and Maps data say where the crossing is.
- **The camera can't see the ground right under or in front of the drone**, which is another reason sudden
  obstacles belong to the webcam. From above and behind a person, the narrow vertical view means only part of
  the body may be in frame. Choose follow height and distance by testing.

## 10. Before every flight

1. **Install while you still have normal internet.** pip and npm downloads won't work once you're on the
   Tello's Wi-Fi.
2. Charged packs. Run `battery?` (or `--ground`) first: ≥ 50 % for a run, never fly below 30 % for tests.
3. **Phone app closed**, phone off the Tello Wi-Fi. The laptop joins `TELLO-xxxxxx`.
4. `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` prints nothing.
5. `python tests/drone_test.py --ground --preview`. Pass if:
   - state packets arrive steadily;
   - video is ~30 fps;
   - frame age stays low and doesn't grow.
6. **Space:**
   - at least 3 × 3 m for the latency test, more for following;
   - ceiling above 2.5 m;
   - textured floor;
   - take off **from the floor**.
7. Roles: operator at the laptop, spotter watching only the drone, the walker. Ctrl+C lands; `emergency` stops
   the motors (the drone falls).
8. **First run of any new behaviour in dry-run** (`make_tello(cfg, dry_run=True)`), with the drone held by hand.
