"""
Tello response-lag test: one flight, djitellopy, everything timestamped.

Measures (each reported as mean ± std [min … max] over repeats):
  command delay   round trip of a read command ('battery?'), 10 repeats on the ground
  takeoff / land  time until the drone answers 'ok'
  state / video   packet rate, frame rate, jitter, worst gap
  yaw, throttle, pitch, roll   (rc steps in both directions, --trials repeats each)
      lag          command -> first reaction in the telemetry (yaw/pitch/roll angle, vertical speed)
      speed-up     first reaction -> 90 % of peak speed
      stop         rc 0 -> speed back under 10 % of peak
      video        command -> first video frame that shows motion
      video delay  video - lag   (how much later the camera shows what the drone already did)
  forward/back N cm   discrete moves: time to 'ok', lag, speed-up, video

Install:  see README.md (the venv with `pip install -e ".[dev]"` provides djitellopy, OpenCV and kite)
Run:      python tests/drone_test.py                        # full flight (~2 min)
          python tests/drone_test.py --preview              # same, with a live camera window
          python tests/drone_test.py --ground --preview     # no takeoff: command delay, stream rates, camera check
          python tests/drone_test.py --analyze results/tello_latency_<stamp>.json
          python tests/drone_test.py --ground --preview --video-backend opencv   # another video reader

Video goes through one of kite's newest-frame readers (--video-backend): pyav = PyAVStream (FFmpeg udp:// + h264
demuxer; the default), opencv = the team's VideoStream (OpenCV/FFmpeg, nobuffer + low_delay), raw =
kite.sources.tello_raw.RawTelloStream (its own UDP socket + PyAV's bare H.264 decoder), auto = raw on macOS. Each
decodes every frame but converts at most --decode-fps per second, so latency does not grow over the flight. The
recorder and the preview window share that one reader. --video-bitrate / --video-resolution / --video-fps set the
drone's stream (Tello SDK setbitrate / setresolution / setfps) before streamon; nothing is sent unless given.
Safety:   3 x 3 m clear space, prop guards, battery >= 30 %. Ctrl+C lands.

The state stream is ~10 Hz, so one measurement is quantised to ~100 ms; the mean over repeats is better
than that, and the std includes the quantisation.
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the kite package next to this file

DROP_CORRUPT = False  # --drop-corrupt (set in main)
FIELDS = ["pitch", "roll", "yaw", "vgx", "vgy", "vgz", "tof", "h", "bat"]

# axis: rc vector (left_right, forward_back, up_down, yaw), reaction channel, its minimum threshold,
#       speed channels (first one that responds is used)
AXES = {
    "yaw":      ((0, 0, 0, 1), "yaw",   2.0, ["yaw_rate"]),
    "throttle": ((0, 0, 1, 0), "vgz",   1.0, ["vgz", "tof_rate"]),
    "pitch":    ((0, 1, 0, 0), "pitch", 2.0, ["vgx"]),
    "roll":     ((1, 0, 0, 0), "roll",  2.0, ["vgy"]),
}

# ------------------------------------------------------------------ recording (background threads)
def log_state(tello, rec, stop):
    """Store every new state packet with the time we first saw it.

    djitellopy replaces the whole state dict per packet and does not timestamp it, so we poll at
    ~500 Hz and stamp on arrival: <= 2 ms of jitter, well under the ~100 ms state period.
    """
    last = None
    while not stop.is_set():
        st = tello.get_current_state()
        if st and st is not last:
            last = st
            rec["state"].append([time.time()] + [float(st.get(k, "nan")) for k in FIELDS])
        time.sleep(0.002)


def probe_video_packets(tello, seconds=3.0):
    """Count raw UDP packets on the video port after one more streamon, without decoding.

    Tells "the drone sends nothing" (0 packets) from "packets arrive but OpenCV can't decode them".
    Returns (packets, bytes), or (None, reason) if the port can't be bound.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", tello.vs_udp_port))
    except OSError as e:
        sock.close()
        return None, f"UDP {tello.vs_udp_port} is held by another program ({e})"
    sock.settimeout(0.2)
    n = size = 0
    try:
        tello.send_command_without_return("streamon")
        end = time.time() + seconds
        while time.time() < end:
            try:
                size += len(sock.recv(65536))
                n += 1
            except socket.timeout:
                pass
    finally:
        sock.close()
    return n, size


def open_video(tello, fps, backend="raw", tries=3):
    """streamon + the chosen reader, re-sending streamon between tries: the Tello often ignores the first one.

    On failure, says what is wrong (no packets, or packets that do not decode); returns None.
    """
    from kite.sources.opencv_sources import stream_class

    Stream = stream_class(backend)
    if backend == "raw":
        return open_raw_video(tello, Stream, fps, tries)
    for i in range(tries):
        try:
            tello.streamon()
        except Exception as e:  # an 'error' reply: try again
            print(f"video try {i + 1}/{tries}: streamon -> {e}")
            continue
        try:
            return Stream(tello.get_udp_video_address(), fps=fps).start()
        except RuntimeError as e:
            print(f"video try {i + 1}/{tries}: {e}")
    n, size = probe_video_packets(tello)
    if n is None:
        print(f"video: {size}. Find it: lsof -nP -iUDP:{tello.vs_udp_port} (macOS/Linux), "
              f"netstat -ano -p udp | findstr {tello.vs_udp_port} (Windows)")
    elif n:
        print(f"video: the drone IS sending ({n} packets, {size // 1024} KB in 3 s), but the {backend} reader could "
              "not decode it. Try --video-backend raw, or record the packets: python -m kite.video_doctor")
    else:
        print(f"video: no packets reached UDP {tello.vs_udp_port} in 3 s: the drone is not streaming. "
              "Close the Tello phone app, power-cycle the drone, rejoin its Wi-Fi and try again.")
    return None


def open_raw_video(tello, Stream, fps, tries=3, wait_s=5.0):
    """kite's own UDP reader: bind first (its constructor never waits), then streamon until a frame decodes.
    streamon is re-sent only while no packet arrives; once packets flow it waits for a keyframe (wait_s per try)."""
    try:
        stream = Stream(tello.get_udp_video_address(), fps=fps, drop_corrupt=DROP_CORRUPT).start()
    except RuntimeError as e:  # the port is held by another program
        print(f"video: {e}")
        return None
    for i in range(tries):
        if i == 0 or not stream.stats()["datagrams"]:  # re-sent only while nothing arrives
            try:
                tello.streamon()
            except Exception as e:  # an 'error' reply: try again
                print(f"video try {i + 1}/{tries}: streamon -> {e}")
                continue
        t_end = time.time() + wait_s
        while time.time() < t_end:
            if stream.read_seq(copy=False)[1] is not None:
                st = stream.stats()
                print(f"video: raw reader, first frame {st['time_to_first_frame_s']} s after start, {st['size']}, "
                      f"SO_RCVBUF {st['rcvbuf']} bytes")
                return stream
            time.sleep(0.05)
        st = stream.stats()
        print(f"video try {i + 1}/{tries}: no frame yet ({st['datagrams']} packets, {st['bytes'] // 1024} KB, "
              f"decode errors {st['decode_errors']}, {st['skipped_bytes']} bytes skipped before a keyframe)")
    st = stream.stats()
    stream.stop()
    if st["datagrams"]:
        print(f"video: the drone IS sending ({st['datagrams']} packets) but no frame decoded: {st}. Record the "
              "packets for analysis: python -m kite.video_doctor")
    else:
        print(f"video: no packets reached UDP {tello.vs_udp_port}: the drone is not streaming. "
              "Close the Tello phone app, power-cycle the drone, rejoin its Wi-Fi and try again.")
    return None


def log_video(stream, rec, stop):
    """Store every new frame's decode time and how much it differs from the previous frame."""
    import cv2

    prev, last_seq = None, -1
    while not stop.is_set():
        try:
            seq, img, ts = stream.read_seq(copy=False)  # read-only, no memcpy
        except Exception as e:  # never let the recorder die unnoticed mid-flight
            print(f"video recorder stopped: {e}")
            return
        if img is None or seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq = seq
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            rec["video"].append([ts, float(np.mean(np.abs(small - prev)))])
        prev = small


def preview_loop(ui, done, scale):
    """Show the live feed in a window until the run ends. Runs on the MAIN thread: macOS only lets the
    main thread create windows (from any other thread cv2.namedWindow fails with "Unknown C++ exception").

    The flight runs in its own thread and hands over its stream through `ui` (ui["video"] is set once
    the video is open or has failed). Closing the window or pressing q only ends the preview; landing
    is still Ctrl+C in the terminal.
    """
    import cv2

    while not ui["video"].wait(0.1):  # stay interruptible: Ctrl+C must reach the main thread
        if done.is_set():
            return
    stream, hud = ui.get("stream"), ui["hud"]
    if stream is None:
        return
    win = "Tello live  -  q closes preview, Ctrl+C in terminal lands"
    last = -1
    stats0 = stream.stats() if hasattr(stream, "stats") else {}
    has_stats = "datagrams" in stats0  # the raw reader's counters (format_rates reads its keys)
    prev, prev_t, last_print, counters = stats0, time.monotonic(), 0.0, ""
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)  # in the try: a HighGUI error closes the preview, never the flight
        while not done.is_set():
            now = time.monotonic()
            if has_stats and now - prev_t >= 1.0:  # the reader's own counters, once a second
                from kite.sources.tello_raw import format_rates, rates

                cur = stream.stats()
                counters = format_rates(rates(prev, cur, now - prev_t))
                prev, prev_t = cur, now
                if now - last_print >= 2.0:
                    print(f"  video: {counters}", flush=True)
                    last_print = now
            seq, img, ts = stream.read_seq(copy=True)  # we draw on it, so take a copy
            if img is None or seq == last:
                cv2.waitKey(1)
                # closed with X: checked here too, before the next imshow (which would reopen it on Windows)
                if last != -1 and cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                time.sleep(0.005)
                continue
            last = seq
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            label = f"{hud.get('phase', '')}   frame age {1000 * (time.time() - ts):.0f} ms"
            for y, text in ((22, label), (44, counters)):
                if text:
                    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except Exception as e:
        print(f"preview closed: {e}")
    finally:
        try:
            cv2.destroyWindow(win)
            cv2.waitKey(1)
        except Exception:
            pass
    print("preview window closed (flight continues)")


# ------------------------------------------------------------------ drone commands
def timed_cmd(tello, cmd, timeout=10.0):
    """Send a command, return (reply, t_sent, t_reply). Polls every 1 ms (djitellopy polls every 100 ms)."""
    replies = tello.get_own_udp_object()["responses"]
    time.sleep(0.1)  # the Tello drops commands sent too close together
    replies.clear()
    t0 = time.time()
    tello.send_command_without_return(cmd)
    while not replies:
        if time.time() - t0 > timeout:
            return None, t0, time.time()
        time.sleep(0.001)
    t1 = time.time()
    return replies.pop(0).decode(errors="ignore").strip(), t0, t1


def hold(tello, rc, seconds):
    """Re-send one rc command at 20 Hz for `seconds` (also keeps the 15 s auto-land from firing)."""
    t_end = time.time() + seconds
    while True:
        tello.send_rc_control(*rc)
        if time.time() >= t_end:
            return
        time.sleep(0.05)


def rc_step(tello, rec, axis, sign, mag, dur, settle):
    rc = [sign * mag * c for c in AXES[axis][0]]
    ev = {"type": "rc", "axis": axis, "sign": sign, "t_cmd": time.time()}
    hold(tello, rc, dur)
    ev["t_release"] = time.time()
    rec["events"].append(ev)
    hold(tello, [0, 0, 0, 0], settle)


def safe_land(tello, rec):
    """Land, retrying until the drone confirms. Ctrl+C during this does not abort it.

    On failure is_flying is left set so tello.end() makes its own attempt; the Tello also
    auto-lands ~15 s after the last command it hears.
    """
    for attempt in range(4):
        try:
            tello.send_rc_control(0, 0, 0, 0)
            reply, t0, t1 = timed_cmd(tello, "land", timeout=8)
            if reply and reply.lower().startswith("ok"):
                rec["meta"].setdefault("land", t1 - t0)
                tello.is_flying = False
                print("landed")
                return True
            print(f"land attempt {attempt + 1}/4: reply {reply!r} - retrying")
        except KeyboardInterrupt:
            print("Ctrl+C ignored - still landing")
    print("!! LAND NOT CONFIRMED - drone may still be airborne; it auto-lands ~15 s after the last command")
    return False


def move(tello, rec, cmd, settle):
    reply, t0, t1 = timed_cmd(tello, cmd, timeout=10)
    rec["events"].append({"type": "move", "cmd": cmd, "axis": "pitch", "t_cmd": t0, "t_release": t1, "reply": reply})
    hold(tello, [0, 0, 0, 0], settle)


# ------------------------------------------------------------------ analysis
def build_series(rec):
    S = np.array(rec["state"], float)
    d = {"t": S[:, 0]}
    for i, k in enumerate(FIELDS):
        d[k] = S[:, i + 1]
    d["yaw"] = np.degrees(np.unwrap(np.radians(d["yaw"])))
    d["yaw_rate"] = np.gradient(d["yaw"], d["t"])
    d["tof_rate"] = np.gradient(d["tof"], d["t"])
    return d


def first_time(t, mask):
    i = np.flatnonzero(mask)
    return float(t[i[0]]) if i.size else None


def analyze_step(d, noise, video, v_thr, ev, t_next):
    _, react, min_thr, speeds = AXES[ev["axis"]]
    t, t_cmd, t_rel = d["t"], ev["t_cmd"], ev["t_release"]
    pre = (t >= t_cmd - 0.5) & (t < t_cmd)
    after = (t >= t_cmd) & (t < t_next)
    out = {"axis": ev["axis"], "cmd": ev.get("cmd", f"{ev['axis']} {'+' if ev.get('sign', 1) > 0 else '-'}")}

    # lag: first telemetry sample that moved away from the pre-command value
    base = np.median(d[react][pre]) if pre.any() else d[react][after][0]
    t_react = first_time(t, after & (np.abs(d[react] - base) >= max(min_thr, 3 * noise.get(react, 0))))
    out["lag"] = None if t_react is None else t_react - t_cmd

    # speed-up and stop, on the first speed channel that really moved
    for ch in speeds:
        v = d[ch]
        dev = v - (np.median(v[pre]) if pre.any() else 0.0)
        win = (t >= t_cmd) & (t <= t_rel + 0.5)
        if win.sum() < 3:
            break
        peak = float(np.median(np.sort(np.abs(dev[win]))[-3:]))
        if peak < max(3 * noise.get(ch, 0), 1e-6):
            continue
        sign = np.sign(dev[win][np.argmax(np.abs(dev[win]))])
        r = dev * sign / peak
        t90 = first_time(t, win & (r >= 0.9))
        t_stop = first_time(t, (t > t_rel) & (t < t_next) & (r <= 0.1))
        out.update(speed_channel=ch, peak=peak,
                   speedup=None if (t90 is None or t_react is None) else max(0.0, t90 - t_react),
                   stop=None if t_stop is None else t_stop - t_rel)
        break

    # video: first frame after the command whose change is well above hover level and the pre-command level
    if len(video):
        vt, ve = video[:, 0], video[:, 1]
        pre_v = ve[(vt >= t_cmd - 0.3) & (vt < t_cmd)]
        thr = max(v_thr, 1.5 * pre_v.max()) if pre_v.size else v_thr
        t_vid = first_time(vt, (vt >= t_cmd) & (vt < t_next) & (ve >= thr))
        out["video"] = None if t_vid is None else t_vid - t_cmd
        out["video_delay"] = None if (t_vid is None or t_react is None) else t_vid - t_react
    return out


def spread(vals, scale=1000.0, unit="ms"):
    v = np.array([x for x in vals if x is not None], float) * scale
    if not v.size:
        return "-"
    sd = v.std(ddof=1) if v.size > 1 else 0.0
    return f"{v.mean():6.0f} ± {sd:4.0f} {unit}  [{v.min():.0f} … {v.max():.0f}]  n={v.size}"


def stream_line(ts):
    if len(ts) < 5:
        return "not enough samples"
    dt = np.diff(ts)
    return (f"{1 / np.median(dt):5.1f} Hz   interval {1000 * np.median(dt):.0f} ms   jitter {1000 * dt.std():.0f} ms   "
            f"worst gap {1000 * dt.max():.0f} ms   n={len(ts)}")


def analyze(rec):
    print("\n" + "=" * 78 + "\nTELLO LATENCY REPORT   " + rec["meta"].get("stamp", "") + f"   battery {rec['meta'].get('battery')}%")
    print("=" * 78)
    rtt = rec["meta"].get("rtt", [])
    print(f"command delay (round trip)   {spread(rtt)}")
    for k in ("takeoff", "land"):
        if rec["meta"].get(k) is not None:
            print(f"{k:<28} 'ok' after {rec['meta'][k] * 1000:.0f} ms")
    video = np.array(rec["video"], float).reshape(-1, 2)
    if rec["state"]:
        print(f"state stream                 {stream_line(np.array(rec['state'])[:, 0])}")
    if len(video):
        print(f"video stream                 {stream_line(video[:, 0])}")
    steps = [e for e in rec["events"] if e["type"] in ("rc", "move")]
    if not steps or len(rec["state"]) < 10:
        print("no flight steps to analyse")
        return {}

    d = build_series(rec)
    h0, h1 = rec["meta"]["hover"]
    hov = (d["t"] >= h0) & (d["t"] <= h1)
    noise = {k: float(np.nanstd(d[k][hov])) for k in d if k != "t"}
    hv = video[(video[:, 0] >= h0) & (video[:, 0] <= h1), 1] if len(video) else np.array([])
    v_thr = float(hv.mean() + 5 * hv.std()) if hv.size > 5 else 5.0

    results = []
    for i, ev in enumerate(steps):
        t_next = steps[i + 1]["t_cmd"] if i + 1 < len(steps) else ev["t_release"] + 3.0
        results.append(analyze_step(d, noise, video, v_thr, ev, t_next))

    groups = [(a, [r for s, r in zip(steps, results) if s["type"] == "rc" and s["axis"] == a]) for a in AXES]
    groups.append(("forward/back move", [r for s, r in zip(steps, results) if s["type"] == "move"]))
    for name, rs in groups:
        if not rs:
            continue
        ch = next((r["speed_channel"] for r in rs if "speed_channel" in r), "?")
        print("-" * 78)
        print(f"{name.upper()}   ({sum('speed_channel' in r for r in rs)}/{len(rs)} steps responded, speed channel {ch}, "
              f"peak {spread([r.get('peak') for r in rs], 1.0, '')})")
        if name.startswith("forward"):
            oks = [s["t_release"] - s["t_cmd"] for s in steps if s["type"] == "move"]
            print(f"  time to 'ok'   {spread(oks)}")
        print(f"  lag            {spread([r.get('lag') for r in rs])}")
        print(f"  speed-up       {spread([r.get('speedup') for r in rs])}")
        if not name.startswith("forward"):
            print(f"  stop           {spread([r.get('stop') for r in rs])}")
        print(f"  video          {spread([r.get('video') for r in rs])}")
        print(f"  video delay    {spread([r.get('video_delay') for r in rs])}")
    print("=" * 78)
    print("lag = command -> first telemetry reaction | speed-up = reaction -> 90 % of peak speed | stop = rc 0 -> < 10 %")
    print("video = command -> first moving frame | video delay = video - lag")
    return {"steps": results, "noise": noise, "video_threshold": v_thr}


# ------------------------------------------------------------------ flight
def run(args):
    """Without --preview the flight runs right here. With it, the flight runs in a thread and the main
    thread shows the window (macOS); Ctrl+C is forwarded to the flight thread, which lands as before."""
    if not args.preview:
        fly(args, None)
        return
    ui = {"video": threading.Event(), "hud": {"phase": "starting"}}
    done = threading.Event()
    failed = []

    def flight():
        try:
            fly(args, ui)
        except BaseException as e:  # reported on the main thread
            failed.append(e)
        finally:
            ui["video"].set()
            done.set()

    th = threading.Thread(target=flight, name="flight")
    th.start()
    interrupted = shown = False
    while not done.is_set():
        try:
            if not (interrupted or shown):
                shown = True  # q or closing the window ends the preview for good
                preview_loop(ui, done, args.preview_scale)
            done.wait(0.1)
        except KeyboardInterrupt:
            if interrupted:
                print("  Ctrl+C ignored: still landing")
            else:
                interrupted = True
                interrupt_thread(th)  # raises KeyboardInterrupt inside fly(): its handler lands
    th.join()
    for e in failed:
        if not isinstance(e, KeyboardInterrupt):
            raise e


def interrupt_thread(th):
    """Raise KeyboardInterrupt inside another thread (delivered at its next Python step, within ~50 ms
    while flying: hold() sleeps in 0.05 s slices)."""
    import ctypes

    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(th.ident), ctypes.py_object(KeyboardInterrupt))


def _drop_binary_reply_noise(record) -> bool:
    """djitellopy logs an ERROR when the Tello answers with one of its binary packets (they start with 0xCC); it
    then retries the command, so the line only looks alarming."""
    return "can't decode byte" not in record.getMessage()


def quiet_ffmpeg() -> None:
    """Hide FFmpeg's decoder chatter (\"non-existing PPS 0 referenced\", \"no frame!\", \"error while decoding MB\"):
    it is printed until the first keyframe and after every lost packet, and the report already measures the video.
    --ffmpeg-log brings it back."""
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # OpenCV's bundled FFmpeg: quiet (read when it starts)
    try:
        import av.logging

        av.logging.set_level(av.logging.PANIC)  # PyAV's FFmpeg (pyav and raw readers)
    except Exception:
        pass


def fly(args, ui):
    import logging

    import cv2  # noqa: F401  - fail here, not silently inside the video thread
    from djitellopy import Tello

    from kite.drone.tello import apply_video_link
    from kite.sources.opencv_sources import stream_class  # likewise: a missing install must not quietly drop the video

    stream_class(args.video_backend)

    Tello.LOGGER.setLevel(logging.WARNING)  # otherwise every rc packet is printed
    Tello.LOGGER.addFilter(_drop_binary_reply_noise)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = {"meta": {"stamp": stamp, "args": vars(args)}, "state": [], "video": [], "events": []}
    stop = threading.Event()

    tello = Tello()
    tello.connect()
    rec["meta"]["battery"] = tello.get_battery()
    print(f"connected, battery {rec['meta']['battery']}%")
    apply_video_link(lambda c: tello.send_command_with_return(c, timeout=3), args.video_bitrate,
                     args.video_resolution, args.video_fps, say=print)
    if not args.ground and rec["meta"]["battery"] < args.min_battery:
        print(f"battery below {args.min_battery}% -> not flying")
        return
    hud = ui["hud"] if ui is not None else {"phase": "starting"}
    threading.Thread(target=log_state, args=(tello, rec, stop), daemon=True).start()

    # One reader for both the recorder and the preview: two readers cannot both bind UDP 11111. Every reader
    # drains the stream and keeps only the newest frame, so decoder backlog never inflates the measured video
    # delay over a long flight.
    stream = None
    try:
        stream = open_video(tello, args.decode_fps, args.video_backend)
    except Exception as e:  # a video problem must never stop the flight or the telemetry run
        print(f"video unavailable ({e})")
    if stream is None:
        print("continuing with telemetry only")
    else:
        threading.Thread(target=log_video, args=(stream, rec, stop), daemon=True).start()
    if ui is not None:  # the main thread draws it
        ui["stream"] = stream
        ui["video"].set()
    time.sleep(3.0)  # let the video decoder start

    rec["meta"]["rtt"] = []
    for _ in range(10):
        reply, t0, t1 = timed_cmd(tello, "battery?")
        if reply is not None:
            rec["meta"]["rtt"].append(t1 - t0)

    if args.ground:
        hud["phase"] = "ground test"
        t = time.time()
        time.sleep(5.0)
        rec["meta"]["hover"] = [t, time.time()]
    else:
        print("taking off - Ctrl+C lands")
        try:
            # armed before the takeoff is sent: a Ctrl+C during the climb or a lost 'ok' must still land it
            # (safe_land below and djitellopy's end() both land only when is_flying)
            tello.is_flying = True
            reply, t0, t1 = timed_cmd(tello, "takeoff", timeout=20)
            if not (reply and reply.lower().startswith("ok")):
                # never run the step sequence on a drone that did not confirm it is airborne; still on the ground
                # (h < 30 cm, e.g. a refused takeoff): nothing to land
                if float((tello.get_current_state() or {}).get("h", 0) or 0) < 30:
                    tello.is_flying = False
                raise RuntimeError(f"takeoff not confirmed (reply {reply!r})")
            rec["meta"]["takeoff"] = t1 - t0
            hud["phase"] = "hover"
            hold(tello, [0, 0, 0, 0], 2.0)
            t = time.time()
            hold(tello, [0, 0, 0, 0], args.hover)
            rec["meta"]["hover"] = [t, time.time()]
            for axis in AXES:
                print(f"{axis}: {args.trials} x (+{args.mag} / -{args.mag}) for {args.dur} s")
                for n in range(args.trials):
                    hud["phase"] = f"{axis} {n + 1}/{args.trials}"
                    rc_step(tello, rec, axis, +1, args.mag, args.dur, args.settle)
                    rc_step(tello, rec, axis, -1, args.mag, args.dur, args.settle)
            if args.forward > 0:
                print(f"forward/back {args.forward} cm x {args.trials}")
                for n in range(args.trials):
                    hud["phase"] = f"forward/back {n + 1}/{args.trials}"
                    move(tello, rec, f"forward {args.forward}", args.settle)
                    move(tello, rec, f"back {args.forward}", args.settle)
        except KeyboardInterrupt:
            print("\nCtrl+C -> landing")
        except Exception as e:
            print(f"\naborting flight: {e}")
        finally:
            hud["phase"] = "landing"
            if tello.is_flying:
                safe_land(tello, rec)
            rec["meta"].setdefault("hover", [time.time() - 1, time.time()])

    time.sleep(0.5)
    stop.set()
    if stream is not None:
        stream.stop()
    tello.end()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"tello_latency_{stamp}.json")
    with open(path, "w") as f:
        json.dump(rec, f)
    summary = analyze(rec)
    with open(path.replace(".json", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"raw data -> {path}")


def check_args(p, args):
    """Reject values that would turn a nudge into a dash across the room."""
    if not 10 <= args.mag <= 100:
        p.error("--mag must be 10-100 (below 10 the drone barely moves, above ~60 it covers a lot of ground)")
    if not 0.5 <= args.dur <= 3.0:
        p.error("--dur must be 0.5-3.0 s (distance scales with it; 1.5 s at --mag 30 is ~0.6-1 m)")
    if not 0.0 <= args.settle <= 10.0:
        p.error("--settle must be 0-10 s")
    if not 0.2 <= args.preview_scale <= 2.0:
        p.error("--preview-scale must be 0.2-2.0")
    if not 5 <= args.decode_fps <= 60:
        p.error("--decode-fps must be 5-60")
    if not 1 <= args.trials <= 10:
        p.error("--trials must be 1-10")
    if args.forward != 0 and not 20 <= args.forward <= 500:
        p.error("--forward must be 0 (skip) or 20-500 cm - the Tello rejects anything else")
    if args.mag * args.dur > 90:  # ~30 x 1.5 x 2, i.e. twice the documented default push
        p.error(f"--mag {args.mag} x --dur {args.dur} is a long, fast push; keep mag*dur <= 90 "
                f"(default is 30 x 1.5 = 45) or you will need more than 3 x 3 m")


def build_parser():
    from kite.config import VIDEO_BACKENDS, add_video_link_args

    p = argparse.ArgumentParser(description="Measure Tello command, motion and video lag in one flight.")
    p.add_argument("--mag", type=int, default=30, help="rc stick value for the steps, 10-100 (default 30)")
    p.add_argument("--dur", type=float, default=1.5, help="seconds each step is held (default 1.5)")
    p.add_argument("--settle", type=float, default=2.0, help="hover seconds between steps (default 2.0)")
    p.add_argument("--trials", type=int, default=3, help="repeats per axis, each = one + and one - step (default 3)")
    p.add_argument("--forward", type=int, default=50, help="cm for the forward/back move test, 0 = skip (default 50)")
    p.add_argument("--hover", type=float, default=3.0, help="hover seconds used to measure sensor noise (default 3)")
    p.add_argument("--min-battery", type=int, default=30)
    p.add_argument("--preview", action="store_true", help="show the live camera feed in a window during the run")
    p.add_argument("--preview-scale", type=float, default=1.0, help="scale the preview window, e.g. 0.5 (default 1.0)")
    p.add_argument("--decode-fps", type=int, default=60, help="decode cap in fps; frames arriving faster are dropped, "
                   "not queued (default 60: it must sit above the ~30 fps stream, a cap of 30 drops ~40 %% of frames "
                   "to arrival jitter)")
    p.add_argument("--video-backend", choices=VIDEO_BACKENDS, default=None,
                   help="video reader: pyav (default), opencv, raw (kite's own UDP reader) or auto (raw on macOS, "
                        "pyav elsewhere)")
    add_video_link_args(p)
    p.add_argument("--drop-corrupt", action="store_true",
                   help="raw reader: hold the last clean frame through packet loss instead of showing smears")
    p.add_argument("--ground", action="store_true", help="no takeoff: command delay and stream rates only")
    p.add_argument("--simple", action="store_true", help="short flight: one + and one - step per axis, no forward/back "
                   "moves (same as --trials 1 --forward 0, about 45 s in the air)")
    p.add_argument("--ffmpeg-log", action="store_true", help="show FFmpeg's decoder messages (hidden by default)")
    p.add_argument("--analyze", metavar="JSON", help="re-analyse a saved run without flying")
    p.add_argument("--out", default="results")
    return p


def parse_args(argv=None):
    """Arguments with --video-backend resolved to the reader that will run (None -> pyav, auto -> this platform's)."""
    from kite.config import resolve_video_backend

    p = build_parser()
    args = p.parse_args(argv)
    args.video_backend = resolve_video_backend(args.video_backend)
    if args.simple:
        args.trials, args.forward = 1, 0
    return p, args


def main(argv=None):
    p, args = parse_args(argv)
    if not args.ffmpeg_log:
        quiet_ffmpeg()
    global DROP_CORRUPT
    DROP_CORRUPT = args.drop_corrupt
    if not args.analyze:
        check_args(p, args)
    if args.analyze:
        with open(args.analyze) as f:
            analyze(json.load(f))
    else:
        run(args)


if __name__ == "__main__":
    main()