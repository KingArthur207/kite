# Tello video on the Mac: why no frame is delivered

Status: Cal Hacks, Sat 2026-10-24, early morning PT. Investigated on Arthur's MacBook Pro (Apple Silicon, macOS 26 /
Darwin 25.6, Python 3.12.14, opencv-python 5.0.0 with FFmpeg 7.1 (libavcodec 61.19), av 19.0.2 with libavcodec 63,
djitellopy 2.5.0). The drone could not be reached from this machine, so every reader was run against a Tello-like
stream simulator on local UDP. A real capture (`python -m kite.video_doctor`) was **not available** when this was
written. Section 6 says how to settle the remaining question from one.

## 1. Summary

1. **The Mac, its sockets and both FFmpeg builds are fine.** Every reader decodes a Tello-like stream over UDP on this
   Mac: 960x720 at 30 fps, one burst of 1460-byte datagrams per frame, joined mid-stream. That holds with both FFmpeg
   builds loaded in one process, so the duplicate `AVFFrameReceiver` warning is harmless. It also holds with up to 20 %
   random datagram loss.
2. **"No frame" means no decodable keyframe reached the decoder within the reader's window.** FFmpeg outputs nothing
   until it has decoded a keyframe (an IDR, an I frame or a recovery point). Random loss does not prevent that. Losing
   20 % of datagrams, keeping only the first 3 datagrams of every frame, or dropping half of every keyframe still gives
   frames after the next keyframe. Zero frames were reproduced only when one of these held:
   - keyframes were sparse (5-8 s apart);
   - every keyframe was lost;
   - there were no keyframes at all.

   The real OpenCV signature is reproduced exactly when keyframes are 8 s apart: `Stream timeout triggered after 5014
   ms`, then `can't be used to capture by name`.
3. **Each reader adds a defect that turns sparse or damaged keyframes into zero frames:**
   - **OpenCV `VideoStream`** (drone_test). It forces no input format, so the H.264 format probe must see
     SPS+PPS+a slice within the first 1 MiB. Opening also waits for a decoded keyframe, but the open timeout is 5 s.
     After one 3 s read timeout the capture never delivers again. If opening ends without a decoded frame, the capture
     has size 0x0 and `retrieve()` fails forever. It cannot be made reliable for the Tello by tuning.
   - **`PyAVStream`** (kite's default, `python -m kite.sources.open tello`). `av.open` runs `find_stream_info`, and
     `fflags=nobuffer` throws away the packets it consumed. The keyframe that ended probing is never shown, so the
     first frame is the second keyframe. With keyframes 5 s apart the first frame comes at 8.6 s, against 3.6 s without
     the drop. With gaps of about 7 s or more, `FIRST_FRAME_S = 15` runs out and you get "frames 0".
   - **djitellopy `BackgroundFrameRead`**. It has the same format probe and a 5 s timeout as OpenCV, and fails the same
     way. kite does not use it for video.
4. **What works on this Mac in every condition where any decoder can work:** a raw UDP socket feeding
   `av.CodecContext('h264').parse()` and `decode()`. There is no format probe, no `find_stream_info`, and no FFmpeg
   timeout or interrupt state. The first frame comes at the first keyframe after listening starts (0.6 s, 3.6 s and
   6.6 s for keyframes every 1 s, 5 s and 8 s). Adding `flags2=+showall` and `flags=+output_corrupt` delivers frames
   0.5 s after the first SPS/PPS even with no keyframe: grey at first, filling in as intra blocks arrive. With
   parameter sets every 1 s but an IDR only every 8 s, that is 0.6 s instead of 6.6 s.
5. **Still open, and only the real capture can settle it:** how often the Tello sends SPS/PPS and keyframes, and how
   intact they arrive on the hackathon Wi-Fi. The real log points to parameter sets arriving, P slices decoding with
   errors, and no keyframe being output: `sps_id 0 out of range`, then `left block unavailable` and `error while
   decoding MB 0 6`, then `missing picture in access unit`. That is the `rare_idr_params1s` pattern, plus some loss.

## 2. Evidence table

`scripts/mac_video_repro.py` runs each reader in its own process, the way the real app does, while
`scripts/tello_video_sim.py` streams from another. The reader opens 1 s after the stream starts. Each cell gives the
time to the first frame, measured from reader process start (imports included), and the frames delivered in a
10-14 s window. **0** means it opened and delivered nothing. "open fails (probe)" is the format probe, ending in
`can't be used to capture by name` or `InvalidDataError`. "open fails (timeout)" is the open interrupt, ending in
`Stream timeout triggered after ~5010 ms`.

Readers:
- **opencv_kite**: `VideoStream` exactly as drone_test uses it (kite's `OPENCV_FFMPEG_CAPTURE_OPTIONS`, open 5 s,
  read 3 s).
- **opencv_kite_15s**: the same with 15 s timeouts, the `TelloVideoSource(backend="opencv")` settings.
- **opencv_noopts**: no capture options.
- **opencv_fast**: probesize 32 K and analyzeduration 0.
- **opencv_probe_15s**: formatprobesize 10 MB and analyzeduration 1 s.
- **pyav_kite**: `PyAVStream`.
- **pyav_noprobe**: `av.open(format="h264")` with probesize 32 K, analyzeduration 0 and no nobuffer.
- **pyav_buf**: `PyAVStream` with `?buffer_size=4M&fifo_size=..&overrun_nonfatal=1`.
- **tello_source_***: `TelloVideoSource`.
- **djitellopy**: `BackgroundFrameRead`.
- **parser**: raw socket with an 8 MB `SO_RCVBUF`, plus `CodecContext.parse`.
- **parser_smallbuf**: the default buffer and 30 ms of work per frame.
- **parser_showall**: parser with `flags2=+showall` and `flags=+output_corrupt`.

Conditions (sim flags in `CONDITIONS` in the script). Unless stated, the stream is 3 Mbit/s with an IDR every 1 s and
SPS/PPS inline before each IDR:

| condition | what the sender does |
|---|---|
| clean | the default stream, no loss |
| loss1 / 5 / 10 / 20 / 35 | random datagram loss of that percentage |
| tail8 / tail3 | only the first 8 or 3 datagrams of every frame arrive |
| bigloss50 | 50 % loss inside keyframe bursts only (`_sep`: SPS/PPS also in their own datagrams) |
| outage | everything lost for 150 ms once per second |
| outage_heavy | everything lost for 200 ms every 500 ms; it lines up with the 1 s keyframes, so every keyframe is lost |
| lowrate | 1 Mbit/s |
| params_sep | SPS and PPS in their own datagrams |
| keyint5s | IDR every 5 s |
| keyint8s | IDR every 8 s at 1 Mbit/s |
| rare_idr_params1s | IDR every 8 s, SPS/PPS every 1 s |
| nonidr | keyframes are non-IDR I frames, no SEI |
| intra_refresh | periodic intra refresh with recovery-point SEI |
| intra_refresh_nosei | intra refresh without SEI; parameter sets and keyframe only at the very start |

| condition | opencv_kite | opencv_kite_15s | opencv_noopts | opencv_fast | opencv_probe_15s | pyav_kite | pyav_noprobe | tello_source_pyav | tello_source_opencv | djitellopy | parser | parser_showall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 3.1 s / 263 | 3.1 s / 323 | (1.9 s, loss1) | **0** (loss1) | 2.1 s / 329 | 2.6 s / 276 | 0.7 s / 347 | 2.6 s / 285 | 3.1 s / 270 | 1.9 s / 304 | 0.6 s / 285 | |
| loss5 | 3.1 s / 263 | | 1.9 s / 301 | **0** | | 2.6 s / 276 | | | | 1.8 s / 307 | 0.6 s / 285 | |
| loss20 | 3.1 s / 262 | 3.1 s / 323 | | | | 2.6 s / 276 | 1.6 s / 319 | 2.6 s / 285 | 3.1 s / 271 | | 1.6 s / 256 | |
| loss35 | open fails (probe) | open fails (probe) | | | **0** | 4.6 s / 277 | | | | | 4.6 s / 165 | |
| tail3 | 3.1 s / 262 | | | | | 2.6 s / 276 | | | | | 0.6 s / 285 | |
| bigloss50_sep | 4.1 s / 272 | | | | | 3.6 s / 271 | | | | | 2.6 s / 225 | |
| outage | 3.2 s / 231 | | | | | 2.6 s / 239 | | | | 2.0 s / 262 | 0.6 s / 245 | |
| outage_heavy | open fails (timeout) | | | | | **0** | | | | open fails (probe) | **0** | |
| params_sep | 3.1 s / 265 | | 1.9 s / 300 | **0** | | 2.6 s / 274 | | | | 1.8 s / 302 | 0.6 s / 285 | |
| lowrate | 3.1 s / 263 | | 1.9 s / 301 | **0** | | 2.6 s / 275 | | | | 1.8 s / 304 | 0.6 s / 285 | |
| nonidr | 3.0 s / 267 | | 1.9 s / 301 | | | 2.5 s / 277 | | | | 1.8 s / 318 | 0.5 s / 288 | |
| intra_refresh | 4.1 s / 234 | | | | | 3.6 s / 246 | | | | 1.8 s / 301 | 1.6 s / 255 | |
| intra_refresh_nosei | **0** | | | | | **0** | | | | **0** | **0** | |
| keyint5s | open fails (probe) | open fails (probe) | open fails (probe) | open fails (probe) | **0** | **8.7 s** / 151 | 3.6 s / 258 | **8.6 s** / 105 | 9.1 s / 91 | open fails (probe) | 3.6 s / 195 | |
| keyint8s | **open fails (timeout), the real signature** | | | | | 6.6 s / 352 | 6.6 s / 412 | 6.6 s / 405 | 15.1 s / 150 | open fails | 6.6 s / 225 | 6.6 s / 224 |
| rare_idr_params1s | **0** (read timeout at 4.7 s, never recovers) | | | | | 6.6 s / 276 | | | | | 6.6 s / 224 | **0.6 s / 404** |

Notes on the table:
- **`pyav_kite` in keyint5s and keyint8s.** At 5 s the IDR arrives inside the probe window, `find_stream_info`
  consumes it and nobuffer drops it, so the first frame is the next IDR: about 8.6 s, 5 s later than `pyav_noprobe`.
  At 8 s the probe gives up first (`could not find codec parameters` after its default 5 s analyzeduration), so
  nothing is dropped.
- **`opencv_fast` and `opencv_probe_15s` with 0 frames.** The capture opened before a frame was decoded. Every
  `retrieve()` then logs `swscaler: Value 0.000000 for parameter 'srcw'` and `retrieveFrame Out of memory issue on
  av_frame_get_buffer`, forever.
- **Socket buffers, measured with `scripts/udp_burst_check.py --ffmpeg`.** FFmpeg's `udp://` socket gets `SO_RCVBUF =
  393216`: netstat `rhiwat` 393216, its circular-buffer thread is active, and `fifo_size` is accepted without a
  warning. Python's default is 786896 (`net.inet.udp.recvspace`), and `kern.ipc.maxsockbuf` is 8 MB. With the reader
  idle during the burst, a back-to-back burst over loopback keeps:

  | burst | default / 384 KB / 8 MB buffer | 64 KB buffer |
  |---|---|---|
  | 200 datagrams | 200 | 43 |
  | 600 datagrams | 527 / 263 / 600 | 43 |

  A Tello keyframe is about 17-60 datagrams, so receive-side overflow on this Mac needs a reader stalled for about
  1 s. Buffer options (`pyav_buf`, `opencv_*_buf`) changed nothing.
- **Reproduced real-log signatures.** `Stream timeout triggered after 5014/5007 ms` plus `can't be used to capture
  by name` in keyint8s and outage_heavy. `non-existing PPS 0 referenced`, `decode_slice_header error` and `no frame!`
  in every mid-stream join. `error while decoding MB x y` with loss. Not reproduced: `sps_id 0 out of range`, which
  needs a PPS decoded before any valid SPS, so SPS and PPS travel in different datagrams on the Tello and the SPS
  datagram was missed. `left block unavailable` was not reproduced verbatim either; it is the same family as `error
  while decoding MB`, bitstream damage from missing data.

## 3. Hypotheses

| | hypothesis | verdict |
|---|---|---|
| a | OpenCV probesize 10 MB and analyzeduration 12 s versus a 5 s open timeout | **Partly confirmed.** With a keyframe in the window, opening ends about 0.9 s after it (open 1.8 s), so the probe options are irrelevant. Without one, `find_stream_info` keeps reading toward 12 s and the 5 s open interrupt fires: the exact real signature. Raising the timeout does not rescue OpenCV, see below. |
| b | UDP receive-side loss on the Mac: FFmpeg buffer, macOS defaults | **Killed for this Mac's receive path.** FFmpeg sets a 384 KB `SO_RCVBUF` and drains it with a thread; buffer options change nothing; FFmpeg survives 20 % loss anyway. Wi-Fi loss before the Mac could not be measured here, which is why the capture is needed. |
| c | Stream structure: SPS/PPS only before IDRs, IDRs far apart, nobuffer, framedrop | **Confirmed as the trigger.** Sparse keyframes produce zero frames with OpenCV (≥5 s) and the real timeout signature. nobuffer delays the first frame by one keyframe interval in both kite readers: OpenCV 3.1 s against 1.9 s without options, PyAV 8.6 s against 3.6 s. |
| d | PyAV specifics | **Confirmed:** `find_stream_info` plus nobuffer drops the first keyframe. The `timeout=(open, read)` tuple behaves as documented: open covers `avformat_open_input` and `find_stream_info`. `packet.decode()` errors are swallowed correctly. Starvation by threads or the GIL: none seen (PyAV kept 30 fps in every row). |
| e | Duplicate libavdevice objc classes | **Killed.** They belong to the AVFoundation capture device only. Every opencv_* and tello_source run imports djitellopy and so loads both builds, and decodes fine. |
| f | djitellopy 2.5.0 and the `0xCC` reply | **Unrelated to video.** `0xCC` is the first byte of the Tello's binary app-protocol packets landing on UDP 8889. djitellopy's `send_command_with_return` pops one reply per command, so that command gets "response decode error" and is retried; kite's `busy()` already decodes with `errors="ignore"`. Video is on 11111. Worth noting: the drone talks the app protocol, so make sure the phone app is closed. |
| g | New: OpenCV read timeout is fatal | **Confirmed.** One `grab()` timeout (3 s in drone_test) stops the capture for good, even when keyframes arrive later (`rare_idr_params1s`: IDR at 6.6 s, 0 frames in 14 s). |
| h | New: OpenCV opens with 0x0 size when probing ends without a decoded frame | **Confirmed** (`opencv_fast`, `opencv_probe_15s`): every `retrieve()` fails forever. |
| i | New: the format probe needs a keyframe in the first 1 MiB | **Confirmed** for OpenCV and djitellopy, which force no format (`InvalidDataError` or `can't capture by name` after about 2.9 s at 3 Mbit/s). PyAVStream forces `format="h264"` and is immune. |
| j | New: non-IDR I frames, or intra refresh with SEI | **Killed.** FFmpeg's heuristics output both (`nonidr`, `intra_refresh`). Only a stream with neither keyframes nor recovery points gives zero frames; `showall` is the only way to see that. |

## 4. What must change

These changes are for the agent working on `kite/sources/*.py` and `kite/drone/*.py`; this branch does not touch those
files.

### 4.1 Tello reader: raw socket plus H.264 parser (recommended)

Replace `PyAVStream` and `VideoStream` for the Tello with a reader that owns the socket:

```python
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)   # kern.ipc.maxsockbuf is 8 MB here
sock.bind(("0.0.0.0", 11111))
sock.settimeout(0.5)                                                    # Python-side timeout: nothing to break in FFmpeg
codec = av.CodecContext.create("h264", "r")
codec.options = {"flags2": "+showall", "flags": "+output_corrupt"}      # frames before the first keyframe (grey at first)
while not stopped:
    try:
        data = sock.recv(65536)
    except socket.timeout:
        continue                                                        # count it; the stream just paused
    for pkt in codec.parse(data):
        try:
            frames = codec.decode(pkt)
        except av.error.FFmpegError:
            continue                                                    # damaged packet: keep going
        for f in frames:
            keyframe_seen |= f.key_frame                                # flag frames before it as unreliable
            ...  # f.to_ndarray(format="bgr24") at most `fps` times per second, publish (seq, img, ts, keyframe_seen)
```

- **Bind it before the first `streamon`.** Then the parameter sets and keyframe the drone sends when streaming
  starts are not lost. Let this reader report "packets arrive" instead of the separate `video_packets_arrive()` probe
  socket in `kite/drone/tello.py`; today that probe swallows the first datagram, then closes, and FFmpeg binds later.
- **Nothing to reopen.** No FFmpeg I/O, no interrupt state. The reconnect logic in `TelloVideoSource` can go for this
  backend. Keep the seq and timestamp interface.
- **Expose counters** (datagrams/s, KB/s, decode errors, keyframes seen, SPS/PPS seen) so the next "no frames"
  explains itself in one line.
- **Whether to keep `showall`.** It turns "no keyframe yet" into grey-smeared frames that clean up as intra blocks
  arrive. Keep it, and gate detectors on `keyframe_seen` if smeared frames hurt them. Without it, the reader behaves
  like `parser`: first frame at the first keyframe.

### 4.2 If PyAVStream stays

Remove `"fflags": "nobuffer"` and skip probing:

```python
av.open(url, format="h264", timeout=(15, 15),
        options={"probesize": "32768", "analyzeduration": "0", "fpsprobesize": "0", "flags": "low_delay"})
```

This is `pyav_noprobe`: open in 0.2 s, first frame at the first keyframe (0.65 s for 1 s keyframes, 3.6 s for 5 s),
and it survives 20 % loss. The h264 decoder still works out the size from the first SPS. `showall` through the
demuxer path was not tested. If the drone's keyframes turn out to be rare, the parser reader (4.1) is the fix.

### 4.3 OpenCV `VideoStream`: stop using it for the Tello

Use it in neither `tests/drone_test.py` nor `TelloVideoSource(backend="opencv")`. Defects 3, g, h and i cannot be
fixed through `OPENCV_FFMPEG_CAPTURE_OPTIONS`:
- Raising `formatprobesize` gets it to open, but at 0x0 size.
- Small probe sizes give the same 0x0 failure.
- Longer timeouts only help if a clean keyframe arrives in time.

drone_test should read video through kite's Tello source (the parser backend). If OpenCV must stay as a fallback,
give it open and read timeouts of at least twice the keyframe gap plus 2 s, and remove `fflags;nobuffer` and
`framedrop`.

### 4.4 Timeouts that assume frequent keyframes

`FIRST_FRAME_S = 15` (`kite/sources/open.py`) and `TelloVideoSource.timeout_ms = 15000` are fine with the parser
reader. With PyAVStream as it is today, they fail once keyframes are about 7 s or more apart.

## 5. Reproduction

Use `/Users/arthurarmaing/Desktop/kite/.venv/bin/python` from the repo root. Needs the ffmpeg CLI, and x264 for the
`nonidr` condition.

```bash
python scripts/tello_video_sim.py --keyint 150 --loss 0.05     # a Tello-like sender on 127.0.0.1:11111 (Ctrl+C stops)
python scripts/mac_video_repro.py                              # the default matrix (about 5 min)
python scripts/mac_video_repro.py --readers opencv_kite,pyav_kite,parser,parser_showall --conditions keyint8s,rare_idr_params1s
python scripts/udp_burst_check.py --ffmpeg                     # socket-buffer burst test, FFmpeg's real SO_RCVBUF
python -m pytest tests/test_mac_video_repro.py -m slow         # 6 tests, about 3 min: the behaviours this report relies on
```

## 6. With a real capture (`python -m kite.video_doctor`)

```bash
python scripts/mac_video_repro.py --analyze recordings/video_doctor_<stamp>   # offline, joining at 0, 1, 2.5, 4, 6, 9 s
python scripts/mac_video_repro.py --replay  recordings/video_doctor_<stamp> --readers opencv_kite,pyav_kite,parser,parser_showall
```

`--analyze` prints the SPS, PPS, IDR and I-frame timeline, and the first-frame delay for the default and `showall`
decoders at each join time. Read it with `summary.json` (`idr_interval_s`, `idr_with_sps_pps_before`,
`decode.pyav_parser.frames`):

| capture shows | meaning | fix |
|---|---|---|
| IDRs at least once per second, offline decode fine, but the live readers fail | live loss or the reader | replay with `--replay`. If the replay decodes too, the loss is on the Wi-Fi (venue 2.4 GHz). Use 4.1 with `showall` and keep the drone close. |
| IDRs 5 s or more apart, or only right after `streamon` | structural (this report's main case) | 4.1, bound before `streamon`, with `showall` |
| SPS/PPS often, IDR rare (`showall` first frame much earlier than default) | the real log's pattern | 4.1 with `showall` |
| No IDR and no I frame at all, offline decode 0 frames | only `showall` can show anything | 4.1 with `showall`, and re-send `streamon` to see whether it triggers a keyframe |
| `packets: 0`, or no H.264 start codes | the drone is not streaming to this laptop | see `NO_VIDEO` in `kite/drone/tello.py` |
