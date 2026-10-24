# Flying and recording the Tello from Windows

Macs can't hold a reliable connection to the Tello (Sat Oct 24), so the drone is flown and recorded from Rex's Windows laptop. The drone layer (`kite/`) is plain Python, OpenCV and PyAV; on Windows it uses the PyAV reader. Nothing here needs a second network: the laptop is on the Tello's Wi-Fi only, and the body cam records on the Arduino Q.

## Set up once (on normal internet, about 20 minutes)

1. Install Python 3.12 from python.org (tick "Add python.exe to PATH") and Git for Windows.
2. In PowerShell:
   ```powershell
   git clone https://github.com/KingArthur207/kite.git
   cd kite
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   pip install -e ".[follow,dev]"
   mkdir data\models
   curl.exe -L -o data\models\yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
   ```
3. Check the install offline: `python -m pytest tests -q -m "not slow"`.
4. Open the firewall for the Tello's ports once, in an admin PowerShell:
   ```powershell
   netsh advfirewall firewall add rule name="kite tello" dir=in action=allow protocol=UDP localport=8890,11111
   ```
   Also click "Allow" for both private and public networks if Windows asks the first time Python opens a socket. The Tello's Wi-Fi counts as a public network.

## Before each flight

1. Power on the drone, join its Wi-Fi (`TELLO-xxxxxx`) from the laptop, and close the Tello phone app on every phone (while the app is connected, the drone streams to the phone).
2. `python -m kite.preflight`: it should say GO for commands, battery and video.
3. On the ground first, no takeoff: `python tests\drone_test.py --ground --preview`. A live picture means video works.
4. Fly and record per [the flight plan](shoot-flight.md), including the corner scout at Howard.

## Body cam

The chest webcam records on the Arduino Q, not on this laptop (for example `ffmpeg -f v4l2 -i /dev/video0 body.mp4`). Clap once in view of both cameras at the start, so the two recordings can be lined up later, then run both through `scripts/pov.sh`.

## If the drone video is corrupted

Commands work but the picture smears or breaks into blocks: the video is losing packets on the radio link. Measure before and after each change below, so you know which one helped:

```powershell
python -m kite.video_doctor --seconds 20
```

It ends with one line, `link GOOD / FAIR / POOR`, giving packets per second, KB/s, the longest pause between packets and the share of frames damaged by loss. On Windows it also lists both Wi-Fi adapters and warns when both are on 2.4 GHz. `python -m kite.preflight --video-backend raw` prints the same line as its `video link` check.

Try these in order:

1. Keep the Tello alone on 2.4 GHz. The Tello only has 2.4 GHz, so move the phone hotspot to 5 GHz (iPhone: Settings > Personal Hotspot, Maximize Compatibility OFF; Android: hotspot band 5 GHz) and have the USB Netgear adapter and the Arduino join it there. The two radios then stop sharing one band.
2. If that isn't enough, swap the adapters: the Netgear (better antenna) on the Tello, the built-in Wi-Fi on the hotspot.
3. Turn power saving off on both adapters: Device Manager > Network adapters > each adapter > Power Management, untick "Allow the computer to turn off this device to save power"; and Control Panel > Power Options > Change plan settings > Change advanced power settings > Wireless Adapter Settings > Power Saving Mode: Maximum Performance (on battery and plugged in).
4. Use kite's own reader, which recovers from lost packets better than PyAV's `udp://` reader: add `--video-backend raw` to `kite.record`, `kite.preflight`, `kite.calibrate` or `tests\drone_test.py`. It works on Windows as is; the log line `video socket UDP 11111: SO_RCVBUF ... granted` shows the receive buffer it got.
5. Send a lighter stream: `--video-bitrate 2 --video-resolution low` (or `--video-bitrate 1`). The same flags work on `kite.record`, `kite.preflight`, `kite.calibrate`, `tests\drone_test.py` and `kite.video_doctor`, plus `--video-fps high|middle|low`. Nothing is sent unless you pass them. Older Tello firmware doesn't know these commands; you'll see "this Tello's firmware doesn't support setbitrate; keeping its default" and the run carries on.
6. Keep the laptop within about 5 m of the drone with a clear line of sight, and close the Tello app on every phone.

`tests\drone_test.py` used to take `--video-fps N` as its decode cap; that is now `--decode-fps N`, and `--video-fps` sets the drone's stream rate like everywhere else.
