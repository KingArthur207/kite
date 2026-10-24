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
