"""Self-check: python sim/test_relay.py  (starts the relay on a spare port, pubs a frame, subs it)."""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from websockets.sync.client import connect

HERE = Path(__file__).parent
PORT = 5029


def main() -> None:
    relay = subprocess.Popen([sys.executable, str(HERE / "relay.py"), "--port", str(PORT)])
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", PORT)).close()
                break
            except OSError:
                time.sleep(0.1)
        with connect(f"ws://127.0.0.1:{PORT}/pub") as pub, connect(f"ws://127.0.0.1:{PORT}/sub") as sub:
            check(pub, sub)
    finally:
        relay.terminate()


def check(pub, sub) -> None:
    time.sleep(0.2)
    try:
        import cv2
        import numpy as np
        jpeg = cv2.imencode(".jpg", np.full((720, 960, 3), (0, 0, 255), np.uint8))[1].tobytes()
    except ImportError:
        cv2, jpeg = None, b"\xff\xd8fake-jpeg\xff\xd9"
    pose = {"type": "pose", "camera": "drone", "x": 1, "y": 2, "heading": 90, "cam_height": 2.0, "pitch": 0}
    pub.send(json.dumps(pose))
    pub.send(bytes([0]) + jpeg)
    assert json.loads(sub.recv(timeout=2)) == pose
    assert sub.recv(timeout=2) == bytes([0]) + jpeg
    print("relay pub -> sub OK")

    if cv2 is None:
        print("cv2 not installed: skipped SimFrameSource check")
        return
    sys.path.insert(0, str(HERE))
    from frames import SimFrameSource
    src = SimFrameSource("drone", f"ws://127.0.0.1:{PORT}/sub")
    time.sleep(0.5)                      # connected; it got the cached latest frame + pose
    ok, img = src.read()
    assert ok and img.shape == (720, 960, 3) and img[0, 0, 2] > 200, "bad frame"
    assert src.pose == pose, src.pose
    pub.send(bytes([1]) + jpeg)          # chest frame must not reach the drone source
    src.timeout = 0.5
    assert src.read() == (False, None)
    src.release()
    print("SimFrameSource OK")


if __name__ == "__main__":
    main()
