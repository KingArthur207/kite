"""board.py: auth, commands, text, status from /proc and /sys, camera choice, MJPEG and the test pattern."""

from __future__ import annotations

import io
import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

import board
import kite_common
import kite_link


def get(port: int, path: str, token: str | None = None, body: dict | None = None, timeout: float = 5):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = json.dumps(body).encode() if body is not None else None
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # local test server  # nosec B310
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def decode_jpeg(b: bytes):
    Image = pytest.importorskip("PIL.Image")
    im = Image.open(io.BytesIO(b))
    im.load()
    return im


# ------------------------------------------------------------------------------------------- auth

@pytest.mark.parametrize("path", ["/hello", "/ping", "/status", "/snapshot.jpg", "/video.mjpg", "/events", "/view"])
def test_every_endpoint_but_health_needs_the_token(make_board, token, path):
    b = make_board(token, None)
    assert get(b.port, path)[0] == 401
    assert get(b.port, path, "wrong-" + token)[0] == 401
    assert get(b.port, path + "?token=nope")[0] == 401
    assert get(b.port, "/health")[0] == 200


def test_token_in_query_works_and_is_never_logged(make_board, token, capsys):
    b = make_board(token, None)
    code, body, _ = get(b.port, f"/hello?token={token}")
    assert code == 200 and json.loads(body)["reply"].startswith("hello from ")
    assert token not in capsys.readouterr().out


def test_post_send_needs_the_token(make_board, token):
    b = make_board(token, None)
    assert get(b.port, "/send", None, {"text": "hi"})[0] == 401
    assert b.board.received == 0


def test_board_refuses_to_start_without_a_token(tmp_path):
    cfg = board.Config({"KITE_TOKEN_FILE": str(tmp_path / "missing")})
    with pytest.raises(SystemExit):
        cfg.load_token()
    (tmp_path / "short").write_text("abc\n")
    cfg = board.Config({"KITE_TOKEN_FILE": str(tmp_path / "short")})
    with pytest.raises(SystemExit):
        cfg.load_token()


# --------------------------------------------------------------------------------------- commands

def test_text_round_trip_with_unicode(make_board, token, capsys):
    b = make_board(token, None)
    text = "héllo ✓ 世界 🪁"
    code, body, _ = get(b.port, "/send", token, {"text": text})
    d = json.loads(body)
    assert code == 200 and d["reply"] == f'got "{text}"' and d["n"] == 1
    assert f"message 1 (direct): {text}" in capsys.readouterr().out


@pytest.mark.parametrize("body,err", [
    ({"text": "x" * 501}, "longer than 500"), ({"text": ""}, "empty"), ({"text": 5}, "string"),
    ({"text": "\x00\x01"}, "empty"), ({"nottext": "hi"}, "string"),
])
def test_bad_text_is_refused(make_board, token, body, err):
    b = make_board(token, None)
    code, raw, _ = get(b.port, "/send", token, body)
    assert code == 400 and err in json.loads(raw)["error"]
    assert b.board.received == 0


def test_text_at_the_cap_is_accepted_and_control_characters_are_stripped(make_board, token):
    b = make_board(token, None)
    code, raw, _ = get(b.port, "/send", token, {"text": "a\x07" + "y" * 499})
    assert code == 200 and json.loads(raw)["reply"] == 'got "a' + "y" * 499 + '"'


def test_body_must_be_small_json_object(make_board, token):
    b = make_board(token, None)
    req = urllib.request.Request(f"http://127.0.0.1:{b.port}/send", data=b"[1,2]",
                                 headers={"Authorization": f"Bearer {token}"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)  # local test server  # nosec B310
    assert e.value.code == 400
    req = urllib.request.Request(f"http://127.0.0.1:{b.port}/send", data=b"x" * 9000,
                                 headers={"Authorization": f"Bearer {token}"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)  # local test server  # nosec B310
    assert e.value.code == 400


def test_command_dispatch(make_board, token):
    b = make_board(token, None).board
    assert b.command("ping")["pong"] is True
    assert b.command("hello")["reply"] == f"hello from {b.host}"
    assert b.command("status")["status"]["camera"]["source"] == "synthetic"
    with pytest.raises(ValueError, match="unknown command"):
        b.command("rm -rf")
    with pytest.raises(ValueError):
        b.command("send", None)


def test_unknown_path_is_404(make_board, token):
    b = make_board(token, None)
    assert get(b.port, "/nope", token)[0] == 404


# ------------------------------------------------------------------------------------------ hello

def test_recurring_hello_arrives_in_order(make_board, token):
    b = make_board(token, None)
    d = kite_link.Direct("127.0.0.1", b.port, token)
    ns = []
    with d.open("/events", timeout=5) as r:
        for ev in kite_link.iter_events(r):
            if ev.get("kind") == "hello":
                ns.append(ev["n"])
            if len(ns) == 4:
                break
    assert ns == sorted(ns) and len(set(ns)) == 4 and ns[-1] - ns[0] == 3


def test_received_text_is_also_announced_to_listeners(make_board, token):
    b = make_board(token, None)
    d = kite_link.Direct("127.0.0.1", b.port, token)
    got = []
    with d.open("/events", timeout=5) as r:
        threading.Timer(0.5, lambda: d.call("/send", {"text": "ping from test"})).start()
        for ev in kite_link.iter_events(r):
            if ev.get("kind") == "got":
                got.append(ev)
                break
    assert got[0]["text"] == "ping from test" and got[0]["via"] == "direct"


def test_heartbeat_is_signed_and_carries_the_address(make_board, token):
    b = make_board(token, None).board
    hb = b.heartbeat()
    assert kite_common.verify(token, hb) and hb["ips"] == ["127.0.0.1"] and hb["port"] == b.cfg.port
    hb["ips"] = ["6.6.6.6"]
    assert not kite_common.verify(token, hb)


# ----------------------------------------------------------------------------------------- status

def test_status_reads_mocked_proc_and_sys(tmp_path):
    (tmp_path / "proc/net").mkdir(parents=True)
    (tmp_path / "proc/uptime").write_text("1234.56 4000.00\n")
    (tmp_path / "proc/net/wireless").write_text(
        "Inter-| sta-|   Quality        |   Discarded packets\n"
        " face | tus | link level noise |  nwid  crypt   frag\n"
        "wlan0: 0000   54.  -56.  -256        0      0      0\n")
    for i, v in enumerate(["45123", "51000"]):
        z = tmp_path / f"sys/class/thermal/thermal_zone{i}"
        z.mkdir(parents=True)
        (z / "temp").write_text(v + "\n")
    bat = tmp_path / "sys/class/power_supply/battery"
    bat.mkdir(parents=True)
    (bat / "voltage_now").write_text("7412000\n")

    def run(cmd):
        return {"iwgetid": "eduroam\n",
                "ip": "1: lo    inet 127.0.0.1/8 scope host lo\n"
                      "3: wlan0    inet 10.142.7.9/16 brd 10.142.255.255 scope global dynamic wlan0\n"
                      "5: tailscale0    inet 100.101.5.6/32 scope global tailscale0\n"
                      "6: docker0    inet 172.17.0.1/16 scope global docker0\n"}.get(cmd[0])

    root = str(tmp_path)
    assert board.uptime_s(root) == 1234
    assert board.cpu_temp_c(root) == 51.0
    assert board.battery_v(root) == 7.41
    assert board.wifi(root, run) == {"ssid": "eduroam", "iface": "wlan0", "signal_dbm": -56}
    assert board.addresses(run) == (["10.142.7.9"], "100.101.5.6")
    assert board.addresses(run, "192.168.1.5, 10.0.0.2") == (["192.168.1.5", "10.0.0.2"], "100.101.5.6")


def test_status_with_nothing_readable(tmp_path):
    root = str(tmp_path)
    assert board.uptime_s(root) is None and board.cpu_temp_c(root) is None and board.battery_v(root) is None
    assert board.wifi(root, lambda c: None) == {"ssid": None, "iface": None, "signal_dbm": None}
    nm = "no:Other\nyes:eduroam\n"
    assert board.wifi(root, lambda c: nm if c[0] == "nmcli" else None)["ssid"] == "eduroam"


def test_usb_camera_detection_skips_the_chip_codec_nodes(tmp_path):
    drivers = tmp_path / "drivers"
    for name in ("qcom-venus", "uvcvideo"):
        (drivers / name).mkdir(parents=True)
    for node, drv in (("video0", "qcom-venus"), ("video1", "qcom-venus"), ("video3", "uvcvideo"),
                      ("video2", "uvcvideo")):
        dev = tmp_path / f"sys/class/video4linux/{node}/device"
        dev.mkdir(parents=True)
        os.symlink(drivers / drv, dev / "driver")
    assert board.usb_cameras(str(tmp_path)) == ["/dev/video2", "/dev/video3"]
    cam, reason = board.open_camera("auto", 320, 240, 10, str(tmp_path / "empty"))
    assert isinstance(cam, board.SyntheticCamera) and reason == "no USB camera found"


def test_a_camera_that_cannot_open_falls_back_to_the_pattern(monkeypatch):
    def broken(*a, **k):
        raise OSError("can't open /dev/video9")
    monkeypatch.setattr(board, "OpenCVCamera", broken)
    cam, reason = board.open_camera("/dev/video9", 320, 240, 10)
    assert isinstance(cam, board.SyntheticCamera) and "video9" in reason and cam.label == "CAMERA ERROR"


# ------------------------------------------------------------------------------------------ video

def test_test_pattern_is_a_real_jpeg_of_the_right_size():
    for n, (w, h) in enumerate([(640, 480), (320, 240), (1280, 720)]):
        jpeg = board.test_pattern(n, w, h)
        assert kite_link.is_jpeg(jpeg)
        im = decode_jpeg(jpeg)
        assert im.size == (w, h)
    im = decode_jpeg(board.test_pattern(1))
    assert im.getpixel((5, 5)) == (255, 255, 255)          # first colour bar is white


def test_test_pattern_moves():
    assert board.test_pattern(1) != board.test_pattern(2)


def test_mjpeg_framing_round_trips():
    frames = [board.test_pattern(i, 320, 240) for i in range(3)] + [b"\xff\xd8" + b"\xff\xd9" * 3]
    stream = io.BytesIO(b"".join(board.mjpeg_part(f) for f in frames))
    assert list(kite_link.iter_mjpeg(stream)) == frames


def test_mjpeg_parser_without_content_length_and_with_a_truncated_tail():
    jpeg = board.test_pattern(1, 320, 240)
    s = io.BytesIO(b"--kiteframe\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n" +
                   board.mjpeg_part(jpeg)[:-500])
    assert list(kite_link.iter_mjpeg(s)) == [jpeg]


def test_boundary_from_content_type():
    assert kite_link.boundary_of('multipart/x-mixed-replace; boundary="abc"') == "abc"
    assert kite_link.boundary_of("multipart/x-mixed-replace") == "kiteframe"


def read_stream(port: int, token: str, seconds: float) -> list[tuple[float, bytes]]:
    d = kite_link.Direct("127.0.0.1", port, token)
    out = []
    end = time.monotonic() + seconds
    with d.open("/video.mjpg", timeout=5) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        for f in kite_link.iter_mjpeg(r):
            out.append((time.monotonic(), f))
            if time.monotonic() > end:
                break
    return out


def test_video_fps_and_every_frame_decodes(make_board, token):
    b = make_board(token, None)
    frames = read_stream(b.port, token, 3.0)
    fps = (len(frames) - 1) / (frames[-1][0] - frames[0][0])
    assert 8.5 <= fps <= 11, fps
    for _, f in frames:
        assert decode_jpeg(f).size == (640, 480)


def test_two_viewers_at_once(make_board, token):
    b = make_board(token, None)
    got = {}
    threads = [threading.Thread(target=lambda k=k: got.__setitem__(k, read_stream(b.port, token, 2.0)))
               for k in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(got[1]) >= 15 and len(got[2]) >= 15
    assert b.board.hub.viewers == 0 or b.board.hub.viewers <= 2


def test_too_many_viewers_get_503(make_board, token, monkeypatch):
    monkeypatch.setattr(board, "MAX_VIEWERS", 0)
    b = make_board(token, None)
    assert get(b.port, "/video.mjpg", token)[0] == 503


def test_snapshot_and_view_page(make_board, token):
    b = make_board(token, None)
    code, body, headers = get(b.port, "/snapshot.jpg", token)
    assert code == 200 and headers["Content-Type"] == "image/jpeg" and headers["X-Kite-Camera"] == "synthetic"
    assert decode_jpeg(body).size == (640, 480)
    code, page, headers = get(b.port, "/view", token)
    assert code == 200 and b"/video.mjpg?token=" in page and headers["Referrer-Policy"] == "no-referrer"


def test_camera_idles_without_viewers(make_board, token, monkeypatch):
    monkeypatch.setattr(board, "IDLE_S", 0.5)
    b = make_board(token, None)
    get(b.port, "/snapshot.jpg", token)
    time.sleep(1.2)
    seq = b.board.hub.seq
    time.sleep(1.0)
    assert b.board.hub.seq == seq
