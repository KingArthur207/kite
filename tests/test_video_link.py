"""Video link knobs (setbitrate / setresolution / setfps) in the CLIs, and the one-line link verdict
(kite.sources.tello_raw.link_verdict) in kite.preflight and kite.video_doctor. No drone, no ffmpeg."""

from __future__ import annotations

import socket
import threading
import time

import pytest

from kite import preflight, video_doctor
from kite.sources.tello_raw import RawTelloStream, link_verdict


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ link verdict
def stats(**kw):
    base = {"datagrams": 3000, "bytes": 3000 * 1460, "receiving_s": 10.0, "max_gap_s": 0.05, "gaps_over_0.2s": 0,
            "frames_decoded": 300, "corrupt_frames": 3}
    return {**base, **kw}


def test_link_verdict_grades():
    grade, line = link_verdict(stats())
    assert grade == "GOOD" and "300 pkt/s" in line and "1% of frames" in line and "windows-drone" not in line
    assert link_verdict(stats(corrupt_frames=30))[0] == "FAIR"
    assert link_verdict(stats(**{"gaps_over_0.2s": 1}))[0] == "FAIR"
    assert link_verdict(stats(corrupt_frames=90))[0] == "POOR"
    assert link_verdict(stats(bytes=300 * 1460))[0] == "POOR"  # ~43 KB/s: starving
    assert link_verdict(stats(frames_decoded=0, corrupt_frames=0))[0] == "POOR"
    grade, line = link_verdict(stats(**{"gaps_over_0.2s": 5, "max_gap_s": 1.2}))
    assert grade == "POOR" and "max gap 1.20 s" in line and "If the drone video is corrupted" in line
    assert link_verdict({"datagrams": 0})[0] == "NONE"


def test_raw_reader_counts_gaps_between_datagrams():
    port = free_port()
    s = RawTelloStream(port).start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
            for i in range(6):
                tx.sendto(b"\x00" * 100, ("127.0.0.1", port))
                time.sleep(0.3 if i == 2 else 0.01)
        t_end = time.monotonic() + 2
        while s.stats()["datagrams"] < 6 and time.monotonic() < t_end:
            time.sleep(0.01)
        st = s.stats()
    finally:
        s.stop()
    assert st["datagrams"] == 6 and st["gaps_over_0.2s"] == 1 and st["max_gap_s"] >= 0.25
    assert 0.3 <= st["receiving_s"] < 1.5 and st["rcvbuf"] > 0


def test_preflight_link_check():
    good = preflight._link_check({"backend": "raw", **stats()})
    poor = preflight._link_check({"backend": "raw", **stats(corrupt_frames=100)})
    assert good.status == preflight.GO and good.data["grade"] == "GOOD"
    assert poor.status == preflight.WARN and "POOR" in poor.detail


def test_preflight_passes_the_flags_and_hands_them_to_the_next_command(monkeypatch, capsys):
    got = {}

    def fake_run(**kw):
        got.update(kw)
        return [preflight.Check("x", preflight.GO, "")]

    monkeypatch.setattr(preflight, "run", fake_run)
    assert preflight.main(["--quick", "--video-bitrate", "2", "--video-resolution", "low"]) == 0
    a = got["link_args"]
    assert (a.video_bitrate, a.video_resolution, a.video_fps) == ("2", "low", None)
    assert "--video-bitrate 2 --video-resolution low" in capsys.readouterr().out


def test_preflight_drone_checks_build_the_drone_with_the_flags(monkeypatch):
    import argparse

    import kite.drone.tello as tello_mod

    seen = []

    def connect(self):
        seen.append(self.video_link)
        raise RuntimeError("no drone in tests")

    monkeypatch.setattr(tello_mod.TelloDrone, "connect", connect)
    preflight.drone_checks(link_args=argparse.Namespace(video_bitrate="auto", video_resolution=None, video_fps="low"))
    assert seen == [{"bitrate": "auto", "resolution": None, "fps": "low"}]


# ------------------------------------------------------------------ video_doctor on Windows
NETSH = """
There are 2 interfaces on the system:

    Name                   : Wi-Fi
    Description            : Intel(R) Wi-Fi 6 AX201 160MHz
    State                  : connected
    SSID                   : TELLO-5A1B2C
    Radio type             : 802.11n
    Band                   : 2.4 GHz
    Channel                : 6
    Receive rate (Mbps)    : 72
    Signal                 : 88%

    Name                   : Wi-Fi 2
    Description            : NETGEAR A6210 WiFi USB3.0 Adapter
    State                  : connected
    SSID                   : Eddy's iPhone
    Band                   : 2.4 GHz
    Channel                : 11
    Signal                 : 70%
"""


def test_netsh_parse_and_shared_band_warning():
    itfs = video_doctor.parse_netsh_interfaces(NETSH)
    assert [i["Name"] for i in itfs] == ["Wi-Fi", "Wi-Fi 2"] and itfs[0]["Channel"] == "6"
    link = {"wifi_interfaces": itfs}
    assert len(video_doctor.shared_24ghz(link)) == 2
    v = video_doctor.verdict({"stream": {"packets": 100, "kbytes_per_s": 150, "nal_counts": {"IDR": 2}},
                              "link_before": link})
    assert any("Two Wi-Fi adapters are on 2.4 GHz" in line for line in v)
    itfs[1]["Channel"] = "149"  # hotspot moved to 5 GHz: no warning
    assert len(video_doctor.shared_24ghz(link)) == 1


class FakeDrone:
    """Answers the command port like a Tello (setbitrate: 'error', like SDK 1.3 firmware) and streams bytes to the
    video port once streamon arrives."""

    def __init__(self, video_port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.1)
        self.port, self.video_port, self.got = self.sock.getsockname()[1], video_port, []
        self._stop = threading.Event()
        self._streaming = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()
        threading.Thread(target=self._stream, daemon=True).start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            cmd = data.decode()
            self.got.append(cmd)
            if cmd == "streamon":
                self._streaming.set()
            reply = "error" if cmd.startswith("setbitrate") else ("87" if cmd == "battery?" else "ok")
            self.sock.sendto(reply.encode(), addr)

    def _stream(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
            while not self._stop.wait(0.005):
                if self._streaming.is_set():
                    tx.sendto(b"\x00" * 1460, ("127.0.0.1", self.video_port))

    def close(self):
        self._stop.set()
        self.sock.close()


def test_video_doctor_sends_the_link_settings_before_streamon(tmp_path, capsys):
    video_port = free_port()
    drone = FakeDrone(video_port)
    try:
        rc = video_doctor.main(["--host", "127.0.0.1", "--cmd-port", str(drone.port), "--local-cmd-port",
                                str(free_port()), "--video-port", str(video_port), "--seconds", "0.3",
                                "--out", str(tmp_path), "--video-bitrate", "2", "--video-resolution", "low"])
    finally:
        drone.close()
    out = capsys.readouterr().out
    assert rc == 0
    assert drone.got[:5] == ["command", "battery?", "setbitrate 2", "setresolution low", "streamon"]
    assert "doesn't support setbitrate" in out and "keeping its default" in out
    assert "\nlink " in out  # the one-line verdict
    import json

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["video_link"] == {"setbitrate 2": "error", "setresolution low": "ok"}
    assert summary["link_grade"] in ("GOOD", "FAIR", "POOR")


@pytest.mark.parametrize("argv", [["--video-bitrate", "6"], ["--video-resolution", "medium"], ["--video-fps", "30"]])
def test_cli_rejects_values_the_tello_does_not_know(argv):
    with pytest.raises(SystemExit):
        video_doctor.main(argv)
