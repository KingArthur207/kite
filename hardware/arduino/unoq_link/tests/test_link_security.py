"""Security regressions for kite_link.py: where the token may go, replayed heartbeats, and hostile streams."""

from __future__ import annotations

import io
import socket
import time

import kite_common
import kite_link


def heartbeat(token: str, ips, port: int = 8765, ts_ms: int | None = None) -> dict:
    ts = int(time.time() * 1000) if ts_ms is None else ts_ms
    return kite_common.sign(token, {"msg": "hello", "n": 1, "host": "kite-unoq", "ips": ips, "tailscale": None,
                                    "port": port, "v": 1, "ts": ts})


# ------------------------------------------------------------------------- replayed heartbeats

def test_a_replayed_old_heartbeat_does_not_redirect_the_token(spacetime, token):
    now = time.time()
    spacetime.insert("heartbeat", "unoq-test", heartbeat(token, ["10.0.0.2"]), t=now - 5)
    # a genuine heartbeat from a day ago (other network, other IPs), pushed again by anyone as the newest row
    old = heartbeat(token, ["192.168.1.66"], ts_ms=int((now - 86400) * 1000))
    spacetime.insert("heartbeat", "unoq-test", old, t=now)
    info = kite_link.find_heartbeat(kite_common.Spacetime(spacetime.url), "unoq-test", token, now)
    assert info["valid"] and info["hb"]["ips"] == ["10.0.0.2"] and info["unsigned"] == 1


def test_only_replays_means_no_heartbeat(spacetime, token):
    now = time.time()
    spacetime.insert("heartbeat", "unoq-test", heartbeat(token, ["192.168.1.66"], ts_ms=int((now - 3600) * 1000)),
                     t=now)
    info = kite_link.find_heartbeat(kite_common.Spacetime(spacetime.url), "unoq-test", token, now)
    assert info["found"] and not info["valid"]


def test_a_heartbeat_without_ts_is_not_trusted(spacetime, token):
    d = kite_common.sign(token, {"msg": "hello", "ips": ["10.0.0.3"], "port": 8765})
    spacetime.insert("heartbeat", "unoq-test", d)
    info = kite_link.find_heartbeat(kite_common.Spacetime(spacetime.url), "unoq-test", token)
    assert not info["valid"]


def test_a_board_clock_a_few_minutes_off_still_works(spacetime, token):
    now = time.time()
    spacetime.insert("heartbeat", "unoq-test", heartbeat(token, ["10.0.0.4"], ts_ms=int((now - 300) * 1000)), t=now)
    info = kite_link.find_heartbeat(kite_common.Spacetime(spacetime.url), "unoq-test", token, now)
    assert info["valid"] and info["hb"]["ips"] == ["10.0.0.4"]


# ------------------------------------------------------------------- the MagicDNS name lookup

def fake_lookup(monkeypatch, ips):
    def getaddrinfo(host, port, *a, **k):
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
                for ip in ips]
    monkeypatch.setattr(kite_link.socket, "getaddrinfo", getaddrinfo)
    dialed = []

    def tcp_ok(host, port, timeout=2.0):
        dialed.append(host)
        return True, "connected"
    monkeypatch.setattr(kite_link, "tcp_ok", tcp_ok)
    return dialed


def test_a_lan_answer_for_the_tailscale_name_is_never_sent_the_token(monkeypatch):
    # Windows asks the LAN (LLMNR, NetBIOS) for a bare name, and some networks register DHCP hostnames in DNS:
    # anyone there can claim to be kite-unoq.
    dialed = fake_lookup(monkeypatch, ["192.168.1.50"])
    (label, host, port, ok, why), = kite_link.reach([("tailscale name", "kite-unoq", 8765)])
    assert not ok and "not a Tailscale address" in why and dialed == []


def test_the_tailscale_name_is_dialled_by_its_tailscale_address(monkeypatch):
    dialed = fake_lookup(monkeypatch, ["192.168.1.50", "100.101.102.103"])
    (label, host, port, ok, why), = kite_link.reach([("tailscale name", "kite-unoq", 8765)])
    assert ok and host == "100.101.102.103" and dialed == ["100.101.102.103"]


def test_tailscale_ranges():
    assert kite_link.is_tailscale("100.64.0.1") and kite_link.is_tailscale("100.127.255.254")
    assert kite_link.is_tailscale("fd7a:115c:a1e0::1")
    for ip in ("100.63.255.255", "100.128.0.1", "10.0.0.1", "192.168.1.5", "fd00::1", "nonsense", ""):
        assert not kite_link.is_tailscale(ip), ip


# ------------------------------------------------------------------------- hostile MJPEG streams

def test_mjpeg_refuses_a_huge_or_negative_content_length():
    for length in (str(kite_link.MAX_FRAME + 1), "-1", "lots"):
        s = io.BytesIO(b"--kiteframe\r\nContent-Length: " + length.encode() + b"\r\n\r\n" + b"\xff\xd8" * 100)
        assert list(kite_link.iter_mjpeg(s)) == []


def test_mjpeg_without_length_stops_at_the_frame_cap(monkeypatch):
    monkeypatch.setattr(kite_link, "MAX_FRAME", 1000)
    s = io.BytesIO(b"--kiteframe\r\nContent-Type: image/jpeg\r\n\r\n" + b"\x00" * 5000)
    assert list(kite_link.iter_mjpeg(s)) == []


def test_mjpeg_line_reads_are_bounded():
    s = io.BytesIO(b"x" * (kite_link.MAX_LINE * 50))       # no newline anywhere
    assert list(kite_link.iter_mjpeg(s)) == []
