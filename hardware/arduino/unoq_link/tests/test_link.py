"""kite_link.py against a real board.py socket and a fake SpacetimeDB: discovery, fallback, doctor, reconnects."""

from __future__ import annotations

import json
import threading
import time

import pytest
from conftest import free_port

import kite_common
import kite_link


def run(argv: list[str], st=None, token: str = "", tmp_path=None) -> tuple[int, str]:
    out: list[str] = []
    base = ["--source", "unoq-test", "--timeout", "1"]
    if st is not None:
        base += ["--st-host", st.url]
    else:
        base += ["--no-cloud"]
    if token:
        base += [f"--token={token}"]
    if tmp_path is not None:
        base += ["--token-file", str(tmp_path / "tok")]
    rc = kite_link.main(base + argv, out=lambda *a: out.append(" ".join(str(x) for x in a)))
    return rc, "\n".join(out)


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def heartbeat(token: str, ips, port: int, n: int = 1, **extra) -> dict:
    return kite_common.sign(token, {"msg": "hello", "n": n, "host": "kite-unoq", "ips": ips, "tailscale": None,
                                    "port": port, "v": 1, "ts": int(time.time() * 1000), **extra})


# ------------------------------------------------------------------------------------- token file

def test_token_file_round_trip_and_powershell_utf16(tmp_path):
    p = tmp_path / "tok"
    kite_link.save_token("  abcdefghijklmnopqrstuvwx \n", p)
    assert kite_link.read_token(p) == "abcdefghijklmnopqrstuvwx"
    p.write_bytes("abcdefghijklmnopqrstuvwx\r\n".encode("utf-16"))      # what PowerShell's > writes
    assert kite_link.read_token(p) == "abcdefghijklmnopqrstuvwx"
    with pytest.raises(SystemExit):
        kite_link.save_token("short", p)
    assert kite_link.read_token(tmp_path / "missing") == ""


def test_no_token_gives_clear_instructions(tmp_path):
    with pytest.raises(SystemExit, match="kite_link.py token"):
        run(["--board", "127.0.0.1:1", "hello"], tmp_path=tmp_path)


def test_parse_board():
    assert kite_link.parse_board("10.0.0.5", 8765) == ("10.0.0.5", 8765)
    assert kite_link.parse_board("kite-unoq:9000", 8765) == ("kite-unoq", 9000)
    assert kite_link.parse_board("[fd7a::1]:9000", 8765) == ("fd7a::1", 9000)


# -------------------------------------------------------------------------------------- discovery

def test_discovery_picks_the_newest_signed_heartbeat(spacetime, token):
    now = time.time()
    spacetime.insert("heartbeat", "unoq-test", heartbeat(token, ["10.0.0.1"], 8765, n=1), t=now - 60)
    spacetime.insert("heartbeat", "unoq-test", heartbeat(token, ["10.0.0.2"], 8765, n=2), t=now - 5)
    info = kite_link.find_heartbeat(kite_common.Spacetime(spacetime.url), "unoq-test", token, now)
    assert info["valid"] and info["hb"]["ips"] == ["10.0.0.2"] and 4 < info["age_s"] < 6


def test_discovery_ignores_an_impostor_that_overwrote_latest(spacetime, token):
    now = time.time()
    spacetime.insert("heartbeat", "unoq-test", heartbeat(token, ["10.0.0.2"], 8765), t=now - 3)
    spacetime.insert("heartbeat", "unoq-test", {"msg": "hello", "ips": ["6.6.6.6"], "port": 80, "sig": "00"}, t=now)
    spacetime.insert("heartbeat", "unoq-test", heartbeat("someone-elses-token-xx", ["6.6.6.7"], 80), t=now)
    info = kite_link.find_heartbeat(kite_common.Spacetime(spacetime.url), "unoq-test", token, now)
    assert info["valid"] and info["hb"]["ips"] == ["10.0.0.2"] and info["unsigned"] == 2


def test_discovery_with_nothing_signed(spacetime, token):
    spacetime.insert("heartbeat", "unoq-test", heartbeat("other-token-other-token", ["6.6.6.6"], 80))
    info = kite_link.find_heartbeat(kite_common.Spacetime(spacetime.url), "unoq-test", token)
    assert info["found"] and not info["valid"]
    info = kite_link.find_heartbeat(kite_common.Spacetime(spacetime.url), "nobody-here", token)
    assert not info["found"]


def test_board_heartbeat_is_found_and_used(make_board, spacetime, token):
    b = make_board(token, spacetime)
    assert wait_for(lambda: any(r["type"] == "heartbeat" for r in spacetime.events))
    rc, out = run(["where"], spacetime, token)
    assert rc == 0 and f"board: 127.0.0.1:{b.port} (direct)" in out
    rc, out = run(["send", "via", "discovery"], spacetime, token)
    assert rc == 0 and f"via direct 127.0.0.1:{b.port}" in out and 'got "via discovery"' in out


def test_candidates_order(token):
    a = kite_link.build_parser().parse_args([f"--token={token}", "hello"])
    link = kite_link.Link(a)
    link._hb = {"valid": True, "hb": {"ips": ["10.0.0.9", "10.0.0.9"], "tailscale": "100.64.1.2", "port": 9000}}
    assert link.candidates() == [("direct", "10.0.0.9", 9000), ("tailscale", "100.64.1.2", 9000),
                                 ("tailscale name", "kite-unoq", 9000)]


# ------------------------------------------------------------------------------- text and fallback

def test_text_direct_unicode(make_board, token):
    b = make_board(token, None)
    rc, out = run(["--board", f"127.0.0.1:{b.port}", "send", "ünï", "✓", "世界"], token=token)
    assert rc == 0 and 'got "ünï ✓ 世界"' in out


def test_long_text_is_refused_before_sending(make_board, token):
    b = make_board(token, None)
    rc, out = run(["--board", f"127.0.0.1:{b.port}", "send", "x" * 501], token=token)
    assert rc == 1 and "longer than 500" in out and b.board.received == 0


def test_text_falls_back_to_spacetimedb_when_direct_is_blocked(make_board, spacetime, token):
    b = make_board(token, spacetime)
    assert wait_for(lambda: b.board.cloud and spacetime.calls > 2)
    time.sleep(1.2)                                       # the board's inbox has read its starting point
    rc, out = run(["--board", "127.0.0.1:1", "send", "through", "the", "internet ✓"], spacetime, token)
    assert rc == 0 and "via internet (SpacetimeDB)" in out and 'got "through the internet ✓"' in out
    assert b.board.received == 1


def test_cloud_commands_must_be_signed_fresh_and_unreplayed(make_board, spacetime, token):
    b = make_board(token, spacetime, KITE_CLOUD="0")      # drive the inbox by hand
    b.board.cloud = kite_common.Spacetime(spacetime.url)
    seen: dict = {}
    now = time.time()

    def row(payload, t=now):
        return {"id": 1, "payload": json.dumps(payload), "t": [int(t * 1e6)]}

    good = kite_common.sign(token, {"to": "unoq-test", "cmd": "send", "text": "hi", "id": "abc", "ts": int(now * 1000)})
    assert b.board.handle_cloud_row(row(good), seen)["reply"] == 'got "hi"'
    assert b.board.handle_cloud_row(row(good), seen) is None                       # replayed
    forged = dict(good, text="rm", id="abd")
    assert b.board.handle_cloud_row(row(forged), seen) is None                     # bad signature
    stale = kite_common.sign(token, dict(good, id="old", ts=int((now - 600) * 1000)))
    assert b.board.handle_cloud_row(row(stale), seen) is None                      # too old
    other = kite_common.sign(token, dict(good, id="x2", to="other-board"))
    assert b.board.handle_cloud_row(row(other), seen) is None                      # not for us
    bad = kite_common.sign(token, dict(good, id="x3", cmd="reboot"))
    assert b.board.handle_cloud_row(row(bad), seen)["ok"] is False                 # answered with an error
    assert b.board.received == 1
    replies = [json.loads(r["payload"]) for r in spacetime.events if r["source"] == "unoq-test.board"]
    assert all(kite_common.verify(token, r) for r in replies) and len(replies) == 2


def test_cloud_command_times_out_cleanly(spacetime, token):
    with pytest.raises(TimeoutError, match="no reply"):
        kite_link.cloud_command(kite_common.Spacetime(spacetime.url), "unoq-test", token, "ping", wait_s=1.2,
                                poll_s=0.3)


def test_hello_status_ping_direct(make_board, token):
    b = make_board(token, None)
    board = ["--board", f"127.0.0.1:{b.port}"]
    rc, out = run(board + ["hello"], token=token)
    assert rc == 0 and "hello from" in out
    rc, out = run(board + ["status"], token=token)
    assert rc == 0 and '"camera"' in out
    rc, out = run(board + ["ping", "-n", "2"], token=token)
    assert rc == 0 and "round trip min" in out


def test_wrong_token_is_reported(make_board, token):
    b = make_board(token, None)
    with pytest.raises(SystemExit, match="401"):
        run(["--board", f"127.0.0.1:{b.port}", "hello"], token="wrong-" + token)


# ---------------------------------------------------------------------------------- listen, watch

def test_listen_direct_counts_in_order(make_board, token):
    b = make_board(token, None)
    rc, out = run(["--board", f"127.0.0.1:{b.port}", "listen", "--count", "3"], token=token)
    ns = [int(line.split("hello ")[1].split()[0]) for line in out.splitlines() if "direct   hello" in line]
    assert rc == 0 and len(ns) == 3 and ns == sorted(ns)


def test_listen_through_spacetimedb(make_board, spacetime, token):
    make_board(token, spacetime)
    rc, out = run(["--cloud", "listen", "--count", "2", "--seconds", "10"], spacetime, token)
    assert rc == 0 and out.count("internet hello") >= 2


def test_listen_reconnects_after_a_board_restart(make_board, token):
    port = free_port()
    first = make_board(token, None, port=port)
    result = {}
    t = threading.Thread(target=lambda: result.update(zip(("rc", "out"), run(
        ["--board", f"127.0.0.1:{port}", "listen", "--seconds", "5"], token=token))))
    t.start()
    time.sleep(1.5)
    first.stop()
    time.sleep(0.8)
    make_board(token, None, port=port)
    t.join(15)
    lines = result["out"].splitlines()
    i = next(i for i, line in enumerate(lines) if "reconnecting" in line)
    after = [line for line in lines[i:] if "direct   hello" in line]
    assert any("direct   hello" in line for line in lines[:i]) and len(after) >= 3


def test_watch_saves_decodable_frames_and_reconnects(make_board, token, tmp_path):
    port = free_port()
    first = make_board(token, None, port=port)
    result = {}
    t = threading.Thread(target=lambda: result.update(zip(("rc", "out"), run(
        ["--board", f"127.0.0.1:{port}", "watch", "--save", str(tmp_path / "f"), "--seconds", "5"], token=token))))
    t.start()
    time.sleep(1.5)
    first.stop()
    time.sleep(0.5)
    make_board(token, None, port=port)
    t.join(20)
    files = sorted((tmp_path / "f").glob("*.jpg"))
    assert result["rc"] == 0 and len(files) >= 25 and "reconnect" in result["out"]
    Image = pytest.importorskip("PIL.Image")
    for f in files:
        Image.open(f).load()


def test_watch_without_a_direct_path_says_why(token):
    rc, out = run(["--board", "127.0.0.1:1", "watch", "--save", "x"], token=token)
    assert rc == 2 and "eduroam blocks device-to-device" in out and "Tailscale" in out


def test_watch_opens_the_browser(make_board, token, monkeypatch):
    b = make_board(token, None)
    opened = []
    monkeypatch.setattr(kite_link.webbrowser, "open", lambda url: opened.append(url) or True)
    rc, out = run(["--board", f"127.0.0.1:{b.port}", "watch"], token=token)
    assert rc == 0 and opened[0].startswith(f"http://127.0.0.1:{b.port}/view?token=") and token not in out


# ----------------------------------------------------------------------------------------- doctor

def test_doctor_all_good(make_board, spacetime, token):
    b = make_board(token, spacetime)
    assert wait_for(lambda: any(r["type"] == "heartbeat" for r in spacetime.events))
    time.sleep(1.2)
    rc, out = run(["doctor", "--cloud-wait", "8"], spacetime, token)
    assert rc == 0, out
    for step in ("token", "heartbeat", "reach", "health", "token check", "text (direct)", "snapshot", "video",
                 "text (internet)"):
        assert f"PASS  {step}:" in out, step
    assert f"127.0.0.1:{b.port}" in out
    assert "WARN  camera: no camera" in out                    # the synthetic camera is reported, not hidden


def test_doctor_unreachable(token):
    rc, out = run(["--board", "127.0.0.1:1", "doctor", "--skip-cloud"], token=token)
    assert rc == 1 and "FAIL  reach: TCP 127.0.0.1:1" in out and "eduroam blocks device-to-device" in out


def test_doctor_wrong_token(make_board, token):
    b = make_board(token, None)
    rc, out = run(["--board", f"127.0.0.1:{b.port}", "doctor", "--skip-cloud"], token="wrong-" + token)
    assert rc == 1 and "FAIL  token check" in out and "token mismatch" in out


def test_doctor_no_heartbeat(spacetime, token):
    rc, out = run(["doctor", "--skip-cloud"], spacetime, token)
    assert rc == 1 and "FAIL  heartbeat: no heartbeat from 'unoq-test'" in out and "systemctl status kite-link" in out


def test_doctor_stale_heartbeat(spacetime, token):
    spacetime.insert("heartbeat", "unoq-test", heartbeat(token, ["127.0.0.1"], 1), t=time.time() - 300)
    rc, out = run(["doctor", "--skip-cloud"], spacetime, token)
    assert rc == 1 and "FAIL  heartbeat: last one 300 s ago" in out


def test_doctor_heartbeat_with_other_token(spacetime, token):
    spacetime.insert("heartbeat", "unoq-test", heartbeat("other-token-other-token", ["127.0.0.1"], 1))
    rc, out = run(["doctor", "--skip-cloud"], spacetime, token)
    assert rc == 1 and "none signed with your token" in out


def test_doctor_no_internet(token):
    class Offline:
        url = "http://127.0.0.1:1"
    rc, out = run(["doctor", "--skip-cloud"], Offline, token)
    assert rc == 1 and "FAIL  internet: can't reach SpacetimeDB" in out


def test_doctor_no_camera_is_a_warning_and_a_real_camera_passes(make_board, token, monkeypatch):
    b = make_board(token, None)
    rc, out = run(["--board", f"127.0.0.1:{b.port}", "doctor", "--skip-cloud"], token=token)
    assert rc == 0 and "WARN  camera: no camera: the board streams its test pattern" in out
    assert "powered USB-C hub" in out
    b.board.hub.camera.name = "usb camera /dev/video2"
    rc, out = run(["--board", f"127.0.0.1:{b.port}", "doctor", "--skip-cloud"], token=token)
    assert rc == 0 and "PASS  camera: usb camera /dev/video2" in out


def test_doctor_cloud_path_fails_when_board_is_offline(spacetime, token):
    rc, out = run(["--board", "127.0.0.1:1", "doctor", "--cloud-wait", "1.5"], spacetime, token)
    assert rc == 1 and "FAIL  text (internet)" in out
