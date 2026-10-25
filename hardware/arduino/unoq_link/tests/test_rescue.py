"""unoq_rescue.py against a simulated board over a fake adb: every root route, the token, and that the new
password never reaches a command line or stays on the board."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import unoq_rescue as rescue

TOKEN = "0123456789abcdef0123456789abcdef"
NEW = "kite-walks-2026"


class Board:
    """What adb would do with the commands unoq_rescue sends, on a board in a chosen state."""

    def __init__(self, uid=1000, nopasswd=False, password="forgotten-pw", docker=True,
                 images=("ghcr.io/arduino/app-bricks/python-apps-base:0.5.0",), token=TOKEN, devices=("ABC123\tdevice",),
                 sudo_ok=True, cams=("qcom-venus-encoder", "qcom-venus-decoder")) -> None:
        self.uid, self.nopasswd, self.password, self.docker = uid, nopasswd, password, docker
        self.images, self.token, self.devices, self.sudo_ok = list(images), token, list(devices), sudo_ok
        self.files: dict[str, str] = {}
        self.argv: list[list[str]] = []
        self.cams = list(cams)
        self.restarts = 0
        self.pulled = False
        self.late: list[str] = []

    def __call__(self, argv, capture_output=True, text=True, timeout=None):
        self.argv.append(list(argv))
        args = argv[1:]
        if args[:1] == ["-s"]:
            args = args[2:]
        if args[0] in ("kill-server", "start-server"):
            self.restarts += 1
            if args[0] == "start-server" and self.late:
                self.devices = list(self.late)
            return self._done(0, "")
        if args[0] == "devices":
            return self._done(0, "List of devices attached\n" + "\n".join(self.devices) + "\n\n")
        if args[0] == "push":
            self.files[args[2]] = Path(args[1]).read_text(encoding="utf-8")
            return self._done(0, "1 file pushed")
        assert args[0] == "shell" and len(args) == 2
        script, marker = args[1].rsplit("; echo ", 1)
        assert marker == rescue.RC + "$?"
        rc, out = self.shell(script)
        return self._done(0, f"{out}\r\n{rescue.RC}{rc}\r\n" if out else f"{rescue.RC}{rc}\r\n")

    def _done(self, rc, out):
        return subprocess.CompletedProcess([], rc, out, "")

    def _file(self, script):
        m = re.search(r"(/tmp/\.kite-[0-9a-f]+)", script)
        return m.group(1) if m else ""

    def _sudo_pw_ok(self, given: str) -> bool:
        return self.sudo_ok and given.strip() == self.password

    def _chpasswd(self, f):
        user, pw = self.files[f].rstrip("\n").split(":", 1)
        assert user == "arduino"
        self.password = pw
        return 0, ""

    def shell(self, s: str):
        f = self._file(s)
        simple = {"id -u": str(self.uid), "id -un": "root" if self.uid == 0 else "arduino", "hostname": "kite-unoq",
                  "id -Gn": "arduino dialout video docker" if self.docker else "arduino dialout video",
                  "systemctl is-active kite-link 2>/dev/null": "active",
                  "cat /sys/class/video4linux/*/name 2>/dev/null": "\n".join(self.cams)}
        if s in simple:
            return 0, simple[s]
        if s.startswith(("chmod 600 ", "rm -f ")):
            if s.startswith("rm -f "):
                self.files.pop(f, None)
            return 0, ""
        if s == "sudo -n true 2>/dev/null":
            return (0 if self.nopasswd else 1), ""
        if s.startswith("sudo -S -k -p '' true < "):
            return (0 if self._sudo_pw_ok(self.files[f]) else 1), ""
        if s == "docker info >/dev/null 2>&1":
            return (0 if self.docker else 1), ""
        if s.startswith("docker images"):
            return 0, "\n".join(self.images)
        if s.startswith("docker pull "):
            self.pulled = True
            self.images.append(rescue.FALLBACK_IMAGE)
            return 0, ""
        if s.startswith("docker run "):
            assert self.docker and "--user 0:0" in s and "-v /:/host" in s and "--network none" in s
            assert s.endswith(f"/host /usr/sbin/chpasswd < {f}")
            return self._chpasswd(f)
        if s.startswith("sudo -n /usr/sbin/chpasswd < "):
            assert self.nopasswd
            return self._chpasswd(f)
        if s.startswith("/usr/sbin/chpasswd < "):
            assert self.uid == 0
            return self._chpasswd(f)
        if s.startswith("{ echo arduino; cat "):
            if self.password != "arduino":
                return 1, "sudo: incorrect password"
            return self._chpasswd(f)
        if s == "cat /etc/kite/token 2>/dev/null":
            return (0, self.token) if self.token else (1, "")
        raise AssertionError(f"unexpected board command: {s}")


def run(board: Board, tmp_path, *extra, answers=(NEW, NEW)):
    out: list[str] = []
    asked = list(answers)
    rc = rescue.main(["--adb", "adb", "--token-file", str(tmp_path / "tok"), *extra], out=out.append,
                     prompt=lambda _p: asked.pop(0), runner=board)
    return rc, "\n".join(out)


def never_leaked(board: Board) -> None:
    assert not any(NEW in " ".join(a) for a in board.argv), "the password was on a command line"
    assert board.files == {}, "a password file was left on the board"


def test_docker_route_resets_the_password_and_saves_the_token(tmp_path):
    b = Board()
    rc, text = run(b, tmp_path)
    assert rc == 0, text
    assert b.password == NEW
    assert "through Docker" in text and "sudo accepts it" in text
    assert (tmp_path / "tok").read_text().strip() == TOKEN
    assert TOKEN not in text
    assert NEW not in text
    never_leaked(b)


def test_passwordless_sudo_is_used_before_docker(tmp_path):
    b = Board(nopasswd=True)
    rc, text = run(b, tmp_path)
    assert rc == 0 and b.password == NEW and "sudo without a password" in text
    assert not any("docker run" in " ".join(a) for a in b.argv)
    never_leaked(b)


def test_factory_password_route(tmp_path):
    b = Board(password="arduino", docker=False)
    rc, text = run(b, tmp_path)
    assert rc == 0 and b.password == NEW and "factory password" in text
    never_leaked(b)


def test_root_adb_route(tmp_path):
    b = Board(uid=0, docker=False)
    rc, text = run(b, tmp_path)
    assert rc == 0 and b.password == NEW and "adb is root" in text


def test_no_way_in_changes_nothing_and_never_asks(tmp_path):
    b = Board(docker=False)
    asked: list[str] = []
    out: list[str] = []
    rc = rescue.main(["--adb", "adb", "--token-file", str(tmp_path / "tok")], out=out.append,
                     prompt=lambda p: asked.append(p) or NEW, runner=b)
    text = "\n".join(out)
    assert rc == 1 and b.password == "forgotten-pw" and not asked
    assert "reflashing" in text and "Nothing was changed" in text
    assert (tmp_path / "tok").read_text().strip() == TOKEN          # the token is still recovered
    assert b.files == {}


def test_check_changes_nothing(tmp_path):
    b = Board()
    rc, text = run(b, tmp_path, "--check", answers=())
    assert rc == 0 and b.password == "forgotten-pw" and "can be set through Docker" in text


def test_board_missed_by_an_old_adb_server_is_found_after_a_restart(tmp_path):
    b = Board(devices=())
    b.late = ["ABC123\tdevice"]
    rc, text = run(b, tmp_path)
    assert rc == 0 and b.restarts == 2 and b.password == NEW


def test_the_socs_video_codecs_are_not_taken_for_a_camera(tmp_path):
    _, text = run(Board(), tmp_path)
    assert "no camera now" in text and "PASS  camera" not in text
    _, text = run(Board(cams=("qcom-venus-encoder", "UVC Camera (046d:0825)", "UVC Camera (046d:0825)")), tmp_path)
    assert "PASS  camera: UVC Camera (046d:0825)" in text


def test_pulls_a_tiny_image_when_the_board_has_none(tmp_path):
    b = Board(images=())
    rc, _ = run(b, tmp_path)
    assert rc == 0 and b.pulled and b.password == NEW


def test_sudo_refusing_the_new_password_is_a_warning(tmp_path):
    b = Board(sudo_ok=False)
    rc, text = run(b, tmp_path)
    assert rc == 0 and b.password == NEW and "WARN" in text and "may not be allowed to use sudo" in text


def test_mismatched_or_short_passwords_change_nothing(tmp_path):
    b = Board()
    rc, text = run(b, tmp_path, answers=("short", NEW, "other-one-x", NEW, "nope-nope-1"))
    assert rc == 1 and b.password == "forgotten-pw" and "nothing changed" in text and b.files == {}


@pytest.mark.parametrize("devices,needle", [((), "carries data"), (("ABC123\tunauthorized",), "unauthorized"),
                                             (("A\tdevice", "B\tdevice"), "--serial")])
def test_device_problems_are_explained(tmp_path, devices, needle):
    rc, text = run(Board(devices=devices), tmp_path, answers=())
    assert rc == 1 and needle in text


def test_missing_token_is_a_warning_not_a_failure(tmp_path):
    rc, text = run(Board(token=""), tmp_path)
    assert rc == 0 and "no Kite token" in text and not (tmp_path / "tok").exists()


def test_exit_code_marker_survives_windows_line_endings():
    adb = rescue.Adb("adb", runner=lambda *a, **k: subprocess.CompletedProcess([], 0, f"hi\r\n{rescue.RC}3\r\n", ""))
    assert adb.sh("x") == (3, "hi")


def test_finds_adb_installed_by_winget(tmp_path):
    exe = tmp_path / "Microsoft" / "WinGet" / "Packages" / "Google.PlatformTools_x" / "platform-tools" / "adb.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    found = rescue.adb_candidates({"LOCALAPPDATA": str(tmp_path), "USERPROFILE": str(tmp_path)}, system="Windows")
    assert str(exe) in found
