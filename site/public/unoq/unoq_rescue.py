#!/usr/bin/env python3
"""Get back into the UNO Q over its USB-C cable when nobody knows its Linux password. Standard library only.

    python unoq_rescue.py                 set a new password, save the board's token here, check the camera
    python unoq_rescue.py --check         only report: who adb logs in as, the token, the camera, kite-link

Plug the board into this computer with a USB-C cable that carries data, and close Arduino App Lab (it runs its own
adb). Nothing on the board is erased or reflashed. Afterwards unplug it from this computer: its one USB-C port
goes back to the camera hub, and kite-link switches the port to host mode at every start.

How it gets in without the old password:
  1. `adb shell` over USB needs no password once the board's first setup is done; it logs in as `arduino`.
  2. To change the password it needs root, tried in this order: adb already root, `sudo` without a password,
     the factory password `arduino`, then Docker. App Lab runs its apps in Docker, so the `arduino` user can
     start a container, and a container that mounts the board's disk sets the password with the board's own
     `chpasswd`.
  3. The new password goes to the board in a file that only `arduino` can read, deleted straight after. It is
     never on a command line and never printed.

The token in /etc/kite/token belongs to `arduino`, so it is read and saved to ~/.kite_unoq_token with no root.
If adb is missing, Google's platform-tools are downloaded once into ~/.kite/platform-tools.
"""

from __future__ import annotations

import argparse
import getpass
import glob
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

TOKEN_FILE = Path.home() / ".kite_unoq_token"
TOOLS_DIR = Path.home() / ".kite" / "platform-tools"
PLATFORM_TOOLS = "https://dl.google.com/android/repository/platform-tools-latest-{}.zip"
FACTORY_PASSWORD = "arduino"
FALLBACK_IMAGE = "busybox:stable"
RC = "__KITE_RC="
CODEC = re.compile(r"venus|codec|encoder|decoder", re.I)   # the SoC's video codecs, not cameras
REFLASH = "https://docs.arduino.cc/tutorials/uno-q/update-image/"


class Fail(SystemExit):
    pass


# ---------------------------------------------------------------------------------------------- adb

def adb_candidates(env=os.environ, system: str = "") -> list[str]:
    """Where adb usually lives: PATH, Android Studio, winget, Homebrew, App Lab, and our own download."""
    system = system or platform.system()
    exe = "adb.exe" if system == "Windows" else "adb"
    found = [shutil.which("adb") or ""]
    home = Path(env.get("USERPROFILE") or env.get("HOME") or Path.home())
    local = Path(env.get("LOCALAPPDATA", home / "AppData" / "Local"))
    found += [str(TOOLS_DIR / exe), str(local / "Android" / "Sdk" / "platform-tools" / exe),
              str(home / "Library" / "Android" / "sdk" / "platform-tools" / exe),
              str(home / "Android" / "Sdk" / "platform-tools" / exe), "/opt/homebrew/bin/adb", "/usr/local/bin/adb"]
    for pattern in (local / "Microsoft" / "WinGet" / "Packages" / "Google.PlatformTools*" / "platform-tools" / exe,
                    local / "Programs" / "*[Aa]pp*[Ll]ab*" / "**" / exe,
                    Path("/Applications") / "*App Lab*.app" / "**" / exe):
        found += sorted(glob.glob(str(pattern), recursive=True))
    out: list[str] = []
    for f in found:
        if f and f not in out and os.path.isfile(f):
            out.append(f)
    return out


def download_adb(system: str = "", out=print) -> str:
    system = system or platform.system()
    flavour = {"Windows": "windows", "Darwin": "darwin"}.get(system, "linux")
    TOOLS_DIR.parent.mkdir(parents=True, exist_ok=True)
    out(f"adb not found: downloading Google's platform-tools into {TOOLS_DIR.parent} (about 7 MB, once)")
    with tempfile.TemporaryDirectory() as tmp:
        z = Path(tmp) / "pt.zip"
        urllib.request.urlretrieve(PLATFORM_TOOLS.format(flavour), z)  # noqa: S310  # nosec B310 - fixed https URL
        with zipfile.ZipFile(z) as zf:
            zf.extractall(TOOLS_DIR.parent)                              # the zip holds platform-tools/
    exe = TOOLS_DIR / ("adb.exe" if system == "Windows" else "adb")
    if not exe.is_file():
        raise Fail("downloaded platform-tools, but adb isn't in it; install it by hand: winget install Google.PlatformTools")
    exe.chmod(0o755)
    return str(exe)


class Adb:
    """One board over adb. `runner` is swappable so the tests need no board."""

    def __init__(self, exe: str, serial: str = "", runner=subprocess.run) -> None:
        self.exe, self.serial, self.runner = exe, serial, runner

    def _base(self) -> list[str]:
        return [self.exe] + (["-s", self.serial] if self.serial else [])

    def raw(self, *args: str, timeout: float = 60) -> tuple[int, str]:
        try:
            p = self.runner(self._base() + list(args), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "timed out"
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    def sh(self, script: str, timeout: float = 120) -> tuple[int, str]:
        """Run a shell line on the board. The exit code comes back in a marker line, so it survives adb
        versions that don't pass it through."""
        rc, text = self.raw("shell", f"{script}; echo {RC}$?", timeout=timeout)
        lines = text.replace("\r\n", "\n").rstrip("\n").split("\n")
        if lines and lines[-1].startswith(RC):
            try:
                return int(lines[-1][len(RC):]), "\n".join(lines[:-1]).strip()
            except ValueError:
                pass
        return (rc or 1), text.strip()

    def push_secret(self, content: str) -> str:
        """Put `content` in a file only the adb user can read, and return its path on the board."""
        remote = f"/tmp/.kite-{secrets.token_hex(6)}"  # nosec B108 - random name, pushed as 0600, removed after use
        fd, local = tempfile.mkstemp(prefix="kite-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            rc, text = self.raw("push", local, remote)
            if rc != 0:
                raise Fail(f"couldn't copy to the board over adb: {text.strip()}")
        finally:
            try:
                os.remove(local)
            except OSError:
                pass
        self.sh(f"chmod 600 {remote}")
        return remote


def list_devices(adb: Adb) -> list[list[str]]:
    rc, text = adb.raw("devices", timeout=30)
    if rc != 0:
        raise Fail(f"adb didn't start: {text.strip()}\nClose Arduino App Lab (it runs its own adb) and try again.")
    return [ln.split() for ln in text.splitlines()[1:] if ln.strip() and not ln.startswith("*")]


def pick_device(adb: Adb) -> str:
    rows = list_devices(adb)
    if not rows:                    # an adb server started before the board was plugged in can miss it
        adb.raw("kill-server", timeout=30)
        adb.raw("start-server", timeout=30)
        rows = list_devices(adb)
    ready = [r[0] for r in rows if len(r) >= 2 and r[1] == "device"]
    if adb.serial:
        if adb.serial not in ready:
            raise Fail(f"device {adb.serial} isn't ready over adb (found: {rows or 'nothing'})")
        return adb.serial
    if not rows:
        raise Fail("no board over USB. Use a USB-C cable that carries data (not charge-only), plug it straight "
                   "into this computer, wait 30 s after power-on, and close Arduino App Lab.")
    if not ready:
        states = ", ".join(f"{r[0]} is {' '.join(r[1:])}" for r in rows)
        raise Fail(f"the board is there but not ready: {states}. 'unauthorized' or 'no permissions': unplug, "
                   "plug back in, and run this again.")
    if len(ready) > 1:
        raise Fail(f"more than one device over adb ({', '.join(ready)}); pick one with --serial")
    return ready[0]


# ------------------------------------------------------------------------------------------- root

def find_root(adb: Adb, out=print) -> tuple[str, str]:
    """How to run a command as root: (route, prefix). The prefix reads a password file on stdin if needed."""
    rc, uid = adb.sh("id -u")
    if rc == 0 and uid.strip() == "0":
        return "adb is root", ""
    if adb.sh("sudo -n true 2>/dev/null")[0] == 0:
        return "sudo without a password", "sudo -n"
    f = adb.push_secret(FACTORY_PASSWORD + "\n")
    try:
        if adb.sh(f"sudo -S -k -p '' true < {f} 2>/dev/null")[0] == 0:
            return "the factory password", "factory"
    finally:
        adb.sh(f"rm -f {f}")
    if adb.sh("docker info >/dev/null 2>&1")[0] == 0:
        return "Docker (App Lab's container runtime)", "docker"
    rc, groups = adb.sh("id -Gn")
    raise Fail("this board gives no way to root without the old password "
               f"(adb user's groups: {groups or 'unknown'}; no passwordless sudo; no Docker access).\n"
               f"The only way left is reflashing it, which erases it: {REFLASH}\nNothing was changed.")


def docker_image(adb: Adb, out=print) -> str:
    """An image already on the board (App Lab's), so this works offline; else a tiny one from Docker Hub."""
    rc, text = adb.sh("docker images --format '{{.Repository}}:{{.Tag}}'")
    images = [i for i in text.split() if rc == 0 and ":" in i and "<none>" not in i]
    for want in ("busybox", "debian", "python", "ubuntu", "alpine"):
        for i in images:
            if want in i:
                return i
    if images:
        return images[0]
    out(f"no container image on the board yet: pulling {FALLBACK_IMAGE} (needs the board online)")
    rc, text = adb.sh(f"docker pull {FALLBACK_IMAGE}", timeout=300)
    if rc != 0:
        raise Fail(f"couldn't get a container image: {text[-300:]}\nPut the board on Wi-Fi and try again.")
    return FALLBACK_IMAGE


def as_root(adb: Adb, route: str, command: str, stdin_file: str, image: str = "", password_file: str = "") -> tuple[int, str]:
    """Run `command` as root on the board, with `stdin_file` on its standard input."""
    if route == "":
        return adb.sh(f"{command} < {stdin_file}")
    if route == "sudo -n":
        return adb.sh(f"sudo -n {command} < {stdin_file}")
    if route == "factory":
        # sudo reads the factory password first, then the command reads the rest of stdin
        return adb.sh(f"{{ echo {FACTORY_PASSWORD}; cat {stdin_file}; }} | sudo -S -k -p '' {command}")
    # Docker: the container mounts the board's disk and runs the board's own binary in it
    return adb.sh(f"docker run --rm -i --user 0:0 --network none -v /:/host --entrypoint chroot {image} "
                  f"/host {command} < {stdin_file}", timeout=180)


# ------------------------------------------------------------------------------------------- steps

def ask_password(prompt=getpass.getpass, out=print) -> str:
    for _ in range(3):
        pw = prompt("New board password (hidden): ")
        if len(pw) < 8 or "\n" in pw or "\r" in pw:
            out("use at least 8 characters, on one line")
            continue
        if prompt("Same again: ") != pw:
            out("those didn't match")
            continue
        return pw
    raise Fail("no password set; nothing changed")


def set_password(adb: Adb, user: str, pw: str, route: str, prefix: str, out=print) -> None:
    image = docker_image(adb, out) if prefix == "docker" else ""
    f = adb.push_secret(f"{user}:{pw}\n")
    try:
        rc, text = as_root(adb, prefix, "/usr/sbin/chpasswd", f, image)
        if rc != 0:
            raise Fail(f"setting the password through {route} failed: {text[-400:]}\nNothing was changed.")
    finally:
        adb.sh(f"rm -f {f}")
    f = adb.push_secret(pw + "\n")
    try:
        ok = adb.sh(f"sudo -S -k -p '' true < {f} 2>/dev/null")[0] == 0
    finally:
        adb.sh(f"rm -f {f}")
    if not ok:
        _, groups = adb.sh("id -Gn")
        out(f"WARN  new password set for '{user}' (through {route}), but sudo doesn't accept it: '{user}' may not "
            f"be allowed to use sudo (groups: {groups.strip()}). App Lab and ssh use the new password.")
        return
    out(f"PASS  new password set for '{user}' (through {route}), and sudo accepts it")
    out("      write it down now: App Lab, ssh and sudo all use it")


def save_token(adb: Adb, path: Path = TOKEN_FILE, out=print) -> bool:
    rc, tok = adb.sh("cat /etc/kite/token 2>/dev/null")
    tok = tok.strip()
    if rc != 0 or len(tok) < 16 or any(c.isspace() for c in tok):
        out("WARN  no Kite token on the board (kite-link isn't installed). Once it is: "
            "curl -fsSL https://trykite.pages.dev/unoq/install.sh | sh")
        return False
    old = path.read_text(encoding="utf-8", errors="replace").strip() if path.is_file() else ""
    path.write_text(tok + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    out(f"PASS  token saved to {path}" + (" (it matched the one already here)" if old == tok else
                                            " (it was missing or different here: that was the 401)" if old else ""))
    return True


def report(adb: Adb, out=print) -> str:
    _, user = adb.sh("id -un")
    user = user.strip() or "arduino"
    _, host = adb.sh("hostname")
    out(f"PASS  adb shell on '{host.strip()}' as '{user}', no password needed")
    _, active = adb.sh("systemctl is-active kite-link 2>/dev/null")
    out(("PASS" if active.strip() == "active" else "WARN") + f"  kite-link service: {active.strip() or 'not installed'}")
    _, names = adb.sh("cat /sys/class/video4linux/*/name 2>/dev/null")
    cams = [n.strip() for n in names.splitlines() if n.strip() and not CODEC.search(n)]
    if cams:
        out(f"PASS  camera: {', '.join(dict.fromkeys(cams))}")
    else:
        out("INFO  no camera now, as expected while the board's one USB-C port is on this computer. Afterwards: "
            "unplug it here, webcam into a POWERED USB-C hub, hub into the board, power the board up")
    return user


def main(argv=None, out=print, prompt=getpass.getpass, runner=subprocess.run) -> int:
    ap = argparse.ArgumentParser(description="Reset the UNO Q's password and save its token, over USB.")
    ap.add_argument("--check", action="store_true", help="only report; change nothing")
    ap.add_argument("--serial", default="", help="adb serial, if more than one device is plugged in")
    ap.add_argument("--adb", default="", help="path to adb")
    ap.add_argument("--user", default="", help="whose password to set (default: the adb user, or arduino if that is root)")
    ap.add_argument("--token-file", type=Path, default=TOKEN_FILE)
    a = ap.parse_args(argv)
    try:
        exe = a.adb or next(iter(adb_candidates()), "") or download_adb(out=out)
        adb = Adb(exe, a.serial, runner)
        adb.serial = pick_device(adb)
        user = report(adb, out)
        user = a.user or ("arduino" if user == "root" else user)
        save_token(adb, a.token_file, out)
        if a.check:
            route, _ = find_root(adb, out)
            out(f"PASS  a new password can be set through {route}: run this again without --check")
            return 0
        route, prefix = find_root(adb, out)
        pw = ask_password(prompt, out)
        set_password(adb, user, pw, route, prefix, out)
        out("next: unplug the board from this computer and plug the camera hub back in; then python kite_link.py doctor")
        return 0
    except Fail as e:
        out(f"FAIL  {e.code}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
