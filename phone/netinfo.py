"""Which networks this laptop is on, read-only: never changes a setting.

    python -m phone.netinfo          # addresses, the phone's URL candidates, Tello Wi-Fi, default route, service order
"""

from __future__ import annotations

import ipaddress
import re
import socket
import subprocess
import sys

TELLO_IP = "192.168.10.1"
HOTSPOT_NET = ipaddress.ip_network("172.20.10.0/28")    # iPhone Personal Hotspot (USB, Wi-Fi or Bluetooth)
WIN_HOTSPOT_NET = ipaddress.ip_network("192.168.137.0/24")  # Windows Mobile Hotspot / Internet Connection Sharing
TELLO_NET = ipaddress.ip_network("192.168.10.0/24")


def _parse_ifconfig(text: str) -> list[tuple[str, str]]:
    """macOS / BSD / Linux `ifconfig` output."""
    out, iface = [], None
    for line in text.splitlines():
        m = re.match(r"^([a-zA-Z0-9_.-]+):? (?:flags=|Link )", line)
        if m:
            iface = m.group(1).rstrip(":")
            continue
        m = re.match(r"^\s+inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)", line)
        if m and iface:
            out.append((iface, m.group(1)))
    return out


def _parse_ipconfig(text: str) -> list[tuple[str, str]]:
    """Windows `ipconfig` output (English; other languages still match the address lines)."""
    out, iface = [], None
    for line in text.splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith(":"):
            iface = line.strip().rstrip(":")
            continue
        m = re.match(r"^\s+IPv4[^:]*:\s*(\d+\.\d+\.\d+\.\d+)", line)
        if m:
            out.append((iface or "?", m.group(1)))
    return out


def interface_addresses(text: str | None = None, windows: bool | None = None) -> list[tuple[str, str]]:
    """[(interface, IPv4)] on this machine, loopback and link-local excluded. Works on macOS, Windows and Linux:
    psutil when it is installed, else `ifconfig` / `ipconfig`, else the address used for the default route."""
    windows = sys.platform == "win32" if windows is None else windows
    pairs: list[tuple[str, str]] = []
    if text is None:
        try:
            import psutil
            pairs = [(name, a.address) for name, addrs in psutil.net_if_addrs().items() for a in addrs
                     if a.family == socket.AF_INET]
        except ImportError:
            cmd = ["ipconfig"] if windows else (["/sbin/ifconfig"] if sys.platform == "darwin" else ["ifconfig"])
            text = _run(cmd)
    if text is not None:
        pairs = _parse_ipconfig(text) if windows else _parse_ifconfig(text)
    if not pairs:
        ip = local_ip_toward("8.8.8.8")
        pairs = [("default", ip)] if ip else []
    out = []
    for iface, ip in pairs:
        a = ipaddress.ip_address(ip)
        if not (a.is_loopback or a.is_link_local) and (iface, ip) not in out:
            out.append((iface, ip))
    return out


def rank(addrs: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Addresses the phone could reach, best first, with what each one is: the iPhone hotspot (USB), the Tello's
    Wi-Fi (the phone joined it too), then any other private network."""
    def kind(ip: str) -> tuple[int, str]:
        a = ipaddress.ip_address(ip)
        if a in HOTSPOT_NET:
            return 0, "iPhone Personal Hotspot"
        if a in WIN_HOTSPOT_NET:
            return 0, "Windows Mobile Hotspot"
        if a in TELLO_NET:
            return 1, "Tello Wi-Fi"
        if a.is_private:
            return 2, "local network"
        return 3, "public"
    return [(i, ip, kind(ip)[1]) for i, ip in sorted(addrs, key=lambda a: kind(a[1])[0])]


def local_ip_toward(host: str, port: int = 9) -> str | None:
    """The address this laptop would use to reach `host` (a UDP connect sends no packet)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, port))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def on_tello_wifi() -> bool:
    ip = local_ip_toward(TELLO_IP, 8889)
    return bool(ip) and ipaddress.ip_address(ip) in TELLO_NET


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def default_route() -> dict:
    if sys.platform != "darwin":
        ip = local_ip_toward("8.8.8.8")
        return {"interface": ip or "?", "gateway": "?"}
    text = _run(["/sbin/route", "-n", "get", "default"])
    return {k: v for k, v in re.findall(r"^\s*(gateway|interface): (\S+)", text, re.M)}


def service_order() -> list[str]:
    if sys.platform != "darwin":
        return []
    text = _run(["/usr/sbin/networksetup", "-listnetworkserviceorder"])
    # disabled services are listed as "(*) Name"; -ordernetworkservices needs every one of them
    return re.findall(r"^\((?:\d+|\*)\) (.+)$", text, re.M)


def main() -> int:
    addrs = interface_addresses()
    print("addresses the phone could reach (best first):")
    for iface, ip, what in rank(addrs) or []:
        print(f"  {iface:8} {ip:16} {what}")
    if not addrs:
        print("  none")
    print(f"Tello Wi-Fi joined: {'yes' if on_tello_wifi() else 'no'}")
    r = default_route()
    print(f"default route: {r.get('interface', '?')} via {r.get('gateway', '?')}  (internet goes this way)")
    order = service_order()
    if order:
        print("network service order: " + " > ".join(order))
        if "iPhone USB" in order and "Wi-Fi" in order and order.index("iPhone USB") > order.index("Wi-Fi"):
            print("  iPhone USB is below Wi-Fi: on the Tello's Wi-Fi the laptop would have no internet. To fix it once")
            print("  (docs/phone.md), run:")
            rest = [s for s in order if s != "iPhone USB"]
            print("    sudo networksetup -ordernetworkservices " + " ".join(f'"{s}"' for s in ["iPhone USB", *rest]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
