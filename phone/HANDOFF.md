# Phone link: handoff (work in progress)

Started by Karthik on Oct 24; continued by Eddy. This is unfinished: read this first.

## Goal
The walker's phone sends GPS and compass heading (and optionally chest-camera frames) to the computer, which
forwards them to the earpiece as `position` events (`earpiece/earpiece/events.py`). Blind-first: the iPhone
**Action Button** opens it with a sound, and one tap anywhere connects.

## Decided
- **Primary: Wi-Fi to the Windows computer, plain Safari or Chrome.** The Windows computer stays on the Tello Wi-Fi
  and runs **Windows Mobile Hotspot** at the same time (check `netsh wlan show drivers`; a USB Wi-Fi adapter if the
  built-in one can't do both). The phone joins that hotspot and opens the bridge's **HTTPS** page (Safari only
  allows GPS, compass and camera on secure pages, so the bridge serves a local CA the phone installs once as a
  profile: Settings > General > VPN & Device Management, then Settings > General > About > Certificate Trust
  Settings). Windows Mobile Hotspot usually gives itself 192.168.137.1.
- **Forward to the earpiece:** the earpiece runs on the Mac (macOS-only audio and AirPods controls). The Mac joins
  the Windows hotspot; the bridge forwards `position` (with heading) over UDP to `KITE_EARPIECE_HOST`, port 5010,
  the same protocol as `earpiece/earpiece/client.py`. Start the earpiece with `--bind 0.0.0.0`.
- **Fallbacks:** Bluetooth LE via the free **Bluefy** iOS browser (Safari has no Web Bluetooth) or Android Chrome,
  to `python -m phone.ble` on the computer; or a USB cable (iPhone Personal Hotspot over USB, which also gives the
  computer internet).
- **Action Button:** a Shortcut that first **Speaks** "Kite opening" (Shortcuts can always make sound; web pages
  only after a tap), then opens the HTTPS URL in Safari. The page is one full-screen Connect target, announces status
  with `aria-live` and speechSynthesis after the tap (which unlocks audio), and uses distinct sounds for connected,
  lost and reconnected. iOS web pages have no haptics (`navigator.vibrate` is unsupported).
- **Internet:** while a computer is on the Tello Wi-Fi it has none; Cartesia and Claude need it. Options: the Mac on a
  phone's USB tether, or Windows sharing a cellular link. Not settled.
- **Arduino UNO Q** (`hardware/arduino/`) talks to the Windows computer over USB serial; not part of this link yet.

## Done in this branch (compiles; untested)
| File | What |
|---|---|
| `__init__.py` | Entry points, GATT service/characteristic UUIDs, paths (`phone/.state/` holds the CA and token, git-ignored) |
| `fix.py` | Parse the phone's JSON fix `{t, lat, lng, acc, alt, speed, course, heading, heading_acc}`, turn it into a `position` event, and format the live terminal line (position, accuracy, compass with cardinal, speed, ages) |
| `ble.py` | `python -m phone.ble`: a BLE GATT peripheral with `bless` that receives fixes and prints the live line |
| `certs.py` | Local CA + leaf certificate for the laptop's IPs, iOS `.mobileconfig` profile, SSL context |
| `netinfo.py` | Interface addresses from `ifconfig` (macOS) and `ipconfig` (Windows), ranking, Tello-Wi-Fi detection, routes |
| `link.py`, `earpiece_watch.py` | Sending to the earpiece and watching that it's alive |

## Still to do
1. The HTTPS + WebSocket **bridge server** for the Wi-Fi path (`python -m phone`, Windows-first: Proactor event loop,
   bind to the hotspot address, print a QR code and URL, firewall rule
   `New-NetFirewallRule -DisplayName "Kite phone" -Direction Inbound -Protocol TCP -LocalPort <port> -Action Allow`).
2. The **phone page** (Connect, `watchPosition`, compass via `webkitCompassHeading` after
   `DeviceOrientationEvent.requestPermission()`, Screen Wake Lock, reconnect, sounds). The docstring in `__init__.py`
   expected it at `site/src/phone.body.html` + `site/public/phone.js` (built by `site/tools/build_pages.py`) for the
   BLE path; for the Wi-Fi path the bridge must serve it itself over HTTPS (an https page can't talk to a `ws://` LAN
   address).
3. UDP forwarding to the earpiece on the Mac, plus a "lost the phone" status.
4. `docs/phone.md` (setup steps, the Windows hotspot check, the Shortcut steps, a 5-minute table checklist).
5. Tests: fix parsing, terminal line, cert generation, adapter detection with mocked `ipconfig`, UDP forwarding to a
   local socket, and a Playwright run with fake geolocation (no real Bluetooth, mic, camera or sound).
