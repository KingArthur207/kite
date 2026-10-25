# UNO Q link: text and video between the board and the computer

The UNO Q (on a battery, on Wi-Fi) and the computer exchange text both ways, and the board streams its USB
camera to the computer. No processing. The computer finds the board by itself: no IP to type.

## Board (once)

1. **App Lab wizard** (USB-C cable): name the board `kite-unoq`, set a password, give it any Wi-Fi with internet.
2. **Paste one line** into App Lab's terminal (or `ssh arduino@kite-unoq.local`):

   ```
   curl -fsSL https://trykite.pages.dev/unoq/install.sh | sh
   ```

   It installs the `kite-link` service (starts at every power-on), prints the board's **token**, and asks
   whether to join eduroam (username, password, school domain).
3. **For video:** camera and battery go into a **USB-C hub with power pass-through (PD in)**, and the hub into
   the board's only USB-C port. Then `sudo systemctl restart kite-link`. With no camera the board streams a
   labelled test pattern.

## Computer (Windows, macOS or Linux; Python 3.9+, nothing to install)

From the repo folder `hardware/arduino/unoq_link/` (or download the two files:
`curl.exe -O https://trykite.pages.dev/unoq/kite_link.py -O https://trykite.pages.dev/unoq/kite_common.py`):

```
python kite_link.py token <token the board printed>     once
python kite_link.py doctor                               checks every step, says what is wrong
python kite_link.py send hello                           the board logs it and replies
python kite_link.py listen                               the board's hello, every second
python kite_link.py watch                                the camera, in your browser
python kite_link.py watch --save frames --seconds 10     or saved as JPEG files
```

Also `hello`, `ping`, `status`, `where`. `--board IP[:PORT]` skips discovery.

## How it connects

| Order | Path | Text | Video |
|---|---|---|---|
| 1 | Direct to the board's Wi-Fi IP, port 8765 (found through the board's signed heartbeat in SpacetimeDB) | yes | yes |
| 2 | Tailscale (`kite-unoq`), if set up | yes | yes |
| 3 | Through SpacetimeDB (signed messages, 2 to 4 s) | yes | no |

Everything needs the token: HTTP calls send it, and SpacetimeDB messages are signed with it (the tables are
public). eduroam often blocks device-to-device traffic; then text still works through 3, and `doctor` says so.

## Plan B: Tailscale (when eduroam blocks the direct path)

Tailscale reaches between networks and relays over HTTPS (TCP 443) when direct traffic is blocked.

- Board: `curl -fsSL https://tailscale.com/install.sh | sh`, then `sudo tailscale up --hostname kite-unoq`
  and open the printed link to sign in.
- Windows: install from https://tailscale.com/download and sign in with **the same account**.

Nothing else changes: the heartbeat then carries the Tailscale IP, and `kite_link.py` tries it.

## Troubleshooting

| `doctor` says | Do |
|---|---|
| `no heartbeat` | Board off, or no internet. On the board: `systemctl status kite-link`, `nmcli dev status`. |
| `FAIL reach ... eduroam blocks device-to-device` | Plan B (Tailscale), or put both on a phone hotspot. |
| `token mismatch` / `401` | On the board `cat /etc/kite/token`; here `python kite_link.py token <it>`. |
| `WARN camera: no camera` | Plug the camera in through the powered hub; `sudo systemctl restart kite-link`. |
| `none signed with your token` | Same as token mismatch. |
| Nobody knows the board's password | `python unoq_rescue.py` (below). |

**Board password lost?** Plug the board into the computer with a USB-C data cable, close App Lab, and run
`python unoq_rescue.py` (or download it: `curl -fsSLO https://trykite.pages.dev/unoq/unoq_rescue.py`). It logs in
over USB with `adb` (no password needed after the first setup; it downloads adb if it's missing), saves the
board's token here, checks the camera, and asks for a new password. To set it, it uses the first of: adb as root,
passwordless sudo, the factory password `arduino`, or App Lab's Docker (a container that mounts the board's disk
runs the board's own `chpasswd`). The new password goes over in a file only `arduino` can read, deleted at once;
it is never on a command line. Nothing is erased. `--check` only reports; `--restart` also restarts `kite-link`.

Board log: `journalctl -u kite-link -f`. Change Wi-Fi: `sudo nmcli dev wifi connect "<name>" password "<pw>"`.
Rejoin eduroam: `curl -fsSL https://trykite.pages.dev/unoq/install.sh | sh -s -- --eduroam`.
Video size and rate: `KITE_VIDEO_SIZE` and `KITE_VIDEO_FPS` in `/etc/systemd/system/kite-link.service`
(640x480 at 10 fps by default, sized for relayed links).

## Files

- `board.py`: the board service (standard library; OpenCV from apt only for a real camera).
- `unoq_rescue.py`: over USB, sets a new board password without the old one and saves the token here.
- `kite_link.py`: the computer client. `kite_common.py`: SpacetimeDB calls and signing, used by both.
- `install.sh`, `kite-link.service`: the installer and the service. `publish.py` writes the installer's
  SHA-256 list and copies these files to `site/public/unoq/`; run it after any change, then deploy the site.
- `e2e_local.sh`: end-to-end on one computer (board and client as separate processes over the LAN IP, plus
  real SpacetimeDB as `unoq-test`). `tests/`: `scripts/check.sh hardware`.

Replaces `../unoq_wifi/` on the board (the installer turns the old `kite-hello` service off; the heartbeat
keeps the same shape, so `../unoq_wifi/listen.py` still works).
