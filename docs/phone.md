# Phone link (SpacetimeDB)

The walker's iPhone runs the native app in `phone-app/`. Each GPS fix goes straight from the phone to SpacetimeDB
over the phone's own internet (cellular or Wi-Fi): one HTTPS call to the `push` reducer, about once a second, and
again every 2 s while standing still so Kite knows the phone is there. The background location task keeps it
going with the screen locked. No Bluetooth, no board in between, no shared network with the laptop.

```
iPhone app (phone-app/)  --HTTPS push-->  SpacetimeDB (event + latest tables)  -->  python -m phone.cloud  --UDP 5010-->  earpiece
                                                                               -->  the live dashboard map, python -m phone.latest, kite.navigate
```

Each fix is a `position` event with source `phone`: `lat`, `lng`, `accuracy` (m), `speed` (m/s), `course` (deg, GPS
direction of travel) when known, and `fix_t`, the fix's own time in ms. Building and installing the app:
[`phone-app/README.md`](../phone-app/README.md).

## Run on the laptop

```
earpiece/.venv/bin/python -m phone.cloud                          # forward to the earpiece on this computer
earpiece/.venv/bin/python -m phone.cloud --ear-host 192.168.137.2 # earpiece on another computer: start it with --bind 0.0.0.0
earpiece/.venv/bin/python -m phone.cloud --no-send                # print the walker's position only
python -m phone.latest --watch                                    # any computer: the newest position, every second
```

`phone.cloud` polls SpacetimeDB twice a second and hands each new fix to the same link logic the Bluetooth receiver
uses (`phone/link.py`): the earpiece gets `position` events, "Kite has lost the phone. Directions are paused." after
5 s with nothing from the phone, and "The phone is back." when it returns (`--no-announce` turns those off). It
prints the same live line as before:

```
37.871850, -122.272700  +/-5 m  |  1.2 m/s toward N  |  fix 1.6 s ago  |  phone ok  |  sent 6
```

It reads the append-only `event` table, not `latest`. The earpiece mirrors every position it is given back to
SpacetimeDB under the same source, and that copy has no `fix_t`, so `phone.cloud` never forwards it again. On start it
only notes where the log ends, so a fix from an earlier walk is never replayed. `SPACETIME_HOST` and `SPACETIME_DB`
point it at another database (a local `spacetime start` for testing).

## Testing without the phone

`phone/tests/test_cloud.py` drives `phone.cloud` against a fake SQL endpoint and a local UDP socket, and
`phone-app/cloud.test.ts` drives the app's link against a fake `fetch` and clock (`npx bun test` in `phone-app/`).
For the whole chain on one computer, start a local SpacetimeDB, publish `dashboard/spacetimedb` to it, run
`phone.cloud` with `SPACETIME_HOST=http://127.0.0.1:3000`, and call the app's `startLink` from a bun script with the
same host.

## Troubleshooting

- The app stays on "Connecting": the phone has no internet, or no GPS fix yet (it connects with its first fix).
- The app says "Lost connection. Reconnecting.": two pushes in a row failed. It retries, waiting up to 5 s between
  tries, and always sends the newest fix.
- `phone.cloud` stays at `waiting for the phone ...` while the app says Connected: both must use the same database.
  The app uses `kite-sync-pl6rp` on Maincloud (`phone-app/cloud.ts`); check `SPACETIME_DB` on the laptop.
- `can't reach SpacetimeDB`: the laptop has no internet (on the Tello Wi-Fi, for example).

## Older path: the Bluetooth receiver

The app no longer uses Bluetooth. The receiver below, `python -m phone.ble`, still works with any Bluetooth LE
sender that writes the same fixes (`phone/fake_phone.py` is one), and the UNO Q install in this section still runs
it, but nothing on the walker's phone talks to it now.

### Install

Mac or Windows, from the repo root, with [uv](https://docs.astral.sh/uv/):

```
uv venv --python 3.12 earpiece/.venv
uv pip install --python earpiece/.venv -e "earpiece[ble]"
```

The `ble` extra installs `bless` (the peripheral, on CoreBluetooth or WinRT) and `bleak` (for `phone/fake_phone.py`).
`scripts/check.sh` installs it too.

### Run

```
earpiece/.venv/bin/python -m phone.ble --no-send          # Mac: print positions only
earpiece\.venv\Scripts\python -m phone.ble --no-send      # Windows
python -m phone.ble                                        # also forward to the earpiece (KITE_EARPIECE_HOST, default 127.0.0.1)
python -m phone.ble --ear-host 192.168.137.2               # earpiece on another computer: start it with --bind 0.0.0.0
python -m phone.ble --no-send --seconds 15                 # quick check that advertising starts
python -m phone.ble --walker 3fa2                          # this phone guides; the others only show on the map
```

A good start looks like this:

```
15:26:23 [phone] advertising 'Kite', service 4b495445-0000-4b69-7465-50686f6e6500
15:26:23         on the phone: open the Kite app (phone-app/) and connect
waiting for the phone ...
```

macOS: the first run asks whether your terminal app (Terminal, iTerm, VS Code) may use Bluetooth. Say yes. If you
missed the prompt, turn the app on in System Settings > Privacy & Security > Bluetooth and restart the terminal.

Windows: Bluetooth must be on (Settings > Bluetooth & devices). The adapter must support the peripheral role. Most
built-in Intel and Realtek adapters on Windows 10 and 11 do, but some cheap USB dongles don't.

### The terminal line

```
37.778352, -122.408768  +/-5 m  |  1.2 m/s toward N  |  fix 0.3 s ago  |  phone ok  |  sent 42  |  earpiece up
```

From left to right: latitude and longitude, GPS accuracy in metres, speed (plus the direction of travel once the
walker moves faster than 0.6 m/s), how long ago the newest GPS fix arrived, whether the phone is still writing
(`PHONE LOST (6 s ago)` after 5 s of silence), how many positions went to the earpiece, and whether the earpiece's
WebSocket answers. A heading column shows up only if the app sends a compass heading; it currently sends GPS only.

When the phone goes quiet for 5 s the earpiece says "Kite has lost the phone. Directions are paused.", and "The
phone is back." when it returns. `--no-announce` turns that off.

### Several phones

Any number of phones can connect at once. Each one gets its own source, `phone-<id>`, where `<id>` is the first 4
hex digits of a SHA-1 of the phone's CoreBluetooth central UUID (the same for a given phone on a given Mac), e.g.
`phone-3fa2`. Every phone's position goes to the earpiece and on to SpacetimeDB, so the dashboard map shows all of
them. Each `position` carries `accuracy` (m), `speed` (m/s), `course` (deg, GPS direction of travel, null when
standing) and `walker` (true or false).

Only one phone guides. The walker is the first phone to connect, or the one named with `--walker 3fa2`. If it goes
quiet for 5 s while another phone is still writing, that phone takes over without a word. The earpiece drops
`walker:false` positions from routing, and only the walker's phone says "Kite has lost the phone" or "The phone is
back". The log names the phone on every line:

```
16:58:02 [phone] walker is now phone-3fa2
16:58:02 phone-3fa2 (walker) #57 37.866012, -122.258310  +/-5 m  |  1.2 m/s toward N  |  ...
16:58:02 phone-91c0 #12 37.866101, -122.258402  +/-8 m  |  - m/s  |  ...
```

On Windows, bless's WinRT backend doesn't pass the central through, so all phones share one link and the source
`phone`. Use one phone there.

### Testing without the phone

`python -m phone.fake_phone` is a stand-in for the app. It scans for the service, connects, and writes a fake walk
north at 2 Hz, chunked to the MTU like the real app. Run it on a second computer: a Mac can't connect to its own
peripheral (tried: the scan never sees it). The receiver's logic is covered by `phone/tests` with no radio:

```
earpiece/.venv/bin/python -m pytest -q phone/tests
```

### Troubleshooting

- `could not start the Bluetooth peripheral`: on a Mac, Bluetooth permission for the terminal app or Bluetooth is
  off; on Windows, Bluetooth is off or the adapter can't act as a peripheral.
- `started (advertising not confirmed)`: the stack accepted the service but says it isn't advertising yet. Restart
  Bluetooth, then the receiver.
- The app can't find "Kite": check the receiver is running, and that no other computer nearby is also advertising
  as Kite (`--name` gives it another name). On iPhone, Settings > Bluetooth must be on and the app needs Bluetooth
  permission.
- Line stuck at `waiting for the phone ...` after the app says connected: the app is writing to the wrong
  characteristic. Fixes go to `4b495445-0001-4b69-7465-50686f6e6500`, one JSON object per line, newline at the end.
- `bad` count rising in the stop summary: the app is sending JSON the parser rejects (see `phone/fix.py` for the
  format).
- `sent 0` while `phone ok`: the fixes have no `lat`/`lng` yet (the phone has no GPS fix).

### On the Arduino UNO Q (the phone talks to the board on the walker)

The phone and the UNO Q both ride on the walker, so Bluetooth range stops mattering. The board runs the same
receiver and forwards every position over Wi-Fi to the main computer's earpiece (UDP 5010).

```
phone/unoq/install.sh                                     # on the main computer, from the repo root: uses its own IP
phone/unoq/install.sh <main-computer-ip> arduino@<board-ip>
```

It copies `phone/` and `earpiece/earpiece/` to `~/kite-phone` on the board, installs BlueZ, numpy, scipy and
bless, and starts the `kite-phone` service (as root, which BlueZ needs to advertise). On the main computer:
`python -m earpiece run --bind 0.0.0.0 --spacetime`. Board log: `journalctl -u kite-phone -f`. Turn the Mac's
receiver off first, or the phone may connect to whichever "Kite" it finds first. On Linux every phone shares
the source `phone` (only macOS tells phones apart), so use one phone with the board.

### Reading the GPS from another computer (any network)

eduroam doesn't always route between laptops, so the receiver can also put each phone's newest position in
SpacetimeDB (about once a second, newest only), and any computer with internet reads it:

```
python -m phone.ble --spacetime            # on the computer the phone connects to (add --no-send if no earpiece)
python -m phone.latest --watch             # on the main computer: each phone's newest position, every second
```

In a script: `from phone.latest import latest` gives `[{"source", "lat", "lng", "accuracy", "speed", "course", "age_s", ...}]`,
newest first. Standard library only.
