# UNO Q: hello over Wi-Fi (on a battery, no cable)

The UNO Q's Linux side sends `hello 1`, `hello 2`, ... once a second as a UDP broadcast on port 5099.
Any computer on the **same Wi-Fi** prints it with `listen.py`. No Arduino IDE and no sketch: it starts by itself
whenever the board powers on, and keeps trying until the Wi-Fi is up.

**The network:** put the UNO Q and the computer on one Wi-Fi, e.g. **Arthur's iPhone Personal Hotspot**.
iPhone hotspots let the devices on them talk to each other; campus Wi-Fi like eduroam usually doesn't.

## One time only (needs a USB-C cable, about 10 minutes)

A new board doesn't know any Wi-Fi, so it has to be told once.

1. Turn on Arthur's Personal Hotspot (Settings > Personal Hotspot > Allow Others to Join; keep that screen open
   while the board joins).
2. Plug the UNO Q into a computer with USB-C, open **Arduino App Lab**, choose the USB connection, and follow the
   first-setup wizard: give the board a name (say `kite-unoq`), a password, and Arthur's hotspot name and password.
   The wizard also turns on SSH.
3. Copy these files to the board and install the service. From a terminal on the computer (Windows 10/11 have
   `ssh` and `scp` built in), in the repo folder:

   ```
   scp hardware/arduino/unoq_wifi/hello_sender.py hardware/arduino/unoq_wifi/kite-hello.service hardware/arduino/unoq_wifi/install.sh arduino@kite-unoq.local:/home/arduino/
   ssh arduino@kite-unoq.local "sh install.sh"
   ```

   If `kite-unoq.local` isn't found, use the board's IP from App Lab instead. Without Wi-Fi you can use
   `adb shell` over the USB cable (Android platform-tools), or App Lab's built-in terminal.

The board is done: unplug it, put it on the battery, and it says hello as soon as it joins the hotspot.

## Every time

On a computer on the same hotspot: `python hardware/arduino/unoq_wifi/listen.py` (no installs).

```
listening for the UNO Q on UDP 5099 (Ctrl+C to stop)
14:02:12  172.20.10.4      hello 1 from kite-unoq
14:02:13  172.20.10.4      hello 2 from kite-unoq
```

- **Windows firewall:** allow the port once (PowerShell as admin):
  `New-NetFirewallRule -DisplayName "Kite UNO Q hello" -Direction Inbound -Protocol UDP -LocalPort 5099 -Action Allow`
- **If nothing arrives but both devices are on the hotspot:** some networks drop broadcasts. Send to the computer
  directly by setting its IP in `kite-hello.service` (`Environment=KITE_HELLO_HOST=172.20.10.x`), then on the board
  `sudo systemctl daemon-reload && sudo systemctl restart kite-hello`.
- **Change the board's Wi-Fi later:** on the board, `sudo nmcli dev wifi connect "<name>" password "<password>"`.
- **Board log:** `journalctl -u kite-hello -f`. Stop it for good: `sudo systemctl disable --now kite-hello`.

Verified on a laptop (sender pointed at itself, listener printing every message and the quiet warning). Not yet
run on a UNO Q: the service assumes the board's user is `arduino` with home `/home/arduino`, as on Arduino's image.
