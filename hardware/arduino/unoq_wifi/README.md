# UNO Q: hello over Wi-Fi, from a battery (works on eduroam)

> Superseded by [`../unoq_link/`](../unoq_link/README.md): one-line install, text both ways and video. Its
> installer turns this `kite-hello` service off and sends the same heartbeat, so `listen.py` keeps working.

The UNO Q's Linux side sends `hello 1`, `hello 2`, ... once a second **through the internet**: it pushes each
hello to Kite's SpacetimeDB database, and `listen.py` on any computer with internet prints it. That works on
eduroam, venue Wi-Fi or a phone hotspot, because campus Wi-Fi usually stops devices from reaching each other
directly but not from reaching the internet. The board also appears as a live device (`unoq`) on the caregiver
dashboard. No Arduino IDE, no sketch, no extra libraries: it starts by itself whenever the board powers on and
keeps retrying until it has internet.

## One time only (USB-C cable, about 15 minutes)

A new board doesn't know any Wi-Fi, so it has to be told once.

1. **First setup.** Plug the UNO Q into a computer with USB-C, open **Arduino App Lab**, choose the USB
   connection and follow the wizard: name the board `kite-unoq`, set a password, and give it **any normal Wi-Fi**
   it can join right now (a phone hotspot is easiest; the wizard may not handle eduroam's login). This also turns
   on SSH.
2. **Copy Kite's files to the board**, from a terminal on that computer in the repo folder (Windows 10/11 have
   `scp` and `ssh` built in; use the board's IP from App Lab if `kite-unoq.local` isn't found):

   ```
   scp hardware/arduino/unoq_wifi/hello_sender.py hardware/arduino/unoq_wifi/kite-hello.service hardware/arduino/unoq_wifi/install.sh hardware/arduino/unoq_wifi/join_eduroam.sh arduino@kite-unoq.local:/home/arduino/
   ssh arduino@kite-unoq.local
   ```
3. **On the board** (in that ssh session):

   ```
   sh install.sh                                                     # starts hello at every power-on
   sh join_eduroam.sh you@berkeley.edu 'your-eduroam-password' berkeley.edu   # only if it should use eduroam
   ```

   No Wi-Fi at all? Use `adb shell` over the USB cable (Android platform-tools) or App Lab's built-in terminal
   for the same commands.

Done: unplug it, put it on the battery. It says hello as soon as it has internet.

## Every time

On any computer with internet: `python hardware/arduino/unoq_wifi/listen.py` (no installs).

```
listening for 'unoq' through SpacetimeDB (kite-sync-pl6rp); Ctrl+C to stop
15:30:44  via internet     hello 1 from kite-unoq
15:30:45  via internet     hello 2 from kite-unoq
15:30:53                   ... nothing for 5 s (board off, or lost its Wi-Fi?)
```

- **Board log:** `journalctl -u kite-hello -f` (on the board). Stop it for good: `sudo systemctl disable --now kite-hello`.
- **Change the board's Wi-Fi later:** `sudo nmcli dev wifi connect "<name>" password "<password>"`, or
  `join_eduroam.sh` again.
- **Direct mode, no internet:** if both are on one network that allows device-to-device traffic (a phone hotspot),
  set `Environment=KITE_HELLO_MODE=udp` (or `both`) in `kite-hello.service`, restart it
  (`sudo systemctl daemon-reload && sudo systemctl restart kite-hello`), and run `python listen.py --udp` on the
  computer (Windows firewall: allow inbound UDP 5099 once).

## What's verified
- Through Maincloud from a laptop on the Berkeley campus Wi-Fi: every hello arrived about a second later, and the
  quiet warning fired 5 s after the sender stopped.
- UDP mode: sender and listener on one laptop.
- Not yet run on a UNO Q. The service assumes the board's user is `arduino` with home `/home/arduino`, as on
  Arduino's image, and `join_eduroam.sh` assumes PEAP + MSCHAPv2 (what most eduroam schools use, including
  Berkeley); if your school differs, its IT page lists the settings.
