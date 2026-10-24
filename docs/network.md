# Kite networking

At the judging table everything runs on one laptop, and the earpiece address is `127.0.0.1`. You only need the steps below when the drone laptop and the wearable laptop (the one running the earpiece) are separate machines. They talk over a Thunderbolt cable, so the safety path never depends on Wi-Fi.

Check any setup with:

```bash
python3 scripts/netcheck.py              # uses KITE_EARPIECE_HOST and SPACETIME_HOST from .env
python3 scripts/netcheck.py --host 169.254.12.34
```

Each line is PASS or FAIL, and every FAIL comes with a fix. The exit code is non-zero if anything failed. It never prints key values and never writes to SpacetimeDB.

## Two laptops over Thunderbolt

1. Plug a Thunderbolt (or USB4) cable between the two Macs. A plain USB-C charging cable won't work.
2. On both Macs, open System Settings > Network > Thunderbolt Bridge. Leave Configure IPv4 on "Using DHCP". Nothing on the cable hands out addresses, so after about ten seconds each Mac gives itself a 169.254.x.x address. The status dot turns yellow with "Self-assigned IP", which is what you want.
3. On the wearable laptop, find its address. Click Details > TCP/IP under Thunderbolt Bridge, or run `python3 scripts/netcheck.py` and look for the line marked `bridge0`.
4. On the wearable laptop, start the earpiece listening on every interface:

   ```bash
   cd earpiece && .venv/bin/python -m earpiece run --bind 0.0.0.0
   ```

   The first time, macOS asks whether Python may accept incoming network connections. Click Allow. If you missed the prompt, go to System Settings > Network > Firewall > Options and set Python to "Allow incoming connections", or turn the firewall off for the weekend.
5. On the drone laptop, put the wearable laptop's address in `.env`:

   ```
   KITE_EARPIECE_HOST=169.254.12.34
   ```

6. Run `python3 scripts/netcheck.py` on both laptops. On the wearable laptop it checks the earpiece at `127.0.0.1`; on the drone laptop it checks it over the cable. Both earpiece lines should pass: the WebSocket on 5011 sends back a `state` message, and a UDP heartbeat sent to 5010 shows up in that state's `net.udp` count.

The 169.254 address can change after a reboot or a replug. If netcheck starts failing on the drone laptop, look up the address again and update `.env`.

If the WebSocket passes and UDP fails, the firewall is letting TCP through and dropping UDP. Allow Python again in the firewall settings.

## Venue Wi-Fi

The cable carries earpiece traffic. Wi-Fi is only for the cloud services: Claude, Cartesia, Google Routes and SpacetimeDB Maincloud. With Maincloud, the two laptops don't need to share a network at all, they just need internet. Join "Cal Hacks 13.0" on each laptop that runs something cloud-backed and run netcheck. The internet lines only test that DNS and TLS reach each API, so they pass without keys.

Venue networks often block traffic between clients, so don't use Wi-Fi in place of the cable. If the venue Wi-Fi is down or slow, a phone hotspot is the next thing to try.

## Offline fallback

When Maincloud is unreachable, run SpacetimeDB on the laptop itself:

```bash
scripts/offline.sh
```

It starts `spacetime start --listen-addr 127.0.0.1:3000` if no server is running yet, publishes the Kite module to it, and writes `VITE_SPACETIMEDB_HOST=ws://127.0.0.1:3000` to `dashboard/.env.local`. It then prints the line to put in `.env` for the Python side:

```
SPACETIME_HOST=http://127.0.0.1:3000
```

Restart `npm run dev` and anything that publishes to Spacetime, then run netcheck to confirm. Running the script twice does no harm.

To go back to Maincloud, run `scripts/offline.sh off`. It removes the dashboard override. Then set `SPACETIME_HOST=https://maincloud.spacetimedb.com` in `.env` again. The local server keeps running until you stop it with `pkill -f 'spacetimedb-standalone start'`.

The local server only helps when everything runs on one laptop, which is the judging setup. With Claude and Cartesia offline, the earpiece falls back to cached phrases and macOS `say`, and the sonar, `critical` and `dont_walk` cues still work.
