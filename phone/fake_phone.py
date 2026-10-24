"""A stand-in for the walker's phone app: a BLE central (bleak) that finds Kite's service, connects and writes fake
GPS fixes at 2 Hz, newline-terminated and chunked to the MTU, the same way the native app does.

    python -m phone.fake_phone                  # walk north from the Cal Hacks default point until Ctrl+C
    python -m phone.fake_phone --seconds 20

Run it on a second computer: a Mac usually cannot connect to a peripheral it is advertising itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from . import FIX_CHAR_UUID, SERVICE_UUID

RATE_HZ = 2.0
ATT_OVERHEAD = 3            # ATT write header bytes inside the MTU
STEP_DEG = 0.00001          # ~1.1 m of latitude per fix: a brisk walk at 2 Hz


def fake_fix(i: int, lat0: float, lng0: float) -> bytes:
    fix = {"t": int(time.time() * 1000), "lat": round(lat0 + i * STEP_DEG, 7), "lng": lng0, "acc": 5.0,
           "alt": 12.0, "speed": 1.2 if i else None, "course": 0.0 if i else None}
    return (json.dumps(fix, separators=(",", ":")) + "\n").encode()


def chunks(data: bytes, size: int) -> list[bytes]:
    return [data[i:i + size] for i in range(0, len(data), size)]


async def run(args) -> int:
    from bleak import BleakClient, BleakScanner

    print(f"[fake phone] scanning for {SERVICE_UUID} ...")
    dev = await BleakScanner.find_device_by_filter(
        lambda d, ad: SERVICE_UUID in [u.lower() for u in ad.service_uuids], timeout=args.scan)
    if dev is None:
        print("[fake phone] no Kite peripheral found (is `python -m phone.ble` running on ANOTHER computer?)")
        return 1
    print(f"[fake phone] found {dev.name!r} {dev.address}, connecting")
    async with BleakClient(dev) as client:
        size = max(20, client.mtu_size - ATT_OVERHEAD)
        print(f"[fake phone] connected, MTU {client.mtu_size}: {size}-byte chunks")
        t_end = time.monotonic() + args.seconds if args.seconds else None
        i = 0
        while t_end is None or time.monotonic() < t_end:
            for part in chunks(fake_fix(i, args.lat, args.lng), size):
                await client.write_gatt_char(FIX_CHAR_UUID, part, response=False)
            i += 1
            await asyncio.sleep(1.0 / RATE_HZ)
        print(f"[fake phone] sent {i} fixes")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m phone.fake_phone", description=__doc__.splitlines()[0])
    p.add_argument("--lat", type=float, default=37.7749)
    p.add_argument("--lng", type=float, default=-122.4194)
    p.add_argument("--scan", type=float, default=10.0, help="seconds to scan for the peripheral")
    p.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    try:
        return asyncio.run(run(p.parse_args(argv)))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
