"""Kite's phone link: the walker's phone sends GPS and compass heading (and, over USB, camera frames) to the laptop.

    python -m phone.ble             # Bluetooth LE (primary): the native app in phone-app/ connects (docs/phone.md)
    python -m phone usb             # USB Personal Hotspot fallback: HTTPS page served by this laptop, + camera frames
    python -m phone.detect -- ...   # the stand-in detector on the phone's camera (USB fallback only)

The phone side is one page (site/src/phone.body.html + site/public/phone.js) with two transports: Web Bluetooth when
opened from trykite.pages.dev, a WebSocket when served by `python -m phone usb`. Both send the same JSON fix
({t, lat, lng, acc, alt, speed, course, heading, heading_acc}); phone/fix.py turns it into an earpiece `position`.
Design, the comparison behind it and the table checklist: docs/phone.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(__file__).resolve().parent / ".state"    # CA, token, local frame URL (gitignored)
SITE_PUBLIC = ROOT / "site" / "public"

# Kite's own 128-bit GATT UUIDs ("KITE" in the first group). The phone page filters on the service UUID.
SERVICE_UUID = "4b495445-0000-4b69-7465-50686f6e6500"
FIX_CHAR_UUID = "4b495445-0001-4b69-7465-50686f6e6500"      # phone -> laptop: newline-framed JSON fixes
STATUS_CHAR_UUID = "4b495445-0002-4b69-7465-50686f6e6500"   # laptop -> phone: read / notify, small JSON status


def _earpiece_on_path() -> None:
    """`earpiece/` at the repo root has no __init__.py, so from the repo root `import earpiece` would find that folder
    as an empty namespace package. Put the real package's parent first, as sim/standin_detect.py does."""
    p = str(ROOT / "earpiece")
    if p not in sys.path:
        sys.path.insert(0, p)


_earpiece_on_path()
