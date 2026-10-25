"""Kite's phone link: the walker's phone sends GPS to Kite.

    python -m phone.cloud           # primary: the app's fixes from SpacetimeDB to the earpiece (docs/phone.md)
    python -m phone.latest --watch  # the newest phone position, from any computer with internet
    python -m phone.ble             # older path: a Bluetooth LE receiver (phone/fake_phone.py stands in for a sender)

The phone side is the native app in phone-app/: it pushes each fix to SpacetimeDB over the phone's own internet as a
`position` event. phone/fix.py and phone/link.py turn fixes into earpiece `position` events and announce a lost
phone, whichever way the fixes arrive. Setup and troubleshooting: docs/phone.md.
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
