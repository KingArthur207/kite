#!/usr/bin/env python3
"""Publish the board files to the site, so `curl -fsSL https://trykite.pages.dev/unoq/install.sh | sh` works.

    python3 hardware/arduino/unoq_link/publish.py           write the hashes into install.sh, copy to site/public/unoq/
    python3 hardware/arduino/unoq_link/publish.py --check   exit 1 if anything is out of date (tests run this)

The files in site/public/unoq/ are byte-for-byte copies of the ones here; install.sh carries the SHA-256 of
each board file it downloads and refuses any that don't match. Run this after changing any of them, then
deploy the site (scripts/deploy-site.sh).
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parents[2] / "site" / "public" / "unoq"
BOARD_FILES = ["board.py", "kite_common.py", "kite-link.service"]        # what install.sh downloads
PUBLISHED = BOARD_FILES + ["install.sh", "kite_link.py"]                   # kite_link.py: for the computer
BLOCK = re.compile(r"(# --- hashes: written by publish.py, do not edit by hand ---\nHASHES='\n)(.*?)('\n# --- end hashes ---)",
                   re.S)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def installer_text() -> str:
    text = (HERE / "install.sh").read_text()
    lines = "".join(f"{sha256(HERE / f)}  {f}\n" for f in BOARD_FILES)
    if not BLOCK.search(text):
        raise SystemExit("publish: install.sh has no hash block")
    return BLOCK.sub(lambda m: m.group(1) + lines + m.group(3), text)


def stale() -> list[str]:
    out = []
    if installer_text() != (HERE / "install.sh").read_text():
        out.append("install.sh hashes")
    for f in PUBLISHED:
        dst = SITE / f
        if not dst.exists() or dst.read_bytes() != (HERE / f).read_bytes():
            out.append(f"site/public/unoq/{f}")
    return out


def main() -> int:
    if "--check" in sys.argv[1:]:
        bad = stale()
        if bad:
            print("out of date: " + ", ".join(bad) + "\nrun: python3 hardware/arduino/unoq_link/publish.py")
            return 1
        print("site/public/unoq is up to date")
        return 0
    (HERE / "install.sh").write_text(installer_text())
    SITE.mkdir(parents=True, exist_ok=True)
    for f in PUBLISHED:
        (SITE / f).write_bytes((HERE / f).read_bytes())
    print(f"published {', '.join(PUBLISHED)} to {SITE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
