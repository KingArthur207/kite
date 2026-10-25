"""Digests of the code behind a behind-follow episode, so a cached result is never reused across a code change.

    episode_digest()   the sources an episode's METRICS depend on (the sim, the follow, the governor, the brain and
                       its data): signoff's episode cache and the trainer's FLOWN cache are keyed by it
    run_digest()       episode_digest + how they are SCORED and searched (fitness.py; params.py is already in): a
                       training run refuses to resume when it changed (its FLOWN normalizer and its CMA state would
                       mix two objectives, or decode a stored x with other bounds)

Both hash the relative path and the bytes of every file matched by SOURCE_GLOBS under the repo (iCloud conflict copies
"x 2.py" and __pycache__ ignored), so the same code gives the same digest on this laptop and in a Modal container,
whose image ships these files at the same relative paths (modal_app.EXTRA_FILES / PY_TREES / DATA_GLOBS cover every
glob here; tests/test_behind_modal.py checks it against the import probe). The driver-only modules (signoff, modal
app, train) are left out: editing a report or a launcher changes no episode.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Every file an episode imports or reads (the import probe of modal_app sees the .py ones; FlySteer reads the data)
SOURCE_GLOBS = ("kite/__init__.py", "kite/config.py", "kite/types.py", "kite/camera.py", "kite/follow.py",
                "kite/predict.py", "kite/follow_behind.py", "kite/follow_tuned.py", "kite/sources/__init__.py",
                "kite/sources/base.py", "kite/sources/opencv_sources.py", "kite/sources/pose.py",
                "kite/sources/recording.py",
                "kite/drone/__init__.py", "kite/drone/base.py", "kite/drone/safety.py", "kite/fly/**/*.py",
                "kite/flybrain/**/*.py", "kite/flybrain/data/brains/**/*.npz", "kite/flybrain/data/brains/**/*.json",
                "kite/flybrain/configs/*.yaml")
DRIVER_ONLY = frozenset({"kite/fly/behind/signoff.py", "kite/fly/behind/modal_app.py", "kite/fly/behind/train.py"})
SCORING = frozenset({"kite/fly/behind/fitness.py"})  # in run_digest, not in episode_digest
_ICLOUD_DUP = re.compile(r" \d+(\.[^./]*)?$")


def _skipped(rel: Path) -> bool:
    return any(p == "__pycache__" or _ICLOUD_DUP.search(p) for p in rel.parts)


def source_files(repo: Path = REPO, *, scoring: bool = False) -> list[Path]:
    """The files a digest covers, as paths relative to repo, sorted."""
    root = Path(repo).resolve()
    out = set()
    for g in SOURCE_GLOBS:
        for p in root.glob(g):
            if not p.is_file():
                continue
            rel = p.resolve().relative_to(root)
            key = rel.as_posix()
            if _skipped(rel) or key in DRIVER_ONLY or (key in SCORING and not scoring):
                continue
            out.add(rel)
    return sorted(out, key=lambda r: r.as_posix())


def _digest(repo: Path, scoring: bool) -> str:
    root = Path(repo).resolve()
    h = hashlib.sha256()
    for rel in source_files(root, scoring=scoring):
        h.update(rel.as_posix().encode())
        h.update(b"\0")
        h.update((root / rel).read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:20]


def episode_digest(repo: Path = REPO) -> str:
    """sha256 (20 hex) of what an episode's metrics depend on."""
    return _digest(repo, scoring=False)


def run_digest(repo: Path = REPO) -> str:
    """episode_digest plus the fitness: what a training run's scores depend on."""
    return _digest(repo, scoring=True)
