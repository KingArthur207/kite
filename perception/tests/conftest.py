import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import perception  # noqa: E402,F401  (puts earpiece/, sim/ on sys.path)
import live_walk  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m") and "slow" in config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="model test: run with -m slow (needs weights/, see perception/README.md)")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)


def plane_depth(h=252, w=448, fx=336.0, cam_h=1.2, pitch=0.0, sky=80.0, scale=1.0):
    """Depth map (z, metres) of flat ground seen by a camera pitched `pitch` deg down at cam_h, sky beyond.
    `scale` mimics a depth model whose metric scale is off (0.5 = reads everything at half)."""
    z = live_walk.floor_depth(h, fx, cam_h, pitch)
    z = np.where(np.isfinite(z), z, sky)
    return np.repeat(z[:, None], w, axis=1).astype(np.float32) * scale
