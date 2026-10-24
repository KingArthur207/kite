"""Open-vocabulary street perception for Kite: YOLOE prompts + metric depth -> earpiece events.

Drop-in for sim/standin_detect.py on files and cameras (same events, same sources); see README.md.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# earpiece.events (the wire contract), kite.sources (camera/file opening) and the stand-in's geometry
# (sim/standin_detect.py, sim/live_walk.py) are imported read-only from their own directories.
for _p in (_ROOT / "earpiece", _ROOT, _ROOT / "sim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
