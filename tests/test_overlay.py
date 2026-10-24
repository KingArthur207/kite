import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "video"))

import overlay  # noqa: E402


def test_timeline_aligns_events_and_expires_hazards_and_stale_obstacles():
    evs = overlay.timed_events([
        {"t": 50.0, "type": "say", "text": "Hello."},
        {"t": 51.0, "type": "hazard", "kind": "crosswalk", "state": "dont_walk", "bearing": 3, "ttl": 2},
        {"t": 51.0, "type": "obstacles", "objects": [{"bearing": -30, "distance": 1.0}]},
    ], offset=10.0)
    assert [t for t, _ in evs] == [10.0, 11.0, 11.0]
    tl = overlay.Timeline(evs, overlay.captions_from_events(evs))

    assert tl.at(10.5)["caption"] == "Hello."
    st = tl.at(11.2)
    assert [overlay.hazard_text(h) for h in st["hazards"]] == ["Don't walk."]
    assert st["objects"] and st["caption"] == "Don't walk."
    st = tl.at(13.5)
    assert st["hazards"] == [] and st["objects"] == []
