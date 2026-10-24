import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import replay  # noqa: E402
from earpiece.spacetime import SpacetimeBridge  # noqa: E402


def test_schedule_makes_epoch_times_relative_and_fills_gaps():
    plan = replay.schedule([{"type": "say", "t": 1000.0}, {"type": "say"}, {"type": "say", "t": 1002.5}])
    assert [round(t, 2) for t, _ in plan] == [0.0, 0.1, 2.5]
    assert all("t" not in ev for _, ev in plan)


def test_bridge_throttles_perception_frames_but_not_hazards():
    st = SpacetimeBridge(host="http://127.0.0.1:9", log=lambda *_: None)
    queued: list = []
    st._q.put_nowait = queued.append   # capture instead of sending
    for _ in range(10):
        st.publish({"type": "obstacles", "source": "webcam", "objects": []})
        st.publish({"type": "hazard", "source": "drone", "kind": "crowd"})
    assert sum(o["type"] == "obstacles" for o in queued) == 1
    assert sum(o["type"] == "hazard" for o in queued) == 10


def test_bridge_throttles_sim_positions_to_one_per_second():
    st = SpacetimeBridge(host="http://127.0.0.1:9", log=lambda *_: None)
    queued: list = []
    st._q.put_nowait = queued.append
    for _ in range(20):
        st.publish({"type": "position", "source": "sim", "lat": 37.7, "lng": -122.4})
    assert len(queued) == 1
