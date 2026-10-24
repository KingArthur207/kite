import json
import math

import numpy as np
import pytest

from earpiece import events, spatial
from earpiece.beacon import Beacon
from earpiece.config import BeaconCfg, SonarCfg
from earpiece.events import BadMessage, Obstacle, clock_phrase, distance_phrase, parse
from earpiece.mixer import Engine
from earpiece.sonar import Sonar

DT = 1024 / 48000


# ---------------------------------------------------------------- events
def test_obstacles_frame_parses_with_aliases_and_sorts():
    m = parse(json.dumps({"type": "obstacles", "source": "webcam", "seq": 5, "objects": [
        {"id": 1, "class": "Pedestrian", "bearing_deg": 370, "distance_m": 4},
        {"id": "b", "kind": "car", "bearing": -20, "distance": 2.5, "closing_speed": 1.2, "confidence": 3}]}))
    assert m.type == "obstacles" and m.seq == 5 and m.source == "webcam"
    assert [o.id for o in m.objects] == ["b", "1"]
    car, person = m.objects
    assert car.kind == "vehicle" and car.confidence == 1.0 and car.closing_speed == 1.2
    assert person.kind == "person" and person.bearing == 10.0


@pytest.mark.parametrize("raw", [
    b"\xff\xfe", b"not json", b"[1,2]", b'{"type":"nope"}', b'{"type":"guidance"}',
    b'{"type":"guidance","text":""}', b'{"type":"beacon"}', b'{"type":"control","action":"selfdestruct"}',
    b'{"type":"say","text":42}', b"x" * 300000,
])
def test_bad_messages_are_rejected(raw):
    with pytest.raises(BadMessage):
        parse(raw)


def test_bad_objects_are_dropped_not_the_frame():
    m = parse({"type": "obstacles", "objects": [
        {"id": 1, "bearing": float("nan"), "distance": 2}, {"id": 2, "bearing": 0},
        {"id": 3, "bearing": "left", "distance": 1}, "junk", {"id": 4, "bearing": 5, "distance": -3},
        {"id": 5, "bearing": 5, "distance": True}]})
    assert [o.id for o in m.objects] == ["4"]
    assert m.objects[0].distance == 0.0


def test_object_cap_and_text_limits():
    m = parse({"type": "obstacles", "objects": [{"id": i, "bearing": 0, "distance": i} for i in range(500)]})
    assert len(m.objects) == events.MAX_OBJECTS and m.objects[0].distance == 0
    g = parse({"type": "guidance", "text": "a  b\n" + "x" * 1000})
    assert g.text.startswith("a b ") and len(g.text) == events.MAX_TEXT


def test_ids_are_sanitised():
    m = parse({"type": "obstacles", "source": "web cam;rm -rf", "objects": [{"id": "../x y", "bearing": 0, "distance": 1}]})
    assert m.source == "web_cam_rm_-rf" and m.objects[0].id == ".._x_y"


@pytest.mark.parametrize("b,want", [(0, "ahead"), (10, "ahead"), (-20, "slightly left"), (30, "slightly right"),
                                    (90, "on your right"), (-85, "on your left"), (60, "at 2 o'clock"),
                                    (-60, "at 10 o'clock"), (130, "at 4 o'clock"), (175, "behind you")])
def test_clock_phrases(b, want):
    assert clock_phrase(b) == want


def test_distance_phrases():
    assert distance_phrase(0.5) == "very close"
    assert distance_phrase(1.0) == "1 meter"
    assert distance_phrase(2.3) == "2.5 meters"
    assert distance_phrase(23) == "25 meters"
    assert distance_phrase(123) == "120 meters"


# ---------------------------------------------------------------- sonar
_clock = {"t": 0.0, "seq": 0}


def run(sonar: Sonar, eng: Engine, frames, seconds: float, fps=15, source="cam"):
    """frames(t) -> list[Obstacle] with t relative to this call; the clock and the frame
    counter keep running across calls like a real sender. Returns audio + chosen tracks."""
    out, chosen, t, fr = [], [], 0.0, 0
    t0 = _clock["t"]
    while t < seconds:
        if t * fps >= fr:
            fr += 1
            _clock["seq"] += 1
            sonar.update_frame(source, frames(t), seq=_clock["seq"], now=t0 + t)
        chosen.append([c.id for c in sonar.tick(now=t0 + t)])
        out.append(eng.render(1024))
        t += DT
    _clock["t"] = t0 + t
    return np.concatenate(out), chosen


def beep_times(plays):
    return [p[0] for p in plays]


def make(speak=None):
    eng = Engine()
    plays = []
    orig = eng.play

    def spy(st, at=None, bus="cue", gain=1.0, tag=None):
        plays.append((at / eng.rate if at is not None else None, bus, gain, tag))
        orig(st, at, bus, gain, tag)
    eng.play = spy
    return eng, Sonar(eng, SonarCfg(), speak=speak or (lambda *a, **k: None)), plays


def test_approaching_person_speeds_up_then_goes_critical():
    said = []
    eng, s, plays = make(lambda t, p, **k: said.append((t, p)))
    run(s, eng, lambda t: [Obstacle("p", "person", 0, max(0.4, 5 - 1.2 * t), None, 0.9)], 4.5)
    times = beep_times(plays)
    gaps = np.diff(times)
    assert len(times) > 10
    assert gaps[0] > 0.4 and gaps[-1] < 0.13          # slow far away, fast up close
    assert all(b <= a + 0.02 for a, b in zip(gaps, gaps[1:]))   # never slows down while approaching
    assert any(p[1] == "alert" for p in plays)        # critical uses the unduckable bus
    assert ("Stop. Person ahead.", "critical") in said


def test_object_beside_the_path_stays_quiet():
    eng, s, plays = make()
    run(s, eng, lambda t: [Obstacle("p", "person", 60, 3.0, None, 0.9)], 3)    # 2.6 m to the side
    assert plays == []


def test_close_object_beside_you_is_heard():
    eng, s, plays = make()
    run(s, eng, lambda t: [Obstacle("p", "person", 95, 0.6, None, 0.9)], 2)
    assert plays


def test_cue_comes_from_the_objects_side():
    eng, s, plays = make()
    audio, _ = run(s, eng, lambda t: [Obstacle("p", "pole", -25, 2.0, None, 1.0)], 2)
    l, r = np.abs(audio[:, 0]).sum(), np.abs(audio[:, 1]).sum()
    assert l > r * 1.4


def test_stale_track_fades_out():
    eng, s, plays = make()
    run(s, eng, lambda t: [Obstacle("p", "pole", 0, 1.5, None, 1.0)] if t < 1 else [], 3)
    assert max(beep_times(plays)) < 1.0 + SonarCfg().stale_track_s + 0.1
    assert s.tracks == {}


def test_out_of_order_frames_are_ignored():
    eng, s, _ = make()
    s.update_frame("cam", [Obstacle("p", "person", 0, 1.0)], seq=10, now=0)
    s.update_frame("cam", [Obstacle("p", "person", 0, 5.0)], seq=9, now=0.01)
    assert s.tracks["cam:p"].distance == pytest.approx(1.0)
    assert s.stats["out_of_order"] == 1


def test_crowd_sonifies_at_most_two():
    eng, s, plays = make()
    objs = lambda t: [Obstacle(f"c{i}", "person", -40 + i * 4, 1.2 + 0.1 * i, None, 0.9) for i in range(20)]
    _, chosen = run(s, eng, objs, 2)
    assert max(len(c) for c in chosen) <= 2
    assert len({p[3] for p in plays}) <= 3


def test_closing_speed_is_estimated():
    eng, s, _ = make()
    for i in range(10):
        s.update_frame("cam", [Obstacle("p", "person", 0, 6 - 2.0 * i / 15)], seq=i + 1, now=i / 15)
    assert s.tracks["cam:p"].est_speed == pytest.approx(2.0, abs=0.3)


def test_fast_car_goes_critical_before_it_is_close():
    eng, s, plays = make()
    s.update_frame("cam", [Obstacle("c", "vehicle", 0, 3.0, 4.0, 1.0)], seq=1, now=0)
    s.tick(now=0)
    assert s.tracks["cam:c"].critical            # 0.75 s to contact


def test_muted_still_warns_when_critical():
    eng, s, plays = make()
    s.muted = True
    run(s, eng, lambda t: [Obstacle("p", "pole", 0, 3.0)], 1.5)
    assert plays == []
    run(s, eng, lambda t: [Obstacle("p", "pole", 0, 0.5)], 1.0)
    assert plays and all(p[1] == "alert" for p in plays)


def test_disabled_sonar_is_silent():
    eng, s, plays = make()
    s.enabled = False
    run(s, eng, lambda t: [Obstacle("p", "pole", 0, 0.5)], 1.0)
    assert plays == []


def test_labels_are_rate_limited():
    said = []
    eng, s, _ = make(lambda t, p, **k: said.append(t))
    run(s, eng, lambda t: [Obstacle("a", "person", -10, 2.2), Obstacle("b", "pole", 10, 2.4)], 4.0)
    labels = [x for x in said if not x.startswith("Stop")]
    assert 1 <= len(labels) <= 2


# ---------------------------------------------------------------- beacon
def test_beacon_regions():
    b = Beacon(Engine(), BeaconCfg())
    assert [b.classify(x) for x in (0, 14, -30, 70, -120, 150, -179)] == \
        ["on", "on", "near", "side", "side", "behind", "behind"]


def beacon_ticks(b, eng, seconds, speech=False):
    times = []
    orig = eng.play

    def spy(st, at=None, bus="cue", gain=1.0, tag=None):
        if tag == "beacon":
            times.append(at / eng.rate)
        orig(st, at, bus, gain, tag)
    eng.play = spy
    t = 0.0
    while t < seconds:
        b.tick(now=t, speech_playing=speech)
        eng.render(1024)
        t += DT
    return times


def test_beacon_tempo_tracks_distance():
    eng = Engine()
    far = Beacon(eng, BeaconCfg())
    far.set(0, 40, "x", now=0)
    gf = np.diff(beacon_ticks(far, eng, 5)).mean()
    eng2 = Engine()
    near = Beacon(eng2, BeaconCfg())
    near.set(0, 5, "x", now=0)
    gn = np.diff(beacon_ticks(near, eng2, 5)).mean()
    assert gf == pytest.approx(BeaconCfg().period_far_s, abs=0.03)
    assert gn == pytest.approx(BeaconCfg().period_near_s, abs=0.03)


def test_beacon_quiet_during_speech_and_when_stale():
    eng = Engine()
    b = Beacon(eng, BeaconCfg(), stale_s=1.0)
    b.set(20, 30, "x", now=0)
    assert beacon_ticks(b, eng, 2, speech=True) == []
    eng2 = Engine()
    b2 = Beacon(eng2, BeaconCfg(), stale_s=1.0)
    b2.set(20, 30, "x", now=0)
    ticks = beacon_ticks(b2, eng2, 4)
    assert ticks and max(ticks) < 1.2


def test_arrival_announced_once_with_hysteresis():
    said = []
    eng = Engine()
    b = Beacon(eng, BeaconCfg(), speak=lambda t, p, **k: said.append(t))
    for d in (10, 5, 2.5, 2.0, 3.2, 2.4, 2.8):
        b.set(0, d, "the cafe", now=0)
        b.tick(now=0)
    assert said == ["You have arrived at the cafe."]
    b.set(0, 12, "the cafe", now=0)      # walked away
    b.set(0, 2, "the cafe", now=0)
    b.tick(now=0)
    assert len(said) == 2


def test_sender_restart_resets_sequence():
    eng, s, _ = make()
    s.update_frame("cam", [Obstacle("p", "person", 0, 4.0)], seq=500, now=0)
    s.update_frame("cam", [Obstacle("p", "person", 0, 1.0)], seq=1, now=0.05)   # restarted process
    assert s.tracks["cam:p"].distance < 2.0
    s.update_frame("cam", [Obstacle("p", "person", 0, 3.0)], seq=1, now=2.0)    # after a pause
    assert s.stats["out_of_order"] == 0


# ---------------------------------------------------------------- cue pre-warm
def test_sonar_prewarm_covers_every_critical_cue_and_matches_lazy_render():
    from earpiece import sonar as sonar_mod
    cfg = SonarCfg()
    n = sonar_mod.prewarm(cfg, 48000)
    assert n == len(sonar_mod.PREWARM_KINDS) * len(spatial.BEARINGS_5DEG) == 6 * 73
    pq = sonar_mod.critical_pitch_q(cfg)
    assert pq == 1250
    for kind in sonar_mod.PREWARM_KINDS:
        for bq in (-180, -95, 0, 35, 180):
            warm = sonar_mod._cue(kind, pq, bq, 48000)
            assert warm is sonar_mod._WARM[(kind, pq, bq, 48000)]
            assert np.array_equal(warm, sonar_mod._cue_render(kind, pq, bq, 48000))
    assert sonar_mod.prewarm(cfg, 44100, kinds=("person",)) == 73     # resampled path


def test_critical_beeps_play_from_the_warm_cache():
    from earpiece import sonar as sonar_mod
    sonar_mod.prewarm(SonarCfg(), 48000)
    warm = {id(a) for a in sonar_mod._WARM.values()}
    eng, s, _ = make()
    alerts = []
    orig = eng.play

    def spy(st, at=None, bus="cue", gain=1.0, tag=None):
        if bus == "alert":
            alerts.append(id(st))
        orig(st, at, bus, gain, tag)
    eng.play = spy
    run(s, eng, lambda t: [Obstacle("p", "person", 12, 0.5, 1.0, 1.0)], 1.0)
    assert alerts and all(a in warm for a in alerts)


def test_beacon_prewarm_covers_every_region_and_bearing_tick_can_ask_for():
    from earpiece import beacon as beacon_mod
    cfg = BeaconCfg()
    n = beacon_mod.prewarm(cfg, 48000)
    assert 73 <= n < 2 * 73
    for b10 in range(-1800, 1801):
        b = spatial.wrap_deg(b10 / 10)
        key = (beacon_mod.region_of(b, cfg), int(round(b / 5.0) * 5), 48000)
        assert key in beacon_mod._WARM, key
    for key in [("on", 0, 48000), ("behind", 180, 48000), ("side", -90, 48000)]:
        assert np.array_equal(beacon_mod._beacon_sound(*key), beacon_mod._beacon_render(*key))
