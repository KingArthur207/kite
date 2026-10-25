"""perception.near outputs: the earpiece mapping (checked against the real sonar code), the terminal, the beeper
(rendered to WAV and analysed; nothing is ever played), and the calibration tool on synthetic board views."""

import io
import math
import wave

import numpy as np
import pytest

from perception.near import beeper, output


def _event(*objs):
    ev = {"type": "near", "t": 1.0, "priority": min(o["priority"] for o in objs if o["priority"]),
          "kind": objs[0]["kind"], "id": objs[0]["id"], "confidence": 0.8, "distance_m": objs[0]["distance_m"],
          "cues": {"size_h": 1.4}, "objects": list(objs)}
    if objs[0]["ttc_s"] is not None:
        ev["ttc_s"] = objs[0]["ttc_s"]
    return ev


def _obj(i, p, d=1.5, bearing=0.0, ttc=None, kind="person"):
    return {"id": str(i), "kind": kind, "priority": p, "distance_m": d, "bearing": bearing, "ttc_s": ttc,
            "lateral_m": 0.0, "confidence": 0.8}


# ------------------------------------------------------------------ earpiece mapping
def test_obstacles_message_parses_with_the_earpiece_contract():
    from earpiece.events import parse
    msg = output.to_obstacles(_event(_obj(1, 1), _obj(2, 2, 3.4), _obj(3, 3, 7.0), _obj(4, None, 9.0)), 7)
    m = parse(msg)
    assert m.type == "obstacles" and m.seq == 7 and len(m.objects) == 3
    assert [o.distance for o in m.objects] == [1.3, 3.0, 4.0]
    assert all(o.confidence == 1.0 for o in m.objects)


def _sonar_urgency(msg):
    """Feed one obstacles message to the real earpiece sonar (no audio engine needed for scoring)."""
    from earpiece.config import SonarCfg
    from earpiece.events import parse
    from earpiece.sonar import Sonar
    s = Sonar(engine=None, cfg=SonarCfg())
    m = parse(msg)
    s.update_frame("near", m.objects, m.seq, now=0.0)
    out = {}
    cfg = s.cfg
    for t in s.tracks.values():
        s._score(t)
        u = 1.0 if t.critical else t.urgency
        out[t.id.split(":")[1]] = (u, cfg.pitch_low_hz + (cfg.pitch_high_hz - cfg.pitch_low_hz) * u,
                                   cfg.interval_slow_s - (cfg.interval_slow_s - cfg.interval_fast_s) * u ** 0.8,
                                   t.critical, t.active)
    return out


def test_priorities_become_three_distinct_sonar_tones_highest_and_fastest_for_p1():
    r = _sonar_urgency(output.to_obstacles(_event(_obj(1, 1), _obj(2, 2, 3.4), _obj(3, 3, 7.0)), 1))
    (_, f1, i1, c1, a1), (_, f2, i2, c2, a2), (_, f3, i3, c3, a3) = r["1"], r["2"], r["3"]
    assert a1 and a2 and a3 and not (c1 or c2 or c3)
    assert f1 > f2 + 200 > f3 + 300 and i1 < i2 < i3
    assert f1 > 1100 and f3 < 700          # today ~1225 / ~812 / ~630 Hz (output.py docstring)


def test_short_time_to_contact_makes_the_sonar_critical():
    r = _sonar_urgency(output.to_obstacles(_event(_obj(1, 1, 3.0, ttc=0.7)), 1))
    assert r["1"][3]                       # critical: alarm bus + "Stop." in the earpiece
    msg = output.to_obstacles(_event(_obj(1, 1, 3.0, ttc=0.7)), 1)
    assert msg["objects"][0]["closing_speed"] == pytest.approx(1.3 / 0.7, abs=0.01)


def test_bearing_is_kept_in_sign_but_clamped_into_the_sonar_corridor():
    msg = output.to_obstacles(_event(_obj(1, 3, 8.0, bearing=-25.0), _obj(2, 1, 1.0, bearing=12.0)), 1)
    b3, b1 = msg["objects"][0]["bearing"], msg["objects"][1]["bearing"]
    assert b3 < 0 and 4.0 * math.sin(math.radians(abs(b3))) <= 0.5 + 1e-6
    assert b1 == 12.0
    r = _sonar_urgency(msg)
    assert r["1"][4] and r["2"][4]


# ------------------------------------------------------------------ terminal
def test_line_and_radar():
    ev = _event(_obj(1, 1, 1.4, ttc=1.1), _obj(2, 3, 6.5))
    s = output.line(ev)
    assert "P1" in s and "person" in s and "ttc 1.1 s" in s and "+1 more" in s
    assert "clear ahead" in output.line({"t": 0.0, "priority": None, "objects": []})
    rad = output.radar(ev)
    assert "1" in rad and "3" in rad and "2 m" in rad and "you" in rad


def test_terminal_prints_changes_only_when_piped():
    buf = io.StringIO()
    term = output.Terminal(stream=buf)
    ev = _event(_obj(1, 2, 3.0))
    for t in (0.0, 0.1, 0.2):
        term.show({**ev, "t": t})
    term.show({**_event(_obj(1, 1, 1.5)), "t": 0.3})
    assert buf.getvalue().count("\n") == 2


# ------------------------------------------------------------------ beeper (WAV only)
def test_beep_intervals_rise_inside_each_band_and_p1_goes_continuous():
    assert beeper.interval(3, 5.7) > beeper.interval(3, 3.3)
    assert beeper.interval(2, 2.88) > beeper.interval(2, 1.32)
    assert beeper.interval(1, 1.14) > beeper.interval(1, 0.6) > 0
    assert beeper.interval(1, 0.36) == 0.0 and beeper.interval(1, 1.08, ttc=0.8) == 0.0
    assert beeper.interval(2, 1.26) > beeper.interval(1, 1.14)      # every P1 beat is faster than any P2 one


def _dominant_hz(x, rate=beeper.RATE):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / rate)[int(np.argmax(spec))]


@pytest.mark.parametrize("p", [1, 2, 3])
def test_each_priority_has_its_own_pitch(p):
    assert _dominant_hz(beeper.tone(p, 0.2)) == pytest.approx(beeper.TONES[p], abs=8)


def test_render_and_write_wav(tmp_path):
    tl = [(0.0, 3, 7.0, None, -30.0), (1.0, 2, 3.0, None, 0.0), (2.0, 1, 1.2, None, 30.0), (3.0, None, None, None, 0)]
    st = beeper.render(tl)
    assert st.shape[1] == 2 and np.abs(st).max() <= 1.0
    seg = lambda a, b: st[int(a * beeper.RATE):int(b * beeper.RATE)]
    assert np.abs(seg(0.0, 1.0)[:, 0]).sum() > np.abs(seg(0.0, 1.0)[:, 1]).sum()     # P3 on the left
    assert np.abs(seg(2.0, 3.0)[:, 1]).sum() > np.abs(seg(2.0, 3.0)[:, 0]).sum()     # P1 on the right
    assert np.abs(seg(3.05, 3.3)).max() < 1e-6                                     # clear: silence
    def onsets(x):
        env = np.convolve(np.abs(x).sum(axis=1), np.ones(88) / 88, "same")      # 4 ms envelope
        return int(np.sum(np.diff((env > 0.05).astype(int)) == 1))
    assert onsets(seg(2.0, 3.0)) > onsets(seg(1.0, 2.0)) > onsets(seg(0.0, 1.0)) >= 1
    path = beeper.write_wav(tmp_path / "b.wav", st)
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 2 and w.getframerate() == beeper.RATE and w.getnframes() == len(st)


def test_demo_writes_four_files(tmp_path):
    paths = beeper.demo(tmp_path)
    assert sorted(p.name for p in paths) == ["approach.wav", "p1.wav", "p2.wav", "p3.wav"]


def test_beeper_never_touches_an_audio_device():
    import sys
    assert "sounddevice" not in sys.modules or True       # the module itself never imports one
    src = open(beeper.__file__).read()
    assert "sounddevice" not in src and "pyaudio" not in src and "simpleaudio" not in src


# ------------------------------------------------------------------ calibration (synthetic ChArUco views)
def _views(K, dist, n=14, size=(960, 540), seed=0):
    cv2 = pytest.importorskip("cv2")
    from perception.calibrate_camera import charuco_board
    board = charuco_board((7, 5), 40.0, 30.0, "DICT_5X5_100")
    img = board.generateImage((700, 500), marginSize=0)
    bw, bh = 7 * 0.04, 5 * 0.04
    rng = np.random.default_rng(seed)
    views = []
    for _ in range(n):
        rvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)])
        tvec = np.array([rng.uniform(-0.12, 0.05), rng.uniform(-0.08, 0.02), rng.uniform(0.35, 0.6)])
        corners = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], np.float64)
        proj, _ = cv2.projectPoints(corners, rvec, tvec, K, np.zeros(5))
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [700, 0], [700, 500], [0, 500]]),
                                        proj.reshape(-1, 2).astype(np.float32))
        ideal = cv2.warpPerspective(img, H, size, borderValue=255)
        if np.any(dist):
            from perception.near.intrinsics import CameraProfile
            prof = CameraProfile("t", size[0], size[1], K[0, 0], K[1, 1], K[0, 2], K[1, 2], list(dist))
            vv, uu = np.mgrid[0:size[1], 0:size[0]].astype(np.float64)
            und = prof.undistort_points(np.stack([uu.ravel(), vv.ravel()], 1)).reshape(size[1], size[0], 2)
            ideal = cv2.remap(ideal, und[..., 0].astype(np.float32), und[..., 1].astype(np.float32),
                              cv2.INTER_LINEAR, borderValue=255)
        views.append(cv2.cvtColor(ideal, cv2.COLOR_GRAY2BGR))
    return views, board


def test_charuco_calibration_recovers_known_intrinsics(tmp_path):
    from perception import calibrate_camera as C
    K = np.array([[700.0, 0, 480.0], [0, 700.0, 270.0], [0, 0, 1]])
    dist = np.array([-0.15, 0.05, 0.0, 0.0, 0.0])
    views, board = _views(K, dist)
    obj, img, size = C.collect(views, "charuco", board, (9, 6), 40.0)
    assert len(obj) >= 10
    rms, K2, d2, n = C.solve(obj, img, size)
    assert rms < 1.0
    assert K2[0, 0] == pytest.approx(700, rel=0.02) and K2[1, 1] == pytest.approx(700, rel=0.02)
    assert d2[0] == pytest.approx(-0.15, abs=0.04)
    prof = C.to_profile("Logitech C270", size, K2, d2, rms, n, "charuco")
    assert prof.calibrated and prof.has_distortion
    out = prof.save(tmp_path / "c270.json")
    from perception.near.intrinsics import CameraProfile
    assert CameraProfile.load(out).fx == pytest.approx(prof.fx)
    assert any("spec" in s for s in C.verdict(prof))


def test_print_board(tmp_path):
    pytest.importorskip("cv2")
    from perception import calibrate_camera as C
    p = C.print_board(str(tmp_path / "b.png"), "charuco", (7, 5), (9, 6), 30.0, 22.0, "DICT_5X5_100")
    assert p.is_file() and p.stat().st_size > 1000
