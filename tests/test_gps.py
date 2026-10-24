"""kite.gps: phone GPS track -> position events, per-frame walker pose, sim metres. All synthetic, no phone."""

import csv
import importlib.util
import json
import math
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from kite import gps
from kite.record import FileReplay, Recorder, RecordCfg

ROOT = Path(__file__).resolve().parents[1]
PDT = timezone(timedelta(hours=-7))  # Oct 24 2026 in San Francisco
T0 = datetime(2026, 10, 24, 7, 0, 0, tzinfo=PDT).timestamp()


def earpiece_parse():
    """earpiece.events.parse from the repo. earpiece.spatial imports scipy for its audio filter, which kite's venv
    doesn't have; events.py only uses its pure-math wrap_deg, so a stand-in scipy is enough to import it."""
    sys.path.insert(0, str(ROOT / "earpiece"))
    stubs = {}
    try:
        import scipy.signal  # noqa: F401
    except ImportError:
        for name in ("scipy", "scipy.signal"):
            stubs[name] = types.ModuleType(name)
        stubs["scipy.signal"].lfilter = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub"))
        stubs["scipy"].signal = stubs["scipy.signal"]
        sys.modules.update(stubs)
    try:
        from earpiece.events import parse
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
    return parse


def load_replay():
    sys.path.insert(0, str(ROOT / "earpiece"))
    spec = importlib.util.spec_from_file_location("kite_replay_under_test", ROOT / "scripts" / "replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ a synthetic walk: the shoot route itself
WALK1_S, STOP_S, WALK2_S, SPEED = 116.0, 12.0, 30.0, 0.6
START = gps.planned_route()[0][1:]  # NE sidewalk of Moss, 70 m SE of the crossing


def truth(t_rel):
    """Walker (x, y) at seconds since T0: NW at 0.6 m/s, stand still, turn left (SW) at 0.6 m/s."""
    t = np.asarray(t_rel, float)
    nw, sw = np.radians(315.0), np.radians(225.0)
    a = np.clip(t, 0, WALK1_S) * SPEED
    b = np.clip(t - WALK1_S - STOP_S, 0, WALK2_S) * SPEED
    return START[0] + a * np.sin(nw) + b * np.sin(sw), START[1] + a * np.cos(nw) + b * np.cos(sw)


def noisy_fixes(seed=0, outlier=True, drift=1.0):
    """1 Hz fixes with phone-like noise: a slow drift of a few metres plus 0.5 m jitter, and one 40 m glitch.
    The drift moves at up to ~0.1 m/s, which no smoother can tell from walking: at 0.6 m/s it tilts the heading by
    up to ~10 deg (a real limit of phone GPS at walking speed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, WALK1_S + STOP_S + WALK2_S + 0.5, 1.0)
    x, y = truth(t)
    x = x + drift * (3 * np.sin(2 * np.pi * t / 240) + 1.5 * np.cos(2 * np.pi * t / 300 + 1)) + rng.normal(0, 0.5, len(t))
    y = y + drift * (2 * np.cos(2 * np.pi * t / 200) + 1.5 * np.sin(2 * np.pi * t / 180 + 2)) + rng.normal(0, 0.5, len(t))
    if outlier:
        x[30] += 40.0
    lat, lng = gps.sim_to_latlng(x, y)
    acc = rng.uniform(3, 10, len(t))
    return T0 + t, lat, lng, acc


def write_gpx(path, t, lat, lng, acc):
    """iOS-logger style: times with a -07:00 offset, Garmin TrackPointExtension speed, accuracy in extensions."""
    pts = "\n".join(
        f'<trkpt lat="{la:.8f}" lon="{ln:.8f}"><ele>12.0</ele>'
        f"<time>{datetime.fromtimestamp(ti, PDT).isoformat(timespec='milliseconds')}</time>"
        f"<extensions><gpxtpx:TrackPointExtension><gpxtpx:speed>0.6</gpxtpx:speed></gpxtpx:TrackPointExtension>"
        f"<accuracy>{a:.1f}</accuracy></extensions></trkpt>"
        for ti, la, ln, a in zip(t, lat, lng, acc))
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="test" '
                    'xmlns="http://www.topografix.com/GPX/1/1" '
                    'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">'
                    f"<trk><trkseg>\n{pts}\n</trkseg></trk></gpx>\n")
    return path


def write_csv(path, t, lat, lng, acc):
    """Android-logger style: local times WITHOUT a zone (needs --tz), 'Latitude' / 'Longitude' / 'Accuracy (m)'."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Time", "Latitude", "Longitude", "Accuracy (m)", "Altitude"])
        for ti, la, ln, a in zip(t, lat, lng, acc):
            w.writerow([datetime.fromtimestamp(ti, PDT).replace(tzinfo=None).isoformat(sep=" ", timespec="milliseconds"),
                        f"{la:.8f}", f"{ln:.8f}", f"{a:.1f}", "12"])
    return path


# ------------------------------------------------------------------ projection and times
def test_sim_projection_round_trip_and_convention():
    assert gps.latlng_to_sim(*gps.CROSSING) == (0.0, 0.0)
    rng = np.random.default_rng(1)
    lat = gps.CROSSING[0] + rng.uniform(-0.003, 0.003, 200)
    lng = gps.CROSSING[1] + rng.uniform(-0.003, 0.003, 200)
    x, y = gps.latlng_to_sim(lat, lng)
    la2, ln2 = gps.sim_to_latlng(x, y)
    assert np.abs(la2 - lat).max() < 1e-12 and np.abs(ln2 - lng).max() < 1e-12
    # x east, y north, metres; matches sim/src/config.ts toLatLng (111320 m/deg, cos(origin lat) for longitude)
    la, ln = gps.sim_to_latlng(100.0, 50.0)
    assert la == pytest.approx(gps.CROSSING[0] + 50 / 111320)
    assert ln == pytest.approx(gps.CROSSING[1] + 100 / (111320 * math.cos(math.radians(gps.CROSSING[0]))))
    # and it is a good tangent plane: 100 m east and north agree with the haversine distance to 0.5 %
    r = 6371008.8
    p1, p2 = map(math.radians, (gps.CROSSING[0], la))
    dl = math.radians(ln - gps.CROSSING[1])
    hav = 2 * r * math.asin(math.sqrt(math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))
    assert hav == pytest.approx(math.hypot(100, 50), rel=5e-3)
    assert gps.compass(1, 0) == 90 and gps.compass(0, 1) == 0 and gps.compass(-1, -1) == 225


def test_time_parsing_zones_and_epochs():
    want = datetime(2026, 10, 24, 14, 3, 12, 500000, tzinfo=timezone.utc).timestamp()
    la = gps.parse_tz("America/Los_Angeles")
    for s, tz in [("2026-10-24T14:03:12.5Z", timezone.utc), ("2026-10-24T07:03:12.500-07:00", timezone.utc),
                  ("2026-10-24T14:03:12.5+0000", timezone.utc), ("2026-10-24 07:03:12.5", la),
                  ("2026/10/24 07:03:12.5", gps.parse_tz("-07:00")), ("2026-10-24 14:03:12.5 UTC", la),
                  (f"{want:.1f}", timezone.utc), (f"{want * 1000:.0f}", timezone.utc),
                  (f"{want * 1e9:.0f}", timezone.utc)]:
        assert gps.parse_time(s, tz) == pytest.approx(want, abs=1e-3), s


def test_a_row_with_a_garbled_time_is_skipped_not_fatal(tmp_path):
    """One unreadable time (a logger killed mid-write, a hand edit) used to abort the whole track."""
    t, lat, lng, acc = noisy_fixes(seed=1)
    good = gps.read_track(write_csv(tmp_path / "w.csv", t, lat, lng, acc), gps.parse_tz("-07:00"))
    lines = (tmp_path / "w.csv").read_text().splitlines()
    lines[5] = "2026-10-24 07:0x:00.000," + lines[5].split(",", 1)[1]
    (tmp_path / "bad.csv").write_text("\n".join(lines) + "\n")
    fixes = gps.read_track(tmp_path / "bad.csv", gps.parse_tz("-07:00"))
    assert len(fixes) == len(good) - 1
    gpx = write_gpx(tmp_path / "w.gpx", t, lat, lng, acc).read_text()
    first = gpx.index("<time>")
    (tmp_path / "bad.gpx").write_text(gpx[:first] + "<time>not a time</time>" + gpx[gpx.index("</time>", first) + 7:])
    assert len(gps.read_track(tmp_path / "bad.gpx")) == len(good) - 1
    (tmp_path / "none.csv").write_text("time,lat,lon\nsoon,37.7783,-122.4087\nlater,37.7784,-122.4087\n")
    with pytest.raises(ValueError, match="unreadable time 'soon'"):  # nothing readable: still a clear error
        gps.read_track(tmp_path / "none.csv")


def test_a_fix_with_an_impossible_date_cannot_blow_up_the_grid(tmp_path):
    """Track sampled the whole time span at 10 Hz: one 1970 placeholder fix (loggers write epoch 0 before the first
    satellite fix) made it allocate ~1.8e10 grid points and die of MemoryError (or swap the laptop to a halt)."""
    t, lat, lng, acc = noisy_fixes(seed=2)
    good = len(gps.read_track(write_csv(tmp_path / "w.csv", t, lat, lng, acc), gps.parse_tz("-07:00")))
    t = t.copy()
    t[0], t[-1] = 0.0, 5e9  # epoch 0, and the year 2128
    fixes = gps.read_track(write_csv(tmp_path / "bad.csv", t, lat, lng, acc), gps.parse_tz("-07:00"))
    assert len(fixes) == good - 2
    gps.Track(fixes)  # fits
    wrong_year = fixes[:-1] + [gps.Fix(fixes[-1].t + 365 * 86400.0, fixes[-1].lat, fixes[-1].lng)]
    with pytest.raises(ValueError, match="wrong date"):  # a fix a year off: a clear error, not a 300 M point grid
        gps.Track(wrong_year)


def test_gpx_with_offset_and_csv_with_local_naive_times_read_the_same(tmp_path):
    t, lat, lng, acc = noisy_fixes()
    a = gps.read_track(write_gpx(tmp_path / "w.gpx", t, lat, lng, acc))
    b = gps.read_track(write_csv(tmp_path / "w.csv", t, lat, lng, acc), gps.parse_tz("America/Los_Angeles"))
    assert len(a) == len(b) == len(t)
    assert np.allclose([f.t for f in a], t, atol=1e-3) and np.allclose([f.t for f in b], t, atol=1e-3)
    assert np.allclose([f.lat for f in a], [f.lat for f in b]) and np.allclose([f.lng for f in a], lng)
    assert a[0].speed == 0.6 and a[0].ele == 12.0 and a[5].accuracy == pytest.approx(acc[5], abs=0.06)
    assert b[5].accuracy == pytest.approx(acc[5], abs=0.06)
    # the same CSV read as UTC is 7 h off: exactly the mistake the overlap report flags
    c = gps.read_track(tmp_path / "w.csv")
    assert c[0].t - a[0].t == pytest.approx(-7 * 3600, abs=1e-3)


# ------------------------------------------------------------------ smoothing, speed, heading
@pytest.mark.parametrize("drift, med_deg, p95_deg", [(0.0, 4, 12), (1.0, 15, 25)])
def test_speed_heading_stop_and_left_turn(tmp_path, drift, med_deg, p95_deg):
    tr = gps.Track(gps.read_track(write_gpx(tmp_path / "w.gpx", *noisy_fixes(drift=drift))))
    assert tr.n_outliers >= 1  # the 40 m glitch
    rel = np.arange(0, WALK1_S + STOP_S + WALK2_S, 0.2)
    s = tr.at(T0 + rel)
    assert s["valid"].all()

    def seg(lo, hi):
        return (rel >= lo) & (rel <= hi)

    def ang_err(h, want):
        return np.abs((h - want + 180) % 360 - 180)

    # smoothing (sigma 3 s) blurs each start and stop over a few seconds: test away from the transitions
    walk1, stop, walk2 = seg(4, WALK1_S - 4), seg(WALK1_S + 4, WALK1_S + STOP_S - 4), seg(WALK1_S + STOP_S + 4, 154)
    for seg_, want in ((walk1, 315), (walk2, 225)):  # drift biases the heading (see noisy_fixes); jitter alone not
        err = ang_err(s["heading"][seg_], want)
        assert np.median(err) < med_deg and np.percentile(err, 95) < p95_deg, (want, np.median(err))
    assert abs(np.median(s["speed"][walk1]) - SPEED) < 0.1 and abs(np.median(s["speed"][walk2]) - SPEED) < 0.1
    assert s["moving"][walk1].mean() > 0.95 and s["moving"][walk2].mean() > 0.95
    assert not s["moving"][stop].any() and np.isnan(s["heading"][stop]).all()  # None while standing still
    assert np.median(ang_err(s["heading_hold"][stop], 315)) < 10  # still facing the way they walked
    # smoothing pulls the noisy fixes towards the truth
    tx, ty = truth(rel)
    raw_err = np.hypot(tr.raw_x - truth(tr.t - T0)[0], tr.raw_y - truth(tr.t - T0)[1])
    assert np.median(np.hypot(s["x"] - tx, s["y"] - ty)) <= np.median(raw_err) + 0.1
    st = tr.stats()
    assert 80 < st["distance_m"] < 100  # 69.6 m + 18 m walked
    assert 0.5 < st["mean_walking_speed_mps"] < 0.7
    assert st["closest_to_crossing"]["distance_m"] < 6  # walked across the node, give or take the GPS drift
    assert st["duration_s"] == pytest.approx(WALK1_S + STOP_S + WALK2_S, abs=1)


def test_noise_free_heading_and_speed_are_exact():
    t = np.arange(0.0, 60.0, 1.0)
    x, y = 0.7 * t * math.sin(math.radians(40)), 0.7 * t * math.cos(math.radians(40))
    lat, lng = gps.sim_to_latlng(x, y)
    tr = gps.Track([gps.Fix(T0 + a, b, c) for a, b, c in zip(t, lat, lng)])
    s = tr.at(T0 + np.arange(5, 55, 0.5))
    assert np.allclose(s["heading"], 40, atol=0.01) and np.allclose(s["speed"], 0.7, atol=0.001)
    assert not tr.at([T0 - 1, T0 + 61])["valid"].any()


# ------------------------------------------------------------------ alignment with a kite.record take
def make_take(tmp_path, n=45, fps=30.0):
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for i in range(n):
        w.write(np.full((120, 160, 3), 5 * i % 250, np.uint8))
    w.release()
    src = FileReplay(path)
    try:
        return Recorder(src, None, RecordCfg(out_dir=str(tmp_path / "rec"), fps=src.fps),
                        {"source": str(path), "source_kind": "file"}).run()
    finally:
        src.stop()


def test_a_frames_csv_row_torn_by_a_crash_is_ignored(tmp_path):
    """A hard crash cuts frames.csv mid-row. A row cut inside t_wall kept a truncated number ("17617") as the last
    frame's wall time, which made the video span negative: no overlap, no position events."""
    rec = make_take(tmp_path)
    _, t_wall, _ = gps.read_recording(rec)
    text = (rec / "frames.csv").read_text()
    cut = text.rstrip("\n").rsplit("\n", 1)[0] + "\n45,46,123.456789,17617"  # torn inside t_wall, no src_t
    (rec / "frames.csv").write_text(cut)
    idx, tw, _ = gps.read_recording(rec)
    assert len(tw) == len(t_wall) - 1 and tw[-1] == pytest.approx(t_wall[-2])


def test_alignment_with_a_recording_and_replayable_events(tmp_path):
    rec = make_take(tmp_path)
    _, t_wall, meta = gps.read_recording(rec)
    assert len(t_wall) == 45 and abs(t_wall[0] - meta["started_wall"]) < 2
    offset = 12.5  # the phone's clock is 12.5 s ahead of the laptop's
    # noise-free straight walk NW in true (phone) time, from 30 s before the take to 60 s after
    t = np.arange(t_wall[0] + offset - 30, t_wall[0] + offset + 60, 1.0)
    walk = lambda tt: (-0.6 * (tt - t[0]) * math.sqrt(0.5), 0.6 * (tt - t[0]) * math.sqrt(0.5))  # noqa: E731
    lat, lng = gps.sim_to_latlng(*walk(t))
    track = tmp_path / "walk.csv"
    with open(track, "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["time", "lat", "lon", "accuracy"])
        for ti, la, ln in zip(t, lat, lng):
            cw.writerow([datetime.fromtimestamp(ti, timezone.utc).isoformat(), f"{la:.9f}", f"{ln:.9f}", 5])

    s = gps.process(track, rec, tmp_path / "out", offset_s=offset)
    v = s["recording"]
    assert v["overlap_frac_of_video"] == 1.0 and not v["warnings"] and v["frames_with_gps"] == 45
    assert v["gps_start_minus_video_start_s"] == pytest.approx(-30, abs=0.2)
    rows = list(csv.DictReader(open(tmp_path / "out" / "frames_gps.csv")))
    assert [int(r["index"]) for r in rows] == list(range(45))
    x, y = np.array([float(r["x"]) for r in rows]), np.array([float(r["y"]) for r in rows])
    tx, ty = walk(t_wall + offset)
    assert np.abs(x - tx).max() < 0.05 and np.abs(y - ty).max() < 0.05  # the pose OF THAT FRAME
    assert all(float(r["heading"]) == pytest.approx(315, abs=0.5) for r in rows)
    assert all(float(r["speed"]) == pytest.approx(0.6, abs=0.01) and r["moving"] == "1" for r in rows)

    # wrong offset: the walker is in the wrong place by offset * speed
    s0 = gps.process(track, rec, tmp_path / "out0", offset_s=0.0)
    rows0 = list(csv.DictReader(open(tmp_path / "out0" / "frames_gps.csv")))
    shift = math.hypot(float(rows0[0]["x"]) - x[0], float(rows0[0]["y"]) - y[0])
    assert shift == pytest.approx(offset * 0.6, abs=0.05) and s0["recording"]["overlap_frac_of_video"] == 1.0

    # every event is valid earpiece wire format, and replay.py schedules it on the video's clock
    parse = earpiece_parse()
    lines = (tmp_path / "out" / "positions.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    for ev in events:
        m = parse(dict(ev))
        assert m.type == "position" and m.source == "gps"
        assert m.data["lat"] == ev["lat"] and m.data["heading"] == ev["heading"] and m.data["place"]
    assert events[0]["t"] == 0.0 and events[0]["frame"] == 0
    plan = load_replay().schedule(load_replay().load(str(tmp_path / "out" / "positions.jsonl")))
    assert [round(at, 3) for at, _ in plan] == [ev["t"] for ev in events]
    assert all("t" not in ev for _, ev in plan)
    assert len(events) == int((t_wall[-1] - t_wall[0]) * 5) + 1


def test_no_overlap_and_time_zone_mistake_are_reported(tmp_path, capsys):
    rec = make_take(tmp_path, n=10)
    _, t_wall, _ = gps.read_recording(rec)
    t = np.arange(t_wall[0] - 7 * 3600 - 20, t_wall[0] - 7 * 3600 + 60, 1.0)  # a CSV read as UTC that was PDT
    lat, lng = gps.sim_to_latlng(0.5 * (t - t[0]), 0 * t)
    track = tmp_path / "walk.gpx"
    write_gpx(track, t, lat, lng, np.full(len(t), 5.0))
    assert gps.main([str(track), "--recording", str(rec), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "NO OVERLAP" in out and "time-zone" in out
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert s["recording"]["overlap_s"] == 0 and s["recording"]["frames_with_gps"] == 0
    assert (tmp_path / "o" / "positions.jsonl").read_text() == ""
    # the suggested offset fixes it
    sugg = s["recording"]["gps_start_minus_video_start_s"]
    s2 = gps.process(track, rec, tmp_path / "o2", offset_s=sugg)
    assert s2["recording"]["overlap_frac_of_video"] == 1.0


# ------------------------------------------------------------------ CLI on a track alone, and the planned route
def test_cli_track_only_outputs(tmp_path):
    parse = earpiece_parse()
    gpx = write_gpx(tmp_path / "w.gpx", *noisy_fixes(seed=3))
    assert gps.main([str(gpx), "--rate", "2"]) == 0
    out = tmp_path / "w_gps"
    events = [json.loads(line) for line in (out / "positions.jsonl").read_text().splitlines()]
    assert events[0]["t"] == 0.0 and len(events) == pytest.approx(2 * (WALK1_S + STOP_S + WALK2_S), abs=2)
    for ev in events:
        parse(ev)
    assert all("heading" in ev for ev in events[20:])
    rows = list(csv.DictReader(open(out / "track.csv")))
    assert len(rows) == len(noisy_fixes()[0]) - 1  # the glitch is dropped
    x, y = gps.latlng_to_sim(float(rows[0]["lat"]), float(rows[0]["lng"]))
    assert (x, y) == (pytest.approx(float(rows[0]["x"]), abs=0.01), pytest.approx(float(rows[0]["y"]), abs=0.01))
    s = json.loads((out / "summary.json").read_text())
    assert s["vs_planned_route"]["median_m"] < 6 and "closest_to_crossing" in s["track"]


def test_planned_route(tmp_path):
    parse = earpiece_parse()
    info = gps.write_route(tmp_path)
    assert info["length_m"] == pytest.approx(70 + 10 + 20, abs=0.1)
    corners = np.array([c[1:] for c in info["corners"]])
    # the zebra leg passes over the crossing node and is perpendicular to Moss; first leg runs NW (315 deg)
    assert gps.distance_to_polyline(0.0, 0.0, corners)[0] < 1e-9
    d = np.diff(corners, axis=0)
    assert gps.compass(*d[0]) == pytest.approx(315, abs=0.1) and gps.compass(*d[1]) == pytest.approx(225, abs=0.1)
    assert gps.compass(*d[2]) == pytest.approx(225, abs=0.1)
    for line in (tmp_path / "shoot_route.jsonl").read_text().splitlines():
        parse(line)
    fixes = gps.read_track(tmp_path / "shoot_route.gpx")
    assert len(fixes) == info["points"] and fixes[-1].t - fixes[0].t == pytest.approx(info["duration_s"], abs=0.1)
    # the committed copy in data/ is what this code writes
    committed = ROOT / "data" / "shoot_route.jsonl"
    if committed.exists():
        assert committed.read_text() == (tmp_path / "shoot_route.jsonl").read_text()
