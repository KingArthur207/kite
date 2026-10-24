"""scripts/imagine_scenes.py with the xAI API, the detector and the earpiece all faked: no network, no audio."""

import base64
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("imagine_scenes", ROOT / "scripts" / "imagine_scenes.py")
imagine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imagine)

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 64


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(imagine, "OUT", tmp_path)
    monkeypatch.setattr(imagine.time, "sleep", lambda s: None)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    return tmp_path


def test_every_scene_says_what_kite_should_do():
    assert set(imagine.SCENES) >= {"cyclist", "car_turning", "doorway", "oncoming", "scooter"}
    for name, (still, motion, expect) in imagine.SCENES.items():
        assert "chest" in still and motion and expect, name


def test_generate_writes_still_clip_and_manifest(out, monkeypatch):
    calls = []

    def fake_call(method, path, body=None, timeout=120):
        calls.append((method, path, body))
        if path == "/images/generations":
            return {"data": [{"b64_json": base64.b64encode(JPEG).decode()}]}
        if path == "/videos/generations":
            assert body["image"]["url"].startswith("data:image/jpeg;base64,")
            return {"request_id": "r1"}
        return {"status": "done", "video": {"url": "https://vidgen.x.ai/r1.mp4"}}

    monkeypatch.setattr(imagine, "call", fake_call)
    monkeypatch.setattr(imagine, "download", lambda url: b"MP4DATA")
    imagine.generate("cyclist", duration=6, force=False)

    assert (out / "cyclist.jpg").read_bytes() == JPEG
    assert (out / "cyclist.mp4").read_bytes() == b"MP4DATA"
    m = json.loads((out / "manifest.json").read_text())["cyclist"]
    assert m["image_model"] == imagine.IMAGE_MODEL and m["video_model"] == imagine.VIDEO_MODEL
    assert m["duration_s"] == 6 and "cyclist" in m["still_prompt"]
    assert [c[1] for c in calls] == ["/images/generations", "/videos/generations", "/videos/r1"]


def test_generate_skips_existing_files_unless_forced(out, monkeypatch):
    (out / "doorway.jpg").write_bytes(JPEG)
    (out / "doorway.mp4").write_bytes(b"old")
    monkeypatch.setattr(imagine, "call", lambda *a, **k: pytest.fail("no API call expected"))
    imagine.generate("doorway", duration=6, force=False)
    assert (out / "doorway.mp4").read_bytes() == b"old"


def test_failed_video_job_stops_with_a_message(out, monkeypatch):
    def fake_call(method, path, body=None, timeout=120):
        if path == "/images/generations":
            return {"data": [{"b64_json": base64.b64encode(JPEG).decode()}]}
        if path == "/videos/generations":
            return {"request_id": "r2"}
        return {"status": "failed", "error": "moderation"}

    monkeypatch.setattr(imagine, "call", fake_call)
    with pytest.raises(SystemExit, match="failed"):
        imagine.generate("scooter", duration=6, force=False)


@pytest.mark.parametrize("url", ["http://vidgen.x.ai/a.mp4", "file:///etc/passwd", "ftp://x/y"])
def test_download_refuses_anything_but_https(url):
    with pytest.raises(ValueError, match="non-https"):
        imagine.download(url)


def test_rehearse_summarises_detector_and_earpiece(out, monkeypatch):
    (out / "oncoming.mp4").write_bytes(b"clip")
    events = [
        {"type": "heartbeat", "t": 100.0},
        {"type": "obstacles", "t": 100.5, "objects": [{"kind": "person", "distance": 6.0}]},
        {"type": "obstacles", "t": 101.5, "objects": [{"kind": "person", "distance": 1.2}]},
    ]

    def fake_run(cmd, check, cwd, stdout=None):
        if any("standin_detect.py" in str(c) for c in cmd):   # the detector
            Path(cmd[cmd.index("--jsonl") + 1]).write_text("\n".join(json.dumps(e) for e in events))
        else:                                     # the earpiece render
            wav = Path(cmd[-1])
            wav.with_suffix(".captions.jsonl").write_text(json.dumps({"t": 1.4, "text": "Stop. Person ahead."}) + "\n")

    monkeypatch.setattr(imagine.subprocess, "run", fake_run)
    s = imagine.rehearse("oncoming")
    assert s["kinds"] == ["person"]
    assert s["first_seen_s"] == 0.5
    assert s["closest_m"] == 1.2
    assert s["would_stop"] is True
    assert s["kite_said"] == ["1.4s Stop. Person ahead."]
    assert json.loads((out / "oncoming.rehearsal.json").read_text()) == s


def test_rehearse_needs_the_clip(out):
    with pytest.raises(SystemExit, match="generate oncoming"):
        imagine.rehearse("oncoming")
