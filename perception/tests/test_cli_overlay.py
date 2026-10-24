"""The perception CLI's guards, the review overlay and the weight-path helpers, with no model loaded."""

import json

import numpy as np
import pytest

from perception import __main__ as cli
from perception import models
from perception.fuse import Calib, Det
from perception.overlay import draw
from perception.pipeline import FrameResult


# ------------------------------------------------------------------ CLI guards (exit before any model loads)
def no_models(monkeypatch):
    monkeypatch.setattr(cli, "build", lambda *a, **k: pytest.fail("no model should load"))


def test_print_vocab_is_json_and_takes_extra_prompts(monkeypatch, capsys):
    no_models(monkeypatch)
    cli.main(["--print-vocab", "--prompt", "e_scooter=rental scooter on its side"])
    vocab = json.loads(capsys.readouterr().out)
    assert isinstance(vocab, dict) and "rental scooter on its side" in vocab["e_scooter"]


def test_a_source_is_required(monkeypatch):
    no_models(monkeypatch)
    with pytest.raises(SystemExit, match="--source is required"):
        cli.main([])


@pytest.mark.parametrize("source", ["webcam:0", "tello"])
def test_live_cameras_never_write_an_overlay(monkeypatch, source, tmp_path):
    """Privacy rule (spec 5.3): frames from a live camera are never stored, so --overlay is refused
    before a camera or a model is opened."""
    no_models(monkeypatch)
    with pytest.raises(SystemExit, match="never stored"):
        cli.main(["--source", source, "--overlay", str(tmp_path / "x.mp4"), "--no-send"])
    assert not (tmp_path / "x.mp4").exists()


def test_an_unknown_source_kind_exits_with_the_reason(monkeypatch):
    no_models(monkeypatch)
    with pytest.raises(SystemExit) as e:
        cli.main(["--source", "ftp://camera/feed", "--no-send"])
    assert str(e.value)                                   # a message, not a traceback


# ------------------------------------------------------------------ overlay
def result(objects, dets, events=()):
    return FrameResult(events=list(events), objects=objects, dets=dets, calib=Calib(4.0, 1.2, source="depth"),
                       timings={"det": 12.0, "depth": 30.0})


def test_overlay_draws_without_touching_the_frame():
    frame = np.full((240, 320, 3), 40, np.uint8)
    mask = np.zeros((240, 320), bool)
    mask[100:150, 100:150] = True
    dets = [Det(100, 100, 150, 150, "person", 0.9, tid="c1", mask=mask), Det(10, 10, 20, 20, "pole", 0.5, tid="o9")]
    objects = [{"id": "c1", "kind": "person", "distance": 1.2, "bearing": -3.0, "range": "ground", "stop": True}]
    events = [{"type": "hazard", "text": "Pole ahead."}, {"type": "obstacles", "objects": []}]
    before = frame.copy()
    out = draw(frame, result(objects, dets, events), "chest")
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert np.array_equal(frame, before)                  # the caller's frame is left alone
    assert not np.array_equal(out, frame)                 # something was drawn


def test_overlay_copes_with_an_empty_frame_result_and_odd_fields():
    frame = np.zeros((60, 80, 3), np.uint8)
    out = draw(frame, result([], []), "drone", title="t")
    assert out.shape == frame.shape
    # a detection whose object was dropped (no matching id) and an unknown kind / range are skipped or labelled
    objects = [{"id": "0", "kind": "unicorn", "distance": 3.0, "bearing": 10.0}]
    out = draw(frame, result(objects, [Det(1, 1, 30, 30, "unicorn", 0.4), Det(2, 2, 3, 3, "x", 0.1, tid="gone")]),
               "chest")
    assert out.shape == frame.shape


# ------------------------------------------------------------------ weights
def test_missing_weights_say_how_to_fetch_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        models._weights(str(tmp_path / "nope.pt"))
    with pytest.raises(FileNotFoundError, match="--download"):
        models._weights("definitely-not-downloaded.pt")
    real = tmp_path / "w.pt"
    real.write_bytes(b"x")
    assert models._weights(str(real)) == real


def test_embedded_weights_are_keyed_by_the_exact_prompt_list():
    a = models.embedded_path("yoloe-11s-seg.pt", ["pole", "curb"])
    assert a == models.embedded_path("yoloe-11s-seg.pt", ["pole", "curb"])
    assert a != models.embedded_path("yoloe-11s-seg.pt", ["curb", "pole"])          # order is part of the model
    assert a != models.embedded_path("yoloe-11s-seg.pt", ["pole", "curb", "stairs"])
    assert a.parent == models.WEIGHTS and a.name.startswith("world-yoloe-11s-seg-")
