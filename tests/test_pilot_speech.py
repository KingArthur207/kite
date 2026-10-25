"""run/pilot.py speaks the drone's lines through PowerShell on Windows. The text must never become part of the
command: PowerShell also closes a '...' string at a curly quote, so escaping ' alone is not enough."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1] / "run" / "pilot.py"


def load_pilot():
    spec = importlib.util.spec_from_file_location("kite_run_pilot", PILOT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod              # dataclasses look their module up while the file runs
    spec.loader.exec_module(mod)
    return mod


def test_windows_speech_passes_text_out_of_band(monkeypatch):
    pilot = load_pilot()
    calls = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: calls.append((args, kw)))
    say = pilot.speaker()
    evil = "Holding\u2019); Remove-Item -Recurse C:\\ ; ('"
    say(evil)
    (args, kw), = calls
    assert args[-1] == pilot.WIN_SAY and "$env:KITE_SAY" in args[-1]
    assert not any(evil in a or "Remove-Item" in a for a in args)
    assert kw["env"]["KITE_SAY"] == evil
