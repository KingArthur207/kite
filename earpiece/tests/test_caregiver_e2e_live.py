"""The caregiver line end to end, live: Grok Voice, a local SpacetimeDB and a silent earpiece.

Skipped unless KITE_LIVE_TESTS=1 with XAI_API_KEY available (environment, repo-root .env, or the
main checkout's .env in a git worktree) and the spacetime CLI, macOS `say` and openssl on PATH.
One Grok session, two short spoken turns; nothing is played aloud (see caregiver_e2e.py).
KITE_E2E_OUT=DIR keeps the audio, transcripts and logs there instead of a pytest temp dir."""

import os
import shutil
from pathlib import Path

import pytest

import caregiver_e2e

if os.environ.get("KITE_LIVE_TESTS"):
    caregiver_e2e.load_keys()
    os.environ["PATH"] = f"{Path.home() / '.local' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
LIVE = bool(os.environ.get("KITE_LIVE_TESTS") and os.environ.get("XAI_API_KEY")
            and shutil.which("spacetime") and shutil.which("say") and shutil.which("openssl"))


@pytest.mark.skipif(not LIVE, reason="live end-to-end test: set KITE_LIVE_TESTS=1 and XAI_API_KEY "
                                     "(needs the spacetime CLI, `say` and openssl)")
def test_caregiver_line_reaches_the_walkers_earpiece(tmp_path):
    out = Path(os.environ.get("KITE_E2E_OUT") or tmp_path)
    res = caregiver_e2e.run(out)
    failed = [f"{name}: {detail}" for name, ok, detail in res["checks"] if not ok]
    assert res["checks"] and not failed, "\n".join(failed) + f"\n(logs in {out})"
