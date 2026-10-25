"""Downloaded checkpoints are data, not code: every torch.load in perception/ and kite/ passes weights_only=True,
so a tampered .pth can't run a pickle payload. A source check, so it needs no torch."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def torch_loads():
    for d in ("perception", "kite", "sim", "run"):
        for f in (ROOT / d).rglob("*.py"):
            if "/tests/" in f.as_posix() or "/.venv/" in f.as_posix():
                continue
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load"
                        and "torch" in ast.unparse(node.func.value) and "hub" not in ast.unparse(node.func.value)):
                    yield f.relative_to(ROOT), node


def test_every_torch_load_is_weights_only():
    found = list(torch_loads())
    assert found, "expected at least one torch.load (perception/depth_zoo.py)"
    for path, call in found:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw.get("weights_only"), ast.Constant) and kw["weights_only"].value is True, \
            f"{path}:{call.lineno} torch.load without weights_only=True"
