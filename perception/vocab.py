"""What the open-vocabulary detector is asked to find, and what the earpiece is told it found.

The detector gets a flat list of text prompts. Several prompts can name the same thing ("pole",
"street light pole", "utility pole"); each prompt maps to one earpiece `kind`. The earpiece's sonar
has its own sounds for person, vehicle, bike, curb and stairs_down; every other kind plays the
generic obstacle pip and is spoken as "Obstacle" (sonar.py `_SPOKEN_KIND`), so a specific kind such
as `pole` costs nothing on the audio side and still reaches the dashboard and the judgment layer.

A vocabulary file (--vocab) is JSON: {"kind": ["prompt", ...], ...}. It replaces the default.
`--prompt kind=a,b` adds prompts on the command line.
"""

from __future__ import annotations

import json
from pathlib import Path

# kind -> prompts. Order matters only for readability. Prompts were picked on the benchmark frames
# (perception/README.md): a prompt that fired on nothing true or kept firing on the wrong thing was dropped.
DEFAULT_VOCAB: dict[str, list[str]] = {
    "person": ["person"],
    "bike": ["bicycle"],
    "vehicle": ["car", "bus", "truck", "motorcycle", "van"],
    "e_scooter": ["parked scooter", "rental scooter", "electric scooter"],
    "pole": ["pole", "street light", "utility pole", "sign post", "parking meter", "traffic light pole"],
    "bollard": ["bollard"],
    "fire_hydrant": ["fire hydrant"],
    "construction": ["traffic cone", "construction barrier", "road barricade", "hole in the ground",
                     "open trench", "scaffolding"],
    "overhead": ["tree branch", "awning"],
    "tree": ["street tree", "tree trunk"],
    "bench": ["bench"],
    "trash_can": ["trash can", "wheelie bin"],
    "sign_board": ["sandwich board sign", "a-frame sign"],
    "street_furniture": ["mailbox", "newspaper box", "planter", "bike rack", "fire alarm box"],
    "stroller": ["stroller"],
    "dog": ["dog"],
    "stairs": ["stairs"],
    "curb": ["curb"],
}

# Typical real height (m) of each kind, for range from box size when no ground row or depth is usable.
TYPICAL_H = {"person": 1.7, "bike": 1.1, "vehicle": 1.5, "e_scooter": 1.1, "pole": 3.0, "bollard": 0.9,
             "fire_hydrant": 0.75, "construction": 0.9, "tree": 3.0, "bench": 0.8, "trash_can": 1.0,
             "sign_board": 1.0, "street_furniture": 1.1, "stroller": 1.0, "dog": 0.6, "stairs": 1.0,
             "curb": 0.15, "overhead": 0.6}

# Kinds that move by themselves; the rest are fixed street objects (they never get a crowd/vehicle hazard).
MOVING = {"person", "bike", "vehicle", "dog"}

# Kinds a cane may miss because they are not at ground level when seen from the chest; only `overhead`
# is decided by geometry (see fuse.height_band), the prompt list just names likely candidates.
ELEVATED_HINT = {"overhead"}


def load(path: str | Path | None = None, extra: list[str] | None = None) -> dict[str, list[str]]:
    """Default (or a JSON file's) vocabulary, plus `kind=prompt,prompt` strings from the command line."""
    vocab = {k: list(v) for k, v in DEFAULT_VOCAB.items()}
    if path:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
            raise ValueError(f"{path}: expected {{kind: [prompt, ...]}}")
        vocab = {str(k): [str(p) for p in v] for k, v in data.items()}
    for item in extra or []:
        kind, _, prompts = item.partition("=")
        if not kind or not prompts:
            raise ValueError(f"--prompt {item!r}: use kind=prompt[,prompt]")
        vocab.setdefault(kind.strip(), []).extend(p.strip() for p in prompts.split(",") if p.strip())
    return vocab


def flatten(vocab: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    """-> (prompts in a stable order without duplicates, prompt -> kind). A prompt listed under two
    kinds keeps the first."""
    prompts, kind_of = [], {}
    for kind, ps in vocab.items():
        for p in ps:
            p = " ".join(p.lower().split())
            if p and p not in kind_of:
                kind_of[p] = kind
                prompts.append(p)
    return prompts, kind_of


def spoken(kind: str) -> str:
    """Kind -> words for scene and hazard text ("e_scooter" -> "e-scooter")."""
    return {"e_scooter": "e-scooter", "fire_hydrant": "fire hydrant", "trash_can": "trash can",
            "sign_board": "sign board", "street_furniture": "street furniture", "stairs_down": "step down",
            "stairs_up": "steps up", "overhead": "overhead obstacle"}.get(kind, kind.replace("_", " "))
