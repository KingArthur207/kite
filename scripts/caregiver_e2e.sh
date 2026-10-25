#!/usr/bin/env bash
# Headless end-to-end test of the Grok caregiver line (no speaker, no microphone, no camera).
#   scripts/caregiver_e2e.sh [--out DIR]
# Starts a throwaway SpacetimeDB on a free 127.0.0.1 port (own data dir, keys and CLI config;
# Maincloud and ~/.config/spacetime are never touched), publishes dashboard/spacetimedb to it,
# runs the walker's earpiece with --null against it, streams two `say`-synthesised questions to
# Grok as the microphone, and checks that the caregiver's message reached the earpiece.
# Prints PASS/FAIL per check; exit 0 only if all pass. Audio, transcripts and logs stay in --out
# (default earpiece/.cache/caregiver-e2e/<time>/).
# Keys: the repo-root .env, or the main checkout's .env when run from a git worktree.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
MAIN="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null | sed 's#/\.git$##' || true)"

for f in "$ROOT/.env" "${MAIN:+$MAIN/.env}"; do
  if [ -n "$f" ] && [ -f "$f" ]; then
    # .env is the user's own git-ignored file, so there is nothing for shellcheck to follow
    set -a
    # shellcheck source=/dev/null
    . "$f"                          # never echoed
    set +a
    break
  fi
done
[ -n "${XAI_API_KEY:-}" ] || { echo "FAIL  XAI_API_KEY is not set (repo-root .env)"; exit 2; }
command -v spacetime >/dev/null || { echo "FAIL  spacetime CLI not found (expected ~/.local/bin/spacetime)"; exit 2; }

PY="${KITE_PYTHON:-}"
for c in "$ROOT/earpiece/.venv/bin/python" "${MAIN:+$MAIN/earpiece/.venv/bin/python}"; do
  [ -z "$PY" ] && [ -n "$c" ] && [ -x "$c" ] && PY="$c"
done
PY="${PY:-python3}"

cd "$ROOT/earpiece"
PYTHONPATH="$ROOT/earpiece" exec "$PY" tests/caregiver_e2e.py "$@"
