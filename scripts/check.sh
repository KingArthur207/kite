#!/usr/bin/env bash
# Every check Kite has, run locally. Replaces GitHub Actions; run it before you merge.
#
#   scripts/check.sh            everything: tests, lint, build, security
#   scripts/check.sh quick      earpiece tests (Rust engine), Rust lint, secret scan of the working tree
#   scripts/check.sh <step>...  any of: earpiece rust drone dashboard secrets audit
#
# Needs: uv, cargo, npm. Optional: gitleaks and cargo-audit (brew install gitleaks; cargo install cargo-audit).
# Exit code is non-zero if any step failed; every step still runs.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EAR="$ROOT/earpiece"
PY="$EAR/.venv/bin/python"
DRONE_VENV="$ROOT/.venv"
FAILED=()
SKIPPED=()

bold() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run() {  # run <label> <command...>
  local label="$1"; shift
  printf -- '-- %s\n' "$label"
  if ! "$@"; then FAILED+=("$label"); fi
}
skip() { SKIPPED+=("$1 ($2)"); printf -- '-- %s: skipped, %s\n' "$1" "$2"; }

earpiece_venv() {
  if [ ! -x "$PY" ]; then
    (cd "$EAR" && uv venv -q --python 3.12 .venv && uv pip install -q --python .venv/bin/python -e ".[brain,memory,dev]")
  fi
  if ! "$PY" -c "import kite_audio" 2>/dev/null; then
    uv pip install -q --python "$PY" maturin
    (cd "$EAR" && VIRTUAL_ENV="$EAR/.venv" .venv/bin/maturin develop -q --release -m native/Cargo.toml)
  fi
}

step_earpiece() {
  bold "earpiece (Python, both audio engines)"
  earpiece_venv
  run "earpiece, rust engine" bash -c "cd '$EAR' && KITE_AUDIO_ENGINE=rust '$PY' -m pytest -q"
  if [ "${1:-}" != quick ]; then
    run "earpiece, python engine" bash -c "cd '$EAR' && KITE_AUDIO_ENGINE=python '$PY' -m pytest -q"
  fi
}

step_rust() {
  bold "rust (fmt, clippy, unit tests)"
  local d="$EAR/native"
  run "cargo fmt" bash -c "cd '$d' && cargo fmt --check"
  run "cargo clippy" bash -c "cd '$d' && cargo clippy -q --release --all-targets -- -D warnings"
  [ "${1:-}" = quick ] || run "cargo test" bash -c "cd '$d' && PYO3_PYTHON='$PY' cargo test -q --release --lib"
}

step_drone() {
  bold "drone layer (kite/)"
  if [ ! -x "$DRONE_VENV/bin/python" ]; then
    uv venv -q --python 3.12 "$DRONE_VENV"
  fi
  uv pip install -q --python "$DRONE_VENV/bin/python" -e "$ROOT[dev]"
  # a few drone tests measure wall-clock timing; rerun failures once before calling them real
  run "drone tests" bash -c "cd '$ROOT' && { '$DRONE_VENV/bin/python' -m pytest -q -m 'not slow' || '$DRONE_VENV/bin/python' -m pytest -q -m 'not slow' --lf; }"
  run "sim scene-describe tests" bash -c "cd '$ROOT' && '$PY' -m pytest -q sim/test_scene_describe.py"
}

step_dashboard() {
  bold "dashboard (build, module typecheck)"
  run "dashboard build" bash -c "cd '$ROOT/dashboard' && npm ci --silent && npm run -s build"
  run "spacetime module typecheck" bash -c "cd '$ROOT/dashboard/spacetimedb' && npm ci --silent && npx tsc --noEmit -p ."
}

step_secrets() {
  bold "secret scan (gitleaks)"
  if ! command -v gitleaks >/dev/null; then skip "gitleaks" "not installed: brew install gitleaks"; return; fi
  # committed history, then uncommitted changes to tracked files (never .env, venvs or node_modules)
  run "gitleaks, full history" bash -c "cd '$ROOT' && gitleaks git --redact --no-banner --log-level warn --exit-code 1 ."
  run "gitleaks, uncommitted changes" bash -c "cd '$ROOT' && gitleaks git --pre-commit --redact --no-banner --log-level warn --exit-code 1 . && gitleaks git --pre-commit --staged --redact --no-banner --log-level warn --exit-code 1 ."
}

step_audit() {
  bold "dependency and code audit"
  earpiece_venv
  run "pip-audit" uvx -q pip-audit --path "$EAR/.venv/lib/python3.12/site-packages" --progress-spinner off
  run "bandit" uvx -q bandit -q -r "$EAR/earpiece" -ll
  run "npm audit, dashboard" bash -c "cd '$ROOT/dashboard' && npm audit --audit-level=high"
  run "npm audit, spacetime module" bash -c "cd '$ROOT/dashboard/spacetimedb' && npm audit --audit-level=high"
  if command -v cargo-audit >/dev/null; then
    run "cargo audit" bash -c "cd '$EAR/native' && cargo audit -q"
  else
    skip "cargo audit" "not installed: cargo install cargo-audit"
  fi
}

case "${1:-all}" in
  all)   step_earpiece; step_rust; step_drone; step_dashboard; step_secrets; step_audit ;;
  quick) step_earpiece quick; step_rust quick; step_secrets quick ;;
  *)     for s in "$@"; do
           case "$s" in
             earpiece|rust|drone|dashboard|secrets|audit) "step_$s" ;;
             *) echo "check: unknown step '$s'" >&2; exit 2 ;;
           esac
         done ;;
esac

echo
[ ${#SKIPPED[@]} -eq 0 ] || printf 'skipped: %s\n' "${SKIPPED[@]}"
if [ ${#FAILED[@]} -eq 0 ]; then
  printf '\033[32mall checks passed\033[0m\n'
else
  printf '\033[31mfailed:\033[0m %s\n' "${FAILED[@]}"
  exit 1
fi
