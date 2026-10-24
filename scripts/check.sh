#!/usr/bin/env bash
# Every check Kite has, run locally. Replaces GitHub Actions; run it before you merge.
#
#   scripts/check.sh            everything: tests, lint, build, security
#   scripts/check.sh quick      earpiece tests (Rust engine, in parallel), Python and Rust lint, site, secret scan
#   scripts/check.sh <step>...  any of: earpiece rust drone dashboard site lint secrets audit
#   scripts/check.sh hooks      install a git pre-commit hook (ruff + gitleaks on what you stage, ~2 s)
#
# Needs: uv, cargo, npm. Optional: gitleaks and cargo-audit (brew install gitleaks; cargo install cargo-audit).
# Exit code is non-zero if any step failed; every step still runs.
# KITE_NO_DEVICE_TESTS=1 skips the one test that plays sound on the real output device.
# HYPOTHESIS_PROFILE=thorough runs the fuzz tests with thousands of examples instead of 60.
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
  [ -x "$PY" ] || (cd "$EAR" && uv venv -q --python 3.12 .venv)
  # every run: a no-op when satisfied, and it picks up dev dependencies added since the venv was made
  # (camera = OpenCV, for the sim scene-describe tests that run on this venv; ble = bless, for phone/)
  uv pip install -q --python "$PY" -e "${EAR}[brain,memory,dev,camera,ble]" maturin
  # rebuild the Rust engine when it is missing or older than its sources, so tests never run a stale build
  local so
  so="$("$PY" -c 'import glob, os, kite_audio; d = os.path.dirname(kite_audio.__file__); print(max(glob.glob(d + "/*.so"), key=os.path.getmtime))' 2>/dev/null)"
  if [ -z "$so" ] || [ -n "$(find "$EAR/native/src" "$EAR/native/Cargo.toml" -newer "$so" -print -quit)" ]; then
    (cd "$EAR" && VIRTUAL_ENV="$EAR/.venv" .venv/bin/maturin develop -q --release -m native/Cargo.toml)
  fi
}

pytest_earpiece() {  # pytest_earpiece <engine>: 4 workers, then failures once more on their own
  # a few tests measure wall-clock timing and can lose under load; a real failure fails twice
  local env="KITE_AUDIO_ENGINE=$1"
  bash -c "cd '$EAR' && $env '$PY' -m pytest -q -n 4" && return 0
  echo "-- rerunning the failures serially (wall-clock tests can lose under parallel load)"
  bash -c "cd '$EAR' && $env '$PY' -m pytest -q --lf"
}

step_earpiece() {
  bold "earpiece (Python, both audio engines)"
  earpiece_venv
  run "earpiece, rust engine" pytest_earpiece rust
  if [ "${1:-}" != quick ]; then
    run "earpiece, python engine" pytest_earpiece python
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
  uv pip install -q --python "$DRONE_VENV/bin/python" -e "${ROOT}[dev]"
  # a few drone tests measure wall-clock timing; rerun failures once before calling them real
  run "drone tests" bash -c "cd '$ROOT' && { '$DRONE_VENV/bin/python' -m pytest -q -m 'not slow' || '$DRONE_VENV/bin/python' -m pytest -q -m 'not slow' --lf; }"
  earpiece_venv
  run "sim scene-describe tests" bash -c "cd '$ROOT' && '$PY' -m pytest -q sim/test_scene_describe.py"
  # geometry, depth fusion, hazards and event contract with fake models; `-m slow` adds the real models
  run "perception tests" bash -c "cd '$ROOT' && '$PY' -m pytest -q perception/tests"
  # the phone's Bluetooth link with no radio: fix parsing, chunk reassembly, GATT writes, server build (bless)
  run "phone link tests" bash -c "cd '$ROOT' && '$PY' -m pytest -q phone/tests"
  # self-check scripts (no pytest functions): the relay's origin rule, the stand-in detector's geometry
  run "sim relay self-check" bash -c "cd '$ROOT' && '$PY' sim/test_relay.py"
  run "sim stand-in geometry self-check" bash -c "cd '$ROOT' && '$PY' sim/test_standin_geometry.py"
}

step_dashboard() {
  bold "dashboard (build, module typecheck)"
  run "dashboard build" bash -c "cd '$ROOT/dashboard' && npm ci --silent && npm run -s build"
  run "spacetime module typecheck" bash -c "cd '$ROOT/dashboard/spacetimedb' && npm ci --silent && npx tsc --noEmit -p ."
}

step_site() {
  bold "site (generated pages in sync, Markdown negotiation, 404s, structured data)"
  run "site pages rebuilt from site/src" bash -c "cd '$ROOT' && python3 site/tools/build_pages.py >/dev/null && git diff --quiet -- site/public || { echo 'site/public is stale: run python3 site/tools/build_pages.py and commit'; exit 1; }"
  run "site tests" bash -c "cd '$ROOT' && node --test site/tests/*.test.mjs"
}

step_lint() {
  bold "python lint (ruff, config in pyproject.toml)"
  run "ruff" uvx -q ruff check "$ROOT"
}

step_precommit() {  # what the git hook runs: fast checks on the staged change only
  local files
  files="$(cd "$ROOT" && git diff --cached --name-only --diff-filter=ACM -- '*.py')"
  if [ -n "$files" ]; then
    run "ruff, staged files" bash -c "cd '$ROOT' && git diff --cached --name-only --diff-filter=ACM -z -- '*.py' | xargs -0 uvx -q ruff check --force-exclude"
  fi
  if command -v gitleaks >/dev/null; then
    run "gitleaks, staged" bash -c "cd '$ROOT' && gitleaks git --pre-commit --staged --redact --no-banner --log-level warn --exit-code 1 ."
  else
    skip "gitleaks" "not installed: brew install gitleaks"
  fi
}

install_hooks() {
  local hook
  hook="$(cd "$ROOT" && git rev-parse --git-path hooks)/pre-commit"
  case "$hook" in /*) ;; *) hook="$ROOT/$hook" ;; esac
  if [ -e "$hook" ] && ! grep -q "scripts/check.sh precommit" "$hook"; then
    echo "check: $hook already exists and is not ours; leaving it alone" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$hook")"
  printf '#!/usr/bin/env bash\n# installed by scripts/check.sh hooks\nexec "$(git rev-parse --show-toplevel)/scripts/check.sh" precommit\n' > "$hook"
  chmod +x "$hook"
  echo "installed $hook (skip once with: git commit --no-verify)"
  exit 0
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
  all)   step_earpiece; step_rust; step_drone; step_dashboard; step_site; step_lint; step_secrets; step_audit ;;
  quick) step_earpiece quick; step_rust quick; step_lint; step_site; step_secrets quick ;;
  hooks) install_hooks ;;
  *)     for s in "$@"; do
           case "$s" in
             earpiece|rust|drone|dashboard|site|lint|precommit|secrets|audit) "step_$s" ;;
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
