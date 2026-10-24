#!/usr/bin/env bash
# One command to bring the Kite demo up and down on this Mac (docs/demo-runbook.md).
#
#   scripts/demo.sh sim [--output <device>]              earpiece, relay, sim, dashboard, stand-in detector
#   scripts/demo.sh live [camera-index] [--output <dev>] earpiece, dashboard, detector on a real camera
#   scripts/demo.sh replay <file.jsonl> [--output <dev>] earpiece, dashboard, scripts/replay.py
#   scripts/demo.sh stop                                 stops only what this script started
#   scripts/demo.sh status                               what is up, last log lines, netcheck
#
# Env: KITE_OUTPUT=<device> (same as --output), KITE_SPACETIME=0 (no SpacetimeDB mirroring),
#      KITE_OFFLINE=1 (run scripts/offline.sh first and use the local SpacetimeDB),
#      KITE_NO_OPEN=1 (don't open the browser), KITE_DETECT_ARGS (extra detector flags, e.g. --depth),
#      KITE_EARPIECE_ARGS (extra earpiece flags, e.g. --null --no-stem),
#      KITE_PORT_OFFSET=N (testing only: add N to every port; the sim page still talks to 5011 and 5020).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/logs"
PIDS="$LOGS/pids"
PY="$ROOT/earpiece/.venv/bin/python"
OFF="${KITE_PORT_OFFSET:-0}"
UDP_PORT=$((5010 + OFF)) WS_PORT=$((5011 + OFF)) RELAY_PORT=$((5020 + OFF))
SIM_PORT=$((5173 + OFF)) DASH_PORT=$((5174 + OFF))
export PYTHONUNBUFFERED=1   # so logs/ fills as things happen
DETECT_DEPS="--with ultralytics --with lap --with websockets --with python-dotenv --with numpy --with scipy"

say() { printf '%s\n' "$*"; }
die() { printf 'demo: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- process bookkeeping
started() { ps -o lstart= -p "$1" 2>/dev/null | tr -s ' '; }

alive_ours() {  # pid recorded-start-time: true only if that exact process is still running
  [ -n "$1" ] && [ "$(started "$1")" = "$2" ]
}

any_running() {
  [ -f "$PIDS" ] || return 1
  while IFS='|' read -r name pid start; do
    alive_ours "$pid" "$start" && return 0
  done < "$PIDS"
  return 1
}

launch() {  # name dir cmd...   runs in its own process group so stop gets its children too
  local name="$1" dir="$2"; shift 2
  local log="$LOGS/$name.log"
  set -m
  ( cd "$dir" && exec nohup "$@" ) >"$log" 2>&1 </dev/null &
  local pid=$!
  set +m
  printf '%s|%s|%s\n' "$name" "$pid" "$(started "$pid")" >> "$PIDS"
  say "  started $name (pid $pid, log logs/$name.log)"
}

holder() {  # port proto: "pid command" of whatever listens there, empty if free
  local p
  if [ "$2" = udp ]; then p=$(lsof -nP -t -iUDP:"$1" 2>/dev/null | head -1)
  else p=$(lsof -nP -t -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | head -1); fi
  [ -n "$p" ] && printf '%s %s' "$p" "$(ps -o command= -p "$p" | sed "s|$ROOT/||g" | cut -c1-80)"
}

wait_port() {  # name port proto seconds
  local i
  for ((i = 0; i < $4 * 2; i++)); do
    [ -n "$(holder "$2" "$3")" ] && { say "  ok   $1 is listening on $2"; return 0; }
    sleep 0.5
  done
  say "  FAIL $1 did not open port $2 within $4 s. Last log lines:"
  tail -n 5 "$LOGS/$1.log" | sed 's/^/       /'
  return 1
}

# ---------------------------------------------------------------- prerequisites
check_prereqs() {  # what the mode needs: words from venv sim dashboard uv
  local bad=0 need
  for need in "$@"; do
    case "$need" in
      venv)
        [ -x "$PY" ] || { say "missing earpiece/.venv. Fix: cd earpiece && uv venv --python 3.12 .venv && uv pip install --python .venv -e \".[mac,brain,memory,dev]\""; bad=1; } ;;
      sim|dashboard)
        [ -d "$ROOT/$need/node_modules" ] || { say "missing $need/node_modules. Fix: cd $need && npm install"; bad=1; } ;;
      uv)
        command -v uv >/dev/null || { say "uv not found. Fix: curl -LsSf https://astral.sh/uv/install.sh | sh"; bad=1; } ;;
    esac
  done
  [ -f "$ROOT/.env" ] || say "note: no .env at the repo root, so no Cartesia or Claude (cp .env.example .env and fill in keys)"
  return $bad
}

check_ports() {  # "name:port:proto" ...  refuses to start over anything already listening
  local bad=0 spec name port proto h
  for spec in "$@"; do
    IFS=: read -r name port proto <<< "$spec"
    h=$(holder "$port" "$proto")
    [ -n "$h" ] && { say "port $port ($name) is held by pid $h. Not starting over it: stop that process (scripts/demo.sh stop, if this script started it)."; bad=1; }
  done
  return $bad
}

# ---------------------------------------------------------------- pieces
earpiece_cmd() {
  local cmd=("$PY" -m earpiece run --udp-port "$UDP_PORT" --ws-port "$WS_PORT")
  [ "${KITE_SPACETIME:-1}" = 0 ] || cmd+=(--spacetime)
  [ -n "$OUTPUT" ] && cmd+=(--output "$OUTPUT")
  # shellcheck disable=SC2206
  [ -n "${KITE_EARPIECE_ARGS:-}" ] && cmd+=(${KITE_EARPIECE_ARGS})
  printf '%s\n' "${cmd[@]}"
}

start_earpiece() {
  local cmd=(); while IFS= read -r a; do cmd+=("$a"); done < <(earpiece_cmd)
  launch earpiece "$ROOT/earpiece" "${cmd[@]}"
}

start_vite() {  # sim|dashboard port
  launch "$1" "$ROOT/$1" ./node_modules/.bin/vite --port "$2" --strictPort
}

start_detector() {  # extra standin flags...
  if [ ! -f "$ROOT/sim/standin_detect.py" ]; then
    say "  skip detector: sim/standin_detect.py is not on this branch yet (PR #14). Everything else runs."
    return 1
  fi
  if [ ! -f "$ROOT/weights/yolo11n.pt" ]; then
    say "  skip detector: weights/yolo11n.pt missing. Fix (once, with internet): uv run --no-project $DETECT_DEPS --with opencv-python-headless python sim/standin_detect.py --download"
    return 1
  fi
  local deps="$DETECT_DEPS --with opencv-python-headless" extra=()
  [ "${1:-}" = --source ] && deps="$DETECT_DEPS --with opencv-python --with transformers"  # --show needs GUI OpenCV
  # The detector always sends to port 5010, so with a port offset it must not send (it would feed someone else's earpiece).
  [ "$OFF" != 0 ] && { extra+=(--no-send); say "  note: KITE_PORT_OFFSET is set, so the detector runs with --no-send"; }
  # shellcheck disable=SC2206
  [ -n "${KITE_DETECT_ARGS:-}" ] && extra+=(${KITE_DETECT_ARGS})
  # shellcheck disable=SC2086
  launch detector "$ROOT" uv run --no-project $deps python sim/standin_detect.py "$@" ${extra[@]+"${extra[@]}"}
}

open_url() {
  [ "${KITE_NO_OPEN:-0}" = 1 ] && { say "  open $1 in the browser"; return; }
  open "$1"
}

go_offline() {
  [ "${KITE_OFFLINE:-0}" = 1 ] || return 0
  say "KITE_OFFLINE=1: switching SpacetimeDB to the local server"
  "$ROOT/scripts/offline.sh" || die "scripts/offline.sh failed, see above"
  export SPACETIME_HOST=http://127.0.0.1:3000   # the earpiece and replay read this before .env
}

# ---------------------------------------------------------------- modes
begin() {  # common start checks; args are check_prereqs words, then -- then port specs
  local pre=() ports=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do pre+=("$1"); shift; done
  shift; ports=("$@")
  any_running && die "Kite is already running from this script. Run scripts/demo.sh stop first (or status to see it)."
  local ok=0
  check_prereqs "${pre[@]}" || ok=1
  check_ports "${ports[@]}" || ok=1
  [ $ok = 0 ] || die "not starting. Fix the lines above and run it again."
  mkdir -p "$LOGS"; : > "$PIDS"
  go_offline
}

cmd_sim() {
  begin venv sim dashboard -- earpiece:$UDP_PORT:udp earpiece:$WS_PORT:tcp \
    relay:$RELAY_PORT:tcp sim:$SIM_PORT:tcp dashboard:$DASH_PORT:tcp
  say "starting the sim demo"
  start_earpiece
  launch relay "$ROOT" "$PY" sim/relay.py --port "$RELAY_PORT"
  start_vite sim "$SIM_PORT"
  start_vite dashboard "$DASH_PORT"
  start_detector --relay "ws://127.0.0.1:$RELAY_PORT/sub" || true
  wait_port relay "$RELAY_PORT" tcp 15; wait_port sim "$SIM_PORT" tcp 20
  wait_port dashboard "$DASH_PORT" tcp 20; wait_port earpiece "$WS_PORT" tcp 45
  open_url "http://localhost:$SIM_PORT"; open_url "http://localhost:$DASH_PORT"
  say "up. Sim: arrow keys walk, B blacks out, V reveals. Stop everything with: scripts/demo.sh stop"
}

cmd_live() {
  local cam="${1:-0}"
  [[ "$cam" =~ ^[0-9]+$ ]] || die "camera index must be a number (try: uv run --no-project --with opencv-python python sim/standin_detect.py --list-cameras)"
  begin venv dashboard uv -- earpiece:$UDP_PORT:udp earpiece:$WS_PORT:tcp dashboard:$DASH_PORT:tcp
  say "starting the live walk on camera $cam"
  start_earpiece
  start_vite dashboard "$DASH_PORT"
  # sim/live_walk.py is a helper module the stand-in imports, not a program; the live path is the stand-in on a camera.
  local det=1
  start_detector --source "$cam" --camera chest --show --cam-height 1.2 --pitch 15 || det=0
  wait_port dashboard "$DASH_PORT" tcp 20; wait_port earpiece "$WS_PORT" tcp 45
  open_url "http://localhost:$DASH_PORT"
  if [ $det = 1 ]; then
    say "up. The reveal window is the detector's: r replays the last 20 s, l goes live. Stop with: scripts/demo.sh stop"
  else
    say "earpiece and dashboard are up, but there is no detector, so the live walk will not stop anyone. Stop with: scripts/demo.sh stop"
  fi
}

cmd_replay() {
  local file="${1:-}"
  [ -n "$file" ] || die "usage: scripts/demo.sh replay <file.jsonl>   (try data/sample_walk.jsonl)"
  [ -f "$file" ] || die "no such file: $file"
  file="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"
  begin venv dashboard -- earpiece:$UDP_PORT:udp earpiece:$WS_PORT:tcp dashboard:$DASH_PORT:tcp
  say "starting a replay of $file"
  start_earpiece
  start_vite dashboard "$DASH_PORT"
  wait_port dashboard "$DASH_PORT" tcp 20; wait_port earpiece "$WS_PORT" tcp 45 || die "earpiece is not up, not replaying. scripts/demo.sh stop, then fix and retry."
  local rp=("$PY" scripts/replay.py "$file" --port "$UDP_PORT")
  [ "${KITE_SPACETIME:-1}" = 0 ] || rp+=(--spacetime)
  launch replay "$ROOT" "${rp[@]}"
  open_url "http://localhost:$DASH_PORT"
  say "up. The replay plays once (logs/replay.log). Stop with: scripts/demo.sh stop"
}

cmd_stop() {
  [ -f "$PIDS" ] || { say "nothing to stop (no logs/pids)"; return; }
  local name pid start n=0
  while IFS='|' read -r name pid start; do
    if alive_ours "$pid" "$start"; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      say "stopped $name (pid $pid)"; n=$((n + 1))
    fi
  done < "$PIDS"
  sleep 1
  while IFS='|' read -r name pid start; do  # anything that ignored TERM
    alive_ours "$pid" "$start" && { kill -KILL -- "-$pid" 2>/dev/null; say "killed $name (pid $pid)"; }
  done < "$PIDS"
  rm -f "$PIDS"
  [ $n -gt 0 ] || say "nothing was running"
}

cmd_status() {
  local spec name port proto h mine
  say "== ports"
  for spec in earpiece-udp:$UDP_PORT:udp earpiece-ws:$WS_PORT:tcp relay:$RELAY_PORT:tcp \
              sim:$SIM_PORT:tcp dashboard:$DASH_PORT:tcp; do
    IFS=: read -r name port proto <<< "$spec"
    h=$(holder "$port" "$proto")
    if [ -z "$h" ]; then say "  down  $name :$port"; else say "  up    $name :$port  (pid $h)"; fi
  done
  say "== started by this script"
  if [ -f "$PIDS" ]; then
    while IFS='|' read -r name pid start; do
      if alive_ours "$pid" "$start"; then mine=running; else mine=exited; fi
      say "  $name (pid $pid) $mine"
      [ -f "$LOGS/$name.log" ] && tail -n 3 "$LOGS/$name.log" | sed 's/^/      /'
    done < "$PIDS"
  else
    say "  nothing (no logs/pids)"
  fi
  say "== netcheck"
  python3 "$ROOT/scripts/netcheck.py" --udp-port "$UDP_PORT" --ws-port "$WS_PORT"
}

# ---------------------------------------------------------------- main
MODE="${1:-}"; [ $# -gt 0 ] && shift
OUTPUT="${KITE_OUTPUT:-}"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --output) [ $# -ge 2 ] || die "--output needs a device name"; OUTPUT="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

case "$MODE" in
  sim) cmd_sim ;;
  live) cmd_live "$@" ;;
  replay) cmd_replay "$@" ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
