#!/usr/bin/env bash
# One command to bring the Kite demo up and down on this Mac (docs/internal/demo-runbook.md).
#
#   scripts/demo.sh sim [--output <device>]              earpiece, relay, sim, dashboard, stand-in detector
#   scripts/demo.sh streetview [--output <device>]       earpiece, relay, Street View bridge, sim, dashboard, detector;
#                                                        the Moss St Street View walk opens in a new Chrome window
#   scripts/demo.sh live [camera] [--output <dev>]       earpiece, dashboard, detector on a camera (index, name or video file)
#   scripts/demo.sh replay <file.jsonl> [--output <dev>] earpiece, dashboard, scripts/replay.py
#   scripts/demo.sh bodycam [unoq|HOST|file] [--output <dev>]
#                                                        earpiece, dashboard, perception.near on the UNO Q body camera
#                                                        (found on the network; or its IP, or a video file to rehearse)
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
NEAR_DEPS="--with ultralytics --with lap --with transformers --with torch --with scipy --with opencv-python-headless --with python-dotenv"

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

still_up() {  # name: true if the process this script started under that name is running
  local name pid start
  while IFS='|' read -r name pid start; do
    [ "$name" = "$1" ] && alive_ours "$pid" "$start" && return 0
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
  [ "${1:-}" = --source ] && deps="$DETECT_DEPS --with opencv-python --with transformers --with av"  # --show needs GUI OpenCV; av opens cameras by name
  # shellcheck disable=SC2206
  [ -n "${KITE_DETECT_ARGS:-}" ] && extra+=(${KITE_DETECT_ARGS})
  # shellcheck disable=SC2086
  launch detector "$ROOT" env KITE_EARPIECE_PORT="$UDP_PORT" uv run --no-project $deps python sim/standin_detect.py "$@" ${extra[@]+"${extra[@]}"}
}

start_near() {  # source: unoq, unoq:HOST[:PORT] or a video file
  if [ ! -f "$ROOT/weights/yolo11n.pt" ]; then
    say "  skip near: weights/yolo11n.pt missing. Fix (once, with internet): uv run --no-project $NEAR_DEPS python -m perception --download"
    return 1
  fi
  local extra=()
  # shellcheck disable=SC2206
  [ -n "${KITE_NEAR_ARGS:-}" ] && extra+=(${KITE_NEAR_ARGS})
  # shellcheck disable=SC2086
  launch near "$ROOT" env KITE_EARPIECE_PORT="$UDP_PORT" uv run --no-project --python 3.12 $NEAR_DEPS \
    python -m perception.near --source "$1" ${extra[@]+"${extra[@]}"}
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
  # Every mode starts an earpiece, and the team's Cartesia plan allows one at a time, on any port.
  local ep; ep=$(pgrep -fl 'python.* -m earpiece run' | grep -v -- '--udp-port' | awk '{print $1}' | head -1)   # test earpieces on other ports (agents, --null, no key) don't count
  [ -n "$ep" ] && die "an earpiece is already running (pid $ep). Cartesia allows one: use that one, or stop it first."
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

cmd_streetview() {
  [ -f "$ROOT/sim/.env.local" ] || die "no sim/.env.local (the page's Google key). Fix: sh sim/tiles_env.sh"
  begin venv sim dashboard uv -- earpiece:$UDP_PORT:udp earpiece:$WS_PORT:tcp \
    relay:$RELAY_PORT:tcp sim:$SIM_PORT:tcp dashboard:$DASH_PORT:tcp
  say "starting the Street View walk on Moss St"
  start_earpiece
  launch relay "$ROOT" "$PY" sim/relay.py --port "$RELAY_PORT"
  launch bridge "$ROOT" "$PY" sim/streetview_bridge.py --relay "ws://127.0.0.1:$RELAY_PORT"
  start_vite sim "$SIM_PORT"
  start_vite dashboard "$DASH_PORT"
  start_detector --relay "ws://127.0.0.1:$RELAY_PORT/sub" || true
  wait_port relay "$RELAY_PORT" tcp 15; wait_port sim "$SIM_PORT" tcp 20
  wait_port dashboard "$DASH_PORT" tcp 20; wait_port earpiece "$WS_PORT" tcp 45
  still_up bridge || { say "  FAIL bridge exited. Last log lines:"; tail -n 3 "$LOGS/bridge.log" | sed 's/^/       /'; }
  local url="http://localhost:$SIM_PORT/?world=streetview&relay=ws://127.0.0.1:$RELAY_PORT/pub&ear=ws://127.0.0.1:$WS_PORT"
  if [ "${KITE_NO_OPEN:-0}" = 1 ]; then say "  open $url in Chrome"
  else open -na "Google Chrome" --args --new-window "$url"; fi
  open_url "http://localhost:$DASH_PORT"
  say "up. Street View: Up/Down next/previous panorama, Left/Right turn, B blacks out, V reveals. Stop with: scripts/demo.sh stop"
}

cmd_live() {
  local cam="${1:-0}"   # an index, a name (iPhone) or webcam:<index or name>; see: python -m kite.sources.open --list-cameras
  if [ -f "$cam" ]; then cam="$(cd "$(dirname "$cam")" && pwd)/$(basename "$cam")"  # a video file: rehearse without a camera
  elif [[ ! "$cam" =~ ^[0-9]+$ && "$cam" != webcam:* ]]; then cam="webcam:$cam"; fi
  begin venv dashboard uv -- earpiece:$UDP_PORT:udp earpiece:$WS_PORT:tcp dashboard:$DASH_PORT:tcp
  say "starting the live walk on camera $cam"
  start_earpiece
  start_vite dashboard "$DASH_PORT"
  # sim/live_walk.py is a helper module the stand-in imports, not a program; the live path is the stand-in on a camera.
  local det=1
  start_detector --source "$cam" --camera chest --show --cam-height 1.2 --pitch 15 || det=0
  wait_port dashboard "$DASH_PORT" tcp 20; wait_port earpiece "$WS_PORT" tcp 45
  open_url "http://localhost:$DASH_PORT"
  if [ $det = 1 ]; then  # it prints "[standin] cams=" once the camera is open; a camera that won't open ends it
    local i
    for ((i = 0; i < 120; i++)); do
      grep -q '^\[standin\] cams=' "$LOGS/detector.log" 2>/dev/null && { say "  ok   detector has the camera"; break; }
      still_up detector || { det=0; say "  FAIL detector exited. Last log lines:"; tail -n 2 "$LOGS/detector.log" | sed 's/^/       /'; break; }
      sleep 0.5
    done
  fi
  if [ $det = 1 ]; then
    say "up. The reveal window is the detector's: r replays the last 20 s, l goes live. Stop with: scripts/demo.sh stop"
  else
    say "earpiece and dashboard are up, but there is no detector, so the live walk will not stop anyone. Stop with: scripts/demo.sh stop"
  fi
}

cmd_bodycam() {
  local src="${1:-unoq}"
  if [ -f "$src" ]; then src="$(cd "$(dirname "$src")" && pwd)/$(basename "$src")"   # rehearse on a clip
  elif [[ "$src" != unoq && "$src" != unoq:* ]]; then src="unoq:$src"; fi          # a bare IP or host name
  begin venv dashboard uv -- earpiece:$UDP_PORT:udp earpiece:$WS_PORT:tcp dashboard:$DASH_PORT:tcp
  say "starting body camera -> priorities -> beeps on $src"
  start_earpiece
  start_vite dashboard "$DASH_PORT"
  local ok=1
  start_near "$src" || ok=0
  wait_port dashboard "$DASH_PORT" tcp 20; wait_port earpiece "$WS_PORT" tcp 45
  open_url "http://localhost:$DASH_PORT"
  if [ $ok = 1 ]; then   # it prints "[near] device=" once the board and the models are ready
    local i
    for ((i = 0; i < 240; i++)); do
      grep -q '^\[near\] device=' "$LOGS/near.log" 2>/dev/null && { say "  ok   near is reading the body camera"; break; }
      still_up near || { ok=0; say "  FAIL near exited. Last log lines:"; tail -n 3 "$LOGS/near.log" | sed 's/^/       /'; break; }
      sleep 0.5
    done
  fi
  if [ $ok = 1 ]; then
    say "up. P1 / P2 / P3 beep at 1228 / 812 / 630 Hz; watch: tail -f logs/near.log. Stop with: scripts/demo.sh stop"
  else
    say "earpiece and dashboard are up, but near is not running, so nothing will beep. Stop with: scripts/demo.sh stop"
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
  streetview) cmd_streetview ;;
  live) cmd_live "$@" ;;
  replay) cmd_replay "$@" ;;
  bodycam) cmd_bodycam "$@" ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
