#!/usr/bin/env bash
# End-to-end test of the Kite link on one computer, with no UNO Q: board.py (synthetic camera) and
# kite_link.py run as separate processes and talk over real sockets on this machine's LAN IP, and over the
# real SpacetimeDB (Maincloud) for discovery and the text fallback, as source `unoq-test` (about a hundred
# small calls in total). Prints PASS or FAIL per check; exits non-zero if any failed.
#
#   hardware/arduino/unoq_link/e2e_local.sh
#
# Needs python3 and uv (Pillow decodes the frames, as an independent JPEG decoder). Opens no camera, mic or
# speaker. The token is generated here and never printed.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${KITE_E2E_PORT:-18765}"
SOURCE="unoq-test"
WORK="$(mktemp -d)"
TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
export KITE_TOKEN="$TOKEN"
IP="$(python3 -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("192.0.2.1", 9)); print(s.getsockname()[0])')"
PASSED=0
FAILED=0
BOARD_PID=""

check() {  # check <name> <condition exit status> <detail>
  if [ "$2" -eq 0 ]; then
    PASSED=$((PASSED + 1)); printf 'PASS  %s: %s\n' "$1" "$3"
  else
    FAILED=$((FAILED + 1)); printf 'FAIL  %s: %s\n' "$1" "$3"
  fi
}
start_board() {
  KITE_PORT="$PORT" KITE_SOURCE="$SOURCE" KITE_CAMERA=synthetic KITE_VIDEO_FPS=10 \
    python3 "$HERE/board.py" >> "$WORK/board.log" 2>&1 &
  BOARD_PID=$!
  for _ in $(seq 1 50); do
    curl -fs "http://$IP:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 0.2
  done
  return 1
}
stop_board() { [ -n "$BOARD_PID" ] && kill "$BOARD_PID" 2>/dev/null; wait "$BOARD_PID" 2>/dev/null; BOARD_PID=""; }
cleanup() { stop_board; rm -rf "$WORK"; }
trap cleanup EXIT
link() { python3 "$HERE/kite_link.py" --source "$SOURCE" --token-file "$WORK/token" "$@"; }
decode() {  # decode <folder>: "decoded/total WxH" using Pillow
  uv run -q --no-project --with pillow python - "$1" <<'PY'
import sys, pathlib
from PIL import Image
files = sorted(pathlib.Path(sys.argv[1]).glob("*.jpg"))
ok, size = 0, None
for f in files:
    try:
        im = Image.open(f); im.load(); ok += 1; size = im.size
    except Exception:
        pass
print(f"{ok}/{len(files)} {size[0] if size else 0}x{size[1] if size else 0}")
PY
}

echo "kite link end-to-end on $IP:$PORT, SpacetimeDB source '$SOURCE'"

start_board; check "board starts" $? "board.py answers /health on $IP:$PORT (bound to every interface)"

out="$(link token "$TOKEN")"; check "token saved" $? "kite_link.py token stores it for the client"

# discovery: the board's signed heartbeat in SpacetimeDB carries its LAN IP
found=1
for _ in $(seq 1 15); do
  out="$(link where 2>&1)"
  if echo "$out" | grep -q "board: $IP:$PORT (direct)"; then found=0; break; fi
  sleep 1
done
check "discovery via SpacetimeDB" $found "$(echo "$out" | grep -m1 heartbeat || echo "$out" | tail -1)"

out="$(link send "hello from the e2e test, ünïcødé ✓ 世界" 2>&1)"
echo "$out" | grep -q "via direct $IP:$PORT" && echo "$out" | grep -q 'got "hello from the e2e test, ünïcødé ✓ 世界"'
check "text computer to board (direct)" $? "$out"
grep -q "message 1 (direct): hello from the e2e test, ünïcødé ✓ 世界" "$WORK/board.log"
check "board logged the text" $? "$(grep -m1 'message 1' "$WORK/board.log")"

long="$(python3 -c 'print("x" * 600)')"
out="$(link send "$long" 2>&1)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q "longer than 500"
check "long text refused" $? "client: $out"
code="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"text\": \"$long\"}" "http://$IP:$PORT/send")"
if [ "$code" = 400 ]; then rc=0; else rc=1; fi; check "long text refused by the board too" "$rc" "HTTP $code"

out="$(link ping -n 3 2>&1)"; echo "$out" | grep -q "round trip min"
check "ping" $? "$(echo "$out" | tail -1)"

out="$(link listen --count 3 2>&1)"
ns="$(echo "$out" | sed -n 's/.*direct   hello \([0-9]*\) from.*/\1/p' | tr '\n' ' ')"
# shellcheck disable=SC2086  # one argument per hello number
python3 -c "import sys; n=[int(x) for x in sys.argv[1:]]; sys.exit(0 if len(n)==3 and n==sorted(n) and len(set(n))==3 else 1)" $ns
check "recurring hello board to computer (direct, in order)" $? "hello $ns"

out="$(link --cloud listen --count 1 --seconds 20 2>&1)"; echo "$out" | grep -q "internet hello"
check "recurring hello through SpacetimeDB" $? "$(echo "$out" | grep -m1 'internet hello')"

out="$(link --cloud send "through the internet" 2>&1)"
echo "$out" | grep -q 'via internet (SpacetimeDB)' && echo "$out" | grep -q 'got "through the internet"'
check "text through SpacetimeDB (forced)" $? "$out"

out="$(link --board 127.0.0.1:1 send "fallback when direct is blocked" 2>&1)"
echo "$out" | grep -q 'via internet (SpacetimeDB)'
check "automatic fallback to SpacetimeDB" $? "direct path closed: $out"

code="$(curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer wrong-token-wrong-token' "http://$IP:$PORT/status")"
code2="$(curl -s -o /dev/null -w '%{http_code}' "http://$IP:$PORT/video.mjpg")"
if [ "$code" = 401 ] && [ "$code2" = 401 ]; then rc=0; else rc=1; fi; check "wrong or missing token gives 401" "$rc" "status $code, video $code2"

out="$(link watch --save "$WORK/frames" --seconds 8 2>&1)"
fps="$(echo "$out" | sed -n 's/.*: \([0-9.]*\) fps.*/\1/p')"
dec="$(decode "$WORK/frames")"
python3 -c "import sys; f=float(sys.argv[1] or 0); sys.exit(0 if 8.5 <= f <= 11 else 1)" "$fps"
check "video fps over the network" $? "$out"
n_ok="${dec%%/*}"; rest="${dec#*/}"; n_all="${rest%% *}"
[ "$n_ok" = "$n_all" ] && [ "$n_all" -ge 60 ] && echo "$dec" | grep -q "640x480"
check "every video frame decodes" $? "$dec frames decoded by Pillow"

curl -s -o "$WORK/snap.jpg" -H "Authorization: Bearer $TOKEN" "http://$IP:$PORT/snapshot.jpg"
mkdir -p "$WORK/snap" && mv "$WORK/snap.jpg" "$WORK/snap/"
dec="$(decode "$WORK/snap")"; if [ "${dec%% *}" = "1/1" ]; then rc=0; else rc=1; fi; check "snapshot decodes" "$rc" "$dec"

link watch --save "$WORK/v1" --seconds 5 > "$WORK/v1.txt" 2>&1 &
p1=$!
link watch --save "$WORK/v2" --seconds 5 > "$WORK/v2.txt" 2>&1 &
p2=$!
wait $p1 $p2
a="$(sed -n 's/saved \([0-9]*\) frames.*/\1/p' "$WORK/v1.txt")"; b="$(sed -n 's/saved \([0-9]*\) frames.*/\1/p' "$WORK/v2.txt")"
if [ "${a:-0}" -ge 25 ] && [ "${b:-0}" -ge 25 ]; then rc=0; else rc=1; fi; check "two viewers at once" "$rc" "viewer 1: ${a:-0} frames, viewer 2: ${b:-0} frames in 5 s"

out="$(link doctor 2>&1)"; rc=$?
echo "$out" > "$WORK/doctor.txt"
[ $rc -eq 0 ] && ! echo "$out" | grep -q '^FAIL' && echo "$out" | grep -q 'PASS  heartbeat' \
  && echo "$out" | grep -q 'PASS  text (internet)' && echo "$out" | grep -q 'PASS  video'
check "doctor: all good" $? "$(echo "$out" | grep -c '^PASS') PASS, $(echo "$out" | grep -c '^WARN') WARN (camera: test pattern)"
sed 's/^/        /' "$WORK/doctor.txt"

out="$(link --token wrong-token-wrong-token --board "$IP:$PORT" doctor --skip-cloud 2>&1)"
echo "$out" | grep -q 'FAIL  token check' && echo "$out" | grep -q 'token mismatch'
check "doctor: wrong token" $? "$(echo "$out" | grep -m1 'FAIL  token check')"

out="$(link --board 192.0.2.1:"$PORT" doctor --skip-cloud 2>&1)"
echo "$out" | grep -q 'FAIL  reach' && echo "$out" | grep -q 'eduroam blocks device-to-device'
check "doctor: board unreachable" $? "$(echo "$out" | grep -m1 'FAIL  reach')"

out="$(link --source unoq-test-nobody doctor --skip-cloud 2>&1)"
echo "$out" | grep -q 'FAIL  heartbeat: no heartbeat'
check "doctor: no heartbeat" $? "$(echo "$out" | grep -m1 'FAIL  heartbeat')"

# board restart while a client listens: the client reconnects and keeps printing hellos
link listen --seconds 12 > "$WORK/listen.txt" 2>&1 &
lp=$!
sleep 3
stop_board
sleep 1
start_board >/dev/null
wait $lp
after="$(sed -n '/reconnecting/,$p' "$WORK/listen.txt" | grep -c 'direct   hello')"
grep -q reconnecting "$WORK/listen.txt" && [ "$after" -ge 2 ]
check "board restart: client reconnects" $? "$after hellos after the restart"

stop_board
echo
echo "$PASSED passed, $FAILED failed"
[ "$FAILED" -eq 0 ]
