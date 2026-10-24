#!/usr/bin/env bash
# POV demo video in one command: detector on the footage -> what the walker hears -> overlay + audio.
#
#   scripts/pov.sh out/shots/pov-moss.mp4                       # sim POV (record_shot.py pov), defaults fit it
#   scripts/pov.sh clips/moss_phone.mp4 --fx 700 --pitch 15     # phone at chest height
#   scripts/pov.sh clip.mp4 --jsonl events.jsonl                # skip the detector, use these events
#   scripts/pov.sh clip.mp4 -o out/pov/take2.mp4 --seconds 20   # quick try
#
# Writes <out>.events.jsonl (detector), <out>.wav and <out>.captions.jsonl (earpiece render), <out> (mp4).
# Default out: out/pov/<video name>.mp4. The detector is sim/standin_detect.py on the chest path with the
# Depth Anything fallback (catches bins and signs YOLO has no class for). Lens defaults are the sim POV shot's:
# 1.3 m up, 10 deg down, fx 700 px at 960 wide (an iPhone 1x is about the same; measure the phone's pitch).
# The render speaks through Cartesia when CARTESIA_API_KEY is in .env, one request at a time, cached phrases
# first; else the offline voice. KITE_EARPIECE_PY picks the earpiece venv's python (default earpiece/.venv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${KITE_EARPIECE_PY:-$ROOT/earpiece/.venv/bin/python}"
DETECT_DEPS=(--with ultralytics --with lap --with websockets --with python-dotenv --with numpy --with scipy
             --with opencv-python-headless --with transformers)

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
[ $# -ge 1 ] || usage 1
case "$1" in -h|--help) usage 0 ;; esac
VIDEO="$1"; shift
EVENTS="" OUT="" FX=700 PITCH=10 HEIGHT=1.3 SECONDS_ARG=() BADGE="Kite sees"
while [ $# -gt 0 ]; do
  case "$1" in
    --jsonl) EVENTS="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    --fx) FX="$2"; shift 2 ;;
    --pitch) PITCH="$2"; shift 2 ;;
    --cam-height) HEIGHT="$2"; shift 2 ;;
    --badge) BADGE="$2"; shift 2 ;;
    --seconds) SECONDS_ARG=(--seconds "$2"); shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "pov.sh: unknown option $1" >&2; usage 1 ;;
  esac
done
[ -f "$VIDEO" ] || { echo "pov.sh: no video at $VIDEO" >&2; exit 1; }
[ -x "$PY" ] || { echo "pov.sh: no earpiece python at $PY (set KITE_EARPIECE_PY)" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "pov.sh: ffmpeg missing (brew install ffmpeg)" >&2; exit 1; }
abs() { case "$1" in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }
VIDEO="$(abs "$VIDEO")"
OUT="$(abs "${OUT:-$ROOT/out/pov/$(basename "${VIDEO%.*}").mp4}")"
[ "$OUT" != "$VIDEO" ] || { echo "pov.sh: -o would overwrite the input video" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
STEM="${OUT%.mp4}"

if [ -z "$EVENTS" ]; then
  EVENTS="$STEM.events.jsonl"
  echo "== 1/3 detector -> $EVENTS"
  (cd "$ROOT" && uv run --no-project "${DETECT_DEPS[@]}" python sim/standin_detect.py --source "$VIDEO" \
     --camera chest --cam-height "$HEIGHT" --fx "$FX" --pitch "$PITCH" --depth \
     --jsonl "$EVENTS" --no-send --fast)   # whole video: --max-seconds is wall time, not video time
else
  EVENTS="$(abs "$EVENTS")"
  echo "== 1/3 events from $EVENTS (detector skipped)"
fi

echo "== 2/3 earpiece render -> $STEM.wav (real time: a 60 s walk takes 60 s)"
(cd "$ROOT/earpiece" && KITE_CARTESIA_PREWARM=1 "$PY" -m earpiece render --jsonl "$EVENTS" "$STEM.wav" \
   ${SECONDS_ARG[@]+"${SECONDS_ARG[@]}"})

echo "== 3/3 overlay -> $OUT"
(cd "$ROOT" && uv run --no-project --with ultralytics --with opencv-python-headless --with numpy python video/overlay.py \
   "$VIDEO" --events "$EVENTS" --wav "$STEM.wav" --blur-faces --badge "$BADGE" -o "$OUT" \
   ${SECONDS_ARG[@]+"${SECONDS_ARG[@]}"})
ffprobe -v error -show_entries format=duration:stream=codec_type -of compact "$OUT"
