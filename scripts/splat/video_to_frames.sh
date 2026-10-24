#!/usr/bin/env bash
# Phone walkthrough video(s) -> sharp frames for COLMAP.
#
#   scripts/splat/video_to_frames.sh <scene_dir> <video>... [--frames 300] [--long-side 1920]
#
# Writes <scene_dir>/images/c<clip>_<n>.jpg. For each clip it decodes 3 candidates per kept frame,
# spread evenly over the clip, and keeps the sharpest of each 3 (variance of the Laplacian), so motion
# blur from footsteps drops out. --frames is the total across clips, split by clip length.
# Needs ffmpeg, ffprobe, bc and uv.
set -euo pipefail

FRAMES=300 LONG=1920 SCENE="" VIDEOS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --frames) FRAMES="$2"; shift 2 ;;
    --long-side) LONG="$2"; shift 2 ;;
    *) if [ -z "$SCENE" ]; then SCENE="$1"; else VIDEOS+=("$1"); fi; shift ;;
  esac
done
[ -n "$SCENE" ] && [ ${#VIDEOS[@]} -gt 0 ] || { sed -n 2,9p "$0"; exit 2; }
for v in "${VIDEOS[@]}"; do [ -f "$v" ] || { echo "no such video: $v" >&2; exit 1; }; done

dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
TOTAL=0
for v in "${VIDEOS[@]}"; do TOTAL=$(echo "$TOTAL + $(dur "$v")" | bc -l); done

mkdir -p "$SCENE/images"
CAND="$(mktemp -d)"
trap 'rm -rf "$CAND"' EXIT
i=0
for v in "${VIDEOS[@]}"; do
  d=$(dur "$v")
  keep=$(echo "($FRAMES * $d / $TOTAL + 0.5) / 1" | bc)
  fps=$(echo "3 * $keep / $d" | bc -l)
  echo "clip $i: $(basename "$v"), ${d%.*} s, keeping $keep frames (candidates at $(printf %.2f "$fps") fps)"
  mkdir -p "$CAND/$i"
  # ffmpeg autorotates phone video. Scale the long side to $LONG, never up.
  ffmpeg -v error -hwaccel videotoolbox -i "$v" -an \
    -vf "fps=$fps,scale='if(gt(iw,ih),min($LONG,iw),-2)':'if(gt(iw,ih),-2,min($LONG,ih))'" \
    -q:v 2 "$CAND/$i/%06d.jpg"
  i=$((i + 1))
done

uv run -q --no-project --with opencv-python-headless python -I - "$CAND" "$SCENE/images" <<'PY'
import os, shutil, sys
import cv2

cand, out = sys.argv[1], sys.argv[2]
sharp = lambda p: cv2.Laplacian(cv2.imread(p, cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var()
kept = 0
for clip in sorted(os.listdir(cand), key=int):
    files = sorted(os.listdir(os.path.join(cand, clip)))
    for g in range(0, len(files), 3):
        best = max((os.path.join(cand, clip, f) for f in files[g:g + 3]), key=sharp)
        shutil.copy(best, os.path.join(out, f"c{clip}_{g // 3:05d}.jpg"))
        kept += 1
print(f"{kept} frames -> {out}")
PY
