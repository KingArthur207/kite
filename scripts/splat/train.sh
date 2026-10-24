#!/usr/bin/env bash
# Train a Gaussian splat with Brush (github.com/ArthurBrussee/brush, Rust + WebGPU, runs on Metal).
#
#   scripts/splat/train.sh <scene_dir> [--steps 30000] [--max-splats 3000000] [--res 1600] [--viewer]
#
# Reads <scene_dir>/images + sparse/0 (poses.sh). Writes <scene_dir>/out/splat_<step>.ply, a checkpoint
# every 5000 steps, so a run cut short still leaves something to load. Downloads the Brush v0.3.0
# macOS binary to ~/.local/splat-tools/brush the first time (checksum verified).
# Settings for a street walk: 3M splats cap (about 50 fps in the browser on an M-series Mac),
# 1600 px training images, growth stops at half the run so the end of the budget sharpens instead
# of adding floaters. --viewer opens Brush's window to watch it train.
set -euo pipefail

SCENE="" STEPS=30000 MAX=3000000 RES=1600 VIEWER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS="$2"; shift 2 ;;
    --max-splats) MAX="$2"; shift 2 ;;
    --res) RES="$2"; shift 2 ;;
    --viewer) VIEWER=--with-viewer; shift ;;
    *) SCENE="$1"; shift ;;
  esac
done
[ -n "$SCENE" ] || { sed -n 2,12p "$0"; exit 2; }
[ -d "$SCENE/sparse/0" ] || { echo "no $SCENE/sparse/0: run scripts/splat/poses.sh first" >&2; exit 1; }

VER=v0.3.0
DIR="$HOME/.local/splat-tools/brush"
BRUSH="$DIR/brush-app-aarch64-apple-darwin/brush_app"
if [ ! -x "$BRUSH" ]; then
  mkdir -p "$DIR"
  (cd "$DIR" && gh release download "$VER" -R ArthurBrussee/brush -p 'brush-app-aarch64-apple-darwin.tar.xz*' --clobber \
    && [ "$(shasum -a 256 brush-app-aarch64-apple-darwin.tar.xz | cut -d' ' -f1)" = "$(cut -d' ' -f1 brush-app-aarch64-apple-darwin.tar.xz.sha256)" ] \
    && tar xf brush-app-aarch64-apple-darwin.tar.xz && xattr -dr com.apple.quarantine brush-app-aarch64-apple-darwin || true)
  [ -x "$BRUSH" ] || { echo "could not install Brush to $DIR" >&2; exit 1; }
fi

OUT="$SCENE/out"
mkdir -p "$OUT"
s=$SECONDS
"$BRUSH" "$SCENE" $VIEWER \
  --total-steps "$STEPS" --max-splats "$MAX" --max-resolution "$RES" \
  --growth-stop-iter $((STEPS / 2)) \
  --export-every 5000 --export-path "$OUT" --export-name 'splat_{iter}.ply'
echo "== brush: $((SECONDS - s)) s for $STEPS steps; latest: $(ls -t "$OUT"/splat_*.ply | head -1)"
