#!/usr/bin/env bash
# Camera poses for a video walk with COLMAP (brew install colmap; 4.x has GLOMAP built in).
#
#   scripts/splat/poses.sh <scene_dir> [--mapper global|incremental] [--no-loops]
#
# Reads <scene_dir>/images (from video_to_frames.sh). One shared camera (one phone, focus and zoom
# locked), SIFT on the CPU (no CUDA on a Mac), sequential matching for a walk plus vocab-tree loop
# detection so the two sidewalk passes and the zebra crossing tie together, then the global mapper
# (GLOMAP; minutes, where the incremental mapper can take much longer), then undistortion to pinhole.
# Leaves <scene_dir>/images (undistorted) and <scene_dir>/sparse/0, the layout Brush reads; the raw
# frames move to <scene_dir>/frames.
set -euo pipefail

SCENE="" MAPPER=global LOOPS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --mapper) MAPPER="$2"; shift 2 ;;
    --no-loops) LOOPS=0; shift ;;
    *) SCENE="$1"; shift ;;
  esac
done
[ -n "$SCENE" ] || { sed -n 2,11p "$0"; exit 2; }
command -v colmap >/dev/null || { echo "colmap not found: brew install colmap" >&2; exit 1; }
[ -d "$SCENE/frames" ] || mv "$SCENE/images" "$SCENE/frames"
W="$SCENE/colmap"
rm -rf "$W" "$SCENE/sparse" "$SCENE/images"
mkdir -p "$W/sparse"
t() { local s=$SECONDS; "$@"; echo "== $1 $2: $((SECONDS - s)) s"; }

t colmap feature_extractor --database_path "$W/db.db" --image_path "$SCENE/frames" \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 0 --FeatureExtraction.max_image_size 1600 --SiftExtraction.max_num_features 4096 --log_level 1

t colmap sequential_matcher --database_path "$W/db.db" --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 15 --SequentialMatching.loop_detection "$LOOPS" --log_level 1

if [ "$MAPPER" = global ]; then
  t colmap global_mapper --database_path "$W/db.db" --image_path "$SCENE/frames" --output_path "$W/sparse" --log_level 1
else
  t colmap mapper --database_path "$W/db.db" --image_path "$SCENE/frames" --output_path "$W/sparse" --log_level 1
fi

# The mapper can split a walk into several models when a stretch fails to match; keep the biggest.
best="" best_n=0
for m in "$W"/sparse/*/; do
  n=$(colmap model_analyzer --path "$m" 2>&1 | sed -n 's/.*Registered images: \([0-9]*\).*/\1/p' | head -1)
  echo "model $(basename "$m"): ${n:-0} images registered"
  if [ "${n:-0}" -gt "$best_n" ]; then best="$m" best_n="$n"; fi
done
[ -n "$best" ] || { echo "COLMAP registered no images. Check the frames: blur, glare, too little overlap." >&2; exit 1; }
echo "using $(basename "$best"): $best_n of $(ls "$SCENE/frames" | wc -l | tr -d ' ') frames"

t colmap image_undistorter --image_path "$SCENE/frames" --input_path "$best" --output_path "$W/undistorted" \
  --output_type COLMAP --log_level 1
mv "$W/undistorted/images" "$SCENE/images"
mkdir -p "$SCENE/sparse/0"
mv "$W/undistorted/sparse/"* "$SCENE/sparse/0/"
echo "poses ready: $SCENE/images and $SCENE/sparse/0"
