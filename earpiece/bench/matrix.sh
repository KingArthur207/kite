#!/bin/sh
# Run the earpiece suite under every engine combination (no devices, nothing played aloud).
#   bench/matrix.sh [logdir]      from earpiece/, with the venv's python first on PATH
set -u
cd "$(dirname "$0")/.."
LOG=${1:-.cache/matrix}
mkdir -p "$LOG"
PY=${PYTHON:-.venv/bin/python}
run() {
    name=$1; shift
    env KITE_NO_DEVICE_TESTS=1 "$@" "$PY" -m pytest -q -p no:cacheprovider > "$LOG/$name.log" 2>&1
    echo "$name: $(tail -1 "$LOG/$name.log")"
}
run audio-rust_sonar-rust           KITE_AUDIO_ENGINE=rust   KITE_SONAR_ENGINE=rust
run audio-rust_sonar-python         KITE_AUDIO_ENGINE=rust   KITE_SONAR_ENGINE=python
run audio-python_sonar-rust         KITE_AUDIO_ENGINE=python KITE_SONAR_ENGINE=rust
run audio-python_sonar-python       KITE_AUDIO_ENGINE=python KITE_SONAR_ENGINE=python
run audio-rust_sonar-rust_thread-native   KITE_AUDIO_ENGINE=rust KITE_SONAR_ENGINE=rust KITE_SONAR_THREAD=native
run audio-python_sonar-rust_thread-native KITE_AUDIO_ENGINE=python KITE_SONAR_ENGINE=rust KITE_SONAR_THREAD=native
