#!/usr/bin/env bash
# Standalone Metal IO probe — no omlx build required.
#
# Builds the two C++ probes against the metal-cpp headers shipped inside the
# mlx wheel and runs the cold/warm A/B against a model directory.
#
#   ./run.sh "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-JANG_4M"
#
# Args after the model dir are forwarded to metal_io_probe:
#   <N_slices> <gap_slices> <reps> <start_file_index>
set -euo pipefail

MODEL_DIR="${1:?usage: run.sh <model-dir> [N] [gap_slices] [reps] [start_file]}"
shift
MODEL_DIR="${MODEL_DIR%/}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

METAL_CPP="$("$PY" -c 'import mlx.core, pathlib; print(pathlib.Path(mlx.core.__file__).parent / "include" / "metal_cpp")')"

OUT_DIR="$HERE/../results/metal_io_probe"
mkdir -p "$OUT_DIR"

echo "metal_cpp: $METAL_CPP"
clang++ -std=c++17 -O2 -fno-objc-arc \
  -I"$METAL_CPP" \
  -framework Foundation -framework Metal -framework QuartzCore \
  -o "$OUT_DIR/metal_io_probe" "$HERE/metal_io_probe.cpp"
clang++ -std=c++17 -O2 -fno-objc-arc \
  -I"$METAL_CPP" \
  -framework Foundation -framework Metal -framework QuartzCore \
  -o "$OUT_DIR/feasibility_probe" "$HERE/feasibility_probe.cpp"

# Feasibility: byte-exactness vs pread on one shard.
SHARD="$(ls "$MODEL_DIR"/model-*-of-*.safetensors 2>/dev/null | head -1)"
if [ -n "$SHARD" ]; then
  echo "--- feasibility (byte-exactness vs pread) ---"
  "$OUT_DIR/feasibility_probe" "$SHARD" 1048576 4194304 2>&1 | tee "$OUT_DIR/feasibility.log"
fi

echo "--- cold/warm A/B ---"
"$OUT_DIR/metal_io_probe" "$MODEL_DIR" "$@" 2>&1 | tee "$OUT_DIR/metal_io_ab.log"
