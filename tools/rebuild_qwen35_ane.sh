#!/usr/bin/env bash
# Rebuild ONLY the qwen35_prefill native extension and install it in place.
#
# The workspace path contains a space ("/Volumes/SSD 4TB/..."), which setup.py
# cannot quote correctly when it injects -DPython_EXECUTABLE into CMAKE_ARGS.
# The pre-configured CMake build directory already pins a space-free
# interpreter (/tmp/omlxvenv/bin/python), so drive cmake directly instead.
#
# Usage: tools/rebuild_qwen35_ane.sh [cmake-build-args...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/temp.macosx-11.0-arm64-cpython-313/omlx.custom_kernels.qwen35_prefill._ext"
STAGE_DIR="$REPO_ROOT/build/lib.macosx-11.0-arm64-cpython-313/omlx/custom_kernels/qwen35_prefill"
PKG_DIR="$REPO_ROOT/omlx/custom_kernels/qwen35_prefill"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "error: $BUILD_DIR is not configured. Run the full" >&2
    echo "       OMLX_WITH_CUSTOM_KERNEL=1 python setup.py build_ext --inplace first." >&2
    exit 1
fi

JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(sysctl -n hw.ncpu)}"
cmake --build "$BUILD_DIR" -j"$JOBS" "$@"

install -m 0644 "$STAGE_DIR/omlx_qwen35_prefill_kernels.metallib" \
    "$PKG_DIR/omlx_qwen35_prefill_kernels.metallib"
install -m 0644 "$STAGE_DIR/omlx_qwen35_prefill_kernels_nax.metallib" \
    "$PKG_DIR/omlx_qwen35_prefill_kernels_nax.metallib"
install -m 0755 "$STAGE_DIR/libomlx_qwen35_prefill_kernel_ops.dylib" \
    "$PKG_DIR/libomlx_qwen35_prefill_kernel_ops.dylib"
install -m 0755 "$STAGE_DIR/_ext.cpython-313-darwin.so" \
    "$PKG_DIR/_ext.cpython-313-darwin.so"

echo "qwen35_prefill extension rebuilt and installed into $PKG_DIR"
