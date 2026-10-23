#!/bin/bash
# Convenience wrapper: the cxx/Makefile now builds render_bench/render_matrix
# (including the vendored rasterizer sources). Kept for habit continuity.
set -e
cd "$(dirname "$0")/.."
ARCH=${1:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')}
make -C cxx SM=sm_${ARCH} render_bench render_matrix
