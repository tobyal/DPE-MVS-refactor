#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${DPE_CUDA_ARCH:-86}"

CUDA_COMPILER="${CUDACXX:-}"
if [[ -z "$CUDA_COMPILER" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_COMPILER="$(command -v nvcc)"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    elif [[ -x /usr/local/cuda-12.6/bin/nvcc ]]; then
        CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
    else
        echo "nvcc not found; set CUDACXX to the CUDA compiler path" >&2
        exit 1
    fi
fi

cmake -S "$ROOT" -B "$ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DDPE_CUDA_ARCH="$ARCH"
cmake --build "$ROOT/build" -j"$(nproc)"
