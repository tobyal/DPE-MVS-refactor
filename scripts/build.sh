#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${DPE_CUDA_ARCH:-86}"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release -DDPE_CUDA_ARCH="$ARCH"
cmake --build "$ROOT/build" -j"$(nproc)"
