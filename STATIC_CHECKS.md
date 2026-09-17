# Static audit status

This package was reviewed against the official DPE-MVS implementation and the existing `DPE-MVS-fast` runtime changes before packaging.

## Checks completed

- Source delimiter balance (`{}`, `()`, `[]`) across all `.cpp/.h/.cu/.cuh` files.
- All project-local `#include "..."` paths resolve.
- Every source listed in `CMakeLists.txt` exists.
- `scripts/build.sh` passes `bash -n`.
- CUDA allocation/free calls are confined to `runtime/gpu_workspace.cu`; texture lifetime is confined to `runtime/gpu_scene.cu`.
- DPE algorithm/runtime layers contain no disk I/O calls.
- Normal execution does not call `cudaDeviceSynchronize()` after every kernel; kernel ordering uses one CUDA stream, with stream synchronization only at host result/lifetime boundaries. `DPE_DEBUG_SYNC` can restore per-kernel synchronization for debugging.
- Active pyramid images/guidance are bounded to one scale at a time; raw scene images/cameras and the compact coarsest fine-edge maps persist across scales.
- Rechecked the official flow for initial photometric view selection, strong-path photometric evaluation, weak-path geometric consistency, the two RANSAC stages, adaptive patch radius, reliability classification, and final fusion.
- Rechecked the high-resolution edge-crossing behavior: the coarsest fine-edge map is used as the low-resolution Bresenham map.
- Perception Range Expansion uses the paper-consistent Eq. (4) clamp.
- Diagnostics are opt-in through `DiagnosticSink`; normal runs perform no stage downloads or diagnostic I/O.
- The default diagnostics policy limits full stage tracing to one view at the final scale/final pass while retaining pyramid summaries and Fusion Fate for all views.

## Important behavior notes

The goal is algorithmic fidelity with a cleaner architecture, not bitwise identity with every incidental behavior of the released source. A few defensive choices are explicit:

- Eq. (4) uses `max(1, min(2*eta-1, raw))` instead of the released source's reversed clamp.
- Cost arrays are explicitly filled with their intended sentinel values rather than relying on partial aggregate initialization.
- Selected-view bit removal clears only the requested bit.
- Joint-view selection has a uniform fallback if its probability mass degenerates to zero, preventing a NaN CDF.
- Checkerboard launch coverage includes the final row for odd image heights.

These cases should be kept in mind when comparing exact numerical output with the released executable.

## Build validation

The target machine configured and compiled the complete project successfully with CUDA 12.6,
OpenCV, Boost, CMake Release mode, and `DPE_CUDA_ARCH=86`.

A full ETH3D reconstruction was not launched as part of this code build. Before large-scale
experiments, run one small scene and compare final depth statistics plus ETH3D 2 cm / 10 cm
accuracy, completeness, and F1 against the current baseline.
