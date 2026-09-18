# DPE-MVS Refactored

This package reorganizes DPE-MVS around the algorithmic flow of the AAAI'25 paper while keeping data lifetime and CUDA execution explicit and local to the layers that own them.

Source basis:

- Official DPE-MVS main branch inspected at commit `129e6530cd80ad4ab7697b1b7a8c9a305986c1a7`.
- The existing `DPE-MVS-fast` implementation was inspected for the useful runtime ideas already validated there: persistent CPU scene data, reconstruction state kept in memory, reusable GPU allocations, and avoiding unconditional device-wide synchronization.
- The Perception Range Expansion sample-count clamp follows paper Eq. (4): `max(1, min(2*eta-1, raw_step))`.

## Code organization

```text
src/
├── main.cpp
├── common/                 basic types, camera/bin I/O, CUDA checks
├── scene/                  input scene + in-memory reconstruction state
├── preprocessing/          fine edge / coarse region construction
├── runtime/                CUDA stream, pooled buffers/arrays, GPU scene textures
├── dpe/
│   ├── dpe.cpp             host-side solver state preparation / transfer
│   ├── dpe.cu              high-level GPU PatchMatch schedule
│   └── modules/
│       ├── initialization.cuh
│       ├── edge_guidance.cuh
│       ├── matching_cost.cuh
│       ├── view_selection.cuh
│       ├── strong_propagation.cuh
│       ├── anchor_search.cuh
│       ├── plane_construction.cuh
│       ├── weak_propagation.cuh
│       ├── consistency.cuh
│       ├── reliability.cuh
│       └── refinement.cuh
├── pipeline/               pyramid / init / refinement scheduling
└── fusion/                 final multi-view fusion
```

The intended reading order is:

```text
main
  -> DPEPipeline
      -> DPESolver::Run
          -> RunDpeKernels
              -> strong path
              -> plane construction
              -> weak path
              -> reliability/refinement
      -> Fusion
```

## Runtime model

`Scene` owns CPU input data and an active-pyramid-level image/guidance cache. `ReconstructionState` owns depth/normal/reliability/view-selection state between pyramid passes. `CudaContext` owns one CUDA stream and a reusable `GpuWorkspace`. `GpuScene` owns the GPU textures/cameras for one solver invocation. `DPESolver` owns only the per-view PatchMatch working state.

This removes file I/O as an internal communication mechanism. Intermediate files are written only when `--debug` is requested.

## Build

Requirements are the same family as the official project: CUDA, OpenCV, Boost filesystem/system, and CMake.

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j$(nproc)
```

Set the CUDA architecture explicitly when needed, for example:

```bash
cmake .. -DDPE_CUDA_ARCH=89
```

## Run

```bash
./build/DPE /path/to/dense_folder 0
```

Optional intermediate output:

```bash
./build/DPE /path/to/dense_folder 0 --debug
```

Choose a separate output directory with:

```bash
./build/DPE /path/to/dense_folder 0 --output=/path/to/output
```

By default the final point cloud is written to `<dense_folder>/DPE/DPE.ply`; with `--output`, it is written to `<output>/DPE.ply`.

## Important validation step

This is a structural refactor/reimplementation, so before using it for paper experiments, compare it against your current baseline on a small scene first. Recommended checks are final depth statistics and ETH3D 2 cm / 10 cm accuracy, completeness, and F1. See `STATIC_CHECKS.md` for the completed static audit and the explicit behavior notes.

## ETH3D evaluation projection

`eth3d_projection` is an independent visualization tool for the colored PLY files produced by
the official `ETH3DMultiViewEvaluation` executable. It does not call or modify DPE, PatchMatch,
or fusion, and it does not recompute Accuracy or Completeness labels.

```bash
./build/eth3d_projection \
  --dense-folder /path/to/dense_folder \
  --output /path/to/projection_output \
  --accuracy-2cm /path/to/accuracy.tolerance_0.02.ply \
  --accuracy-10cm /path/to/accuracy.tolerance_0.1.ply \
  --completeness-2cm /path/to/completeness.tolerance_0.02.ply \
  --completeness-10cm /path/to/completeness.tolerance_0.1.ply
```

The tool reads reference image IDs from `pair.txt`, reuses the project's camera parser, applies
`X_cam = R * X_world + t`, projects with the complete 3x3 intrinsic matrix, and resolves visibility
with an independent per-view Z-buffer. The default circular splat radius is 2 pixels and the
default overlay alpha is 0.7. Use `--radius`, `--alpha`, and `--threads` to override them.

Only the requested PNG directories are created:

```text
accuracy_2cm/<id>.png
accuracy_10cm/<id>.png
completeness_2cm/<id>.png
completeness_10cm/<id>.png
```
