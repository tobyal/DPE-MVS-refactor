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

This is a structural refactor/reimplementation, so before using it for paper experiments, compare it against your current baseline on a small scene first. Recommended checks are final depth statistics and ETH3D 2 cm / 10 cm accuracy, completeness, and F1. The execution environment used to prepare this package does not contain a CUDA/OpenCV C++ development toolchain, so an actual `nvcc` build could not be run here. See `STATIC_CHECKS.md` for the completed static audit and the explicit behavior notes.
