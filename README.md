# DPE-MVS Studio + ETH3D GPU Telemetry

This package extends the refactored DPE-MVS codebase with a decoupled experiment/telemetry layer and a local-browser 2D/3D analysis tool.

The architecture remains split into three independent parts:

```text
C++/CUDA algorithm core
        ↓
Experiment + GT telemetry artifacts
        ↓
FastAPI artifact server
        ↓
React / Three.js local browser
```

The normal `DPE` executable does not require GT and remains independent from the Studio.

## Source basis

- DPE-MVS algorithm structure follows the refactored `tobyal/DPE-MVS-refactor` organization.
- ETH3D GT input is the laser-scan PLY plus `scan_alignment.mlp`.
- `scan_alignment.mlp` is interpreted as a MeshLab project: each `MLMesh` contains a 4x4 `MLMatrix44` that maps scan-local coordinates into the global ETH3D coordinate system.

## Layout

```text
src/
├── common/
├── scene/
├── preprocessing/
├── runtime/
├── dpe/
│   ├── dpe.cpp
│   ├── dpe.cu
│   └── modules/
├── pipeline/
├── fusion/
├── experiments/
│   ├── experiment_case.h
│   ├── experiment_runner.*
│   ├── experiment_suite.*
│   ├── main.cpp
│   ├── ground_truth/
│   │   └── ground_truth_provider.*
│   └── telemetry/
│       ├── telemetry.h
│       └── gpu_telemetry.cuh
└── studio/
    └── artifact_writer.*

studio/
├── backend/
│   └── server.py
└── frontend/
    └── src/
        ├── main.jsx
        └── style.css
```

## ETH3D GT provider

`GroundTruthProvider`:

1. reads ASCII or binary-little-endian PLY vertices;
2. parses every `MLMesh` / `MLMatrix44` in `scan_alignment.mlp`;
3. transforms every available scan into ETH3D global coordinates;
4. projects the aligned scan to the current reference camera with a z-buffer;
5. optionally splats each laser point by a small pixel radius;
6. derives per-view GT depth, valid mask, approximate GT normal, geometry edge, and connected surface labels.

Important: these per-view maps are **diagnostic projections of the aligned laser scans**, not a replacement for the official ETH3D multi-view evaluator. Official final 3D Accuracy / Completeness / F1 should still be computed with `ETH3DMultiViewEvaluation` and the original `scan_alignment.mlp`.

## GPU telemetry

GT-backed telemetry is captured during the final finest-scale `REFINE_ITER` pass by default. This keeps the overhead bounded while still exposing the complete final DPE decision chain.

Current GPU telemetry includes:

```text
DPE fine edge / coarse region / texture complexity
Strong-path ES candidate count + same-GT-surface ratio
PRE / anchor candidate count + same-GT-surface ratio
Selected anchor count + same-GT-surface ratio
Fitted plane center-depth error
Fitted plane normal error
Adaptive patch radius
GT-surface violation ratio inside the patch
Final DPE depth error
Final matching cost
```

Sparse per-pixel anchor coordinates and fitted plane parameters are exported separately for interactive debugging.

By default GT telemetry is enabled only for `baseline_current`. Use `--telemetry-all` if you explicitly want full GT telemetry for every ablation case; this is much heavier in GPU memory, runtime, and disk usage.

## Build

```bash
cmake -S . -B build -DDPE_CUDA_ARCH=86
cmake --build build -j
```

## Normal DPE reconstruction

```bash
./build/DPE /path/to/dense_folder 0 --output=/path/to/output
```

## Run experiments with ETH3D GT telemetry

Single scan / explicit PLY:

```bash
./build/DPEExperiment \
    /path/to/dense \
    0 \
    --output=/path/to/experiments/pipes \
    --gt-scan=/path/to/dslr_scan_eval/scan.ply \
    --gt-mlp=/path/to/dslr_scan_eval/scan_alignment.mlp \
    --gt-splat-radius=1
```

For multi-scan ETH3D scenes, `scan_alignment.mlp` may reference several PLY files. The provider loads all referenced files that exist beside the MLP and applies each 4x4 alignment matrix. `--gt-scan` is also kept as a fallback for a single explicitly supplied scan.

If you intentionally want telemetry for every case:

```bash
./build/DPEExperiment ... --telemetry-all
```

## Output artifacts

For `baseline_current/views/00000017/`, the Studio can export:

```text
depth / normal / state
fine_edge / coarse_region / texture_complexity
gt_depth / gt_normal / gt_geometry_edge / gt_surface_label
edge_relation
es_candidate_count / es_same_ratio
candidate_count / candidate_same_ratio
anchor_same_ratio
plane_depth_error / plane_normal_error
radius_violation
final_depth_error
matching_cost
adaptive_radius
anchors.bin
planes.bin
```

`edge_relation.png` encodes the important distinction between image edges and GT geometry edges, so texture-only edges and missed geometry boundaries can be inspected directly.

A decimated aligned GT scan preview is written once to:

```text
<experiment_root>/_ground_truth/aligned_scan_preview.ply
```

## Studio server

Build the frontend and install the lightweight backend:

```bash
./scripts/build_studio.sh
python3 -m pip install -r studio/backend/requirements.txt
```

Start on the CUDA server:

```bash
./scripts/run_studio.sh \
    /path/to/experiments/pipes \
    /path/to/dense \
    8765
```

On the local machine:

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

Open:

```text
http://127.0.0.1:8765
```

## Studio interaction

The browser provides:

- DPE point cloud + ETH3D aligned GT scan in the same 3D scene;
- camera frusta and camera selection;
- RGB / depth / normal / DPE edge / GT edge / surface / telemetry heat-map layers;
- image-edge vs geometry-edge relation view;
- 2D pixel selection linked to the corresponding 3D ray and reconstructed point;
- GT point at the same pixel;
- selected anchors in both 2D and 3D;
- same-surface anchors vs cross-surface anchors;
- fitted plane patch in 3D;
- adaptive patch rectangle in the 2D image;
- per-pixel PRE / anchor / plane / radius / final-depth telemetry;
- per-view 2 cm / 10 cm diagnostic depth-hit ratios and intermediate statistics;
- WebSocket experiment status.

The UI is offline-first: heavy arrays and point clouds are loaded only on demand. WebSocket traffic is limited to light run status.

## Validation status

This environment does not contain the CUDA/OpenCV C++ development stack required for a real `nvcc` build, so the package does **not** claim CUDA compilation success here. Python backend syntax, API binary readers, local include resolution, delimiter balance, and artifact layout are statically checked. See `STATIC_CHECKS.md`.
