# ETH3D GT + DPE Telemetry Design

## Why `scan.ply + scan_alignment.mlp` is enough for diagnostic GT

ETH3D stores each laser scan in its own local scanner coordinate system. `scan_alignment.mlp` is a MeshLab project containing one `MLMesh` entry per scan and a 4x4 `MLMatrix44` that maps that scan into the shared global coordinate system. The provider therefore reconstructs one aligned global laser cloud before projecting it into DPE cameras.

For the Studio this produces **diagnostic per-view GT** directly from the scan data:

```text
aligned laser cloud
      ↓ z-buffer projection
GT depth + valid mask
      ↓ local finite differences
GT normal
      ↓ depth/normal discontinuity
GT geometry edge
      ↓ depth+normal connected components
GT surface label
```

This is intentionally separate from the official ETH3D 3D evaluator. It is designed to answer DPE-internal questions such as “did PRE cross a true surface?”, not to redefine the benchmark metric.

## What runs on GPU

During the final finest-scale DPE pass, GT maps are uploaded once with the reference view. Telemetry is then written directly inside the existing CUDA modules:

```text
strong_propagation.cuh
    → ES selected candidate count
    → ES same-GT-surface count

anchor_search.cuh
    → PRE candidate count
    → PRE same-GT-surface count
    → selected anchor count
    → same-GT-surface anchor count

plane_construction.cuh
    → fitted-plane depth error
    → fitted-plane normal error
    → adaptive-radius surface violation

dpe.cu final telemetry kernel
    → final depth error
    → matching cost
    → final radius
```

The algorithm still owns algorithm state. Telemetry owns only diagnostics. The frontend never calls CUDA or DPESolver directly.

## Why capture only the final finest pass by default

A full ETH3D DSLR image can contain tens of millions of pixels. Keeping GT depth, normal, surface label, telemetry maps, and sparse anchor/plane data for every pyramid level and every refinement pass would create unnecessary GPU/CPU/disk pressure.

The default therefore captures one complete decision chain in the final finest-scale `REFINE_ITER`. This is enough to diagnose the final reconstruction state and keeps experiment overhead bounded. `--telemetry-all` applies this same final-pass capture to all ablation cases.

## Interpretation cautions

- A projected scan is sparse relative to an image. `--gt-splat-radius` controls a small z-buffer splat used to improve local support.
- Geometry edges are generated only where GT samples exist; invalid-neighbor transitions are not automatically called geometry edges because that would turn scan sparsity into false boundaries.
- Surface labels are derived diagnostic connected components based on projected GT depth and normal continuity. They are not native ETH3D semantic or plane labels.
- Plane-normal error is computed against the derived projected GT normal; pixels without a valid GT normal remain unscored for that quantity.
- Final official ETH3D Accuracy / Completeness / F1 should still come from `ETH3DMultiViewEvaluation`.
