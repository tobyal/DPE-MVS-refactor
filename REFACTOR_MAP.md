# Refactor map

| Original responsibility | Refactored location |
|---|---|
| `main.cpp` pair parsing + pyramid scheduling | `scene/scene.cpp` + `pipeline/dpe_pipeline.cpp` |
| repeated `imread` / camera read | `Scene` |
| `StateStore`-style intermediate state | `ReconstructionState` |
| repeated `cudaMalloc/cudaFree` | `GpuWorkspace` |
| texture/camera upload | `GpuScene` |
| `DataPassHelper` | `DPEGpuContext` split into scene/state/guidance |
| `RunPatchMatch()` sequence | `dpe/dpe.cu::RunDpeKernels()` |
| `ComputeBilateralNCCOld` | `StandardPatchCost` |
| `ComputeBilateralNCCNew` | `DeformablePatchCost` |
| reliable-pixel propagation / ES | `strong_propagation.cuh` |
| `GenEdgeInform` | `edge_guidance.cuh` |
| weak anchor search + PE | `anchor_search.cuh` |
| first RANSAC (candidate -> anchors) | `anchor_search.cuh` |
| second RANSAC / AA | `plane_construction.cuh` |
| weak deformable PM | `weak_propagation.cuh` |
| geometric consistency | `consistency.cuh` |
| `DepthToWeak` | `reliability.cuh` |
| strong filtering + local refine | `refinement.cuh` |
| final point cloud fusion | `fusion/fusion.cpp` |

## DPE GPU flow

```text
Init RNG
  -> initialize/restore plane hypotheses
  -> build edge/region guidance
  -> nearest strong points
  -> weak anchor generation
  -> repeat PM iterations:
       strong propagation
       weak-plane construction (RANSAC + adaptive radius)
       weak/deformable propagation
  -> depth/normal finalization
  -> strong depth filtering
  -> reliability classification
  -> local depth refinement
```

## Eq. (4) correction

The official source currently contains a clamp in the coarse-region extension that reduces the directional sample count to one. In this package the implementation is located in:

```text
src/dpe/modules/anchor_search.cuh
```

and uses:

```cpp
s = max(1, min(2 * rotate_time - 1, s));
```

which matches the range constraint described around Eq. (4).
