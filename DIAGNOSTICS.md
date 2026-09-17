# Reconstruction diagnostics

The diagnostics subsystem follows the reconstruction result backwards from fusion to the PatchMatch stages. It is deliberately optional: a normal run passes no sink to the pipeline, launches the original kernel sequence, and performs no stage downloads.

## Capture levels

The default `--diagnostics` policy balances causal coverage with ETH3D data volume:

| Data | Coverage |
|---|---|
| Guidance | every scale and reference view |
| Pyramid final state | final pass of every scale and reference view |
| Strong / Plane / Weak trace | final scale and final pass for one reference view |
| Fusion fate and support | every final reference pixel |
| Diagnostic point clouds | every accepted point |

The traced reference defaults to the first entry in `pair.txt`. Use `--diagnostic-view=<id>` to select a view. `--diagnostic-view=all` enables full-view stage traces and can produce very large output.

## Solver stage sequence

Each captured solver invocation has this ordered sequence:

```text
00 input
01 strong iter 0
02 plane iter 0
03 weak iter 0
04 strong iter 1
05 plane iter 1
06 weak iter 1
07 strong iter 2
08 plane iter 2
09 weak iter 2
10 finalized
11 filtered
12 classified
13 refined
```

Each stage directory can contain:

| Artifact | Meaning |
|---|---|
| `depth.dmb` | depth implied by the current or fitted plane |
| `normal.dmb` | world-space normal |
| `reliability.dmb` | STRONG / WEAK / UNKNOWN state |
| `matching_cost.dmb` | current aggregated matching cost |
| `selected_views.dmb` | selected-source bit mask |
| `adaptive_radius.dmb` | center-patch radius |
| `anchor_count.dmb` | valid reliable anchors for each weak pixel |
| `texture_complexity.dmb` | guidance complexity at solver input |
| `update_mask.dmb` | pixels whose depth changed from the previous stage |
| `stage_summary.csv` | compact counts and means for the full stage sequence |

The pass-level `update_events.csv` is the sparse event table. It contains one row for each
changed pixel with its stage, pixel coordinate, before/after depth and matching cost,
reliability state, selected-view mask, adaptive radius, and anchor count. This is the
bridge between dense-map inspection and single-pixel causal tracing.

The Plane stage uses `fitted_planes`, not the current hypothesis. Comparing Plane depth with the following Weak depth therefore answers whether a good fitted plane was retained or rejected by deformable PatchMatch.

## Pyramid evolution

`pass_summary.csv` and each `final_state/` directory retain one result per pyramid level. This is enough to compare the input of a finer level with the output of the preceding coarser level without retaining all outer passes.

## Fusion fate

Every final reference pixel receives one fate code:

| Code | Meaning |
|---:|---|
| 0 | accepted into the point cloud |
| 1 | invalid reference depth |
| 2 | no valid source depth support |
| 3 | reprojection threshold failed |
| 4 | relative-depth threshold failed |
| 5 | normal-angle threshold failed |
| 6 | support exists but dynamic score failed |
| 7 | pixel already consumed by another accepted point |

Per-view maps are written below `fusion/ref_XXXXXXXX/`; aggregate counts are appended to `fusion/fusion_summary.csv`.

The final cloud is also emitted with three diagnostic colorings:

- `point_support.ply`: source support count;
- `point_reliability.ply`: STRONG / WEAK origin;
- `point_reference.ply`: originating reference view.

## Ground-truth extension boundary

Ground-truth rendering and metrics are intentionally not part of `FrameState`, `DPEGpuContext`, or the normal fusion API. A later GT provider can consume the artifacts above to derive depth-error maps, Fix/Break transitions, Correct Depth Survival Rate, and Wrong Depth Survival Rate without changing the reconstruction algorithm or its resource lifetime.
