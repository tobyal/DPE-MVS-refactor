# DPE reliability evolution study

This mode is instrumentation only. It does not change reliability
classification, PatchMatch, propagation, RANSAC, plane construction, local
refinement, pyramid scheduling, or fusion. Without `--reliability-study`, no
study writer is created and no additional device copy or synchronization is
performed.

## Capture boundaries

Every pyramid pass captures three states:

```text
FinalizeDepthNormal
StrongFilterBlack
StrongFilterRed
    -> depth_pre / normal_pre
ClassifyReliability
    -> reliability
LocalRefine
    -> depth_post / normal_post
```

`normal_pre` and `normal_post` are both retained even though the current local
refinement kernel only changes depth. Normals are exported in world
coordinates.

The pipeline assigns one global stage per pass. With three pyramid levels this
produces `S00` through `S11`; other pyramid depths produce the corresponding
dynamic number of stages. A directory name also contains the zero-based level
and pass name, for example:

```text
S00_L00_coarse_init
S01_L00_refine1
S04_L01_dpe_init
```

## Run

```bash
./build/DPE /path/to/dense_folder 0 \
  --output=/path/to/output \
  --reliability-study
```

Results are written below:

```text
<output>/reliability_study/
  ref_<8-digit-id>/
    Sxx_Lxx_<pass>/
      depth_pre.dmb
      depth_pre.png
      normal_pre.dmb
      normal_pre.png
      reliability.bin
      reliability.png
      depth_post.dmb
      depth_post.png
      normal_post.dmb
      normal_post.png
      stage.json
```

`reliability.bin` uses the same versioned OpenCV-matrix binary header as other
DPE `.dmb`/`.bin` files. Reliability values are `WEAK=0`, `STRONG=1`, and
`UNKNOWN=2`. `outer_refine` in `stage.json` is `-1` for init passes and `1..3`
for refinement passes.

## Offline GT analysis

The analysis script expects per-view GT depth, optional GT normal, and optional
valid mask already aligned with each reference camera. It does not project or
reclassify ETH3D point clouds.

Default file discovery supports layouts such as:

```text
<gt_root>/ref_<id>/gt_depth.dmb
<gt_root>/ref_<id>/gt_normal.dmb
<gt_root>/ref_<id>/valid_mask.dmb
<gt_root>/views/<id>/gt_depth.dmb
```

Custom layouts can use `--gt-depth-pattern`, `--gt-normal-pattern`, and
`--gt-mask-pattern`; patterns may contain `{gt_root}`, `{ref_id}`, and
`{ref_id8}`.

Official ETH3D GT consists of local `scan*.ply` files plus the global scan poses
in `scan_alignment.mlp`. Convert it once into camera-aligned depth maps with:

```bash
python3 scripts/prepare_eth3d_gt.py \
  --dense-folder /path/to/dense_folder \
  --ground-truth-mlp /path/to/dslr_scan_eval/scan_alignment.mlp \
  --output /path/to/aligned_gt
```

The converter applies each MLP `global_T_mesh`, projects through the DPE camera
definition (`X_cam = R * X_world + t` and the complete 3x3 `K`), and resolves
visibility with a per-camera Z-buffer. It writes only:

```text
<aligned_gt>/ref_<8-digit-id>/gt_depth.dmb
```

The default `--splat-radius 0` preserves the official scan sampling. A positive
radius can increase valid-pixel coverage, but it also spreads laser samples
across object boundaries and should be recorded with the experiment. Use
`--view <id>` for a one-view alignment check before converting the full scene.

```bash
python3 scripts/analyze_reliability.py \
  --study-root /path/to/output/reliability_study \
  --gt-root /path/to/aligned_gt \
  --depth-thresholds 0.02 0.10 \
  --normal-thresholds 5 10 20 \
  --depth-correct-threshold 0.02 \
  --normal-correct-threshold 10 \
  --depth-error-vmax 0.10 \
  --normal-error-vmax 30
```

If GT normals are unavailable, derive them from aligned GT depth and the DPE
camera files:

```bash
python3 scripts/analyze_reliability.py \
  --study-root /path/to/output/reliability_study \
  --gt-root /path/to/aligned_gt \
  --derive-gt-normals \
  --camera-root /path/to/dense_folder
```

Derived normals are oriented toward the reference camera, matching DPE's plane
normal convention, and are then transformed to world coordinates. The default
angular metric uses signed `dot(n_pred, n_gt)`. Use
`--normal-sign-mode unsigned` only when the supplied GT convention is genuinely
sign-ambiguous; the selected convention is recorded in `analysis_config.json`.
Use `--gt-normal-coordinates camera --camera-root ...` when supplied normals
are in camera coordinates.

To compare the start and end of each pyramid scale for one reference view, join
the same analysis map from `S00`, `S03`, `S04`, `S07`, `S08`, and `S11`:

```bash
python3 scripts/stitch_reliability_stages.py \
  --analysis-root /path/to/reliability_study/analysis \
  --view 0 \
  --image-name reliability_vs_joint_gt.png
```

The default output is
`<analysis-root>/comparisons/ref_<id>/<image>_S00_S03_S04_S07_S08_S11.png`. Both
`--view 0` and `--view ref_00000000` are accepted. The command reports any
stage image that has not been generated yet.

Use `--all-views` instead of `--view <id>` to generate one comparison for every
reference view. Lower-resolution stage images are enlarged to the highest stage
height with nearest-neighbor sampling before composition.

Raw snapshot images such as `reliability.png` live directly below the study
root instead of `analysis/maps`. Select them with
`--source-root /path/to/reliability_study`; comparison output is still written
below the supplied analysis root.

The script writes:

```text
analysis/
  analysis_config.json
  metrics/per_stage.csv
  metrics/per_view.csv
  metrics/summary.csv
  plots/depth_evolution.png
  plots/normal_evolution.png
  plots/reliability_evolution.png
  plots/geometry_vs_reliability.png
  plots/strong_vs_weak_depth_error.png
  plots/strong_vs_weak_normal_error.png
  plots/strong_vs_weak_depth_correctness.png
  plots/strong_vs_weak_normal_correctness.png
  plots/strong_vs_weak_joint_correctness.png
  maps/ref_.../S.../
```

Reliability correctness always uses pre-classification geometry. UNKNOWN is
reported separately and never merged into STRONG or WEAK. Disable per-stage
error maps with `--skip-error-maps` when only CSVs and trend plots are needed.

Each stage analysis directory includes reliability-conditioned continuous maps
(`strong_depth_error.png`, `weak_depth_error.png`,
`strong_normal_error.png`, and `weak_normal_error.png`), binary correctness
maps for STRONG/WEAK depth and normal, and fixed-color four-class maps:

```text
reliability_vs_depth_gt.png
reliability_vs_normal_gt.png
reliability_vs_joint_gt.png
```

Continuous maps use the configured fixed maxima and include colorbars in meters
or degrees, so colors remain comparable across all stages and reference views.
Four-class maps distinguish STRONG+Correct, STRONG+Wrong, WEAK+Correct, and
WEAK+Wrong; UNKNOWN and invalid GT share a neutral ignored color.

The correctness options also accept the earlier aliases
`--reliability-depth-threshold` and `--reliability-normal-threshold`.
