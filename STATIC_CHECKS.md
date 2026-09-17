# Static validation report

## Completed

- Verified all quoted local C++/CUDA includes resolve under `src/`.
- Checked balanced `()`, `[]`, and `{}` after lexically removing C/C++ comments, string literals, and character literals.
- Python backend passes `python -m py_compile`.
- Backend DMB random-access reader smoke-tested with synthetic CV_32F/CV_8U matrices.
- Sparse `anchors.bin` and `planes.bin` readers smoke-tested with synthetic records.
- FastAPI application construction smoke-tested; routes are registered successfully.
- Experiment output paths remain case-isolated.
- GT telemetry is disabled for ordinary `DPE` and only attached through `DPEExperiment`.
- Baseline-only telemetry default prevents all ablation cases from paying GT telemetry cost unless `--telemetry-all` is requested.

## Environment limitations

- `nvcc` is not installed in this execution environment, so CUDA compilation was not executed here.
- OpenCV C++ development headers/pkg-config are not installed here, so host C++ compilation was not executed here.
- Frontend dependency installation was not performed because package download/network access is unavailable in this runtime.

## Required first server-side validation

Run on the CUDA server:

```bash
cmake -S . -B build -DDPE_CUDA_ARCH=86
cmake --build build -j
```

Then start with one small ETH3D scene and the baseline telemetry case. Inspect:

1. aligned GT point cloud overlaps the DPE cameras and reconstruction;
2. GT depth is non-zero on expected scan-visible pixels;
3. image-edge / GT-edge relation map is geometrically plausible;
4. anchor markers and fitted planes appear at the selected weak pixel;
5. telemetry summary has non-zero GT-valid pixels and finite plane/depth statistics.

Only after this geometry sanity check should the full experiment suite be used for quantitative conclusions.
