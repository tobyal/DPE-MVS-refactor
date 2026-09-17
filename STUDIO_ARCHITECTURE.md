# DPE Studio architecture

```text
                        CUDA server
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  dpe_core                                                    │
│  ┌─────────┐  ┌──────────┐  ┌──────────────┐                │
│  │ Scene   │→ │ Pipeline │→ │ DPESolver GPU│                │
│  └─────────┘  └──────────┘  └──────┬───────┘                │
│                                     │                        │
│  ETH3D scan.ply + scan_alignment.mlp│                        │
│         ↓                           │                        │
│  GroundTruthProvider                │                        │
│         ↓ GT depth/normal/surface   │                        │
│         └──────────────→ GPU Telemetry                       │
│                                     ↓                        │
│                          ReconstructionState                  │
│                                     ↓                        │
│                             ArtifactWriter                   │
│                                     ↓                        │
│       manifest / DMB / PNG / sparse anchors / planes / PLY  │
│                                     ↓                        │
│                              FastAPI server                   │
└─────────────────────────────────────┬────────────────────────┘
                                      │  SSH -L
                                      ↓
                          Local browser / Three.js
```

## Dependency rule

- `dpe_core` never depends on FastAPI, React, or web concepts.
- Studio frontend never calls CUDA or C++ objects directly.
- GT is optional and enters only through the experiment path.
- Telemetry reads algorithm state but does not own or alter the DPE decision logic.
- Large arrays are persisted as DMB/binary artifacts; JSON is limited to manifests, summaries, and small per-pixel API responses.

## Visualization levels

```text
Experiment
  ↓
Scene          DPE cloud + ETH3D aligned scan + cameras
  ↓
View           RGB / depth / normal / edge / surface / telemetry heatmaps
  ↓
Pixel          DPE point + GT point + anchors + fitted plane + patch radius
```

The default GT-backed capture point is the final finest-scale refinement pass. This keeps the telemetry useful while avoiding the memory and disk multiplier of recording every pyramid pass.
