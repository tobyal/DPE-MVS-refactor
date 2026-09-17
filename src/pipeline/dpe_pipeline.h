#pragma once

#include "scene/scene.h"
#include "scene/reconstruction_state.h"
#include "runtime/cuda_context.h"
#include "diagnostics/diagnostic_sink.h"

#include <vector>

namespace dpe {

class DPEPipeline {
public:
    DPEPipeline(Scene& scene, ReconstructionState& reconstruction, CudaContext& cuda,
                DiagnosticSink* diagnostics = nullptr);

    void Run(std::vector<Problem>& problems);
    int PyramidLevels() const { return pyramid_levels_; }

private:
    void PrepareGuidanceForLevel(std::vector<Problem>& problems, int level);
    void RunPass(std::vector<Problem>& problems, int level, int outer_refine,
                 RunState state, bool geom_consistency);
    void ProcessView(Problem& problem);
    void DumpDebug(const Problem& problem, const FrameState& state) const;

    Scene& scene_;
    ReconstructionState& reconstruction_;
    CudaContext& cuda_;
    DiagnosticSink* diagnostics_ = nullptr;
    int pyramid_levels_ = 0;
};

}  // namespace dpe
