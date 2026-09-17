#pragma once

#include "scene/scene.h"
#include "scene/reconstruction_state.h"
#include "runtime/cuda_context.h"
#include "experiments/experiment_case.h"
#include "experiments/ground_truth/ground_truth_provider.h"

#include <vector>

namespace dpe {

class DPEPipeline {
public:
    DPEPipeline(Scene& scene, ReconstructionState& reconstruction, CudaContext& cuda,
                AlgorithmPolicy policy = {}, GroundTruthProvider* ground_truth = nullptr);

    void Run(std::vector<Problem>& problems);
    int PyramidLevels() const { return pyramid_levels_; }

private:
    void PrepareGuidanceForLevel(std::vector<Problem>& problems, int level);
    void RunPass(std::vector<Problem>& problems, int level, int outer_refine,
                 RunState state, bool geom_consistency);
    void ProcessView(Problem& problem, bool capture_telemetry);
    void DumpDebug(const Problem& problem, const FrameState& state) const;

    Scene& scene_;
    ReconstructionState& reconstruction_;
    CudaContext& cuda_;
    int pyramid_levels_ = 0;
    AlgorithmPolicy policy_;
    GroundTruthProvider* ground_truth_ = nullptr;
};

}  // namespace dpe
