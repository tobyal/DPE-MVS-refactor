#pragma once
#include "experiments/experiment_case.h"
#include "scene/scene.h"
#include "runtime/cuda_context.h"
#include "experiments/ground_truth/ground_truth_provider.h"
#include <boost/filesystem.hpp>
#include <vector>

namespace dpe {
class ExperimentRunner {
public:
    ExperimentRunner(Scene& scene, CudaContext& cuda,
                     boost::filesystem::path experiment_root,
                     GroundTruthProvider* ground_truth = nullptr);
    void Run(const std::vector<Problem>& base_problems,
             const std::vector<ExperimentCase>& cases);
private:
    Scene& scene_;
    CudaContext& cuda_;
    boost::filesystem::path root_;
    GroundTruthProvider* ground_truth_ = nullptr;
};
}
