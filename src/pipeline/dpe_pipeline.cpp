#include "pipeline/dpe_pipeline.h"
#include "dpe/dpe.h"
#include "common/io.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <stdexcept>

namespace dpe {

DPEPipeline::DPEPipeline(Scene& scene, ReconstructionState& reconstruction, CudaContext& cuda,
                         AlgorithmPolicy policy, GroundTruthProvider* ground_truth)
    : scene_(scene), reconstruction_(reconstruction), cuda_(cuda), policy_(policy), ground_truth_(ground_truth) {}

void DPEPipeline::PrepareGuidanceForLevel(std::vector<Problem>& problems, int level) {
    const int scale = 1 << (pyramid_levels_ - 1 - level);
    scene_.SetActiveScale(scale);
    for (auto& problem : problems) {
        DPEParams guide_params = problem.params;
        guide_params.use_edge = true;
        guide_params.use_limit = true;
        guide_params.use_label = true;
        scene_.Guidance(problem.ref_image_id, scale, guide_params);
    }
}

void DPEPipeline::ProcessView(Problem& problem, bool capture_telemetry) {
    const auto begin = std::chrono::steady_clock::now();
    SceneView view = scene_.MakeView(problem);
    // Keep the effective depth interval in the Problem as well as in the solver
    // so previews/checkpoints use the same range as PatchMatch.
    problem.params.depth_min = view.cameras.front().depth_min * 0.6f;
    problem.params.depth_max = view.cameras.front().depth_max * 1.2f;
    const EdgeGuidanceHost& guidance = scene_.Guidance(problem.ref_image_id, problem.scale_size, problem.params);
    const GroundTruthFrame* gt = nullptr;
    if (ground_truth_ && capture_telemetry) gt = &ground_truth_->Frame(problem.ref_image_id, view.cameras.front());
    DPESolver solver(problem, view, guidance, reconstruction_, cuda_, gt);
    FrameState result = solver.Run();
    reconstruction_.Put(problem.ref_image_id, result);
    if (problem.show_medium_result) DumpDebug(problem, result);
    const auto end = std::chrono::steady_clock::now();
    std::cout << "view " << FormatIndex(problem.ref_image_id)
              << " scale=" << problem.scale_size
              << " iter=" << problem.iteration
              << " time=" << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
              << " ms\n";
}

void DPEPipeline::RunPass(std::vector<Problem>& problems, int level, int outer_refine,
                          RunState state, bool geom_consistency) {
    const int scale = 1 << (pyramid_levels_ - 1 - level);
    for (auto& problem : problems) {
        problem.scale_size = scale;
        problem.params.scale_size = scale;
        problem.params.state = state;
        problem.params.geom_consistency = geom_consistency;
        problem.params.max_iterations = 3;
        if (state == RunState::FirstInit) {
            problem.params.use_apd = false;
            problem.params.use_edge = false;
            problem.params.weak_peak_radius = 6;
        } else if (state == RunState::RefineInit) {
            problem.params.use_apd = true;
            problem.params.use_edge = true;
            problem.params.ransac_threshold = 0.01f - level * 0.00125f;
            problem.params.rotate_time = std::min(1 << level, 4);
            problem.params.weak_peak_radius = 6;
        } else {
            problem.params.use_apd = level != 0;
            problem.params.use_edge = level != 0;
            problem.params.ransac_threshold = 0.01f - level * 0.00125f;
            problem.params.rotate_time = std::min(1 << level, 4);
            problem.params.weak_peak_radius = std::max(4 - 2 * outer_refine, 2);
        }
        // Experiment policy is applied after the paper-faithful schedule so
        // ablations cannot be accidentally overwritten by level logic.
        policy_.Apply(problem.params);
        const bool capture = ground_truth_ && level == pyramid_levels_ - 1 &&
                             state == RunState::RefineIter && outer_refine == 2;
        ProcessView(problem, capture);
    }
}

void DPEPipeline::Run(std::vector<Problem>& problems) {
    if (!scene_.CheckImageSizes(problems)) throw std::runtime_error("Input images have inconsistent sizes");
    pyramid_levels_ = scene_.ComputePyramidLevels(problems);
    const int max_scale = 1 << (pyramid_levels_ - 1);
    for (auto& p : problems) p.params.max_scale_size = max_scale;

    int iteration = 0;
    for (int level = 0; level < pyramid_levels_; ++level) {
        PrepareGuidanceForLevel(problems, level);
        for (auto& p : problems) p.iteration = iteration;
        RunPass(problems, level, -1,
                level == 0 ? RunState::FirstInit : RunState::RefineInit,
                false);
        ++iteration;

        for (int refine = 0; refine < 3; ++refine) {
            for (auto& p : problems) p.iteration = iteration;
            RunPass(problems, level, refine, RunState::RefineIter, true);
            ++iteration;
        }
        std::cout << "pyramid level " << level << " done\n";
    }
}

void DPEPipeline::DumpDebug(const Problem& problem, const FrameState& state) const {
    boost::filesystem::create_directories(problem.result_folder);
    WriteDepthPreview(problem.result_folder / ("depth_" + std::to_string(problem.iteration) + ".jpg"),
                      state.depth, problem.params.depth_min, problem.params.depth_max);
    WriteNormalPreview(problem.result_folder / ("normal_" + std::to_string(problem.iteration) + ".jpg"), state.normal);
    WriteReliabilityPreview(problem.result_folder / ("weak_" + std::to_string(problem.iteration) + ".jpg"), state.reliability);
    WriteBinMat(problem.result_folder / "depths.dmb", state.depth);
    WriteBinMat(problem.result_folder / "normals.dmb", state.normal);
    WriteBinMat(problem.result_folder / "weak.bin", state.reliability);
    WriteBinMat(problem.result_folder / "selected_views.bin", state.selected_views);
}

}  // namespace dpe
