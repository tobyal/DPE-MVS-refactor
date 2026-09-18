#include "pipeline/dpe_pipeline.h"
#include "dpe/dpe.h"
#include "common/io.h"
#include "diagnostics/reliability_study.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace dpe {

namespace {

using Clock = std::chrono::steady_clock;

std::string FormatDuration(std::chrono::milliseconds value) {
    const auto total_seconds = value.count() / 1000;
    const auto minutes = total_seconds / 60;
    const auto seconds = total_seconds % 60;
    std::ostringstream out;
    out << std::setfill('0') << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
    return out.str();
}

std::string ProgressBar(int completed, int total) {
    constexpr int width = 20;
    const double ratio = total > 0 ? static_cast<double>(completed) / total : 1.0;
    const int filled = std::max(0, std::min(width, static_cast<int>(ratio * width + 0.5)));
    return '[' + std::string(filled, '#') + std::string(width - filled, '-') + ']';
}

std::string StageName(RunState state, int outer_refine) {
    if (state == RunState::FirstInit) return "Coarse Init";
    if (state == RunState::RefineInit) return "DPE Init";
    return "PM Refine " + std::to_string(outer_refine + 1) + "/3";
}

void PrintStageProgress(const std::string& name, int completed, int total,
                        const Clock::time_point& started) {
    const auto now = Clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - started);
    const int percent = total > 0 ? static_cast<int>((100.0 * completed / total) + 0.5) : 100;

    std::cout << "\r  " << std::left << std::setw(16) << name << ' '
              << ProgressBar(completed, total) << ' '
              << std::right << std::setw(3) << percent << "%  "
              << std::setw(3) << completed << '/' << total << "  "
              << FormatDuration(elapsed);

    if (completed > 0 && completed < total) {
        const auto eta_ms = elapsed.count() * (total - completed) / completed;
        std::cout << "  ETA " << FormatDuration(std::chrono::milliseconds(eta_ms));
    }
    std::cout << std::flush;
}

bool ShouldRefreshProgress(int completed, int total, int* last_bucket) {
    if (completed == 0 || completed == total) return true;
    if (total <= 20) return true;
    const int bucket = static_cast<int>((100.0 * completed / total) / 5.0);
    if (bucket <= *last_bucket) return false;
    *last_bucket = bucket;
    return true;
}

}  // namespace

DPEPipeline::DPEPipeline(Scene& scene, ReconstructionState& reconstruction, CudaContext& cuda,
                         ReliabilityStudyWriter* reliability_study)
    : scene_(scene), reconstruction_(reconstruction), cuda_(cuda),
      reliability_study_(reliability_study) {}

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

void DPEPipeline::ProcessView(Problem& problem, int level, int outer_refine) {
    SceneView view = scene_.MakeView(problem);
    // Keep the effective depth interval in the Problem as well as in the solver
    // so previews/checkpoints use the same range as PatchMatch.
    problem.params.depth_min = view.cameras.front().depth_min * 0.6f;
    problem.params.depth_max = view.cameras.front().depth_max * 1.2f;
    const EdgeGuidanceHost& guidance = scene_.Guidance(problem.ref_image_id, problem.scale_size, problem.params);
    ReliabilityStudyStage stage;
    stage.global_iteration = problem.iteration;
    stage.pyramid_level = level;
    stage.scale = problem.scale_size;
    stage.run_state = problem.params.state;
    stage.outer_refine = problem.params.state == RunState::RefineIter ? outer_refine + 1 : -1;
    stage.ref_image_id = problem.ref_image_id;
    stage.depth_min = problem.params.depth_min;
    stage.depth_max = problem.params.depth_max;
    stage.use_apd = problem.params.use_apd;
    stage.use_edge = problem.params.use_edge;
    stage.geom_consistency = problem.params.geom_consistency;
    DPESolver solver(problem, view, guidance, reconstruction_, cuda_, reliability_study_,
                     reliability_study_ ? &stage : nullptr);
    FrameState result = solver.Run();
    reconstruction_.Put(problem.ref_image_id, result);
    if (problem.show_medium_result) DumpDebug(problem, result);
}

void DPEPipeline::RunPass(std::vector<Problem>& problems, int level, int outer_refine,
                          RunState state, bool geom_consistency) {
    const int scale = 1 << (pyramid_levels_ - 1 - level);
    const std::string stage = StageName(state, outer_refine);
    const auto started = Clock::now();
    int completed = 0;
    int last_bucket = 0;
    PrintStageProgress(stage, completed, static_cast<int>(problems.size()), started);

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
        ProcessView(problem, level, outer_refine);
        ++completed;
        if (ShouldRefreshProgress(completed, static_cast<int>(problems.size()), &last_bucket)) {
            PrintStageProgress(stage, completed, static_cast<int>(problems.size()), started);
        }
    }
    PrintStageProgress(stage, completed, static_cast<int>(problems.size()), started);
    std::cout << '\n';
}

void DPEPipeline::Run(std::vector<Problem>& problems) {
    if (!scene_.CheckImageSizes(problems)) throw std::runtime_error("Input images have inconsistent sizes");
    pyramid_levels_ = scene_.ComputePyramidLevels(problems);
    const int max_scale = 1 << (pyramid_levels_ - 1);
    for (auto& p : problems) p.params.max_scale_size = max_scale;

    std::cout << "DPE-MVS Reconstruction\n"
              << "Images : " << problems.size() << "\n"
              << "Pyramid: " << pyramid_levels_ << " scales\n";

    int iteration = 0;
    for (int level = 0; level < pyramid_levels_; ++level) {
        const int scale = 1 << (pyramid_levels_ - 1 - level);
        const auto& ref = scene_.ScaledGrayFloat(problems.front().ref_image_id, scale);
        const auto scale_started = Clock::now();

        std::cout << "\n----------------------------------------\n"
                  << "Scale " << (level + 1) << '/' << pyramid_levels_
                  << " | 1/" << scale
                  << " | " << ref.cols << 'x' << ref.rows << "\n"
                  << "Views " << problems.size() << "\n"
                  << "----------------------------------------\n";

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

        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - scale_started);
        std::cout << "Scale " << (level + 1) << " completed in " << FormatDuration(elapsed) << "\n";
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
