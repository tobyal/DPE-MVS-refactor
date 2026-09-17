#include "experiments/experiment_runner.h"
#include "common/io.h"
#include "pipeline/dpe_pipeline.h"
#include "scene/reconstruction_state.h"
#include "fusion/fusion.h"
#include "studio/artifact_writer.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <stdexcept>

namespace dpe {
ExperimentRunner::ExperimentRunner(Scene& scene, CudaContext& cuda,
                                   boost::filesystem::path experiment_root,
                                   GroundTruthProvider* ground_truth)
    : scene_(scene), cuda_(cuda), root_(std::move(experiment_root)), ground_truth_(ground_truth) {}

void ExperimentRunner::Run(const std::vector<Problem>& base_problems,
                           const std::vector<ExperimentCase>& cases) {
    ArtifactWriter writer(root_, scene_.Root());
    if (ground_truth_) writer.WriteGroundTruthPreview(*ground_truth_);
    int completed = 0;
    int total = 0;
    for (const auto& c : cases) if (c.enabled) ++total;

    for (const auto& c : cases) {
        if (!c.enabled) continue;
        if (c.policy.RequiresExternalPrior()) {
            throw std::runtime_error("External/oracle priors require a GroundTruthProvider; case: " + c.name);
        }
        writer.WriteStatus(c.name, "reconstruction", completed, total, c.description);
        auto problems = base_problems; // strict case isolation
        const auto case_dir = root_ / c.name;
        boost::filesystem::create_directories(case_dir / "views");
        for (auto& p : problems) {
            p.result_folder = case_dir / "views" / FormatIndex(p.ref_image_id);
        }

        ReconstructionState state;
        GroundTruthProvider* case_gt = (c.telemetry ? ground_truth_ : nullptr);
        DPEPipeline pipeline(scene_, state, cuda_, c.policy, case_gt);
        pipeline.Run(problems);
        const auto ply = case_dir / "DPE.ply";
        RunFusion(scene_, state, problems, ply);
        writer.WriteFinalViews(c.name, problems, state);
        writer.WriteTelemetrySummary(c.name, problems, state);
        writer.WriteCaseManifest(c.name, c.description, problems, scene_, ply, "completed", case_gt);
        ++completed;
        writer.WriteStatus(c.name, "completed", completed, total, c.description);
        std::cout << "experiment " << c.name << " complete\n";
    }
}
}  // namespace dpe
