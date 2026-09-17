#pragma once

#include "scene/scene.h"
#include "scene/reconstruction_state.h"
#include "common/types.h"
#include "experiments/ground_truth/ground_truth_provider.h"
#include <boost/filesystem.hpp>
#include <string>
#include <vector>

namespace dpe {

class ArtifactWriter {
public:
    ArtifactWriter(const boost::filesystem::path& experiment_root,
                   const boost::filesystem::path& dense_root);

    void WriteCaseManifest(const std::string& case_name,
                           const std::string& description,
                           const std::vector<Problem>& problems,
                           Scene& scene,
                           const boost::filesystem::path& point_cloud,
                           const std::string& status = "completed",
                           GroundTruthProvider* ground_truth = nullptr);

    void WriteFinalViews(const std::string& case_name,
                         const std::vector<Problem>& problems,
                         const ReconstructionState& reconstruction);

    boost::filesystem::path WriteGroundTruthPreview(const GroundTruthProvider& ground_truth);

    void WriteTelemetrySummary(const std::string& case_name,
                               const std::vector<Problem>& problems,
                               const ReconstructionState& reconstruction);

    void WriteStatus(const std::string& case_name, const std::string& phase,
                     int completed, int total, const std::string& message = "");

private:
    boost::filesystem::path root_;
    boost::filesystem::path dense_;
};

}  // namespace dpe
