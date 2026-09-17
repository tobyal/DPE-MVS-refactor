#pragma once

#include "diagnostics/diagnostic_sink.h"

#include <boost/filesystem/path.hpp>
#include <string>

namespace dpe {

class FileDiagnosticSink final : public DiagnosticSink {
public:
    FileDiagnosticSink(boost::filesystem::path root, int traced_view_id,
                       bool trace_all_views);

    bool ShouldCaptureStages(const Problem& problem) const override;
    void BeginRun(int image_count, int pyramid_levels) override;
    void RecordGuidance(const Problem& problem, int scale,
                        const EdgeGuidanceHost& guidance) override;
    void RecordStage(const Problem& problem, const StageSnapshot& snapshot) override;
    void RecordFinalState(const Problem& problem, const FrameState& state) override;
    void RecordFusionView(const FusionViewSnapshot& snapshot) override;
    void RecordPointCloud(const std::vector<DiagnosticPoint>& points) override;

private:
    boost::filesystem::path ScaleViewRoot(int scale, int ref_image_id) const;
    boost::filesystem::path PassRoot(const Problem& problem) const;
    static std::string StageDirectoryName(DiagnosticStage stage, int inner_iteration);

    void WriteStageSummary(const Problem& problem, const StageSnapshot& snapshot,
                           const cv::Mat& update_mask);
    void WriteUpdateEvents(const Problem& problem, const StageSnapshot& snapshot,
                           const cv::Mat& update_mask);
    void WriteFusionSummary(const FusionViewSnapshot& snapshot);

    boost::filesystem::path root_;
    int traced_view_id_ = -1;
    bool trace_all_views_ = false;
    int final_pass_ = -1;
    std::string active_pass_key_;
    cv::Mat previous_depth_;
    cv::Mat previous_cost_;
};

}  // namespace dpe
