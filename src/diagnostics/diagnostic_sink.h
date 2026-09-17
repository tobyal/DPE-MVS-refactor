#pragma once

#include "common/types.h"
#include "preprocessing/edge_detection.h"

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace dpe {

struct FrameState;

enum class DiagnosticStage {
    Input,
    Strong,
    Plane,
    Weak,
    Finalized,
    Filtered,
    Classified,
    Refined,
};

enum class FusionFate : unsigned char {
    Accept = 0,
    InvalidReferenceDepth = 1,
    NoSourceSupport = 2,
    ReprojectionFail = 3,
    DepthConsistencyFail = 4,
    NormalConsistencyFail = 5,
    ScoreFail = 6,
    Duplicate = 7,
};

struct StageSnapshot {
    DiagnosticStage stage = DiagnosticStage::Input;
    int inner_iteration = -1;
    cv::Mat depth;
    cv::Mat normal;
    cv::Mat reliability;
    cv::Mat cost;
    cv::Mat selected_views;
    cv::Mat adaptive_radius;
    cv::Mat anchor_count;
    cv::Mat texture_complexity;
};

struct FusionViewSnapshot {
    int ref_image_id = -1;
    cv::Mat fate;
    cv::Mat support_count;
};

struct DiagnosticPoint {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    unsigned char red = 0;
    unsigned char green = 0;
    unsigned char blue = 0;
    unsigned char reliability = 0;
    int support_count = 0;
    int ref_image_id = -1;
};

class DiagnosticSink {
public:
    virtual ~DiagnosticSink() = default;

    virtual bool ShouldCaptureStages(const Problem& problem) const = 0;
    virtual void BeginRun(int image_count, int pyramid_levels) = 0;
    virtual void RecordGuidance(const Problem& problem, int scale,
                                const EdgeGuidanceHost& guidance) = 0;
    virtual void RecordStage(const Problem& problem, const StageSnapshot& snapshot) = 0;
    virtual void RecordFinalState(const Problem& problem, const FrameState& state) = 0;
    virtual void RecordFusionView(const FusionViewSnapshot& snapshot) = 0;
    virtual void RecordPointCloud(const std::vector<DiagnosticPoint>& points) = 0;
};

}  // namespace dpe
