#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "experiments/telemetry/telemetry.h"

namespace dpe {

struct FrameState {
    cv::Mat depth;           // CV_32F
    cv::Mat normal;          // CV_32FC3, world coordinates
    cv::Mat reliability;     // CV_8U, PixelState
    cv::Mat selected_views;  // CV_32S bit mask
    FrameTelemetry telemetry;

    bool Empty() const { return depth.empty(); }
};

class ReconstructionState {
public:
    bool Has(int image_id) const;
    const FrameState* Find(int image_id) const;
    FrameState* Find(int image_id);
    void Put(int image_id, const FrameState& state);
    void Clear();

private:
    std::unordered_map<int, FrameState> frames_;
};

}  // namespace dpe
