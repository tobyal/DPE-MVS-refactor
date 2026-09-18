#pragma once

#include "common/types.h"

#include <boost/filesystem/path.hpp>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <string>

namespace dpe {

struct ReliabilityStudyStage {
    int global_iteration = -1;
    int pyramid_level = -1;
    int scale = 1;
    RunState run_state = RunState::FirstInit;
    int outer_refine = -1;
    int ref_image_id = -1;
    float depth_min = 0.0f;
    float depth_max = 0.0f;
    bool use_apd = false;
    bool use_edge = false;
    bool geom_consistency = false;
};

struct ReliabilityStudySnapshot {
    cv::Mat depth_pre;
    cv::Mat normal_pre;
    cv::Mat reliability;
    cv::Mat depth_post;
    cv::Mat normal_post;
};

// Captures device state at host-side kernel boundaries. It is instantiated only
// in reliability-study mode, so normal reconstruction adds no copies or syncs.
class ReliabilityStudyCapture {
public:
    ReliabilityStudyCapture(int width, int height, const float4* device_planes,
                            const unsigned char* device_reliability);

    void CapturePre(cudaStream_t stream);
    void CaptureReliability(cudaStream_t stream);
    void CapturePost(cudaStream_t stream);

    const ReliabilityStudySnapshot& Snapshot() const { return snapshot_; }

private:
    void CaptureGeometry(const float4* device_planes, cudaStream_t stream,
                         cv::Mat* depth, cv::Mat* normal);

    int width_ = 0;
    int height_ = 0;
    const float4* device_planes_ = nullptr;
    const unsigned char* device_reliability_ = nullptr;
    ReliabilityStudySnapshot snapshot_;
};

class ReliabilityStudyWriter {
public:
    explicit ReliabilityStudyWriter(boost::filesystem::path root);

    void Write(const ReliabilityStudyStage& stage,
               const ReliabilityStudySnapshot& snapshot) const;

private:
    boost::filesystem::path root_;
};

std::string ReliabilityStageId(int global_iteration);
std::string ReliabilityStageDirectoryName(const ReliabilityStudyStage& stage);
const char* ReliabilityRunStateName(RunState state);

}  // namespace dpe
