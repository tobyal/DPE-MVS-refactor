#pragma once

#include "common/types.h"
#include "dpe/gpu_types.cuh"
#include "runtime/cuda_context.h"
#include "runtime/gpu_scene.h"
#include "scene/scene.h"
#include "scene/reconstruction_state.h"
#include "preprocessing/edge_detection.h"

#include <memory>
#include <vector>

namespace dpe {

void RunDpeKernels(DPEGpuContext* device_context, cudaStream_t stream,
                   const DPEParams& params, int width, int height);

class DPESolver {
public:
    DPESolver(const Problem& problem,
              const SceneView& view,
              const EdgeGuidanceHost& guidance,
              ReconstructionState& reconstruction,
              CudaContext& cuda);
    ~DPESolver();

    FrameState Run();

private:
    void PrepareHostState();
    void AllocateAndUpload();
    FrameState DownloadResult();
    void ReleaseDevice();

    template <typename T>
    T* Allocate(size_t count) {
        T* ptr = AcquireTyped<T>(cuda_.Workspace(), count);
        allocations_.push_back(ptr);
        return ptr;
    }

    const Problem& problem_;
    const SceneView& view_;
    const EdgeGuidanceHost& guidance_;
    ReconstructionState& reconstruction_;
    CudaContext& cuda_;

    DPEParams params_;
    int width_ = 0;
    int height_ = 0;
    int weak_count_ = 0;

    std::vector<float4> host_planes_;
    cv::Mat host_reliability_;
    cv::Mat host_selected_views_;
    cv::Mat host_anchor_map_;
    cv::Mat lowres_edges_;

    std::unique_ptr<GpuScene> gpu_scene_;
    DPEGpuContext host_gpu_{};
    DPEGpuContext* device_gpu_ = nullptr;
    DPEParams* device_params_ = nullptr;
    std::vector<void*> allocations_;
};

}  // namespace dpe
