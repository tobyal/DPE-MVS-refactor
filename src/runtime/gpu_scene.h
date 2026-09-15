#pragma once

#include "common/types.h"
#include "runtime/cuda_context.h"
#include "scene/scene.h"
#include "scene/reconstruction_state.h"

#include <vector>

namespace dpe {

struct GpuTextureSet {
    cudaTextureObject_t images[kMaxImages];
};

struct GpuSceneData {
    Camera* cameras = nullptr;
    GpuTextureSet* image_textures = nullptr;
    GpuTextureSet* depth_textures = nullptr;
    int num_images = 0;
};

class GpuScene {
public:
    GpuScene(CudaContext& cuda, const SceneView& view,
             const ReconstructionState& state, bool upload_depths);
    ~GpuScene();

    GpuScene(const GpuScene&) = delete;
    GpuScene& operator=(const GpuScene&) = delete;

    const GpuSceneData& Data() const { return data_; }

private:
    cudaTextureObject_t CreateTexture(cudaArray_t array) const;

    CudaContext& cuda_;
    const SceneView& view_;
    GpuSceneData data_;
    GpuTextureSet image_handles_{};
    GpuTextureSet depth_handles_{};
    std::vector<cudaArray_t> image_arrays_;
    std::vector<cudaArray_t> depth_arrays_;
};

}  // namespace dpe
