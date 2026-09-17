#pragma once

#include "common/types.h"
#include "runtime/gpu_scene.h"
#include <curand_kernel.h>

namespace dpe {

struct GpuPatchMatchState {
    float4* planes = nullptr;
    float4* fitted_planes = nullptr;
    float* costs = nullptr;

    unsigned char* reliability = nullptr;
    unsigned char* weak_reliable = nullptr;
    unsigned int* selected_views = nullptr;
    unsigned char* view_weights = nullptr;

    short2* anchors = nullptr;
    int* anchor_map = nullptr;
    short2* nearest_strong = nullptr;

    int* radius = nullptr;
    curandState* random_states = nullptr;
};

struct GpuGuidance {
    unsigned char* fine_edges = nullptr;
    unsigned char* lowres_edges = nullptr;
    int low_width = 0;
    int low_height = 0;

    int* region_labels = nullptr;
    short2* region_boundaries = nullptr;
    short2* nearest_edges = nullptr;
    float* texture_complexity = nullptr;
};

struct DPEGpuContext {
    int width = 0;
    int height = 0;
    int ref_id = -1;
    int num_images = 0;
    int weak_count = 0;

    Camera* cameras = nullptr;
    GpuTextureSet* image_textures = nullptr;
    GpuTextureSet* depth_textures = nullptr;

    GpuPatchMatchState state;
    GpuGuidance guidance;
    DPEParams* params = nullptr;
};

}  // namespace dpe
