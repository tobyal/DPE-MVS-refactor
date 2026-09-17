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


struct GpuTelemetry {
    bool enabled = false;
    float* gt_depth = nullptr;
    float3* gt_normal = nullptr;
    unsigned char* gt_valid = nullptr;
    unsigned char* gt_geometry_edge = nullptr;
    int* gt_surface_label = nullptr;

    int* es_candidate_count = nullptr;
    int* es_same_surface = nullptr;
    int* candidate_count = nullptr;
    int* same_surface_candidates = nullptr;
    unsigned char* anchor_count = nullptr;
    unsigned char* same_surface_anchors = nullptr;
    float* plane_depth_error = nullptr;
    float* plane_normal_error = nullptr;
    float* radius_violation = nullptr;
    float* final_depth_error = nullptr;
    float* matching_cost = nullptr;
    int* adaptive_radius = nullptr;
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
    GpuTelemetry telemetry;
    DPEParams* params = nullptr;
};

}  // namespace dpe
