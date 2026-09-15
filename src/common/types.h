#pragma once

#include <cuda_runtime.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace dpe {

constexpr int kMaxImages = 32;
constexpr int kNeighbourNum = 9;
constexpr int kMaxSearchRadius = 4096;

struct Camera {
    float K[9]{};
    float R[9]{};
    float t[3]{};
    float c[3]{};
    int height = 0;
    int width = 0;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
};

struct PointList {
    float3 coord;
    float3 color;
};

enum class RunState : int {
    FirstInit = 0,
    RefineInit = 1,
    RefineIter = 2,
};

enum PixelState : unsigned char {
    WEAK = 0,
    STRONG = 1,
    UNKNOWN = 2,
};

struct DPEParams {
    int max_iterations = 3;
    int num_images = 5;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int top_k = 4;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    bool geom_consistency = false;

    int strong_radius = 5;
    int strong_increment = 2;
    int weak_radius = 5;
    int weak_increment = 5;

    bool use_apd = true;
    bool use_edge = true;
    bool use_limit = true;
    bool use_label = true;
    bool use_radius = true;
    bool high_res_img = true;

    int max_scale_size = 1;
    int scale_size = 1;
    int weak_peak_radius = 2;
    int rotate_time = 4;
    float ransac_threshold = 0.005f;
    float geom_factor = 0.2f;
    RunState state = RunState::FirstInit;
};

struct Problem {
    int index = 0;
    int ref_image_id = -1;
    std::vector<int> src_image_ids;
    boost::filesystem::path dense_folder;
    boost::filesystem::path result_folder;
    int scale_size = 1;
    DPEParams params;
    bool show_medium_result = false;
    int iteration = 0;
};

}  // namespace dpe
