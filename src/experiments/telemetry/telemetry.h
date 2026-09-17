#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace dpe {

struct AnchorRecord {
    int pixel_index = -1;
    short xy[16]{};  // 8 anchor points: x0,y0,...,x7,y7; -1 means absent
};

struct PlaneRecord {
    int pixel_index = -1;
    float plane[4]{}; // camera-coordinate plane normal xyz + distance w
    int radius = 0;
};

struct FrameTelemetry {
    cv::Mat gt_depth;                 // CV_32F
    cv::Mat gt_normal;                // CV_32FC3
    cv::Mat gt_valid;                 // CV_8U
    cv::Mat gt_geometry_edge;         // CV_8U
    cv::Mat gt_surface_label;         // CV_32S

    cv::Mat fine_edge;                // CV_8U, DPE image edge
    cv::Mat coarse_region;            // CV_32S, DPE coarse region label
    cv::Mat texture_complexity;       // CV_32F, stochastic edge relaxation score
    cv::Mat es_candidate_count;       // CV_32S, strong-path selected candidates
    cv::Mat es_same_surface;          // CV_32S
    cv::Mat candidate_count;          // CV_32S, PRE/anchor candidate set
    cv::Mat same_surface_candidates;  // CV_32S
    cv::Mat anchor_count;             // CV_8U
    cv::Mat same_surface_anchors;     // CV_8U
    cv::Mat fitted_plane_depth_error; // CV_32F, meters
    cv::Mat fitted_plane_normal_error;// CV_32F, degrees
    cv::Mat radius_violation;         // CV_32F, [0,1]
    cv::Mat final_depth_error;        // CV_32F, meters
    cv::Mat matching_cost;            // CV_32F
    cv::Mat adaptive_radius;          // CV_32S

    std::vector<AnchorRecord> anchors; // sparse, one record per weak pixel with anchor-map entry
    std::vector<PlaneRecord> planes;  // sparse fitted plane records for the same pixels

    bool Empty() const { return gt_depth.empty(); }
};

}  // namespace dpe
