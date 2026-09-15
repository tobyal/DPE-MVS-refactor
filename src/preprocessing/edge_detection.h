#pragma once

#include "common/types.h"
#include <opencv2/opencv.hpp>

namespace dpe {

struct EdgeGuidanceHost {
    cv::Mat fine_edges;       // CV_8U, current pyramid size
    cv::Mat lowres_edges;     // CV_8U, coarsest fine-edge map used by edge-crossing tests
    cv::Mat coarse_regions;   // CV_32S, current pyramid size
};

cv::Mat RobertsGradient(const cv::Mat& image);
cv::Mat ConnectedRegionLabels(const cv::Mat& binary, int weak_tex_num);
cv::Mat ComputeFineEdges(const cv::Mat& image_u8);
cv::Mat ComputeCoarseRegions(const cv::Mat& full_res_u8, int scale_log2, bool high_res_img);
EdgeGuidanceHost BuildEdgeGuidance(const cv::Mat& full_res_u8,
                                   const cv::Mat& scaled_u8,
                                   int scale_log2,
                                   const DPEParams& params);

}  // namespace dpe
