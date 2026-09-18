#pragma once

#include "common/types.h"
#include "evaluation_types.h"

#include <opencv2/core.hpp>
#include <cstddef>
#include <vector>

namespace dpe {
namespace eth3d {

struct ProjectionResult {
    cv::Mat image;
    std::size_t projected_points = 0;
    std::size_t visible_pixels = 0;
};

ProjectionResult ProjectEvaluationCloud(const std::vector<EvaluationPoint>& points,
                                        const Camera& camera,
                                        const cv::Mat& source_image,
                                        EvaluationKind kind,
                                        double tolerance_meters,
                                        int splat_radius,
                                        double alpha);

}  // namespace eth3d
}  // namespace dpe
