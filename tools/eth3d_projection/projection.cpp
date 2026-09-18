#include "projection.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dpe {
namespace eth3d {
namespace {

struct LegendEntry {
    cv::Scalar color;
    std::string label;
};

void DrawLegend(cv::Mat* image, EvaluationKind kind, double tolerance_meters) {
    const double ui_scale = std::max(
        1.0, std::min(1.8, std::min(image->cols / 1920.0, image->rows / 1080.0)));
    const int margin = static_cast<int>(std::round(16.0 * ui_scale));
    const int row_height = static_cast<int>(std::round(30.0 * ui_scale));
    const int marker_radius = static_cast<int>(std::round(6.0 * ui_scale));
    const double font_scale = 0.65 * ui_scale;
    const int thickness = std::max(1, static_cast<int>(std::round(ui_scale)));

    std::vector<LegendEntry> entries;
    std::string title;
    if (kind == EvaluationKind::Accuracy) {
        title = "Accuracy | tolerance " + cv::format("%.2f m", tolerance_meters);
        entries = {
            {cv::Scalar(0, 255, 0), "Green = Accurate"},
            {cv::Scalar(0, 0, 255), "Red = Inaccurate"},
            {cv::Scalar(255, 0, 0), "Blue = Unobserved"},
        };
    } else {
        title = "Completeness | tolerance " + cv::format("%.2f m", tolerance_meters);
        entries = {
            {cv::Scalar(0, 255, 0), "Green = Complete"},
            {cv::Scalar(0, 0, 255), "Red = Incomplete"},
        };
    }

    int max_text_width = cv::getTextSize(
        title, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, nullptr).width;
    for (const LegendEntry& entry : entries) {
        max_text_width = std::max(max_text_width, cv::getTextSize(
            entry.label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, nullptr).width + row_height);
    }
    const int panel_width = std::min(image->cols, max_text_width + 2 * margin);
    const int panel_height = std::min(
        image->rows, margin * 2 + row_height * (static_cast<int>(entries.size()) + 1));

    cv::Mat shaded = image->clone();
    cv::rectangle(shaded, cv::Rect(0, 0, panel_width, panel_height), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(shaded, 0.62, *image, 0.38, 0.0, *image);

    int baseline_y = margin + row_height * 3 / 4;
    cv::putText(*image, title, cv::Point(margin, baseline_y), cv::FONT_HERSHEY_SIMPLEX,
                font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    for (const LegendEntry& entry : entries) {
        baseline_y += row_height;
        cv::circle(*image, cv::Point(margin + marker_radius, baseline_y - marker_radius),
                   marker_radius, entry.color, cv::FILLED, cv::LINE_AA);
        cv::putText(*image, entry.label,
                    cv::Point(margin + row_height, baseline_y), cv::FONT_HERSHEY_SIMPLEX,
                    font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
}

}  // namespace

ProjectionResult ProjectEvaluationCloud(const std::vector<EvaluationPoint>& points,
                                        const Camera& camera,
                                        const cv::Mat& source_image,
                                        EvaluationKind kind,
                                        double tolerance_meters,
                                        int splat_radius,
                                        double alpha) {
    if (source_image.empty() || source_image.type() != CV_8UC3) {
        throw std::runtime_error("Projection source image must be a non-empty 8-bit color image");
    }
    if (splat_radius < 0) throw std::runtime_error("Splat radius must be non-negative");
    if (alpha < 0.0 || alpha > 1.0) throw std::runtime_error("Alpha must be in [0, 1]");

    cv::Mat depth(source_image.size(), CV_32F,
                  cv::Scalar(std::numeric_limits<float>::infinity()));
    cv::Mat evaluation_color(source_image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat covered(source_image.size(), CV_8U, cv::Scalar(0));

    ProjectionResult result;
    const long long radius_squared =
        static_cast<long long>(splat_radius) * splat_radius;
    for (const EvaluationPoint& point : points) {
        const double camera_x = camera.R[0] * point.x + camera.R[1] * point.y +
                                camera.R[2] * point.z + camera.t[0];
        const double camera_y = camera.R[3] * point.x + camera.R[4] * point.y +
                                camera.R[5] * point.z + camera.t[1];
        const double camera_z = camera.R[6] * point.x + camera.R[7] * point.y +
                                camera.R[8] * point.z + camera.t[2];
        if (!(camera_z > 0.0) || !std::isfinite(camera_z)) continue;

        const double projected_x = camera.K[0] * camera_x + camera.K[1] * camera_y +
                                   camera.K[2] * camera_z;
        const double projected_y = camera.K[3] * camera_x + camera.K[4] * camera_y +
                                   camera.K[5] * camera_z;
        const double projected_z = camera.K[6] * camera_x + camera.K[7] * camera_y +
                                   camera.K[8] * camera_z;
        if (!std::isfinite(projected_z) || std::fabs(projected_z) < 1e-12) continue;

        const double image_x = projected_x / projected_z;
        const double image_y = projected_y / projected_z;
        if (!std::isfinite(image_x) || !std::isfinite(image_y)) continue;
        if (image_x < -splat_radius - 0.5 || image_y < -splat_radius - 0.5 ||
            image_x > source_image.cols - 0.5 + splat_radius ||
            image_y > source_image.rows - 0.5 + splat_radius) {
            continue;
        }

        const long center_x_long = std::lround(image_x);
        const long center_y_long = std::lround(image_y);
        if (center_x_long + splat_radius < 0 || center_y_long + splat_radius < 0 ||
            center_x_long - splat_radius >= source_image.cols ||
            center_y_long - splat_radius >= source_image.rows) {
            continue;
        }

        const int center_x = static_cast<int>(center_x_long);
        const int center_y = static_cast<int>(center_y_long);
        bool projects_to_image = false;
        for (int offset_y = -splat_radius; offset_y <= splat_radius; ++offset_y) {
            const int pixel_y = center_y + offset_y;
            if (pixel_y < 0 || pixel_y >= source_image.rows) continue;
            for (int offset_x = -splat_radius; offset_x <= splat_radius; ++offset_x) {
                const long long squared_distance =
                    static_cast<long long>(offset_x) * offset_x +
                    static_cast<long long>(offset_y) * offset_y;
                if (squared_distance > radius_squared) continue;
                const int pixel_x = center_x + offset_x;
                if (pixel_x < 0 || pixel_x >= source_image.cols) continue;
                projects_to_image = true;

                float& nearest_depth = depth.at<float>(pixel_y, pixel_x);
                if (camera_z >= nearest_depth) continue;
                nearest_depth = static_cast<float>(camera_z);
                evaluation_color.at<cv::Vec3b>(pixel_y, pixel_x) =
                    cv::Vec3b(point.blue, point.green, point.red);
                covered.at<unsigned char>(pixel_y, pixel_x) = 255;
            }
        }
        if (projects_to_image) ++result.projected_points;
    }

    result.visible_pixels = static_cast<std::size_t>(cv::countNonZero(covered));
    result.image = source_image.clone();
    cv::Mat blended;
    cv::addWeighted(source_image, 1.0 - alpha, evaluation_color, alpha, 0.0, blended);
    blended.copyTo(result.image, covered);
    DrawLegend(&result.image, kind, tolerance_meters);
    return result;
}

}  // namespace eth3d
}  // namespace dpe
