#include "fusion/fusion.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <tuple>
#include <unordered_map>

namespace dpe {
namespace {

struct Vec3 { float x, y, z; };

Vec3 CamPoint(const Camera& camera, float x, float y, float depth) {
    return {depth * (x - camera.K[2]) / camera.K[0],
            depth * (y - camera.K[5]) / camera.K[4], depth};
}

Vec3 CamToWorld(const Camera& camera, const Vec3& point) {
    const float x = point.x - camera.t[0];
    const float y = point.y - camera.t[1];
    const float z = point.z - camera.t[2];
    return {camera.R[0] * x + camera.R[3] * y + camera.R[6] * z,
            camera.R[1] * x + camera.R[4] * y + camera.R[7] * z,
            camera.R[2] * x + camera.R[5] * y + camera.R[8] * z};
}

Vec3 WorldToCam(const Camera& camera, const Vec3& point) {
    return {camera.R[0] * point.x + camera.R[1] * point.y + camera.R[2] * point.z + camera.t[0],
            camera.R[3] * point.x + camera.R[4] * point.y + camera.R[5] * point.z + camera.t[1],
            camera.R[6] * point.x + camera.R[7] * point.y + camera.R[8] * point.z + camera.t[2]};
}

cv::Point2f Project(const Camera& camera, const Vec3& point) {
    const float z = camera.K[6] * point.x + camera.K[7] * point.y + camera.K[8] * point.z;
    return {(camera.K[0] * point.x + camera.K[1] * point.y + camera.K[2] * point.z) / z,
            (camera.K[3] * point.x + camera.K[4] * point.y + camera.K[5] * point.z) / z};
}

float NormalAngle(const cv::Vec3f& first, const cv::Vec3f& second) {
    const float first_norm = std::sqrt(first.dot(first));
    const float second_norm = std::sqrt(second.dot(second));
    if (first_norm < 1e-8f || second_norm < 1e-8f) return 3.14159f;
    float dot = first.dot(second) / (first_norm * second_norm);
    dot = std::max(-1.0f, std::min(1.0f, dot));
    return std::acos(dot);
}

void WriteRgbPly(const boost::filesystem::path& path,
                 const std::vector<DiagnosticPoint>& points) {
    std::ofstream out(path.string());
    out << "ply\nformat ascii 1.0\nelement vertex " << points.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (const auto& point : points) {
        out << point.x << ' ' << point.y << ' ' << point.z << ' '
            << static_cast<int>(point.red) << ' '
            << static_cast<int>(point.green) << ' '
            << static_cast<int>(point.blue) << '\n';
    }
}

}  // namespace

void RunFusion(Scene& scene,
               const ReconstructionState& reconstruction,
               const std::vector<Problem>& problems,
               const boost::filesystem::path& output_ply,
               DiagnosticSink* diagnostics) {
    std::vector<DiagnosticPoint> cloud;
    std::unordered_map<int, cv::Mat> used;

    for (const auto& problem : problems) {
        const FrameState* ref_state = reconstruction.Find(problem.ref_image_id);
        if (!ref_state || ref_state->depth.empty()) continue;

        const cv::Mat& ref_image = scene.ColorImage(problem.ref_image_id);
        const Camera ref_camera = scene.ScaledCamera(
            problem.ref_image_id, ref_state->depth.cols, ref_state->depth.rows);
        if (used.find(problem.ref_image_id) == used.end()) {
            used[problem.ref_image_id] = cv::Mat(ref_state->depth.size(), CV_8U, cv::Scalar(0));
        }

        FusionViewSnapshot fusion_snapshot;
        fusion_snapshot.ref_image_id = problem.ref_image_id;
        if (diagnostics) {
            fusion_snapshot.fate = cv::Mat(
                ref_state->depth.size(), CV_8U,
                cv::Scalar(static_cast<unsigned char>(FusionFate::InvalidReferenceDepth)));
            fusion_snapshot.support_count = cv::Mat(ref_state->depth.size(), CV_16U, cv::Scalar(0));
        }

        for (int y = 0; y < ref_state->depth.rows; ++y) {
            for (int x = 0; x < ref_state->depth.cols; ++x) {
                auto set_fate = [&](FusionFate fate) {
                    if (diagnostics) {
                        fusion_snapshot.fate.at<unsigned char>(y, x) =
                            static_cast<unsigned char>(fate);
                    }
                };

                if (used[problem.ref_image_id].at<unsigned char>(y, x)) {
                    set_fate(FusionFate::Duplicate);
                    continue;
                }
                const float depth = ref_state->depth.at<float>(y, x);
                if (depth <= 0.0f || !std::isfinite(depth)) {
                    set_fate(FusionFate::InvalidReferenceDepth);
                    continue;
                }

                const Vec3 world_point = CamToWorld(ref_camera, CamPoint(ref_camera, x, y, depth));
                const cv::Vec3f ref_normal = ref_state->normal.at<cv::Vec3f>(y, x);
                int consistent = 0;
                float dynamic_score = 0.0f;
                bool has_source_depth = false;
                bool passed_reprojection = false;
                bool passed_depth = false;
                bool passed_normal = false;

                const int ref_x = std::min(ref_image.cols - 1, std::max(0,
                    static_cast<int>(x * ref_image.cols / static_cast<float>(ref_state->depth.cols))));
                const int ref_y = std::min(ref_image.rows - 1, std::max(0,
                    static_cast<int>(y * ref_image.rows / static_cast<float>(ref_state->depth.rows))));
                const cv::Vec3b ref_color = ref_image.at<cv::Vec3b>(ref_y, ref_x);
                cv::Vec3i color_sum(ref_color[0], ref_color[1], ref_color[2]);
                std::vector<std::tuple<int, int, int>> matches;

                for (int source_id : problem.src_image_ids) {
                    const FrameState* source_state = reconstruction.Find(source_id);
                    if (!source_state || source_state->depth.empty()) continue;
                    const Camera source_camera = scene.ScaledCamera(
                        source_id, source_state->depth.cols, source_state->depth.rows);
                    const Vec3 source_point = WorldToCam(source_camera, world_point);
                    if (source_point.z <= 0.0f) continue;
                    const cv::Point2f source_pixel = Project(source_camera, source_point);
                    const int source_x = static_cast<int>(std::round(source_pixel.x));
                    const int source_y = static_cast<int>(std::round(source_pixel.y));
                    if (source_x < 0 || source_y < 0 ||
                        source_x >= source_state->depth.cols || source_y >= source_state->depth.rows) continue;

                    const float source_depth = source_state->depth.at<float>(source_y, source_x);
                    if (source_depth <= 0.0f || !std::isfinite(source_depth)) continue;
                    has_source_depth = true;

                    const Vec3 source_world = CamToWorld(
                        source_camera, CamPoint(source_camera, source_x, source_y, source_depth));
                    const Vec3 roundtrip_point = WorldToCam(ref_camera, source_world);
                    if (roundtrip_point.z <= 0.0f) continue;
                    const cv::Point2f roundtrip_pixel = Project(ref_camera, roundtrip_point);
                    const float reprojection_error = std::hypot(roundtrip_pixel.x - x, roundtrip_pixel.y - y);
                    if (reprojection_error >= 2.0f) continue;
                    passed_reprojection = true;

                    const float relative_depth_error =
                        std::fabs(roundtrip_point.z - depth) / std::max(depth, 1e-6f);
                    if (relative_depth_error >= 0.01f) continue;
                    passed_depth = true;

                    const float normal_error = NormalAngle(
                        ref_normal, source_state->normal.at<cv::Vec3f>(source_y, source_x));
                    if (normal_error >= 0.174533f) continue;
                    passed_normal = true;

                    ++consistent;
                    dynamic_score += std::exp(-(
                        reprojection_error + 200.0f * relative_depth_error + 10.0f * normal_error));
                    matches.emplace_back(source_id, source_x, source_y);

                    const cv::Mat& source_image = scene.ColorImage(source_id);
                    const int image_x = std::min(source_image.cols - 1, std::max(0,
                        static_cast<int>(source_x * source_image.cols /
                                         static_cast<float>(source_state->depth.cols))));
                    const int image_y = std::min(source_image.rows - 1, std::max(0,
                        static_cast<int>(source_y * source_image.rows /
                                         static_cast<float>(source_state->depth.rows))));
                    const cv::Vec3b source_color = source_image.at<cv::Vec3b>(image_y, image_x);
                    color_sum[0] += source_color[0];
                    color_sum[1] += source_color[1];
                    color_sum[2] += source_color[2];
                }

                if (diagnostics) {
                    fusion_snapshot.support_count.at<unsigned short>(y, x) =
                        static_cast<unsigned short>(consistent);
                }
                const unsigned char reliability = ref_state->reliability.at<unsigned char>(y, x);
                const float acceptance_factor = reliability == WEAK ? 0.45f : 0.30f;
                if (consistent >= 1 && dynamic_score > acceptance_factor * consistent) {
                    set_fate(FusionFate::Accept);
                    const cv::Vec3b color(
                        static_cast<unsigned char>(color_sum[0] / (consistent + 1)),
                        static_cast<unsigned char>(color_sum[1] / (consistent + 1)),
                        static_cast<unsigned char>(color_sum[2] / (consistent + 1)));
                    DiagnosticPoint point;
                    point.x = world_point.x;
                    point.y = world_point.y;
                    point.z = world_point.z;
                    point.red = color[2];
                    point.green = color[1];
                    point.blue = color[0];
                    point.reliability = reliability;
                    point.support_count = consistent;
                    point.ref_image_id = problem.ref_image_id;
                    cloud.push_back(point);
                    used[problem.ref_image_id].at<unsigned char>(y, x) = 1;
                    for (const auto& match : matches) {
                        const int source_id = std::get<0>(match);
                        const int source_x = std::get<1>(match);
                        const int source_y = std::get<2>(match);
                        const FrameState* source_state = reconstruction.Find(source_id);
                        if (used.find(source_id) == used.end()) {
                            used[source_id] = cv::Mat(source_state->depth.size(), CV_8U, cv::Scalar(0));
                        }
                        used[source_id].at<unsigned char>(source_y, source_x) = 1;
                    }
                } else if (!has_source_depth) {
                    set_fate(FusionFate::NoSourceSupport);
                } else if (!passed_reprojection) {
                    set_fate(FusionFate::ReprojectionFail);
                } else if (!passed_depth) {
                    set_fate(FusionFate::DepthConsistencyFail);
                } else if (!passed_normal) {
                    set_fate(FusionFate::NormalConsistencyFail);
                } else {
                    set_fate(FusionFate::ScoreFail);
                }
            }
        }
        if (diagnostics) diagnostics->RecordFusionView(fusion_snapshot);
    }

    WriteRgbPly(output_ply, cloud);
    if (diagnostics) diagnostics->RecordPointCloud(cloud);
    std::cout << "fusion: " << cloud.size() << " points -> " << output_ply.string() << '\n';
}

}  // namespace dpe
