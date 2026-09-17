#include "diagnostics/file_diagnostic_sink.h"

#include "common/io.h"
#include "scene/reconstruction_state.h"

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <utility>

namespace dpe {
namespace {

std::string Padded(int value, int width) {
    std::ostringstream out;
    out << std::setfill('0') << std::setw(width) << value;
    return out.str();
}

const char* RunStateName(RunState state) {
    switch (state) {
        case RunState::FirstInit: return "coarse_init";
        case RunState::RefineInit: return "dpe_init";
        case RunState::RefineIter: return "pm_refine";
    }
    return "unknown";
}

void WriteMat(const boost::filesystem::path& directory, const char* name, const cv::Mat& mat) {
    if (!mat.empty()) WriteBinMat(directory / (std::string(name) + ".dmb"), mat);
}

cv::Mat BuildUpdateMask(const cv::Mat& previous, const cv::Mat& current) {
    if (previous.empty() || previous.size() != current.size()) return cv::Mat();
    cv::Mat mask(current.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < current.rows; ++y) {
        for (int x = 0; x < current.cols; ++x) {
            const float before = previous.at<float>(y, x);
            const float after = current.at<float>(y, x);
            const bool changed = (before > 0.0f) != (after > 0.0f) ||
                                 (before > 0.0f && after > 0.0f && std::fabs(before - after) > 1e-6f);
            if (changed) mask.at<unsigned char>(y, x) = 255;
        }
    }
    return mask;
}

void WritePly(const boost::filesystem::path& path,
              const std::vector<DiagnosticPoint>& points,
              const std::function<cv::Vec3b(const DiagnosticPoint&)>& color) {
    std::ofstream out(path.string());
    out << "ply\nformat ascii 1.0\nelement vertex " << points.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (const auto& point : points) {
        const cv::Vec3b c = color(point);
        out << point.x << ' ' << point.y << ' ' << point.z << ' '
            << static_cast<int>(c[2]) << ' '
            << static_cast<int>(c[1]) << ' '
            << static_cast<int>(c[0]) << '\n';
    }
}

cv::Vec3b IndexColor(int value) {
    const unsigned int hash = static_cast<unsigned int>(value) * 2654435761u;
    return cv::Vec3b(
        static_cast<unsigned char>(64 + (hash & 0x7f)),
        static_cast<unsigned char>(64 + ((hash >> 8) & 0x7f)),
        static_cast<unsigned char>(64 + ((hash >> 16) & 0x7f)));
}

}  // namespace

FileDiagnosticSink::FileDiagnosticSink(boost::filesystem::path root,
                                       int traced_view_id,
                                       bool trace_all_views)
    : root_(std::move(root)),
      traced_view_id_(traced_view_id),
      trace_all_views_(trace_all_views) {
    boost::filesystem::create_directories(root_);
}

bool FileDiagnosticSink::ShouldCaptureStages(const Problem& problem) const {
    const bool selected_view = trace_all_views_ || problem.ref_image_id == traced_view_id_;
    return selected_view && problem.scale_size == 1 && problem.iteration == final_pass_;
}

void FileDiagnosticSink::BeginRun(int image_count, int pyramid_levels) {
    final_pass_ = pyramid_levels * 4 - 1;
    boost::filesystem::create_directories(root_ / "levels");
    boost::filesystem::create_directories(root_ / "fusion");
    std::ofstream manifest((root_ / "manifest.txt").string());
    manifest << "schema=1\n"
             << "images=" << image_count << "\n"
             << "pyramid_levels=" << pyramid_levels << "\n"
             << "stage_trace_view=" << (trace_all_views_ ? "all" : std::to_string(traced_view_id_)) << "\n"
             << "stage_trace_pass=" << final_pass_ << "\n"
             << "fusion_fate_codes=0:accept,1:invalid_reference_depth,2:no_source_support,"
                "3:reprojection_fail,4:depth_consistency_fail,5:normal_consistency_fail,"
                "6:score_fail,7:duplicate\n";
}

boost::filesystem::path FileDiagnosticSink::ScaleViewRoot(int scale, int ref_image_id) const {
    return root_ / "levels" / ("scale_" + Padded(scale, 2)) /
           ("ref_" + FormatIndex(ref_image_id));
}

boost::filesystem::path FileDiagnosticSink::PassRoot(const Problem& problem) const {
    return ScaleViewRoot(problem.scale_size, problem.ref_image_id) /
           ("pass_" + Padded(problem.iteration, 3) + "_" + RunStateName(problem.params.state));
}

std::string FileDiagnosticSink::StageDirectoryName(DiagnosticStage stage, int inner_iteration) {
    if (stage == DiagnosticStage::Input) return "00_input";
    if (stage == DiagnosticStage::Strong) return Padded(1 + inner_iteration * 3, 2) + "_strong_iter_" + std::to_string(inner_iteration);
    if (stage == DiagnosticStage::Plane) return Padded(2 + inner_iteration * 3, 2) + "_plane_iter_" + std::to_string(inner_iteration);
    if (stage == DiagnosticStage::Weak) return Padded(3 + inner_iteration * 3, 2) + "_weak_iter_" + std::to_string(inner_iteration);
    if (stage == DiagnosticStage::Finalized) return "10_finalized";
    if (stage == DiagnosticStage::Filtered) return "11_filtered";
    if (stage == DiagnosticStage::Classified) return "12_classified";
    return "13_refined";
}

void FileDiagnosticSink::RecordGuidance(const Problem& problem, int scale,
                                        const EdgeGuidanceHost& guidance) {
    const auto directory = ScaleViewRoot(scale, problem.ref_image_id) / "guidance";
    boost::filesystem::create_directories(directory);
    WriteMat(directory, "fine_edge", guidance.fine_edges);
    WriteMat(directory, "coarse_region", guidance.coarse_regions);
    WriteMat(directory, "lowres_edge", guidance.lowres_edges);
    if (!guidance.fine_edges.empty()) cv::imwrite((directory / "fine_edge.png").string(), guidance.fine_edges);
}

void FileDiagnosticSink::RecordStage(const Problem& problem, const StageSnapshot& snapshot) {
    const std::string pass_key = PassRoot(problem).string();
    if (active_pass_key_ != pass_key || snapshot.stage == DiagnosticStage::Input) {
        active_pass_key_ = pass_key;
        previous_depth_.release();
        previous_cost_.release();
    }

    const auto directory = PassRoot(problem) / "stages" /
                           StageDirectoryName(snapshot.stage, snapshot.inner_iteration);
    boost::filesystem::create_directories(directory);

    const cv::Mat update_mask = BuildUpdateMask(previous_depth_, snapshot.depth);
    WriteMat(directory, "depth", snapshot.depth);
    WriteMat(directory, "normal", snapshot.normal);
    WriteMat(directory, "reliability", snapshot.reliability);
    WriteMat(directory, "matching_cost", snapshot.cost);
    WriteMat(directory, "selected_views", snapshot.selected_views);
    WriteMat(directory, "adaptive_radius", snapshot.adaptive_radius);
    WriteMat(directory, "anchor_count", snapshot.anchor_count);
    WriteMat(directory, "texture_complexity", snapshot.texture_complexity);
    WriteMat(directory, "update_mask", update_mask);

    if (!snapshot.depth.empty()) {
        WriteDepthPreview(directory / "depth.jpg", snapshot.depth,
                          problem.params.depth_min, problem.params.depth_max);
    }
    if (!snapshot.normal.empty()) WriteNormalPreview(directory / "normal.jpg", snapshot.normal);
    if (!snapshot.reliability.empty()) {
        WriteReliabilityPreview(directory / "reliability.jpg", snapshot.reliability);
    }
    if (!update_mask.empty()) cv::imwrite((directory / "update_mask.png").string(), update_mask);

    WriteStageSummary(problem, snapshot, update_mask);
    WriteUpdateEvents(problem, snapshot, update_mask);
    previous_depth_ = snapshot.depth.clone();
    previous_cost_ = snapshot.cost.clone();
}

void FileDiagnosticSink::WriteStageSummary(const Problem& problem,
                                           const StageSnapshot& snapshot,
                                           const cv::Mat& update_mask) {
    const auto path = PassRoot(problem) / "stage_summary.csv";
    const bool create_header = !boost::filesystem::exists(path);
    std::ofstream out(path.string(), std::ios::app);
    if (create_header) {
        out << "stage,inner_iteration,pixels,valid_depth,updated,strong,weak,unknown,mean_depth,mean_cost\n";
    }

    long long valid = 0;
    long long updated = update_mask.empty() ? 0 : cv::countNonZero(update_mask);
    long long strong = 0;
    long long weak = 0;
    long long unknown = 0;
    double depth_sum = 0.0;
    double cost_sum = 0.0;
    long long cost_count = 0;
    for (int y = 0; y < snapshot.depth.rows; ++y) {
        for (int x = 0; x < snapshot.depth.cols; ++x) {
            const float depth = snapshot.depth.at<float>(y, x);
            if (depth > 0.0f && std::isfinite(depth)) { ++valid; depth_sum += depth; }
            if (!snapshot.reliability.empty()) {
                const unsigned char state = snapshot.reliability.at<unsigned char>(y, x);
                if (state == STRONG) ++strong;
                else if (state == WEAK) ++weak;
                else ++unknown;
            }
            if (!snapshot.cost.empty()) {
                const float cost = snapshot.cost.at<float>(y, x);
                if (std::isfinite(cost)) { cost_sum += cost; ++cost_count; }
            }
        }
    }
    out << StageDirectoryName(snapshot.stage, snapshot.inner_iteration) << ','
        << snapshot.inner_iteration << ',' << snapshot.depth.total() << ','
        << valid << ',' << updated << ',' << strong << ',' << weak << ',' << unknown << ','
        << (valid ? depth_sum / valid : 0.0) << ','
        << (cost_count ? cost_sum / cost_count : 0.0) << '\n';
}

void FileDiagnosticSink::WriteUpdateEvents(const Problem& problem,
                                           const StageSnapshot& snapshot,
                                           const cv::Mat& update_mask) {
    if (update_mask.empty()) return;

    const auto path = PassRoot(problem) / "update_events.csv";
    const bool create_header = !boost::filesystem::exists(path);
    std::ofstream out(path.string(), std::ios::app);
    if (create_header) {
        out << "stage,inner_iteration,x,y,before_depth,after_depth,before_cost,after_cost,"
               "reliability,selected_view_mask,adaptive_radius,anchor_count\n";
    }

    for (int y = 0; y < update_mask.rows; ++y) {
        for (int x = 0; x < update_mask.cols; ++x) {
            if (update_mask.at<unsigned char>(y, x) == 0) continue;

            const float before_cost = previous_cost_.empty()
                                          ? 0.0f
                                          : previous_cost_.at<float>(y, x);
            const float after_cost = snapshot.cost.empty()
                                         ? 0.0f
                                         : snapshot.cost.at<float>(y, x);
            const int reliability = snapshot.reliability.empty()
                                        ? -1
                                        : snapshot.reliability.at<unsigned char>(y, x);
            const int selected_view_mask = snapshot.selected_views.empty()
                                               ? -1
                                               : snapshot.selected_views.at<int>(y, x);
            const int adaptive_radius = snapshot.adaptive_radius.empty()
                                            ? -1
                                            : snapshot.adaptive_radius.at<int>(y, x);
            const int anchor_count = snapshot.anchor_count.empty()
                                         ? -1
                                         : snapshot.anchor_count.at<unsigned char>(y, x);

            out << StageDirectoryName(snapshot.stage, snapshot.inner_iteration) << ','
                << snapshot.inner_iteration << ',' << x << ',' << y << ','
                << previous_depth_.at<float>(y, x) << ',' << snapshot.depth.at<float>(y, x) << ','
                << before_cost << ',' << after_cost << ',' << reliability << ','
                << selected_view_mask << ',' << adaptive_radius << ',' << anchor_count << '\n';
        }
    }
}

void FileDiagnosticSink::RecordFinalState(const Problem& problem, const FrameState& state) {
    // One final state per pyramid level is enough to reconstruct pyramid evolution.
    if (problem.iteration % 4 != 3) return;
    const auto directory = PassRoot(problem) / "final_state";
    boost::filesystem::create_directories(directory);
    WriteMat(directory, "depth", state.depth);
    WriteMat(directory, "normal", state.normal);
    WriteMat(directory, "reliability", state.reliability);
    WriteMat(directory, "selected_views", state.selected_views);

    const auto summary_path = ScaleViewRoot(problem.scale_size, problem.ref_image_id) / "pass_summary.csv";
    const bool create_header = !boost::filesystem::exists(summary_path);
    std::ofstream out(summary_path.string(), std::ios::app);
    if (create_header) out << "pass,state,scale,valid_depth,strong,weak,unknown\n";
    long long valid = 0, strong = 0, weak = 0, unknown = 0;
    for (int y = 0; y < state.depth.rows; ++y) {
        for (int x = 0; x < state.depth.cols; ++x) {
            if (state.depth.at<float>(y, x) > 0.0f) ++valid;
            const unsigned char value = state.reliability.at<unsigned char>(y, x);
            if (value == STRONG) ++strong;
            else if (value == WEAK) ++weak;
            else ++unknown;
        }
    }
    out << problem.iteration << ',' << RunStateName(problem.params.state) << ','
        << problem.scale_size << ',' << valid << ',' << strong << ',' << weak << ',' << unknown << '\n';
}

void FileDiagnosticSink::RecordFusionView(const FusionViewSnapshot& snapshot) {
    const auto directory = root_ / "fusion" / ("ref_" + FormatIndex(snapshot.ref_image_id));
    boost::filesystem::create_directories(directory);
    WriteMat(directory, "fate", snapshot.fate);
    WriteMat(directory, "support_count", snapshot.support_count);

    if (!snapshot.fate.empty()) {
        static const std::array<cv::Vec3b, 8> colors = {{
            {70, 190, 70}, {0, 0, 0}, {160, 160, 160}, {220, 120, 40},
            {40, 80, 220}, {180, 60, 180}, {40, 200, 220}, {100, 100, 100},
        }};
        cv::Mat preview(snapshot.fate.size(), CV_8UC3);
        for (int y = 0; y < snapshot.fate.rows; ++y)
            for (int x = 0; x < snapshot.fate.cols; ++x)
                preview.at<cv::Vec3b>(y, x) = colors[std::min<int>(7, snapshot.fate.at<unsigned char>(y, x))];
        cv::imwrite((directory / "fate.png").string(), preview);
    }
    if (!snapshot.support_count.empty()) {
        double maximum = 0.0;
        cv::minMaxLoc(snapshot.support_count, nullptr, &maximum);
        cv::Mat preview;
        snapshot.support_count.convertTo(preview, CV_8U, maximum > 0.0 ? 255.0 / maximum : 0.0);
        cv::imwrite((directory / "support_count.png").string(), preview);
    }
    WriteFusionSummary(snapshot);
}

void FileDiagnosticSink::WriteFusionSummary(const FusionViewSnapshot& snapshot) {
    const auto path = root_ / "fusion" / "fusion_summary.csv";
    const bool create_header = !boost::filesystem::exists(path);
    std::ofstream out(path.string(), std::ios::app);
    if (create_header) {
        out << "ref,accept,invalid_reference_depth,no_source_support,reprojection_fail,"
               "depth_consistency_fail,normal_consistency_fail,score_fail,duplicate\n";
    }
    std::array<long long, 8> counts{};
    for (int y = 0; y < snapshot.fate.rows; ++y)
        for (int x = 0; x < snapshot.fate.cols; ++x)
            ++counts[std::min<int>(7, snapshot.fate.at<unsigned char>(y, x))];
    out << snapshot.ref_image_id;
    for (long long count : counts) out << ',' << count;
    out << '\n';
}

void FileDiagnosticSink::RecordPointCloud(const std::vector<DiagnosticPoint>& points) {
    const auto directory = root_ / "fusion";
    boost::filesystem::create_directories(directory);
    WritePly(directory / "point_rgb.ply", points, [](const DiagnosticPoint& p) {
        return cv::Vec3b(p.blue, p.green, p.red);
    });
    WritePly(directory / "point_support.ply", points, [](const DiagnosticPoint& p) {
        const int value = std::min(255, 40 + p.support_count * 45);
        return cv::Vec3b(255 - value / 2, value, 40);
    });
    WritePly(directory / "point_reliability.ply", points, [](const DiagnosticPoint& p) {
        if (p.reliability == STRONG) return cv::Vec3b(80, 210, 80);
        if (p.reliability == WEAK) return cv::Vec3b(80, 80, 230);
        return cv::Vec3b(100, 100, 100);
    });
    WritePly(directory / "point_reference.ply", points, [](const DiagnosticPoint& p) {
        return IndexColor(p.ref_image_id);
    });
}

}  // namespace dpe
