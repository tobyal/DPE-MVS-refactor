#include "preprocessing/edge_detection.h"

#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace dpe {

static cv::Mat ResizeNearestInt(const cv::Mat& src, cv::Size size) {
    if (src.size() == size) return src.clone();
    cv::Mat dst(size, CV_32S);
    for (int y = 0; y < size.height; ++y) {
        const int sy = std::min(src.rows - 1, static_cast<int>((y + 0.5) * src.rows / static_cast<double>(size.height)));
        for (int x = 0; x < size.width; ++x) {
            const int sx = std::min(src.cols - 1, static_cast<int>((x + 0.5) * src.cols / static_cast<double>(size.width)));
            dst.at<int>(y, x) = src.at<int>(sy, sx);
        }
    }
    return dst;
}

cv::Mat RobertsGradient(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8U);
    cv::Mat dst(src.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y + 1 < src.rows; ++y) {
        for (int x = 0; x + 1 < src.cols; ++x) {
            const int a = src.at<unsigned char>(y, x);
            const int b = src.at<unsigned char>(y + 1, x + 1);
            const int c = src.at<unsigned char>(y + 1, x);
            const int d = src.at<unsigned char>(y, x + 1);
            const int g1 = a - b;
            const int g2 = c - d;
            dst.at<unsigned char>(y, x) = static_cast<unsigned char>(
                std::min(255.0, std::sqrt(static_cast<double>(g1 * g1 + g2 * g2))));
        }
    }
    return dst;
}

static cv::Mat ConnectComponents(const cv::Mat& edge_binary, std::vector<int>* counts) {
    CV_Assert(edge_binary.type() == CV_8U);
    cv::Mat free_mask;
    cv::compare(edge_binary, 0, free_mask, cv::CMP_EQ);
    cv::Mat labels;
    const int n = cv::connectedComponents(free_mask, labels, 4, CV_32S);
    if (counts) {
        counts->assign(n, 0);
        for (int y = 0; y < labels.rows; ++y)
            for (int x = 0; x < labels.cols; ++x)
                ++(*counts)[labels.at<int>(y, x)];
    }
    // Edge itself is label 0 in the original semantics.
    labels.setTo(0, edge_binary != 0);
    return labels;
}

cv::Mat ConnectedRegionLabels(const cv::Mat& binary, int weak_tex_num) {
    std::vector<int> counts;
    cv::Mat labels = ConnectComponents(binary, &counts);
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            const int l = labels.at<int>(y, x);
            if (l != 0 && l < static_cast<int>(counts.size()) && counts[l] <= weak_tex_num)
                labels.at<int>(y, x) = -1;
        }
    }
    return labels;
}

cv::Mat ComputeFineEdges(const cv::Mat& image_u8) {
    CV_Assert(image_u8.type() == CV_8U);
    int hist[256] = {0};
    for (int y = 0; y < image_u8.rows; ++y) {
        const unsigned char* p = image_u8.ptr<unsigned char>(y);
        for (int x = 0; x < image_u8.cols; ++x) ++hist[p[x]];
    }
    const int half = image_u8.rows * image_u8.cols / 2;
    int acc = 0, median = 0;
    for (int i = 0; i < 256; ++i) {
        acc += hist[i];
        if (acc > half) { median = i; break; }
    }
    // Keep the official-code threshold behavior: [0.33*median, median].
    const double low = (1.0 - 0.67) * median;
    const double high = median;
    cv::Mat edge;
    cv::Canny(image_u8, edge, low, high, 3, true);
    cv::threshold(edge, edge, 4, 255, cv::THRESH_BINARY);
    return edge;
}

cv::Mat ComputeCoarseRegions(const cv::Mat& full_res_u8, int scale_log2, bool high_res_img) {
    CV_Assert(full_res_u8.type() == CV_8U);
    cv::Mat src;
    if (high_res_img) {
        cv::resize(full_res_u8, src,
                   cv::Size(std::max(1, full_res_u8.cols / 2), std::max(1, full_res_u8.rows / 2)),
                   0, 0, cv::INTER_LINEAR);
    } else {
        src = full_res_u8.clone();
    }
    cv::resize(src, src, cv::Size(std::max(1, src.cols / 2), std::max(1, src.rows / 2)),
               0, 0, cv::INTER_LINEAR);

    const int rob_thr = high_res_img ? 4 : 6;
    cv::Mat edge = RobertsGradient(src);
    cv::threshold(edge, edge, rob_thr, 255, cv::THRESH_BINARY);

    const int weak_tex_num = std::max(1,
        static_cast<int>(static_cast<double>(full_res_u8.rows) * full_res_u8.cols /
                         static_cast<double>(1024 << (2 * scale_log2))));

    // Reproduce the coarse-edge construction in the official implementation:
    // find large free-space components, extract their outer one-pixel boundary,
    // then regularize that boundary with Hough line segments.
    std::vector<int> counts;
    cv::Mat initial_labels = ConnectComponents(edge, &counts);
    const int line_scale = std::max(1, std::min(src.cols, src.rows) / 30);
    for (int label = 1; label < static_cast<int>(counts.size()); ++label) {
        if (counts[label] < weak_tex_num) continue;
        cv::Mat boundary(edge.size(), CV_8U, cv::Scalar(0));
        for (int y = 0; y < boundary.rows; ++y) {
            for (int x = 0; x < boundary.cols; ++x) {
                if (initial_labels.at<int>(y, x) == label) continue;
                bool adjacent = false;
                if (x > 0 && initial_labels.at<int>(y, x - 1) == label) adjacent = true;
                if (x + 1 < boundary.cols && initial_labels.at<int>(y, x + 1) == label) adjacent = true;
                if (y > 0 && initial_labels.at<int>(y - 1, x) == label) adjacent = true;
                if (y + 1 < boundary.rows && initial_labels.at<int>(y + 1, x) == label) adjacent = true;
                if (adjacent) boundary.at<unsigned char>(y, x) = 255;
            }
        }
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(boundary, lines, 1, CV_PI / 180.0,
                        line_scale, line_scale, line_scale);
        for (const auto& l : lines)
            cv::line(edge, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255), 1);
    }

    const float factor = 1.0f / static_cast<float>(1 << scale_log2);
    const cv::Size target(std::max(1, static_cast<int>(std::round(full_res_u8.cols * factor))),
                          std::max(1, static_cast<int>(std::round(full_res_u8.rows * factor))));
    cv::resize(edge, edge, target, 0, 0, cv::INTER_LINEAR);
    cv::threshold(edge, edge, rob_thr, 255, cv::THRESH_BINARY);

    // The original implementation propagates a free-space border pixel to the
    // outer image border before connected-component labeling.
    if (edge.cols >= 2) {
        for (int y = 0; y < edge.rows; ++y) {
            if (edge.at<unsigned char>(y, 1) == 0) edge.at<unsigned char>(y, 0) = 0;
            if (edge.at<unsigned char>(y, edge.cols - 2) == 0) edge.at<unsigned char>(y, edge.cols - 1) = 0;
        }
    }
    if (edge.rows >= 2) {
        for (int x = 0; x < edge.cols; ++x) {
            if (edge.at<unsigned char>(1, x) == 0) edge.at<unsigned char>(0, x) = 0;
            if (edge.at<unsigned char>(edge.rows - 2, x) == 0) edge.at<unsigned char>(edge.rows - 1, x) = 0;
        }
    }

    return ConnectedRegionLabels(edge, weak_tex_num);
}

EdgeGuidanceHost BuildEdgeGuidance(const cv::Mat& full_res_u8,
                                   const cv::Mat& scaled_u8,
                                   int scale_log2,
                                   const DPEParams& params) {
    EdgeGuidanceHost out;
    if (params.use_edge || params.use_limit) {
        out.fine_edges = ComputeFineEdges(scaled_u8);
    } else {
        out.fine_edges = cv::Mat(scaled_u8.size(), CV_8U, cv::Scalar(0));
    }
    if (params.use_label) {
        out.coarse_regions = ComputeCoarseRegions(full_res_u8, scale_log2, params.high_res_img);
        if (out.coarse_regions.size() != scaled_u8.size())
            out.coarse_regions = ResizeNearestInt(out.coarse_regions, scaled_u8.size());
    } else {
        out.coarse_regions = cv::Mat(scaled_u8.size(), CV_32S, cv::Scalar(0));
    }
    return out;
}

}  // namespace dpe
