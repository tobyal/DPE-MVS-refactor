#include "common/io.h"

#include <boost/filesystem/fstream.hpp>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>

namespace dpe {

std::string FormatIndex(int index) {
    std::ostringstream os;
    os << std::setw(8) << std::setfill('0') << index;
    return os.str();
}

bool ReadCamera(const boost::filesystem::path& path, Camera& camera) {
    std::ifstream in(path.string());
    if (!in) return false;
    std::string token;
    in >> token;  // extrinsic
    for (int r = 0; r < 3; ++r) {
        in >> camera.R[3*r+0] >> camera.R[3*r+1] >> camera.R[3*r+2] >> camera.t[r];
    }
    float bottom[4];
    in >> bottom[0] >> bottom[1] >> bottom[2] >> bottom[3];
    in >> token;  // intrinsic
    for (int r = 0; r < 3; ++r)
        in >> camera.K[3*r+0] >> camera.K[3*r+1] >> camera.K[3*r+2];
    float interval = 0.0f, depth_num = 0.0f;
    in >> camera.depth_min >> interval >> depth_num >> camera.depth_max;

    camera.c[0] = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    camera.c[1] = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    camera.c[2] = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    return true;
}

bool ReadBinMat(const boost::filesystem::path& path, cv::Mat& mat) {
    std::ifstream in(path.string(), std::ios::binary);
    if (!in) return false;
    int version = 0, rows = 0, cols = 0, type = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(int));
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));
    in.read(reinterpret_cast<char*>(&type), sizeof(int));
    if (!in || rows <= 0 || cols <= 0) return false;
    mat.create(rows, cols, type);
    const size_t row_bytes = static_cast<size_t>(cols) * mat.elemSize();
    for (int r = 0; r < rows; ++r) {
        in.read(reinterpret_cast<char*>(mat.ptr(r)), static_cast<std::streamsize>(row_bytes));
        if (!in) return false;
    }
    return true;
}

bool WriteBinMat(const boost::filesystem::path& path, const cv::Mat& mat) {
    std::ofstream out(path.string(), std::ios::binary);
    if (!out) return false;
    const int version = 1;
    const int rows = mat.rows;
    const int cols = mat.cols;
    const int type = mat.type();
    out.write(reinterpret_cast<const char*>(&version), sizeof(int));
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(&type), sizeof(int));
    const size_t row_bytes = static_cast<size_t>(cols) * mat.elemSize();
    for (int r = 0; r < rows; ++r) {
        out.write(reinterpret_cast<const char*>(mat.ptr(r)), static_cast<std::streamsize>(row_bytes));
        if (!out) return false;
    }
    return true;
}

void WriteDepthPreview(const boost::filesystem::path& path, const cv::Mat& depth,
                       float depth_min, float depth_max) {
    cv::Mat clipped = depth.clone();
    cv::Mat vis(depth.size(), CV_8U, cv::Scalar(0));
    const float denom = std::max(1e-6f, depth_max - depth_min);
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            const float d = depth.at<float>(y, x);
            if (d <= 0) continue;
            const float v = std::max(0.0f, std::min(1.0f, (d - depth_min) / denom));
            vis.at<unsigned char>(y, x) = static_cast<unsigned char>(255.0f * v);
        }
    }
    cv::imwrite(path.string(), vis);
}

void WriteNormalPreview(const boost::filesystem::path& path, const cv::Mat& normal) {
    cv::Mat vis(normal.size(), CV_8UC3);
    for (int y = 0; y < normal.rows; ++y) {
        for (int x = 0; x < normal.cols; ++x) {
            const cv::Vec3f n = normal.at<cv::Vec3f>(y, x);
            vis.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<unsigned char>(127.5f * (n[2] + 1.0f)),
                static_cast<unsigned char>(127.5f * (n[1] + 1.0f)),
                static_cast<unsigned char>(127.5f * (n[0] + 1.0f)));
        }
    }
    cv::imwrite(path.string(), vis);
}

void WriteReliabilityPreview(const boost::filesystem::path& path, const cv::Mat& reliability) {
    cv::Mat vis(reliability.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < reliability.rows; ++y) {
        for (int x = 0; x < reliability.cols; ++x) {
            const auto s = reliability.at<unsigned char>(y, x);
            if (s == STRONG) vis.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
            else if (s == WEAK) vis.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
            else vis.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
        }
    }
    cv::imwrite(path.string(), vis);
}

}  // namespace dpe
