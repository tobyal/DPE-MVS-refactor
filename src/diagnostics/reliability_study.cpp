#include "diagnostics/reliability_study.h"

#include "common/cuda_utils.cuh"
#include "common/io.h"

#include <boost/filesystem.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dpe {
namespace {

std::string PaddedNumber(char prefix, int value) {
    std::ostringstream out;
    out << prefix << std::setfill('0') << std::setw(2) << value;
    return out.str();
}

std::string StageSuffix(const ReliabilityStudyStage& stage) {
    if (stage.run_state == RunState::FirstInit) return "coarse_init";
    if (stage.run_state == RunState::RefineInit) return "dpe_init";
    return "refine" + std::to_string(stage.outer_refine);
}

void RequireMap(const cv::Mat& map, const cv::Size& expected, int type,
                const char* name) {
    if (map.size() != expected || map.type() != type)
        throw std::runtime_error(std::string("Incomplete reliability-study map: ") + name);
}

void RequireWrite(bool success, const boost::filesystem::path& path) {
    if (!success) throw std::runtime_error("Cannot write reliability-study file: " + path.string());
}

}  // namespace

ReliabilityStudyCapture::ReliabilityStudyCapture(
    int width, int height, const float4* device_planes,
    const unsigned char* device_reliability)
    : width_(width), height_(height), device_planes_(device_planes),
      device_reliability_(device_reliability) {}

void ReliabilityStudyCapture::CaptureGeometry(
    const float4* device_planes, cudaStream_t stream,
    cv::Mat* depth, cv::Mat* normal) {
    const size_t count = static_cast<size_t>(width_) * height_;
    std::vector<float4> planes(count);
    DPE_CUDA_CHECK(cudaMemcpyAsync(planes.data(), device_planes,
                                   sizeof(float4) * count,
                                   cudaMemcpyDeviceToHost, stream));
    DPE_CUDA_CHECK(cudaStreamSynchronize(stream));

    depth->create(height_, width_, CV_32F);
    normal->create(height_, width_, CV_32FC3);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const float4 plane = planes[static_cast<size_t>(y) * width_ + x];
            depth->at<float>(y, x) = plane.w;
            normal->at<cv::Vec3f>(y, x) = cv::Vec3f(plane.x, plane.y, plane.z);
        }
    }
}

void ReliabilityStudyCapture::CapturePre(cudaStream_t stream) {
    CaptureGeometry(device_planes_, stream, &snapshot_.depth_pre, &snapshot_.normal_pre);
}

void ReliabilityStudyCapture::CaptureReliability(cudaStream_t stream) {
    snapshot_.reliability.create(height_, width_, CV_8U);
    const size_t count = static_cast<size_t>(width_) * height_;
    DPE_CUDA_CHECK(cudaMemcpyAsync(snapshot_.reliability.data, device_reliability_,
                                   count, cudaMemcpyDeviceToHost, stream));
    DPE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void ReliabilityStudyCapture::CapturePost(cudaStream_t stream) {
    CaptureGeometry(device_planes_, stream, &snapshot_.depth_post, &snapshot_.normal_post);
}

ReliabilityStudyWriter::ReliabilityStudyWriter(boost::filesystem::path root)
    : root_(std::move(root)) {
    if (root_.empty()) throw std::runtime_error("Reliability-study output root is empty");
    boost::filesystem::create_directories(root_);
}

void ReliabilityStudyWriter::Write(
    const ReliabilityStudyStage& stage,
    const ReliabilityStudySnapshot& snapshot) const {
    const cv::Size size = snapshot.depth_pre.size();
    if (size.empty()) throw std::runtime_error("Empty reliability-study snapshot");
    RequireMap(snapshot.depth_pre, size, CV_32F, "depth_pre");
    RequireMap(snapshot.normal_pre, size, CV_32FC3, "normal_pre");
    RequireMap(snapshot.reliability, size, CV_8U, "reliability");
    RequireMap(snapshot.depth_post, size, CV_32F, "depth_post");
    RequireMap(snapshot.normal_post, size, CV_32FC3, "normal_post");

    const boost::filesystem::path directory =
        root_ / ("ref_" + FormatIndex(stage.ref_image_id)) /
        ReliabilityStageDirectoryName(stage);
    boost::filesystem::create_directories(directory);

    const boost::filesystem::path depth_pre = directory / "depth_pre.dmb";
    const boost::filesystem::path normal_pre = directory / "normal_pre.dmb";
    const boost::filesystem::path reliability = directory / "reliability.bin";
    const boost::filesystem::path depth_post = directory / "depth_post.dmb";
    const boost::filesystem::path normal_post = directory / "normal_post.dmb";
    RequireWrite(WriteBinMat(depth_pre, snapshot.depth_pre), depth_pre);
    RequireWrite(WriteBinMat(normal_pre, snapshot.normal_pre), normal_pre);
    RequireWrite(WriteBinMat(reliability, snapshot.reliability), reliability);
    RequireWrite(WriteBinMat(depth_post, snapshot.depth_post), depth_post);
    RequireWrite(WriteBinMat(normal_post, snapshot.normal_post), normal_post);

    WriteDepthPreview(directory / "depth_pre.png", snapshot.depth_pre,
                      stage.depth_min, stage.depth_max);
    WriteNormalPreview(directory / "normal_pre.png", snapshot.normal_pre);
    WriteReliabilityPreview(directory / "reliability.png", snapshot.reliability);
    WriteDepthPreview(directory / "depth_post.png", snapshot.depth_post,
                      stage.depth_min, stage.depth_max);
    WriteNormalPreview(directory / "normal_post.png", snapshot.normal_post);

    std::ofstream metadata((directory / "stage.json").string());
    if (!metadata) throw std::runtime_error("Cannot write reliability-study stage metadata");
    metadata << std::boolalpha
             << "{\n"
             << "  \"stage_id\": \"" << ReliabilityStageId(stage.global_iteration) << "\",\n"
             << "  \"global_iteration\": " << stage.global_iteration << ",\n"
             << "  \"pyramid_level\": " << stage.pyramid_level << ",\n"
             << "  \"scale\": " << stage.scale << ",\n"
             << "  \"width\": " << size.width << ",\n"
             << "  \"height\": " << size.height << ",\n"
             << "  \"run_state\": \"" << ReliabilityRunStateName(stage.run_state) << "\",\n"
             << "  \"outer_refine\": " << stage.outer_refine << ",\n"
             << "  \"ref_image_id\": " << stage.ref_image_id << ",\n"
             << "  \"depth_min\": " << stage.depth_min << ",\n"
             << "  \"depth_max\": " << stage.depth_max << ",\n"
             << "  \"use_apd\": " << stage.use_apd << ",\n"
             << "  \"use_edge\": " << stage.use_edge << ",\n"
             << "  \"geom_consistency\": " << stage.geom_consistency << ",\n"
             << "  \"normal_coordinates\": \"world\"\n"
             << "}\n";
}

std::string ReliabilityStageId(int global_iteration) {
    return PaddedNumber('S', global_iteration);
}

std::string ReliabilityStageDirectoryName(const ReliabilityStudyStage& stage) {
    return ReliabilityStageId(stage.global_iteration) + "_" +
           PaddedNumber('L', stage.pyramid_level) + "_" + StageSuffix(stage);
}

const char* ReliabilityRunStateName(RunState state) {
    if (state == RunState::FirstInit) return "coarse_init";
    if (state == RunState::RefineInit) return "dpe_init";
    return "refine_iter";
}

}  // namespace dpe
