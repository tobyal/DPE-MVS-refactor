#include "dpe/dpe.h"
#include "common/cuda_utils.cuh"

#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace dpe {

namespace {

template <typename T>
cv::Mat ResizeNearestTyped(const cv::Mat& src, const cv::Size& size, int type) {
    if (src.size() == size) return src.clone();
    cv::Mat dst(size, type);
    for (int y = 0; y < size.height; ++y) {
        const int sy = std::min(src.rows - 1, static_cast<int>((y + 0.5) * src.rows / static_cast<double>(size.height)));
        for (int x = 0; x < size.width; ++x) {
            const int sx = std::min(src.cols - 1, static_cast<int>((x + 0.5) * src.cols / static_cast<double>(size.width)));
            dst.at<T>(y, x) = src.at<T>(sy, sx);
        }
    }
    return dst;
}

}  // namespace

DPESolver::DPESolver(const Problem& problem,
                     const SceneView& view,
                     const EdgeGuidanceHost& guidance,
                     ReconstructionState& reconstruction,
                     CudaContext& cuda,
                     DiagnosticSink* diagnostics)
    : problem_(problem), view_(view), guidance_(guidance), reconstruction_(reconstruction), cuda_(cuda),
      diagnostics_(diagnostics),
      params_(problem.params), width_(view.width), height_(view.height) {
    params_.num_images = static_cast<int>(view.image_ids.size());
    params_.depth_min = view.cameras.front().depth_min * 0.6f;
    params_.depth_max = view.cameras.front().depth_max * 1.2f;
}

DPESolver::~DPESolver() { ReleaseDevice(); }

void DPESolver::PrepareHostState() {
    const size_t count = static_cast<size_t>(width_) * height_;
    host_planes_.assign(count, make_float4(0, 0, 0, 0));
    host_reliability_ = cv::Mat(height_, width_, CV_8U, cv::Scalar(STRONG));
    host_selected_views_ = cv::Mat(height_, width_, CV_32S, cv::Scalar(0));

    const FrameState* previous = reconstruction_.Find(problem_.ref_image_id);
    if (params_.state != RunState::FirstInit && previous && !previous->depth.empty()) {
        cv::Mat depth, normal, reliability, views;
        depth = ResizeNearestTyped<float>(previous->depth, cv::Size(width_, height_), CV_32F);
        normal = ResizeNearestTyped<cv::Vec3f>(previous->normal, cv::Size(width_, height_), CV_32FC3);
        if (params_.use_apd && !previous->reliability.empty())
            reliability = ResizeNearestTyped<unsigned char>(previous->reliability, cv::Size(width_, height_), CV_8U);
        else reliability = cv::Mat(height_, width_, CV_8U, cv::Scalar(STRONG));
        if (!previous->selected_views.empty())
            views = ResizeNearestTyped<int>(previous->selected_views, cv::Size(width_, height_), CV_32S);
        else views = cv::Mat(height_, width_, CV_32S, cv::Scalar(0));

        host_reliability_ = reliability.clone();
        host_selected_views_ = views.clone();
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                const cv::Vec3f n = normal.at<cv::Vec3f>(y, x);
                host_planes_[static_cast<size_t>(y) * width_ + x] =
                    make_float4(n[0], n[1], n[2], depth.at<float>(y, x));
            }
        }
    }

    if (!params_.use_apd) host_reliability_.setTo(STRONG);

    host_anchor_map_ = cv::Mat(height_, width_, CV_32S, cv::Scalar(-1));
    weak_count_ = 0;
    for (int y = 0; y < height_; ++y)
        for (int x = 0; x < width_; ++x)
            if (host_reliability_.at<unsigned char>(y, x) == WEAK)
                host_anchor_map_.at<int>(y, x) = weak_count_++;

    if (!guidance_.lowres_edges.empty()) {
        lowres_edges_ = guidance_.lowres_edges;
    } else if (!guidance_.fine_edges.empty()) {
        lowres_edges_ = guidance_.fine_edges;
    } else {
        lowres_edges_ = cv::Mat(height_, width_, CV_8U, cv::Scalar(0));
    }
}

void DPESolver::AllocateAndUpload() {
    const size_t count = static_cast<size_t>(width_) * height_;
    const auto stream = cuda_.Stream();
    gpu_scene_.reset(new GpuScene(cuda_, view_, reconstruction_, params_.geom_consistency));

    host_gpu_.width = width_;
    host_gpu_.height = height_;
    host_gpu_.ref_id = problem_.ref_image_id;
    host_gpu_.num_images = params_.num_images;
    host_gpu_.weak_count = weak_count_;
    host_gpu_.cameras = gpu_scene_->Data().cameras;
    host_gpu_.image_textures = gpu_scene_->Data().image_textures;
    host_gpu_.depth_textures = gpu_scene_->Data().depth_textures;

    host_gpu_.state.planes = Allocate<float4>(count);
    host_gpu_.state.fitted_planes = Allocate<float4>(count);
    host_gpu_.state.costs = Allocate<float>(count);
    host_gpu_.state.reliability = Allocate<unsigned char>(count);
    host_gpu_.state.weak_reliable = Allocate<unsigned char>(count);
    host_gpu_.state.selected_views = Allocate<unsigned int>(count);
    host_gpu_.state.view_weights = Allocate<unsigned char>(count * kMaxImages);
    host_gpu_.state.anchor_map = Allocate<int>(count);
    host_gpu_.state.nearest_strong = Allocate<short2>(count);
    host_gpu_.state.radius = Allocate<int>(count);
    host_gpu_.state.random_states = Allocate<curandState>(count);
    host_gpu_.state.anchors = weak_count_ > 0 ? Allocate<short2>(static_cast<size_t>(weak_count_) * kNeighbourNum) : nullptr;

    host_gpu_.guidance.fine_edges = Allocate<unsigned char>(count);
    host_gpu_.guidance.low_width = lowres_edges_.cols;
    host_gpu_.guidance.low_height = lowres_edges_.rows;
    host_gpu_.guidance.lowres_edges = Allocate<unsigned char>(lowres_edges_.total());
    host_gpu_.guidance.region_labels = Allocate<int>(count);
    host_gpu_.guidance.nearest_edges = Allocate<short2>(count * 8);
    host_gpu_.guidance.texture_complexity = Allocate<float>(count);
    host_gpu_.guidance.region_boundaries = weak_count_ > 0 ? Allocate<short2>(static_cast<size_t>(weak_count_) * 8) : nullptr;

    device_params_ = Allocate<DPEParams>(1);
    device_gpu_ = Allocate<DPEGpuContext>(1);
    host_gpu_.params = device_params_;

    DPE_CUDA_CHECK(cudaMemcpyAsync(host_gpu_.state.planes, host_planes_.data(), sizeof(float4) * count, cudaMemcpyHostToDevice, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(host_gpu_.state.reliability, host_reliability_.data, count, cudaMemcpyHostToDevice, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(host_gpu_.state.selected_views, host_selected_views_.data, sizeof(unsigned int) * count, cudaMemcpyHostToDevice, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(host_gpu_.state.anchor_map, host_anchor_map_.data, sizeof(int) * count, cudaMemcpyHostToDevice, stream));
    DPE_CUDA_CHECK(cudaMemsetAsync(host_gpu_.state.fitted_planes, 0, sizeof(float4) * count, stream));
    DPE_CUDA_CHECK(cudaMemsetAsync(host_gpu_.state.costs, 0, sizeof(float) * count, stream));
    DPE_CUDA_CHECK(cudaMemsetAsync(host_gpu_.state.weak_reliable, 0, count, stream));
    DPE_CUDA_CHECK(cudaMemsetAsync(host_gpu_.state.view_weights, 0, count * kMaxImages, stream));
    DPE_CUDA_CHECK(cudaMemsetAsync(host_gpu_.state.radius, 0, sizeof(int) * count, stream));
    if (host_gpu_.state.anchors)
        DPE_CUDA_CHECK(cudaMemsetAsync(host_gpu_.state.anchors, 0xff, sizeof(short2) * weak_count_ * kNeighbourNum, stream));

    cv::Mat fine = guidance_.fine_edges.empty() ? cv::Mat(height_, width_, CV_8U, cv::Scalar(0)) : guidance_.fine_edges;
    cv::Mat labels = guidance_.coarse_regions.empty() ? cv::Mat(height_, width_, CV_32S, cv::Scalar(0)) : guidance_.coarse_regions;
    if (fine.size() != cv::Size(width_, height_)) cv::resize(fine, fine, cv::Size(width_, height_), 0, 0, cv::INTER_NEAREST);
    if (labels.size() != cv::Size(width_, height_))
        labels = ResizeNearestTyped<int>(labels, cv::Size(width_, height_), CV_32S);
    DPE_CUDA_CHECK(cudaMemcpyAsync(host_gpu_.guidance.fine_edges, fine.data, count, cudaMemcpyHostToDevice, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(host_gpu_.guidance.lowres_edges, lowres_edges_.data, lowres_edges_.total(), cudaMemcpyHostToDevice, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(host_gpu_.guidance.region_labels, labels.data, sizeof(int) * count, cudaMemcpyHostToDevice, stream));
    if (host_gpu_.guidance.region_boundaries)
        DPE_CUDA_CHECK(cudaMemsetAsync(host_gpu_.guidance.region_boundaries, 0xff, sizeof(short2) * weak_count_ * 8, stream));

    DPE_CUDA_CHECK(cudaMemcpyAsync(device_params_, &params_, sizeof(DPEParams), cudaMemcpyHostToDevice, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(device_gpu_, &host_gpu_, sizeof(DPEGpuContext), cudaMemcpyHostToDevice, stream));
}

FrameState DPESolver::DownloadResult() {
    const size_t count = static_cast<size_t>(width_) * height_;
    std::vector<float4> planes(count);
    cv::Mat reliability(height_, width_, CV_8U);
    cv::Mat views(height_, width_, CV_32S);
    const auto stream = cuda_.Stream();
    DPE_CUDA_CHECK(cudaMemcpyAsync(planes.data(), host_gpu_.state.planes, sizeof(float4) * count, cudaMemcpyDeviceToHost, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(reliability.data, host_gpu_.state.reliability, count, cudaMemcpyDeviceToHost, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(views.data, host_gpu_.state.selected_views, sizeof(unsigned int) * count, cudaMemcpyDeviceToHost, stream));
    cuda_.Synchronize();

    FrameState state;
    state.depth = cv::Mat(height_, width_, CV_32F);
    state.normal = cv::Mat(height_, width_, CV_32FC3);
    state.reliability = reliability;
    state.selected_views = views;
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const float4 p = planes[static_cast<size_t>(y) * width_ + x];
            float d = p.w;
            if (d < params_.depth_min || d > params_.depth_max || !std::isfinite(d)) {
                d = 0.0f;
                state.reliability.at<unsigned char>(y, x) = UNKNOWN;
            }
            state.depth.at<float>(y, x) = d;
            state.normal.at<cv::Vec3f>(y, x) = cv::Vec3f(p.x, p.y, p.z);
        }
    }
    return state;
}

void DPESolver::CaptureStage(DiagnosticStage stage, int inner_iteration) {
    if (!diagnostics_) return;

    const size_t count = static_cast<size_t>(width_) * height_;
    const bool fitted_plane_stage = stage == DiagnosticStage::Plane;
    const bool world_representation = stage == DiagnosticStage::Finalized ||
                                      stage == DiagnosticStage::Filtered ||
                                      stage == DiagnosticStage::Classified ||
                                      stage == DiagnosticStage::Refined;
    std::vector<float4> planes(count);
    cv::Mat reliability(height_, width_, CV_8U);
    cv::Mat cost(height_, width_, CV_32F);
    cv::Mat selected_views(height_, width_, CV_32S);
    cv::Mat radius(height_, width_, CV_32S);
    cv::Mat anchor_count(height_, width_, CV_8U, cv::Scalar(0));
    cv::Mat texture_complexity;

    const auto stream = cuda_.Stream();
    const float4* source_planes = fitted_plane_stage
        ? host_gpu_.state.fitted_planes
        : host_gpu_.state.planes;
    DPE_CUDA_CHECK(cudaMemcpyAsync(planes.data(), source_planes,
                                  sizeof(float4) * count, cudaMemcpyDeviceToHost, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(reliability.data, host_gpu_.state.reliability,
                                  count, cudaMemcpyDeviceToHost, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(cost.data, host_gpu_.state.costs,
                                  sizeof(float) * count, cudaMemcpyDeviceToHost, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(selected_views.data, host_gpu_.state.selected_views,
                                  sizeof(unsigned int) * count, cudaMemcpyDeviceToHost, stream));
    DPE_CUDA_CHECK(cudaMemcpyAsync(radius.data, host_gpu_.state.radius,
                                  sizeof(int) * count, cudaMemcpyDeviceToHost, stream));

    std::vector<short2> anchors;
    if (weak_count_ > 0 && host_gpu_.state.anchors) {
        anchors.resize(static_cast<size_t>(weak_count_) * kNeighbourNum);
        DPE_CUDA_CHECK(cudaMemcpyAsync(anchors.data(), host_gpu_.state.anchors,
                                      sizeof(short2) * anchors.size(),
                                      cudaMemcpyDeviceToHost, stream));
    }
    if (stage == DiagnosticStage::Input) {
        texture_complexity = cv::Mat(height_, width_, CV_32F);
        DPE_CUDA_CHECK(cudaMemcpyAsync(texture_complexity.data,
                                      host_gpu_.guidance.texture_complexity,
                                      sizeof(float) * count, cudaMemcpyDeviceToHost, stream));
    }
    cuda_.Synchronize();

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const int map = host_anchor_map_.at<int>(y, x);
            if (map < 0 || anchors.empty()) continue;
            unsigned char valid = 0;
            for (int index = 1; index < kNeighbourNum; ++index) {
                if (anchors[static_cast<size_t>(map) * kNeighbourNum + index].x >= 0) ++valid;
            }
            anchor_count.at<unsigned char>(y, x) = valid;
        }
    }

    StageSnapshot snapshot;
    snapshot.stage = stage;
    snapshot.inner_iteration = inner_iteration;
    snapshot.depth = cv::Mat(height_, width_, CV_32F, cv::Scalar(0));
    snapshot.normal = cv::Mat(height_, width_, CV_32FC3, cv::Scalar(0, 0, 0));
    snapshot.reliability = reliability;
    snapshot.cost = cost;
    snapshot.selected_views = selected_views;
    snapshot.adaptive_radius = radius;
    snapshot.anchor_count = anchor_count;
    snapshot.texture_complexity = texture_complexity;

    const Camera& camera = view_.cameras.front();
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const float4 plane = planes[static_cast<size_t>(y) * width_ + x];
            float depth = 0.0f;
            cv::Vec3f normal;
            if (world_representation) {
                depth = plane.w;
                normal = cv::Vec3f(plane.x, plane.y, plane.z);
            } else {
                const float denominator = (x - camera.K[2]) * plane.x +
                    (camera.K[0] / camera.K[4]) * (y - camera.K[5]) * plane.y +
                    camera.K[0] * plane.z;
                if (std::fabs(denominator) > 1e-12f) {
                    depth = -plane.w * camera.K[0] / denominator;
                }
                normal = cv::Vec3f(
                    camera.R[0] * plane.x + camera.R[3] * plane.y + camera.R[6] * plane.z,
                    camera.R[1] * plane.x + camera.R[4] * plane.y + camera.R[7] * plane.z,
                    camera.R[2] * plane.x + camera.R[5] * plane.y + camera.R[8] * plane.z);
            }
            if (!std::isfinite(depth) || depth < params_.depth_min || depth > params_.depth_max) {
                depth = 0.0f;
            }
            snapshot.depth.at<float>(y, x) = depth;
            snapshot.normal.at<cv::Vec3f>(y, x) = normal;
        }
    }
    diagnostics_->RecordStage(problem_, snapshot);
}

FrameState DPESolver::Run() {
    PrepareHostState();
    AllocateAndUpload();
    KernelStageCallback callback;
    if (diagnostics_ && diagnostics_->ShouldCaptureStages(problem_)) {
        callback = [this](DiagnosticStage stage, int inner_iteration) {
            CaptureStage(stage, inner_iteration);
        };
    }
    RunDpeKernels(device_gpu_, cuda_.Stream(), params_, width_, height_, callback);
    FrameState result = DownloadResult();
    return result;
}

void DPESolver::ReleaseDevice() {
    if (!allocations_.empty()) cuda_.Synchronize();
    for (void* p : allocations_) cuda_.Workspace().Release(p);
    allocations_.clear();
    device_gpu_ = nullptr;
    device_params_ = nullptr;
    gpu_scene_.reset();
}

}  // namespace dpe
