#include "runtime/gpu_scene.h"
#include "common/cuda_utils.cuh"

#include <opencv2/imgproc.hpp>
#include <cstring>

namespace dpe {

cudaTextureObject_t GpuScene::CreateTexture(cudaArray_t array) const {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = array;
    cudaTextureDesc tex{};
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.readMode = cudaReadModeElementType;
    tex.normalizedCoords = 0;
    cudaTextureObject_t object = 0;
    DPE_CUDA_CHECK(cudaCreateTextureObject(&object, &res, &tex, nullptr));
    return object;
}

GpuScene::GpuScene(CudaContext& cuda, const SceneView& view,
                   const ReconstructionState& state, bool upload_depths)
    : cuda_(cuda), view_(view) {
    auto& ws = cuda_.Workspace();
    const auto stream = cuda_.Stream();
    data_.num_images = static_cast<int>(view_.image_ids.size());
    data_.cameras = AcquireTyped<Camera>(ws, data_.num_images);
    DPE_CUDA_CHECK(cudaMemcpyAsync(data_.cameras, view_.cameras.data(),
                                  sizeof(Camera) * data_.num_images,
                                  cudaMemcpyHostToDevice, stream));

    for (int i = 0; i < data_.num_images; ++i) {
        const cv::Mat& image = *view_.images_float[i];
        cudaArray_t array = ws.AcquireArray(image.cols, image.rows);
        image_arrays_.push_back(array);
        DPE_CUDA_CHECK(cudaMemcpy2DToArrayAsync(array, 0, 0, image.ptr<float>(), image.step,
                                               image.cols * sizeof(float), image.rows,
                                               cudaMemcpyHostToDevice, stream));
        image_handles_.images[i] = CreateTexture(array);
    }
    data_.image_textures = AcquireTyped<GpuTextureSet>(ws, 1);
    DPE_CUDA_CHECK(cudaMemcpyAsync(data_.image_textures, &image_handles_, sizeof(GpuTextureSet),
                                  cudaMemcpyHostToDevice, stream));

    if (upload_depths) {
        for (int i = 0; i < data_.num_images; ++i) {
            cv::Mat depth(view_.cameras[i].height, view_.cameras[i].width, CV_32F, cv::Scalar(0));
            if (const FrameState* frame = state.Find(view_.image_ids[i])) {
                if (!frame->depth.empty()) {
                    if (frame->depth.size() == depth.size()) depth = frame->depth;
                    else cv::resize(frame->depth, depth, depth.size(), 0, 0, cv::INTER_NEAREST);
                }
            }
            cudaArray_t array = ws.AcquireArray(depth.cols, depth.rows);
            depth_arrays_.push_back(array);
            DPE_CUDA_CHECK(cudaMemcpy2DToArrayAsync(array, 0, 0, depth.ptr<float>(), depth.step,
                                                   depth.cols * sizeof(float), depth.rows,
                                                   cudaMemcpyHostToDevice, stream));
            depth_handles_.images[i] = CreateTexture(array);
        }
        data_.depth_textures = AcquireTyped<GpuTextureSet>(ws, 1);
        DPE_CUDA_CHECK(cudaMemcpyAsync(data_.depth_textures, &depth_handles_, sizeof(GpuTextureSet),
                                      cudaMemcpyHostToDevice, stream));
    }
}

GpuScene::~GpuScene() {
    cuda_.Synchronize();
    auto& ws = cuda_.Workspace();
    for (int i = 0; i < data_.num_images; ++i) {
        if (image_handles_.images[i]) cudaDestroyTextureObject(image_handles_.images[i]);
        if (depth_handles_.images[i]) cudaDestroyTextureObject(depth_handles_.images[i]);
    }
    for (size_t i = 0; i < image_arrays_.size(); ++i) {
        const cv::Mat& image = *view_.images_float[i];
        ws.ReleaseArray(image_arrays_[i], image.cols, image.rows);
    }
    for (size_t i = 0; i < depth_arrays_.size(); ++i)
        ws.ReleaseArray(depth_arrays_[i], view_.cameras[i].width, view_.cameras[i].height);
    ws.Release(data_.cameras);
    ws.Release(data_.image_textures);
    ws.Release(data_.depth_textures);
}

}  // namespace dpe
