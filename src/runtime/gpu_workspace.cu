#include "runtime/gpu_workspace.h"
#include "common/cuda_utils.cuh"

namespace dpe {

GpuWorkspace::~GpuWorkspace() {
    for (const auto& x : free_blocks_) cudaFree(x.second);
    for (const auto& x : live_blocks_) cudaFree(x.first);
    for (const auto& bucket : free_arrays_)
        for (cudaArray_t a : bucket.second) cudaFreeArray(a);
    for (const auto& x : live_arrays_) cudaFreeArray(x.first);
}

void* GpuWorkspace::Acquire(size_t bytes) {
    if (bytes == 0) return nullptr;
    auto it = free_blocks_.lower_bound(bytes);
    void* ptr = nullptr;
    size_t capacity = bytes;
    if (it != free_blocks_.end()) {
        ptr = it->second;
        capacity = it->first;
        free_blocks_.erase(it);
    } else {
        DPE_CUDA_CHECK(cudaMalloc(&ptr, bytes));
        reserved_bytes_ += bytes;
    }
    live_blocks_[ptr] = capacity;
    return ptr;
}

void GpuWorkspace::Release(void* ptr) {
    if (!ptr) return;
    auto it = live_blocks_.find(ptr);
    if (it == live_blocks_.end()) return;
    free_blocks_.emplace(it->second, ptr);
    live_blocks_.erase(it);
}

cudaArray_t GpuWorkspace::AcquireArray(int width, int height) {
    const auto key = std::make_pair(width, height);
    auto& bucket = free_arrays_[key];
    cudaArray_t array = nullptr;
    if (!bucket.empty()) {
        array = bucket.back();
        bucket.pop_back();
    } else {
        const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        DPE_CUDA_CHECK(cudaMallocArray(&array, &desc, width, height));
        reserved_bytes_ += static_cast<size_t>(width) * height * sizeof(float);
    }
    live_arrays_[array] = key;
    return array;
}

void GpuWorkspace::ReleaseArray(cudaArray_t array, int width, int height) {
    if (!array) return;
    auto it = live_arrays_.find(array);
    const auto key = it == live_arrays_.end() ? std::make_pair(width, height) : it->second;
    if (it != live_arrays_.end()) live_arrays_.erase(it);
    free_arrays_[key].push_back(array);
}

}  // namespace dpe
