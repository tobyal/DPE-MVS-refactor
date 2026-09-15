#pragma once

#include <cuda_runtime.h>
#include <map>
#include <unordered_map>
#include <vector>
#include <utility>
#include <cstddef>

namespace dpe {

class GpuWorkspace {
public:
    GpuWorkspace() = default;
    ~GpuWorkspace();

    void* Acquire(size_t bytes);
    void Release(void* ptr);

    cudaArray_t AcquireArray(int width, int height);
    void ReleaseArray(cudaArray_t array, int width, int height);

    size_t ReservedBytes() const { return reserved_bytes_; }

private:
    std::multimap<size_t, void*> free_blocks_;
    std::unordered_map<void*, size_t> live_blocks_;
    std::map<std::pair<int, int>, std::vector<cudaArray_t>> free_arrays_;
    std::map<cudaArray_t, std::pair<int, int>> live_arrays_;
    size_t reserved_bytes_ = 0;
};

template <typename T>
T* AcquireTyped(GpuWorkspace& ws, size_t count) {
    return static_cast<T*>(ws.Acquire(sizeof(T) * count));
}

}  // namespace dpe
