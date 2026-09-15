#pragma once

#include "runtime/gpu_workspace.h"
#include <cuda_runtime.h>

namespace dpe {

class CudaContext {
public:
    explicit CudaContext(int gpu_index = 0);
    ~CudaContext();

    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    cudaStream_t Stream() const { return stream_; }
    GpuWorkspace& Workspace() { return workspace_; }
    const GpuWorkspace& Workspace() const { return workspace_; }
    void Synchronize() const;

private:
    int gpu_index_ = 0;
    cudaStream_t stream_ = nullptr;
    GpuWorkspace workspace_;
};

}  // namespace dpe
