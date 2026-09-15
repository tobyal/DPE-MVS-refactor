#include "runtime/cuda_context.h"
#include "common/cuda_utils.cuh"

namespace dpe {

CudaContext::CudaContext(int gpu_index) : gpu_index_(gpu_index) {
    DPE_CUDA_CHECK(cudaSetDevice(gpu_index_));
    DPE_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

CudaContext::~CudaContext() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
}

void CudaContext::Synchronize() const {
    DPE_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace dpe
