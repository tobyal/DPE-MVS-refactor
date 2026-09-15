#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

namespace dpe {

inline void CheckCuda(cudaError_t err, const char* file, int line) {
    if (err == cudaSuccess) return;
    std::ostringstream os;
    os << "CUDA error at " << file << ':' << line << ": " << cudaGetErrorString(err);
    throw std::runtime_error(os.str());
}

#define DPE_CUDA_CHECK(expr) ::dpe::CheckCuda((expr), __FILE__, __LINE__)

#ifdef DPE_DEBUG_SYNC
#define DPE_KERNEL_CHECK(stream) do { \
    DPE_CUDA_CHECK(cudaPeekAtLastError()); \
    DPE_CUDA_CHECK(cudaStreamSynchronize(stream)); \
} while (0)
#else
#define DPE_KERNEL_CHECK(stream) DPE_CUDA_CHECK(cudaPeekAtLastError())
#endif

}  // namespace dpe
