#include "dpe/dpe.h"
#include "common/cuda_utils.cuh"

#include "dpe/modules/device_math.cuh"
#include "dpe/modules/matching_cost.cuh"
#include "dpe/modules/consistency.cuh"
#include "dpe/modules/view_selection.cuh"
#include "dpe/modules/initialization.cuh"
#include "dpe/modules/edge_guidance.cuh"
#include "dpe/modules/anchor_search.cuh"
#include "dpe/modules/strong_propagation.cuh"
#include "dpe/modules/plane_construction.cuh"
#include "dpe/modules/weak_propagation.cuh"
#include "dpe/modules/refinement.cuh"
#include "dpe/modules/reliability.cuh"

namespace dpe {

void RunDpeKernels(DPEGpuContext* ctx, cudaStream_t stream,
                   const DPEParams& params, int width, int height) {
    const dim3 block_full(16,16);
    const dim3 grid_full((width+15)/16,(height+15)/16);
    const dim3 block_half(32,16);
    const dim3 grid_half((width+31)/32,((height+1)/2+15)/16);

    InitRandomStatesKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);

    // Convert/recreate current hypotheses before any geometry module consumes them.
    InitializeHypothesesKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);

    GenerateGuidanceKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    FindNearestStrongKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    GenerateAnchorsKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    UpdateUnreliableWeakKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);

    for(int iter=0;iter<params.max_iterations;++iter){
        StrongBlackKernel<<<grid_half,block_half,0,stream>>>(iter,ctx);
        DPE_KERNEL_CHECK(stream);
        StrongRedKernel<<<grid_half,block_half,0,stream>>>(iter,ctx);
        DPE_KERNEL_CHECK(stream);

        ConstructWeakPlaneKernel<<<grid_full,block_full,0,stream>>>(ctx);
        DPE_KERNEL_CHECK(stream);

        WeakBlackKernel<<<grid_half,block_half,0,stream>>>(iter,ctx);
        DPE_KERNEL_CHECK(stream);
        WeakRedKernel<<<grid_half,block_half,0,stream>>>(iter,ctx);
        DPE_KERNEL_CHECK(stream);
    }

    FinalizeDepthNormalKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    StrongFilterBlackKernel<<<grid_half,block_half,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    StrongFilterRedKernel<<<grid_half,block_half,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    ClassifyReliabilityKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    LocalRefineKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
}

}  // namespace dpe
