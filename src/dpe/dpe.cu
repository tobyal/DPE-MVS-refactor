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
                   const DPEParams& params, int width, int height,
                   const KernelStageCallback& stage_callback) {
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
    if (stage_callback) stage_callback(DiagnosticStage::Input, -1);

    for(int iter=0;iter<params.max_iterations;++iter){
        StrongBlackKernel<<<grid_half,block_half,0,stream>>>(iter,ctx);
        DPE_KERNEL_CHECK(stream);
        StrongRedKernel<<<grid_half,block_half,0,stream>>>(iter,ctx);
        DPE_KERNEL_CHECK(stream);
        if (stage_callback) stage_callback(DiagnosticStage::Strong, iter);

        ConstructWeakPlaneKernel<<<grid_full,block_full,0,stream>>>(ctx);
        DPE_KERNEL_CHECK(stream);
        if (stage_callback) stage_callback(DiagnosticStage::Plane, iter);

        WeakBlackKernel<<<grid_half,block_half,0,stream>>>(iter,ctx);
        DPE_KERNEL_CHECK(stream);
        WeakRedKernel<<<grid_half,block_half,0,stream>>>(iter,ctx);
        DPE_KERNEL_CHECK(stream);
        if (stage_callback) stage_callback(DiagnosticStage::Weak, iter);
    }

    FinalizeDepthNormalKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    if (stage_callback) stage_callback(DiagnosticStage::Finalized, -1);
    StrongFilterBlackKernel<<<grid_half,block_half,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    StrongFilterRedKernel<<<grid_half,block_half,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    if (stage_callback) stage_callback(DiagnosticStage::Filtered, -1);
    ClassifyReliabilityKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    if (stage_callback) stage_callback(DiagnosticStage::Classified, -1);
    LocalRefineKernel<<<grid_full,block_full,0,stream>>>(ctx);
    DPE_KERNEL_CHECK(stream);
    if (stage_callback) stage_callback(DiagnosticStage::Refined, -1);
}

}  // namespace dpe
