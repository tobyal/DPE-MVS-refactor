#pragma once

#include "dpe/modules/view_selection.cuh"

namespace dpe {

__global__ void InitRandomStatesKernel(DPEGpuContext* ctx) {
    const int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    if(!InImage(p,ctx->width,ctx->height)) return;
    const int c=p.y*ctx->width+p.x;
    curand_init(clock64(),p.y,p.x,&ctx->state.random_states[c]);
}

__global__ void InitializeHypothesesKernel(DPEGpuContext* ctx) {
    const int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    if(!InImage(p,ctx->width,ctx->height)) return;
    const int c=p.y*ctx->width+p.x;
    float4 plane=ctx->state.planes[c];
    if(ctx->params->state==RunState::FirstInit){
        plane=RandomPlane(ctx->cameras[0],p,&ctx->state.random_states[c],ctx->params->depth_min,ctx->params->depth_max);
        ctx->state.planes[c]=plane;
        unsigned char weights[kMaxImages]; unsigned int mask=0;
        ctx->state.costs[c]=SelectInitialViews(p,plane,weights,&mask,ctx);
        ctx->state.selected_views[c]=mask;
    }else{
        // Host state stores world normal + depth. Convert to camera-plane representation.
        const float depth=plane.w;
        plane=WorldNormalToCamera(ctx->cameras[0],plane);
        plane.w=PlaneDistanceForDepth(ctx->cameras[0],p,depth,plane);
        ctx->state.planes[c]=plane;
        unsigned char weights[kMaxImages]={0};
        unsigned int mask=ctx->state.selected_views[c];
        const float cost=SelectedViewPhotometricCost(p,plane,weights,&mask,ctx);
        ctx->state.selected_views[c]=mask;
        // Reference DPE initialization remains photometric even in REFINE_ITER.
        ctx->state.costs[c]=cost;
    }
}

}  // namespace dpe
