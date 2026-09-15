#pragma once

#include "dpe/modules/view_selection.cuh"

namespace dpe {

__global__ void FinalizeDepthNormalKernel(DPEGpuContext* ctx){
    const int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    if(!InImage(p,ctx->width,ctx->height))return;
    const int c=p.y*ctx->width+p.x;
    float4 plane=ctx->state.planes[c];
    const float d=ComputeDepthFromPlane(ctx->cameras[0],plane,p);
    plane=CameraNormalToWorld(ctx->cameras[0],plane);
    plane.w=d;
    ctx->state.planes[c]=plane;
}

__device__ inline void StrongDepthMedianFilter(int2 p,DPEGpuContext* ctx){
    const int center=p.y*ctx->width+p.x;
    if(ctx->state.reliability[center]==WEAK)return;
    if(ctx->state.costs[center]<0.001f)return;

    // Same sparse 21-neighbour pattern used by the official implementation.
    float values[21];
    int n=0;
    values[n++]=ctx->state.planes[center].w;
    const int offsets[20][2]={
        {0,-1},{0,-3},{0,-5},{0,1},{0,3},{0,5},
        {-1,0},{-3,0},{-5,0},{1,0},{3,0},{5,0},
        {2,-1},{2,1},{-2,-1},{-2,1},{-1,-2},{1,-2},{-1,2},{1,2}
    };
    for(int i=0;i<20;++i){
        const int2 q=make_int2(p.x+offsets[i][0],p.y+offsets[i][1]);
        if(!InImage(q,ctx->width,ctx->height))continue;
        const int qi=q.y*ctx->width+q.x;
        if(ctx->state.reliability[qi]==STRONG)values[n++]=ctx->state.planes[qi].w;
    }
    SortSmall(values,n);
    ctx->state.planes[center].w=(n&1)?values[n/2]:0.5f*(values[n/2-1]+values[n/2]);
}

__global__ void StrongFilterBlackKernel(DPEGpuContext* ctx){
    int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    p.y=p.y*2+((threadIdx.x&1)?1:0);
    if(InImage(p,ctx->width,ctx->height))StrongDepthMedianFilter(p,ctx);
}
__global__ void StrongFilterRedKernel(DPEGpuContext* ctx){
    int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    p.y=p.y*2+((threadIdx.x&1)?0:1);
    if(InImage(p,ctx->width,ctx->height))StrongDepthMedianFilter(p,ctx);
}

__device__ inline float CostAtDepthWorldNormal(
    int2 p,float depth,const float4& world_n,const unsigned char* weights,DPEGpuContext* ctx){
    float4 n=WorldNormalToCamera(ctx->cameras[0],world_n);
    n.w=PlaneDistanceForDepth(ctx->cameras[0],p,depth,n);
    return CandidateWeightedCost(p,n,weights,false,ctx);
}

__global__ void LocalRefineKernel(DPEGpuContext* ctx){
    const int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    if(!InImage(p,ctx->width,ctx->height))return;
    const int center=p.y*ctx->width+p.x;
    const float4 world=ctx->state.planes[center];
    const float origin_depth=world.w;
    if(origin_depth==0.0f)return;

    const unsigned int mask=ctx->state.selected_views[center];
    const unsigned char* weights=&ctx->state.view_weights[center*kMaxImages];
    float weight_norm=0.0f;
    float baseline=0.0f;
    int valid=0;
    const Camera& ref=ctx->cameras[0];
    for(int src=1;src<ctx->num_images;++src){
        const int v=src-1;
        if(!BitSet(mask,v))continue;
        weight_norm+=weights[v];
        const Camera& sc=ctx->cameras[src];
        const float dx=ref.c[0]-sc.c[0],dy=ref.c[1]-sc.c[1],dz=ref.c[2]-sc.c[2];
        baseline+=sqrtf(dx*dx+dy*dy+dz*dz);
        ++valid;
    }
    if(weight_norm<=0.0f||valid==0)return;
    baseline/=valid;

    const float cost_now=CostAtDepthWorldNormal(p,origin_depth,world,weights,ctx);
    const float disp=ref.K[0]*baseline/origin_depth;
    float min_cost=2.0f;
    float best_depth=origin_depth;
    for(int delta=-5;delta<=5;++delta){
        const float denom=disp+delta;
        if(fabsf(denom)<1e-8f)continue;
        const float depth=ref.K[0]*baseline/denom;
        if(depth<ctx->params->depth_min||depth>ctx->params->depth_max)continue;
        const float cost=CostAtDepthWorldNormal(p,depth,world,weights,ctx);
        if(cost<min_cost){min_cost=cost;best_depth=depth;}
    }
    if(cost_now-min_cost>0.1f)ctx->state.planes[center].w=best_depth;
}

}  // namespace dpe
