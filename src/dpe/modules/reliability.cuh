#pragma once

#include "dpe/modules/refinement.cuh"

namespace dpe {

__global__ void ClassifyReliabilityKernel(DPEGpuContext* ctx){
    const int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    if(!InImage(p,ctx->width,ctx->height))return;
    const int center=p.y*ctx->width+p.x;
    if(p.x<6||p.y<6||p.x>=ctx->width-6||p.y>=ctx->height-6){
        ctx->state.reliability[center]=UNKNOWN;
        return;
    }

    const float4 world=ctx->state.planes[center];
    const float origin_depth=world.w;
    if(origin_depth<=0.0f||origin_depth<ctx->params->depth_min||origin_depth>ctx->params->depth_max){
        ctx->state.reliability[center]=UNKNOWN;
        return;
    }

    const unsigned int mask=ctx->state.selected_views[center];
    const unsigned char* weights=&ctx->state.view_weights[center*kMaxImages];
    const Camera& ref=ctx->cameras[0];
    float baseline=0.0f;
    float weight_norm=0.0f;
    int valid=0;
    for(int src=1;src<ctx->num_images;++src){
        const int v=src-1;
        if(!BitSet(mask,v))continue;
        weight_norm+=weights[v];
        const Camera& sc=ctx->cameras[src];
        const float dx=ref.c[0]-sc.c[0],dy=ref.c[1]-sc.c[1],dz=ref.c[2]-sc.c[2];
        baseline+=sqrtf(dx*dx+dy*dy+dz*dz);
        ++valid;
    }
    if(weight_norm<=0.0f||valid==0){
        ctx->state.reliability[center]=UNKNOWN;
        return;
    }
    baseline/=valid;

    const float disp=ref.K[0]*baseline/origin_depth;
    constexpr int radius=30;
    constexpr int count=2*radius+1;
    float costs[count];
    for(int delta=-radius;delta<=radius;++delta){
        const float denom=disp+delta;
        float c=2.0f;
        if(fabsf(denom)>1e-8f){
            const float depth=ref.K[0]*baseline/denom;
            if(depth>=ctx->params->depth_min&&depth<=ctx->params->depth_max)
                c=fminf(2.0f,CostAtDepthWorldNormal(p,depth,world,weights,ctx));
        }
        costs[delta+radius]=c;
    }

    bool peak[count];
    for(int i=0;i<count;++i)peak[i]=false;
    int peak_count=0;
    int min_peak=0;
    float min_cost=2.0f;
    for(int i=2;i<count-2;++i){
        if(costs[i-1]>costs[i]&&costs[i+1]>costs[i]){
            peak[i]=true;
            ++peak_count;
            if(costs[i]<min_cost){min_cost=costs[i];min_peak=i;}
        }
    }

    if(peak_count==0||abs(min_peak-radius)>ctx->params->weak_peak_radius||costs[min_peak]>0.5f){
        ctx->state.reliability[center]=WEAK;
        return;
    }
    if(peak_count==1){
        ctx->state.reliability[center]=(costs[min_peak]<=0.15f)?STRONG:WEAK;
        return;
    }

    float var=0.0f;
    for(int i=2;i<count-2;++i){
        if(peak[i]&&i!=min_peak){
            const float d=costs[i]-min_cost;
            var+=d*d;
        }
    }
    var=sqrtf(var)/(peak_count-1);
    ctx->state.reliability[center]=(var>0.2f)?STRONG:WEAK;
}

}  // namespace dpe
