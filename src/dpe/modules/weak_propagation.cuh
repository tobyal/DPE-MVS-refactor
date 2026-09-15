#pragma once

#include "dpe/modules/view_selection.cuh"

namespace dpe {

__device__ inline void RefineWeakHypothesis(int2 p,float4* plane,float* cost,
                                             const unsigned char* weights,DPEGpuContext* ctx){
    const int center=p.y*ctx->width+p.x;
    curandState* rng=&ctx->state.random_states[center];
    const Camera& cam=ctx->cameras[0];
    const float dmin=ctx->params->depth_min,dmax=ctx->params->depth_max;
    float depth=ComputeDepthFromPlane(cam,*plane,p);

    // The fitted plane is explicitly part of the weak-pixel hypothesis pool.
    // The official implementation stops weak refinement when no fitted plane exists.
    const float4 fitted=ctx->state.fitted_planes[center];
    if(fitted.x==0&&fitted.y==0&&fitted.z==0)return;
    const float fd=ComputeDepthFromPlane(cam,fitted,p);
    if(fd>=dmin&&fd<=dmax){
        const float c=CandidateWeightedCost(p,fitted,weights,true,ctx);
        if(c<*cost){*cost=c;*plane=fitted;depth=fd;}
    }
    const float random_depth=dmin+curand_uniform(rng)*(dmax-dmin);
    const float pert_depth=depth*(0.98f+0.04f*curand_uniform(rng));
    const float4 random_n=RandomNormal(cam,p,rng);
    const float4 pert_n=PerturbNormal(cam,p,*plane,rng,0.02f*M_PI);
    float depths[5]={random_depth,depth,random_depth,depth,pert_depth};
    float4 normals[5]={*plane,random_n,random_n,pert_n,*plane};
    for(int i=0;i<5;++i){
        float4 q=normals[i];q.w=PlaneDistanceForDepth(cam,p,depths[i],q);
        const float c=CandidateWeightedCost(p,q,weights,true,ctx);
        const float d=ComputeDepthFromPlane(cam,q,p);
        if(d>=dmin&&d<=dmax&&c<*cost){*cost=c;*plane=q;}
    }
}

__device__ inline void WeakPropagation(int2 p,int iter,DPEGpuContext* ctx){
    const int center=p.y*ctx->width+p.x;
    const int map=ctx->state.anchor_map[center];
    if(map<0)return;
    float4 candidate[8];bool valid[8]={false};float cost_matrix[8][kMaxImages];
    for(int i=0;i<8;++i){for(int v=0;v<kMaxImages;++v)cost_matrix[i][v]=2.0f;}
    int n=0;
    for(int k=1;k<kNeighbourNum&&n<8;++k){
        const short2 a=ctx->state.anchors[map*kNeighbourNum+k];
        if(a.x<0)continue;
        const int ai=a.y*ctx->width+a.x;
        if(ctx->state.reliability[ai]!=STRONG)continue;
        candidate[n]=ctx->state.planes[ai];valid[n]=true;
        MultiViewCostVector(p,candidate[n],true,cost_matrix[n],ctx);++n;
    }
    unsigned char weights[kMaxImages]={0};unsigned int mask=0;
    JointViewSelectionWeak(p,cost_matrix,valid,iter,weights,&mask,ctx);
    for(int v=0;v<kMaxImages;++v)
        ctx->state.view_weights[center*kMaxImages+v]=weights[v];

    const float4 original=ctx->state.planes[center];
    const float baseline=CandidateWeightedCost(p,original,weights,true,ctx);
    ctx->state.costs[center]=baseline;
    float4 best=original;
    float best_cost=baseline;
    bool propagated=false;
    for(int i=0;i<n;++i){
        const float d=ComputeDepthFromPlane(ctx->cameras[0],candidate[i],p);
        if(d<ctx->params->depth_min||d>ctx->params->depth_max)continue;
        const float c=CandidateWeightedCost(p,candidate[i],weights,true,ctx);
        if(c<best_cost){best_cost=c;best=candidate[i];propagated=true;}
    }
    if(propagated)ctx->state.selected_views[center]=mask;

    RefineWeakHypothesis(p,&best,&best_cost,weights,ctx);
    if(ctx->params->state==RunState::RefineInit){
        if(best_cost<baseline-0.1f){ctx->state.planes[center]=best;ctx->state.costs[center]=best_cost;}
    }else{
        ctx->state.planes[center]=best;ctx->state.costs[center]=best_cost;
    }
    // Re-evaluate with standard photometric matching for the next strong stage,
    // matching the reference weak-path cost reset.
    float vec[kMaxImages];MultiViewCostVector(p,ctx->state.planes[center],false,vec,ctx);
    ctx->state.costs[center]=WeightedCost(vec,weights,ctx->num_images-1);
}
__global__ void WeakBlackKernel(int iter,DPEGpuContext* ctx){
    int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    p.y=p.y*2+((threadIdx.x&1)?1:0);
    if(!InImage(p,ctx->width,ctx->height))return;
    if(ctx->state.reliability[p.y*ctx->width+p.x]==WEAK)WeakPropagation(p,iter,ctx);
}
__global__ void WeakRedKernel(int iter,DPEGpuContext* ctx){
    int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    p.y=p.y*2+((threadIdx.x&1)?0:1);
    if(!InImage(p,ctx->width,ctx->height))return;
    if(ctx->state.reliability[p.y*ctx->width+p.x]==WEAK)WeakPropagation(p,iter,ctx);
}

}  // namespace dpe
