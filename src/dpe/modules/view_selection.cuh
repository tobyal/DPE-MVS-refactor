#pragma once

#include "dpe/modules/matching_cost.cuh"
#include "dpe/modules/consistency.cuh"

namespace dpe {

// Initial top-k photometric view selection used by FIRST_INIT.
// This mirrors ComputeMultiViewInitialCostandSelectedViews in the reference code:
// the returned cost is the average of exactly the best top-k costs, while the
// selected-view mask includes all ties at the kth threshold.
__device__ inline float SelectInitialViews(int2 p, const float4& plane,
                                           unsigned char* weights,
                                           unsigned int* mask,
                                           const DPEGpuContext* ctx) {
    float costs[kMaxImages];
    float sorted[kMaxImages];
    const int n=ctx->num_images-1;
    for(int i=0;i<kMaxImages;++i){costs[i]=2.0f;sorted[i]=2.0f;weights[i]=0;}
    MultiViewCostVector(p,plane,false,costs,ctx);
    int valid=0;
    for(int i=0;i<n;++i){sorted[i]=costs[i];if(costs[i]<2.0f)++valid;}
    SortSmall(sorted,n);
    *mask=0;
    const int top=min(valid,ctx->params->top_k);
    if(top<=0)return 2.0f;
    float total=0.0f;
    for(int i=0;i<top;++i) total+=sorted[i];
    const float threshold=sorted[top-1];
    for(int i=0;i<n;++i){
        if(costs[i]<=threshold){weights[i]=1;SetBit(mask,i);}
    }
    return total/top;
}

// Photometric re-evaluation of an existing selected-view mask. Invalid views
// are removed exactly as in ComputeMultiViewInitialCost in the reference code.
__device__ inline float SelectedViewPhotometricCost(int2 p,const float4& plane,
                                                     unsigned char* weights,
                                                     unsigned int* mask,
                                                     const DPEGpuContext* ctx){
    const int n=ctx->num_images-1;
    float total=0.0f; int count=0;
    for(int i=0;i<kMaxImages;++i) weights[i]=0;
    for(int v=0;v<n;++v){
        if(!BitSet(*mask,v)) continue;
        const float c=StandardPatchCost(p,v+1,plane,ctx);
        if(c<2.0f){total+=c;++count;weights[v]=1;}
        else UnsetBit(mask,v);
    }
    return count>0?total/count:2.0f;
}

__device__ inline void SampleJointViewWeights(const float candidate_costs[8][kMaxImages],
                                               const bool valid[8],int iter,
                                               const float* priors,
                                               unsigned char* out_weights,
                                               unsigned int* out_mask,
                                               int center,
                                               const DPEGpuContext* ctx){
    const int n=ctx->num_images-1;
    float probs[kMaxImages];
    for(int i=0;i<kMaxImages;++i){probs[i]=0.0f;out_weights[i]=0;}
    const float threshold=0.8f*expf(iter*iter/-90.0f);
    float sum=0.0f;
    for(int v=0;v<n;++v){
        int good=0,bad=0; float w=0.0f;
        for(int c=0;c<8;++c){
            const float cost=valid[c]?candidate_costs[c][v]:2.0f;
            if(cost<threshold){w+=expf(cost*cost/-0.18f);++good;}
            if(cost>1.2f)++bad;
        }
        if(good>2&&bad<3) probs[v]=(w/good)*priors[v];
        else if(bad<3) probs[v]=expf(threshold*threshold/-0.32f)*priors[v];
        sum+=probs[v];
    }
    // The original code assumes a non-zero PDF. Keep a deterministic safe
    // fallback so degenerate scenes cannot create NaNs in the CDF.
    if(sum<=1e-12f){
        for(int v=0;v<n;++v) probs[v]=1.0f/n;
    }else{
        for(int v=0;v<n;++v) probs[v]/=sum;
    }
    for(int v=1;v<n;++v) probs[v]+=probs[v-1];
    curandState* rng=&ctx->state.random_states[center];
    for(int s=0;s<15;++s){
        const float r=curand_uniform(rng)-FLT_EPSILON;
        for(int v=0;v<n;++v){if(probs[v]>r){out_weights[v]++;break;}}
    }
    *out_mask=0;
    for(int v=0;v<n;++v)if(out_weights[v])SetBit(out_mask,v);
}

__device__ inline void JointViewSelectionStrong(int2 p,
                                                 const float candidate_costs[8][kMaxImages],
                                                 const bool valid[8],int iter,
                                                 unsigned char* out_weights,
                                                 unsigned int* out_mask,
                                                 const DPEGpuContext* ctx){
    const int n=ctx->num_images-1;
    const int center=p.y*ctx->width+p.x;
    const int neighbor[4]={center-ctx->width,center+ctx->width,center-1,center+1};
    float priors[kMaxImages]={0};
    // Official strong path: only near slots 0,2,4,6 contribute priors.
    for(int k=0;k<4;++k){
        if(!valid[2*k])continue;
        const int q=neighbor[k];
        if(q<0||q>=ctx->width*ctx->height)continue;
        const unsigned int m=ctx->state.selected_views[q];
        for(int v=0;v<n;++v)priors[v]+=BitSet(m,v)?0.9f:0.1f;
    }
    SampleJointViewWeights(candidate_costs,valid,iter,priors,out_weights,out_mask,center,ctx);
}

__device__ inline void JointViewSelectionWeak(int2 p,
                                               const float candidate_costs[8][kMaxImages],
                                               const bool valid[8],int iter,
                                               unsigned char* out_weights,
                                               unsigned int* out_mask,
                                               const DPEGpuContext* ctx){
    const int n=ctx->num_images-1;
    const int center=p.y*ctx->width+p.x;
    float priors[kMaxImages]={0};
    const int map=ctx->state.anchor_map[center];
    if(map>=0){
        // Official weak path: priors are accumulated from all eight anchors.
        for(int k=1;k<kNeighbourNum;++k){
            const short2 a=ctx->state.anchors[map*kNeighbourNum+k];
            if(a.x<0||a.y<0)continue;
            const unsigned int m=ctx->state.selected_views[a.y*ctx->width+a.x];
            for(int v=0;v<n;++v)priors[v]+=BitSet(m,v)?0.9f:0.1f;
        }
    }
    SampleJointViewWeights(candidate_costs,valid,iter,priors,out_weights,out_mask,center,ctx);
}

__device__ inline float PhotometricWeightedCost(int2 p,const float4& plane,
                                                 const unsigned char* weights,
                                                 bool deformable,const DPEGpuContext* ctx){
    float v[kMaxImages];MultiViewCostVector(p,plane,deformable,v,ctx);
    return WeightedCost(v,weights,ctx->num_images-1);
}

__device__ inline float CandidateWeightedCost(int2 p,const float4& plane,
                                               const unsigned char* weights,
                                               bool deformable,const DPEGpuContext* ctx){
    float v[kMaxImages]; MultiViewCostVector(p,plane,deformable,v,ctx);
    float total=0,norm=0;
    for(int i=0;i<ctx->num_images-1;++i){
        if(!weights[i]) continue;
        float c=v[i];
        if(ctx->params->geom_consistency)
            c+=ctx->params->geom_factor*GeometryConsistencyCost(p,i+1,plane,ctx);
        total+=weights[i]*c; norm+=weights[i];
    }
    return norm>0?total/norm:2.0f;
}

}  // namespace dpe
