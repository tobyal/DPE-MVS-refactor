#pragma once

#include "dpe/modules/view_selection.cuh"
#include "experiments/telemetry/gpu_telemetry.cuh"

namespace dpe {

__device__ inline void RefineStrongHypothesis(int2 p,float4* plane,float* cost,
                                               const unsigned char* weights,DPEGpuContext* ctx){
    const int center=p.y*ctx->width+p.x;
    curandState* rng=&ctx->state.random_states[center];
    const Camera& cam=ctx->cameras[0];
    float depth=ComputeDepthFromPlane(cam,*plane,p);
    const float dmin=ctx->params->depth_min,dmax=ctx->params->depth_max;
    const float random_depth=dmin+curand_uniform(rng)*(dmax-dmin);
    const float perturbed_depth=depth*(0.98f+0.04f*curand_uniform(rng));
    const float4 random_normal=RandomNormal(cam,p,rng);
    const float4 pert_normal=PerturbNormal(cam,p,*plane,rng,0.02f*M_PI);
    const float depths[5]={random_depth,depth,random_depth,depth,perturbed_depth};
    const float4 normals[5]={*plane,random_normal,random_normal,pert_normal,*plane};
    for(int i=0;i<5;++i){
        float4 candidate=normals[i];
        candidate.w=PlaneDistanceForDepth(cam,p,depths[i],candidate);
        const float c=PhotometricWeightedCost(p,candidate,weights,false,ctx);
        const float d=ComputeDepthFromPlane(cam,candidate,p);
        if(d>=dmin&&d<=dmax&&c<*cost){depth=d;*cost=c;*plane=candidate;}
    }
}

__device__ inline bool BestCostOnRay(int2 p,int dx,int dy,int start_x,int start_y,
                                      int step_len,int step_num,int dir_index,
                                      int* best_position,const DPEGpuContext* ctx){
    float best=FLT_MAX;
    int pos=-1;
    for(int step=0;step<step_num;++step){
        int fx=0,fy=0;
        if(dir_index>4){if(dir_index&1)fx=dx;else fy=dy;}
        const int2 q=make_int2(p.x+start_x+step*step_len*dx+fx,
                               p.y+start_y+step*step_len*dy+fy);
        if(!InImage(q,ctx->width,ctx->height))continue;
        const int qi=q.y*ctx->width+q.x;
        if(ctx->state.costs[qi]<best){best=ctx->state.costs[qi];pos=qi;}
    }
    if(pos<0)return false;
    *best_position=pos;
    return true;
}

__device__ inline void CollectEdgeGuidedCandidates(
    int2 p,int iter,int positions[8],bool valid[8],float costs[8][kMaxImages],DPEGpuContext* ctx){
    const int dirs[8][2]={{0,-1},{0,1},{-1,0},{1,0},{-1,-1},{1,1},{-1,1},{1,-1}};
    const int center=p.y*ctx->width+p.x;
    const bool center_edge=ctx->guidance.fine_edges&&ctx->guidance.fine_edges[center];
    const float max_edge_dist=max(ctx->height,ctx->width)/30.0f;
    constexpr int min_step_len=2;

    for(int d=0;d<8;++d){
        const int dx=dirs[d][0],dy=dirs[d][1];
        const int sx=max(1,5-2*iter)*dx;
        const int sy=max(1,5-2*iter)*dy;
        const short2 e=ctx->guidance.nearest_edges[center*8+d];
        float dist=(e.x>=0&&e.y>=0)?hypotf(e.x-p.x,e.y-p.y):max_edge_dist;
        if(d>=4)dist/=sqrtf(2.0f);
        if(center_edge)dist=11*min_step_len;
        else if(e.x<0||e.y<0||dist>max_edge_dist){dist=max_edge_dist;if(d>=4)dist/=sqrtf(2.0f);}
        const int step_num=min(max(11,static_cast<int>(dist/min_step_len)),22);
        int step_len=max(static_cast<int>(dist/step_num),min_step_len);
        if(d<4&&(step_len&1))--step_len;
        int pos=-1;
        if(BestCostOnRay(p,dx,dy,sx,sy,step_len,step_num,d,&pos,ctx)){
            valid[d]=true;positions[d]=pos;
            MultiViewCostVector(p,ctx->state.planes[pos],false,costs[d],ctx);
        }
    }

    // Non-edge pixels also use the fixed 11-step non-local sampler from DPE.
    if(!center_edge){
        const float good_threshold=0.8f*expf(iter*iter/-90.0f);
        constexpr float bad_threshold=1.2f;
        for(int d=0;d<8;++d){
            const int dx=dirs[d][0],dy=dirs[d][1];
            const int sx=max(1,5-2*iter)*dx;
            const int sy=max(1,5-2*iter)*dy;
            int pos=-1;
            if(!BestCostOnRay(p,dx,dy,sx,sy,min_step_len,11,d,&pos,ctx))continue;
            float alt[kMaxImages];for(int v=0;v<kMaxImages;++v)alt[v]=2.0f;
            MultiViewCostVector(p,ctx->state.planes[pos],false,alt,ctx);
            int good_old=0,bad_old=0,good_new=0,bad_new=0;
            for(int v=0;v<ctx->num_images-1;++v){
                if(valid[d]){if(costs[d][v]<good_threshold)++good_old;if(costs[d][v]>bad_threshold)++bad_old;}
                if(alt[v]<good_threshold)++good_new;if(alt[v]>bad_threshold)++bad_new;
            }
            if(!valid[d]||good_new>good_old||(good_new==good_old&&bad_new<bad_old)){
                valid[d]=true;positions[d]=pos;
                for(int v=0;v<ctx->num_images-1;++v)costs[d][v]=alt[v];
            }
        }
    }
}

__device__ inline void ConsiderPosition(int2 p,int candidate,int slot,
                                        int positions[8],bool valid[8],float costs[8][kMaxImages],
                                        DPEGpuContext* ctx){
    if(candidate<0||candidate>=ctx->width*ctx->height)return;
    valid[slot]=true;positions[slot]=candidate;
    MultiViewCostVector(p,ctx->state.planes[candidate],false,costs[slot],ctx);
}

__device__ inline void CollectCheckerboardCandidates(
    int2 p,int positions[8],bool valid[8],float costs[8][kMaxImages],DPEGpuContext* ctx){
    const int w=ctx->width,h=ctx->height,center=p.y*w+p.x;
    // Slot order follows the official implementation:
    // up-near, up-far, down-near, down-far, left-near, left-far, right-near, right-far.
    if(p.y>2){
        int best=center-3*w;float c=ctx->state.costs[best];
        for(int i=1;i<11;++i)if(p.y>2+2*i){const int q=center-(3+2*i)*w;if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
        ConsiderPosition(p,best,1,positions,valid,costs,ctx);
    }
    if(p.y<h-3){
        int best=center+3*w;float c=ctx->state.costs[best];
        for(int i=1;i<11;++i)if(p.y<h-3-2*i){const int q=center+(3+2*i)*w;if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
        ConsiderPosition(p,best,3,positions,valid,costs,ctx);
    }
    if(p.x>2){
        int best=center-3;float c=ctx->state.costs[best];
        for(int i=1;i<11;++i)if(p.x>2+2*i){const int q=center-(3+2*i);if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
        ConsiderPosition(p,best,5,positions,valid,costs,ctx);
    }
    if(p.x<w-3){
        int best=center+3;float c=ctx->state.costs[best];
        for(int i=1;i<11;++i)if(p.x<w-3-2*i){const int q=center+(3+2*i);if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
        ConsiderPosition(p,best,7,positions,valid,costs,ctx);
    }

    if(p.y>0){
        int best=center-w;float c=ctx->state.costs[best];
        for(int i=0;i<3;++i){const int off=1+i;
            if(p.y>off&&p.x>i){const int q=center-(1+off)*w-off;if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
            if(p.y>off&&p.x<w-1-i){const int q=center-(1+off)*w+off;if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
        }
        ConsiderPosition(p,best,0,positions,valid,costs,ctx);
    }
    if(p.y<h-1){
        int best=center+w;float c=ctx->state.costs[best];
        for(int i=0;i<3;++i){const int off=1+i;
            if(p.y<h-1-off&&p.x>i){const int q=center+(1+off)*w-off;if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
            if(p.y<h-1-off&&p.x<w-1-i){const int q=center+(1+off)*w+off;if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
        }
        ConsiderPosition(p,best,2,positions,valid,costs,ctx);
    }
    if(p.x>0){
        int best=center-1;float c=ctx->state.costs[best];
        for(int i=0;i<3;++i){const int off=1+i;
            if(p.x>off&&p.y>i){const int q=center-(1+off)-off*w;if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
            if(p.x>off&&p.y<h-1-i){const int q=center-(1+off)+off*w;if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
        }
        ConsiderPosition(p,best,4,positions,valid,costs,ctx);
    }
    if(p.x<w-1){
        int best=center+1;float c=ctx->state.costs[best];
        for(int i=0;i<3;++i){const int off=1+i;
            if(p.x<w-1-off&&p.y>i){const int q=center+(1+off)-off*w;if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
            if(p.x<w-1-off&&p.y<h-1-i){const int q=center+(1+off)+off*w;if(ctx->state.costs[q]<c){c=ctx->state.costs[q];best=q;}}
        }
        ConsiderPosition(p,best,6,positions,valid,costs,ctx);
    }
}

__device__ inline void StrongPropagation(int2 p,int iter,DPEGpuContext* ctx){
    const int center=p.y*ctx->width+p.x;
    int positions[8];for(int i=0;i<8;++i)positions[i]=-1;
    bool valid[8]={false};
    float cost_matrix[8][kMaxImages];
    for(int i=0;i<8;++i)for(int v=0;v<kMaxImages;++v)cost_matrix[i][v]=2.0f;

    if(ctx->params->use_edge)CollectEdgeGuidedCandidates(p,iter,positions,valid,cost_matrix,ctx);
    else CollectCheckerboardCandidates(p,positions,valid,cost_matrix,ctx);
    TelemetryStrongCandidates(ctx,center,positions,valid);

    unsigned char weights[kMaxImages]={0};
    unsigned int mask=0;
    JointViewSelectionStrong(p,cost_matrix,valid,iter,weights,&mask,ctx);
    for(int v=0;v<kMaxImages;++v)ctx->state.view_weights[center*kMaxImages+v]=weights[v];

    const float4 original=ctx->state.planes[center];
    const float baseline=PhotometricWeightedCost(p,original,weights,false,ctx);
    // The reference implementation writes the re-evaluated baseline before
    // propagation. REFINE_INIT then requires a 0.1 improvement over it.
    ctx->state.costs[center]=baseline;

    float4 best=original;
    float best_cost=baseline;
    bool propagated=false;
    for(int i=0;i<8;++i){
        if(!valid[i])continue;
        const float4 q=ctx->state.planes[positions[i]];
        const float d=ComputeDepthFromPlane(ctx->cameras[0],q,p);
        if(d<ctx->params->depth_min||d>ctx->params->depth_max)continue;
        const float c=PhotometricWeightedCost(p,q,weights,false,ctx);
        if(c<best_cost){best_cost=c;best=q;propagated=true;}
    }
    // In reference DPE, the new view mask is committed when a propagated
    // candidate wins; random refinement does not independently change it.
    if(propagated)ctx->state.selected_views[center]=mask;

    RefineStrongHypothesis(p,&best,&best_cost,weights,ctx);
    if(ctx->params->state==RunState::RefineInit){
        if(best_cost<baseline-0.1f){
            ctx->state.planes[center]=best;
            ctx->state.costs[center]=best_cost;
        }
    }else{
        ctx->state.planes[center]=best;
        ctx->state.costs[center]=best_cost;
    }
}
__global__ void StrongBlackKernel(int iter,DPEGpuContext* ctx){
    int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    p.y=p.y*2+((threadIdx.x&1)?1:0);
    if(!InImage(p,ctx->width,ctx->height))return;
    if(ctx->state.reliability[p.y*ctx->width+p.x]!=WEAK)StrongPropagation(p,iter,ctx);
}
__global__ void StrongRedKernel(int iter,DPEGpuContext* ctx){
    int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    p.y=p.y*2+((threadIdx.x&1)?0:1);
    if(!InImage(p,ctx->width,ctx->height))return;
    if(ctx->state.reliability[p.y*ctx->width+p.x]!=WEAK)StrongPropagation(p,iter,ctx);
}

}  // namespace dpe
