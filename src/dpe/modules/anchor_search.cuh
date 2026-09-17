#pragma once

#include "dpe/modules/edge_guidance.cuh"

namespace dpe {

__device__ inline float3 PointFromDepth(const Camera& cam,short2 p,float d){
    return make_float3(d*(p.x-cam.K[2])/cam.K[0],d*(p.y-cam.K[5])/cam.K[4],d);
}

__device__ inline void AddCandidate(
    short2 q,
    short2* pts,
    int* count) {

    if(q.x < 0 || q.y < 0 || *count >= 64)
        return;

    pts[(*count)++] = q;
}

__device__ inline bool CandidateEdgeAllowed(int2 center,short2 q,bool edge_limit,const DPEGpuContext* ctx){
    return !edge_limit || !EdgeCrossing(center,make_int2(q.x,q.y),ctx);
}

__global__ void GenerateAnchorsKernel(DPEGpuContext* ctx){
    const int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    if(!InImage(p,ctx->width,ctx->height)) return;
    const int center=p.y*ctx->width+p.x;
    if(ctx->state.reliability[center]!=WEAK) return;
    const int map=ctx->state.anchor_map[center];
    if(map<0) return;
    short2* out=&ctx->state.anchors[map*kNeighbourNum];
    for(int i=0;i<kNeighbourNum;++i) out[i]=make_short2(-1,-1);
    out[0]=make_short2(p.x,p.y);

    bool edge_limit=ctx->params->use_limit;
    if(edge_limit&&ctx->params->use_edge){
        const float prob=curand_uniform(&ctx->state.random_states[center])-FLT_EPSILON;
        const float complexity=ctx->guidance.texture_complexity[center];
        if(prob<complexity) edge_limit=false;
        else ctx->guidance.texture_complexity[center]=fmaxf(0.99f,complexity);
    }

    short2 candidates[64]; int count=0;
    for(int i=0;i<64;++i)candidates[i]=make_short2(-1,-1);
    const int rotate_time=max(1,min(4,ctx->params->rotate_time));
    const float angle=(45.0f/rotate_time)*M_PI/180.0f;
    const float cang=cosf(angle),sang=sinf(angle);
    const float cos_thresh=cosf((45.0f/rotate_time/2.0f)*M_PI/180.0f);
    const int shift_range=max(static_cast<int>(tanf((45.0f/rotate_time/2.0f)*M_PI/180.0f)*20.0f),1);
    const int base_dirs[8][2]={{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
    curandState* rng=&ctx->state.random_states[center];
    for(int od=0;od<8;++od){
        float2 dir=make_float2(base_dirs[od][0],base_dirs[od][1]);Normalize2(&dir);
        for(int rot=0;rot<rotate_time;++rot){
            bool found=false;
            for(int radius=2;radius<=kMaxSearchRadius;radius=min(radius*2,radius+25)){
                const float2 test=make_float2(p.x+dir.x*radius,p.y+dir.y*radius);
                if(test.x<0||test.y<0||test.x>=ctx->width||test.y>=ctx->height) break;
                for(int trial=0;trial<4;++trial){
                    const int rx=(curand(rng)&1u)?1:-1;
                    const int ry=(curand(rng)&1u)?1:-1;
                    const int xshift=rx*static_cast<int>(curand(rng)%shift_range);
                    const int yshift=ry*static_cast<int>(curand(rng)%shift_range);
                    float2 probe=make_float2(dir.x*20.0f+xshift,dir.y*20.0f+yshift);Normalize2(&probe);
                    short2 q=make_short2(static_cast<short>(p.x+probe.x*radius),
                                         static_cast<short>(p.y+probe.y*radius));
                    if(!InImage(make_int2(q.x,q.y),ctx->width,ctx->height,6)) continue;
                    int qid=q.y*ctx->width+q.x;
                    if(ctx->state.reliability[qid]!=STRONG) q=ctx->state.nearest_strong[qid];
                    if(q.x<0||q.y<0) continue;
                    float2 td=make_float2(q.x-p.x,q.y-p.y);Normalize2(&td);
                    if(td.x*dir.x+td.y*dir.y>cos_thresh && CandidateEdgeAllowed(p,q,edge_limit,ctx)){
                        AddCandidate(q,candidates,&count); found=true; break;
                    }
                }
                if(found) break;
            }
            const float x=dir.x*cang-dir.y*sang, y=dir.x*sang+dir.y*cang;
            dir=make_float2(x,y);Normalize2(&dir);
        }
    }

    // Perception Range Expansion (paper Eq. 4). This intentionally uses the corrected clamp:
    // step = max(1, min(2*eta-1, floor(2*eta*D/(D+D_opposite)))).
    if(ctx->params->use_label && ctx->guidance.region_labels && ctx->guidance.region_labels[center]>0){
        const int dirs[8][2]={{0,-1},{0,1},{-1,0},{1,0},{-1,-1},{1,1},{-1,1},{1,-1}};
        float dist[8]={0}; int nsteps[8]={0};
        for(int i=0;i<8;++i){
            const short2 b=ctx->guidance.region_boundaries[map*8+i];
            if(b.x>=0){dist[i]=hypotf(p.x-b.x,p.y-b.y);if(i>=4)dist[i]/=sqrtf(2.0f);}
            if(i%2==1){
                const float denom=dist[i]+dist[i-1];
                int s=1;
                if(denom>1e-6f) s=static_cast<int>(2*rotate_time*dist[i]/denom);
                s=max(1,min(2*rotate_time-1,s));
                nsteps[i]=s; nsteps[i-1]=2*rotate_time-s;
            }
        }
        const int label=ctx->guidance.region_labels[center];
        for(int i=0;i<8;++i){
            const int gap=nsteps[i]+1;
            const int step_len=max(1,static_cast<int>(floorf(dist[i]/gap)));
            for(int s=1;s<=nsteps[i];++s){
                short2 q=make_short2(p.x+s*step_len*dirs[i][0],p.y+s*step_len*dirs[i][1]);
                if(!InImage(make_int2(q.x,q.y),ctx->width,ctx->height,6)) continue;
                int qid=q.y*ctx->width+q.x;
                if(ctx->state.reliability[qid]!=STRONG) q=ctx->state.nearest_strong[qid];
                if(q.x<0) continue;
                qid=q.y*ctx->width+q.x;
                const int qlabel=ctx->guidance.region_labels[qid];
                if(qlabel!=0&&qlabel!=label) continue;
                AddCandidate(q,candidates,&count);
            }
        }
    }

    if(count<=3){ctx->state.weak_reliable[center]=0;return;}

    // First RANSAC: candidate set -> robust anchor set.  This keeps the
    // original DPE selection policy (valid-hypothesis budget, normal-priority,
    // dynamic residual threshold and center-depth tie break) while exposing it
    // as an isolated anchor-construction stage.
    const Camera& cam=ctx->cameras[0];
    const float ransac_threshold=ctx->params->ransac_threshold*(ctx->params->depth_max-ctx->params->depth_min);
    float3 xyz[64]; float3 normals[64];
    for(int i=0;i<count;++i){
        const short2 q=candidates[i]; const int qid=q.y*ctx->width+q.x;
        const float d=ComputeDepthFromPlane(cam,ctx->state.planes[qid],make_int2(q.x,q.y));
        xyz[i]=PointFromDepth(cam,q,d);
        const float4 n4=ctx->state.planes[qid]; normals[i]=make_float3(n4.x,n4.y,n4.z);
    }
    const float center_depth=ComputeDepthFromPlane(cam,ctx->state.planes[center],p);
    const float3 center_xyz=make_float3(center_depth*(p.x-cam.K[2])/cam.K[0],
                                        center_depth*(p.y-cam.K[5])/cam.K[4],center_depth);

    int valid_hypotheses=50;
    int max_trials=ctx->params->high_res_img?200:125;
    int best_inliers=3;
    float best_center_error=FLT_MAX;
    float4 best_plane=make_float4(0,0,0,0);
    bool has_valid_plane=false;
    bool has_consistent_normal_plane=false;
    bool must_in_triangle=!(ctx->params->use_label && ctx->guidance.region_labels &&
                            ctx->guidance.region_labels[center]>0 && edge_limit);
    float temp_threshold=ransac_threshold;

    while(valid_hypotheses>0 && max_trials>0){
        --max_trials;
        const int a=curand(rng)%count,b=curand(rng)%count,c=curand(rng)%count;
        if(a==b||a==c||b==c) continue;
        if(must_in_triangle && !PointInTriangle(candidates[a],candidates[b],candidates[c],p)) continue;
        if(edge_limit && (EdgeCrossing(make_int2(candidates[a].x,candidates[a].y),make_int2(candidates[b].x,candidates[b].y),ctx)||
                          EdgeCrossing(make_int2(candidates[b].x,candidates[b].y),make_int2(candidates[c].x,candidates[c].y),ctx)||
                          EdgeCrossing(make_int2(candidates[c].x,candidates[c].y),make_int2(candidates[a].x,candidates[a].y),ctx))) continue;

        bool normal_consistency=false;
        if(ctx->params->geom_consistency&&edge_limit){
            normal_consistency=Dot3(normals[a],normals[b])>=0.8660254f &&
                               Dot3(normals[a],normals[c])>=0.8660254f &&
                               Dot3(normals[b],normals[c])>=0.8660254f;
            if(has_consistent_normal_plane&&!normal_consistency) continue;
        }
        --valid_hypotheses;

        float3 n=Cross3(make_float3(xyz[a].x-xyz[c].x,xyz[a].y-xyz[c].y,xyz[a].z-xyz[c].z),
                        make_float3(xyz[b].x-xyz[c].x,xyz[b].y-xyz[c].y,xyz[b].z-xyz[c].z));
        if(Dot3(n,n)<1e-12f||!isfinite(n.x)||!isfinite(n.y)||!isfinite(n.z)) continue;
        Normalize3(&n);
        const float w=-(n.x*xyz[a].x+n.y*xyz[a].y+n.z*xyz[a].z);

        int inliers=0;
        float residuals[64];
        for(int j=0;j<count;++j){
            const short2 q=candidates[j];
            const float fx=(q.x-cam.K[2])/cam.K[0],fy=(q.y-cam.K[5])/cam.K[4];
            const float denom=n.x*fx+n.y*fy+n.z;
            const float d=fabsf(denom)>1e-12f?-w/denom:1e30f;
            const float e=fabsf(d-xyz[j].z);
            residuals[j]=e;
            if(e<temp_threshold) ++inliers;
        }
        if(inliers<6) continue;

        const float center_denom=n.x*(p.x-cam.K[2])/cam.K[0]+n.y*(p.y-cam.K[5])/cam.K[4]+n.z;
        const float fit_center_depth=fabsf(center_denom)>1e-12f?-w/center_denom:1e30f;
        const float center_error=fabsf(fit_center_depth-center_xyz.z);

        if(inliers>best_inliers){
            if(!must_in_triangle&&PointInTriangle(candidates[a],candidates[b],candidates[c],p)) must_in_triangle=true;
            if(!has_consistent_normal_plane&&normal_consistency) has_consistent_normal_plane=true;
            best_plane=make_float4(n.x,n.y,n.z,w);
            best_inliers=inliers;
            best_center_error=center_error;
            has_valid_plane=true;

            const float floor_threshold=ctx->params->high_res_img?0.05f:0.005f;
            if(temp_threshold>floor_threshold && count>kNeighbourNum){
                float sorted[64]; for(int j=0;j<count;++j) sorted[j]=residuals[j];
                SortSmall(sorted,count);
                if(temp_threshold>=sorted[kNeighbourNum]){
                    temp_threshold=sorted[kNeighbourNum]-1e-6f;
                    int kept=0;for(int j=0;j<count;++j)if(sorted[j]<temp_threshold)++kept;
                    best_inliers=kept;
                }
            }
        }else if(inliers==best_inliers&&center_error<best_center_error){
            if(!must_in_triangle&&PointInTriangle(candidates[a],candidates[b],candidates[c],p)) must_in_triangle=true;
            best_plane=make_float4(n.x,n.y,n.z,w);
            best_center_error=center_error;
            has_valid_plane=true;
        }
    }
    if(!has_valid_plane){ctx->state.weak_reliable[center]=0;return;}

    float residual[64];
    for(int i=0;i<count;++i){
        const short2 q=candidates[i];
        const float fx=(q.x-cam.K[2])/cam.K[0],fy=(q.y-cam.K[5])/cam.K[4];
        const float denom=best_plane.x*fx+best_plane.y*fy+best_plane.z;
        const float fd=fabsf(denom)>1e-12f?-best_plane.w/denom:1e30f;
        const float e=fabsf(fd-xyz[i].z);
        residual[i]=(e<ransac_threshold)?e:FLT_MAX;
    }
    bool used[64]={false};
    for(int k=1;k<kNeighbourNum;++k){
        int best=-1;float e=FLT_MAX;
        for(int i=0;i<count;++i)if(!used[i]&&residual[i]<e){best=i;e=residual[i];}
        if(best<0) break;
        used[best]=true; out[k]=candidates[best];
    }
    ctx->state.weak_reliable[center]=1;
}

__global__ void UpdateUnreliableWeakKernel(DPEGpuContext* ctx){
    const int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    if(!InImage(p,ctx->width,ctx->height)) return;
    const int center=p.y*ctx->width+p.x;
    if(ctx->state.reliability[center]==WEAK && ctx->state.weak_reliable[center]!=1)
        ctx->state.reliability[center]=UNKNOWN;
}

}  // namespace dpe
