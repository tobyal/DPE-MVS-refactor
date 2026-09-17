#pragma once

#include "dpe/modules/anchor_search.cuh"

namespace dpe {

// Second RANSAC stage in DPE: the anchor set is fixed by GenerateAnchorsKernel,
// then the freshly updated anchor hypotheses are used to construct a plane for
// the weak pixel. This is intentionally separate from anchor-selection RANSAC.
__global__ void ConstructWeakPlaneKernel(DPEGpuContext* ctx){
    const int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    if(!InImage(p,ctx->width,ctx->height))return;
    const int center=p.y*ctx->width+p.x;

    if(ctx->state.reliability[center]!=WEAK){
        ctx->state.fitted_planes[center]=ctx->state.planes[center];
        return;
    }

    const int map=ctx->state.anchor_map[center];
    if(map<0||!ctx->state.weak_reliable[center]){
        ctx->state.fitted_planes[center]=make_float4(0,0,0,0);
        ctx->state.radius[center]=ctx->params->strong_radius;
        return;
    }

    short2 pts[kNeighbourNum-1];
    float3 xyz[kNeighbourNum-1];
    float3 normals[kNeighbourNum-1];
    int n=0;
    const Camera& cam=ctx->cameras[0];
    for(int i=1;i<kNeighbourNum;++i){
        const short2 q=ctx->state.anchors[map*kNeighbourNum+i];
        if(q.x<0||q.y<0)continue;
        const int qi=q.y*ctx->width+q.x;
        if(ctx->state.reliability[qi]!=STRONG)continue;
        const float d=ComputeDepthFromPlane(cam,ctx->state.planes[qi],make_int2(q.x,q.y));
        pts[n]=q;
        xyz[n]=PointFromDepth(cam,q,d);
        const float4 nn=ctx->state.planes[qi];
        normals[n]=make_float3(nn.x,nn.y,nn.z);
        ++n;
    }
    if(n<3){
        ctx->state.fitted_planes[center]=ctx->state.planes[center];
        ctx->state.radius[center]=ctx->params->strong_radius;
        return;
    }

    bool edge_limit=ctx->params->use_limit;
    curandState* rng=&ctx->state.random_states[center];
    if(edge_limit&&ctx->params->use_edge){
        const float r=curand_uniform(rng)-FLT_EPSILON;
        if(r<ctx->guidance.texture_complexity[center]) edge_limit=false;
    }

    // DPE relaxes the center-in-triangle requirement inside a detected coarse
    // region while edge constraints are active; once a valid enclosing triangle
    // is encountered, the requirement becomes active again.
    bool must_in_triangle=!(ctx->params->use_label && ctx->guidance.region_labels &&
                            ctx->guidance.region_labels[center]>0 && edge_limit);
    bool has_strong_plane=false;
    float min_cost=FLT_MAX;
    float4 best=make_float4(0,0,0,0);
    bool has_best=false;
    int best_a=-1,best_b=-1,best_c=-1;

    for(int iter=0;iter<50;++iter){
        const int a=curand(rng)%n;
        const int b=curand(rng)%n;
        const int c=curand(rng)%n;
        if(a==b||a==c||b==c)continue;

        bool strong_plane=false;
        if(ctx->params->geom_consistency&&edge_limit){
            strong_plane=Dot3(normals[a],normals[b])>=0.8660254f &&
                         Dot3(normals[a],normals[c])>=0.8660254f &&
                         Dot3(normals[b],normals[c])>=0.8660254f;
            if(has_strong_plane&&!strong_plane)continue;
        }

        if(must_in_triangle&&!PointInTriangle(pts[a],pts[b],pts[c],p))continue;
        if(edge_limit&&(
            EdgeCrossing(make_int2(pts[a].x,pts[a].y),make_int2(pts[b].x,pts[b].y),ctx)||
            EdgeCrossing(make_int2(pts[b].x,pts[b].y),make_int2(pts[c].x,pts[c].y),ctx)||
            EdgeCrossing(make_int2(pts[c].x,pts[c].y),make_int2(pts[a].x,pts[a].y),ctx)))continue;

        float3 normal=Cross3(
            make_float3(xyz[a].x-xyz[c].x,xyz[a].y-xyz[c].y,xyz[a].z-xyz[c].z),
            make_float3(xyz[b].x-xyz[c].x,xyz[b].y-xyz[c].y,xyz[b].z-xyz[c].z));
        if(Dot3(normal,normal)<1e-12f||!isfinite(normal.x)||!isfinite(normal.y)||!isfinite(normal.z))continue;
        Normalize3(&normal);
        const float w=-(normal.x*xyz[a].x+normal.y*xyz[a].y+normal.z*xyz[a].z);

        if(!has_strong_plane&&strong_plane)has_strong_plane=true;

        // The official second RANSAC ranks a hypothesis by summed depth residual
        // on the remaining anchors, rather than by an inlier-count threshold.
        float cost=0.0f;
        for(int j=0;j<n;++j){
            if(j==a||j==b||j==c)continue;
            const float fx=(pts[j].x-cam.K[2])/cam.K[0];
            const float fy=(pts[j].y-cam.K[5])/cam.K[4];
            const float denom=normal.x*fx+normal.y*fy+normal.z;
            if(fabsf(denom)<1e-12f){cost=FLT_MAX;break;}
            const float d=-w/denom;
            cost+=fabsf(d-xyz[j].z);
        }
        if(cost<min_cost){
            if(!must_in_triangle&&PointInTriangle(pts[a],pts[b],pts[c],p))must_in_triangle=true;
            min_cost=cost;
            best=make_float4(normal.x,normal.y,normal.z,w);
            best_a=a;best_b=b;best_c=c;
            has_best=true;
        }
    }

    if(!has_best){
        ctx->state.fitted_planes[center]=make_float4(0,0,0,0);
        ctx->state.radius[center]=ctx->params->strong_radius;
        return;
    }

    // Orient the fitted normal toward the reference camera, matching DPE.
    const float4 view=ViewDirection(cam,p);
    if(best.x*view.x+best.y*view.y+best.z*view.z>0.0f){
        best.x=-best.x;best.y=-best.y;best.z=-best.z;best.w=-best.w;
    }
    ctx->state.fitted_planes[center]=best;
    TelemetryPlane(ctx,p,best);

    if(!ctx->params->use_radius){
        TelemetryRadiusViolation(ctx,p,ctx->params->strong_radius);
        return;
    }
    if(!must_in_triangle||best_a<0||best_b<0||best_c<0){
        ctx->state.radius[center]=ctx->params->strong_radius;
        return;
    }

    const short2 A=pts[best_a],B=pts[best_b],C=pts[best_c];
    const float a=hypotf(A.x-B.x,A.y-B.y);
    const float b=hypotf(B.x-C.x,B.y-C.y);
    const float c=hypotf(C.x-A.x,C.y-A.y);
    const float sem=(a+b+c)*0.5f;
    const float area=sqrtf(fmaxf(0.0f,sem*(sem-a)*(sem-b)*(sem-c)));
    int radius=static_cast<int>(floorf(sqrtf(area)/2.0f));

    const float da=hypotf(A.x-p.x,A.y-p.y);
    const float db=hypotf(B.x-p.x,B.y-p.y);
    const float dc=hypotf(C.x-p.x,C.y-p.y);
    const float min_anchor=fminf(da,fminf(db,dc));
    if(2.5f*min_anchor<radius)radius=static_cast<int>(min_anchor);

    if(edge_limit){
        if(ctx->params->use_edge){
            float min_edge=FLT_MAX;
            for(int d=0;d<8;++d){
                const short2 e=ctx->guidance.nearest_edges[center*8+d];
                if(e.x>=0&&e.y>=0)min_edge=fminf(min_edge,hypotf(e.x-p.x,e.y-p.y));
            }
            if(min_edge<radius)radius=static_cast<int>(min_edge);
        }
        if(ctx->params->use_label&&ctx->guidance.region_labels&&ctx->guidance.region_labels[center]>0){
            float min_boundary=FLT_MAX;
            for(int d=0;d<8;++d){
                const short2 q=ctx->guidance.region_boundaries[map*8+d];
                if(q.x>=0&&q.y>=0)min_boundary=fminf(min_boundary,hypotf(q.x-p.x,q.y-p.y));
            }
            if(min_boundary<radius)radius=static_cast<int>(min_boundary);
        }
    }

    while(radius>0&&((radius<<1)%5)!=0)--radius;
    if(!edge_limit){
        ctx->state.radius[center]=radius>ctx->params->strong_radius?0:ctx->params->strong_radius;
    }else{
        ctx->state.radius[center]=radius>ctx->params->strong_radius?radius:ctx->params->strong_radius;
    }
    TelemetryRadiusViolation(ctx,p,ctx->state.radius[center]);

}

}  // namespace dpe
