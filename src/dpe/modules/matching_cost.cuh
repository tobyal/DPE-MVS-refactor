#pragma once

#include "dpe/modules/device_math.cuh"

namespace dpe {

__device__ inline float BilateralWeight(float dx, float dy, float pix, float center,
                                        float sigma_spatial, float sigma_color) {
    const float spatial = sqrtf(dx*dx+dy*dy);
    const float color = fabsf(pix-center);
    return expf(-spatial/(2.0f*sigma_spatial*sigma_spatial)
                -color/(2.0f*sigma_color*sigma_color));
}

__device__ inline short2 AnchorPoint(const DPEGpuContext* ctx, int2 p, int k) {
    const int center=p.y*ctx->width+p.x;
    const int map=ctx->state.anchor_map[center];
    if (map < 0) return make_short2(-1,-1);
    return ctx->state.anchors[map*kNeighbourNum+k];
}

__device__ inline float PatchNccAt(const DPEGpuContext* ctx,
                                   int2 patch_center,
                                   int2 bilateral_center,
                                   int src_idx,
                                   const float H[9],
                                   int radius,
                                   int increment) {
    const auto ref_tex=ctx->image_textures->images[0];
    const auto src_tex=ctx->image_textures->images[src_idx];
    const float center_pix=tex2D<float>(ref_tex, bilateral_center.x+0.5f, bilateral_center.y+0.5f);

    float sr=0, ss=0, srr=0, sss=0, srs=0, sw=0;
    for (int dy=-radius; dy<=radius; dy+=increment) {
        for (int dx=-radius; dx<=radius; dx+=increment) {
            const int2 rp=make_int2(patch_center.x+dx, patch_center.y+dy);
            const float rv=tex2D<float>(ref_tex,rp.x+0.5f,rp.y+0.5f);
            const float2 sp=WarpPoint(H,rp);
            // Once the patch center is valid, the reference implementation
            // relies on CUDA texture clamp behavior for border samples.
            const float sv=tex2D<float>(src_tex,sp.x+0.5f,sp.y+0.5f);
            const float w=BilateralWeight(dx,dy,rv,center_pix,
                                          ctx->params->sigma_spatial,ctx->params->sigma_color);
            sr+=w*rv; ss+=w*sv; srr+=w*rv*rv; sss+=w*sv*sv; srs+=w*rv*sv; sw+=w;
        }
    }
    if (sw<=1e-12f) return 2.0f;
    const float inv=1.0f/sw;
    sr*=inv; ss*=inv; srr*=inv; sss*=inv; srs*=inv;
    const float vr=srr-sr*sr, vs=sss-ss*ss;
    if (vr<1e-5f||vs<1e-5f) return 2.0f;
    const float cov=srs-sr*ss;
    return fminf(2.0f,fmaxf(0.0f,1.0f-cov/sqrtf(vr*vs)));
}

__device__ inline float StandardPatchCost(int2 p, int src_idx, const float4& plane,
                                          const DPEGpuContext* ctx) {
    float H[9]; ComputeHomography(ctx->cameras[0],ctx->cameras[src_idx],plane,H);
    const float2 q=WarpPoint(H,p);
    const Camera& src_cam=ctx->cameras[src_idx];
    if (q.x<0||q.y<0||q.x>=src_cam.width||q.y>=src_cam.height) return 2.0f;
    return PatchNccAt(ctx,p,p,src_idx,H,ctx->params->strong_radius,ctx->params->strong_increment);
}

__device__ inline float DeformablePatchCost(int2 p, int src_idx, const float4& plane,
                                            const DPEGpuContext* ctx) {
    const int center=p.y*ctx->width+p.x;
    if (ctx->state.reliability[center] != WEAK) return StandardPatchCost(p,src_idx,plane,ctx);

    float H[9]; ComputeHomography(ctx->cameras[0],ctx->cameras[src_idx],plane,H);
    float center_cost=2.0f, anchor_sum=0.0f;
    int anchor_count=0;
    for (int k=0;k<kNeighbourNum;++k) {
        const short2 a=AnchorPoint(ctx,p,k);
        if (a.x<0||a.y<0) continue;
        const int2 ap=make_int2(a.x,a.y);
        const float2 q=WarpPoint(H,ap);
        const Camera& src_cam=ctx->cameras[src_idx];
        if (q.x<0||q.y<0||q.x>=src_cam.width||q.y>=src_cam.height) {
            if (k==0) return 2.0f;
            const unsigned int views=ctx->state.selected_views[ap.y*ctx->width+ap.x];
            if (BitSet(views,src_idx-1)) { anchor_sum+=2.0f; ++anchor_count; }
            continue;
        }
        int radius=(k==0?ctx->params->strong_radius:ctx->params->weak_radius);
        int inc=(k==0?ctx->params->strong_increment:ctx->params->weak_increment);
        if (k==0 && ctx->params->use_radius) {
            radius=ctx->state.radius[center];
            inc=max(2,static_cast<int>(2.0f*radius/5.0f));
        }
        // DPE's deformable NCC keeps the weak center pixel as the bilateral
        // color reference even when the spatial patch is centered on an anchor.
        const float c=PatchNccAt(ctx,ap,p,src_idx,H,radius,inc);
        if (k==0) center_cost=c;
        else { anchor_sum+=c; ++anchor_count; }
    }
    if (anchor_count==0) return center_cost;
    return 0.25f*center_cost+0.75f*fminf(2.0f,anchor_sum/anchor_count);
}

__device__ inline void MultiViewCostVector(int2 p, const float4& plane, bool deformable,
                                          float* costs, const DPEGpuContext* ctx) {
    for (int i=1;i<ctx->num_images;++i)
        costs[i-1]=deformable?DeformablePatchCost(p,i,plane,ctx):StandardPatchCost(p,i,plane,ctx);
}

__device__ inline float WeightedCost(const float* vector, const unsigned char* weights, int n) {
    float total=0.0f, norm=0.0f;
    for(int i=0;i<n;++i){ if(weights[i]){ total+=weights[i]*vector[i]; norm+=weights[i]; } }
    return norm>0?total/norm:2.0f;
}

}  // namespace dpe
