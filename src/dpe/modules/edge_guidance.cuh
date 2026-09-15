#pragma once

#include "dpe/modules/device_math.cuh"

namespace dpe {

__device__ inline bool EdgeCrossing(int2 A,int2 B,const DPEGpuContext* ctx){
    if(!ctx->guidance.lowres_edges) return false;

    const int full_a=A.y*ctx->width+A.x;
    const int full_b=B.y*ctx->width+B.x;
    // Preserve the official DPE semantics: an endpoint lying on a fine edge is
    // not itself treated as a crossed edge segment.
    if(ctx->guidance.fine_edges &&
       (ctx->guidance.fine_edges[full_a] || ctx->guidance.fine_edges[full_b])) return false;

    const float scale_x=ctx->guidance.low_width/static_cast<float>(ctx->width);
    const float scale_y=ctx->guidance.low_height/static_cast<float>(ctx->height);
    const int max_step=ctx->params->high_res_img
        ? max(1,static_cast<int>(roundf(max(ctx->guidance.low_width,ctx->guidance.low_height)/60.0f)))
        : max(ctx->guidance.low_width,ctx->guidance.low_height);

    // The official implementation scans from both endpoints. On high-resolution
    // images each scan is deliberately truncated, so both directions matter.
    for(int pass=0;pass<2;++pass){
        const int2 from=pass==0?A:B;
        const int2 to=pass==0?B:A;
        int x0=min(ctx->guidance.low_width-1,max(0,static_cast<int>(roundf(from.x*scale_x))));
        int y0=min(ctx->guidance.low_height-1,max(0,static_cast<int>(roundf(from.y*scale_y))));
        const int x1=min(ctx->guidance.low_width-1,max(0,static_cast<int>(roundf(to.x*scale_x))));
        const int y1=min(ctx->guidance.low_height-1,max(0,static_cast<int>(roundf(to.y*scale_y))));
        const int dx=abs(x1-x0), sx=x0<x1?1:-1;
        const int dy=abs(y1-y0), sy=y0<y1?1:-1;
        int err=(dx>dy?dx:dy)/2;
        bool tagx=true,tagy=true;
        int step=0;
        while(tagx||tagy){
            if(x0==x1) tagx=false;
            if(y0==y1) tagy=false;
            const int e2=err;
            if(e2>-dx){err-=dy;x0+=sx;}
            if(e2<dy){err+=dx;y0+=sy;}
            if(x0<0||y0<0||x0>=ctx->guidance.low_width||y0>=ctx->guidance.low_height) break;
            if(ctx->guidance.lowres_edges[y0*ctx->guidance.low_width+x0]) return true;
            if(++step>=max_step) break;
        }
    }
    return false;
}

__global__ void GenerateGuidanceKernel(DPEGpuContext* ctx){
    const int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    if(!InImage(p,ctx->width,ctx->height)) return;
    const int center=p.y*ctx->width+p.x;
    const int dirs[8][2]={{0,-1},{0,1},{-1,0},{1,0},{-1,-1},{1,1},{-1,1},{1,-1}};
    const int edge_offset=center*8;
    for(int d=0;d<8;++d){
        short2 hit=make_short2(-1,-1);
        for(int s=1;;++s){
            const int2 q=make_int2(p.x+s*dirs[d][0],p.y+s*dirs[d][1]);
            if(!InImage(q,ctx->width,ctx->height)) break;
            if(ctx->guidance.fine_edges && ctx->guidance.fine_edges[q.y*ctx->width+q.x]){hit=make_short2(q.x,q.y);break;}
        }
        ctx->guidance.nearest_edges[edge_offset+d]=hit;
    }

    int edge_count=0,total=0,bound_count=0;
    const int r=max(1,ctx->params->strong_radius);
    for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx){
        const int2 q=make_int2(p.x+dx,p.y+dy);
        if(!InImage(q,ctx->width,ctx->height)) continue;
        ++total;
        const int qid=q.y*ctx->width+q.x;
        if(ctx->guidance.fine_edges && ctx->guidance.fine_edges[qid]) ++edge_count;
        if(ctx->guidance.region_labels && ctx->guidance.region_labels[qid]==0) ++bound_count;
    }
    const float density=total?fmaxf(edge_count/static_cast<float>(total),bound_count/static_cast<float>(total)):0.0f;
    ctx->guidance.texture_complexity[center]=1.0f/(1.0f+expf(-25.0f*(density-0.35f)));

    if(ctx->state.reliability[center]==WEAK && ctx->guidance.region_labels && ctx->guidance.region_labels[center]>0){
        const int map=ctx->state.anchor_map[center];
        if(map>=0){
            const int label=ctx->guidance.region_labels[center];
            for(int d=0;d<8;++d){
                short2 last=make_short2(-1,-1);
                for(int s=1;;++s){
                    const int2 q=make_int2(p.x+s*dirs[d][0],p.y+s*dirs[d][1]);
                    if(!InImage(q,ctx->width,ctx->height)) break;
                    const int l=ctx->guidance.region_labels[q.y*ctx->width+q.x];
                    if(l!=label) break;
                    last=make_short2(q.x,q.y);
                }
                ctx->guidance.region_boundaries[map*8+d]=last;
            }
        }
    }
}

__global__ void FindNearestStrongKernel(DPEGpuContext* ctx){
    const int2 p=make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    if(!InImage(p,ctx->width,ctx->height)) return;
    const int center=p.y*ctx->width+p.x;
    ctx->state.nearest_strong[center]=make_short2(-1,-1);
    if(ctx->state.reliability[center]!=WEAK) return;

    // Same square-ring search used by the official implementation (ETH: radius 100).
    constexpr int max_radius=100;
    for(int radius=0;radius<=max_radius;++radius){
        for(int dx=-radius;dx<=radius;++dx){
            for(int dy=-radius;dy<=radius;++dy){
                if(abs(dx)!=radius&&abs(dy)!=radius) continue;
                const int2 q=make_int2(p.x+dx,p.y+dy);
                if(!InImage(q,ctx->width,ctx->height)) continue;
                if(ctx->state.reliability[q.y*ctx->width+q.x]==STRONG){
                    ctx->state.nearest_strong[center]=make_short2(q.x,q.y);
                    return;
                }
            }
        }
    }
}

}  // namespace dpe
