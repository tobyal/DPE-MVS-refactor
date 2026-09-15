#pragma once

#include "dpe/modules/device_math.cuh"

namespace dpe {

__device__ inline float3 CameraPoint(const Camera& cam,int2 p,float depth){
    return make_float3(depth*(p.x-cam.K[2])/cam.K[0],depth*(p.y-cam.K[5])/cam.K[4],depth);
}
__device__ inline float3 CameraPoint(const Camera& cam,float x,float y,float depth){
    return make_float3(depth*(x-cam.K[2])/cam.K[0],depth*(y-cam.K[5])/cam.K[4],depth);
}
__device__ inline float3 CameraToWorld(const Camera& cam,const float3& X){
    const float3 v=make_float3(X.x-cam.t[0],X.y-cam.t[1],X.z-cam.t[2]);
    return make_float3(cam.R[0]*v.x+cam.R[3]*v.y+cam.R[6]*v.z,
                       cam.R[1]*v.x+cam.R[4]*v.y+cam.R[7]*v.z,
                       cam.R[2]*v.x+cam.R[5]*v.y+cam.R[8]*v.z);
}
__device__ inline float3 WorldToCamera(const Camera& cam,const float3& X){
    return make_float3(cam.R[0]*X.x+cam.R[1]*X.y+cam.R[2]*X.z+cam.t[0],
                       cam.R[3]*X.x+cam.R[4]*X.y+cam.R[5]*X.z+cam.t[1],
                       cam.R[6]*X.x+cam.R[7]*X.y+cam.R[8]*X.z+cam.t[2]);
}
__device__ inline float2 Project(const Camera& cam,const float3& X){
    const float z=cam.K[6]*X.x+cam.K[7]*X.y+cam.K[8]*X.z;
    if(fabsf(z)<1e-12f)return make_float2(-1e6f,-1e6f);
    return make_float2((cam.K[0]*X.x+cam.K[1]*X.y+cam.K[2]*X.z)/z,
                       (cam.K[3]*X.x+cam.K[4]*X.y+cam.K[5]*X.z)/z);
}

__device__ inline float GeometryConsistencyCost(int2 p,int src_idx,const float4& plane,
                                                const DPEGpuContext* ctx){
    if(!ctx->depth_textures) return 0.0f;
    const Camera& ref=ctx->cameras[0];
    const Camera& src=ctx->cameras[src_idx];
    const float d=ComputeDepthFromPlane(ref,plane,p);
    if(d<=0) return 3.0f;
    const float3 Xw=CameraToWorld(ref,CameraPoint(ref,p,d));
    const float3 Xs=WorldToCamera(src,Xw);
    if(Xs.z<=0) return 3.0f;
    const float2 q=Project(src,Xs);
    if(q.x<0||q.y<0||q.x>=src.width||q.y>=src.height) return 3.0f;
    const int qx=static_cast<int>(q.x),qy=static_cast<int>(q.y);
    const float sd=tex2D<float>(ctx->depth_textures->images[src_idx],qx+0.5f,qy+0.5f);
    if(sd<=0) return 3.0f;
    const float3 Xw2=CameraToWorld(src,CameraPoint(src,q.x,q.y,sd));
    const float3 Xr2=WorldToCamera(ref,Xw2);
    if(Xr2.z<=0) return 3.0f;
    const float2 rp=Project(ref,Xr2);
    const float err=hypotf(rp.x-p.x,rp.y-p.y);
    return fminf(3.0f,err);
}

}  // namespace dpe
