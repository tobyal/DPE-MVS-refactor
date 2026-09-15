#pragma once

#include "dpe/gpu_types.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cmath>

namespace dpe {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ inline float Dot3(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ inline float Dot3(const float4& a, const float4& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ inline float3 Cross3(const float3& a, const float3& b) {
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ inline void Normalize3(float3* v) {
    const float inv = rsqrtf(v->x*v->x + v->y*v->y + v->z*v->z + 1e-20f);
    v->x*=inv; v->y*=inv; v->z*=inv;
}
__device__ inline void Normalize4(float4* v) {
    const float inv = rsqrtf(v->x*v->x + v->y*v->y + v->z*v->z + 1e-20f);
    v->x*=inv; v->y*=inv; v->z*=inv;
}
__device__ inline void Normalize2(float2* v) {
    const float inv = rsqrtf(v->x*v->x + v->y*v->y + 1e-20f);
    v->x*=inv; v->y*=inv;
}

__device__ inline bool InImage(int2 p, int w, int h, int margin=0) {
    return p.x >= margin && p.y >= margin && p.x < w-margin && p.y < h-margin;
}

__device__ inline float ComputeDepthFromPlane(const Camera& cam, const float4& plane, int2 p) {
    return -plane.w * cam.K[0] /
        ((p.x-cam.K[2])*plane.x + (cam.K[0]/cam.K[4])*(p.y-cam.K[5])*plane.y + cam.K[0]*plane.z);
}

__device__ inline float PlaneDistanceForDepth(const Camera& cam, int2 p, float depth, const float4& normal) {
    const float x = depth * (p.x - cam.K[2]) / cam.K[0];
    const float y = depth * (p.y - cam.K[5]) / cam.K[4];
    return -(normal.x*x + normal.y*y + normal.z*depth);
}

__device__ inline float4 ViewDirection(const Camera& cam, int2 p) {
    float4 v = make_float4((p.x-cam.K[2])/cam.K[0], (p.y-cam.K[5])/cam.K[4], 1.0f, 0.0f);
    Normalize4(&v);
    return v;
}

__device__ inline float4 RandomNormal(const Camera& cam, int2 p, curandState* rng) {
    float q1=1, q2=1, s=2;
    while (s >= 1.0f || s < 1e-12f) {
        q1 = 2.0f*curand_uniform(rng)-1.0f;
        q2 = 2.0f*curand_uniform(rng)-1.0f;
        s = q1*q1+q2*q2;
    }
    const float sq = sqrtf(1.0f-s);
    float4 n = make_float4(2*q1*sq, 2*q2*sq, 1.0f-2*s, 0.0f);
    const float4 v = ViewDirection(cam,p);
    if (Dot3(n,v)>0) { n.x=-n.x; n.y=-n.y; n.z=-n.z; }
    Normalize4(&n);
    return n;
}

__device__ inline float4 RandomPlane(const Camera& cam, int2 p, curandState* rng, float dmin, float dmax) {
    const float d = dmin + curand_uniform(rng)*(dmax-dmin);
    float4 plane = RandomNormal(cam,p,rng);
    plane.w = PlaneDistanceForDepth(cam,p,d,plane);
    return plane;
}

__device__ inline float4 PerturbNormal(const Camera& cam, int2 p, const float4& n,
                                      curandState* rng, float max_angle) {
    const float ax=(curand_uniform(rng)-0.5f)*max_angle;
    const float ay=(curand_uniform(rng)-0.5f)*max_angle;
    const float az=(curand_uniform(rng)-0.5f)*max_angle;
    const float sx=sinf(ax), cx=cosf(ax), sy=sinf(ay), cy=cosf(ay), sz=sinf(az), cz=cosf(az);
    float4 out;
    out.x=(cy*cz)*n.x + (cz*sx*sy-cx*sz)*n.y + (sx*sz+cx*cz*sy)*n.z;
    out.y=(cy*sz)*n.x + (cx*cz+sx*sy*sz)*n.y + (cx*sy*sz-cz*sx)*n.z;
    out.z=(-sy)*n.x + (cy*sx)*n.y + (cx*cy)*n.z;
    out.w=n.w;
    const float4 v=ViewDirection(cam,p);
    if (Dot3(out,v)>=0) return n;
    Normalize4(&out);
    return out;
}

__device__ inline float4 WorldNormalToCamera(const Camera& cam, float4 n) {
    return make_float4(cam.R[0]*n.x+cam.R[1]*n.y+cam.R[2]*n.z,
                       cam.R[3]*n.x+cam.R[4]*n.y+cam.R[5]*n.z,
                       cam.R[6]*n.x+cam.R[7]*n.y+cam.R[8]*n.z,
                       n.w);
}

__device__ inline float4 CameraNormalToWorld(const Camera& cam, float4 n) {
    return make_float4(cam.R[0]*n.x+cam.R[3]*n.y+cam.R[6]*n.z,
                       cam.R[1]*n.x+cam.R[4]*n.y+cam.R[7]*n.z,
                       cam.R[2]*n.x+cam.R[5]*n.y+cam.R[8]*n.z,
                       n.w);
}

__device__ inline void ComputeHomography(const Camera& ref, const Camera& src,
                                         const float4& plane, float H[9]) {
    float Rrel[9];
    Rrel[0]=src.R[0]*ref.R[0]+src.R[1]*ref.R[1]+src.R[2]*ref.R[2];
    Rrel[1]=src.R[0]*ref.R[3]+src.R[1]*ref.R[4]+src.R[2]*ref.R[5];
    Rrel[2]=src.R[0]*ref.R[6]+src.R[1]*ref.R[7]+src.R[2]*ref.R[8];
    Rrel[3]=src.R[3]*ref.R[0]+src.R[4]*ref.R[1]+src.R[5]*ref.R[2];
    Rrel[4]=src.R[3]*ref.R[3]+src.R[4]*ref.R[4]+src.R[5]*ref.R[5];
    Rrel[5]=src.R[3]*ref.R[6]+src.R[4]*ref.R[7]+src.R[5]*ref.R[8];
    Rrel[6]=src.R[6]*ref.R[0]+src.R[7]*ref.R[1]+src.R[8]*ref.R[2];
    Rrel[7]=src.R[6]*ref.R[3]+src.R[7]*ref.R[4]+src.R[8]*ref.R[5];
    Rrel[8]=src.R[6]*ref.R[6]+src.R[7]*ref.R[7]+src.R[8]*ref.R[8];

    const float Cr[3] = {
        -(ref.R[0]*ref.t[0]+ref.R[3]*ref.t[1]+ref.R[6]*ref.t[2]),
        -(ref.R[1]*ref.t[0]+ref.R[4]*ref.t[1]+ref.R[7]*ref.t[2]),
        -(ref.R[2]*ref.t[0]+ref.R[5]*ref.t[1]+ref.R[8]*ref.t[2])};
    const float Cs[3] = {
        -(src.R[0]*src.t[0]+src.R[3]*src.t[1]+src.R[6]*src.t[2]),
        -(src.R[1]*src.t[0]+src.R[4]*src.t[1]+src.R[7]*src.t[2]),
        -(src.R[2]*src.t[0]+src.R[5]*src.t[1]+src.R[8]*src.t[2])};
    const float dC[3]={Cr[0]-Cs[0],Cr[1]-Cs[1],Cr[2]-Cs[2]};
    const float tr[3]={src.R[0]*dC[0]+src.R[1]*dC[1]+src.R[2]*dC[2],
                       src.R[3]*dC[0]+src.R[4]*dC[1]+src.R[5]*dC[2],
                       src.R[6]*dC[0]+src.R[7]*dC[1]+src.R[8]*dC[2]};
    const float iw = 1.0f / plane.w;
    float A[9];
    A[0]=Rrel[0]-tr[0]*plane.x*iw; A[1]=Rrel[1]-tr[0]*plane.y*iw; A[2]=Rrel[2]-tr[0]*plane.z*iw;
    A[3]=Rrel[3]-tr[1]*plane.x*iw; A[4]=Rrel[4]-tr[1]*plane.y*iw; A[5]=Rrel[5]-tr[1]*plane.z*iw;
    A[6]=Rrel[6]-tr[2]*plane.x*iw; A[7]=Rrel[7]-tr[2]*plane.y*iw; A[8]=Rrel[8]-tr[2]*plane.z*iw;

    // H = Ks * A * Kr^-1 for pinhole K=[fx 0 cx;0 fy cy;0 0 1]
    const float ifx=1.0f/ref.K[0], ify=1.0f/ref.K[4];
    float B[9];
    B[0]=A[0]*ifx; B[1]=A[1]*ify; B[2]=A[2]-A[0]*ref.K[2]*ifx-A[1]*ref.K[5]*ify;
    B[3]=A[3]*ifx; B[4]=A[4]*ify; B[5]=A[5]-A[3]*ref.K[2]*ifx-A[4]*ref.K[5]*ify;
    B[6]=A[6]*ifx; B[7]=A[7]*ify; B[8]=A[8]-A[6]*ref.K[2]*ifx-A[7]*ref.K[5]*ify;
    H[0]=src.K[0]*B[0]+src.K[2]*B[6]; H[1]=src.K[0]*B[1]+src.K[2]*B[7]; H[2]=src.K[0]*B[2]+src.K[2]*B[8];
    H[3]=src.K[4]*B[3]+src.K[5]*B[6]; H[4]=src.K[4]*B[4]+src.K[5]*B[7]; H[5]=src.K[4]*B[5]+src.K[5]*B[8];
    H[6]=src.K[8]*B[6]; H[7]=src.K[8]*B[7]; H[8]=src.K[8]*B[8];
}

__device__ inline float2 WarpPoint(const float H[9], int2 p) {
    const float z=H[6]*p.x+H[7]*p.y+H[8];
    if (fabsf(z)<1e-12f) return make_float2(-1e6f,-1e6f);
    return make_float2((H[0]*p.x+H[1]*p.y+H[2])/z,
                       (H[3]*p.x+H[4]*p.y+H[5])/z);
}

__device__ inline bool BitSet(unsigned int mask, int bit) { return ((mask>>bit)&1u)!=0; }
__device__ inline void SetBit(unsigned int* mask, int bit) { *mask |= (1u<<bit); }
__device__ inline void UnsetBit(unsigned int* mask, int bit) { *mask &= ~(1u<<bit); }

__device__ inline bool PointInTriangle(short2 A, short2 B, short2 C, int2 P) {
    const float ab=hypotf(static_cast<float>(A.x-B.x),static_cast<float>(A.y-B.y));
    const float bc=hypotf(static_cast<float>(B.x-C.x),static_cast<float>(B.y-C.y));
    const float ca=hypotf(static_cast<float>(C.x-A.x),static_cast<float>(C.y-A.y));
    if(ab<=2.0f||bc<=2.0f||ca<=2.0f) return false;
    if(!(ab+bc>ca&&bc+ca>ab&&ab+ca>bc)) return false;
    const float2 a=make_float2(A.x-P.x,A.y-P.y), b=make_float2(B.x-P.x,B.y-P.y), c=make_float2(C.x-P.x,C.y-P.y);
    const float t1=a.x*b.y-a.y*b.x, t2=b.x*c.y-b.y*c.x, t3=c.x*a.y-c.y*a.x;
    return t1*t2>=0 && t1*t3>=0;
}

__device__ inline void SortSmall(float* x, int n) {
    for (int i=1;i<n;++i){ float v=x[i]; int j=i; while(j>0&&v<x[j-1]){x[j]=x[j-1];--j;} x[j]=v; }
}

}  // namespace dpe
