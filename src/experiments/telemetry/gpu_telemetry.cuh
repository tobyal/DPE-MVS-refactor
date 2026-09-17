#pragma once

#include "dpe/gpu_types.cuh"
#include "dpe/modules/device_math.cuh"

namespace dpe {

__device__ inline bool GtPixelValid(const DPEGpuContext* ctx, int idx) {
    return ctx->telemetry.enabled && ctx->telemetry.gt_valid && ctx->telemetry.gt_valid[idx] != 0;
}

__device__ inline bool SameGtSurface(const DPEGpuContext* ctx, int a, int b) {
    if (!ctx->telemetry.enabled || !ctx->telemetry.gt_surface_label) return false;
    const int la = ctx->telemetry.gt_surface_label[a];
    const int lb = ctx->telemetry.gt_surface_label[b];
    return la > 0 && la == lb;
}


__device__ inline void TelemetryStrongCandidates(const DPEGpuContext* ctx, int center, const int positions[8], const bool valid[8]) {
    if (!ctx->telemetry.enabled || !ctx->telemetry.es_candidate_count) return;
    int total = 0, same = 0;
    for (int i=0;i<8;++i) {
        if (!valid[i] || positions[i] < 0) continue;
        ++total;
        if (SameGtSurface(ctx, center, positions[i])) ++same;
    }
    atomicAdd(&ctx->telemetry.es_candidate_count[center], total);
    atomicAdd(&ctx->telemetry.es_same_surface[center], same);
}

__device__ inline void TelemetryCandidate(const DPEGpuContext* ctx, int center, short2 q) {
    if (!ctx->telemetry.enabled || !ctx->telemetry.candidate_count) return;
    atomicAdd(&ctx->telemetry.candidate_count[center], 1);
    if (q.x < 0 || q.y < 0 || q.x >= ctx->width || q.y >= ctx->height) return;
    const int qid = q.y * ctx->width + q.x;
    if (SameGtSurface(ctx, center, qid) && ctx->telemetry.same_surface_candidates)
        atomicAdd(&ctx->telemetry.same_surface_candidates[center], 1);
}

__device__ inline void TelemetryAnchors(const DPEGpuContext* ctx, int center, int map) {
    if (!ctx->telemetry.enabled || !ctx->telemetry.anchor_count || map < 0) return;
    int total = 0, same = 0;
    for (int i = 1; i < kNeighbourNum; ++i) {
        const short2 q = ctx->state.anchors[map * kNeighbourNum + i];
        if (q.x < 0 || q.y < 0) continue;
        ++total;
        const int qid = q.y * ctx->width + q.x;
        if (SameGtSurface(ctx, center, qid)) ++same;
    }
    ctx->telemetry.anchor_count[center] = static_cast<unsigned char>(min(total, 255));
    ctx->telemetry.same_surface_anchors[center] = static_cast<unsigned char>(min(same, 255));
}

__device__ inline float AngleDeg(float3 a, float3 b) {
    const float na = sqrtf(fmaxf(1e-20f, Dot3(a, a)));
    const float nb = sqrtf(fmaxf(1e-20f, Dot3(b, b)));
    float d = Dot3(a, b) / (na * nb);
    d = fmaxf(-1.0f, fminf(1.0f, d));
    return acosf(d) * 57.29577951308232f;
}

__device__ inline void TelemetryPlane(const DPEGpuContext* ctx, int2 p, float4 plane) {
    if (!ctx->telemetry.enabled) return;
    const int center = p.y * ctx->width + p.x;
    if (!GtPixelValid(ctx, center)) return;
    if (ctx->telemetry.plane_depth_error && ctx->telemetry.gt_depth) {
        const float d = ComputeDepthFromPlane(ctx->cameras[0], plane, p);
        ctx->telemetry.plane_depth_error[center] = fabsf(d - ctx->telemetry.gt_depth[center]);
    }
    if (ctx->telemetry.plane_normal_error && ctx->telemetry.gt_normal) {
        const float3 est_cam = make_float3(plane.x, plane.y, plane.z);
        // plane normals inside PatchMatch are in reference-camera coordinates.
        const Camera& c = ctx->cameras[0];
        float3 est_world = make_float3(
            c.R[0]*est_cam.x + c.R[3]*est_cam.y + c.R[6]*est_cam.z,
            c.R[1]*est_cam.x + c.R[4]*est_cam.y + c.R[7]*est_cam.z,
            c.R[2]*est_cam.x + c.R[5]*est_cam.y + c.R[8]*est_cam.z);
        const float3 gt = ctx->telemetry.gt_normal[center];
        if (Dot3(gt, gt) > 1e-10f && Dot3(est_world, est_world) > 1e-10f)
            ctx->telemetry.plane_normal_error[center] = AngleDeg(est_world, gt);
    }
}

__device__ inline void TelemetryRadiusViolation(const DPEGpuContext* ctx, int2 p, int radius) {
    if (!ctx->telemetry.enabled || !ctx->telemetry.radius_violation || !ctx->telemetry.gt_surface_label) return;
    const int center = p.y * ctx->width + p.x;
    const int label = ctx->telemetry.gt_surface_label[center];
    if (label <= 0 || radius <= 0) return;
    int total = 0, wrong = 0;
    const int step = max(1, radius / 8);
    for (int dy = -radius; dy <= radius; dy += step) {
        for (int dx = -radius; dx <= radius; dx += step) {
            const int x = p.x + dx, y = p.y + dy;
            if (x < 0 || y < 0 || x >= ctx->width || y >= ctx->height) continue;
            ++total;
            const int q = y * ctx->width + x;
            if (ctx->telemetry.gt_surface_label[q] != label) ++wrong;
        }
    }
    if (total > 0) ctx->telemetry.radius_violation[center] = wrong / static_cast<float>(total);
}

__global__ void FinalTelemetryKernel(DPEGpuContext* ctx) {
    const int2 p = make_int2(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y);
    if (!InImage(p, ctx->width, ctx->height) || !ctx->telemetry.enabled) return;
    const int center = p.y * ctx->width + p.x;
    if (ctx->telemetry.matching_cost) ctx->telemetry.matching_cost[center] = ctx->state.costs[center];
    if (ctx->telemetry.adaptive_radius) ctx->telemetry.adaptive_radius[center] = ctx->state.radius[center];
    if (!GtPixelValid(ctx, center) || !ctx->telemetry.final_depth_error || !ctx->telemetry.gt_depth) return;
    const float d = ctx->state.planes[center].w; // after FinalizeDepthNormal: w stores depth
    if (d > 0.0f && isfinite(d)) ctx->telemetry.final_depth_error[center] = fabsf(d - ctx->telemetry.gt_depth[center]);
}

}  // namespace dpe
