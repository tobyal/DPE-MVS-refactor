#include "scene/reconstruction_state.h"

namespace dpe {

bool ReconstructionState::Has(int image_id) const {
    return frames_.find(image_id) != frames_.end();
}

const FrameState* ReconstructionState::Find(int image_id) const {
    auto it = frames_.find(image_id);
    return it == frames_.end() ? nullptr : &it->second;
}

FrameState* ReconstructionState::Find(int image_id) {
    auto it = frames_.find(image_id);
    return it == frames_.end() ? nullptr : &it->second;
}

void ReconstructionState::Put(int image_id, const FrameState& state) {
    FrameState copy;
    copy.depth = state.depth.clone();
    copy.normal = state.normal.clone();
    copy.reliability = state.reliability.clone();
    copy.selected_views = state.selected_views.clone();
    copy.telemetry.gt_depth = state.telemetry.gt_depth.clone();
    copy.telemetry.gt_normal = state.telemetry.gt_normal.clone();
    copy.telemetry.gt_valid = state.telemetry.gt_valid.clone();
    copy.telemetry.gt_geometry_edge = state.telemetry.gt_geometry_edge.clone();
    copy.telemetry.gt_surface_label = state.telemetry.gt_surface_label.clone();
    copy.telemetry.fine_edge = state.telemetry.fine_edge.clone();
    copy.telemetry.coarse_region = state.telemetry.coarse_region.clone();
    copy.telemetry.texture_complexity = state.telemetry.texture_complexity.clone();
    copy.telemetry.es_candidate_count = state.telemetry.es_candidate_count.clone();
    copy.telemetry.es_same_surface = state.telemetry.es_same_surface.clone();
    copy.telemetry.candidate_count = state.telemetry.candidate_count.clone();
    copy.telemetry.same_surface_candidates = state.telemetry.same_surface_candidates.clone();
    copy.telemetry.anchor_count = state.telemetry.anchor_count.clone();
    copy.telemetry.same_surface_anchors = state.telemetry.same_surface_anchors.clone();
    copy.telemetry.fitted_plane_depth_error = state.telemetry.fitted_plane_depth_error.clone();
    copy.telemetry.fitted_plane_normal_error = state.telemetry.fitted_plane_normal_error.clone();
    copy.telemetry.radius_violation = state.telemetry.radius_violation.clone();
    copy.telemetry.final_depth_error = state.telemetry.final_depth_error.clone();
    copy.telemetry.matching_cost = state.telemetry.matching_cost.clone();
    copy.telemetry.adaptive_radius = state.telemetry.adaptive_radius.clone();
    copy.telemetry.anchors = state.telemetry.anchors;
    copy.telemetry.planes = state.telemetry.planes;
    frames_[image_id] = std::move(copy);
}

void ReconstructionState::Clear() { frames_.clear(); }

}  // namespace dpe
