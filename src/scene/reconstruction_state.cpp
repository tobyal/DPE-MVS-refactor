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
    frames_[image_id] = std::move(copy);
}

void ReconstructionState::Clear() { frames_.clear(); }

}  // namespace dpe
