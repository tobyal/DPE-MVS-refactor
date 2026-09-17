#pragma once

#include "scene/scene.h"
#include "scene/reconstruction_state.h"
#include "diagnostics/diagnostic_sink.h"
#include <boost/filesystem.hpp>
#include <vector>

namespace dpe {

void RunFusion(Scene& scene,
               const ReconstructionState& reconstruction,
               const std::vector<Problem>& problems,
               const boost::filesystem::path& output_ply,
               DiagnosticSink* diagnostics = nullptr);

}  // namespace dpe
