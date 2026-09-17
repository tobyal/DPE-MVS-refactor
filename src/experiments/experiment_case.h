#pragma once

#include "common/types.h"
#include <string>

namespace dpe {

enum class SourceMode { Native, Disabled, External };

struct AlgorithmPolicy {
    SourceMode edge = SourceMode::Native;
    SourceMode apd = SourceMode::Native;
    SourceMode label = SourceMode::Native;
    SourceMode radius = SourceMode::Native;
    SourceMode edge_limit = SourceMode::Native;

    void Apply(DPEParams& p) const {
        if (edge == SourceMode::Disabled) p.use_edge = false;
        if (apd == SourceMode::Disabled) p.use_apd = false;
        if (label == SourceMode::Disabled) p.use_label = false;
        if (radius == SourceMode::Disabled) p.use_radius = false;
        if (edge_limit == SourceMode::Disabled) p.use_limit = false;
    }

    bool RequiresExternalPrior() const {
        return edge == SourceMode::External || apd == SourceMode::External ||
               label == SourceMode::External || radius == SourceMode::External ||
               edge_limit == SourceMode::External;
    }
};

struct ExperimentCase {
    std::string name;
    std::string description;
    AlgorithmPolicy policy;
    bool enabled = true;
    bool telemetry = false;  // capture GT-backed GPU telemetry on final finest pass
};

}  // namespace dpe
