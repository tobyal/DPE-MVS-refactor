#include "experiments/experiment_suite.h"

namespace dpe {
std::vector<ExperimentCase> MakeDefaultExperimentSuite() {
    std::vector<ExperimentCase> out;
    ExperimentCase baseline{"baseline_current", "Native DPE-MVS refactored pipeline", {}, true};
    baseline.telemetry = true;
    out.push_back(baseline);

    ExperimentCase edge{"edge_off", "Disable fine-edge guided propagation", {}, true};
    edge.policy.edge = SourceMode::Disabled;
    out.push_back(edge);

    ExperimentCase apd{"apd_off", "Disable weak-region APD/DPE path", {}, true};
    apd.policy.apd = SourceMode::Disabled;
    out.push_back(apd);

    ExperimentCase label{"label_off", "Disable coarse-region labels", {}, true};
    label.policy.label = SourceMode::Disabled;
    out.push_back(label);

    ExperimentCase radius{"radius_fixed", "Disable adaptive patch radius", {}, true};
    radius.policy.radius = SourceMode::Disabled;
    out.push_back(radius);

    ExperimentCase limit{"edge_limit_off", "Disable edge-crossing limit", {}, true};
    limit.policy.edge_limit = SourceMode::Disabled;
    out.push_back(limit);
    return out;
}
}  // namespace dpe
