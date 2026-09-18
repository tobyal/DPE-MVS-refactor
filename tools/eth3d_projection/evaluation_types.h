#pragma once

#include <cstdint>

namespace dpe {
namespace eth3d {

struct EvaluationPoint {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    std::uint8_t red = 0;
    std::uint8_t green = 0;
    std::uint8_t blue = 0;
};

enum class EvaluationKind {
    Accuracy,
    Completeness,
};

}  // namespace eth3d
}  // namespace dpe
