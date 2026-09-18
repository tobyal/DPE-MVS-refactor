#pragma once

#include "evaluation_types.h"

#include <boost/filesystem/path.hpp>
#include <vector>

namespace dpe {
namespace eth3d {

std::vector<EvaluationPoint> ReadEvaluationPly(const boost::filesystem::path& path);

}  // namespace eth3d
}  // namespace dpe
