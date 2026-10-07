#pragma once

// Compatibility aliases for the viewer and its geometry tests. Training uses
// this same evaluator; neither path maintains a separate overlap definition.
#include "Renderer/SurfaceOverlap.h"

namespace viewer {
    using Pale::SurfaceFootprint;
    using Pale::SurfaceOverlapSettings;
    using Pale::SurfaceOverlapView;
    using Pale::SurfaceOverlapScore;
    using Pale::SurfaceOverlap;
}
