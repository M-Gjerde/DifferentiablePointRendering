// Pale/Render/PathTracer.Config.ixx
module;

#include <cstdint>

export module Pale.Render.PathTracerConfig;

export namespace Pale {
     struct RenderBatch {
        std::uint64_t samples{100000};  // how many paths to trace
        std::uint32_t maxBounces{6};    // path length cap
        std::uint64_t seed{0};          // deterministic base seed
        bool accumulate{true};          // false = clear outputs first
    };
} // namespace Pale::Render