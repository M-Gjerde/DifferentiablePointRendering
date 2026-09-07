#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace viewer {
    // A fixed grid in normalized image coordinates. Count centers, not disk
    // coverage: opacity, orientation and radius do not change the point budget.
    struct SurfelDensity {
        int gridSize;
        std::vector<std::uint32_t> counts;
        std::size_t total = 0;
        std::size_t inView = 0;
        std::uint32_t peak = 0;

        explicit SurfelDensity(int size)
            : gridSize(std::clamp(size, 1, 256)),
              counts(static_cast<std::size_t>(gridSize) * gridSize, 0u) {}

        void addCameraCenter(float x, float y, float z,
                             float fx, float fy, float cx, float cy,
                             std::uint32_t width, std::uint32_t height) {
            ++total;
            // Inverse of makePrimaryRayFromPixelJitteredFov: camera looks down -Z.
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
                z >= 0.0f || width == 0u || height == 0u) return;
            const double u = (cx + static_cast<double>(fx) * x / -z) / width;
            const double v = (cy - static_cast<double>(fy) * y / -z) / height;
            if (!(u >= 0.0 && u < 1.0 && v >= 0.0 && v < 1.0)) return;
            const auto column = static_cast<std::size_t>(u * gridSize);
            const auto row = static_cast<std::size_t>(v * gridSize);
            peak = std::max(peak, ++counts[row * gridSize + column]);
            ++inView;
        }
    };
}
