#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace viewer {
    // Use the selected view's RGB with the renderer's coverage alpha. Neither
    // viewport background composition nor the force-opaque toggle belongs here.
    inline std::vector<uint8_t> transparentScreenshotPixels(
        const std::vector<uint8_t>& viewRgba,
        const std::vector<uint8_t>& renderedRgba,
        uint32_t width, uint32_t height) {
        const auto size = static_cast<std::size_t>(width) * height * 4u;
        if (width == 0 || height == 0 || viewRgba.size() != size || renderedRgba.size() != size) {
            throw std::invalid_argument("Screenshot buffers do not match the rendered image dimensions");
        }
        auto result = viewRgba;
        for (std::size_t i = 0; i < size; i += 4u) {
            result[i + 3u] = renderedRgba[i + 3u];
            if (result[i + 3u] == 0u) {
                result[i] = result[i + 1u] = result[i + 2u] = 0u;
            }
        }
        return result;
    }
}
