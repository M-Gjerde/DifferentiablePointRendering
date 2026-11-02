#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace viewer {
    // Numeric linear RGB preview, not a color-managed rendering. Preserve the
    // radiance already weighted by coverage; never multiply or divide by alpha.
    inline std::vector<uint8_t> rawLinearPreview(const std::vector<float>& rgba) {
        if (rgba.size() % 4u != 0u) throw std::invalid_argument("Expected packed raw RGBA");
        std::vector<uint8_t> pixels(rgba.size(), 255u);
        for (std::size_t i = 0; i < rgba.size(); i += 4u) {
            if (!std::isfinite(rgba[i]) || !std::isfinite(rgba[i+1]) || !std::isfinite(rgba[i+2])) {
                pixels[i] = 255u; pixels[i+1] = 0u; pixels[i+2] = 255u;
                continue;
            }
            for (std::size_t c = 0; c < 3u; ++c) {
                pixels[i+c] = static_cast<uint8_t>(std::clamp(rgba[i+c], 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
        return pixels;
    }
}
