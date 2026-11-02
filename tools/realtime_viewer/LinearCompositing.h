#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace viewer {
    struct DisplayTransform {
        float exposure = 1.0f;
        float gamma = 1.0f;
        bool srgb = true;
    };

    inline float encodeDisplay(float linear, const DisplayTransform& transform) {
        linear = std::max(linear, 0.0f);
        if (!transform.srgb) return std::pow(linear, transform.gamma > 0 ? 1.0f / transform.gamma : 1.0f);
        return linear <= .0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 1.0f / 2.4f) - .055f;
    }

    inline float decodeDisplay(float encoded, const DisplayTransform& transform) {
        if (!transform.srgb) return std::pow(encoded, transform.gamma > 0 ? transform.gamma : 1.0f);
        return encoded <= .04045f ? encoded / 12.92f : std::pow((encoded + .055f) / 1.055f, 2.4f);
    }

    inline uint8_t displayByte(float encoded) {
        if (!std::isfinite(encoded)) return encoded > 0 ? 255u : 0u;
        return static_cast<uint8_t>(std::clamp(encoded, 0.0f, 1.0f) * 255.0f + .5f);
    }

    // Background is a display-referred UI color/grid: decode it for linear
    // composition, keeping its appearance independent of scene exposure.
    // raw RGB is already associated with coverage: C + (1-A)B, never A*C.
    inline void compositeLinearOverBackground(
        const std::vector<float>& raw, std::vector<uint8_t>& background,
        const DisplayTransform& transform, bool forceOpaque = false) {
        if (raw.size() != background.size() || raw.size() % 4u != 0u)
            throw std::invalid_argument("Raw RGBA and background dimensions differ");
        for (std::size_t i = 0; i < raw.size(); i += 4u) {
            const bool hasContribution = raw[i+3] > 0 || raw[i] != 0 || raw[i+1] != 0 || raw[i+2] != 0;
            const float alpha = forceOpaque && hasContribution ? 1.0f : std::clamp(raw[i+3], 0.0f, 1.0f);
            for (std::size_t c = 0; c < 3u; ++c) {
                const float bg = decodeDisplay(background[i+c] / 255.0f, transform);
                const float linear = raw[i+c] * transform.exposure + (1.0f - alpha) * bg;
                background[i+c] = displayByte(encodeDisplay(linear, transform));
            }
            background[i+3] = 255u; // Prevent ImGui from compositing coverage again.
        }
    }

    // PNG uses straight alpha. Unassociate linear radiance before encoding,
    // not encoded RGB afterward. HDR straight colors can clip in an 8-bit PNG.
    inline std::vector<uint8_t> straightAlphaScreenshot(
        const std::vector<float>& raw, const DisplayTransform& transform) {
        if (raw.size() % 4u != 0u) throw std::invalid_argument("Expected raw RGBA");
        std::vector<uint8_t> pixels(raw.size(), 0u);
        for (std::size_t i = 0; i < raw.size(); i += 4u) {
            const float alpha = std::clamp(raw[i+3], 0.0f, 1.0f);
            pixels[i+3] = displayByte(alpha);
            if (pixels[i+3] == 0u) continue;
            for (std::size_t c = 0; c < 3u; ++c)
                pixels[i+c] = displayByte(encodeDisplay(raw[i+c] / alpha * transform.exposure, transform));
        }
        return pixels;
    }
}
