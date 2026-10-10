#pragma once

#include <optional>

#include <glm/glm.hpp>

#include "Renderer/Kernels/KernelHelpers.h"

namespace viewer {
    struct PickRay {
        glm::vec3 origin{0.0f};
        glm::vec3 direction{0.0f, 0.0f, -1.0f};
    };

    [[nodiscard]] inline std::optional<PickRay> makePickRay(
        const Pale::CameraGPU& displayedCamera,
        glm::vec2 mouse,
        glm::vec2 imageMin,
        glm::vec2 imageSize) {
        if (displayedCamera.width == 0u || displayedCamera.height == 0u ||
            !(imageSize.x > 0.0f && imageSize.y > 0.0f)) {
            return std::nullopt;
        }
        const glm::vec2 uv = (mouse - imageMin) / imageSize;
        if (!(uv.x >= 0.0f && uv.x < 1.0f && uv.y >= 0.0f && uv.y < 1.0f)) {
            return std::nullopt;
        }

        // Map the displayed texture, even while a resized render is pending.
        // The renderer samples pixel (x, y) with zero jitter, without a +0.5 offset.
        const auto pixelX = static_cast<uint32_t>(uv.x * displayedCamera.width);
        const auto pixelY = static_cast<uint32_t>(uv.y * displayedCamera.height);
        const Pale::Ray ray = Pale::makePrimaryRayFromPixelJitteredFov(
            displayedCamera, static_cast<float>(pixelX), static_cast<float>(pixelY), 0.0f, 0.0f);
        return PickRay{Pale::sycl2glm(ray.origin), Pale::sycl2glm(ray.direction)};
    }
}
