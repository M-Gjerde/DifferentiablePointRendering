#pragma once

#include <cmath>
#include <optional>

#include <glm/glm.hpp>

#include "Renderer/Kernels/KernelHelpers.h"

namespace viewer {
    // Project with the same intrinsics as primary rays, rather than the UI's
    // current camera (a resized or newly selected camera may still be rendering).
    [[nodiscard]] inline std::optional<glm::vec2> projectViewportPoint(
        const Pale::CameraGPU& camera, glm::vec3 worldPoint,
        glm::vec2 imageMin, glm::vec2 imageSize) {
        if (camera.width == 0u || camera.height == 0u ||
            !(imageSize.x > 0.0f && imageSize.y > 0.0f)) return std::nullopt;
        const auto point = camera.view * Pale::float4{worldPoint.x, worldPoint.y, worldPoint.z, 1.0f};
        if (!(point.z() < -1.0e-6f)) return std::nullopt;
        glm::vec2 uv;
        if (camera.orthographic) {
            const auto ndc = camera.proj * point;
            uv = {0.5f + 0.5f * ndc.x(), 0.5f - 0.5f * ndc.y()};
        } else {
            const float focal = 0.5f * camera.height / std::tan(0.5f * glm::radians(camera.fovy));
            const bool intrinsics = camera.hasPinholeIntrinsics && camera.fx > 0 && camera.fy > 0;
            const float fx = intrinsics ? camera.fx : focal;
            const float fy = intrinsics ? camera.fy : focal;
            const float cx = intrinsics ? camera.cx : 0.5f * camera.width;
            const float cy = intrinsics ? camera.cy : 0.5f * camera.height;
            if (!(fx > 0 && fy > 0 && std::isfinite(fx) && std::isfinite(fy))) return std::nullopt;
            uv = {(cx + fx * point.x() / -point.z()) / camera.width,
                  (cy - fy * point.y() / -point.z()) / camera.height};
        }
        if (!(uv.x >= 0 && uv.x < 1 && uv.y >= 0 && uv.y < 1)) return std::nullopt;
        return imageMin + uv * imageSize;
    }

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
        if (!displayedCamera.orthographic) {
            const bool intrinsics = displayedCamera.hasPinholeIntrinsics &&
                displayedCamera.fx > 0 && displayedCamera.fy > 0;
            if (intrinsics) {
                if (!std::isfinite(displayedCamera.fx) || !std::isfinite(displayedCamera.fy)) return std::nullopt;
            } else if (!(displayedCamera.fovy > 0 && displayedCamera.fovy < 180)) {
                return std::nullopt;
            }
        }

        // Map the displayed texture, even while a resized render is pending.
        // The renderer samples pixel (x, y) with zero jitter, without a +0.5 offset.
        const auto pixelX = static_cast<uint32_t>(uv.x * displayedCamera.width);
        const auto pixelY = static_cast<uint32_t>(uv.y * displayedCamera.height);
        const Pale::Ray ray = Pale::makePrimaryRayFromPixelJitteredFov(
            displayedCamera, static_cast<float>(pixelX), static_cast<float>(pixelY), 0.0f, 0.0f);
        const PickRay result{Pale::sycl2glm(ray.origin), Pale::sycl2glm(ray.direction)};
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(result.origin[axis]) || !std::isfinite(result.direction[axis])) return std::nullopt;
        }
        if (!(glm::dot(result.direction, result.direction) > 0)) return std::nullopt;
        return result;
    }
}
