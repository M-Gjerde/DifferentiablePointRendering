#pragma once

#include "Renderer/GPUDataStructures.h"

#include <cmath>
#include <optional>

namespace viewer {

struct PickRay {
    glm::vec3 origin{0.0f};
    glm::vec3 direction{0.0f, 0.0f, -1.0f};
};

// The displayed image uses top-left screen coordinates. Match the renderer's
// makePrimaryRayFromPixelJitteredFov convention, without stochastic pixel jitter.
// Use the displayed camera, which may lag behind interactive camera updates.
[[nodiscard]] inline std::optional<PickRay> makePickRay(
    const Pale::CameraGPU &camera, glm::vec2 mousePosition,
    glm::vec2 imageMin, glm::vec2 imageSize) {
    const auto finite2 = [](glm::vec2 v) {
        return std::isfinite(v.x) && std::isfinite(v.y);
    };
    const auto finite3 = [](glm::vec3 v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    };
    if (!finite2(mousePosition) || !finite2(imageMin) || !finite2(imageSize) ||
        imageSize.x <= 0.0f || imageSize.y <= 0.0f ||
        camera.width == 0 || camera.height == 0) return std::nullopt;

    const glm::vec2 uv = (mousePosition - imageMin) / imageSize;
    if (!finite2(uv) || uv.x < 0.0f || uv.x >= 1.0f || uv.y < 0.0f || uv.y >= 1.0f)
        return std::nullopt;

    const bool useIntrinsics = camera.hasPinholeIntrinsics != 0u &&
        camera.fx > 0.0f && camera.fy > 0.0f;
    float fx, fy, cx, cy;
    if (useIntrinsics) {
        fx = camera.fx; fy = camera.fy; cx = camera.cx; cy = camera.cy;
    } else {
        if (!std::isfinite(camera.fovy) || camera.fovy <= 0.0f || camera.fovy >= 180.0f)
            return std::nullopt;
        fx = fy = 0.5f * static_cast<float>(camera.height) /
            std::tan(0.5f * glm::radians(camera.fovy));
        cx = 0.5f * static_cast<float>(camera.width);
        cy = 0.5f * static_cast<float>(camera.height);
    }
    if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) || !std::isfinite(cy) ||
        fx <= 0.0f || fy <= 0.0f) return std::nullopt;

    const glm::vec3 cameraDirection{
        (uv.x * static_cast<float>(camera.width) - cx) / fx,
        (cy - uv.y * static_cast<float>(camera.height)) / fy,
        -1.0f
    };
    glm::mat4 worldFromCamera{1.0f};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            const float value = camera.invView.row[row][column];
            if (!std::isfinite(value)) return std::nullopt;
            worldFromCamera[column][row] = value;
        }
    }
    const glm::vec4 origin = worldFromCamera * glm::vec4{0.0f, 0.0f, 0.0f, 1.0f};
    const glm::vec3 direction = glm::vec3(worldFromCamera * glm::vec4{cameraDirection, 0.0f});
    const float length = glm::length(direction);
    if (!std::isfinite(origin.w) || std::abs(origin.w) <= 1.0e-8f ||
        !finite3(glm::vec3(origin)) || !finite3(direction) || !std::isfinite(length) || length <= 0.0f)
        return std::nullopt;
    const glm::vec3 worldOrigin = glm::vec3(origin) / origin.w;
    if (!finite3(worldOrigin)) return std::nullopt;
    return PickRay{worldOrigin, direction / length};
}

} // namespace viewer
