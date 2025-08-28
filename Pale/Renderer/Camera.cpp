//
// Created by magnus on 8/28/25.
//

module;

#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>

module Pale.Camera;

namespace Pale {
        Camera::Camera(const glm::mat4& projection)
            : m_projectionMatrix(projection)
        {
        }

        Camera::Camera(const float degFov, const float width, const float height, const float nearP, const float farP)
            : m_projectionMatrix(glm::perspectiveFov(glm::radians(degFov), width, height, farP, nearP))
        {
        }

}
