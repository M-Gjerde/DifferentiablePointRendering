//
// Created by magnus on 8/28/25.
//
module;

#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>

export module Pale.Camera;


export namespace Pale {
    class Camera {
    public:
        Camera() = default;
        Camera(const glm::mat4& projection);
        Camera(const float degFov, const float width, const float height, const float nearP, const float farP);

        [[nodiscard]] const glm::mat4& getProjectionMatrix() const;

    private:
        glm::mat4 m_projectionMatrix = glm::mat4(1.0f);

    };

}