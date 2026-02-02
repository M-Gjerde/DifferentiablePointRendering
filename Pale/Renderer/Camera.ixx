//
// Created by magnus on 8/28/25.
//
module;

#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>

export module Pale.Camera;


export namespace Pale {

    struct PinholeIntrinsics {
        bool isValid = false;
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };


    class Camera {
    public:
        Camera() = default;
        Camera(const glm::mat4& projection);
        Camera(const float degFov, const float width, const float height, const float nearP, const float farP);

        [[nodiscard]] const glm::mat4& getProjectionMatrix() const { return m_projectionMatrix; }
        void setProjectionMatrix(const glm::mat4& mat) { m_projectionMatrix = mat; }

        // New: pinhole intrinsics storage
        void setPinholeIntrinsics(float fxPixels, float fyPixels, float cxPixels, float cyPixels) {
            m_hasPinholeIntrinsics = true;
            m_fxPixels = fxPixels;
            m_fyPixels = fyPixels;
            m_cxPixels = cxPixels;
            m_cyPixels = cyPixels;
        }

        [[nodiscard]] bool hasPinholeIntrinsics() const { return m_hasPinholeIntrinsics; }
        [[nodiscard]] float getFxPixels() const { return m_fxPixels; }
        [[nodiscard]] float getFyPixels() const { return m_fyPixels; }
        [[nodiscard]] float getCxPixels() const { return m_cxPixels; }
        [[nodiscard]] float getCyPixels() const { return m_cyPixels; }

        float width = 0.0f;
        float height = 0.0f;

    private:
        glm::mat4 m_projectionMatrix = glm::mat4(1.0f);

        bool m_hasPinholeIntrinsics = false;
        float m_fxPixels = 0.0f;
        float m_fyPixels = 0.0f;
        float m_cxPixels = 0.0f;
        float m_cyPixels = 0.0f;
    };

}