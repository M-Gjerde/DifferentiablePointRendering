//
// Created by magnus on 8/28/25.
module;
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module Pale.Scene.Components;

import Pale.UUID;
import Pale.Camera;
import Pale.Assets.Core;

export namespace Pale {
    struct TagComponent {
        std::string tag;
        std::string &getTag() { return tag; }

        void setTag(const std::string &newTag) { tag = newTag; }
    };

    struct IDComponent {
        UUID ID{};

        IDComponent() = default;

        explicit IDComponent(const UUID &uuid) : ID(uuid) {
        }
    };

    struct TransformComponent {
        // Transformation components
        glm::vec3 translation = {0.0f, 0.0f, 0.0f};
        glm::vec3 rotationEuler = {0.0f, 0.0f, 0.0f}; // Euler angles in degrees
        glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity quaternion
        glm::vec3 scale = {1.0f, 1.0f, 1.0f};

        // Constructors
        TransformComponent() = default;

        // Constructor that decomposes a transform matrix into translation, rotation, and scale.
        explicit TransformComponent(const glm::mat4 &transform) {
            // --- Translation ---
            // The translation is stored in the 4th column of the matrix.
            translation = glm::vec3(transform[3]);
            // --- Scale ---
            // The scale factors are the lengths of the first three columns.
            scale.x = glm::length(glm::vec3(transform[0]));
            scale.y = glm::length(glm::vec3(transform[1]));
            scale.z = glm::length(glm::vec3(transform[2]));
            // --- Rotation ---
            // To extract the rotation, first remove the scaling from the matrix.
            glm::mat3 rotationMatrix;
            if (scale.x != 0) rotationMatrix[0] = glm::vec3(transform[0]) / scale.x;
            else rotationMatrix[0] = glm::vec3(transform[0]);
            if (scale.y != 0) rotationMatrix[1] = glm::vec3(transform[1]) / scale.y;
            else rotationMatrix[1] = glm::vec3(transform[1]);
            if (scale.z != 0) rotationMatrix[2] = glm::vec3(transform[2]) / scale.z;
            else rotationMatrix[2] = glm::vec3(transform[2]);
            // Convert the rotation matrix to a quaternion.
            rotation = glm::quat_cast(rotationMatrix);
            // Convert the quaternion to Euler angles (in degrees) for storage.
            rotationEuler = glm::degrees(glm::eulerAngles(rotation));
        }


        // Get the transformation matrix
        glm::mat4 getTransform() const {
            glm::mat4 rotMat = glm::mat4_cast(rotation);

            return glm::translate(glm::mat4(1.0f), translation) * rotMat *
                   glm::scale(glm::mat4(1.0f), scale);
        }


        void setTransform(const glm::mat4 &transform) {
            // --- Translation ---
            // The translation is stored in the 4th column of the matrix.
            translation = glm::vec3(transform[3]);
            // --- Scale ---
            // The scale factors are the lengths of the first three columns.
            scale.x = glm::length(glm::vec3(transform[0]));
            scale.y = glm::length(glm::vec3(transform[1]));
            scale.z = glm::length(glm::vec3(transform[2]));
            // --- Rotation ---
            // To extract the rotation, first remove the scaling from the matrix.
            glm::mat3 rotationMatrix;
            if (scale.x != 0) rotationMatrix[0] = glm::vec3(transform[0]) / scale.x;
            else rotationMatrix[0] = glm::vec3(transform[0]);
            if (scale.y != 0) rotationMatrix[1] = glm::vec3(transform[1]) / scale.y;
            else rotationMatrix[1] = glm::vec3(transform[1]);
            if (scale.z != 0) rotationMatrix[2] = glm::vec3(transform[2]) / scale.z;
            else rotationMatrix[2] = glm::vec3(transform[2]);
            // Convert the rotation matrix to a quaternion.
            rotation = glm::quat_cast(rotationMatrix);
            // Convert the quaternion to Euler angles (in degrees) for storage.
            rotationEuler = glm::degrees(glm::eulerAngles(rotation));
        }

        // Set rotation using Euler angles (degrees)
        void setRotationEuler(const glm::vec3 &eulerAnglesDegrees) {
            rotationEuler = eulerAnglesDegrees;
            glm::vec3 radians = glm::radians(rotationEuler);
            rotation = glm::quat(radians);
        }

        // Get rotation as Euler angles (degrees)
        glm::vec3 &getRotationEuler() {
            return rotationEuler;
        }

        void updateFromEulerRotation() {
            setRotationEuler(rotationEuler);
        }

        // Set rotation using quaternion
        void setRotationQuaternion(const glm::quat &q) {
            rotation = q;
            rotationEuler = glm::degrees(glm::eulerAngles(rotation));
        }

        // Get rotation as quaternion
        glm::quat &getRotationQuaternion() {
            return rotation;
        }

        // Position setters and getters
        void setPosition(const glm::vec3 &v) {
            translation = v;
        }

        glm::vec3 &getPosition() {
            return translation;
        }

        glm::vec3 getPosition() const {
            return translation;
        }

        // Scale setters and getters
        void setScale(const glm::vec3 &s) {
            scale = s;
        }

        glm::vec3 &getScale() {
            return scale;
        }
    };

    struct AreaLightComponent {
        float flux = 100.0f; // total radiant flux Φ in watts
        glm::vec3 radiance = glm::vec3(0.0f);
    };

    struct VisibleComponent {
        bool visible = true;
    };

    struct CameraComponent {
        enum class Type { None = -1, Perspective, Orthographic };

        Type projectionType;

        Camera camera;
        bool primary = true;

        CameraComponent() = default;

        CameraComponent(const CameraComponent &other) = default;

        operator Camera &() { return camera; }
        operator const Camera &() const { return camera; }
    };

    struct MeshComponent {
        AssetHandle meshID{};
    };

    struct MaterialComponent {
        AssetHandle material;
    };
}
