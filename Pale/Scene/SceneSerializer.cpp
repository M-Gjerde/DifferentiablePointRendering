//
// Created by magnus on 8/28/25.
//
module;

#include <pugixml.hpp>
#include <filesystem>
#include <optional>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

module Pale.SceneSerializer;

import Pale.Log;
import Pale.Scene.Components;

namespace Pale {
    // --- small utils ---
    static inline float attrf(const pugi::xml_node &n, const char *name, float def = 0.f) {
        if (auto a = n.attribute(name)) return a.as_float(def);
        return def;
    }

    static inline int attri(const pugi::xml_node &n, const char *name, int def = 0) {
        if (auto a = n.attribute(name)) return a.as_int(def);
        return def;
    }

    static inline std::string attrs(const pugi::xml_node &n, const char *name, const std::string &def = {}) {
        if (auto a = n.attribute(name)) return a.value();
        return def;
    }

    static glm::vec3 parse_csv_vec3(const std::string &s, glm::vec3 def = glm::vec3(0)) {
        glm::vec3 v = def;
        float a = def.x, b = def.y, c = def.z;
        if (sscanf(s.c_str(), "%f , %f , %f", &a, &b, &c) == 3) v = glm::vec3(a, b, c);
        return v;
    }

    static glm::vec3 parse_rgb(const pugi::xml_node &n) {
        return parse_csv_vec3(attrs(n, "value", "1,1,1"));
    }

    static inline bool parseXmlBoolValue(const char* valueString, bool defaultValue) {
        if (valueString == nullptr) return defaultValue;

        std::string text = valueString;
        for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if (text == "true" || text == "1" || text == "yes" || text == "on")  return true;
        if (text == "false" || text == "0" || text == "no"  || text == "off") return false;

        char* endPtr = nullptr;
        const float numericValue = std::strtof(text.c_str(), &endPtr);
        if (endPtr != text.c_str()) return numericValue != 0.0f;

        return defaultValue;
    }

    static inline bool readAdjointSourceFlag(const pugi::xml_node& sensorNode, bool defaultValue = true) {
        if (auto booleanNode = sensorNode.find_child_by_attribute("boolean", "name", "adjoint_source")) {
            return parseXmlBoolValue(booleanNode.attribute("value").as_string(), defaultValue);
        }
        if (auto floatNode = sensorNode.find_child_by_attribute("float", "name", "adjoint_source")) {
            return parseXmlBoolValue(floatNode.attribute("value").as_string(), defaultValue);
        }
        return defaultValue;
    }

    static float computeFovYDegrees(float fovDegrees, const std::string &fovAxis,
                                    int filmWidth, int filmHeight) {
        const float aspect = static_cast<float>(filmWidth) / static_cast<float>(filmHeight);
        if (fovAxis == "y") return fovDegrees;
        if (fovAxis == "x") {
            const float fovXrad = glm::radians(fovDegrees);
            const float fovYrad = 2.0f * std::atan(std::tan(fovXrad * 0.5f) / aspect);
            return glm::degrees(fovYrad);
        }
        if (filmWidth <= filmHeight) {
            const float fovXrad = glm::radians(fovDegrees);
            const float fovYrad = 2.0f * std::atan(std::tan(fovXrad * 0.5f) / aspect);
            return glm::degrees(fovYrad);
        }
        return fovDegrees;
    }

    static std::optional<float> readFloatByName(const pugi::xml_node& parentNode,
                                                const char* nodeTagName,
                                                const char* nameAttribute,
                                                const char* desiredName,
                                                const char* valueAttribute = "value") {
        if (auto n = parentNode.find_child_by_attribute(nodeTagName, nameAttribute, desiredName)) {
            if (auto a = n.attribute(valueAttribute)) {
                return a.as_float();
            }
        }
        return std::nullopt;
    }

    static std::optional<std::string> readStringByName(const pugi::xml_node& parentNode,
                                                       const char* nodeTagName,
                                                       const char* nameAttribute,
                                                       const char* desiredName,
                                                       const char* valueAttribute = "value") {
        if (auto n = parentNode.find_child_by_attribute(nodeTagName, nameAttribute, desiredName)) {
            if (auto a = n.attribute(valueAttribute)) {
                return std::string(a.as_string());
            }
        }
        return std::nullopt;
    }

    // RH, depth 0..1, camera forward -Z (OpenGL-style view)
    static glm::mat4 perspectiveOffCenterRH_ZO(float left, float right,
                                              float bottom, float top,
                                              float nearClip, float farClip) {
        glm::mat4 m(0.0f);
        m[0][0] = 2.0f * nearClip / (right - left);
        m[1][1] = 2.0f * nearClip / (top - bottom);
        m[2][0] = (right + left) / (right - left);
        m[2][1] = (top + bottom) / (top - bottom);
        m[2][2] = farClip / (nearClip - farClip);
        m[2][3] = -1.0f;
        m[3][2] = (farClip * nearClip) / (nearClip - farClip);
        return m;
    }

    bool SceneSerializer::deserialize(const std::filesystem::path &xmlPath) {
        Log::PA_INFO("Loading xml file: {}", xmlPath.string());

        pugi::xml_document doc;
        pugi::xml_parse_result ok = doc.load_file(xmlPath.string().c_str());
        if (!ok) {
            Log::PA_INFO("XML parse error: {}", ok.description());
            return false;
        }

        pugi::xml_node scene = doc.child("scene");
        if (!scene) {
            Log::PA_INFO("Missing <scene> root");
            return false;
        }

        // (2) sensor -> camera
        for (auto sensor: scene.children("sensor")) {
            std::string name = attrs(sensor, "id", "");
            if (name.empty()) return false;

            Entity cameraEntity = m_scene->createEntity(name);
            auto &cameraComponent = cameraEntity.addComponent<CameraComponent>();

            const std::string sensorType = sensor.attribute("type").as_string();
            cameraComponent.projectionType =
                    (sensorType == "orthographic")
                        ? CameraComponent::Type::Orthographic
                        : CameraComponent::Type::Perspective;

            // film
            const auto filmNode = sensor.child("film");
            const int filmWidth =
                filmNode.child("integer").attribute("name").as_string() == std::string("width")
                    ? filmNode.child("integer").attribute("value").as_int()
                    : filmNode.find_child_by_attribute("integer", "name", "width").attribute("value").as_int();
            const int filmHeight =
                filmNode.find_child_by_attribute("integer", "name", "height").attribute("value").as_int();

            // frustum
            const float nearClip = sensor.find_child_by_attribute("float", "name", "near_clip").attribute("value").as_float(0.01f);
            const float farClip  = sensor.find_child_by_attribute("float", "name", "far_clip").attribute("value").as_float(1000.0f);

            // Optional explicit pinhole intrinsics (pixels)
            const std::optional<float> fxOpt = readFloatByName(sensor, "float", "name", "fx");
            const std::optional<float> fyOpt = readFloatByName(sensor, "float", "name", "fy");
            const std::optional<float> cxOpt = readFloatByName(sensor, "float", "name", "cx");
            const std::optional<float> cyOpt = readFloatByName(sensor, "float", "name", "cy");

            const bool hasPinholeIntrinsics = fxOpt.has_value() && fyOpt.has_value() && cxOpt.has_value() && cyOpt.has_value();

            // FOV fallback (legacy)
            const float fovDegreesRaw =
                sensor.find_child_by_attribute("float", "name", "fov").attribute("value").as_float(45.0f);
            const std::string fovAxis =
                sensor.find_child_by_attribute("string", "name", "fov_axis").attribute("value").as_string("smaller");

            const float fovYDegrees = computeFovYDegrees(fovDegreesRaw, fovAxis, filmWidth, filmHeight);

            // to_world via <lookat>
            const auto toWorld = sensor.child("transform");
            const auto lookAt = toWorld.child("lookat");
            auto parseVec3 = [](const char *s) {
                float x = 0, y = 0, z = 0;
                std::sscanf(s, "%f,%f,%f", &x, &y, &z);
                return glm::vec3{x, y, z};
            };
            const glm::vec3 cameraOrigin = parseVec3(lookAt.attribute("origin").as_string());
            const glm::vec3 cameraTarget = parseVec3(lookAt.attribute("target").as_string());
            const glm::vec3 cameraUpHint = parseVec3(lookAt.attribute("up").as_string());

            // View matrix: world→camera (RH, -Z forward)
            const glm::mat4 viewMatrix = glm::lookAt(cameraOrigin, cameraTarget, cameraUpHint);
            const glm::mat4 worldFromCamera = glm::inverse(viewMatrix);

            glm::mat4 projectionMatrix(1.0f);

            if (cameraComponent.projectionType == CameraComponent::Type::Perspective) {
                if (hasPinholeIntrinsics) {
                    const float fx = *fxOpt;
                    const float fy = *fyOpt;
                    const float cx = *cxOpt;
                    const float cy = *cyOpt;

                    // Store pinhole intrinsics in component and camera
                    cameraComponent.pinholeIntrinsics.isValid = true;
                    cameraComponent.pinholeIntrinsics.fx = fx;
                    cameraComponent.pinholeIntrinsics.fy = fy;
                    cameraComponent.pinholeIntrinsics.cx = cx;
                    cameraComponent.pinholeIntrinsics.cy = cy;

                    cameraComponent.camera.setPinholeIntrinsics(fx, fy, cx, cy);

                    // Off-center frustum from intrinsics:
                    // u = fx * x / (-z) + cx
                    // v = fy * y / (-z) + cy
                    // Here we assume:
                    // - camera forward is -Z
                    // - pixel coordinates: u right, v down
                    // - camera space: +Y up
                    //
                    // Convert pixel extents to camera-plane extents at near:
                    const float left   = -(cx) * nearClip / fx;
                    const float right  = (static_cast<float>(filmWidth) - cx) * nearClip / fx;

                    // y: v grows downward; camera-space y grows upward => top uses +cy, bottom uses -(H-cy)
                    const float top    = (cy) * nearClip / fy;
                    const float bottom = -(static_cast<float>(filmHeight) - cy) * nearClip / fy;

                    projectionMatrix = perspectiveOffCenterRH_ZO(left, right, bottom, top, nearClip, farClip);

                    // For debug/legacy UI, keep a consistent fovy value too:
                    cameraComponent.fovy = fovYDegrees;
                } else {
                    // Legacy symmetric projection from fov
                    cameraComponent.pinholeIntrinsics.isValid = false;

                    const float fovYRadians = glm::radians(fovYDegrees);
                    projectionMatrix = glm::perspectiveFovRH_ZO(
                        fovYRadians,
                        static_cast<float>(filmWidth),
                        static_cast<float>(filmHeight),
                        nearClip,
                        farClip
                    );

                    cameraComponent.fovy = fovYDegrees;
                }
            } else {
                // Orthographic handling (unchanged)
                cameraComponent.pinholeIntrinsics.isValid = false;
            }

            // Store camera state
            cameraComponent.camera.setProjectionMatrix(projectionMatrix);
            cameraComponent.camera.width = static_cast<float>(filmWidth);
            cameraComponent.camera.height = static_cast<float>(filmHeight);

            cameraComponent.primary = true;
            cameraComponent.useForAdjointPass = readAdjointSourceFlag(sensor, true);

            auto &transform = cameraEntity.getComponent<TransformComponent>();
            transform.setTransform(worldFromCamera);
        }

        for (auto shape: scene.children("shape")) {
            std::string type = attrs(shape, "type", "");
            if (type.empty()) return false;
            std::string name = attrs(shape, "id", "");
            if (name.empty()) return false;

            Entity entity = m_scene->createEntity(name);
            auto &transformComponent = entity.getComponent<TransformComponent>();

            if (auto tw = shape.find_child_by_attribute("transform", "name", "to_world")) {
                glm::vec3 translateVec(0.0f), scaleVec(1.0f);
                glm::quat rotationQuat(1.0f, 0.0f, 0.0f, 0.0f);
                for (auto child: tw.children()) {
                    std::string n = child.name();
                    if (n == "translate") {
                        glm::vec3 t(attrf(child, "x", 0), attrf(child, "y", 0), attrf(child, "z", 0));
                        translateVec = t;
                    } else if (n == "scale") {
                        glm::vec3 s(attrf(child, "x", 1), attrf(child, "y", 1), attrf(child, "z", 1));
                        scaleVec = s;
                    } else if (n == "rotate") {
                        glm::vec3 axis(attrf(child, "x", 0), attrf(child, "y", 0), attrf(child, "z", 0));
                        float deg = attrf(child, "angle", 0);
                        float rad = glm::radians(deg);
                        if (glm::length(axis) > 0) axis = glm::normalize(axis);
                        rotationQuat = glm::quat_cast(glm::rotate(glm::mat4(1), rad, axis));
                    }
                }
                transformComponent.setPosition(translateVec);
                transformComponent.setRotationQuaternion(rotationQuat);
                transformComponent.setScale(scaleVec);
            }

            auto &meshComp = entity.addComponent<MeshComponent>();

            auto resolveMeshFromShape = [&](const pugi::xml_node &s) -> std::optional<Pale::AssetHandle> {
                auto filenameNode = s.find_child_by_attribute("string", "name", "filename");
                if (filenameNode) {
                    std::filesystem::path path = filenameNode.attribute("value").as_string();
                    if (auto id = m_assets.findByPath(path)) return *id;
                    return m_assets.importPath(path, Pale::AssetType::Mesh);
                }
                return std::nullopt;
            };

            std::optional<Pale::AssetHandle> meshId;
            if (type == "obj" || type == "ply" || type == "serialized" || type == "plymesh") {
                meshId = resolveMeshFromShape(shape);
            }

            if (meshId) {
                meshComp.meshID = meshId.value();
            }

            std::unordered_map<std::string, AssetHandle> materialById;
            for (auto bsdfNode: scene.children("bsdf")) {
                std::string bsdfId = bsdfNode.attribute("id").as_string();
                if (bsdfId.empty()) continue;

                AssetHandle bakedMaterialHandle =
                        bakeFromMitsuba(bsdfNode, xmlPath, m_assets, /*keyMode=*/BakeKey::ByIdThenHash);

                materialById.emplace(bsdfId, bakedMaterialHandle);
            }

            std::optional<AssetHandle> emissiveMaterialHandle;
            if (auto emitterNode = shape.child("emitter")) {
                std::string emitterType = emitterNode.attribute("type").as_string();
                if (emitterType == "area") {
                    glm::vec3 radianceRgb =
                            parse_rgb(emitterNode.find_child_by_attribute("rgb", "name", "radiance"));

                    auto &areaLightComponent = entity.addComponent<AreaLightComponent>();
                    areaLightComponent.radiance = radianceRgb;

                    AssetHandle bakedEmissiveHandle =
                            bakeFromMitsuba(emitterNode, xmlPath, m_assets,
                                            /*keyMode=*/BakeKey::ByIdThenHash,
                                            /*isEmitter=*/true);

                    emissiveMaterialHandle = bakedEmissiveHandle;
                }
            }

            MaterialComponent &materialComponent = entity.addComponent<MaterialComponent>();
            const bool preferEmissiveMaterial =
                    entity.hasComponent<AreaLightComponent>() && emissiveMaterialHandle.has_value();

            if (preferEmissiveMaterial) {
                materialComponent.materialID = *emissiveMaterialHandle;
            } else if (auto refNode = shape.child("ref")) {
                std::string referencedId = refNode.attribute("id").as_string();
                if (auto iterator = materialById.find(referencedId); iterator != materialById.end()) {
                    materialComponent.materialID = iterator->second;
                } else if (emissiveMaterialHandle.has_value()) {
                    materialComponent.materialID = *emissiveMaterialHandle;
                }
            } else if (emissiveMaterialHandle.has_value()) {
                materialComponent.materialID = *emissiveMaterialHandle;
            }
        }

        return true;
    }
}
