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
        // Mitsuba rgb value="r,g,b"
        return parse_csv_vec3(attrs(n, "value", "1,1,1"));
    }

    static inline bool parseXmlBoolValue(const char* valueString, bool defaultValue) {
        if (valueString == nullptr) return defaultValue;

        std::string text = valueString;
        for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if (text == "true" || text == "1" || text == "yes" || text == "on")  return true;
        if (text == "false" || text == "0" || text == "no"  || text == "off") return false;

        // Fallback: allow numeric strings like "0.0" / "1.0"
        char* endPtr = nullptr;
        const float numericValue = std::strtof(text.c_str(), &endPtr);
        if (endPtr != text.c_str()) return numericValue != 0.0f;

        return defaultValue;
    }

    static inline bool readAdjointSourceFlag(const pugi::xml_node& sensorNode, bool defaultValue = true) {
        // Preferred: <boolean name="adjoint_source" value="false"/>
        if (auto booleanNode = sensorNode.find_child_by_attribute("boolean", "name", "adjoint_source")) {
            return parseXmlBoolValue(booleanNode.attribute("value").as_string(), defaultValue);
        }

        // Back-compat: <float name="adjoint_source" value="False"/> or 0/1
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
        // "smaller": fov applies to the smaller image dimension
        if (filmWidth <= filmHeight) {
            // smaller is width ⇒ given fov is fovX
            const float fovXrad = glm::radians(fovDegrees);
            const float fovYrad = 2.0f * std::atan(std::tan(fovXrad * 0.5f) / aspect);
            return glm::degrees(fovYrad);
        } else {
            // smaller is height ⇒ given fov is fovY
            return fovDegrees;
        }
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
            const int filmWidth = filmNode.child("integer").attribute("name").as_string() == std::string("width")
                                      ? filmNode.child("integer").attribute("value").as_int()
                                      : filmNode.find_child_by_attribute("integer", "name", "width").attribute("value").
                                      as_int();
            const int filmHeight = filmNode.find_child_by_attribute("integer", "name", "height").attribute("value").
                    as_int();
            const float aspectRatio = static_cast<float>(filmWidth) / static_cast<float>(filmHeight);

            // frustum
            const float nearClip = sensor.find_child_by_attribute("float", "name", "near_clip").attribute("value").
                    as_float(0.01f);
            const float farClip = sensor.find_child_by_attribute("float", "name", "far_clip").attribute("value").
                    as_float(1000.0f);
            const float fovDegreesRaw = sensor.find_child_by_attribute("float", "name", "fov").attribute("value").
                    as_float(45.0f);
            const std::string fovAxis =
                    sensor.find_child_by_attribute("string", "name", "fov_axis").attribute("value").
                    as_string("smaller");

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

            // View matrix: world→camera
            const glm::mat4 viewMatrix = glm::lookAt(cameraOrigin, cameraTarget, cameraUpHint);

            // World transform (camera→world) is the inverse of view
            const glm::mat4 worldFromCamera = glm::inverse(viewMatrix);

            // Projection matrix
            const float fovYRadians = glm::radians(fovYDegrees);
            glm::mat4 projectionMatrix = glm::perspectiveFovRH_ZO(fovYRadians, static_cast<float>(filmWidth), static_cast<float>(filmHeight), nearClip, farClip);

            // Optionally fix Vulkan's Y if your NDC expects flipped Y:
            // projectionMatrix[1][1] *= -1.0f;

            // Store
            cameraComponent.camera.setProjectionMatrix(projectionMatrix) ;
            cameraComponent.camera.width = filmWidth;
            cameraComponent.camera.height = filmHeight;
            cameraComponent.primary = true;
            cameraComponent.useForAdjointPass = readAdjointSourceFlag(sensor, true);


            auto &transform = cameraEntity.getComponent<TransformComponent>();
            transform.setTransform(worldFromCamera);

        }


        // (3) shapes
        for (auto shape: scene.children("shape")) {
            std::string type = attrs(shape, "type", "");
            if (type.empty()) return false;
            std::string name = attrs(shape, "id", "");
            if (name.empty()) return false;

            // Create entity per shape
            Entity entity = m_scene->createEntity(name);

            auto &transformComponent = entity.getComponent<TransformComponent>();

            // Transform
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
                        // axis-angle; Mitsuba uses unit axis (x,y,z) and angle in degrees
                        glm::vec3 axis(attrf(child, "x", 0), attrf(child, "y", 0), attrf(child, "z", 0));
                        float deg = attrf(child, "angle", 0);
                        float rad = glm::radians(deg);
                        if (glm::length(axis) > 0)
                            axis = glm::normalize(axis);
                        rotationQuat = glm::quat_cast(glm::rotate(glm::mat4(1), rad, axis));
                    }
                }
                transformComponent.setPosition(translateVec);
                transformComponent.setRotationQuaternion(rotationQuat);
                transformComponent.setScale(scaleVec);
            }

            // MeshComponent with AssetHandle
            auto &meshComp = entity.addComponent<MeshComponent>();

            // Resolve mesh AssetHandle from Mitsuba <shape>
            auto resolveMeshFromShape = [&](const pugi::xml_node &s) -> std::optional<Pale::AssetHandle> {
                // filename is in <string name="filename" value="...">
                auto filenameNode = s.find_child_by_attribute("string", "name", "filename");
                if (filenameNode) {
                    std::filesystem::path path = filenameNode.attribute("value").as_string();

                    // Reuse existing ID or import a new one
                    if (auto id = m_assets.findByPath(path)) return *id;
                    return m_assets.importPath(path, Pale::AssetType::Mesh);
                }

                // Built-in primitives (rectangle/sphere/etc.) — optional; stub for now
                // You can register procedural assets later (e.g., “__primitive__/rectangle”)
                return std::nullopt;
            };

            std::optional<Pale::AssetHandle> meshId;
            if (type == "obj" || type == "ply" || type == "serialized" || type == "plymesh") {
                meshId = resolveMeshFromShape(shape);
            }

            if (meshId) {
                meshComp.meshID = meshId.value(); // <-- your MeshComponent's AssetHandle field
            } else {
                // If there is no mesh for this shape, you may remove MeshComponent or keep it empty.
                // entity.removeComponent<MeshComponent>();
            }


            std::unordered_map<std::string, AssetHandle> materialById;

            // 1) Bake all BSDFs by id (scene-wide)
            for (auto bsdfNode: scene.children("bsdf")) {
                std::string bsdfId = bsdfNode.attribute("id").as_string();
                if (bsdfId.empty()) continue;

                AssetHandle bakedMaterialHandle =
                        bakeFromMitsuba(bsdfNode, xmlPath, m_assets, /*keyMode=*/BakeKey::ByIdThenHash);

                materialById.emplace(bsdfId, bakedMaterialHandle);
            }

            // Track emissive for this specific shape
            std::optional<AssetHandle> emissiveMaterialHandle;

            if (auto emitterNode = shape.child("emitter")) {
                std::string emitterType = emitterNode.attribute("type").as_string();
                if (emitterType == "area") {
                    glm::vec3 radianceRgb =
                            parse_rgb(emitterNode.find_child_by_attribute("rgb", "name", "radiance"));

                    auto &areaLightComponent = entity.addComponent<AreaLightComponent>();
                    areaLightComponent.radiance = radianceRgb;

                    // Bake an emissive material variant for the area emitter
                    AssetHandle bakedEmissiveHandle =
                            bakeFromMitsuba(emitterNode, xmlPath, m_assets,
                                            /*keyMode=*/BakeKey::ByIdThenHash,
                                            /*isEmitter=*/true);

                    emissiveMaterialHandle = bakedEmissiveHandle;
                }
            }

            // Decide which material to attach
            // Preference rule: if entity has the tag `UseEmitterMaterialTag` and we baked an emissive, use that.
            // Otherwise fall back to the <ref id="..."> BSDF link if present.
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
                    // Fallback: if BSDF id not found but emissive exists, attach emissive
                    materialComponent.materialID = *emissiveMaterialHandle;
                }
            } else if (emissiveMaterialHandle.has_value()) {
                // No <ref>, but emissive exists
                materialComponent.materialID = *emissiveMaterialHandle;
            }
        }

        return true;
    }
}
