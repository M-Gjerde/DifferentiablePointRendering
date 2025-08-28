//
// Created by magnus on 8/28/25.
//
module;

#include <pugixml.hpp>
#include <filesystem>
#include <optional>

#include "Components.h"

module Pale.SceneSerializer;

import Pale.Log;
import Pale.Import.MaterialBakerMitsuba;

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
        if (auto sensor = scene.child("sensor")) {
            Entity cameraEntity = m_scene->createEntity("Camera");
            auto &camera = cameraEntity.addComponent<CameraComponent>();
        } else {
            Log::PA_INFO("No <sensor> found; creating default camera");
            // optionally create a default camera entity here…
        }


        // (3) shapes
        for (auto shape: scene.children("shape")) {
            std::string type = attrs(shape, "type", "");
            if (type.empty()) return false;

            // Create entity per shape
            Entity entity = m_scene->createEntity(type);

            auto &transformComponent = entity.getComponent<TransformComponent>();

            // Transform
            glm::mat4 M(1.f);
            if (auto tw = shape.find_child_by_attribute("transform", "name", "to_world")) {
                glm::vec3 translateVec, scaleVec;
                glm::quat rotationQuat;
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
                    // Normalize relative to the scene file if needed:
                    if (path.is_relative()) {
                        auto sceneDir = std::filesystem::path(xmlPath).parent_path();
                        path = std::filesystem::weakly_canonical(sceneDir / path);
                    }
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

            for (auto bsdf: scene.children("bsdf")) {
                std::string id = bsdf.attribute("id").as_string();
                if (id.empty()) continue;
                AssetHandle h = bakeFromMitsuba(bsdf, xmlPath, m_assets,
                                                /*keyMode=*/BakeKey::ByIdThenHash);
                materialById.emplace(id, h);
            }

            if (auto ref = shape.child("ref")) {
                std::string rid = ref.attribute("id").as_string();
                if (auto it = materialById.find(rid); it != materialById.end())
                    entity.addComponent<MaterialComponent>().material = it->second;
            }

            if (auto emitter = shape.child("emitter")) {
                std::string emitterType = emitter.attribute("type").as_string();
                if (emitterType == "area") {
                    glm::vec3 L = parse_rgb(emitter.find_child_by_attribute("rgb","name","radiance"));
                    auto& light = entity.addComponent<AreaLightComponent>();
                    light.radiance = L;
                }
            }

        }

        return true;
    }
}
