//
// Created by magnus on 8/28/25.
//m
module;

#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <glm/glm.hpp>

export module Pale.Assets:MaterialBakerCommon;


import Pale.UUID;

export namespace Pale {
    struct MaterialDesc {
        glm::vec3 baseColor{1, 1, 1};
        float power = 0.0f;
        float roughness{0.5f};
        float metallic{0.0f};
        float ior{1.5f};
        float opacity{1.0f};

        // texture *paths* (baking writes paths; registry will import them)
        std::filesystem::path baseColorTex;
        std::filesystem::path metallicRoughnessTex;
        std::filesystem::path normalTex;
        std::filesystem::path emissiveTex;
        std::filesystem::path opacityTex;
    };

    inline std::string hashDesc(const MaterialDesc &d) {
        // deterministic string; replace with a proper hash if you have one in Pale::UUID
        auto toKey = [&](const std::filesystem::path &p) { return p.empty() ? std::string{} : p.generic_string(); };
        std::string s = std::to_string(d.baseColor.x) + "|" + std::to_string(d.baseColor.y) + "|" +
                        std::to_string(d.baseColor.z) + "|" +
                        std::to_string(d.power)  + "|" +
                        std::to_string(d.roughness) + "|" + std::to_string(d.metallic) + "|" + std::to_string(d.ior) +
                        "|" + std::to_string(d.opacity) + "|" +
                        toKey(d.baseColorTex) + "|" + toKey(d.metallicRoughnessTex) + "|" + toKey(d.normalTex) + "|" +
                        toKey(d.emissiveTex) + "|" + toKey(d.opacityTex);
        return std::string(Pale::UUID(s)); // implement hashString in your UUID module or use another hash
    }

    inline void writeYaml(const std::filesystem::path &out, const MaterialDesc &d) {
        YAML::Emitter e;
        e << YAML::BeginMap;
        e << YAML::Key << "baseColor" << YAML::Value << YAML::Flow << YAML::BeginSeq << d.baseColor.x << d.baseColor.y
                << d.baseColor.z << YAML::EndSeq;
        e << YAML::Key << "power" << YAML::Value << d.power;
        e << YAML::Key << "roughness" << YAML::Value << d.roughness;
        e << YAML::Key << "metallic" << YAML::Value << d.metallic;
        e << YAML::Key << "ior" << YAML::Value << d.ior;
        e << YAML::Key << "opacity" << YAML::Value << d.opacity;
        if (!d.baseColorTex.empty()) e << YAML::Key << "baseColorTex" << YAML::Value << d.baseColorTex.generic_string();
        if (!d.metallicRoughnessTex.empty()) e << YAML::Key << "metallicRoughnessTex" << YAML::Value << d.
                                             metallicRoughnessTex.generic_string();
        if (!d.normalTex.empty()) e << YAML::Key << "normalTex" << YAML::Value << d.normalTex.generic_string();
        if (!d.emissiveTex.empty()) e << YAML::Key << "emissiveTex" << YAML::Value << d.emissiveTex.generic_string();
        if (!d.opacityTex.empty()) e << YAML::Key << "opacityTex" << YAML::Value << d.opacityTex.generic_string();
        e << YAML::EndMap;

        std::filesystem::create_directories(out.parent_path());
        std::ofstream(out) << e.c_str();
    }
} // namespace Pale
