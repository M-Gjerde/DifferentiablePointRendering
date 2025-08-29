module;
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <glm/glm.hpp>
export module Pale.Assets:MaterialYamlLoader;

import :Core;
import :Material;

export namespace Pale {

    struct YamlMaterialLoader : IAssetLoader<Material> {
        AssetPtr<Material> load(const AssetHandle&, const AssetMeta& meta) override {
            if (!std::filesystem::exists(meta.path)) return {};
            YAML::Node n = YAML::LoadFile(meta.path.string());
            auto m = std::make_shared<Material>();

            auto readVec3 = [](const YAML::Node& a, glm::vec3 def)->glm::vec3{
                if (!a || !a.IsSequence() || a.size()<3) return def;
                return { a[0].as<float>(), a[1].as<float>(), a[2].as<float>() };
            };
            m->baseColor = readVec3(n["baseColor"], m->baseColor);
            m->emissive  = readVec3(n["emissive"],  m->emissive);
            m->roughness = n["roughness"] ? n["roughness"].as<float>() : m->roughness;
            m->metallic  = n["metallic"]  ? n["metallic"].as<float>()  : m->metallic;
            m->ior       = n["ior"]       ? n["ior"].as<float>()       : m->ior;
            m->opacity   = n["opacity"]   ? n["opacity"].as<float>()   : m->opacity;

            auto toID = [](const YAML::Node& s)->AssetHandle {
                if (!s) return {};
                // store the baked texture’s path in YAML; you’ll import it via registry before calling get()
                return AssetHandle(s.as<std::string>()); // or leave empty if you store paths not IDs
            };
            // If you prefer storing texture *paths* in YAML, resolve to AssetHandle in your render/upload step.

            return m;
        }
    };

} // namespace Pale