// Pale.Import.MaterialBaker.Mitsuba.ixx
module;
#include <pugixml.hpp>
#include <filesystem>
#include <glm/glm.hpp>
#include <optional>

export module Pale.Assets:MaterialBakerMitsuba;

import :MaterialBakerCommon;
import Pale.Assets.Core;
import :Index;

export namespace Pale {
    inline MaterialDesc descFromBsdf(const pugi::xml_node &bsdf, const std::filesystem::path &sceneDir) {
        MaterialDesc d{};
        auto rgb = [&](const char *name, glm::vec3 def)-> glm::vec3 {
            if (auto n = bsdf.find_child_by_attribute("rgb", "name", name)) {
                float r = def.x, g = def.y, b = def.z;
                std::sscanf(n.attribute("value").as_string(), "%f , %f , %f", &r, &g, &b);
                return {r, g, b};
            }
            return def;
        };

        std::string type = bsdf.attribute("type").as_string("diffuse");
        if (type == "diffuse") {
            d.baseColor = rgb("reflectance", {0.5f, 0.5f, 0.5f});
            if (auto tex = bsdf.find_child_by_attribute("texture", "name", "reflectance")) {
                if (auto fn = tex.find_child_by_attribute("string", "name", "filename")) {
                    auto p = std::filesystem::path(fn.attribute("value").as_string());
                    if (p.is_relative()) p = sceneDir / p;
                    d.baseColorTex = std::filesystem::weakly_canonical(p);
                }
            }
            d.roughness = 1.0f;
            d.metallic = 0.0f;
        }
        return d;
    }

    inline MaterialDesc descFromEmitter(const pugi::xml_node &emitter, const std::filesystem::path &sceneDir) {
        MaterialDesc materialDescription{};
        auto rgb = [&](const char *name, glm::vec3 def)-> glm::vec3 {
            if (auto n = emitter.find_child_by_attribute("rgb", "name", name)) {
                float r = def.x, g = def.y, b = def.z;
                std::sscanf(n.attribute("value").as_string(), "%f , %f , %f", &r, &g, &b);
                return {r, g, b};
            }
            return def;
        };
        auto flt = [&](const char *name, float def)-> float {
            if (auto n = emitter.find_child_by_attribute("float", "name", name)) return n.attribute("value").
                    as_float(def);
            return def;
        };

        std::string type = emitter.attribute("type").as_string("area");
        if (type == "area") {
            materialDescription.baseColor = rgb("color", {0.5f, 0.5f, 0.5f});
            materialDescription.power = flt("power", 0.0f);


            Log::PA_INFO("Material Emissive Color: {}, {}, {}", materialDescription.baseColor.x,
                         materialDescription.baseColor.y, materialDescription.baseColor.z);
            Log::PA_INFO("Material Emissive power: {}", materialDescription.power);
            materialDescription.roughness = 1.0f;
            materialDescription.metallic = 0.0f;
        }
        return materialDescription;
    }

    // In Pale.Import.MaterialBaker.Mitsuba
    enum class BakeKey { ByHash, ById, ByIdThenHash };

    // sceneName: optional foldering per scene file
    inline std::string sceneStem(const std::filesystem::path &xml) {
        return xml.stem().string(); // e.g. "cornell_box"
    }

    inline AssetHandle bakeFromMitsuba(const pugi::xml_node &bsdf,
                                       const std::filesystem::path &sceneXml,
                                       IAssetIndex &assets,
                                       BakeKey keyMode = BakeKey::ByHash,
                                       bool isEmitter = false) {
        MaterialDesc material{};
        if (isEmitter) {
            material = descFromEmitter(bsdf, sceneXml.parent_path());
        } else {
            material = descFromBsdf(bsdf, sceneXml.parent_path());
        }

        std::string idAttr = bsdf.attribute("id").as_string(""); // may be empty
        std::string h = hashDesc(material);
        auto matDir = std::filesystem::path("Materials") / sceneStem(sceneXml);

        std::filesystem::path out;
        std::string matLabel = isEmitter ? "emissive_" : "bsdf_";
        switch (keyMode) {
            case BakeKey::ByHash:
                out = matDir / (matLabel + h + ".mat.yaml");
                break;
            case BakeKey::ById:
                out = matDir / (matLabel + (idAttr.empty() ? h : idAttr) + ".mat.yaml");
                break;
            case BakeKey::ByIdThenHash:
            default:
                // keep one file per id; include hash to make changes visible but still per-id
                out = matDir / (matLabel + (idAttr.empty() ? h : idAttr + "_" + h) + ".mat.yaml");
                break;
        }

        writeYaml(out, material); // write project-relative texture paths inside

        if (auto id = assets.findByPath(out)) return *id;
        return assets.importPath(out, AssetType::Material);
    }
} // namespace Pale
