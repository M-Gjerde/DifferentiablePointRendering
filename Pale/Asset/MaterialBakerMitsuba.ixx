// Pale.Import.MaterialBaker.Mitsuba.ixx
module;
#include <pugixml.hpp>
#include <filesystem>
#include <glm/glm.hpp>
#include <optional>

export module Pale.Import.MaterialBakerMitsuba;

import Pale.Assets.Core;
import Pale.Assets.API;
import Pale.Import.MaterialBakerCommon;

export namespace Pale {


inline MaterialDesc descFromBsdf(const pugi::xml_node& bsdf, const std::filesystem::path& sceneDir) {
    MaterialDesc d{};
    auto rgb = [&](const char* name, glm::vec3 def)->glm::vec3{
        if (auto n = bsdf.find_child_by_attribute("rgb","name",name)) {
            float r=def.x,g=def.y,b=def.z;
            std::sscanf(n.attribute("value").as_string(), "%f , %f , %f", &r,&g,&b);
            return {r,g,b};
        }
        return def;
    };
    auto flt = [&](const char* name, float def)->float{
        if (auto n = bsdf.find_child_by_attribute("float","name",name)) return n.attribute("value").as_float(def);
        return def;
    };

    std::string type = bsdf.attribute("type").as_string("diffuse");
    if (type == "diffuse") {
        d.baseColor = rgb("reflectance", {0.5f,0.5f,0.5f});
        if (auto tex = bsdf.find_child_by_attribute("texture","name","reflectance")) {
            if (auto fn = tex.find_child_by_attribute("string","name","filename")) {
                auto p = std::filesystem::path(fn.attribute("value").as_string());
                if (p.is_relative()) p = sceneDir / p;
                d.baseColorTex = std::filesystem::weakly_canonical(p);
            }
        }
        d.roughness = 1.0f; d.metallic = 0.0f;
    } else if (type == "roughconductor") {
        d.metallic = 1.0f;
        d.roughness = flt("alpha", 0.2f);
        d.ior = flt("eta", 1.5f);
    } else if (type == "dielectric" || type == "roughdielectric") {
        d.metallic = 0.0f;
        d.roughness = (type == "roughdielectric") ? flt("alpha", 0.2f) : 0.0f;
        d.ior = flt("int_ior", 1.5f);
    } // extend as needed

    return d;
}

    // In Pale.Import.MaterialBaker.Mitsuba
    enum class BakeKey { ByHash, ById, ByIdThenHash };

    // sceneName: optional foldering per scene file
    inline std::string sceneStem(const std::filesystem::path& xml) {
        return xml.stem().string(); // e.g. "cornell_box"
    }

    inline AssetHandle bakeFromMitsuba(const pugi::xml_node& bsdf,
                                          const std::filesystem::path& sceneXml,
                                          IAssetProvider& assets,
                                          BakeKey keyMode = BakeKey::ByHash)
{
    MaterialDesc d = descFromBsdf(bsdf, sceneXml.parent_path());
    std::string idAttr = bsdf.attribute("id").as_string("");   // may be empty
    std::string h = hashDesc(d);
    auto matDir = std::filesystem::path("Materials") / sceneStem(sceneXml);

    std::filesystem::path out;
    switch (keyMode) {
        case BakeKey::ByHash:
            out = matDir / ("bsdf_" + h + ".mat.yaml");
            break;
        case BakeKey::ById:
            out = matDir / ("bsdf_" + (idAttr.empty()? h : idAttr) + ".mat.yaml");
            break;
        case BakeKey::ByIdThenHash:
        default:
            // keep one file per id; include hash to make changes visible but still per-id
            out = matDir / ("bsdf_" + (idAttr.empty()? h : idAttr + "_" + h) + ".mat.yaml");
            break;
    }

    if (!std::filesystem::exists(out))
        writeYaml(out, d);  // write project-relative texture paths inside

    if (auto id = assets.findByPath(out)) return *id;
    return assets.importPath(out, AssetType::Material);
}

} // namespace Pale
