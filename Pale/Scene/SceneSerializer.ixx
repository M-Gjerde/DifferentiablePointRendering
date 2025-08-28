//
// Created by magnus on 8/27/25.
//
module;

#include <filesystem>
#include <pugixml.hpp>

export module Pale.SceneSerializer;

import Pale.Scene;
import Pale.Assets.Core;
import Pale.Assets.API;

export namespace Pale {

    class SceneSerializer {
        public:

        SceneSerializer(std::shared_ptr<Scene> scene, IAssetProvider& assets)
          : m_scene(std::move(scene)), m_assets(assets) {}

        bool deserialize(const std::filesystem::path& xmlPath);

    private:
        std::shared_ptr<Scene> m_scene;
        IAssetProvider& m_assets;

        AssetHandle loadMeshFromMitsubaNode(const pugi::xml_node& shape) {
            // Mitsuba: <shape type="obj"><string name="filename" value="meshes/rect.obj"/></shape>
            if (std::string_view{shape.attribute("type").as_string()} == "obj") {
                auto s = shape.find_child_by_attribute("string","name","filename");
                std::filesystem::path path = s.attribute("value").as_string();
                return m_assets.importPath(path, AssetType::Mesh);
            }
            // other shapes/materials as needed...
            return {}; // or AssetID{} if you prefer optional
        }

    };

}