//
// Created by magnus on 8/27/25.
//
module;

#include <filesystem>

export module MG.SceneSerializer;

import pugixml;
import MG.Scene;

export namespace MG {

    class SceneSerializer {
        public:

        SceneSerializer(std::shared_ptr<Scene> scene) : m_scene(scene) {}

        void deserialize(const std::filesystem::path& xmlPath) {

        }

    private:

        std::shared_ptr<Scene> m_scene;
    };

}