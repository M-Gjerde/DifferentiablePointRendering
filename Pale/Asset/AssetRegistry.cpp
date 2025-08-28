//
// Created by magnus on 8/28/25.
//
module;
#include <fstream>
#include <filesystem>
#include <yaml-cpp/yaml.h>

// ---------- partition implementation ----------
module Pale.Assets.Registry;


import Pale.Assets.Core;
import Pale.UUID;
import Pale.Assets.API;


namespace Pale {


    AssetHandle AssetRegistry::import(const std::filesystem::path& path, AssetType type) {
        AssetHandle id = UUID();
        AssetMeta m; m.type = type; m.path = path;
        if (std::filesystem::exists(path)) m.lastWrite = std::filesystem::last_write_time(path);
        m_meta.emplace(id, m);
        m_reverse[path] = id;
        return id;
    }


    void AssetRegistry::save(const std::filesystem::path& file) const {
        YAML::Emitter out;
        out << YAML::BeginMap << YAML::Key << "assets" << YAML::Value << YAML::BeginSeq;
        for (auto& [id, m] : m_meta) {
            out << YAML::BeginMap;
            out << YAML::Key << "id" << YAML::Value << static_cast<uint64_t>(id);
            out << YAML::Key << "type" << YAML::Value << static_cast<int>(m.type);
            out << YAML::Key << "path" << YAML::Value << m.path.string();
            out << YAML::EndMap;
        }
        out << YAML::EndSeq << YAML::EndMap;
        std::ofstream(file) << out.c_str();
    }


    void AssetRegistry::load(const std::filesystem::path& file) {
        if (!std::filesystem::exists(file)) return;
        YAML::Node root = YAML::LoadFile(file.string());
        if (!root["assets"]) return;
        m_meta.clear();
        m_reverse.clear();
        for (const auto& n : root["assets"]) {
            auto idStr = n["id"].as<std::string>();
            AssetHandle id = UUID(idStr);
            AssetMeta m;
            m.type = static_cast<AssetType>(n["type"].as<int>());
            m.path = n["path"].as<std::string>();
            if (std::filesystem::exists(m.path))
                m.lastWrite = std::filesystem::last_write_time(m.path);
            m_meta[id] = m;
            m_reverse[m.path] = id;
        }
    }

} // namespace Pale
