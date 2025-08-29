//
// Created by magnus on 8/28/25.
//
module;
#include <fstream>
#include <filesystem>
#include <yaml-cpp/yaml.h>

// ---------- partition implementation ----------
module Pale.Assets:Registry;


import :Core;
import Pale.UUID;
import :API;

namespace YAML {

    // AssetType as string (with int fallback on decode)
    template<> struct convert<Pale::AssetType> {
        static Node encode(const Pale::AssetType& rhs) {
            Node n; n = Pale::AssetRegistry::toString(rhs); return n;
        }
        static bool decode(const Node& node, Pale::AssetType& rhs) {
            if (!node.IsScalar()) return false;
            const auto s = node.as<std::string>();
            // try string first
            rhs = Pale::AssetRegistry::assetTypeFromString(s);
            if (rhs != Pale::AssetType::Unknown) return true;
            // fallback: legacy int
            try { rhs = static_cast<Pale::AssetType>(node.as<int>()); return true; }
            catch (...) { rhs = Pale::AssetType::Unknown; return true; } // tolerate unknown
        }
    };

    // UUID as hex string
    template<> struct convert<Pale::UUID> {
        static Node encode(const Pale::UUID& rhs) {
            Node n; n = std::string(rhs); return n;
        }
        static bool decode(const Node& node, Pale::UUID& rhs) {
            if (!node.IsScalar()) return false;
            rhs = Pale::UUID(node.as<std::string>());
            return true;
        }
    };

} // namespace YAML

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
            out << YAML::Key << "id"   << YAML::Value << YAML::convert<UUID>::encode(id);
            out << YAML::Key << "type" << YAML::Value << YAML::convert<AssetType>::encode(m.type);
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
            const auto id = n["id"].as<UUID>();            // string → UUID
            AssetMeta m;
            // accept "type" as "Mesh" or legacy int
            m.type = n["type"].as<AssetType>();
            m.path = n["path"].as<std::string>();
            if (std::filesystem::exists(m.path))
                m.lastWrite = std::filesystem::last_write_time(m.path);

            m_meta[id] = m;
            m_reverse[m.path] = id;
        }
    }

} // namespace Pale
