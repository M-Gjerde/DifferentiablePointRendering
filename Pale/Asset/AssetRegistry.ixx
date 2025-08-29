// ================================
// File: Pale/Asset/AssetRegistry.ixx
// ================================
module; // global module fragment for headers
#include <fstream>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <algorithm>
#include <yaml-cpp/yaml.h>

export module Pale.Assets:Registry;

// partitions you depend on
export import :Core;      // re-export Core to consumers of :Registry
import Pale.UUID;         // internal use only (not re-exported)


// -------------------- exported API --------------------
export namespace Pale {
    class AssetRegistry {
    public:
        bool has(const AssetHandle &assetHandle) const { return m_meta.contains(assetHandle); }

        const AssetMeta *meta(const AssetHandle &assetHandle) const {
            if (auto iterator = m_meta.find(assetHandle); iterator != m_meta.end()) return &iterator->second;
            return nullptr;
        }

        std::optional<AssetHandle> findByPath(const std::filesystem::path &filePath) const {
            if (auto iterator = m_reverse.find(filePath); iterator != m_reverse.end()) return iterator->second;
            return std::nullopt;
        }

        AssetHandle import(const std::filesystem::path &filePath, AssetType assetType);

        void save(const std::filesystem::path &filePath) const;

        void load(const std::filesystem::path &filePath);

        std::unordered_map<AssetHandle, AssetMeta> snapshot() const { return m_meta; }

        static std::string toString(AssetType assetType) {
            switch (assetType) {
                case AssetType::Mesh: return "Mesh";
                case AssetType::Material: return "Material";
                case AssetType::Shader: return "Shader";
                default: return "Unknown";
            }
        }

        static AssetType assetTypeFromString(std::string_view text) {
            auto lower = std::string(text);
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
            if (lower == "mesh") {
                return AssetType::Mesh;
            }
            if (lower == "material") {
                return AssetType::Material;
            }
            if (lower == "shader") {
                return AssetType::Shader;
            }
            return AssetType::Unknown;
        }

    private:
        std::unordered_map<AssetHandle, AssetMeta> m_meta{};
        std::unordered_map<std::filesystem::path, AssetHandle> m_reverse{};
    };
} // namespace Pale

namespace YAML {
    // AssetType as string, with legacy int fallback
    template<>
    struct convert<Pale::AssetType> {
        static Node encode(const Pale::AssetType &rhs) {
            Node node;
            node = Pale::AssetRegistry::toString(rhs);
            return node;
        }

        static bool decode(const Node &node, Pale::AssetType &rhs) {
            if (!node.IsScalar()) {
                return false;
            }
            const auto text = node.as<std::string>();
            rhs = Pale::AssetRegistry::assetTypeFromString(text);
            if (rhs != Pale::AssetType::Unknown) {
                return true;
            }
            try {
                rhs = static_cast<Pale::AssetType>(node.as<int>());
                return true;
            } catch (...) {
                rhs = Pale::AssetType::Unknown;
                return true;
            }
        }
    };

    // UUID as hex/string
    template<>
    struct convert<Pale::UUID> {
        static Node encode(const Pale::UUID &rhs) {
            Node node;
            node = std::string(rhs);
            return node;
        }

        static bool decode(const Node &node, Pale::UUID &rhs) {
            if (!node.IsScalar()) {
                return false;
            }
            rhs = Pale::UUID(node.as<std::string>());
            return true;
        }
    };
}

// -------------------- non-exported definitions --------------------
namespace Pale {
    inline AssetHandle AssetRegistry::import(const std::filesystem::path &filePath, AssetType assetType) {
        AssetHandle assetHandle = UUID();
        AssetMeta assetMeta;
        assetMeta.type = assetType;
        assetMeta.path = filePath;
        if (std::filesystem::exists(filePath)) {
            assetMeta.lastWrite = std::filesystem::last_write_time(filePath);
        }
        m_reverse[filePath] = assetHandle;
        m_meta.emplace(assetHandle, assetMeta);
        return assetHandle;
    }

    inline void AssetRegistry::save(const std::filesystem::path &filePath) const {
        YAML::Emitter emitter;
        emitter << YAML::BeginMap << YAML::Key << "assets" << YAML::Value << YAML::BeginSeq;
        for (const auto &[assetHandle, assetMeta]: m_meta) {
            emitter << YAML::BeginMap;
            emitter << YAML::Key << "id" << YAML::Value << YAML::convert<UUID>::encode(assetHandle);
            emitter << YAML::Key << "type" << YAML::Value << YAML::convert<AssetType>::encode(assetMeta.type);
            emitter << YAML::Key << "path" << YAML::Value << assetMeta.path.string();
            emitter << YAML::EndMap;
        }
        emitter << YAML::EndSeq << YAML::EndMap;
        std::ofstream(filePath) << emitter.c_str();
    }

    inline void AssetRegistry::load(const std::filesystem::path &filePath) {
        if (!std::filesystem::exists(filePath)) return;
        YAML::Node root = YAML::LoadFile(filePath.string());
        if (!root["assets"]) return;

        m_meta.clear();
        m_reverse.clear();

        for (const auto &node: root["assets"]) {
            const auto assetHandle = node["id"].as<UUID>();
            AssetMeta assetMeta;
            assetMeta.type = node["type"].as<AssetType>(); // string or legacy int
            assetMeta.path = node["path"].as<std::string>();
            if (std::filesystem::exists(assetMeta.path))
                assetMeta.lastWrite = std::filesystem::last_write_time(assetMeta.path);

            m_reverse[assetMeta.path] = assetHandle;
            m_meta[assetHandle] = assetMeta;
        }
    }
} // namespace Pale
