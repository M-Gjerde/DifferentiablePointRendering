// ================================
// File: Pale.Assets.Registry.ixx
// ================================
module;

#include <filesystem>
#include <optional>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

export module Pale.Assets.Registry;

export import Pale.Assets.Core;

export namespace Pale {


    class AssetRegistry {
    public:
        bool has(const AssetHandle& id) const { return m_meta.contains(id); }


        const AssetMeta* meta(const AssetHandle& id) const {
            if (auto it = m_meta.find(id); it != m_meta.end()) return &it->second;
            return nullptr;
        }


        std::optional<AssetHandle> findByPath(const std::filesystem::path& p) const {
            if (auto it = m_reverse.find(p); it != m_reverse.end()) return it->second;
            return std::nullopt;
        }


        AssetHandle import(const std::filesystem::path& path, AssetType type);


        void save(const std::filesystem::path& file) const;
        void load(const std::filesystem::path& file);


        // Expose a shallow copy for watcher (avoid long-held locks)
        std::unordered_map<AssetHandle, AssetMeta> snapshot() const { return m_meta; }

        static inline std::string toString(AssetType t) {
            switch (t) {
                case AssetType::Mesh:     return "Mesh";
                case AssetType::Material: return "Material";
                case AssetType::Shader:   return "Shader";
                default:                  return "Unknown";
            }
        }

        static inline AssetType assetTypeFromString(std::string_view s) {
            auto lower = std::string(s);
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return std::tolower(c); });
            if (lower == "mesh")     return AssetType::Mesh;
            if (lower == "material") return AssetType::Material;
            if (lower == "shader")   return AssetType::Shader;
            return AssetType::Unknown;
        }


    private:
        std::unordered_map<AssetHandle, AssetMeta> m_meta;
        std::unordered_map<std::filesystem::path, AssetHandle> m_reverse{};
    };



} // namespace Pale

