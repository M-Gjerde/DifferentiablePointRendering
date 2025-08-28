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


    private:
        std::unordered_map<AssetHandle, AssetMeta> m_meta{};
        std::unordered_map<std::filesystem::path, AssetHandle> m_reverse{};
    };



} // namespace Pale