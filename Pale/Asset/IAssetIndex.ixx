module;

#include <filesystem>
#include <optional>

export module Pale.Assets.API;

import Pale.Assets.Core;
import Pale.Assets.Registry;

export namespace Pale {
    struct IAssetIndex {
        virtual ~IAssetIndex() = default;
        virtual AssetHandle importPath(const std::filesystem::path&, AssetType) = 0;
        virtual std::optional<AssetHandle> findByPath(const std::filesystem::path&) const = 0;
    };

    struct AssetIndexFromRegistry final : IAssetIndex {
        AssetRegistry& registry;
        explicit AssetIndexFromRegistry(AssetRegistry& r) : registry(r) {}
        AssetHandle importPath(const std::filesystem::path& p, AssetType t) override {
            if (auto id = registry.findByPath(p)) return *id;
            return registry.import(p, t);
        }
        std::optional<AssetHandle> findByPath(const std::filesystem::path& p) const override {
            return registry.findByPath(p);
        }
    };
}