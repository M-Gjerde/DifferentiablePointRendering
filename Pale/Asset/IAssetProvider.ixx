module;

#include <filesystem>
#include <optional>

export module Pale.Assets:API;
import :Core;
import :Registry;

export namespace Pale {
    struct IAssetProvider {
        virtual ~IAssetProvider() = default;
        virtual AssetHandle importPath(const std::filesystem::path&, AssetType) = 0;
        virtual std::optional<AssetHandle> findByPath(const std::filesystem::path&) const = 0;
    };


    struct AssetProviderFromRegistry : IAssetProvider {
        AssetRegistry& reg;
        AssetProviderFromRegistry(AssetRegistry& r) : reg(r) {}
        AssetHandle importPath(const std::filesystem::path& p, AssetType t) override {
            if (auto id = reg.findByPath(p)) return *id;
            return reg.import(p, t);
        }
        std::optional<AssetHandle> findByPath(const std::filesystem::path& p) const override {
            return reg.findByPath(p);
        }
    };
}