//
// Created by magnus on 8/29/25.
//
module;
#include <memory>

export module Pale.Assets:Provider;

import :API;
import :Manager;
import :Mesh;
import :Material;
import :Core;

export namespace Pale {
    struct IAssetAccess {
        virtual ~IAssetAccess() = default;
        virtual AssetPtr<Mesh>     getMesh(AssetHandle) = 0;
        virtual AssetPtr<Material> getMaterial(AssetHandle) = 0;
    };

    class AssetAccessFromManager final : public IAssetAccess {
    public:
        explicit AssetAccessFromManager(AssetManager& m) : m_assetManager(m) {}
        AssetPtr<Mesh>     getMesh(AssetHandle h) override     { return m_assetManager.get<Mesh>(h); }
        AssetPtr<Material> getMaterial(AssetHandle h) override { return m_assetManager.get<Material>(h); }
    private:
        AssetManager& m_assetManager;
    };
}
