module;

#include <glm/glm.hpp>

export module Pale.Assets:Material;
import :Core;

export namespace Pale {

    struct Material : IAsset {
        // Core PBR-ish knobs (tune to your renderer)
        glm::vec3 baseColor{1,1,1};
        float roughness{0.5f};
        float metallic{0.0f};
        float ior{1.5f};
        glm::vec3 emissive{0,0,0};
        float opacity{1.0f};

        // Texture handles (optional)
        AssetHandle baseColorTex{};
        AssetHandle metallicRoughnessTex{};
        AssetHandle normalTex{};
        AssetHandle emissiveTex{};
        AssetHandle opacityTex{};
    };

} // namespace Pale