module;
#include <cstdint>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "glm/fwd.hpp"


export module Pale.Assets:Point;


import Pale.Assets.Core;


export namespace Pale {


    struct PointGeometry {
        std::vector<glm::vec3> positions{};
        std::vector<glm::quat> quat{};
        std::vector<glm::vec2> scales{};
        std::vector<glm::vec3> albedos{};
        std::vector<float>     opacities{};
        std::vector<float>     betas{};
        std::vector<float>     shapes{};
        std::vector<float>     powers{};
        // 0=initial/unknown, 1=clone, 2=position-gradient split, 3=curvature split.
        // This is diagnostic metadata; rendering does not use it.
        std::vector<std::uint8_t> densificationOrigins{};
        // Number of optimization iterations since this primitive was created
        // or last split. Diagnostic metadata only.
        std::vector<std::uint32_t> primitiveAges{};
        // Exact optimizer position-densification statistics at snapshot time.
        // These are diagnostic metadata and are not consumed by rendering.
        std::vector<float> densificationPositionSignals{};
        std::vector<std::uint32_t> densificationPositionSampleCounts{};
        std::vector<float> densificationPositionThresholds{};
        // Optional inputs needed to recompute radiance-biased thresholds in
        // diagnostic viewers. The base threshold is repeated per vertex in PLY.
        std::vector<float> densificationPositionRadianceRms{};
        std::vector<float> densificationPositionBaseThresholds{};
    };


    struct PointAsset : IAsset {
        std::vector<PointGeometry> points{};
    };


} // namespace Pale
