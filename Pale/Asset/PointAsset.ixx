module;
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
    };


    struct PointAsset : IAsset {
        std::vector<PointGeometry> points{};
    };


} // namespace Pale
