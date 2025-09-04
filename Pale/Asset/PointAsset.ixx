module;
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>


export module Pale.Assets:Point;


import Pale.Assets.Core;


export namespace Pale {


    struct PointGeometry {
        std::vector<glm::vec3> positions{};
        std::vector<glm::vec3> tanU{};
        std::vector<glm::vec3> tanV{};
        std::vector<glm::vec2> scale{};

        std::vector<glm::vec3> colors{};
        std::vector<float> opacities{};
    };


    struct Point : IAsset {
        std::vector<PointGeometry> points{};
    };


} // namespace Pale
