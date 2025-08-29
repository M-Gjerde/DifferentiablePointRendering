module;
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>


export module Pale.Assets:Mesh;


import :Core;


export namespace Pale {


    struct Submesh {
        std::vector<glm::vec3> positions{};
        std::vector<glm::vec3> normals{};
        std::vector<glm::vec2> uvs{};
        std::vector<std::uint32_t> indices{};
        int materialIndex{-1};
    };


    struct Mesh : IAsset {
        std::vector<Submesh> submeshes{};
    };


} // namespace Pale