// Pale/Render/SceneGPU.ixx
module;

#include <cstdint>
#include <vector>
#include <memory>
#include <sycl/sycl.hpp>

export module Pale.Render.SceneGPU;

import Pale.Scene;
import Pale.Assets:Provider;

export namespace Pale {

    // device pointers & counts (triangles, vertices, instances, materials, lights, cameras, BVHs…)
    class SceneGPU {
    public:
        static SceneGPU upload(const std::shared_ptr<Pale::Scene>& scene, sycl::queue queue) {


            return {};
        };

    private:
    };

    struct BuildProducts {

        /*
        // BLAS
        std::vector<Pale::PackedMesh>             packedMeshes;
        std::vector<Pale::BvhNode>                bottomLevelBvhs;

        // TLAS
        std::vector<Pale::TlasNode>               topLevelBvh;
        std::vector<Pale::InstanceRecord>         instanceRecords;

        // Shading
        std::vector<Pale::PackedMaterial>         packedMaterials;
        std::vector<Pale::LightRecord>            lightRecords;
        Pale::AliasTable                          lightAlias;

        // Textures
        Pale::TextureAtlasCPU                     textureAtlas;
        */
    };

    struct BuildOptions {

    };


    class SceneBuild {
    public:
        static BuildProducts build(const std::shared_ptr<Pale::Scene>& scene, IAssetAccess& assets,
                                   const Pale::BuildOptions& buildOptions = {}) {

            return {};
        };
    };

}
