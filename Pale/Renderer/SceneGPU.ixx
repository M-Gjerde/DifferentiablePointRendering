// Pale/Render/SceneGPU.ixx
module;

#include <cstdint>
#include <vector>
#include <memory>
#include <sycl/sycl.hpp>

export module Pale.Render.SceneGPU;

import Pale.UUID;
import Pale.Scene;
import Pale.Assets;
import Pale.Render.GPUDataStructures;

export namespace Pale {

    // device pointers & counts (triangles, vertices, instances, materials, lights, cameras, BVHs…)
    class SceneGPU {
    public:
        static SceneGPU upload(const std::shared_ptr<Pale::Scene>& scene, sycl::queue queue) {


            return {};
        };

    private:
    };


    class SceneBuild {
    public:

        struct InstanceRecord {
            GeometryType geometryType{GeometryType::Mesh};
            uint32_t     geometryIndex{0};      // meshRanges index or pointRanges index
            uint32_t     materialIndex{0};      // dense index into GPUMaterial array
            uint32_t     transformIndex{0};     // index into transforms
            std::string  name;
        };

        struct BLASResult {
            std::vector<BVHNode> nodes;   // the BLAS node array
            BLASRange range;              // [firstNode, nodeCount] in the global BVH node array
            std::vector<uint32_t> triPermutation; // local triangle reordering
            uint32_t meshIndex;           // to associate back with meshRanges
        };

        struct TLASResult {
            std::vector<TLASNode> nodes;  // top-level BVH nodes
            uint32_t rootIndex;           // index of the root (often 0)
        };


        struct BuildProducts {

            std::vector<BVHNode> bottomLevelNodes;     // concatenated BLAS nodes
            std::vector<BLASRange> bottomLevelRanges;  // [offset,count] per mesh
            std::vector<TLASNode> topLevelNodes;       // single TLAS
            std::vector<uint32_t> trianglePermutation; // maps old->new tri order

            // optional packing for upload
            uint32_t nodeStrideBytes{0};
            uint32_t blasNodeBaseOffset{0};
            uint32_t tlasNodeBaseOffset{0};

            std::vector<Vertex>        vertices;
            std::vector<Triangle>      triangles;
            std::vector<MeshRange>     meshRanges;
            std::unordered_map<UUID,uint32_t> meshIndexById;

            std::vector<Transform>     transforms;    // index by transformIndex
            std::vector<GPUMaterial>   materials;     // index by materialIndex
            std::vector<InstanceRecord> instances;
        };

        struct BuildOptions {
            uint32_t bvhMaxLeafTriangles{4};
            uint32_t sahBinCount{32};
            uint32_t tlasMaxLeafInstances{1};
        };


        static BuildProducts build(const std::shared_ptr<Scene>& scene, IAssetAccess& assetAccess,
                                   const  BuildOptions& buildOptions) {
            BuildProducts buildProducts;

            collectGeometry(scene, assetAccess, buildProducts);

            collectInstances(scene,
                             assetAccess,
                             buildProducts.meshIndexById,
                             buildProducts);

            for (uint32_t meshIndex = 0; meshIndex < buildProducts.meshRanges.size(); ++meshIndex) {
                BLASResult blasResult = buildMeshBLAS(meshIndex, buildProducts.meshRanges[meshIndex], buildOptions);
                appendBLAS(buildProducts, blasResult);
            }

            // TLAS over instances
            TLASResult tlasResult =
                buildTLAS(buildProducts.instances,
                          buildProducts.bottomLevelRanges,
                          buildOptions);
            buildProducts.topLevelNodes = std::move(tlasResult.nodes);

            // finalize permutation and packing metadata
            finalizePermutation(buildProducts);
            computePacking(buildProducts);

            return buildProducts;

        };

    private:

        static void collectGeometry(const std::shared_ptr<Scene>& scene,
                                    IAssetAccess& assetAccess,
                                    BuildProducts& outBuildProducts);

        static void collectInstances(const std::shared_ptr<Scene>& scene,
                                     IAssetAccess& assetAccess,
                                     const std::unordered_map<UUID,uint32_t>& meshIndexById,
                                     BuildProducts& outBuildProducts);

        static BLASResult buildMeshBLAS(uint32_t meshIndex, const MeshRange&, const BuildOptions&);
        static TLASResult buildTLAS(const std::vector<InstanceRecord>&,
                                    const std::vector<BLASRange>&,
                                    const BuildOptions&);

        static void appendBLAS(BuildProducts& buildProducts, const BLASResult& blasResult);
        static void finalizePermutation(BuildProducts& buildProducts);
        static void computePacking(BuildProducts& buildProducts);

    };

}
