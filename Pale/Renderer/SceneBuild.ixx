// Pale/Render/SceneGPU.ixx
module;

#include <cstdint>
#include <vector>
#include <memory>
#include <sycl/sycl.hpp>

#include "Renderer/GPUDataStructures.h"

export module Pale.Render.SceneBuild;

import Pale.UUID;
import Pale.Scene;
import Pale.Assets;

export namespace Pale {

    class SceneBuild {
    public:
        struct InstanceRecord {
            GeometryType geometryType{GeometryType::Mesh};
            uint32_t geometryIndex{0}; // meshRanges index or pointRanges index
            uint32_t materialIndex{0}; // dense index into GPUMaterial array
            uint32_t transformIndex{0}; // index into transforms
            std::string name;
        };

        struct BLASResult {
            std::vector<BVHNode> nodes; // the BLAS node array
            BLASRange range; // [firstNode, nodeCount] in the global BVH node array
            std::vector<uint32_t> triPermutation; // local triangle reordering
            uint32_t meshIndex; // to associate back with meshRanges
        };

        struct TLASResult {
            std::vector<TLASNode> nodes; // top-level BVH nodes
            uint32_t rootIndex; // index of the root (often 0)
        };


        struct BuildProducts {
            std::vector<BVHNode> bottomLevelNodes; // concatenated BLAS nodes
            std::vector<BLASRange> bottomLevelRanges; // [offset,count] per mesh
            std::vector<TLASNode> topLevelNodes; // single TLAS
            std::vector<uint32_t> trianglePermutation; // maps old->new tri order

            std::vector<Vertex> vertices;
            std::vector<Triangle> triangles;
            std::vector<MeshRange> meshRanges;
            std::unordered_map<UUID, uint32_t> meshIndexById;

            std::vector<Transform> transforms; // index by transformIndex
            std::vector<GPUMaterial> materials; // index by materialIndex
            std::vector<InstanceRecord> instances;
        };

        struct BuildOptions {
            uint32_t bvhMaxLeafTriangles{4};
            uint32_t sahBinCount{32};
            uint32_t tlasMaxLeafInstances{1};
        };


        static void write_tlas_dot(const std::vector<TLASNode>& nodes, const char* path) {
            std::ofstream f(path);
            f << "digraph TLAS{\nnode[shape=box,fontname=mono];\n";
            for (size_t i = 0; i < nodes.size(); ++i) {
                const auto& n = nodes[i];
                f << i << " [label=\"#" << i
                  << "\\ncount=" << n.count
                  << "\\nmin=(" << n.aabbMin.x() << "," << n.aabbMin.y() << "," << n.aabbMin.z() << ")"
                  << "\\nmax=(" << n.aabbMax.x() << "," << n.aabbMax.y() << "," << n.aabbMax.z() << ")\"];\n";
                if (n.count == 0) { f << i << " -> " << n.leftChild  << ";\n"
                                      << i << " -> " << n.rightChild << ";\n"; }
            }
            f << "}\n";
        }

        static void write_tlas_csv(const std::vector<TLASNode>& nodes, const char* path) {
            std::ofstream f(path);
            f << "node,left,right,count,minx,miny,minz,maxx,maxy,maxz\n";
            for (size_t i = 0; i < nodes.size(); ++i) {
                const auto& n = nodes[i];
                f << i << "," << n.leftChild << "," << n.rightChild << "," << n.count << ","
                  << n.aabbMin.x() << "," << n.aabbMin.y() << "," << n.aabbMin.z() << ","
                  << n.aabbMax.x() << "," << n.aabbMax.y() << "," << n.aabbMax.z() << "\n";
            }
        }

        static BuildProducts build(const std::shared_ptr<Scene> &scene, IAssetAccess &assetAccess,
                                   const BuildOptions &buildOptions) {
            BuildProducts buildProducts;

            collectGeometry(scene, assetAccess, buildProducts);

            collectInstances(scene,
                             assetAccess,
                             buildProducts.meshIndexById,
                             buildProducts);

            for (uint32_t meshIndex = 0; meshIndex < buildProducts.meshRanges.size(); ++meshIndex) {
                const MeshRange& meshRange = buildProducts.meshRanges[meshIndex];
                BLASResult blasResult = buildMeshBLAS(meshIndex,
                                                      meshRange,
                                                      buildProducts.triangles,
                                                      buildProducts.vertices,
                                                      buildOptions);
                appendBLAS(buildProducts, blasResult);
            }
            TLASResult tlasResult = buildTLAS(buildProducts.instances,
                                              buildProducts.bottomLevelRanges,
                                              buildProducts.bottomLevelNodes,
                                              buildProducts.transforms,
                                              buildOptions);
            buildProducts.topLevelNodes = std::move(tlasResult.nodes);

            write_tlas_dot(buildProducts.topLevelNodes, "tlas.dot");
            write_tlas_csv(buildProducts.topLevelNodes, "tlas.csv");
            // finalize permutation and packing metadata

            computePacking(buildProducts);

            return buildProducts;
        };

    private:
        static void collectGeometry(const std::shared_ptr<Scene> &scene,
                                    IAssetAccess &assetAccess,
                                    BuildProducts &outBuildProducts);

        static void collectInstances(const std::shared_ptr<Scene> &scene,
                                     IAssetAccess &assetAccess,
                                     const std::unordered_map<UUID, uint32_t> &meshIndexById,
                                     BuildProducts &outBuildProducts);

        static BLASResult buildMeshBLAS(
            uint32_t meshIndex,
            const MeshRange &meshRange,
            const std::vector<Triangle> &allTriangles,
            const std::vector<Vertex> &allVertices,
            const BuildOptions &buildOptions);


        static TLASResult buildTLAS(const std::vector<InstanceRecord>& instances,
                                    const std::vector<BLASRange>& blasRanges,
                                    const std::vector<BVHNode>& blasNodes,
                                    const std::vector<Transform>& transforms,
                                    const BuildOptions& opts);

        static void appendBLAS(BuildProducts &buildProducts, const BLASResult &blasResult);

        static void computePacking(BuildProducts &buildProducts);
    };
}
