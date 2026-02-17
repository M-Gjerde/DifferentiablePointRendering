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
        struct BLASResult {
            std::vector<Triangle> localTriangles;
            std::vector<Point> localPoints; // points

            std::vector<BVHNode> nodes; // the BLAS node array
            BLASRange range; // [firstNode, nodeCount] in the global BVH node array
            std::vector<uint32_t> triPermutation; // local triangle reordering
            std::vector<uint32_t> pointPermutation; // local point reordering

            uint32_t meshIndex; // to associate back with meshRanges
            uint32_t pointCloudIndex; // Associate with point clouds
        };

        struct TLASResult {
            std::vector<TLASNode> nodes; // top-level BVH nodes
            uint32_t rootIndex; // index of the root (often 0)
        };

        struct PointCloudRange {
            uint32_t firstPoint;
            uint32_t pointCount;
        };

        struct BuildProducts {
            std::vector<BVHNode> bottomLevelNodes; // concatenated BLAS nodes
            std::vector<BLASRange> bottomLevelRanges; // [offset,count] per mesh
            std::vector<TLASNode> topLevelNodes; // single TLAS

            std::vector<Vertex> vertices;
            std::vector<Triangle> triangles;
            std::vector<MeshRange> meshRanges;
            std::unordered_map<UUID, uint32_t> meshIndexById;

            std::vector<Point> points; // global SoA/AoS slice
            std::vector<PointCloudRange> pointCloudRanges;
            std::unordered_map<UUID, uint32_t> pointCloudIndexById;

            std::vector<Transform> transforms; // index by transformIndex
            std::vector<GPUMaterial> materials; // index by materialIndex
            std::vector<InstanceRecord> instances;

            std::vector<GPULightRecord> lights;
            std::vector<GPUEmissiveTriangle> emissiveTriangles;

            std::vector<CameraGPU> cameraGPUs; // camera data for sensors

            float diffuseSurfaceArea = 0.0f;

            [[nodiscard]] std::size_t cameraCount() const { return cameraGPUs.size(); }
            [[nodiscard]] const std::vector<CameraGPU>& cameras() const { return cameraGPUs; }
        };

        struct BuildOptions {
            uint32_t bvhMaxLeafTriangles = 4;
            uint32_t bvhMaxLeafPoints = 4;
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
                if (n.count == 0) {
                    f << i << " -> " << n.leftChild << ";\n"
                        << i << " -> " << n.rightChild << ";\n";
                }
            }
            f << "}\n";
        }
        static void write_blas_csv(const std::vector<BVHNode>& allNodes,
                                   BLASRange range,
                                   const char* path) {
            std::ofstream f(path);

            f << "left right primitiveIdx aabb_min_x aabb_min_y aabb_min_z "
                 "aabb_max_x aabb_max_y aabb_max_z\n";

            const uint32_t first = range.firstNode;
            const uint32_t end = range.firstNode + range.nodeCount;

            for (uint32_t globalNodeIndex = first; globalNodeIndex < end; ++globalNodeIndex) {
                const BVHNode& node = allNodes[globalNodeIndex];
                const bool isLeaf = node.isLeaf();

                // Use indices local to this BLAS for readability in external tools
                const int localNodeIndex = static_cast<int>(globalNodeIndex - first);

                int left = -1;
                int right = -1;
                int primitiveIdx = -1;

                if (isLeaf) {
                    // leftFirst is already patched to global primitive base (triangles or points)
                    primitiveIdx = static_cast<int>(node.leftFirst);
                } else {
                    // Assumption: right child is implicit
                    // leftFirst stores the *global* index of the left child node.
                    left  = static_cast<int>(node.leftFirst - first);
                    right = static_cast<int>(node.leftFirst + 1u - first);
                }

                f << left << " "
                  << right << " "
                  << primitiveIdx << " "
                  << node.aabbMin.x() << " "
                  << node.aabbMin.y() << " "
                  << node.aabbMin.z() << " "
                  << node.aabbMax.x() << " "
                  << node.aabbMax.y() << " "
                  << node.aabbMax.z() << "\n";
            }
        }

        static void write_tlas_csv(const std::vector<TLASNode>& nodes, const char* path) {
            std::ofstream f(path);

            // Exact header required
            f << "left right primitiveIdx aabb_min_x aabb_min_y aabb_min_z "
                 "aabb_max_x aabb_max_y aabb_max_z\n";

            for (size_t i = 0; i < nodes.size(); ++i) {
                const TLASNode& n = nodes[i];

                const bool isLeaf = (n.count == 1);

                const int left  = isLeaf ? -1 : static_cast<int>(n.leftChild);
                const int right = isLeaf ? -1 : static_cast<int>(n.rightChild);

                // For TLAS: primitiveIdx = instance index for leaf, -1 otherwise
                const int primitiveIdx = isLeaf ? static_cast<int>(n.leftChild) : -1;

                f << left << " "
                  << right << " "
                  << primitiveIdx << " "
                  << n.aabbMin.x() << " "
                  << n.aabbMin.y() << " "
                  << n.aabbMin.z() << " "
                  << n.aabbMax.x() << " "
                  << n.aabbMax.y() << " "
                  << n.aabbMax.z() << "\n";
            }
        }


        // Existing top-level builder (now a thin wrapper)
        static BuildProducts build(const std::shared_ptr<Scene>& scene,
                                   IAssetAccess& assetAccess,
                                   const BuildOptions& buildOptions);

        // New: only (re)build BLAS/TLAS on existing BuildProducts
        static void rebuildBVHs(const std::shared_ptr<Scene> &scene, IAssetAccess &assetAccess, BuildProducts &buildProducts,
                                 const BuildOptions &buildOptions);


        /*
        static BuildProducts build(const std::shared_ptr<Scene>& scene, IAssetAccess& assetAccess,
                                   const BuildOptions& buildOptions) {
            BuildProducts buildProducts;
            collectGeometry(scene, assetAccess, buildProducts);
            collectInstances(scene,
                             assetAccess,
                             buildProducts.meshIndexById,
                             buildProducts);
            collectPointCloudGeometry(scene, assetAccess, buildProducts);
            collectPointCloudInstances(scene, buildProducts);
            collectLights(scene, assetAccess, buildProducts);
            collectCameras(scene, buildProducts);


            std::vector<uint32_t> meshRangeToBlasRange(buildProducts.meshRanges.size(), UINT32_MAX);
            for (uint32_t meshIndex = 0; meshIndex < buildProducts.meshRanges.size(); ++meshIndex) {
                const MeshRange& meshRange = buildProducts.meshRanges[meshIndex];
                BLASResult blasResult = buildMeshBLAS(meshIndex,
                                                      meshRange,
                                                      buildProducts.triangles,
                                                      buildProducts.vertices,
                                                      buildOptions);
                uint32_t globalTriStart = meshRange.firstTri;
                // 1.  Temporary copy that will hold triangles in BVH order
                std::vector<Triangle> reordered;
                reordered.reserve(blasResult.localTriangles.size());
                for (unsigned int i : blasResult.triPermutation) {
                    Triangle T = blasResult.localTriangles[i];
                    // convert vertex indices back to GLOBAL space
                    T.v0 += meshRange.firstVert;
                    T.v1 += meshRange.firstVert;
                    T.v2 += meshRange.firstVert;
                    reordered.push_back(T);
                }
                // 2.  Overwrite the slice in m_tris with the reordered triangles
                std::copy(reordered.begin(), reordered.end(),
                          buildProducts.triangles.begin() + globalTriStart);
                // 3.  Now patch the BVH nodes ----------------------------------------------
                //     (children still contiguous, only need global offset)
                uint32_t firstNode = uint32_t(buildProducts.bottomLevelNodes.size());
                // 4C) Patch all *leaf* nodes so their triangle slices shift by globalTriStart
                for (BVHNode& N : blasResult.nodes) {
                    if (N.isLeaf())
                        N.leftFirst += globalTriStart;
                }
                //---------------- 5.  Append to the big BLAS pool & record range -----
                buildProducts.bottomLevelNodes.insert(buildProducts.bottomLevelNodes.end(),
                                                      blasResult.nodes.begin(), blasResult.nodes.end());

                const uint32_t blasRangeIndex = static_cast<uint32_t>(buildProducts.bottomLevelRanges.size());

                buildProducts.bottomLevelRanges.push_back({
                    firstNode,
                    uint32_t(blasResult.nodes.size())
                });

                meshRangeToBlasRange[meshIndex] = blasRangeIndex;

            }

            std::vector<uint32_t> pointRangeToBlasRange(buildProducts.pointCloudRanges.size(), UINT32_MAX);
            for (uint32_t pointCloudIndex = 0; pointCloudIndex < buildProducts.pointCloudRanges.size(); ++
                 pointCloudIndex) {
                const PointCloudRange& pointCloudRange = buildProducts.pointCloudRanges[pointCloudIndex];
                BLASResult blasResult = buildPointCloudBLAS(pointCloudIndex,
                                                            pointCloudRange,
                                                            buildProducts.points,
                                                            buildOptions);

                const uint32_t globalPointStart = pointCloudRange.firstPoint;

                // Reorder points into BVH order
                std::vector<Point> reorderedPoints;
                reorderedPoints.reserve(blasResult.localPoints.size());
                for (uint32_t localIdx : blasResult.pointPermutation) {
                    reorderedPoints.push_back(blasResult.localPoints[localIdx]);
                }
                std::copy(reorderedPoints.begin(), reorderedPoints.end(),
                          buildProducts.points.begin() + globalPointStart);

                // Patch leaf node ranges from local to global
                const uint32_t firstNode = static_cast<uint32_t>(buildProducts.bottomLevelNodes.size());
                for (BVHNode& node : blasResult.nodes) {
                    if (node.isLeaf()) node.leftFirst += globalPointStart;
                }

                // Append nodes and record BLAS range
                buildProducts.bottomLevelNodes.insert(buildProducts.bottomLevelNodes.end(),
                                                      blasResult.nodes.begin(), blasResult.nodes.end());
                const uint32_t nodeCount = static_cast<uint32_t>(blasResult.nodes.size());
                const uint32_t blasRangeIndex = static_cast<uint32_t>(buildProducts.bottomLevelRanges.size());
                buildProducts.bottomLevelRanges.push_back({ firstNode, nodeCount });

                pointRangeToBlasRange[pointCloudIndex] = blasRangeIndex;

            }

            for (auto& instanceRecord : buildProducts.instances) {
                if (instanceRecord.geometryType == GeometryType::Mesh) {
                    // If you didn’t keep a meshRange->BLASRange map, compute it where you append mesh BLAS
                    // and store here. Assuming you have meshRangeToBlasRange:
                    instanceRecord.blasRangeIndex = meshRangeToBlasRange[instanceRecord.geometryIndex];
                } else {
                    instanceRecord.blasRangeIndex = pointRangeToBlasRange[instanceRecord.geometryIndex];
                }
            }

            TLASResult tlasResult = buildTLAS(buildProducts.instances,
                                              buildProducts.bottomLevelRanges,
                                              buildProducts.bottomLevelNodes,
                                              buildProducts.transforms,
                                              buildOptions);
            buildProducts.topLevelNodes = std::move(tlasResult.nodes);
            //write_tlas_dot(buildProducts.topLevelNodes, "tlas.dot");
            //write_tlas_csv(buildProducts.topLevelNodes, "tlas.csv");
            // finalize permutation and packing metadata

            buildProducts.diffuseSurfaceArea = computeDiffuseSurfaceAreaWorld(buildProducts);

            return buildProducts;
        };
        */

    private:
        static void collectGeometry(const std::shared_ptr<Scene>& scene,
                                    IAssetAccess& assetAccess,
                                    BuildProducts& outBuildProducts);

        static void collectInstances(const std::shared_ptr<Scene>& scene,
                                     IAssetAccess& assetAccess,
                                     const std::unordered_map<UUID, uint32_t>& meshIndexById,
                                     BuildProducts& outBuildProducts);

        static void collectPointCloudGeometry(const std::shared_ptr<Scene>& scene,
                                              IAssetAccess& assetAccess,
                                              BuildProducts& outBuildProducts);

        static void collectPointCloudInstances(const std::shared_ptr<Scene>& scene,
                                               BuildProducts& outBuildProducts);

        /*
        BLASResult buildPointCloudBLAS(uint32_t pointCloudIndex,
                                                   const PointCloudRange& pointCloudRange,
                                                   const std::vector<Surfel>& allSurfels,
                                                   const BuildOptions& buildOptions);
        */

        static void collectLights(const std::shared_ptr<Scene>& scene,
                                  IAssetAccess& assetAccess,
                                  BuildProducts& out);
        static float triangleArea(const float3& p0, const float3& p1, const float3& p2);
        static void collectCameras(const std::shared_ptr<Scene>& scene,
                                   BuildProducts& outBuildProducts);

        static BLASResult buildMeshBLAS(
            uint32_t meshIndex,
            const MeshRange& meshRange,
            const std::vector<Triangle>& allTriangles,
            const std::vector<Vertex>& allVertices,
            const BuildOptions& buildOptions);

        static BLASResult buildPointCloudBLAS(
            uint32_t pcIndex,
            const PointCloudRange& pcRange,
            const std::vector<Point>& allPoints,
            const BuildOptions& buildOptions);


        static TLASResult buildTLAS(const std::vector<InstanceRecord>& instances,
                                    const std::vector<BLASRange>& blasRanges,
                                    const std::vector<BVHNode>& blasNodes,
                                    const std::vector<Transform>& transforms,
                                    const BuildOptions& opts);

        // New: high-level BVH phases
        static void buildBottomLevelBVHs(BuildProducts& buildProducts,
                                         const BuildOptions& buildOptions);

        static void buildTopLevelBVH(BuildProducts& buildProducts,
                                     const BuildOptions& buildOptions);

        static float computeDiffuseSurfaceAreaWorld(const BuildProducts& buildProducts);

    };
}
