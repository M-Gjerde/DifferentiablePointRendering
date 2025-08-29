//
// Created by magnus on 8/29/25.
//

module;

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <glm/glm.hpp>

module Pale.Render.SceneBuild;

import Pale.Scene.Components;
import Pale.Render.GPUDataTypes;
import Pale.Render.BVH;

namespace Pale {
    // ==== Geometry collector ====
    void SceneBuild::collectGeometry(const std::shared_ptr<Scene> &scene,
                                     IAssetAccess &assetAccess,
                                     BuildProducts &outBuildProducts) {
        std::vector<Vertex> vertices;
        std::vector<Triangle> triangles;

        // 1) Gather unique mesh IDs in scene order
        std::vector<UUID> uniqueMeshIds; {
            std::unordered_set<UUID> seen;
            for (auto [e, mesh]: scene->getAllEntitiesWith<MeshComponent>().each()) {
                if (seen.insert(mesh.meshID).second) uniqueMeshIds.push_back(mesh.meshID);
            }
        }

        // 2) For each unique mesh, append geometry once and record ALL ranges
        for (const UUID &meshId: uniqueMeshIds) {
            const auto meshAsset = assetAccess.getMesh(meshId);

            Submesh mesh = meshAsset->submeshes.front();

            const uint32_t vertexBaseIndex = static_cast<uint32_t>(vertices.size());
            const uint32_t triangleBaseIndex = static_cast<uint32_t>(triangles.size());

            for (size_t i = 0; i < mesh.positions.size(); ++i) {
                Vertex gpuVertex{};
                gpuVertex.pos = float3{mesh.positions[i].x, mesh.positions[i].y, mesh.positions[i].z};
                gpuVertex.norm = float3{mesh.normals[i].x, mesh.normals[i].y, mesh.normals[i].z};
                vertices.push_back(gpuVertex);
            }
            constexpr float oneThird = 1.0f / 3.0f;
            for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                const uint32_t i0 = mesh.indices[i + 0] + vertexBaseIndex;
                const uint32_t i1 = mesh.indices[i + 1] + vertexBaseIndex;
                const uint32_t i2 = mesh.indices[i + 2] + vertexBaseIndex;

                Triangle tri{};
                tri.v0 = i0;
                tri.v1 = i1;
                tri.v2 = i2;

                const float3 p0 = vertices[i0].pos;
                const float3 p1 = vertices[i1].pos;
                const float3 p2 = vertices[i2].pos;
                tri.centroid = (p0 + p1 + p2) * oneThird;

                triangles.push_back(tri);
            }

            MeshRange meshRange{};
            meshRange.firstVert = vertexBaseIndex;
            meshRange.vertCount = static_cast<uint32_t>(mesh.positions.size());
            meshRange.firstTri = triangleBaseIndex;
            meshRange.triCount = static_cast<uint32_t>(mesh.indices.size() / 3);

            const uint32_t rangeIndex = static_cast<uint32_t>(outBuildProducts.meshRanges.size());
            outBuildProducts.meshRanges.push_back(meshRange);

            outBuildProducts.meshIndexById[meshId] = rangeIndex;
        }
        outBuildProducts.vertices = std::move(vertices);
        outBuildProducts.triangles = std::move(triangles);
    }

    void SceneBuild::collectInstances(const std::shared_ptr<Pale::Scene> &scene,
                                      IAssetAccess &assetAccess,
                                      const std::unordered_map<UUID, uint32_t> &meshIndexById,
                                      BuildProducts &outBuildProducts) {
        std::unordered_map<UUID, uint32_t> materialIndexByUuid;

        auto view = scene->getAllEntitiesWith<MeshComponent, MaterialComponent, TransformComponent, TagComponent>();
        for (auto [entityId, meshComponent, materialComponent, transformComponent, tagComponent]: view.each()) {
            auto it = meshIndexById.find(meshComponent.meshID);
            if (it == meshIndexById.end()) continue;
            const uint32_t geometryIndex = it->second;

            // material de-dup
            uint32_t materialIndex;
            if (auto mit = materialIndexByUuid.find(materialComponent.materialID); mit != materialIndexByUuid.end()) {
                materialIndex = mit->second;
            } else {
                const auto materialAsset = assetAccess.getMaterial(materialComponent.materialID);
                GPUMaterial gpuMaterial{};
                gpuMaterial.baseColor = materialAsset->baseColor;
                gpuMaterial.specular = materialAsset->metallic;
                gpuMaterial.diffuse = materialAsset->roughness;
                gpuMaterial.phongExp = 16;

                materialIndex = static_cast<uint32_t>(outBuildProducts.materials.size());
                outBuildProducts.materials.push_back(gpuMaterial);
                materialIndexByUuid.emplace(materialComponent.materialID, materialIndex);
            }

            // transform
            Transform gpuTransform{};
            const glm::mat4 objectToWorldGLM = transformComponent.getTransform();
            gpuTransform.objectToWorld = glm2sycl(objectToWorldGLM);
            gpuTransform.worldToObject = glm2sycl(glm::inverse(objectToWorldGLM));
            const uint32_t transformIndex = static_cast<uint32_t>(outBuildProducts.transforms.size());
            outBuildProducts.transforms.push_back(gpuTransform);

            outBuildProducts.instances.push_back(InstanceRecord{
                .geometryType = GeometryType::Mesh,
                .geometryIndex = geometryIndex,
                .materialIndex = materialIndex,
                .transformIndex = transformIndex,
                .name = tagComponent.tag
            });
        }
    }

    // ---- BLAS build: localize -> build -> return nodes + permutation ----
    SceneBuild::BLASResult
    SceneBuild::buildMeshBLAS(uint32_t meshIndex,
                              const MeshRange &meshRange,
                              const std::vector<Triangle> &allTriangles,
                              const std::vector<Vertex> &allVertices,
                              const BuildOptions &buildOptions) {
        // 1) local vertices
        std::vector<Vertex> localVertices;
        localVertices.reserve(meshRange.vertCount);
        for (uint32_t v = 0; v < meshRange.vertCount; ++v)
            localVertices.push_back(allVertices[meshRange.firstVert + v]);

        // 2) local triangles with re-indexed vertices
        std::vector<Triangle> localTriangles;
        localTriangles.reserve(meshRange.triCount);
        for (uint32_t t = 0; t < meshRange.triCount; ++t) {
            Triangle tri = allTriangles[meshRange.firstTri + t];
            tri.v0 -= meshRange.firstVert;
            tri.v1 -= meshRange.firstVert;
            tri.v2 -= meshRange.firstVert;
            localTriangles.push_back(tri);
        }

        // 3) build BVH over local data
        std::vector<BVHNode> localNodes;
        std::vector<uint32_t> localTriangleOrder; // permutation: new_order[i] = old_local_index
        BasicBVH::build(localTriangles,
                        localVertices,
                        localNodes,
                        localTriangleOrder,
                        buildOptions.bvhMaxLeafTriangles);

        // range.firstNode is filled in appendBLAS
        BLASResult result{};
        result.nodes = std::move(localNodes);
        result.range = {0u, 0u};
        result.triPermutation = std::move(localTriangleOrder);
        result.meshIndex = meshIndex;
        return result;
    }


    // ---- splice into global pools, reorder tris, patch leaves, record range ----
    void SceneBuild::appendBLAS(BuildProducts &buildProducts,
                                const BLASResult &blasResult) {
        const MeshRange &meshRange = buildProducts.meshRanges[blasResult.meshIndex];
        const uint32_t globalTriangleStart = meshRange.firstTri;
        const uint32_t localTriangleCount = meshRange.triCount;

        // A) reorder the mesh triangle slice using the permutation
        //    triPermutation[i_new] = i_old (local indices)
        std::vector<Triangle> reorderedTriangles;
        reorderedTriangles.reserve(localTriangleCount);
        for (uint32_t iNew = 0; iNew < localTriangleCount; ++iNew) {
            const uint32_t oldLocal = blasResult.triPermutation[iNew];
            reorderedTriangles.push_back(
                buildProducts.triangles[globalTriangleStart + oldLocal]
            );
        }
        std::copy(reorderedTriangles.begin(),
                  reorderedTriangles.end(),
                  buildProducts.triangles.begin() + globalTriangleStart);

        // Also fill a global old->new mapping if you want it later
        if (buildProducts.trianglePermutation.empty())
            buildProducts.trianglePermutation.resize(buildProducts.triangles.size());
        for (uint32_t iNew = 0; iNew < localTriangleCount; ++iNew) {
            const uint32_t iOld = blasResult.triPermutation[iNew];
            buildProducts.trianglePermutation[globalTriangleStart + iOld] =
                    globalTriangleStart + iNew;
        }

        // B) patch leaf ranges from local to global triangle indices
        std::vector<BVHNode> patchedNodes = blasResult.nodes;
        for (BVHNode &node: patchedNodes) {
            if (node.isLeaf())
                node.leftFirst += globalTriangleStart; // shift leaf's first triangle
        }

        // C) append nodes and record BLAS range
        const uint32_t firstNode = static_cast<uint32_t>(buildProducts.bottomLevelNodes.size());
        buildProducts.bottomLevelNodes.insert(buildProducts.bottomLevelNodes.end(),
                                              patchedNodes.begin(), patchedNodes.end());
        buildProducts.bottomLevelRanges.push_back({
            firstNode,
            static_cast<uint32_t>(patchedNodes.size())
        });
    }

    static inline float get(const float3 &v, int axis) { return axis == 0 ? v.x() : axis == 1 ? v.y() : v.z(); }

    SceneBuild::TLASResult
    SceneBuild::buildTLAS(const std::vector<InstanceRecord> &instances,
                          const std::vector<BLASRange> &blasRanges,
                          const std::vector<BVHNode> &blasNodes,
                          const std::vector<Transform> &transforms,
                          const BuildOptions &opts) {
        struct Box {
            float3 bmin, bmax;
            uint32_t inst;
        };
        std::vector<Box> boxes;
        boxes.reserve(instances.size());

        // 1) gather world-space AABBs per instance
        for (uint32_t i = 0; i < instances.size(); ++i) {
            const auto &inst = instances[i];
            const auto &xf = transforms[inst.transformIndex];

            const BLASRange br = blasRanges[inst.geometryIndex];
            const BVHNode &root = blasNodes[br.firstNode];

            float3 wmin{FLT_MAX, FLT_MAX, FLT_MAX};
            float3 wmax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (int c = 0; c < 8; ++c) {
                const bool bx = c & 4, by = c & 2, bz = c & 1;
                const float4 pObj{
                    bx ? root.aabbMax.x() : root.aabbMin.x(),
                    by ? root.aabbMax.y() : root.aabbMin.y(),
                    bz ? root.aabbMax.z() : root.aabbMin.z(),
                    0.0f
                };
                const float3 pW = float3(xf.objectToWorld * pObj);
                wmin = min(wmin, pW);
                wmax = max(wmax, pW);
            }
            boxes.push_back({wmin, wmax, i});
        }

        // 2) median-split build
        TLASResult R{};
        R.nodes.clear();
        R.nodes.reserve(std::max<size_t>(1, boxes.size() * 2));

        std::function<uint32_t(uint32_t, uint32_t)> build = [&](uint32_t start, uint32_t end)-> uint32_t {
            const uint32_t n = static_cast<uint32_t>(R.nodes.size());
            R.nodes.emplace_back();
            TLASNode &N = R.nodes.back();

            float3 bmin{FLT_MAX, FLT_MAX, FLT_MAX}, bmax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (uint32_t i = start; i < end; ++i) {
                bmin = min(bmin, boxes[i].bmin);
                bmax = max(bmax, boxes[i].bmax);
            }
            N.aabbMin = bmin;
            N.aabbMax = bmax;

            const uint32_t count = end - start;
            if (count <= std::max(1u, opts.tlasMaxLeafInstances)) {
                N.count = count; // leaf
                N.leftChild = boxes[start].inst; // instance index
                N.rightChild = (count == 2) ? boxes[start + 1].inst : 0;
                return n;
            }

            float3 cmin{FLT_MAX, FLT_MAX, FLT_MAX}, cmax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (uint32_t i = start; i < end; ++i) {
                const float3 c = {
                    (boxes[i].bmin.x() + boxes[i].bmax.x()) * 0.5f,
                    (boxes[i].bmin.y() + boxes[i].bmax.y()) * 0.5f,
                    (boxes[i].bmin.z() + boxes[i].bmax.z()) * 0.5f
                };
                cmin = min(cmin, c);
                cmax = max(cmax, c);
            }
            const float3 ext = {cmax.x() - cmin.x(), cmax.y() - cmin.y(), cmax.z() - cmin.z()};
            const int axis = (ext.x() >= ext.y() && ext.x() >= ext.z()) ? 0 : (ext.y() >= ext.z() ? 1 : 2);
            const float pivot = 0.5f * (get(cmin, axis) + get(cmax, axis));

            auto midIt = std::partition(boxes.begin() + start, boxes.begin() + end,
                                        [&](const Box &b) {
                                            const float3 c = {
                                                (b.bmin.x() + b.bmax.x()) * 0.5f,
                                                (b.bmin.y() + b.bmax.y()) * 0.5f,
                                                (b.bmin.z() + b.bmax.z()) * 0.5f
                                            };
                                            return get(c, axis) < pivot;
                                        });
            uint32_t mid = static_cast<uint32_t>(midIt - boxes.begin());
            if (mid == start || mid == end) mid = start + count / 2;

            N.count = 0; // internal
            N.leftChild = build(start, mid);
            N.rightChild = build(mid, end);
            return n;
        };

        R.rootIndex = boxes.empty() ? UINT32_MAX : build(0, static_cast<uint32_t>(boxes.size()));
        return R;
    }


    void SceneBuild::computePacking(BuildProducts &buildProducts) {
    }
}
