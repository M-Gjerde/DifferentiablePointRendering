//
// Created by magnus on 8/29/25.
//

module;

#include <cmath>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"
#include "Renderer/GPUDataStructures.h"
#include "Renderer/Kernels/KernelHelpers.h"


module Pale.Render.SceneBuild;
import Pale.Scene.Components;
import Pale.Render.BVH;
import Pale.Log;

namespace Pale {
    // ==== Geometry collector ====
    void SceneBuild::collectGeometry(const std::shared_ptr<Scene>& scene,
                                     IAssetAccess& assetAccess,
                                     BuildProducts& outBuildProducts) {
        std::vector<Vertex> vertices;
        std::vector<Triangle> triangles;
        // 1) Gather unique mesh IDs in scene order
        std::vector<UUID> uniqueMeshIds;
        {
            std::unordered_set<UUID> seen;
            for (auto [e, mesh] : scene->getAllEntitiesWith<MeshComponent>().each()) {
                if (seen.insert(mesh.meshID).second) uniqueMeshIds.push_back(mesh.meshID);
            }
        }
        // 2) For each unique mesh, append geometry once and record ALL ranges
        for (const UUID& meshId : uniqueMeshIds) {
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

    void SceneBuild::collectInstances(const std::shared_ptr<Pale::Scene>& scene,
                                      IAssetAccess& assetAccess,
                                      const std::unordered_map<UUID, uint32_t>& meshIndexById,
                                      BuildProducts& outBuildProducts) {
        std::unordered_map<UUID, uint32_t> materialIndexByUuid;
        auto view = scene->getAllEntitiesWith<MeshComponent, MaterialComponent, TransformComponent, TagComponent>();
        for (auto [entityId, meshComponent, materialComponent, transformComponent, tagComponent] : view.each()) {
            auto it = meshIndexById.find(meshComponent.meshID);
            if (it == meshIndexById.end()) continue;
            const uint32_t geometryIndex = it->second;
            // material de-dup
            uint32_t materialIndex;
            if (auto mit = materialIndexByUuid.find(materialComponent.materialID); mit != materialIndexByUuid.end()) {
                materialIndex = mit->second;
            }
            else {
                const auto materialAsset = assetAccess.getMaterial(materialComponent.materialID);
                GPUMaterial gpuMaterial{};
                gpuMaterial.baseColor = materialAsset->baseColor;
                gpuMaterial.specular = materialAsset->metallic;
                gpuMaterial.diffuse = materialAsset->roughness;
                gpuMaterial.emissive = materialAsset->emissive;
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

            InstanceRecord record{
                .geometryType = GeometryType::Mesh,
                .geometryIndex = geometryIndex,
                .materialIndex = materialIndex,
                .transformIndex = transformIndex,
            };
            copyName(record.name, tagComponent.tag);


            outBuildProducts.instances.push_back(record);

            // Print entity name and index
            uint32_t instanceIndex = static_cast<uint32_t>(outBuildProducts.instances.size() - 1);
            Log::PA_INFO("Instance [{}]: {}", instanceIndex, tagComponent.tag);
        }
    }

    void SceneBuild::collectPointCloudGeometry(const std::shared_ptr<Scene>& scene,
                                               IAssetAccess& assetAccess,
                                               BuildProducts& outBuildProducts) {
        std::vector<Point> collectedPoints;
        std::vector<UUID> uniquePointCloudIds;
        {    
            // de-dup assets
            std::unordered_set<UUID> seen;
            for (auto [e, pc] : scene->getAllEntitiesWith<PointCloudComponent>().each()) {
                if (seen.insert(pc.pointCloudID).second)
                    uniquePointCloudIds.push_back(pc.pointCloudID);
            }
        }

        for (const UUID& pointCloudID : uniquePointCloudIds) {
            const auto pointCloudAsset = assetAccess.getPointCloud(pointCloudID);
            // assuming one geometry block per asset for now
            const PointGeometry& pointGeometry = pointCloudAsset->points.front();

            const uint32_t firstPointIndex = static_cast<uint32_t>(collectedPoints.size());
            collectedPoints.reserve(firstPointIndex + static_cast<uint32_t>(pointGeometry.positions.size()));

            for (size_t i = 0; i < pointGeometry.positions.size(); ++i) {
                Point gpuPoint{};
                gpuPoint.position = pointGeometry.positions[i];
                gpuPoint.tanU = normalize(pointGeometry.tanU[i]);
                gpuPoint.tanV = normalize(pointGeometry.tanV[i]); // assume orthonormal input
                gpuPoint.scale = {pointGeometry.scales[i].x, pointGeometry.scales[i].y};
                gpuPoint.color = pointGeometry.colors[i];
                gpuPoint.opacity = pointGeometry.opacities[i];
                gpuPoint.shape = pointGeometry.shapes[i];
                gpuPoint.beta = pointGeometry.betas[i];

                // Print entity name and index
                uint32_t instanceIndex = i;
                Log::PA_INFO("Point Instance color: [{}]: {}", instanceIndex, glm::to_string(pointGeometry.colors[i]));

                collectedPoints.push_back(gpuPoint);
            }

            const uint32_t pointCount = static_cast<uint32_t>(pointGeometry.positions.size());
            const uint32_t rangeIndex = static_cast<uint32_t>(outBuildProducts.pointCloudRanges.size());
            outBuildProducts.pointCloudRanges.push_back(PointCloudRange{firstPointIndex, pointCount});
            outBuildProducts.pointCloudIndexById.emplace(pointCloudID, rangeIndex);
        }

        outBuildProducts.points = std::move(collectedPoints);
    }

    void SceneBuild::collectPointCloudInstances(const std::shared_ptr<Scene>& scene,
                                                BuildProducts& outBuildProducts) {
        auto view = scene->getAllEntitiesWith<PointCloudComponent, TransformComponent, TagComponent>();
        for (auto [entityId, pointCloudComponent, transformComponent, tagComponent] : view.each()) {
            auto it = outBuildProducts.pointCloudIndexById.find(pointCloudComponent.pointCloudID);
            if (it == outBuildProducts.pointCloudIndexById.end()) continue;

            Transform gpuTransform{};
            const glm::mat4 objectToWorldGLM = transformComponent.getTransform();
            gpuTransform.objectToWorld = glm2sycl(objectToWorldGLM);
            gpuTransform.worldToObject = glm2sycl(glm::inverse(objectToWorldGLM));
            const uint32_t transformIndex = static_cast<uint32_t>(outBuildProducts.transforms.size());
            outBuildProducts.transforms.push_back(gpuTransform);

            InstanceRecord record{
                .geometryType = GeometryType::PointCloud,
                .geometryIndex = it->second,
                .materialIndex = kInvalidMaterialIndex,
                .transformIndex = transformIndex,
            };
            copyName(record.name, tagComponent.tag);
            outBuildProducts.instances.push_back(record);
        }
    }

    inline float SceneBuild::triangleArea(const float3& p0,
                                          const float3& p1,
                                          const float3& p2) {
        const float3 e0 = p1 - p0;
        const float3 e1 = p2 - p0;
        return 0.5f * length(cross(e0, e1));
    }


    void SceneBuild::collectLights(const std::shared_ptr<Pale::Scene>& scene,
                                   IAssetAccess& assetAccess,
                                   BuildProducts& out) {
        out.lights.clear();
        out.emissiveTriangles.clear();

        for (const InstanceRecord& instanceRecord : out.instances) {
            if (instanceRecord.geometryType == GeometryType::PointCloud) continue;
            const GPUMaterial& gpuMaterial = out.materials[instanceRecord.materialIndex];
            const bool isEmissive = (gpuMaterial.emissive.x() > 0.f) ||
                (gpuMaterial.emissive.y() > 0.f) ||
                (gpuMaterial.emissive.z() > 0.f);
            if (!isEmissive) continue;

            const MeshRange& meshRange = out.meshRanges[instanceRecord.geometryIndex];
            const uint32_t triangleOffset = static_cast<uint32_t>(out.emissiveTriangles.size());

            float totalArea = 0.f;
            out.emissiveTriangles.reserve(out.emissiveTriangles.size() + meshRange.triCount);

            for (uint32_t localTri = 0; localTri < meshRange.triCount; ++localTri) {
                const uint32_t globalTriIndex = meshRange.firstTri + localTri;
                const Triangle& tri = out.triangles[globalTriIndex];

                const float3 p0 = out.vertices[tri.v0].pos;
                const float3 p1 = out.vertices[tri.v1].pos;
                const float3 p2 = out.vertices[tri.v2].pos;

                const float area = triangleArea(p0, p1, p2);
                totalArea += area;

                out.emissiveTriangles.push_back(GPUEmissiveTriangle{globalTriIndex});
            }

            GPULightRecord light{};
            light.lightType = 0u; // mesh area
            light.geometryIndex = instanceRecord.geometryIndex;
            light.transformIndex = instanceRecord.transformIndex;
            light.triangleOffset = triangleOffset;
            light.triangleCount = meshRange.triCount;
            light.emissionRgb = gpuMaterial.emissive;
            light.totalArea = totalArea;

            out.lights.push_back(light);
        }
    }

    void SceneBuild::collectCameras(const std::shared_ptr<Scene>& scene,
                                    BuildProducts& outBuildProducts) {
        auto view = scene->getAllEntitiesWith<CameraComponent, TransformComponent>();
        for (auto [entityId, cameraComponent, transformComponent] : view.each()) {
            CameraGPU gpuCam{};
            const glm::mat4 world = transformComponent.getTransform();
            const glm::mat4 viewMat = glm::inverse(world);
            const glm::mat4 projMat = cameraComponent.camera.getProjectionMatrix();
            gpuCam.view = glm2sycl(viewMat);
            gpuCam.proj = glm2sycl(projMat);
            gpuCam.invView = glm2sycl(world);
            gpuCam.invProj = glm2sycl(glm::inverse(projMat));
            const glm::vec3 pos = transformComponent.getPosition();
            gpuCam.pos = float3{pos.x, pos.y, pos.z};
            glm::vec3 forward = glm::mat3(world) * glm::vec3(0.0f, 0.0f, -1.0f);
            forward = glm::normalize(forward);
            gpuCam.forward = float3{forward.x, forward.y, forward.z};
            gpuCam.width = cameraComponent.camera.width;
            gpuCam.height = cameraComponent.camera.height;
            outBuildProducts.cameraGPUs.push_back(gpuCam);
        }
    }



    inline AABB surfelObjectAabb(const Point& surfel,
                                  float kStdDevs = 2.8f, // Should be similar to the same kSigmas as in intersect surfels
                                  float sigmaNormal = 0.0f) // set >0 model thickness
    {
        const float3 tangentU = normalize(surfel.tanU);
        const float3 tangentV = normalize(surfel.tanV);
        const float3 normalObject = normalize(cross(tangentU, tangentV));

        const float suK = kStdDevs * std::fmax(surfel.scale.x(), 1e-8f);
        const float svK = kStdDevs * std::fmax(surfel.scale.y(), 1e-8f);
        const float snK = kStdDevs * std::fmax(sigmaNormal, 0.0f);

        auto axisExtent = [&](int axis)->float {
            const float tu = axis==0? tangentU.x(): axis==1? tangentU.y(): tangentU.z();
            const float tv = axis==0? tangentV.x(): axis==1? tangentV.y(): tangentV.z();
            const float nn = axis==0? std::fabs(normalObject.x()): axis==1? std::fabs(normalObject.y()): std::fabs(normalObject.z());
            const float projInPlane = sycl::sqrt((suK*tu)*(suK*tu) + (svK*tv)*(svK*tv));
            return projInPlane + snK * nn; // add normal thickness if used
        };

        const float3 halfExtent{ axisExtent(0), axisExtent(1), axisExtent(2) };
        return { surfel.position - halfExtent, surfel.position + halfExtent };
    }

    // ---- BLAS build: localize -> build -> return nodes + permutation ----
    SceneBuild::BLASResult SceneBuild::buildPointCloudBLAS(uint32_t pointCloudIndex,
                                               const PointCloudRange& pointCloudRange,
                                               const std::vector<Point>& allPoints,
                                               const BuildOptions& buildOptions) {
        // 1) Localize points
        std::vector<Point> localPoints;
        localPoints.reserve(pointCloudRange.pointCount);
        for (uint32_t i = 0; i < pointCloudRange.pointCount; ++i)
            localPoints.push_back(allPoints[pointCloudRange.firstPoint + i]);

        // 2) Build AABBs and centroids
        std::vector<AABB>   localAabbs(localPoints.size());
        std::vector<float3> localCentroids(localPoints.size());
        for (uint32_t i = 0; i < localPoints.size(); ++i) {
            const AABB aabb = surfelObjectAabb(localPoints[i]);
            localAabbs[i]   = aabb;
            localCentroids[i]= (aabb.minP + aabb.maxP) * 0.5f;
        }

        // 3) Build BVH over boxes
        std::vector<BVHNode> localNodes;
        std::vector<uint32_t> localPointOrder; // permutation: new_order[i] = old_local_index
        BasicBVH::buildFromBoxes(localAabbs,
                                 localCentroids,
                                 localNodes,
                                 localPointOrder,
                                 buildOptions.bvhMaxLeafPoints);

        // 4) Package
        BLASResult result{};
        result.localPoints        = std::move(localPoints);
        result.pointPermutation   = std::move(localPointOrder);
        result.nodes              = std::move(localNodes);
        result.range              = {0u, 0u}; // filled when appended
        result.pointCloudIndex    = pointCloudIndex;
        return result;
    }

    // ---- BLAS build: localize -> build -> return nodes + permutation ----
    SceneBuild::BLASResult
    SceneBuild::buildMeshBLAS(uint32_t meshIndex,
                              const MeshRange& meshRange,
                              const std::vector<Triangle>& allTriangles,
                              const std::vector<Vertex>& allVertices,
                              const BuildOptions& buildOptions) {
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
        result.localTriangles = std::move(localTriangles);
        result.nodes = std::move(localNodes);
        result.range = {0u, 0u};
        result.triPermutation = std::move(localTriangleOrder);
        result.meshIndex = meshIndex;
        return result;
    }



    SceneBuild::TLASResult
    SceneBuild::buildTLAS(const std::vector<InstanceRecord>& instances,
                          const std::vector<BLASRange>& blasRanges,
                          const std::vector<BVHNode>& blasNodes,
                          const std::vector<Transform>& transforms,
                          const BuildOptions& opts) {
        struct Box {
            float3 bmin, bmax;
            uint32_t inst;
        };
        std::vector<Box> boxes;
        boxes.reserve(instances.size());
        // 1) gather world-space AABBs per instance
        for (uint32_t i = 0; i < instances.size(); ++i) {
            const auto& inst = instances[i];
            const auto& xf = transforms[inst.transformIndex];
            uint32_t blasRangeIndex = inst.blasRangeIndex;
            const BLASRange br = blasRanges[blasRangeIndex];
            const BVHNode& root = blasNodes[br.firstNode];
            float3 wmin{FLT_MAX, FLT_MAX, FLT_MAX};
            float3 wmax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (int c = 0; c < 8; ++c) {
                const bool bx = c & 4, by = c & 2, bz = c & 1;
                const float3 pObj{
                    bx ? root.aabbMax.x() : root.aabbMin.x(),
                    by ? root.aabbMax.y() : root.aabbMin.y(),
                    bz ? root.aabbMax.z() : root.aabbMin.z()
                };
                float3 pW = toWorldPoint(pObj, xf);
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
            TLASNode& N = R.nodes.back();
            float3 bmin{FLT_MAX, FLT_MAX, FLT_MAX}, bmax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (uint32_t i = start; i < end; ++i) {
                bmin = min(bmin, boxes[i].bmin);
                bmax = max(bmax, boxes[i].bmax);
            }
            N.aabbMin = bmin;
            N.aabbMax = bmax;
            const uint32_t count = end - start;
            if (count == 1) {
                N.count = 1; // leaf
                N.leftChild = boxes[start].inst; // points to Instance index
                N.rightChild = 0;
            }
            else {
                N.count = 0; // internal

                float3 cmin{FLT_MAX, FLT_MAX, FLT_MAX}, cmax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
                for (uint32_t i = start; i < end; ++i) {
                    float3 cent = (boxes[i].bmin + boxes[i].bmax) * 0.5f;

                    cmin = min(cmin, cent);
                    cmax = max(cmax, cent);
                }
                float3 ext = cmax - cmin;
                int axis = (ext.x() > ext.y() && ext.x() > ext.z()) ? 0 : (ext.y() > ext.z()) ? 1 : 2;
                float pivot = (cmin[axis] + cmax[axis]) * 0.5f;

                auto midIter = std::partition(boxes.begin() + start, boxes.begin() + end,
                                              [&](const Box& b) {
                                                  float3 cc = (b.bmin + b.bmax) * 0.5f;
                                                  return cc[axis] < pivot;
                                              });
                int mid = int(midIter - boxes.begin());
                if (mid == start || mid == end) mid = start + count / 2;

                N.leftChild = build(start, mid);
                N.rightChild = build(mid, end);
            }
            return n;
        };
        R.rootIndex = boxes.empty() ? UINT32_MAX : build(0, static_cast<uint32_t>(boxes.size()));
        return R;
    }


    void SceneBuild::computePacking(BuildProducts& buildProducts) {
    }

    float SceneBuild::computeDiffuseSurfaceAreaWorld(const BuildProducts &buildProducts) {
        float totalDiffuseArea = 0.0f;

        for (const InstanceRecord& instanceRecord : buildProducts.instances) {
            if (instanceRecord.geometryType != GeometryType::Mesh)
                continue;

            const GPUMaterial& material = buildProducts.materials[instanceRecord.materialIndex];
            if (material.emissive != float3(0.0f))
                continue;

            const MeshRange& meshRange = buildProducts.meshRanges[instanceRecord.geometryIndex];
            const Transform& xf        = buildProducts.transforms[instanceRecord.transformIndex];

            for (uint32_t localTri = 0; localTri < meshRange.triCount; ++localTri) {
                const uint32_t triIndex = meshRange.firstTri + localTri;
                const Triangle& tri     = buildProducts.triangles[triIndex];

                const float3 p0W = toWorldPoint(buildProducts.vertices[tri.v0].pos, xf);
                const float3 p1W = toWorldPoint(buildProducts.vertices[tri.v1].pos, xf);
                const float3 p2W = toWorldPoint(buildProducts.vertices[tri.v2].pos, xf);

                totalDiffuseArea += triangleArea(p0W, p1W, p2W);
            }
        }
        return totalDiffuseArea;
    }
}
