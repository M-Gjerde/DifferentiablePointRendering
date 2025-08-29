//
// Created by magnus on 8/29/25.
//

module;

#include <memory>
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>

module Pale.Render.SceneGPU;

import Pale.Scene.Components;
import Pale.Render.GPUDataTypes;

namespace Pale {
    // ==== Geometry collector ====
    void SceneBuild::collectGeometry(const std::shared_ptr<Scene> &scene,
                                     IAssetAccess &assetAccess,
                                     BuildProducts &outBuildProducts) {
        std::vector<Vertex> vertices;
        std::vector<Triangle> triangles;

        auto view = scene->getAllEntitiesWith<MeshComponent>();

        for (auto [entID, meshComponent]: view.each()) {
            auto meshAsset = assetAccess.getMesh(meshComponent.meshID);

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
            outBuildProducts.meshRanges.push_back(meshRange);
            outBuildProducts.meshIndexById[meshComponent.meshID] =
                static_cast<uint32_t>(outBuildProducts.meshRanges.size() - 1);
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

    SceneBuild::BLASResult SceneBuild::buildMeshBLAS(uint32_t, const MeshRange &, const BuildOptions &) {
        return {};
    }

    SceneBuild::TLASResult SceneBuild::buildTLAS(const std::vector<InstanceRecord> &, const std::vector<BLASRange> &,
                                                 const BuildOptions &) {
        return {};
    }

    void SceneBuild::appendBLAS(BuildProducts &buildProducts, const BLASResult &blasResult) {
    }

    void SceneBuild::finalizePermutation(BuildProducts &buildProducts) {
    }

    void SceneBuild::computePacking(BuildProducts &buildProducts) {
    }
}
