//
// Created by magnus on 8/29/25.
//

module;

#include <memory>
#include <vector>
#include <unordered_map>

module Pale.Render.SceneGPU;

import Pale.Scene.Components;
import Pale.Render.GPUDataTypes;

namespace Pale {
    // ==== Geometry collector ====
    SceneBuild::GeometryInput
    SceneBuild::collectGeometry(const std::shared_ptr<Pale::Scene> &scene,
                                IAssetAccess &assetAccess) {
        GeometryInput geometryInput;

        std::vector<Vertex> vertices;
        std::vector<Triangle> triangles;
        std::vector<MeshRange> meshRanges;
        std::vector<OrientedPoint> orientedPoints;
        std::unordered_map<std::string, uint32_t> meshIndexById;
        std::vector<UUID> meshNames;
        /*
        auto view = scene->getRegistry().view<MeshComponent>();
        for (auto entId: view) {
            Entity entity(entId, scene.get());

            if (!entity.isVisible()) continue;

            auto &meshComponent = entity.getComponent<MeshComponent>();
            auto meshData = MeshManager::instance().getMeshData(meshComponent);
            if (!meshData) continue;

            const std::string meshCacheId = meshComponent.getCacheIdentifier();
            if (meshIndexById.count(meshCacheId)) continue;


            const uint32_t vertexBaseIndex = static_cast<uint32_t>(vertices.size());
            const uint32_t triangleBaseIndex = static_cast<uint32_t>(triangles.size());

            for (const auto &v: meshData->m_vertices) {
                Vertex gpuVertex{};
                gpuVertex.pos = float3{v.pos.x, v.pos.y, v.pos.z};
                gpuVertex.norm = float3{v.normal.x, v.normal.y, v.normal.z};
                vertices.push_back(gpuVertex);
            }

            constexpr float oneThird = 1.0f / 3.0f;
            for (size_t i = 0; i < meshData->m_indices.size(); i += 3) {
                const uint32_t i0 = meshData->m_indices[i + 0] + vertexBaseIndex;
                const uint32_t i1 = meshData->m_indices[i + 1] + vertexBaseIndex;
                const uint32_t i2 = meshData->m_indices[i + 2] + vertexBaseIndex;

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
            meshRange.vertCount = static_cast<uint32_t>(meshData->m_vertices.size());
            meshRange.firstTri = triangleBaseIndex;
            meshRange.triCount = static_cast<uint32_t>(meshData->m_indices.size() / 3);
            meshRanges.push_back(meshRange);
            meshNames.push_back(meshCacheId);
            meshIndexById[meshCacheId] = static_cast<uint32_t>(meshRanges.size() - 1);

        }
        */
        // pack into GeometryInput (extend it with the fields you need)
        geometryInput.meshes.reserve(meshRanges.size());
        for (size_t i = 0; i < meshRanges.size(); ++i) {
            geometryInput.meshes.push_back(MeshInput{
                .meshId = meshNames[i],
                .meshRange = meshRanges[i],
            });
        }

        // Store into global BuildProducts later (vertices, triangles, ranges, points)
        // Return whatever your pipeline expects; keep the arrays around in your builder state.
        return geometryInput;
    }

    // ==== Instance collector ====
    SceneBuild::InstanceInput
    SceneBuild::collectInstances(const std::shared_ptr<Pale::Scene> &scene,
                                 IAssetAccess &assetAccess) {
        InstanceInput instanceInput;

        std::vector<InstanceRecord> instanceRecords;
        std::vector<Transform> transforms;
        std::vector<GPUMaterial> gpuMaterials;
        std::vector<std::string> materialNames;

        /*
    // You must have a meshId → index map produced in collectGeometry().
    // Expose it from GeometryInput or keep it as builder state.
    const auto& meshIndexById = /* your map from collectGeometry */
        ;

        /*
    auto view = scene->getRegistry().view<MeshComponent, MaterialComponent, TransformComponent>(
        entt::exclude<RasterizerRenderingComponent, LightSourceComponent, CameraComponent, QuadricCollectionComponent>);

    for (auto entId : view) {
        Entity entity(entId, scene.get());
        if (!entity.isVisible()) continue;

        auto& meshComponent = entity.getComponent<MeshComponent>();
        if (!MeshManager::instance().getMeshData(meshComponent)) continue;

        const bool isPointType =
            meshComponent.meshDataType() == QUADRIC ||
            meshComponent.meshDataType() == GAUSSIAN_2D;

        const std::string meshCacheId = meshComponent.getCacheIdentifier();
        if (!meshIndexById.contains(meshCacheId)) continue;

        const uint32_t geometryIndex = meshIndexById.at(meshCacheId);

        auto& materialComponent = entity.getComponent<MaterialComponent>();
        GPUMaterial gpuMaterial{};
        gpuMaterial.baseColor = materialComponent.albedo.x;
        gpuMaterial.specular  = materialComponent.specular;
        gpuMaterial.diffuse   = materialComponent.diffuse;
        gpuMaterial.phongExp  = materialComponent.phongExponent;

        const uint32_t materialIndex = static_cast<uint32_t>(gpuMaterials.size());
        gpuMaterials.push_back(gpuMaterial);
        materialNames.push_back(entity.getName());

        auto& transformComponent = entity.getComponent<TransformComponent>();
        Transform objectTransform{};
        objectTransform.objectToWorld = glm2sycl(transformComponent.getTransform());
        objectTransform.worldToObject = glm2sycl(glm::inverse(transformComponent.getTransform()));
        const uint32_t transformIndex = static_cast<uint32_t>(transforms.size());
        transforms.push_back(objectTransform);

        instanceRecords.push_back(InstanceRecord{
            .geometryType   = isPointType ? GeometryType::PointCloud : GeometryType::Mesh,
            .geometryIndex  = geometryIndex,
            .materialIndex  = materialIndex,
            .transformIndex = transformIndex,
            .name           = entity.getName(),
        });
    }

    // Fill InstanceInput
    instanceInput.instances = std::move(instanceRecords);

    // Stash transforms and materials into your BuildProducts later:
    // buildProducts.transforms   = std::move(transforms);
    // buildProducts.materialBlocks = std::move(gpuMaterials);

*/
        return instanceInput;
    }

    SceneBuild::BLASResult SceneBuild::buildMeshBLAS(const MeshInput &, const BuildOptions &) {

        return {};

    }

    SceneBuild::TLASResult SceneBuild::buildTLAS(const InstanceInput &, const std::vector<BLASRange> &,
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
