#pragma once

#include "Renderer/GPUDataStructures.h"
#include "Renderer/SurfaceOverlap.h"

namespace Pale {
    inline glm::mat4 surfaceOverlapMatrix(const float4x4& matrix) {
        glm::mat4 result{1};
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) result[c][r] = matrix.row[r][c];
        return result;
    }

    inline SurfaceOverlapSettings surfaceOverlapSettings(const PathTracerSettings& settings) {
        return {rendererDebugLocalLayerDepthEpsilon(settings),
                rendererDebugLocalLayerNormalCosineThreshold(settings),
                settings.rendererDebugLocalLayerDepthMode,
                rendererDebugMaxLocalSurfelHits(settings), RayEpsilon};
    }

    inline SurfaceOverlapView surfaceOverlapView(const CameraGPU& camera) {
        const bool intrinsics = camera.hasPinholeIntrinsics != 0u && camera.fx > 0 && camera.fy > 0;
        const float focal = 0.5f * camera.height / std::tan(0.5f * glm::radians(camera.fovy));
        return {glm::vec3(surfaceOverlapMatrix(camera.invView)[3]),
                surfaceOverlapMatrix(camera.view),
                intrinsics ? camera.fx : focal, intrinsics ? camera.fy : focal,
                intrinsics ? camera.cx : 0.5f * camera.width,
                intrinsics ? camera.cy : 0.5f * camera.height, camera.width, camera.height};
    }

    // Keep placement, footprint radii and instance membership identical in the
    // viewer and training. Result rows are ordered by instance, then point row.
    template<class SceneBuildProducts>
    std::vector<SurfaceFootprint> surfaceOverlapFootprints(
            const SceneBuildProducts& scene, std::vector<std::size_t>& instanceOffsets) {
        std::vector<SurfaceFootprint> footprints;
        instanceOffsets.assign(scene.instances.size(), 0u);
        for (std::size_t instanceIndex = 0; instanceIndex < scene.instances.size(); ++instanceIndex) {
            const auto& instance = scene.instances[instanceIndex];
            if (instance.geometryType != GeometryType::PointCloud) continue;
            instanceOffsets[instanceIndex] = footprints.size();
            const auto& range = scene.pointCloudRanges.at(instance.geometryIndex);
            const auto worldFromObject = surfaceOverlapMatrix(scene.transforms.at(instance.transformIndex).objectToWorld);
            for (std::size_t i = range.firstPoint; i < static_cast<std::size_t>(range.firstPoint) + range.pointCount; ++i) {
                const auto& point = scene.points.at(i);
                const SurfaceFootprint footprint{sycl2glm(point.position),
                    sycl2glm(point.tanU) * point.scale.x(), sycl2glm(point.tanV) * point.scale.y(),
                    !point.isEmissive(), static_cast<std::uint32_t>(instanceIndex)};
                footprints.push_back(footprint.transformed(worldFromObject));
            }
        }
        return footprints;
    }
}
