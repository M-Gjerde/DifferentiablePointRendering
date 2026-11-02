#pragma once

#include "IntersectionKernels.h"
#include <stdexcept>

namespace Pale {
    inline void requireCameraCoverageSupport(const PathTracerSettings &settings) {
        if (settings.sharedHeightEnabled || settings.integratorKind != IntegratorKind::photonMapping ||
            settings.cameraGatherKernelKind != CameraGatherKernelKind::CameraGatherKernel2)
            throw std::invalid_argument("Alpha adjoints require photon mapping with CameraGatherKernel2 (without shared height)");
    }

    struct CameraCoverageResult {
        bool interacted = false;
        bool complete = false;
        bool opaqueMesh = false;
        float transmission = 1.0f;
    };

    // Replay CameraGatherKernel2's exact slab membership and batching. The
    // callback receives every geometric member, including zero-opacity members.
    // No lighting, shadow, or secondary path is part of camera coverage.
    template<class Visit>
    inline CameraCoverageResult traceCameraCoverage(
        Ray ray, const GPUSceneBuffers &scene, const PathTracerSettings &settings, Visit visit) {
        CameraCoverageResult result;
        constexpr float alphaEpsilon = 1.0e-8f;
        const auto maxEvents = rendererDebugMaxSplatEventsPerRay(settings);
        const auto batchSize = rendererDebugPointHitBatchSize(settings);
        const auto apply = [&](const PointCloudLocalLayer &layer, uint32_t instanceIndex) {
            result.interacted |= layer.hitCount != 0u;
            visit(layer, instanceIndex);
            result.transmission *= layer.transmission;
        };
        uint32_t singleInstance = kInvalidIndex;
        if (batchSize > 1u && tryGetSinglePointCloudInstance(scene, singleInstance)) {
            const auto capacity = rendererDebugPointHitBatchLookaheadCapacity(settings);
            for (uint32_t event = 0u; event < maxEvents;) {
                LocalSurfelLayerHit hits[kMaxPointHitBatchWithLookahead];
                uint32_t instanceIndex = kInvalidIndex;
                const auto count = collectScenePointHitsDirect(ray, scene, RayEpsilon,
                    std::numeric_limits<float>::infinity(), hits, capacity, instanceIndex);
                if (count == 0u) { result.complete = true; return result; }
                const auto coreCount = sycl::min(count, batchSize);
                uint32_t cursor = 0u;
                float consumed = 0.0f;
                while (cursor < coreCount && event < maxEvents) {
                    const auto previous = cursor;
                    const auto layer = buildPointCloudLocalLayerFromHits(ray, hits[cursor],
                        hits + cursor, count - cursor, scene,
                        rendererDebugLocalLayerDepthEpsilon(settings), rendererDebugMaxLocalSurfelHits(settings),
                        rendererDebugLocalLayerNormalCosineThreshold(settings), settings.rendererDebugLocalLayerDepthMode);
                    apply(layer, instanceIndex);
                    ++event;
                    consumed = sycl::fmax(consumed, layer.furthestT);
                    while (cursor < count && hits[cursor].tWorld <= layer.furthestT + RayEpsilon) ++cursor;
                    if (cursor == previous) ++cursor;
                    if (result.transmission <= alphaEpsilon) { result.complete = true; return result; }
                }
                if (consumed <= 0.0f) return result;
                advanceRayPast(ray, consumed, RayEpsilon);
                if (cursor >= count && count < capacity) { result.complete = true; return result; }
            }
        } else {
            for (uint32_t event = 0u; event < maxEvents; ++event) {
                WorldHit hit{};
                intersectScene(ray, &hit, scene, SurfelIntersectMode::FirstHit);
                if (!hit.hit) { result.complete = true; return result; }
                const auto &instance = scene.instances[hit.instanceIndex];
                if (instance.geometryType == GeometryType::Mesh) {
                    result.opaqueMesh = result.complete = true;
                    result.transmission = 0.0f;
                    return result;
                }
                if (instance.geometryType != GeometryType::PointCloud) return result;
                buildIntersectionNormal(scene, hit);
                const auto layer = collectPointCloudLocalLayer(ray, hit, instance, scene,
                    rendererDebugLocalLayerDepthEpsilon(settings), rendererDebugMaxLocalSurfelHits(settings),
                    rendererDebugLocalLayerNormalCosineThreshold(settings), settings.rendererDebugLocalLayerDepthMode);
                if (layer.hitCount == 0u) return result;
                apply(layer, hit.instanceIndex);
                advanceRayPast(ray, layer.furthestT, RayEpsilon);
                if (result.transmission <= alphaEpsilon) { result.complete = true; return result; }
            }
        }
        // Reaching the iteration limit is not evidence of scene escape. A final
        // query distinguishes a genuinely exhausted ray from an incomplete one.
        WorldHit remaining{};
        intersectScene(ray, &remaining, scene, SurfelIntersectMode::FirstHit);
        result.complete = !remaining.hit;
        return result;
    }
}
