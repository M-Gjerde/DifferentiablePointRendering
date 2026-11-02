#pragma once

#include "Renderer/Kernels/IntersectionKernels.h"

namespace viewer {
    // Unit-weight squared attenuation deficit. Untouched rays are excluded.
    inline float terminalTransmittanceLoss(float transmission, float targetAttenuation) {
        if (transmission >= 1.0f) return 0.0f;
        const float deficit = sycl::fmax(0.0f, transmission - (1.0f - targetAttenuation));
        return deficit * deficit;
    }

    // Camera-only deterministic slab compositing, matching the viewer's surface
    // footprints. Negative values mean traversal was incomplete, not scene escape.
    inline float traceTerminalTransmittanceLoss(
        Pale::Ray ray, const Pale::GPUSceneBuffers& scene,
        const Pale::PathTracerSettings& settings, float targetAttenuation) {
        float transmission = 1.0f;
        const auto maxEvents = Pale::rendererDebugMaxSplatEventsPerRay(settings);
        for (uint32_t event = 0; ; ++event) {
            Pale::WorldHit hit{};
            Pale::intersectScene(ray, &hit, scene, Pale::SurfelIntersectMode::FirstHit);
            if (!hit.hit) return terminalTransmittanceLoss(transmission, targetAttenuation);
            const auto& instance = scene.instances[hit.instanceIndex];
            // Meshes terminate the camera ray opaquely in the renderer.
            if (instance.geometryType == Pale::GeometryType::Mesh) return 0.0f;
            if (event >= maxEvents) return -1.0f;
            const auto layer = Pale::collectPointCloudLocalLayer(
                ray, hit, instance, scene,
                Pale::rendererDebugLocalLayerDepthEpsilon(settings),
                Pale::rendererDebugMaxLocalSurfelHits(settings),
                Pale::rendererDebugLocalLayerNormalCosineThreshold(settings),
                settings.rendererDebugLocalLayerDepthMode);
            if (layer.hitCount == 0u) return -1.0f;
            transmission *= sycl::clamp(layer.transmission, 0.0f, 1.0f);
            // Further attenuation cannot increase this loss.
            if (transmission <= 1.0f - targetAttenuation) return 0.0f;
            Pale::advanceRayPast(ray, layer.furthestT, Pale::RayEpsilon);
        }
    }
}
