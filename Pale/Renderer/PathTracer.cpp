//
// Created by magnus-desktop on 8/28/25.
//
module;

#include <sycl/sycl.hpp>
#include <filesystem>
#include "Kernels/SyclBridge.h"

module Pale.Render.PathTracer;

import Pale.Utils.ImageIO;
import Pale.Log;
import Pale.Render.BVH;

namespace Pale {
    PathTracer::PathTracer(sycl::queue q, const PathTracerSettings &settings) : m_queue(q), m_settings(settings) {
#ifdef NDEBUG
        // Release
        m_settings.photonsPerLaunch = 2e7; // 1e7
        m_settings.maxBounces = 8;

#else
        // Debug
        m_settings.photonsPerLaunch = 1e6; // 1e6
        m_settings.maxBounces = 3;
#endif
    }

    // Call this before first render, or inside submitKernel() after computing capacity.
    void PathTracer::ensureCapacity(uint32_t requiredRayQueueCapacity) {
        if (requiredRayQueueCapacity <= m_rayQueueCapacity) return;
        // grow to next power of two to avoid frequent reallocations
        uint32_t newCapacity = 1u;
        while (newCapacity < requiredRayQueueCapacity) newCapacity <<= 1u;
        allocateIntermediates(newCapacity);
    }

    void PathTracer::allocateIntermediates(uint32_t newCapacity) {
        freeIntermediates();
        m_rayQueueCapacity = newCapacity;

        m_intermediates.primaryRays = sycl::malloc_device<RayState>(m_rayQueueCapacity, m_queue);
        m_intermediates.extensionRaysA = sycl::malloc_device<RayState>(m_rayQueueCapacity, m_queue);
        m_intermediates.hitRecords = sycl::malloc_device<WorldHit>(m_rayQueueCapacity, m_queue);
        m_intermediates.countPrimary = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.countExtensionOut = sycl::malloc_device<uint32_t>(1, m_queue);

        m_intermediates.map.photons = sycl::malloc_device<DevicePhotonSurface>(
            m_rayQueueCapacity * m_settings.maxBounces, m_queue);
        m_intermediates.map.photonCountDevicePtr = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.map.photonCapacity = m_rayQueueCapacity * m_settings.maxBounces;

        // optional: zero counters once
        m_queue.memset(m_intermediates.countPrimary, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countExtensionOut, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.map.photonCountDevicePtr, 0, sizeof(uint32_t));

        m_queue.fill(m_intermediates.map.photons, DevicePhotonSurface(), m_rayQueueCapacity * m_settings.maxBounces);
    }

    void PathTracer::freeIntermediates() {
        if (!m_rayQueueCapacity) return;
        sycl::free(m_intermediates.primaryRays, m_queue);
        sycl::free(m_intermediates.map.photons, m_queue);
        sycl::free(m_intermediates.map.photonCountDevicePtr, m_queue);
        sycl::free(m_intermediates.extensionRaysA, m_queue);
        sycl::free(m_intermediates.hitRecords, m_queue);
        sycl::free(m_intermediates.countPrimary, m_queue);
        sycl::free(m_intermediates.countExtensionOut, m_queue);
        m_intermediates = {};
        m_rayQueueCapacity = 0;
    }

    void PathTracer::configurePhotonGrid(const AABB& sceneAabb, float gatherRadiusWorld)
    {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        auto& grid = m_intermediates.map;
        grid.gatherRadiusWorld = gatherRadiusWorld;
        grid.cellSizeWorld     = float3{gatherRadiusWorld, gatherRadiusWorld, gatherRadiusWorld};

        const float3 pad = float3{gatherRadiusWorld, gatherRadiusWorld, gatherRadiusWorld};
        grid.gridOriginWorld   = sceneAabb.minP - pad;
        const float3 gridMax   = sceneAabb.maxP + pad;

        const float3 extent    = gridMax - grid.gridOriginWorld;
        const auto cells = [&](float e, float s){ return static_cast<int>(sycl::ceil(e / sycl::fmax(s, 1e-6f))); };
        grid.gridResolution = sycl::int3{ cells(extent.x(), grid.cellSizeWorld.x()),
                                          cells(extent.y(), grid.cellSizeWorld.y()),
                                          cells(extent.z(), grid.cellSizeWorld.z()) };

        const std::uint64_t nx = static_cast<std::uint64_t>(grid.gridResolution.x());
        const std::uint64_t ny = static_cast<std::uint64_t>(grid.gridResolution.y());
        const std::uint64_t nz = static_cast<std::uint64_t>(grid.gridResolution.z());
        const std::uint64_t totalCells = nx*ny*nz;
        if (totalCells == 0 || totalCells > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("Photon grid too large; increase r or tighten AABB.");

        grid.totalCellCount = static_cast<std::uint32_t>(totalCells);

        grid.cellHeadIndexArray   = sycl::malloc_device<std::uint32_t>(grid.totalCellCount, m_queue);
        grid.photonNextIndexArray = sycl::malloc_device<std::uint32_t>(grid.photonCapacity, m_queue);

        m_queue.fill(grid.cellHeadIndexArray, kInvalidIndex, grid.totalCellCount).wait();
    }


    void PathTracer::setScene(const GPUSceneBuffers &scene, SceneBuild::BuildProducts bp) {
        m_scene = scene;
        ensureCapacity(m_settings.photonsPerLaunch);

        auto topTLAS = bp.topLevelNodes.front();
        AABB sceneAabb{.minP = topTLAS.aabbMin, topTLAS.aabbMax};
        float diag = length(topTLAS.aabbMax - topTLAS.aabbMin);
        const float Adiff = bp.diffuseSurfaceArea;
        const float N     = float(m_settings.photonsPerLaunch);

#ifdef NDEBUG
        const float k     = 20.0f;
#else
        const float k     = 5.0f;
#endif

        const float r0    = sycl::sqrt((k * Adiff) / (N * float(M_PI)));
        configurePhotonGrid(sceneAabb, r0);

    }

    void PathTracer::renderForward(SensorGPU &sensor) {
        m_settings.rayGenMode = RayGenMode::Emitter;

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_scene,
            .intermediates = m_intermediates,
            .sensor = sensor
        };

        submitKernel(renderPackage);

        m_queue.wait();
    }

    void PathTracer::renderBackward(SensorGPU &sensor, AdjointGPU &adjoint) {
        if (!adjoint.framebuffer) {
            Log::PA_WARN("Adjoint image is not set but renderBackward is called");
            return;
        }

        const uint32_t requiredRayCapacity = sensor.width * sensor.height;
        ensureCapacity(requiredRayCapacity);

        m_settings.rayGenMode = RayGenMode::Adjoint;

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_scene,
            .intermediates = m_intermediates,
            .sensor = sensor,
            .adjoint = adjoint
        };

        submitKernel(renderPackage);

        m_queue.wait();
    }

    void PathTracer::reset() {
    }
}
