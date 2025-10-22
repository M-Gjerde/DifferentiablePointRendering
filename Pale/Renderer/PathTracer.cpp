//
// Created by magnus-desktop on 8/28/25.
//
module;

#include <sycl/sycl.hpp>
#include <filesystem>
#include "Kernels/SyclBridge.h"
#include "Core/ScopedTimer.h"

module Pale.Render.PathTracer;

import Pale.Utils.ImageIO;
import Pale.Utils.StringFormatting;
import Pale.Log;
import Pale.Render.BVH;

namespace Pale {
    PathTracer::PathTracer(sycl::queue q, const PathTracerSettings &settings) : m_queue(q), m_settings(settings) {
#ifdef NDEBUG
        // Release
        m_settings.photonsPerLaunch = 1e7; // 1e7
        m_settings.maxBounces = 4;
        m_settings.maxAdjointBounces = 4;
        m_settings.adjointSamplesPerPixel = 4;
        m_settings.samplesPerRay = 4;

#else
        //  cuda/rocm
        m_settings.photonsPerLaunch = 1e6; // 1e6
        m_settings.maxBounces = 4;
        m_settings.numForwardPasses = 8;
        m_settings.numGatherPasses = 8;
        m_settings.maxAdjointBounces = 1;
        m_settings.adjointSamplesPerPixel = 1;
        // omp
        //m_settings.photonsPerLaunch = 1e6; // 1e6
        //m_settings.maxBounces = 4;
        //m_settings.numForwardPasses = 1;
        //m_settings.maxAdjointBounces = 2;
        //m_settings.adjointSamplesPerPixel = 1;
#endif
    }

    // Call this before first render, or inside submitKernel() after computing capacity.
    void PathTracer::ensureCapacity(uint32_t requiredRayQueueCapacity) {
        if (requiredRayQueueCapacity <= m_rayQueueCapacity) return;
        // grow to next power of two to avoid frequent reallocations
        uint32_t newCapacity = 1u;
        while (newCapacity < requiredRayQueueCapacity)
            newCapacity <<= 1u;

        Log::PA_INFO("Required RayQueueCapacity {}M, Allocated {}M", std::round(requiredRayQueueCapacity / 1e6),
                     std::round(newCapacity / 1e6));
        allocateIntermediates(newCapacity);
    }

    void PathTracer::allocateIntermediates(uint32_t newCapacity) {
        freeIntermediates();
        m_rayQueueCapacity = newCapacity;

        // --- primary buffers ---
        std::size_t sizePrimaryRaysBytes = sizeof(RayState) * m_rayQueueCapacity;
        m_intermediates.primaryRays = sycl::malloc_device<RayState>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated primaryRays: {}", Utils::formatBytes(sizePrimaryRaysBytes));

        std::size_t sizeExtensionRaysBytes = sizeof(RayState) * m_rayQueueCapacity;
        m_intermediates.extensionRaysA = sycl::malloc_device<RayState>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated extensionRaysA: {}", Utils::formatBytes(sizeExtensionRaysBytes));

        std::size_t sizeHitRecordsBytes = sizeof(WorldHit) * m_rayQueueCapacity;
        m_intermediates.hitRecords = sycl::malloc_device<WorldHit>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated hitRecords: {}", Utils::formatBytes(sizeHitRecordsBytes));

        m_intermediates.countPrimary = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.countExtensionOut = sycl::malloc_device<uint32_t>(1, m_queue);

        constexpr std::size_t maxPhotonBytes = 2ull * 1024ull * 1024ull * 1024ull; // 2GB
        std::size_t photonSize = sizeof(DevicePhotonSurface);

        // desired photon count
        std::size_t requestedPhotons = m_rayQueueCapacity * static_cast<uint64_t>(
                                           m_settings.maxBounces * m_settings.numForwardPasses);

        // clamp to what fits
        std::size_t maxPhotons = maxPhotonBytes / photonSize;
        std::size_t finalPhotonCount = std::min(requestedPhotons, maxPhotons);

        m_intermediates.map.photons = sycl::malloc_device<DevicePhotonSurface>(
            finalPhotonCount, m_queue);
        m_intermediates.map.photonCountDevicePtr = sycl::malloc_device<uint32_t>(1, m_queue);

        m_intermediates.map.photonCapacity = static_cast<uint32_t>(finalPhotonCount);

        Log::PA_INFO("Photon map size: {}M photons (~{})",
                     finalPhotonCount / 1e6,
                     Utils::formatBytes(finalPhotonCount * photonSize));


        // --- zero init ---
        m_queue.memset(m_intermediates.countPrimary, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countExtensionOut, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.map.photonCountDevicePtr, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.map.photons, 0, sizeof(DevicePhotonSurface) * finalPhotonCount);
        m_queue.wait();

        // --- totals ---
        std::size_t intermediatesTotalBytes =
                sizePrimaryRaysBytes +
                sizeExtensionRaysBytes +
                sizeHitRecordsBytes +
                sizeof(uint32_t) * 2; // countPrimary + countExtensionOut

        std::size_t photonMapTotalBytes = sizeof(DevicePhotonSurface) * finalPhotonCount;

        Log::PA_INFO("Total intermediates memory: {}", Utils::formatBytes(intermediatesTotalBytes));
        Log::PA_INFO("Total photon map memory: {}", Utils::formatBytes(photonMapTotalBytes));
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

    void PathTracer::configurePhotonGrid(const AABB &sceneAabb, float gatherRadiusWorld) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        auto &grid = m_intermediates.map;
        grid.gatherRadiusWorld = gatherRadiusWorld;
        grid.cellSizeWorld = float3{gatherRadiusWorld, gatherRadiusWorld, gatherRadiusWorld};

        const float3 pad = float3{gatherRadiusWorld, gatherRadiusWorld, gatherRadiusWorld};
        grid.gridOriginWorld = sceneAabb.minP - pad;
        const float3 gridMax = sceneAabb.maxP + pad;

        const float3 extent = gridMax - grid.gridOriginWorld;
        const auto cells = [&](float e, float s) { return static_cast<int>(sycl::ceil(e / sycl::fmax(s, 1e-6f))); };
        grid.gridResolution = sycl::int3{
            cells(extent.x(), grid.cellSizeWorld.x()),
            cells(extent.y(), grid.cellSizeWorld.y()),
            cells(extent.z(), grid.cellSizeWorld.z())
        };

        const std::uint64_t nx = static_cast<std::uint64_t>(grid.gridResolution.x());
        const std::uint64_t ny = static_cast<std::uint64_t>(grid.gridResolution.y());
        const std::uint64_t nz = static_cast<std::uint64_t>(grid.gridResolution.z());
        const std::uint64_t totalCells = nx * ny * nz;
        if (totalCells == 0 || totalCells > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("Photon grid too large; increase r or tighten AABB.");

        grid.totalCellCount = static_cast<std::uint32_t>(totalCells);

        std::size_t sizeCellHeadIndexArrayBytes = sizeof(std::uint32_t) * grid.totalCellCount;
        grid.cellHeadIndexArray = sycl::malloc_device<std::uint32_t>(grid.totalCellCount, m_queue);
        Log::PA_TRACE("Allocated cellHeadIndexArray: {}", Utils::formatBytes(sizeCellHeadIndexArrayBytes));

        std::size_t sizePhotonNextIndexArrayBytes = sizeof(std::uint32_t) * grid.photonCapacity;
        grid.photonNextIndexArray = sycl::malloc_device<std::uint32_t>(grid.photonCapacity, m_queue);
        Log::PA_TRACE("Allocated photonNextIndexArray: {}", Utils::formatBytes(sizePhotonNextIndexArrayBytes));

        std::vector<uint32_t> zeroInitHost(grid.totalCellCount, 0);
        m_queue.memcpy(grid.cellHeadIndexArray, zeroInitHost.data(), sizeCellHeadIndexArrayBytes).wait();

        // Optional: report grid-side total as part of photon map footprint
        std::size_t photonGridArraysTotalBytes =
                sizeCellHeadIndexArrayBytes + sizePhotonNextIndexArrayBytes;
        Log::PA_INFO("Photon grid arrays total: {}", Utils::formatBytes(photonGridArraysTotalBytes));
        Log::PA_INFO("Photon grid radius: {}", gatherRadiusWorld);
    }

    void PathTracer::setScene(const GPUSceneBuffers &scene, SceneBuild::BuildProducts bp) {
        m_scene = scene;
        uint32_t requiredCapacity = m_settings.photonsPerLaunch;

        ensureCapacity(requiredCapacity);

        auto topTLAS = bp.topLevelNodes.front();
        AABB sceneAabb{
            topTLAS.aabbMin,
            topTLAS.aabbMax
        };
        const float Adiff = bp.diffuseSurfaceArea;
        const float N = static_cast<float>(requiredCapacity);

        const float k = 20.0f;
        const float r0 = sycl::sqrt((k * Adiff) / (N * M_PIf));
        configurePhotonGrid(sceneAabb, r0);
    }

    void PathTracer::renderForward(SensorGPU &sensor, SensorGPU &sensor2) {
        ScopedTimer forwardTimer("Forward pass total", spdlog::level::debug);
        m_settings.rayGenMode = RayGenMode::Emitter;

        if (sensor.width * sensor.height > m_rayQueueCapacity) {
            Log::PA_ERROR("Not enough rays. Required capacity is 2*width*height");
            exit(1); // TODO graceful exit or something else
        }

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_scene,
            .intermediates = m_intermediates,
            .sensor = sensor,
            .photonMapSensor = sensor2
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

        if (requiredRayCapacity > m_rayQueueCapacity) {
            Log::PA_ERROR("RayQueue Capacity not sufficient. Try reducing image size");
            return;
        }
        m_settings.rayGenMode = RayGenMode::Adjoint;

        ScopedTimer adjointTimer("Adjoint pass total", spdlog::level::debug);

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
