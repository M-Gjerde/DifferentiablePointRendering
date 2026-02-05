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
    }

    void PathTracer::setScene(const GPUSceneBuffers &scene, SceneBuild::BuildProducts bp) {
        m_sceneGPU = scene;
        uint32_t requiredCapacity = m_settings.photonsPerLaunch;
        ensureRayCapacity(requiredCapacity);

        if (!m_intermediates.map.photons) {
            allocatePhotonMap();

            auto topTLAS = bp.topLevelNodes.front();
            AABB sceneAabb = {topTLAS.aabbMin, topTLAS.aabbMax};

            const float3 sceneMin = sceneAabb.minP;
            const float3 sceneMax = sceneAabb.maxP;
            const float3 sceneExtent = sceneMax - sceneMin;

            Log::PA_INFO(
                "Scene AABB min = ({:.6f}, {:.6f}, {:.6f}), "
                "max = ({:.6f}, {:.6f}, {:.6f}), "
                "extent = ({:.6f}, {:.6f}, {:.6f})",
                sceneMin.x(), sceneMin.y(), sceneMin.z(),
                sceneMax.x(), sceneMax.y(), sceneMax.z(),
                sceneExtent.x(), sceneExtent.y(), sceneExtent.z()
            );

            const float Adiff = bp.diffuseSurfaceArea;
            const float N = static_cast<float>(m_settings.photonsPerLaunch);
            const float k = 10.0f;
            const float r0 = sycl::sqrt((k * Adiff) / (N * M_PIf));

            configurePhotonGrid(sceneAabb);
        }
    }

    // Call this before first render, or inside submitKernel() after computing capacity.
    void PathTracer::ensureRayCapacity(uint32_t requiredRayQueueCapacity) {
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

        // --- zero init ---
        m_queue.memset(m_intermediates.countPrimary, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countExtensionOut, 0, sizeof(uint32_t));
        m_queue.wait();

        // --- totals ---
        std::size_t intermediatesTotalBytes =
                sizePrimaryRaysBytes +
                sizeExtensionRaysBytes +
                sizeHitRecordsBytes +
                sizeof(uint32_t) * 2; // countPrimary + countExtensionOut


        Log::PA_INFO("Total intermediates memory: {}", Utils::formatBytes(intermediatesTotalBytes));
    }

    void PathTracer::allocatePhotonMap() {
        freePhotonMap();

        constexpr std::size_t maxPhotonBytes = 6ull * 1024ull * 1024ull * 1024ull; // 3GB
        std::size_t photonSize = sizeof(DevicePhotonSurface);


        // desired photon count
        std::size_t requestedPhotons = m_rayQueueCapacity * static_cast<uint64_t>(
                                           m_settings.numForwardPasses * m_settings.maxBounces);


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


        m_queue.memset(m_intermediates.map.photonCountDevicePtr, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.map.photons, 0,
                       sizeof(DevicePhotonSurface) * m_intermediates.map.photonCapacity);
        m_queue.wait();

        std::size_t photonMapTotalBytes = sizeof(DevicePhotonSurface) * finalPhotonCount;
        Log::PA_INFO("Total photon map memory: {}", Utils::formatBytes(photonMapTotalBytes));
    }

    void PathTracer::freeIntermediates() {
        if (!m_rayQueueCapacity) return;
        sycl::free(m_intermediates.primaryRays, m_queue);
        sycl::free(m_intermediates.extensionRaysA, m_queue);
        sycl::free(m_intermediates.hitRecords, m_queue);
        sycl::free(m_intermediates.countPrimary, m_queue);
        sycl::free(m_intermediates.countExtensionOut, m_queue);
        m_intermediates.primaryRays = nullptr;
        m_intermediates.extensionRaysA = nullptr;
        m_intermediates.hitRecords = nullptr;
        m_intermediates.countPrimary = nullptr;
        m_intermediates.countExtensionOut = nullptr;
        m_rayQueueCapacity = 0;
    }

    void PathTracer::freePhotonMap() {
        if (!m_intermediates.map.photons) return;
        sycl::free(m_intermediates.map.photons, m_queue);
        sycl::free(m_intermediates.map.photonCountDevicePtr, m_queue);
        m_intermediates.map.photons = nullptr;
        m_intermediates.map.photonCountDevicePtr = nullptr;
    }

    void PathTracer::configurePhotonGrid(const AABB &sceneAabb) {
        auto &grid = m_intermediates.map;

        grid.gatherRadiusWorld = 0.01f;
        const float gatherRadiusWorld = grid.gatherRadiusWorld;
        const float cellSizeWorld = 0.5f * gatherRadiusWorld;

        grid.cellSizeWorld = float3{cellSizeWorld, cellSizeWorld, cellSizeWorld};

        const float3 pad = float3{gatherRadiusWorld, gatherRadiusWorld, gatherRadiusWorld};
        grid.gridOriginWorld = sceneAabb.minP - pad;
        const float3 gridMax = sceneAabb.maxP + pad;

        const float3 extent = gridMax - grid.gridOriginWorld;

        auto cellCountAxis = [](float extentAxis, float cellSize) -> std::int32_t {
            const float safeCellSize = sycl::fmax(cellSize, 1e-6f);
            return static_cast<std::int32_t>(sycl::ceil(extentAxis / safeCellSize));
        };

        grid.gridResolution = sycl::int3{
            cellCountAxis(extent.x(), cellSizeWorld),
            cellCountAxis(extent.y(), cellSizeWorld),
            cellCountAxis(extent.z(), cellSizeWorld)
        };

        const std::uint64_t nx = static_cast<std::uint64_t>(grid.gridResolution.x());
        const std::uint64_t ny = static_cast<std::uint64_t>(grid.gridResolution.y());
        const std::uint64_t nz = static_cast<std::uint64_t>(grid.gridResolution.z());
        const std::uint64_t totalCells64 = nx * ny * nz;

        if (totalCells64 == 0 || totalCells64 > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("Photon grid resolution too high; increase r or tighten AABB.");

        grid.totalCellCount = static_cast<std::uint32_t>(totalCells64);

        ensurePhotonGridBuffersAllocatedAndInitialized(grid);
    }

    void PathTracer::ensurePhotonGridBuffersAllocatedAndInitialized(DeviceSurfacePhotonMapGrid &grid) {

        auto allocateU32 = [&](std::uint32_t *&devicePtr, std::size_t elementCount, const char *name) {
            devicePtr = sycl::malloc_device<std::uint32_t>(elementCount, m_queue);
            if (!devicePtr) throw std::runtime_error(std::string("Failed to allocate ") + name);
        };

        auto freeU32 = [&](std::uint32_t *&devicePtr) {
            if (devicePtr) {
                sycl::free(devicePtr, m_queue);
                devicePtr = nullptr;
            }
        };

        const std::uint32_t requiredCellCount = grid.totalCellCount;
        const std::uint32_t requiredPhotonCapacity = grid.photonCapacity;

        // Choose a scan block size (power of two)
        static constexpr std::uint32_t kScanBlockSize = 1024;
        const std::uint32_t requiredBlockCount =
                (requiredCellCount + kScanBlockSize - 1u) / kScanBlockSize;

        const bool needReallocCells = (grid.allocatedCellCount != requiredCellCount);
        const bool needReallocPhotons = (grid.allocatedPhotonCapacity != requiredPhotonCapacity);
        const bool needReallocBlocks = (grid.allocatedBlockCount != requiredBlockCount);

        // Reallocate per-cell buffers if cellCount changes
        if (needReallocCells) {
            freeU32(grid.cellStart);
            freeU32(grid.cellEnd);
            freeU32(grid.cellCount);
            freeU32(grid.cellWriteOffset);

            allocateU32(grid.cellStart, requiredCellCount, "cellStart");
            allocateU32(grid.cellEnd, requiredCellCount, "cellEnd");
            allocateU32(grid.cellCount, requiredCellCount, "cellCount");
            allocateU32(grid.cellWriteOffset, requiredCellCount, "cellWriteOffset");

            grid.allocatedCellCount = requiredCellCount;
        }

        // Reallocate per-photon buffers if capacity changes
        if (needReallocPhotons) {
            freeU32(grid.photonCellId);
            freeU32(grid.photonIndex);
            freeU32(grid.sortedPhotonIndex);

            allocateU32(grid.photonCellId, requiredPhotonCapacity, "photonCellId");
            allocateU32(grid.photonIndex, requiredPhotonCapacity, "photonIndex");
            allocateU32(grid.sortedPhotonIndex, requiredPhotonCapacity, "sortedPhotonIndex");

            grid.allocatedPhotonCapacity = requiredPhotonCapacity;
        }

        // Reallocate scan temporaries if block count changes
        if (needReallocBlocks) {
            freeU32(grid.blockSums);
            freeU32(grid.blockPrefix);

            allocateU32(grid.blockSums, requiredBlockCount, "blockSums");
            allocateU32(grid.blockPrefix, requiredBlockCount, "blockPrefix");

            grid.allocatedBlockCount = requiredBlockCount;
        }


    }


    void PathTracer::renderForward(std::vector<SensorGPU> &sensor) {
        ScopedTimer forwardTimer("Rendering time", spdlog::level::debug);
        m_settings.rayGenMode = RayGenMode::Emitter;

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_sceneGPU,
            .intermediates = m_intermediates,
            .gradients = {},
            .sensor = sensor,
            .debugImages = nullptr,
            .numSensors = static_cast<uint32_t>(sensor.size())
        };

        Log::PA_INFO("Rendering {} point(s)", renderPackage.scene.pointCount);

        switch (m_settings.integratorKind) {
            case IntegratorKind::lightTracing:
                submitLightTracingKernel(renderPackage);
                break;
            case IntegratorKind::lightTracingCylinderRay:
                //submitLightTracingKernelCylinderRay(renderPackage);
                break;
            case IntegratorKind::photonMapping:
                submitPhotonMappingKernel(renderPackage);
                break;
        }

        m_queue.wait();
    }

    void PathTracer::renderBackward(std::vector<SensorGPU> &sensors, PointGradients &gradients,
                                    DebugImages *debugImages) {
        for (const auto &sensor: sensors) {
            const uint32_t requiredRayCapacity = sensor.width * sensor.height;
            if (requiredRayCapacity > m_rayQueueCapacity) {
                Log::PA_INFO("RayQueue Capacity too small for per pixel adjoint pass. Resizing queue capacity..");
                ensureRayCapacity(requiredRayCapacity);
            }
        }


        m_settings.rayGenMode = RayGenMode::Adjoint;

        ScopedTimer adjointTimer("Adjoint pass total", spdlog::level::debug);

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_sceneGPU,
            .intermediates = m_intermediates,
            .gradients = gradients,
            .sensor = sensors,
            .debugImages = debugImages,
            .numSensors = static_cast<uint32_t>(sensors.size()),
        };

        submitAdjointKernel(renderPackage);

        m_queue.wait();
    }

    void PathTracer::reset() {
    }
}
