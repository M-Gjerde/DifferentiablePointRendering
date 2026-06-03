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
    PathTracer::PathTracer(sycl::queue q, const PathTracerSettings& settings) : m_queue(q), m_settings(settings),
        m_sessionSeed(settings.random.seed) {
    }

    void PathTracer::setScene(const GPUSceneBuffers& scene, SceneBuild::BuildProducts bp) {
        m_sceneGPU = scene;

        const uint32_t requiredCapacity = m_settings.photonsPerLaunch;
        ensureRayCapacity(requiredCapacity);

        if (m_settings.integratorKind == IntegratorKind::photonMapping) {
            freePhotonMap();
            freePhotonGridBuffers();

            allocatePhotonMap();

            const auto& topTLAS = bp.topLevelNodes.front();
            const AABB sceneAabb{topTLAS.aabbMin, topTLAS.aabbMax};

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

            const float diffuse_surface_area = bp.diffuseSurfaceArea;
            const float photon_count = static_cast<float>(m_settings.photonsPerLaunch);

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
        m_intermediates.maxRayQueueCapacity = m_rayQueueCapacity;
        // --- primary buffers ---
        const std::size_t sizePrimaryRaysBytes =
            sizeof(RayState) * m_rayQueueCapacity;
        m_intermediates.primaryRays =
            sycl::malloc_device<RayState>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated primaryRays: {}", Utils::formatBytes(sizePrimaryRaysBytes));

        const std::size_t sizeExtensionRaysBytes =
            sizeof(RayState) * m_rayQueueCapacity;
        m_intermediates.extensionRaysA =
            sycl::malloc_device<RayState>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated extensionRaysA: {}", Utils::formatBytes(sizeExtensionRaysBytes));

        const std::size_t sizeHitRecordsBytes =
            sizeof(WorldHit) * m_rayQueueCapacity;
        m_intermediates.hitRecords =
            sycl::malloc_device<WorldHit>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated hitRecords: {}", Utils::formatBytes(sizeHitRecordsBytes));

        const std::size_t sizeContributionRecordsBytes =
            sizeof(HitInfoContribution) * m_rayQueueCapacity;
        m_intermediates.hitContribution =
            sycl::malloc_device<HitInfoContribution>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated hitContribution records: {}", Utils::formatBytes(sizeContributionRecordsBytes));
        m_intermediates.maxHitContributionCount = m_rayQueueCapacity;

        // --- compact adjoint event buffers ---
        const std::size_t sizeMeasurementEventsBytes =
            sizeof(MeasurementGradientEvent) * m_rayQueueCapacity;
        m_intermediates.measurementEvents =
            sycl::malloc_device<MeasurementGradientEvent>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated sizeMeasurementEventsBytes: {}", Utils::formatBytes(sizeMeasurementEventsBytes));

        const std::size_t sizeMeasurementEventsTwoPointBytes =
            sizeof(MeasurementGradientEventXY) * m_rayQueueCapacity * m_settings.numAdjointPathShadowRays * 2;
        m_intermediates.measurementTwoPointEvents =
            sycl::malloc_device<MeasurementGradientEventXY>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated sizeMeasurementEventsTwoPointBytes: {}",
                      Utils::formatBytes(sizeMeasurementEventsTwoPointBytes));
        m_intermediates.maxMeasurementTwoPointEventCount = m_rayQueueCapacity * m_settings.numAdjointPathShadowRays * 2;

        const std::size_t cameraAttachedBridgeEventSize =
            sizeof(MaterialVertexGradientEvent) * m_rayQueueCapacity;
        m_intermediates.materialVertexEvents =
            sycl::malloc_device<MaterialVertexGradientEvent>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated projectionScatterEvents: {}", Utils::formatBytes(cameraAttachedBridgeEventSize));
        m_intermediates.maxMaterialVertexEventCount = m_rayQueueCapacity;

        const std::size_t materialEndEdgeEventsSize =
            sizeof(MaterialEdgeGradientEvent) * m_rayQueueCapacity;
        m_intermediates.materialEndEdgeEvents =
            sycl::malloc_device<MaterialEdgeGradientEvent>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated reflectScatterEvents: {}", Utils::formatBytes(materialEndEdgeEventsSize));
        m_intermediates.maxMaterialEndEdgeEventCount = m_rayQueueCapacity;

        const std::size_t materialStartEdgeEventsSize =
            sizeof(MaterialEdgeGradientEvent) * m_rayQueueCapacity;
        m_intermediates.materialStartEdgeEvents =
            sycl::malloc_device<MaterialEdgeGradientEvent>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated reflectScatterEvents: {}", Utils::formatBytes(materialStartEdgeEventsSize));
        m_intermediates.maxMaterialStartEdgeEventCount = m_rayQueueCapacity;

        // --- pending adjoint states ---
        const std::size_t sizePendingAdjointStatesXBytes =
            sizeof(PendingAdjointStageX) * m_rayQueueCapacity;
        m_intermediates.pendingStageX =
            sycl::malloc_device<PendingAdjointStageX>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated pendingStageX: {}", Utils::formatBytes(sizePendingAdjointStatesXBytes));

        m_intermediates.maxPendingAdjointStateCount = m_rayQueueCapacity;

        const std::size_t sizePendingCameraSegmentBytes =
            sizeof(PendingCameraSegment) * m_rayQueueCapacity;
        m_intermediates.pendingCameraSegments =
            sycl::malloc_device<PendingCameraSegment>(m_rayQueueCapacity, m_queue);
        Log::PA_TRACE("Allocated pendingStageXY: {}", Utils::formatBytes(sizePendingCameraSegmentBytes));
        m_intermediates.maxMeasurementEventCount = m_rayQueueCapacity;;

        const uint32_t gradientRecordCapacity = m_rayQueueCapacity * m_settings.numAdjointPathShadowRays * 20; // TODO depends on number of sensors
        const std::size_t sizeGradientRecordsBytes =
            sizeof(SurfelGradientRecord) * gradientRecordCapacity;
        m_intermediates.gradientRecords =
            sycl::malloc_device<SurfelGradientRecord>(gradientRecordCapacity, m_queue);
        m_intermediates.maxGradientRecordCount = gradientRecordCapacity;
        Log::PA_INFO("Allocated gradientRecords: QueueCapacity: {}, adjointSPP {}, adjointShadowSPP {}, {}",
                     m_rayQueueCapacity, m_settings.adjointSamplesPerPixel, m_settings.numAdjointPathShadowRays,
                     Utils::formatBytes(sizeGradientRecordsBytes));

        Log::PA_INFO("sizeof(SurfelGradientRecord) = {}, Count: {}", sizeof(SurfelGradientRecord), gradientRecordCapacity);

        // --- counters ---
        m_intermediates.countPrimary = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.countContributions = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.countMaterialVertexEvents = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.countExtensionOut = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.countMeasurementEvents = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.countMeasurementTwoPointEvents = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.countMaterialEndEdgeEvents = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.countMaterialStartEdgeEvents = sycl::malloc_device<uint32_t>(1, m_queue);

        m_queue.memset(m_intermediates.countPrimary, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countContributions, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countMaterialVertexEvents, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countExtensionOut, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countMeasurementEvents, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countMeasurementTwoPointEvents, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countMaterialEndEdgeEvents, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countMaterialStartEdgeEvents, 0, sizeof(uint32_t));
        m_queue.wait();

        const std::size_t counterBytes =
            sizeof(uint32_t) * 6;

        const std::size_t intermediatesTotalBytes =
            sizePrimaryRaysBytes +
            sizeExtensionRaysBytes +
            sizeHitRecordsBytes +
            sizeContributionRecordsBytes +
            sizeMeasurementEventsBytes +
            sizeMeasurementEventsTwoPointBytes +
            cameraAttachedBridgeEventSize +
            materialStartEdgeEventsSize +
            materialEndEdgeEventsSize +
            sizePendingAdjointStatesXBytes +
            sizeGradientRecordsBytes +
            counterBytes;

        Log::PA_INFO("Total intermediates memory: {}", Utils::formatBytes(intermediatesTotalBytes));
    }

    void PathTracer::allocatePhotonMap() {
        freePhotonMap();
        constexpr std::size_t maxPhotonBytes = 8ull * 1024ull * 1024ull * 1024ull; // 10GB
        std::size_t photonSize = sizeof(DevicePhotonSurface);
        // desired photon count
        std::size_t requestedPhotons = m_settings.photonsPerLaunch * static_cast<uint64_t>(
            m_settings.numForwardPasses * m_settings.maxBounces);
        // clamp to what fits
        std::size_t maxPhotons = maxPhotonBytes / photonSize;
        std::size_t finalPhotonCount = std::min(requestedPhotons, maxPhotons);
        m_intermediates.map.photons = sycl::malloc_device<DevicePhotonSurface>(
            finalPhotonCount, m_queue);
        m_intermediates.map.photonCountDevicePtr = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.map.photonCapacity = static_cast<uint32_t>(finalPhotonCount);

        Log::PA_INFO(
            "Photon map max size: {}M photons (~{}). Launching {}M photons should require storage for {}M photons",
            maxPhotons / 1e6f,
            Utils::formatBytes(finalPhotonCount * photonSize),
            m_settings.numForwardPasses * m_settings.photonsPerLaunch / 1e6f,
            m_settings.numForwardPasses * m_settings.photonsPerLaunch * m_settings.maxBounces / 1e6f);

        Log::PA_INFO("Used Storage Capacity: {}%",
                     (m_settings.numForwardPasses * m_settings.photonsPerLaunch * m_settings.maxBounces / static_cast<
                         float>(maxPhotons)) * 100.0f);

        m_queue.memset(m_intermediates.map.photonCountDevicePtr, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.map.photons, 0,
                       sizeof(DevicePhotonSurface) * m_intermediates.map.photonCapacity);
        m_queue.wait();

        std::size_t photonMapTotalBytes = sizeof(DevicePhotonSurface) * finalPhotonCount;
        Log::PA_INFO("Total photon map memory: {}", Utils::formatBytes(photonMapTotalBytes));
    }

    template <typename T>
    static void freeDevicePtr(T*& devicePointer, sycl::queue& queue) {
        if (devicePointer) {
            sycl::free(devicePointer, queue);
            devicePointer = nullptr;
        }
    }

    void PathTracer::freeIntermediates() {
        if (!m_rayQueueCapacity) {
            return;
        }

        freeDevicePtr(m_intermediates.primaryRays, m_queue);
        freeDevicePtr(m_intermediates.extensionRaysA, m_queue);
        freeDevicePtr(m_intermediates.hitRecords, m_queue);
        freeDevicePtr(m_intermediates.hitContribution, m_queue);

        freeDevicePtr(m_intermediates.measurementEvents, m_queue);
        freeDevicePtr(m_intermediates.measurementTwoPointEvents, m_queue);
        freeDevicePtr(m_intermediates.materialVertexEvents, m_queue);
        freeDevicePtr(m_intermediates.materialEndEdgeEvents, m_queue);
        freeDevicePtr(m_intermediates.materialStartEdgeEvents, m_queue);
        freeDevicePtr(m_intermediates.pendingCameraSegments, m_queue);
        freeDevicePtr(m_intermediates.countMeasurementEvents, m_queue);
        freeDevicePtr(m_intermediates.countMeasurementTwoPointEvents, m_queue);


        freeDevicePtr(m_intermediates.pendingStageX, m_queue);

        freeDevicePtr(m_intermediates.countPrimary, m_queue);
        freeDevicePtr(m_intermediates.countContributions, m_queue);
        freeDevicePtr(m_intermediates.countMaterialVertexEvents, m_queue);
        freeDevicePtr(m_intermediates.countMaterialEndEdgeEvents, m_queue);
        freeDevicePtr(m_intermediates.countMaterialStartEdgeEvents, m_queue);
        freeDevicePtr(m_intermediates.countExtensionOut, m_queue);
        freeDevicePtr(m_intermediates.gradientRecords, m_queue);

        m_intermediates.primaryRays = nullptr;
        m_intermediates.extensionRaysA = nullptr;
        m_intermediates.hitRecords = nullptr;
        m_intermediates.hitContribution = nullptr;

        m_intermediates.measurementEvents = nullptr;
        m_intermediates.measurementTwoPointEvents = nullptr;
        m_intermediates.materialVertexEvents = nullptr;
        m_intermediates.materialEndEdgeEvents = nullptr;
        m_intermediates.materialStartEdgeEvents = nullptr;

        m_intermediates.pendingStageX = nullptr;

        m_intermediates.countPrimary = nullptr;
        m_intermediates.countContributions = nullptr;
        m_intermediates.countMaterialVertexEvents = nullptr;
        m_intermediates.countMaterialEndEdgeEvents = nullptr;
        m_intermediates.countMaterialStartEdgeEvents = nullptr;
        m_intermediates.countExtensionOut = nullptr;
        m_intermediates.pendingCameraSegments = nullptr;
        m_intermediates.countMeasurementEvents = nullptr;
        m_intermediates.countMeasurementTwoPointEvents = nullptr;
        m_intermediates.maxMaterialVertexEventCount = 0;
        m_intermediates.maxHitContributionCount = 0;
        m_intermediates.maxMeasurementEventCount = 0;
        m_intermediates.maxPendingAdjointStateCount = 0;
        m_intermediates.gradientRecords = nullptr;
        m_intermediates.maxGradientRecordCount = 0;
        m_intermediates.maxMeasurementTwoPointEventCount = 0;
        m_intermediates.maxRayQueueCapacity = 0;
        m_intermediates.maxMaterialStartEdgeEventCount = 0;
        m_intermediates.maxMaterialEndEdgeEventCount = 0;

        m_rayQueueCapacity = 0;
    }

    void PathTracer::freePhotonMap() {
        freeDevicePtr(m_intermediates.map.photons, m_queue);
        freeDevicePtr(m_intermediates.map.photonCountDevicePtr, m_queue);

        m_intermediates.map.photons = nullptr;
        m_intermediates.map.photonCountDevicePtr = nullptr;
        m_intermediates.map.photonCapacity = 0;
    }

    void PathTracer::freePhotonGridBuffers() {
        auto& grid = m_intermediates.map;

        freeDevicePtr(grid.cellStart, m_queue);
        freeDevicePtr(grid.cellEnd, m_queue);
        freeDevicePtr(grid.cellCount, m_queue);
        freeDevicePtr(grid.cellWriteOffset, m_queue);

        freeDevicePtr(grid.photonCellId, m_queue);
        freeDevicePtr(grid.photonIndex, m_queue);
        freeDevicePtr(grid.sortedPhotonIndex, m_queue);

        freeDevicePtr(grid.blockSums, m_queue);
        freeDevicePtr(grid.blockPrefix, m_queue);

        grid.allocatedCellCount = 0;
        grid.allocatedPhotonCapacity = 0;
        grid.allocatedBlockCount = 0;
        grid.totalCellCount = 0;
    }

    void PathTracer::configurePhotonGrid(const AABB& sceneAabb) {
        auto& grid = m_intermediates.map;

        grid.minimumGatherRadiusWorld = 0.01f;
        grid.maximumGatherRadiusWorld = 0.04f;
        grid.gatherPadWorld = 0.04f;
        const float cellSizeWorld = 0.01f;

        grid.cellSizeWorld = float3{cellSizeWorld, cellSizeWorld, cellSizeWorld};

        const float3 pad = float3{grid.gatherPadWorld};
        grid.gridOriginWorld = (sceneAabb.minP - pad);
        const float3 gridMax = (sceneAabb.maxP + pad);

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


        if (totalCells64 == 0 || totalCells64 > std::numeric_limits<std::uint32_t>::max()) {
            std::ostringstream errorStream;
            errorStream
                << "Photon grid resolution too high; increase cell size or tighten AABB.\n"
                << "sceneAabb.minP = (" << sceneAabb.minP.x() << ", " << sceneAabb.minP.y() << ", " << sceneAabb.minP.
                z() << ")\n"
                << "sceneAabb.maxP = (" << sceneAabb.maxP.x() << ", " << sceneAabb.maxP.y() << ", " << sceneAabb.maxP.
                z() << ")\n"
                << "gridOriginWorld = (" << grid.gridOriginWorld.x() << ", " << grid.gridOriginWorld.y() << ", " << grid
                .gridOriginWorld.z() << ")\n"
                << "gridMax = (" << gridMax.x() << ", " << gridMax.y() << ", " << gridMax.z() << ")\n"
                << "extent = (" << extent.x() << ", " << extent.y() << ", " << extent.z() << ")\n"
                << "cellSizeWorld = " << cellSizeWorld << "\n"
                << "gridResolution = (" << grid.gridResolution.x() << ", " << grid.gridResolution.y() << ", " << grid.
                gridResolution.z() << ")\n"
                << "totalCells64 = " << totalCells64;

            throw std::runtime_error(errorStream.str());
        }
        grid.totalCellCount = static_cast<std::uint32_t>(totalCells64);

        ensurePhotonGridBuffersAllocatedAndInitialized(grid);
    }

    void PathTracer::ensurePhotonGridBuffersAllocatedAndInitialized(DeviceSurfacePhotonMapGrid& grid) {
        auto allocateU32 = [&](std::uint32_t*& devicePtr, std::size_t elementCount, const char* name) {
            devicePtr = sycl::malloc_device<std::uint32_t>(elementCount, m_queue);
            if (!devicePtr) throw std::runtime_error(std::string("Failed to allocate ") + name);
        };

        auto freeU32 = [&](std::uint32_t*& devicePtr) {
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


    void PathTracer::renderForward(std::vector<SensorGPU>& sensor) {
        ScopedTimer forwardTimer("Rendering time", spdlog::level::debug);
        m_settings.rayGenMode = RayGenMode::Emitter;


        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .random.seed = m_sessionSeed,
            .scene = m_sceneGPU,
            .intermediates = m_intermediates,
            .gradients = {},
            .sensors = sensor,
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

    void PathTracer::renderBackward(std::vector<SensorGPU>& sensors, PointGradients& gradients,
                                    DebugImages* debugImages) {
        for (const auto& sensor : sensors) {
            const uint32_t requiredRayCapacity = sensor.width * sensor.height;
            if (requiredRayCapacity > m_rayQueueCapacity) {
                Log::PA_INFO("RayQueue Capacity too small for per pixel adjoint pass. Resizing queue capacity..");
                ensureRayCapacity(requiredRayCapacity);
            }
        }

        m_settings.rayGenMode = RayGenMode::Adjoint;
        Log::PA_DEBUG("Submitting Adjoint rendering pass");

        ScopedTimer adjointTimer("Adjoint pass total", spdlog::level::debug);

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_sceneGPU,
            .intermediates = m_intermediates,
            .gradients = gradients,
            .sensors = sensors,
            .debugImages = debugImages,
            .numSensors = static_cast<uint32_t>(sensors.size()),
        };
        submitAdjointKernel(renderPackage);
        m_queue.wait();
    }

    void PathTracer::renderDepthDistortionBackward(std::vector<SensorGPU>& sensors, PointGradients& gradients) {
        m_settings.rayGenMode = RayGenMode::Adjoint;
        Log::PA_DEBUG("Submitting Distortion loss kernel");

        ScopedTimer adjointTimer("Distortion pass total", spdlog::level::debug);

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_sceneGPU,
            .intermediates = m_intermediates,
            .gradients = gradients,
            .sensors = sensors,
            .numSensors = static_cast<uint32_t>(sensors.size()),
        };

        submitDepthDistortionKernel(renderPackage);

        m_queue.wait();
    }

    void PathTracer::renderNormalConsistencyBackward(std::vector<SensorGPU>& sensors, PointGradients& gradients) {
        m_settings.rayGenMode = RayGenMode::Adjoint;
        Log::PA_DEBUG("Submitting Distortion loss kernel");

        ScopedTimer adjointTimer("Distortion pass total", spdlog::level::debug);

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_sceneGPU,
            .intermediates = m_intermediates,
            .gradients = gradients,
            .sensors = sensors,
            .numSensors = static_cast<uint32_t>(sensors.size()),
        };

        submitNormalConsistencyKernel(renderPackage);

        m_queue.wait();
    }

    void PathTracer::renderSurfaceRegularizersBackward(
        std::vector<SensorGPU> &sensors,
        PointGradients &depthDistortionGradients,
        PointGradients &normalConsistencyGradients,
        PointGradients &visibilityOpacityGradients,
        DebugImages *debugImages) {
        m_settings.rayGenMode = RayGenMode::Adjoint;
        Log::PA_DEBUG("Submitting surface regularizer backward pass");

        ScopedTimer regularizerTimer("Surface regularizer backward total", spdlog::level::debug);

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_sceneGPU,
            .intermediates = m_intermediates,
            .gradients = depthDistortionGradients,
            .depthDistortionGradients = depthDistortionGradients,
            .normalConsistencyGradients = normalConsistencyGradients,
            .visibilityOpacityGradients = visibilityOpacityGradients,
            .sensors = sensors,
            .debugImages = debugImages,
            .numSensors = static_cast<uint32_t>(sensors.size()),
        };

        submitSurfaceRegularizersKernel(renderPackage);
        m_queue.wait();
    }

    void PathTracer::reset() {
    }
}
