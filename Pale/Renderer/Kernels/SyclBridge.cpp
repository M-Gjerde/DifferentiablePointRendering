// SyclWarmup.cpp (no imports of your modules)
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <string>

#include "PostprocessKernel.h"
#include "Core/ScopedTimer.h"
#include "Renderer/GPUDataStructures.h"
#include "Renderer/Kernels/Utils.h"
#include "Renderer/Kernels/KernelHelpers.h"
#include "Renderer/Kernels/PrimalKernels.h"
#include "Renderer/Kernels/AdjointKernels.h"
#include "Renderer/Kernels/SyclBridge.h"

import Pale.Log;

namespace Pale {
    static void waitForProfiling(sycl::queue& queue) {
        if (ScopedTimerDetail::isProfilingEnabled()) {
            queue.wait();
        }
    }

    void submitLightTracingKernel(RenderPackage& pkg) {
        {
            ScopedTimer forwardTimer("Forward Pass Total", spdlog::level::debug);
            for (uint32_t forwardPass = 0; forwardPass < pkg.settings.numForwardPasses; forwardPass++) {
                {
                    ScopedTimer timer("Forward light tracing: reset primary count", spdlog::level::debug);
                    pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();
                }
                {
                    ScopedTimer timer("Forward light tracing: ray gen emitter launch", spdlog::level::debug);
                    launchRayGenEmitterKernel(pkg, forwardPass);
                }

                uint32_t activeCount = 0;
                {
                    ScopedTimer timer("Forward light tracing: read active ray count", spdlog::level::debug);
                    pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();
                }
                {
                    ScopedTimer forwardTimer("Traced forward pass", spdlog::level::debug);

                    {
                        ScopedTimer timer(
                            "Forward light tracing: emitter visible contributions launch+wait",
                            spdlog::level::debug);
                        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                            launchContributionEmitterVisibleKernel(pkg, activeCount, cameraIndex);
                        }
                        waitForProfiling(pkg.queue);
                    }

                    for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                        ScopedTimer bounceTimer("Bounce: " + std::to_string(bounce));
                        {
                            ScopedTimer timer("Forward light tracing: reset extension count", spdlog::level::debug);
                            pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                            //pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                            pkg.queue.wait();
                        }
                        uint32_t contributionCount = 0;
                        {
                            ScopedTimer timer(
                                "Forward light tracing: intersect + read contribution count",
                                spdlog::level::debug);
                            launchIntersectKernel(pkg, activeCount);
                            pkg.queue.memcpy(
                                &contributionCount,
                                pkg.intermediates.countContributions,
                                sizeof(uint32_t)).wait();
                        }
                        {
                            ScopedTimer timer("ContributionKernels total");
                            for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                                //if (pkg.sensors[cameraIndex].name[6] != '2')
                                //    continue;
                                launchContributionKernel(pkg, contributionCount, cameraIndex);
                            }
                            waitForProfiling(pkg.queue);
                        }
                        {
                            ScopedTimer timer("Forward light tracing: reset contribution count", spdlog::level::debug);
                            pkg.queue.fill(pkg.intermediates.countContributions, static_cast<uint32_t>(0), 1);
                        }

                        uint32_t nextCount = 0;
                        {
                            ScopedTimer timer("Forward light tracing: read next ray count", spdlog::level::debug);
                            pkg.queue.memcpy(&nextCount, pkg.intermediates.countExtensionOut, sizeof(uint32_t)).wait();
                        }
                        {
                            ScopedTimer timer("Forward light tracing: copy extension rays", spdlog::level::debug);
                            pkg.queue.memcpy(pkg.intermediates.primaryRays, pkg.intermediates.extensionRaysA,
                                             nextCount * sizeof(RayState));
                            pkg.queue.wait();
                        }
                        activeCount = nextCount;
                    }
                }
            }
        }
        // Gamma, exposure and rgb8 conversion
        {
            ScopedTimer timer("Post Processing launch+wait", spdlog::level::debug);
            launchPostProcessKernel(pkg);
            waitForProfiling(pkg.queue);
        }
    }

    void submitLightTracingKernelCylinderRay(RenderPackage& pkg) {
        /*
        for (uint32_t cameraIndex = 0u;
             cameraIndex < pkg.numSensors;
             ++cameraIndex) {
            for (uint32_t sampleIndex = 0u;
                 sampleIndex < pkg.settings.numGatherPasses;
                 ++sampleIndex) {
                launchPointSampledPathTracingCameraKernel(
                    pkg,
                    cameraIndex,
                    sampleIndex);
            }
        }
        */

        launchPostProcessKernel(pkg);
    }

    void submitPhotonMappingKernel(RenderPackage& pkg) {
        {
            ScopedTimer timer(
                "Forward photon mapping: camera gather total",
                spdlog::level::debug);

            for (size_t gatherPass = 0; gatherPass < pkg.settings.numGatherPasses; ++gatherPass) {
                ScopedTimer passTimer(
                    "Forward photon mapping: gather pass launch+wait",
                    spdlog::level::debug);
                for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                    const bool useCameraGatherKernel2 =
                            pkg.settings.cameraGatherKernelKind == CameraGatherKernelKind::CameraGatherKernel2;
                    const std::string kernelName =
                            useCameraGatherKernel2 ? "CameraGatherKernel2" : "CameraGatherKernel";
                    ScopedTimer cameraTimer(
                        "Forward photon mapping: " + kernelName +
                        " camera " + std::to_string(cameraIndex + 1u) + "/" +
                        std::to_string(pkg.numSensors) + " launch+wait",
                        spdlog::level::debug);
                    {
                        ScopedTimer launchTimer(
                            "Forward photon mapping: " + kernelName + " launch",
                            spdlog::level::debug);
                        if (useCameraGatherKernel2) {
                            launchCameraGatherKernel2(pkg, cameraIndex, gatherPass);
                        } else {
                            launchCameraGatherKernel(pkg, cameraIndex, gatherPass);
                        }
                    }
                    pkg.queue.wait();
                }
            }
        }

        {
            ScopedTimer timer("Post Processing launch+wait", spdlog::level::debug);
            launchPostProcessKernel(pkg);
            waitForProfiling(pkg.queue);
        }
    }

    static void clearAdjointPointGradients(RenderPackage& pkg) {
        auto& queue = pkg.queue;
        auto& gradients = pkg.gradients;
        const size_t pointCount = gradients.numPoints;
        const size_t cameraSlotCount = gradients.cameraSlotCount;
        const size_t primitiveCameraCount = pointCount * cameraSlotCount;
        if (pointCount == 0u) {
            return;
        }
        queue.fill(gradients.gradPosition, float3{0.0f, 0.0f, 0.0f}, pointCount);
        queue.fill(gradients.cloneSignal, float3{0.0f, 0.0f, 0.0f}, pointCount);
        queue.fill(gradients.gradRotation, float3{0.0f, 0.0f, 0.0f}, pointCount);
        queue.fill(gradients.gradScale, float2{0.0f, 0.0f}, pointCount);
        queue.fill(gradients.gradAlbedo, float3{0.0f, 0.0f, 0.0f}, pointCount);
        queue.fill(gradients.gradOpacity, 0.0f, pointCount);
        queue.fill(gradients.gradBeta, 0.0f, pointCount);
        queue.fill(gradients.gradShape, 0.0f, pointCount);
        queue.fill(gradients.cloneSignalMeanNorm, 0.0f, pointCount);
        queue.fill(gradients.cloneSignalStd, 0.0f, pointCount);
        queue.fill(gradients.cloneSignalCoherence, 0.0f, pointCount);
        queue.fill(gradients.cloneSignalDisagreement, 0.0f, pointCount);
        queue.fill(gradients.cloneSignalActiveCameraCount, 0u, pointCount);
        if (cameraSlotCount > 0u) {
            queue.fill(gradients.gradPositionPerPrimitivePerCamera, float3{0.0f, 0.0f, 0.0f}, primitiveCameraCount);
            queue.fill(gradients.gradPositionRecordCountPerPrimitivePerCamera, 0u, primitiveCameraCount);
            queue.fill(gradients.cloneSignalPerPrimitivePerCamera, float3{0.0f, 0.0f, 0.0f}, primitiveCameraCount);
            queue.fill(gradients.cloneSignalRecordCountPerPrimitivePerCamera, 0u, primitiveCameraCount);
            queue.fill(gradients.cloneRadianceRmsSumPerPrimitivePerCamera, 0.0f, primitiveCameraCount);
        }
    }

    // ---- Orchestrator -------------------------------------------------------
    void submitAdjointKernel(RenderPackage& pkg) {
        {
            ScopedTimer timer("Adjoint setup: clear point gradients", spdlog::level::debug);
            clearAdjointPointGradients(pkg);
        }

        const bool enableAdjointDirectLight = pkg.settings.enableAdjointDirectLight;
        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            if (pkg.settings.renderDebugGradientImages && pkg.debugImages != nullptr) {
                ScopedTimer timer("Adjoint setup: clear debug gradient images", spdlog::level::debug);
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosX, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosY, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosZ, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferRotX, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferRotY, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferRotZ, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferScaleU, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferScaleV, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferOpacity, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferAlbedo, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferBeta, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferDepthLoss, float4{0.0f},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferDepthLossPos, float4{0.0f},
                               pkg.debugImages[cameraIndex].numPixels).wait();
            }

            const uint32_t samplesPerPixel = pkg.settings.adjointSamplesPerPixel;

            for (uint32_t spp = 0; spp < samplesPerPixel; ++spp) {
                ScopedTimer forwardTimer("Traced adjoint pass", spdlog::level::debug);
                const uint32_t raysPerFrame = pkg.sensors[cameraIndex].width * pkg.sensors[cameraIndex].height;
                {
                    ScopedTimer timer("launchRayGenAdjointKernel", spdlog::level::debug);
                    Log::PA_TRACE("Generating adjoint rays");
                    launchRayGenAdjointKernel(pkg, spp, static_cast<uint32_t>(cameraIndex));
                }
                uint32_t activeRayCount = raysPerFrame;
                for (uint32_t adjointBounceIndex = 0;
                     adjointBounceIndex < pkg.settings.maxAdjointBounces && activeRayCount > 0;
                     ++adjointBounceIndex) {
                    {
                        ScopedTimer timer("Adjoint bounce setup", spdlog::level::debug);
                        pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.countMeasurementEvents, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.countMeasurementTwoPointEvents, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.countMaterialVertexEvents, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.countMaterialEndEdgeEvents, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.countMaterialStartEdgeEvents, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.countGradientRecords, static_cast<uint32_t>(0), 1);
                    }
                    {
                        Log::PA_TRACE("Launching adjoint intersect kernel");
                        ScopedTimer timer("launchAdjointIntersectKernel", spdlog::level::debug);
                        launchAdjointIntersectKernel(pkg, spp, activeRayCount, cameraIndex);
                    }
                    uint32_t measurementEventCount = 0u;
                    uint32_t measurementTwoPointEventCount = 0u;
                    uint32_t materialVertexEventCount = 0u;
                    uint32_t materialEndEdgeEventCount = 0u;
                    uint32_t materialStartEdgeEventCount = 0u;
                    {
                        ScopedTimer timer("Adjoint read event counters", spdlog::level::debug);
                        pkg.queue.memcpy(&measurementEventCount, pkg.intermediates.countMeasurementEvents,
                                         sizeof(uint32_t));
                        pkg.queue.memcpy(&measurementTwoPointEventCount, pkg.intermediates.countMeasurementTwoPointEvents,
                                         sizeof(uint32_t));
                        pkg.queue.memcpy(&materialVertexEventCount, pkg.intermediates.countMaterialVertexEvents,
                                         sizeof(uint32_t));
                        pkg.queue.memcpy(&materialEndEdgeEventCount, pkg.intermediates.countMaterialEndEdgeEvents,
                                         sizeof(uint32_t));
                        pkg.queue.memcpy(&materialStartEdgeEventCount, pkg.intermediates.countMaterialStartEdgeEvents,
                                         sizeof(uint32_t));
                        pkg.queue.wait();
                    }
                    const auto requireEventCapacity = [](uint32_t count, uint32_t capacity, const char *name) {
                        if (count > capacity) {
                            throw std::runtime_error(std::string(name) +
                                " capacity exceeded; refusing an incomplete adjoint pass");
                        }
                    };
                    requireEventCapacity(measurementEventCount, pkg.intermediates.maxMeasurementEventCount,
                                         "Camera measurement events");
                    requireEventCapacity(measurementTwoPointEventCount, pkg.intermediates.maxMeasurementTwoPointEventCount,
                                         "Camera slab/light events");
                    requireEventCapacity(materialVertexEventCount, pkg.intermediates.maxMaterialVertexEventCount,
                                         "Material vertex events");
                    requireEventCapacity(materialEndEdgeEventCount, pkg.intermediates.maxMaterialEndEdgeEventCount,
                                         "Material end-edge events");
                    requireEventCapacity(materialStartEdgeEventCount, pkg.intermediates.maxMaterialStartEdgeEventCount,
                                         "Material start-edge events");
                    {
                        ScopedTimer timer(
                            "reduceFusedFirstBounceMeasurementGradientRecords",
                            spdlog::level::debug);
                        reduceFusedFirstBounceMeasurementGradientRecords(
                            pkg,
                            static_cast<uint32_t>(cameraIndex));
                    }
                    if (measurementEventCount > 0u ||
                        measurementTwoPointEventCount > 0u ||
                        materialVertexEventCount > 0u ||
                        materialEndEdgeEventCount > 0u ||
                        materialStartEdgeEventCount > 0u) {
                        ScopedTimer timer(
                            "Total adjointContributionKernels bounce: " + std::to_string(adjointBounceIndex),
                            spdlog::level::debug);
                        adjointContributionKernels(pkg, measurementEventCount, measurementTwoPointEventCount,
                                                   materialVertexEventCount, materialEndEdgeEventCount,
                                                   materialStartEdgeEventCount, static_cast<uint32_t>(cameraIndex));
                    }

                    uint32_t nextRayCountRaw = 0u;
                    {
                        ScopedTimer timer("Adjoint read next ray count", spdlog::level::debug);
                        pkg.queue.memcpy(&nextRayCountRaw, pkg.intermediates.countExtensionOut, sizeof(uint32_t)).wait();
                    }
                    const uint32_t nextRayCount = std::min(nextRayCountRaw, pkg.intermediates.maxRayQueueCapacity);
                    if (nextRayCountRaw > pkg.intermediates.maxRayQueueCapacity) {
                        Log::PA_ERROR("Overflow: nextRayCount={} max={}", nextRayCountRaw,
                                      pkg.intermediates.maxRayQueueCapacity);
                    }
                    if (nextRayCount > 0u) {
                        ScopedTimer timer("Adjoint copy extension rays", spdlog::level::debug);
                        pkg.queue.memcpy(pkg.intermediates.primaryRays,
                                         pkg.intermediates.extensionRaysA,
                                         nextRayCount * sizeof(RayState)).wait();
                    }

                    activeRayCount = nextRayCount;
                }
            }
        }

        {
            ScopedTimer timer("Adjoint compute clone signal stats", spdlog::level::debug);
            computePerPrimitiveCloneSignalStats(pkg);
        }
    }


    static void clearPointGradients(sycl::queue& queue, PointGradients& gradients) {
        const size_t pointCount = gradients.numPoints;
        if (pointCount == 0) {
            return;
        }
        if (gradients.gradPosition) {
            queue.fill(gradients.gradPosition, float3{0.0f, 0.0f, 0.0f}, pointCount);
        }
        if (gradients.cloneSignal) {
            queue.fill(gradients.cloneSignal, float3{0.0f, 0.0f, 0.0f}, pointCount);
        }
        if (gradients.gradRotation) {
            queue.fill(gradients.gradRotation, float3{0.0f, 0.0f, 0.0f}, pointCount);
        }
        if (gradients.gradScale) {
            queue.fill(gradients.gradScale, float2{0.0f, 0.0f}, pointCount);
        }
        if (gradients.gradAlbedo) {
            queue.fill(gradients.gradAlbedo, float3{0.0f, 0.0f, 0.0f}, pointCount);
        }
        if (gradients.gradOpacity) {
            queue.fill(gradients.gradOpacity, 0.0f, pointCount);
        }
        if (gradients.gradBeta) {
            queue.fill(gradients.gradBeta, 0.0f, pointCount);
        }
        if (gradients.gradShape) {
            queue.fill(gradients.gradShape, 0.0f, pointCount);
        }
    }

    void submitSurfaceRegularizersKernel(RenderPackage& pkg) {
        {
            ScopedTimer timer("Surface regularizer clear gradients", spdlog::level::debug);
            clearPointGradients(pkg.queue, pkg.depthDistortionGradients);
            clearPointGradients(pkg.queue, pkg.normalConsistencyGradients);
            clearPointGradients(pkg.queue, pkg.visibilityOpacityGradients);
            clearPointGradients(pkg.queue, pkg.intraSlabDepthGradients);
            clearPointGradients(pkg.queue, pkg.curvatureScaleGradients);
            pkg.queue.wait();
        }

        for (uint32_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            auto& sensor = pkg.sensors[cameraIndex];
            const uint32_t pixelCount = sensor.width * sensor.height;

            {
                ScopedTimer timer("Surface regularizer clear median depth adjoint", spdlog::level::debug);
                pkg.queue.fill(sensor.medianDepthAdjointBuffer, 0.0f, pixelCount).wait();
            }

            if (pkg.settings.normalConsistencyWeight != 0.0f) {
                ScopedTimer timer("launchNormalFromDepthAdjointKernel", spdlog::level::debug);
                launchNormalFromDepthAdjointKernel(pkg, cameraIndex);
            }

            {
                ScopedTimer timer("launchSurfaceRegularizersBackwardKernel", spdlog::level::debug);
                launchSurfaceRegularizersBackwardKernel(pkg, cameraIndex);
            }
        }

        {
            ScopedTimer timer("Surface regularizer final wait", spdlog::level::debug);
            pkg.queue.wait();
        }
    }

    void submitDepthDistortionKernel(RenderPackage& pkg) {
        pkg.queue.fill(pkg.gradients.gradPosition, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.cloneSignal, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradRotation, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradScale, float2{0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradAlbedo, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradOpacity, 0.0f, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradBeta, 0.0f, pkg.gradients.numPoints);
        pkg.queue.wait();
        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            //launchDepthDistortionBackwardKernel(pkg, cameraIndex);
        }
    }

    void submitNormalConsistencyKernel(RenderPackage& pkg) {
        pkg.queue.fill(pkg.gradients.gradPosition, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.cloneSignal, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradRotation, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradScale, float2{0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradAlbedo, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradOpacity, 0.0f, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradBeta, 0.0f, pkg.gradients.numPoints);
        pkg.queue.wait();
        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            auto& sensor = pkg.sensors[cameraIndex];
            const uint32_t pixelCount = sensor.width * sensor.height;
            pkg.queue.fill(sensor.medianDepthAdjointBuffer, 0.0f, pixelCount).wait();
            //launchNormalFromDepthAdjointKernel(pkg, cameraIndex);
            //launchNormalConsistencyBackwardKernel(pkg, cameraIndex);
        }
        pkg.queue.wait();
    }
}
