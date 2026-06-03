// SyclWarmup.cpp (no imports of your modules)
#include <sycl/sycl.hpp>

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
    void submitLightTracingKernel(RenderPackage &pkg) { {
            ScopedTimer forwardTimer("Forward Pass Total", spdlog::level::debug);
            for (uint32_t forwardPass = 0; forwardPass < pkg.settings.numForwardPasses; forwardPass++) {
                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait(); {
                    ScopedTimer timer("launchRayGenEmitterKernel");
                    launchRayGenEmitterKernel(pkg, forwardPass);
                }

                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait(); {
                    ScopedTimer forwardTimer("Traced forward pass", spdlog::level::debug);

                    for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                        launchContributionEmitterVisibleKernel(pkg, activeCount, cameraIndex);
                    }

                    for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                        ScopedTimer bounceTimer("Bounce: " + std::to_string(bounce));
                        pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                        //pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                        pkg.queue.wait(); {
                            ScopedTimer timer("launchIntersectKernel");
                            launchIntersectKernel(pkg, activeCount);
                        }
                        ScopedTimer timer("ContributionKernels total");
                        uint32_t contributionCount = 0;
                        pkg.queue.memcpy(&contributionCount, pkg.intermediates.countContributions,
                                         sizeof(uint32_t)).wait();
                        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                            //if (pkg.sensors[cameraIndex].name[6] != '2')
                            //    continue;
                            launchContributionKernel(pkg, contributionCount, cameraIndex);
                        }
                        pkg.queue.fill(pkg.intermediates.countContributions, static_cast<uint32_t>(0), 1);

                        uint32_t nextCount = 0;
                        pkg.queue.memcpy(&nextCount, pkg.intermediates.countExtensionOut, sizeof(uint32_t)).wait();
                        pkg.queue.memcpy(pkg.intermediates.primaryRays, pkg.intermediates.extensionRaysA,
                                         nextCount * sizeof(RayState));
                        pkg.queue.wait();
                        activeCount = nextCount;
                        pkg.queue.wait();
                    }
                }
            }
        }
        // Gamma, exposure and rgb8 conversion
        {
            ScopedTimer timer("Post Processing", spdlog::level::debug);
            launchPostProcessKernel(pkg);
        }
    }

    void submitPhotonMappingKernel(RenderPackage &pkg) {
        const uint64_t renderSeed = pkg.random.seed; // capture value

        pkg.queue.fill(pkg.intermediates.map.photonCountDevicePtr, 0u, 1).wait(); {
            ScopedTimer forwardTimer("Forward Pass Total", spdlog::level::debug);
            for (int forwardPass = 0; forwardPass < pkg.settings.numForwardPasses; forwardPass++) {
                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait(); {
                    ScopedTimer timer("launchRayGenEmitterKernel");
                    launchRayGenEmitterKernel(pkg, forwardPass);
                }

                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait(); {
                    ScopedTimer forwardTimer("Traced forward pass", spdlog::level::debug);

                    for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces; ++bounce) {
                        pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                        pkg.queue.wait(); {
                            ScopedTimer timer("launchIntersectKernel");
                            launchIntersectKernel(pkg, activeCount);
                        }
                        uint32_t nextCount = 0;
                        pkg.queue.memcpy(&nextCount, pkg.intermediates.countExtensionOut, sizeof(uint32_t)).wait();
                        pkg.queue.memcpy(pkg.intermediates.primaryRays, pkg.intermediates.extensionRaysA,
                                         nextCount * sizeof(RayState));
                        pkg.queue.wait();
                        activeCount = nextCount;
                        pkg.queue.wait();
                        if (!activeCount)
                            break;
                    }
                }
            }
        }
        uint32_t photonMapCount = 0;
        pkg.queue.memcpy(&photonMapCount,
                         pkg.intermediates.map.photonCountDevicePtr,
                         sizeof(uint32_t)).wait();

        const uint32_t photonCount = std::min(photonMapCount, pkg.intermediates.map.photonCapacity); {
            ScopedTimer timer("buildPhotonCellRangesAndOrdering", spdlog::level::debug);
            buildPhotonCellRangesAndOrdering(pkg.queue, pkg.intermediates.map, photonCount);
        } {
            ScopedTimer timer("Camera Gather for " + std::to_string(pkg.numSensors) + " cameras", spdlog::level::debug);

            for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                //if (pkg.sensors[cameraIndex].name[6] != '1')
                //    continue;

                ScopedTimer timer(
                    "launchCameraGatherKernel: " + std::to_string(cameraIndex) + "/" +
                    std::to_string(pkg.numSensors), spdlog::level::debug);
                launchCameraGatherKernel(pkg, cameraIndex); // generate image from photon map
                pkg.queue.wait();
            }

            // Photon map stats:
            // Percent full
            uint32_t photonMapCount = 0;
            pkg.queue.memcpy(&photonMapCount,
                             pkg.intermediates.map.photonCountDevicePtr,
                             sizeof(uint32_t)).wait();

            const uint32_t photonCapacity = pkg.intermediates.map.photonCapacity;

            const float percentFull =
                    photonCapacity > 0
                        ? 100.0f * static_cast<float>(photonMapCount) / static_cast<float>(photonCapacity)
                        : 0.0f;

            if (percentFull < 99.0f) Log::PA_INFO("Photonmap is at {:.2f}% capacity", percentFull);
            else {
                Log::PA_ERROR("Photonmap is at {:.2f}% capacity", percentFull);
            }


            /*
            {

                ScopedTimer timer("dumpPhotonMapToPLY");
                dumpPhotonMapToPLY(pkg.queue,
                                  pkg.intermediates.map.photons,
                                  photonMapCount,
                                  std::filesystem::path("Output/photon_map.ply"));

            }
            */
        }

        // Save photon map to disk:
        {
            ScopedTimer timer("Post Processing", spdlog::level::debug);
            launchPostProcessKernel(pkg);
        }
    }

    // ---- Orchestrator -------------------------------------------------------
    void submitAdjointKernel(RenderPackage &pkg) {
        pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();
        pkg.queue.fill(pkg.gradients.gradPosition, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradTanU, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradTanV, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradScale, float2{0.0f, 0.0f}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradAlbedo, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradOpacity, 0.0f, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradBeta, 0.0f, pkg.gradients.numPoints).wait();
        const bool enableAdjointDirectLight = pkg.settings.enableAdjointDirectLight;
        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            if (pkg.settings.renderDebugGradientImages) {
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosX, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosY, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosZ, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferRot, 0.0f,
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferScale, 0.0f,
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
                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();
                const uint32_t raysPerFrame = pkg.sensors[cameraIndex].width * pkg.sensors[cameraIndex].height;
                pkg.queue.fill(pkg.intermediates.pendingCameraSegments, PendingCameraSegment{}, raysPerFrame).wait(); {
                    ScopedTimer timer("launchRayGenAdjointKernel");
                    Log::PA_TRACE("Generating adjoint rays");
                    launchRayGenAdjointKernel(pkg, spp, static_cast<uint32_t>(cameraIndex));
                }
                uint32_t activeRayCount = raysPerFrame;
                pkg.queue.fill(pkg.intermediates.pendingStageX, PendingAdjointStageX{}, raysPerFrame).wait();
                for (uint32_t adjointBounceIndex = 0;
                     adjointBounceIndex < pkg.settings.maxAdjointBounces && activeRayCount > 0;
                     ++adjointBounceIndex) {
                    pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.countMeasurementEvents, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.countMeasurementTwoPointEvents, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.countMaterialVertexEvents, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.countMaterialEndEdgeEvents, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.countMaterialStartEdgeEvents, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit{}, activeRayCount);
                    pkg.queue.wait(); {
                        Log::PA_TRACE("Launching adjoint intersect kernel");
                        ScopedTimer timer("launchAdjointIntersectKernel", spdlog::level::debug);
                        launchAdjointIntersectKernel(pkg, spp, activeRayCount, cameraIndex);
                    }
                    uint32_t measurementEventCount = 0u;
                    uint32_t measurementTwoPointEventCount = 0u;
                    uint32_t materialVertexEventCount = 0u;
                    uint32_t materialEndEdgeEventCount = 0u;
                    uint32_t materialStartEdgeEventCount = 0u;
                    pkg.queue.memcpy(&measurementEventCount, pkg.intermediates.countMeasurementEvents,
                                     sizeof(uint32_t)).wait();
                    pkg.queue.memcpy(&measurementTwoPointEventCount, pkg.intermediates.countMeasurementTwoPointEvents,
                                     sizeof(uint32_t)).wait();
                    pkg.queue.memcpy(&materialVertexEventCount, pkg.intermediates.countMaterialVertexEvents,
                                     sizeof(uint32_t)).wait();
                    pkg.queue.memcpy(&materialEndEdgeEventCount, pkg.intermediates.countMaterialEndEdgeEvents,
                                     sizeof(uint32_t)).wait();
                    pkg.queue.memcpy(&materialStartEdgeEventCount, pkg.intermediates.countMaterialStartEdgeEvents,
                                     sizeof(uint32_t)).wait();
                    measurementEventCount = std::min(measurementEventCount, pkg.intermediates.maxMeasurementEventCount);
                    measurementTwoPointEventCount = std::min(measurementTwoPointEventCount,
                                                             pkg.intermediates.maxMeasurementTwoPointEventCount);
                    materialVertexEventCount = std::min(materialVertexEventCount,
                                                        pkg.intermediates.maxMaterialVertexEventCount);
                    materialEndEdgeEventCount = std::min(materialEndEdgeEventCount,
                                                         pkg.intermediates.maxMaterialEndEdgeEventCount);
                    materialStartEdgeEventCount = std::min(materialStartEdgeEventCount,
                                                           pkg.intermediates.maxMaterialStartEdgeEventCount);
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
                    pkg.queue.memset(pkg.intermediates.gradientRecords, 0x00,
                                     pkg.intermediates.maxGradientRecordCount * sizeof(SurfelGradientRecord));
                    uint32_t nextRayCountRaw = 0u;
                    pkg.queue.memcpy(&nextRayCountRaw, pkg.intermediates.countExtensionOut, sizeof(uint32_t)).wait();const uint32_t nextRayCount =                            std::min(nextRayCountRaw, pkg.intermediates.maxRayQueueCapacity);
                    if (nextRayCountRaw > pkg.intermediates.maxRayQueueCapacity) {
                        Log::PA_ERROR("Overflow: nextRayCount={} max={}",                                      nextRayCountRaw,                                      pkg.intermediates.maxRayQueueCapacity);
                    }
                    if (nextRayCount > 0u) {
                        pkg.queue.memcpy(pkg.intermediates.primaryRays,
                                         pkg.intermediates.extensionRaysA,
                                         nextRayCount * sizeof(RayState)).wait();
                    }

                    activeRayCount = nextRayCount;
                }
            }
        }
    }

    static void clearPointGradients(sycl::queue &queue, PointGradients &gradients) {
        const size_t pointCount = gradients.numPoints;
        if (pointCount == 0) {
            return;
        }

        if (gradients.gradPosition) {
            queue.fill(gradients.gradPosition, float3{0.0f, 0.0f, 0.0f}, pointCount);
        }
        if (gradients.gradTanU) {
            queue.fill(gradients.gradTanU, float3{0.0f, 0.0f, 0.0f}, pointCount);
        }
        if (gradients.gradTanV) {
            queue.fill(gradients.gradTanV, float3{0.0f, 0.0f, 0.0f}, pointCount);
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

    void submitSurfaceRegularizersKernel(RenderPackage &pkg) {
        clearPointGradients(pkg.queue, pkg.depthDistortionGradients);
        clearPointGradients(pkg.queue, pkg.normalConsistencyGradients);
        clearPointGradients(pkg.queue, pkg.visibilityOpacityGradients);
        pkg.queue.wait();

        for (uint32_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            auto &sensor = pkg.sensors[cameraIndex];
            const uint32_t pixelCount = sensor.width * sensor.height;

            pkg.queue.fill(sensor.medianDepthAdjointBuffer, 0.0f, pixelCount).wait();

            if (pkg.settings.normalConsistencyWeight != 0.0f) {
                launchNormalFromDepthAdjointKernel(pkg, cameraIndex);
            }

            launchSurfaceRegularizersBackwardKernel(pkg, cameraIndex);
        }

        pkg.queue.wait();
    }

    void submitDepthDistortionKernel(RenderPackage &pkg) {
        pkg.queue.fill(pkg.gradients.gradPosition, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradTanU, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradTanV, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradScale, float2{0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradAlbedo, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradOpacity, 0.0f, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradBeta, 0.0f, pkg.gradients.numPoints);
        pkg.queue.wait();
        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            launchDepthDistortionBackwardKernel(pkg, cameraIndex);
        }
    }

    void submitNormalConsistencyKernel(RenderPackage &pkg) {
        pkg.queue.fill(pkg.gradients.gradPosition, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradTanU, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradTanV, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradScale, float2{0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradAlbedo, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradOpacity, 0.0f, pkg.gradients.numPoints);
        pkg.queue.fill(pkg.gradients.gradBeta, 0.0f, pkg.gradients.numPoints);
        pkg.queue.wait();
        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            auto &sensor = pkg.sensors[cameraIndex];
            const uint32_t pixelCount = sensor.width * sensor.height;
            pkg.queue.fill(sensor.medianDepthAdjointBuffer, 0.0f, pixelCount).wait();
            launchNormalFromDepthAdjointKernel(pkg, cameraIndex);
            launchNormalConsistencyBackwardKernel(pkg, cameraIndex);
        }
        pkg.queue.wait();
    }
}
