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
    void submitLightTracingKernel(RenderPackage& pkg) {
        {
            ScopedTimer forwardTimer("Forward Pass Total", spdlog::level::debug);
            for (uint32_t forwardPass = 0; forwardPass < pkg.settings.numForwardPasses; forwardPass++) {
                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();
                {
                    ScopedTimer timer("launchRayGenEmitterKernel");
                    launchRayGenEmitterKernel(pkg, forwardPass);
                }

                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();
                {
                    ScopedTimer forwardTimer("Traced forward pass", spdlog::level::debug);

                    for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                        launchContributionEmitterVisibleKernel(pkg, activeCount, cameraIndex);
                    }

                    for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                        ScopedTimer bounceTimer("Bounce: " + std::to_string(bounce));
                        pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                        //pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                        pkg.queue.wait();
                        {
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

    void submitPhotonMappingKernel(RenderPackage& pkg) {
        const uint64_t renderSeed = pkg.random.seed; // capture value

        pkg.queue.fill(pkg.intermediates.map.photonCountDevicePtr, 0u, 1).wait();
        {
            ScopedTimer forwardTimer("Forward Pass Total", spdlog::level::debug);
            for (int forwardPass = 0; forwardPass < pkg.settings.numForwardPasses; forwardPass++) {
                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();
                {
                    ScopedTimer timer("launchRayGenEmitterKernel");
                    launchRayGenEmitterKernel(pkg, forwardPass);
                }

                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();
                {
                    ScopedTimer forwardTimer("Traced forward pass", spdlog::level::debug);

                    for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces; ++bounce) {
                        if (activeCount > 0) {
                            pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                            pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                            pkg.queue.wait();
                            {
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
                        }
                    }
                }
            }
        }
        uint32_t photonMapCount = 0;
        pkg.queue.memcpy(&photonMapCount,
                         pkg.intermediates.map.photonCountDevicePtr,
                         sizeof(uint32_t)).wait();
        const uint32_t photonCount = std::min(photonMapCount, pkg.intermediates.map.photonCapacity);
        {
            ScopedTimer timer("buildPhotonCellRangesAndOrdering", spdlog::level::debug);
            buildPhotonCellRangesAndOrdering(pkg.queue, pkg.intermediates.map, photonCount);
        }
        {
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

            Log::PA_INFO("Photonmap is at {:.2f}% capacity", percentFull);


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
    // ---- Orchestrator -------------------------------------------------------
    void submitAdjointKernel(RenderPackage& pkg) {
        std::mt19937_64 seedGenerator(pkg.random.seed);

        pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();

        pkg.queue.fill(pkg.gradients.gradPosition, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradTanU, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradTanV, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradScale, float2{0.0f, 0.0f}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradAlbedo, float3{0.0f, 0.0f, 0.0f}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradOpacity, 0.0f, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradBeta, 0.0f, pkg.gradients.numPoints).wait();

        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            if (pkg.settings.renderDebugGradientImages) {
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosX, float4{0.0f},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosY, float4{0.0f},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosZ, float4{0.0f},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferRot, float4{0.0f},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferScale, float4{0.0f},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferOpacity, float4{0.0f},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferAlbedo, float4{0.0f},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferBeta, float4{0.0f},
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

                {
                    ScopedTimer timer("launchRayGenAdjointKernel");
                    Log::PA_TRACE("Generating adjoint rays");
                    launchRayGenAdjointKernel(pkg, spp, static_cast<uint32_t>(cameraIndex));
                }

                const uint32_t raysPerFrame =
                    pkg.sensors[cameraIndex].width * pkg.sensors[cameraIndex].height;

                uint32_t activeCount = raysPerFrame;

                // One pending state slot per pathId/pixel.
                // Clear the full pending arrays once per spp.
                pkg.queue.fill(pkg.intermediates.pendingStageX, PendingAdjointStageX{}, raysPerFrame).wait();
                pkg.queue.fill(pkg.intermediates.pendingStageXY, PendingAdjointStageXY{}, raysPerFrame).wait();

                for (uint32_t bounce = 0;
                     bounce < pkg.settings.maxAdjointBounces && activeCount > 0;
                     ++bounce) {
                    pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.countProjectionEvents, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.countProjectionScatterEvents, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.countReflectScatterEvents, static_cast<uint32_t>(0), 1);

                    pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit{}, activeCount);
                    pkg.queue.wait();

                    {
                        Log::PA_TRACE("Launching adjoint intersect kernel");
                        ScopedTimer timer("launchAdjointIntersectKernel", spdlog::level::debug);
                        launchAdjointIntersectKernel(pkg, spp, activeCount, bounce);
                    }

                    uint32_t projectionEventCount = 0;
                    uint32_t projectionScatterEventCount = 0;
                    uint32_t reflectScatterEventCount = 0;

                    pkg.queue.memcpy(
                        &projectionEventCount,
                        pkg.intermediates.countProjectionEvents,
                        sizeof(uint32_t)).wait();

                    pkg.queue.memcpy(
                        &projectionScatterEventCount,
                        pkg.intermediates.countProjectionScatterEvents,
                        sizeof(uint32_t)).wait();

                    pkg.queue.memcpy(
                        &reflectScatterEventCount,
                        pkg.intermediates.countReflectScatterEvents,
                        sizeof(uint32_t)).wait();

                    projectionEventCount = sycl::min(
                        projectionEventCount,
                        pkg.intermediates.maxProjectionEventCount);

                    projectionScatterEventCount = sycl::min(
                        projectionScatterEventCount,
                        pkg.intermediates.maxProjectionScatterEventCount);

                    reflectScatterEventCount = sycl::min(
                        reflectScatterEventCount,
                        pkg.intermediates.maxReflectScatterEventCount);

                    if (projectionEventCount > 0 ||
                        projectionScatterEventCount > 0 ||
                        reflectScatterEventCount > 0) {
                        ScopedTimer timer("adjointContributionKernels", spdlog::level::debug);

                        adjointContributionKernels(
                            pkg,
                            projectionEventCount,
                            projectionScatterEventCount,
                            reflectScatterEventCount,
                            static_cast<uint32_t>(cameraIndex));
                    }

                    uint32_t nextCount = 0;
                    pkg.queue.memcpy(
                        &nextCount,
                        pkg.intermediates.countExtensionOut,
                        sizeof(uint32_t)).wait();

                    if (nextCount > 0) {
                        pkg.queue.memcpy(
                            pkg.intermediates.primaryRays,
                            pkg.intermediates.extensionRaysA,
                            nextCount * sizeof(RayState)).wait();
                    }

                    activeCount = nextCount;
                }
            }
        }
    }
}
