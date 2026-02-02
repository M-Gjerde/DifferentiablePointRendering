// SyclWarmup.cpp (no imports of your modules)
#include <sycl/sycl.hpp>
#include "Renderer/Kernels/SyclBridge.h"

#include "PostprocessKernel.h"
#include "Renderer/Kernels/AdjointKernels.h"

#include "Core/ScopedTimer.h"

#include "Renderer/GPUDataStructures.h"

#include "Renderer/Kernels/Utils.h"
#include "Renderer/Kernels/KernelHelpers.h"
#include "Renderer/Kernels/PrimalKernels.h"
#include "Renderer/Kernels/CylinderRayKernels.h"

import Pale.Log;

namespace Pale {
    void submitLightTracingKernelCylinderRay(RenderPackage &pkg) {
        std::mt19937_64 seedGen(pkg.settings.randomSeed); // define once before the loop

        pkg.queue.fill(pkg.intermediates.map.photonCountDevicePtr, 0u, 1).wait();

        for (int forwardPass = 0; forwardPass < pkg.settings.numForwardPasses; forwardPass++) {
            pkg.settings.randomSeed = seedGen(); // new high-entropy seed each pass

            pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait(); {
                ScopedTimer timer("launchRayGenEmitterKernel");
                launchRayGenEmitterKernel(pkg);
            }

            uint32_t activeCount = 0;
            pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait(); {
                ScopedTimer forwardTimer("Traced forward pass", spdlog::level::debug);

                for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                    launchContributionEmitterVisibleKernel(pkg, activeCount, cameraIndex);
                }

                for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                    pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                    pkg.queue.wait(); {
                        ScopedTimer timer("launchIntersectKernel");
                        launchCylinderRayIntersectKernel(pkg, activeCount);
                    }
                    for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                        launchCylinderContributionKernel(pkg, activeCount, cameraIndex);
                    } {
                        ScopedTimer timer("generateNextRays");
                        generateNextRays(pkg, activeCount);
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
        // Gamma, exposure and rgb8 conversion
        launchPostProcessKernel(pkg);
    }

    void submitLightTracingKernel(RenderPackage &pkg) {
        std::mt19937_64 seedGen(pkg.settings.randomSeed); // define once before the loop

        pkg.queue.fill(pkg.intermediates.map.photonCountDevicePtr, 0u, 1).wait();
        {
            ScopedTimer forwardTimer("Forward Pass Total", spdlog::level::debug);
            for (int forwardPass = 0; forwardPass < pkg.settings.numForwardPasses; forwardPass++) {
                pkg.settings.randomSeed = seedGen(); // new high-entropy seed each pass

                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait(); {
                    ScopedTimer timer("launchRayGenEmitterKernel");
                    launchRayGenEmitterKernel(pkg);
                }

                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait(); {
                    ScopedTimer forwardTimer("Traced forward pass", spdlog::level::debug);

                    for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                        launchContributionEmitterVisibleKernel(pkg, activeCount, cameraIndex);
                    }

                    for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                        pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                        pkg.queue.wait(); {
                            ScopedTimer timer("launchIntersectKernel");
                            launchIntersectKernel(pkg, activeCount);
                        }
                        if (pkg.settings.integratorKind == IntegratorKind::lightTracing) {
                            ScopedTimer timer("ContributionKernels total");
                            for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
                                launchContributionKernel(pkg, activeCount, cameraIndex);
                            }
                        } {
                            ScopedTimer timer("generateNextRays");
                            generateNextRays(pkg, activeCount);
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
        // Gamma, exposure and rgb8 conversion
        {
            ScopedTimer timer("Post Processing", spdlog::level::debug);
            launchPostProcessKernel(pkg);
        }
    }

    void submitPhotonMappingKernel(RenderPackage &pkg) {
        std::mt19937_64 seedGen(pkg.settings.randomSeed); // define once before the loop

        pkg.queue.fill(pkg.intermediates.map.photonCountDevicePtr, 0u, 1).wait(); {
            ScopedTimer forwardTimer("Forward Pass Total", spdlog::level::debug);
            for (int forwardPass = 0; forwardPass < pkg.settings.numForwardPasses; forwardPass++) {
                pkg.settings.randomSeed = seedGen(); // new high-entropy seed each pass

                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait(); {
                    ScopedTimer timer("launchRayGenEmitterKernel");
                    launchRayGenEmitterKernel(pkg);
                }

                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait(); {
                    ScopedTimer forwardTimer("Traced forward pass", spdlog::level::debug);

                    for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                        pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                        pkg.queue.wait(); {
                            ScopedTimer timer("launchIntersectKernel");
                            launchIntersectKernel(pkg, activeCount);
                        } {
                            ScopedTimer timer("generateNextRays");
                            generateNextRays(pkg, activeCount);
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
                ScopedTimer timer(
                    "launchCameraGatherKernel: " + std::to_string(cameraIndex) + "/" +
                    std::to_string(pkg.numSensors), spdlog::level::debug);
                launchCameraGatherKernel(pkg, cameraIndex); // generate image from photon map
                pkg.queue.wait();
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
        std::mt19937_64 seedGen(pkg.settings.randomSeed); // define once before the loop

        pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();
        pkg.queue.fill(pkg.gradients.gradPosition, float3{0, 0, 0}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradTanU, float3{0, 0, 0}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradTanV, float3{0, 0, 0}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradScale, float2{0, 0}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradAlbedo, float3{0}, pkg.gradients.numPoints).wait();
        pkg.queue.fill(pkg.gradients.gradOpacity, 0.0f, pkg.gradients.numPoints).wait();

        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            if (pkg.settings.renderDebugGradientImages) {
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosX, float4{0},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosY, float4{0},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferPosZ, float4{0},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferRot, float4{0},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferScale, float4{0},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferOpacity, float4{0},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferAlbedo, float4{0},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferBeta, float4{0},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferDepthLoss, float4{0},
                               pkg.debugImages[cameraIndex].numPixels).wait();
                pkg.queue.fill(pkg.debugImages[cameraIndex].framebufferDepthLossPos, float4{0},
                               pkg.debugImages[cameraIndex].numPixels).wait();
            }

            int samplesPerPixel = pkg.settings.adjointSamplesPerPixel;
            for (int spp = 0; spp < samplesPerPixel; ++spp) {
                pkg.settings.randomSeed = pkg.settings.randomSeed * spp;
                ScopedTimer forwardTimer("Traced adjoint pass", spdlog::level::debug);

                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait(); {
                    ScopedTimer timer("launchRayGenAdjointKernel");
                    std::mt19937_64 seedGen(pkg.settings.randomSeed); // define once before the loop
                    pkg.settings.randomSeed = seedGen(); // new high-entropy seed each pass
                    launchRayGenAdjointKernel(pkg, spp, cameraIndex);
                }

                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();

                for (uint32_t bounce = 0; bounce < pkg.settings.maxAdjointBounces && activeCount > 0; ++bounce) {
                    pkg.settings.randomSeed = pkg.settings.randomSeed + bounce;
                    pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                    pkg.queue.wait(); {
                        ScopedTimer timer("launchIntersectKernel");
                        launchIntersectKernel(pkg, activeCount);
                    } {
                        ScopedTimer timer("launchAdjointKernel");
                        if (bounce == 0) {
                            launchAdjointProjectionKernel(pkg, activeCount, cameraIndex);
                        } else {
                            launchAdjointTransportKernel(pkg, activeCount, cameraIndex);
                        }
                    } {
                        ScopedTimer timer("generateNextRays");
                        //generateNextAdjointRays(pkg, activeCount);
                        generateNextRays(pkg, activeCount);
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


            /*
            {
                Point point;
                float3 grad;

                pkg.queue.memcpy(&point, &pkg.scene.points[2], sizeof(Point));
                pkg.queue.memcpy(&grad, &pkg.gradients.gradPosition[2], sizeof(float3));
                pkg.queue.wait();

                Log::PA_WARN("Pos: ({},{},{}) | grad: ({},{},{})", point.position.x(), point.position.y(),
                             point.position.z(), grad.x(), grad.y(), grad.z());
            }
            {
                Point point;
                float3 grad;

                pkg.queue.memcpy(&point, &pkg.scene.points[pkg.scene.pointCount - 1], sizeof(Point));
                pkg.queue.memcpy(&grad, &pkg.gradients.gradPosition[pkg.scene.pointCount - 1], sizeof(float3));
                pkg.queue.wait();

                Log::PA_WARN("Pos: ({},{},{}) | grad: ({},{},{})", point.position.x(), point.position.y(),
                             point.position.z(), grad.x(), grad.y(), grad.z());
            }
            */
        }
    }
}
