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

import Pale.Log;

namespace Pale {
    // ---- Orchestrator -------------------------------------------------------
    void submitKernel(RenderPackage& pkg) {
        std::mt19937_64 seedGen(pkg.settings.randomSeed); // define once before the loop

        if (pkg.settings.rayGenMode == RayGenMode::Emitter) {

            for (int forwardPass = 0; forwardPass < pkg.settings.numForwardPasses; forwardPass++) {
                pkg.settings.randomSeed = seedGen(); // new high-entropy seed each pass

                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();
                {
                    ScopedTimer timer("launchRayGenEmitterKernel");
                    launchRayGenEmitterKernel(pkg);
                }

                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();

                {
                    ScopedTimer forwardTimer("Traced forward pass", spdlog::level::debug);

                    for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                        pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                        pkg.queue.wait();
                        {
                            ScopedTimer timer("launchIntersectKernel");
                            launchIntersectKernel(pkg, activeCount);
                        }
                        {
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
            uint32_t photonMapCount = 0;
            pkg.queue.memcpy(&photonMapCount, pkg.intermediates.map.photonCountDevicePtr, sizeof(uint32_t)).wait();
            {
                ScopedTimer timer("clearGridHeads");
                clearGridHeads(pkg.queue, pkg.intermediates.map);
            }
            {
                ScopedTimer timer("buildPhotonGridLinkedLists", spdlog::level::debug);
                size_t photonCount = std::min(photonMapCount, pkg.intermediates.map.photonCapacity);
                buildPhotonGridLinkedLists(pkg.queue, pkg.intermediates.map, photonCount);
            }

            // Save photon map to disk:
            {
                /*
                ScopedTimer timer("dumpPhotonMapToPLY");
                dumpPhotonMapToPLY(pkg.queue,
                                  pkg.intermediates.map.photons,
                                  photonMapCount,
                                  std::filesystem::path("Output/photon_map.ply"));
                */
            }
            {
                ScopedTimer timer("launchCameraGatherKernel", spdlog::level::debug);
                int cameraGatherSPP = pkg.settings.numGatherPasses;
                for (int i = 0; i < cameraGatherSPP; i++) {
                    ScopedTimer timer("CameraGatherSample", spdlog::level::debug);
                    std::mt19937_64 seedGen(pkg.settings.randomSeed); // define once before the loop
                    pkg.settings.randomSeed = seedGen(); // new high-entropy seed each pass
                    launchCameraGatherKernel(pkg, cameraGatherSPP); // generate image from photon map
                }
            }
            // Post processing:
            // Gamma, exposure and rgb8 conversion
            launchPostProcessKernel(pkg);
        }
        else if (pkg.settings.rayGenMode == RayGenMode::Adjoint) {
            pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();

            pkg.queue.fill(pkg.gradients.gradPosition, float3{0, 0, 0}, pkg.gradients.numPoints).wait();
            pkg.queue.fill(pkg.gradients.gradTanU, float3{0, 0, 0}, pkg.gradients.numPoints).wait();
            pkg.queue.fill(pkg.gradients.gradTanV, float3{0, 0, 0}, pkg.gradients.numPoints).wait();
            pkg.queue.fill(pkg.gradients.gradScale, float2{0, 0}, pkg.gradients.numPoints).wait();
            pkg.queue.fill(pkg.gradients.gradColor, float3{0}, pkg.gradients.numPoints).wait();
            pkg.queue.fill(pkg.gradients.gradOpacity, 0.0f, pkg.gradients.numPoints).wait();

            for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {

                if (pkg.settings.renderDebugGradientImages) {
                    pkg.queue.fill(pkg.debugImages[cameraIndex].framebuffer_pos, float4{0}, pkg.debugImages[cameraIndex].numPixels).wait();
                    pkg.queue.fill(pkg.debugImages[cameraIndex].framebuffer_rot, float4{0}, pkg.debugImages[cameraIndex].numPixels).wait();
                    pkg.queue.fill(pkg.debugImages[cameraIndex].framebuffer_scale, float4{0}, pkg.debugImages[cameraIndex].numPixels).wait();
                    pkg.queue.fill(pkg.debugImages[cameraIndex].framebuffer_opacity, float4{0}, pkg.debugImages[cameraIndex].numPixels).wait();
                    pkg.queue.fill(pkg.debugImages[cameraIndex].framebuffer_albedo, float4{0}, pkg.debugImages[cameraIndex].numPixels).wait();
                    pkg.queue.fill(pkg.debugImages[cameraIndex].framebuffer_beta, float4{0}, pkg.debugImages[cameraIndex].numPixels).wait();
                }

            int samplesPerPixel = pkg.settings.adjointSamplesPerPixel;
            for (int spp = 0; spp < samplesPerPixel; ++spp) {
                pkg.settings.randomSeed = pkg.settings.randomSeed * spp;
                ScopedTimer forwardTimer("Traced adjoint pass", spdlog::level::debug);

                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();
                {
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
                    pkg.queue.wait();
                    {
                        ScopedTimer timer("launchIntersectKernel");
                        launchIntersectKernel(pkg, activeCount);
                    }
                    {
                        ScopedTimer timer("launchAdjointKernel");
                        if (bounce == 0) {
                            launchAdjointKernel(pkg, activeCount, cameraIndex);
                        }
                        else {
                            launchAdjointKernel2(pkg, activeCount, cameraIndex);
                        }
                    }
                    {
                        ScopedTimer timer("generateNextRays");
                        generateNextAdjointRays(pkg, activeCount);
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
        }}
    }
}
