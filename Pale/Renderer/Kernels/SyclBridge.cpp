// SyclWarmup.cpp (no imports of your modules)
#include <sycl/sycl.hpp>
#include "Renderer/Kernels/SyclBridge.h"
#include "Renderer/Kernels/AdjointKernels.h"

#include "Core/ScopedTimer.h"

#include "Renderer/GPUDataStructures.h"

#include "Renderer/Kernels/Utils.h"
#include "Renderer/Kernels/KernelHelpers.h"
#include "Renderer/Kernels/PrimalKernels.h"

namespace Pale {
    // ---- Orchestrator -------------------------------------------------------
    void submitKernel(RenderPackage &pkg) {
        pkg.queue.fill(pkg.sensor.framebuffer, sycl::float4{0, 0, 0, 0}, pkg.sensor.height * pkg.sensor.width).wait();
        pkg.queue.fill(pkg.photonMapSensor.framebuffer, sycl::float4{0, 0, 0, 0},
                       pkg.photonMapSensor.height * pkg.photonMapSensor.width).wait();

        std::mt19937_64 seedGen(pkg.settings.randomSeed); // define once before the loop

        if (pkg.settings.rayGenMode == RayGenMode::Emitter) {
            for (int forwardPass = 0; forwardPass < pkg.settings.numForwardPasses; forwardPass++) {
                pkg.settings.randomSeed = seedGen(); // new high-entropy seed each pass

                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait(); {
                    ScopedTimer timer("launchRayGenEmitterKernel");
                    launchRayGenEmitterKernel(pkg);
                }

                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();
                if (forwardPass == 0)
                {
                    ScopedTimer timer("launchDirectContributionKernel");
                    launchDirectContributionKernel(pkg, activeCount);
                }
                {
                    ScopedTimer forwardTimer("Traced forward pass", spdlog::level::debug);

                    for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                        pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                        pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                        pkg.queue.wait(); {
                            ScopedTimer timer("launchIntersectKernel");
                            launchIntersectKernel(pkg, activeCount);
                        } {
                            ScopedTimer timer("launchContributionKernel");
                            launchContributionKernel(pkg, activeCount);
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
            uint32_t photonMapCount = 0;
            pkg.queue.memcpy(&photonMapCount, pkg.intermediates.map.photonCountDevicePtr, sizeof(uint32_t)).wait(); {
                ScopedTimer timer("clearGridHeads");
                clearGridHeads(pkg.queue, pkg.intermediates.map);
            } {
                ScopedTimer timer("buildPhotonGridLinkedLists", spdlog::level::debug);
                size_t photonCount = std::min(photonMapCount, pkg.intermediates.map.photonCapacity);
                buildPhotonGridLinkedLists(pkg.queue, pkg.intermediates.map, photonCount);
            }

            // Save photon map to disk:
            {


                ScopedTimer timer("dumpPhotonMapToPLY");
                dumpPhotonMapToPLY(pkg.queue,
                                  pkg.intermediates.map.photons,
                                  photonMapCount,
                                  std::filesystem::path("Output/photon_map.ply"));

            } {
                ScopedTimer timer("launchCameraGatherKernel", spdlog::level::debug);
                int cameraGatherSPP = pkg.settings.numForwardPasses;
                std::mt19937_64 seedGen(pkg.settings.randomSeed); // define once before the loop
                pkg.settings.randomSeed = seedGen(); // new high-entropy seed each pass
                launchCameraGatherKernel(pkg, cameraGatherSPP); // generate image from photon map

            }
        } else if (pkg.settings.rayGenMode == RayGenMode::Adjoint) {
            pkg.queue
                    .fill(pkg.sensor.framebuffer, sycl::float4{0, 0, 0, 0}, pkg.sensor.height * pkg.sensor.width)
                    .wait();
            pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();

            int samplesPerPixel = pkg.settings.adjointSamplesPerPixel;
            for (int spp = 0; spp < samplesPerPixel; ++spp) {
                pkg.settings.randomSeed = pkg.settings.randomSeed * spp;
                ScopedTimer forwardTimer("Traced adjoint pass", spdlog::level::debug);

                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait(); {
                    ScopedTimer timer("launchRayGenAdjointKernel");
                    launchRayGenAdjointKernel(pkg, spp);
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
                        launchAdjointKernel(pkg, activeCount);
                    } {
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
        }
    }
}
