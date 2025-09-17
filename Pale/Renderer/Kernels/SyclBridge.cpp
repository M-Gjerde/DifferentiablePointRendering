// SyclWarmup.cpp (no imports of your modules)
#include <sycl/sycl.hpp>
#include "Renderer/Kernels/SyclBridge.h"
#include "Renderer/Kernels/AdjointKernels.h"

#include "Core/ScopedTimer.h"

#include "Renderer/GPUDataStructures.h"

#include "Renderer/Kernels/IntersectionKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"
#include "Renderer/Kernels/PrimalKernels.h"

namespace Pale {
    // ---- Orchestrator -------------------------------------------------------
    void submitKernel(RenderPackage &pkg) {
        pkg.queue.fill(pkg.sensor.framebuffer, sycl::float4{0, 0, 0, 0}, pkg.sensor.height * pkg.sensor.width).wait();

        if (pkg.settings.rayGenMode == RayGenMode::Emitter) {
            pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait(); {
                ScopedTimer timer("launchRayGenEmitterKernel");
                launchRayGenEmitterKernel(pkg);
            }

            uint32_t activeCount = 0;
            pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait(); {
                ScopedTimer timer("launchDirectContributionKernel");
                launchDirectContributionKernel(pkg, activeCount);
            }

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

            uint32_t photonMapCount = 0;
            pkg.queue.memcpy(&photonMapCount, pkg.intermediates.map.photonCountDevicePtr, sizeof(uint32_t)).wait(); {
                ScopedTimer timer("clearGridHeads");
                clearGridHeads(pkg.queue, pkg.intermediates.map);
            } {
                ScopedTimer timer("buildPhotonGridLinkedLists");
                buildPhotonGridLinkedLists(pkg.queue, pkg.intermediates.map, photonMapCount);
            }

            //launchCameraGatherKernel(pkg); // generate image from photon map
        } else if (pkg.settings.rayGenMode == RayGenMode::Adjoint) {
            pkg.queue
                    .fill(pkg.sensor.framebuffer, sycl::float4{0, 0, 0, 0}, pkg.sensor.height * pkg.sensor.width)
                    .wait();
            pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();

            int samplesPerPixel = pkg.settings.adjointSamplesPerPixel;
            for (int spp = 0; spp < samplesPerPixel; ++spp) {
                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait(); {
                    ScopedTimer timer("launchRayGenAdjointKernel");
                    launchRayGenAdjointKernel(pkg);
                }

                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();

                for (uint32_t bounce = 0; bounce < pkg.settings.maxAdjointBounces && activeCount > 0; ++bounce) {
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
}
