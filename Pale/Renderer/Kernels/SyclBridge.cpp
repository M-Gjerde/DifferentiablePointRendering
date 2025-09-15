// SyclWarmup.cpp (no imports of your modules)
#include <sycl/sycl.hpp>
#include "Renderer/Kernels/SyclBridge.h"
#include "Renderer/Kernels/AdjointKernels.h"

#include "Renderer/GPUDataStructures.h"

#include "Renderer/Kernels/IntersectionKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"
#include "Renderer/Kernels/PrimalKernels.h"

namespace Pale {
    // ---- Orchestrator -------------------------------------------------------
    void submitKernel(RenderPackage &pkg) {
        pkg.queue.fill(pkg.sensor.framebuffer, sycl::float4{0, 0, 0, 0}, pkg.sensor.height * pkg.sensor.width).wait();
        // Ray generation mode
        pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();


        if (pkg.settings.rayGenMode == RayGenMode::Emitter) {
            launchRayGenEmitterKernel(pkg);

            uint32_t activeCount = 0;
            pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();

            launchDirectContributionKernel(pkg, activeCount);

            for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                pkg.queue.wait();
                launchIntersectKernel(pkg, activeCount);
                launchContributionKernel(pkg, activeCount);
                generateNextRays(pkg, activeCount);
                uint32_t nextCount = 0;
                pkg.queue.memcpy(&nextCount, pkg.intermediates.countExtensionOut, sizeof(uint32_t)).wait();
                pkg.queue.memcpy(pkg.intermediates.primaryRays, pkg.intermediates.extensionRaysA,
                                 nextCount * sizeof(RayState));
                pkg.queue.wait();
                activeCount = nextCount;
                pkg.queue.wait();
            }
            uint32_t photonMapCount = 0;
            pkg.queue.memcpy(&photonMapCount, pkg.intermediates.map.photonCountDevicePtr, sizeof(uint32_t)).wait();
            clearGridHeads(pkg.queue, pkg.intermediates.map);
            buildPhotonGridLinkedLists(pkg.queue, pkg.intermediates.map, photonMapCount);

            //launchCameraGatherKernel(pkg); // generate image from photon map
        } else if (pkg.settings.rayGenMode == RayGenMode::Adjoint) {
            int samplesPerPixel = pkg.settings.adjointSamplesPerPixel;
            for (int spp = 0; spp < samplesPerPixel; ++spp) {
                pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();

                launchRayGenAdjointKernel(pkg.queue, pkg.settings, pkg.sensor, pkg.adjoint, pkg.scene,
                                          pkg.intermediates);
                uint32_t activeCount = 0;
                pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();

                for (uint32_t bounce = 0; bounce < pkg.settings.maxAdjointBounces && activeCount > 0; ++bounce) {
                    pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                    pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                    pkg.queue.wait();
                    launchIntersectKernel(pkg, activeCount);

                    launchAdjointKernel(pkg, activeCount);
                    launchAdjointContributionKernel(pkg, activeCount);

                    generateNextRays(pkg, activeCount);

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
