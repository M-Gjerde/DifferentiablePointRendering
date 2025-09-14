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

        switch (pkg.settings.rayGenMode) {
            case RayGenMode::Emitter:
                launchRayGenEmitterKernel(pkg);
                break;
            case RayGenMode::Adjoint:
                launchRayGenAdjointKernel(pkg.queue, pkg.settings, pkg.sensor, pkg.adjoint, pkg.scene,
                                          pkg.intermediates);
                break;
            default:
                ;
        }
        uint32_t activeCount = 0;
        pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();

        if (pkg.settings.rayGenMode == RayGenMode::Emitter) {
            //launchDirectShadeKernel(pkg.queue, pkg.scene, pkg.sensor, pkg.intermediates.primaryRays, activeCount, pkg.settings);
            for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                pkg.queue.wait();
                launchIntersectKernel(pkg, activeCount);
                //launchVolumeKernel(pkg, activeCount);

                //launchContributionKernel(pkg, activeCount);

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

        uint32_t photonMapCount = 0;
        pkg.queue.memcpy(&photonMapCount, pkg.intermediates.map.photonCountDevicePtr, sizeof(uint32_t)).wait();
        clearGridHeads(pkg.queue, pkg.intermediates.map);
        buildPhotonGridLinkedLists(pkg.queue, pkg.intermediates.map, photonMapCount);

        launchCameraGatherKernel(pkg);


        /*
        else if (pkg.settings.rayGenMode == RayGenMode::Adjoint) {
            //for (uint32_t bounce = 0; bounce < 1  && activeCount > 0; ++bounce) {
            for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                pkg.queue.wait();
                launchIntersectKernel(pkg, activeCount);

                launchAdjointShadeKernel(pkg.queue, pkg.scene, pkg.sensor, pkg.adjoint, pkg.intermediates.hitRecords,
                                         pkg.intermediates.primaryRays, activeCount,
                                         pkg.intermediates.extensionRaysA, pkg.intermediates, pkg.settings);

                uint32_t nextCount = 0;
                pkg.queue.memcpy(&nextCount, pkg.intermediates.countExtensionOut, sizeof(uint32_t)).wait();
                pkg.queue.memcpy(pkg.intermediates.primaryRays, pkg.intermediates.extensionRaysA,
                                 nextCount * sizeof(RayState));
                pkg.queue.wait();
                activeCount = nextCount;
                pkg.queue.wait();
            }
            // Launch intersect kernel

            // Launch shade kernel//
        }
        */
    }
}
