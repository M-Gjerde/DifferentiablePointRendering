//
// Created by magnus on 9/12/25.
//

#pragma once

#include "Renderer/GPUDataStructures.h"
#include "Renderer/RenderPackage.h"


namespace Pale {
    void launchRayGenEmitterKernel(RenderPackage& pkg, uint32_t forwardPass);

    void launchIntersectKernel(RenderPackage &pkg, uint32_t activeRayCount);

    void computePhotonCellIdsAndPermutation(sycl::queue &q, DeviceSurfacePhotonMapGrid g, uint32_t photonCount);

    void buildPhotonCellRangesAndOrdering(sycl::queue &q, DeviceSurfacePhotonMapGrid g, uint32_t photonCount);

    void clearGridHeads(sycl::queue &q, DeviceSurfacePhotonMapGrid &g);

    void launchVolumeKernel(RenderPackage &pkg, uint32_t activeRayCount);

    void launchDirectContributionKernel(RenderPackage &pkg, uint32_t activeRayCount);

    static void launchContributionKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex){};
    static void launchContributionEmitterVisibleKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex){};

    void launchCameraGatherKernel(RenderPackage &pkg, uint32_t cameraIndex);

    void generateNextRays(RenderPackage &pkg, uint32_t activeRayCount);
} // Pale
