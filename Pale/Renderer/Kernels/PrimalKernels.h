//
// Created by magnus on 9/12/25.
//

#pragma once

#include "Renderer/GPUDataStructures.h"
#include "Renderer/RenderPackage.h"


namespace Pale {
    void launchRayGenEmitterKernel(RenderPackage& pkg);

    void launchIntersectKernel(RenderPackage &pkg, uint32_t activeRayCount);

    void computePhotonCellIdsAndPermutation(sycl::queue &q, DeviceSurfacePhotonMapGrid g, uint32_t photonCount);

    void buildPhotonCellRangesAndOrdering(sycl::queue &q, DeviceSurfacePhotonMapGrid g, uint32_t photonCount);

    void clearGridHeads(sycl::queue &q, DeviceSurfacePhotonMapGrid &g);

    void launchVolumeKernel(RenderPackage &pkg, uint32_t activeRayCount);

    void launchDirectContributionKernel(RenderPackage &pkg, uint32_t activeRayCount);

    void launchContributionKernel(RenderPackage &pkg, uint32_t activeRayCount);

    void launchCameraGatherKernel(RenderPackage &pkg, int sppTotal, uint32_t cameraIndex);

    void generateNextRays(RenderPackage &pkg, uint32_t activeRayCount);
} // Pale
