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

    void launchPointSampledPathTracingCameraKernel(    RenderPackage &pkg,    uint32_t cameraIndex,    uint32_t sampleIndex);

    static void launchContributionKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex){};
    static void launchContributionEmitterVisibleKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex){};

    void launchCameraGatherKernel(RenderPackage &pkg, uint32_t cameraIndex, uint32_t gatherPass);

    void generateNextRays(RenderPackage &pkg, uint32_t activeRayCount);
} // Pale
