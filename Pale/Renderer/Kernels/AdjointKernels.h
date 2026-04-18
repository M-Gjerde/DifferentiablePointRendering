//
// Created by magnus on 9/8/25.
//
#pragma once

#include "Renderer/GPUDataStructures.h"
#include "Renderer/RenderPackage.h"


namespace Pale {
    void launchRayGenAdjointKernel(RenderPackage &pkg, int spp, uint32_t cameraIndex);

    void launchAdjointProjectionKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex);

    void launchAdjointDirectLightKernel(
        RenderPackage& pkg,
        uint32_t spp,
        uint32_t activeQueryCount,
        uint32_t cameraIndex);

    void adjointContributionKernels(
        RenderPackage& pkg,
        uint32_t measurementEventCount,
        uint32_t measurementTwoPointCount,
        uint32_t cameraAttachedBridgeEventCount,
        uint32_t recursiveBridgeEventCount,
        uint32_t directLightEventCount,
        uint32_t cameraIndex);

    void generateNextAdjointRays(RenderPackage &pkg, uint32_t activeRayCount);

    void launchAdjointIntersectKernel(RenderPackage &pkg, uint32_t spp, uint32_t activeRayCount, uint32_t cameraIndex);
}
