//
// Created by magnus on 9/8/25.
//
#pragma once

#include "Renderer/GPUDataStructures.h"
#include "Renderer/RenderPackage.h"


namespace Pale {
    void launchRayGenAdjointKernel(RenderPackage &pkg, int spp, uint32_t cameraIndex);

    void adjointContributionKernels(
        RenderPackage& pkg,
        uint32_t measurementEventCount,
        uint32_t measurementTwoPointCount,
        uint32_t materialMaterialEventCount,
        uint32_t materialEndEdgeEventCount,
        uint32_t materialStartEdgeEventCount,
        uint32_t cameraIndex);

    void computePerPrimitiveCloneSignalStats(RenderPackage &pkg);

    void reduceFusedFirstBounceMeasurementGradientRecords(RenderPackage &pkg, uint32_t cameraIndex);

    void launchDepthDistortionBackwardKernel(RenderPackage& pkg, uint32_t cameraIndex);
    void launchNormalConsistencyBackwardKernel(RenderPackage& pkg, uint32_t cameraIndex);
    void launchNormalFromDepthAdjointKernel(RenderPackage& pkg, uint32_t cameraIndex);
    void launchAdjointIntersectKernel(RenderPackage &pkg, uint32_t spp, uint32_t activeRayCount, uint32_t cameraIndex);
    void launchSurfaceRegularizersBackwardKernel(RenderPackage &pkg, uint32_t cameraIndex) ;
}
