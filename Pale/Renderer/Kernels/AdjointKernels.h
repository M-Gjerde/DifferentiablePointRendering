//
// Created by magnus on 9/8/25.
//
#pragma once

#include "Renderer/GPUDataStructures.h"
#include "Renderer/RenderPackage.h"


namespace Pale {

    void launchRayGenAdjointKernel(RenderPackage & pkg, int spp, uint32_t cameraIndex);

    void launchAdjointProjectionKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex);

    void launchAdjointTransportKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex);

    void generateNextAdjointRays(RenderPackage &pkg, uint32_t activeRayCount);

}

