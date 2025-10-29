//
// Created by magnus on 9/8/25.
//
#pragma once

#include "Renderer/GPUDataStructures.h"
#include "Renderer/RenderPackage.h"


namespace Pale {

    void launchRayGenAdjointKernel(RenderPackage & pkg, int spp);

    void launchAdjointKernel(RenderPackage &pkg, uint32_t activeRayCount);

    void generateNextAdjointRays(RenderPackage &pkg, uint32_t activeRayCount);

}

