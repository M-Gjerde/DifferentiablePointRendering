//
// Created by magnus on 9/12/25.
//

#pragma once

#include "Renderer/GPUDataStructures.h"
namespace Pale {

    void launchIntersectKernel(RenderPackage &pkg, uint32_t activeRayCount);

    void launchVolumeKernel(RenderPackage &pkg, uint32_t activeRayCount);

    void launchContributionKernel(RenderPackage &pkg, uint32_t activeRayCount);


    void generateNextRays(RenderPackage &pkg, uint32_t activeRayCount);

} // Pale
