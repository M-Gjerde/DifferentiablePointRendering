//
// Created by magnus on 11/21/25.
//

#pragma once
#include "Renderer/RenderPackage.h"

namespace Pale {

    void launchPostProcessKernel(RenderPackage& pkg);
    void accumulatePhotonEnergyPerSurfelDebug(RenderPackage& pkg);

}
