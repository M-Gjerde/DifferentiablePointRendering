//
// Created by magnus on 9/8/25.
//
#pragma once

#include "Renderer/GPUDataStructures.h"


namespace Pale {
    void launchRayGenAdjointKernel(sycl::queue queue,
                                   PathTracerSettings settings,
                                   GPUSceneBuffers scene,
                                   RenderIntermediatesGPU renderIntermediates);

}

