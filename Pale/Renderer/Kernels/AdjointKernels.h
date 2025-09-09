//
// Created by magnus on 9/8/25.
//
#pragma once

#include "Renderer/GPUDataStructures.h"


namespace Pale {
    void launchRayGenAdjointKernel(sycl::queue queue,
                                   PathTracerSettings settings,
                                   SensorGPU cameraSensor,
                                   AdjointGPU adjointSensor,
                                   GPUSceneBuffers scene,
                                   RenderIntermediatesGPU renderIntermediates);

    void launchAdjointShadeKernel(sycl::queue &queue,
                                  GPUSceneBuffers scene,
                                  SensorGPU sensor,
                                  AdjointGPU adjointSensor,
                                  const WorldHit *hitRecords,
                                  const RayState *raysIn,
                                  uint32_t rayCount,
                                  RayState *raysOut,
                                  RenderIntermediatesGPU renderIntermediates,
                                  const PathTracerSettings &settings

    );
}

