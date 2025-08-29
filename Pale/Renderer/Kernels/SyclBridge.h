// SyclWarmup.h (bridge, no modules)
#pragma once
#include <cstddef>
#include <sycl/sycl.hpp>
#include <glm/glm.hpp>

#include "Renderer/GPUDataStructures.h"


namespace Pale {
    void warmup_kernel_submit(void* queue_ptr, std::size_t n);
    void submitKernel(sycl::queue& device_queue,  GPUSceneBuffers scene, SensorGPU sensor);

}

