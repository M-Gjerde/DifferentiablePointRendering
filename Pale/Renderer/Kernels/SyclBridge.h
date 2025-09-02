// SyclWarmup.h (bridge, no modules)
#pragma once
#include <glm/glm.hpp>

#include "Renderer/GPUDataStructures.h"


namespace Pale {
    void warmup_kernel_submit(void* queue_ptr, std::size_t n);
    void submitKernel(RenderPackage& renderPackage);

}

