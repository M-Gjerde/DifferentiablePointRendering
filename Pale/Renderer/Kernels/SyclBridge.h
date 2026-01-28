// SyclWarmup.h (bridge, no modules)
#pragma once
#include <glm/glm.hpp>

#include "Renderer/GPUDataStructures.h"
#include "Renderer/RenderPackage.h"


namespace Pale {
    void warmup_kernel_submit(void* queue_ptr, std::size_t n);
    void submitPhotonMappingKernel(RenderPackage& renderPackage);
    void submitLightTracingKernel(RenderPackage& renderPackage);
    void submitLightTracingKernelCylinderRay(RenderPackage& renderPackage);
    void submitAdjointKernel(RenderPackage& renderPackage);

}

