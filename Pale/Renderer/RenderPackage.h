#pragma once

#include "Renderer/GPUDataStructures.h"

namespace Pale {

    struct SensorGPU {
        CameraGPU camera; // camera parameters
        float4 *framebuffer{nullptr}; // device pointer
        uint32_t width{}, height{};
    };

    struct AdjointGPU {
        float4 *framebuffer{nullptr}; // input adjoint image
        float4 *framebufferGrad{nullptr}; // ouput gradient image
        uint32_t width{}, height{};
        float3 *gradient_pk{nullptr};
    };

    struct RenderPackage {
        sycl::queue queue;
        PathTracerSettings settings{};
        GPUSceneBuffers scene{};
        RenderIntermediatesGPU intermediates{};
        SensorGPU sensor{};
    };

}
