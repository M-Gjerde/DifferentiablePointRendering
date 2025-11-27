#pragma once

#include "Renderer/GPUDataStructures.h"

namespace Pale {

    struct SensorGPU {
        CameraGPU camera; // camera parameters
        float4 *framebuffer = nullptr; // device pointer
        uint32_t width{}, height{};

        float gammaCorrection = 2.2f;
        float exposureCorrection = 2.0f;
        float* ldrFramebuffer = nullptr;
        sycl::uchar4* outputFramebuffer = nullptr;
    };

    // GPU Struct
    struct PointGradients {
        float3 *gradPosition = nullptr;
        float3 *gradTanU = nullptr;
        float3 *gradTanV = nullptr;
        float2 *gradScale = nullptr;
        float3 *gradColor = nullptr;
        float  *gradOpacity = nullptr;
        float  *gradBeta = nullptr;
        float  *gradShape = nullptr;
        size_t numPoints{0};

        // debug
        float4 *framebuffer_pos = nullptr; // gradient image
        float4 *framebuffer_rot = nullptr; // gradient image
        float4 *framebuffer_scale = nullptr; // gradient image
        float4 *framebuffer_opacity = nullptr; // gradient image
        float4 *framebuffer_albedo = nullptr; // gradient image
        float4 *framebuffer_beta = nullptr; // gradient image

    };

    struct AdjointGPU {
        float4 *framebuffer = nullptr; // input adjoint image
        float4 *framebufferGrad = nullptr; // ouput gradient image
        uint32_t width{}, height{};
        float3 *gradient_pk = nullptr;
    };


    struct RenderPackage {
        sycl::queue queue;
        PathTracerSettings settings{};
        GPUSceneBuffers scene{};
        RenderIntermediatesGPU intermediates{};
        SensorGPU sensor{};
        PointGradients gradients{};
    };

}
