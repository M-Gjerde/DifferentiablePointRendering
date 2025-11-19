#pragma once

#include "Renderer/GPUDataStructures.h"

namespace Pale {

    struct SensorGPU {
        CameraGPU camera; // camera parameters
        float4 *framebuffer{nullptr}; // device pointer
        uint32_t width{}, height{};
    };


    /*
    struct alignas(16) Point {
    float3 position{0.0f};
    float3 tanU{0.0f};
    float3 tanV{0.0f};
    float2 scale{0.0f};
    float3 color{0.0f};
    float opacity{0.0f};
    float beta{0.0f};
    float shape{0.0f};
    };
    */

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
        PointGradients gradients{};
    };

}
