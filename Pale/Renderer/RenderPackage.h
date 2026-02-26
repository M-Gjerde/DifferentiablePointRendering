#pragma once

#include "Renderer/GPUDataStructures.h"

namespace Pale {


    struct SensorGPU {
        CameraGPU camera; // camera parameters
        float4 *framebuffer = nullptr; // RAW framebuffer
        uint32_t width{}, height{};

        float gammaCorrection = 2.2f;
        float exposureCorrection = 1.0f;
        float* ldrFramebuffer = nullptr; // Low Dynamic Range framebuffer
        sycl::uchar4* outputFramebuffer = nullptr; // uint8 converted framebuffer
        char name[16];
    };

    // GPU Struct
    struct PointGradients {
        float3 *gradPosition = nullptr;
        float3 *gradTanU = nullptr;
        float3 *gradTanV = nullptr;
        float2 *gradScale = nullptr;
        float3 *gradAlbedo = nullptr;
        float  *gradOpacity = nullptr;
        float  *gradBeta = nullptr;
        float  *gradShape = nullptr;
        size_t numPoints{0};
    };

    struct DebugImages {
        // debug
        float4 *framebufferPosX = nullptr; // gradient image
        float4 *framebufferPosY = nullptr; // gradient image
        float4 *framebufferPosZ = nullptr; // gradient image
        float4 *framebufferRot = nullptr; // gradient image
        float4 *framebufferScale = nullptr; // gradient image
        float4 *framebufferOpacity = nullptr; // gradient image
        float4 *framebufferAlbedo = nullptr; // gradient image
        float4 *framebufferBeta = nullptr; // gradient image

        float4 * framebufferDepthLoss = nullptr;
        float4 * framebufferNormalLoss = nullptr;
        float4 * framebufferDepthLossPos = nullptr;
        uint32_t numPixels = 0;
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
        Random random;
        GPUSceneBuffers scene{};
        RenderIntermediatesGPU intermediates{};
        PointGradients gradients{};
        std::vector<SensorGPU> sensors{};
        DebugImages* debugImages{};
        uint32_t numSensors{};

    };

}
