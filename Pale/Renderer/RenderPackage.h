#pragma once

#include "Renderer/GPUDataStructures.h"

namespace Pale {


    struct SensorGPU {
        CameraGPU camera; // camera parameters
        float4 *framebuffer = nullptr; // RAW framebuffer
        uint32_t width{}, height{};

        float gammaCorrection = 1.0f;
        float exposureCorrection = 1.0f;
        float* ldrFramebuffer = nullptr; // Low Dynamic Range framebuffer
        sycl::uchar4* outputFramebuffer = nullptr; // uint8 converted framebuffer
        char name[16];

        // Depth distortion buffer
        float* depthDistortionBuffer = nullptr;
        float* depthDistortionAdjointBuffer = nullptr;

        float*  medianDepthBuffer;        // scalar visualization depth
        float4* medianWorldPositionBuffer;        // xyz = world-space median point, w = valid
        float4* visibleNormalBuffer;        // xyz = world-space median point, w = valid
        float4* normalFromDepthBuffer;       // xyz = normal, w = valid

        float * medianDepthAdjointBuffer;
        float4 * visibleNormalAdjointBuffer;
        float4 * normalFromDepthAdjointBuffer;
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

    struct Storage {
        float *distortionBuffer = nullptr;
        float pixelCount;
        char name[16];
    };

    struct DebugImages {
        // debug
        float *framebufferPosX = nullptr; // gradient image
        float *framebufferPosY = nullptr; // gradient image
        float *framebufferPosZ = nullptr; // gradient image
        float *framebufferRot = nullptr; // gradient image
        float *framebufferScale = nullptr; // gradient image
        float *framebufferOpacity = nullptr; // gradient image
        float *framebufferAlbedo = nullptr; // gradient image
        float *framebufferBeta = nullptr; // gradient image

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
