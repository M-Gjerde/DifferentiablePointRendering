#pragma once

#include "Renderer/GPUDataStructures.h"

namespace Pale {


    struct SensorGPU {
        CameraGPU camera; // camera parameters
        float4 *framebuffer = nullptr; // RAW framebuffer
        uint32_t width{}, height{};
        uint32_t cameraSlotIndex; // original scene.xml camera index

        float gammaCorrection = 1.0f;
        float exposureCorrection = 1.0f;
        bool useSrgbEncoding = true;
        float* ldrFramebuffer = nullptr; // Low Dynamic Range framebuffer
        sycl::uchar4* outputFramebuffer = nullptr; // uint8 converted framebuffer
        char name[16];

        // Depth distortion buffer
        float* depthDistortionBuffer = nullptr;
        float* depthDistortionAdjointBuffer = nullptr;
        float* visibilityWeightedOpacityBuffer = nullptr;
        float* intraSlabDepthBuffer = nullptr;
        float* intraSlabDepthAdjointBuffer = nullptr;
        uint32_t* intraSlabDepthActiveSlabCountBuffer = nullptr;
        float* curvatureScaleBuffer = nullptr;
        float* curvatureScaleAdjointBuffer = nullptr;
        uint32_t* curvatureScaleActiveSlabCountBuffer = nullptr;
        // Optional debug output: dominant primitive from the exact visible slab
        // selected by the curvature-scale regularizer, or UINT32_MAX.
        uint32_t* curvaturePrimitiveIndexBuffer = nullptr;

        float*  medianDepthBuffer;        // scalar visualization depth
        float*  meanDepthBuffer;        // scalar visualization depth
        float4* medianWorldPositionBuffer;        // xyz = world-space median point, w = valid
        float4* visibleNormalBuffer;        // xyz = world-space median point, w = valid
        float4* normalFromDepthBuffer;       // xyz = normal, w = valid

        float * medianDepthAdjointBuffer;
        float4 * visibleNormalAdjointBuffer;
        float4 * normalFromDepthAdjointBuffer;
    };

    // GPU Struct
    // WE use SoA for actual optimizer gradients since  pytorch likely consumes each parameter gradient as a contiguous array
    struct PointGradients {
        float3 *gradPosition = nullptr;
        float3 *cloneSignal = nullptr;
        float3 *gradRotation = nullptr;
        float2 *gradScale = nullptr;
        float3 *gradAlbedo = nullptr;
        float *gradOpacity = nullptr;
        float *gradBeta = nullptr;
        float *gradShape = nullptr;

        // Per-primitive/per-camera translation accumulation.
        float3 *gradPositionPerPrimitivePerCamera = nullptr;
        uint32_t *gradPositionRecordCountPerPrimitivePerCamera = nullptr;
        float3 *cloneSignalPerPrimitivePerCamera = nullptr;
        uint32_t *cloneSignalRecordCountPerPrimitivePerCamera = nullptr;

        // Final per-primitive clone-signal stats.
        float *cloneSignalMeanNorm = nullptr;
        float *cloneSignalStd = nullptr;
        float *cloneSignalCoherence = nullptr;
        float *cloneSignalDisagreement = nullptr;
        uint32_t *cloneSignalActiveCameraCount = nullptr;

        size_t numPoints{0};
        size_t cameraSlotCount{0};
    };

    // Forward-only structural statistics used by Python densification. These
    // are not adjoints and are reset once at the start of every forward render.
    struct CurvatureDensificationStats {
        float *violationSum = nullptr;
        uint32_t *violationCount = nullptr;
        float *directionTensorUu = nullptr;
        float *directionTensorUv = nullptr;
        float *directionTensorVv = nullptr;
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
        float *framebufferRotX = nullptr;
        float *framebufferRotY = nullptr;
        float *framebufferRotZ = nullptr;
        float *framebufferScaleU = nullptr; // gradient image
        float *framebufferScaleV = nullptr; // gradient image
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
        PointGradients gradients{}; // photometric
        PointGradients depthDistortionGradients{};
        PointGradients normalConsistencyGradients{};
        PointGradients visibilityOpacityGradients{};
        PointGradients intraSlabDepthGradients{};
        PointGradients curvatureScaleGradients{};
        CurvatureDensificationStats curvatureDensificationStats{};
        std::vector<SensorGPU> sensors{};
        DebugImages* debugImages{};
        uint32_t numSensors{};

    };

}
