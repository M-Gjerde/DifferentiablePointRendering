//
// Created by magnus-desktop on 8/28/25.
//
module;
#include <sycl/sycl.hpp>
#include <Renderer/GPUDataStructures.h>
#include <Renderer/RenderPackage.h>

export module Pale.Render.Sensors;

import Pale.Render.SceneBuild;
import Pale.Log;

export namespace Pale {
    SensorGPU
    makeSensorsForScene(sycl::queue queue, const SceneBuild::BuildProducts &buildProducts,
                        bool initializeData = false) {
        SensorGPU out{};
        if (buildProducts.cameraCount() == 0) {
            return out;
        }

        const auto &cam = buildProducts.cameras().front();
        const size_t pixelCount = static_cast<size_t>(cam.width) * static_cast<size_t>(cam.height);
        float4 *dev = reinterpret_cast<float4 *>(sycl::malloc_device(pixelCount * sizeof(float4), queue));
        sycl::uchar4 *dev2 = reinterpret_cast<sycl::uchar4 *>(sycl::malloc_device(
            pixelCount * sizeof(sycl::uchar4), queue));

        float *dev3 = reinterpret_cast<float *>(sycl::malloc_device(pixelCount * sizeof(float) * 3, queue));

        queue.wait();

        //queue.fill(dev, float4{0, 0, 0, 0}, pixelCount).wait();

        if (initializeData) {
            queue.memset(dev, 1e6f, pixelCount * sizeof(float4)).wait();
            queue.memset(dev2, 0.0f, pixelCount * sizeof(sycl::uchar4)).wait();
            queue.memset(dev3, 0, pixelCount * 3u * sizeof(float)).wait();
        }

        out.camera = cam;
        out.framebuffer = dev;
        out.outputFramebuffer = dev2;
        out.ldrFramebuffer = dev3;
        out.width = cam.width;
        out.height = cam.height;
        return out;
    }


    PointGradients
    makeGradientsForScene(sycl::queue queue,
                          const SceneBuild::BuildProducts &buildProducts, bool allocateDebugImages = false) {
        PointGradients out{};

        const uint32_t numPoints =
                static_cast<uint32_t>(buildProducts.points.size());

        if (buildProducts.cameraCount() == 0) {
            Pale::Log::PA_WARN(
                "makeGradientsForScene: no cameras in buildProducts; "
                "gradient framebuffer will not be allocated."
            );
        }

        Pale::Log::PA_INFO(
            "makeGradientsForScene: allocating gradients for {} points",
            numPoints
        );

        out.numPoints = numPoints;

        if (numPoints > 0) {
            out.gradPosition =
                    static_cast<float3 *>(
                        sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradTanU =
                    static_cast<float3 *>(
                        sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradTanV =
                    static_cast<float3 *>(
                        sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradScale =
                    static_cast<float2 *>(
                        sycl::malloc_device(numPoints * sizeof(float2), queue));
            out.gradColor =
                    static_cast<float3 *>(
                        sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradOpacity =
                    static_cast<float *>(
                        sycl::malloc_device(numPoints * sizeof(float), queue));
            out.gradBeta =
                    static_cast<float *>(
                        sycl::malloc_device(numPoints * sizeof(float), queue));
            out.gradShape =
                    static_cast<float *>(
                        sycl::malloc_device(numPoints * sizeof(float), queue));
        }

        // Allocate adjoint framebuffer (same resolution as first camera)
        if (allocateDebugImages && buildProducts.cameraCount() > 0) {
            const auto &cam = buildProducts.cameras().front();
            const size_t pixelCount =
                    static_cast<size_t>(cam.width) *
                    static_cast<size_t>(cam.height);

            Pale::Log::PA_INFO(
                "makeGradientsForScene: allocating adjoint framebuffer for {}x{} ({} pixels)",
                cam.width, cam.height, pixelCount
            );

            out.framebuffer_pos =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float4), queue));
            out.framebuffer_rot =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float4), queue));
            out.framebuffer_scale =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float4), queue));
            out.framebuffer_opacity =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float4), queue));
            out.framebuffer_albedo =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float4), queue));

            out.framebuffer_beta =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float4), queue));


        } else {
            out.framebuffer_pos = nullptr;
            out.framebuffer_rot = nullptr;
            out.framebuffer_scale = nullptr;
            out.framebuffer_opacity = nullptr;
            out.framebuffer_albedo = nullptr;
            out.framebuffer_beta = nullptr;
        }

        queue.wait();
        return out;
    }

    inline void freeGradientsForScene(sycl::queue queue, PointGradients &g)
    {
        if (g.gradPosition)  { sycl::free(g.gradPosition, queue);  g.gradPosition  = nullptr; }
        if (g.gradTanU)      { sycl::free(g.gradTanU,     queue);  g.gradTanU      = nullptr; }
        if (g.gradTanV)      { sycl::free(g.gradTanV,     queue);  g.gradTanV      = nullptr; }
        if (g.gradScale)     { sycl::free(g.gradScale,    queue);  g.gradScale     = nullptr; }
        if (g.gradColor)     { sycl::free(g.gradColor,    queue);  g.gradColor     = nullptr; }
        if (g.gradOpacity)   { sycl::free(g.gradOpacity,  queue);  g.gradOpacity   = nullptr; }
        if (g.gradBeta)      { sycl::free(g.gradBeta,     queue);  g.gradBeta      = nullptr; }
        if (g.gradShape)     { sycl::free(g.gradShape,    queue);  g.gradShape     = nullptr; }
        if (g.framebuffer_pos)   { sycl::free(g.framebuffer_pos,  queue);  g.framebuffer_pos   = nullptr; }
        if (g.framebuffer_rot)   { sycl::free(g.framebuffer_rot,  queue);  g.framebuffer_rot   = nullptr; }
        if (g.framebuffer_scale)   { sycl::free(g.framebuffer_scale,  queue);  g.framebuffer_scale   = nullptr; }
        if (g.framebuffer_opacity)   { sycl::free(g.framebuffer_opacity,  queue);  g.framebuffer_opacity   = nullptr; }
        if (g.framebuffer_albedo)   { sycl::free(g.framebuffer_albedo,  queue);  g.framebuffer_albedo   = nullptr; }
        if (g.framebuffer_beta)   { sycl::free(g.framebuffer_beta,  queue);  g.framebuffer_beta   = nullptr; }

        g.numPoints = 0;
    }

    inline std::vector<float>
    downloadSensorRGBARaw(sycl::queue queue, const SensorGPU &sensorGpu) {
        // Total number of float elements = width * height * 4 (RGBA channels)
        const size_t totalFloatCount = static_cast<size_t>(sensorGpu.width)
                                       * static_cast<size_t>(sensorGpu.height)
                                       * 4u;
        std::vector<float> hostSideFramebuffer(totalFloatCount);


        // Allocate host-side buffer
        queue.wait();
        // Copy device framebuffer → host buffer
        queue.memcpy(
            hostSideFramebuffer.data(), // destination
            sensorGpu.framebuffer, // source (device pointer)
            totalFloatCount * sizeof(float) // size in bytes
        ).wait();

        return hostSideFramebuffer;
    }

    inline std::vector<float>
    downloadSensorLDR(sycl::queue queue, const SensorGPU &sensorGpu) {
        // Total number of float elements = width * height * 3 (RGB channels)
        const size_t totalFloatCount = static_cast<size_t>(sensorGpu.width)
                                       * static_cast<size_t>(sensorGpu.height)
                                       * 3u;
        std::vector<float> hostSideFramebuffer(totalFloatCount);


        // Allocate host-side buffer
        queue.wait();
        // Copy device framebuffer → host buffer
        queue.memcpy(
            hostSideFramebuffer.data(), // destination
            sensorGpu.ldrFramebuffer, // source (device pointer)
            totalFloatCount * sizeof(float) // size in bytes
        ).wait();

        return hostSideFramebuffer;
    }

    inline std::vector<uint8_t>
    downloadSensorRGBA(sycl::queue queue, const SensorGPU &sensorGpu) {
        // Total number of float elements = width * height * 4 (RGBA channels)
        const size_t totalFloatCount = static_cast<size_t>(sensorGpu.width)
                                       * static_cast<size_t>(sensorGpu.height)
                                       * 4u;
        std::vector<uint8_t> hostSideFramebuffer(totalFloatCount);


        // Allocate host-side buffer
        queue.wait();
        // Copy device framebuffer → host buffer
        queue.memcpy(
            hostSideFramebuffer.data(), // destination
            sensorGpu.outputFramebuffer, // source (device pointer)
            totalFloatCount * sizeof(uint8_t) // size in bytes
        ).wait();

        return hostSideFramebuffer;
    }

struct DebugGradientImages {
    // Each buffer has size: width * height * 4 (RGBA)
    std::vector<float> position;  // framebuffer_pos
    std::vector<float> rotation;  // framebuffer_rot
    std::vector<float> scale;     // framebuffer_scale
    std::vector<float> opacity;   // framebuffer_opacity
    std::vector<float> albedo;    // framebuffer_albedo
    std::vector<float> beta;    // framebuffer_albedo
};

inline DebugGradientImages downloadDebugGradientImages(
    sycl::queue queue,
    const SensorGPU &sensorGpu,
    const PointGradients &gradients
) {
    const std::size_t pixelCount = static_cast<std::size_t>(sensorGpu.width)
                                   * static_cast<std::size_t>(sensorGpu.height);
    const std::size_t totalFloatCount = pixelCount * 4u; // RGBA

    DebugGradientImages images;
    images.position.resize(totalFloatCount);
    images.rotation.resize(totalFloatCount);
    images.scale.resize(totalFloatCount);
    images.opacity.resize(totalFloatCount);
    images.albedo.resize(totalFloatCount);
    images.beta.resize(totalFloatCount);

    auto copyBuffer = [&] (std::vector<float> &hostBuffer, const float4 *deviceBuffer) {
        queue.memcpy(
            hostBuffer.data(),       // destination (host)
            deviceBuffer,            // source (device)
            totalFloatCount * sizeof(float)
        );
    };

    copyBuffer(images.position, gradients.framebuffer_pos);
    copyBuffer(images.rotation, gradients.framebuffer_rot);
    copyBuffer(images.scale,    gradients.framebuffer_scale);
    copyBuffer(images.opacity,  gradients.framebuffer_opacity);
    copyBuffer(images.albedo,   gradients.framebuffer_albedo);
    copyBuffer(images.beta,   gradients.framebuffer_beta);

    // Ensure all copies are completed before returning
    queue.wait();

    return images;
}


    inline std::vector<float>
    uploadSensorRGBA(sycl::queue queue, const SensorGPU &sensorGpu, std::vector<float> hostSideFramebuffer) {
        // Allocate host-side buffer
        queue.wait();
        // Copy device framebuffer → host buffer
        queue.memcpy(
            sensorGpu.framebuffer, // destination
            hostSideFramebuffer.data(), // source (device pointer)
            hostSideFramebuffer.size() * sizeof(float) // size in bytes
        ).wait();

        return hostSideFramebuffer;
    }
}
