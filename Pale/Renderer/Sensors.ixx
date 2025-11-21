//
// Created by magnus-desktop on 8/28/25.
//
module;
#include <sycl/sycl.hpp>
#include <Renderer/GPUDataStructures.h>
#include <Renderer/RenderPackage.h>

export module Pale.Render.Sensors;

import Pale.Render.SceneBuild;

export namespace Pale {
    SensorGPU
    makeSensorsForScene(sycl::queue queue, const SceneBuild::BuildProducts &buildProducts, bool initializeData = false) {
        SensorGPU out{};
        if (buildProducts.cameraCount() == 0) {
            return out;
        }

        const auto &cam = buildProducts.cameras().front();
        const size_t pixelCount = static_cast<size_t>(cam.width) * static_cast<size_t>(cam.height);
        float4 *dev = reinterpret_cast<float4 *>(sycl::malloc_device(pixelCount * sizeof(float4), queue));
        sycl::uchar4 *dev2 = reinterpret_cast<sycl::uchar4 *>(sycl::malloc_device(pixelCount * sizeof(sycl::uchar4), queue));

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
    makeGradientsForScene(sycl::queue queue, const SceneBuild::BuildProducts &buildProducts) {
        PointGradients out{};


        uint32_t numPoints = buildProducts.points.size();
        out.gradPosition = static_cast<float3 *>(sycl::malloc_device(numPoints * sizeof(float3), queue));
        out.gradTanU = static_cast<float3 *>(sycl::malloc_device(numPoints * sizeof(float3), queue));
        out.gradTanV = static_cast<float3 *>(sycl::malloc_device(numPoints * sizeof(float3), queue));
        out.gradScale = static_cast<float2 *>(sycl::malloc_device(numPoints * sizeof(float2), queue));
        out.gradColor = static_cast<float3 *>(sycl::malloc_device(numPoints * sizeof(float3), queue));
        out.gradOpacity = static_cast<float *>(sycl::malloc_device(numPoints * sizeof(float), queue));

        const auto &cam = buildProducts.cameras().front();
        const size_t pixelCount = static_cast<size_t>(cam.width) * static_cast<size_t>(cam.height);
        out.framebuffer = reinterpret_cast<float4 *>(sycl::malloc_device(pixelCount * sizeof(float4), queue));

        queue.wait();

        //queue.fill(out.gradPosition, float3{0}, numPoints).wait();
        //queue.fill(out.gradTanU, float3{0}, numPoints).wait();
        //queue.fill(out.gradTanV, float3{0}, numPoints).wait();
        //queue.fill(out.gradScale, float2{0}, numPoints).wait();
        //queue.fill(out.framebuffer, float4{0}, pixelCount).wait();

        out.numPoints = numPoints;
        queue.wait();
        return out;
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

    inline std::vector<float>
    downloadDebugGradientImage(sycl::queue queue, const SensorGPU &sensorGpu, PointGradients& gradients) {
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
            gradients.framebuffer, // source (device pointer)
            totalFloatCount * sizeof(float) // size in bytes
        ).wait();

        return hostSideFramebuffer;
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
