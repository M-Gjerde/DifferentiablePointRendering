//
// Created by magnus-desktop on 8/28/25.
//
module;
#include <sycl/sycl.hpp>
#include <Renderer/GPUDataStructures.h>

export module Pale.Render.Sensors;

import Pale.Render.SceneBuild;

export namespace Pale {


    SensorGPU
    makeSensorsForScene(sycl::queue queue, const SceneBuild::BuildProducts& buildProducts) {
        SensorGPU out{};
        if (buildProducts.cameraCount() == 0) {
            return out;
        }

        const auto& cam = buildProducts.cameras().front();
        const size_t pixelCount = static_cast<size_t>(cam.width) * static_cast<size_t>(cam.height);
        float4* dev = reinterpret_cast<float4*>(sycl::malloc_device(pixelCount * sizeof(float4), queue));

        queue.wait();

        //queue.fill(dev, float4{0, 0, 0, 0}, pixelCount).wait();

        out.camera = cam;
        out.framebuffer = dev;
        out.width = cam.width;
        out.height = cam.height;
        return out;
    }

    inline std::vector<float>
    downloadSensorRGBA(sycl::queue queue, const SensorGPU& sensorGpu)
    {
        // Total number of float elements = width * height * 4 (RGBA channels)
        const size_t totalFloatCount = static_cast<size_t>(sensorGpu.width)
                                     * static_cast<size_t>(sensorGpu.height)
                                     * 4u;
        std::vector<float> hostSideFramebuffer(totalFloatCount);


        // Allocate host-side buffer
        queue.wait();
        // Copy device framebuffer → host buffer
        queue.memcpy(
            hostSideFramebuffer.data(),       // destination
            sensorGpu.framebuffer,            // source (device pointer)
            totalFloatCount * sizeof(float)   // size in bytes
        ).wait();

        return hostSideFramebuffer;
    }
}
