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
    makeSensorsForScene(sycl::queue q, const SceneBuild::BuildProducts& scene) {
        SensorGPU out;

        /*
        out.reserve(scene.cameraCount());

        for (auto& cam : scene.cameras()) {             // assumes accessors exist
            const size_t n = size_t(cam.width) * cam.height;
            auto* dev = (sycl::float4*)sycl::malloc_device(n * sizeof(sycl::float4), q);
            q.fill(dev, sycl::float4{0,0,0,0}, n).wait();
            out.push_back({dev, cam.width, cam.height});
        }
        */
        return out;
    }

    inline std::vector<float>
    downloadSensorRGBA(sycl::queue queue, const SensorGPU& sensorGpu)
    {
        // Total number of float elements = width * height * 4 (RGBA channels)
        const size_t totalFloatCount = static_cast<size_t>(sensorGpu.width)
                                     * static_cast<size_t>(sensorGpu.height)
                                     * 4u;
        // Allocate host-side buffer
        std::vector<float> hostSideFramebuffer(totalFloatCount);
        // Copy device framebuffer → host buffer
        queue.memcpy(
            hostSideFramebuffer.data(),       // destination
            sensorGpu.framebuffer,            // source (device pointer)
            totalFloatCount * sizeof(float)   // size in bytes
        ).wait();

        return hostSideFramebuffer;
    }
}
