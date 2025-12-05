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
    std::vector<SensorGPU>
    makeSensorsForScene(sycl::queue queue,
                        const SceneBuild::BuildProducts& buildProducts,
                        bool clearData = true, bool simulateAdjoint = false) {
        std::vector<SensorGPU> sensorDevices;
        const auto& cameraList = buildProducts.cameras();
        if (cameraList.empty()) {
            return sensorDevices;
        }
        sensorDevices.reserve(cameraList.size());

        for (const auto& camera : cameraList) {
            SensorGPU sensorGpu{};
            copyName(sensorGpu.name, camera.name);

            const size_t pixelCount =
                static_cast<size_t>(camera.width) * static_cast<size_t>(camera.height);

            float4* deviceHighDynamicRangeFramebuffer =
                reinterpret_cast<float4*>(
                    sycl::malloc_device(pixelCount * sizeof(float4), queue));

            sycl::uchar4* deviceOutputFramebuffer =
                reinterpret_cast<sycl::uchar4*>(
                    sycl::malloc_device(pixelCount * sizeof(sycl::uchar4), queue));

            float* deviceLdrFramebuffer =
                reinterpret_cast<float*>(
                    sycl::malloc_device(pixelCount * sizeof(float) * 3, queue));

            // Optional: check allocations
            if (deviceHighDynamicRangeFramebuffer == nullptr ||
                deviceOutputFramebuffer == nullptr ||
                deviceLdrFramebuffer == nullptr) {
                // Handle allocation failure: free what succeeded, skip this camera or throw
                if (deviceHighDynamicRangeFramebuffer) {
                    sycl::free(deviceHighDynamicRangeFramebuffer, queue);
                }
                if (deviceOutputFramebuffer) {
                    sycl::free(deviceOutputFramebuffer, queue);
                }
                if (deviceLdrFramebuffer) {
                    sycl::free(deviceLdrFramebuffer, queue);
                }
                continue;
            }

            if (clearData) {

                if (simulateAdjoint)
                    queue.fill(deviceHighDynamicRangeFramebuffer, float4{1.0f}, pixelCount);
                else {
                    queue.fill(deviceHighDynamicRangeFramebuffer, float4{0.0f}, pixelCount);
                }
                // Output framebuffer initialized to black / zero alpha
                queue.memset(deviceOutputFramebuffer,
                             0,
                             pixelCount * sizeof(sycl::uchar4));

                // LDR framebuffer initialized to zero
                queue.memset(deviceLdrFramebuffer,
                             0,
                             pixelCount * 3u * sizeof(float));
                queue.wait();
            }

            sensorGpu.camera = camera;
            sensorGpu.framebuffer = deviceHighDynamicRangeFramebuffer;
            sensorGpu.outputFramebuffer = deviceOutputFramebuffer;
            sensorGpu.ldrFramebuffer = deviceLdrFramebuffer;
            sensorGpu.width = camera.width;
            sensorGpu.height = camera.height;

            sensorDevices.push_back(sensorGpu);
        }

        return sensorDevices;
    }

    PointGradients
    makeGradientsForScene(sycl::queue queue,
                          const SceneBuild::BuildProducts& buildProducts, DebugImages* debugImages) {
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
                static_cast<float3*>(
                    sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradTanU =
                static_cast<float3*>(
                    sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradTanV =
                static_cast<float3*>(
                    sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradScale =
                static_cast<float2*>(
                    sycl::malloc_device(numPoints * sizeof(float2), queue));
            out.gradColor =
                static_cast<float3*>(
                    sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradOpacity =
                static_cast<float*>(
                    sycl::malloc_device(numPoints * sizeof(float), queue));
            out.gradBeta =
                static_cast<float*>(
                    sycl::malloc_device(numPoints * sizeof(float), queue));
            out.gradShape =
                static_cast<float*>(
                    sycl::malloc_device(numPoints * sizeof(float), queue));
        }

        const auto& cameraList = buildProducts.cameras();

        // Allocate adjoint framebuffer (same resolution as first camera)
        for (size_t id = 0; const auto& camera : cameraList) {
            const size_t pixelCount =
                static_cast<size_t>(camera.width) *
                static_cast<size_t>(camera.height);

            Pale::Log::PA_INFO(
                "makeGradientsForScene: allocating adjoint framebuffer for {}x{} ({} pixels)",
                camera.width, camera.height, pixelCount
            );

            debugImages[id].framebuffer_posX =
                reinterpret_cast<float4*>(
                    sycl::malloc_device(pixelCount * sizeof(float4), queue));
            debugImages[id].framebuffer_posY =
                reinterpret_cast<float4*>(
                    sycl::malloc_device(pixelCount * sizeof(float4), queue));
            debugImages[id].framebuffer_posZ =
                reinterpret_cast<float4*>(
                    sycl::malloc_device(pixelCount * sizeof(float4), queue));
            debugImages[id].framebuffer_rot =
                reinterpret_cast<float4*>(
                    sycl::malloc_device(pixelCount * sizeof(float4), queue));
            debugImages[id].framebuffer_scale =
                reinterpret_cast<float4*>(
                    sycl::malloc_device(pixelCount * sizeof(float4), queue));
            debugImages[id].framebuffer_opacity =
                reinterpret_cast<float4*>(
                    sycl::malloc_device(pixelCount * sizeof(float4), queue));
            debugImages[id].framebuffer_albedo =
                reinterpret_cast<float4*>(
                    sycl::malloc_device(pixelCount * sizeof(float4), queue));

            debugImages[id].framebuffer_beta =
                reinterpret_cast<float4*>(
                    sycl::malloc_device(pixelCount * sizeof(float4), queue));
            debugImages[id].numPixels = pixelCount;

            id++;
        }

        queue.wait();
        return out;
    }

    inline void freeGradientsForScene(sycl::queue queue, PointGradients& g) {
        if (g.gradPosition) {
            sycl::free(g.gradPosition, queue);
            g.gradPosition = nullptr;
        }
        if (g.gradTanU) {
            sycl::free(g.gradTanU, queue);
            g.gradTanU = nullptr;
        }
        if (g.gradTanV) {
            sycl::free(g.gradTanV, queue);
            g.gradTanV = nullptr;
        }
        if (g.gradScale) {
            sycl::free(g.gradScale, queue);
            g.gradScale = nullptr;
        }
        if (g.gradColor) {
            sycl::free(g.gradColor, queue);
            g.gradColor = nullptr;
        }
        if (g.gradOpacity) {
            sycl::free(g.gradOpacity, queue);
            g.gradOpacity = nullptr;
        }
        if (g.gradBeta) {
            sycl::free(g.gradBeta, queue);
            g.gradBeta = nullptr;
        }
        if (g.gradShape) {
            sycl::free(g.gradShape, queue);
            g.gradShape = nullptr;
        }
        g.numPoints = 0;
    }

    inline void freeDebugImagesForScene(sycl::queue queue, DebugImages* g, size_t numDebugImages) {
        for (size_t id = 0; id < numDebugImages; id++) {
            if (g[id].framebuffer_posX) {
                sycl::free(g[id].framebuffer_posX, queue);
                g[id].framebuffer_posX = nullptr;
            }
            if (g[id].framebuffer_posY) {
                sycl::free(g[id].framebuffer_posY, queue);
                g[id].framebuffer_posY = nullptr;
            }
            if (g[id].framebuffer_posZ) {
                sycl::free(g[id].framebuffer_posZ, queue);
                g[id].framebuffer_posZ = nullptr;
            }
            if (g[id].framebuffer_rot) {
                sycl::free(g[id].framebuffer_rot, queue);
                g[id].framebuffer_rot = nullptr;
            }
            if (g[id].framebuffer_scale) {
                sycl::free(g[id].framebuffer_scale, queue);
                g[id].framebuffer_scale = nullptr;
            }
            if (g[id].framebuffer_opacity) {
                sycl::free(g[id].framebuffer_opacity, queue);
                g[id].framebuffer_opacity = nullptr;
            }
            if (g[id].framebuffer_albedo) {
                sycl::free(g[id].framebuffer_albedo, queue);
                g[id].framebuffer_albedo = nullptr;
            }
            if (g[id].framebuffer_beta) {
                sycl::free(g[id].framebuffer_beta, queue);
                g[id].framebuffer_beta = nullptr;
            }
        }
    }

    inline std::vector<float>
    downloadSensorRGBARaw(sycl::queue queue, const SensorGPU& sensorGpu) {
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
    downloadSensorLDR(sycl::queue queue, const SensorGPU& sensorGpu) {
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
    downloadSensorRGBA(sycl::queue queue, const SensorGPU& sensorGpu) {
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

    struct DebugGradientImagesHost {
        // Each buffer has size: width * height * 4 (RGBA)
        std::vector<float> positionX; // framebuffer_pos
        std::vector<float> positionY; // framebuffer_pos
        std::vector<float> positionZ; // framebuffer_pos
        std::vector<float> rotation; // framebuffer_rot
        std::vector<float> scale; // framebuffer_scale
        std::vector<float> opacity; // framebuffer_opacity
        std::vector<float> albedo; // framebuffer_albedo
        std::vector<float> beta; // framebuffer_albedo
    };

    inline DebugGradientImagesHost downloadDebugGradientImages(
        sycl::queue queue,
        const SensorGPU& sensorGpu,
        const DebugImages& debugImages
    ) {
        const std::size_t pixelCount = static_cast<std::size_t>(sensorGpu.width)
            * static_cast<std::size_t>(sensorGpu.height);
        const std::size_t totalFloatCount = pixelCount * 4u; // RGBA

        DebugGradientImagesHost images;
        images.positionX.resize(totalFloatCount);
        images.positionY.resize(totalFloatCount);
        images.positionZ.resize(totalFloatCount);
        images.rotation.resize(totalFloatCount);
        images.scale.resize(totalFloatCount);
        images.opacity.resize(totalFloatCount);
        images.albedo.resize(totalFloatCount);
        images.beta.resize(totalFloatCount);

        auto copyBuffer = [&](std::vector<float>& hostBuffer, const float4* deviceBuffer) {
            queue.memcpy(
                hostBuffer.data(), // destination (host)
                deviceBuffer, // source (device)
                totalFloatCount * sizeof(float)
            );
        };

        copyBuffer(images.positionX, debugImages.framebuffer_posX);
        copyBuffer(images.positionY, debugImages.framebuffer_posY);
        copyBuffer(images.positionZ, debugImages.framebuffer_posZ);
        copyBuffer(images.rotation, debugImages.framebuffer_rot);
        copyBuffer(images.scale, debugImages.framebuffer_scale);
        copyBuffer(images.opacity, debugImages.framebuffer_opacity);
        copyBuffer(images.albedo, debugImages.framebuffer_albedo);
        copyBuffer(images.beta, debugImages.framebuffer_beta);

        // Ensure all copies are completed before returning
        queue.wait();

        return images;
    }


    inline std::vector<float>
    uploadSensorRGBA(sycl::queue queue, const SensorGPU& sensorGpu, std::vector<float> hostSideFramebuffer) {
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
