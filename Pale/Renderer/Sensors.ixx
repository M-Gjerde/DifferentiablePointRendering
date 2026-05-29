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
                        const SceneBuild::BuildProducts &buildProducts,
                        bool clearData = true, bool simulateAdjoint = false) {
        std::vector<SensorGPU> sensorDevices;
        const auto &cameraList = buildProducts.cameras();
        if (cameraList.empty()) {
            return sensorDevices;
        }
        sensorDevices.reserve(cameraList.size());

        for (const auto &camera: cameraList) {
            if (simulateAdjoint && !camera.useForAdjointPass)
                continue;

            SensorGPU sensorGpu{};
            copyName(sensorGpu.name, camera.name);

            const size_t pixelCount =
                    static_cast<size_t>(camera.width) * static_cast<size_t>(camera.height);

            float4 *deviceHighDynamicRangeFramebuffer =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float4), queue));

            sycl::uchar4 *deviceOutputFramebuffer =
                    reinterpret_cast<sycl::uchar4 *>(
                        sycl::malloc_device(pixelCount * sizeof(sycl::uchar4), queue));

            float *deviceLdrFramebuffer =
                    reinterpret_cast<float *>(
                        sycl::malloc_device(pixelCount * sizeof(float) * 4, queue));

            float *depthDistortionBuffer =
                    reinterpret_cast<float *>(
                        sycl::malloc_device(pixelCount * sizeof(float), queue));

            float *depthDistortionAdjointBuffer =
                    reinterpret_cast<float *>(
                        sycl::malloc_device(pixelCount * sizeof(float), queue));

            float *medianDepthBuffer =
                    reinterpret_cast<float *>(
                        sycl::malloc_device(pixelCount * sizeof(float), queue));


            float *meanDepthBuffer =
                    reinterpret_cast<float *>(
                        sycl::malloc_device(pixelCount * sizeof(float), queue));

            float *medianDepthAdjointBuffer =
                    reinterpret_cast<float *>(
                        sycl::malloc_device(pixelCount * sizeof(float), queue));

            float4 *medianWorldPositionBuffer =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float) * 4, queue));

            float4 *visibleNormalBuffer =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float) * 4, queue));

            float4 *normalFromDepthBuffer =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float) * 4, queue));

            float4 *normalFromDepthAdjointBuffer =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float) * 4, queue));

            float4 *visibleNormalAdjointBuffer =
                    reinterpret_cast<float4 *>(
                        sycl::malloc_device(pixelCount * sizeof(float) * 4, queue));

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
                             pixelCount * 4u * sizeof(float));


                queue.memset(depthDistortionBuffer,
                             0,
                             pixelCount * sizeof(float));
                queue.memset(depthDistortionAdjointBuffer,
                             0,
                             pixelCount * sizeof(float));

                queue.memset(medianDepthBuffer,
                             0,
                             pixelCount * sizeof(float));

                queue.memset(meanDepthBuffer,
                             0,
                             pixelCount * sizeof(float));

                queue.memset(medianDepthAdjointBuffer,
                             0,
                             pixelCount * sizeof(float));

                queue.memset(medianWorldPositionBuffer,
                             0,
                             pixelCount * 4u * sizeof(float));

                queue.memset(visibleNormalBuffer,
                             0,
                             pixelCount * 4u * sizeof(float));

                queue.memset(normalFromDepthBuffer,
                             0,
                             pixelCount * 4u * sizeof(float));
                queue.memset(normalFromDepthAdjointBuffer,
                             0,
                             pixelCount * 4u * sizeof(float));
                queue.memset(visibleNormalAdjointBuffer,
                             0,
                             pixelCount * 4u * sizeof(float));

                queue.wait();
            }

            sensorGpu.camera = camera;
            sensorGpu.framebuffer = deviceHighDynamicRangeFramebuffer;
            sensorGpu.outputFramebuffer = deviceOutputFramebuffer;
            sensorGpu.ldrFramebuffer = deviceLdrFramebuffer;
            sensorGpu.depthDistortionBuffer = depthDistortionBuffer;
            sensorGpu.depthDistortionAdjointBuffer = depthDistortionAdjointBuffer;
            sensorGpu.medianDepthBuffer = medianDepthBuffer;
            sensorGpu.meanDepthBuffer = meanDepthBuffer;
            sensorGpu.medianWorldPositionBuffer = medianWorldPositionBuffer;
            sensorGpu.visibleNormalBuffer = visibleNormalBuffer;
            sensorGpu.normalFromDepthBuffer = normalFromDepthBuffer;

            sensorGpu.medianDepthAdjointBuffer = medianDepthAdjointBuffer;
            sensorGpu.normalFromDepthAdjointBuffer = normalFromDepthAdjointBuffer;
            sensorGpu.visibleNormalAdjointBuffer = visibleNormalAdjointBuffer;
            sensorGpu.width = camera.width;
            sensorGpu.height = camera.height;

            sensorDevices.push_back(sensorGpu);
        }

        return sensorDevices;
    }

    void setBackgroundColor(sycl::queue queue, std::vector<SensorGPU> sensors, float4 color) {
        for (auto &sensor: sensors) {
            queue.fill(sensor.framebuffer, color, sensor.width * sensor.height);
            queue.wait();
        }
    }

    PointGradients
    makeGradientsForScene(sycl::queue queue, const SceneBuild::BuildProducts &buildProducts, DebugImages *debugImages) {
        PointGradients out{};
        const uint32_t numPoints = static_cast<uint32_t>(buildProducts.points.size());
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
            out.gradPosition = static_cast<float3 *>(sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradTanU = static_cast<float3 *>(sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradTanV = static_cast<float3 *>(sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradScale = static_cast<float2 *>(sycl::malloc_device(numPoints * sizeof(float2), queue));
            out.gradAlbedo = static_cast<float3 *>(sycl::malloc_device(numPoints * sizeof(float3), queue));
            out.gradOpacity = static_cast<float *>(sycl::malloc_device(numPoints * sizeof(float), queue));
            out.gradBeta = static_cast<float *>(sycl::malloc_device(numPoints * sizeof(float), queue));
            out.gradShape = static_cast<float *>(sycl::malloc_device(numPoints * sizeof(float), queue));
        }

        const auto &cameraList = buildProducts.cameras();

        // Allocate adjoint framebuffer (same resolution as first camera)
        for (size_t id = 0; const auto &camera: cameraList) {
            if (!camera.useForAdjointPass)
                continue;
            const size_t pixelCount =
                    static_cast<size_t>(camera.width) *
                    static_cast<size_t>(camera.height);

            Pale::Log::PA_INFO(
                "makeGradientsForScene: allocating adjoint framebuffer for {}x{} ({} pixels)",
                camera.width, camera.height, pixelCount
            );

            debugImages[id].framebufferPosX = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            debugImages[id].framebufferPosY = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            debugImages[id].framebufferPosZ = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            debugImages[id].framebufferRot = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            debugImages[id].framebufferScale = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            debugImages[id].framebufferOpacity = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            debugImages[id].framebufferAlbedo = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            debugImages[id].framebufferBeta = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            debugImages[id].framebufferDepthLoss = reinterpret_cast<float4 *>(sycl::malloc_device(
                pixelCount * sizeof(float4), queue));
            debugImages[id].framebufferDepthLossPos = reinterpret_cast<float4 *>(sycl::malloc_device(
                pixelCount * sizeof(float4), queue));
            debugImages[id].framebufferNormalLoss = reinterpret_cast<float4 *>(sycl::malloc_device(
                pixelCount * sizeof(float4), queue));
            debugImages[id].numPixels = pixelCount;

            id++;
        }

        queue.wait();
        return out;
    }

    inline void freeGradientsForScene(sycl::queue queue, PointGradients &g) {
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
        if (g.gradAlbedo) {
            sycl::free(g.gradAlbedo, queue);
            g.gradAlbedo = nullptr;
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

    inline void freeDebugImagesForScene(sycl::queue queue, DebugImages *g, size_t numDebugImages) {
        for (size_t id = 0; id < numDebugImages; id++) {
            if (g[id].framebufferPosX) {
                sycl::free(g[id].framebufferPosX, queue);
                g[id].framebufferPosX = nullptr;
            }
            if (g[id].framebufferPosY) {
                sycl::free(g[id].framebufferPosY, queue);
                g[id].framebufferPosY = nullptr;
            }
            if (g[id].framebufferPosZ) {
                sycl::free(g[id].framebufferPosZ, queue);
                g[id].framebufferPosZ = nullptr;
            }
            if (g[id].framebufferRot) {
                sycl::free(g[id].framebufferRot, queue);
                g[id].framebufferRot = nullptr;
            }
            if (g[id].framebufferScale) {
                sycl::free(g[id].framebufferScale, queue);
                g[id].framebufferScale = nullptr;
            }
            if (g[id].framebufferOpacity) {
                sycl::free(g[id].framebufferOpacity, queue);
                g[id].framebufferOpacity = nullptr;
            }
            if (g[id].framebufferAlbedo) {
                sycl::free(g[id].framebufferAlbedo, queue);
                g[id].framebufferAlbedo = nullptr;
            }
            if (g[id].framebufferBeta) {
                sycl::free(g[id].framebufferBeta, queue);
                g[id].framebufferBeta = nullptr;
            }
            if (g[id].framebufferDepthLoss) {
                sycl::free(g[id].framebufferDepthLoss, queue);
                g[id].framebufferDepthLoss = nullptr;
            }
            if (g[id].framebufferDepthLossPos) {
                sycl::free(g[id].framebufferDepthLossPos, queue);
                g[id].framebufferDepthLossPos = nullptr;
            }
            if (g[id].framebufferNormalLoss) {
                sycl::free(g[id].framebufferNormalLoss, queue);
                g[id].framebufferNormalLoss = nullptr;
            }
        }
    }

    inline std::vector<float>
    downloadSensorLDR(sycl::queue queue, const SensorGPU &sensorGpu) {
        // Total number of float elements = width * height * 4 (RGBa channels)
        const size_t totalFloatCount = static_cast<size_t>(sensorGpu.width)
                                       * static_cast<size_t>(sensorGpu.height)
                                       * 4u;
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
    downloadSensorRGBARAW(sycl::queue queue, const SensorGPU &sensorGpu) {
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

    inline std::vector<float> downloadFloatBuffer(
        sycl::queue queue,
        const float *devicePtr,
        std::size_t count) {
        std::vector<float> host(count);
        if (devicePtr != nullptr && count > 0) {
            queue.memcpy(host.data(), devicePtr, count * sizeof(float)).wait();
        }
        return host;
    }

    inline std::vector<float> downloadFloat4Buffer(
        sycl::queue queue,
        const float4 *devicePtr,
        std::size_t count) {
        std::vector<float> host(count * 4);
        if (devicePtr != nullptr && count > 0) {
            queue.memcpy(host.data(), devicePtr, count * sizeof(float) * 4).wait();
        }
        return host;
    }

    inline void uploadFloatImage(
        sycl::queue &queue,
        float *devicePtr,
        const std::vector<float> &hostData) {
        if (!devicePtr) {
            throw std::runtime_error("uploadFloatImage: devicePtr is null");
        }
        if (!hostData.empty()) {
            queue.memcpy(
                devicePtr,
                hostData.data(),
                hostData.size() * sizeof(float)
            ).wait();
        }
    }

    inline std::vector<float>
    downloadSensorDepthDistortionRAW(sycl::queue queue, const SensorGPU &sensorGpu) {
        // Total number of float elements = width * height * 4 (RGBA channels)
        const size_t totalFloatCount = static_cast<size_t>(sensorGpu.width)
                                       * static_cast<size_t>(sensorGpu.height);
        std::vector<float> hostSideFramebuffer(totalFloatCount);


        // Allocate host-side buffer
        queue.wait();
        // Copy device framebuffer → host buffer
        queue.memcpy(
            hostSideFramebuffer.data(), // destination
            sensorGpu.depthDistortionBuffer, // source (device pointer)
            totalFloatCount * sizeof(float) // size in bytes
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
        std::vector<float> depthLoss; // framebuffer_albedo
        std::vector<float> depthLossPos; // framebuffer_albedo
        std::vector<float> normalLoss; // framebuffer_albedo
    };

    inline DebugGradientImagesHost downloadDebugGradientImages(
        sycl::queue queue,
        const SensorGPU &sensorGpu,
        const DebugImages &debugImages
    ) {
        const std::size_t pixelCount =
                static_cast<std::size_t>(sensorGpu.width) *
                static_cast<std::size_t>(sensorGpu.height);

        const std::size_t rgbaFloatCount = pixelCount * 4u;

        DebugGradientImagesHost images;
        images.positionX.resize(rgbaFloatCount);
        images.positionY.resize(rgbaFloatCount);
        images.positionZ.resize(rgbaFloatCount);
        images.rotation.resize(rgbaFloatCount);
        images.scale.resize(rgbaFloatCount);
        images.opacity.resize(rgbaFloatCount);
        images.albedo.resize(rgbaFloatCount);
        images.beta.resize(rgbaFloatCount);
        images.depthLoss.resize(rgbaFloatCount);
        images.depthLossPos.resize(rgbaFloatCount);
        images.normalLoss.resize(rgbaFloatCount);

        auto downloadScalarImageAsGrayscaleRgba =
                [&](std::vector<float> &hostRgbaBuffer, const float *deviceScalarBuffer) {
            std::vector<float> hostScalarBuffer(pixelCount);

            queue.memcpy(
                hostScalarBuffer.data(),
                deviceScalarBuffer,
                pixelCount * sizeof(float)
            ).wait();

            for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
                const float value = hostScalarBuffer[pixelIndex];
                const std::size_t rgbaIndex = pixelIndex * 4u;

                hostRgbaBuffer[rgbaIndex + 0u] = value;
                hostRgbaBuffer[rgbaIndex + 1u] = value;
                hostRgbaBuffer[rgbaIndex + 2u] = value;
                hostRgbaBuffer[rgbaIndex + 3u] = 1.0f;
            }
        };

        downloadScalarImageAsGrayscaleRgba(images.positionX, debugImages.framebufferPosX);
        downloadScalarImageAsGrayscaleRgba(images.positionY, debugImages.framebufferPosY);
        downloadScalarImageAsGrayscaleRgba(images.positionZ, debugImages.framebufferPosZ);
        downloadScalarImageAsGrayscaleRgba(images.rotation, debugImages.framebufferRot);
        downloadScalarImageAsGrayscaleRgba(images.scale, debugImages.framebufferScale);
        downloadScalarImageAsGrayscaleRgba(images.opacity, debugImages.framebufferOpacity);
        downloadScalarImageAsGrayscaleRgba(images.albedo, debugImages.framebufferAlbedo);
        downloadScalarImageAsGrayscaleRgba(images.beta, debugImages.framebufferBeta);
        //downloadScalarImageAsGrayscaleRgba(images.depthLoss, debugImages.framebufferDepthLoss);
        //downloadScalarImageAsGrayscaleRgba(images.depthLossPos, debugImages.framebufferDepthLossPos);
        //downloadScalarImageAsGrayscaleRgba(images.normalLoss, debugImages.framebufferNormalLoss);

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
