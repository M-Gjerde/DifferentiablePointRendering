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
import Pale.Utils.StringFormatting;

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

        for (std::size_t cameraIndex = 0; cameraIndex < cameraList.size(); ++cameraIndex) {
            const auto& camera = cameraList[cameraIndex];

            if (simulateAdjoint && !camera.useForAdjointPass)
                continue;
            SensorGPU sensorGpu{};
            copyName(sensorGpu.name, camera.name);

            const size_t pixelCount = static_cast<size_t>(camera.width) * static_cast<size_t>(camera.height);
            float4 *deviceHighDynamicRangeFramebuffer = reinterpret_cast<float4 *>(sycl::malloc_device(
                pixelCount * sizeof(float4), queue));
            sycl::uchar4 *deviceOutputFramebuffer = reinterpret_cast<sycl::uchar4 *>(sycl::malloc_device(
                pixelCount * sizeof(sycl::uchar4), queue));
            float *deviceLdrFramebuffer = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float) * 4, queue));
            float *depthDistortionBuffer = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            float *depthDistortionAdjointBuffer = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            float *medianDepthBuffer = reinterpret_cast<float *>(
                sycl::malloc_device(pixelCount * sizeof(float), queue));
            float *visibilityWeightedOpacityBuffer = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            float *intraSlabDepthBuffer = sycl::malloc_device<float>(pixelCount, queue);
            float *intraSlabDepthAdjointBuffer = sycl::malloc_device<float>(pixelCount, queue);
            uint32_t *intraSlabDepthActiveSlabCountBuffer =
                    sycl::malloc_device<uint32_t>(pixelCount, queue);
            float *curvatureScaleBuffer = sycl::malloc_device<float>(pixelCount, queue);
            float *curvatureScaleAdjointBuffer = sycl::malloc_device<float>(pixelCount, queue);
            uint32_t *curvatureScaleActiveSlabCountBuffer =
                    sycl::malloc_device<uint32_t>(pixelCount, queue);
            float *meanDepthBuffer = reinterpret_cast<float *>(sycl::malloc_device(pixelCount * sizeof(float), queue));
            float *medianDepthAdjointBuffer = reinterpret_cast<float *>(sycl::malloc_device(
                pixelCount * sizeof(float), queue));
            float4 *medianWorldPositionBuffer = reinterpret_cast<float4 *>(sycl::malloc_device(
                pixelCount * sizeof(float) * 4, queue));
            float4 *visibleNormalBuffer = reinterpret_cast<float4 *>(sycl::malloc_device(
                pixelCount * sizeof(float) * 4, queue));
            float4 *normalFromDepthBuffer = reinterpret_cast<float4 *>(sycl::malloc_device(
                pixelCount * sizeof(float) * 4, queue));
            float4 *normalFromDepthAdjointBuffer = reinterpret_cast<float4 *>(sycl::malloc_device(
                pixelCount * sizeof(float) * 4, queue));
            float4 *visibleNormalAdjointBuffer = reinterpret_cast<float4 *>(sycl::malloc_device(
                pixelCount * sizeof(float) * 4, queue));

            // Optional: check allocations
            if (deviceHighDynamicRangeFramebuffer == nullptr ||
                deviceOutputFramebuffer == nullptr ||
                deviceLdrFramebuffer == nullptr ||
                depthDistortionBuffer == nullptr ||
                depthDistortionAdjointBuffer == nullptr ||
                visibilityWeightedOpacityBuffer == nullptr ||
                intraSlabDepthBuffer == nullptr ||
                intraSlabDepthAdjointBuffer == nullptr ||
                intraSlabDepthActiveSlabCountBuffer == nullptr ||
                curvatureScaleBuffer == nullptr ||
                curvatureScaleAdjointBuffer == nullptr ||
                curvatureScaleActiveSlabCountBuffer == nullptr) {
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
                if (depthDistortionBuffer) sycl::free(depthDistortionBuffer, queue);
                if (depthDistortionAdjointBuffer) sycl::free(depthDistortionAdjointBuffer, queue);
                if (visibilityWeightedOpacityBuffer) sycl::free(visibilityWeightedOpacityBuffer, queue);
                if (intraSlabDepthBuffer) sycl::free(intraSlabDepthBuffer, queue);
                if (intraSlabDepthAdjointBuffer) sycl::free(intraSlabDepthAdjointBuffer, queue);
                if (intraSlabDepthActiveSlabCountBuffer) {
                    sycl::free(intraSlabDepthActiveSlabCountBuffer, queue);
                }
                if (curvatureScaleBuffer) sycl::free(curvatureScaleBuffer, queue);
                if (curvatureScaleAdjointBuffer) sycl::free(curvatureScaleAdjointBuffer, queue);
                if (curvatureScaleActiveSlabCountBuffer) {
                    sycl::free(curvatureScaleActiveSlabCountBuffer, queue);
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
                queue.memset(deviceOutputFramebuffer, 0, pixelCount * sizeof(sycl::uchar4));
                // LDR framebuffer initialized to zero
                queue.memset(deviceLdrFramebuffer, 0, pixelCount * 4u * sizeof(float));
                queue.memset(depthDistortionBuffer, 0, pixelCount * sizeof(float));
                queue.memset(depthDistortionAdjointBuffer, 0, pixelCount * sizeof(float));
                queue.memset(medianDepthBuffer, 0, pixelCount * sizeof(float));
                queue.memset(visibilityWeightedOpacityBuffer, 0, pixelCount * sizeof(float));
                queue.memset(intraSlabDepthBuffer, 0, pixelCount * sizeof(float));
                queue.memset(intraSlabDepthAdjointBuffer, 0, pixelCount * sizeof(float));
                queue.memset(intraSlabDepthActiveSlabCountBuffer, 0, pixelCount * sizeof(uint32_t));
                queue.memset(curvatureScaleBuffer, 0, pixelCount * sizeof(float));
                queue.memset(curvatureScaleAdjointBuffer, 0, pixelCount * sizeof(float));
                queue.memset(curvatureScaleActiveSlabCountBuffer, 0, pixelCount * sizeof(uint32_t));
                queue.memset(meanDepthBuffer, 0, pixelCount * sizeof(float));
                queue.memset(medianDepthAdjointBuffer, 0, pixelCount * sizeof(float));
                queue.memset(medianWorldPositionBuffer, 0, pixelCount * 4u * sizeof(float));
                queue.memset(visibleNormalBuffer, 0, pixelCount * 4u * sizeof(float));
                queue.memset(normalFromDepthBuffer, 0, pixelCount * 4u * sizeof(float));
                queue.memset(normalFromDepthAdjointBuffer, 0, pixelCount * 4u * sizeof(float));
                queue.memset(visibleNormalAdjointBuffer, 0, pixelCount * 4u * sizeof(float));
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
            sensorGpu.visibilityWeightedOpacityBuffer = visibilityWeightedOpacityBuffer;
            sensorGpu.intraSlabDepthBuffer = intraSlabDepthBuffer;
            sensorGpu.intraSlabDepthAdjointBuffer = intraSlabDepthAdjointBuffer;
            sensorGpu.intraSlabDepthActiveSlabCountBuffer = intraSlabDepthActiveSlabCountBuffer;
            sensorGpu.curvatureScaleBuffer = curvatureScaleBuffer;
            sensorGpu.curvatureScaleAdjointBuffer = curvatureScaleAdjointBuffer;
            sensorGpu.curvatureScaleActiveSlabCountBuffer = curvatureScaleActiveSlabCountBuffer;
            sensorGpu.medianWorldPositionBuffer = medianWorldPositionBuffer;
            sensorGpu.visibleNormalBuffer = visibleNormalBuffer;
            sensorGpu.normalFromDepthBuffer = normalFromDepthBuffer;

            sensorGpu.medianDepthAdjointBuffer = medianDepthAdjointBuffer;
            sensorGpu.normalFromDepthAdjointBuffer = normalFromDepthAdjointBuffer;
            sensorGpu.visibleNormalAdjointBuffer = visibleNormalAdjointBuffer;
            sensorGpu.width = camera.width;
            sensorGpu.height = camera.height;
            sensorGpu.cameraSlotIndex = static_cast<uint32_t>(cameraIndex);

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

    PointGradients makeGradientsForScene(
        sycl::queue queue,
        const SceneBuild::BuildProducts &buildProducts,
        DebugImages *debugImages) {
        PointGradients out{};

        const uint32_t numPoints = static_cast<uint32_t>(buildProducts.points.size());
        const auto &cameraList = buildProducts.cameras();
        const uint32_t cameraSlotCount = static_cast<uint32_t>(cameraList.size());

        out.numPoints = numPoints;
        out.cameraSlotCount = cameraSlotCount;

        Pale::Log::PA_INFO(
            "makeGradientsForScene: allocating gradients for {} points and {} camera slots",
            numPoints,
            cameraSlotCount);

        if (numPoints > 0u) {
            out.gradPosition = sycl::malloc_device<float3>(numPoints, queue);
            out.cloneSignal = sycl::malloc_device<float3>(numPoints, queue);
            out.gradRotation = sycl::malloc_device<float3>(numPoints, queue);
            out.gradScale = sycl::malloc_device<float2>(numPoints, queue);
            out.gradAlbedo = sycl::malloc_device<float3>(numPoints, queue);
            out.gradOpacity = sycl::malloc_device<float>(numPoints, queue);
            out.gradBeta = sycl::malloc_device<float>(numPoints, queue);
            out.gradShape = sycl::malloc_device<float>(numPoints, queue);

            out.cloneSignalMeanNorm = sycl::malloc_device<float>(numPoints, queue);
            out.cloneSignalStd = sycl::malloc_device<float>(numPoints, queue);
            out.cloneSignalCoherence = sycl::malloc_device<float>(numPoints, queue);
            out.cloneSignalDisagreement = sycl::malloc_device<float>(numPoints, queue);
            out.cloneSignalActiveCameraCount = sycl::malloc_device<uint32_t>(numPoints, queue);

            const size_t primitiveCameraCount =
                    static_cast<size_t>(numPoints) * static_cast<size_t>(cameraSlotCount);

            if (cameraSlotCount > 0u) {
                out.gradPositionPerPrimitivePerCamera =
                        sycl::malloc_device<float3>(primitiveCameraCount, queue);

                out.gradPositionRecordCountPerPrimitivePerCamera =
                        sycl::malloc_device<uint32_t>(primitiveCameraCount, queue);

                out.cloneSignalPerPrimitivePerCamera =
                        sycl::malloc_device<float3>(primitiveCameraCount, queue);

                out.cloneSignalRecordCountPerPrimitivePerCamera =
                        sycl::malloc_device<uint32_t>(primitiveCameraCount, queue);
            }

            if (!out.gradPosition || !out.cloneSignal || !out.gradRotation || !out.gradScale ||
                !out.gradAlbedo || !out.gradOpacity || !out.gradBeta || !out.gradShape ||
                !out.cloneSignalMeanNorm || !out.cloneSignalStd ||
                !out.cloneSignalCoherence || !out.cloneSignalDisagreement ||
                !out.cloneSignalActiveCameraCount ||
                (cameraSlotCount > 0u && (!out.gradPositionPerPrimitivePerCamera ||
                                          !out.gradPositionRecordCountPerPrimitivePerCamera ||
                                          !out.cloneSignalPerPrimitivePerCamera ||
                                          !out.cloneSignalRecordCountPerPrimitivePerCamera))) {
                throw std::runtime_error("makeGradientsForScene: failed to allocate one or more gradient buffers");
            }

            queue.fill(out.gradPosition, float3{0.0f, 0.0f, 0.0f}, numPoints);
            queue.fill(out.cloneSignal, float3{0.0f, 0.0f, 0.0f}, numPoints);
            queue.fill(out.gradRotation, float3{0.0f, 0.0f, 0.0f}, numPoints);
            queue.fill(out.gradScale, float2{0.0f, 0.0f}, numPoints);
            queue.fill(out.gradAlbedo, float3{0.0f, 0.0f, 0.0f}, numPoints);
            queue.fill(out.gradOpacity, 0.0f, numPoints);
            queue.fill(out.gradBeta, 0.0f, numPoints);
            queue.fill(out.gradShape, 0.0f, numPoints);

            queue.fill(out.cloneSignalMeanNorm, 0.0f, numPoints);
            queue.fill(out.cloneSignalStd, 0.0f, numPoints);
            queue.fill(out.cloneSignalCoherence, 0.0f, numPoints);
            queue.fill(out.cloneSignalDisagreement, 0.0f, numPoints);
            queue.fill(out.cloneSignalActiveCameraCount, 0u, numPoints);

            if (cameraSlotCount > 0u) {
                queue.fill(out.gradPositionPerPrimitivePerCamera, float3{0.0f, 0.0f, 0.0f}, primitiveCameraCount);
                queue.fill(out.gradPositionRecordCountPerPrimitivePerCamera, 0u, primitiveCameraCount);
                queue.fill(out.cloneSignalPerPrimitivePerCamera, float3{0.0f, 0.0f, 0.0f}, primitiveCameraCount);
                queue.fill(out.cloneSignalRecordCountPerPrimitivePerCamera, 0u, primitiveCameraCount);
            }

            Pale::Log::PA_INFO(
                "makeGradientsForScene: gradient memory: paramPosition={}, perCameraPosition={}, stats={}",
                Pale::Utils::formatBytes(sizeof(float3) * static_cast<size_t>(numPoints)),
                Pale::Utils::formatBytes(sizeof(float3) * primitiveCameraCount),
                Pale::Utils::formatBytes(
                    sizeof(float) * static_cast<size_t>(numPoints) * 4u +
                    sizeof(uint32_t) * static_cast<size_t>(numPoints)));
        }

        if (cameraList.empty()) {
            Pale::Log::PA_WARN(
                "makeGradientsForScene: no cameras in buildProducts; debug images will not be allocated");
            queue.wait();
            return out;
        }

        if (!debugImages) {
            queue.wait();
            return out;
        }

        for (size_t cameraIndex = 0; cameraIndex < cameraList.size(); ++cameraIndex) {
            const auto &camera = cameraList[cameraIndex];

            DebugImages &debugImage = debugImages[cameraIndex];
            debugImage = DebugImages{};

            if (!camera.useForAdjointPass) {
                continue;
            }

            const size_t pixelCount =
                    static_cast<size_t>(camera.width) *
                    static_cast<size_t>(camera.height);

            Pale::Log::PA_INFO(
                "makeGradientsForScene: allocating debug gradient images for camera '{}' {}x{} ({} pixels)",
                camera.name,
                camera.width,
                camera.height,
                pixelCount);

            debugImage.framebufferPosX = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferPosY = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferPosZ = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferRotX = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferRotY = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferRotZ = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferScaleU = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferScaleV = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferOpacity = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferAlbedo = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferBeta = sycl::malloc_device<float>(pixelCount, queue);
            debugImage.framebufferDepthLoss = sycl::malloc_device<float4>(pixelCount, queue);
            debugImage.framebufferDepthLossPos = sycl::malloc_device<float4>(pixelCount, queue);
            debugImage.framebufferNormalLoss = sycl::malloc_device<float4>(pixelCount, queue);
            debugImage.numPixels = pixelCount;

            if (!debugImage.framebufferPosX || !debugImage.framebufferPosY || !debugImage.framebufferPosZ ||
                !debugImage.framebufferRotX ||  !debugImage.framebufferRotY ||  !debugImage.framebufferRotZ || !debugImage.framebufferScaleU ||  !debugImage.framebufferScaleV || !debugImage.framebufferOpacity ||
                !debugImage.framebufferAlbedo || !debugImage.framebufferBeta ||
                !debugImage.framebufferDepthLoss || !debugImage.framebufferDepthLossPos ||
                !debugImage.framebufferNormalLoss) {
                throw std::runtime_error("makeGradientsForScene: failed to allocate one or more debug image buffers");
            }

            queue.fill(debugImage.framebufferPosX, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferPosY, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferPosZ, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferRotX, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferRotY, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferRotZ, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferScaleU, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferScaleV, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferOpacity, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferAlbedo, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferBeta, 0.0f, pixelCount);
            queue.fill(debugImage.framebufferDepthLoss, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount);
            queue.fill(debugImage.framebufferDepthLossPos, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount);
            queue.fill(debugImage.framebufferNormalLoss, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount);
        }

        queue.wait();
        return out;
    }

    CurvatureDensificationStats makeCurvatureDensificationStatsForScene(
        sycl::queue queue,
        const SceneBuild::BuildProducts &buildProducts) {
        CurvatureDensificationStats out{};
        out.numPoints = buildProducts.points.size();
        if (out.numPoints == 0u) {
            return out;
        }

        out.violationSum = sycl::malloc_device<float>(out.numPoints, queue);
        out.violationCount = sycl::malloc_device<uint32_t>(out.numPoints, queue);
        out.directionTensorUu = sycl::malloc_device<float>(out.numPoints, queue);
        out.directionTensorUv = sycl::malloc_device<float>(out.numPoints, queue);
        out.directionTensorVv = sycl::malloc_device<float>(out.numPoints, queue);

        if (!out.violationSum || !out.violationCount ||
            !out.directionTensorUu || !out.directionTensorUv || !out.directionTensorVv) {
            const auto freeDevicePtr = [&queue]<typename T>(T *&devicePtr) {
                if (devicePtr) {
                    sycl::free(devicePtr, queue);
                    devicePtr = nullptr;
                }
            };
            freeDevicePtr(out.violationSum);
            freeDevicePtr(out.violationCount);
            freeDevicePtr(out.directionTensorUu);
            freeDevicePtr(out.directionTensorUv);
            freeDevicePtr(out.directionTensorVv);
            out.numPoints = 0u;
            throw std::runtime_error(
                "makeCurvatureDensificationStatsForScene: failed to allocate buffers");
        }

        queue.fill(out.violationSum, 0.0f, out.numPoints);
        queue.fill(out.violationCount, 0u, out.numPoints);
        queue.fill(out.directionTensorUu, 0.0f, out.numPoints);
        queue.fill(out.directionTensorUv, 0.0f, out.numPoints);
        queue.fill(out.directionTensorVv, 0.0f, out.numPoints);
        queue.wait();
        return out;
    }

    void clearCurvatureDensificationStats(
        sycl::queue queue,
        const CurvatureDensificationStats &stats) {
        if (stats.numPoints == 0u) {
            return;
        }
        queue.fill(stats.violationSum, 0.0f, stats.numPoints);
        queue.fill(stats.violationCount, 0u, stats.numPoints);
        queue.fill(stats.directionTensorUu, 0.0f, stats.numPoints);
        queue.fill(stats.directionTensorUv, 0.0f, stats.numPoints);
        queue.fill(stats.directionTensorVv, 0.0f, stats.numPoints);
        queue.wait();
    }

    void freeCurvatureDensificationStats(
        sycl::queue queue,
        CurvatureDensificationStats &stats) {
        const auto freeDevicePtr = [&queue]<typename T>(T *&devicePtr) {
            if (devicePtr) {
                sycl::free(devicePtr, queue);
                devicePtr = nullptr;
            }
        };
        freeDevicePtr(stats.violationSum);
        freeDevicePtr(stats.violationCount);
        freeDevicePtr(stats.directionTensorUu);
        freeDevicePtr(stats.directionTensorUv);
        freeDevicePtr(stats.directionTensorVv);
        stats.numPoints = 0u;
        queue.wait();
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

    inline std::vector<uint32_t> downloadUint32Buffer(
        sycl::queue queue,
        const uint32_t *devicePtr,
        std::size_t count) {
        std::vector<uint32_t> host(count);
        if (devicePtr != nullptr && count > 0) {
            queue.memcpy(host.data(), devicePtr, count * sizeof(uint32_t)).wait();
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

    inline std::vector<float>
    downloadSensorVisibilityOpacityRAW(sycl::queue queue, const SensorGPU &sensorGpu) {
        // Total number of float elements = width * height * 4 (RGBA channels)
        const size_t totalFloatCount = static_cast<size_t>(sensorGpu.width)
                                       * static_cast<size_t>(sensorGpu.height);
        std::vector<float> hostSideFramebuffer(totalFloatCount);


        // Allocate host-side buffer
        queue.wait();
        // Copy device framebuffer → host buffer
        queue.memcpy(
            hostSideFramebuffer.data(), // destination
            sensorGpu.visibilityWeightedOpacityBuffer, // source (device pointer)
            totalFloatCount * sizeof(float) // size in bytes
        ).wait();

        return hostSideFramebuffer;
    }

    struct DebugGradientImagesHost {
        // Each buffer has size: width * height * 4 (RGBA)
        std::vector<float> positionX; // framebuffer_pos
        std::vector<float> positionY; // framebuffer_pos
        std::vector<float> positionZ; // framebuffer_pos
        std::vector<float> rotationX;
        std::vector<float> rotationY;
        std::vector<float> rotationZ;
        std::vector<float> scaleU; // framebuffer_scale
        std::vector<float> scaleV; // framebuffer_scale
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
        images.rotationX.resize(rgbaFloatCount);
        images.rotationY.resize(rgbaFloatCount);
        images.rotationZ.resize(rgbaFloatCount);
        images.scaleU.resize(rgbaFloatCount);
        images.scaleV.resize(rgbaFloatCount);
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
        downloadScalarImageAsGrayscaleRgba(images.rotationX, debugImages.framebufferRotX);
        downloadScalarImageAsGrayscaleRgba(images.rotationY, debugImages.framebufferRotY);
        downloadScalarImageAsGrayscaleRgba(images.rotationZ, debugImages.framebufferRotZ);
        downloadScalarImageAsGrayscaleRgba(images.scaleU, debugImages.framebufferScaleU);
        downloadScalarImageAsGrayscaleRgba(images.scaleV, debugImages.framebufferScaleV);
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

    inline void freeGradientsForScene(sycl::queue queue, PointGradients &gradients) {
        const auto freeDevicePtr = [&queue]<typename T>(T *&devicePtr) {
            if (devicePtr) {
                sycl::free(devicePtr, queue);
                devicePtr = nullptr;
            }
        };

        freeDevicePtr(gradients.gradPosition);
        freeDevicePtr(gradients.cloneSignal);
        freeDevicePtr(gradients.gradRotation);
        freeDevicePtr(gradients.gradScale);
        freeDevicePtr(gradients.gradAlbedo);
        freeDevicePtr(gradients.gradOpacity);
        freeDevicePtr(gradients.gradBeta);
        freeDevicePtr(gradients.gradShape);

        freeDevicePtr(gradients.gradPositionPerPrimitivePerCamera);
        freeDevicePtr(gradients.gradPositionRecordCountPerPrimitivePerCamera);
        freeDevicePtr(gradients.cloneSignalPerPrimitivePerCamera);
        freeDevicePtr(gradients.cloneSignalRecordCountPerPrimitivePerCamera);

        freeDevicePtr(gradients.cloneSignalMeanNorm);
        freeDevicePtr(gradients.cloneSignalStd);
        freeDevicePtr(gradients.cloneSignalCoherence);
        freeDevicePtr(gradients.cloneSignalDisagreement);
        freeDevicePtr(gradients.cloneSignalActiveCameraCount);

        gradients.numPoints = 0;
        gradients.cameraSlotCount = 0;

        queue.wait();
    }

    inline void freeDebugImagesForScene(
        sycl::queue queue,
        DebugImages *debugImages,
        std::size_t debugImageCount) {
        if (!debugImages) {
            return;
        }

        for (std::size_t imageIndex = 0; imageIndex < debugImageCount; ++imageIndex) {
            DebugImages &debugImage = debugImages[imageIndex];

            if (debugImage.framebufferPosX) {
                sycl::free(debugImage.framebufferPosX, queue);
                debugImage.framebufferPosX = nullptr;
            }
            if (debugImage.framebufferPosY) {
                sycl::free(debugImage.framebufferPosY, queue);
                debugImage.framebufferPosY = nullptr;
            }
            if (debugImage.framebufferPosZ) {
                sycl::free(debugImage.framebufferPosZ, queue);
                debugImage.framebufferPosZ = nullptr;
            }
            if (debugImage.framebufferRotX) {
                sycl::free(debugImage.framebufferRotX, queue);
                debugImage.framebufferRotX = nullptr;
            }
            if (debugImage.framebufferRotY) {
                sycl::free(debugImage.framebufferRotY, queue);
                debugImage.framebufferRotY = nullptr;
            }
            if (debugImage.framebufferRotZ) {
                sycl::free(debugImage.framebufferRotZ, queue);
                debugImage.framebufferRotZ = nullptr;
            }
            if (debugImage.framebufferScaleU) {
                sycl::free(debugImage.framebufferScaleU, queue);
                debugImage.framebufferScaleU = nullptr;
            }
            if (debugImage.framebufferScaleV) {
                sycl::free(debugImage.framebufferScaleV, queue);
                debugImage.framebufferScaleV = nullptr;
            }
            if (debugImage.framebufferOpacity) {
                sycl::free(debugImage.framebufferOpacity, queue);
                debugImage.framebufferOpacity = nullptr;
            }
            if (debugImage.framebufferAlbedo) {
                sycl::free(debugImage.framebufferAlbedo, queue);
                debugImage.framebufferAlbedo = nullptr;
            }
            if (debugImage.framebufferBeta) {
                sycl::free(debugImage.framebufferBeta, queue);
                debugImage.framebufferBeta = nullptr;
            }
            if (debugImage.framebufferDepthLoss) {
                sycl::free(debugImage.framebufferDepthLoss, queue);
                debugImage.framebufferDepthLoss = nullptr;
            }
            if (debugImage.framebufferDepthLossPos) {
                sycl::free(debugImage.framebufferDepthLossPos, queue);
                debugImage.framebufferDepthLossPos = nullptr;
            }
            if (debugImage.framebufferNormalLoss) {
                sycl::free(debugImage.framebufferNormalLoss, queue);
                debugImage.framebufferNormalLoss = nullptr;
            }

            debugImage.numPixels = 0;
        }

        queue.wait();
    }
}
