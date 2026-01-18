//
// Created by magnus on 11/21/25.
//

#include "Renderer/Kernels/PostprocessKernel.h"

#include "KernelHelpers.h"

namespace Pale {

    void launchPostProcessKernel(RenderPackage& pkg) {
        auto& queue = pkg.queue;
        auto sensor = pkg.sensor;
        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            SensorGPU sensor = pkg.sensor[cameraIndex];
            const uint32_t imageWidth = sensor.camera.width;
            const uint32_t imageHeight = sensor.camera.height;
            const uint32_t raysPerSet = imageWidth * imageHeight;


            queue.submit([&](sycl::handler& commandGroupHandler) {
                const float exposureCorrection = sensor.exposureCorrection;
                const float gammaCorrection = sensor.gammaCorrection;
                const float inverseGamma =
                    (gammaCorrection > 0.0f) ? (1.0f / gammaCorrection) : 1.0f;

                commandGroupHandler.parallel_for<class PostProcessKernelTag>(
                    sycl::range<1>(raysPerSet),
                    // ReSharper disable once CppDFAUnusedValue
                    [=](sycl::id<1> globalId) {
                        const uint32_t globalRayIndex = static_cast<uint32_t>(globalId[0]);

                        // Map to pixel (linear index; X/Y only needed if you want them)
                        const uint32_t linearIndex = globalRayIndex;

                        // ------------------------------------------------------------
                        // 2. Compute flipped Y coordinate for LDR output
                        // ------------------------------------------------------------
                        const uint32_t linearIndexFlipped = flippedYLinearIndex(linearIndex, sensor.width, sensor.height);


                        // Read HDR color
                        const uint32_t pixelIndex = linearIndex;
                        const float4 hdrRgba = sensor.framebuffer[pixelIndex];

                        float redLinear = hdrRgba.x();
                        float greenLinear = hdrRgba.y();
                        float blueLinear = hdrRgba.z();
                        float alphaLinear = hdrRgba.w(); // if you care about alpha

                        // Apply exposure
                        redLinear *= exposureCorrection;
                        greenLinear *= exposureCorrection;
                        blueLinear *= exposureCorrection;

                        // Clamp to [0, +inf) before gamma
                        redLinear = sycl::fmax(redLinear, 0.0f);
                        greenLinear = sycl::fmax(greenLinear, 0.0f);
                        blueLinear = sycl::fmax(blueLinear, 0.0f);

                        // Apply gamma (convert to display space)
                        redLinear = sycl::pow(redLinear, inverseGamma);
                        greenLinear = sycl::pow(greenLinear, inverseGamma);
                        blueLinear = sycl::pow(blueLinear, inverseGamma);

                        // Clamp to [0,1] before quantization
                        redLinear = sycl::clamp(redLinear, 0.0f, 1.0f);
                        greenLinear = sycl::clamp(greenLinear, 0.0f, 1.0f);
                        blueLinear = sycl::clamp(blueLinear, 0.0f, 1.0f);

                        // If you want alpha fixed to 1.0 (opaque), do that here:
                        alphaLinear = sycl::clamp(alphaLinear, 0.0f, 1.0f);

                        auto convertChannelToUint8 = [](float channelValue) -> unsigned char {
                            // scale to [0,255] with simple rounding
                            const float scaledValue =
                                sycl::clamp(channelValue * 255.0f + 0.5f, 0.0f, 255.0f);
                            return static_cast<unsigned char>(scaledValue);
                        };

                        const unsigned char redU8 = convertChannelToUint8(redLinear);
                        const unsigned char greenU8 = convertChannelToUint8(greenLinear);
                        const unsigned char blueU8 = convertChannelToUint8(blueLinear);

                        sycl::uchar4 outputPixel(
                            redU8,
                            greenU8,
                            blueU8,
                            255
                        );

                        // 3 floats per pixel, packed
                        const uint32_t ldrBase = linearIndexFlipped * 3u;
                        sensor.ldrFramebuffer[ldrBase + 0u] = redLinear;
                        sensor.ldrFramebuffer[ldrBase + 1u] = greenLinear;
                        sensor.ldrFramebuffer[ldrBase + 2u] = blueLinear;


                        sensor.outputFramebuffer[linearIndexFlipped] = outputPixel;
                    }
                );
            });
            queue.wait();
        }
    }

void accumulatePhotonEnergyPerSurfelDebug(RenderPackage& renderPackage)
{
    auto& computeQueue = renderPackage.queue;
    auto& scene = renderPackage.scene;          // GPUSceneBuffers
    auto& photonMap = renderPackage.intermediates.map;

    const uint32_t surfelCount = scene.pointCount;
    if (surfelCount == 0) {
        return;
    }

    // Temporary device buffers
    float3* surfelEnergyDevicePtr =
        sycl::malloc_device<float3>(surfelCount, computeQueue);
    float* surfelAreaDevicePtr =
        sycl::malloc_device<float>(surfelCount, computeQueue);

    // Initialize energy and area to zero
    computeQueue
        .fill(surfelEnergyDevicePtr, float3{0.0f, 0.0f, 0.0f}, surfelCount)
        .wait();
    computeQueue
        .fill(surfelAreaDevicePtr, 0.0f, surfelCount)
        .wait();

    // Traverse all photons and accumulate per-surfel data
    computeQueue.submit([&](sycl::handler& commandGroupHandler) {
        const uint32_t photonCapacity = photonMap.photonCapacity;
        DevicePhotonSurface* photonsDevicePtr = photonMap.photons;
        GPUSceneBuffers deviceScene = scene; // capture by value for device

        commandGroupHandler.parallel_for<class AccumulatePhotonEnergyAndAreaPerSurfelDebugKernel>(
            sycl::range<1>(photonCapacity),
            [=](sycl::id<1> globalId) {
                const uint32_t photonIndex = globalId[0];
                const DevicePhotonSurface photon = photonsDevicePtr[photonIndex];

                // Skip unused photon slots (adapt sentinel as needed)
                if (photon.power.x() == 0.0f &&
                    photon.power.y() == 0.0f &&
                    photon.power.z() == 0.0f)
                {
                    return;
                }

                // Only surfel photons
                if (photon.geometryType != GeometryType::PointCloud) {
                    return;
                }

                const uint32_t surfelIndex = photon.primitiveIndex;
                if (surfelIndex >= deviceScene.pointCount) {
                    return;
                }

                // --- Surfel area from GPU scene.points[] ---
                const Point surfel = deviceScene.points[surfelIndex];
                const float su = surfel.scale.x();
                const float sv = surfel.scale.y();
                const float surfelArea = su * sv; // as requested: su * sv

                {
                    sycl::atomic_ref<float,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomicArea(surfelAreaDevicePtr[surfelIndex]);
                    // su*sv is the same for all photons on this surfel, so store is fine
                    atomicArea.store(surfelArea);
                }

                // --- Accumulate RGB energy ---
                {
                    sycl::atomic_ref<float,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomicEnergyR(surfelEnergyDevicePtr[surfelIndex].x());
                    atomicEnergyR.fetch_add(photon.power.x());
                }
                {
                    sycl::atomic_ref<float,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomicEnergyG(surfelEnergyDevicePtr[surfelIndex].y());
                    atomicEnergyG.fetch_add(photon.power.y());
                }
                {
                    sycl::atomic_ref<float,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomicEnergyB(surfelEnergyDevicePtr[surfelIndex].z());
                    atomicEnergyB.fetch_add(photon.power.z());
                }
            });
    }).wait();

    // Copy back to host
    std::vector<float3> surfelEnergyHost(surfelCount);
    std::vector<float> surfelAreaHost(surfelCount);

    computeQueue.memcpy(surfelEnergyHost.data(),
                        surfelEnergyDevicePtr,
                        sizeof(float3) * surfelCount).wait();
    computeQueue.memcpy(surfelAreaHost.data(),
                        surfelAreaDevicePtr,
                        sizeof(float) * surfelCount).wait();

    // Free device buffers
    sycl::free(surfelEnergyDevicePtr, computeQueue);
    sycl::free(surfelAreaDevicePtr, computeQueue);

    // Write CSV
    std::ofstream csvFileStream("surfel_energy_debug.csv");
    if (!csvFileStream.is_open()) {
        return;
    }

    csvFileStream << "surfel_index,area,energyR,energyG,energyB\n";
    for (uint32_t surfelIndex = 0; surfelIndex < surfelCount; ++surfelIndex) {
        const float area = surfelAreaHost[surfelIndex];
        const float3 energy = surfelEnergyHost[surfelIndex];

        csvFileStream << surfelIndex << ","
                      << area << ","
                      << energy.x() << ","
                      << energy.y() << ","
                      << energy.z() << "\n";
    }
    csvFileStream.close();
}

} // namespace Pale

