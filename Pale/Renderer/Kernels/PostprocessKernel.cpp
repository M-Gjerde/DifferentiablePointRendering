//
// Created by magnus on 11/21/25.
//

#include "Renderer/Kernels/PostprocessKernel.h"

namespace Pale {
    void launchPostProcessKernel(RenderPackage &pkg) {
        auto &queue = pkg.queue;
        auto sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {

            SensorGPU sensor = pkg.sensor[cameraIndex];
            const uint32_t imageWidth =  sensor.camera.width;
            const uint32_t imageHeight = sensor.camera.height;

            const uint32_t raysPerSet = imageWidth * imageHeight;

            queue
                    .memcpy(pkg.intermediates.countPrimary, &raysPerSet, sizeof(uint32_t))
                    .wait();

            queue.submit([&](sycl::handler &commandGroupHandler) {
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
                        // 1. Recover pixel coordinates (unflipped)
                        // ------------------------------------------------------------
                        const uint32_t pixelX = linearIndex % imageWidth;
                        const uint32_t pixelY = linearIndex / imageWidth;

                        // ------------------------------------------------------------
                        // 2. Compute flipped Y coordinate for LDR output
                        // ------------------------------------------------------------
                        const uint32_t flippedY = (imageHeight - 1u) - pixelY;
                        const uint32_t flippedLinearIndex = flippedY * imageWidth + pixelX;


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
                        const uint32_t ldrBase = flippedLinearIndex * 3u;
                        sensor.ldrFramebuffer[ldrBase + 0u] = redLinear;
                        sensor.ldrFramebuffer[ldrBase + 1u] = greenLinear;
                        sensor.ldrFramebuffer[ldrBase + 2u] = blueLinear;


                        sensor.outputFramebuffer[flippedLinearIndex] = outputPixel;




                    }
                );
            });
            queue.wait();
        }
    }
}
