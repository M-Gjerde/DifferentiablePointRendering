//
// Created by magnus-desktop on 8/28/25.
//
module;

#include <sycl/sycl.hpp>
#include <filesystem>
#include "Kernels/SyclBridge.h"

module Pale.Render.PathTracer;

import Pale.Utils.ImageIO;
import Pale.Log;

namespace Pale {
    PathTracer::PathTracer(sycl::queue q, const PathTracerSettings& settings) : m_queue(q), m_settings(settings) {

#ifdef NDEBUG
        // Release
        m_settings.photonsPerLaunch = 1e7; // 1e7
#else
        // Debug
        m_settings.photonsPerLaunch = 1e6;  // 1e6
#endif

        m_settings.maxBounces = 2;
    }

    // Call this before first render, or inside submitKernel() after computing capacity.
    void PathTracer::ensureCapacity(uint32_t requiredRayQueueCapacity) {
        if (requiredRayQueueCapacity <= m_rayQueueCapacity) return;
        // grow to next power of two to avoid frequent reallocations
        uint32_t newCapacity = 1u;
        while (newCapacity < requiredRayQueueCapacity) newCapacity <<= 1u;
        allocateIntermediates(newCapacity);
    }

    void PathTracer::allocateIntermediates(uint32_t newCapacity) {
        freeIntermediates();
        m_rayQueueCapacity = newCapacity;

        m_intermediates.primaryRays = sycl::malloc_device<RayState>(m_rayQueueCapacity, m_queue);
        m_intermediates.extensionRaysA = sycl::malloc_device<RayState>(m_rayQueueCapacity, m_queue);
        m_intermediates.hitRecords = sycl::malloc_device<WorldHit>(m_rayQueueCapacity, m_queue);
        m_intermediates.countPrimary = sycl::malloc_device<uint32_t>(1, m_queue);
        m_intermediates.countExtensionOut = sycl::malloc_device<uint32_t>(1, m_queue);

        // optional: zero counters once
        m_queue.memset(m_intermediates.countPrimary, 0, sizeof(uint32_t));
        m_queue.memset(m_intermediates.countExtensionOut, 0, sizeof(uint32_t));
    }

    void PathTracer::freeIntermediates() {
        if (!m_rayQueueCapacity) return;
        sycl::free(m_intermediates.primaryRays, m_queue);
        sycl::free(m_intermediates.extensionRaysA, m_queue);
        sycl::free(m_intermediates.hitRecords, m_queue);
        sycl::free(m_intermediates.countPrimary, m_queue);
        sycl::free(m_intermediates.countExtensionOut, m_queue);
        m_intermediates = {};
        m_rayQueueCapacity = 0;
    }


    void PathTracer::setScene(const GPUSceneBuffers& scene) {
        m_scene = scene;
    }

    void PathTracer::renderForward(SensorGPU& sensor) {


        const uint32_t numberOfImagePixels = sensor.width * sensor.height;
        const uint32_t photonBudget = m_settings.photonsPerLaunch;
        const uint32_t requiredCapacity = std::max(numberOfImagePixels, photonBudget);
        ensureCapacity(requiredCapacity);

        m_settings.rayGenMode = RayGenMode::Emitter;

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_scene,
            .intermediates = m_intermediates,
            .sensor = sensor
        };

        submitKernel(renderPackage);

        m_queue.wait();
    }

    void PathTracer::setResiduals(SensorGPU & sensor, const std::filesystem::path &targetImagePath) {
    // 1) Download current predicted image (RGBA, linear)
    std::vector<float> predictedRgba; // size = W*H*4
    uint32_t predictedWidth = 0, predictedHeight = 0;
    {
        predictedRgba = Pale::downloadSensorRGBA(m_queue, sensor);
        predictedWidth  = sensor.width;
        predictedHeight = sensor.height;
    }

    // 2) Load target reference image (RGB, linear)
    std::vector<float> targetRgb; // size = W*H*3
    uint32_t targetWidth = 0, targetHeight = 0;
    if (!Utils::loadPFM_RGB(targetImagePath, targetRgb, targetWidth, targetHeight)) {
        Log::PA_ERROR("Failed to find target image at {}", targetImagePath.string());
        return;
    };

    // 3) Validate dimensions
    if (predictedWidth != targetWidth || predictedHeight != targetHeight) {
        Log::PA_ERROR("setResiduals(): predicted and target image sizes differ");
        return;
    }
    const uint32_t imageWidth  = predictedWidth;
    const uint32_t imageHeight = predictedHeight;
    const size_t   pixelCount  = static_cast<size_t>(imageWidth) * imageHeight;

    // 4) Extract predicted RGB from RGBA
    std::vector<float> predictedRgb(pixelCount * 3u);
    for (size_t i = 0, j = 0; i < pixelCount; ++i, j += 4) {
        predictedRgb[i * 3 + 0] = predictedRgba[j + 0];
        predictedRgb[i * 3 + 1] = predictedRgba[j + 1];
        predictedRgb[i * 3 + 2] = predictedRgba[j + 2];
    }

    // 5) Compute per-pixel residuals and adjoint (for 0.5 * L2)
    //     residual = predicted - target
    //     adjoint  = residual  (∂L/∂predicted)
    std::vector<float> residualRgb(pixelCount * 3u);
    for (size_t k = 0; k < residualRgb.size(); ++k) {
        residualRgb[k] = predictedRgb[k] - targetRgb[k];
    }

    // Optional: if you use mean squared error over N pixels, scale by 1/N
    // const float invPixelCount = 1.0f / static_cast<float>(pixelCount);
    // for (float& v : residualRgb) v *= invPixelCount;

        // 6) Save adjoint image to disk (PFM, RGB)
        const std::filesystem::path adjointPath = "Output/adjoint/adjoint_rgb.pfm";
        Utils::savePFM(adjointPath, residualRgb, imageWidth, imageHeight,
                       /*channels=*/3, /*flipY=*/true);

        // 6b) Also save each RGB component separately
        std::vector<float> residualR(pixelCount);
        std::vector<float> residualG(pixelCount);
        std::vector<float> residualB(pixelCount);

        for (size_t i = 0; i < pixelCount; ++i) {
            residualR[i] = residualRgb[i * 3 + 0];
            residualG[i] = residualRgb[i * 3 + 1];
            residualB[i] = residualRgb[i * 3 + 2];
        }

        Utils::savePFM("Output/adjoint/adjoint_r.pfm", residualR, imageWidth, imageHeight,
                       /*channels=*/1, /*flipY=*/true);
        Utils::savePFM("Output/adjoint/adjoint_g.pfm", residualG, imageWidth, imageHeight,
                       /*channels=*/1, /*flipY=*/true);
        Utils::savePFM("Output/adjoint/adjoint_b.pfm", residualB, imageWidth, imageHeight,
                       /*channels=*/1, /*flipY=*/true);

    // 7) (Optional) also save residual magnitude for debugging
    {
        std::vector<float> residualLuminance(pixelCount);
        for (size_t i = 0; i < pixelCount; ++i) {
            const float r = residualRgb[i * 3 + 0];
            const float g = residualRgb[i * 3 + 1];
            const float b = residualRgb[i * 3 + 2];
            residualLuminance[i] = std::sqrt(r*r + g*g + b*b);
        }
        Utils::savePFM("Output/adjoint/adjoint_mag.pfm", residualLuminance, imageWidth, imageHeight, /*channels=*/1, /*flipY=*/true);
    }

    }

    void PathTracer::renderBackward(SensorGPU& sensor) {

        m_settings.rayGenMode = RayGenMode::Adjoint;

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_scene,
            .intermediates = m_intermediates,
            .sensor = sensor
        };

        submitKernel(renderPackage);

        m_queue.wait();

    }

    void PathTracer::reset() {
    }
}
