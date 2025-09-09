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
    PathTracer::PathTracer(sycl::queue q, const PathTracerSettings &settings) : m_queue(q), m_settings(settings) {
#ifdef NDEBUG
        // Release
        m_settings.photonsPerLaunch = 1e7; // 1e7
        m_settings.maxBounces = 6;

#else
        // Debug
        m_settings.photonsPerLaunch = 1e6; // 1e6
        m_settings.maxBounces = 2;
#endif

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


    void PathTracer::setScene(const GPUSceneBuffers &scene) {
        m_scene = scene;
    }

    void PathTracer::renderForward(SensorGPU &sensor) {
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

    void PathTracer::renderBackward(SensorGPU &sensor, AdjointGPU& adjoint) {

        if (!adjoint.framebuffer) {
            Log::PA_WARN("Adjoint image is not set but renderBackward is called");
            return;
        }

        m_settings.rayGenMode = RayGenMode::Adjoint;

        RenderPackage renderPackage{
            .queue = m_queue,
            .settings = m_settings,
            .scene = m_scene,
            .intermediates = m_intermediates,
            .sensor = sensor,
            .adjoint = adjoint
        };

        submitKernel(renderPackage);

        m_queue.wait();
    }

    void PathTracer::reset() {
    }
}
