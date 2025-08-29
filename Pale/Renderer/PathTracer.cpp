//
// Created by magnus-desktop on 8/28/25.
//
module;

#include <sycl/sycl.hpp>

#include "Kernels/SyclBridge.h"

module Pale.Render.PathTracer;


namespace Pale {
    PathTracer::PathTracer(sycl::queue q) : m_queue(q) {
    }

    void PathTracer::setScene(const GPUSceneBuffers &scene) {
        m_scene = scene;
    }

    void PathTracer::renderForward(const RenderBatch &batch, SensorGPU &sensors) {

        Pale::submitKernel(m_queue, m_scene, sensors);
    }

    void PathTracer::renderBackward(SensorGPU &sensors) {
    }

    void PathTracer::reset() {
    }
}
