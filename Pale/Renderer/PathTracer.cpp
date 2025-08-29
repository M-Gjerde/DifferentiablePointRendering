//
// Created by magnus-desktop on 8/28/25.
//
module;

#include <sycl/sycl.hpp>

module Pale.Render.PathTracer;


namespace Pale {
    PathTracer::PathTracer(sycl::queue q) : m_queue(q) {
    }

    void PathTracer::setScene(const SceneUpload::GPUSceneBuffers& s) {
    }

    void PathTracer::renderForward(const RenderBatch& b, std::span<const SensorGPU> sensors) {
    }

    void PathTracer::renderBackward(std::span<const SensorGPU> sensors) {
    }

    void PathTracer::reset() {
    }

}
