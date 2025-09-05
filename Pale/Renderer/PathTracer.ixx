//
// Created by magnus-desktop on 8/28/25.
//
module;

#include <sycl/sycl.hpp>
#include "Kernels/SyclBridge.h"
#include <filesystem>
export module Pale.Render.PathTracer;

import Pale.Render.Sensors;
import Pale.Render.SceneUpload;

export namespace Pale {

    class PathTracer {
    public:
        explicit PathTracer(sycl::queue q, const PathTracerSettings& settings = {});
        void setScene(const GPUSceneBuffers& s);
        void renderForward(SensorGPU& sensors);
        void setResiduals(SensorGPU& sensors, const std::filesystem::path& targetImage);
        void renderBackward(SensorGPU& sensors);
        void reset();

    private:
        void ensureCapacity(uint32_t requiredRayQueueCapacity);
        void allocateIntermediates(uint32_t newCapacity);
        void freeIntermediates();
    private:
        sycl::queue m_queue;
        GPUSceneBuffers m_scene{};
        RenderIntermediatesGPU m_intermediates{};
        PathTracerSettings m_settings{};
        uint32_t m_rayQueueCapacity = 0;
    };

}
