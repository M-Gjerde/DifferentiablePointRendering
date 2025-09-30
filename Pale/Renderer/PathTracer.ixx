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
import Pale.Render.BVH;
import Pale.Render.SceneBuild;

export namespace Pale {

    class PathTracer {
    public:
        explicit PathTracer(sycl::queue q, const PathTracerSettings& settings = {});
        void setScene(const GPUSceneBuffers &scene, SceneBuild::BuildProducts bp);
        void renderForward(SensorGPU& sensors, SensorGPU &sensor2);
        void renderBackward(SensorGPU &sensor, AdjointGPU& adjoint);
        void reset();

        PathTracerSettings& getSettings() { return m_settings; }

    private:
        void ensureCapacity(uint32_t requiredRayQueueCapacity);
        void allocateIntermediates(uint32_t newCapacity);
        void freeIntermediates();
        void configurePhotonGrid(const AABB& sceneAabb, float gatherRadiusWorld);
    private:
        sycl::queue m_queue;
        GPUSceneBuffers m_scene{};
        RenderIntermediatesGPU m_intermediates{};
        PathTracerSettings m_settings{};
        uint32_t m_rayQueueCapacity = 0;
    };

}
