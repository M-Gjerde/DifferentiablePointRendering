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
        explicit PathTracer(sycl::queue q, const PathTracerSettings &settings = {});

        void setScene(const GPUSceneBuffers &scene, SceneBuild::BuildProducts bp);

        void setCurvatureDensificationStats(CurvatureDensificationStats *stats) {
            m_curvatureDensificationStats = stats;
        }

        void renderForward(std::vector<SensorGPU> &sensors);

        void renderBackward(std::vector<SensorGPU> &sensor, PointGradients &gradients, DebugImages *debugImages);

        void renderDepthDistortionBackward(std::vector<SensorGPU> &sensor, PointGradients &gradients);

        void renderNormalConsistencyBackward(std::vector<SensorGPU> &sensor, PointGradients &gradients);

        void renderSurfaceRegularizersBackward(std::vector<SensorGPU> &sensors,
                                               PointGradients &depthDistortionGradients,
                                               PointGradients &normalConsistencyGradients,
                                               PointGradients &visibilityOpacityGradients,
                                               PointGradients &intraSlabDepthGradients,
                                               PointGradients &curvatureScaleGradients,
                                               DebugImages *debugImages);

        void reset();

        PathTracerSettings &getSettings() { return m_settings; }

    private:
        void ensureRayCapacity(uint32_t requiredRayQueueCapacity);

        void ensurePhotonGridBuffersAllocatedAndInitialized(DeviceSurfacePhotonMapGrid &grid);

        void allocateIntermediates(uint32_t newCapacity);

        void allocatePhotonMap();

        void freeIntermediates();

        void freePhotonMap();

        void freePhotonGridBuffers();

        void configurePhotonGrid(const AABB &sceneAabb);

    private:
        sycl::queue m_queue;
        GPUSceneBuffers m_sceneGPU{};
        RenderIntermediatesGPU m_intermediates{};
        PathTracerSettings m_settings{};
        CurvatureDensificationStats *m_curvatureDensificationStats = nullptr;
        uint32_t m_rayQueueCapacity = 0;
        uint64_t m_sessionSeed = 42;
    };
}
