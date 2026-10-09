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

        ~PathTracer();
        PathTracer(const PathTracer&) = delete;
        PathTracer& operator=(const PathTracer&) = delete;

        uint32_t rayQueueCapacity() const { return m_rayQueueCapacity; }
        bool hasAdjointScratch() const { return m_intermediates.gradientRecords != nullptr; }
        uint32_t adjointPrimarySlabCacheCapacity() const {
            return m_intermediates.adjointPrimarySlabCacheCapacity;
        }

        void setScene(const GPUSceneBuffers &scene, const SceneBuild::BuildProducts &bp);


        void setPrimalActivityStats(PrimalActivityStats *stats) {
            m_primalActivityStats = stats;
        }

        void renderForward(std::vector<SensorGPU> &sensors, bool waitForCompletion = true);

        void renderBackward(std::vector<SensorGPU> &sensor, PointGradients &gradients,
                            DebugImages *debugImages, bool waitForCompletion = true,
                            bool computeCloneStatistics = true);

        void renderDepthDistortionBackward(std::vector<SensorGPU> &sensor, PointGradients &gradients);

        void renderNormalConsistencyBackward(std::vector<SensorGPU> &sensor, PointGradients &gradients);

        void renderSurfaceRegularizersBackward(std::vector<SensorGPU> &sensors,
                                               PointGradients &depthDistortionGradients,
                                               PointGradients &normalConsistencyGradients,
                                               PointGradients &intraSlabDepthGradients,
                                               DebugImages *debugImages, bool waitForCompletion = true);

        void reset();

        PathTracerSettings &getSettings() { return m_settings; }

    private:
        void ensureRayCapacity(uint32_t requiredRayQueueCapacity, bool adjoint = false);

        void ensureMeasurementTwoPointEventCapacity(uint32_t cameraRayCount);
        void ensureAdjointPrimarySlabCacheCapacity(uint32_t cameraRayCount);

        void ensurePhotonGridBuffersAllocatedAndInitialized(DeviceSurfacePhotonMapGrid &grid);

        void allocateIntermediates(uint32_t newCapacity);

        void allocateAdjointIntermediates();

        void allocatePhotonMap();

        void freeIntermediates();

        void freePhotonMap();

        void freePhotonGridBuffers();

        void configurePhotonGrid(const AABB &sceneAabb);

    private:
        sycl::queue m_queue;
        GPUSceneBuffers m_sceneGPU{};
        bool m_singlePointCloudInstance = false;
        RenderIntermediatesGPU m_intermediates{};
        PathTracerSettings m_settings{};
        PrimalActivityStats *m_primalActivityStats = nullptr;
        uint32_t m_rayQueueCapacity = 0;
        uint64_t m_sessionSeed = 42;
    };
}
