//
// Created by magnus-desktop on 8/28/25.
//
module;

#include <sycl/sycl.hpp>
#include "Kernels/SyclBridge.h"

export module Pale.Render.PathTracer;

import Pale.Render.Sensors;
import Pale.Render.SceneUpload;
import Pale.Render.PathTracerConfig;

export namespace Pale {

    class PathTracer {
    public:
        explicit PathTracer(sycl::queue q);
        void setScene(const GPUSceneBuffers& s);
        void renderForward(const RenderBatch& b,
                           SensorGPU& sensors);

        void renderBackward(SensorGPU& sensors);
        void reset();

    private:

        sycl::queue m_queue;
        GPUSceneBuffers m_scene{};

    };

}
