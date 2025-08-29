//
// Created by magnus-desktop on 8/28/25.
//
module;
#include <sycl/sycl.hpp>

export module Pale.Render.PathTracer;

import Pale.Render.Sensors;
import Pale.Render.SceneUpload;
import Pale.Render.PathTracerConfig;

export namespace Pale {

    class PathTracer {
    public:
        explicit PathTracer(sycl::queue q);
        void setScene(const SceneUpload::GPUSceneBuffers& s);
        void renderForward(const RenderBatch& b,
                           std::span<const SensorGPU> sensors);

        void renderBackward(std::span<const SensorGPU> sensors);
        void reset();

    private:

        sycl::queue& m_queue;
        SceneUpload::GPUSceneBuffers m_scene{};

    };

}
