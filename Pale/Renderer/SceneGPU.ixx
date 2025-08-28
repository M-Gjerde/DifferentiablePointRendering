// Pale/Render/SceneGPU.ixx
module;

#include <cstdint>
#include <vector>
#include <memory>
#include <sycl/sycl.hpp>

export module Pale.Render.SceneGPU;

import Pale.Scene;

export namespace Pale {

    // device pointers & counts (triangles, vertices, instances, materials, lights, cameras, BVHs…)
    class SceneGPU {
    public:
        static SceneGPU upload(const std::shared_ptr<Pale::Scene>& scene, sycl::queue queue) {


            return {};
        };

    private:
    };
}
