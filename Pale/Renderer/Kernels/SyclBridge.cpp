// SyclWarmup.cpp (no imports of your modules)
#include <sycl/sycl.hpp>
#include "SyclBridge.h"

#include "Renderer/Kernels/PathTracerKernels.h"
#include "Renderer/GPUDataStructures.h"

namespace Pale {
    struct WarmupKernelTag {};
    void warmup_kernel_submit(void* queue_ptr, std::size_t n) {
        auto& q = *static_cast<sycl::queue*>(queue_ptr);
        q.submit([&](sycl::handler& h) {
            h.parallel_for<WarmupKernelTag>(sycl::range<1>(n), [](sycl::id<1>) {});
            int stack[10];

            for (int i = 1; i < 10000; i++) {

            }

        }).wait();
    }

    struct ClearSensorKernelTag {};
    void submitKernel(sycl::queue& queue, GPUSceneBuffers scene, SensorGPU sensor) {

        queue.submit([&](sycl::handler& commandGroupHandler) {
            PathTracerMeshKernel kernel(scene, sensor);
            commandGroupHandler.parallel_for<PathTracerMeshKernel>(sycl::range<1>(1000), kernel);
        });

    }


}
