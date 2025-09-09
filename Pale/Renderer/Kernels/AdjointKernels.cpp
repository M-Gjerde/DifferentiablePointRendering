//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"


namespace Pale {
    void launchRayGenAdjointKernel(sycl::queue queue,
                                   PathTracerSettings settings,
                                   GPUSceneBuffers scene,
                                   SensorGPU sensor,
                                   RenderIntermediatesGPU renderIntermediates) {
        const uint32_t photonCount = settings.photonsPerLaunch;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            uint64_t baseSeed = settings.randomSeed;

            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(photonCount),
                [=](sycl::id<1> globalId) {
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId);
                    // Choose any generator you like:
                    rng::Xorshift128 rng128(perItemSeed);
                    auto& adjointCamera = sensor.camera;
                    // Generate rays
                });
        });
    }

    void launchAdjointShadeKernel(sycl::queue &queue,
                                  GPUSceneBuffers scene,
                                  SensorGPU sensor,
                                  const WorldHit *hitRecords,
                                  const RayState *raysIn,
                                  uint32_t rayCount,
                                  RayState *raysOut,
                                  RenderIntermediatesGPU renderIntermediates,
                                  const PathTracerSettings &settings

    ) {
        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<struct AdjointShaderKernel>(
                sycl::range<1>(rayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    constexpr float kEps = 1e-4f;
                    auto &camera = sensor.camera;

                    /* 2)  raster coords (clamp to avoid the right/top fenceposts) */
                    uint32_t px = camera.width / 2 + rng128.nextFloat() * 100;
                    uint32_t py = camera.height / 2 + rng128.nextFloat() * 100;

                    // FLIP Y
                    const uint32_t idx = (camera.height - 1u - py) * camera.width + px;

                    float4 &dst = sensor.framebuffer[idx];
                    const sycl::atomic_ref<float,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            r(dst.x());
                    const sycl::atomic_ref<float,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            g(dst.y());
                    const sycl::atomic_ref<float,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            b(dst.z());
                    const sycl::atomic_ref<float,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            a(dst.w());


                    float3 color(1.0f);

                    r.fetch_add(color.x());
                    g.fetch_add(color.y());
                    b.fetch_add(color.z());
                    a.store(1.0f);
                }
            );
        });
    }
}
