// SyclWarmup.cpp (no imports of your modules)
#include <sycl/sycl.hpp>
#include "SyclBridge.h"

#include "Renderer/Kernels/KernelHelpers.h"
#include "Renderer/Kernels/PathTracerKernels.h"
#include "Renderer/GPUDataStructures.h"

namespace Pale {
    // ---- Tags ---------------------------------------------------------------
    struct WarmupKernelTag {
    };

    struct RayGenEmitterKernelTag {
    };

    struct IntersectKernelTag {
    };

    struct ShadeKernelTag {
    };

    // ---- Warmup -------------------------------------------------------------
    void warmupKernelSubmit(void* queuePtr, std::size_t totalWorkItems) {
        auto& queue = *static_cast<sycl::queue*>(queuePtr);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<WarmupKernelTag>(
                sycl::range<1>(totalWorkItems),
                [](sycl::id<1>) {
                });
        }).wait();
    }

    // ---- Helpers ------------------------------------------------------------
    inline void resetDeviceCounter(sycl::queue& queue, uint32_t* counterPtr) {
        queue.memset(counterPtr, 0, sizeof(uint32_t));
    }

    // ---- Kernels ------------------------------------------------------------
    void launchRayGenEmitterKernel(sycl::queue queue,
                                          PathTracerSettings settings,
                                          GPUSceneBuffers scene,
                                          SensorGPU sensor,
                                          RenderIntermediatesGPU renderIntermediates
                                          ) {
        const uint32_t photonCount = settings.photonsPerLaunch;

        queue.submit([&](sycl::handler& commandGroupHandler) {
            uint64_t baseSeed = settings.randomSeed;

            commandGroupHandler.parallel_for<RayGenEmitterKernelTag>(
                sycl::range<1>(photonCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t photonIndex = globalId[0];

                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId);

                    // Choose any generator you like:
                   rng::Xorshift128 rng128(perItemSeed);

                    /*
                    // 1) pick emitter by power CDF
                    const float u = rng128.nextFloat();
                    const uint32_t emitterIdx = sampleEmitterIndex(scene.emitterCdf, scene.emitterCount, u);

                    const EmitterGPU emitter = scene.emitters[emitterIdx];
                    */

                    // 2) sample position and direction on the emitter
                    // TODO: implement for your emitter types (area, point, directional)
                    sycl::float3 x = /* sample position on emitter */ sycl::float3{0, 0, 0};
                    sycl::float3 n = /* emitter normal           */ sycl::float3{0, 0, 1};
                    sycl::float3 wo; // sampled emission direction
                    {
                        // cosine hemisphere as placeholder
                        const float r1 = rng128.nextFloat();
                        const float r2 = rng128.nextFloat();
                        const float phi = 2.f * 3.1415926535f * r1;
                        const float z = r2;
                        const float r = sycl::sqrt(1.f - z * z);
                        wo = sycl::float3{r * sycl::cos(phi), r * sycl::sin(phi), z};
                    }


                    // 3) initial photon throughput = emitter radiance * jacobian / pdf
                    sycl::float3 initialThroughput = sycl::float3{1.f, 1.f, 1.f}; // placeholder

                    RayState ray{};
                    ray.rayOrigin = x;
                    ray.rayDirection = wo;
                    ray.pathThroughput = initialThroughput;
                    ray.pixelIndex = 0u; // not tied to a pixel yet
                    ray.bounceIndex = 0u;

                    auto counter = sycl::atomic_ref<uint32_t,
                                                    sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>(
                        *renderIntermediates.countPrimary);
                    const uint32_t slot = counter.fetch_add(1);
                    renderIntermediates.primaryRays[slot] = ray;
                });
        });
    }



    inline void launchIntersectKernel(sycl::queue& queue,
                                      GPUSceneBuffers scene,
                                      const RayState* raysIn,
                                      uint32_t rayCount,
                                      HitRecord* hitRecords) {
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<IntersectKernelTag>(
                sycl::range<1>(rayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t i = globalId[0];
                    HitRecord hit{};
                    // TODO: BVH traversal
                    hit.didHit = 0;
                    hitRecords[i] = hit;
                });
        });
    }
    /*
    inline void launchShadeKernel(sycl::queue& queue,
                                  GPUSceneBuffers scene,
                                  SensorGPU sensor,
                                  const HitRecord* hitRecords,
                                  const RayState* raysIn,
                                  uint32_t rayCount,
                                  RayState* raysOut) {
        resetDeviceCounter(queue, sensor.countExtensionOut);

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<ShadeKernelTag>(
                sycl::range<1>(rayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t i = globalId[0];

                    const HitRecord hit = hitRecords[i];
                    const RayState  rs  = raysIn[i];

                    if (!hit.didHit) {
                        // TODO: accumulate environment
                        return;
                    }

                    // TODO: accumulate direct lighting

                    RayState next{};
                    next.rayOrigin      = hit.hitPosition;
                    next.rayDirection   = sycl::float3{0.f, 1.f, 0.f}; // placeholder
                    next.pathThroughput = rs.pathThroughput;
                    next.pixelIndex     = rs.pixelIndex;
                    next.bounceIndex    = rs.bounceIndex + 1;

                    auto counter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::ext_intel_global_device_space>(*sensor.countExtensionOut);
                    uint32_t slot = counter.fetch_add(1);
                    raysOut[slot] = next;
                });
        });
    }
    */

    // ---- Orchestrator -------------------------------------------------------
    void submitKernel(RenderPackage& pkg) {

        pkg.queue.fill(pkg.sensor.framebuffer, sycl::float4{0, 0, 0, 0}, pkg.sensor.height * pkg.sensor.width).wait();


        pkg.queue.submit([&](sycl::handler& commandGroupHandler) {
            PathTracerMeshKernel kernel(pkg.scene, pkg.sensor);
            commandGroupHandler.parallel_for<PathTracerMeshKernel>(sycl::range<1>(pkg.sensor.width * pkg.sensor.height), kernel);
        });

        // Ray generation mode
        switch (pkg.settings.rayGenMode) {
        case RayGenMode::Emitter:
            launchRayGenEmitterKernel(pkg.queue, pkg.settings, pkg.scene, pkg.sensor, pkg.intermediates);
            pkg.queue.wait();
            break;
        case RayGenMode::Adjoint:

            break;
        default:
            ;
        }



        RayState* raysIn  = pkg.intermediates.primaryRays;
        RayState* raysOut = pkg.intermediates.extensionRaysA;

        uint32_t activeCount = 0;
        pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();

        for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
            /*
            launchIntersectKernel(queue, scene, raysIn, activeCount, sensor.hitRecords);
            launchShadeKernel(queue, scene, sensor, sensor.hitRecords, raysIn, activeCount, raysOut);

            uint32_t nextCount = 0;
            queue.memcpy(&nextCount, sensor.countExtensionOut, sizeof(uint32_t)).wait();

            std::swap(raysIn, raysOut);
            raysOut = (raysOut == sensor.extensionRaysA) ? sensor.extensionRaysB : sensor.extensionRaysA;
            activeCount = nextCount;
            */
        }

    }
}
