//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"


namespace Pale {
    void launchRayGenAdjointKernel(sycl::queue queue,
                                   PathTracerSettings settings,
                                   SensorGPU cameraSensor,
                                   AdjointGPU adjointSensor,
                                   GPUSceneBuffers scene,
                                   RenderIntermediatesGPU renderIntermediates) {
        const uint32_t imageWidth = cameraSensor.camera.width;
        const uint32_t imageHeight = cameraSensor.camera.height;
        const uint32_t rayCount = imageWidth * imageHeight;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            const uint64_t baseSeed = settings.randomSeed;
            const float invWidth = 1.f / imageWidth;
            const float invHeight = 1.f / imageHeight;

            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(rayCount),
                [=](sycl::id<1> globalId) {
                    const auto pixelLinearIndex = static_cast<uint32_t>(globalId[0]);

                    const uint32_t pixelX = pixelLinearIndex % imageWidth;
                    const uint32_t pixelY = pixelLinearIndex / imageWidth;

                    // RNG per pixel
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId[0]);
                    rng::Xorshift128 rng128(perItemSeed);
                    const float jitterX = rng128.nextFloat();
                    const float jitterY = rng128.nextFloat();

                    const CameraGPU adjointCamera = cameraSensor.camera;

                    // Subpixel sample in [0,1]^2
                    const float sampleX = (static_cast<float>(pixelX) + jitterX) * invWidth;
                    const float sampleY = (static_cast<float>(pixelY) + jitterY) * invHeight;

                    // NDC → clip (OpenGL-style NDC in [-1,1])
                    const float ndcX = 2.f * sampleX - 1.f;
                    const float ndcY = 1.f - 2.f * sampleY;

                    // Ray in view space: unproject near and far, then get direction
                    const float3 pNearView = transformPoint(adjointCamera.invProj, {ndcX, ndcY, -1.f}, 1.f);
                    const float3 pFarView = transformPoint(adjointCamera.invProj, {ndcX, ndcY, 1.f}, 1.f);
                    const float3 dirView = normalize(pFarView - pNearView);

                    // To world
                    const float3 rayOriginWorld = transformPoint(adjointCamera.invView, {0.f, 0.f, 0.f}, 1.f);
                    // camera position
                    const float3 rayDirectionWorld = transformDirection(adjointCamera.invView, dirView);

                    // Initial adjoint throughput = residual (linear RGB). Alpha ignored.
                    const uint32_t pixelIndex = pixelLinearIndex;
                    const float4 residualRgba = adjointSensor.framebuffer[pixelIndex]; // linear RGB adjoint source

                    float3 initialAdjointWeight = {
                        residualRgba.x(), // R
                        residualRgba.y(), // G
                        residualRgba.z() // B
                    };

                    const float weightLen2 =
                            initialAdjointWeight.x() * initialAdjointWeight.x() +
                            initialAdjointWeight.y() * initialAdjointWeight.y() +
                            initialAdjointWeight.z() * initialAdjointWeight.z();

                    //if (residualRgba.x() == 0.0f) return;

                    RayState rayState{};
                    rayState.ray.origin = rayOriginWorld;
                    rayState.ray.direction = rayDirectionWorld;
                    rayState.pathThroughput = float3(weightLen2);
                    rayState.bounceIndex = 0;

                    auto outputCounter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*renderIntermediates.countPrimary);

                    const uint32_t outputSlot = outputCounter.fetch_add(1u);
                    renderIntermediates.primaryRays[outputSlot] = rayState;
                });
        });
        queue.wait();
    }

    void launchAdjointShadeKernel(sycl::queue &queue,
                                  GPUSceneBuffers scene,
                                  SensorGPU cameraSensor,
                                  AdjointGPU adjointSensor,
                                  const WorldHit *hitRecords,
                                  const RayState *raysIn,
                                  uint32_t rayCount,
                                  RayState *raysOut,
                                  RenderIntermediatesGPU renderIntermediates,
                                  const PathTracerSettings &settings

    ) {
        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<struct AjointShadeKernelTag>(
                sycl::range<1>(rayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    constexpr float kEps = 1e-4f;
                    // Choose any generator you like:

                    const WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];

                    if (worldHit.t == FLT_MAX) {
                        return;
                    }

                    auto &instance = scene.instances[worldHit.instanceIndex];
                    GPUMaterial material;
                    switch (instance.geometryType) {
                        case GeometryType::Mesh:
                            material = scene.materials[instance.materialIndex];
                            break;
                        case GeometryType::PointCloud: {
                            auto val = scene.points[worldHit.primitiveIndex];
                            material.baseColor = val.color;
                        }
                        break;
                    }

                    auto &camera = cameraSensor.camera;

                    float4 clip = camera.proj * (camera.view * float4(worldHit.hitPositionW, 1.f));

                    if (clip.w() > 0.0f) {
                        float2 ndc = {clip.x() / clip.w(), clip.y() / clip.w()};
                        if (ndc.x() >= -1.f && ndc.x() <= 1.f && ndc.y() >= -1.f && ndc.y() <= 1.f) {
                            /* 2)  raster coords (clamp to avoid the right/top fenceposts) */
                            uint32_t px = sycl::clamp(
                                static_cast<uint32_t>((ndc.x() * 0.5f + 0.5f) * camera.width),
                                0u, camera.width - 1);
                            uint32_t py = sycl::clamp(
                                static_cast<uint32_t>((ndc.y() * 0.5f + 0.5f) * camera.height),
                                0u, camera.height - 1);

                            // FLIP Y
                            const uint32_t idx = (camera.height - 1u - py) * camera.width + px;

                            float4 &dst = cameraSensor.framebuffer[idx];
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
                            float3 color = material.baseColor;
                            color = float3(rayState.pathThroughput);
                            r.fetch_add(color.x());
                            g.fetch_add(color.y());
                            b.fetch_add(color.z());
                            a.store(1.0f);
                        }
                    }
                });
        });
        queue.wait();
    }
}
